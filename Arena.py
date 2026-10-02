r"""
Arena.py

Standalone arena match: loads two model checkpoints, plays them against
each other for N games at N sims/move (batched via mcts_core, same
mechanism as self-play/AnalyzeUI), and saves every game to one PGN file.
Colors alternate game-to-game so neither model gets a first-move-advantage
bias in the result.

Pulled out standalone (rather than reusing TrainingLoop.py's arena_match
directly) so you can run ad-hoc head-to-head comparisons -- e.g. epoch 104
vs epoch 130, or old-architecture vs the new redesigned network -- without
touching the main training loop. Edit the config block below and run.
"""

import os
import time
import random
import chess
import chess.pgn
import torch
import numpy as np

import Network
import Network2
from Encoder import IN_PLANES, ACTION_SIZE, encode_board
from mcts_core_old import MCTS, batch_search, init_tablebase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/, syzygy/, Human Data/ ... live next to this script


# ── Config ────────────────────────────────────────────────────────────────

CHAMPION_PATH   = os.path.join(BASE_DIR, "Models", "chess_selfplay3_epoch_6.pt")
CHALLENGER_PATH = os.path.join(BASE_DIR, "Models", "model_human_pretrained_7-7.pt")

# Architecture per model -- set these separately in case champion/challenger
# come from different network shapes (e.g. comparing the old 96x6 net
# against the redesigned 64x8+SE one). If both checkpoints share the same
# architecture, just set both pairs the same.
CHAMPION_CHANNELS,   CHAMPION_BLOCKS   = 96, 6
CHALLENGER_CHANNELS, CHALLENGER_BLOCKS = 64, 8

NUM_GAMES       = 2    # rounded up to even so colors split exactly in half
MCTS_SIMS_CHAMP       = 800
MCTS_SIMS_CHALLENGER = 1000
MAX_GAME_LENGTH = 210   # plies; games not finished by then are scored as draws

MAX_BATCH_SIZE = 200     # GPU batch size per infer call (many games in parallel here, unlike AnalyzeUI)
MAX_WAIT_S     = 0.01

ARENA_ARGS = {
    'CPUCT':             2,
    'DIRICHLET_ALPHA':   0.15,
    'DIRICHLET_EPSILON': 0.0,
    'MIN_PRIOR':         1e-2,
    'VIRTUAL_LOSS':       1.0,
    'VIRTUAL_LOSS_VALUE': -1.0,
    'FORCED_PLAYOUT_K': 1,      # try 1-2; higher = more forced exploration, slower convergence per move
    'TACTICAL_CHECK_MIN_PRIOR': 0.30,   # vs. your existing MIN_PRIOR of 2e-3 -- a meaningfully bigger floor
    'TACTICAL_CAPTURE_MIN_PRIOR': 0.05,   # vs. your existing MIN_PRIOR of 2e-3 -- a meaningfully bigger floor
}

# Temperature for the first few plies of each game -- samples moves
# proportional to visit_count^(1/temperature) instead of always taking the
# single most-visited move. This just diversifies openings across the match
# (so every champion-vs-challenger pairing doesn't replay the identical
# "best" line every single game); it does NOT affect strength evaluation,
# since it only applies briefly before falling back to fully deterministic
# best_move() selection for the rest of the game.
OPENING_TEMPERATURE = 0.8
OPENING_TEMPERATURE_PLIES = 6   # ~3 moves per side; set to 0 to disable entirely

# Optional: enable tablebase-corrected leaf evaluation during the match too.
# Set to None to disable, or a path / list of paths (e.g. separate WDL/DTZ
# folders) to enable -- see mcts_core.init_tablebase for details.
SYZYGY_PATH = [
    os.path.join(BASE_DIR, "syzygy", "Syzygy345WDL"),
    os.path.join(BASE_DIR, "syzygy", "Syzygy345DTZ"),
]
MAX_TABLEBASE_PIECES = 5

OUT_PGN_PATH = os.path.join(BASE_DIR, "Arena Game PGNs", "arena_match.pgn")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────

def load_model_champion(path, channels, blocks):
    model = Network.AZNetChess(in_planes=IN_PLANES, channels=channels, blocks=blocks, action_size=ACTION_SIZE).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_challenger(path, channels, blocks):
    model = Network2.AZNetChess(in_planes=IN_PLANES, channels=channels, blocks=blocks, action_size=ACTION_SIZE).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_infer_batch_fn(model):
    def infer_batch_fn(batch_items):
        x = torch.from_numpy(np.stack(batch_items)).to(DEVICE)
        with torch.no_grad():
            logits, values = model(x)
        return logits.cpu().numpy(), values.reshape(-1).cpu().numpy()
    return infer_batch_fn

 
def select_move(mcts: MCTS, temperature: float, top_k: int = 3) -> chess.Move:
    """
    temperature <= 0:
        Deterministic — return the most visited child (mcts.best_move()).

    temperature > 0:
        Restrict to the top_k most-visited children.
        Sample one proportional to visits^(1/temperature).
        This ensures exploration but prevents low-visit moves from ever being chosen.
    """
    root = mcts.root
    if root is None or not root.children:
        raise RuntimeError("select_move called with no children -- did you run search first?")

    # Deterministic path
    if temperature <= 0:
        return mcts.best_move()

    # Top‑k restriction
    children = sorted(root.children, key=lambda c: c.visits, reverse=True)[:top_k]

    # Extract visit counts
    visits = np.array([c.visits for c in children], dtype=np.float64)

    # If somehow all visits are zero (extremely rare), fall back to uniform top‑k
    if visits.sum() == 0:
        return random.choice(children).action_taken

    # Temperature sampling: visits^(1/temperature)
    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()

    idx = np.random.choice(len(children), p=probs)
    return children[idx].action_taken

# ── Per-game state ────────────────────────────────────────────────────────

class GameState:
    def __init__(self, game_id: int, white_is_champion: bool):
        self.game_id = game_id
        self.board = chess.Board()
        self.mcts = MCTS(ARENA_ARGS)
        self.mcts.new_root(self.board)
        self.white_is_champion = white_is_champion
        self.result = None  # None while ongoing, else '1-0' / '0-1' / '1/2-1/2'
        self.ply = 0

    def champion_to_move(self) -> bool:
        return (self.board.turn == chess.WHITE) == self.white_is_champion


# ── Arena loop ────────────────────────────────────────────────────────────

def run_arena():
    if SYZYGY_PATH:
        init_tablebase(SYZYGY_PATH, MAX_TABLEBASE_PIECES)

    print(f"Device: {DEVICE}")
    print(f"Loading champion:   {CHAMPION_PATH}")
    champion = load_model_champion(CHAMPION_PATH, CHAMPION_CHANNELS, CHAMPION_BLOCKS)
    print(f"Loading challenger: {CHALLENGER_PATH}")
    challenger = load_model_challenger(CHALLENGER_PATH, CHALLENGER_CHANNELS, CHALLENGER_BLOCKS)

    infer_champion = make_infer_batch_fn(champion)
    infer_challenger = make_infer_batch_fn(challenger)

    n_games = NUM_GAMES if NUM_GAMES % 2 == 0 else NUM_GAMES + 1
    games = [GameState(i, white_is_champion=(i % 2 == 0)) for i in range(n_games)]

    start = time.time()
    round_num = 0
    while True:
        active = [g for g in games if g.result is None]
        if not active:
            break
        round_num += 1

        champ_turn = [g for g in active if g.champion_to_move()]
        chal_turn = [g for g in active if not g.champion_to_move()]

        if champ_turn:
            batch_search(
                [g.mcts for g in champ_turn], MCTS_SIMS_CHAMP, infer_champion,
                max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_S,
                min_prior=ARENA_ARGS['MIN_PRIOR'],
            )
        if chal_turn:
            batch_search(
                [g.mcts for g in chal_turn], MCTS_SIMS_CHALLENGER, infer_challenger,
                max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_S,
                min_prior=ARENA_ARGS['MIN_PRIOR'],
            )

        for g in active:
            temperature = OPENING_TEMPERATURE if g.ply < OPENING_TEMPERATURE_PLIES else 0.0
            move = select_move(g.mcts, temperature)
            g.board.push(move)
            g.mcts.advance_root(move)
            g.ply += 1

            outcome = g.board.outcome(claim_draw=True)
            if outcome is not None:
                g.result = outcome.result()
            elif g.ply >= MAX_GAME_LENGTH:
                g.result = '1/2-1/2'  # truncated -- scored as a draw, not discarded

        done = sum(1 for g in games if g.result is not None)
        print(f"[Arena] round {round_num}: {done}/{n_games} games finished "
              f"({time.time() - start:.1f}s elapsed)")

    save_pgn(games)
    print_summary(games)


# ── Output ────────────────────────────────────────────────────────────────

def save_pgn(games):
    with open(OUT_PGN_PATH, "w", encoding="utf-8") as f:
        for g in games:
            game = chess.pgn.Game()
            game.headers["Event"] = "Arena Match"
            game.headers["Round"] = str(g.game_id + 1)
            game.headers["White"] = "Champion" if g.white_is_champion else "Challenger"
            game.headers["Black"] = "Challenger" if g.white_is_champion else "Champion"
            game.headers["Result"] = g.result or "*"

            node = game
            for move in g.board.move_stack:
                node = node.add_variation(move)

            print(game, file=f, end="\n\n")

    print(f"[Arena] Saved {len(games)} games to {OUT_PGN_PATH}")


def print_summary(games):
    champ_wins = losses = draws = 0
    for g in games:
        if g.result == '1/2-1/2':
            draws += 1
        elif (g.result == '1-0') == g.white_is_champion:
            champ_wins += 1
        else:
            losses += 1

    total = len(games)
    score = (champ_wins + 0.5 * draws) / total if total else 0.0
    print(f"\nChampion vs Challenger: {champ_wins}W {losses}L {draws}D "
          f"out of {total} games (champion score {score:.3f})")


if __name__ == "__main__":
    run_arena()