"""
Self-play training loop for the chess AlphaZero bot.
Adapted from TrainingLoop.py (the Go version) with chess-specific changes:

- chess.Board instead of numpy arrays; game ends via board.outcome(claim_draw=True)
- MCTS is a generator (run_search / run_batched from MCTSChess.py)
- run_batched is used for self-play: all SELFPLAY_GAMES_PER_ITER games share
  one batched GPU call per simulation round -- this is the leaf-batching speedup
  (discussed before adding this loop) that turns 200 NPS into ~2000+ NPS.
- No board augmentations: chess move encoding is not rotationally symmetric.
- Policy targets are full pi distributions (visit counts), not single move
  indices -- KL loss, same as the Go version.
- Value targets: +1.0 winner's perspective, -1.0 loser's, 0.0 draw.
- Arena: both models maintain their own MCTS tree over the shared board,
  advancing roots after every move regardless of whose turn it was.
"""

# https://lichess.org/paste
'''
virtual loss?
make cpu and gpu run at same time
make cpu have 100% utilization
do a time check analysis with that one library thing fro go i used

visualize model


make synthetic endgame data and mix with human data

what about also giving an exploration value for checks, captures, attacks, and forced moves


'''

import os
import csv
import copy
import time
import random
import math
import warnings
import numpy as np
import chess; import chess.svg; import chess.pgn
import torch; import torch.nn.functional as F; from torch.nn.utils import clip_grad_norm_

from colorama import Fore
from datetime import datetime
try:
    import winsound  # Windows only: beep when a checkpoint is saved
except ImportError:
    winsound = None

from Network2 import AZNetChess, init_weights
from Encoder import IN_PLANES, ACTION_SIZE, encode_board, move_to_index
from mcts_core import (
    MCTS, batch_search, softmax_legal, cpuct_from_sims,
    init_tablebase, pruned_action_probs, probe_tablebase_value,
)
# from Tablebase import SYZYGY_PATH, MAX_TABLEBASE_PIECES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/, syzygy/, Human Data/ ... live next to this script

init_tablebase([
    os.path.join(BASE_DIR, "syzygy", "Syzygy345WDL"),
    os.path.join(BASE_DIR, "syzygy", "Syzygy345DTZ"),
], max_pieces=5)


# ── Architecture ──────────────────────────────────────────────────────────────
CHANNELS = 64
BLOCKS   = 8

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR                = 2e-4  # ---------------------
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_WEIGHT = 1.0
MAX_NORM          = 1.0

# ── Self-play ─────────────────────────────────────────────────────────────────
SELFPLAY_GAMES_IN_PARALELL = 32      # parallel games per epoch (all share one GPU call/sim)
TOTAL_GAMES_PER_EPOCH = 101
AVG_POSITIONS_PER_GAME  = 98    # adjust after first few games (now its less  with game pool)
FRACTION_DISCARDED = 0.10        # percent of observed games that reached max-length and were discarded
MCTS_SIMS               = 900  # simulations per move
ADJUST_SIMS             = False   # scale MCTS_SIMs in self play depending on how many pieces left
SIM_SCHEDULE = [
    (int(MCTS_SIMS),  32),   # opening: 32-23 pieces
    (int(MCTS_SIMS * 1),  22),   # middlegame: 22-13 pieces
    (int(MCTS_SIMS * 1.3),  12),   # endgame: 12-9 pieces
    (int(MCTS_SIMS * 2),  8),   # endgame: 8-5 pieces    
    (int(MCTS_SIMS * 4),   4),   # critical endgame: 4 pieces and below
]

MAX_GAME_LENGTH         = 200    # half-moves; safety cap 

# Temperature: (temp, up_to_half_move_number)
# High temp early = exploration; greedy late = decisive play
TEMPERATURE_SCHEDULE = [(1.0, 14), (0.8, 20), (0.5, 30), (0.0, 9999)]

DIRICHLET_ALPHA   = 0.3          # AlphaZero paper value for chess
DIRICHLET_EPSILON = 0.25

MAX_BATCH_SIZE = 256
MAX_WAIT_S = 0.018
# ── Arena ─────────────────────────────────────────────────────────────────────
ARENA_GAMES        = 20
ARENA_MCTS_SIMS    = MCTS_SIMS
WINRATE_THRESHOLD  = 0.55        # challenger win rate to replace champion
ARENA_INTERVAL     = 2           # run arena every N epochs
TRAINING_ALLOWANCE = 999          # epochs without improvement before reset to champion

ARENA_TEMPERATURE_SCHEDULE = [(0.35, 4),(0.2, 6), (0.0, 9999)]

# ── Loop / saving ─────────────────────────────────────────────────────────────
TRAIN_STEPS    = 999
SAVE_INTERVAL  = 1 # which checkpoints to save
KEEP_INTERVAL = 1 # which to keep when saving new ones
TRAIN_INTERVAL = 1  # train network every N epochs

# ── Replay buffer ─────────────────────────────────────────────────────────────
EPOCHS_TO_KEEP = TRAIN_INTERVAL * 3 * (1 - FRACTION_DISCARDED) # account for maxlen games
MAX_BUFFER_SIZE = int((TOTAL_GAMES_PER_EPOCH + SELFPLAY_GAMES_IN_PARALELL - 1) * AVG_POSITIONS_PER_GAME * EPOCHS_TO_KEEP) # 47_241
BATCH_SIZE      = 256
TRAINING_EPOCHS = 2              # effective passes over buffer per train step

# ── Paths ─────────────────────────────────────────────────────────────────────
HUMAN_PRETRAINED_PATH = os.path.join(BASE_DIR, "Models", "model_human_pretrained_7-7.pt")
MODEL_DIR        = os.path.join(BASE_DIR, "Models")
CURR_EPOCH_PATH  = os.path.join(BASE_DIR, "paths", "curr_epoch.txt")
CHAMP_EPOCH_PATH = os.path.join(BASE_DIR, "paths", "champion_epoch.txt")
SELFPLAY_PGN_DIR = os.path.join(BASE_DIR, "SelfPlay Game PGNs")
ARENA_PGN_DIR    = os.path.join(BASE_DIR, "Arena Game PGNs")
EPOCH_STATS_PATH = os.path.join(BASE_DIR, "paths", "epoch_stats.csv")

# ── MCTS args dicts ───────────────────────────────────────────────────────────
# FORCED_PLAYOUT_K / TACTICAL_CHECK_MIN_PRIOR / TACTICAL_CAPTURE_MIN_PRIOR
# are copied unchanged from AnalyzeUI2.py's ARGS, not rescaled for the lower
# self-play/arena sim count (750 vs. the thousands AnalyzeUI accumulates
# interactively). The forced-playout target is forced_k * sqrt(prior *
# parent_visits), so as a FRACTION of the sim budget (target / N =
# forced_k * sqrt(prior / N)) it already grows, not shrinks, as N drops --
# the same constant enforces a *larger* relative exploration share at lower
# sim counts automatically, no manual rescaling needed. If self-play ends
# up feeling tactically-skewed or slow at 750 sims once you've watched a
# few epochs, the knob to turn is lowering FORCED_PLAYOUT_K specifically
# here, not the tactical floors.
SELFPLAY_ARGS = {
    'CPUCT':             1.8,
    'DIRICHLET_ALPHA':   DIRICHLET_ALPHA,
    'DIRICHLET_EPSILON': DIRICHLET_EPSILON,
    'MIN_PRIOR':         1e-3,
    'VIRTUAL_LOSS':      1.0,
    'VIRTUAL_LOSS_VALUE': -1.0,
    'FORCED_PLAYOUT_K':            0.4,
    'TACTICAL_CHECK_MIN_PRIOR':    0.45,
    'TACTICAL_CAPTURE_MIN_PRIOR':  0.2,
    # 'TACTICAL_FEW_REPLIES_MIN_PRIOR': 0.65,
    'TACTICAL_ONE_REPLY_MIN_PRIOR': 0.5,   # move leaves opponent exactly 1 legal reply
    'TACTICAL_TWO_REPLY_MIN_PRIOR': 0.4,  # move leaves opponent exactly 2 legal replies
    'FORCED_PLAYOUT_MIN_ABSOLUTE': 0,   # every child guaranteed this many visits, regardless of prior
    
}

ARENA_ARGS = {
    'CPUCT':             1.8,
    'DIRICHLET_ALPHA':   0.0,
    'DIRICHLET_EPSILON': 0.0,
    'MIN_PRIOR':         1e-3,
    'VIRTUAL_LOSS':      1.0,
    'VIRTUAL_LOSS_VALUE': -1.0,
    'FORCED_PLAYOUT_K':            0.4,
    'TACTICAL_CHECK_MIN_PRIOR':    0.45,
    'TACTICAL_CAPTURE_MIN_PRIOR':  0.2,
    # 'TACTICAL_FEW_REPLIES_MIN_PRIOR': 0.65,
    'TACTICAL_ONE_REPLY_MIN_PRIOR': 0.5,   # move leaves opponent exactly 1 legal reply
    'TACTICAL_TWO_REPLY_MIN_PRIOR': 0.4,  # move leaves opponent exactly 2 legal replies
    'FORCED_PLAYOUT_MIN_ABSOLUTE': 0,   # every child guaranteed this many visits, regardless of prior
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def win_percent(root_value):
    return 100 / (1 + math.exp(-4 * root_value))

def mcts_confidence(root):
    # Extract visit counts of all children
    visits = [child.visits for child in root.children]

    # If fewer than 2 moves exist, confidence is absolute
    if len(visits) < 2:
        return 1.0

    # Sort descending: best first, second-best second
    visits.sort(reverse=True)
    N1, N2 = visits[0], visits[1]

    # Avoid division by zero
    if N2 == 0:
        return 1.0

    # Ratio of best to second-best
    ratio = N1 / N2

    # Logistic confidence curve centered at ratio = 1
    return 1 / (1 + math.exp(-1 * (ratio - 1)))

def eval_points(x):
    sign = 1 if x >= 0 else -1
    ax = abs(x)
    return sign * (4.4 * (math.exp(ax) - 1))

def get_temperature(half_move_number: int, schedule) -> float:
    for temp, upto in schedule:
        if half_move_number <= upto:
            return temp
    return schedule[-1][0]


def apply_temperature(pi: np.ndarray, tau: float) -> np.ndarray:
    pi = np.array(pi, dtype=np.float64)
    if tau < 1e-8:
        out = np.zeros_like(pi)
        out[np.argmax(pi)] = 1.0
        return out
    log_pi = np.log(np.clip(pi, 1e-12, 1.0))
    scaled = np.exp(log_pi / tau)
    return scaled / scaled.sum()


def sample_move_from_pi(pi_t, mcts, board):
    sampled_idx = int(np.random.choice(ACTION_SIZE, p=pi_t))
    for child in mcts.root.children:
        if move_to_index(child.action_taken, board) == sampled_idx:
            return child.action_taken
    return mcts.best_move()   # was best_move(mcts.root)

def make_infer_fn(model, device):
    """Single-board inference -- used by run_search in arena games."""
    def infer_fn(encoded):
        x = torch.from_numpy(encoded).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = model(x)
        return logits.squeeze(0).cpu().numpy(), float(value.item())
    return infer_fn


def make_infer_batch_fn(model, device):
    def infer_batch_fn(encoded_boards):
        x = torch.from_numpy(np.stack(encoded_boards)).to(device)
        with torch.no_grad():
            logits, values = model(x)
        return logits.cpu().numpy(), values.cpu().numpy()  # array not .tolist()
    return infer_batch_fn

# def make_infer_batch_fn(model, device):
#     call_count = [0]
#     total_leaves = [0]
#     last_print = [time.time()]

#     def infer_batch_fn(encoded_boards):
#         batch_size = len(encoded_boards)
#         call_count[0] += 1
#         total_leaves[0] += batch_size

#         # Print rolling average every 5 seconds
#         now = time.time()
#         if now - last_print[0] >= 5.0:
#             avg = total_leaves[0] / call_count[0]
#             print(f"  [GPU] avg batch size: {avg:.1f} leaves/call "
#                   f"({call_count[0]} calls, {total_leaves[0]} total leaves)")
#             call_count[0] = 0
#             total_leaves[0] = 0
#             last_print[0] = now

#         x = torch.from_numpy(np.stack(encoded_boards)).to(device)
#         with torch.no_grad():
#             logits, values = model(x)
#         return logits.cpu().numpy(), values.cpu().numpy()

#     return infer_batch_fn

def checkpoint_path(epoch: int) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"chess_selfplay4_epoch_{epoch}.pt")


def save_checkpoint(epoch, model, opt, replay_buffer):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'replay_buffer':        replay_buffer,
        'epoch':                epoch,
    }, checkpoint_path(epoch))
    with open(CURR_EPOCH_PATH, "w") as f:
        f.write(str(epoch))
    print(f"\t\t\t\t{Fore.YELLOW}\033[1mCheckpoint saved at epoch {epoch}\033[0m{Fore.RESET}")
    cleanup_old_checkpoints(epoch)

    if winsound:
        winsound.Beep(1000, 500) # sound
        winsound.Beep(500, 500) # sound
        winsound.Beep(1500, 500) # sound


def load_checkpoint(path, model, opt=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if opt is not None and 'optimizer_state_dict' in ckpt:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('replay_buffer', []), ckpt.get('epoch', 0)


def cleanup_old_checkpoints(current_epoch, keep_every=KEEP_INTERVAL):
    """Delete checkpoint files that aren't multiples of keep_every,
    keeping the 2 most recent regardless so you always have a fallback."""
    if not os.path.exists(MODEL_DIR):
        return

    # Find all selfplay checkpoints
    files = {}
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("chess_selfplay4_epoch_") and fname.endswith(".pt"):
            try:
                epoch = int(fname.replace("chess_selfplay4_epoch_", "").replace(".pt", ""))
                files[epoch] = os.path.join(MODEL_DIR, fname)
            except ValueError:
                continue

    sorted_epochs = sorted(files.keys())

    # Always keep the 2 most recent (current + previous as fallback)
    always_keep = set(sorted_epochs[-2:]) if len(sorted_epochs) >= 2 else set(sorted_epochs)

    for epoch in sorted_epochs:
        if epoch == current_epoch:
            continue
        if epoch in always_keep:
            continue
        if epoch % keep_every == 0:
            continue
        os.remove(files[epoch])
        print(f"  Deleted checkpoint epoch {epoch}")


def new_model():
    m = AZNetChess(
        in_planes=IN_PLANES, channels=CHANNELS,
        blocks=BLOCKS, action_size=ACTION_SIZE,
    ).to(DEVICE)
    m.apply(init_weights)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# PGN saving / epoch stats logging
# ─────────────────────────────────────────────────────────────────────────────

def board_to_pgn_string(board: chess.Board, headers: dict) -> str:
    """Replays board.move_stack into a chess.pgn.Game and returns it as a
    PGN string, ready to append to a file. `headers` is a dict of PGN
    header tags (Event, White, Black, Result, etc.)."""
    game = chess.pgn.Game()
    for key, value in headers.items():
        game.headers[key] = str(value)

    node = game
    for move in board.move_stack:
        node = node.add_variation(move)

    game.headers["Result"] = board.result(claim_draw=True)
    return str(game)


def append_pgn(path: str, pgn_string: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(pgn_string + "\n\n")


def log_epoch_stats(epoch: int, n_games: int, white_wins: int, black_wins: int,
                     draws: int, discarded: int, avg_half_moves: float, elapsed: float):
    """Appends one row to a running CSV. Creates the file with a header
    row if it doesn't exist yet."""
    os.makedirs(os.path.dirname(EPOCH_STATS_PATH), exist_ok=True)
    file_exists = os.path.exists(EPOCH_STATS_PATH)

    with open(EPOCH_STATS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "n_games", "white_wins", "black_wins", "draws",
                "discarded_maxlen", "white_wr", "black_wr", "draw_rate",
                "avg_half_moves", "elapsed_s"
            ])
        writer.writerow([
            epoch, n_games, white_wins, black_wins, draws, discarded,
            f"{white_wins/n_games:.3f}" if n_games else 0,
            f"{black_wins/n_games:.3f}" if n_games else 0,
            f"{draws/n_games:.3f}" if n_games else 0,
            f"{avg_half_moves:.1f}", f"{elapsed:.0f}",
        ])

def update_constants(n_games, discarded, avg_half_moves):
    global FRACTION_DISCARDED, AVG_POSITIONS_PER_GAME, MAX_BUFFER_SIZE, EPOCHS_TO_KEEP, TRAIN_INTERVAL

    if n_games <= 0:
        return

    new_fraction_discarded = discarded / n_games

    # Smooth the estimates so they don't jump too hard from one epoch
    FRACTION_DISCARDED = max(0.0, min(0.95, 0.6 * FRACTION_DISCARDED + 0.4 * new_fraction_discarded))
    AVG_POSITIONS_PER_GAME = max(1, int(round(0.6 * AVG_POSITIONS_PER_GAME + 0.4 * avg_half_moves)))

    EPOCHS_TO_KEEP = max(1, TRAIN_INTERVAL * 3 * (1 - FRACTION_DISCARDED))
    MAX_BUFFER_SIZE = max(
        BATCH_SIZE * 4,
        int((TOTAL_GAMES_PER_EPOCH + SELFPLAY_GAMES_IN_PARALELL - 1) *
            AVG_POSITIONS_PER_GAME * EPOCHS_TO_KEEP)
    )

    print(f"[constants] discarded={FRACTION_DISCARDED:.3f}, "
          f"avg_positions={AVG_POSITIONS_PER_GAME}, "
          f"epochs_to_keep={EPOCHS_TO_KEEP}, "
          f"max_buffer={MAX_BUFFER_SIZE}")
# ─────────────────────────────────────────────────────────────────────────────
# Self-play  (run_batched: all games share one GPU callper simulation leaf)
# ─────────────────────────────────────────────────────────────────────────────

def self_play_games(model, pool_size: int, total_games: int, args: dict, epoch: int, verbose=True):
    """
    Persistent-pool self-play: always keep pool_size slots running simultaneously.

    When a game ends its slot immediately restarts with a fresh game instead
    of sitting idle waiting for the slowest game in the batch. Runs until
    total_games games have completed.

    pool_size  -- number of concurrent games (e.g. 12)
    total_games -- how many completed games to collect before stopping (e.g. 24)

    The speedup is largest when game-length variance is high (which it is:
    games regularly range from 15 to 200 half-moves in the same epoch).
    """
    model.eval()
    infer_batch_fn = make_infer_batch_fn(model, DEVICE)

    boards    = [chess.Board() for _ in range(pool_size)]
    mcts_list = [MCTS(args)    for _ in range(pool_size)]
    histories = [[]            for _ in range(pool_size)]
    move_nums = [1]            * pool_size
    game_ids  = list(range(pool_size))   # tracks which "game number" each slot is on
    active    = [True]         * pool_size

    for mcts, board in zip(mcts_list, boards):
        mcts.new_root(board)

    os.makedirs(SELFPLAY_PGN_DIR, exist_ok=True)

    all_data     = []
    n_completed  = 0
    next_game_id = pool_size      # ID counter for games beyond the initial batch
    game_start   = time.time()

    # ── Stats accumulators for the epoch-level CSV log ────────────────────
    stat_white_wins = 0
    stat_black_wins = 0
    stat_draws      = 0
    stat_discarded  = 0
    stat_half_move_sum = 0

    while any(active):
        live_idx  = [i for i in range(pool_size) if active[i]]
        live_mcts = [mcts_list[i] for i in live_idx]
        
        batch_results = batch_search(
            live_mcts, MCTS_SIMS, infer_batch_fn,
            max_batch_size=MAX_BATCH_SIZE,   # ~5 leaves/game with 12 games
            max_wait_s=MAX_WAIT_S,
            min_prior=SELFPLAY_ARGS['MIN_PRIOR'],
        )
        

        for k, i in enumerate(live_idx):
            board = boards[i]
            mcts  = mcts_list[i]
            pi, _ = batch_results[k]

            # Training target uses pruned visits (strips out FORCED_PLAYOUT_K's
            # guaranteed-minimum visits on non-best children) so the network
            # isn't taught a generic bias toward checks/captures just because
            # the search was forced to double-check them. The move actually
            # PLAYED still samples from the full, unpruned `pi` below -- it
            # should keep benefiting from whatever forced exploration found.
            pi_target = pruned_action_probs(mcts)
            histories[i].append((encode_board(board), pi_target, board.turn))

            # Sample move with temperature
            tau  = get_temperature(move_nums[i], TEMPERATURE_SCHEDULE)
            pi_t = apply_temperature(pi, tau)
            move = sample_move_from_pi(pi_t, mcts, board)

            board.push(move)
            mcts.advance_root(move)
            move_nums[i] += 1

            outcome = board.outcome(claim_draw=True)
            endgame_tablebase = chess.popcount(board.occupied) <= 5
            if outcome is not None or move_nums[i] > MAX_GAME_LENGTH or endgame_tablebase:
                n_completed += 1
                gid = game_ids[i]

                resolved_by_tablebase = False
                winner_color = outcome.winner if outcome is not None else None
                if outcome is None or endgame_tablebase:
                    tb_value = probe_tablebase_value(board)
                    if tb_value is not None:
                        resolved_by_tablebase = True
                        if tb_value > 0.0:
                            winner_color = board.turn
                        elif tb_value < 0.0:
                            winner_color = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
                        else:
                            winner_color = None

                if outcome is None and not resolved_by_tablebase:
                    result_str = "maxlen"
                elif outcome is not None and outcome.winner is None:
                    result_str = "draw"
                elif resolved_by_tablebase and winner_color is None:
                    result_str = "tablebase-draw"
                elif resolved_by_tablebase:
                    result_str = "tablebase-win"
                else:
                    result_str = "white" if outcome.winner == chess.WHITE else "black"

                # ── Tally stats for this game ──────────────────────────
                stat_half_move_sum += (move_nums[i] - 1)
                if outcome is None and not resolved_by_tablebase:
                    stat_discarded += 1
                elif winner_color is None:
                    stat_draws += 1
                elif winner_color == chess.WHITE:
                    stat_white_wins += 1
                else:
                    stat_black_wins += 1

                # ── Save full game as PGN (appended to this epoch's file) ──
                pgn_path = os.path.join(SELFPLAY_PGN_DIR, f"epoch{epoch:04d}_games.pgn")
                headers = {
                    "Event":       f"SelfPlay Epoch {epoch}",
                    "White":       "Bot",
                    "Black":       "Bot",
                    "GameID":      gid,
                    "PlyCount":    move_nums[i] - 1,
                    "Termination": result_str,
                }
                append_pgn(pgn_path, board_to_pgn_string(board, headers))

                if verbose:
                    if outcome is None and not resolved_by_tablebase:
                        result = f"{Fore.RED}max-length draw{Fore.RESET}"
                    elif winner_color is None:
                        result = f"{Fore.LIGHTRED_EX}draw{Fore.RESET}"
                    elif resolved_by_tablebase:
                        result = f"tablebase win for {('white' if winner_color == chess.WHITE else 'black')}"
                    else:
                        result = "white wins" if winner_color == chess.WHITE else "black wins"
                    print(f"  game {gid} | {move_nums[i]-1} half-moves | {result} | "
                          f"{time.time()-game_start:.0f}s")

                # Collect training data -- use tablebase when available
                if outcome is not None or resolved_by_tablebase:
                    for encoded, pi_h, color in histories[i]:
                        z = (0.0 if winner_color is None
                             else 1.0 if color == winner_color
                             else -1.0)
                        all_data.append((encoded, pi_h, z))
                elif verbose:
                    print(f"\t\t\t game {gid} discarded (max-length)")

                # ── Restart or retire this slot ───────────────────────────
                if n_completed < total_games:
                    # Still need more games -- restart immediately
                    boards[i]    = chess.Board()
                    mcts_list[i] = MCTS(args)
                    mcts_list[i].new_root(boards[i])
                    histories[i] = []
                    move_nums[i] = 1
                    game_ids[i]  = next_game_id
                    next_game_id += 1
                else:
                    # Target reached -- retire this slot
                    active[i] = False

    avg_half_moves = stat_half_move_sum / n_completed if n_completed else 0.0
    log_epoch_stats(
        epoch, n_games=n_completed,
        white_wins=stat_white_wins, black_wins=stat_black_wins,
        draws=stat_draws, discarded=stat_discarded,
        avg_half_moves=avg_half_moves, elapsed=time.time() - game_start
    )

    update_constants(
        n_games=n_completed, discarded=stat_discarded, avg_half_moves=avg_half_moves
    )
    return all_data


def arena_match(champion, challenger, champion_epoch, challenger_epoch, verbose=True):
    """
    Play ARENA_GAMES games in parallel using batched MCTS inference.

    All games run simultaneously. Each round:
      1. Split live games by whose turn it is (champion or challenger).
      2. run_batched through champion's network for champion-to-move games.
      3. run_batched through challenger's network for challenger-to-move games.
      4. Make moves, advance both trees in every game, check outcomes.

    Both models maintain their own MCTS tree per game and call advance_root
    after every move (including the opponent's), so tree reuse stays valid
    exactly as in the sequential version.

    Colors are assigned upfront: even-indexed games give champion White,
    odd-indexed games give champion Black -- same alternation as before,
    just all at once rather than one game at a time.

    champion_epoch / challenger_epoch are used only for PGN headers/filename
    so saved arena games are identifiable later.
    """
    champion.eval()
    challenger.eval()

    champ_infer_batch = make_infer_batch_fn(champion, DEVICE)
    chal_infer_batch  = make_infer_batch_fn(challenger, DEVICE)

    # Assign colors upfront
    champion_colors = [chess.WHITE if g % 2 == 0 else chess.BLACK
                       for g in range(ARENA_GAMES)]

    boards          = [chess.Board()   for _ in range(ARENA_GAMES)]
    champ_mcts_list = [MCTS(ARENA_ARGS) for _ in range(ARENA_GAMES)]
    chal_mcts_list  = [MCTS(ARENA_ARGS) for _ in range(ARENA_GAMES)]

    for g in range(ARENA_GAMES):
        champ_mcts_list[g].new_root(boards[g])
        chal_mcts_list[g].new_root(boards[g])

    half_moves     = [1]    * ARENA_GAMES
    alive          = [True] * ARENA_GAMES
    outcomes       = [None] * ARENA_GAMES

    champion_wins   = 0
    challenger_wins = 0
    draws           = 0
    early_stopped   = False

    os.makedirs(ARENA_PGN_DIR, exist_ok=True)
    arena_pgn_path = os.path.join(
        ARENA_PGN_DIR, f"arena_champ{champion_epoch}_vs_chal{challenger_epoch}.pgn"
    )

    while any(alive) and not early_stopped:
        live_idx = [i for i in range(ARENA_GAMES) if alive[i]]

        # Split by whose turn it is this half-move
        champ_turn_idx = [i for i in live_idx
                          if boards[i].turn == champion_colors[i]]
        chal_turn_idx  = [i for i in live_idx
                          if boards[i].turn != champion_colors[i]]

        # Batched MCTS for champion's games
        champ_results = {}
        if champ_turn_idx:
            results = batch_search(
                [champ_mcts_list[i] for i in champ_turn_idx],
                ARENA_MCTS_SIMS, champ_infer_batch,
                max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_S,
                min_prior=ARENA_ARGS['MIN_PRIOR'],
            )
            for k, i in enumerate(champ_turn_idx):
                champ_results[i] = results[k]

        # Batched MCTS for challenger's games
        chal_results = {}
        if chal_turn_idx:
            results = batch_search(
                [chal_mcts_list[i] for i in chal_turn_idx],
                ARENA_MCTS_SIMS, chal_infer_batch,
                max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_S,
                min_prior=ARENA_ARGS['MIN_PRIOR'],
            )
            for k, i in enumerate(chal_turn_idx):
                chal_results[i] = results[k]

        # Make moves for all live games
        for i in live_idx:
            board = boards[i]

            if i in champ_results:
                pi, _       = champ_results[i]
                active_mcts = champ_mcts_list[i]
            else:
                pi, _       = chal_results[i]
                active_mcts = chal_mcts_list[i]

            tau  = get_temperature(half_moves[i], ARENA_TEMPERATURE_SCHEDULE)
            pi_t = apply_temperature(pi, tau)
            move = sample_move_from_pi(pi_t, active_mcts, board)

            board.push(move)
            champ_mcts_list[i].advance_root(move)
            chal_mcts_list[i].advance_root(move)
            half_moves[i] += 1

            outcome = board.outcome(claim_draw=True)
            if outcome is not None or half_moves[i] > MAX_GAME_LENGTH * 2:
                alive[i]    = False
                outcomes[i] = outcome

                if outcome is None or outcome.winner is None:
                    draws += 1
                    result_str = "Draw"
                elif outcome.winner == champion_colors[i]:
                    champion_wins += 1
                    result_str = "Champion wins"
                else:
                    challenger_wins += 1
                    result_str = "Challenger wins"

                # ── Save this arena game as PGN, appended to the run's file ──
                if champion_colors[i] == chess.WHITE:
                    white_label, black_label = f"Champion_{champion_epoch}", f"Challenger_{challenger_epoch}"
                else:
                    white_label, black_label = f"Challenger_{challenger_epoch}", f"Champion_{champion_epoch}"

                headers = {
                    "Event":       f"Arena Champ{champion_epoch} vs Chal{challenger_epoch}",
                    "White":       white_label,
                    "Black":       black_label,
                    "Round":       i + 1,
                    "PlyCount":    half_moves[i] - 1,
                    "Termination": result_str,
                }
                append_pgn(arena_pgn_path, board_to_pgn_string(board, headers))

                if verbose:
                    champ_color_str = "White" if champion_colors[i] == chess.WHITE else "Black"
                    print(f"  Game {i+1}: {result_str} ({half_moves[i]-1} half-moves) | "
                          f"Champ {champ_color_str} | "
                          f"Record C{champion_wins}-Ch{challenger_wins}-D{draws}")

                # Early stop: same logic as sequential version
                remaining      = sum(alive)   # games still running
                total_decisive = champion_wins + challenger_wins

                if total_decisive + remaining > 0:
                    best_chal_rate  = (challenger_wins + remaining) / (total_decisive + remaining)
                    best_champ_rate =  challenger_wins               / (total_decisive + remaining)

                    if best_chal_rate < WINRATE_THRESHOLD:
                        print(f"  Early stop: challenger cannot reach "
                              f"{WINRATE_THRESHOLD:.0%} even winning all remaining.")
                        early_stopped = True
                        break
                    if best_champ_rate >= WINRATE_THRESHOLD:
                        print(f"  Early stop: challenger has already secured "
                              f"{WINRATE_THRESHOLD:.0%} win rate.")
                        early_stopped = True
                        break

    total    = champion_wins + challenger_wins
    win_rate = challenger_wins / total if total > 0 else 0.0
    print(f"\nChampion: {champion_wins} | Challenger: {challenger_wins} | "
          f"Draws: {draws} | Challenger WR: {win_rate:.2f}")

    return win_rate >= WINRATE_THRESHOLD, [champion_wins, challenger_wins, draws]



def champion_color_str(color):
    return "White" if color == chess.WHITE else "Black"


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_on_buffer(model, opt, scaler, replay_buffer):
    """
    Sample TRAINING_EPOCHS effective passes over the replay buffer.
    Policy loss: KL divergence against visit-count distribution (same as Go).
    Value loss:  MSE.
    """
    model.train()
    n_steps   = int(len(replay_buffer) / BATCH_SIZE * TRAINING_EPOCHS)
    log_every = max(1, n_steps // 10)

    for step in range(n_steps):
        batch     = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
        states    = torch.tensor(np.array([s for s, _, _ in batch]),
                                 dtype=torch.float32, device=DEVICE)
        target_pi = torch.tensor(np.array([p for _, p, _ in batch]),
                                 dtype=torch.float32, device=DEVICE)
        target_v  = torch.tensor(np.array([z for _, _, z in batch]),
                                 dtype=torch.float32, device=DEVICE)

        with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
            pred_pi, pred_v = model(states)
            log_probs    = F.log_softmax(pred_pi, dim=1)
            policy_loss  = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss   = F.mse_loss(pred_v, target_v)
            loss         = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        opt.zero_grad(set_to_none=True)

        if not torch.isfinite(loss) or loss == 0:
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_norm_(model.parameters(), MAX_NORM)
        try:
            scaler.step(opt)
            scaler.update()
        except AssertionError:
            pass

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  [{step+1}/{n_steps}] "
                  f"total={loss.item():.3f}  "
                  f"policy={policy_loss.item():.3f}  "
                  f"value={value_loss.item():.3f}")


    with torch.no_grad():
        sample = random.sample(replay_buffer, min(1000, len(replay_buffer)))
        s = torch.tensor(np.array([x[0] for x in sample]), dtype=torch.float32, device=DEVICE)
        z = np.array([x[2] for x in sample])
        _, pred_v = model(s)
        pred = pred_v.cpu().numpy()

        draws  = pred[np.abs(z) < 0.1]
        wins   = pred[z > 0.5]
        losses = pred[z < -0.5]

        print(f"  value predictions by outcome:")
        print(f"    draws  (n={len(draws):,}): mean={draws.mean():+.3f}  std={draws.std():.3f}")
        print(f"    wins   (n={len(wins):,}): mean={wins.mean():+.3f}  std={wins.std():.3f}")
        print(f"    losses (n={len(losses):,}): mean={losses.mean():+.3f}  std={losses.std():.3f}")
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def train_loop():
    for path in [CURR_EPOCH_PATH, CHAMP_EPOCH_PATH]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    time_tracker   = [time.time()]
    model          = new_model()
    champion       = new_model()
    opt            = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler         = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))
    replay_buffer  = []

    # ── Load champion ─────────────────────────────────────────
    champion_epoch = 0
    if os.path.exists(CHAMP_EPOCH_PATH):
        with open(CHAMP_EPOCH_PATH) as f:
            champion_epoch = int(f.read().strip())
    champ_path = checkpoint_path(champion_epoch)
    if os.path.exists(champ_path):
        print(f"Loading champion: {champ_path}")
        load_checkpoint(champ_path, champion)
    elif os.path.exists(HUMAN_PRETRAINED_PATH):
        print(f"No selfplay champion found, loading human pretrained weights.")
        ckpt = torch.load(HUMAN_PRETRAINED_PATH, map_location=DEVICE, weights_only=False)
        champion.load_state_dict(ckpt['model_state_dict'])
    else:
        print("No champion or pretrained weights found, starting random.")

    # ── Load current model ────────────────────────────────────
    last_epoch = 0
    if os.path.exists(CURR_EPOCH_PATH):
        with open(CURR_EPOCH_PATH) as f:
            last_epoch = int(f.read().strip())
    curr_path = checkpoint_path(last_epoch)
    if os.path.exists(curr_path):
        print(f"Loading current: {curr_path}")
        replay_buffer, _ = load_checkpoint(curr_path, model, opt)
    else:
        model = copy.deepcopy(champion).to(DEVICE)
        print("No current checkpoint, starting from champion weights. Replay buffer not loaded.")

    print(f"\nDevice: {DEVICE}")
    print(datetime.now().strftime("%m/%d/%y %I:%M %p"))
    print(f"Buffer: {len(replay_buffer):,} positions. (Max of {MAX_BUFFER_SIZE})")
    print(f"Cpuct: {SELFPLAY_ARGS['CPUCT']}")
    print(f"Selfplay games in paralell: {SELFPLAY_GAMES_IN_PARALELL}")
    print(f"Total selfplay games: {(TOTAL_GAMES_PER_EPOCH + SELFPLAY_GAMES_IN_PARALELL - 1) * TRAIN_INTERVAL}")
    print(f"Arena Interval: {ARENA_INTERVAL}")
    print(f"Arena games: {ARENA_GAMES}")
    print(f"Max batch size: {MAX_BATCH_SIZE}")
    print(f"Max wait time: {MAX_WAIT_S}")

    epoch = last_epoch + 1
    
    # replay_buffer = [] # temporary fix to clear the buffer at the start of training loop

    while epoch <= last_epoch + TRAIN_STEPS:
        print(f"\n{Fore.GREEN}Epoch {epoch}/{last_epoch + TRAIN_STEPS} | "
              f"buffer: {len(replay_buffer):,}{Fore.RESET}")

        # ── Self-play ─────────────────────────────────────────
        sp_start = time.time()
        print(f"Self-play ({TOTAL_GAMES_PER_EPOCH + SELFPLAY_GAMES_IN_PARALELL - 1} games, {MCTS_SIMS} sims/move, batched)...")
        data = self_play_games(champion, SELFPLAY_GAMES_IN_PARALELL, TOTAL_GAMES_PER_EPOCH, SELFPLAY_ARGS, epoch, verbose=True)
        replay_buffer.extend(data)
        if len(replay_buffer) > MAX_BUFFER_SIZE:
            replay_buffer = replay_buffer[-MAX_BUFFER_SIZE:]
        print(f"Self-play: +{len(data):,} positions | total {len(replay_buffer):,} | {time.time()-sp_start:.0f}s")

        new_training = len(replay_buffer) < int(MAX_BUFFER_SIZE * 0.7)
        # ── Save checkpoint ───────────────────────────────────
        if epoch % SAVE_INTERVAL == 0:
            save_checkpoint(epoch, model, opt, replay_buffer)

        # ── Train on buffer ───────────────────────────────────
        if epoch % TRAIN_INTERVAL == 0 and len(replay_buffer) >= BATCH_SIZE and not new_training:
            print("\nStarting training...")
            t_start = time.time()
            train_on_buffer(model, opt, scaler, replay_buffer)
            print(f"Training took: {time.time()-t_start:.0f}s")

            # resave after train
            if epoch % SAVE_INTERVAL == 0:
                save_checkpoint(epoch, model, opt, replay_buffer)

        # ── Arena ─────────────────────────────────────────────
        if epoch % ARENA_INTERVAL == 0 and not new_training:
            print(f"\n{Fore.CYAN}=== Arena: champion (epoch {champion_epoch}) "
                  f"vs challenger (epoch {epoch}) ==={Fore.RESET}")
            a_start = time.time()
            challenger_won, record = arena_match(champion, model, champion_epoch, epoch, verbose=True)

            print(f"\n\033[1mCHALLENGER "
                  f"{Fore.GREEN+'PROMOTED'+Fore.RESET if challenger_won else Fore.RED+'NOT PROMOTED'+Fore.RESET}"
                  f"\033[0m")

            if challenger_won:
                with open(CHAMP_EPOCH_PATH, "w") as f:
                    f.write(str(epoch))
                champion       = copy.deepcopy(model).to(DEVICE)
                champion_epoch = epoch

            if epoch - champion_epoch >= TRAINING_ALLOWANCE:
                print(f"{Fore.RED}\033[1mNo improvement in {TRAINING_ALLOWANCE} epochs. "
                      f"Resetting to champion.\033[0m{Fore.RESET}")
                model  = copy.deepcopy(champion).to(DEVICE)
                opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

            print(f"Arena took: {time.time()-a_start:.0f}s")

            # resave after arena
            if epoch % SAVE_INTERVAL == 0:
                save_checkpoint(epoch, model, opt, replay_buffer)

        # ── Timing ───────────────────────────────────────────
        time_tracker.append(time.time())
        avg = (time_tracker[-1] - time_tracker[0]) / (len(time_tracker) - 1)
        now = datetime.now().strftime("%I:%M %p")
        print(f"{Fore.BLUE}epoch: {time_tracker[-1]-time_tracker[-2]:.0f}s | "
              f"avg: {avg:.0f}s{Fore.RESET} — {now}")

        epoch += 1

    total = time_tracker[-1] - time_tracker[0]
    print(f"\n{Fore.BLUE}Done. Total: {total/3600:.2f}h for {TRAIN_STEPS} epochs.{Fore.RESET}")

def profile_selfplay():
    """
    Profiles a short self-play run to find what's actually eating CPU time.
    Loads the current champion (or starts random if none exists), runs a
    small batch of games under cProfile, then prints two sorted views:
      - cumulative time (best for finding the big outer bottlenecks)
      - total time / "tottime" (best for finding hot leaf-level functions,
        e.g. board.copy, legal_moves_mask, move_to_index)

    Also dumps a .prof file you can open with snakeviz for a visual
    flame-graph view:
        pip install snakeviz
        snakeviz selfplay_profile.prof
    """
    import cProfile
    import pstats

    # ── Load a model to profile against (same logic as train_loop's champion load) ──
    model = new_model()
    champion_epoch = 0
    if os.path.exists(CHAMP_EPOCH_PATH):
        with open(CHAMP_EPOCH_PATH) as f:
            champion_epoch = int(f.read().strip())
    champ_path = checkpoint_path(champion_epoch)
    if os.path.exists(champ_path):
        print(f"Profiling with champion: {champ_path}")
        load_checkpoint(champ_path, model)
    elif os.path.exists(HUMAN_PRETRAINED_PATH):
        print("No selfplay champion found, profiling with human pretrained weights.")
        ckpt = torch.load(HUMAN_PRETRAINED_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("No champion or pretrained weights found, profiling with random weights.")

    # ── Keep this small -- it's just for timing breakdown, not real training data ──
    PROFILE_POOL_SIZE  = 12
    PROFILE_TOTAL_GAMES = 6

    profiler = cProfile.Profile()
    profiler.enable()

    data = self_play_games(
        model, PROFILE_POOL_SIZE, PROFILE_TOTAL_GAMES,
        SELFPLAY_ARGS, epoch=0, verbose=True
    )

    profiler.disable()

    print(f"\nCollected {len(data):,} positions from {PROFILE_TOTAL_GAMES} games.\n")

    stats = pstats.Stats(profiler)

    print("=" * 80)
    print("TOP 25 BY CUMULATIVE TIME (best for finding big outer bottlenecks)")
    print("=" * 80)
    stats.sort_stats('cumulative').print_stats(25)

    print("=" * 80)
    print("TOP 25 BY TOTAL TIME / tottime (best for finding hot leaf functions)")
    print("=" * 80)
    stats.sort_stats('tottime').print_stats(25)

    prof_path = "selfplay_profile.prof"
    profiler.dump_stats(prof_path)
    print(f"\nSaved profile to '{prof_path}'.")

    # ── Auto-launch the snakeviz flame graph in your browser ──────────────
    # Requires: pip install snakeviz
    try:
        from snakeviz.cli import main as snakeviz_main
        import sys
        print("Launching snakeviz in your browser...")
        sys.argv = ["snakeviz", prof_path]
        snakeviz_main()
    except ImportError:
        print("snakeviz not installed. Run 'pip install snakeviz', then:")
        print(f"  python -m snakeviz selfplay_profile.prof")


if __name__ == "__main__":
    # profile_selfplay()
    train_loop()