r"""
TablebaseData.py

Generates synthetic endgame training examples from perfect Syzygy tablebase
play, to patch the gap human GM game data leaves behind: strong human
players resign long before a won position is actually converted to mate, or
before a lost one is fully defended, so a pure human-game dataset barely
shows the network "how to finish". This produces exactly that -- random
legal <= N-piece positions, played to completion with tablebase-optimal
moves.

Writes the SAME on-disk format as HumanData.py's build_dataset_to_disk():
<out_prefix>_states.bin (float32, IN_PLANES x 8 x 8)
<out_prefix>_moves.bin  (int64, single played-move index)
<out_prefix>_values.bin (float32, side-to-move perspective)

So this is a drop-in second cache, loadable with the exact same
MemmapChessDataset from TrainHuman.py -- and since these are flat arrays
read in shuffled order by DataLoader anyway, you can also just concatenate
the raw files with your existing human_data_cache_*.bin to mix them into
one combined cache (see concatenate_caches() below) instead of juggling two
datasets.

Usage (standalone CLI):
    python TablebaseData.py --syzygy-path "...\syzygy\WDL" "...\syzygy\DTZ" \
        --num-games 20000 --max-pieces 5 \
        --out "Human Data/tablebase_endgames"
"""

import math
import os
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import chess
import chess.syzygy as syzygy

from Encoder import encode_board, move_to_index, IN_PLANES


# ─────────────────────────────────────────────────────────────────────────
# Random legal position sampling
# ─────────────────────────────────────────────────────────────────────────

# Rough material weighting so common instructive endgames (K+P, K+R, K+Q vs
# lone king, and same-piece endgames) come up often, not just chaotic
# 5-piece soups. Tune freely.
MATERIAL_TEMPLATES = [
    # (white_extra_pieces, black_extra_pieces) piece-type lists, excluding kings
    ([chess.PAWN], []),
    ([chess.ROOK], []),
    ([chess.QUEEN], []),
    ([chess.ROOK], [chess.PAWN]),
    ([chess.QUEEN], [chess.PAWN]),
    ([chess.ROOK], [chess.ROOK]),
    ([chess.ROOK, chess.PAWN], []),
    ([chess.BISHOP, chess.PAWN], []),
    ([chess.KNIGHT, chess.PAWN], []),
    ([chess.QUEEN], [chess.ROOK]),
    ([chess.ROOK, chess.PAWN], [chess.PAWN]),
]


def _random_square(exclude: set) -> int:
    while True:
        sq = random.randrange(64)
        if sq not in exclude:
            return sq


def sample_random_position(max_pieces: int = 5, max_attempts: int = 200) -> Optional[chess.Board]:
    """
    Returns a random *legal* position with <= max_pieces total pieces
    (including both kings), no castling rights, drawn from a lightly-curated
    set of instructive material balances. Returns None on repeated failure
    (rare -- most attempts succeed within a handful of tries).
    """
    for _ in range(max_attempts):
        templates = [t for t in MATERIAL_TEMPLATES if 2 + len(t[0]) + len(t[1]) <= max_pieces]
        if not templates:
            templates = [([], [])]
        white_extra, black_extra = random.choice(templates)

        board = chess.Board.empty()
        occupied = set()

        wk = _random_square(occupied)
        occupied.add(wk)
        bk = _random_square(occupied)
        occupied.add(bk)
        if chess.square_distance(wk, bk) <= 1:
            continue  # kings can't be adjacent -- resample
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        ok = True
        for pt in white_extra:
            sq = _random_square(occupied)
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
        if not ok:
            continue
        for pt in black_extra:
            sq = _random_square(occupied)
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            occupied.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
        if not ok:
            continue

        board.turn = random.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1

        if not board.is_valid():
            continue
        return board

    return None


# ─────────────────────────────────────────────────────────────────────────
# Tablebase-optimal move selection
# ─────────────────────────────────────────────────────────────────────────

def _dtz_to_value(dtz: int) -> float:
    if dtz == 0:
        return 0.0
    sign = 1.0 if dtz > 0 else -1.0
    return sign * (0.5 + 0.49 * math.exp(-abs(dtz) / 25.0))


@dataclass
class TBEval:
    wdl: int              # -2..2, from perspective of side to move
    dtz: Optional[int]    # signed halfmoves-to-zeroing, or None if unavailable
    value: float          # scalar in (-1, 1), matches mcts_core's convention


def probe(tb, board: chess.Board) -> Optional[TBEval]:
    try:
        wdl = tb.probe_wdl(board)
    except Exception:
        return None
    dtz = None
    value = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)
    try:
        dtz = tb.probe_dtz(board)
        value = _dtz_to_value(dtz)
    except Exception:
        pass
    return TBEval(wdl=wdl, dtz=dtz, value=value)


def best_moves(tb, board: chess.Board) -> Tuple[List[chess.Move], Optional[TBEval]]:
    """
    Returns (list_of_equally_optimal_moves, position_eval). Selection rule:
    among all legal moves, prefer the ones giving the opponent the worst
    resulting WDL; among those, prefer the smallest |DTZ| (fastest
    conversion if winning, longest resistance if losing -- standard
    tablebase "optimal play" convention). Falls back to WDL-only ranking if
    DTZ files aren't loaded.
    """
    legal = list(board.legal_moves)
    if not legal:
        return [], None

    scored = []
    for mv in legal:
        board.push(mv)
        child = probe(tb, board)
        board.pop()
        if child is None:
            continue
        our_wdl = -child.wdl
        our_dtz = -child.dtz if child.dtz is not None else None
        scored.append((mv, our_wdl, our_dtz))

    if not scored:
        return [], None

    best_wdl = max(s[1] for s in scored)
    candidates = [s for s in scored if s[1] == best_wdl]

    if any(s[2] is not None for s in candidates):
        candidates_with_dtz = [c for c in candidates if c[2] is not None]
        min_abs_dtz = min(abs(c[2]) for c in candidates_with_dtz)
        chosen = [c[0] for c in candidates_with_dtz if abs(c[2]) == min_abs_dtz]
    else:
        chosen = [c[0] for c in candidates]

    pos_eval = probe(tb, board)  # eval of the *current* position, not a child
    return chosen, pos_eval


# ─────────────────────────────────────────────────────────────────────────
# Streaming generation, same on-disk format as HumanData.build_dataset_to_disk
# ─────────────────────────────────────────────────────────────────────────

def build_tablebase_dataset_to_disk(
    syzygy_paths,
    out_prefix: str,
    num_games: Optional[int] = None,
    target_positions: Optional[int] = None,
    max_pieces: int = 5,
    max_plies_per_game: int = 80,
    seed: Optional[int] = None,
) -> int:
    """
    Streams tablebase-optimal-play positions straight to
    <out_prefix>_states.bin / _moves.bin / _values.bin, in the exact same
    layout as HumanData.build_dataset_to_disk(), so it opens with the
    identical MemmapChessDataset from TrainHuman.py. Returns n_positions.

    Pass exactly one of:
      num_games        -- stop after this many games (position count varies
                           per game -- a quick queen mate is ~10-15 plies,
                           a blocked pawn ending can run much longer -- so
                           this doesn't give you an exact position count).
      target_positions  -- stop once this many positions have been written,
                           regardless of how many games that takes. Use this
                           if you're aiming for a specific mix ratio against
                           an existing human-data cache (see
                           positions_needed_for_ratio() below).

    Where multiple moves are equally optimal (common in simple endgames --
    e.g. several rook checks that all mate equally fast), one is picked at
    random each ply, both as the game's continuation and as that position's
    policy label -- this avoids teaching a false preference between
    genuinely equal moves while still fitting the single-move-index schema.
    """
    if (num_games is None) == (target_positions is None):
        raise ValueError("Pass exactly one of num_games or target_positions")

    if seed is not None:
        random.seed(seed)

    tb = None
    for p in ([syzygy_paths] if isinstance(syzygy_paths, str) else syzygy_paths):
        if tb is None:
            tb = syzygy.open_tablebase(p)
        else:
            tb.add_directory(p)
    if tb is None:
        raise ValueError("No valid syzygy path(s) given")

    n_positions = 0
    games_done = 0
    attempts = 0
    # generous upper bound on attempts so a bad config can't loop forever
    max_attempts = (num_games * 20) if num_games is not None else (target_positions * 2 + 10000)

    with open(f"{out_prefix}_states.bin", "wb") as states_f, \
         open(f"{out_prefix}_moves.bin", "wb") as moves_f, \
         open(f"{out_prefix}_values.bin", "wb") as values_f:

        while attempts < max_attempts:
            if num_games is not None and games_done >= num_games:
                break
            if target_positions is not None and n_positions >= target_positions:
                break

            attempts += 1
            board = sample_random_position(max_pieces=max_pieces)
            if board is None:
                continue

            wrote_any = False
            for _ply in range(max_plies_per_game):
                if target_positions is not None and n_positions >= target_positions:
                    break
                if board.is_game_over(claim_draw=False):
                    break
                moves, pos_eval = best_moves(tb, board)
                if not moves or pos_eval is None:
                    break  # out of tablebase range -- shouldn't happen, piece count only shrinks

                state = encode_board(board)
                mv = random.choice(moves)
                idx = move_to_index(mv, board)

                states_f.write(np.asarray(state, dtype=np.float32).tobytes())
                moves_f.write(np.int64(idx).tobytes())
                values_f.write(np.float32(pos_eval.value).tobytes())
                n_positions += 1
                wrote_any = True

                board.push(mv)

            if wrote_any:
                games_done += 1

    print(f"[TablebaseData] Wrote {n_positions:,} positions from {games_done:,} games "
          f"({attempts:,} sampling attempts) to {out_prefix}_*.bin")
    return n_positions


def positions_needed_for_ratio(existing_positions: int, target_ratio: float) -> int:
    """
    How many synthetic positions to generate so that, once combined with an
    existing cache of `existing_positions`, the synthetic slice makes up
    `target_ratio` of the total. E.g. positions_needed_for_ratio(23_000_000,
    0.15) -> ~4,058,824.
    """
    if not (0 < target_ratio < 1):
        raise ValueError("target_ratio must be between 0 and 1 (exclusive)")
    return round((target_ratio / (1 - target_ratio)) * existing_positions)


def count_cache_positions(prefix: str) -> int:
    """Position count of an existing build_dataset_to_disk-style cache,
    read the same way TrainHuman.py does (via the moves file's shape)."""
    return int(np.memmap(f"{prefix}_moves.bin", dtype=np.int64, mode='r').shape[0])


# ─────────────────────────────────────────────────────────────────────────
# Merging with an existing human-data cache (or any other same-schema cache)
# ─────────────────────────────────────────────────────────────────────────

def concatenate_caches(prefixes: List[str], out_prefix: str, chunk_bytes: int = 64 * 1024 * 1024) -> int:
    """
    Concatenates any number of build_dataset_to_disk-style caches (human
    data, tablebase data, whatever -- same schema) into one combined cache,
    streaming in chunks rather than loading anything fully into RAM. Since
    DataLoader shuffles at read time, position order in the combined file
    doesn't matter.

    Returns the combined position count (read back from the moves file,
    same pattern TrainHuman.py already uses).
    """
    for suffix in ("_states.bin", "_moves.bin", "_values.bin"):
        with open(f"{out_prefix}{suffix}", "wb") as out_f:
            for prefix in prefixes:
                in_path = f"{prefix}{suffix}"
                with open(in_path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(chunk_bytes)
                        if not chunk:
                            break
                        out_f.write(chunk)

    n_positions = np.memmap(f"{out_prefix}_moves.bin", dtype=np.int64, mode='r').shape[0]
    print(f"[TablebaseData] Combined {len(prefixes)} caches into {out_prefix}_*.bin "
          f"({n_positions:,} total positions)")
    return n_positions


# ─────────────────────────────────────────────────────────────────────────
# Config -- edit these instead of passing CLI args, same style as
# TrainHuman.py's PGN_PATH / CACHE_PREFIX block up top.
# ─────────────────────────────────────────────────────────────────────────

# One directory, or a list of directories (e.g. separate WDL/DTZ folders --
# add_directory() gets called for each one, same as mcts_core.init_tablebase).
SYZYGY_PATH = [
    r"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\syzygy\Syzygy345WDL",
    r"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\syzygy\Syzygy345DTZ",
]

OUT_PREFIX = r"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\Human Data\tablebase_endgames"

# Existing human-data cache to match ratio against. Set HUMAN_CACHE_PREFIX
# and TARGET_RATIO to auto-compute how many positions are needed; or set
# NUM_GAMES directly instead if you'd rather just generate a fixed number
# of games and not worry about hitting an exact ratio.
HUMAN_CACHE_PREFIX = r"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\Human Data\human_data_cache"
TARGET_RATIO = 0.15   # synthetic positions as a fraction of the combined total

NUM_GAMES = None      # set an int (and leave TARGET_RATIO/HUMAN_CACHE_PREFIX unused) for game-count mode instead

MAX_PIECES = 5           # must not exceed what your syzygy files actually cover
MAX_PLIES_PER_GAME = 80
SEED = None              # set an int for reproducible generation

# Optional: existing same-schema cache prefix(es) to merge this new data
# with (e.g. your human_data_cache). Leave as None/[] to skip merging --
# you'd then just generate tablebase_endgames_*.bin on its own.
MERGE_WITH = [HUMAN_CACHE_PREFIX]
COMBINED_OUT_PREFIX = OUT_PREFIX + "_combined"


if __name__ == "__main__":
    start = time.time()
    if NUM_GAMES is not None:
        print(f"[TablebaseData] Generating {NUM_GAMES} tablebase games")
        n = build_tablebase_dataset_to_disk(
            syzygy_paths=SYZYGY_PATH,
            out_prefix=OUT_PREFIX,
            num_games=NUM_GAMES,
            max_pieces=MAX_PIECES,
            max_plies_per_game=MAX_PLIES_PER_GAME,
            seed=SEED,
        )
    else:
        print(f"[TablebaseData] Targeting {TARGET_RATIO:.0%} synthetic positions")
        existing = count_cache_positions(HUMAN_CACHE_PREFIX)
        target = positions_needed_for_ratio(existing, TARGET_RATIO)
        print(f"[TablebaseData] Human cache has {existing:,} positions; "
              f"targeting {target:,} synthetic positions for a {TARGET_RATIO:.0%} mix")
        n = build_tablebase_dataset_to_disk(
            syzygy_paths=SYZYGY_PATH,
            out_prefix=OUT_PREFIX,
            target_positions=target,
            max_pieces=MAX_PIECES,
            max_plies_per_game=MAX_PLIES_PER_GAME,
            seed=SEED,
        )

    print(f"[TablebaseData] Generated {n:,} tablebase positions in {time.time() - start:.1f}s")
    
    if MERGE_WITH:
        concatenate_caches(prefixes=MERGE_WITH + [OUT_PREFIX], out_prefix=COMBINED_OUT_PREFIX)