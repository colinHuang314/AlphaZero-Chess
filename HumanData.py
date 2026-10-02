"""
Builds a supervised training set from human games (PGN), to sanity-check the
network/encoder before investing in the (still unbatched) self-play loop.

For each position in each game: state = encode_board(board) (the position
before the move was made), the policy target is the single move index of
the move actually played, and the value target is +1/-1/0 for whoever is
to move at that position, derived from the game's final result.

Elo filtering only works if the PGN has WhiteElo/BlackElo headers -- true
for Lichess exports, not for most TWIC files (those are all titled-player
games by virtue of the event, with no numeric Elo tag at all). Leave
min_elo=None for TWIC-style files.

No board-symmetry augmentation here, unlike the Go version. Go's 8-fold
rotation/reflection symmetry doesn't exist for chess -- the board only has
a left/right mirror at most, and using it correctly means also swapping
kingside/queenside castling rights and re-deriving move indices through a
mirrored direction table. Skipped for now to keep this first pass simple
and obviously correct.
"""

import chess
import chess.pgn
import numpy as np

from Encoder import encode_board, move_to_index

_RESULT_TO_WINNER = {
    "1-0": chess.WHITE,
    "0-1": chess.BLACK,
    "1/2-1/2": None,
}


def iter_positions(pgn_path, max_games=None, min_elo=None):
    """Generator: yields (state, move_index, value) one position at a time.
    This is the actual extraction logic; build_dataset() and
    build_dataset_to_disk() below just differ in what they do with each
    position -- collect it in RAM, or write it straight to disk."""
    n_games = 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while max_games is None or n_games < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # end of file

            result = game.headers.get("Result")
            if result not in _RESULT_TO_WINNER:
                continue  # aborted/unknown-result games ("*"), skip
            winner = _RESULT_TO_WINNER[result]

            if min_elo is not None:
                try:
                    w_elo = int(game.headers.get("WhiteElo", "0"))
                    b_elo = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    continue
                if w_elo < min_elo or b_elo < min_elo:
                    continue

            board = game.board()
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    break  # corrupt/non-standard game, stop using it

                state = encode_board(board)
                idx = move_to_index(move, board)
                value = 0.0 if winner is None else (1.0 if winner == board.turn else -1.0)
                yield state, idx, value

                board.push(move)

            n_games += 1


def build_dataset(pgn_path, max_games=None, min_elo=None):
    """Holds the whole dataset in RAM. Fine up to a few million positions;
    beyond that, use build_dataset_to_disk() instead -- 1M positions is
    already ~4.6GB just for the states array, before counting the transient
    doubling np.stack() causes at the end."""
    states, move_indices, target_vs = [], [], []
    for state, idx, value in iter_positions(pgn_path, max_games, min_elo):
        states.append(state)
        move_indices.append(idx)
        target_vs.append(value)

    states = np.stack(states).astype(np.float32)
    move_indices = np.array(move_indices, dtype=np.int64)
    target_vs = np.array(target_vs, dtype=np.float32)
    return states, move_indices, target_vs


def build_dataset_to_disk(pgn_path, out_prefix, max_games=None, min_elo=None):
    """Streams positions straight to disk instead of materializing them in
    RAM. Peak memory while this runs is roughly one position's worth,
    regardless of how many games you process -- dataset size becomes a
    disk-space question, not a RAM question. Writes
    <out_prefix>_states.bin / _moves.bin / _values.bin as flat binary files;
    returns the position count, which you need (along with out_prefix) to
    open them afterward via MemmapChessDataset in train_human.py.

    Disk space still scales the same way RAM did before: at this encoding
    size (IN_PLANES x 8 x 8 float32) it's about 4.6KB/position for states
    alone, so a few million positions is multiple GB on disk, not free.
    """
    n_positions = 0
    with open(f"{out_prefix}_states.bin", "wb") as states_f, \
         open(f"{out_prefix}_moves.bin", "wb") as moves_f, \
         open(f"{out_prefix}_values.bin", "wb") as values_f:

        for state, idx, value in iter_positions(pgn_path, max_games, min_elo):
            states_f.write(np.asarray(state, dtype=np.float32).tobytes())
            moves_f.write(np.int64(idx).tobytes())
            values_f.write(np.float32(value).tobytes())
            n_positions += 1

    return n_positions