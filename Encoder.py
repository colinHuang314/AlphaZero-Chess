"""
Board and move encoders for the chess net.

Design choice: everything is canonicalized to "the player about to move is
white." If it's actually black's turn, we encode board.mirror() instead of
board directly, and move_to_index()/index_to_move() silently mirror move
squares through chess.square_mirror() on the way in/out. Callers never deal
with this -- you always pass the real board and a real, pushable move; the
flip is fully contained in this module.

Board encoding: 18 planes, no history stack.
  0-5   my pieces   (P, N, B, R, Q, K)
  6-11  opp pieces  (P, N, B, R, Q, K)
  12    my kingside castling rights
  13    my queenside castling rights
  14    opp kingside castling rights
  15    opp queenside castling rights
  16    en passant target square (one-hot)
  17    halfmove clock / 100  (no-progress counter, for the fifty-move rule)

Move encoding: AlphaZero's 73-move-type scheme, flattened.
  flat_index = (rank * 8 + file) * 73 + move_type      -- (rank, file) of the
  FROM square in canonical coordinates. move_type is 0-72:
    0-55   "queen-like" moves: 8 directions x 7 distances
    56-63  knight moves
    64-72  underpromotions: 3 forward offsets x 3 piece types
  Queen promotions are NOT a separate move_type -- they reuse the ordinary
  forward queen-like slot, and get promotion=QUEEN attached automatically
  when decoding, because a pawn moving to the back rank can only mean that.
  This also means castling needs no special case: O-O is just a king moving
  two squares east, which is already a valid queen-like move_type.
"""


import numpy as np
import chess

IN_PLANES = 18
NUM_MOVE_TYPES = 73
ACTION_SIZE = 8 * 8 * NUM_MOVE_TYPES  # 4672

# Compass order: N, NE, E, SE, S, SW, W, NW, as (delta_rank, delta_file)
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

_KNIGHT_OFFSETS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_INDEX = {off: i for i, off in enumerate(_KNIGHT_OFFSETS)}

# Forward-only (canonical pawns always move +1 rank): left-capture, push, right-capture
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def encode_board(board: chess.Board) -> np.ndarray:
    """Returns float32 array of shape (IN_PLANES, 8, 8)."""
    canonical = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((IN_PLANES, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(chess.PIECE_TYPES):  # 0-5: my pieces
        for sq in canonical.pieces(piece_type, chess.WHITE):
            planes[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    for i, piece_type in enumerate(chess.PIECE_TYPES):  # 6-11: opponent pieces
        for sq in canonical.pieces(piece_type, chess.BLACK):
            planes[6 + i, chess.square_rank(sq), chess.square_file(sq)] = 1.0

    planes[12, :, :] = float(canonical.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(canonical.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(canonical.has_kingside_castling_rights(chess.BLACK))
    planes[15, :, :] = float(canonical.has_queenside_castling_rights(chess.BLACK))

    if canonical.ep_square is not None:
        planes[16, chess.square_rank(canonical.ep_square), chess.square_file(canonical.ep_square)] = 1.0

    planes[17, :, :] = canonical.halfmove_clock / 100.0

    return planes


def _queen_dir_and_dist(dr: int, df: int):
    dist = max(abs(dr), abs(df))
    unit = (dr // dist, df // dist)
    return _QUEEN_DIRS.index(unit), dist


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    mirror = board.turn == chess.BLACK
    frm = chess.square_mirror(move.from_square) if mirror else move.from_square
    to = chess.square_mirror(move.to_square) if mirror else move.to_square

    r0, f0 = chess.square_rank(frm), chess.square_file(frm)
    r1, f1 = chess.square_rank(to), chess.square_file(to)
    dr, df = r1 - r0, f1 - f0

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        file_idx = df + 1  # df in {-1, 0, 1} -> {0, 1, 2}
        move_type = 64 + file_idx * 3 + piece_idx
    elif (dr, df) in _KNIGHT_INDEX:
        move_type = 56 + _KNIGHT_INDEX[(dr, df)]
    else:
        direction_idx, dist = _queen_dir_and_dist(dr, df)
        move_type = direction_idx * 7 + (dist - 1)

    return (r0 * 8 + f0) * NUM_MOVE_TYPES + move_type


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    mirror = board.turn == chess.BLACK
    square_idx, move_type = divmod(index, NUM_MOVE_TYPES)
    r0, f0 = divmod(square_idx, 8)

    promotion = None
    if move_type < 56:
        direction_idx, dist0 = divmod(move_type, 7)
        dr, df = _QUEEN_DIRS[direction_idx]
        dr, df = dr * (dist0 + 1), df * (dist0 + 1)
    elif move_type < 64:
        dr, df = _KNIGHT_OFFSETS[move_type - 56]
    else:
        sub = move_type - 64
        file_idx, piece_idx = divmod(sub, 3)
        dr, df = 1, file_idx - 1
        promotion = _UNDERPROMO_PIECES[piece_idx]

    r1, f1 = r0 + dr, f0 + df
    frm = chess.square(f0, r0)
    to = chess.square(f1, r1)

    if mirror:
        frm, to = chess.square_mirror(frm), chess.square_mirror(to)

    if promotion is None:
        piece = board.piece_at(frm)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(frm, to, promotion=promotion)


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = 1.0
    return mask