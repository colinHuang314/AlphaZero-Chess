import os
import chess.syzygy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/, syzygy/, Human Data/ ... live next to this script

SYZYGY_PATH = os.path.join(BASE_DIR, "syzygy", "Syzygy345WDL")
MAX_TABLEBASE_PIECES = 5 # 3-4-5 piece only
_tablebase = None

def get_tablebase():
    global _tablebase
    if _tablebase is None:
        _tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
    return _tablebase

def should_use_tablebase(board: chess.Board) -> bool:
    return len(board.piece_map()) <= MAX_TABLEBASE_PIECES