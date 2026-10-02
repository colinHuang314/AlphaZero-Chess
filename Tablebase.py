import chess.syzygy

SYZYGY_PATH = fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\syzygy\Syzygy345WDL"
MAX_TABLEBASE_PIECES = 5 # 3-4-5 piece only
_tablebase = None

def get_tablebase():
    global _tablebase
    if _tablebase is None:
        _tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
    return _tablebase

def should_use_tablebase(board: chess.Board) -> bool:
    return len(board.piece_map()) <= MAX_TABLEBASE_PIECES