"""
AnalyzeUI.py  --  Lichess-style visual analysis board for your chess bot.

The MCTS runs continuously in a background thread. The browser polls for
updates every 300ms and re-renders the eval bar, arrows, and candidate list.
advance_root() is called automatically when you make a move on the board.

Requirements:
    pip install flask

Usage:
    python AnalyzeUI.py
    Browser opens automatically at http://localhost:5000
"""

import sys
import os
import re
import time
import math
import json
import threading
import webbrowser
import warnings
import numpy as np
import chess
import torch
from flask import Flask, jsonify, request, render_template_string

from Network2 import AZNetChess, init_weights
from Encoder import IN_PLANES, ACTION_SIZE, encode_board
from mcts_core import MCTS, batch_search, init_tablebase
from TrainHuman import CHANNELS, BLOCKS
from Tablebase import SYZYGY_PATH, MAX_TABLEBASE_PIECES

init_tablebase([
    fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\syzygy\Syzygy345WDL",
    fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\syzygy\Syzygy345DTZ",
], max_pieces=5)

# ── Config ────────────────────────────────────────────────────────────────────
# 106 was fine
# CHECKPOINT_PATH = fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\Models\chess_selfplay_epoch_104.pt"
CHECKPOINT_PATH = fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\Models\model_human_pretrained_7-7.pt"

# CHECKPOINT_PATH = fr"C:\Users\colin\OneDrive\Desktop\VS Code\Projects\Chess Bot\Models\chess_selfplay4_epoch_12.pt"
PORT     = 5000
BOARD_PX = 720      # must be divisible by 8

ARGS = {    
    'CPUCT':             2,
    'DIRICHLET_ALPHA':   0.15,
    'DIRICHLET_EPSILON': 0.0,
    'MIN_PRIOR':         1e-3,
    'VIRTUAL_LOSS':       1.0,
    'VIRTUAL_LOSS_VALUE': -1.0,
    'FORCED_PLAYOUT_K': 0.4,      # try 1-2; higher = more forced exploration, slower convergence per move
    'TACTICAL_CHECK_MIN_PRIOR': 0.45,   
    'TACTICAL_CAPTURE_MIN_PRIOR': 0.2,   
    # 'TACTICAL_FEW_REPLIES_MIN_PRIOR': 0.5,   # tune independently of check/capture floors
    'TACTICAL_ONE_REPLY_MIN_PRIOR': 0.5,   # move leaves opponent exactly 1 legal reply
    'TACTICAL_TWO_REPLY_MIN_PRIOR': 0.4,  # move leaves opponent exactly 2 legal replies
    'FORCED_PLAYOUT_MIN_ABSOLUTE': 5,   # every child guaranteed this many visits, regardless of prior
}

# batch_search tuning for continuous single-game analysis
SEARCH_CHUNK   = 8     # sims requested per batch_search() call
MAX_BATCH_SIZE = 8      # leaves collected per GPU call
MAX_WAIT_S     = 0.002  # time to wait collecting leaves before firing a GPU call --- 0.002?
PV_MAX_DEPTH   = 12

# ── App + device ──────────────────────────────────────────────────────────────
app    = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract epoch label from checkpoint filename for display
_epoch_label = "custom"
_m = re.search(r'epoch[_\s](\d+)', os.path.basename(CHECKPOINT_PATH), re.IGNORECASE)
if _m:
    _epoch_label = f"epoch {_m.group(1)}"

# ── Global state  (always access under _lock) ─────────────────────────────────
_lock    = threading.Lock()
_board   = chess.Board()
_mcts    = MCTS(ARGS)
_flipped = False

_total_sims     = 0
_nps            = 0
_sims_bucket    = 0
_last_nps_time  = time.time()
_searching      = True

_move_history = []   # list of {'uci': ..., 'san': ...}, oldest first
_base_fen = chess.Board().fen()   # the position _move_history's moves are replayed from -- NOT
                                    # always the standard start; updated by /reset and /set_fen

# ── Model ─────────────────────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    _model = AZNetChess(
        in_planes=IN_PLANES, channels=CHANNELS,
        blocks=BLOCKS, action_size=ACTION_SIZE,
    ).to(DEVICE)
    _model.apply(init_weights)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        _model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded: {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"  WARNING: checkpoint not found, using random weights.")
    _model.eval()


def _infer_batch_fn(batch_items):
    """
    batch_items: list of encoded board arrays (as produced by encode_board()).
    Returns (policy_logits_batch, value_batch) as numpy arrays, indexable by
    row -- this is the contract mcts_core.batch_search expects.
    """
    assert _model is not None, "Model not loaded. Call load_model() first."
    x = torch.from_numpy(np.stack(batch_items)).to(DEVICE)
    with torch.no_grad():
        logits, values = _model(x)
    return logits.cpu().numpy(), values.reshape(-1).cpu().numpy()


# ── Search thread ─────────────────────────────────────────────────────────────
def _search_worker():
    global _total_sims, _nps, _sims_bucket, _last_nps_time

    while _searching:
        with _lock:
            root = _mcts.root
            game_over = _board.outcome(claim_draw=True) is not None
            if root is not None and not game_over:
                # Collects up to SEARCH_CHUNK leaves via virtual loss, batching
                # GPU calls up to MAX_BATCH_SIZE at a time -- same mechanism
                # used in self-play/arena, just against a single MCTS instance.
                batch_search(
                    [_mcts], SEARCH_CHUNK, _infer_batch_fn,
                    max_batch_size=MAX_BATCH_SIZE,
                    max_wait_s=MAX_WAIT_S,
                    min_prior=ARGS['MIN_PRIOR'],
                )
                _total_sims += SEARCH_CHUNK
                _sims_bucket += SEARCH_CHUNK

        # NPS update (outside lock so we don't hold it for timing)
        now = time.time()
        if now - _last_nps_time >= 1.0:
            _nps = _sims_bucket
            _sims_bucket = 0
            _last_nps_time = now
            # debug_root_children(_mcts)

        time.sleep(0)  # yield GIL to Flask request threads


def _get_pv(root, board_at_root, max_depth=PV_MAX_DEPTH):
    """Walk best_child (max visits) down the tree, returning a SAN move list.
    Must be called with _lock held. Uses a throwaway board copy so the live
    root_board used by the search thread is never touched."""
    pv = []
    node = root
    board = board_at_root.copy(stack=False)
    depth = 0
    while node is not None and node.children and depth < max_depth:
        best = max(node.children, key=lambda c: c.visits)
        if best.visits <= 0:
            break
        if best.action_taken not in board.legal_moves:
            # Defensive: should not happen now that expand() is idempotent,
            # but never let a stale/duplicate node crash the /state endpoint.
            break
        try:
            pv.append(board.san(best.action_taken))
        except Exception:
            pv.append(best.action_taken.uci())
        board.push(best.action_taken)
        node = best
        depth += 1
    return pv


# ── State helper ──────────────────────────────────────────────────────────────
def _build_state():
    """Builds the JSON payload for /state. Must be called with _lock held."""
    fen     = _board.fen()
    turn    = 'white' if _board.turn == chess.WHITE else 'black'
    outcome = _board.outcome(claim_draw=True)
    game_over = outcome is not None

    root = _mcts.root
    if root is None or not root.children or game_over:
        return {
            'fen': fen, 'turn': turn,
            'eval': 0.0, 'eval_display': '0.00',
            'top_moves': [], 'pv': [],
            'total_sims': _total_sims, 'nps': _nps,
            'flipped': _flipped, 'game_over': game_over,
            'move_history': _move_history,
        }

    best = max(root.children, key=lambda c: c.visits)
    if best.visits == 0:
        root_value = 0.0
    else:
        root_value = -(best.value / best.visits)

    # Convert to white-absolute perspective
    white_eval = root_value if _board.turn == chess.WHITE else -root_value

    # Human-readable string in pseudo-pawn units
    try:
        v = max(-0.9999, min(0.9999, white_eval))
        pawns = 4.5 * math.log((1 + v) / (1 - v))
        if abs(white_eval) > 0.97:
            eval_str = ('+M' if white_eval > 0 else '-M')
        else:
            eval_str = f"{pawns:+.2f}"
    except Exception:
        eval_str = '0.00'

    # Always include at least the top 3 candidates,
    # but also include any moves within 50% of the top move's visit count.
    threshold = best.visits * 0.5
    sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
    within_threshold = [c for c in sorted_children if c.visits >= threshold]
    candidates = sorted_children[:max(3, len(within_threshold))][:8]

    top_moves = []
    for c in candidates:
        try:
            san = _board.san(c.action_taken)
        except Exception:
            san = c.action_taken.uci()
        q = -(c.value / c.visits) if c.visits > 0 else 0.0
        top_moves.append({
            'san':    san,
            'uci':    c.action_taken.uci(),
            'visits': int(c.visits),
            'pct':    round(c.visits / best.visits, 3),
            'q':      round(q, 3),
        })

    pv = _get_pv(root, _mcts.root_board)

    return {
        'fen': fen, 'turn': turn,
        'eval': round(white_eval, 4), 'eval_display': eval_str,
        'top_moves': top_moves, 'pv': pv,
        'total_sims': _total_sims, 'nps': _nps,
        'flipped': _flipped, 'game_over': game_over,
        'move_history': _move_history,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    cfg = json.dumps({'board_px': BOARD_PX, 'square_px': BOARD_PX // 8,
                      'epoch_label': _epoch_label})
    return render_template_string(HTML_TEMPLATE, cfg=cfg)


@app.route('/state')
def get_state():
    with _lock:
        return jsonify(_build_state())


@app.route('/move', methods=['POST'])
def make_move():
    global _total_sims
    uci = request.get_json().get('uci', '')
    with _lock:
        try:
            move = chess.Move.from_uci(uci)
            if move not in _board.legal_moves:
                return jsonify({'ok': False, 'error': 'Illegal move'})
            san = _board.san(move)
            _mcts.advance_root(move)
            _board.push(move)
            _move_history.append({'uci': uci, 'san': san})
            _total_sims = 0
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})


@app.route('/reset', methods=['POST'])
def reset_board():
    global _board, _total_sims, _move_history, _base_fen
    with _lock:
        _board = chess.Board()
        _base_fen = _board.fen()
        _mcts.new_root(_board)
        _total_sims = 0
        _move_history = []
    return jsonify({'ok': True})


@app.route('/undo', methods=['POST'])
def undo_move():
    global _board, _total_sims, _move_history
    with _lock:
        if not _move_history:
            return jsonify({'ok': False, 'error': 'No moves to undo'})
        _move_history.pop()
        new_board = chess.Board(_base_fen)   # rebuild from the actual starting
                                              # position, not always the default
        for m in _move_history:
            new_board.push(chess.Move.from_uci(m['uci']))
        _board = new_board
        _mcts.new_root(_board)
        _total_sims = 0
    return jsonify({'ok': True})


@app.route('/flip', methods=['POST'])
def flip_board():
    global _flipped
    with _lock:
        _flipped = not _flipped
    return jsonify({'ok': True, 'flipped': _flipped})


@app.route('/set_fen', methods=['POST'])
def set_fen():
    global _board, _total_sims, _move_history, _base_fen
    fen = request.get_json().get('fen', '').strip()
    with _lock:
        try:
            new_board = chess.Board(fen)
            _board = new_board
            _base_fen = new_board.fen()
            _mcts.new_root(_board)
            _total_sims = 0
            _move_history = []
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)})


# ── HTML / CSS / JS template ──────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chess Analyzer</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: #161512;
    color: #c8c8c8;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* ── Header ── */
header {
    background: #1e1b18;
    border-bottom: 1px solid #2c2a27;
    padding: 15px 30px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-shrink: 0;
}
header h1 {
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.3px;
}
.tag {
    font-size: 17px;
    color: #777;
    background: #2a2826;
    border: 1px solid #333;
    padding: 3px 12px;
    border-radius: 5px;
    font-family: 'Courier New', monospace;
}
.pulse {
    width: 11px; height: 11px;
    border-radius: 50%;
    background: #5d9a2a;
    margin-left: auto;
    flex-shrink: 0;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Main layout ── */
.layout {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 30px;
    flex: 1;
    min-height: 0;
}

/* ── Eval bar ── */
.eval-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-right: 15px;
}
.eval-bar {
    width: 21px;
    height: 720px;
    background: #111;
    border-radius: 3px;
    border: 1px solid #2c2c2c;
    position: relative;
    overflow: hidden;
}
.eval-fill {
    position: absolute;
    top: 0; left: 0; right: 0;
    background: #e8e8e8;
    transition: height 0.35s ease;
}
.eval-score {
    font-size: 17px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #999;
    letter-spacing: -0.4px;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    transform: rotate(180deg);
}

/* ── Board ── */
.board-wrap {
    position: relative;
    flex-shrink: 0;
    box-shadow: 0 12px 40px rgba(0,0,0,0.7);
}
#board { width: 720px; }

#arrow-layer {
    position: absolute;
    top: 0; left: 0;
    width: 720px; height: 720px;
    pointer-events: none;
}

/* ── Right panel ── */
.panel {
    width: 315px;
    margin-left: 24px;
    display: flex;
    flex-direction: column;
    gap: 15px;
    height: 720px;
    overflow-y: auto;
}
.card {
    background: #1e1b18;
    border: 1px solid #2c2a27;
    border-radius: 8px;
    padding: 17px 20px;
    flex-shrink: 0;
}
.card-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.7px;
    color: #555;
    margin-bottom: 12px;
}

/* Position card */
.pos-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 9px;
}
.dot {
    width: 17px; height: 17px;
    border-radius: 50%;
    border: 1px solid #444;
    flex-shrink: 0;
}
.dot.white { background: #fff; }
.dot.black { background: #111; }
#turn-text { font-size: 20px; font-weight: 500; color: #d0d0d0; }
#eval-big {
    font-size: 33px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.75px;
    margin-top: 3px;
}
#eval-big.pos  { color: #e0e0e0; }
#eval-big.neg  { color: #888; }
#eval-big.even { color: #aaa; }

/* Candidates card */
.cand-card { flex: 1 1 auto; min-height: 285px; overflow-y: auto; }
.cand-list { display: flex; flex-direction: column; gap: 0; }
.cand-row {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 8px 0;
    border-bottom: 1px solid #252320;
}
.cand-row:last-child { border: none; }
.cand-san {
    font-size: 20px;
    font-weight: 600;
    font-family: 'Courier New', monospace;
    color: #d8d8d8;
    width: 69px;
    flex-shrink: 0;
}
.cand-row.top-move .cand-san { color: #fff; }
.bar-bg {
    flex: 1;
    height: 8px;
    background: #2a2826;
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background: #3d72a8;
    border-radius: 3px;
    transition: width 0.3s ease;
}
.cand-row.top-move .bar-fill { background: #4a8fcc; }
.cand-vis {
    font-size: 15px;
    color: #555;
    width: 57px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
}

/* Stats card */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 18px;
}
.stat-k { color: #555; }
.stat-v { color: #999; font-variant-numeric: tabular-nums; font-weight: 500; }

/* ── Controls bar ── */
.controls {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 30px;
    background: #1a1714;
    border-top: 1px solid #222;
    flex-shrink: 0;
}
.btn {
    background: #2a2826;
    border: 1px solid #3a3835;
    color: #b8b8b8;
    padding: 9px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 20px;
    transition: background 0.12s, color 0.12s;
    white-space: nowrap;
}
.btn:hover { background: #343230; color: #fff; }
.fen-label { font-size: 17px; color: #555; white-space: nowrap; margin-left: 6px; }
.fen-in {
    flex: 1;
    background: #1e1b18;
    border: 1px solid #333;
    color: #b8b8b8;
    padding: 9px 14px;
    border-radius: 6px;
    font-size: 17px;
    font-family: 'Courier New', monospace;
    min-width: 0;
}
.fen-in:focus { outline: none; border-color: #3d72a8; }
.btn-go {
    background: #2a3d1a;
    border-color: #3d5626;
    color: #9bc86a;
}
.btn-go:hover { background: #344d20; color: #b8e080; }

/* Game over overlay */
.game-over-card {
    background: #2a1e1e;
    border: 1px solid #5a3030;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    font-size: 20px;
    color: #e09090;
}
</style>
</head>
<body>

<script>const CFG = {{ cfg | safe }};</script>

<header>
    <h1>♟ Chess Analyzer</h1>
    <span class="tag" id="epoch-tag">loading...</span>
    <div class="pulse" id="pulse-dot"></div>
</header>

<div class="layout">

    <!-- Eval bar -->
    <div class="eval-wrap">
        <div class="eval-bar">
            <div class="eval-fill" id="eval-fill" style="height:50%"></div>
        </div>
    </div>

    <!-- Board -->
    <div class="board-wrap">
        <div id="board"></div>
        <svg id="arrow-layer" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>

    <!-- Analysis panel -->
    <div class="panel">

        <!-- Position + eval -->
        <div class="card">
            <div class="card-label">Position</div>
            <div class="pos-row">
                <div class="dot" id="turn-dot"></div>
                <span id="turn-text">White to move</span>
            </div>
            <div id="eval-big" class="even">+0.00</div>
        </div>

        <!-- Top candidates -->
        <div class="card cand-card">
            <div class="card-label">Top Candidates</div>
            <div class="cand-list" id="cand-list">
                <div style="color:#444;font-size:18px">Searching...</div>
            </div>
        </div>

        <!-- Principal variation -->
        <div class="card">
            <div class="card-label">Engine Line</div>
            <div id="pv-line" style="font-family:'Courier New',monospace;font-size:18px;color:#9bb;line-height:1.5;">–</div>
        </div>

        <!-- Move history -->
        <div class="card">
            <div class="card-label">Moves</div>
            <div id="move-history" style="font-family:'Courier New',monospace;font-size:18px;color:#999;line-height:1.6;max-height:105px;overflow-y:auto;">–</div>
        </div>

        <!-- Search stats -->
        <div class="card">
            <div class="card-label">Search</div>
            <div class="stat-row">
                <span class="stat-k">Total sims</span>
                <span class="stat-v" id="stat-sims">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-k">Sims / sec</span>
                <span class="stat-v" id="stat-nps">0</span>
            </div>
        </div>

        <!-- Game over (hidden until needed) -->
        <div class="game-over-card" id="game-over-card" style="display:none"></div>

    </div>
</div>

<div class="controls">
    <button class="btn" onclick="doReset()">↺ Reset</button>
    <button class="btn" onclick="doUndo()">⤺ Undo</button>
    <button class="btn" onclick="doFlip()">⇅ Flip</button>
    <span class="fen-label">FEN:</span>
    <input class="fen-in" id="fen-in" type="text" placeholder="Paste FEN to jump to a position…">
    <button class="btn btn-go" onclick="doSetFen()">Go</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
'use strict';

const BOARD_PX  = CFG.board_px;
const SQUARE_PX = CFG.square_px;

document.getElementById('epoch-tag').textContent = CFG.epoch_label;

// ── Chess.js + Chessboard.js ──────────────────────────────────────────────
const game = new Chess();
let flipped = false;

const board = Chessboard('board', {
    draggable:   true,
    position:    'start',
    pieceTheme:  'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
    snapbackSpeed: 180,
    snapSpeed:   60,
    onDragStart(source, piece) {
        if (game.game_over()) return false;
        if (game.turn() === 'w' && piece[0] === 'b') return false;
        if (game.turn() === 'b' && piece[0] === 'w') return false;
    },
    onDrop(source, target) {
        if (source === target) return 'snapback';
        const mv = game.move({ from: source, to: target, promotion: 'q' });
        if (!mv) return 'snapback';
        const uci = source + target + (mv.promotion || '');
        fetch('/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uci }),
        }).then(r => r.json()).then(d => {
            if (!d.ok) { game.undo(); board.position(game.fen()); }
            else { lastFen = null; }   // force board refresh from server
        });
    },
    onSnapEnd() { board.position(game.fen()); },
});

// ── Square → pixel center ─────────────────────────────────────────────────
function sqToXY(sq, flip) {
    const fi = sq.charCodeAt(0) - 97;          // file index 0-7
    const ri = parseInt(sq[1]) - 1;             // rank index 0-7
    return {
        x: (flip ? 7 - fi : fi) * SQUARE_PX + SQUARE_PX / 2,
        y: (flip ? ri : 7 - ri) * SQUARE_PX + SQUARE_PX / 2,
    };
}

// ── Arrow rendering ───────────────────────────────────────────────────────
function drawArrow(svg, x1, y1, x2, y2, pct, isTop) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;               // perpendicular unit vector

    const shaftW = isTop ? 14 : 9;
    const headL  = isTop ? 33 : 24;
    const headW  = isTop ? 24 : 17;
    const alpha  = isTop ? 0.85 : 0.42 + pct * 0.35;

    // Shaft end (leave room for arrowhead)
    const ex = x2 - ux * headL, ey = y2 - uy * headL;

    // Shaft corners
    const shaft = [
        [x1 + px * shaftW / 2, y1 + py * shaftW / 2],
        [ex  + px * shaftW / 2, ey  + py * shaftW / 2],
        [ex  - px * shaftW / 2, ey  - py * shaftW / 2],
        [x1  - px * shaftW / 2, y1  - py * shaftW / 2],
    ].map(p => p.join(',')).join(' ');

    // Arrowhead corners
    const head = [
        [x2, y2],
        [ex + px * headW, ey + py * headW],
        [ex - px * headW, ey - py * headW],
    ].map(p => p.join(',')).join(' ');

    const color = isTop ? '#4a9ae0' : '#4a8acc';

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('opacity', alpha.toFixed(2));

    ['polygon', 'polygon'].forEach((tag, i) => {
        const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        el.setAttribute('points', i === 0 ? shaft : head);
        el.setAttribute('fill', color);
        g.appendChild(el);
    });

    svg.appendChild(g);
}

function refreshArrows(moves, flip) {
    const svg = document.getElementById('arrow-layer');
    svg.innerHTML = '';
    if (!moves || !moves.length) return;
    // Draw weaker moves first so the top move renders on top
    for (let i = moves.length - 1; i >= 0; i--) {
        const m = moves[i];
        const from = m.uci.slice(0, 2), to = m.uci.slice(2, 4);
        const a = sqToXY(from, flip), b = sqToXY(to, flip);
        drawArrow(svg, a.x, a.y, b.x, b.y, m.pct, i === 0);
    }
}

// ── Eval bar ──────────────────────────────────────────────────────────────
function refreshEval(whiteEval, display) {
    const pct = ((1 + whiteEval) / 2 * 100).toFixed(2);
    document.getElementById('eval-fill').style.height = pct + '%';

    const el = document.getElementById('eval-big');
    el.textContent = display;
    el.className = whiteEval > 0.04 ? 'pos' : whiteEval < -0.04 ? 'neg' : 'even';
}

// ── Candidate list ────────────────────────────────────────────────────────
function fmt(n) {
    return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
}

function refreshCandidates(moves) {
    const el = document.getElementById('cand-list');
    if (!moves || !moves.length) {
        el.innerHTML = '<div style="color:#444;font-size:18px">Searching…</div>';
        return;
    }
    el.innerHTML = moves.map((m, i) => `
        <div class="cand-row${i === 0 ? ' top-move' : ''}">
            <span class="cand-san">${m.san}</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width:${(m.pct * 100).toFixed(1)}%"></div>
            </div>
            <span class="cand-vis">${fmt(m.visits)}</span>
        </div>`
    ).join('');
}

// ── PV / move history ─────────────────────────────────────────────────────
function refreshPv(pv) {
    const el = document.getElementById('pv-line');
    el.textContent = (pv && pv.length) ? pv.join(' ') : '–';
}

function refreshMoveHistory(history) {
    const el = document.getElementById('move-history');
    if (!history || !history.length) { el.textContent = '–'; return; }
    let out = '';
    for (let i = 0; i < history.length; i++) {
        if (i % 2 === 0) out += `${(i / 2 + 1)}. `;
        out += history[i].san + ' ';
    }
    el.textContent = out.trim();
    el.scrollTop = el.scrollHeight;
}

// ── Main poll loop ────────────────────────────────────────────────────────
let lastFen = null;

function poll() {
    fetch('/state').then(r => r.json()).then(s => {

        // Sync board only when FEN actually changed (avoid flicker)
        if (s.fen !== lastFen) {
            game.load(s.fen);
            board.position(s.fen, false);
            document.getElementById('fen-in').value = s.fen;
            lastFen = s.fen;
        }

        // Board orientation
        if (s.flipped !== flipped) {
            flipped = s.flipped;
            board.orientation(flipped ? 'black' : 'white');
        }

        // Turn dot + label
        const isWhite = s.turn === 'white';
        const dotEl = document.getElementById('turn-dot');
        dotEl.className = 'dot ' + s.turn;
        document.getElementById('turn-text').textContent =
            s.game_over ? 'Game over'
            : isWhite   ? 'White to move'
            :              'Black to move';

        refreshEval(s.eval, s.eval_display);
        refreshArrows(s.top_moves, flipped);
        refreshCandidates(s.top_moves);
        refreshPv(s.pv);
        refreshMoveHistory(s.move_history);

        document.getElementById('stat-sims').textContent = fmt(s.total_sims);
        document.getElementById('stat-nps').textContent  = s.nps.toLocaleString();

        const goCard = document.getElementById('game-over-card');
        goCard.style.display = s.game_over ? 'block' : 'none';
        if (s.game_over) goCard.textContent = 'Game over';

    }).catch(() => { /* server not ready yet */ });
}

// ── Controls ──────────────────────────────────────────────────────────────
function doReset() {
    fetch('/reset', { method: 'POST' }).then(() => {
        game.reset();
        board.start();
        lastFen = null;
    });
}

function doFlip() {
    fetch('/flip', { method: 'POST' }).then(r => r.json()).then(d => {
        flipped = d.flipped;
        board.orientation(flipped ? 'black' : 'white');
    });
}

function doUndo() {
    fetch('/undo', { method: 'POST' }).then(r => r.json()).then(d => {
        if (d.ok) lastFen = null;   // force board/PV/history refresh from server
    });
}

document.addEventListener('keydown', e => {
    if (document.activeElement === document.getElementById('fen-in')) return;
    if (e.key === 'r') doReset();
    else if (e.key === 'f') doFlip();
    else if (e.key === 'u') doUndo();
});

function doSetFen() {
    const fen = document.getElementById('fen-in').value.trim();
    if (!fen) return;
    fetch('/set_fen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen }),
    }).then(r => r.json()).then(d => {
        if (!d.ok) alert('Invalid FEN: ' + d.error);
        else lastFen = null;
    });
}

document.getElementById('fen-in').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSetFen();
});

// Start polling at 300ms
setInterval(poll, 300);
poll();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print("Loading model...")
    load_model()

    print("Initialising MCTS root...")
    with _lock:
        _mcts.new_root(_board)

    t = threading.Thread(target=_search_worker, daemon=True)
    t.start()
    print("Search thread started.")

    threading.Timer(1.4, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()

    print(f"\nServer: http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)