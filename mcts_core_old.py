"""
mcts_core.py

Optimized MCTS with:
- Shared board per MCTS instance (push/pop, no per-node board copies)
- Lightweight nodes (move, parent, children, priors, stats, virtual loss, lock)
- Virtual loss to safely reserve nodes during multi-leaf selection
- Multi-leaf batching per game and multi-game batching
- No tablebase, no generator API
"""

import time
import math
import random
import threading
from collections import deque, defaultdict
from typing import Callable, List, Tuple, Optional

import numpy as np
import chess
import copy

# -------------------------
# Tablebase (Syzygy WDL/DTZ) support
# -------------------------
try:
    import chess.syzygy as syzygy
except Exception:
    syzygy = None

_TABLEBASE = None
_TABLEBASE_MAX_PIECES = 5


def init_tablebase(path, max_pieces: int = 5) -> bool:
    r"""
    Open one or more directories of Syzygy tablebase files (.rtbw and/or
    .rtbz) for use at leaf evaluation. Call once at startup, e.g.:

        from mcts_core import init_tablebase

        # Single directory containing both .rtbw and .rtbz files:
        init_tablebase(r"...\syzygy\Syzygy345WDL", max_pieces=5)

        # Or, if WDL and DTZ files live in separate subfolders (this does
        # NOT recurse into subfolders on its own, so pass them explicitly):
        init_tablebase([
            r"...\syzygy\WDL",
            r"...\syzygy\DTZ",
        ], max_pieces=5)

    Safe to call with path=None/""/[] to leave the tablebase disabled --
    probing then silently no-ops and mcts_core falls back to the network's
    own value everywhere.

    Returns True if at least one directory was opened successfully.
    """
    global _TABLEBASE, _TABLEBASE_MAX_PIECES
    _TABLEBASE_MAX_PIECES = max_pieces

    if not path or syzygy is None:
        _TABLEBASE = None
        return False

    paths = [path] if isinstance(path, str) else list(path)

    tb = None
    any_loaded = False
    for p in paths:
        if not p:
            continue
        try:
            if tb is None:
                tb = syzygy.open_tablebase(p)
            else:
                tb.add_directory(p)
            print(f"[mcts_core] Tablebase directory loaded: {p}")
            any_loaded = True
        except Exception as e:
            print(f"[mcts_core] Failed to load tablebase directory {p}: {e}")

    if not any_loaded:
        _TABLEBASE = None
        return False

    _TABLEBASE = tb
    print(f"[mcts_core] Tablebase ready (<= {max_pieces} pieces)")
    return True


def _dtz_to_value(dtz: int) -> float:
    """Map a DTZ (distance-to-zeroing) count to a value in (-1, 1). Closer
    wins/losses are scored with more confidence than distant ones, per the
    project's dtz_to_value convention (0.5 + 0.49 * exp(-|dtz| / 25))."""
    if dtz == 0:
        return 0.0
    sign = 1.0 if dtz > 0 else -1.0
    return sign * (0.5 + 0.49 * math.exp(-abs(dtz) / 25.0))


def probe_tablebase_value(board: chess.Board) -> Optional[float]:
    """
    Ground-truth leaf value from the perspective of the side to move, or
    None if the tablebase isn't loaded or this position is out of its range
    (too many pieces, castling rights still available, etc -- Syzygy tables
    assume castling has already been resolved).

    Prefers DTZ (if .rtbz files are available) since it also grades how
    close the win/loss is; falls back to plain WDL (win/draw/loss only).
    """
    if _TABLEBASE is None:
        return None
    if chess.popcount(board.occupied) > _TABLEBASE_MAX_PIECES:
        return None
    if board.castling_rights:
        return None

    try:
        dtz = _TABLEBASE.probe_dtz(board)
        return _dtz_to_value(dtz)
    except Exception:
        pass

    try:
        wdl = _TABLEBASE.probe_wdl(board)  # -2..2, from side-to-move perspective
        if wdl > 0:
            return 1.0
        elif wdl < 0:
            return -1.0
        else:
            return 0.0
    except Exception:
        return None


def terminal_value(board: chess.Board, claim_draw: bool = True) -> Optional[float]:
    """
    Rules-based ground truth (side-to-move perspective) if `board` is
    already a completed game: checkmate, stalemate, insufficient material,
    fivefold repetition / 75-move draw, and -- with claim_draw=True (the
    default) -- the *claimable* threefold-repetition and 50-move draws too.

    Outranks both the tablebase and the network -- there's no more certain
    signal than "0 legal moves", "K vs K", or "this exact position has now
    occurred 3 times", so it should always win when it applies.

    claim_draw=True costs more per call (repetition detection scans the
    move stack, a measured hotspot in the old MCTSChess profiling), but
    without it the search's own simulated lines never see a repeated
    position as a forced draw -- it'll happily walk toward one it still
    "likes" per the network, since nothing tells it otherwise. That's worse
    than the CPU cost: pass claim_draw=False here only if you've confirmed
    it's a real bottleneck for your setup and are OK with the search being
    blind to repetition-based draws in its own lookahead.
    """
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0

# -------------------------
# Encoder / action helpers (fallbacks)
# -------------------------
try:
    from Encoder import encode_board, move_to_index, ACTION_SIZE, legal_moves_mask  # type: ignore
except Exception:
    ACTION_SIZE = 4096

    def encode_board(board: chess.Board):
        # Placeholder: logits zeros, value 0.0
        return np.zeros(ACTION_SIZE, dtype=np.float32), 0.0

    def move_to_index(move: chess.Move, board: chess.Board) -> int:
        return abs(hash(move.uci())) % ACTION_SIZE

    def legal_moves_mask(board: chess.Board) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=np.int32)
        for mv in board.legal_moves:
            mask[move_to_index(mv, board)] = 1
        return mask

# -------------------------
# Utility: softmax over legal moves
# -------------------------
def softmax_legal(logits: np.ndarray, board: chess.Board, min_prior: float) -> np.ndarray:
    mask = legal_moves_mask(board)
    exp = np.exp(logits - np.max(logits))
    probs = exp / (exp.sum() + 1e-12)
    probs = probs * mask
    if min_prior > 0:
        probs = np.where(mask > 0, np.clip(probs, min_prior, None), 0.0)
    s = probs.sum()
    if s <= 0:
        legal = np.where(mask > 0)[0]
        if legal.size == 0:
            return probs
        probs[legal] = 1.0 / legal.size
        return probs
    return probs / s

# -------------------------
# MCTSNode
# -------------------------
class MCTSNode:
    __slots__ = (
        "args", "action_taken", "parent", "children",
        "prior", "visits", "value", "virtual_visits", "virtual_value", "lock"
    )

    def __init__(self, args: dict, action_taken: Optional[chess.Move] = None, parent: Optional["MCTSNode"] = None, prior: float = 0.0):
        self.args = args
        self.action_taken = action_taken
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.prior = float(prior)

        self.visits = 0.0
        self.value = 0.0

        self.virtual_visits = 0.0
        self.virtual_value = 0.0

        self.lock = threading.Lock()

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    # normal
    def get_ucb(self, child: "MCTSNode") -> float:
        with child.lock:
            child_visits = child.visits
            child_value = child.value
        if child_visits == 0:
            return float('inf')
        q = -(child_value / child_visits)
        parent_visits = max(1.0, self.visits)
        u = self.args.get('CPUCT', 1.5) * child.prior * math.sqrt(parent_visits) / (1.0 + child_visits)
        return q + u
    

    def select_child(self) -> "MCTSNode":
        best_ucb = -float('inf')
        best_children: List[MCTSNode] = []
        for c in self.children:
            ucb = self.get_ucb(c)
            if ucb > best_ucb:
                best_ucb = ucb
                best_children = [c]
            elif ucb == best_ucb:
                best_children.append(c)
        return random.choice(best_children)

    def expand(self, probs: np.ndarray, board: chess.Board):
        if self.children:
            # Already expanded -- this leaf was reserved more than once in the
            # same multi-leaf collection round (possible when few siblings
            # exist to route virtual loss around). Re-expanding would append
            # duplicate children and inflate visit counts, so no-op here;
            # the caller still backpropagates the value it computed.
            return
        legal = list(board.legal_moves)
        if not legal:
            return
        move_priors = []
        for m in legal:
            idx = move_to_index(m, board)
            move_priors.append((m, float(probs[idx])))

        if self.parent is None:
            alpha = self.args.get('DIRICHLET_ALPHA', 0.0)
            eps = self.args.get('DIRICHLET_EPSILON', 0.0)
            if eps > 0 and alpha > 0 and len(move_priors) > 0:
                noise = np.random.dirichlet([alpha] * len(move_priors))
                for i, (m, p) in enumerate(move_priors):
                    move_priors[i] = (m, (1 - eps) * p + eps * float(noise[i]))

        for m, p in move_priors:
            if p <= 0:
                continue
            child = MCTSNode(self.args, action_taken=m, parent=self, prior=float(p))
            self.children.append(child)

    def add_root_noise(self):
        if not self.children:
            return
        alpha = self.args.get('DIRICHLET_ALPHA', 0.0)
        eps = self.args.get('DIRICHLET_EPSILON', 0.0)
        if eps <= 0 or alpha <= 0:
            return
        noise = np.random.dirichlet([alpha] * len(self.children))
        for child, n in zip(self.children, noise):
            with child.lock:
                child.prior = (1 - eps) * child.prior + eps * float(n)

    def apply_virtual_loss(self, vl: float, vl_value: float):
        with self.lock:
            self.visits += vl
            self.value += vl_value
            self.virtual_visits += vl
            self.virtual_value += vl_value

    def remove_virtual_loss(self):
        with self.lock:
            self.visits -= self.virtual_visits
            self.value -= self.virtual_value
            self.virtual_visits = 0.0
            self.virtual_value = 0.0

    def backpropagate_threadsafe(self, value: float):
        with self.lock:
            self.value += value
            self.visits += 1.0
        if self.parent is not None:
            self.parent.backpropagate_threadsafe(-value)

# -------------------------
# MCTS (single game)
# -------------------------
class MCTS:
    def __init__(self, args: dict, debug: bool = False):
        self.args = args
        self.debug = debug
        self.root: Optional[MCTSNode] = None
        self.root_board: Optional[chess.Board] = None

    def new_root(self, board: chess.Board):
        self.root = MCTSNode(self.args)
        try:
            # Full move-stack copy, not a truncated one: every other call site
            # (fresh self-play games, /reset, /set_fen) passes a brand-new
            # chess.Board() with zero prior moves anyway, so this costs
            # nothing there. The one place it isn't fresh -- rebuilding after
            # an undo -- is exactly the place a truncated stack silently
            # broke threefold-repetition detection: the moves proving a
            # repetition happened could fall outside the kept window,
            # making the position look repetition-free when it wasn't.
            self.root_board = board.copy(stack=True)
        except Exception:
            self.root_board = copy.deepcopy(board)

    def advance_root(self, move: chess.Move):
        if self.root is None or self.root_board is None:
            raise RuntimeError("advance_root called before new_root")
        self.root_board.push(move)
        for child in self.root.children:
            if child.action_taken == move:
                child.parent = None
                self.root = child
                self.root.add_root_noise()
                return
        self.root = MCTSNode(self.args)

    def action_probs(self) -> np.ndarray:
        probs = np.zeros(ACTION_SIZE, dtype=np.float64)
        if self.root is None or self.root_board is None:
            return probs
        for child in self.root.children:
            idx = move_to_index(child.action_taken, self.root_board)
            probs[idx] = child.visits
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs

    def best_move(self) -> chess.Move:
        if self.root is None:
            raise RuntimeError("best_move called before search")
        best = None
        best_visits = -1.0
        for c in self.root.children:
            if c.visits > best_visits:
                best_visits = c.visits
                best = c
        if best is None:
            raise RuntimeError("no children to choose from")
        return best.action_taken

# -------------------------
# Selection traversal
# -------------------------
def select_and_reserve_leaf(root: MCTSNode, board: chess.Board, vl: float, vl_value: float) -> Tuple[MCTSNode, List[MCTSNode]]:
    """
    Assumes board is at root when called.
    Descend using UCB, apply virtual loss, push moves, return leaf and path.
    Board is left at leaf; caller must undo_path to return to root.
    """
    node = root
    path: List[MCTSNode] = []
    while node.is_expanded():
        child = node.select_child()
        child.apply_virtual_loss(vl, vl_value * vl)
        path.append(child)
        board.push(child.action_taken)
        node = child
    return node, path

def undo_path(board: chess.Board, path: List[MCTSNode]):
    for _ in path:
        board.pop()

# -------------------------
# Batch search (multi-instance, multi-leaf)
# -------------------------
def batch_search(
    mcts_list: List[MCTS],
    num_simulations: int,
    infer_batch_fn: Callable[[List[Tuple[np.ndarray, float]]], Tuple[np.ndarray, np.ndarray]],
    max_batch_size: int = 64,
    max_wait_s: float = 0.005,
    min_prior: float = 1e-6,
    verbose: bool = False,
    claim_draw: bool = True,
) -> List[Tuple[np.ndarray, float]]:
    n = len(mcts_list)
    sims_done = [0] * n

    VIRTUAL_LOSS = float(mcts_list[0].args.get('VIRTUAL_LOSS', 1.0)) if n > 0 else 1.0
    VIRTUAL_LOSS_VALUE = float(mcts_list[0].args.get('VIRTUAL_LOSS_VALUE', -1.0)) if n > 0 else -1.0

    queue = deque()

    while True:
        all_done = True
        for i in range(n):
            if sims_done[i] < num_simulations:
                all_done = False
                break
        if all_done and not queue:
            break

        start_collect = time.time()
        while len(queue) < max_batch_size and (time.time() - start_collect) < max_wait_s:
            made_progress = False
            for i, mcts in enumerate(mcts_list):
                if sims_done[i] >= num_simulations:
                    continue
                root = mcts.root
                board = mcts.root_board
                if root is None or board is None:
                    sims_done[i] = num_simulations
                    continue

                # board must be at root here
                leaf, path = select_and_reserve_leaf(root, board, VIRTUAL_LOSS, VIRTUAL_LOSS_VALUE)
                # encode at leaf
                encoded = encode_board(board)
                # immediately undo path to return board to root for next selection
                undo_path(board, path)

                queue.append((i, leaf, path, encoded))
                made_progress = True
                if len(queue) >= max_batch_size:
                    break
            if not made_progress:
                break

        if not queue:
            continue

        batch_items = []
        batch_map = []
        per_game_counts = defaultdict(int)
        while queue and len(batch_items) < max_batch_size:
            i, leaf, path, encoded = queue.popleft()
            batch_items.append(encoded)
            batch_map.append((i, leaf, path))
            per_game_counts[i] += 1

        t0 = time.time()
        policy_batch, value_batch = infer_batch_fn(batch_items)
        batch_latency = time.time() - t0

        for k, (i, leaf, path) in enumerate(batch_map):
            # replay path: board at root, push moves again
            board = mcts_list[i].root_board
            for node in path:
                board.push(node.action_taken)

            # remove virtual loss
            for node in path:
                node.remove_virtual_loss()

            # --- FIX: check terminal BEFORE expansion ---
            term_value = terminal_value(board, claim_draw=claim_draw)
            if term_value is not None:
                # terminal: DO NOT expand
                value = term_value
            else:
                # non-terminal: safe to expand
                policy_logits = policy_batch[k]
                probs = softmax_legal(np.asarray(policy_logits), board, min_prior)
                leaf.expand(probs, board)

                # tablebase > NN
                tb_value = probe_tablebase_value(board)
                value = tb_value if tb_value is not None else float(value_batch[k])

            leaf.backpropagate_threadsafe(value)

            # pop back to root
            undo_path(board, path)

            sims_done[i] += 1

        if verbose:
            contrib_summary = ", ".join(f"{gi}:{cnt}" for gi, cnt in sorted(per_game_counts.items()))
            print(f"[batch] size={len(batch_items)} latency={batch_latency:.4f}s ms/sim={batch_latency*1000.0/len(batch_items):.3f} contribs={contrib_summary}")

    outputs = []
    for mcts in mcts_list:
        outputs.append((mcts.action_probs(), _best_move_value(mcts)))
    return outputs

# -------------------------
# Single-instance batched search
# -------------------------
def search_single_batched(
    mcts: MCTS,
    num_simulations: int,
    infer_batch_fn: Callable[[List[Tuple[np.ndarray, float]]], Tuple[np.ndarray, np.ndarray]],
    max_batch_size: int = 8,
    max_wait_s: float = 0.005,
    min_prior: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    outputs = batch_search([mcts], num_simulations, infer_batch_fn, max_batch_size=max_batch_size, max_wait_s=max_wait_s, min_prior=min_prior, verbose=verbose)
    return outputs[0]

# -------------------------
# Helper: best move value
# -------------------------
def _best_move_value(mcts: MCTS) -> float:
    root = mcts.root
    if root is None:
        return 0.0
    best = None
    best_visits = -1.0
    for c in root.children:
        if c.visits > best_visits:
            best_visits = c.visits
            best = c
    if best is None or best_visits <= 0:
        return 0.0
    return -best.value / best_visits

# other

def cpuct_from_sims(n_sims):
    N = n_sims

    if N <= 100:
        return 1.5

    elif N <= 300:
        # 100: 1.5 → 300: 2.0
        return 1.5 + (N - 100) * (2.0 - 1.5) / (300 - 100)

    elif N <= 600:
        # 300: 2.0 → 600: 2.5
        return 2.0 + (N - 300) * (2.5 - 2.0) / (600 - 300)

    elif N <= 2000:
        # 600: 2.5 → 2000: 3.0
        return 2.5 + (N - 600) * (3.0 - 2.5) / (2000 - 600)

    else:
        return 3.0
    # else:
    #     # Beyond 2000, grow slowly in log-space from 3.0
    #     return 3.0 * (1 + 0.1 * math.log(N / 2000.0))