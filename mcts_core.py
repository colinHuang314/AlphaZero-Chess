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
        "prior", "visits", "value", "virtual_visits", "virtual_value", "lock",
        "is_tactical",
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

        # Whether the move that created this node (action_taken, played from
        # the parent) was a check, capture, or left the opponent very few
        # replies. Set by expand() when this node is created as a child.
        # Used to gate FORCED_PLAYOUT_MIN_ABSOLUTE -- the absolute floor
        # only cascades to a node's OWN children once the node itself is
        # already inside a forcing line, rather than applying everywhere.
        self.is_tactical = False

    def is_expanded(self) -> bool:
        return len(self.children) > 0

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
        forced_k = self.args.get('FORCED_PLAYOUT_K', 0.0)
        forced_min_absolute = self.args.get('FORCED_PLAYOUT_MIN_ABSOLUTE', 0.0)
        # Only cascade the absolute floor into a node's children if the node
        # itself was reached via a tactical move (check/capture/narrow-reply)
        # -- i.e. we're already inside a forcing line. A blanket floor
        # applied everywhere guarantees depth in quiet positions too, which
        # wastes sim budget the search doesn't need to spend there.
        apply_absolute_floor = forced_min_absolute > 0 and self.is_tactical
        parent_visits = max(1.0, self.visits)

        for c in self.children:
            if forced_k > 0 or apply_absolute_floor:
                with c.lock:
                    child_visits = c.visits
                # Forced playouts (KataGo/Leela-style): guarantee each child
                # a minimum visit count proportional to sqrt(prior *
                # parent_visits), not just the single obligatory visit a
                # fresh (visits==0) child already gets from get_ucb's inf
                # case. Without this, a low-prior move that looks bad after
                # one shallow visit never earns enough follow-up visits to
                # reach a tactic a few plies deeper -- its exploration
                # bonus stays small forever regardless of what's actually
                # down that line.
                forced_target = forced_k * math.sqrt(max(c.prior, 0.0) * parent_visits)
                if apply_absolute_floor:
                    # Non-prior-scaled floor, cascading only through an
                    # already-forcing line. The prior-scaled target alone
                    # still collapses toward zero for a genuinely low-prior
                    # move -- e.g. the quiet link in a forced mating
                    # sequence that doesn't look special on its own -- since
                    # sqrt(tiny_prior * N) stays tiny no matter how large N
                    # gets. Capped by parent_visits so it can never force
                    # more visits than the parent itself has accumulated.
                    forced_target = max(forced_target, min(parent_visits, forced_min_absolute))
                ucb = float('inf') if child_visits < forced_target else self.get_ucb(c)
            else:
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
        check_min_prior = self.args.get('TACTICAL_CHECK_MIN_PRIOR', 0.0)
        capture_min_prior = self.args.get('TACTICAL_CAPTURE_MIN_PRIOR', 0.0)
        one_reply_min_prior = self.args.get('TACTICAL_ONE_REPLY_MIN_PRIOR', 0.0)
        two_reply_min_prior = self.args.get('TACTICAL_TWO_REPLY_MIN_PRIOR', 0.0)
        forced_min_absolute = self.args.get('FORCED_PLAYOUT_MIN_ABSOLUTE', 0.0)
        out_of_tb_range = len(board.piece_map()) > _TABLEBASE_MAX_PIECES

        move_priors = []
        for m in legal:
            idx = move_to_index(m, board)
            p = float(probs[idx])
            # Separate floors for checks vs. captures -- give checks/captures
            # a higher exploration floor than ordinary moves get from
            # MIN_PRIOR alone, since both types are prone to being starved
            # of exploration budget when the network underrates them. This
            # doesn't renormalize the rest of the distribution back down --
            # it's an additive nudge on top of whatever the network already
            # believes, not a redistribution. A move that's both a check and
            # a capture gets the higher of the two floors, not both added.
            is_check = board.gives_check(m)
            is_capture_move = board.is_capture(m)
            # Also compute reply count when FORCED_PLAYOUT_MIN_ABSOLUTE is
            # active, even if the explicit narrow-reply prior floors aren't
            # set -- the tactical flag it feeds needs narrow-reply detection
            # independent of whether those specific floors are configured.
            needs_reply_count = out_of_tb_range and (
                (check_min_prior > 0 and is_check)
                or one_reply_min_prior > 0
                or two_reply_min_prior > 0
                or forced_min_absolute > 0
            )

            n_replies = None
            if needs_reply_count:
                board.push(m)
                n_replies = board.legal_moves.count()
                board.pop()

            if check_min_prior > 0 and is_check and out_of_tb_range:
                # Scale by how forcing the check actually is, rather than a
                # flat floor -- a check the opponent can only answer one way
                # deserves more guaranteed exploration than one they have a
                # dozen replies to. This is a genuinely different signal
                # from material value, so unlike curving captures by piece
                # value, it doesn't quietly reintroduce the "grade by how
                # good it looks" bias the floor exists to bypass.
                forcing_scale = 1.0 / max(1, (n_replies ** 1.4) * 0.7)
                p = max(p, check_min_prior * forcing_scale)

            if capture_min_prior > 0 and is_capture_move and out_of_tb_range:
                p = max(p, capture_min_prior)

            if out_of_tb_range and n_replies is not None:
                # Catches forcing moves that AREN'T themselves a check or
                # capture -- e.g. the quiet middle move of a mating
                # sequence that sets up the actual mate two plies later
                # without directly threatening anything this ply. Checks
                # and captures already get their own floor above; these
                # exist specifically for the gap between them, where a move
                # constrains the opponent to almost nothing but has no
                # other signal marking it as tactically relevant. Two
                # independent constants (rather than one floor + a
                # threshold) so a nearly-forced move (1 reply) and a
                # merely-narrow one (2 replies) can be weighted completely
                # separately instead of sharing a single curve.
                if n_replies == 1 and one_reply_min_prior > 0:
                    p = max(p, one_reply_min_prior)
                elif n_replies == 2 and two_reply_min_prior > 0:
                    p = max(p, two_reply_min_prior)

            move_is_tactical = (
                is_check or is_capture_move or (n_replies is not None and n_replies <= 2)
            )
            move_priors.append((m, p, move_is_tactical))

        if self.parent is None:
            alpha = self.args.get('DIRICHLET_ALPHA', 0.0)
            eps = self.args.get('DIRICHLET_EPSILON', 0.0)
            if eps > 0 and alpha > 0 and len(move_priors) > 0:
                noise = np.random.dirichlet([alpha] * len(move_priors))
                for i, (m, p, tac) in enumerate(move_priors):
                    move_priors[i] = (m, (1 - eps) * p + eps * float(noise[i]), tac)

        for m, p, tac in move_priors:
            if p <= 0:
                continue
            child = MCTSNode(self.args, action_taken=m, parent=self, prior=float(p))
            child.is_tactical = tac
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


def pruned_action_probs(mcts: "MCTS") -> np.ndarray:
    """
    KataGo-style policy target pruning -- a pragmatic approximation, not
    the exact utility-matching binary search from the paper, but the same
    spirit: strip the portion of each non-best child's visit count that
    forced playouts (FORCED_PLAYOUT_K's prior-scaled target AND
    FORCED_PLAYOUT_MIN_ABSOLUTE's flat floor) guaranteed it regardless of
    merit, before turning visit counts into a policy TRAINING target.

    Use this only for what gets stored as the supervised label. The move
    actually played should still benefit from forced exploration finding a
    hidden tactic -- keep using mcts.action_probs() (unpruned) for
    temperature sampling / move selection. This function exists so that a
    forced "let's double check this sacrifice" visit that didn't pan out
    doesn't also teach the network a generic bias toward checks/captures,
    since the training target is a different consumer than move choice --
    one wants "what did search conclude", the other wants "what should
    still be explored".

    The single most-visited child is never pruned -- its visit count
    reflects what the search actually concluded was best, forced or not.
    """
    probs = np.zeros(ACTION_SIZE, dtype=np.float64)
    root = mcts.root
    if root is None or mcts.root_board is None or not root.children:
        return probs

    forced_k = mcts.args.get('FORCED_PLAYOUT_K', 0.0)
    forced_min_absolute = mcts.args.get('FORCED_PLAYOUT_MIN_ABSOLUTE', 0.0)
    apply_absolute_floor = forced_min_absolute > 0 and root.is_tactical
    parent_visits = max(1.0, root.visits)
    best = max(root.children, key=lambda c: c.visits)

    pruned = {}
    for c in root.children:
        if c is best or (forced_k <= 0 and not apply_absolute_floor):
            pruned[c] = c.visits
            continue
        forced_target = forced_k * math.sqrt(max(c.prior, 0.0) * parent_visits)
        if apply_absolute_floor:
            forced_target = max(forced_target, min(parent_visits, forced_min_absolute))
        pruned[c] = max(0.0, c.visits - forced_target)

    total = sum(pruned.values())
    if total <= 0:
        # Everything but the best move got pruned to zero -- fall back to
        # a one-hot target on the best move rather than an all-zero vector.
        idx = move_to_index(best.action_taken, mcts.root_board)
        probs[idx] = 1.0
        return probs

    for c, v in pruned.items():
        if v <= 0:
            continue
        idx = move_to_index(c.action_taken, mcts.root_board)
        probs[idx] = v / total

    return probs

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

            term_value = terminal_value(board, claim_draw=claim_draw)
            if term_value is not None:
                value = term_value
                # no expand() call -- leaf.children stays empty, is_expanded() stays False
            else:
                policy_logits = policy_batch[k]
                probs = softmax_legal(np.asarray(policy_logits), board, min_prior)
                leaf.expand(probs, board)

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
        return 1.4

    elif N <= 300:
        # 100: 1.4 → 300: 1.7
        return 1.4 + (N - 100) * (1.7 - 1.4) / (300 - 100)

    elif N <= 600:
        # 300: 1.7 → 600: 2.0
        return 1.7 + (N - 300) * (2.0 - 1.7) / (600 - 300)

    elif N <= 2000:
        # 600: 2.0 → 2000: 3.0
        return 2.0 + (N - 600) * (3.0 - 2.0) / (2000 - 600)

    else:
        return 3.0