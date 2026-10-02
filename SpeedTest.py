"""
SpeedTest.py

Benchmarks raw mcts_core throughput (sims/sec) with a completely random
policy/value "network" -- no model, no GPU, no torch needed at all. This
isolates pure search/tree-traversal + selection overhead from anything to
do with actual network inference cost, so you can measure what mcts_core
itself can do on your CPU, independent of model size or GPU speed.

Moves are picked uniformly at random (not by search quality) purely to
advance games and keep the pool full -- this script measures throughput,
not move quality.
"""

import time
import random
import numpy as np
import chess

from Encoder import ACTION_SIZE
from mcts_core import MCTS, batch_search

# ── Config ────────────────────────────────────────────────────────────────
POOL_SIZE       = 32     # parallel games, matches SELFPLAY_GAMES_IN_PARALELL
MCTS_SIMS       = 300    # sims per move
TEST_SECONDS    = 20     # how long to run before printing the final report
MAX_BATCH_SIZE  = 256
MAX_WAIT_S      = 0.01
MAX_GAME_LENGTH = 200    # half-moves safety cap, matches self-play

ARGS = {
    'CPUCT':              1.8,
    'DIRICHLET_ALPHA':    0.3,
    'DIRICHLET_EPSILON':  0.25,
    'MIN_PRIOR':          1e-3,
    'VIRTUAL_LOSS':       1.0,
    'VIRTUAL_LOSS_VALUE': -1.0,
    # Leave FORCED_PLAYOUT_K / TACTICAL_* / FORCED_PLAYOUT_MIN_ABSOLUTE unset
    # (defaults to 0, i.e. off) for a baseline throughput number. Add them
    # here with your real training values if you want to measure the
    # throughput COST of those features specifically, by comparing against
    # a run with them left at 0.
}


# Set True to profile instead of just reporting sims/sec. Profiling adds
# real overhead of its own, so use a shorter duration for this than the
# plain throughput run above -- you're looking at RELATIVE time between
# functions here, not an accurate absolute sims/sec number.
PROFILE          = True
PROFILE_SECONDS  = 10


def random_infer_batch_fn(batch_items):
    """
    Ignores the actual encoded boards entirely -- returns uniform-random
    policy logits and random values in [-1, 1]. This is the whole point:
    zero model/GPU cost, so whatever sims/sec this reports is purely
    mcts_core's own overhead (tree traversal, virtual loss, expand/backprop
    bookkeeping), not anything about your network's inference speed.
    """
    n = len(batch_items)
    logits = np.random.randn(n, ACTION_SIZE).astype(np.float32)
    values = (np.random.rand(n).astype(np.float32) * 2 - 1)
    return logits, values


def _run_loop(duration_s, verbose=True):
    """The actual test loop, factored out so both run_speed_test() and
    run_profiled_speed_test() share identical logic -- profiling a
    DIFFERENT code path than the one you actually measure defeats the
    point."""
    boards    = [chess.Board() for _ in range(POOL_SIZE)]
    mcts_list = [MCTS(ARGS)    for _ in range(POOL_SIZE)]
    move_nums = [1] * POOL_SIZE

    for mcts, board in zip(mcts_list, boards):
        mcts.new_root(board)

    total_sims      = 0
    total_moves     = 0
    games_completed = 0

    start = time.time()
    last_print = start

    while time.time() - start < duration_s:
        batch_search(
            mcts_list, MCTS_SIMS, random_infer_batch_fn,
            max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_S,
            min_prior=ARGS['MIN_PRIOR'],
        )

        for i, mcts in enumerate(mcts_list):
            board = boards[i]
            move = random.choice(list(board.legal_moves))
            board.push(move)
            mcts.advance_root(move)
            move_nums[i] += 1
            total_moves += 1
            total_sims += MCTS_SIMS

            if board.outcome(claim_draw=True) is not None or move_nums[i] > MAX_GAME_LENGTH:
                games_completed += 1
                boards[i]    = chess.Board()
                mcts_list[i] = MCTS(ARGS)
                mcts_list[i].new_root(boards[i])
                move_nums[i] = 1

        if verbose:
            now = time.time()
            if now - last_print >= 2.0:
                elapsed = now - start
                print(f"  {elapsed:5.1f}s | {total_sims/elapsed:8.1f} sims/sec | "
                      f"{total_moves/elapsed:6.1f} moves/sec | {games_completed} games completed")
                last_print = now

    elapsed = time.time() - start
    return total_sims, total_moves, games_completed, elapsed


def run_speed_test():
    print(f"Running for {TEST_SECONDS}s with {POOL_SIZE} parallel games, "
          f"{MCTS_SIMS} sims/move, random policy/value (no model)...\n")

    total_sims, total_moves, games_completed, elapsed = _run_loop(TEST_SECONDS, verbose=True)

    print(f"\nDone. {elapsed:.1f}s elapsed.")
    print(f"Total sims:       {total_sims:,}")
    print(f"Total moves:      {total_moves:,}")
    print(f"Games completed:  {games_completed}")
    print(f"Average:          {total_sims/elapsed:,.1f} sims/sec across {POOL_SIZE} games")


def run_profiled_speed_test():
    """
    Same test, wrapped in cProfile. Prints two sorted views:
      - cumulative time: best for finding big outer bottlenecks (which
        function's whole call tree, including what it calls, eats the most
        wall-clock time)
      - tottime: best for finding hot LEAF functions specifically (time
        spent in that function's own code, excluding what it calls out to
        -- this is where board.copy, legal_moves, push/pop, select_child's
        UCB loop etc. will show up)
    Also dumps a .prof file for a visual flame-graph view via snakeviz:
        pip install snakeviz
        snakeviz mcts_core_speedtest_profile.prof
    """
    import cProfile
    import pstats

    print(f"Profiling for {PROFILE_SECONDS}s with {POOL_SIZE} parallel games, "
          f"{MCTS_SIMS} sims/move, random policy/value (no model)...\n"
          f"(profiling overhead means the sims/sec number below isn't a real "
          f"throughput measurement -- use the plain, unprofiled run for that.)\n")

    profiler = cProfile.Profile()
    profiler.enable()
    total_sims, total_moves, games_completed, elapsed = _run_loop(PROFILE_SECONDS, verbose=False)
    profiler.disable()

    print(f"Done. {elapsed:.1f}s elapsed. {total_sims:,} sims, {total_moves:,} moves, "
          f"{games_completed} games completed.\n")

    stats = pstats.Stats(profiler)

    print("=" * 80)
    print("TOP 25 BY CUMULATIVE TIME (best for finding big outer bottlenecks)")
    print("=" * 80)
    stats.sort_stats('cumulative').print_stats(25)

    print("=" * 80)
    print("TOP 25 BY TOTAL TIME / tottime (best for finding hot leaf functions)")
    print("=" * 80)
    stats.sort_stats('tottime').print_stats(25)

    prof_path = "mcts_core_speedtest_profile.prof"
    profiler.dump_stats(prof_path)
    print(f"\nSaved profile to '{prof_path}'.")

    try:
        from snakeviz.cli import main as snakeviz_main
        import sys
        print("Launching snakeviz in your browser...")
        sys.argv = ["snakeviz", prof_path]
        snakeviz_main()
    except ImportError:
        print("snakeviz not installed. Run 'pip install snakeviz', then:")
        print(f"  python -m snakeviz {prof_path}")


if __name__ == "__main__":
    if PROFILE:
        run_profiled_speed_test()
    else:
        run_speed_test()