"""Bound a useful profile of the unchanged original R1 consumer, not a pass.

SIGPROF bounds CPU profiling independently of NRA's unchanged real-time scan
deadline. Call under an outer60s/512MiB/oneCPU supervisor with a new owned cache.
The consumer still stages both exact revisions and its complete source context.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import runpy
import signal
import sys
import traceback
from pathlib import Path


class PreparationProfileBudgetReached(BaseException):
    """Diagnostic terminal condition; never a successful audit result."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--cpu-seconds", type=float, default=20.0)
    parser.add_argument("consumer_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    import nominal_refactor_advisor

    assert Path(nominal_refactor_advisor.__file__).is_relative_to(source_root)
    sys.argv = [str(args.consumer), *args.consumer_arguments]
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        del sys.argv[1]
    profiler = cProfile.Profile()

    def finish_profile(_signal, frame):
        traceback.print_stack(frame, file=sys.stderr)
        raise PreparationProfileBudgetReached()

    signal.signal(signal.SIGPROF, finish_profile)
    signal.setitimer(signal.ITIMER_PROF, args.cpu_seconds)
    profiler.enable()
    try:
        runpy.run_path(str(args.consumer), run_name="__main__")
    except PreparationProfileBudgetReached:
        print("INCOMPLETE: diagnostic CPU profiling bound reached", file=sys.stderr)
        return 75
    finally:
        profiler.disable()
        signal.setitimer(signal.ITIMER_PROF, 0.0)
        profiler.dump_stats(str(args.profile))
        pstats.Stats(profiler, stream=sys.stderr).sort_stats("cumulative").print_stats(
            50
        )
        pstats.Stats(profiler, stream=sys.stderr).sort_stats("tottime").print_stats(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
