from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
import gc
import multiprocessing
import os
from pathlib import Path
import pickle
import selectors
import signal
import sys
import traceback
from typing import TypeVar, cast

from nominal_refactor_advisor.scan_cache import ScanCache

WorkItemT = TypeVar("WorkItemT")
WorkResultT = TypeVar("WorkResultT")


def _analysis_process_pool_mp_context() -> multiprocessing.context.BaseContext | None:
    """Use copy-on-write workers for AST-heavy analysis pools on Linux.

    Python 3.14 defaults process pools to forkserver, making every cold worker
    import the full detector registry again and deserialize its own copy of the
    parsed repository.  Analysis pools receive immutable state and are created
    only after earlier parser pools have closed, so Linux fork preserves the
    existing isolation while sharing both the imported detector authority and
    the read-only AST graph.  Other platforms retain their supported default
    start method.
    """

    return (
        multiprocessing.get_context("fork")
        if sys.platform.startswith("linux")
        else None
    )


def _fork_map_batches(
    items: Sequence[WorkItemT],
    worker_count: int,
    run_batch: Callable[[tuple[WorkItemT, ...]], list[WorkResultT]],
) -> list[WorkResultT]:
    """Run round-robin batches across parent + `worker_count - 1` direct forked
    children on Linux, avoiding ProcessPoolExecutor queue/feeder overhead."""

    w = min(worker_count, len(items))
    if w <= 1:
        return run_batch(tuple(items))
    batches = tuple(tuple(items[offset::w]) for offset in range(w))
    froze_here = gc.isenabled() and gc.get_freeze_count() == 0
    if froze_here:
        gc.freeze()
    children: dict[int, tuple[int, int, bytearray]] = {}
    results_by_offset: list[list[WorkResultT] | None] = [None] * w
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for offset in range(1, w):
            reader, writer = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(reader)
                for open_fd in children:
                    os.close(open_fd)
                gc.disable()
                if ScanCache.active() is None:
                    ScanCache.initialize_worker()
                try:
                    try:
                        payload = pickle.dumps((True, run_batch(batches[offset])))
                    except BaseException:  # noqa: BLE001
                        payload = pickle.dumps((False, traceback.format_exc()))
                    with os.fdopen(writer, "wb") as out:
                        out.write(payload)
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os._exit(0)
            os.close(writer)
            children[reader] = (offset, pid, bytearray())

        results_by_offset[0] = run_batch(batches[0])

        with selectors.DefaultSelector() as selector:
            for fd in children:
                selector.register(fd, selectors.EVENT_READ)
            while children:
                for sel_key, _ in selector.select():
                    fd = sel_key.fd
                    offset, pid, buf = children[fd]
                    chunk = os.read(fd, 1 << 20)
                    if chunk:
                        buf += chunk
                        continue
                    selector.unregister(fd)
                    os.close(fd)
                    del children[fd]
                    _, status = os.waitpid(pid, 0)
                    if not buf:
                        raise RuntimeError(
                            f"forked analysis worker exited without result (wait status {status})"
                        )
                    ok, value = pickle.loads(bytes(buf))
                    if not ok:
                        raise RuntimeError(f"forked analysis worker failed:\n{value}")
                    results_by_offset[offset] = value
    except BaseException:
        for fd, (_offset, pid, _buf) in list(children.items()):
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        raise
    finally:
        if froze_here:
            gc.unfreeze()

    ordered: list[WorkResultT] = [cast(WorkResultT, None)] * len(items)
    for offset, batch_results in enumerate(results_by_offset):
        assert batch_results is not None
        ordered[offset::w] = batch_results
    return ordered


@dataclass(frozen=True)
class DetectorAnalysisWorkerPlan:
    """Resolve detector-analysis process parallelism for one scan."""

    requested_worker_count: int
    work_item_count: int
    minimum_auto_work_items: int = 4
    max_auto_worker_count: int = 16

    def map(
        self,
        function: Callable[[WorkItemT], WorkResultT],
        items: Iterable[WorkItemT],
        *,
        initializer: Callable[..., None] = ScanCache.initialize_worker,
        initargs: tuple[object, ...] = (),
    ) -> list[WorkResultT]:
        """Run independent preparation tasks under this plan's worker budget."""
        item_list = list(items)
        if not self.uses_process_pool or len(item_list) <= 1:
            return [function(item) for item in item_list]
        if (
            _analysis_process_pool_mp_context() is not None
            and initializer is ScanCache.initialize_worker
            and not initargs
        ):
            return _fork_map_batches(
                item_list,
                self.effective_worker_count,
                lambda batch: [function(item) for item in batch],
            )
        with self.pool(initializer, initargs) as executor:
            return list(executor.map(function, item_list, chunksize=1))

    @contextmanager
    def pool(
        self, initializer: Callable[..., None], initargs: tuple[object, ...] = ()
    ) -> Iterator[ProcessPoolExecutor]:
        """The plan's worker pool. Workers fork from this process and share its parsed
        tree copy-on-write; the cyclic collector would walk every tracked object in
        each of them and dirty those pages, so the tree is frozen out of its reach
        while the pool runs."""

        froze_here = gc.isenabled() and gc.get_freeze_count() == 0
        if froze_here:
            gc.freeze()
        try:
            with ProcessPoolExecutor(
                max_workers=self.effective_worker_count,
                mp_context=_analysis_process_pool_mp_context(),
                initializer=initializer,
                initargs=initargs,
            ) as executor:
                yield executor
        finally:
            if froze_here:
                gc.unfreeze()

    @property
    def effective_worker_count(self) -> int:
        if self.requested_worker_count == 0:
            if self.work_item_count < self.minimum_auto_work_items:
                return 1
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            return max(
                1,
                min(
                    self.max_auto_worker_count,
                    cpu_count,
                    self.work_item_count,
                    self.memory_worker_ceiling,
                ),
            )
        return max(1, min(self.requested_worker_count, self.work_item_count))

    @cached_property
    def memory_worker_ceiling(self) -> int:
        """How many workers the free memory holds when each, at worst, ends up
        holding its own copy of everything this process holds now (its resident
        size): `MemAvailable` over that, read once so every answer this plan gives
        agrees. It is a floor on the true cost: on NRA's own source each worker held
        about 1.2 times the parent's resident size. Without `/proc` there is no
        ceiling."""

        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
            statm = Path("/proc/self/statm").read_text(encoding="ascii")
        except OSError:
            return self.max_auto_worker_count
        available_kib = next(
            int(line.split()[1])
            for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        )
        resident_kib = int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
        return available_kib // max(1, resident_kib)

    @property
    def uses_process_pool(self) -> bool:
        return self.effective_worker_count > 1

    @property
    def process_map_chunksize(self) -> int:
        if not self.uses_process_pool:
            return 1
        return max(1, self.work_item_count // (self.effective_worker_count * 2))
