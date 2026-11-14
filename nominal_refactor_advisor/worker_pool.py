"""Process pools for one scan: how many workers the work and the free memory
hold, and the state each worker holds for one map."""

from __future__ import annotations

import gc
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, cast

from nominal_refactor_advisor.scan_cache import ScanCache


WorkItemT = TypeVar("WorkItemT")
WorkResultT = TypeVar("WorkResultT")


@dataclass(frozen=True)
class WorkerPlan:
    """How many pool workers one stage of a scan runs: what the work and the free
    memory hold, or what the caller asked for."""

    requested_worker_count: int
    work_item_count: int
    minimum_auto_work_items: int = 4
    max_auto_worker_count: int = 16

    def map(
        self,
        function: Callable[[WorkItemT], WorkResultT],
        items: Iterable[WorkItemT],
    ) -> list[WorkResultT]:
        """Run independent preparation tasks under this plan's worker budget."""
        if not self.uses_process_pool:
            return [function(item) for item in items]
        with self.pool(ScanCache.initialize_worker) as executor:
            return list(executor.map(function, items, chunksize=1))

    def map_state(
        self,
        state: PoolWorkerState[WorkItemT, WorkResultT],
        items: Iterable[WorkItemT],
    ) -> list[WorkResultT]:
        """Run `state` over `items` under this plan's worker budget. Each worker
        installs the state once; a task carries only its item."""
        if not self.uses_process_pool:
            return [state.run(item) for item in items]
        with self.pool(PoolWorkerState.install, (state,)) as executor:
            return [
                cast(WorkResultT, result)
                for result in executor.map(
                    PoolWorkerState.run_installed,
                    items,
                    chunksize=self.process_map_chunksize,
                )
            ]

    @contextmanager
    def pool(
        self, initializer: Callable[..., None], initargs: tuple[object, ...] = ()
    ) -> Iterator[ProcessPoolExecutor]:
        """The plan's worker pool. Workers fork from this process and share its parsed
        tree copy-on-write; the cyclic collector would walk every tracked object in
        each of them and dirty those pages, so the tree is frozen out of its reach
        while the pool runs."""

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
            int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:")
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


class PoolWorkerState(ABC, Generic[WorkItemT, WorkResultT]):
    """What a pool's workers hold for one map: installed once per worker by the
    plan's initializer, so the parsed source and config cross the process
    boundary once, never per task."""

    installed: ClassVar[PoolWorkerState[object, object] | None] = None

    @abstractmethod
    def run(self, item: WorkItemT) -> WorkResultT:
        raise NotImplementedError

    def install(self) -> None:
        """Make this the worker's state, with an invocation cache of its own."""
        ScanCache.initialize_worker()
        PoolWorkerState.installed = cast(PoolWorkerState[object, object], self)

    @staticmethod
    def run_installed(item: object) -> object:
        state = PoolWorkerState.installed
        if state is None:
            raise RuntimeError("pool worker started without its state")
        return state.run(item)


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
