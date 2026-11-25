"""How a goal run proves a concept's stages.

A stage is proved by rerunning the concept's own detectors over its projected
sources. A concept declares which per-module findings a stage carries, and
whether a state with no targets left is certified again by a scan of every
detector over the whole tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

ScanT = TypeVar("ScanT")
ChangeT = TypeVar("ChangeT")


class TrajectoryScans(ABC, Generic[ScanT]):
    """What a goal run can rescan: every detector over one source state, or some
    detectors over the files the stages rewrote."""

    @abstractmethod
    def exact_scan(self, scan: ScanT) -> ScanT:
        raise NotImplementedError

    @abstractmethod
    def settled(self, scan: ScanT) -> ScanT:
        """`scan` with every finding its coverage still owes paid."""
        raise NotImplementedError


class StageProof(ABC):
    """What proves each stage of a concept's trajectory and its terminal state."""

    prewarms_full_rescan: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def carried_detector_ids(cls, per_module: frozenset[str]) -> frozenset[str]:
        """Per-module detectors whose findings a stage carries forward and
        recomputes only for the files it touched."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def certified(cls, scans: TrajectoryScans[ScanT], scan: ScanT) -> ScanT:
        """The scan a terminal state and its report are judged on."""
        raise NotImplementedError

    @classmethod
    def reprove_complete_branch(
        cls,
        scans: TrajectoryScans[ScanT],
        unproved: Callable[[], ScanT],
        projected: Callable[[], ScanT],
    ) -> ScanT:
        """Re-prove a lone branch whose plan covers every current target."""

        del scans, unproved
        return projected()

    @classmethod
    def unjustified(cls, increased: tuple[ChangeT, ...]) -> tuple[ChangeT, ...]:
        """The finding classes a terminal state grew that its stages answer for:
        every one, since a stage may have written what a new finding cites."""

        return increased


class RescannedStages(StageProof):
    """A terminal is rescanned by every detector: the stages' edits may change
    any fact any detector reads."""

    prewarms_full_rescan: ClassVar[bool] = True

    @classmethod
    def carried_detector_ids(cls, per_module: frozenset[str]) -> frozenset[str]:
        return frozenset()

    @classmethod
    def certified(cls, scans: TrajectoryScans[ScanT], scan: ScanT) -> ScanT:
        return scans.exact_scan(scan)

    @classmethod
    def reprove_complete_branch(
        cls,
        scans: TrajectoryScans[ScanT],
        unproved: Callable[[], ScanT],
        projected: Callable[[], ScanT],
    ) -> ScanT:
        """Skip a separate target-only pass when the terminal scan already runs
        every detector over the projected tree."""

        del projected
        return scans.exact_scan(unproved())


class OperationCertifiedStages(StageProof):
    """A stage's operations rewrite spellings of the moved module's name (its
    imports, import strings, manifest entries and mentions) and the bodies they
    move; their preflights refuse a spelling they cannot trace. Each stage reruns
    the concept's own detectors over the projected tree, and every per-module
    detector over the files it touched, carrying the rest: a per-module answer
    depends only on its own file, so those detectors stay exact over the whole
    tree. A rewritten file is re-proved once, when the trajectory settles: no
    stage plans from a per-module finding. Detectors that read across files are
    not rerun; the next full scan measures them."""

    @classmethod
    def carried_detector_ids(cls, per_module: frozenset[str]) -> frozenset[str]:
        return per_module

    @classmethod
    def certified(cls, scans: TrajectoryScans[ScanT], scan: ScanT) -> ScanT:
        return scans.settled(scan)


class ImportScopeStages(RescannedStages):
    """Stages that only move import statements, each proved to close no load-time
    cycle and to shadow no binding: they bind the same objects earlier, so they
    change which names a detector can resolve and nothing any line does. A
    finding class they grow cites code that was there before; it is revealed,
    not debt, and the next scan reports it."""

    @classmethod
    def unjustified(cls, increased: tuple[ChangeT, ...]) -> tuple[ChangeT, ...]:
        return ()


__all__ = (
    "ImportScopeStages",
    "OperationCertifiedStages",
    "RescannedStages",
    "StageProof",
    "TrajectoryScans",
)
