"""How a goal run proves a concept's stages.

A stage is proved by rerunning detectors over its projected sources. A concept
declares which: whether a stage reruns the semantic-mirror detectors beside its
own, and whether a state with no targets left is certified again by a scan of
every detector over the whole tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ScanT = TypeVar("ScanT")


class TrajectoryScans(ABC, Generic[ScanT]):
    """What a goal run can rescan: every detector over one source state."""

    @abstractmethod
    def exact_scan(self, scan: ScanT) -> ScanT:
        raise NotImplementedError


class StageProof(ABC):
    """What proves each stage of a concept's trajectory and its terminal state."""

    @classmethod
    @abstractmethod
    def reproof_detector_ids(
        cls, own: frozenset[str], semantic_mirrors: frozenset[str]
    ) -> frozenset[str]:
        """The detectors a stage reruns over its projected sources."""
        raise NotImplementedError

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


class RescannedStages(StageProof):
    """A stage reruns the semantic mirrors beside its own detectors, and a
    terminal is rescanned by every detector: the stage's edits may change any
    fact any detector reads."""

    @classmethod
    def reproof_detector_ids(
        cls, own: frozenset[str], semantic_mirrors: frozenset[str]
    ) -> frozenset[str]:
        return own | semantic_mirrors

    @classmethod
    def carried_detector_ids(cls, per_module: frozenset[str]) -> frozenset[str]:
        return frozenset()

    @classmethod
    def certified(cls, scans: TrajectoryScans[ScanT], scan: ScanT) -> ScanT:
        return scans.exact_scan(scan)


class OperationCertifiedStages(StageProof):
    """A stage's operations rewrite spellings of the moved module's name (its
    imports, import strings, manifest entries and mentions) and the bodies they
    move; their preflights refuse a spelling they cannot trace. Each stage reruns
    the concept's own detectors over the projected tree, and every per-module
    detector over the files it touched, carrying the rest: a per-module answer
    depends only on its own file, so those detectors stay exact over the whole
    tree. Detectors that read across files are not rerun; the next full scan
    measures them."""

    @classmethod
    def reproof_detector_ids(
        cls, own: frozenset[str], semantic_mirrors: frozenset[str]
    ) -> frozenset[str]:
        return own

    @classmethod
    def carried_detector_ids(cls, per_module: frozenset[str]) -> frozenset[str]:
        return per_module

    @classmethod
    def certified(cls, scans: TrajectoryScans[ScanT], scan: ScanT) -> ScanT:
        return scan


__all__ = (
    "OperationCertifiedStages",
    "RescannedStages",
    "StageProof",
    "TrajectoryScans",
)
