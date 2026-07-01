"""Nominal count records derived from emitted refactoring findings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .models import RefactorFinding, SemanticRecord


@dataclass(frozen=True)
class FindingPatternCount(SemanticRecord):
    """Cached finding count for one canonical pattern id."""

    pattern_id: int
    count: int


@dataclass(frozen=True)
class FindingDetectorCount(SemanticRecord):
    """Observed finding count for one detector family."""

    detector_id: str
    count: int


@dataclass(frozen=True)
class FindingSummary(SemanticRecord):
    """Compact count projection for advisor loop payloads."""

    finding_count: int
    pattern_counts: tuple[FindingPatternCount, ...]
    detector_counts: tuple[FindingDetectorCount, ...]

    @classmethod
    def empty(cls) -> "FindingSummary":
        return cls(
            finding_count=0,
            pattern_counts=(),
            detector_counts=(),
        )

    @classmethod
    def from_findings(
        cls,
        findings: list[RefactorFinding] | tuple[RefactorFinding, ...],
    ) -> "FindingSummary":
        return cls(
            finding_count=len(findings),
            pattern_counts=tuple(
                FindingPatternCount(pattern_id, count)
                for pattern_id, count in sorted(
                    Counter(int(finding.pattern_id) for finding in findings).items()
                )
            ),
            detector_counts=FindingDetectorCountsAuthority(findings).detector_counts(),
        )

    def detector_count(self, detector_id: str) -> int:
        return next(
            (
                detector_count.count
                for detector_count in self.detector_counts
                if detector_count.detector_id == detector_id
            ),
            0,
        )

    def detector_counts_payload(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.detector_counts]

    def detector_counts_text(self) -> str:
        return ", ".join(
            (
                f"{detector_count.detector_id}={detector_count.count}"
                for detector_count in self.detector_counts
            )
        )


@dataclass(frozen=True)
class FindingDetectorCountsAuthority:
    """Project emitted findings into detector multiplicity records."""

    findings: Iterable[RefactorFinding]

    def detector_counts(self) -> tuple[FindingDetectorCount, ...]:
        counts = Counter((finding.detector_id for finding in self.findings))
        return tuple(
            (
                FindingDetectorCount(detector_id=detector_id, count=count)
                for detector_id, count in sorted(counts.items())
            )
        )
