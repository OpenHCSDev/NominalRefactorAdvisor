from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SourceLocation:
    file_path: str
    line: int
    symbol: str


@dataclass(frozen=True)
class RefactorFinding:
    detector_id: str
    pattern_id: int
    title: str
    summary: str
    why: str
    capability_gap: str
    confidence: str
    relation_context: str
    evidence: tuple[SourceLocation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
