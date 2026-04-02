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
    scaffold: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FindingSpec:
    pattern_id: int
    title: str
    why: str
    capability_gap: str
    relation_context: str
    confidence: str = "medium"
    scaffold_template: str | None = None

    def build(
        self,
        *,
        detector_id: str,
        summary: str,
        evidence: tuple[SourceLocation, ...],
        scaffold: str | None = None,
    ) -> RefactorFinding:
        return RefactorFinding(
            detector_id=detector_id,
            pattern_id=self.pattern_id,
            title=self.title,
            summary=summary,
            why=self.why,
            capability_gap=self.capability_gap,
            confidence=self.confidence,
            relation_context=self.relation_context,
            evidence=evidence,
            scaffold=scaffold,
        )
