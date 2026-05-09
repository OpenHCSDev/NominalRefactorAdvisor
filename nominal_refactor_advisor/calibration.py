"""Golden-target calibration for detector portfolios.

Calibration targets are small algebraic contracts over detector multiplicities,
scan time, and certified payoff.  They let the advisor prove that a detector
family still recognizes the semantic objects it was introduced to recognize,
without hard-coding those objects into the detector implementation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Generic, TypeVar, cast

from .analysis import analyze_modules
from .ast_tools import parse_python_modules
from .collection_algebra import sorted_tuple
from .detectors import DetectorConfig
from .economics import ScanEconomicsProof
from .models import RefactorFinding, SemanticRecord
from .planner import build_refactor_plans
from .record_algebra import materialize_product_record, product_record_spec

_DEFAULT_CALIBRATION_SCAN_BUDGET_SECONDS = 20.0

ManifestT = TypeVar("ManifestT")


class ManifestDecoder(ABC, Generic[ManifestT]):
    """Fail-loud typed decoder for calibration manifest fields."""

    @abstractmethod
    def decode(self, value: object, context: str) -> ManifestT:
        raise NotImplementedError

    def __call__(self, value: object, context: str) -> ManifestT:
        return self.decode(value, context)


@dataclass(frozen=True)
class TypedManifestDecoder(ManifestDecoder[ManifestT]):
    expected_types: tuple[type[object], ...]
    type_label: str
    projection: Callable[[object], ManifestT]
    reject_bool: bool = False
    allow_none: bool = False
    none_value: ManifestT | None = None
    validation_error: Callable[[ManifestT], str | None] = lambda value: None

    def decode(self, value: object, context: str) -> ManifestT:
        if value is None and self.allow_none:
            return cast(ManifestT, self.none_value)
        if self.reject_bool and isinstance(value, bool):
            raise TypeError(f"{context} must be {self.type_label}")
        if not isinstance(value, self.expected_types):
            raise TypeError(f"{context} must be {self.type_label}")
        result = self.projection(value)
        if error := self.validation_error(result):
            raise ValueError(f"{context} must be {error}")
        return result


MANIFEST_MAPPING = TypedManifestDecoder[Mapping[str, object]](
    expected_types=(Mapping,),
    type_label="an object",
    projection=lambda value: cast(Mapping[str, object], value),
)
MANIFEST_SEQUENCE = TypedManifestDecoder[tuple[object, ...]](
    expected_types=(list, tuple),
    type_label="a sequence",
    projection=lambda value: tuple(cast(Iterable[object], value)),
    allow_none=True,
    none_value=(),
)
MANIFEST_STRING = TypedManifestDecoder[str](
    expected_types=(str,),
    type_label="a non-empty string",
    projection=lambda value: cast(str, value),
    validation_error=lambda value: None if value else "a non-empty string",
)
MANIFEST_INT = TypedManifestDecoder[int](
    expected_types=(int,),
    type_label="an integer",
    projection=lambda value: cast(int, value),
    reject_bool=True,
)
MANIFEST_FLOAT = TypedManifestDecoder[float](
    expected_types=(int, float),
    type_label="a number",
    projection=float,
    reject_bool=True,
    validation_error=lambda value: None if value >= 0 else "non-negative",
)
MANIFEST_BOOL = TypedManifestDecoder[bool](
    expected_types=(bool,),
    type_label="a boolean",
    projection=lambda value: cast(bool, value),
)


def _non_negative_int(value: object, context: str) -> int:
    number = MANIFEST_INT(value, context)
    if number < 0:
        raise ValueError(f"{context} must be non-negative")
    return number


def _optional_non_negative_int(value: object | None, context: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, context)


@dataclass(frozen=True)
class DetectorExpectation(SemanticRecord):
    """Allowed multiplicity interval for one detector family."""

    detector_id: str
    min_count: int = 1
    max_count: int | None = None

    @classmethod
    def from_manifest(cls, value: object) -> "DetectorExpectation":
        if isinstance(value, str):
            return cls(detector_id=value)
        row = MANIFEST_MAPPING(value, "detector expectation")
        detector_id = MANIFEST_STRING(row.get("detector_id"), "detector_id")
        min_count = _non_negative_int(row.get("min_count", 1), "min_count")
        max_count = _optional_non_negative_int(row.get("max_count"), "max_count")
        if max_count is not None and max_count < min_count:
            raise ValueError("max_count must be greater than or equal to min_count")
        return cls(
            detector_id=detector_id,
            min_count=min_count,
            max_count=max_count,
        )

    def underflow_reason(self, observed_count: int) -> str | None:
        if observed_count == 0 and self.min_count > 0:
            return f"missing_detector:{self.detector_id}"
        if observed_count < self.min_count:
            return f"under_min:{self.detector_id}:{observed_count}<{self.min_count}"
        return None

    def overflow_reason(self, observed_count: int) -> str | None:
        if self.max_count is not None and observed_count > self.max_count:
            return f"over_max:{self.detector_id}:{observed_count}>{self.max_count}"
        return None


materialize_product_record(
    product_record_spec(
        "DetectorCount",
        "detector_id: str; count: int",
        "SemanticRecord",
        doc="Observed multiplicity for one emitted detector family.",
    )
)


@dataclass(frozen=True)
class CalibrationTarget(SemanticRecord):
    """One corpus target with detector and payoff contracts."""

    name: str
    path: str
    expected_detectors: tuple[DetectorExpectation, ...] = field(default_factory=tuple)
    forbidden_detectors: tuple[str, ...] = field(default_factory=tuple)
    max_production_findings: int | None = None
    max_test_only_findings: int | None = None
    require_payoff_guard: bool = True
    max_scan_seconds: float = _DEFAULT_CALIBRATION_SCAN_BUDGET_SECONDS
    min_certified_description_length_savings: int = 0
    min_backend_lower_bound_removable_loc: int = 0

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        manifest_dir: Path,
    ) -> "CalibrationTarget":
        row = MANIFEST_MAPPING(value, "calibration target")
        name = MANIFEST_STRING(row.get("name"), "target.name")
        raw_path = MANIFEST_STRING(row.get("path"), "target.path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_dir / path
        expected = tuple(
            (
                DetectorExpectation.from_manifest(item)
                for item in MANIFEST_SEQUENCE(
                    row.get("expected_detectors"), "expected_detectors"
                )
            )
        )
        forbidden = sorted_tuple(
            (
                MANIFEST_STRING(item, "forbidden_detectors item")
                for item in MANIFEST_SEQUENCE(
                    row.get("forbidden_detectors"), "forbidden_detectors"
                )
            )
        )
        return cls(
            name=name,
            path=str(path),
            expected_detectors=expected,
            forbidden_detectors=forbidden,
            max_production_findings=_optional_non_negative_int(
                row.get("max_production_findings"), "max_production_findings"
            ),
            max_test_only_findings=_optional_non_negative_int(
                row.get("max_test_only_findings"), "max_test_only_findings"
            ),
            require_payoff_guard=MANIFEST_BOOL(
                row.get("require_payoff_guard", True), "require_payoff_guard"
            ),
            max_scan_seconds=MANIFEST_FLOAT(
                row.get("max_scan_seconds", _DEFAULT_CALIBRATION_SCAN_BUDGET_SECONDS),
                "max_scan_seconds",
            ),
            min_certified_description_length_savings=_non_negative_int(
                row.get("min_certified_description_length_savings", 0),
                "min_certified_description_length_savings",
            ),
            min_backend_lower_bound_removable_loc=_non_negative_int(
                row.get("min_backend_lower_bound_removable_loc", 0),
                "min_backend_lower_bound_removable_loc",
            ),
        )


@dataclass(frozen=True)
class CalibrationManifest(SemanticRecord):
    """Typed manifest for a detector calibration corpus."""

    targets: tuple[CalibrationTarget, ...]

    @classmethod
    def from_file(cls, manifest_path: Path) -> "CalibrationManifest":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = MANIFEST_MAPPING(manifest, "calibration manifest")
        targets = tuple(
            (
                CalibrationTarget.from_manifest(
                    item,
                    manifest_dir=manifest_path.resolve().parent,
                )
                for item in MANIFEST_SEQUENCE(row.get("targets"), "targets")
            )
        )
        return cls(targets=targets)


@dataclass(frozen=True)
class CalibrationTargetResult(SemanticRecord):
    """Observed scan result for one calibration target."""

    target: CalibrationTarget
    scan: ScanEconomicsProof
    detector_counts: tuple[DetectorCount, ...] = field(default_factory=tuple)
    unavailable_reason: str | None = None

    def detector_count(self, detector_id: str) -> int:
        return next(
            (
                detector_count.count
                for detector_count in self.detector_counts
                if detector_count.detector_id == detector_id
            ),
            0,
        )

    @property
    def regression_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.unavailable_reason is not None:
            reasons.append(f"target_unavailable:{self.target.name}")
        for expectation in self.target.expected_detectors:
            observed_count = self.detector_count(expectation.detector_id)
            reasons.extend(
                (
                    reason
                    for reason in (
                        expectation.underflow_reason(observed_count),
                        expectation.overflow_reason(observed_count),
                    )
                    if reason is not None
                )
            )
        reasons.extend(
            (
                f"forbidden_detector:{detector_id}:{observed_count}"
                for detector_id in self.target.forbidden_detectors
                for observed_count in (self.detector_count(detector_id),)
                if observed_count
            )
        )
        if not self.scan.scan_budget_passes:
            reasons.append(
                f"scan_budget:{self.scan.elapsed_seconds:.3f}>"
                f"{self.scan.scan_budget_seconds:.3f}"
            )
        if (
            self.target.max_production_findings is not None
            and self.scan.production_finding_count > self.target.max_production_findings
        ):
            reasons.append(
                "production_finding_budget:"
                f"{self.scan.production_finding_count}>"
                f"{self.target.max_production_findings}"
            )
        if (
            self.target.max_test_only_findings is not None
            and self.scan.test_only_finding_count > self.target.max_test_only_findings
        ):
            reasons.append(
                "test_only_finding_budget:"
                f"{self.scan.test_only_finding_count}>"
                f"{self.target.max_test_only_findings}"
            )
        if (
            self.target.require_payoff_guard
            and not self.scan.economics.payoff_guard_passes
        ):
            reasons.append("payoff_guard")
        if (
            self.scan.economics.certified_description_length_savings
            < self.target.min_certified_description_length_savings
        ):
            reasons.append(
                "semantic_savings:"
                f"{self.scan.economics.certified_description_length_savings}<"
                f"{self.target.min_certified_description_length_savings}"
            )
        if (
            self.scan.economics.backend_lower_bound_removable_loc
            < self.target.min_backend_lower_bound_removable_loc
        ):
            reasons.append(
                "backend_loc_savings:"
                f"{self.scan.economics.backend_lower_bound_removable_loc}<"
                f"{self.target.min_backend_lower_bound_removable_loc}"
            )
        return tuple(reasons)

    @property
    def passes(self) -> bool:
        return not self.regression_reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "scan": self.scan.to_dict(),
            "detector_counts": [item.to_dict() for item in self.detector_counts],
            "unavailable_reason": self.unavailable_reason,
            "passes": self.passes,
            "regression_reasons": self.regression_reasons,
        }


@dataclass(frozen=True)
class CalibrationReport(SemanticRecord):
    """Full detector calibration result."""

    manifest_path: str
    target_results: tuple[CalibrationTargetResult, ...]

    @property
    def passes(self) -> bool:
        return all((target_result.passes for target_result in self.target_results))

    @property
    def regression_reasons(self) -> tuple[str, ...]:
        return tuple(
            (
                f"{target_result.target.name}:{reason}"
                for target_result in self.target_results
                for reason in target_result.regression_reasons
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "passes": self.passes,
            "regression_reasons": self.regression_reasons,
            "target_results": [
                target_result.to_dict() for target_result in self.target_results
            ],
        }


def _detector_counts(findings: Iterable[RefactorFinding]) -> tuple[DetectorCount, ...]:
    counts = Counter((finding.detector_id for finding in findings))
    return tuple(
        (
            DetectorCount(detector_id=detector_id, count=count)
            for detector_id, count in sorted(counts.items())
        )
    )


def _empty_scan_for_target(
    target: CalibrationTarget,
    *,
    unavailable_reason: str,
) -> CalibrationTargetResult:
    path = Path(target.path)
    scan = ScanEconomicsProof.from_findings_and_plans(
        label=target.name,
        path=path,
        elapsed_seconds=0.0,
        scan_budget_seconds=target.max_scan_seconds,
        findings=(),
        plans=(),
    )
    return CalibrationTargetResult(
        target=target,
        scan=scan,
        unavailable_reason=unavailable_reason,
    )


def run_calibration_target(
    target: CalibrationTarget,
    *,
    config: DetectorConfig | None = None,
) -> CalibrationTargetResult:
    """Run one calibration target and return its proof result."""

    path = Path(target.path)
    if not path.exists():
        return _empty_scan_for_target(
            target,
            unavailable_reason=f"path does not exist: {path}",
        )
    started = perf_counter()
    modules = parse_python_modules(path)
    findings = analyze_modules(modules, config)
    plans = build_refactor_plans(findings, path)
    elapsed = perf_counter() - started
    scan = ScanEconomicsProof.from_findings_and_plans(
        label=target.name,
        path=path,
        elapsed_seconds=elapsed,
        scan_budget_seconds=target.max_scan_seconds,
        findings=findings,
        plans=plans,
    )
    return CalibrationTargetResult(
        target=target,
        scan=scan,
        detector_counts=_detector_counts(findings),
    )


def run_calibration_manifest(
    manifest_path: Path,
    *,
    config: DetectorConfig | None = None,
) -> CalibrationReport:
    """Run all targets from a JSON calibration manifest."""

    manifest = CalibrationManifest.from_file(manifest_path)
    detector_config = config or DetectorConfig()
    return CalibrationReport(
        manifest_path=str(manifest_path),
        target_results=tuple(
            (
                run_calibration_target(target, config=detector_config)
                for target in manifest.targets
            )
        ),
    )


def format_calibration_markdown(report: CalibrationReport) -> str:
    """Render a calibration report in concise Markdown."""

    lines = [
        "Calibration:",
        f"   - Overall: {'pass' if report.passes else 'fail'}",
    ]
    if report.regression_reasons:
        lines.append("   - Regression reasons: " + ", ".join(report.regression_reasons))
    for target_result in report.target_results:
        scan = target_result.scan
        economics = scan.economics
        counts = ", ".join(
            (
                f"{detector_count.detector_id}={detector_count.count}"
                for detector_count in target_result.detector_counts
            )
        )
        lines.append(
            f"   - {target_result.target.name}: "
            f"{'pass' if target_result.passes else 'fail'}; "
            f"{scan.finding_count} finding(s), "
            f"{scan.production_finding_count} production, "
            f"{scan.test_only_finding_count} test-only; "
            f"{scan.elapsed_seconds:.3f}s/{scan.scan_budget_seconds:.3f}s"
        )
        lines.append(
            "     payoff: "
            f"{'pass' if economics.payoff_guard_passes else 'fail'}; "
            "semantic savings "
            f"{economics.certified_description_length_savings}; "
            "backend LOC "
            f"{economics.backend_lower_bound_removable_loc}-"
            f"{economics.backend_upper_bound_removable_loc}"
        )
        if counts:
            lines.append(f"     detectors: {counts}")
        if target_result.unavailable_reason is not None:
            lines.append(f"     unavailable: {target_result.unavailable_reason}")
        if target_result.regression_reasons:
            lines.append(
                "     target reasons: " + ", ".join(target_result.regression_reasons)
            )
    return "\n".join(lines)
