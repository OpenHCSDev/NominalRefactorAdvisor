"""Portfolio-level payoff accounting for refactor recommendations.

The advisor should prove that abstraction work compounds instead of hiding cost
in nicer names.  This module keeps recommendation payoff, semantic compression,
and working-tree change budgets as separate dimensions so reports do not mix
detector/test growth with backend savings.
"""

from __future__ import annotations

import subprocess
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .analysis import analyze_modules
from .ast_tools import parse_python_modules
from .collection_algebra import sorted_tuple
from .detectors import DetectorConfig
from .models import ImpactDelta, RefactorFinding, RefactorPlan, SemanticRecord
from .planner import build_refactor_plans

_DEFAULT_SCAN_BUDGET_SECONDS = 20.0
_READABILITY_DETECTOR_IDS = frozenset(
    {
        "excessive_blank_line_run",
        "readability_compressed_line",
    }
)


@dataclass(frozen=True)
class ImpactDeltaClassifier:
    def has_positive_impact(self, delta: ImpactDelta) -> bool:
        return any(
            (
                delta.lower_bound_removable_loc > 0,
                delta.upper_bound_removable_loc > 0,
                delta.repeated_mappings_centralized > 0,
                delta.dispatch_sites_eliminated > 0,
                delta.registration_sites_removed > 0,
                delta.shared_algorithm_sites_centralized > 0,
                delta.description_length_savings > 0,
            )
        )


IMPACT_DELTA_CLASSIFIER = ImpactDeltaClassifier()


def _sum_impacts(impacts: Iterable[ImpactDelta]) -> ImpactDelta:
    total = ImpactDelta()
    for impact in impacts:
        total += impact
    return total


def _finding_requires_payoff_proof(finding: RefactorFinding) -> bool:
    return bool(finding.scaffold or finding.codemod_patch)


def _finding_has_payoff_proof(finding: RefactorFinding) -> bool:
    certificate = finding.compression_certificate
    if certificate is not None and certificate.pays_rent:
        return True
    return IMPACT_DELTA_CLASSIFIER.has_positive_impact(finding.metrics.impact_delta)


def _is_test_file_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/")
    return (
        normalized.startswith("tests/")
        or "/tests/" in normalized
        or normalized.endswith("_test.py")
        or normalized.endswith("_tests.py")
        or normalized.startswith("test_")
        or "/test_" in normalized
    )


def _is_test_only_finding(finding: RefactorFinding) -> bool:
    return bool(finding.evidence) and all(
        (_is_test_file_path(item.file_path) for item in finding.evidence)
    )


def _run_git_command(start_path: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(start_path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _git_root(start_path: Path) -> Path:
    result = _run_git_command(
        start_path,
        "rev-parse",
        "--show-toplevel",
    )
    if result.returncode != 0:
        return start_path.resolve()
    return Path(result.stdout.strip()).resolve()


@dataclass(frozen=True)
class RecommendationEconomics(SemanticRecord):
    """Aggregated proof that emitted recommendations carry payoff evidence."""

    finding_count: int = 0
    plan_count: int = 0
    evidence_site_count: int = 0
    certificate_count: int = 0
    proven_finding_count: int = 0
    loc_payoff_finding_count: int = 0
    semantic_payoff_finding_count: int = 0
    unproven_infrastructure_finding_count: int = 0
    unproven_infrastructure_detector_ids: tuple[str, ...] = field(default_factory=tuple)
    backend_lower_bound_removable_loc: int = 0
    backend_upper_bound_removable_loc: int = 0
    description_length_before: int = 0
    description_length_after: int = 0
    certified_description_length_savings: int = 0
    payoff_guard_passes: bool = True
    has_long_term_signal: bool = False

    @classmethod
    def from_findings_and_plans(
        cls,
        findings: Iterable[RefactorFinding],
        plans: Iterable[RefactorPlan] = (),
    ) -> "RecommendationEconomics":
        finding_tuple = tuple(findings)
        plan_tuple = tuple(plans)
        certificates = tuple(
            (
                finding.compression_certificate
                for finding in finding_tuple
                if finding.compression_certificate is not None
            )
        )
        plan_outcome = _sum_impacts((plan.outcome for plan in plan_tuple))
        finding_outcome = _sum_impacts(
            (finding.metrics.impact_delta for finding in finding_tuple)
        )
        outcome = plan_outcome if plan_tuple else finding_outcome
        if certificates:
            description_length_before = sum(
                (certificate.before_description_length for certificate in certificates)
            )
            description_length_after = sum(
                (
                    certificate.description_cost.description_length
                    for certificate in certificates
                )
            )
            certified_description_length_savings = sum(
                (
                    certificate.certified_description_length_savings
                    for certificate in certificates
                )
            )
        else:
            description_length_before = outcome.description_length_before
            description_length_after = outcome.description_length_after
            certified_description_length_savings = outcome.description_length_savings

        unproven = tuple(
            (
                finding
                for finding in finding_tuple
                if _finding_requires_payoff_proof(finding)
                and not _finding_has_payoff_proof(finding)
            )
        )
        proven_finding_count = sum(
            (1 for finding in finding_tuple if _finding_has_payoff_proof(finding))
        )
        loc_payoff_finding_count = sum(
            (
                1
                for finding in finding_tuple
                if IMPACT_DELTA_CLASSIFIER.has_positive_impact(
                    finding.metrics.impact_delta
                )
            )
        )
        semantic_payoff_finding_count = sum(
            (
                1
                for finding in finding_tuple
                if finding.compression_certificate is not None
                and finding.compression_certificate.pays_rent
            )
        )
        evidence_sites = frozenset(
            (
                (item.file_path, item.line, item.symbol)
                for finding in finding_tuple
                for item in finding.evidence
            )
        )
        return cls(
            finding_count=len(finding_tuple),
            plan_count=len(plan_tuple),
            evidence_site_count=len(evidence_sites),
            certificate_count=len(certificates),
            proven_finding_count=proven_finding_count,
            loc_payoff_finding_count=loc_payoff_finding_count,
            semantic_payoff_finding_count=semantic_payoff_finding_count,
            unproven_infrastructure_finding_count=len(unproven),
            unproven_infrastructure_detector_ids=sorted_tuple(
                {finding.detector_id for finding in unproven}
            ),
            backend_lower_bound_removable_loc=outcome.lower_bound_removable_loc,
            backend_upper_bound_removable_loc=outcome.upper_bound_removable_loc,
            description_length_before=description_length_before,
            description_length_after=description_length_after,
            certified_description_length_savings=certified_description_length_savings,
            payoff_guard_passes=not unproven,
            has_long_term_signal=(
                outcome.lower_bound_removable_loc > 0
                or certified_description_length_savings > 0
            ),
        )


@dataclass(frozen=True)
class ScanEconomicsProof(SemanticRecord):
    """One timed scan summarized as an economics gate."""

    label: str
    path: str
    elapsed_seconds: float
    scan_budget_seconds: float
    finding_count: int
    production_finding_count: int
    test_only_finding_count: int
    semantic_production_finding_count: int
    readability_finding_count: int
    plan_count: int
    detector_ids: tuple[str, ...] = field(default_factory=tuple)
    production_detector_ids: tuple[str, ...] = field(default_factory=tuple)
    economics: RecommendationEconomics = field(default_factory=RecommendationEconomics)

    @classmethod
    def from_findings_and_plans(
        cls,
        *,
        label: str,
        path: Path,
        elapsed_seconds: float,
        scan_budget_seconds: float,
        findings: Iterable[RefactorFinding],
        plans: Iterable[RefactorPlan],
    ) -> "ScanEconomicsProof":
        finding_tuple = tuple(findings)
        plan_tuple = tuple(plans)
        production_findings = tuple(
            (finding for finding in finding_tuple if not _is_test_only_finding(finding))
        )
        test_only_findings = tuple(
            (finding for finding in finding_tuple if _is_test_only_finding(finding))
        )
        readability_findings = tuple(
            (
                finding
                for finding in production_findings
                if finding.detector_id in _READABILITY_DETECTOR_IDS
            )
        )
        semantic_production_findings = tuple(
            (
                finding
                for finding in production_findings
                if finding.detector_id not in _READABILITY_DETECTOR_IDS
            )
        )
        return cls(
            label=label,
            path=str(path),
            elapsed_seconds=round(elapsed_seconds, 3),
            scan_budget_seconds=scan_budget_seconds,
            finding_count=len(finding_tuple),
            production_finding_count=len(production_findings),
            test_only_finding_count=len(test_only_findings),
            semantic_production_finding_count=len(semantic_production_findings),
            readability_finding_count=len(readability_findings),
            plan_count=len(plan_tuple),
            detector_ids=sorted_tuple(
                {finding.detector_id for finding in finding_tuple}
            ),
            production_detector_ids=sorted_tuple(
                {finding.detector_id for finding in production_findings}
            ),
            economics=RecommendationEconomics.from_findings_and_plans(
                finding_tuple, plan_tuple
            ),
        )

    @property
    def scan_budget_passes(self) -> bool:
        return self.elapsed_seconds <= self.scan_budget_seconds

    @property
    def production_scan_clean(self) -> bool:
        return self.production_finding_count == 0

    @property
    def proof_passes(self) -> bool:
        return (
            self.production_scan_clean
            and self.scan_budget_passes
            and self.economics.payoff_guard_passes
        )

    @property
    def regression_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.production_scan_clean:
            reasons.append(f"{self.label}_production_findings")
        if not self.scan_budget_passes:
            reasons.append(f"{self.label}_scan_budget")
        if not self.economics.payoff_guard_passes:
            reasons.append(f"{self.label}_payoff_guard")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path": self.path,
            "elapsed_seconds": self.elapsed_seconds,
            "scan_budget_seconds": self.scan_budget_seconds,
            "scan_budget_passes": self.scan_budget_passes,
            "finding_count": self.finding_count,
            "production_finding_count": self.production_finding_count,
            "test_only_finding_count": self.test_only_finding_count,
            "semantic_production_finding_count": self.semantic_production_finding_count,
            "readability_finding_count": self.readability_finding_count,
            "plan_count": self.plan_count,
            "detector_ids": self.detector_ids,
            "production_detector_ids": self.production_detector_ids,
            "production_scan_clean": self.production_scan_clean,
            "payoff_guard_passes": self.economics.payoff_guard_passes,
            "proof_passes": self.proof_passes,
            "regression_reasons": self.regression_reasons,
            "economics": self.economics.to_dict(),
        }


@dataclass(frozen=True)
class EconomicsProofReport(SemanticRecord):
    """Two-scan proof that separates cleanliness, payoff, timing, and LOC budget."""

    package_scan: ScanEconomicsProof
    repository_scan: ScanEconomicsProof
    change_budget: "RepositoryChangeBudget"

    @property
    def proof_passes(self) -> bool:
        return (
            self.package_scan.proof_passes
            and self.repository_scan.proof_passes
            and self.change_budget.unavailable_reason is None
        )

    @property
    def regression_reasons(self) -> tuple[str, ...]:
        reasons = [
            *self.package_scan.regression_reasons,
            *self.repository_scan.regression_reasons,
        ]
        if self.change_budget.unavailable_reason is not None:
            reasons.append("change_budget_unavailable")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "proof_passes": self.proof_passes,
            "regression_reasons": self.regression_reasons,
            "package_scan": self.package_scan.to_dict(),
            "repository_scan": self.repository_scan.to_dict(),
            "change_budget": self.change_budget.to_dict(),
        }


@dataclass(frozen=True)
class LineChangeBudget(SemanticRecord):
    """Added/deleted line budget for one repository category."""

    added: int = 0
    deleted: int = 0

    @property
    def net_added(self) -> int:
        return self.added - self.deleted

    def to_dict(self) -> dict[str, object]:
        return {
            "added": self.added,
            "deleted": self.deleted,
            "net_added": self.net_added,
        }


@dataclass(frozen=True)
class RepositoryChangeBudget(SemanticRecord):
    """Working-tree line budget split by role instead of one misleading total."""

    advisor_backend: LineChangeBudget = field(default_factory=LineChangeBudget)
    detectors: LineChangeBudget = field(default_factory=LineChangeBudget)
    tests: LineChangeBudget = field(default_factory=LineChangeBudget)
    docs: LineChangeBudget = field(default_factory=LineChangeBudget)
    generated: LineChangeBudget = field(default_factory=LineChangeBudget)
    other: LineChangeBudget = field(default_factory=LineChangeBudget)
    compare_ref: str = "HEAD"
    unavailable_reason: str | None = None

    @classmethod
    def unavailable(
        cls, reason: str, *, compare_ref: str = "HEAD"
    ) -> "RepositoryChangeBudget":
        return cls(compare_ref=compare_ref, unavailable_reason=reason)

    @classmethod
    def from_numstat_rows(
        cls,
        rows: Iterable[str],
        *,
        compare_ref: str = "HEAD",
    ) -> "RepositoryChangeBudget":
        budgets = {
            "advisor_backend": LineChangeBudget(),
            "detectors": LineChangeBudget(),
            "tests": LineChangeBudget(),
            "docs": LineChangeBudget(),
            "generated": LineChangeBudget(),
            "other": LineChangeBudget(),
        }
        for row in rows:
            parts = row.rstrip("\n").split("\t")
            if len(parts) < 3 or "-" in parts[:2]:
                continue
            added = int(parts[0])
            deleted = int(parts[1])
            path = parts[2]
            category = cls.category_for_path(path)
            current = budgets[category]
            budgets[category] = LineChangeBudget(
                added=current.added + added,
                deleted=current.deleted + deleted,
            )
        return cls(compare_ref=compare_ref, **budgets)

    @classmethod
    def from_git_diff(
        cls,
        start_path: Path,
        *,
        compare_ref: str = "HEAD",
    ) -> "RepositoryChangeBudget":
        repo_result = _run_git_command(
            start_path,
            "rev-parse",
            "--show-toplevel",
        )
        if repo_result.returncode != 0:
            return cls.unavailable(
                repo_result.stderr.strip() or "not a git repository",
                compare_ref=compare_ref,
            )
        repo_root = repo_result.stdout.strip()
        diff_result = _run_git_command(
            Path(repo_root),
            "diff",
            "--numstat",
            compare_ref,
        )
        if diff_result.returncode != 0:
            return cls.unavailable(
                diff_result.stderr.strip() or "git diff failed",
                compare_ref=compare_ref,
            )
        return cls.from_numstat_rows(
            diff_result.stdout.splitlines(), compare_ref=compare_ref
        )

    @staticmethod
    def category_for_path(path: str) -> str:
        normalized = path.replace("\\", "/")
        if normalized.startswith("tests/") or "/tests/" in normalized:
            return "tests"
        if normalized.startswith("nominal_refactor_advisor/detectors/"):
            return "detectors"
        if normalized.startswith("nominal_refactor_advisor/"):
            return "advisor_backend"
        if normalized.startswith("docs/"):
            return "docs"
        if (
            normalized.startswith("build/")
            or normalized.startswith("dist/")
            or ".egg-info/" in normalized
        ):
            return "generated"
        return "other"

    def to_dict(self) -> dict[str, object]:
        return {
            "advisor_backend": self.advisor_backend.to_dict(),
            "detectors": self.detectors.to_dict(),
            "tests": self.tests.to_dict(),
            "docs": self.docs.to_dict(),
            "generated": self.generated.to_dict(),
            "other": self.other.to_dict(),
            "compare_ref": self.compare_ref,
            "unavailable_reason": self.unavailable_reason,
        }


def _scan_modules_for_proof(
    label: str,
    path: Path,
    modules: list,
    config: DetectorConfig,
    scan_budget_seconds: float,
) -> ScanEconomicsProof:
    started = perf_counter()
    findings = analyze_modules(modules, config)
    plans = build_refactor_plans(findings, path)
    elapsed = perf_counter() - started
    return ScanEconomicsProof.from_findings_and_plans(
        label=label,
        path=path,
        elapsed_seconds=elapsed,
        scan_budget_seconds=scan_budget_seconds,
        findings=findings,
        plans=plans,
    )


def build_economics_proof_report(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    compare_ref: str = "HEAD",
    scan_budget_seconds: float = _DEFAULT_SCAN_BUDGET_SECONDS,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> EconomicsProofReport:
    """Run the standard proof scans for long-term advisor economics."""

    config = config or DetectorConfig()
    repo_root = _git_root(root)
    package_path = repo_root / "nominal_refactor_advisor"
    if not package_path.exists():
        package_path = root
        parsed_modules = parse_python_modules(
            root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
        )
    else:
        parsed_modules = parse_python_modules(
            repo_root,
            cache_dir=cache_dir,
            use_parse_cache=use_parse_cache,
            parse_workers=parse_workers,
        )
    package_root = package_path.resolve()
    package_modules = [
        module
        for module in parsed_modules
        if module.path.resolve().is_relative_to(package_root)
    ]
    package_scan = _scan_modules_for_proof(
        "package",
        package_path,
        package_modules,
        config,
        scan_budget_seconds,
    )
    repository_scan = _scan_modules_for_proof(
        "repository",
        repo_root,
        parsed_modules,
        config,
        scan_budget_seconds,
    )
    return EconomicsProofReport(
        package_scan=package_scan,
        repository_scan=repository_scan,
        change_budget=RepositoryChangeBudget.from_git_diff(
            repo_root, compare_ref=compare_ref
        ),
    )
