from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from operator import attrgetter
from pathlib import Path

from .models import (
    CERTIFIED,
    ImpactDelta,
    STRONG_HEURISTIC,
    OutcomeEstimate,
    RefactorAction,
    RefactorFinding,
    RefactorPlan,
    SourceLocation,
)
from .patterns import PATTERN_SPECS
from .taxonomy import (
    CapabilityTag,
    CertificationLevel,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ObservationTag,
)


_PATTERN_DEPENDENCIES = {
    3: {1, 2, 4, 6},
    5: {1, 4},
    8: {4, 7, 13},
    9: {10},
    12: {10},
    13: {7},
    14: {5, 6, 7, 13},
}

_PATTERN_SYNERGY = {
    1: {3, 4, 5},
    2: {3, 5},
    3: {1, 2, 4, 6, 14},
    4: {1, 3, 5, 8, 14},
    5: {1, 2, 4, 6, 14},
    6: {3, 5, 13, 14},
    7: {8, 13, 14},
    8: {4, 7, 13},
    9: {10, 11, 12},
    10: {9, 11, 12},
    11: {9, 10, 12},
    12: {9, 10, 11},
    13: {6, 7, 8, 14},
    14: {3, 4, 5, 6, 7, 13},
}

_PATTERN_PRIORITY = {
    1: 95,
    2: 92,
    4: 90,
    5: 88,
    6: 84,
    7: 80,
    10: 76,
    9: 74,
    11: 72,
    12: 70,
    13: 66,
    8: 62,
    3: 58,
    14: 40,
}


@dataclass(frozen=True)
class _FindingCluster:
    subsystem: str
    findings: tuple[RefactorFinding, ...]
    evidence: tuple[SourceLocation, ...]


class PatternPlanStepBuilder(ABC):
    @abstractmethod
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        raise NotImplementedError


class GenericPatternPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        pattern = PATTERN_SPECS[pattern_id]
        return f"Apply Pattern {pattern_id} in `{subsystem}`: {pattern.prescription}"


class TemplateMethodPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum(
            finding.metrics.shared_algorithm_sites_for_plan() for finding in findings
        )
        return (
            f"Create one ABC template-method family for `{subsystem}` and move the shared orchestration from "
            f"{site_count or len(findings)} duplicated method site(s) into the base class."
        )


class AutoRegisterPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum(
            finding.metrics.registration_sites_for_plan() for finding in findings
        )
        return (
            f"Introduce `AutoRegisterMeta` for `{subsystem}` and replace {site_count or len(findings)} "
            "manual registration site(s) with declarative class hooks."
        )


class AuthoritativeMappingPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum(
            finding.metrics.mapping_sites_for_plan() for finding in findings
        )
        return (
            f"Declare one authoritative builder/schema for `{subsystem}` and route {site_count or len(findings)} "
            "repeated mapping site(s) through it."
        )


class ClosedFamilyDispatchPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum(
            finding.metrics.dispatch_sites_for_plan(len(finding.evidence))
            for finding in findings
        )
        return (
            f"Replace {site_count or len(findings)} branch or dispatch site(s) in `{subsystem}` with one "
            "enum/type-keyed registry or rule table."
        )


class BidirectionalRegistryPlanStepBuilder(PatternPlanStepBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> str:
        site_count = sum(
            finding.metrics.registration_sites_for_plan() for finding in findings
        )
        return (
            f"Centralize forward/reverse lookup for `{subsystem}` in one bidirectional registry and delete "
            f"{site_count or len(findings)} mirrored update site(s)."
        )


def _combined_evidence(
    findings: tuple[RefactorFinding, ...],
) -> tuple[SourceLocation, ...]:
    seen: set[tuple[str, int, str]] = set()
    evidence: list[SourceLocation] = []
    for finding in findings:
        for item in finding.evidence:
            key = (item.file_path, item.line, item.symbol)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    return tuple(sorted(evidence, key=lambda item: (item.file_path, item.line))[:8])


def _evidence_symbols(findings: tuple[RefactorFinding, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for finding in findings:
        for item in finding.evidence:
            if item.symbol in seen:
                continue
            seen.add(item.symbol)
            ordered.append(item.symbol)
    return tuple(ordered)


class PatternActionBuilder(ABC):
    @abstractmethod
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class ActionTemplate:
    kind: str
    description: str
    confidence: object


class GenericPatternActionBuilder(PatternActionBuilder):
    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        template = ActionTemplate(
            kind="apply_pattern",
            description=f"Apply Pattern {pattern_id}: {PATTERN_SPECS[pattern_id].prescription}",
            confidence=MEDIUM_CONFIDENCE,
        )
        return _actions_from_templates(subsystem, findings, (template,))


class TemplatedPatternActionBuilder(PatternActionBuilder):
    def __init__(self, templates: tuple[ActionTemplate, ...]) -> None:
        self.templates = templates

    def build(
        self,
        subsystem: str,
        pattern_id: int,
        findings: tuple[RefactorFinding, ...],
    ) -> tuple[RefactorAction, ...]:
        return _actions_from_templates(subsystem, findings, self.templates)


def _actions_from_templates(
    subsystem: str,
    findings: tuple[RefactorFinding, ...],
    templates: tuple[ActionTemplate, ...],
) -> tuple[RefactorAction, ...]:
    evidence = _combined_evidence(findings)
    symbols = _evidence_symbols(findings)
    return tuple(
        RefactorAction(
            kind=template.kind,
            description=template.description.format(subsystem=subsystem),
            target=subsystem,
            symbols=symbols,
            evidence=evidence,
            confidence=template.confidence,
        )
        for template in templates
    )


_GENERIC_PATTERN_PLAN_STEP_BUILDER = GenericPatternPlanStepBuilder()
_PATTERN_PLAN_STEP_BUILDERS: dict[int, PatternPlanStepBuilder] = {
    3: ClosedFamilyDispatchPlanStepBuilder(),
    5: TemplateMethodPlanStepBuilder(),
    6: AutoRegisterPlanStepBuilder(),
    13: BidirectionalRegistryPlanStepBuilder(),
    14: AuthoritativeMappingPlanStepBuilder(),
}

_GENERIC_PATTERN_ACTION_BUILDER = GenericPatternActionBuilder()
_PATTERN_ACTION_BUILDERS: dict[int, PatternActionBuilder] = {
    3: TemplatedPatternActionBuilder(
        (
            ActionTemplate(
                kind="create_dispatch_authority",
                description="Create one enum/type-keyed dispatch authority for `{subsystem}`.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="replace_branch_sites",
                description="Replace the repeated branch/lookup sites with a single dispatch entry point.",
                confidence=HIGH_CONFIDENCE,
            ),
        )
    ),
    5: TemplatedPatternActionBuilder(
        (
            ActionTemplate(
                kind="create_abc_base",
                description="Create one ABC base in `{subsystem}` for the repeated behavior family.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="extract_template_method",
                description="Move the shared orchestration statements from the repeated methods into one concrete template method.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="leave_residual_hooks",
                description="Leave only irreducible per-class residue as abstract hooks or mixin-provided concerns.",
                confidence=MEDIUM_CONFIDENCE,
            ),
        )
    ),
    6: TemplatedPatternActionBuilder(
        (
            ActionTemplate(
                kind="create_metaclass",
                description="Create `AutoRegisterMeta` or an equivalent registry base for `{subsystem}`.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="add_declarative_hooks",
                description="Add declarative class-level registration hooks such as `registry_key` on the participating classes.",
                confidence=MEDIUM_CONFIDENCE,
            ),
            ActionTemplate(
                kind="delete_manual_registration",
                description="Delete the manual registration writes after routing the family through the metaclass/base.",
                confidence=HIGH_CONFIDENCE,
            ),
        )
    ),
    13: TemplatedPatternActionBuilder(
        (
            ActionTemplate(
                kind="create_bidirectional_registry",
                description="Create one authoritative forward/reverse registry for `{subsystem}`.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="delete_mirrored_updates",
                description="Delete mirrored forward/reverse update sites once the bijective registry is in place.",
                confidence=HIGH_CONFIDENCE,
            ),
        )
    ),
    14: TemplatedPatternActionBuilder(
        (
            ActionTemplate(
                kind="create_authoritative_schema",
                description="Create one authoritative builder/schema in `{subsystem}` for the repeated field mapping family.",
                confidence=HIGH_CONFIDENCE,
            ),
            ActionTemplate(
                kind="replace_mapping_sites",
                description="Replace each repeated constructor/export/projection site with the authoritative builder/schema.",
                confidence=HIGH_CONFIDENCE,
            ),
        )
    ),
}


def build_refactor_plans(
    findings: list[RefactorFinding], root: Path
) -> list[RefactorPlan]:
    clusters = _cluster_findings(findings, root)
    plans = [_plan_for_cluster(cluster) for cluster in clusters]
    return sorted(plans, key=lambda plan: (plan.subsystem, plan.primary_pattern_id))


def _cluster_findings(
    findings: list[RefactorFinding], root: Path
) -> list[_FindingCluster]:
    if not findings:
        return []

    parents = list(range(len(findings)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, right_index in combinations(range(len(findings)), 2):
        if _relation_score(findings[left_index], findings[right_index], root) >= 3:
            union(left_index, right_index)

    grouped: dict[int, list[RefactorFinding]] = defaultdict(list)
    for index, finding in enumerate(findings):
        grouped[find(index)].append(finding)

    clusters: list[_FindingCluster] = []
    for group_findings in grouped.values():
        ordered_findings = tuple(
            sorted(
                group_findings,
                key=lambda finding: (
                    _subsystem_name((finding,), root),
                    finding.pattern_id,
                    finding.title,
                ),
            )
        )
        evidence = _cluster_evidence(ordered_findings)
        clusters.append(
            _FindingCluster(
                subsystem=_subsystem_name(ordered_findings, root),
                findings=ordered_findings,
                evidence=evidence,
            )
        )
    return sorted(
        clusters, key=lambda cluster: (cluster.subsystem, len(cluster.findings))
    )


def _relation_score(left: RefactorFinding, right: RefactorFinding, root: Path) -> int:
    score = 0
    left_paths = set(_evidence_paths(left))
    right_paths = set(_evidence_paths(right))
    if left_paths & right_paths:
        score += 3
    common_depth = _max_common_dir_depth(left_paths, right_paths, root)
    if common_depth:
        score += min(common_depth, 2)
    if set(left.capability_tags) & set(right.capability_tags):
        score += 1
    if _patterns_are_synergistic(left.pattern_id, right.pattern_id):
        score += 1
    if _shared_symbol_roots(left, right):
        score += 1
    return score


def _patterns_are_synergistic(left: int, right: int) -> bool:
    return right in _PATTERN_SYNERGY.get(left, set()) or left in _PATTERN_SYNERGY.get(
        right, set()
    )


def _shared_symbol_roots(left: RefactorFinding, right: RefactorFinding) -> bool:
    return bool(_symbol_roots(left) & _symbol_roots(right))


def _symbol_roots(finding: RefactorFinding) -> set[str]:
    roots: set[str] = set()
    for item in finding.evidence:
        symbol = item.symbol.replace(":", ".")
        root = symbol.split(".", maxsplit=1)[0]
        if root and not root.startswith("<"):
            roots.add(root)
    return roots


def _max_common_dir_depth(
    left_paths: set[Path], right_paths: set[Path], root: Path
) -> int:
    depth = 0
    for left in left_paths:
        left_parts = _safe_relative(left, root).parent.parts
        for right in right_paths:
            right_parts = _safe_relative(right, root).parent.parts
            depth = max(depth, _common_prefix_length(left_parts, right_parts))
    return depth


def _subsystem_name(findings: tuple[RefactorFinding, ...], root: Path) -> str:
    paths = [path for finding in findings for path in _evidence_paths(finding)]
    if not paths:
        return root.name

    parents = [_safe_relative(path, root).parent.parts for path in paths]
    prefix: list[str] = []
    for parts in zip(*parents):
        if all(part == parts[0] for part in parts):
            prefix.append(parts[0])
        else:
            break

    if prefix:
        return str(Path(*prefix))

    first = _safe_relative(paths[0], root)
    if first.parent != Path("."):
        return str(first.parent)
    return first.stem


def _cluster_evidence(
    findings: tuple[RefactorFinding, ...],
) -> tuple[SourceLocation, ...]:
    seen: set[tuple[str, int, str]] = set()
    evidence: list[SourceLocation] = []
    for finding in findings:
        for item in finding.evidence:
            key = (item.file_path, item.line, item.symbol)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    return tuple(sorted(evidence, key=lambda item: (item.file_path, item.line))[:8])


def _plan_for_cluster(cluster: _FindingCluster) -> RefactorPlan:
    selected_patterns = _select_pattern_cover(cluster.findings)
    ordered_patterns = _order_patterns(selected_patterns, cluster.findings)
    primary_pattern_id = ordered_patterns[0]
    outcome = _estimate_outcome(cluster.findings, ordered_patterns)
    capabilities = _unique_capabilities(cluster.findings)
    missing_capabilities = _render_tag_values(capabilities, attrgetter("label"))
    collapsed_distinctions = _render_tag_values(capabilities, attrgetter("distinction"))
    current_partial_view = _current_partial_view(cluster.findings)
    summary = _plan_summary(cluster.subsystem, ordered_patterns, cluster.findings)
    supporting_findings = tuple(
        _dedupe_preserve_order(finding.title for finding in cluster.findings)
    )
    canonical_normal_form = _canonical_normal_form(ordered_patterns)
    plan_steps = _build_plan_steps(
        cluster.subsystem, ordered_patterns, cluster.findings
    )
    actions = _build_plan_actions(cluster.subsystem, ordered_patterns, cluster.findings)
    return RefactorPlan(
        subsystem=cluster.subsystem,
        summary=summary,
        current_partial_view=current_partial_view,
        collapsed_distinctions=collapsed_distinctions,
        missing_capabilities=missing_capabilities,
        certification=_aggregate_certification(cluster.findings),
        primary_pattern_id=primary_pattern_id,
        secondary_pattern_ids=tuple(ordered_patterns[1:]),
        application_order=tuple(ordered_patterns),
        canonical_normal_form=canonical_normal_form,
        plan_steps=plan_steps,
        supporting_findings=supporting_findings,
        evidence=cluster.evidence,
        outcome=outcome,
        actions=actions,
    )


def _select_pattern_cover(findings: tuple[RefactorFinding, ...]) -> tuple[int, ...]:
    pattern_ids = tuple(sorted({finding.pattern_id for finding in findings}))
    required_capabilities = set(_unique_capabilities(findings))
    if not pattern_ids:
        return ()
    if not required_capabilities:
        return pattern_ids

    pattern_counts = Counter(finding.pattern_id for finding in findings)
    certified_counts = Counter(
        finding.pattern_id for finding in findings if finding.certification == CERTIFIED
    )

    best_subset: tuple[int, ...] | None = None
    best_score: tuple[int, int, int, tuple[int, ...]] | None = None
    for size in range(1, len(pattern_ids) + 1):
        for subset in combinations(pattern_ids, size):
            covered = set()
            for pattern_id in subset:
                covered.update(PATTERN_SPECS[pattern_id].witness_capabilities)
            if not required_capabilities <= covered:
                continue
            score = (
                sum(pattern_counts[pattern_id] for pattern_id in subset),
                sum(certified_counts[pattern_id] for pattern_id in subset),
                sum(_PATTERN_PRIORITY.get(pattern_id, 0) for pattern_id in subset),
                tuple(pattern_counts[pattern_id] for pattern_id in subset),
            )
            if best_score is None or score > best_score:
                best_subset = subset
                best_score = score
        if best_subset is not None:
            return best_subset
    return pattern_ids


def _order_patterns(
    pattern_ids: tuple[int, ...], findings: tuple[RefactorFinding, ...]
) -> list[int]:
    if not pattern_ids:
        return []

    pattern_set = set(pattern_ids)
    dependencies = {
        pattern_id: set(_PATTERN_DEPENDENCIES.get(pattern_id, set())) & pattern_set
        for pattern_id in pattern_ids
    }
    pattern_counts = Counter(finding.pattern_id for finding in findings)
    certified_counts = Counter(
        finding.pattern_id for finding in findings if finding.certification == CERTIFIED
    )

    ordered: list[int] = []
    ready = [pattern_id for pattern_id in pattern_ids if not dependencies[pattern_id]]
    while ready:
        ready.sort(
            key=lambda pattern_id: (
                _PATTERN_PRIORITY.get(pattern_id, 0),
                pattern_counts[pattern_id],
                certified_counts[pattern_id],
                -pattern_id,
            ),
            reverse=True,
        )
        pattern_id = ready.pop(0)
        if pattern_id in ordered:
            continue
        ordered.append(pattern_id)
        for candidate in pattern_ids:
            if pattern_id in dependencies[candidate]:
                dependencies[candidate].remove(pattern_id)
                if not dependencies[candidate] and candidate not in ordered:
                    ready.append(candidate)

    if len(ordered) != len(pattern_ids):
        remaining = [
            pattern_id for pattern_id in pattern_ids if pattern_id not in ordered
        ]
        remaining.sort(
            key=lambda pattern_id: (_PATTERN_PRIORITY.get(pattern_id, 0), -pattern_id),
            reverse=True,
        )
        ordered.extend(remaining)
    return ordered


def _estimate_outcome(
    findings: tuple[RefactorFinding, ...], ordered_patterns: list[int]
) -> OutcomeEstimate:
    total = ImpactDelta()

    for finding in findings:
        total += finding.metrics.outcome_delta(len(finding.evidence))

    loci_before = total.loci_of_change_before
    if loci_before == 0:
        loci_before = len(
            {
                (item.file_path, item.line)
                for finding in findings
                for item in finding.evidence
            }
        )
    loci_after = max(
        total.loci_of_change_after,
        len(ordered_patterns),
        1 if findings else 0,
    )
    upper_bound = max(
        total.lower_bound_removable_loc,
        total.upper_bound_removable_loc,
    )

    return OutcomeEstimate(
        lower_bound_removable_loc=total.lower_bound_removable_loc,
        upper_bound_removable_loc=upper_bound,
        loci_of_change_before=loci_before,
        loci_of_change_after=loci_after,
        repeated_mappings_centralized=total.repeated_mappings_centralized,
        dispatch_sites_eliminated=total.dispatch_sites_eliminated,
        registration_sites_removed=total.registration_sites_removed,
        shared_algorithm_sites_centralized=(total.shared_algorithm_sites_centralized),
    )


def _aggregate_certification(
    findings: tuple[RefactorFinding, ...],
) -> CertificationLevel:
    certifications = {finding.certification for finding in findings}
    if certifications == {CERTIFIED}:
        return CERTIFIED
    if CertificationLevel.SPECULATIVE in certifications:
        return CertificationLevel.SPECULATIVE
    return STRONG_HEURISTIC


def _plan_summary(
    subsystem: str, ordered_patterns: list[int], findings: tuple[RefactorFinding, ...]
) -> str:
    primary = PATTERN_SPECS[ordered_patterns[0]]
    if len(ordered_patterns) == 1:
        return (
            f"`{subsystem}` clusters {len(findings)} finding(s) into Pattern {primary.pattern_id} "
            f"as the authoritative refactor witness."
        )
    secondary = ", ".join(
        f"Pattern {pattern_id}" for pattern_id in ordered_patterns[1:]
    )
    return (
        f"`{subsystem}` needs Pattern {primary.pattern_id} as the primary witness, "
        f"with {secondary} as supporting helpers."
    )


def _current_partial_view(findings: tuple[RefactorFinding, ...]) -> str:
    observations = _render_tag_values(
        sorted({tag for finding in findings for tag in finding.observation_tags}),
        attrgetter("label"),
    )
    if not observations:
        return "The subsystem is currently described by mixed structural observations."
    return (
        "The subsystem is currently observed through "
        f"{_human_join(observations)}, which leaves semantic distinctions to later recovery."
    )


def _canonical_normal_form(pattern_ids: list[int]) -> str:
    primary = PATTERN_SPECS[pattern_ids[0]].canonical_shape
    if len(pattern_ids) == 1:
        return primary
    supporting = "; then ".join(
        PATTERN_SPECS[pattern_id].canonical_shape for pattern_id in pattern_ids[1:]
    )
    return f"{primary}; then {supporting}"


def _build_plan_steps(
    subsystem: str, pattern_ids: list[int], findings: tuple[RefactorFinding, ...]
) -> tuple[str, ...]:
    steps = [
        _pattern_step(subsystem, pattern_id, findings) for pattern_id in pattern_ids
    ]
    steps.append(
        f"Delete superseded partial views in `{subsystem}` and route call sites through the new authorities."
    )
    return tuple(steps)


def _pattern_step(
    subsystem: str, pattern_id: int, findings: tuple[RefactorFinding, ...]
) -> str:
    supporting = [finding for finding in findings if finding.pattern_id == pattern_id]
    builder = _PATTERN_PLAN_STEP_BUILDERS.get(
        pattern_id, _GENERIC_PATTERN_PLAN_STEP_BUILDER
    )
    return builder.build(subsystem, pattern_id, tuple(supporting))


def _build_plan_actions(
    subsystem: str, pattern_ids: list[int], findings: tuple[RefactorFinding, ...]
) -> tuple[RefactorAction, ...]:
    actions: list[RefactorAction] = []
    for pattern_id in pattern_ids:
        supporting = tuple(
            finding for finding in findings if finding.pattern_id == pattern_id
        )
        builder = _PATTERN_ACTION_BUILDERS.get(
            pattern_id, _GENERIC_PATTERN_ACTION_BUILDER
        )
        actions.extend(builder.build(subsystem, pattern_id, supporting))
    return tuple(actions)


def _render_tag_values(items, projector) -> tuple[str, ...]:
    return tuple(_dedupe_preserve_order(projector(item) for item in items))


def _unique_capabilities(findings: tuple[RefactorFinding, ...]) -> list[CapabilityTag]:
    capabilities = sorted(
        {tag for finding in findings for tag in finding.capability_tags}
    )
    return capabilities


def _human_join(items: tuple[str, ...] | list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _dedupe_preserve_order(items) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _evidence_paths(finding: RefactorFinding) -> tuple[Path, ...]:
    paths = {Path(item.file_path) for item in finding.evidence}
    return tuple(sorted(paths))


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    depth = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        depth += 1
    return depth
