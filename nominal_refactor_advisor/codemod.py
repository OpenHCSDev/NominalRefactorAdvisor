"""Codemod planning primitives anchored to source-index AST geometry.

The advisor does not apply edits here. It represents target-level rewrite plans,
simulates their effect over source text, and validates the resulting source with
the best parser available in the local environment.

The carrier-factorization signal is intentionally algebraic rather than tied to
carrier names: it detects cancelable compositions where a function maps product
fields through pack/forward/unpack steps without changing those fields or owning
an invariant. In categorical terms, these are identity-like morphisms between
product carriers whose common factors can be cancelled before a codemod runner
materializes a rewrite.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from .collection_algebra import sorted_tuple
from .impact_ranking import (
    RefactorImpactKey,
    RefactorImpactOpportunity,
    RefactorImpactRankingReport,
)
from .models import ImpactDelta
from .patterns import PatternId
from .source_index import AstTargetDigest, SourceIndex


class RewriteOperation(StrEnum):
    """Supported source-index anchored rewrite operations."""

    REPLACE_TARGET = "replace_target"


class CodemodBackend(StrEnum):
    """Parser backend used to validate simulated rewrite output."""

    AST_SPAN = "ast_span"
    LIBCST = "libcst"


class CodemodCandidateOrigin(StrEnum):
    """Where an advisor codemod candidate came from."""

    IMPACT_OPPORTUNITY = "impact_opportunity"
    TRAJECTORY_STEP = "trajectory_step"


class CodemodAutomationLevel(StrEnum):
    """How much executable authority the advisor has for a candidate."""

    SAFE_MECHANICAL = "safe_mechanical"
    SIMULATABLE_REWRITE = "simulatable_rewrite"
    ADVISORY_ONLY = "advisory_only"


class CodemodSimulationStatus(StrEnum):
    """Whether a candidate currently has source rewrites that can be simulated."""

    NO_REWRITE_PLAN = "no_rewrite_plan"
    READY_TO_SIMULATE = "ready_to_simulate"


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions that can be factored away."""

    PRODUCT_PACK_FORWARD = "product_pack_forward"
    PACK_UNPACK_FORWARD = "pack_unpack_forward"


@dataclass(frozen=True)
class PlannedSourceRewrite:
    """One planned source rewrite against an AST target digest."""

    target_id: str
    replacement_source: str
    operation: RewriteOperation = RewriteOperation.REPLACE_TARGET
    rationale: str = ""


@dataclass(frozen=True)
class CodemodStrategy:
    """Registry metadata for one codemod strategy."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    reason: str
    safe_to_apply: bool = False


@dataclass(frozen=True)
class CodemodApplicability:
    """Concrete codemod applicability for one candidate."""

    strategy_id: str
    automation_level: CodemodAutomationLevel
    simulation_status: CodemodSimulationStatus
    safe_to_apply: bool
    target_count: int
    planned_rewrite_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "automation_level": self.automation_level.value,
            "simulation_status": self.simulation_status.value,
            "safe_to_apply": self.safe_to_apply,
            "target_count": self.target_count,
            "planned_rewrite_count": self.planned_rewrite_count,
            "reason": self.reason,
        }


SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="semantic-structural-advisory",
    automation_level=CodemodAutomationLevel.ADVISORY_ONLY,
    safe_to_apply=False,
    reason=(
        "Semantic structural findings identify source targets and refactor shape, "
        "but choosing the new authority boundary is not a safe mechanical edit."
    ),
)


MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY = CodemodStrategy(
    strategy_id="mixed-semantic-structural-advisory",
    automation_level=CodemodAutomationLevel.ADVISORY_ONLY,
    safe_to_apply=False,
    reason=(
        "The opportunity spans multiple semantic pattern families, so the advisor "
        "keeps it advisory until a caller supplies an explicit rewrite plan."
    ),
)


class CodemodStrategyRegistry:
    """Lookup table for codemod strategy metadata by refactoring pattern."""

    def __init__(
        self,
        pattern_strategies: Mapping[PatternId, CodemodStrategy] | None = None,
        *,
        fallback_strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
        mixed_strategy: CodemodStrategy = MIXED_SEMANTIC_ADVISORY_CODEMOD_STRATEGY,
    ) -> None:
        self._pattern_strategies = dict(pattern_strategies or {})
        self._fallback_strategy = fallback_strategy
        self._mixed_strategy = mixed_strategy

    def strategy_for_opportunity(
        self, opportunity: RefactorImpactOpportunity
    ) -> CodemodStrategy:
        strategies = {
            self._pattern_strategies[pattern_id]
            for pattern_id in _opportunity_pattern_ids(opportunity)
            if pattern_id in self._pattern_strategies
        }
        if len(strategies) == 1:
            return next(iter(strategies))
        if len(strategies) > 1:
            return self._mixed_strategy
        return self._fallback_strategy

    def applicability_for_candidate(
        self, candidate: "CodemodCandidate"
    ) -> CodemodApplicability:
        strategy = candidate.strategy
        simulation_status = (
            CodemodSimulationStatus.READY_TO_SIMULATE
            if candidate.planned_rewrites
            else CodemodSimulationStatus.NO_REWRITE_PLAN
        )
        return CodemodApplicability(
            strategy_id=strategy.strategy_id,
            automation_level=strategy.automation_level,
            simulation_status=simulation_status,
            safe_to_apply=strategy.safe_to_apply,
            target_count=candidate.target_count,
            planned_rewrite_count=len(candidate.planned_rewrites),
            reason=strategy.reason,
        )


DEFAULT_CODEMOD_STRATEGY_REGISTRY = CodemodStrategyRegistry()


@dataclass(frozen=True)
class SimulatedSourceRewrite:
    """Resolved source span and replacement preview for one planned rewrite."""

    target_id: str
    file_path: str
    qualname: str
    operation: RewriteOperation
    line: int
    end_line: int
    original_source: str
    replacement_source: str
    rationale: str = ""


@dataclass(frozen=True)
class CodemodSimulationReport:
    """Result of simulating planned rewrites without writing files."""

    backend: CodemodBackend
    rewrites: tuple[SimulatedSourceRewrite, ...]
    rewritten_sources: dict[str, str]

    @property
    def applied_rewrite_count(self) -> int:
        return len(self.rewrites)

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.rewritten_sources))


@dataclass(frozen=True)
class CodemodCandidate:
    """Impact-ranked rewrite candidate with optional executable rewrite plans."""

    origin: CodemodCandidateOrigin
    opportunity: RefactorImpactOpportunity
    target_ids: tuple[str, ...]
    planned_rewrites: tuple[PlannedSourceRewrite, ...] = ()
    strategy: CodemodStrategy = SEMANTIC_ADVISORY_CODEMOD_STRATEGY

    @property
    def candidate_id(self) -> str:
        return _candidate_id(self.opportunity, self.target_ids)

    @property
    def opportunity_key(self) -> RefactorImpactKey:
        return self.opportunity.key

    @property
    def covered_finding_ids(self) -> tuple[str, ...]:
        return self.opportunity.covered_finding_ids

    @property
    def predicted_removed_finding_count(self) -> int:
        return self.opportunity.predicted_removed_finding_count

    @property
    def impact_delta(self) -> ImpactDelta:
        return self.opportunity.impact_delta

    @property
    def load_bearing_score(self) -> int:
        return self.opportunity.load_bearing_score

    @property
    def target_count(self) -> int:
        return len(self.target_ids)

    @property
    def has_planned_rewrites(self) -> bool:
        return bool(self.planned_rewrites)

    @property
    def applicability(self) -> CodemodApplicability:
        return DEFAULT_CODEMOD_STRATEGY_REGISTRY.applicability_for_candidate(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "opportunity_key": self.opportunity_key.to_dict(),
            "target_ids": self.target_ids,
            "covered_finding_ids": self.covered_finding_ids,
            "predicted_removed_finding_count": self.predicted_removed_finding_count,
            "load_bearing_score": self.load_bearing_score,
            "has_planned_rewrites": self.has_planned_rewrites,
            "planned_rewrite_count": len(self.planned_rewrites),
            "applicability": self.applicability.to_dict(),
        }

    def with_planned_rewrites(
        self, rewrites: Iterable[PlannedSourceRewrite]
    ) -> "CodemodCandidate":
        return replace(self, planned_rewrites=tuple(rewrites))

    def with_replacement(
        self,
        target_id: str,
        replacement_source: str,
        *,
        rationale: str = "",
    ) -> "CodemodCandidate":
        if target_id not in self.target_ids:
            raise ValueError(
                f"Target {target_id!r} is not covered by candidate {self.candidate_id}"
            )
        rewrite = PlannedSourceRewrite(
            target_id=target_id,
            replacement_source=replacement_source,
            rationale=rationale,
        )
        return replace(self, planned_rewrites=(*self.planned_rewrites, rewrite))

    def simulate(
        self,
        source_index: SourceIndex,
        source_by_path: Mapping[str, str],
        *,
        backend: CodemodBackend | None = None,
    ) -> CodemodSimulationReport:
        if not self.planned_rewrites:
            raise ValueError(
                f"Candidate {self.candidate_id} has no planned source rewrites"
            )
        return simulate_planned_rewrites(
            source_index,
            self.planned_rewrites,
            source_by_path,
            backend=backend,
        )


@dataclass(frozen=True)
class CancelableCompositionSignal:
    """Generic factorable morphism over product carrier fields."""

    target_id: str
    file_path: str
    qualname: str
    line: int
    end_line: int
    composition_kind: CancelableCompositionKind
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]
    covered_finding_ids: tuple[str, ...] = ()

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def covered_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def load_bearing_score(self) -> int:
        immediate_unpack_bonus = (
            75
            if self.composition_kind == CancelableCompositionKind.PACK_UNPACK_FORWARD
            else 25
        )
        return (
            self.field_count * 50
            + self.covered_finding_count * 100
            + immediate_unpack_bonus
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def codemod_candidates_from_impact_ranking(
    impact_ranking: RefactorImpactRankingReport,
    source_index: SourceIndex,
    *,
    include_trajectory_steps: bool = True,
    strategy_registry: CodemodStrategyRegistry | None = None,
) -> tuple[CodemodCandidate, ...]:
    """Project impact-ranking opportunities into source-index codemod candidates."""

    registry = strategy_registry or DEFAULT_CODEMOD_STRATEGY_REGISTRY
    candidates_by_id: dict[str, CodemodCandidate] = {}
    for opportunity in impact_ranking.opportunities:
        candidate = _candidate_from_opportunity(
            opportunity,
            source_index,
            CodemodCandidateOrigin.IMPACT_OPPORTUNITY,
            registry,
        )
        if candidate is not None:
            candidates_by_id[candidate.candidate_id] = candidate

    if include_trajectory_steps:
        for trajectory in impact_ranking.trajectories:
            for step in trajectory.steps:
                candidate = _candidate_from_opportunity(
                    step.opportunity,
                    source_index,
                    CodemodCandidateOrigin.TRAJECTORY_STEP,
                    registry,
                )
                if candidate is not None:
                    candidates_by_id.setdefault(candidate.candidate_id, candidate)

    return sorted_tuple(
        candidates_by_id.values(),
        key=lambda item: (
            -item.load_bearing_score,
            -item.predicted_removed_finding_count,
            item.opportunity_key.kind,
            item.opportunity_key.value,
            item.target_ids,
        ),
    )


def detect_cancelable_composition_signals(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> tuple[CancelableCompositionSignal, ...]:
    """Detect generic pack/unpack/forward compositions worth factoring away."""

    nodes_by_target_id = _function_nodes_by_target_id(source_index, source_by_path)
    signals = []
    for target in source_index.ast_targets:
        if target.node_type not in {"function", "method"}:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if node is None:
            continue
        signal = _cancelable_composition_signal_for_target(source_index, target, node)
        if signal is not None:
            signals.append(signal)
    return sorted_tuple(
        signals,
        key=lambda item: (
            -item.load_bearing_score,
            item.file_path,
            item.line,
            item.qualname,
        ),
    )


def libcst_available() -> bool:
    """Return whether LibCST is importable in the current environment."""

    return importlib.util.find_spec("libcst") is not None


def select_codemod_backend(*, prefer_libcst: bool = True) -> CodemodBackend:
    """Select the validation backend without requiring optional dependencies."""

    if prefer_libcst and libcst_available():
        return CodemodBackend.LIBCST
    return CodemodBackend.AST_SPAN


def simulate_planned_rewrites(
    source_index: SourceIndex,
    rewrites: Iterable[PlannedSourceRewrite],
    source_by_path: Mapping[str, str],
    *,
    backend: CodemodBackend | None = None,
) -> CodemodSimulationReport:
    """Simulate source-index target replacements over in-memory source text."""

    selected_backend = backend or select_codemod_backend()
    resolved = _resolve_rewrites(source_index, tuple(rewrites), source_by_path)
    _validate_non_overlapping(resolved)

    sources = dict(source_by_path)
    simulated: list[SimulatedSourceRewrite] = []
    for file_path in sorted({target.file_path for _, target in resolved}):
        file_rewrites = [
            (rewrite, target)
            for rewrite, target in resolved
            if target.file_path == file_path
        ]
        lines = sources[file_path].splitlines(keepends=True)
        for rewrite, target in sorted(
            file_rewrites,
            key=lambda item: (item[1].line, item[1].end_line),
            reverse=True,
        ):
            start_index = target.line - 1
            end_index = target.end_line
            if start_index < 0 or end_index > len(lines):
                raise ValueError(
                    f"Target {target.target_id!r} span is outside {file_path!r}"
                )
            original_source = "".join(lines[start_index:end_index])
            replacement_lines = _replacement_lines(rewrite.replacement_source)
            lines[start_index:end_index] = replacement_lines
            simulated.append(
                SimulatedSourceRewrite(
                    target_id=target.target_id,
                    file_path=file_path,
                    qualname=target.qualname,
                    operation=rewrite.operation,
                    line=target.line,
                    end_line=target.end_line,
                    original_source=original_source,
                    replacement_source="".join(replacement_lines),
                    rationale=rewrite.rationale,
                )
            )
        sources[file_path] = "".join(lines)
        _validate_source(sources[file_path], file_path, selected_backend)

    changed_sources = {
        file_path: sources[file_path]
        for file_path in sorted({target.file_path for _, target in resolved})
    }
    return CodemodSimulationReport(
        backend=selected_backend,
        rewrites=sorted_tuple(
            simulated,
            key=lambda item: (item.file_path, item.line, item.end_line, item.qualname),
        ),
        rewritten_sources=changed_sources,
    )


def _resolve_rewrites(
    source_index: SourceIndex,
    rewrites: tuple[PlannedSourceRewrite, ...],
    source_by_path: Mapping[str, str],
) -> tuple[tuple[PlannedSourceRewrite, AstTargetDigest], ...]:
    resolved = []
    for rewrite in rewrites:
        if rewrite.operation != RewriteOperation.REPLACE_TARGET:
            raise ValueError(f"Unsupported rewrite operation: {rewrite.operation}")
        target = source_index.target_by_id.get(rewrite.target_id)
        if target is None:
            raise KeyError(f"Unknown source-index target id: {rewrite.target_id}")
        if target.file_path not in source_by_path:
            raise KeyError(f"Missing source text for {target.file_path!r}")
        resolved.append((rewrite, target))
    return tuple(resolved)


def _candidate_from_opportunity(
    opportunity: RefactorImpactOpportunity,
    source_index: SourceIndex,
    origin: CodemodCandidateOrigin,
    strategy_registry: CodemodStrategyRegistry,
) -> CodemodCandidate | None:
    target_ids = source_index.target_ids_for_finding_ids(
        opportunity.covered_finding_ids
    )
    if not target_ids:
        return None
    return CodemodCandidate(
        origin=origin,
        opportunity=opportunity,
        target_ids=target_ids,
        strategy=strategy_registry.strategy_for_opportunity(opportunity),
    )


def _candidate_id(
    opportunity: RefactorImpactOpportunity, target_ids: tuple[str, ...]
) -> str:
    payload = "|".join(
        (
            opportunity.key.kind,
            opportunity.key.value,
            *opportunity.covered_finding_ids,
            *target_ids,
        )
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


def _opportunity_pattern_ids(
    opportunity: RefactorImpactOpportunity,
) -> tuple[PatternId, ...]:
    pattern_ids: list[PatternId] = []
    for pattern_id in opportunity.pattern_ids:
        try:
            pattern_ids.append(PatternId(pattern_id))
        except ValueError:
            continue
    return tuple(pattern_ids)


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _ProductForward:
    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


class _FunctionNodeIndexer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.nodes_by_geometry: dict[tuple[str, int, int], _FunctionNode] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: _FunctionNode) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        end_line = node.end_lineno or node.lineno
        self.nodes_by_geometry[(qualname, node.lineno, end_line)] = node
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()


def _function_nodes_by_target_id(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> dict[str, _FunctionNode]:
    nodes_by_file_geometry: dict[str, dict[tuple[str, int, int], _FunctionNode]] = {}
    for file_path, source in source_by_path.items():
        tree = ast.parse(source, filename=file_path)
        indexer = _FunctionNodeIndexer()
        indexer.visit(tree)
        nodes_by_file_geometry[file_path] = indexer.nodes_by_geometry

    nodes_by_target_id: dict[str, _FunctionNode] = {}
    for target in source_index.ast_targets:
        if target.node_type not in {"function", "method"}:
            continue
        geometry = (target.qualname, target.line, target.end_line)
        node = nodes_by_file_geometry.get(target.file_path, {}).get(geometry)
        if node is not None:
            nodes_by_target_id[target.target_id] = node
    return nodes_by_target_id


def _cancelable_composition_signal_for_target(
    source_index: SourceIndex,
    target: AstTargetDigest,
    node: _FunctionNode,
) -> CancelableCompositionSignal | None:
    pack_forward = _return_pack_forward(node)
    if pack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PRODUCT_PACK_FORWARD,
            pack_forward,
        )

    pack_unpack_forward = _pack_then_unpack_forward(node)
    if pack_unpack_forward is not None:
        return _cancelable_signal(
            source_index,
            target,
            CancelableCompositionKind.PACK_UNPACK_FORWARD,
            pack_unpack_forward,
        )
    return None


def _cancelable_signal(
    source_index: SourceIndex,
    target: AstTargetDigest,
    composition_kind: CancelableCompositionKind,
    product_forward: _ProductForward,
) -> CancelableCompositionSignal:
    return CancelableCompositionSignal(
        target_id=target.target_id,
        file_path=target.file_path,
        qualname=target.qualname,
        line=target.line,
        end_line=target.end_line,
        composition_kind=composition_kind,
        carrier_name=product_forward.carrier_name,
        source_name=product_forward.source_name,
        field_names=product_forward.field_names,
        covered_finding_ids=source_index.finding_ids_for_target_id(target.target_id),
    )


def _return_pack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return None
    value = node.body[0].value
    if not isinstance(value, ast.Call):
        return None
    return _product_forward_from_call(value)


def _pack_then_unpack_forward(node: _FunctionNode) -> _ProductForward | None:
    if len(node.body) != 2:
        return None
    assignment, returned = node.body
    if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
        return None
    assigned_name = assignment.targets[0]
    if not isinstance(assigned_name, ast.Name):
        return None
    if not isinstance(assignment.value, ast.Call):
        return None
    if not isinstance(returned, ast.Return) or returned.value is None:
        return None

    pack = _product_forward_from_call(assignment.value)
    if pack is None:
        return None
    unpacked_fields = _unpacked_fields_from_return(returned.value, assigned_name.id)
    if len(unpacked_fields) < 2:
        return None
    common_fields = sorted_tuple(set(pack.field_names) & set(unpacked_fields))
    if len(common_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=pack.carrier_name,
        source_name=pack.source_name,
        field_names=common_fields,
    )


def _product_forward_from_call(call: ast.Call) -> _ProductForward | None:
    carrier_name = _call_name(call.func)
    if carrier_name is None:
        return None

    source_name: str | None = None
    field_names: list[str] = []
    for argument in call.args:
        projected = _attribute_projection(argument)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        projected = _attribute_projection(keyword.value)
        if projected is None:
            return None
        candidate_source_name, field_name = projected
        if keyword.arg != field_name:
            return None
        source_name = _consistent_source_name(source_name, candidate_source_name)
        if source_name is None:
            return None
        field_names.append(field_name)

    unique_fields = sorted_tuple(set(field_names))
    if source_name is None or len(unique_fields) < 2:
        return None
    return _ProductForward(
        carrier_name=carrier_name,
        source_name=source_name,
        field_names=unique_fields,
    )


def _unpacked_fields_from_return(
    value: ast.expr, carrier_variable_name: str
) -> tuple[str, ...]:
    if isinstance(value, ast.Call):
        fields: list[str] = []
        for argument in value.args:
            field_name = _field_from_carrier_attribute(argument, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        for keyword in value.keywords:
            if keyword.arg is None:
                return ()
            field_name = _field_from_carrier_attribute(
                keyword.value, carrier_variable_name
            )
            if field_name is None or keyword.arg != field_name:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))

    if isinstance(value, (ast.Tuple, ast.List)):
        fields = []
        for element in value.elts:
            field_name = _field_from_carrier_attribute(element, carrier_variable_name)
            if field_name is None:
                return ()
            fields.append(field_name)
        return sorted_tuple(set(fields))
    return ()


def _field_from_carrier_attribute(
    node: ast.expr, carrier_variable_name: str
) -> str | None:
    projected = _attribute_projection(node)
    if projected is None:
        return None
    source_name, field_name = projected
    if source_name != carrier_variable_name:
        return None
    return field_name


def _attribute_projection(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    return ast.unparse(node.value), node.attr


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def _consistent_source_name(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    if current == candidate:
        return current
    return None


def _validate_non_overlapping(
    resolved: tuple[tuple[PlannedSourceRewrite, AstTargetDigest], ...],
) -> None:
    spans_by_file: dict[str, list[tuple[int, int, str]]] = {}
    for _, target in resolved:
        spans_by_file.setdefault(target.file_path, []).append(
            (target.line - 1, target.end_line, target.target_id)
        )
    for file_path, spans in spans_by_file.items():
        ordered_spans = sorted(spans)
        _, previous_end, previous_id = ordered_spans[0]
        for start, end, target_id in ordered_spans[1:]:
            if start < previous_end:
                raise ValueError(
                    "Overlapping rewrites for "
                    f"{file_path!r}: {previous_id!r} and {target_id!r}"
                )
            previous_end, previous_id = end, target_id


def _replacement_lines(replacement_source: str) -> list[str]:
    if replacement_source and not replacement_source.endswith(("\n", "\r")):
        replacement_source = f"{replacement_source}\n"
    return replacement_source.splitlines(keepends=True)


def _validate_source(source: str, file_path: str, backend: CodemodBackend) -> None:
    if backend == CodemodBackend.LIBCST:
        import libcst as cst

        cst.parse_module(source)
        return
    ast.parse(source, filename=file_path)
