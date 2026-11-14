"""Surface and refinement detector implementations.

This module holds the later detector classes plus the public detector factory
surface.
"""

from __future__ import annotations

from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Sequence, Self, Iterable, ClassVar, cast
from nominal_refactor_advisor.ast_tools import ParsedModule, statements_without_docstring, ScopedAstObservation
from nominal_refactor_advisor.collection_algebra import sorted_tuple
from nominal_refactor_advisor.models import (
    FindingBuildContext,
    MappingMetrics,
    RefactorFinding,
    RepeatedMethodMetrics,
    SourceLocation,
    EvidenceLocated,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.detectors._base import (
    CandidateFindingDetector,
    declare_candidate_rule_detector,
    DetectorConfig,
    high_confidence_spec,
    IssueDetector,
    ModuleCollectorCandidateDetector,
    PerModuleIssueDetector,
    CollectsItself,
    ZippedSourceLocationEvidenceProperty,
    _module_builder_call_shapes,
)
from nominal_refactor_advisor.detectors._helpers import (
    DerivedQuerySpecsBySignature,
    NamedStringSequenceSpecsByName,
    HELPER_SUPPORT_PROJECTION_AUTHORITY,
    _IMPLICIT_METHOD_PARAMETER_NAMES,
    _SEQUENCE_WRAPPER_CALL_NAMES,
    _call_display_name,
    _name_mentions,
    _raised_exception_name,
)
from nominal_refactor_advisor.detectors._substrate_support import ModuleNamedValues
import ast
from nominal_refactor_advisor.class_families.parsed_index import ClassStatement, ModuleClassStatements
from nominal_refactor_advisor.ast_projection import StatementValues, CallKeywords, ExpressionReference, nodes_of
from nominal_refactor_advisor.detectors._candidates import WitnessCarrierCandidate
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.observation_shapes import BuilderCallShape
from nominal_refactor_advisor.semantic_match import Maybe


@dataclass(frozen=True)
class FragmentedFamilyAuthorityCandidate(CollectsItself):
    file_path: str
    mapping_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    key_family_name: str
    shared_keys: tuple[str, ...]
    total_keys: tuple[str, ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        family_maps: NamedStringSequenceSpecsByName = defaultdict(list)
        for binding in module.fact(ModuleNamedValues).bindings:
            if not isinstance(binding.value, ast.Dict):
                continue
            key_pairs = tuple(
                (
                    key_pair
                    for key_pair in (
                        cls._enum_key_family(key)
                        for key in binding.value.keys
                        if key is not None
                    )
                    if key_pair is not None
                )
            )
            if len(key_pairs) < 2 or len(key_pairs) != len(binding.value.keys):
                continue
            family_names = {family_name for family_name, _ in key_pairs}
            if len(family_names) != 1:
                continue
            family_name = next(iter(family_names))
            key_names = sorted_tuple((member_name for _, member_name in key_pairs))
            family_maps[family_name].append((binding.name, binding.line, key_names))

        candidates: list[Self] = []
        for family_name, entries in family_maps.items():
            if len(entries) < 2:
                continue
            key_counter: Counter[str] = Counter(
                (key_name for _, _, key_names in entries for key_name in set(key_names))
            )
            shared_keys = sorted_tuple(
                (key for key, count in key_counter.items() if count >= 2)
            )
            if len(shared_keys) < 3:
                continue
            total_keys = sorted_tuple(key_counter)
            ordered_entries = sorted(entries, key=lambda item: item[1])
            candidates.append(
                cls(
                    file_path=module.file_path,
                    mapping_names=tuple((item[0] for item in ordered_entries)),
                    line_numbers=tuple((item[1] for item in ordered_entries)),
                    key_family_name=family_name,
                    shared_keys=shared_keys,
                    total_keys=total_keys,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _enum_key_family(node: ast.AST) -> tuple[str, str] | None:
        if not isinstance(node, ast.Attribute):
            return None
        if not isinstance(node.value, ast.Name):
            return None
        return (node.value.id, node.attr)


class FragmentedFamilyAuthorityDetector(
    ModuleCollectorCandidateDetector[FragmentedFamilyAuthorityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel key-family tables should become one authoritative record",
        "Several dicts keyed by the same family together describe one record. Merge them into one record that owns all the fields.",
        "one enum-keyed record that holds everything",
        "one key family is split across parallel metadata tables",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _finding_for_candidate(
        self, authority_candidate: FragmentedFamilyAuthorityCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(authority_candidate.file_path, line, name)
                for name, line in zip(
                    authority_candidate.mapping_names,
                    authority_candidate.line_numbers,
                    strict=True,
                )
            )
        )
        return self.build_finding(
            (
                f"Tables {', '.join(authority_candidate.mapping_names)} split one `{authority_candidate.key_family_name}` metadata family across {len(authority_candidate.mapping_names)} authorities."
            ),
            evidence[:6],
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(authority_candidate.mapping_names),
                mapping_name=f"{authority_candidate.key_family_name} spec",
                field_names=authority_candidate.shared_keys,
            ),
        )


@dataclass(frozen=True)
class DerivedQueryIndexCandidate(CollectsItself, EvidenceLocated):
    file_path: str
    line_numbers: tuple[int, ...]
    function_names: tuple[str, ...]
    source_expression: str
    query_key_names: tuple[str, ...]
    return_expressions: tuple[str, ...]
    exception_names: tuple[str, ...]

    evidence = ZippedSourceLocationEvidenceProperty("line_numbers", "function_names")

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        grouped: DerivedQuerySpecsBySignature = defaultdict(list)
        for qualname, function in module.syntax_index.named_functions:
            signature = cls._linear_query_signature(function)
            if signature is None:
                continue
            source_expression, query_key_names, return_expression, exception_name = (
                signature
            )
            grouped[
                (
                    cls._linear_query_source_owner(qualname, source_expression),
                    source_expression,
                    return_expression,
                    exception_name,
                )
            ].append((qualname, function.lineno, query_key_names))
        candidates: list[Self] = []
        for (
            _source_owner,
            source_expression,
            return_expression,
            exception_name,
        ), entries in grouped.items():
            if len(entries) < 2:
                continue
            ordered = sorted_tuple(entries, key=lambda item: (item[1], item[0]))
            query_key_names = sorted_tuple(
                {
                    key_name
                    for _, _, entry_query_key_names in ordered
                    for key_name in entry_query_key_names
                }
            )
            candidates.append(
                cls(
                    file_path=module.file_path,
                    line_numbers=tuple((item[1] for item in ordered)),
                    function_names=tuple((item[0] for item in ordered)),
                    source_expression=source_expression,
                    query_key_names=query_key_names,
                    return_expressions=tuple((return_expression for _ in ordered)),
                    exception_names=(exception_name,),
                )
            )
        return sorted_tuple(
            candidates,
            key=lambda item: (item.file_path, item.source_expression, item.function_names),
        )

    @staticmethod
    def _linear_query_signature(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, tuple[str, ...], str, str] | None:
        body = statements_without_docstring(node.body)
        return (
            Maybe.of(DerivedQueryIndexCandidate._linear_query_loop(body))
            .combine(
                lambda loop: DerivedQueryIndexCandidate._linear_query_return_expr(
                    loop, cast(ast.Name, loop.target).id
                ),
                lambda loop, return_expr: (loop, return_expr),
            )
            .combine(
                lambda _context: DerivedQueryIndexCandidate._linear_query_raised_exception(body),
                lambda context, raised: (context[0], context[1], raised),
            )
            .combine(
                lambda context: DerivedQueryIndexCandidate._linear_query_key_names(
                    node, context[0], context[1], context[2][1]
                ),
                lambda context, query_key_names: (
                    (
                        ast.unparse(context[0].iter),
                        query_key_names,
                        ast.unparse(context[1]),
                        context[2][0],
                    )
                    if query_key_names
                    else None
                ),
            )
            .unwrap_or_none()
        )

    @staticmethod
    def _linear_query_source_owner(
        qualname: str,
        source_expression: str,
    ) -> str | None:
        expression = ast.parse(source_expression, mode="eval").body
        if not (
            HELPER_SUPPORT_PROJECTION_AUTHORITY.expression_root_names(expression)
            & _IMPLICIT_METHOD_PARAMETER_NAMES
        ):
            return None
        owner_name, separator, _method_name = qualname.rpartition(".")
        return owner_name if separator else qualname

    @staticmethod
    def _linear_query_loop(body: list[ast.stmt]) -> ast.For | None:
        if len(body) < 2:
            return None
        loop = next(
            (statement for statement in body if isinstance(statement, ast.For)), None
        )
        if loop is None or not isinstance(loop.target, ast.Name):
            return None
        return loop

    @staticmethod
    def _linear_query_return_expr(loop: ast.For, result_name: str) -> ast.AST | None:
        return_exprs = StatementValues.of(nodes_of(ast.Return, loop)).values
        if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
            return None
        return return_exprs[0]

    @staticmethod
    def _linear_query_raised_exception(
        body: list[ast.stmt],
    ) -> tuple[str, tuple[str, ...]] | None:
        raised = next(
            (
                _raised_exception_name(statement)
                for statement in body
                if _raised_exception_name(statement) is not None
            ),
            None,
        )
        if raised is None:
            return None
        exception_name, exception_names = raised
        if exception_name not in {"KeyError", "LookupError", "ValueError"}:
            return None
        return exception_name, exception_names

    @staticmethod
    def _linear_query_key_names(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        loop: ast.For,
        return_expr: ast.AST,
        exception_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        parameter_names = tuple(
            (
                arg.arg
                for arg in tuple(node.args.posonlyargs)
                + tuple(node.args.args)
                + tuple(node.args.kwonlyargs)
                if arg.arg not in {"self", "cls"}
            )
        )
        query_key_names = sorted_tuple(
            (
                name
                for name in parameter_names
                if _name_mentions(return_expr, name)
                or name in exception_names
                or any(
                    (
                        isinstance(current, ast.If) and _name_mentions(current.test, name)
                        for current in ast.walk(loop)
                    )
                )
            )
        )
        return query_key_names


declare_candidate_rule_detector(
    DerivedQueryIndexCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated lookup helpers should build keyed indexes from the one source of truth",
        "Several lookup helpers rescan the same immutable data to answer different key queries. Build the indexes once and look them up.",
        "one immutable family plus keyed indexes built from it",
        "the same immutable data is rescanned by several query helpers with different keys",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    ),
    summary=lambda query_candidate: (
        f"Helpers {', '.join(query_candidate.function_names[:5])} repeatedly rescan `{query_candidate.source_expression}` for keys {query_candidate.query_key_names}."
    ),
    metrics=lambda query_candidate: MappingMetrics(
        mapping_site_count=len(query_candidate.function_names),
        field_count=max(len(query_candidate.query_key_names), 1),
        mapping_name=query_candidate.function_names[0],
        field_names=query_candidate.query_key_names,
        source_name=query_candidate.source_expression,
        identity_field_names=query_candidate.query_key_names,
    ),
    detector_name="DerivedQueryIndexSurfaceDetector",
)


@dataclass(frozen=True)
class FindingAssemblyPipelineCandidate(CollectsItself, WitnessCarrierCandidate):
    method_name: str
    candidate_source_name: str
    metrics_type_name: str | None

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return tuple(
            candidate
            for declared in module.fact(ModuleClassStatements).statements
            for candidate in cls._for_class(module, declared)
        )

    @classmethod
    def _for_class(
        cls,
        module: ParsedModule,
        declared: ClassStatement,
    ) -> Iterable[Self]:
        if not cls._is_detectorish_class(declared):
            return
        node = declared.node
        method = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "_findings_for_module"
            ),
            None,
        )
        if method is None:
            return
        build_call = cls._finding_build_call(method)
        if build_call is None:
            return
        candidate_source_name = cls._candidate_source_name_from_method(method)
        if candidate_source_name is None:
            return
        metrics_type_name = cls._build_call_keyword_helper_name(build_call, "metrics")
        if metrics_type_name is None:
            return
        yield cls(
            file_path=module.file_path,
            line=method.lineno,
            subject_name=node.name,
            name_family=tuple(
                item
                for item in (
                    candidate_source_name,
                    metrics_type_name,
                )
                if item is not None
            ),
            method_name=method.name,
            candidate_source_name=candidate_source_name,
            metrics_type_name=metrics_type_name,
        )

    @staticmethod
    def _is_detectorish_class(declared: ClassStatement) -> bool:
        if declared.node.name.endswith("Detector"):
            return True
        return bool(
            IssueDetector.detector_family_base_names()
            & set(declared.base_terminal_names)
        )

    @staticmethod
    def _finding_build_call(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.Call | None:
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "build":
                continue
            value = node.func.value
            if not (
                isinstance(value, ast.Attribute)
                and value.attr == "finding_spec"
                and isinstance(value.value, ast.Name)
                and (value.value.id == "self")
            ):
                continue
            return node
        return None

    @staticmethod
    def _build_call_keyword_helper_name(
        build_call: ast.Call, keyword_name: str
    ) -> str | None:
        for keyword in build_call.keywords:
            if keyword.arg != keyword_name:
                continue
            if isinstance(keyword.value, ast.Call):
                return _call_display_name(keyword.value)
        return None

    @staticmethod
    def _candidate_source_name_from_method(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str | None:
        assigned_calls: dict[str, str] = {}
        for statement in statements_without_docstring(method.body):
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and isinstance(statement.value, ast.Call):
                    call_name = _call_display_name(statement.value)
                    if call_name is not None:
                        assigned_calls[target.id] = call_name
            if isinstance(statement, ast.For):
                iterator = statement.iter
                if isinstance(iterator, ast.Call):
                    return _call_display_name(iterator)
                if isinstance(iterator, ast.Name):
                    return assigned_calls.get(iterator.id)
        return None


class FindingAssemblyPipelineDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated finding-assembly pipeline should move into a detector base",
        "Several detectors repeat the same candidate-to-finding pipeline and differ only in small hooks. Put the pipeline in one template base and let each detector supply the hooks.",
        "candidate-driven detector base with abstract hooks and mixins",
        "the same finding-assembly steps repeat across sibling detector classes",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = FindingAssemblyPipelineCandidate.from_module(module)
        if len(candidates) < 3:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    f"{candidate.class_name}.{candidate.method_name}",
                )
                for candidate in candidates[:6]
            )
        )
        collector_names = sorted_tuple(
            {candidate.candidate_source_name for candidate in candidates}
        )
        return [
            self.build_finding(
                (
                    f"Detectors {', '.join(candidate.class_name for candidate in candidates[:5])} repeat the same candidate-to-finding pipeline over collectors {', '.join(collector_names[:4])}."
                ),
                evidence,
                FindingBuildContext(
                    metrics=RepeatedMethodMetrics.from_duplicate_family(
                        duplicate_site_count=len(candidates),
                        statement_count=3,
                        class_count=len(candidates),
                        method_symbols=tuple(
                            f"{candidate.class_name}.{candidate.method_name}"
                            for candidate in candidates
                        ),
                    ),
                ),
            )
        ]


class ProjectionBuilderAuthorityDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Rebuilding a record by projecting fields should live in one builder",
        "Several call sites build the same record by copying fields from overlapping sources, often with guards and defaults mixed in. Move that projection into one constructor, classmethod or helper.",
        "one projection builder for a repeated record family",
        "the same record is rebuilt from overlapping sources at several call sites",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for builders in self._projection_builder_groups(module, config):
            callee_name = builders[0].callee_name
            field_names = builders[0].field_names
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in builders[:6]
                )
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{callee_name}` is rebuilt across {len(builders)} projection sites over field family {field_names}, "
                        "with guards/defaults varying per site."
                    ),
                    evidence,
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(builders),
                        mapping_name=callee_name,
                        field_names=field_names,
                    ),
                )
            )
        return findings

    @staticmethod
    def _projection_builder_groups(
        module: ParsedModule,
        config: DetectorConfig,
    ) -> tuple[tuple[BuilderCallShape, ...], ...]:
        grouped: dict[tuple[str, tuple[str, ...]], list[BuilderCallShape]] = defaultdict(
            list
        )
        for builder in _module_builder_call_shapes(module):
            if builder.is_nominal_owner_projection:
                continue
            if len(builder.field_names) < max(config.min_builder_keywords, 6):
                continue
            if not all(
                (
                    ProjectionBuilderAuthorityDetector._is_projection_like_builder_value(value)
                    for value in builder.value_fingerprint
                )
            ):
                continue
            grouped[(builder.callee_name, builder.field_names)].append(builder)
        candidates: list[Sequence[BuilderCallShape]] = []
        for builders in grouped.values():
            if len(builders) < 3:
                continue
            source_arities = tuple(builder.source_arity for builder in builders)
            if min(source_arities) != 1 or max(source_arities) > 2:
                continue
            if len({builder.value_fingerprint for builder in builders}) < 2:
                continue
            if len({builder.symbol for builder in builders}) < 2:
                continue
            candidates.append(
                sorted_tuple(builders, key=lambda item: (item.file_path, item.lineno))
            )
        return sorted_tuple(
            candidates,
            key=lambda group: (group[0].file_path, group[0].lineno, group[0].callee_name),
        )

    @staticmethod
    def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
        return value_fingerprint.startswith(("Name(", "Attribute(", "IfExp(", "Constant("))


@dataclass(frozen=True)
class GuardedDelegatorCandidate(CollectsItself, WitnessCarrierCandidate):
    method_name: str
    guard_role: str
    delegate_name: str
    scope_role: str

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return tuple(
            candidate
            for declared in module.fact(ModuleClassStatements).statements
            for candidate in cls._for_class(module, declared)
        )

    @classmethod
    def _for_class(
        cls,
        module: ParsedModule,
        declared: ClassStatement,
    ) -> Iterable[Self]:
        if not cls._is_observation_spec_class(declared) or declared.is_abstract:
            return
        node = declared.node
        guarded_returns: list[tuple[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.If], ast.Return]] = []
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if statement.name not in {
                "build_from_function",
                "build_from_assign",
                "build_from_observation",
                "build_from_context",
            }:
                continue
            body = statements_without_docstring(statement.body)
            while body and isinstance(body[0], ast.Assign):
                body = body[1:]
            if len(body) != 2:
                continue
            guard, return_stmt = body
            if not isinstance(
                guard, ast.If
            ) or not HELPER_SUPPORT_PROJECTION_AUTHORITY.if_returns_none_only(guard):
                continue
            if not isinstance(return_stmt, ast.Return):
                continue
            guarded_returns.append(((statement, guard), return_stmt))
        # A bare return delegates to nothing.
        for (statement, guard), returned in StatementValues.keyed(guarded_returns).supplied:
            delegate_name = cls._delegate_name_from_return(returned.value)
            if delegate_name is None:
                continue
            yield cls(
                file_path=module.file_path,
                line=statement.lineno,
                subject_name=node.name,
                name_family=(
                    guard.test.__class__.__name__,
                    delegate_name,
                    cls._scope_role_name(guard.test),
                ),
                method_name=statement.name,
                guard_role=cls._guard_role_name(guard.test),
                delegate_name=delegate_name,
                scope_role=cls._scope_role_name(guard.test),
            )

    @staticmethod
    def _is_observation_spec_class(declared: ClassStatement) -> bool:
        if declared.node.name.endswith("ObservationSpec"):
            return True
        return bool(
            {
                "ObservationShapeSpec",
                "FunctionObservationSpec",
                "AssignObservationSpec",
                "ContextForwardingShapeSpec",
            }
            & set(declared.base_terminal_names)
        )

    @staticmethod
    def _delegate_name_from_return(node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            outer_name = _call_display_name(node)
            if outer_name in _SEQUENCE_WRAPPER_CALL_NAMES and len(node.args) == 1:
                inner = node.args[0]
                if isinstance(inner, ast.Call):
                    return _call_display_name(inner)
            return outer_name
        return None

    @staticmethod
    def _guard_role_name(node: ast.AST) -> str:
        return ScopedAstObservation.guard_role_name_from_text(ast.unparse(node))

    @staticmethod
    def _scope_role_name(node: ast.AST) -> str:
        return ScopedAstObservation.scope_role_name_from_text(ast.unparse(node))


class GuardedDelegatorSpecDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated guarded spec wrappers should collapse into mixins",
        "Several observation-spec methods differ only by a scope guard and one delegate call. Share one wrapper and add the scope guards as mixins.",
        "one shared wrapper with scope mixins",
        "guard-and-delegate logic repeats across sibling observation specs",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        candidates = GuardedDelegatorCandidate.from_module(module)
        if len(candidates) < 2:
            return []
        evidence = tuple(
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    f"{candidate.class_name}.{candidate.method_name}",
                )
                for candidate in candidates[:6]
            )
        )
        scope_roles = sorted_tuple({candidate.scope_role for candidate in candidates})
        return [
            self.build_finding(
                (
                    f"Observation specs {', '.join(candidate.class_name for candidate in candidates[:5])} repeat guarded delegation over scope roles {', '.join(scope_roles)}."
                ),
                evidence,
                FindingBuildContext(
                    metrics=RepeatedMethodMetrics.from_duplicate_family(
                        duplicate_site_count=len(candidates),
                        statement_count=2,
                        class_count=len(
                            {candidate.class_name for candidate in candidates}
                        ),
                        method_symbols=tuple(
                            f"{candidate.class_name}.{candidate.method_name}"
                            for candidate in candidates
                        ),
                    ),
                ),
            )
        ]


@dataclass(frozen=True)
class StructuralObservationPropertyCandidate(CollectsItself, WitnessCarrierCandidate):
    property_name: str
    constructor_name: str
    keyword_names: ClassVar[AliasProperty[tuple[str, ...]]] = AliasProperty(
        "name_family"
    )

    @classmethod
    def from_module(cls, module: ParsedModule) -> tuple[Self, ...]:
        return tuple(
            candidate
            for declared in module.fact(ModuleClassStatements).statements
            for candidate in cls._for_class(
                module, declared
            )
        )

    @classmethod
    def _for_class(
        cls,
        module: ParsedModule,
        declared: ClassStatement,
    ) -> Iterable[Self]:
        node = declared.node
        for method in declared.methods:
            if not method.spells("property"):
                continue
            statement = method.node
            body = statements_without_docstring(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not isinstance(returned, ast.Call):
                continue
            constructor_name = ExpressionReference.decoded(returned.func).terminal_name
            if constructor_name is None:
                continue
            keyword_names = sorted_tuple(CallKeywords.of(returned.keywords).names)
            if len(keyword_names) < 6:
                continue
            yield cls(
                file_path=module.file_path,
                line=statement.lineno,
                subject_name=node.name,
                name_family=keyword_names,
                property_name=statement.name,
                constructor_name=constructor_name,
            )


@dataclass(frozen=True)
class StructuralObservationProjectionGroup:
    property_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    candidates: tuple[StructuralObservationPropertyCandidate, ...]


class StructuralObservationProjectionDetector(
    CandidateFindingDetector[StructuralObservationProjectionGroup]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated property projection builders should share one projection substrate",
        "Several classes repeat the same property-backed constructor projection and differ only in role hooks. Use one projection template.",
        "one projection builder with role hooks",
        "the same property-backed constructor is rebuilt by hand across many classes",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[StructuralObservationProjectionGroup]:
        del config
        grouped: dict[
            (
                tuple[str, str, tuple[str, ...]],
                list[StructuralObservationPropertyCandidate],
            )
        ] = defaultdict(list)
        for candidate in StructuralObservationPropertyCandidate.from_module(module):
            grouped[
                candidate.property_name,
                candidate.constructor_name,
                candidate.keyword_names,
            ].append(candidate)
        return tuple(
            (
                StructuralObservationProjectionGroup(
                    property_name=group_key[0],
                    constructor_name=group_key[1],
                    keyword_names=group_key[2],
                    candidates=tuple(candidates),
                )
                for group_key, candidates in grouped.items()
                if len(candidates) >= 3
            )
        )

    def _finding_for_candidate(
        self, candidate: StructuralObservationProjectionGroup
    ) -> RefactorFinding:
        grouped_candidates = candidate.candidates
        evidence = tuple(
            (
                item.named(item.class_name)
                for item in grouped_candidates[:6]
            )
        )
        return self.build_finding(
            (
                f"Classes {', '.join(item.class_name for item in grouped_candidates[:5])} rebuild property `{candidate.property_name}` with the same `{candidate.constructor_name}` schema over roles {candidate.keyword_names}."
            ),
            evidence,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(grouped_candidates),
                mapping_name=candidate.constructor_name,
                field_names=candidate.keyword_names,
            ),
        )


def default_detectors() -> tuple[IssueDetector, ...]:
    """Instantiate all registered detectors in stable declaration order."""
    return tuple(
        (detector_type() for detector_type in IssueDetector.registered_detector_types())
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
