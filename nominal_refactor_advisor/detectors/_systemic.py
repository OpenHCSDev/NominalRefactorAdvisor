"""Systemic detector implementations.

This module holds the earlier detector families that focus on orchestration,
axis authority, registration, and other repo-wide architectural smells.
"""

from __future__ import annotations

import ast
from abc import ABC
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..ast_tools import (
    AstExpressionProjection,
    CompactModuleIdentity,
    PythonSourcePathPolicy,
    SourceModule,
)
from ..class_index import (
    CompactClassFamilyIndex,
    CompactClosedAxisBranchFunction,
    CompactIndexedClass,
    CompactNamedProjectionSurface,
    CompactSortedKeyCall,
)
from ..codemod import (
    AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer,
    CandidateCollectorBoilerplateFindingRecipeSynthesizer,
    DeclarativeDetectorClassFindingRecipeSynthesizer,
    DirectBuildFindingRendererFindingRecipeSynthesizer,
    InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer,
    SemanticMirrorFindingRecipeEvaluator,
)
from ..native_syntax import NativePythonSyntaxIndex
from ..registry_identity import (
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
)
from ..taxonomy import CapabilityTag, ObservationTag

from ._base import *
from ._finding_spec_defaults import FindingSpecConstructionCandidateCollector
from ._helpers import *


@dataclass(frozen=True)
class CompactConcreteTypeCaseFunctionFact(CompactModuleIdentity):
    line: int
    function_name: str
    subject_expression: str
    subject_role: str
    type_names_by_check: tuple[tuple[str, ...], ...]
    union_aliases: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class CompactImplicitSelfMixinFact:
    file_path: str
    qualname: str
    line: int
    method_names: tuple[str, ...]
    method_lines: tuple[int, ...]
    cast_type_names: tuple[str, ...]
    accessed_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactRemainingSystemicModuleProjection(CompactModuleIdentity):
    concrete_type_functions: tuple[CompactConcreteTypeCaseFunctionFact, ...]
    implicit_self_mixins: tuple[CompactImplicitSelfMixinFact, ...]


def _compact_concrete_type_case_function_facts(
    module: ParsedModule,
) -> tuple[CompactConcreteTypeCaseFunctionFact, ...]:
    union_aliases = tuple(sorted(_module_union_type_aliases(module).items()))
    facts: list[CompactConcreteTypeCaseFunctionFact] = []
    for qualname, function in _iter_named_functions(module):
        alias_sources = _top_level_attribute_aliases(function)
        grouped_checks: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for subnode in _walk_nodes(function):
            if not (
                isinstance(subnode, ast.Call)
                and len(subnode.args) == 2
                and not subnode.keywords
                and AstExpressionProjection.terminal_name(subnode.func)
                == "isinstance"
            ):
                continue
            subject_expression = _attribute_family_subject_expression(
                subnode.args[0], alias_sources=alias_sources
            )
            if subject_expression is None:
                continue
            type_node = subnode.args[1]
            type_items = (
                type_node.elts if isinstance(type_node, ast.Tuple) else (type_node,)
            )
            type_names = sorted_tuple(
                {
                    type_name
                    for item in type_items
                    if (type_name := AstExpressionProjection.terminal_name(item))
                    not in {None, "None", "NoneType"}
                }
            )
            if type_names:
                grouped_checks[subject_expression].append(type_names)
        for subject_expression, checks in sorted(grouped_checks.items()):
            facts.append(
                CompactConcreteTypeCaseFunctionFact(
                    file_path=module.file_path,
                    module_name=module.module_name,
                    line=function.lineno,
                    function_name=qualname,
                    subject_expression=subject_expression,
                    subject_role=subject_expression.rsplit(".", 1)[-1],
                    type_names_by_check=tuple(checks),
                    union_aliases=union_aliases,
                )
            )
    return sorted_tuple(
        facts,
        key=lambda item: (item.file_path, item.subject_role, item.line),
    )


def _iter_qualified_classes(
    statements: Sequence[ast.stmt], parent: str | None = None
) -> Iterator[tuple[str, ast.ClassDef]]:
    for statement in statements:
        if not isinstance(statement, ast.ClassDef):
            continue
        qualname = statement.name if parent is None else f"{parent}.{statement.name}"
        yield qualname, statement
        yield from _iter_qualified_classes(statement.body, qualname)


def _compact_implicit_self_mixin_facts(
    module: ParsedModule,
) -> tuple[CompactImplicitSelfMixinFact, ...]:
    facts: list[CompactImplicitSelfMixinFact] = []
    for qualname, class_node in _iter_qualified_classes(module.module.body):
        if not class_node.name.endswith("Mixin") or CLASS_NODE_AUTHORITY.is_abstract(
            class_node
        ):
            continue
        method_names: list[str] = []
        method_lines: list[int] = []
        cast_type_names: set[str] = set()
        accessed_attribute_names: set[str] = set()
        for method in CLASS_NODE_AUTHORITY.methods(class_node):
            if _is_abstract_method(method):
                continue
            alias_names, method_cast_types = _self_cast_alias_names(method)
            if not alias_names:
                continue
            method_names.append(method.name)
            method_lines.append(method.lineno)
            cast_type_names.update(method_cast_types)
            accessed_attribute_names.update(
                SYNTAX_PROJECTION_AUTHORITY.attribute_names_for_roots(
                    method, root_names=set(alias_names)
                )
            )
        if method_names:
            facts.append(
                CompactImplicitSelfMixinFact(
                    file_path=module.file_path,
                    qualname=qualname,
                    line=class_node.lineno,
                    method_names=tuple(method_names),
                    method_lines=tuple(method_lines),
                    cast_type_names=sorted_tuple(cast_type_names),
                    accessed_attribute_names=sorted_tuple(accessed_attribute_names),
                )
            )
    return tuple(facts)


class CompactRemainingSystemicModuleProjectionFamily(
    CollectedFamily[CompactRemainingSystemicModuleProjection]
):
    item_type = CompactRemainingSystemicModuleProjection
    cache_payload_max_bytes = 1_000_000

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[CompactRemainingSystemicModuleProjection]:
        del cls
        return [
            CompactRemainingSystemicModuleProjection(
                file_path=parsed_module.file_path,
                module_name=parsed_module.module_name,
                concrete_type_functions=_compact_concrete_type_case_function_facts(
                    parsed_module
                ),
                implicit_self_mixins=_compact_implicit_self_mixin_facts(parsed_module),
            )
        ]


def _compact_class_for_detector_name(
    class_index: CompactClassFamilyIndex,
    *,
    module_name: str,
    class_name: str,
) -> CompactIndexedClass | None:
    local_class = class_index.class_for(f"{module_name}.{class_name}")
    if local_class is not None:
        return local_class
    symbols = class_index.symbols_by_simple_name.get(class_name, ())
    if len(symbols) != 1:
        return None
    return class_index.class_for(symbols[0])


def _compact_class_display_name(
    indexed_class: CompactIndexedClass,
    class_index: CompactClassFamilyIndex,
) -> str:
    if len(class_index.symbols_by_simple_name.get(indexed_class.simple_name, ())) <= 1:
        return indexed_class.simple_name
    return indexed_class.symbol


def _compact_concrete_type_function_candidates(
    projection: CompactRemainingSystemicModuleProjection,
    class_index: CompactClassFamilyIndex,
) -> tuple[ConcreteTypeCaseFunctionCandidate, ...]:
    candidates: list[ConcreteTypeCaseFunctionCandidate] = []
    for fact in projection.concrete_type_functions:
        resolved_checks: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for type_names in fact.type_names_by_check:
            concrete_names: list[str] = []
            abstract_names: list[str] = []
            for type_name in type_names:
                indexed_class = _compact_class_for_detector_name(
                    class_index,
                    module_name=fact.module_name,
                    class_name=type_name,
                )
                if indexed_class is None:
                    continue
                display_name = _compact_class_display_name(indexed_class, class_index)
                if indexed_class.is_abstract:
                    abstract_names.append(display_name)
                else:
                    concrete_names.append(display_name)
            concrete = sorted_tuple(set(concrete_names))
            if concrete:
                resolved_checks.append((concrete, sorted_tuple(set(abstract_names))))
        concrete_class_names = sorted_tuple(
            {
                name
                for concrete_names, _abstract_names in resolved_checks
                for name in concrete_names
            }
        )
        if len(concrete_class_names) < 2:
            continue
        union_alias_names = sorted_tuple(
            alias_name
            for alias_name, member_names in fact.union_aliases
            if set(concrete_class_names) <= set(member_names)
        )
        candidates.append(
            ConcreteTypeCaseFunctionCandidate(
                file_path=fact.file_path,
                line=fact.line,
                function_name=fact.function_name,
                subject_expression=fact.subject_expression,
                subject_role=fact.subject_role,
                concrete_class_names=concrete_class_names,
                abstract_class_names=sorted_tuple(
                    {
                        name
                        for _concrete_names, abstract_names in resolved_checks
                        for name in abstract_names
                    }
                ),
                union_alias_names=union_alias_names,
                case_site_count=len(resolved_checks),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.subject_role, item.line),
    )


def _compact_common_abstract_base_names(
    projection: CompactRemainingSystemicModuleProjection,
    class_index: CompactClassFamilyIndex,
    class_names: tuple[str, ...],
) -> tuple[str, ...]:
    indexed_classes = tuple(
        indexed_class
        for class_name in class_names
        if (
            indexed_class := _compact_class_for_detector_name(
                class_index,
                module_name=projection.module_name,
                class_name=class_name,
            )
        )
        is not None
    )
    if len(indexed_classes) < 2:
        return ()
    common_symbols = set(class_index.ancestor_symbols(indexed_classes[0].symbol))
    for indexed_class in indexed_classes[1:]:
        common_symbols &= set(class_index.ancestor_symbols(indexed_class.symbol))
    return sorted_tuple(
        _compact_class_display_name(indexed_class, class_index)
        for symbol in sorted(common_symbols)
        if (indexed_class := class_index.class_for(symbol)) is not None
        and indexed_class.is_abstract
    )


def _compact_repeated_concrete_type_case_candidates(
    projections: tuple[CompactRemainingSystemicModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[RepeatedConcreteTypeCaseAnalysisCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(class_projections)
    min_function_count = max(3, config.min_registration_sites)
    min_class_count = max(2, config.min_reflective_selector_values)
    candidates: list[RepeatedConcreteTypeCaseAnalysisCandidate] = []
    for projection in projections:
        grouped: dict[str, list[ConcreteTypeCaseFunctionCandidate]] = defaultdict(list)
        for function_candidate in _compact_concrete_type_function_candidates(
            projection, class_index
        ):
            grouped[function_candidate.subject_role].append(function_candidate)
        for subject_role, functions in sorted(grouped.items()):
            if len(functions) < min_function_count:
                continue
            concrete_class_names = sorted_tuple(
                class_name
                for function in functions
                for class_name in function.concrete_class_names
            )
            concrete_class_names = sorted_tuple(set(concrete_class_names))
            if len(concrete_class_names) < min_class_count:
                continue
            abstract_base_names = _compact_common_abstract_base_names(
                projection, class_index, concrete_class_names
            )
            union_alias_names = sorted_tuple(
                {
                    alias_name
                    for function in functions
                    for alias_name in function.union_alias_names
                }
            )
            shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
                concrete_class_names
            )
            shared_prefix = CLASS_NAME_ALGEBRA.longest_common_prefix(
                concrete_class_names
            )
            if (
                not abstract_base_names
                and not union_alias_names
                and max(len(shared_suffix), len(shared_prefix)) < 6
            ):
                continue
            candidates.append(
                RepeatedConcreteTypeCaseAnalysisCandidate(
                    file_path=projection.file_path,
                    functions=sorted_tuple(
                        functions, key=lambda item: (item.line, item.function_name)
                    ),
                    abstract_base_names=abstract_base_names,
                )
            )
    return tuple(candidates)


def _compact_implicit_self_contract_mixin_candidates(
    projections: tuple[CompactRemainingSystemicModuleProjection, ...],
    class_projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[ImplicitSelfContractMixinCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(class_projections)
    min_consumer_count = max(2, config.min_registration_sites)
    facts_by_class_symbol = {
        class_symbol: fact
        for projection in projections
        for fact in projection.implicit_self_mixins
        if (
            class_symbol := class_index.symbol_for(
                file_path=fact.file_path, qualname=fact.qualname
            )
        )
        is not None
    }
    candidates: list[ImplicitSelfContractMixinCandidate] = []
    for class_symbol, fact in sorted(facts_by_class_symbol.items()):
        indexed_class = class_index.class_for(class_symbol)
        if indexed_class is None:
            continue
        consumers = tuple(
            descendant
            for descendant_symbol in class_index.descendant_symbols(class_symbol)
            if (descendant := class_index.class_for(descendant_symbol)) is not None
            and not descendant.is_abstract
        )
        if len(consumers) < min_consumer_count:
            continue
        candidates.append(
            ImplicitSelfContractMixinCandidate(
                file_path=fact.file_path,
                line=fact.line,
                mixin_name=_compact_class_display_name(indexed_class, class_index),
                method_names=fact.method_names,
                method_lines=fact.method_lines,
                cast_type_names=fact.cast_type_names,
                consumer_class_names=sorted_tuple(
                    _compact_class_display_name(consumer, class_index)
                    for consumer in consumers
                ),
                consumer_lines=tuple(consumer.line for consumer in consumers),
                accessed_attribute_names=fact.accessed_attribute_names,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class _DataclassNamespaceProjection:
    file_path: str
    line: int
    class_name: str
    field_names: tuple[str, ...]
    namespace_field_names: tuple[str, ...]
    from_namespace_line: int


@dataclass(frozen=True)
class _CliArgumentSpecProjection:
    file_path: str
    name: str
    line: int
    field_names: tuple[str, ...]


@dataclass(frozen=True)
class _DataclassNamespaceCliModuleProjection:
    dataclasses: tuple[_DataclassNamespaceProjection, ...]
    cli_specs: tuple[_CliArgumentSpecProjection, ...]


class _DataclassNamespaceCliModuleProjectionFamily(
    CollectedFamily[_DataclassNamespaceCliModuleProjection]
):
    item_type = _DataclassNamespaceCliModuleProjection
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            item.dataclasses or item.cli_specs
            for item in items
            if isinstance(item, _DataclassNamespaceCliModuleProjection)
        )
    )
    source_collector = staticmethod(
        lambda source_module, syntax_index: _native_dataclass_namespace_cli_projections(
            source_module,
            syntax_index,
        )
    )

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[_DataclassNamespaceCliModuleProjection]:
        del cls
        file_path = parsed_module.file_path
        dataclasses: list[_DataclassNamespaceProjection] = []
        for node in parsed_module.module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            field_names = _dataclass_config_field_names(node)
            if not field_names:
                continue
            namespace_assignment = _from_namespace_keyword_names(node)
            if namespace_assignment is None:
                continue
            from_namespace_line, namespace_field_names = namespace_assignment
            dataclasses.append(
                _DataclassNamespaceProjection(
                    file_path=file_path,
                    line=node.lineno,
                    class_name=node.name,
                    field_names=field_names,
                    namespace_field_names=namespace_field_names,
                    from_namespace_line=from_namespace_line,
                )
            )
        cli_specs = tuple(
            _CliArgumentSpecProjection(
                file_path=file_path,
                name=name,
                line=line,
                field_names=field_names,
            )
            for name, line, field_names in _cli_argument_spec_fields(parsed_module)
        )
        if not dataclasses and not cli_specs:
            return []
        return [
            _DataclassNamespaceCliModuleProjection(
                dataclasses=tuple(dataclasses),
                cli_specs=cli_specs,
            )
        ]


def _native_dataclass_namespace_cli_projections(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[_DataclassNamespaceCliModuleProjection] | None:
    """Project sparse dataclass/CLI mirrors from selected declarations."""

    if not syntax_index.is_complete:
        return None
    file_path = source_module.file_path
    try:
        dataclasses: list[_DataclassNamespaceProjection] = []
        for class_node in syntax_index.top_level_declarations("class"):
            class_source = syntax_index.source_for(class_node)
            decorated_source = (
                syntax_index.source_for(class_node.parent)
                if class_node.parent is not None
                and class_node.parent.type == "decorated_definition"
                else class_source
            )
            if b"dataclass" not in decorated_source or b"from_namespace" not in (
                class_source
            ):
                continue
            parsed_class = syntax_index.class_for(class_node)
            field_names = _dataclass_config_field_names(parsed_class)
            if not field_names:
                continue
            namespace_assignment = _from_namespace_keyword_names(parsed_class)
            if namespace_assignment is None:
                continue
            from_namespace_line, namespace_field_names = namespace_assignment
            dataclasses.append(
                _DataclassNamespaceProjection(
                    file_path=file_path,
                    line=parsed_class.lineno,
                    class_name=parsed_class.name,
                    field_names=field_names,
                    namespace_field_names=namespace_field_names,
                    from_namespace_line=from_namespace_line,
                )
            )
        cli_statements = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
            if b"ArgumentSpec" in syntax_index.source_for(node)
        )
        cli_module = source_module.parsed_module(
            ast.Module(body=list(cli_statements), type_ignores=[]),
        )
        cli_specs = tuple(
            _CliArgumentSpecProjection(
                file_path=file_path,
                name=name,
                line=line,
                field_names=field_names,
            )
            for name, line, field_names in _cli_argument_spec_fields(cli_module)
        )
        if not dataclasses and not cli_specs:
            return []
        return [
            _DataclassNamespaceCliModuleProjection(
                dataclasses=tuple(dataclasses),
                cli_specs=cli_specs,
            )
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _dataclass_namespace_cli_mirror_candidates_from_projections(
    projections: tuple[_DataclassNamespaceCliModuleProjection, ...],
) -> tuple[DataclassNamespaceCliMirrorCandidate, ...]:
    cli_specs = tuple(
        cli_spec for projection in projections for cli_spec in projection.cli_specs
    )
    candidates: list[DataclassNamespaceCliMirrorCandidate] = []
    for projection in projections:
        for dataclass_projection in projection.dataclasses:
            mirrored_fields = tuple(
                name
                for name in dataclass_projection.namespace_field_names
                if name in dataclass_projection.field_names
            )
            if len(mirrored_fields) < 4:
                continue
            for cli_spec in cli_specs:
                mirrored_cli_fields = tuple(
                    name
                    for name in cli_spec.field_names
                    if name in dataclass_projection.field_names
                )
                shared_fields = tuple(
                    name for name in mirrored_fields if name in mirrored_cli_fields
                )
                if len(shared_fields) < 4:
                    continue
                candidates.append(
                    DataclassNamespaceCliMirrorCandidate(
                        file_path=dataclass_projection.file_path,
                        line=dataclass_projection.line,
                        class_name=dataclass_projection.class_name,
                        argument_spec_name=cli_spec.name,
                        field_names=mirrored_fields,
                        cli_field_names=mirrored_cli_fields,
                        from_namespace_line=dataclass_projection.from_namespace_line,
                        argument_spec_file_path=cli_spec.file_path,
                        argument_spec_line=cli_spec.line,
                    )
                )
    return tuple(candidates)


_SINGLE_TEMPLATE_CALL_METRICS = OrchestrationMetrics(
    function_line_count=0,
    branch_site_count=0,
    call_site_count=1,
    parameter_count=1,
    callee_family_count=1,
)






@dataclass(frozen=True)
class ProjectionPropertyFamilyCandidate(ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    base_names: tuple[str, ...]

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(
                    self.file_path, line, f"{self.class_name}.{property_name}"
                )
                for line, property_name in zip(
                    self.line_numbers, self.property_names, strict=True
                )
            )
        )


@dataclass(frozen=True)
class SelfAttributeAuthority:
    def attr_name(self, node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id == "self")
        ):
            return node.attr
        return None


SELF_ATTRIBUTE_AUTHORITY = SelfAttributeAuthority()


def _is_path_projection_part(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(
            (
                isinstance(value, ast.Constant)
                or (
                    isinstance(value, ast.FormattedValue)
                    and SELF_ATTRIBUTE_AUTHORITY.attr_name(value.value) is not None
                )
                for value in node.values
            )
        )
    return False


def _path_projection_base(returned: ast.AST) -> str | None:
    node = returned
    saw_path_part = False
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not _is_path_projection_part(node.right):
            return None
        saw_path_part = True
        node = node.left
    if not saw_path_part:
        return None
    return SELF_ATTRIBUTE_AUTHORITY.attr_name(node)


def _projection_property_family_candidates(
    module: ParsedModule,
) -> tuple[ProjectionPropertyFamilyCandidate, ...]:
    candidates: list[ProjectionPropertyFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                (
                    AstExpressionProjection.terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            body = statements_without_docstring(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            base_name = _path_projection_base(body[0].value)
            if base_name is None:
                continue
            properties.append((statement, base_name))
        if len(properties) < 3:
            continue
        ordered = sorted_tuple(properties, key=lambda item: item[0].lineno)
        candidates.append(
            ProjectionPropertyFamilyCandidate(
                file_path=module.file_path,
                line=class_node.lineno,
                class_name=class_node.name,
                property_names=tuple((function.name for function, _ in ordered)),
                line_numbers=tuple((function.lineno for function, _ in ordered)),
                base_names=sorted_tuple({base_name for _, base_name in ordered}),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


declare_candidate_rule_detector(
    ProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Path projection properties should be derived descriptors",
        "Several properties project Path-valued views from owned base fields through the same `/` algebra. That is a descriptor-derived view family: the varying suffixes should be data while the projection algorithm lives in one reusable descriptor.",
        "single descriptor authority for repeated Path projection properties",
        "same class repeats Path projection properties over owned base fields",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda projection_candidate: (
        f"`{projection_candidate.class_name}` repeats Path projection properties {', '.join(projection_candidate.property_names)} over bases {', '.join(projection_candidate.base_names)}."
    ),
    evidence=lambda projection_candidate: projection_candidate.evidence_locations,
    metrics=lambda projection_candidate: MappingMetrics(
        mapping_site_count=len(projection_candidate.property_names),
        field_count=len(projection_candidate.base_names),
        mapping_name=f"{projection_candidate.class_name} path projection",
        field_names=projection_candidate.property_names,
        source_name=", ".join(projection_candidate.base_names),
    ),
    candidate_collector=_projection_property_family_candidates,
)


def _collection_projection_property_shape(
    returned: ast.AST,
) -> tuple[str, str] | None:
    return (
        Maybe.of(as_ast(returned, ast.Call))
        .filter(
            lambda call: (
                name_id(call.func) in {"tuple", "list", "set", "frozenset"}
                and len(call.args) == 1
                and not call.keywords
            )
        )
        .project(lambda call: as_ast(call.args[0], ast.GeneratorExp))
        .filter(lambda generator: len(generator.generators) == 1)
        .map(lambda generator: (generator, generator.generators[0]))
        .filter(lambda context: not context[1].ifs)
        .combine(
            lambda context: SELF_ATTRIBUTE_AUTHORITY.attr_name(context[1].iter),
            lambda context, collection_name: (
                collection_name,
                as_ast(context[0].elt, ast.Attribute),
                name_id(context[1].target),
            ),
        )
        .project(
            lambda context: (
                (
                    context[0],
                    context[1].attr,
                )
                if context[1] is not None
                and context[2] is not None
                and name_id(context[1].value) == context[2]
                else None
            )
        )
        .unwrap_or_none()
    )


def _is_property_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        (
            AstExpressionProjection.terminal_name(decorator) == "property"
            for decorator in method.decorator_list
        )
    )


def _collection_projection_property_family_candidates(
    module: ParsedModule,
) -> tuple[CollectionProjectionPropertyFamilyCandidate, ...]:
    candidates: list[CollectionProjectionPropertyFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]] = []
        for statement in CLASS_NODE_AUTHORITY.methods(class_node):
            if not _is_property_method(statement):
                continue
            body = statements_without_docstring(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            shape = _collection_projection_property_shape(body[0].value)
            if shape is None:
                continue
            collection_name, projected_attribute_name = shape
            properties.append((statement, collection_name, projected_attribute_name))
        grouped: dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]] = (
            defaultdict(list)
        )
        for statement, collection_name, projected_attribute_name in properties:
            grouped[collection_name].append((statement, projected_attribute_name))
        for collection_name, grouped_properties in grouped.items():
            if len(grouped_properties) < 2:
                continue
            ordered = sorted_tuple(grouped_properties, key=lambda item: item[0].lineno)
            candidates.append(
                CollectionProjectionPropertyFamilyCandidate(
                    file_path=module.file_path,
                    line=class_node.lineno,
                    class_name=class_node.name,
                    property_names=tuple((statement.name for statement, _ in ordered)),
                    line_numbers=tuple((statement.lineno for statement, _ in ordered)),
                    collection_name=collection_name,
                    projected_attribute_names=tuple(
                        (attribute_name for _, attribute_name in ordered)
                    ),
                    line_count=(class_node.end_lineno or class_node.lineno)
                    - class_node.lineno
                    + 1,
                )
            )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


declare_candidate_rule_detector(
    CollectionProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Collection projection properties should be derived descriptors",
        "Sibling properties that only map one owned collection to member attributes are descriptor-derived views. Repeating `tuple(item.attr for item in self.collection)` per property makes each projected attribute look like behavior when the actual semantic object is the collection projection relation.",
        "single collection-projection descriptor parameterized by collection and member attribute",
        "same class repeats collection projection properties over one owned collection",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats collection projection properties "
        f"{candidate.property_names} over `self.{candidate.collection_name}` "
        f"for member attributes {candidate.projected_attribute_names}."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=f"{candidate.class_name}.{candidate.collection_name}",
        field_names=candidate.projected_attribute_names,
    ),
    candidate_collector=_collection_projection_property_family_candidates,
)


class ResidualClosedAxisIndirectionDetector(
    ModuleCollectorCandidateDetector[ResidualClosedAxisIndirectionCandidate]
):
    candidate_collector = staticmethod(_residual_closed_axis_indirection_candidates)
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Enum-keyed table with residual branching should become a nominal strategy family",
        "A function that indexes an enum-keyed table and still branches on the same enum axis is not using the table as an authority. The table is a degenerate projection over behavior that still lives in branches. The stronger normal form is an ABC-backed strategy family keyed by the enum, with `AutoRegisterMeta` owning import-time registration and any table-like views derived from the family.",
        "metaclass-registry-backed nominal strategy family instead of enum table plus residual branching",
        "same function indexes an enum-keyed table and branches on that enum axis",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.PROJECTION_DICT,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
    )

    def _finding_for_candidate(
        self, axis_candidate: ResidualClosedAxisIndirectionCandidate
    ) -> RefactorFinding:
        residual_cases = ", ".join(axis_candidate.residual_case_names)
        table_cases = ", ".join(axis_candidate.table_case_names)
        value_summary = ", ".join(axis_candidate.table_value_summaries[:4])
        return self.build_finding(
            (
                f"`{axis_candidate.qualname}` indexes `{axis_candidate.table_name}` by "
                f"`{axis_candidate.dispatch_axis_expression}` for `{axis_candidate.enum_name}` cases {table_cases}, "
                f"but still branches on residual cases {residual_cases}."
            ),
            axis_candidate.evidence,
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=axis_candidate.enum_name,
                literal_cases=axis_candidate.table_case_names,
            ),
        )


class DerivedWrapperSpecShadowDetector(
    ModuleCollectorCandidateDetector[DerivedWrapperSpecShadowCandidate]
):
    candidate_collector = staticmethod(_derived_wrapper_spec_shadow_candidates)
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Generated wrapper spec family should collapse into the authoritative spec family",
        "The docs treat writable wrapper-spec tables as secondary authorities when they just point back at an existing spec family and feed code generation. Wrapper metadata should live on the authoritative spec records so generated wrappers are derived from one source rather than synchronized across parallel tables.",
        "single authoritative spec family carrying wrapper-generation metadata",
        "secondary spec table references an authoritative spec family entry-by-entry and is only consumed by wrapper generation",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.SCOPED_SHAPE_WRAPPER,
        ),
    )

    def _finding_for_candidate(
        self, shadow_candidate: DerivedWrapperSpecShadowCandidate
    ) -> RefactorFinding:
        primary_family_label = (
            shadow_candidate.primary_family_name
            or shadow_candidate.primary_constructor_name
        )
        constant_preview = ", ".join(shadow_candidate.primary_constant_names[:4])
        builder_preview = ", ".join(shadow_candidate.builder_names[:3])
        return self.build_finding(
            (
                f"`{shadow_candidate.derived_family_name}` re-encodes wrapper metadata over authoritative family "
                f"`{primary_family_label}` through link field `{shadow_candidate.link_field_name}` for {constant_preview}, "
                f"then feeds generated wrappers via {builder_preview}."
            ),
            shadow_candidate.evidence,
            metrics=MappingMetrics(
                mapping_site_count=len(shadow_candidate.primary_constant_names),
                field_count=max(len(shadow_candidate.extra_field_names), 1),
                mapping_name=shadow_candidate.derived_family_name,
                field_names=shadow_candidate.extra_field_names,
                source_name=primary_family_label,
                identity_field_names=(shadow_candidate.link_field_name,),
            ),
        )


def _compact_keyed_family_axis_specs_from_context(
    context: object | None,
) -> tuple[_KeyedFamilyAxisSpec, ...]:
    repository = CompactClassRepositoryContext.require(context)
    return repository.cached(
        _compact_keyed_family_axis_specs_from_index,
        lambda: _compact_keyed_family_axis_specs_from_index(repository.class_index),
    )


def _target_has_keyed_family_axis_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """A keyed-family report names one of the roots that owns the axis."""

    del config
    return any(
        indexed_class.keyed_family_key_type_name is not None
        and "registry_key_attr" in indexed_class.assignments_by_name
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
        for indexed_class in projection.classes
    )


def _target_has_manual_selector_axis(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    del config
    return any(
        projection.manual_selector_axes
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


def _target_has_closed_axis_branch(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    del config
    return any(
        projection.closed_axis_branch_functions
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


def _target_has_keyed_table_axis(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    del config
    return any(
        projection.keyed_table_axes
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


def _target_has_axis_shadow_evidence(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    return _target_has_keyed_family_axis_root(
        projections_by_family, config
    ) or _target_has_manual_selector_axis(projections_by_family, config)


def _target_has_residual_axis_evidence(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    return _target_has_keyed_family_axis_root(
        projections_by_family, config
    ) or _target_has_closed_axis_branch(projections_by_family, config)


class CompactCrossModuleAxisShadowFamilyCandidateBase(
    CompactClassRepositoryCandidateDetector[CrossModuleAxisShadowFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_axis_shadow_evidence
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[CrossModuleAxisShadowFamilyCandidate, ...]:
        del config
        return _cross_module_axis_shadow_family_candidates_from_specs(
            _compact_keyed_family_axis_specs_from_context(context),
            _compact_manual_selector_axis_specs(context.projections),
        )


declare_candidate_rule_detector(
    CrossModuleAxisShadowFamilyCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Cross-module shadow family should collapse into one axis authority",
        "The docs require one authoritative owner per closed semantic axis. When one module already owns an enum/keyed family nominally and another module reintroduces a second family over the same cases, the axis has split authority and local behavior should derive from the authoritative family instead.",
        "single authoritative closed-axis family reused across modules",
        "same keyed enum axis is modeled by an authoritative family in one module and a shadow selector family in another",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    ),
    summary=lambda shadow_candidate: (
        f"Axis `{shadow_candidate.key_type_name}` is already owned by `{shadow_candidate.authoritative.family_name}` but re-encoded by `{shadow_candidate.shadow.family_name}.{shadow_candidate.selector_method_name}` across cases {', '.join(shadow_candidate.shared_case_names[:4])}."
    ),
    evidence=lambda shadow_candidate: shadow_candidate.evidence,
    metrics=lambda shadow_candidate: DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
        shadow_candidate.shared_case_names, shadow_candidate.key_type_name
    ),
    detector_base=CompactCrossModuleAxisShadowFamilyCandidateBase,
)


class ResidualClosedAxisBranchingDetector(
    CompactClassRepositoryCandidateDetector[ResidualClosedAxisBranchingCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_residual_axis_evidence
    )
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Manual closed-axis branching should derive from existing keyed authority",
        "The docs require one authoritative owner per closed enum/key axis. When a keyed nominal family already owns that axis, downstream `if`/`match` ladders over the same cases become residual shadow dispatch.",
        "behavior derived from authoritative keyed family rather than downstream enum branching",
        "function branches on an enum axis already owned by a keyed nominal family in another module",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ResidualClosedAxisBranchingCandidate, ...]:
        del config
        return _residual_closed_axis_branching_candidates_from_compact_specs(
            context.projections,
            _compact_keyed_family_axis_specs_from_context(context),
        )

    def _finding_for_candidate(
        self, residual_candidate: ResidualClosedAxisBranchingCandidate
    ) -> RefactorFinding:
        authoritative_family_names = ", ".join(
            (
                family_name
                for family_name, _, _ in residual_candidate.authoritative_families[:4]
            )
        )
        return self.build_finding(
            (
                f"`{residual_candidate.qualname}` branches {residual_candidate.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            metrics=DispatchCountMetrics(
                dispatch_site_count=residual_candidate.branch_site_count,
                dispatch_axis=residual_candidate.key_type_name,
                literal_cases=residual_candidate.case_names,
            ),
        )


@dataclass(frozen=True)
class ExternalEnumCaseRecoverySite:
    """One external function recovering cases from an enum declaration."""

    function: CompactClosedAxisBranchFunction
    branch_site_count: int
    case_names: tuple[str, ...]

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(
            self.function.file_path,
            self.function.line,
            self.function.qualname,
        )


@dataclass(frozen=True)
class ExternalEnumCaseRecoveryCandidate:
    """One enum whose cases are recovered outside its declaration."""

    enum_class: CompactIndexedClass
    sites: tuple[ExternalEnumCaseRecoverySite, ...]
    case_names: tuple[str, ...]

    @property
    def branch_site_count(self) -> int:
        return sum(site.branch_site_count for site in self.sites)

    @property
    def authority_evidence(self) -> SourceLocation:
        return SourceLocation(
            self.enum_class.file_path,
            self.enum_class.line,
            self.enum_class.qualname,
        )

    @property
    def projection_evidence(self) -> SourceLocation:
        if not self.sites:
            raise ValueError("external enum recovery requires one projection site")
        return self.sites[0].evidence

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            self.authority_evidence,
            *(site.evidence for site in self.sites[:5]),
        )


def _unique_direct_enum_classes(
    class_index: CompactClassFamilyIndex,
) -> dict[str, CompactIndexedClass]:
    return {
        simple_name: enum_class
        for simple_name, symbols in class_index.symbols_by_simple_name.items()
        if len(symbols) == 1
        if (
            enum_class := class_index.classes_by_symbol[symbols[0]]
        ).direct_enum_member_names
    }


def _external_enum_case_recovery_candidates(
    context: CompactClassRepositoryContext,
    config: DetectorConfig,
) -> tuple[ExternalEnumCaseRecoveryCandidate, ...]:
    enum_classes = _unique_direct_enum_classes(context.class_index)
    keyed_axis_names = {
        spec.key_type_name
        for spec in _compact_keyed_family_axis_specs_from_context(context)
    }
    sites_by_enum_name: dict[str, list[ExternalEnumCaseRecoverySite]] = defaultdict(
        list
    )
    for projection in context.projections:
        for function in projection.closed_axis_branch_functions:
            if PythonSourcePathPolicy.is_test_path(Path(function.file_path)):
                continue
            for axis in function.axes:
                if axis.key_type_name in keyed_axis_names:
                    continue
                enum_class = enum_classes.get(axis.key_type_name)
                if enum_class is None:
                    continue
                if (
                    function.file_path == enum_class.file_path
                    and function.qualname.startswith(f"{enum_class.qualname}.")
                ):
                    continue
                declared_case_names = {
                    f"{enum_class.simple_name}.{member_name}"
                    for member_name in enum_class.direct_enum_member_names
                }
                case_names = sorted_tuple(
                    set(axis.case_names).intersection(declared_case_names)
                )
                if not case_names:
                    continue
                sites_by_enum_name[enum_class.simple_name].append(
                    ExternalEnumCaseRecoverySite(
                        function=function,
                        branch_site_count=axis.branch_site_count,
                        case_names=case_names,
                    )
                )
    candidates: list[ExternalEnumCaseRecoveryCandidate] = []
    for enum_name, sites in sites_by_enum_name.items():
        case_names = sorted_tuple(
            {case_name for site in sites for case_name in site.case_names}
        )
        if len(case_names) < config.min_string_cases:
            continue
        candidates.append(
            ExternalEnumCaseRecoveryCandidate(
                enum_class=enum_classes[enum_name],
                sites=tuple(sites),
                case_names=case_names,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: candidate.enum_class.symbol,
    )


def _target_has_external_enum_case_recovery(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    del config
    projections = projections_by_family.get(CompactModuleClassProjectionFamily, ())
    return any(
        projection.closed_axis_branch_functions
        for projection in projections
        if isinstance(projection, CompactModuleClassProjection)
    ) and any(
        indexed_class.direct_enum_member_names
        for projection in projections
        if isinstance(projection, CompactModuleClassProjection)
        for indexed_class in projection.classes
    )


class ExternalEnumCaseRecoveryDetector(
    SemanticMirrorIssueDetector,
    CompactClassRepositoryCandidateDetector[ExternalEnumCaseRecoveryCandidate],
    SemanticMirrorFindingRecipeEvaluator,
):
    """Find closed enum case semantics recovered outside their declaration."""

    compact_report_context_promotion_predicate = staticmethod(
        _target_has_external_enum_case_recovery
    )
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "External enum case recovery should move to its nominal owner",
        "A function outside an enum declaration recovers multiple cases from that closed axis. The enum or a keyed strategy family should own those case semantics so consumers query behavior instead of rediscovering member identity.",
        "one nominal owner for closed-axis behavior with consumers deriving its result",
        "multiple declared enum members are inspected outside their nominal owner",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ExternalEnumCaseRecoveryCandidate, ...]:
        return _external_enum_case_recovery_candidates(context, config)

    def _finding_for_candidate(
        self,
        candidate: ExternalEnumCaseRecoveryCandidate,
    ) -> RefactorFinding:
        return self.build_finding(
            (
                f"Enum `{candidate.enum_class.simple_name}` has "
                f"{candidate.branch_site_count} external branch site(s) across "
                f"{len(candidate.sites)} function(s) for cases "
                f"{', '.join(candidate.case_names)}."
            ),
            candidate.evidence,
            projection_evidence=candidate.projection_evidence,
            authority_evidence=candidate.authority_evidence,
            metrics=DispatchCountMetrics(
                dispatch_site_count=candidate.branch_site_count,
                dispatch_axis=candidate.enum_class.simple_name,
                literal_cases=candidate.case_names,
            ),
        )


class ParallelKeyedAxisFamilyDetector(
    CompactClassRepositoryCandidateDetector[ParallelKeyedAxisFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_family_axis_root
    )
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Parallel keyed families should collapse into one axis authority",
        "The docs require one authoritative nominal owner per closed semantic axis. When two modules each define a keyed family over the same enum/key cases, the axis has split ownership even if both sides are nominal.",
        "single cross-module keyed-axis authority with module-local adapters derived from it",
        "same keyed enum axis is modeled by multiple nominal families across modules",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ParallelKeyedAxisFamilyCandidate, ...]:
        del config
        return _parallel_keyed_axis_family_candidates_from_specs(
            _compact_keyed_family_axis_specs_from_context(context)
        )

    def _finding_for_candidate(
        self, family_candidate: ParallelKeyedAxisFamilyCandidate
    ) -> RefactorFinding:
        shared_cases = ", ".join(family_candidate.shared_case_names[:4])
        label_clause = ""
        if (
            family_candidate.left.family_label is not None
            and family_candidate.left.family_label
            == family_candidate.right.family_label
        ):
            label_clause = (
                f" Both declare family label `{family_candidate.left.family_label}`."
            )
        return self.build_finding(
            (
                f"Axis `{family_candidate.key_type_name}` is owned in parallel by "
                f"`{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` "
                f"across cases {shared_cases}.{label_clause}"
            ),
            family_candidate.evidence,
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                family_candidate.shared_case_names,
                family_candidate.key_type_name,
            ),
        )


class ParallelKeyedTableAndFamilyDetector(
    CompactClassRepositoryCandidateDetector[ParallelKeyedTableAndFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_table_axis
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Keyed table and keyed family should collapse into one auto-registered axis family",
        "The docs require one authoritative owner per closed semantic axis. When a module keeps one keyed table of per-case records and a second keyed nominal family over the same cases, the axis is split across data and behavior. If the family already carries the runtime behavior boundary, the table should derive from that family instead of competing with it.",
        "single authoritative metaclass-registry axis family with derived table/view projections",
        "same enum/key axis is encoded by both a keyed table and a keyed nominal family",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ParallelKeyedTableAndFamilyCandidate, ...]:
        del config
        return _parallel_keyed_table_and_family_candidates_from_specs(
            _compact_keyed_family_axis_specs_from_context(context),
            _compact_keyed_table_axis_specs(context.projections),
        )

    def _finding_for_candidate(
        self, table_candidate: ParallelKeyedTableAndFamilyCandidate
    ) -> RefactorFinding:
        shape_clause = (
            ""
            if table_candidate.value_shape_name is None
            else f" of `{table_candidate.value_shape_name}` records"
        )
        return self.build_finding(
            (
                f"Axis `{table_candidate.key_type_name}` is split between keyed table `{table_candidate.table_name}`"
                f"{shape_clause} and keyed family `{table_candidate.family_name}` across cases "
                f"{', '.join(table_candidate.shared_case_names[:4])}."
            ),
            table_candidate.evidence,
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                table_candidate.shared_case_names,
                table_candidate.key_type_name,
            ),
        )


class InheritedAutoRegisterConfigBoilerplateDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
    InheritedAutoRegisterConfigBoilerplateFindingRecipeSynthesizer,
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )
    detector_id = "inherited_autoregister_config_boilerplate"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister root repeats inherited registry configuration",
        "An AutoRegisterMeta root that directly repeats registry protocol fields already declared by a base is carrying boilerplate instead of relying on inheritance. The registry key, skip policy, and related protocol fields should be inherited from the shared nominal base. If AutoRegisterMeta cannot honor inherited registry config, fix the metaclass package rather than repeating the fields on every root.",
        "inherited AutoRegister registry protocol configuration",
        "AutoRegisterMeta class repeats registry protocol assignments from an inherited base",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        class_index = build_compact_class_family_index(projections)
        return self._findings_from_class_index(class_index)

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del projections, config
        return self._findings_from_class_index(
            CompactClassRepositoryContext.require(context).class_index
        )

    def _findings_from_class_index(
        self,
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            if not indexed_class.declares_autoregister_meta:
                continue
            repeated_fields = class_index.assignments_repeated_from_ancestors(
                indexed_class.symbol,
                INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
            )
            if not repeated_fields:
                continue
            field_list = ", ".join(repeated_fields)
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` repeats inherited AutoRegister "
                        f"registry field(s) {field_list}."
                    ),
                    (
                        SourceLocation(
                            indexed_class.file_path,
                            indexed_class.line,
                            indexed_class.simple_name,
                        ),
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(repeated_fields),
                        mapping_name=indexed_class.simple_name,
                        field_names=repeated_fields,
                    ),
                )
            )
        return findings


_EXPLICIT_CLASS_ORDER_AXIS_NAMES = ("priority", "precedence", "rank", "order")


class AutoRegisterExplicitPriorityOrderingDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
    AutoRegisterExplicitPriorityOrderingFindingRecipeSynthesizer,
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )
    detector_id = "autoregister_explicit_priority_ordering"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister family uses explicit priority ordering instead of MRO",
        "An AutoRegisterMeta family whose registered leaves carry a `priority`, `precedence`, `rank`, or `order` class attribute is maintaining a second ordering authority beside the inheritance graph. If ordering is semantic, the nominal hierarchy and MRO should carry it; if ordering is only presentation, it should be a derived view outside the registered family.",
        "MRO-owned ordering for registered semantic families",
        "AutoRegisterMeta family declares or consumes a class-level priority-like axis to sort registered implementations",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        class_index = build_compact_class_family_index(projections)
        return self._findings_from_class_index(projections, class_index)

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self._findings_from_class_index(
            projections,
            CompactClassRepositoryContext.require(context).class_index,
        )

    def _findings_from_class_index(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            if not indexed_class.declares_autoregister_meta:
                continue
            order_axis_sites = self._order_axis_sites(class_index, indexed_class)
            order_axis_names = tuple(
                axis_name
                for axis_name, locations in order_axis_sites.items()
                if locations
            )
            if not order_axis_names:
                continue
            if not self._sorts_registry_by_order_axis(
                indexed_class.simple_name,
                projections,
                order_axis_names,
            ):
                continue
            evidence_sites = tuple(
                location
                for locations in order_axis_sites.values()
                for location in locations
            )
            axis_label = _class_order_axis_label(order_axis_names)
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` orders registered implementations "
                        f"through explicit class-level {axis_label} values."
                    ),
                    (
                        SourceLocation(
                            indexed_class.file_path,
                            indexed_class.line,
                            indexed_class.simple_name,
                        ),
                        *evidence_sites,
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(evidence_sites),
                        field_count=len(evidence_sites),
                        mapping_name=indexed_class.simple_name,
                        field_names=order_axis_names,
                    ),
                )
            )
        return findings

    def _order_axis_sites(
        self,
        class_index: CompactClassFamilyIndex,
        indexed_class: CompactIndexedClass,
    ) -> dict[str, tuple[SourceLocation, ...]]:
        symbols = (
            indexed_class.symbol,
            *class_index.descendant_symbols(indexed_class.symbol),
        )
        sites_by_axis: dict[str, list[SourceLocation]] = {
            axis_name: [] for axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES
        }
        for symbol in symbols:
            candidate = class_index.class_for(symbol)
            if candidate is None:
                continue
            assignment_lines = candidate.assignment_lines_by_name
            for axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES:
                line = assignment_lines.get(axis_name)
                if line is None:
                    continue
                sites_by_axis[axis_name].append(
                    SourceLocation(candidate.file_path, line, candidate.simple_name)
                )
        return {
            axis_name: tuple(locations)
            for axis_name, locations in sites_by_axis.items()
            if locations
        }

    @staticmethod
    def _sorts_registry_by_order_axis(
        root_name: str,
        projections: tuple[CompactModuleClassProjection, ...],
        axis_names: tuple[str, ...],
    ) -> bool:
        axis_name_set = frozenset(axis_names)
        return any(
            ({"cls", root_name} & frozenset(call.registry_owner_names))
            and (axis_name_set & frozenset(call.key_attribute_names))
            for projection in projections
            for call in projection.sorted_key_calls
        )


def _class_order_axis_label(axis_names: tuple[str, ...]) -> str:
    if len(axis_names) == 1:
        return f"`{axis_names[0]}`"
    return " / ".join(f"`{axis_name}`" for axis_name in axis_names)


class NominalInstanceExplicitOrderingDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Nominal declaration catalog uses explicit ordering instead of MRO",
        "An abstract nominal value family carries a `priority`, `precedence`, `rank`, or `order` instance field, while class-owned declarations supply that field and a consumer sorts by it. The field is a second ordering authority beside the inheritance graph. Give each declaration one nominal catalog node and derive the sequence from its MRO.",
        "MRO-owned sequence for nominal declaration catalogs",
        "class-owned instances of an abstract family supply an explicit ordering axis consumed by sorted(..., key=...)",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self._findings_from_class_index(
            projections,
            build_compact_class_family_index(projections),
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return self._findings_from_class_index(
            projections,
            CompactClassRepositoryContext.require(context).class_index,
        )

    def _findings_from_class_index(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        indexed_classes = tuple(class_index.classes_by_symbol.values())
        sorted_key_calls = tuple(
            call for projection in projections for call in projection.sorted_key_calls
        )
        for family_root in sorted(indexed_classes, key=lambda item: item.symbol):
            if not family_root.is_abstract or family_root.declares_autoregister_meta:
                continue
            root_axis_lines = {
                axis_name: line
                for axis_name, line in family_root.assignment_lines_by_name.items()
                if axis_name in _EXPLICIT_CLASS_ORDER_AXIS_NAMES
            }
            if not root_axis_lines:
                continue
            descendant_names = frozenset(
                descendant.simple_name
                for symbol in class_index.descendant_symbols(family_root.symbol)
                if (descendant := class_index.class_for(symbol)) is not None
            )
            if len(descendant_names) < 2:
                continue
            axis_evidence = {
                axis_name: self._axis_evidence(
                    axis_name,
                    descendant_names,
                    indexed_classes,
                    sorted_key_calls,
                )
                for axis_name in root_axis_lines
            }
            active_axes = tuple(
                axis_name
                for axis_name, evidence in axis_evidence.items()
                if evidence is not None
            )
            if not active_axes:
                continue
            evidence_sites = tuple(
                location
                for axis_name in active_axes
                for location in axis_evidence[axis_name] or ()
            )
            axis_label = _class_order_axis_label(active_axes)
            findings.append(
                self.build_finding(
                    (
                        f"`{family_root.simple_name}` declarations are sequenced "
                        f"through explicit {axis_label} instance values."
                    ),
                    (
                        SourceLocation(
                            family_root.file_path,
                            family_root.line,
                            family_root.simple_name,
                        ),
                        *(
                            SourceLocation(
                                family_root.file_path,
                                root_axis_lines[axis_name],
                                axis_name,
                            )
                            for axis_name in active_axes
                        ),
                        *evidence_sites[:6],
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(evidence_sites),
                        field_count=len(evidence_sites),
                        mapping_name=family_root.simple_name,
                        field_names=active_axes,
                    ),
                )
            )
        return findings

    @staticmethod
    def _axis_evidence(
        axis_name: str,
        descendant_names: frozenset[str],
        indexed_classes: tuple[CompactIndexedClass, ...],
        sorted_key_calls: tuple[CompactSortedKeyCall, ...],
    ) -> tuple[SourceLocation, ...] | None:
        constructions = tuple(
            (owner, construction)
            for owner in indexed_classes
            for construction in owner.direct_value_constructions
            if construction.constructor_name in descendant_names
            and axis_name in construction.keyword_names
        )
        if len(constructions) < 2:
            return None
        declaration_files = frozenset(owner.file_path for owner, _ in constructions)
        matching_calls = tuple(
            call
            for call in sorted_key_calls
            if axis_name in call.key_attribute_names
            and call.file_path in declaration_files
        )
        if not matching_calls:
            return None
        return (
            *(
                SourceLocation(
                    owner.file_path,
                    construction.line,
                    f"{owner.simple_name}.{construction.assigned_name}",
                )
                for owner, construction in constructions[:4]
            ),
            SourceLocation(
                matching_calls[0].file_path,
                matching_calls[0].line,
                f"sorted(..., key=...{axis_name})",
            ),
        )


def _target_has_repeated_keyed_family_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Repeated-keyed-family evidence consists exclusively of its roots."""

    del config
    return any(
        projection.repeated_keyed_family_roots
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


class RepeatedKeyedFamilyDetector(
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        RepeatedKeyedFamilyCandidate,
    ],
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_repeated_keyed_family_root
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Repeated keyed family scaffolding should collapse into one typed metaclass-registry base",
        "The docs encourage aggressive metaprogramming when several nominal families repeat the same class-level registration and lookup shell. When many roots restate `registry_key_attr`, `_registry`, and `for_*` lookup methods, the family algorithm should live in one typed `metaclass-registry` base.",
        "single typed metaclass-registry substrate for keyed nominal registries",
        "same keyed family registration and lookup shell repeated across nominal family roots",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    @staticmethod
    def _candidates_from_compact_projections(
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> tuple[RepeatedKeyedFamilyCandidate, ...]:
        return RepeatedKeyedFamilyCandidate.from_roots(
            tuple(
                root
                for projection in projections
                for root in projection.repeated_keyed_family_roots
            ),
            minimum_root_count=max(3, config.min_registration_sites),
        )

    def _finding_for_candidate(
        self, family_candidate: RepeatedKeyedFamilyCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (root.class_name for root in family_candidate.roots[:8])
        )
        lookup_names = ", ".join(
            sorted({root.lookup_method_name for root in family_candidate.roots[:8]})
        )
        registry_keys = ", ".join(
            sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]})
        )
        evidence = tuple(
            SourceLocation(root.file_path, root.line, root.class_name)
            for root in family_candidate.roots[:8]
        )
        return self.build_finding(
            (
                f"Registry roots {class_names} each repeat `{registry_keys}` + `_registry` + "
                f"`{lookup_names}` over `{family_candidate.family_base_name}`."
            ),
            evidence,
        )


def _compact_registry_class_display_name(
    indexed_class: CompactIndexedClass,
    class_index: CompactClassFamilyIndex,
) -> str:
    if len(class_index.symbols_by_simple_name.get(indexed_class.simple_name, ())) <= 1:
        return indexed_class.simple_name
    return indexed_class.symbol


def _compact_string_literal(expression: str | None) -> str | None:
    if expression is None:
        return None
    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _compact_registry_reference_edges(
    encoded_edges: str,
) -> Iterator[tuple[int, int, int]]:
    for encoded_edge in encoded_edges.split(";"):
        if not encoded_edge:
            continue
        function_index, receiver_index, attribute_index = encoded_edge.split(",")
        yield int(function_index), int(receiver_index), int(attribute_index)


def _compact_registry_consumer_index(
    projections: tuple[CompactModuleClassProjection, ...],
    relevant_keys: frozenset[tuple[str, str]],
) -> dict[tuple[str, str], frozenset[str]]:
    consumers: dict[tuple[str, str], set[str]] = {}
    for projection in projections:
        reference_index = projection.autoregister_reference_index
        if reference_index is None:
            continue
        for (
            function_index,
            receiver_index,
            attribute_index,
        ) in _compact_registry_reference_edges(reference_index.encoded_edges):
            key = (
                reference_index.receiver_names[receiver_index],
                reference_index.attribute_names[attribute_index],
            )
            if key not in relevant_keys:
                continue
            consumers.setdefault(key, set()).add(
                reference_index.function_qualnames[function_index]
            )
    return {key: frozenset(symbols) for key, symbols in consumers.items()}


def _compact_registry_consumer_symbols(
    consumer_index: dict[tuple[str, str], frozenset[str]],
    *,
    family_name: str,
    lookup_method_names: tuple[str, ...],
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            qualname
            for method_name in lookup_method_names
            for qualname in consumer_index.get((family_name, method_name), ())
            if not qualname.startswith(f"{family_name}.")
        }
    )


def _compact_keyed_registry_axis_facts(
    projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[KeyedRegistryAxisFact, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(projections)
    registry_classes = tuple(
        indexed_class
        for indexed_class in class_index.classes_by_symbol.values()
        if indexed_class.keyed_family_key_type_name is not None
        and _compact_string_literal(
            indexed_class.assignments_by_name.get("registry_key_attr")
        )
        is not None
    )
    relevant_consumer_keys = frozenset(
        (family_name, method_name)
        for indexed_class in registry_classes
        for family_name in (
            _compact_registry_class_display_name(indexed_class, class_index),
        )
        for method_name in indexed_class.keyed_registry_lookup_method_names
    )
    consumer_index = _compact_registry_consumer_index(
        projections, relevant_consumer_keys
    )
    min_case_count = max(2, config.min_registration_sites)
    min_consumer_count = max(2, config.min_registration_sites)
    facts: list[KeyedRegistryAxisFact] = []
    for indexed_class in sorted(registry_classes, key=lambda item: item.symbol):
        if PythonSourcePathPolicy.is_test_path(Path(indexed_class.file_path)):
            continue
        key_type_name = indexed_class.keyed_family_key_type_name
        registry_key_attr_name = _compact_string_literal(
            indexed_class.assignments_by_name.get("registry_key_attr")
        )
        if key_type_name is None or registry_key_attr_name is None:
            continue
        family_name = _compact_registry_class_display_name(indexed_class, class_index)
        lookup_method_names = indexed_class.keyed_registry_lookup_method_names
        consumer_symbols = _compact_registry_consumer_symbols(
            consumer_index,
            family_name=family_name,
            lookup_method_names=lookup_method_names,
        )
        descendants = tuple(
            descendant
            for symbol in class_index.descendant_symbols(indexed_class.symbol)
            if (descendant := class_index.class_for(symbol)) is not None
        )
        concrete_descendants = tuple(
            descendant for descendant in descendants if not descendant.is_abstract
        )
        registered_case_names = sorted_tuple(
            {
                expression
                for descendant in descendants
                if (
                    expression := descendant.assignments_by_name.get(
                        registry_key_attr_name
                    )
                )
                is not None
            }
        )
        type_names_by_key: dict[str, list[str]] = {}
        for descendant in concrete_descendants:
            expression = descendant.assignments_by_name.get(registry_key_attr_name)
            if expression is None:
                continue
            type_names_by_key.setdefault(expression, []).append(
                _compact_registry_class_display_name(descendant, class_index)
            )
        injectivity_proof = InjectiveTypeRegistryProof.from_type_map(
            key_axis_name=key_type_name,
            type_names_by_key={
                key_name: sorted_tuple(type_names)
                for key_name, type_names in sorted(type_names_by_key.items())
            },
            registered_type_names=tuple(
                _compact_registry_class_display_name(descendant, class_index)
                for descendant in concrete_descendants
            ),
            reverse_lookup_names=indexed_class.keyed_registry_reverse_lookup_method_names,
            consumer_symbols=consumer_symbols,
        )
        facts.append(
            KeyedRegistryAxisFact(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                class_name=family_name,
                key_type_name=key_type_name,
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_names=lookup_method_names,
                registered_case_names=registered_case_names,
                consumer_symbols=consumer_symbols,
                missing_maturity_signals=_registry_maturity_missing_signals(
                    registered_case_count=len(registered_case_names),
                    lookup_method_names=lookup_method_names,
                    consumer_count=len(consumer_symbols),
                    min_case_count=min_case_count,
                    min_consumer_count=min_consumer_count,
                ),
                injectivity_proof=injectivity_proof,
            )
        )
    return tuple(facts)


def _compact_keyed_registry_axis_facts_from_context(
    context: object | None,
) -> tuple[KeyedRegistryAxisFact, ...]:
    repository = CompactClassRepositoryContext.require(context)
    return repository.cached(
        _compact_keyed_registry_axis_facts,
        lambda: _compact_keyed_registry_axis_facts(
            repository.projections,
            repository.config,
            class_index=repository.class_index,
        ),
    )


def _compact_registry_projection_import_aliases(
    projection: CompactModuleClassProjection,
    *,
    registry_projection: CompactModuleClassProjection,
    fact: KeyedRegistryAxisFact,
) -> dict[str, str]:
    canonical_names = frozenset(
        (
            fact.class_name,
            fact.key_type_name,
            *fact.injectivity_proof.registered_type_names,
        )
    )
    return {
        local_name: canonical_name
        for local_name, qualified_name in projection.import_aliases
        for canonical_name in canonical_names
        if qualified_name == f"{registry_projection.module_name}.{canonical_name}"
    }


def _compact_registry_projection_reference_name(
    reference: tuple[str, str | None],
    import_aliases: dict[str, str],
) -> str:
    value, alias_head = reference
    if alias_head is None or alias_head not in import_aliases:
        return value
    canonical_head = import_aliases[alias_head]
    if value == alias_head:
        return canonical_head
    return f"{canonical_head}{value[len(alias_head) :]}"


def _compact_registry_projection_surface_candidate(
    surface: CompactNamedProjectionSurface,
    *,
    fact: KeyedRegistryAxisFact,
    import_aliases: dict[str, str],
) -> RegistryProjectionSurfaceCandidate | None:
    proof = fact.injectivity_proof
    if surface.sequence_references:
        reference_names = tuple(
            _compact_registry_projection_reference_name(reference, import_aliases)
            for reference in surface.sequence_references
        )
        shared_key_names = sorted_tuple(
            frozenset(reference_names) & frozenset(proof.key_names)
        )
        shared_type_names = sorted_tuple(
            frozenset(reference_names) & frozenset(proof.registered_type_names)
        )
        evidence = RegistryProjectionSurfaceEvidence(
            surface_name=surface.surface_name,
            shared_key_names=shared_key_names,
            shared_type_names=shared_type_names,
            has_key_to_type_pairs=False,
            has_type_to_key_pairs=False,
        )
        surface_kind = RegistryProjectionSurfaceKind.for_evidence(evidence)
        if surface_kind is None or len(shared_key_names) + len(shared_type_names) < 2:
            return None
        return _REGISTRY_PROJECTION_SURFACE_ANALYZER.candidate(
            file_path=surface.file_path,
            fact=fact,
            evidence=evidence,
            line=surface.line,
            surface_kind=surface_kind,
            projected_names=reference_names,
        )

    key_names = tuple(
        _compact_registry_projection_reference_name(reference, import_aliases)
        for reference in surface.dict_key_references
    )
    value_names = tuple(
        _compact_registry_projection_reference_name(reference, import_aliases)
        for reference in surface.dict_value_references
    )
    proof_key_names = frozenset(proof.key_names)
    proof_type_names = frozenset(proof.registered_type_names)
    shared_key_names = sorted_tuple(
        (frozenset(key_names) | frozenset(value_names)) & proof_key_names
    )
    shared_type_names = sorted_tuple(
        (frozenset(key_names) | frozenset(value_names)) & proof_type_names
    )
    has_key_to_type_pairs = bool(
        len(frozenset(key_names) & proof_key_names) >= 2
        and len(frozenset(value_names) & proof_type_names) >= 2
    )
    has_type_to_key_pairs = bool(
        len(frozenset(key_names) & proof_type_names) >= 2
        and len(frozenset(value_names) & proof_key_names) >= 2
    )
    evidence = RegistryProjectionSurfaceEvidence(
        surface_name=surface.surface_name,
        shared_key_names=shared_key_names,
        shared_type_names=shared_type_names,
        has_key_to_type_pairs=has_key_to_type_pairs,
        has_type_to_key_pairs=has_type_to_key_pairs,
    )
    surface_kind = RegistryProjectionSurfaceKind.for_evidence(evidence)
    if surface_kind is None or len(shared_key_names) + len(shared_type_names) < 3:
        return None
    return _REGISTRY_PROJECTION_SURFACE_ANALYZER.candidate(
        file_path=surface.file_path,
        fact=fact,
        evidence=evidence,
        line=surface.line,
        surface_kind=surface_kind,
        projected_names=(*key_names, *value_names),
    )


def _compact_registry_projection_surface_candidates_from_facts(
    projections: tuple[CompactModuleClassProjection, ...],
    facts: tuple[KeyedRegistryAxisFact, ...],
) -> tuple[RegistryProjectionSurfaceCandidate, ...]:
    projections_by_path = {
        projection.file_path: projection for projection in projections
    }
    candidates: list[RegistryProjectionSurfaceCandidate] = []
    for fact in facts:
        if not fact.is_mature_injective:
            continue
        registry_projection = projections_by_path.get(fact.file_path)
        if registry_projection is None:
            continue
        for projection in projections:
            if projection.file_path == fact.file_path:
                import_aliases: dict[str, str] = {}
            else:
                import_aliases = _compact_registry_projection_import_aliases(
                    projection,
                    registry_projection=registry_projection,
                    fact=fact,
                )
                if not (
                    fact.key_type_name in import_aliases.values()
                    or fact.class_name in import_aliases.values()
                    or frozenset(import_aliases.values())
                    & frozenset(fact.injectivity_proof.registered_type_names)
                ):
                    continue
            for surface in projection.named_projection_surfaces:
                candidate = _compact_registry_projection_surface_candidate(
                    surface,
                    fact=fact,
                    import_aliases=import_aliases,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.line,
            item.registry_class_name,
            item.surface_name,
        ),
    )


def _compact_registry_projection_surface_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
) -> tuple[RegistryProjectionSurfaceCandidate, ...]:
    return _compact_registry_projection_surface_candidates_from_facts(
        projections,
        _compact_keyed_registry_axis_facts(projections, config),
    )


def _compact_registry_projection_policy_authority_candidates_from_facts(
    projections: tuple[CompactModuleClassProjection, ...],
    facts: tuple[KeyedRegistryAxisFact, ...],
) -> tuple[RegistryProjectionPolicyAuthorityCandidate, ...]:
    return (
        _REGISTRY_PROJECTION_SURFACE_ANALYZER.policy_authority_candidates_from_surfaces(
            _compact_registry_projection_surface_candidates_from_facts(
                projections, facts
            )
        )
    )


def _compact_registry_projection_policy_authority_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
) -> tuple[RegistryProjectionPolicyAuthorityCandidate, ...]:
    return _compact_registry_projection_policy_authority_candidates_from_facts(
        projections,
        _compact_keyed_registry_axis_facts(projections, config),
    )


def _target_has_keyed_registry_axis_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Registry-axis findings are anchored at the class that owns the axis."""

    del config
    return any(
        indexed_class.keyed_family_key_type_name is not None
        and _compact_string_literal(
            indexed_class.assignments_by_name.get("registry_key_attr")
        )
        is not None
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
        for indexed_class in projection.classes
    )


def _target_has_named_registry_projection_surface(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Projection findings report their manual surface, not the joined registry."""

    del config
    return any(
        projection.named_projection_surfaces
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


CompactKeyedRegistryCandidateT = TypeVar("CompactKeyedRegistryCandidateT")


class _CompactKeyedRegistryCandidateDetectorBase(
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        CompactKeyedRegistryCandidateT,
    ],
    Generic[CompactKeyedRegistryCandidateT],
    ABC,
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )


class _CompactNonInjectiveTypeRegistryDetectorBase(
    _CompactKeyedRegistryCandidateDetectorBase[NonInjectiveTypeRegistryCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_registry_axis_root
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[NonInjectiveTypeRegistryCandidate]:
        return NonInjectiveTypeRegistryCandidate.from_facts(
            _compact_keyed_registry_axis_facts(projections, config)
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del projections
        facts = _compact_keyed_registry_axis_facts_from_context(context)
        return self._findings_for_candidates(
            NonInjectiveTypeRegistryCandidate.from_facts(facts), config
        )


class _CompactInjectiveTypeRegistryDetectorBase(
    _CompactKeyedRegistryCandidateDetectorBase[InjectiveTypeRegistryCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_registry_axis_root
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[InjectiveTypeRegistryCandidate]:
        return InjectiveTypeRegistryCandidate.from_facts(
            _compact_keyed_registry_axis_facts(projections, config)
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del projections
        facts = _compact_keyed_registry_axis_facts_from_context(context)
        return self._findings_for_candidates(
            InjectiveTypeRegistryCandidate.from_facts(facts), config
        )


class _CompactRegistryProjectionSurfaceDetectorBase(
    _CompactKeyedRegistryCandidateDetectorBase[RegistryProjectionSurfaceCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_named_registry_projection_surface
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[RegistryProjectionSurfaceCandidate]:
        return _compact_registry_projection_surface_candidates(projections, config)

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        facts = _compact_keyed_registry_axis_facts_from_context(context)
        return self._findings_for_candidates(
            _compact_registry_projection_surface_candidates_from_facts(
                projections, facts
            ),
            config,
        )


class _CompactRegistryProjectionPolicyAuthorityDetectorBase(
    _CompactKeyedRegistryCandidateDetectorBase[
        RegistryProjectionPolicyAuthorityCandidate
    ],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_named_registry_projection_surface
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[RegistryProjectionPolicyAuthorityCandidate]:
        return _compact_registry_projection_policy_authority_candidates(
            projections, config
        )

    def _findings_from_compact_context(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        facts = _compact_keyed_registry_axis_facts_from_context(context)
        return self._findings_for_candidates(
            _compact_registry_projection_policy_authority_candidates_from_facts(
                projections, facts
            ),
            config,
        )


declare_candidate_rule_detector(
    NonInjectiveTypeRegistryCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Type registry must be injective over its key axis",
        "A nominal registry is only type-safe when each concrete implementation has one canonical key and each key resolves to one implementation. Duplicate keys, duplicate type identities, or concrete descendants without keys mean the registry cannot serve as an injective authority.",
        "injective type registry with one stable key per concrete implementation",
        "registry key axis aliases multiple implementation types or misses concrete descendants",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` registry axis `{candidate.key_type_name}` is not injective: "
        f"duplicate keys {candidate.injectivity_proof.duplicate_key_names}, duplicate types "
        f"{candidate.injectivity_proof.duplicate_type_names}, missing keyed types {candidate.injectivity_proof.missing_type_names}."
    ),
    evidence=lambda candidate: (candidate.evidence,),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactNonInjectiveTypeRegistryDetectorBase,
)


declare_candidate_rule_detector(
    InjectiveTypeRegistryCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Mature injective registry is a metaclass-registration candidate",
        "A registry with a stable key axis, lookup lifecycle, consumer fanout, and an injective type-to-key proof has the shape needed for declaration-owned registration. Replacing its mechanics with AutoRegisterMeta still requires proving that dynamic registration timing and external plugin lifecycle are compatible.",
        "AutoRegisterMeta-backed ABC with an injective type-key proof",
        "registry axis proves one key per implementation type plus mature lookup and consumer fanout, but migration lifecycle remains unproven",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is a mature injective registry over `{candidate.key_type_name}`: "
        f"keys {candidate.registered_case_names}, lookup {candidate.lookup_method_names}, "
        f"consumers {candidate.consumer_symbols}; evaluate AutoRegisterMeta against its registration lifecycle."
    ),
    evidence=lambda candidate: (candidate.evidence,),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactInjectiveTypeRegistryDetectorBase,
)


declare_candidate_rule_detector(
    RegistryProjectionSurfaceCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual registry projection surfaces should derive from the injective registry",
        "Once a registry proves one canonical key per implementation type, export rosters, key/type maps, and option lists are projections of that registry authority. Hand-maintaining those surfaces creates shadow authorities that can drift away from the type-safe registry.",
        "generated projection surface derived from an injective registry proof",
        "manual list or dict surface repeats keys/types already proven by an injective registry",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.surface_name}` is a manual `{candidate.projection_role}` "
        f"`{candidate.surface_kind}` projection "
        f"of injective registry `{candidate.registry_class_name}` over `{candidate.key_type_name}`: "
        f"keys {candidate.shared_key_names}, types {candidate.shared_type_names}, "
        f"coverage {candidate.projection_coverage_ratio:.2f}; "
        f"target `{candidate.projection_target_name}`, "
        f"materialization `{candidate.materialization_rule}`, "
        f"decompression key `{candidate.decompression_key}`."
        + (
            f" Subset policy hint `{candidate.subset_policy_hint}` names the quotient; repeated use should be owned by a projection policy authority."
            if candidate.subset_policy_hint is not None
            and candidate.projection_coverage_ratio < 1.0
            else (
                f" Missing keys {candidate.missing_key_names} and types {candidate.missing_type_names} need a named projection policy."
                if candidate.projection_coverage_ratio < 1.0
                else ""
            )
        )
    ),
    evidence=lambda candidate: (
        SourceLocation(candidate.file_path, candidate.line, candidate.surface_name),
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.projected_names),
        mapping_name=candidate.surface_name,
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            candidate.projection_policy_name,
            candidate.projection_target_name,
            candidate.materialization_rule.value,
        ),
    ),
    detector_base=_CompactRegistryProjectionSurfaceDetectorBase,
)


declare_candidate_rule_detector(
    RegistryProjectionPolicyAuthorityCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated registry subset projections should share a nominal policy authority",
        "A partial projection of an injective registry is a quotient of the registry axis. When several surfaces repeat the same quotient hint, the hint should become a first-class projection policy instead of living as independent allowlists.",
        "nominal registry projection policy reused by generated subset surfaces",
        "multiple registry projection surfaces repeat the same subset hint without one owner",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.registry_class_name}` has repeated `{candidate.policy_hint}` subset projections "
        f"{candidate.surface_names} across roles {tuple(role.value for role in candidate.surface_roles)}; move the quotient into one policy authority "
        f"and materialize targets {candidate.projection_target_names} from specs."
    ),
    evidence=lambda candidate: (candidate.evidence,),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.surface_names),
        mapping_name=f"{candidate.policy_hint}_projection_policy",
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            *(role.value for role in candidate.surface_roles),
            *(rule.value for rule in candidate.materialization_rules),
        ),
    ),
    detector_base=_CompactRegistryProjectionPolicyAuthorityDetectorBase,
)


class KeyedRecordInfrastructureDetector(
    ConfiguredModuleCollectorCandidateDetector[KeyedRecordInfrastructureCandidate]
):
    candidate_collector = _keyed_record_infrastructure_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual keyed-record infrastructure should derive from one substrate",
        "Repeated keyed records, indexes, and lookup mechanics create writable mapping authorities beside their record declarations. The docs prefer one typed keyed-record substrate whose indexes and lookup behavior are derived from declared records.",
        "single authoritative typed keyed-record substrate with derived indexes",
        "manual keyed-record construction or registration and lookup mechanics repeat across tables",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _finding_for_candidate(
        self, candidate: KeyedRecordInfrastructureCandidate
    ) -> RefactorFinding:
        return self.build_finding(
            candidate.finding_summary,
            candidate.finding_evidence,
            metrics=candidate.finding_metrics,
        )


class ManualStructuralRecordMechanicsDetector(
    ConfiguredModuleCollectorCandidateDetector[
        ManualStructuralRecordMechanicsGroupCandidate
    ]
):
    candidate_collector = _manual_structural_record_mechanics_group_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated structural record mechanics should derive from field metadata",
        "When several frozen dataclass records hand-write validation, tuple-style field projection, round-trip reconstruction, and fieldwise transform logic, those mechanics have become a second authority beside the field declarations. The docs prefer one metadata-driven record substrate that derives those mechanics from typed fields.",
        "single typed structural-record substrate with derived validation, projection, and transform mechanics",
        "same dataclass record lifecycle mechanics repeated across sibling structural record classes",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
            CapabilityTag.TYPE_LINEAGE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    )

    def _finding_for_candidate(
        self, group_candidate: ManualStructuralRecordMechanicsGroupCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (item.class_name for item in group_candidate.classes[:6])
        )
        shared_methods = ", ".join(group_candidate.shared_method_names)
        transform_methods = ", ".join(group_candidate.transform_method_names[:6])
        base_names = ", ".join(group_candidate.base_names)
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.build_finding(
            (
                f"Dataclass records {class_names} each hand-roll `{shared_methods}` plus fieldwise transforms "
                f"{transform_methods} on top of base family `{base_names}`."
            ),
            evidence,
        )


class RepeatedConcreteTypeCaseAnalysisDetector(
    CompactClassIndexMultiProjectionDetector,
    ConfiguredCrossModuleCollectorCandidateDetector[
        RepeatedConcreteTypeCaseAnalysisCandidate
    ],
):
    candidate_collector = staticmethod(_repeated_concrete_type_case_analysis_candidates)
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Repeated concrete-type recovery should become nominal family behavior",
        "When several functions repeatedly recover the same semantic family through concrete `isinstance` checks on one carried attribute, the family boundary is still latent. The docs want one nominal ABC and concrete leaf behavior exposed through typed properties or hooks instead of repeated leaf decoding.",
        "single ABC-backed family for the carried subject, with repeated case recovery moved into nominal properties or hooks",
        "same attribute-carried family is re-decoded through repeated concrete runtime type checks across several functions",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        candidates = _compact_repeated_concrete_type_case_candidates(
            cast(
                tuple[CompactRemainingSystemicModuleProjection, ...],
                projections_by_family[CompactRemainingSystemicModuleProjectionFamily],
            ),
            cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
            config,
            class_index=CompactClassFamilyIndex.require(context),
        )
        return self._findings_for_candidates(candidates, config)

    def _finding_for_candidate(
        self, case_candidate: RepeatedConcreteTypeCaseAnalysisCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.function_name for function in case_candidate.functions[:6])
        )
        class_names = ", ".join(case_candidate.concrete_class_names[:6])
        alias_summary = (
            f" Union alias(es): {', '.join(case_candidate.union_alias_names)}."
            if case_candidate.union_alias_names
            else ""
        )
        existing_base_summary = (
            f" Existing abstract base(s): {', '.join(case_candidate.abstract_base_names)}."
            if case_candidate.abstract_base_names
            else ""
        )
        suggested_family_name = _camel_case(case_candidate.subject_role)
        shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
            case_candidate.concrete_class_names
        )
        if (
            shared_suffix
            and len(shared_suffix) >= 6
            and not suggested_family_name.endswith(shared_suffix)
        ):
            suggested_family_name = f"{suggested_family_name}{shared_suffix}"
        elif not suggested_family_name.endswith(("Family", "Witness", "Variant")):
            suggested_family_name = f"{suggested_family_name}Family"
        return self.build_finding(
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(
    CompactClassIndexMultiProjectionDetector,
    ConfiguredCrossModuleCollectorCandidateDetector[ImplicitSelfContractMixinCandidate],
):
    candidate_collector = staticmethod(_implicit_self_contract_mixin_candidates)
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Concrete mixins should not hide consumer contracts behind `self`-casts",
        "The docs reserve mixins for orthogonal reusable concerns that participate in nominal MRO cleanly. When a concrete mixin erases `self` through `cast(..., self)` to reach consumer-owned fields, the mixin is carrying non-orthogonal family logic through a hidden contract instead of a declared base or policy.",
        "declared nominal base or policy row for the shared algorithm instead of a hidden mixin self-contract",
        "concrete mixin methods erase `self` through casts and depend on consumer-owned attributes across several subclasses",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        candidates = _compact_implicit_self_contract_mixin_candidates(
            cast(
                tuple[CompactRemainingSystemicModuleProjection, ...],
                projections_by_family[CompactRemainingSystemicModuleProjectionFamily],
            ),
            cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
            config,
            class_index=CompactClassFamilyIndex.require(context),
        )
        return self._findings_for_candidates(candidates, config)

    def _finding_for_candidate(
        self, mixin_candidate: ImplicitSelfContractMixinCandidate
    ) -> RefactorFinding:
        methods = ", ".join(mixin_candidate.method_names)
        consumers = ", ".join(mixin_candidate.consumer_class_names[:6])
        accessed_attributes = ", ".join(mixin_candidate.accessed_attribute_names[:6])
        cast_types = ", ".join(mixin_candidate.cast_type_names[:6])
        return self.build_finding(
            (
                f"`{mixin_candidate.mixin_name}` uses `cast(..., self)` ({cast_types}) in `{methods}` to reach consumer-owned attributes ({accessed_attributes}) across subclasses {consumers}."
            ),
            mixin_candidate.evidence,
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=len(mixin_candidate.method_names),
                class_count=len(mixin_candidate.consumer_class_names) + 1,
            ),
        )


class RepeatedResultAssemblyPipelineDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedResultAssemblyPipelineCandidate]
):
    candidate_collector = staticmethod(_repeated_result_assembly_pipeline_candidates)
    finding_spec = high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Repeated result-assembly pipeline should collapse into one authoritative assembler",
        "Several owners repeat the same downstream result-assembly stages and differ only in the upstream source or projection that feeds the pipeline. The docs treat that as shared algorithm authority that should move into one template method or authoritative helper with one orthogonal source hook.",
        "single authoritative result-assembly pipeline with one source hook",
        "same staged assembly tail is repeated across sibling functions or methods",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, pipeline_candidate: RepeatedResultAssemblyPipelineCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.qualname for function in pipeline_candidate.functions[:4])
        )
        stage_names = ", ".join(
            (stage.callee_name for stage in pipeline_candidate.shared_tail)
        )
        evidence = tuple(
            (function.evidence for function in pipeline_candidate.functions[:6])
        )
        return self.build_finding(
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(pipeline_candidate.functions),
                statement_count=len(pipeline_candidate.shared_tail),
                class_count=len(
                    {
                        function.qualname.split(".", 1)[0]
                        for function in pipeline_candidate.functions
                        if "." in function.qualname
                    }
                    or {pipeline_candidate.functions[0].qualname}
                ),
                method_symbols=tuple(
                    function.qualname for function in pipeline_candidate.functions
                ),
                shared_statement_texts=tuple(
                    stage.callee_name for stage in pipeline_candidate.shared_tail
                ),
            ),
        )


class CandidateCollectorBoilerplateDetectorBase(
    ModuleCollectorCandidateDetector[CandidateCollectorBoilerplateCandidate],
    CandidateCollectorBoilerplateFindingRecipeSynthesizer,
):
    """Compose collector-boilerplate detection with its proved refactor."""


declare_candidate_rule_detector(
    CandidateCollectorBoilerplateCandidate,
    high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Candidate detector should declare collector strategy",
        "Detector classes repeatedly implement `_candidate_items()` as a one-line forwarding method. That is boilerplate control flow: the detector identity and finding rendering are semantic, while candidate collection is a typed class-level strategy that can be inherited.",
        "typed metaprogrammed detector base that derives candidate collection from a declared strategy",
        "detector class repeats collector forwarding method instead of declaring a collector",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda collector: (
        f"`{collector.class_name}.{collector.method_name}` only forwards to `{collector.collector_name}`; inherit `{collector.recommended_base_name}` and declare `candidate_collector` instead."
    ),
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=0,
        branch_site_count=1,
        call_site_count=1,
        parameter_count=2 if collector.uses_config else 1,
        callee_family_count=1,
    ),
    candidate_collector=CandidateCollectorBoilerplateCandidate.from_module,
    detector_base=CandidateCollectorBoilerplateDetectorBase,
)


declare_candidate_rule_detector(
    TypedCandidateCastBoilerplateCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Candidate template method should receive typed candidates directly",
        "Detector classes repeatedly accept `candidate: object`, immediately cast it to a nominal candidate type, and then never use the object-typed parameter again. That cast belongs in the generic detector base contract: the implementation hook should receive the typed candidate directly.",
        "generic typed candidate detector base with no per-detector cast prelude",
        "candidate-rendering template method starts with a local cast of its only payload parameter",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}.{candidate.method_name}` casts `{candidate.parameter_name}` to `{candidate.candidate_type_name}` before doing real work; parameterize `{candidate.detector_base_name}` and receive `{candidate.local_name}` as that type."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    candidate_collector=_typed_candidate_cast_boilerplate_candidates,
)


class DeclarativeDetectorClassDetectorBase(
    ModuleCollectorCandidateDetector[DeclarativeDetectorClassCandidate],
    DeclarativeDetectorClassFindingRecipeSynthesizer,
):
    """Compose declarative-shell detection with its executable replacement."""


declare_candidate_rule_detector(
    DeclarativeDetectorClassCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Metadata-only detector class should be declared through detector algebra",
        "A detector class whose body only assigns finding metadata and a renderer is not carrying implementation behavior. Its class shell is derivable from the candidate type, detector base, registry key, and declaration line.",
        "one detector-declaration algebra that derives metadata-only detector classes",
        "detector class repeats a nominal class shell around only declarative assignments",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is a {candidate.line_count}-line metadata-only detector over `{candidate.candidate_type_name}` with assignments {candidate.assignment_names}."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=candidate.assignment_names,
        source_name=candidate.base_name,
    ),
    candidate_collector=DeclarativeDetectorClassCandidate.from_module,
    detector_base=DeclarativeDetectorClassDetectorBase,
)


declare_candidate_rule_detector(
    StaticTypedObservationDetectorCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Static observation detector should derive from typed observation algebra",
        "A static detector whose evidence method only collects one typed observation family and maps its line/symbol payload into `SourceLocation` is repeating the same module-observation algorithm. The detector should declare the observation family, item type, evidence threshold, and summary template while the ABC owns collection and evidence projection.",
        "typed observation detector algebra with one shared collection/projection algorithm",
        "detector class shell repeats collect/map/summary mechanics for one observation family",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats a {candidate.line_count}-line static observation shell over `{candidate.observation_family_name}` / `{candidate.observation_type_name}`."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=(
            "finding_spec",
            "observation_family",
            "observation_type",
            "minimum_evidence_count",
            "summary_template",
        ),
        source_name=candidate.observation_family_name,
    ),
    candidate_collector=_static_typed_observation_detector_candidates,
)


declare_candidate_rule_detector(
    SchemaAccessorFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Schema-shaped accessor family should derive from one projection schema",
        "Several public methods on one class fetch one enum-keyed payload field with `self.required(...)` or `self.optional(...)`, then repeat local runtime guards or coercions before returning the value. That is a hidden field schema spread across methods: the enum member, requiredness, accepted type, coercion, and error policy should be declared once and projected through a typed accessor engine.",
        "single authoritative projection schema for enum-keyed payload fields",
        "same class repeats payload accessor methods over one closed enum/key axis",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda family: (
        f"`{family.class_name}` repeats {len(family.method_names)} accessor methods "
        f"over `{family.enum_name}` fields {family.field_names}; requiredness "
        f"{family.requirement_modes} and coercions {family.coercion_kinds} are schema rows."
    ),
    evidence=lambda family: family.evidence_locations,
    compression_certificate=lambda family: family.compression_certificate,
    metrics=lambda family: MappingMetrics.from_field_names(
        mapping_site_count=len(family.method_names),
        mapping_name=family.class_name,
        field_names=family.field_names,
        source_name=family.enum_name,
    ),
    candidate_collector=_schema_accessor_family_candidates,
)


declare_candidate_rule_detector(
    TupleIndexSemanticOpacityCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Carrier tuple context should become a named semantic record",
        "A typed carrier pipeline that accesses context as `pair[0]`, `pair[1]`, or nested numeric tuple paths has collapsed the control-flow smell but introduced a positional data smell. The semantic-compressor normal form is a named product record, authority-owned context object, or step result type whose field names carry the invariant.",
        "named effect context record instead of positional tuple plumbing",
        "effect pipeline stores semantic context in numeric tuple indexes",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.function_name}` uses positional tuple paths "
        f"{candidate.index_expressions} inside carrier pipeline calls "
        f"{candidate.carrier_call_names}."
    ),
    compression_certificate=lambda candidate: CompressionCertificate.from_object_family(
        manual_object_count=candidate.nested_index_count
        + len(candidate.index_expressions),
        replacement_shape=ObjectFamilyShape(shared_objects=("named_effect_context",)),
        semantic_axes=candidate.index_expressions,
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.nested_index_count,
        mapping_name=candidate.function_name,
        field_names=candidate.index_expressions,
        source_name="carrier_tuple_context",
    ),
    candidate_collector=TupleIndexSemanticOpacityCandidateCollector.collect,
    source_candidate_collector=(
        TupleIndexSemanticOpacityCandidateCollector.collect_source
    ),
    detector_base=SourceModuleCollectorCandidateDetector,
)


class FindingSpecConstructionBoilerplateDetector(
    ModuleCollectorCandidateDetector[FindingSpecConstructionCandidate]
):
    candidate_collector = FindingSpecConstructionCandidateCollector.collect
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec construction should use its typed semantic factory",
        "FindingSpec factories own semantic tier defaults and canonical coordinate order. Calling the record constructor directly repeats either that schema, those defaults, or both at every detector declaration.",
        "one typed FindingSpec factory per semantic tier",
        "FindingSpec record is constructed directly instead of through its semantic-tier factory",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _finding_for_candidate(
        self, field_candidate: FindingSpecConstructionCandidate
    ) -> RefactorFinding:
        redundant_defaults = ", ".join(
            (
                f"{name}={value}"
                for name, value in zip(
                    field_candidate.redundant_keyword_names,
                    field_candidate.redundant_keyword_values,
                    strict=True,
                )
            )
        ) or "no explicit semantic defaults"
        return self.build_finding(
            (
                f"`{field_candidate.constructor_name}` constructs FindingSpec directly "
                f"with {redundant_defaults}; use the semantic-tier factory "
                f"`{field_candidate.recommended_builder_name}`."
            ),
            (field_candidate.evidence,),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(field_candidate.redundant_keyword_names),
                mapping_name=field_candidate.constructor_name,
                field_names=field_candidate.redundant_keyword_names,
            ),
        )


declare_candidate_rule_detector(
    ClassMethodLineWitnessCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Detector finding builder should derive detector_id",
        "Concrete detectors repeatedly call `self.finding_spec.build(self.detector_id, ...)`. The detector id is instance-owned template context, not per-finding payload; a shared `build_finding(...)` hook should inject it once.",
        "typed detector template method that injects detector identity into finding construction",
        "finding renderer manually passes detector-owned identity into its own spec builder",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.symbol}` calls `self.finding_spec.build(self.detector_id, ...)`; `build_finding(...)` can derive the detector id from the instance."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_name="FindingSpecBuildBoilerplateDetector",
    candidate_collector=_finding_spec_build_boilerplate_candidates,
)


class DirectBuildFindingRendererDetectorBase(
    ModuleCollectorCandidateDetector[DirectBuildFindingRendererCandidate],
    DirectBuildFindingRendererFindingRecipeSynthesizer,
):
    """Compose direct-renderer detection with its executable replacement."""


declare_candidate_rule_detector(
    DirectBuildFindingRendererCandidate,
    high_confidence_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
        "Direct build_finding renderer should be a typed renderer value",
        "A `_finding_for_candidate` method whose entire body is `return self.build_finding(...)` does not own control flow. It is a data renderer over one candidate type, so the candidate-to-finding algorithm should live once in the ABC and the detector should supply a typed renderer object.",
        "typed candidate finding renderer reused by detector ABC machinery",
        "detector method is only a build_finding payload declaration",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    ),
    summary=lambda renderer: (
        f"`{renderer.class_name}.{renderer.method_name}` is a direct "
        f"`build_finding(...)` renderer with {renderer.positional_arg_count} "
        f"positional payloads and "
        f"{', '.join(renderer.keyword_names) or 'no keywords'}."
    ),
    metrics=lambda renderer: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=renderer.class_name,
        field_names=("summary", "evidence", *renderer.keyword_names),
    ),
    candidate_collector=DirectBuildFindingRendererCandidate.from_module,
    detector_base=DirectBuildFindingRendererDetectorBase,
)


declare_candidate_rule_detector(
    EnumMetadataTableCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Enum metadata table should be carried by enum members",
        "An enum whose properties only index a module-level table by `self` splits member identity from member metadata. The metadata should move into enum construction so each member carries its own typed semantic record.",
        "enum member construction owns the member metadata",
        "enum properties read a parallel metadata table keyed by the same enum family",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.EXPORT_MAPPING,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` reads {candidate.property_names} from `{candidate.table_name}` across {candidate.case_count} enum cases."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.case_count,
        mapping_name=candidate.table_name,
        field_names=candidate.property_names,
        source_name=candidate.class_name,
    ),
    candidate_collector=_enum_metadata_table_candidates,
)


class CompactDataclassNamespaceCliMirrorCandidateBase(
    CompactProjectionCandidateDetector[
        _DataclassNamespaceCliModuleProjection,
        DataclassNamespaceCliMirrorCandidate,
    ],
):
    module_projection_family = _DataclassNamespaceCliModuleProjectionFamily
    compact_report_context_requires_target_projection = True

    def _candidates_from_compact_projections(
        self,
        projections: tuple[_DataclassNamespaceCliModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[DataclassNamespaceCliMirrorCandidate]:
        del config
        return _dataclass_namespace_cli_mirror_candidates_from_projections(projections)


declare_candidate_rule_detector(
    DataclassNamespaceCliMirrorCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Dataclass config surfaces should derive namespace and CLI adapters",
        "A dataclass already owns its field names and defaults. Re-enumerating those fields in a namespace constructor and an argparse specification table creates parallel configuration surfaces that can drift from the typed record.",
        "one dataclass field authority that derives namespace construction and CLI argument rows",
        "dataclass fields are mirrored manually in both from-namespace construction and CLI argument specs",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` mirrors {len(candidate.field_names)} namespace fields and {len(candidate.cli_field_names)} CLI fields through `{candidate.argument_spec_name}` instead of deriving adapters from the dataclass."
    ),
    evidence=lambda candidate: (
        SourceLocation(candidate.file_path, candidate.line, candidate.class_name),
        SourceLocation(
            candidate.file_path,
            candidate.from_namespace_line,
            f"{candidate.class_name}.from_namespace",
        ),
        SourceLocation(
            candidate.argument_spec_file_path,
            candidate.argument_spec_line,
            candidate.argument_spec_name,
        ),
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.field_names) + len(candidate.cli_field_names),
        mapping_name=candidate.class_name,
        field_names=candidate.field_names,
        source_name=candidate.argument_spec_name,
    ),
    detector_base=CompactDataclassNamespaceCliMirrorCandidateBase,
)


declare_candidate_rule_detector(
    ManualFiberTagCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Manual fiber tag should become nominal family",
        "A string-valued instance tag is manually selecting behavior while the same instance still carries fields from several incompatible fibers. That leaves the family above the zero-incoherence threshold and admits disagreement states the host type system could rule out.",
        "host-native nominal fiber decomposition with one subclass per behavior fiber",
        "manual instance tag drives behavior while irrelevant coordinates remain constructible on every fiber",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    ),
    summary=lambda fiber_candidate: (
        f"`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}."
    ),
    evidence=lambda fiber_candidate: (
        SourceLocation(
            fiber_candidate.file_path,
            fiber_candidate.init_line,
            f"{fiber_candidate.class_name}.__init__",
        ),
        SourceLocation(
            fiber_candidate.file_path,
            fiber_candidate.method_line,
            f"{fiber_candidate.class_name}.{fiber_candidate.method_name}",
        ),
    ),
    metrics=lambda fiber_candidate: DispatchCountMetrics.from_literal_family(
        dispatch_axis=f"self.{fiber_candidate.tag_name}",
        literal_cases=fiber_candidate.case_names,
    ),
    candidate_collector=_manual_fiber_tag_candidates,
)


class StructuralConfusabilityDetector(
    ModuleCollectorCandidateDetector[StructuralConfusabilityCandidate]
):
    candidate_collector = staticmethod(_structural_confusability_candidates)
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Consumer observes a confusable duck-typed family",
        "A consumer only observes a partial structural view, and several unrelated classes are confusable under that view. Without a nominal witness, the distortion floor stays above zero and the family boundary remains implicit.",
        "ABC-backed nominal witness for a structurally confusable implementation family",
        "consumer depends on a partial structural view shared by several unrelated classes",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _finding_for_candidate(
        self, confusability_candidate: StructuralConfusabilityCandidate
    ) -> RefactorFinding:
        evidence = (
            SourceLocation(
                confusability_candidate.file_path,
                confusability_candidate.line,
                confusability_candidate.function_name,
            ),
        )
        return self.build_finding(
            f"`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view.",
            evidence,
        )


class SemanticWitnessFamilyDetector(
    ModuleCollectorCandidateDetector[WitnessCarrierFamilyCandidate]
):
    candidate_collector = _witness_carrier_family_candidates
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Renamed semantic carrier roles need one nominal inheritance spine",
        "Several frozen dataclass carriers repeat location and naming roles under distinct field names. One root should own family identity while reusable role mixins own the orthogonal slices and compose through MRO.",
        "one authoritative nominal carrier root plus reusable semantic-role mixins",
        "same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.MRO_ORDERING,
        ),
    )

    def _finding_for_candidate(
        self, witness_candidate: WitnessCarrierFamilyCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(witness_candidate.file_path, line, class_name)
                for class_name, line in zip(
                    witness_candidate.class_names,
                    witness_candidate.line_numbers,
                    strict=True,
                )
            )
        )
        role_summary = "; ".join(
            (
                f"{role.require_mixin_name()} for {field_names}"
                for role, field_names in witness_candidate.role_field_names
            )
        )
        shared_role_names = tuple(
            role.value for role in witness_candidate.shared_role_names
        )
        return self.build_finding(
            f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {shared_role_names} under renamed fields; establish one nominal carrier root and compose {role_summary} through MRO.",
            evidence,
            metrics=WitnessCarrierMetrics(
                class_count=len(witness_candidate.class_names),
                shared_role_count=len(witness_candidate.shared_role_names),
                class_names=witness_candidate.class_names,
                shared_role_names=shared_role_names,
            ),
        )
