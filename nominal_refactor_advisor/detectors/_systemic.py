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
import re
from typing import Generic, TypeVar

from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..ast_tools import (
    CompactModuleIdentity,
    PythonSourcePathPolicy,
    SourceModule,
)
from ..class_index import CompactNamedProjectionSurface, CompactSortedKeyCall
from ..native_syntax import NativePythonSyntaxIndex
from ..registry_normal_form import (
    CanonicalRegistryIdentityStage,
    DerivedRegistryProjectionStage,
    MetaclassRegisteredRegistryStage,
    ProvenRegistryMaturityStage,
    SingleRegistryAuthorityStage,
    UnifiedRegistryAxisFamilyStage,
)
from ..taxonomy import CapabilityTag, ObservationTag

from ._base import *
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
                and _ast_terminal_name(subnode.func) == "isinstance"
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
                    if (type_name := _ast_terminal_name(item))
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
class CompactSpecAxisModuleProjection:
    families: tuple[SpecAxisFamily, ...]


class CompactSpecAxisModuleProjectionFamily(
    CollectedFamily[CompactSpecAxisModuleProjection]
):
    item_type = CompactSpecAxisModuleProjection
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            item.families
            for item in items
            if isinstance(item, CompactSpecAxisModuleProjection)
        )
    )
    source_collector = staticmethod(
        lambda source_module, syntax_index: _native_spec_axis_projections(
            source_module,
            syntax_index,
        )
    )

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[CompactSpecAxisModuleProjection]:
        del cls
        return [CompactSpecAxisModuleProjection(_spec_axis_families(parsed_module))]


def _native_spec_axis_projections(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[CompactSpecAxisModuleProjection] | None:
    """Project spec axes from the shared top-level assignment fragments."""

    if not syntax_index.is_complete:
        return None
    try:
        statements = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
        )
        parsed_module = source_module.parsed_module(
            ast.Module(body=list(statements), type_ignores=[]),
        )
        return [CompactSpecAxisModuleProjection(_spec_axis_families(parsed_module))]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


@dataclass(frozen=True)
class CompactValidateShapeModuleProjection:
    methods: tuple[ValidateShapeGuardMethodCandidate, ...]


class CompactValidateShapeModuleProjectionFamily(
    CollectedFamily[CompactValidateShapeModuleProjection]
):
    item_type = CompactValidateShapeModuleProjection
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            item.methods
            for item in items
            if isinstance(item, CompactValidateShapeModuleProjection)
        )
    )
    source_collector = staticmethod(
        lambda source_module, syntax_index: _native_validate_shape_projections(
            source_module,
            syntax_index,
        )
    )

    @classmethod
    def collect(
        cls, parsed_module: ParsedModule
    ) -> list[CompactValidateShapeModuleProjection]:
        del cls
        return [
            CompactValidateShapeModuleProjection(
                _validate_shape_guard_method_candidates(
                    (parsed_module,), min_guard_count=2
                )
            )
        ]


def _native_validate_shape_projections(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[CompactValidateShapeModuleProjection] | None:
    """Project only direct class ``validate`` methods from native syntax."""

    if not syntax_index.is_complete:
        return None
    parsed_module = source_module.parsed_module(
        ast.Module(body=[], type_ignores=[]),
    )
    methods: list[ValidateShapeGuardMethodCandidate] = []
    try:
        for function_node in sorted(
            syntax_index.common_captures().get("function", ()),
            key=lambda node: (node.start_byte, -node.end_byte),
        ):
            if syntax_index.declared_name(function_node) != "validate":
                continue
            class_node = syntax_index.direct_enclosing_class(function_node)
            if class_node is None:
                continue
            function = syntax_index.function_for(function_node)
            synthetic_class = ast.ClassDef(
                name=syntax_index.declared_name(class_node),
                bases=[],
                keywords=[],
                body=[function],
                decorator_list=[],
            )
            candidate = _validate_shape_guard_method_candidate(
                parsed_module,
                synthetic_class,
                function,
                min_guard_count=2,
            )
            if candidate is not None:
                methods.append(candidate)
        return [CompactValidateShapeModuleProjection(tuple(methods))]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


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


def _closed_axis_conversion_matrix_compression_certificate(
    candidate: ClosedAxisConversionMatrixCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.line_count + len(candidate.function_names),
            len(candidate.function_names) * 2,
        ),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("conversion_dispatcher", "conversion_table"),
            per_axis_objects=("conversion_axis_case",),
        ),
        semantic_axes=(
            *(f"source:{item}" for item in candidate.source_axis_values),
            *(f"target:{item}" for item in candidate.target_axis_values),
        ),
    )


def _option_record_quotient_compression_certificate(
    candidate: OptionRecordQuotientCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.line_count,
            len(candidate.class_names) * max(len(candidate.field_names), 1),
        ),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("option_schema_catalog",),
            per_axis_objects=("option_case",),
        ),
        semantic_axes=(*(f"record:{item}" for item in candidate.class_names),),
        residual_object_count=len(candidate.field_names)
        + len(candidate.default_names)
        + len(candidate.common_base_names),
    )


_SINGLE_TEMPLATE_CALL_METRICS = OrchestrationMetrics(
    function_line_count=0,
    branch_site_count=0,
    call_site_count=1,
    parameter_count=1,
    callee_family_count=1,
)


_PREDICATE_GRAMMAR_AUTHORITY_SUFFIXES = (
    "Authority",
    "Builder",
    "Catalog",
    "Decoder",
    "Extractor",
    "Pipeline",
    "Profile",
    "Projection",
    "Renderer",
)


class AstTypeIsinstanceNameProjection:
    def from_expr(self, node: ast.AST) -> str | None:
        attribute = as_ast(node, ast.Attribute)
        if attribute is not None and name_id(attribute.value) == "ast":
            return attribute.attr
        name = name_id(node)
        return name if name is not None and name[:1].isupper() else None

    def from_isinstance_call(self, call: ast.Call) -> tuple[str, ...]:
        if _call_name(call.func) != "isinstance" or len(call.args) < 2:
            return ()
        type_expr = call.args[1]
        if isinstance(type_expr, ast.Tuple):
            return sorted_tuple(
                (
                    type_name
                    for element in type_expr.elts
                    if (type_name := self.from_expr(element)) is not None
                )
            )
        type_name = self.from_expr(type_expr)
        return () if type_name is None else (type_name,)


AST_TYPE_ISINSTANCE_NAME_PROJECTION = AstTypeIsinstanceNameProjection()


def _uses_ast_traversal(node: ast.AST) -> bool:
    return any(
        (
            isinstance(call, ast.Call)
            and _call_name(call.func) in {"_walk_nodes", "ast.walk"}
            for call in _walk_nodes(node)
        )
    )


def _predicate_grammar_score(method: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return sum(
        (
            isinstance(node, (ast.If, ast.BoolOp, ast.Compare))
            or (
                isinstance(node, ast.Call)
                and bool(AST_TYPE_ISINSTANCE_NAME_PROJECTION.from_isinstance_call(node))
            )
        )
        for node in _walk_nodes(method)
    )


def _inline_ast_predicate_grammar_candidates(
    module: ParsedModule,
) -> tuple[InlineAstPredicateGrammarCandidate, ...]:
    candidates: list[InlineAstPredicateGrammarCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith(
            _PREDICATE_GRAMMAR_AUTHORITY_SUFFIXES
        ):
            continue
        for method in CLASS_NODE_AUTHORITY.methods(node):
            ast_type_names = sorted_tuple(
                {
                    type_name
                    for call in _walk_nodes(method)
                    if isinstance(call, ast.Call)
                    for type_name in (
                        AST_TYPE_ISINSTANCE_NAME_PROJECTION.from_isinstance_call(call)
                    )
                }
            )
            predicate_count = _predicate_grammar_score(method)
            traversal_count = sum(
                (isinstance(loop, (ast.For, ast.While)) for loop in _walk_nodes(method))
            )
            if (
                predicate_count < 6
                or traversal_count == 0
                or not ast_type_names
                or not _uses_ast_traversal(method)
            ):
                continue
            candidates.append(
                InlineAstPredicateGrammarCandidate(
                    file_path=module.file_path,
                    line=method.lineno,
                    class_name=node.name,
                    method_name=method.name,
                    ast_type_names=ast_type_names,
                    predicate_count=predicate_count,
                    traversal_count=traversal_count,
                    line_count=max(
                        1, (method.end_lineno or method.lineno) - method.lineno + 1
                    ),
                )
            )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


def _inline_ast_predicate_grammar_certificate(
    candidate: InlineAstPredicateGrammarCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=candidate.predicate_count + candidate.traversal_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("ast_predicate_grammar", "matcher_runner"),
            per_axis_objects=("node_type_rule",),
        ),
        semantic_axes=candidate.ast_type_names,
        residual_object_count=max(1, len(candidate.ast_type_names)),
    )


declare_candidate_rule_detector(
    InlineAstPredicateGrammarCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Authority method contains inline AST predicate grammar",
        "A nominal authority method that still hand-codes AST traversal, isinstance checks, attribute guards, and boolean predicate ladders has only moved the smell. The deeper normal form is a declarative matcher/effect-step grammar: node types and field predicates are data, while traversal and failure semantics live in one reusable ABC.",
        "declarative AST matcher grammar with traversal and predicate semantics owned once",
        "authority method repeats AST traversal and predicate mechanics inline",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}.{candidate.method_name}` has "
        f"{candidate.predicate_count} inline AST predicates over {candidate.ast_type_names} "
        f"inside {candidate.traversal_count} traversal block(s); move this into a matcher grammar."
    ),
    scaffold=lambda candidate: (
        "class AstPredicateRule(ABC):\n"
        "    node_type: ClassVar[type[ast.AST]]\n"
        "    def matches(self, node: ast.AST) -> bool: ...\n\n"
        "    @classmethod\n"
        "    def concrete_rule_types(cls): ...\n\n"
        "    @classmethod\n"
        "    def matches_anywhere(cls, root: ast.AST):\n"
        "        rule_types = cls.concrete_rule_types()\n"
        "        # Collect all declaration-derived matches and fail on overlap.\n"
        "        ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Replace inline predicate ladder in `{candidate.class_name}.{candidate.method_name}` "
        "with declaration-derived `AstPredicateRule` subclasses and one traversal runner.\n"
        "# Keep node type, field name, operator, and projection residue as typed rule data; "
        "fail closed if rule matches overlap instead of introducing precedence metadata."
    ),
    metrics=lambda candidate: OrchestrationMetrics(
        function_line_count=candidate.line_count,
        branch_site_count=candidate.predicate_count,
        call_site_count=candidate.traversal_count,
        parameter_count=len(candidate.ast_type_names),
        callee_family_count=1,
    ),
    compression_certificate=_inline_ast_predicate_grammar_certificate,
    candidate_collector=_inline_ast_predicate_grammar_candidates,
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
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            body = _trim_docstring_body(statement.body)
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
    scaffold=lambda projection_candidate: (
        "@dataclass(frozen=True)\nclass PathProjection:\n    base_attr: str\n    parts: tuple[str, ...]\n    def __get__(self, instance, owner=None) -> Path: ..."
    ),
    codemod_patch=lambda projection_candidate: (
        "# Replace repeated @property path projections with PathProjection descriptors.\n# Keep only base attribute and path parts as declarative data."
    ),
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
            _ast_terminal_name(decorator) == "property"
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
            body = _trim_docstring_body(statement.body)
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
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class CollectionAttributeProjection:\n"
        "    collection_attr: str\n"
        "    member_attr: str\n"
        "    def __get__(self, instance, owner=None): ..."
    ),
    codemod_patch=lambda candidate: (
        "# Replace repeated collection projection @property methods with one "
        "CollectionAttributeProjection descriptor; keep only collection and "
        "member attribute names as class-level data."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=f"{candidate.class_name}.{candidate.collection_name}",
        field_names=candidate.projected_attribute_names,
    ),
    candidate_collector=_collection_projection_property_family_candidates,
)


class SuffixAxisCompatibilitySurfaceDetector(
    ConfiguredModuleCollectorCandidateDetector[SuffixAxisSurfaceCandidate]
):
    candidate_collector = _suffix_axis_surface_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Mirrored suffix-axis APIs should collapse to one authoritative context",
        "Several operations are exposed once per suffix-named axis, such as `*_for_context` and `*_for_session`. When the same axis split repeats across an owner, the code is usually maintaining adapter surfaces instead of choosing one authoritative request/context record and deriving any compatibility projection at the boundary.",
        "single authoritative context/request record instead of repeated suffix-axis adapter surfaces",
        "same owner repeats an operation family across the same suffix-named axes",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _finding_for_candidate(
        self, surface_candidate: SuffixAxisSurfaceCandidate
    ) -> RefactorFinding:
        axis_summary = " / ".join(surface_candidate.axis_names)
        operation_summary = ", ".join(surface_candidate.operation_names[:5])
        method_names = tuple(method.qualname for method in surface_candidate.methods)
        return self.build_finding(
            (
                f"`{surface_candidate.owner_name}` repeats suffix-axis APIs for axes {axis_summary} "
                f"across operations {operation_summary}."
            ),
            surface_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass OperationContext:\n    ...\n\n# Route operations through one authoritative context/session/request record.\n# Keep at most one boundary adapter that constructs the authority, not one adapter per operation."
            ),
            codemod_patch=(
                f"# Collapse suffix-axis method family {method_names[:8]} onto one authoritative record.\n"
                "# Prefer one conversion point from the secondary axis into the primary axis, then delete per-operation mirrored wrappers."
            ),
            metrics=ParameterThreadMetrics(
                function_count=len(surface_candidate.operation_names),
                shared_parameter_count=len(surface_candidate.axis_names),
                shared_parameter_names=surface_candidate.axis_names,
            ),
        )


class SiblingRoleHelperSymmetryDetector(
    ModuleCollectorCandidateDetector[SiblingRoleHelperSymmetryCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Sibling role helpers should collapse to one local authority",
        "One owner has private helpers whose names differ by a role token but whose control skeletons and parameters are parallel. That is usually one local computation split into symmetrical role-specific helpers, which makes future changes require duplicated edits.",
        "one authoritative local computation instead of parallel role-specific helpers",
        "same owner has role-token sibling helpers with matching control skeletons",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _finding_for_candidate(
        self, helper_candidate: SiblingRoleHelperSymmetryCandidate
    ) -> RefactorFinding:
        helper_summary = ", ".join(helper_candidate.method_names)
        role_summary = " / ".join(helper_candidate.role_tokens)
        shared_summary = "_".join(helper_candidate.shared_tokens)
        return self.build_finding(
            (
                f"`{helper_candidate.owner_name}` splits `{shared_summary}` across role helpers "
                f"{helper_summary} for roles {role_summary}."
            ),
            helper_candidate.evidence,
            scaffold=(
                f"def resolve_{shared_summary}(...):\n    # Compute the role-specific values together while the branch facts are live.\n    ...\n    return left_value, right_value\n\n# Use a small record only if this result crosses a boundary; keep local-only pairs as values."
            ),
            codemod_patch=(
                f"# Collapse sibling helpers {helper_candidate.method_names} into one local authority.\n"
                "# Preserve role names at the assignment site instead of maintaining parallel helper bodies."
            ),
            metrics=ParameterThreadMetrics(
                function_count=len(helper_candidate.methods),
                shared_parameter_count=len(helper_candidate.shared_tokens),
                shared_parameter_names=helper_candidate.shared_tokens,
            ),
        )


class ResidualClosedAxisIndirectionDetector(
    ModuleCollectorCandidateDetector[ResidualClosedAxisIndirectionCandidate]
):
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
            scaffold=(
                f'from abc import ABC, abstractmethod\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\nclass AxisPolicy(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "axis_key"\n    __skip_if_no_key__ = True\n    axis_key: ClassVar[{axis_candidate.enum_name}]\n\n    @classmethod\n    def for_key(cls, key: {axis_candidate.enum_name}):\n        return cls.__registry__[key]()\n\n    @abstractmethod\n    def project(self, source): ...\n\n    @abstractmethod\n    def run(self, ctx): ...\n\n# Move the table projection and residual branch behavior into enum-keyed policy subclasses.\n# Derive table-like views from AxisPolicy.__registry__ only if callers still need them.'
            ),
            codemod_patch=(
                f"# Replace `{axis_candidate.table_name}[{axis_candidate.dispatch_axis_expression}]` plus residual "
                f"`{axis_candidate.enum_name}` branching with `AxisPolicy.for_key({axis_candidate.dispatch_axis_expression})`.\n"
                f"# Move projections ({value_summary}) and per-case behavior into registered `AxisPolicy` subclasses."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=axis_candidate.enum_name,
                literal_cases=axis_candidate.table_case_names,
            ),
        )


class InlineEnumSubsetGuardDetector(
    ModuleCollectorCandidateDetector[InlineEnumSubsetGuardCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Inline enum subset guard should derive from enum-owned policy",
        "A branch that hardcodes an enum-member subset is a closed-axis policy table in disguise. The policy should be owned by the enum member or a typed row family, with any lookup derived exhaustively from that type-safe source.",
        "type-safe enum-owned policy instead of inline enum subset literals",
        "function branches on a hand-enumerated subset of one closed enum axis",
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.PROJECTION_DICT,
        ),
    )

    def _finding_for_candidate(
        self, guard_candidate: InlineEnumSubsetGuardCandidate
    ) -> RefactorFinding:
        cases = ", ".join(
            (
                f"{guard_candidate.enum_name}.{case_name}"
                for case_name in guard_candidate.case_names
            )
        )
        return self.build_finding(
            (
                f"`{guard_candidate.function_name}` checks `{guard_candidate.dispatch_axis_expression} "
                f"{guard_candidate.operator} {{{cases}}}`; move that subset into enum-owned typed policy."
            ),
            (guard_candidate.evidence,),
            scaffold=(
                f"@dataclass(frozen=True)\nclass PolicyRow:\n    key: {guard_candidate.enum_name}\n    requires_policy: bool\n\ndef exhaustive_enum_lookup(enum_type, rows):\n    by_key = {{row.key: row for row in rows}}\n    if set(by_key) != set(enum_type):\n        raise TypeError('incomplete enum policy')\n    return by_key\n\nPOLICY_BY_KEY = exhaustive_enum_lookup(...)\nif POLICY_BY_KEY[{guard_candidate.dispatch_axis_expression}].requires_policy:\n    ..."
            ),
            codemod_patch=(
                f"# Replace inline subset {{{cases}}} with a policy owned by `{guard_candidate.enum_name}`.\n"
                "# Derive any enum-keyed dict from enum members or typed policy rows, and fail if coverage is incomplete."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=guard_candidate.enum_name,
                literal_cases=guard_candidate.case_names,
            ),
        )


class SplitDispatchAuthorityDetector(
    ModuleCollectorCandidateDetector[SplitDispatchAuthorityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Cooperating dispatch layers should collapse into one product-family authority",
        "The docs treat repeated cooperating dispatch layers as split authority. When one orchestration function selects a strategy-family implementation and separately routes another axis through `singledispatch`, the operation usually wants one authoritative product-family policy or one request-dispatched plan.",
        "single authoritative product-family or request-dispatched policy for cooperating dispatch axes",
        "one orchestrator combines a strategy-family selector with a separate singledispatch generic",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _finding_for_candidate(
        self, dispatch_candidate: SplitDispatchAuthorityCandidate
    ) -> RefactorFinding:
        evidence = (
            dispatch_candidate.evidence,
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.selector_line,
                f"{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}",
            ),
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.generic_line,
                dispatch_candidate.generic_function_name,
            ),
        )
        return self.build_finding(
            (
                f"`{dispatch_candidate.qualname}` combines strategy selector "
                f"`{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}({dispatch_candidate.strategy_axis_expression})` "
                f"with singledispatch `{dispatch_candidate.generic_function_name}({dispatch_candidate.generic_axis_expression})` "
                f"through callback `{dispatch_candidate.bridge_callback_name}`, splitting one operation across two dispatch authorities."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass DispatchPlan:\n    strategy: object\n    source_type: type[object]\n\nclass ProductPolicy(ABC):\n    plan_key: ClassVar[DispatchPlan]\n    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# Collapse `{dispatch_candidate.strategy_root_name}` and `{dispatch_candidate.generic_function_name}` under one product-family authority.\n"
                "# Let one nominal plan/policy own both `{dispatch_candidate.strategy_axis_expression}` and `{dispatch_candidate.generic_axis_expression}` so the orchestrator dispatches once."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=2,
                dispatch_axis=(
                    f"{dispatch_candidate.strategy_axis_expression} x "
                    f"{dispatch_candidate.generic_axis_expression}"
                ),
                literal_cases=(
                    *dispatch_candidate.strategy_case_names[:3],
                    *dispatch_candidate.generic_case_names[:3],
                ),
            ),
        )


class EmptyLeafProductFamilyDetector(
    ModuleCollectorCandidateDetector[EmptyLeafProductFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Empty multiple-inheritance leaves should collapse into one product-family authority",
        "The docs allow mixins for orthogonal reusable concerns, but empty leaf classes that merely enumerate all combinations of two reusable axes are usually a handwritten product table in inheritance form. That product should become one keyed authority or one product-family selector.",
        "single authoritative keyed product family instead of empty inheritance combinations",
        "empty leaf classes encode the full Cartesian product of two reusable inheritance axes",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _finding_for_candidate(
        self, product_candidate: EmptyLeafProductFamilyCandidate
    ) -> RefactorFinding:
        left_axis = ", ".join(product_candidate.left_axis_base_names)
        right_axis = ", ".join(product_candidate.right_axis_base_names)
        leaf_preview = ", ".join(product_candidate.leaf_class_names[:6])
        return self.build_finding(
            (
                f"Empty leaf classes {leaf_preview} encode `{left_axis}` x `{right_axis}` through multiple inheritance instead of one product-family authority."
            ),
            product_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass ProductRule:\n    axis_left: object\n    axis_right: object\n    policy_type: type[object]\n\nPRODUCT_RULES = (...)\n"
            ),
            codemod_patch=(
                "# Replace the empty Cartesian-product leaf classes with one keyed product table or one nominal selector family.\n# Keep only irreducible axis-local behavior on the reusable bases; do not encode the cross product as `pass` subclasses."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=(
                    f"{' | '.join(product_candidate.left_axis_base_names)} x {' | '.join(product_candidate.right_axis_base_names)}"
                ),
                literal_cases=product_candidate.leaf_class_names,
            ),
        )


class ClosedConstantSelectorDetector(
    ModuleCollectorCandidateDetector[ClosedConstantSelectorCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Closed selector over sibling constants should derive from one selector table",
        "The docs treat branch ladders that choose among sibling specs, plans, contracts, or other immutable constants as duplicated selector logic once the constant family already exists. The selector should collapse into one authoritative keyed table or selector record so wrappers and downstream views are derived.",
        "single authoritative selector table for a closed constant family",
        "one function branches over a small predicate family and returns sibling constants or one shared wrapper around them",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PREDICATE_CHAIN,
        ),
    )

    def _finding_for_candidate(
        self, selector_candidate: ClosedConstantSelectorCandidate
    ) -> RefactorFinding:
        constants_preview = ", ".join(selector_candidate.constant_names[:4])
        guard_preview = ", ".join(selector_candidate.guard_expressions[:2])
        family_label = (
            selector_candidate.common_constructor_name
            or selector_candidate.family_suffix
            or "selected constant family"
        )
        wrapper_summary = (
            f"`{selector_candidate.wrapper_name}(...)` around "
            if selector_candidate.wrapper_name is not None
            else ""
        )
        guard_summary = (
            f"guards `{guard_preview}` and default fallback"
            if selector_candidate.guard_expressions
            else "a closed fallback ladder"
        )
        return self.build_finding(
            (
                f"`{selector_candidate.qualname}` branches over {guard_summary}, returning {wrapper_summary}"
                f"sibling constants {constants_preview} from `{family_label}`."
            ),
            selector_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass SelectorRule:\n    key: object\n    selected: object\n\nSELECTOR_RULES = (\n    SelectorRule(key=..., selected=...),\n)\n_SELECTED_BY_KEY = {rule.key: rule.selected for rule in SELECTOR_RULES}\n"
            ),
            codemod_patch=(
                f"# Replace manual branches in `{selector_candidate.qualname}` with one authoritative selector table.\n"
                "# Select the sibling constant once, then apply any shared wrapper outside the selector."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(selector_candidate.constant_names),
                field_count=max(len(selector_candidate.guard_expressions), 1),
                mapping_name=selector_candidate.wrapper_name or family_label,
                field_names=selector_candidate.constant_names,
                source_name=selector_candidate.qualname,
            ),
        )


class DerivedWrapperSpecShadowDetector(
    ModuleCollectorCandidateDetector[DerivedWrapperSpecShadowCandidate]
):
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
            scaffold=(
                "@dataclass(frozen=True)\nclass ExecutionSpec:\n    key: object\n    runner: object\n    wrapper_name: str | None = None\n    wrapper_defaults: dict[str, object] = field(default_factory=dict)\n\ndef build_wrapper(spec: ExecutionSpec): ...\n"
            ),
            codemod_patch=(
                f"# Remove parallel family `{shadow_candidate.derived_family_name}`.\n# Move `{', '.join(shadow_candidate.extra_field_names) or 'wrapper metadata'}` onto the authoritative `{shadow_candidate.primary_constructor_name}` records and derive wrappers directly from that family."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(shadow_candidate.primary_constant_names),
                field_count=max(len(shadow_candidate.extra_field_names), 1),
                mapping_name=shadow_candidate.derived_family_name,
                field_names=shadow_candidate.extra_field_names,
                source_name=primary_family_label,
                identity_field_names=(shadow_candidate.link_field_name,),
            ),
        )


declare_candidate_rule_detector(
    ModuleKeyedSelectionHelperCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Local keyed-selection helper should collapse into the generic keyed-record table",
        "The docs push reusable table/index machinery into one authoritative substrate. When a module defines a local selection-rule dataclass, a dict-index builder, and a keyed lookup helper that power multiple rule tables, it is reintroducing a second keyed-table framework instead of reusing the generic keyed-record helper.",
        "single authoritative keyed-record table substrate reused across module-level selector tables",
        "module-local selection helper framework powers multiple keyed rule tables",
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
    ),
    summary=lambda helper_candidate: (
        f"`{helper_candidate.rule_class_name}`, `{helper_candidate.helper_function_name}`, and `{helper_candidate.lookup_function_name}` implement a local keyed-selection substrate for {', '.join(helper_candidate.rule_table_names[:4])} and indexes {', '.join(helper_candidate.index_table_names[:4])}."
    ),
    evidence=lambda helper_candidate: helper_candidate.evidence,
    scaffold=lambda helper_candidate: (
        'KeyT = TypeVar("KeyT")\nRecordT = TypeVar("RecordT")\n\n@dataclass(frozen=True)\nclass KeyedRecordTable(Generic[KeyT, RecordT]):\n    records: tuple[RecordT, ...]\n    key_of: Callable[[RecordT], KeyT]\n    def require(self, key: KeyT, *, missing_error=None) -> RecordT: ...\n'
    ),
    codemod_patch=lambda helper_candidate: (
        f"# Remove local keyed-selection helper `{helper_candidate.rule_class_name}` / `{helper_candidate.helper_function_name}` / `{helper_candidate.lookup_function_name}`.\n# Re-express these rule tables through the shared KeyedRecordTable substrate."
    ),
    metrics=lambda helper_candidate: MappingMetrics(
        mapping_site_count=len(helper_candidate.rule_table_names),
        field_count=1,
        mapping_name=helper_candidate.rule_class_name,
        field_names=(helper_candidate.selected_field_name,),
        source_name=helper_candidate.helper_function_name,
        identity_field_names=("key",),
    ),
    candidate_collector=_module_keyed_selection_helper_candidates,
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
    scaffold=lambda shadow_candidate: (
        _axis_policy_registry_scaffold("invariant(self)")
        + f"\n\ndef run_with_axis(axis: {_AXIS_POLICY_KEY_TYPE_NAME}, ...):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(axis)\n    # derive local execution from authoritative policy facts\n"
    ),
    codemod_patch=lambda shadow_candidate: (
        f"# Remove shadow family `{shadow_candidate.shadow.family_name}`.\n# Derive local behavior from authoritative family `{shadow_candidate.authoritative.family_name}` instead of re-owning axis `{shadow_candidate.key_type_name}`."
    ),
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
            scaffold=(
                _axis_policy_registry_scaffold("apply(self, context)")
                + f"\n\ndef run(context):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(context.axis)\n    return policy.apply(context)\n"
            ),
            codemod_patch=(
                f"# Remove residual `{residual_candidate.key_type_name}` branch ladder in `{residual_candidate.qualname}`.\n"
                "# Delegate through the existing keyed family authority and keep only case-local residue on the policy leaves."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=residual_candidate.branch_site_count,
                dispatch_axis=residual_candidate.key_type_name,
                literal_cases=residual_candidate.case_names,
            ),
        )


class ParallelKeyedAxisFamilyDetector(
    CompactClassRepositoryCandidateDetector[ParallelKeyedAxisFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_family_axis_root
    )
    registry_normal_form_stage = UnifiedRegistryAxisFamilyStage
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
            scaffold=(
                _axis_policy_registry_scaffold(
                    "invariant(self)",
                    "runtime_adapter(self, context)",
                )
                + "\n\n# Keep one authoritative keyed family and let secondary modules derive local adapters/specs from it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` onto one authoritative keyed family.\n"
                "# Move the irreducible case-specific hooks to that family or to a single derived adapter table, not two parallel nominal roots."
            ),
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                family_candidate.shared_case_names,
                family_candidate.key_type_name,
            ),
        )


class CompactParallelKeyedTableAxisCandidateBase(
    CompactProjectionCandidateDetector[
        CompactModuleClassProjection,
        ParallelKeyedTableAxisCandidate,
    ],
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_table_axis
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> tuple[ParallelKeyedTableAxisCandidate, ...]:
        del config
        return _parallel_keyed_table_axis_candidates_from_specs(
            _compact_keyed_table_axis_specs(projections)
        )


declare_candidate_rule_detector(
    ParallelKeyedTableAxisCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Parallel keyed tables should collapse into one auto-registered semantic family",
        "The docs require one authoritative owner per closed semantic axis. When multiple modules maintain keyed tables over the same cases, those tables are usually shadow registries for one semantic family. The stronger default normal form is an ABC plus `AutoRegisterMeta`, with table-like views derived from `Family.__registry__` only when callers still need projections.",
        "single AutoRegisterMeta-backed semantic family with derived module-local projections",
        "same closed enum/key axis is encoded by multiple keyed tables across modules",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.PROJECTION_DICT,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    ),
    summary=lambda table_candidate: (
        f"Axis `{table_candidate.key_type_name}` is restated by `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` across cases {', '.join(table_candidate.shared_case_names[:4])}."
    ),
    evidence=lambda table_candidate: table_candidate.evidence,
    scaffold=lambda table_candidate: (
        _axis_policy_registry_scaffold("run(self, request)")
        + f"\n\ndef run_{table_candidate.key_type_name.lower()}(method, request):\n    return {_AXIS_POLICY_ROOT_NAME}.__registry__[method].run(request)\n\n# Derive table-like projections from {_AXIS_POLICY_ROOT_NAME}.__registry__ only if legacy callers need them.\n"
    ),
    codemod_patch=lambda table_candidate: (
        f"# Collapse `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` onto one AutoRegisterMeta-backed semantic family.\n# Replace hardcoded keyed tables with registered subclasses and route behavior through `Family.__registry__[key].run(...)`.\n# Keep any table-like surface as a derived read-only projection from the registry, not as a writable authority."
    ),
    metrics=lambda table_candidate: MappingMetrics(
        mapping_site_count=2,
        field_count=max(len(table_candidate.shared_case_names), 1),
        mapping_name=table_candidate.left.table_name,
        field_names=table_candidate.shared_case_names,
        source_name=table_candidate.key_type_name,
        identity_field_names=("key",),
    ),
    detector_base=CompactParallelKeyedTableAxisCandidateBase,
    registry_normal_form_stage=DerivedRegistryProjectionStage,
)


class ParallelKeyedTableAndFamilyDetector(
    CompactClassRepositoryCandidateDetector[ParallelKeyedTableAndFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_table_axis
    )
    registry_normal_form_stage = SingleRegistryAuthorityStage
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
            scaffold=(
                _axis_policy_registry_scaffold("build(self)")
                + f"\n\n@dataclass(frozen=True)\nclass DerivedAxisRow:\n    key: {_AXIS_POLICY_KEY_TYPE_NAME}\n    policy_type: type[{_AXIS_POLICY_ROOT_NAME}]\n    config: object\n\ndef build_axis_rows() -> tuple[DerivedAxisRow, ...]:\n    return tuple(\n        DerivedAxisRow(key=key, policy_type=policy_type, config=...)\n        for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n    )"
            ),
            codemod_patch=(
                f"# Collapse `{table_candidate.table_name}` and `{table_candidate.family_name}` onto one authoritative metaclass-registry family.\n"
                "# Keep the runtime boundary on the auto-registered family and derive any keyed rows/views from `AxisPolicy.__registry__`."
            ),
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                table_candidate.shared_case_names,
                table_candidate.key_type_name,
            ),
        )


class CallableMethodAxisRegistryDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Callable method-axis registry should become an auto-registered strategy family",
        "A builder call that maps method-axis member names to callable behavior is a hardcoded strategy family in registry-table form. The canonical shape is an ABC plus `AutoRegisterMeta`, with each method implementation declared as a subclass and dispatch routed through `Family.__registry__[method].run(...)`.",
        "AutoRegisterMeta-backed strategy family instead of callable method-axis registry",
        "module-level registry builder maps closed method-axis cases to callable behavior",
        (
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for statement in module.module.body:
            assignment = self._assignment_target_name(statement)
            value = self._assignment_value(statement)
            if assignment is None or not isinstance(value, ast.Call):
                continue
            if not self._is_method_axis_registry_call(value):
                continue
            axis_name = ast.unparse(value.args[0]) if value.args else "Axis"
            operation_names = tuple(
                keyword.arg
                for keyword in value.keywords
                if keyword.arg is not None
                and isinstance(keyword.value, (ast.Name, ast.Attribute))
            )
            if len(operation_names) < 2:
                continue
            operations = ", ".join(operation_names[:4])
            findings.append(
                self.build_finding(
                    (
                        f"`{assignment}` maps `{axis_name}` member names to callable operations "
                        f"{operations}; this is a hardcoded strategy family."
                    ),
                    (SourceLocation(module.file_path, statement.lineno, assignment),),
                    scaffold=(
                        "from abc import ABC, abstractmethod\n"
                        "from typing import ClassVar\n"
                        "from metaclass_registry import AutoRegisterMeta\n\n"
                        "class MethodStrategy(ABC, metaclass=AutoRegisterMeta):\n"
                        '    __registry_key__ = "method"\n'
                        "    __skip_if_no_key__ = True\n"
                        "    method: ClassVar[object | None] = None\n\n"
                        "    @abstractmethod\n"
                        "    def run(self, request): ...\n\n"
                        "def run_method(method, request):\n"
                        "    return MethodStrategy.__registry__[method].run(request)\n"
                    ),
                    codemod_patch=(
                        f"# Replace callable registry `{assignment}` with an AutoRegisterMeta-backed strategy family.\n"
                        "# Move each callable into a registered subclass and dispatch with `Family.__registry__[method].run(...)`."
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        dispatch_axis=axis_name,
                        literal_cases=operation_names,
                    ),
                )
            )
        return findings

    @staticmethod
    def _is_method_axis_registry_call(call: ast.Call) -> bool:
        if len(call.args) != 1 or len(call.keywords) < 2:
            return False
        func_name = ast.unparse(call.func)
        if not (
            func_name.endswith(".from_member_names")
            or func_name.endswith("from_member_names")
        ):
            return False
        axis_name = ast.unparse(call.args[0])
        return axis_name.endswith("Method") or axis_name.endswith("Axis")

    @staticmethod
    def _assignment_target_name(statement: ast.stmt) -> str | None:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            return statement.target.id
        return None

    @staticmethod
    def _assignment_value(statement: ast.stmt) -> ast.AST | None:
        if isinstance(statement, ast.Assign):
            return statement.value
        if isinstance(statement, ast.AnnAssign):
            return statement.value
        return None


class InheritedAutoRegisterConfigBoilerplateDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
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
            if not self._declares_autoregister_meta(indexed_class):
                continue
            repeated_fields = self._repeated_inherited_fields(
                class_index, indexed_class
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
                    scaffold=(
                        "class RegisteredFamilyBase(ABC):\n"
                        '    __registry_key__ = "method"\n'
                        "    __skip_if_no_key__ = True\n\n"
                        "class ConcreteFamilyRoot(RegisteredFamilyBase, metaclass=AutoRegisterMeta):\n"
                        "    # declare behavior contract only; inherit registry config\n"
                        "    ..."
                    ),
                    codemod_patch=(
                        f"# Delete repeated registry protocol fields {field_list} from `{indexed_class.simple_name}`.\n"
                        "# Keep those fields on the inherited shared base. If the runtime registry does not honor inherited config, fix AutoRegisterMeta inheritance semantics instead of copying boilerplate."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(repeated_fields),
                        mapping_name=indexed_class.simple_name,
                        field_names=repeated_fields,
                    ),
                )
            )
        return findings

    @staticmethod
    def _repeated_inherited_fields(
        class_index: CompactClassFamilyIndex,
        indexed_class: CompactIndexedClass,
    ) -> tuple[str, ...]:
        protocol_fields = (
            "__key_extractor__",
            "__registry_key__",
            "__skip_if_no_key__",
        )
        direct_assignments = indexed_class.assignments_by_name
        repeated: list[str] = []
        for field_name in protocol_fields:
            current_text = direct_assignments.get(field_name)
            if current_text is None:
                continue
            for ancestor_symbol in class_index.ancestor_symbols(indexed_class.symbol):
                ancestor = class_index.class_for(ancestor_symbol)
                if ancestor is None:
                    continue
                ancestor_text = ancestor.assignments_by_name.get(field_name)
                if ancestor_text is None:
                    continue
                if ancestor_text == current_text:
                    repeated.append(field_name)
                    break
        return tuple(repeated)

    @staticmethod
    def _declares_autoregister_meta(indexed_class: CompactIndexedClass) -> bool:
        return any(
            metaclass_name == "AutoRegisterMeta"
            or metaclass_name.endswith("AutoRegisterMeta")
            or HELPER_SUPPORT_PROJECTION_AUTHORITY.registration_authority_base_name(
                metaclass_name
            )
            or ("Registered" in metaclass_name and metaclass_name.endswith("Meta"))
            for metaclass_name in indexed_class.metaclass_names
        )


_EXPLICIT_CLASS_ORDER_AXIS_NAMES = ("priority", "precedence", "rank", "order")


class AutoRegisterExplicitPriorityOrderingDetector(
    CompactModuleProjectionDetectorMixin[CompactModuleClassProjection],
    IssueDetector,
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
            if not InheritedAutoRegisterConfigBoilerplateDetector._declares_autoregister_meta(
                indexed_class
            ):
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
                        *evidence_sites[:5],
                    ),
                    scaffold=(
                        "class RegisteredPolicy(ABC, metaclass=AutoRegisterMeta):\n"
                        "    @classmethod\n"
                        "    def ordered(cls):\n"
                        "        return tuple(cls.__subclasses__())\n\n"
                        "# Encode ordering by inheritance/MRO, not by a parallel priority field."
                    ),
                    codemod_patch=(
                        f"# Delete the {axis_label} class axis from `{indexed_class.simple_name}` and its leaves.\n"
                        "# Replace sorted registry traversal over explicit order fields with an MRO/subclass traversal owned by the nominal hierarchy."
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
        PatternId.ABC_TEMPLATE_METHOD,
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
                    scaffold=(
                        "class FirstDeclarationCatalog(CatalogBase):\n"
                        "    declaration = FirstDeclaration(...)\n\n"
                        "class SecondDeclarationCatalog(CatalogBase):\n"
                        "    declaration = SecondDeclaration(...)\n\n"
                        "class CompleteCatalog(\n"
                        "    FirstDeclarationCatalog, SecondDeclarationCatalog\n"
                        "):\n"
                        "    pass\n\n"
                        "# Traverse CompleteCatalog.__mro__ directly."
                    ),
                    codemod_patch=(
                        f"# Delete the {axis_label} field and its constructor arguments from `{family_root.simple_name}` declarations.\n"
                        "# Give each declaration one catalog node and derive the sequence solely from the catalog MRO."
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


class EnumKeyedTableClassAxisShadowDetector(
    ModuleCollectorCandidateDetector[EnumKeyedTableClassAxisShadowCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Enum-keyed table should derive from auto-registered class-declared axis keys",
        "The docs require a single writable owner per closed semantic axis. If a module already declares that axis through class-level enum assignments, adding a writable enum-keyed table over the same cases creates duplicate authority and a synchronization surface. The class-declared axis should be the primary owner and any enum-keyed lookup should be derived from the family registry.",
        "one authoritative metaclass-registry closed-axis owner with derived table/view projections",
        "module-level enum-keyed table overlaps a class family that already declares the same enum axis",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.PROJECTION_DICT,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, axis_candidate: EnumKeyedTableClassAxisShadowCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(axis_candidate.class_names[:4])
        shared_cases = ", ".join(axis_candidate.shared_case_names[:4])
        value_names = ", ".join(axis_candidate.value_type_names[:4])
        return self.build_finding(
            (
                f"`{axis_candidate.table_name}` maps `{axis_candidate.key_type_name}` cases {shared_cases} "
                f"to {value_names}, while classes {class_names} already declare the same axis via "
                f"`{axis_candidate.key_attr_name}`."
            ),
            axis_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("route_type(self)")
                + f"\n\nAXIS_BY_KEY = {{\n    key: policy_type\n    for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n}}\n"
            ),
            codemod_patch=(
                f"# Remove `{axis_candidate.table_name}` as a second writable authority.\n"
                f"# Derive `{axis_candidate.key_type_name}` lookup views from the auto-registered family keyed by `{axis_candidate.key_attr_name}` instead of hardcoding a parallel table."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(axis_candidate.shared_case_names),
                field_count=1,
                mapping_name=axis_candidate.table_name,
                field_names=(axis_candidate.key_attr_name,),
                source_name=axis_candidate.key_type_name,
                identity_field_names=(axis_candidate.key_attr_name,),
            ),
        )


class TransportShellTemplateMethodDetector(
    ConfiguredModuleCollectorCandidateDetector[TransportShellTemplateCandidate]
):
    candidate_collector = _transport_shell_template_candidates
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Template-method family is a transport shell over a downstream authority",
        "The docs say nominal families should have one authoritative owner. When an ABC template method only materializes an intermediate object from a class-level selector, delegates through one hook, and repackages through another hook, the extra family is usually a transport shell around an already authoritative boundary.",
        "single authoritative materialization/execution family instead of a parallel transport shell",
        "template family varies mostly by class-level selector and result adapter",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, shell_candidate: TransportShellTemplateCandidate
    ) -> RefactorFinding:
        selector_values = ", ".join(shell_candidate.selector_value_names)
        kwargs_clause = (
            f" plus `{shell_candidate.kwargs_helper_name}({shell_candidate.source_param_name})`"
            if shell_candidate.kwargs_helper_name is not None
            else ""
        )
        return self.build_finding(
            (
                f"`{shell_candidate.class_name}.{shell_candidate.driver_method_name}` materializes selector values "
                f"{selector_values} from `{shell_candidate.selector_attr_name}` via `{shell_candidate.constructor_name}`"
                f"{kwargs_clause} across {len(shell_candidate.concrete_class_names)} concrete leaves, then only delegates "
                f"through `{shell_candidate.inner_hook_name}` and `{shell_candidate.outer_hook_name}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\nclass MaterializationSpec:\n    selector: object\n    materializer: object\n    executor: object\n    packager: object\n# Dispatch once on the authoritative selector/spec family."
            ),
            codemod_patch=(
                f"# Collapse `{shell_candidate.class_name}` onto the downstream selector/spec family.\n"
                "# Keep one selection boundary and let that boundary own materialization, execution, and result packaging."
            ),
        )


class CrossModuleSpecAxisAuthorityDetector(
    CompactModuleProjectionDetectorMixin[CompactSpecAxisModuleProjection],
    ConfiguredCrossModuleCollectorCandidateDetector[
        CrossModuleSpecAxisAuthorityCandidate
    ],
):
    module_projection_family = CompactSpecAxisModuleProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Cross-module spec axis should have one authority",
        "The docs say one semantic family should have one authoritative owner. When two modules encode the same identity-axis -> executable-axis spec pairs, one table is a duplicate authority unless it is explicitly derived.",
        "one repository-wide authoritative spec-axis family",
        "same identity/executable spec axis is re-encoded across modules",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactSpecAxisModuleProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        candidates = _cross_module_spec_axis_authority_candidates_from_families(
            tuple(
                family for projection in projections for family in projection.families
            ),
            config,
        )
        return self._findings_for_candidates(candidates, config)

    def _finding_for_candidate(
        self, authority_candidate: CrossModuleSpecAxisAuthorityCandidate
    ) -> RefactorFinding:
        family_names = ", ".join(
            (
                f"{Path(family.file_path).name}:{family.family_name}"
                for family in authority_candidate.families
            )
        )
        pair_names = ", ".join(
            (
                f"{identity}->{executable}"
                for identity, executable in authority_candidate.shared_axis_pairs
            )
        )
        axis_fields = " -> ".join(authority_candidate.axis_field_names)
        evidence = tuple(
            (family.evidence for family in authority_candidate.families[:6])
        )
        return self.build_finding(
            (
                f"Families {family_names} each encode the same `{axis_fields}` pairs {pair_names} across module boundaries."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass AxisExecutionSpec:\n    identity: object\n    executable: object\n# Keep one exported authority and let downstream modules compose from it."
            ),
            codemod_patch=(
                "# Extract one repository-wide spec-axis family.\n# Make downstream wrappers, benchmarks, or adapters reference that authority instead of restating identity/executable pairs."
            ),
        )


class ParallelRegistryProjectionFamilyDetector(
    ModuleCollectorCandidateDetector[ParallelRegistryProjectionFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel registry projection builders should collapse into one family spec",
        "The docs say one semantic family should have one authoritative owner. When several functions differ only in which registry authority feeds which target constructor, the projection-axis mapping should become one declared spec or family authority instead of several hand-wired wrappers.",
        "single authoritative registry-projection family",
        "same registry-authority-to-target projection shape repeated across sibling functions",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.BUILDER_CALL,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _finding_for_candidate(
        self, catalog_candidate: ParallelRegistryProjectionFamilyCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.qualname for function in catalog_candidate.functions[:4])
        )
        extractor_bases = ", ".join(
            (
                function.extractor_base_name
                for function in catalog_candidate.functions[:4]
            )
        )
        catalog_types = ", ".join(
            (function.catalog_type_name for function in catalog_candidate.functions[:4])
        )
        evidence = tuple(
            function.evidence for function in catalog_candidate.functions[:6]
        )
        return self.build_finding(
            (
                f"Functions {function_names} each build {catalog_types} through "
                f"`{catalog_candidate.collector_name}(structure, ExtractorBase.{catalog_candidate.registry_accessor_name}())` "
                f"over parallel extractor bases {extractor_bases}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass RegistryProjectionSpec:\n    registry_authority: type\n    target_type: type\n# One helper should own the registry-authority to target mapping."
            ),
            codemod_patch=(
                "# Extract one registry-projection family spec and one authoritative projection builder.\n# Make per-axis public helpers delegate to that authority instead of reconstructing collector(...registry_accessor())."
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
            scaffold=(
                'from metaclass_registry import AutoRegisterMeta\n\nKeyT = TypeVar("KeyT")\n\nclass KeyedNominalFamily(ABC, Generic[KeyT], metaclass=AutoRegisterMeta):\n    __registry_key__ = "registry_key"\n    __skip_if_no_key__ = True\n    registry_key: ClassVar[KeyT | None] = None\n    family_label: ClassVar[str] = "family"\n    @classmethod\n    def for_key(cls, key: KeyT):\n        try:\n            return cls.__registry__[key]\n        except KeyError as error:\n            raise ValueError(f"Unknown {cls.family_label}: {key}") from error'
            ),
            codemod_patch=(
                "# Extract one typed metaclass-registry base that owns registration lookup, duplicate handling, and error shaping.\n# Leave only declarative key attributes and irreducible hook methods on each family root, and read the registered classes from `cls.__registry__`."
            ),
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


class _CompactPrematureRegistryInfrastructureDetectorBase(
    _CompactKeyedRegistryCandidateDetectorBase[
        PrematureRegistryInfrastructureCandidate
    ],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_keyed_registry_axis_root
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[PrematureRegistryInfrastructureCandidate]:
        return PrematureRegistryInfrastructureCandidate.from_facts(
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
            PrematureRegistryInfrastructureCandidate.from_facts(facts), config
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


def _registry_maturity_fanout_metrics(
    candidate: PrematureRegistryInfrastructureCandidate,
) -> RegistrationMetrics:
    return RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
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
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\nclass InjectiveRegistryRow:\n    key: object\n    implementation_type: type[object]\n\n"
        "# Build the registry from rows only after proving keys and implementation types are one-to-one."
    ),
    codemod_patch=lambda candidate: (
        f"# Repair `{candidate.class_name}` before adding or keeping registry metaprogramming.\n"
        "# Give every concrete implementation exactly one canonical key and delete aliases or duplicate key writes.\n"
        "# If aliases are semantic, model them as an explicit alias projection instead of a second registry identity."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactNonInjectiveTypeRegistryDetectorBase,
    registry_normal_form_stage=CanonicalRegistryIdentityStage,
)


declare_candidate_rule_detector(
    InjectiveTypeRegistryCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Mature injective type registry should use metaclass registration",
        "A registry with a stable key axis, lookup lifecycle, consumer fanout, and an injective type-to-key proof has reached the point where handwritten registration mechanics are declaration noise. The metaclass should own population while implementation classes declare only their canonical key and behavior hooks.",
        "AutoRegisterMeta-backed ABC with an injective type-key proof",
        "registry axis proves one key per implementation type plus mature lookup and consumer fanout",
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
        f"consumers {candidate.consumer_symbols}; replace handwritten registry mechanics with AutoRegisterMeta."
    ),
    evidence=lambda candidate: (candidate.evidence,),
    scaffold=lambda candidate: _metaclass_registry_keyed_family_scaffold(
        root_name="InjectiveRegistryFamily",
        key_attr_name=candidate.registry_key_attr_name,
        key_type_name=candidate.key_type_name,
        method_defs=("run(self)",),
    ),
    codemod_patch=lambda candidate: (
        f"# Replace `{candidate.class_name}` handwritten `_registry` population with `AutoRegisterMeta`.\n"
        f"# Keep `{candidate.registry_key_attr_name}` as the canonical class-level key and let the metaclass prove class-time population."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=_CompactInjectiveTypeRegistryDetectorBase,
    registry_normal_form_stage=MetaclassRegisteredRegistryStage,
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
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class RegistryProjectionSpec:\n"
        "    registry_authority: type[object]\n"
        "    projection_policy: str\n"
        "    projection_target: str\n"
        "    materialization_rule: str\n"
        "    decompression_key: str\n\n"
        "def derive_registry_projection(spec: RegistryProjectionSpec):\n"
        "    return project_from_injective_registry(\n"
        "        spec.registry_authority,\n"
        "        policy=spec.projection_policy,\n"
        "        target=spec.projection_target,\n"
        "        materialization=spec.materialization_rule,\n"
        "    )"
    ),
    codemod_patch=lambda candidate: (
        f"# Delete `{candidate.surface_name}` as a handwritten `{candidate.projection_role}` `{candidate.surface_kind}`.\n"
        f"# Replace it with RegistryProjectionSpec({candidate.registry_class_name}, policy={candidate.projection_policy_name!r}, target={candidate.projection_target_name!r}, materialization={candidate.materialization_rule.value!r}).\n"
        + (
            f"# Its decompression key is `{candidate.decompression_key}`; derive it from the injective key/type registry proof."
            if candidate.projection_coverage_ratio >= 1.0
            else (
                f"# Its decompression key is `{candidate.decompression_key}`; derive it through an explicit `{candidate.subset_policy_hint}` projection policy."
                if candidate.subset_policy_hint is not None
                else f"# Either derive the full surface from `{candidate.registry_class_name}` or add a named projection policy explaining the missing keys/types."
            )
        )
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
    scaffold=lambda candidate: (
        "class RegistryProjectionPolicy(ABC):\n"
        "    @abstractmethod\n"
        "    def includes_key(self, key): ...\n\n"
        f"class {candidate.policy_hint.title()}ProjectionPolicy(RegistryProjectionPolicy):\n"
        "    def includes_key(self, key): ...\n\n"
        "REGISTRY_PROJECTION_SPECS = (...,)"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace repeated `{candidate.policy_hint}` subset surfaces {candidate.surface_names} with one nominal projection policy.\n"
        f"# Generate specs for targets {candidate.projection_target_names} using decompression keys {candidate.decompression_keys}."
    ),
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


declare_candidate_rule_detector(
    PrematureRegistryInfrastructureCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Registry infrastructure should prove key, lifecycle, and fanout maturity",
        "The OpenHCS history showed that registries pay rent only when the key axis is stable, registration lifecycle is explicit, and more than one consumer uses the registry. A registry-shaped class without those signals is likely a premature abstraction boundary.",
        "mature registry authority with stable key axis, class-time lifecycle, and consumer fanout",
        "keyed registry infrastructure exists before registered cases and consumers prove the axis",
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
        f"`{candidate.class_name}` is registry-shaped over `{candidate.key_type_name}` via "
        f"`{candidate.registry_key_attr_name}`, but missing maturity signals "
        f"{candidate.missing_maturity_signals}: cases {candidate.registered_case_names}, "
        f"lookup methods {candidate.lookup_method_names}, consumers {candidate.consumer_symbols}."
    ),
    evidence=lambda candidate: (candidate.evidence,),
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\nclass AxisRow:\n    key: object\n    value: object\n\n"
        "# Keep rows in a small typed table until key cases, lifecycle, and consumer fanout are stable enough for a registry."
    ),
    codemod_patch=lambda candidate: (
        f"# Do not promote `{candidate.class_name}` to registry infrastructure until it proves all three signals:\n"
        "# stable key cases, explicit lookup/class-time lifecycle, and at least two independent consumers.\n"
        "# Replace premature registry infrastructure with a typed table or local strategy map while any signal is missing."
    ),
    metrics=_registry_maturity_fanout_metrics,
    detector_base=_CompactPrematureRegistryInfrastructureDetectorBase,
    registry_normal_form_stage=ProvenRegistryMaturityStage,
)


class ManualKeyedRecordTableDetector(
    ConfiguredModuleCollectorCandidateDetector[ManualKeyedRecordTableGroupCandidate]
):
    candidate_collector = _manual_keyed_record_table_group_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual keyed record tables should collapse into one authoritative spec table",
        "When several frozen record classes repeat `_registry`, `register`, and `for_*` lookup around closed keys, the code is hand-maintaining multiple writable tables. The docs prefer one authoritative spec tuple or generic keyed-record table with derived indexes.",
        "single authoritative keyed-record table or derived index",
        "same manual record registration and keyed lookup shell repeated across data classes",
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
        self, group_candidate: ManualKeyedRecordTableGroupCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (item.class_name for item in group_candidate.classes[:6])
        )
        key_fields = ", ".join(
            sorted({item.key_field_name for item in group_candidate.classes[:6]})
        )
        lookup_names = ", ".join(
            sorted({item.lookup_method_name for item in group_candidate.classes[:6]})
        )
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.build_finding(
            (
                f"Record tables {class_names} each repeat `_registry`, `{group_candidate.classes[0].register_method_name}`, "
                f"and `{lookup_names}` around key fields {key_fields}."
            ),
            evidence,
            scaffold=(
                'KeyT = TypeVar("KeyT")\nRecordT = TypeVar("RecordT")\n\n@dataclass(frozen=True)\nclass KeyedRecordTable(Generic[KeyT, RecordT]):\n    records: tuple[RecordT, ...]\n    key_of: Callable[[RecordT], KeyT]\n\n    def by_key(self) -> dict[KeyT, RecordT]:\n        return {self.key_of(record): record for record in self.records}'
            ),
            codemod_patch=(
                "# Replace per-class mutable `_registry` + `register` shells with one authoritative tuple of record specs.\n# Derive the keyed lookup dict once, or factor the pattern into a generic keyed-record table helper."
            ),
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
            scaffold=(
                "@dataclass_transform(field_specifiers=(field, record_field))\nclass StructuralRecordBase:\n    def validate(self): ...\n    def project_fields(self): ...\n    @classmethod\n    def from_projected(cls, projected, metadata): ...\n    def transformed(self, **changes): ...\n"
            ),
            codemod_patch=(
                "# Move validation constraints, projected-field partitions, and transform semantics into typed field metadata.\n# Derive projection, round-trip reconstruction, and fieldwise transforms from one structural-record base instead of re-encoding them per class."
            ),
        )


def _shared_compact_class_index(context: object | None) -> CompactClassFamilyIndex:
    if not isinstance(context, CompactClassFamilyIndex):
        raise TypeError("shared compact class index is unavailable")
    return context


class RepeatedConcreteTypeCaseAnalysisDetector(
    CompactMultiModuleProjectionDetectorMixin,
    ConfiguredCrossModuleCollectorCandidateDetector[
        RepeatedConcreteTypeCaseAnalysisCandidate
    ],
):
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        compact_class_index_from_projection_groups
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

    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
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
        )
        return self._findings_for_candidates(candidates, config)

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
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
            class_index=_shared_compact_class_index(context),
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
            scaffold=(
                f"class {suggested_family_name}(ABC):\n    @property\n    @abstractmethod\n    def case_label(self) -> str: ...\n\n    def explain_case(self, context):\n        return None\n"
            ),
            codemod_patch=(
                f"# Type `{case_candidate.subject_role}` against one nominal ABC family instead of a concrete union surface.\n# Move repeated concrete `isinstance` recovery into abstract properties or case hooks on that family.\n# Keep only irreducible case-local residue in the concrete subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(
    CompactMultiModuleProjectionDetectorMixin,
    ConfiguredCrossModuleCollectorCandidateDetector[ImplicitSelfContractMixinCandidate],
):
    module_projection_families = (
        CompactRemainingSystemicModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )
    compact_shared_group_context_builder = staticmethod(
        compact_class_index_from_projection_groups
    )
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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

    def _findings_from_compact_projection_groups(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
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
        )
        return self._findings_for_candidates(candidates, config)

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
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
            class_index=_shared_compact_class_index(context),
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
            scaffold=(
                "class FamilyBase(ABC):\n    def run_shared_step(self): ...\n\nclass CasePolicy(ABC):\n    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# `{mixin_candidate.mixin_name}` is not an orthogonal mixin; it hides a consumer contract behind `cast(..., self)`.\n"
                "# Move the shared behavior to a declared nominal base or a keyed policy/spec family, and leave only true orthogonal residue in mixins."
            ),
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=len(mixin_candidate.method_names),
                class_count=len(mixin_candidate.consumer_class_names) + 1,
            ),
        )


class RepeatedGuardValidatorFamilyDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedGuardValidatorFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated guard validators should collapse into one case-policy authority",
        "When several sibling boolean helpers walk the same subject through fail-fast guards and case-local final checks, the algorithm skeleton is split across helper names instead of being owned by one nominal case policy or declarative rule family.",
        "single authoritative case-policy or rule-table validator",
        "same subject and subordinate view validated through repeated fail-fast sibling helpers",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _finding_for_candidate(
        self, family_candidate: RepeatedGuardValidatorFamilyCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.function_name for function in family_candidate.functions[:6])
        )
        shared_attrs = ", ".join(family_candidate.shared_attr_names[:6])
        alias_summary = (
            f" through `{family_candidate.alias_source_attr}`"
            if family_candidate.alias_source_attr is not None
            else ""
        )
        shared_helpers = ", ".join(family_candidate.shared_helper_call_names[:3])
        helper_summary = (
            f" Shared helper calls: {shared_helpers}." if shared_helpers else ""
        )
        return self.build_finding(
            (
                f"Boolean validators {function_names} each guard `{family_candidate.subject_param_name}`{alias_summary} "
                f"with the same fail-fast attribute checks over {shared_attrs}.{helper_summary}"
            ),
            family_candidate.evidence,
            scaffold=(
                "class ValidationCasePolicy(ABC):\n    def validation_error(self, subject):\n        child = self._subject_child(subject)\n        if not self._shared_preconditions(subject, child):\n            return self._shared_failure_message()\n        return self._case_specific_error(subject, child)\n\n    @abstractmethod\n    def _case_specific_error(self, subject, child): ..."
            ),
            codemod_patch=(
                "# Collapse these sibling boolean helpers into one authoritative case-policy family or one declarative rule table.\n# Keep shared fail-fast guards in one concrete validator method, and leave only case-specific predicates or handle sets per case."
            ),
        )


class AllMissingAxisPredicateDetector(
    ConfiguredModuleCollectorCandidateDetector[AllMissingAxisPredicateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "All-missing axis predicates should be named axis authorities",
        "A raw conjunction of several `not axis` clauses is a derived predicate over a semantic axis bundle. Spelling that bundle inline makes the relation easy to fork and hard to audit. The normal form is a named tuple, policy, or context method that owns the axis set and lets the branch ask the derived question once.",
        "one named axis bundle or policy predicate deriving the all-missing condition",
        "three or more sibling axes are checked through an inline all-negative boolean conjunction before appending a missing signal",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, predicate_candidate: AllMissingAxisPredicateCandidate
    ) -> RefactorFinding:
        axis_names = ", ".join(predicate_candidate.predicate_names)
        return self.build_finding(
            (
                f"`{predicate_candidate.function_name}` checks all-missing axes "
                f"{axis_names} inline before appending `{predicate_candidate.signal_name}`."
            ),
            (predicate_candidate.evidence,),
            scaffold=(
                "rent_axes = (behavior_axis, abstract_axis, projection_axis, consumer_axis)\n"
                "if not any(rent_axes):\n"
                '    missing.append("derived_signal")'
            ),
            codemod_patch=(
                f"# Name the axis bundle in `{predicate_candidate.function_name}` before testing it.\n"
                f"# Replace the raw conjunction over {predicate_candidate.predicate_names} with `not any(axis_bundle)` "
                f"or a policy method that owns the same axes, then append `{predicate_candidate.signal_name}`."
            ),
        )


class RepeatedValidateShapeGuardFamilyDetector(
    CompactModuleProjectionDetectorMixin[CompactValidateShapeModuleProjection],
    ConfiguredCrossModuleCollectorCandidateDetector[
        RepeatedValidateShapeGuardFamilyCandidate
    ],
):
    module_projection_family = CompactValidateShapeModuleProjectionFamily
    candidate_collector = _repeated_validate_shape_guard_candidates_for_modules
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated validate() shape guards should collapse into one validated-record authority",
        "Sibling nominal records repeat the same fail-fast shape and dimensional guards in `validate()` while differing only in field names or a small residue check. The docs treat that as duplicated contract authority that should move into one shared validated-record base, field-spec table, or mixin hook.",
        "single authoritative validated-record contract for repeated shape/ndim guards",
        "same nominal record family repeats fail-loud shape validation scaffolding",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[CompactValidateShapeModuleProjection, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        candidates = _group_repeated_validate_shape_guard_candidates(
            tuple(
                method for projection in projections for method in projection.methods
            ),
            config,
        )
        return self._findings_for_candidates(candidates, config)

    def _finding_for_candidate(
        self, family_candidate: RepeatedValidateShapeGuardFamilyCandidate
    ) -> RefactorFinding:
        method_symbols = tuple(method.symbol for method in family_candidate.methods)
        method_summary = ", ".join(method_symbols[:6])
        shared_guard_count = len(family_candidate.shared_shape_guard_signatures)
        shared_guard_preview = ", ".join(
            family_candidate.shared_shape_guard_signatures[:3]
        )
        preview_suffix = (
            f" Shared normalized guards include {shared_guard_preview}."
            if shared_guard_preview
            else ""
        )
        return self.build_finding(
            (
                f"Validate methods {method_summary} repeat {shared_guard_count} shared shape/ndim guard forms."
            ),
            family_candidate.evidence,
            scaffold=(
                f"class ShapeValidatedRecord(ABC):\n    def validate(self):\n        for predicate, message in self._shape_guard_rules():\n            if predicate(self):\n                raise ValueError(message)\n        self._validate_residue()\n\n    @classmethod\n    @abstractmethod\n    def _shape_guard_rules(cls): ...\n\n    def _validate_residue(self):\n        return None{preview_suffix}"
            ),
            codemod_patch=(
                "# Collapse repeated `validate()` shape guards into one authoritative validated-record base or field-spec table.\n# Keep only the truly variable residue checks, messages, or field roster on each concrete record."
            ),
        )


class RepeatedResultAssemblyPipelineDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedResultAssemblyPipelineCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
            scaffold=(
                "class ResultAssembler(ABC):\n    @abstractmethod\n    def supply_inputs(self, request): ...\n\n    def assemble(self, request):\n        supplied = self.supply_inputs(request)\n        # run the shared downstream assembly stages here\n        return result"
            ),
            codemod_patch=(
                "# Extract the shared assignment/return tail into one authoritative helper.\n# Leave only the source-supplier stage variant-specific."
            ),
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
    scaffold=lambda collector: (
        f"class {collector.class_name}({collector.recommended_base_name}):\n    candidate_collector = {collector.collector_name}\n"
    ),
    codemod_patch=lambda collector: (
        f"# Delete the forwarding `_candidate_items()` method.\n# Change the detector base to `{collector.recommended_base_name}` and assign `candidate_collector = {collector.collector_name}`."
    ),
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=0,
        branch_site_count=1,
        call_site_count=1,
        parameter_count=2 if collector.uses_config else 1,
        callee_family_count=1,
    ),
    detector_priority=-19,
    candidate_collector=_candidate_collector_boilerplate_candidates,
)


declare_candidate_rule_detector(
    TypedCandidateCastBoilerplateCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
    scaffold=lambda candidate: (
        f"class {candidate.class_name}({candidate.detector_base_name}[{candidate.candidate_type_name}]):\n    def {candidate.method_name}(self, {candidate.local_name}: {candidate.candidate_type_name}) -> RefactorFinding:\n        ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Change the detector base to `{candidate.detector_base_name}[{candidate.candidate_type_name}]`.\n# Rename the hook argument from `{candidate.parameter_name}` to `{candidate.local_name}` and delete the local `cast(...)` prelude."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_priority=-18,
    candidate_collector=_typed_candidate_cast_boilerplate_candidates,
)


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
    scaffold=lambda candidate: (
        f"declare_module_detector({candidate.candidate_type_name}, finding_spec, finding_renderer, detector_base={candidate.base_name})"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace `{candidate.class_name}` with `declare_module_detector(...)`.\n# Keep only true detector-specific values: spec, renderer, optional collector, base, and priority."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=candidate.assignment_names,
        source_name=candidate.base_name,
    ),
    detector_priority=-17,
    candidate_collector=_declarative_detector_class_candidates,
)


declare_candidate_rule_detector(
    StaticTypedObservationDetectorCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
    scaffold=lambda candidate: (
        f'declare_typed_observation_detector(\n    "{candidate.class_name}",\n    finding_spec,\n    {candidate.observation_family_name},\n    {candidate.observation_type_name},\n    summary_template,\n)'
    ),
    codemod_patch=lambda candidate: (
        f"# Replace `{candidate.class_name}` with `declare_typed_observation_detector(...)`.\n# Keep detector-specific semantics as declarations: finding spec, observation family/type, minimum evidence, and summary template."
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
    detector_priority=-16,
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
    scaffold=lambda family: (
        "@dataclass(frozen=True)\n"
        "class ProjectionFieldSpec:\n"
        "    key: Enum\n"
        "    required: bool\n"
        "    coerce: Callable[[object], object]\n\n"
        "class PayloadProjectionSchema:\n"
        "    fields: ClassVar[tuple[ProjectionFieldSpec, ...]]\n"
        "    def project(self, key): ..."
    ),
    codemod_patch=lambda family: (
        f"# Replace accessor methods {family.method_names} on `{family.class_name}` "
        f"with one authoritative projection schema keyed by `{family.enum_name}`.\n"
        "# Keep required/optional mode, accepted type, coercion, and error text as "
        "field-spec coordinates; derive named accessors only if callers need them."
    ),
    compression_certificate=lambda family: family.compression_certificate,
    metrics=lambda family: MappingMetrics.from_field_names(
        mapping_site_count=len(family.method_names),
        mapping_name=family.class_name,
        field_names=family.field_names,
        source_name=family.enum_name,
    ),
    detector_priority=-13,
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
    scaffold=lambda candidate: (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class PipelineContext:\n"
        "    source: Source\n"
        "    projection: Projection\n"
        "# Replace `pair[0]`/`pair[1]` with named fields derived once by the carrier stage."
    ),
    codemod_patch=lambda candidate: (
        "# Introduce a named product record or authority-owned context for the carrier stage.\n"
        "# Replace numeric tuple paths with named fields; keep the carrier, but stop encoding semantics by position."
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
    detector_priority=-13,
    candidate_collector=_tuple_index_semantic_opacity_candidates,
    source_candidate_collector=lambda module, syntax_index, config: (
        None
        if any(token in module.source for token in _TUPLE_INDEX_OPACITY_CARRIER_CALLS)
        and re.search(r"\[[^\]]*\d[^\]]*\]", module.source) is not None
        else ()
    ),
    detector_base=SourceModuleCollectorCandidateDetector,
)


class FindingSpecDefaultFieldBoilerplateDetector(
    ModuleCollectorCandidateDetector[FindingSpecDefaultFieldCandidate]
):
    candidate_collector = _finding_spec_default_field_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec semantic defaults should be constructor-derived",
        "FindingSpec constructors already encode confidence and certification defaults. Restating those semantic fields in every detector is declaration boilerplate; the constructor should carry the shared semantic tier and leave only true local residue.",
        "constructor-level semantic spec defaults with no repeated confidence/certification payload",
        "FindingSpec call repeats semantic default keywords that can be derived from its constructor",
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
        self, field_candidate: FindingSpecDefaultFieldCandidate
    ) -> RefactorFinding:
        keyword_summary = ", ".join(
            (
                f"{name}={value}"
                for name, value in zip(
                    field_candidate.redundant_keyword_names,
                    field_candidate.redundant_keyword_values,
                    strict=True,
                )
            )
        )
        constructor_note = (
            f" and use `{field_candidate.recommended_constructor_name}`"
            if field_candidate.recommended_constructor_name
            != field_candidate.constructor_name
            else ""
        )
        return self.build_finding(
            (
                f"`{field_candidate.constructor_name}` restates derived semantic defaults "
                f"{keyword_summary}{constructor_note}."
            ),
            (field_candidate.evidence,),
            scaffold=(
                f"{field_candidate.recommended_constructor_name}(\n    pattern_id=...,\n    title=...,\n    ...\n)"
            ),
            codemod_patch=(
                f"# Replace `{field_candidate.constructor_name}` with "
                f"`{field_candidate.recommended_constructor_name}` where needed.\n"
                f"# Delete redundant semantic keywords: {', '.join(field_candidate.redundant_keyword_names)}."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(field_candidate.redundant_keyword_names),
                mapping_name=field_candidate.constructor_name,
                field_names=field_candidate.redundant_keyword_names,
            ),
        )


declare_candidate_rule_detector(
    ClassMethodLineWitnessCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
    scaffold=lambda candidate: (
        "return self.build_finding(\n    summary,\n    evidence,\n    ...\n)"
    ),
    codemod_patch=lambda candidate: (
        "# Replace `self.finding_spec.build(` with `self.build_finding(`.\n# Delete the leading `self.detector_id,` argument."
    ),
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_name="FindingSpecBuildBoilerplateDetector",
    candidate_collector=_finding_spec_build_boilerplate_candidates,
)


class DirectBuildFindingRendererDetector(
    ModuleCollectorCandidateDetector[DirectBuildFindingRendererCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
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
    )

    def _finding_for_candidate(
        self, renderer: DirectBuildFindingRendererCandidate
    ) -> RefactorFinding:
        keyword_summary = ", ".join(renderer.keyword_names) or "no keywords"
        return self.build_finding(
            (
                f"`{renderer.class_name}.{renderer.method_name}` is a direct "
                f"`build_finding(...)` renderer with {renderer.positional_arg_count} "
                f"positional payloads and {keyword_summary}."
            ),
            (renderer.evidence,),
            scaffold=(
                "finding_renderer = CandidateFindingRenderer[Candidate](\n    summary=lambda candidate: ...,\n    evidence=lambda candidate: ...,\n)"
            ),
            codemod_patch=(
                f"# Move the `{renderer.method_name}` payload in `{renderer.class_name}` to a `CandidateFindingRenderer` classvar.\n# Let `CandidateFindingDetector._finding_for_candidate` run the renderer."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=1,
                mapping_name=renderer.class_name,
                field_names=("summary", "evidence", *renderer.keyword_names),
            ),
        )


declare_candidate_rule_detector(
    CanonicalFindingSpecBuilderCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec coordinates should use one typed semantic builder",
        "Detector specs repeatedly enumerate the same semantic coordinate names: pattern, title, why, capability gap, relation context, and tag axes. A typed builder can make that product structure explicit once and leave each detector to provide only its coordinate values.",
        "typed FindingSpec builder with canonical semantic coordinate order",
        "detector repeats FindingSpec keyword schema locally",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` builds `{candidate.constructor_name}` by spelling {len(candidate.keyword_names)} FindingSpec coordinate keywords; use `{candidate.builder_name}`."
    ),
    scaffold=lambda candidate: (
        f"finding_spec = {candidate.builder_name}(\n    PatternId.EXAMPLE,\n    title,\n    why,\n    capability_gap,\n    relation_context,\n)"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace `{candidate.constructor_name}(pattern_id=..., title=..., ...)` with `{candidate.builder_name}(...)` and let the builder own coordinate names."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=candidate.class_name,
        field_names=candidate.keyword_names,
    ),
    candidate_collector=_canonical_finding_spec_builder_candidates,
)


declare_candidate_rule_detector(
    OptionRecordQuotientCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Option record family should derive from one schema catalog",
        "Several small frozen option/config records in the same module are often projections of one closed format axis. Keeping every record as a hand-written class preserves type names, but repeats product mechanics and default surfaces that can be generated from a typed option schema catalog.",
        "typed option schema catalog that derives concrete option records",
        "field-only option/config record family repeats product-record mechanics across a closed format axis",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"{candidate.file_path} declares option record family {candidate.class_names} over fields {candidate.field_names}; derive the records from one typed option schema catalog."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "OPTION_SCHEMAS = (\n    OptionSchema('csv', CsvOptions, fields=(...)),\n    OptionSchema('json', JsonOptions, fields=(...)),\n)\n\n# Derive concrete frozen records and defaults from the schema catalog."
    ),
    codemod_patch=lambda candidate: (
        "# Keep the public option record names, but derive their field/default declarations from one schema catalog.\n# The only per-option residue should be semantic field/default differences."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.class_names),
        mapping_name="option_schema_catalog",
        field_names=candidate.field_names,
        identity_field_names=candidate.class_names,
    ),
    compression_certificate=_option_record_quotient_compression_certificate,
    detector_priority=-8,
    candidate_collector=_option_record_quotient_candidates,
)


declare_candidate_rule_detector(
    ClosedAxisConversionMatrixCandidate,
    high_confidence_certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Conversion matrix should factor into source and target axes",
        "Functions named as pairwise conversions encode a product of two closed axes: source representation and target representation. The advisor should collapse the matrix into one dispatcher/table whose cases are derived from the axes instead of hand-maintaining one function per pair.",
        "closed source/target conversion axes with one derived dispatcher",
        "module declares many pairwise conversion functions whose names form a source-by-target matrix",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"{candidate.file_path} declares conversion matrix {candidate.function_names} over sources {candidate.source_axis_values} and targets {candidate.target_axis_values}; factor it into closed axes."
    ),
    scaffold=lambda candidate: (
        "class SourceMemory(Enum): ...\nclass TargetMemory(Enum): ...\n\nCONVERTERS = {\n    (SourceMemory.CPU, TargetMemory.GPU): convert_cpu_gpu,\n}\n\ndef convert(value, source, target):\n    return CONVERTERS[(source, target)](value)"
    ),
    codemod_patch=lambda candidate: (
        "# Replace pairwise conversion function selection with one source/target axis table.\n# Keep specialized conversion bodies only as private table entries when they carry real backend semantics."
    ),
    metrics=lambda candidate: DispatchCountMetrics(
        dispatch_site_count=len(candidate.function_names),
        dispatch_axis="source,target",
        literal_cases=(*candidate.source_axis_values, *candidate.target_axis_values),
    ),
    compression_certificate=_closed_axis_conversion_matrix_compression_certificate,
    detector_priority=-9,
    candidate_collector=_closed_axis_conversion_matrix_candidates,
)


declare_candidate_rule_detector(
    ArrayProtocolProbeBridgeCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated array capability probes should become a bridge authority",
        "Several operations probe the same array protocol attributes. The bridge normal form is one nominal array bridge that owns capability discovery and exposes typed operation hooks, rather than every operation rediscovering shape/device/dtype semantics.",
        "array bridge ABC with capability properties and operation hooks",
        "multiple operations repeat the same array protocol capability probes",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"Operations {candidate.function_names} repeat array capability probes "
        f"{candidate.attribute_names}; move capability discovery into an array bridge."
    ),
    scaffold=lambda candidate: (
        "class ArrayBridge(ABC):\n"
        + "\n".join(
            (
                f"    @property\n    @abstractmethod\n    def {attribute_name.strip('_')}(self): ..."
                for attribute_name in candidate.attribute_names
            )
        )
        + "\n\n    @abstractmethod\n    def normalize(self, value): ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Replace repeated probes {candidate.attribute_names} in {candidate.function_names} "
        "with one array bridge selected at the boundary.\n"
        "# Keep protocol-specific dtype/device/shape logic behind bridge capability properties."
    ),
    metrics=lambda candidate: ProbeCountMetrics(probe_site_count=candidate.probe_count),
    compression_certificate=lambda candidate: candidate.compression_certificate,
    detector_priority=-9,
    candidate_collector=_array_protocol_probe_bridge_candidates,
)


declare_candidate_rule_detector(
    LifecycleStageSequenceCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated lifecycle stage sequence should move into a template method",
        "Several functions execute the same ordered stage calls. That is a lifecycle skeleton: an ABC should own sequencing, while implementations provide hooks for the irreducible stages or payload residue.",
        "lifecycle ABC template method with stage hooks",
        "multiple operations repeat the same lifecycle stage sequence",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.NORMALIZED_AST,
        ),
    ),
    summary=lambda candidate: (
        f"Functions {candidate.function_names} repeat lifecycle stages "
        f"{candidate.stage_names}; move sequencing into an ABC template method."
    ),
    scaffold=lambda candidate: (
        "class LifecycleTemplate(ABC):\n"
        "    def run(self, request):\n"
        + "\n".join(
            (
                f"        request = self.{stage_name}(request)"
                for stage_name in candidate.stage_names
            )
        )
        + "\n        return request"
    ),
    codemod_patch=lambda candidate: (
        f"# Move repeated stage order {candidate.stage_names} out of {candidate.function_names}.\n"
        "# Put the sequence in one ABC template method and leave only stage hooks or payload residue in implementations."
    ),
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.function_names),
        statement_count=len(candidate.stage_names),
        class_count=1,
        method_symbols=candidate.function_names,
    ),
    compression_certificate=lambda candidate: candidate.compression_certificate,
    detector_priority=-9,
    candidate_collector=_lifecycle_stage_sequence_candidates,
)


declare_candidate_rule_detector(
    NodeVisitorStackBoilerplateCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Manual AST visitor scope stacks should be inherited",
        "Concrete `ast.NodeVisitor` classes that hand-declare multiple scope stacks and repeat push/pop transitions are reimplementing one traversal skeleton. The stack lifecycle belongs in a nominal ABC; concrete visitors should supply hooks for observation-specific work.",
        "one nominal visitor ABC owns stack lifecycle and concrete visitors provide hooks",
        "same visitor class declares multiple stack fields and push/pop transition hooks",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
        ),
    ),
    summary=lambda candidate: (
        f"`{candidate.qualname}` hand-declares visitor stacks {candidate.stack_names} across {candidate.transition_method_names}."
    ),
    scaffold=lambda candidate: (
        "class Visitor(ClassFunctionStackNodeVisitor):\n    def before_visit_function(self, node):\n        ..."
    ),
    codemod_patch=lambda candidate: (
        "# Delete repeated stack lifecycle methods after moving initialization and `visit_ClassDef`/`visit_FunctionDef` push/pop transitions into a nominal ABC such as `ClassFunctionStackNodeVisitor`; keep only visitor-specific hooks and node handlers in the concrete class."
    ),
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.transition_method_names),
        statement_count=candidate.line_count,
        class_count=1,
        method_symbols=tuple(
            (
                f"{candidate.qualname}.{method_name}"
                for method_name in candidate.transition_method_names
            )
        ),
    ),
    candidate_collector=_node_visitor_stack_boilerplate_candidates,
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
    scaffold=lambda candidate: (
        "class MetadataEnum(StrEnum):\n    def __new__(cls, value: str, label: str):\n        obj = str.__new__(cls, value)\n        obj._value_ = value\n        obj.label = label\n        return obj"
    ),
    codemod_patch=lambda candidate: (
        f"# Move `{candidate.table_name}` values into `{candidate.class_name}` member tuples and delete the table-backed property lookups."
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
    scaffold=lambda candidate: (
        "for field in fields(ConfigRecord):\n    value = namespace_values.get(field.name, field.default)\n    ...\n\nCLI_SPECS = tuple(spec_from_field(field) for field in fields(ConfigRecord) if field.name in HELP)"
    ),
    codemod_patch=lambda candidate: (
        f"# Derive `{candidate.class_name}.from_namespace()` and `{candidate.argument_spec_name}` from dataclass fields/defaults.\n# Keep per-option help text as the only CLI-specific residue."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.field_names) + len(candidate.cli_field_names),
        mapping_name=candidate.class_name,
        field_names=candidate.field_names,
        source_name=candidate.argument_spec_name,
    ),
    detector_base=CompactDataclassNamespaceCliMirrorCandidateBase,
)


class NestedBuilderShellDetector(
    ConfiguredModuleCollectorCandidateDetector[NestedBuilderShellCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Nested builder shell should collapse into one authoritative request boundary",
        "A builder forwards a substantial semantic parameter family unchanged into a subordinate nominal builder and only adds a small residue locally. The docs treat that as split request authority: one layer should own the forwarded family instead of rebuilding it inside another shell.",
        "single authoritative request/context builder boundary",
        "one builder nests a forwarded subordinate request builder inside a second nominal shell",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
    )

    def _finding_for_candidate(
        self, shell_candidate: NestedBuilderShellCandidate
    ) -> RefactorFinding:
        forwarded = ", ".join(shell_candidate.forwarded_parameter_names)
        residue_fields = ", ".join(shell_candidate.residue_field_names)
        residue_sources = ", ".join(shell_candidate.residue_source_names)
        return self.build_finding(
            (
                f"`{shell_candidate.qualname}` forwards `{forwarded}` into "
                f"`{shell_candidate.nested_callee_name}` under `{shell_candidate.nested_field_name}` "
                f"while separately deriving `{residue_fields}` from `{residue_sources}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\nclass OuterRequest:\n    child_request: ChildRequest\n\n    @classmethod\n    def from_source(cls, source, *, child_request: ChildRequest):\n        return cls(child_request=child_request, ...)\n"
            ),
            codemod_patch=(
                f"# Stop rebuilding `{shell_candidate.nested_callee_name}` inside `{shell_candidate.qualname}`.\n"
                "# Accept the subordinate request/context directly, or move both layers into one authoritative builder."
            ),
            metrics=ParameterThreadMetrics(
                function_count=1,
                shared_parameter_count=len(shell_candidate.forwarded_parameter_names),
                shared_parameter_names=shell_candidate.forwarded_parameter_names,
            ),
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
    scaffold=lambda fiber_candidate: _manual_fiber_tag_scaffold(fiber_candidate),
    codemod_patch=lambda fiber_candidate: _manual_fiber_tag_patch(fiber_candidate),
    metrics=lambda fiber_candidate: DispatchCountMetrics.from_literal_family(
        dispatch_axis=f"self.{fiber_candidate.tag_name}",
        literal_cases=fiber_candidate.case_names,
    ),
    candidate_collector=_manual_fiber_tag_candidates,
)


class DeferredClassRegistrationDetector(
    ModuleCollectorCandidateDetector[ManualRegistryCandidate]
):
    candidate_collector = _manual_registry_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Class registration is decoupled from class existence",
        "Manual decorator- or helper-based registration leaves a reachable state where a class exists but the registry has not been updated. The host already provides zero-delay registration via `metaclass-registry` or another class-time hook.",
        "zero-delay metaclass-registry class registration with collision checks and runtime provenance",
        "class registration is performed as a separate auxiliary step rather than at class creation time",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _finding_for_candidate(
        self, registry_candidate: ManualRegistryCandidate
    ) -> RefactorFinding:
        evidence = [
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                registry_candidate.decorator_name,
            )
        ]
        evidence.extend(
            (
                SourceLocation(
                    registry_candidate.file_path, registry_candidate.line, class_name
                )
                for class_name in registry_candidate.class_names[:5]
            )
        )
        return self.build_finding(
            f"Registry `{registry_candidate.registry_name}` is updated through manual decorator `{registry_candidate.decorator_name}` for classes {registry_candidate.class_names}, leaving registration structurally decoupled from class creation.",
            tuple(evidence),
            scaffold=_manual_registry_scaffold(registry_candidate),
            codemod_patch=_manual_registry_patch(registry_candidate),
            metrics=RegistrationMetrics(
                registration_site_count=len(registry_candidate.class_names),
                registry_name=registry_candidate.registry_name,
            ),
        )


class StructuralConfusabilityDetector(
    ModuleCollectorCandidateDetector[StructuralConfusabilityCandidate]
):
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
            scaffold=_structural_confusability_scaffold(confusability_candidate),
            codemod_patch=_structural_confusability_patch(confusability_candidate),
        )


class SemanticWitnessFamilyDetector(
    ModuleCollectorCandidateDetector[WitnessCarrierFamilyCandidate]
):
    candidate_collector = _witness_carrier_family_candidates
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Semantic carrier family should share one nominal base",
        "Several frozen dataclass carriers repeat the same location and naming roles under different field names. That leaves one semantic family structurally expanded instead of giving it one nominal carrier root.",
        "one authoritative nominal base for a semantic metadata carrier family",
        "same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
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
        return self.build_finding(
            f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier.",
            evidence,
            scaffold=_witness_carrier_family_scaffold(witness_candidate),
            codemod_patch=_witness_carrier_family_patch(witness_candidate),
            metrics=WitnessCarrierMetrics(
                class_count=len(witness_candidate.class_names),
                shared_role_count=len(witness_candidate.shared_role_names),
                class_names=witness_candidate.class_names,
                shared_role_names=witness_candidate.shared_role_names,
            ),
        )
