"""Shared detector helper functions.

This module contains private analysis helpers that support detector families
across the split implementation modules.
"""

from __future__ import annotations

from ..factorization import (
    FactorizationRow,
    ResidueHookNamesCarrier,
    factorization_axis_catalog_certificate,
)
from ..annotation_semantics import CLASSVAR_ANNOTATION_AUTHORITY
from ..ast_tools import (
    AstExpressionProjection,
    BuiltinCallName,
    SourceModule,
    walk_function_body_nodes,
)
from ..native_syntax import NativePythonSyntaxIndex
from ..models import AutoRegisterMetaRentSignal
from ..semantic_algebra import FiniteAxisSystem, ObjectFamilyShape
from ..semantic_description_length import (
    ClassFamilyCompressionProfile,
    CompressionCertificate,
)
from ..exact_method_authority import (
    ExactLeafMethodAncestorPromotionComponent,
    ExactLeafMethodAncestorPromotionComponentBuilder,
    ExactMethodRoleComponent,
    ExactMethodRoleComponentBuilder,
)
from ..semantic_identity import SemanticRoleIdentityToken
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from typing import Callable, TypeAlias, TypeVar

from ._base import *
from ._substrate_support import *
from ._substrate_support import _class_ancestor_name_map
from ..class_index import (
    CompactClassMethod,
    CompactClassMethodSemanticProfile,
    CompactMethodSemanticCoordinate,
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjection,
    ClassSymbolResolutionAuthority,
    build_compact_class_family_index,
)

BaseBundleClassGroups: TypeAlias = dict[tuple[str, ...], list[ast.ClassDef]]
NamedStringSequenceSpec: TypeAlias = tuple[str, int, tuple[str, ...]]
NamedStringSequenceSpecs: TypeAlias = tuple[NamedStringSequenceSpec, ...]
MutableNamedStringSequenceSpecs: TypeAlias = list[NamedStringSequenceSpec]
NamedStringSequenceSpecsByName: TypeAlias = dict[str, MutableNamedStringSequenceSpecs]
DerivedQuerySignature: TypeAlias = tuple[str | None, str, str, str]
DerivedQuerySpecsBySignature: TypeAlias = dict[
    DerivedQuerySignature, MutableNamedStringSequenceSpecs
]

_APPEND_METHOD_NAME = "append"


_IMPLICIT_METHOD_PARAMETER_NAMES = frozenset({"self", "cls"})
_SEQUENCE_WRAPPER_CALL_NAMES = BuiltinCallName.sequence_wrapper_names()


class HelperSupportProjectionAuthority:
    def declares_autoregister_meta(self, node: ast.ClassDef) -> bool:
        return any(
            (
                (
                    metaclass_name
                    := AstExpressionProjection.terminal_name(keyword.value)
                )
                is not None
                and (
                    metaclass_name == "AutoRegisterMeta"
                    or metaclass_name.endswith("AutoRegisterMeta")
                    or self.registration_authority_base_name(metaclass_name)
                    or (
                        "Registered" in metaclass_name
                        and metaclass_name.endswith("Meta")
                    )
                )
                for keyword in node.keywords
                if keyword.arg == "metaclass"
            )
        )

    @staticmethod
    def registration_authority_base_name(base_name: str) -> bool:
        tokens = frozenset(
            token.lower()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
                base_name,
            )
            if token
        )
        return bool(
            tokens & {"autoregister", "registered", "registry"}
            or (
                "registration" in tokens
                and bool(tokens & {"authority", "base", "family", "meta", "root"})
            )
            or ("stable" in tokens and bool(tokens & {"axis", "key"}))
            or ("key" in tokens and "family" in tokens)
            or (
                "nominal" in tokens
                and "base" in tokens
                and bool(tokens & {"axis", "family", "formula", "policy"})
            )
        )

    def inherits_named_registration_authority(self, node: ast.ClassDef) -> bool:
        return any(
            (
                (base_name := AstExpressionProjection.terminal_name(base)) is not None
                and self.registration_authority_base_name(base_name)
                for base in node.bases
            )
        )

    def declares_named_registration_authority(self, node: ast.ClassDef) -> bool:
        return (
            "AutoRegister" in node.name
            or "Registered" in node.name
            or node.name.endswith("KeyFamily")
            or self.registration_authority_base_name(node.name)
        )

    def declares_registry_protocol_authority(self, node: ast.ClassDef) -> bool:
        assignments = CLASS_NODE_AUTHORITY.direct_assignments(node)
        return "__registry__" in assignments and "__registry_key__" in assignments

    def declares_stable_registry_axis_authority(self, node: ast.ClassDef) -> bool:
        return "stable_key_axis" in CLASS_NODE_AUTHORITY.direct_assignments(node)

    def family_has_autoregister_authority(
        self, class_index: ClassFamilyIndex, indexed_class: IndexedClass
    ) -> bool:
        return any(
            (
                ancestor is not None
                and (
                    self.declares_autoregister_meta(ancestor.node)
                    or self.declares_named_registration_authority(ancestor.node)
                    or self.inherits_named_registration_authority(ancestor.node)
                    or self.declares_registry_protocol_authority(ancestor.node)
                    or self.declares_stable_registry_axis_authority(ancestor.node)
                )
                for ancestor in (
                    class_index.class_for(symbol)
                    for symbol in (
                        indexed_class.symbol,
                        *class_index.ancestor_symbols(indexed_class.symbol),
                    )
                )
            )
        )

    def common_semantic_key_attr_names(
        self, concrete_descendants: tuple[IndexedClass, ...]
    ) -> tuple[str, ...]:
        if not concrete_descendants:
            return ()
        assignment_name_sets = tuple(
            (
                frozenset(
                    name
                    for name, value in CLASS_NODE_AUTHORITY.direct_assignments(
                        descendant.node
                    ).items()
                    if value is not None and _looks_like_semantic_key_attr(name)
                )
                for descendant in concrete_descendants
            )
        )
        common_names = set(assignment_name_sets[0])
        for assignment_names in assignment_name_sets[1:]:
            common_names &= set(assignment_names)
        return sorted_tuple(common_names)

    def derivable_nominal_root_names(
        self, shapes: Sequence[NominalAuthorityShape]
    ) -> tuple[str, ...]:
        root_counts: Counter[str] = Counter()
        for shape in shapes:
            root_counts.update(
                (
                    name
                    for name in {*shape.declared_base_names, *shape.ancestor_names}
                    if name not in _IGNORED_ANCESTOR_NAMES and name != shape.class_name
                )
            )
        return sorted_tuple(
            (root_name for root_name, count in root_counts.items() if count >= 3)
        )

    def constructor_return_call(self, node: ast.FunctionDef) -> ast.Call | None:
        body = statements_without_docstring(node.body)
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            return None
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            return None
        if not isinstance(returned.func, ast.Name) or returned.func.id != "cls":
            return None
        return returned

    def expression_root_names(self, node: ast.AST) -> set[str]:
        roots: set[str] = set()

        class Visitor(ast.NodeVisitor):
            def visit_Attribute(self, node: ast.Attribute) -> None:
                current: ast.AST = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    roots.add(current.id)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                roots.add(node.id)

        Visitor().visit(node)
        return roots

    def concrete_detector_base_name(self, node: ast.ClassDef) -> str | None:
        if not any(
            (
                isinstance(statement, ast.Assign)
                and any(
                    (name_id(target) == "detector_id" for target in statement.targets)
                )
                for statement in node.body
            )
        ):
            return None
        base_names = tuple(
            (
                base_name
                for base_name in HELPER_SYNTAX_PROJECTION_AUTHORITY.class_base_names(
                    node
                )
                if base_name in _TYPED_CANDIDATE_DETECTOR_BASE_NAMES
            )
        )
        return single_item(base_names) if len(base_names) == 1 else None

    def if_returns_none_only(self, node: ast.If) -> bool:
        return bool(
            len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
            and isinstance(node.body[0].value, ast.Constant)
            and (node.body[0].value.value is None)
            and (not node.orelse)
        )

    def module_string_sequence_assignments(
        self, module: ParsedModule
    ) -> NamedStringSequenceSpecs:
        assignments: MutableNamedStringSequenceSpecs = []
        for binding in SUPPORT_PROJECTION_AUTHORITY.module_named_value_bindings(module):
            if not isinstance(binding.value, (ast.Tuple, ast.List)):
                continue
            string_items = tuple(
                (
                    item.value
                    for item in binding.value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
            )
            if len(string_items) != len(binding.value.elts) or len(string_items) < 3:
                continue
            assignments.append((binding.name, binding.line, string_items))
        return tuple(assignments)

    def queue_pop_target_name(self, statement: ast.stmt, queue_name: str) -> str | None:
        assignment_node = as_ast(statement, ast.Assign)
        assignment = named_call_assignment(assignment_node) if assignment_node else None
        pop_call = (
            attribute_call_match(
                assignment.call,
                method_name="pop",
                owner_type=ast.Name,
                owner_name=queue_name,
                single_argument_required=True,
            )
            if assignment is not None
            else None
        )
        if pop_call is None or constant_value(pop_call.single_argument) != 0:
            return None
        return assignment.target_name

    def result_append_args(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, result_name: str
    ) -> tuple[ast.AST, ...]:
        return tuple(
            (
                current.args[0]
                for current in _walk_nodes(node)
                if isinstance(current, ast.Call)
                and isinstance(current.func, ast.Attribute)
                and (current.func.attr == _APPEND_METHOD_NAME)
                and isinstance(current.func.value, ast.Name)
                and (current.func.value.id == result_name)
                and (len(current.args) == 1)
            )
        )

    def module_level_subject_parameter_name(
        self, qualname: str, function: NamedFunctionNode
    ) -> str | None:
        if "." in qualname or function.name.startswith("__"):
            return None
        if function.decorator_list:
            return None
        arguments = _FunctionSignatureView(function).arguments
        if not arguments:
            return None
        parameter_name = arguments[0].arg
        return (
            None
            if parameter_name in _IMPLICIT_METHOD_PARAMETER_NAMES
            else parameter_name
        )

    def shared_family_name(self, class_names: Sequence[str]) -> str | None:
        if not class_names:
            return None
        prefix = class_names[0]
        for name in class_names[1:]:
            while prefix and (not name.startswith(prefix)):
                prefix = prefix[:-1]
        return prefix or None

    def wrapper_delegate_symbol(
        self,
        node: ast.AST,
        *,
        class_name: str | None,
    ) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id in {"self", "cls"})
            and (class_name is not None)
        ):
            return f"{class_name}.{node.attr}"
        return None


HELPER_SUPPORT_PROJECTION_AUTHORITY = HelperSupportProjectionAuthority()


def _class_direct_constant_string_assignments(node: ast.ClassDef) -> dict[str, str]:
    return {
        name: string_value
        for name, value in CLASS_NODE_AUTHORITY.direct_assignments(node).items()
        if (string_value := _constant_string(value)) is not None
    }


def _function_parameter_annotation_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for arg in (
        tuple(node.args.posonlyargs)
        + tuple(node.args.args)
        + tuple(node.args.kwonlyargs)
    ):
        if arg.annotation is None:
            continue
        annotations[arg.arg] = ast.unparse(arg.annotation)
    return annotations


def _nominal_authority_shapes_without_ancestors(
    modules: Sequence[ParsedModule],
) -> tuple[NominalAuthorityShape, ...]:
    shapes_without_ancestors: list[NominalAuthorityShape] = []
    for module in modules:
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            field_type_map = HELPER_SYNTAX_PROJECTION_AUTHORITY.typed_field_map(node)
            shapes_without_ancestors.append(
                NominalAuthorityShape(
                    file_path=module.file_path,
                    class_name=node.name,
                    line=node.lineno,
                    declared_base_names=CLASS_NODE_AUTHORITY.declared_base_names(node),
                    ancestor_names=(),
                    field_names=tuple((name for name, _ in field_type_map)),
                    field_type_map=field_type_map,
                    method_names=sorted_tuple(
                        SYNTAX_PROJECTION_AUTHORITY.method_names(node)
                    ),
                    is_abstract=CLASS_NODE_AUTHORITY.is_abstract(node),
                    is_dataclass_family=_is_dataclass_class(node),
                )
            )

    return tuple(shapes_without_ancestors)


def _nominal_authority_shapes(
    modules: Sequence[ParsedModule],
) -> tuple[NominalAuthorityShape, ...]:
    shapes_without_ancestors = _nominal_authority_shapes_without_ancestors(modules)
    base_lookup: dict[str, set[str]] = defaultdict(set)
    for shape in shapes_without_ancestors:
        base_lookup[shape.class_name].update(shape.declared_base_names)
    ancestor_names_by_class = _class_ancestor_name_map(base_lookup)

    return tuple(
        (
            NominalAuthorityShape(
                file_path=shape.file_path,
                class_name=shape.class_name,
                line=shape.line,
                declared_base_names=shape.declared_base_names,
                ancestor_names=ancestor_names_by_class[shape.class_name],
                field_names=shape.field_names,
                field_type_map=shape.field_type_map,
                method_names=shape.method_names,
                is_abstract=shape.is_abstract,
                is_dataclass_family=shape.is_dataclass_family,
            )
            for shape in shapes_without_ancestors
        )
    )


class NominalAuthorityIndex:
    def __init__(self, modules: Sequence[ParsedModule]) -> None:
        self._shapes = _nominal_authority_shapes(modules)
        self._build_derived_indexes()

    def _build_derived_indexes(self) -> None:
        self._shapes_by_name: dict[str, list[NominalAuthorityShape]] = defaultdict(list)
        for shape in self._shapes:
            self._shapes_by_name[shape.class_name].append(shape)

    @classmethod
    def from_shapes(
        cls,
        shapes: Sequence[NominalAuthorityShape],
        *,
        base_lookup: dict[str, set[str]] | None = None,
    ) -> "NominalAuthorityIndex":
        active_base_lookup: dict[str, set[str]] = defaultdict(set)
        if base_lookup is None:
            for shape in shapes:
                active_base_lookup[shape.class_name].update(shape.declared_base_names)
        else:
            for class_name, base_names in base_lookup.items():
                active_base_lookup[class_name].update(base_names)
        ancestor_names_by_class = _class_ancestor_name_map(active_base_lookup)
        instance = cls.__new__(cls)
        instance._shapes = tuple(
            replace(
                shape,
                ancestor_names=ancestor_names_by_class[shape.class_name],
            )
            for shape in shapes
        )
        instance._build_derived_indexes()
        return instance

    def all_shapes(self) -> tuple[NominalAuthorityShape, ...]:
        return self._shapes

    def shapes_named(self, class_name: str) -> tuple[NominalAuthorityShape, ...]:
        return tuple(self._shapes_by_name.get(class_name, ()))


def _enum_key_family(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    return (node.value.id, node.attr)


def _fragmented_family_authority_candidates(
    module: ParsedModule,
) -> tuple[FragmentedFamilyAuthorityCandidate, ...]:
    family_maps: NamedStringSequenceSpecsByName = defaultdict(list)
    for binding in SUPPORT_PROJECTION_AUTHORITY.module_named_value_bindings(module):
        if not isinstance(binding.value, ast.Dict):
            continue
        key_pairs = tuple(
            (
                key_pair
                for key_pair in (
                    _enum_key_family(key)
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

    candidates: list[FragmentedFamilyAuthorityCandidate] = []
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
            FragmentedFamilyAuthorityCandidate(
                file_path=module.file_path,
                mapping_names=tuple((item[0] for item in ordered_entries)),
                line_numbers=tuple((item[1] for item in ordered_entries)),
                key_family_name=family_name,
                shared_keys=shared_keys,
                total_keys=total_keys,
            )
        )
    return tuple(candidates)


def _is_detectorish_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("Detector"):
        return True
    return bool(
        IssueDetector.detector_family_base_names()
        & set(CLASS_NODE_AUTHORITY.declared_base_names(node))
    )


def _finding_build_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for node in _walk_nodes(method):
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


def _call_display_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _build_call_keyword_helper_name(
    build_call: ast.Call, keyword_name: str
) -> str | None:
    for keyword in build_call.keywords:
        if keyword.arg != keyword_name or keyword.value is None:
            continue
        if isinstance(keyword.value, ast.Call):
            return _call_display_name(keyword.value)
    return None


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


def _finding_assembly_pipeline_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[FindingAssemblyPipelineCandidate]:
    if not _is_detectorish_class(node):
        return
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
    build_call = _finding_build_call(method)
    if build_call is None:
        return
    candidate_source_name = _candidate_source_name_from_method(method)
    if candidate_source_name is None:
        return
    metrics_type_name = _build_call_keyword_helper_name(build_call, "metrics")
    if metrics_type_name is None:
        return
    yield FindingAssemblyPipelineCandidate(
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


def _finding_assembly_pipeline_candidates(
    module: ParsedModule,
) -> tuple[FindingAssemblyPipelineCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _finding_assembly_pipeline_candidates_for_class,
    )


def _is_observation_spec_class(node: ast.ClassDef) -> bool:
    if node.name.endswith("ObservationSpec"):
        return True
    return bool(
        {
            "ObservationShapeSpec",
            "FunctionObservationSpec",
            "AssignObservationSpec",
            "ContextForwardingShapeSpec",
        }
        & set(CLASS_NODE_AUTHORITY.declared_base_names(node))
    )


def _delegate_name_from_return(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        outer_name = _call_display_name(node)
        if outer_name in _SEQUENCE_WRAPPER_CALL_NAMES and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call):
                return _call_display_name(inner)
        return outer_name
    return None


def _guard_role_name(node: ast.AST) -> str:
    return ScopedAstObservation.guard_role_name_from_text(ast.unparse(node))


def _scope_role_name(node: ast.AST) -> str:
    return ScopedAstObservation.scope_role_name_from_text(ast.unparse(node))


def _guarded_delegator_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[GuardedDelegatorCandidate]:
    if not _is_observation_spec_class(node) or CLASS_NODE_AUTHORITY.is_abstract(node):
        return
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
        if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
            continue
        delegate_name = _delegate_name_from_return(return_stmt.value)
        if delegate_name is None:
            continue
        yield GuardedDelegatorCandidate(
            file_path=module.file_path,
            line=statement.lineno,
            subject_name=node.name,
            name_family=(
                guard.test.__class__.__name__,
                delegate_name,
                _scope_role_name(guard.test),
            ),
            method_name=statement.name,
            guard_role=_guard_role_name(guard.test),
            delegate_name=delegate_name,
            scope_role=_scope_role_name(guard.test),
        )


def _guarded_delegator_candidates(
    module: ParsedModule,
) -> tuple[GuardedDelegatorCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _guarded_delegator_candidates_for_class,
    )


def _name_mentions(node: ast.AST, name: str) -> bool:
    return any(
        (
            isinstance(current, ast.Name) and current.id == name
            for current in _walk_nodes(node)
        )
    )


def _raised_exception_name(
    statement: ast.stmt,
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(statement, ast.Raise) or statement.exc is None:
        return None
    exc = statement.exc
    if isinstance(exc, ast.Call):
        exc_name = AstExpressionProjection.terminal_name(exc.func)
        referenced_names = sorted_tuple(
            {
                current.id
                for current in _walk_nodes(exc)
                if isinstance(current, ast.Name)
            }
        )
        if exc_name is not None:
            return (exc_name, referenced_names)
    exc_name = AstExpressionProjection.terminal_name(exc)
    if exc_name is not None:
        return (exc_name, ())
    return None


def _linear_query_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, tuple[str, ...], str, str] | None:
    body = statements_without_docstring(node.body)
    return (
        Maybe.of(_linear_query_loop(body))
        .combine(
            lambda loop: _linear_query_return_expr(
                loop, cast(ast.Name, loop.target).id
            ),
            lambda loop, return_expr: (loop, return_expr),
        )
        .combine(
            lambda _context: _linear_query_raised_exception(body),
            lambda context, raised: (context[0], context[1], raised),
        )
        .combine(
            lambda context: _linear_query_key_names(
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


def _linear_query_loop(body: list[ast.stmt]) -> ast.For | None:
    if len(body) < 2:
        return None
    loop = next(
        (statement for statement in body if isinstance(statement, ast.For)), None
    )
    if loop is None or not isinstance(loop.target, ast.Name):
        return None
    return loop


def _linear_query_return_expr(loop: ast.For, result_name: str) -> ast.AST | None:
    return_exprs = [
        current.value
        for current in _walk_nodes(loop)
        if isinstance(current, ast.Return) and current.value is not None
    ]
    if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
        return None
    return return_exprs[0]


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
                    for current in _walk_nodes(loop)
                )
            )
        )
    )
    return query_key_names


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


def _derived_query_index_candidates(
    module: ParsedModule,
) -> tuple[DerivedQueryIndexCandidate, ...]:
    grouped: DerivedQuerySpecsBySignature = defaultdict(list)
    for qualname, function in _iter_named_functions(module):
        signature = _linear_query_signature(function)
        if signature is None:
            continue
        source_expression, query_key_names, return_expression, exception_name = (
            signature
        )
        grouped[
            (
                _linear_query_source_owner(qualname, source_expression),
                source_expression,
                return_expression,
                exception_name,
            )
        ].append((qualname, function.lineno, query_key_names))
    candidates: list[DerivedQueryIndexCandidate] = []
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
            DerivedQueryIndexCandidate(
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


def _structural_observation_property_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[StructuralObservationPropertyCandidate]:
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not any(
            AstExpressionProjection.terminal_name(decorator) == "property"
            for decorator in statement.decorator_list
        ):
            continue
        body = statements_without_docstring(statement.body)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        returned = body[0].value
        if not isinstance(returned, ast.Call):
            continue
        constructor_name = AstExpressionProjection.terminal_name(returned.func)
        if constructor_name is None:
            continue
        keyword_names = sorted_tuple(
            keyword.arg for keyword in returned.keywords if keyword.arg is not None
        )
        if len(keyword_names) < 6:
            continue
        yield StructuralObservationPropertyCandidate(
            file_path=module.file_path,
            line=statement.lineno,
            subject_name=node.name,
            name_family=keyword_names,
            property_name=statement.name,
            constructor_name=constructor_name,
        )


def _structural_observation_property_candidates(
    module: ParsedModule,
) -> tuple[StructuralObservationPropertyCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _structural_observation_property_candidates_for_class,
    )


class MethodProjection:
    def aliases_to_self_attrs(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for subnode in _walk_nodes(method):
            binding = named_value_binding(subnode)
            if binding is None or binding.value is None:
                continue
            attr_name = None
            if (value_attr_name := _self_attr_name(binding.value)) is not None:
                attr_name = value_attr_name
            elif (value_name := name_id(binding.value)) is not None:
                attr_name = aliases.get(value_name)
            if attr_name is None:
                aliases.pop(binding.name, None)
                continue
            aliases[binding.name] = attr_name
        return aliases


METHOD_PROJECTION = MethodProjection()


def _is_projection_like_builder_value(value_fingerprint: str) -> bool:
    return value_fingerprint.startswith(("Name(", "Attribute(", "IfExp(", "Constant("))


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
                _is_projection_like_builder_value(value)
                for value in builder.value_fingerprint
            )
        ):
            continue
        grouped[(builder.callee_name, builder.field_names)].append(builder)
    candidates: list[tuple[BuilderCallShape, ...]] = []
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


def _projection_helper_groups(
    module: ParsedModule,
) -> tuple[tuple[ProjectionHelperShape, ...], ...]:
    shapes: tuple[ProjectionHelperShape, ...] = (
        CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
            module, ProjectionHelperObservationFamily, ProjectionHelperShape
        )
    )
    graph = ObservationGraph(tuple(shape.structural_observation for shape in shapes))
    lookup = _carrier_lookup(tuple(shapes))
    groups: list[tuple[ProjectionHelperShape, ...]] = []
    for fiber in graph.fibers_with_min_observations(
        ObservationKind.PROJECTION_HELPER,
        StructuralExecutionLevel.FUNCTION_BODY,
        minimum_observations=2,
    ):
        ordered = tuple(
            (
                _as_projection_helper_shape(item)
                for item in SUPPORT_PROJECTION_AUTHORITY.materialize_observations(
                    fiber.observations, lookup
                )
            )
        )
        attributes = {shape.projected_attribute for shape in ordered}
        if len(attributes) < 2:
            continue
        groups.append(ordered)
    return tuple(groups)


def _property_alias_hook_groups(
    module: ParsedModule,
) -> tuple[PropertyAliasHookGroup, ...]:
    grouped: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = tuple(
            (
                name
                for name in CLASS_NODE_AUTHORITY.declared_base_names(node)
                if name not in _IGNORED_BASE_NAMES
            )
        )
        if not base_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                (
                    AstExpressionProjection.terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            if len(statement.args.args) != 1:
                continue
            body = statements_without_docstring(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Name)
                and (returned.value.id == "self")
            ):
                continue
            for base_name in base_names:
                grouped[base_name, statement.name, returned.attr].append(
                    (node.name, statement.lineno)
                )
    return tuple(
        (
            PropertyAliasHookGroup(
                file_path=module.file_path,
                base_name=base_name,
                property_name=property_name,
                returned_attribute=returned_attribute,
                class_names=tuple((class_name for class_name, _ in ordered)),
                line_numbers=tuple((line for _, line in ordered)),
            )
            for (base_name, property_name, returned_attribute), items in sorted(
                grouped.items()
            )
            if len(items) >= 2
            for ordered in [sorted_tuple(items, key=lambda item: (item[1], item[0]))]
        )
    )


def _is_literal_constant_property_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_constant_property_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (
                key is not None
                and _is_literal_constant_property_value(key)
                and _is_literal_constant_property_value(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        )
    return False


def _constant_property_default_methods(
    module: ParsedModule, node: ast.ClassDef
) -> tuple[tuple[str, int, str, int], ...]:
    defaults: list[tuple[str, int, str, int]] = []
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if not any(
            (
                AstExpressionProjection.terminal_name(decorator) == "property"
                for decorator in statement.decorator_list
            )
        ):
            continue
        body = statements_without_docstring(statement.body)
        returned = single_item(body)
        if (
            not isinstance(returned, ast.Return)
            or returned.value is None
            or not _is_literal_constant_property_value(returned.value)
        ):
            continue
        defaults.append(
            (
                statement.name,
                statement.lineno,
                _source_segment(module, returned.value),
                (statement.end_lineno or statement.lineno) - statement.lineno + 1,
            )
        )
    return tuple(defaults)


def _constant_property_default_bundle_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[ConstantPropertyDefaultBundleCandidate]:
    defaults = _constant_property_default_methods(module, node)
    if len(defaults) < 4:
        return
    yield ConstantPropertyDefaultBundleCandidate(
        file_path=module.file_path,
        line=defaults[0][1],
        class_name=node.name,
        property_names=tuple((name for name, _, _, _ in defaults)),
        return_expressions=tuple((expression for _, _, expression, _ in defaults)),
        line_count=sum((line_count for _, _, _, line_count in defaults)),
    )


def _constant_property_default_bundle_candidates(
    module: ParsedModule,
) -> tuple[ConstantPropertyDefaultBundleCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _constant_property_default_bundle_candidates_for_class,
    )


def _reflective_self_attribute_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[ReflectiveSelfAttributeCandidate]:
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call):
                continue
            builtin_name = AstExpressionProjection.terminal_name(subnode.func)
            if builtin_name != _GETATTR_BUILTIN or len(subnode.args) != 2:
                continue
            receiver, attribute_name_node = subnode.args[0], subnode.args[1]
            attribute_name = _constant_string(attribute_name_node)
            if not (
                isinstance(receiver, ast.Name)
                and receiver.id == "self"
                and (attribute_name is not None)
            ):
                continue
            yield ReflectiveSelfAttributeCandidate(
                file_path=module.file_path,
                line=subnode.lineno,
                subject_name=node.name,
                name_family=(attribute_name,),
                method_name=statement.name,
                reflective_builtin=builtin_name,
                attribute_name=attribute_name,
            )


def _reflective_self_attribute_candidates(
    module: ParsedModule,
) -> tuple[ReflectiveSelfAttributeCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _reflective_self_attribute_candidates_for_class,
    )


def _guarded_wrapper_node_types(node: ast.If) -> tuple[str, ...] | None:
    test = node.test
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    operand = test.operand
    if (
        not isinstance(operand, ast.Call)
        or AstExpressionProjection.terminal_name(operand.func) != "isinstance"
        or len(operand.args) != 2
    ):
        return None
    type_node = operand.args[1]
    if isinstance(type_node, ast.Tuple):
        node_types = tuple(ast.unparse(item) for item in type_node.elts)
    else:
        node_types = (ast.unparse(type_node),)
    return tuple(item for item in node_types if item)


def _guarded_wrapper_function_candidates(
    module: ParsedModule,
) -> NamedStringSequenceSpecs:
    candidates: MutableNamedStringSequenceSpecs = []
    for statement in module.module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        body = statements_without_docstring(statement.body)
        while (
            body
            and isinstance(body[0], ast.Assign)
            and (len(body[0].targets) == 1)
            and isinstance(body[0].targets[0], ast.Name)
        ):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(
            guard, ast.If
        ) or not HELPER_SUPPORT_PROJECTION_AUTHORITY.if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
            continue
        node_types = _guarded_wrapper_node_types(guard)
        if not node_types:
            continue
        candidates.append((statement.name, statement.lineno, node_types))
    return tuple(candidates)


def _guarded_wrapper_spec_pairs(
    module: ParsedModule,
) -> tuple[GuardedWrapperSpecPair, ...]:
    wrapper_functions = {
        function_name: (lineno, node_types)
        for function_name, lineno, node_types in _guarded_wrapper_function_candidates(
            module
        )
    }
    pairs: list[GuardedWrapperSpecPair] = []
    for statement in module.module.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
            lineno = statement.lineno
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
            lineno = statement.lineno
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        constructor_name = AstExpressionProjection.terminal_name(value.func)
        if constructor_name is None:
            continue
        referenced_functions = [
            keyword.value.id
            for keyword in value.keywords
            if keyword.arg is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in wrapper_functions
        ]
        if len(referenced_functions) != 1:
            continue
        node_types_node = next(
            (
                keyword.value
                for keyword in value.keywords
                if keyword.arg == "node_types"
            ),
            None,
        )
        if node_types_node is None:
            continue
        if isinstance(node_types_node, ast.Tuple):
            node_types = tuple(ast.unparse(item) for item in node_types_node.elts)
        else:
            node_types = (ast.unparse(node_types_node),)
        function_name = referenced_functions[0]
        function_line, function_node_types = wrapper_functions[function_name]
        if tuple(node_types) != function_node_types:
            continue
        pairs.append(
            GuardedWrapperSpecPair(
                file_path=module.file_path,
                spec_name=target.id,
                spec_line=lineno,
                function_name=function_name,
                function_line=function_line,
                constructor_name=constructor_name,
                node_types=function_node_types,
            )
        )
    return tuple(pairs)


def _dynamic_self_field_selection_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[DynamicSelfFieldSelectionCandidate]:
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for subnode in _walk_nodes(statement):
            if not isinstance(subnode, ast.Call):
                continue
            builtin_name = AstExpressionProjection.terminal_name(subnode.func)
            if builtin_name not in _REFLECTIVE_SELF_BUILTINS:
                continue
            if len(subnode.args) < 2:
                continue
            receiver, selector_node = subnode.args[0], subnode.args[1]
            if not isinstance(receiver, ast.Name) or receiver.id != "self":
                continue
            if _constant_string(selector_node) is not None:
                continue
            selector_expression = ast.unparse(selector_node)
            if not any(
                token in selector_expression
                for token in ("self.", "type(self).", "cls.")
            ):
                continue
            yield DynamicSelfFieldSelectionCandidate(
                file_path=module.file_path,
                line=subnode.lineno,
                subject_name=node.name,
                name_family=(selector_expression,),
                method_name=statement.name,
                reflective_builtin=builtin_name,
                selector_expression=selector_expression,
            )


def _dynamic_self_field_selection_candidates(
    module: ParsedModule,
) -> tuple[DynamicSelfFieldSelectionCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _dynamic_self_field_selection_candidates_for_class,
    )


_SEMANTIC_INHERITANCE_IDENTITY_ATTR_SUFFIXES = (
    SemanticRoleIdentityToken.inheritance_identity_attr_suffixes()
)
_SEMANTIC_INHERITANCE_IDENTITY_ATTR_NAMES = (
    SemanticRoleIdentityToken.inheritance_identity_attr_names()
)


def _looks_like_semantic_key_attr(name: str) -> bool:
    normalized = name.lower()
    return normalized in _SEMANTIC_INHERITANCE_IDENTITY_ATTR_NAMES or any(
        (
            normalized.endswith(f"_{suffix}")
            for suffix in _SEMANTIC_INHERITANCE_IDENTITY_ATTR_SUFFIXES
        )
    )


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if name_id(decorator) == "dataclass":
            return True
        call = as_ast(decorator, ast.Call)
        if call is not None and name_id(call.func) == "dataclass":
            return True
    return False


def _autoregister_membership_object_count(
    *,
    concrete_class_names: tuple[str, ...],
    dynamic_factory_symbols: tuple[str, ...],
    behavior_method_names: tuple[str, ...],
    abstract_method_names: tuple[str, ...],
    registry_projection_names: tuple[str, ...],
    consumer_symbols: tuple[str, ...],
) -> int:
    leaf_objects = max(1, len(behavior_method_names)) + 2
    leaf_axis_count = len(concrete_class_names) + len(dynamic_factory_symbols)
    root_objects = (
        len(abstract_method_names)
        + len(registry_projection_names)
        + len(consumer_symbols)
        + 2
    )
    return leaf_axis_count * leaf_objects + root_objects


def _autoregister_derived_projection_count(
    *,
    registry_key_attr_name: str | None,
    key_extractor_name: str | None,
    behavior_method_names: tuple[str, ...],
    abstract_method_names: tuple[str, ...],
    registry_projection_names: tuple[str, ...],
    consumer_symbols: tuple[str, ...],
) -> int:
    projection_count = 1
    if registry_key_attr_name is not None:
        projection_count += 1
    if key_extractor_name is not None:
        projection_count += 1
    if behavior_method_names:
        projection_count += 1
    if abstract_method_names:
        projection_count += 1
    projection_count += len(registry_projection_names)
    if consumer_symbols:
        projection_count += 1
    return projection_count


def _autoregister_rent_certificate(
    *,
    manual_object_count: int,
    class_name: str,
    registry_axis_name: str,
    semantic_axis_names: tuple[str, ...],
    residual_object_count: int,
    independent_source_count: int,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=manual_object_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("autoregister_meta", "semantic_family_root"),
            per_axis_objects=("registered_leaf_key",),
        ),
        semantic_axes=(
            class_name,
            registry_axis_name,
            *semantic_axis_names,
        ),
        residual_object_count=residual_object_count,
        independent_source_count=independent_source_count,
    )


def _autoregister_missing_rent_signals(
    *,
    concrete_class_names: tuple[str, ...],
    dynamic_factory_symbols: tuple[str, ...],
    registry_key_attr_name: str | None,
    key_extractor_name: str | None,
    behavior_method_names: tuple[str, ...],
    abstract_method_names: tuple[str, ...],
    registry_projection_names: tuple[str, ...],
    consumer_symbols: tuple[str, ...],
    min_leaf_count: int,
) -> tuple[AutoRegisterMetaRentSignal, ...]:
    missing: list[AutoRegisterMetaRentSignal] = []
    if not dynamic_factory_symbols and len(concrete_class_names) < min_leaf_count:
        missing.append(AutoRegisterMetaRentSignal.REGISTERED_LEAF_AXIS)
    if registry_key_attr_name is None and key_extractor_name is None:
        missing.append(AutoRegisterMetaRentSignal.STABLE_KEY_AXIS)
    if not behavior_method_names and not abstract_method_names:
        missing.append(AutoRegisterMetaRentSignal.BEHAVIOR_CONTRACT)
    projection_rent_axes = (
        behavior_method_names,
        abstract_method_names,
        registry_projection_names,
        consumer_symbols,
    )
    if not any(projection_rent_axes):
        missing.append(
            AutoRegisterMetaRentSignal.EXPLICIT_REGISTRY_PROJECTION_OR_CONSUMER
        )
    return tuple(missing)


def _all_missing_axis_predicate_names(test: ast.AST) -> tuple[str, ...]:
    if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.And):
        return ()
    predicate_names: list[str] = []
    for value in test.values:
        if not (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.Not)
            and isinstance(value.operand, ast.Name)
        ):
            return ()
        predicate_names.append(value.operand.id)
    return tuple(predicate_names)


def _single_missing_signal_append(
    statements: list[ast.stmt],
) -> tuple[str, str] | None:
    if len(statements) != 1:
        return None
    statement = statements[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "append"
        and isinstance(statement.value.func.value, ast.Name)
    ):
        return None
    signal_name = _constant_string(single_item(tuple(statement.value.args)))
    if signal_name is None or statement.value.keywords:
        return None
    return statement.value.func.value.id, signal_name


def _all_missing_axis_predicate_for_if(
    module: ParsedModule, node: ast.If, function_name: str
) -> tuple[AllMissingAxisPredicateCandidate, ...]:
    predicate_names = _all_missing_axis_predicate_names(node.test)
    append_shape = _single_missing_signal_append(node.body)
    if len(predicate_names) < 3 or append_shape is None:
        return ()
    append_target_name, signal_name = append_shape
    return (
        AllMissingAxisPredicateCandidate(
            file_path=module.file_path,
            line=node.lineno,
            function_name=function_name,
            predicate_names=predicate_names,
            append_target_name=append_target_name,
            signal_name=signal_name,
            line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
        ),
    )


def _all_missing_axis_predicates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
) -> tuple[AllMissingAxisPredicateCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        function,
        ast.If,
        _all_missing_axis_predicate_for_if,
        qualname,
        traversal=walk_function_body_nodes,
    )


def _all_missing_axis_predicate_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[AllMissingAxisPredicateCandidate, ...]:
    del config
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module, _all_missing_axis_predicates_for_function
    )


def _reflective_lookup_shape(
    node: ast.AST,
) -> tuple[str, str, ast.AST] | None:
    if isinstance(node, ast.Call):
        builtin_name = AstExpressionProjection.terminal_name(node.func)
        if builtin_name == _GETATTR_BUILTIN and len(node.args) >= 2:
            selector_node = node.args[1]
            if _constant_string(selector_node) is None:
                return (_GETATTR_BUILTIN, ast.unparse(node.args[0]), selector_node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 1
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "__dict__"
        ):
            selector_node = node.args[0]
            if _constant_string(selector_node) is None:
                return ("dict.get", ast.unparse(node.func.value.value), selector_node)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and (node.value.func.id in {"globals", "locals"})
        and (not node.value.args)
        and (not node.value.keywords)
        and (_constant_string(node.slice) is None)
    ):
        return (f"{node.value.func.id}[]", f"{node.value.func.id}()", node.slice)
    return None


def _string_backed_reflective_nominal_lookup_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[StringBackedReflectiveNominalLookupCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    class_string_assignments = {
        class_name: _class_direct_constant_string_assignments(node)
        for class_name, node in class_defs_by_name.items()
    }
    candidate_map: dict[
        tuple[str, str, str, str, str], StringBackedReflectiveNominalLookupCandidate
    ] = {}
    for class_name, node in sorted(class_defs_by_name.items()):
        descendants = CLASS_INDEX_PROJECTION.descendant_names(
            class_defs_by_name, class_name
        )
        if len(descendants) < config.min_reflective_selector_values:
            continue
        for method in CLASS_NODE_AUTHORITY.methods(node):
            for subnode in _walk_nodes(method):
                lookup_shape = _reflective_lookup_shape(subnode)
                if lookup_shape is None:
                    continue
                lookup_kind, receiver_expression, selector_node = lookup_shape
                selector_attr_name = _selector_attribute_name(selector_node)
                if selector_attr_name is None:
                    continue
                concrete_class_names = tuple(
                    (
                        descendant
                        for descendant in descendants
                        if selector_attr_name in class_string_assignments[descendant]
                    )
                )
                if len(concrete_class_names) < config.min_reflective_selector_values:
                    continue
                selector_values = sorted_tuple(
                    {
                        class_string_assignments[descendant][selector_attr_name]
                        for descendant in concrete_class_names
                    }
                )
                if len(selector_values) < config.min_reflective_selector_values:
                    continue
                candidate = StringBackedReflectiveNominalLookupCandidate(
                    file_path=module.file_path,
                    line=subnode.lineno,
                    class_name=class_name,
                    method_name=method.name,
                    selector_attr_name=selector_attr_name,
                    lookup_kind=lookup_kind,
                    receiver_expression=receiver_expression,
                    concrete_class_names=concrete_class_names,
                    selector_values=selector_values,
                )
                candidate_map[
                    (
                        class_name,
                        method.name,
                        selector_attr_name,
                        lookup_kind,
                        receiver_expression,
                    )
                ] = candidate
    return sorted_tuple(
        candidate_map.values(),
        key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
    )


def _param_backed_name(expr: ast.AST, parameter_names: set[str]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in parameter_names:
        return expr.id
    if isinstance(expr, ast.IfExp):
        body_name = _param_backed_name(expr.body, parameter_names)
        orelse_name = _param_backed_name(expr.orelse, parameter_names)
        if body_name is not None and orelse_name is None:
            return body_name
        if orelse_name is not None and body_name is None:
            return orelse_name
        if body_name == orelse_name:
            return body_name
    if isinstance(expr, ast.BoolOp):
        names = {
            name
            for value in expr.values
            for name in (_param_backed_name(value, parameter_names),)
            if name is not None
        }
        if len(names) == 1:
            return next(iter(names))
    return None


def _class_init_concrete_param_backed_attrs(node: ast.ClassDef) -> dict[str, str]:
    init_method = CLASS_NODE_AUTHORITY.method_named(node, "__init__")
    if init_method is None:
        return {}
    parameter_type_names = {
        argument.arg: _annotation_type_names(argument.annotation)
        for argument in (
            tuple(init_method.args.posonlyargs)
            + tuple(init_method.args.args)
            + tuple(init_method.args.kwonlyargs)
        )
        if argument.annotation is not None
    }
    parameter_names = set(parameter_type_names)
    attr_type_names: dict[str, str] = {}
    for subnode in _walk_nodes(init_method):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
            target = subnode.targets[0]
            value = subnode.value
        elif isinstance(subnode, ast.AnnAssign):
            target = subnode.target
            value = subnode.value
        attr_name = None if target is None else _self_attr_name(target)
        if attr_name is None or value is None:
            continue
        param_name = _param_backed_name(value, parameter_names)
        if param_name is None:
            continue
        type_names = parameter_type_names.get(param_name, ())
        if len(type_names) != 1:
            continue
        attr_type_names.setdefault(attr_name, type_names[0])
    return attr_type_names


def _receiver_self_attr_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Attribute):
        return _self_attr_name(node)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def _concrete_config_field_probe_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[ConcreteConfigFieldProbeCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    config_field_names = {
        class_name: {
            field_name
            for field_name, _ in HELPER_SYNTAX_PROJECTION_AUTHORITY.typed_field_map(
                node
            )
        }
        for class_name, node in class_defs_by_name.items()
    }
    candidates: list[ConcreteConfigFieldProbeCandidate] = []
    for class_name, node in sorted(class_defs_by_name.items()):
        concrete_config_attrs = _class_init_concrete_param_backed_attrs(node)
        if not concrete_config_attrs:
            continue
        for method in CLASS_NODE_AUTHORITY.methods(node):
            aliases = METHOD_PROJECTION.aliases_to_self_attrs(method)
            grouped_missing_fields: dict[tuple[str, str], set[str]] = defaultdict(set)
            grouped_probe_builtins: dict[tuple[str, str], set[str]] = defaultdict(set)
            grouped_lines: dict[tuple[str, str], int] = {}
            for subnode in _walk_nodes(method):
                if not isinstance(subnode, ast.Call):
                    continue
                builtin_name = AstExpressionProjection.terminal_name(subnode.func)
                if (
                    builtin_name not in {_GETATTR_BUILTIN, _HASATTR_BUILTIN}
                    or len(subnode.args) < 2
                ):
                    continue
                probed_field_name = _constant_string(subnode.args[1])
                if probed_field_name is None:
                    continue
                config_attr_name = _receiver_self_attr_name(subnode.args[0], aliases)
                if config_attr_name is None:
                    continue
                config_type_name = concrete_config_attrs.get(config_attr_name)
                if config_type_name is None:
                    continue
                config_node = class_defs_by_name.get(config_type_name)
                if (
                    config_node is None
                    or CLASS_NODE_AUTHORITY.method_named(config_node, "__getattr__")
                    is not None
                ):
                    continue
                declared_field_names = config_field_names.get(config_type_name, set())
                if (
                    not declared_field_names
                    or probed_field_name in declared_field_names
                ):
                    continue
                key = (config_attr_name, config_type_name)
                grouped_missing_fields[key].add(probed_field_name)
                grouped_probe_builtins[key].add(builtin_name)
                grouped_lines.setdefault(key, subnode.lineno)
            for (config_attr_name, config_type_name), missing_fields in sorted(
                grouped_missing_fields.items()
            ):
                if len(missing_fields) < 2:
                    continue
                candidates.append(
                    ConcreteConfigFieldProbeCandidate(
                        file_path=module.file_path,
                        line=grouped_lines[config_attr_name, config_type_name],
                        class_name=class_name,
                        method_name=method.name,
                        config_attr_name=config_attr_name,
                        config_type_name=config_type_name,
                        missing_field_names=sorted_tuple(missing_fields),
                        probe_builtin_names=sorted_tuple(
                            grouped_probe_builtins[config_attr_name, config_type_name]
                        ),
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.class_name, item.method_name),
    )


_DECLARATIVE_FAMILY_DEFINITION_BASE_NAMES = frozenset(
    {
        "SingleShapeFamilyDefinition",
        "RegisteredShapeFamilyDefinition",
        "RegisteredObservationFamilyDefinition",
        "TypedLiteralObservationFamilyDefinition",
    }
)


def _is_declarative_class_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all((_is_declarative_class_value(item) for item in node.elts))
    if isinstance(node, ast.Dict):
        return all(
            (
                key is not None
                and _is_declarative_class_value(key)
                and _is_declarative_class_value(value)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        )
    if isinstance(node, ast.Call):
        return (
            _is_declarative_class_value(node.func)
            and all((_is_declarative_class_value(item) for item in node.args))
            and all(
                (
                    keyword.arg is not None
                    and _is_declarative_class_value(keyword.value)
                    for keyword in node.keywords
                )
            )
        )
    if isinstance(node, ast.Subscript):
        return _is_declarative_class_value(node.value) and _is_declarative_class_value(
            node.slice
        )
    if isinstance(node, ast.UnaryOp):
        return _is_declarative_class_value(node.operand)
    return False


_ABC_BASE_NAME = "ABC"
_COMPOSABLE_BASE_NAME_SUFFIXES = (
    _ABC_BASE_NAME,
    "Base",
    "Carrier",
    "Contract",
    "Mixin",
    "Template",
)


def _is_composable_base_name(base_name: str) -> bool:
    return base_name == _ABC_BASE_NAME or base_name.endswith(
        _COMPOSABLE_BASE_NAME_SUFFIXES
    )


def _declared_base_name_sequence(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        (
            base_name
            for base_name in (
                AstExpressionProjection.terminal_name(base) for base in node.bases
            )
            if base_name is not None
        )
    )


def _contiguous_base_bundles(
    base_names: tuple[str, ...],
    *,
    minimum_width: int = 3,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            base_names[start:end]
            for start in range(len(base_names))
            for end in range(start + minimum_width, len(base_names) + 1)
            if all(_is_composable_base_name(name) for name in base_names[start:end])
        )
    )


def _is_contiguous_subtuple(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _maximal_repeated_base_bundle_items(
    grouped: BaseBundleClassGroups,
) -> tuple[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]], ...]:
    qualified = tuple(
        (
            (bundle, sorted_tuple(nodes, key=lambda node: node.lineno))
            for bundle, nodes in grouped.items()
            if len(nodes) >= 3
        )
    )
    maximal: list[tuple[tuple[str, ...], tuple[ast.ClassDef, ...]]] = []
    for bundle, nodes in sorted(
        qualified,
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    ):
        class_names = {node.name for node in nodes}
        if any(
            (
                class_names <= {node.name for node in existing_nodes}
                and _is_contiguous_subtuple(bundle, existing_bundle)
                for existing_bundle, existing_nodes in maximal
            )
        ):
            continue
        maximal.append((bundle, nodes))
    return tuple(maximal)


def _repeated_base_bundle_candidates(
    module: ParsedModule,
) -> tuple[RepeatedBaseBundleCandidate, ...]:
    grouped: BaseBundleClassGroups = defaultdict(list)
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        for bundle in _contiguous_base_bundles(_declared_base_name_sequence(node)):
            grouped[bundle].append(node)
    candidates: list[RepeatedBaseBundleCandidate] = []
    for bundle, nodes in _maximal_repeated_base_bundle_items(grouped):
        line_numbers = tuple((node.lineno for node in nodes))
        candidates.append(
            RepeatedBaseBundleCandidate(
                file_path=module.file_path,
                class_names=tuple((node.name for node in nodes)),
                line_numbers=line_numbers,
                base_names=bundle,
                bundle_width=len(bundle),
                class_count=len(nodes),
                line_count=sum(
                    (
                        (node.end_lineno or node.lineno) - node.lineno + 1
                        for node in nodes
                    )
                ),
            )
        )
    return tuple(candidates)


def _is_simple_classvar_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return True
    if isinstance(node, ast.Tuple):
        return all((_is_simple_classvar_value(item) for item in node.elts))
    return False


def _dict_key_kind(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return "type_name"
    if isinstance(value, ast.Attribute):
        return "enum_member"
    if isinstance(value, ast.Constant):
        return type(value.value).__name__
    return None


def _derived_indexed_surface_candidates(
    module: ParsedModule,
) -> tuple[DerivedIndexedSurfaceCandidate, ...]:
    index = NominalAuthorityIndex((module,))
    candidates: list[DerivedIndexedSurfaceCandidate] = []
    for statement in statements_without_docstring(module.module.body):
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 3 or len(value.keys) != len(value.values):
            continue
        key_kinds = {
            key_kind
            for key_kind in (
                _dict_key_kind(key) for key in value.keys if key is not None
            )
            if key_kind is not None
        }
        if len(key_kinds) != 1:
            continue
        value_names = tuple(
            (
                item.id
                for item in value.values
                if isinstance(item, ast.Name)
                and (shapes := index.shapes_named(item.id))
                and (shapes[0].file_path == module.file_path)
            )
        )
        if len(value_names) != len(value.values):
            continue
        local_shapes = [index.shapes_named(value_name)[0] for value_name in value_names]
        shared_roots = HELPER_SUPPORT_PROJECTION_AUTHORITY.derivable_nominal_root_names(
            local_shapes
        )
        if not shared_roots:
            continue
        candidates.append(
            DerivedIndexedSurfaceCandidate(
                file_path=module.file_path,
                surface_name=target_name,
                line=statement.lineno,
                key_kind=next(iter(key_kinds)),
                value_names=value_names,
                derivable_root_names=shared_roots,
            )
        )
    return tuple(candidates)


def _registered_surface_roots(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    calls: list[ast.Call] = []

    def collect_calls(current: ast.AST) -> bool:
        binary = as_ast(current, ast.BinOp)
        if binary is not None and isinstance(binary.op, ast.Add):
            return collect_calls(binary.left) and collect_calls(binary.right)
        call = as_ast(current, ast.Call)
        if call is None:
            return False
        calls.append(call)
        return True

    if not collect_calls(node) or len(calls) < 2:
        return None
    accessors = tuple(
        (
            accessor
            for call in calls
            if (
                accessor := attribute_call_match(
                    call, owner_type=ast.Name, argument_count=0, allow_keywords=False
                )
            )
            is not None
        )
    )
    accessor_names = {accessor.attribute.attr for accessor in accessors}
    if len(accessor_names) != 1:
        return None
    accessor_name = next(iter(accessor_names))
    root_names = sorted_tuple((accessor.owner.id for accessor in accessors))
    return (accessor_name, root_names)


@dataclass(frozen=True)
class _RegisteredUnionSurfaceSource:
    owner_name: str
    value: ast.AST
    line: int

    @classmethod
    def from_node(cls, node: ast.AST) -> "_RegisteredUnionSurfaceSource | None":
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in statements_without_docstring(node.body):
                if isinstance(statement, ast.For):
                    return cls(node.name, statement.iter, statement.lineno)
                if isinstance(statement, ast.Assign):
                    return cls(node.name, statement.value, statement.lineno)
            return None
        if not isinstance(node, ast.Assign):
            return None
        target_name = name_id(single_item(node.targets))
        return (
            None if target_name is None else cls(target_name, node.value, node.lineno)
        )


def _registered_union_surface_candidates_for_node(
    module: ParsedModule,
    node: ast.AST,
    class_defs_by_name: dict[str, ast.ClassDef],
) -> Iterable[RegisteredUnionSurfaceCandidate]:
    source = _RegisteredUnionSurfaceSource.from_node(node)
    if source is None:
        return
    registered_surface = _registered_surface_roots(source.value)
    if registered_surface is None:
        return
    accessor_name, root_names = registered_surface
    if len(root_names) < 2:
        return
    root_nodes = [class_defs_by_name.get(root_name) for root_name in root_names]
    if any((root_node is None for root_node in root_nodes)):
        return
    if any(
        (
            (
                method := CLASS_NODE_AUTHORITY.method_named(
                    cast(ast.ClassDef, root_node), accessor_name
                )
            )
            is None
            or not _is_classmethod(method)
            for root_node in root_nodes
        )
    ):
        return
    yield RegisteredUnionSurfaceCandidate(
        file_path=module.file_path,
        line=source.line,
        owner_name=source.owner_name,
        accessor_name=accessor_name,
        root_names=root_names,
    )


def _registered_union_surface_candidates(
    module: ParsedModule,
) -> tuple[RegisteredUnionSurfaceCandidate, ...]:
    class_defs_by_name = {
        node.name: node for node in module.module.body if isinstance(node, ast.ClassDef)
    }
    source_nodes: tuple[ast.AST, ...] = (
        *(function for _, function in _iter_named_functions(module)),
        *_typed_ast_nodes(module.module, ast.Assign),
    )
    return tuple(
        candidate
        for node in source_nodes
        for candidate in _registered_union_surface_candidates_for_node(
            module, node, class_defs_by_name
        )
    )


_CLASS_OBJECT_TYPE_ANNOTATION_ROOTS = frozenset({"type", "Type"})
_CONCRETE_UNION_CONTRACT_IGNORED_BASE_NAMES = frozenset(
    {"ABC", "ABCMeta", "Generic", "Protocol", "object"}
)
_CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES = (
    "build",
    "create",
    "from_",
    "make",
)


def _annotation_union_members(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            *_annotation_union_members(node.left),
            *_annotation_union_members(node.right),
        )
    if isinstance(node, ast.Subscript):
        root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.value)
        if root_name != "Union":
            return (node,)
        if isinstance(node.slice, ast.Tuple):
            return tuple(node.slice.elts)
        if isinstance(node.slice, ast.List):
            return tuple(node.slice.elts)
    return (node,)


def _class_object_type_argument_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.value)
    if root_name not in _CLASS_OBJECT_TYPE_ANNOTATION_ROOTS:
        return None
    return HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(node.slice)


def _concrete_class_object_union_member_names(
    annotation: ast.AST | None,
) -> tuple[str, ...]:
    if annotation is None:
        return ()
    union_members = _annotation_union_members(annotation)
    member_names = tuple(
        (
            member_name
            for member in union_members
            if (member_name := _class_object_type_argument_name(member)) is not None
        )
    )
    if len(member_names) != len(union_members):
        return ()
    return tuple(dict.fromkeys(member_names))


def _parameter_class_call_attribute_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_name: str,
) -> tuple[str, ...]:
    attribute_names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == parameter_name
            ):
                attribute_names.add(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(
        ast.Module(body=statements_without_docstring(function.body), type_ignores=[])
    )
    return sorted_tuple(attribute_names)


def _class_def_declared_and_ancestor_names(
    class_defs_by_name: dict[str, ast.ClassDef],
    ancestor_names_by_class: dict[str, tuple[str, ...]],
    class_name: str,
) -> set[str]:
    node = class_defs_by_name[class_name]
    return set(CLASS_NODE_AUTHORITY.declared_base_names(node)) | set(
        ancestor_names_by_class[class_name]
    )


def _base_defines_contract_attributes(
    class_defs_by_name: dict[str, ast.ClassDef],
    base_name: str,
    observed_attribute_names: tuple[str, ...],
) -> bool:
    base_node = class_defs_by_name.get(base_name)
    if base_node is None:
        return False
    return all(
        (
            CLASS_NODE_AUTHORITY.method_named(base_node, attribute_name) is not None
            for attribute_name in observed_attribute_names
        )
    )


def _common_contract_base_names(
    class_defs_by_name: dict[str, ast.ClassDef],
    ancestor_names_by_class: dict[str, tuple[str, ...]],
    member_type_names: tuple[str, ...],
    observed_attribute_names: tuple[str, ...],
) -> tuple[str, ...]:
    base_sets = tuple(
        (
            _class_def_declared_and_ancestor_names(
                class_defs_by_name, ancestor_names_by_class, member_type_name
            )
            for member_type_name in member_type_names
        )
    )
    common_base_names = (
        set.intersection(*base_sets) - _CONCRETE_UNION_CONTRACT_IGNORED_BASE_NAMES
    )
    return sorted_tuple(
        (
            base_name
            for base_name in common_base_names
            if _base_defines_contract_attributes(
                class_defs_by_name, base_name, observed_attribute_names
            )
        )
    )


def _shared_edge_tokens(member_type_names: tuple[str, ...]) -> tuple[str, ...]:
    token_rows = tuple(
        (
            CLASS_NAME_ALGEBRA.ordered_tokens(member_type_name)
            for member_type_name in member_type_names
        )
    )
    if not token_rows or not all(token_rows):
        return ()
    prefix_tokens: list[str] = []
    for column in zip(*token_rows, strict=False):
        if len(set(column)) != 1:
            break
        prefix_tokens.append(column[0])
    suffix_tokens: list[str] = []
    reversed_rows = tuple(tuple(reversed(row)) for row in token_rows)
    for column in zip(*reversed_rows, strict=False):
        if len(set(column)) != 1:
            break
        suffix_tokens.append(column[0])
    return (*prefix_tokens, *tuple(reversed(suffix_tokens)))


def _concrete_union_contract_name(
    member_type_names: tuple[str, ...],
    observed_attribute_names: tuple[str, ...],
) -> str:
    edge_tokens = _shared_edge_tokens(member_type_names)
    base_name = (
        "".join((token.title() for token in edge_tokens))
        if edge_tokens
        else f"{member_type_names[0]}Family"
    )
    role_suffix = (
        "Factory"
        if any(
            (
                attribute_name.startswith(_CONCRETE_UNION_FACTORY_ATTRIBUTE_PREFIXES)
                for attribute_name in observed_attribute_names
            )
        )
        else "Contract"
    )
    return base_name if base_name.endswith(role_suffix) else f"{base_name}{role_suffix}"


def _concrete_type_union_contract_candidates_for_function(
    module: ParsedModule,
    function_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    class_defs_by_name: dict[str, ast.ClassDef],
    ancestor_names_by_class: dict[str, tuple[str, ...]],
) -> Iterable[ConcreteTypeUnionContractCandidate]:
    arguments = (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    )
    for argument in arguments:
        if argument.arg in _IMPLICIT_METHOD_PARAMETER_NAMES:
            continue
        member_type_names = _concrete_class_object_union_member_names(
            argument.annotation
        )
        if len(member_type_names) < 2 or any(
            member_type_name not in class_defs_by_name
            for member_type_name in member_type_names
        ):
            continue
        observed_attribute_names = _parameter_class_call_attribute_names(
            function, argument.arg
        )
        if not observed_attribute_names:
            continue
        common_base_names = _common_contract_base_names(
            class_defs_by_name,
            ancestor_names_by_class,
            member_type_names,
            observed_attribute_names,
        )
        yield ConcreteTypeUnionContractCandidate(
            file_path=module.file_path,
            line=argument.lineno,
            function_name=function_name,
            parameter_name=argument.arg,
            member_type_names=member_type_names,
            observed_attribute_names=observed_attribute_names,
            suggested_contract_name=(
                common_base_names[0]
                if common_base_names
                else _concrete_union_contract_name(
                    member_type_names, observed_attribute_names
                )
            ),
            common_base_names=common_base_names,
        )


def _concrete_type_union_contract_candidates(
    module: ParsedModule,
) -> tuple[ConcreteTypeUnionContractCandidate, ...]:
    class_defs_by_name = _module_class_defs_by_name(module)
    base_lookup = {
        class_name: set(CLASS_NODE_AUTHORITY.declared_base_names(node))
        for class_name, node in class_defs_by_name.items()
    }
    ancestor_names_by_class = _class_ancestor_name_map(base_lookup)
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _concrete_type_union_contract_candidates_for_function,
        class_defs_by_name,
        ancestor_names_by_class,
        sort_key=lambda item: (item.file_path, item.line, item.function_name),
    )


def _export_policy_role_names(node: ast.FunctionDef) -> tuple[str, ...]:
    body_text = "\n".join(ast.unparse(statement) for statement in node.body)
    roles: set[str] = set()
    if "name.startswith('_')" in body_text:
        roles.add("exclude_private")
    if (
        "__module__ != __name__" in body_text
        or "getattr(value, '__module__', None) == __name__" in body_text
    ):
        roles.add("module_local")
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        call_name = AstExpressionProjection.terminal_name(current.func)
        if call_name == "isinstance" and len(current.args) == 2:
            type_names = set(
                HELPER_SYNTAX_PROJECTION_AUTHORITY.type_name_set(current.args[1])
            )
            if _TYPE_NAME_LITERAL in type_names:
                roles.add("type_only")
                type_names.discard(_TYPE_NAME_LITERAL)
            elif type_names:
                roles.add("value_type_filter")
            if any((type_name.endswith("Enum") for type_name in type_names)):
                roles.add("enum_ok")
        elif call_name == "callable" and len(current.args) == 1:
            roles.add("callable_ok")
        elif call_name == "issubclass" and len(current.args) == 2:
            roles.add("subclass_constraint")
            type_names = set(
                HELPER_SYNTAX_PROJECTION_AUTHORITY.type_name_set(current.args[1])
            )
            if any((type_name.endswith("Enum") for type_name in type_names)):
                roles.add("enum_ok")
        elif call_name == "isabstract":
            roles.add("exclude_abstract")
    return sorted_tuple(roles)


def _export_policy_root_type_names(node: ast.FunctionDef) -> tuple[str, ...]:
    root_type_names: set[str] = set()
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        if (
            AstExpressionProjection.terminal_name(current.func) != "issubclass"
            or len(current.args) != 2
        ):
            continue
        root_type_names.update(
            (
                type_name
                for type_name in HELPER_SYNTAX_PROJECTION_AUTHORITY.type_name_set(
                    current.args[1]
                )
                if type_name != _TYPE_NAME_LITERAL
            )
        )
    return sorted_tuple(root_type_names)


def _module_function_named(
    module: ParsedModule, function_name: str
) -> ast.FunctionDef | None:
    return next(
        (
            statement
            for statement in statements_without_docstring(module.module.body)
            if isinstance(statement, ast.FunctionDef)
            and statement.name == function_name
        ),
        None,
    )


def _export_all_assignment_value(statement: ast.stmt) -> ast.AST | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is None or name_id(single_item(assignment.targets)) != "__all__":
        return None
    return assignment.value


def _sorted_generator_arg(value: ast.AST) -> ast.GeneratorExp | None:
    call = as_ast(value, ast.Call)
    if (
        call is None
        or AstExpressionProjection.terminal_name(call.func) != "sorted"
    ):
        return None
    return single_ast(call.args, ast.GeneratorExp)


def _single_generator_filter_call(generator: ast.GeneratorExp) -> ast.Call | None:
    comprehension = single_item(generator.generators)
    if comprehension is None:
        return None
    return as_ast(single_item(comprehension.ifs), ast.Call)


def _export_all_predicate_name(statement: ast.stmt) -> str | None:
    return (
        Maybe.of(_export_all_assignment_value(statement))
        .project(_sorted_generator_arg)
        .project(_single_generator_filter_call)
        .project(lambda condition: name_id(condition.func))
        .unwrap_or_none()
    )


def _module_exported_predicate_names(module: ParsedModule) -> frozenset[str]:
    return frozenset(
        (
            predicate_name
            for statement in statements_without_docstring(module.module.body)
            if (predicate_name := _export_all_predicate_name(statement)) is not None
        )
    )


def _module_export_policy_predicate_candidate(
    module: ParsedModule,
) -> ExportPolicyPredicateCandidate | None:
    return (
        Maybe.of(single_item(tuple(_module_exported_predicate_names(module))))
        .combine(
            lambda predicate_name: _module_function_named(module, predicate_name),
            lambda predicate_name, predicate_node: (predicate_name, predicate_node),
        )
        .filter(lambda context: len(context[1].args.args) == 2)
        .combine(
            lambda context: _export_policy_role_names(context[1]),
            lambda context, role_names: (context[0], context[1], role_names),
        )
        .filter(lambda context: len(context[2]) >= 2)
        .map(
            lambda context: ExportPolicyPredicateCandidate(
                file_path=module.file_path,
                line=context[1].lineno,
                subject_name=context[0],
                name_family=context[2],
                role_names=context[2],
                root_type_names=_export_policy_root_type_names(context[1]),
            )
        )
        .unwrap_or_none()
    )


def _returned_sequence_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    for current in _walk_nodes(node):
        if not isinstance(current, ast.Return) or current.value is None:
            continue
        value = current.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "tuple"
            and len(value.args) == 1
        ):
            inner = value.args[0]
            if isinstance(inner, ast.Name):
                return inner.id
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and (inner.func.id == "sorted")
                and inner.args
                and isinstance(inner.args[0], ast.Name)
            ):
                return inner.args[0].id
    return None


class SubclassFilterNameRule(AstPredicateRule[str, ast.AST, str]):
    """Rule family for names used to filter subclass traversal members."""


class NamedSubclassFilterCallRule(SubclassFilterNameRule):
    node_type = ast.Call

    def project_ast(self, node: ast.Call, current_name: str) -> str | None:
        if name_id(node.func) is None:
            return None
        if not any(name_id(argument) == current_name for argument in node.args):
            return None
        return name_id(node.func)


class DictBackedSubclassFilterRule(SubclassFilterNameRule):
    node_type = ast.Call

    def project_ast(self, node: ast.Call, current_name: str) -> str | None:
        match = attribute_call_match(
            node,
            method_name="get",
            owner_type=ast.Attribute,
            argument_count=1,
        )
        if match is None:
            return None
        if match.owner.attr != "__dict__" or name_id(match.owner.value) != current_name:
            return None
        attribute_name = constant_value(match.single_argument)
        return attribute_name if isinstance(attribute_name, str) else None


class SubclassLoopRule(AstPredicateRule[str, ast.AST, str]):
    """Rule family for loop shapes that advance subclass traversal queues."""


class QueueExtendingSubclassLoopRule(SubclassLoopRule):
    node_type = ast.While

    def project_ast(self, node: ast.While, queue_name: str) -> str | None:
        current_name: str | None = None
        extends_queue = False
        for body_statement in node.body:
            current_name = (
                current_name
                or HELPER_SUPPORT_PROJECTION_AUTHORITY.queue_pop_target_name(
                    body_statement, queue_name
                )
            )
            if (
                current_name is not None
                and HELPER_SYNTAX_PROJECTION_AUTHORITY.extends_subclasses_queue(
                    body_statement, queue_name, current_name
                )
            ):
                extends_queue = True
        return current_name if extends_queue else None


def _registry_attribute_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            attribute_name
            for current in _walk_nodes(node)
            if isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and (current.func.attr == "get")
            and isinstance(current.func.value, ast.Attribute)
            and (current.func.value.attr == "__dict__")
            and isinstance(current.func.value.value, ast.Name)
            and (len(current.args) == 1)
            and ((attribute_name := _constant_string(current.args[0])) is not None)
        }
    )


def _registry_traversal_group_from_sites(
    projected_sites: Sequence[SubclassTraversalSite],
) -> SubclassTraversalGroup | None:
    sites = sorted_tuple(
        projected_sites,
        key=lambda item: (item.file_path, item.line, item.symbol),
    )
    if len(sites) < 2:
        return None
    return SubclassTraversalGroup(
        symbols=tuple((site.symbol for site in sites)),
        file_paths=tuple((site.file_path for site in sites)),
        line_numbers=tuple((site.line for site in sites)),
        root_expressions=tuple((site.root_expression for site in sites)),
        materialization_kinds=tuple((site.materialization_kind for site in sites)),
        registry_attribute_names=sorted_tuple(
            {
                attribute_name
                for site in sites
                for attribute_name in site.registry_attribute_names
            }
        ),
        filter_names=sorted_tuple(
            {filter_name for site in sites for filter_name in site.filter_names}
        ),
    )


class SubclassTraversalSiteFamily(CollectedFamily[SubclassTraversalSite]):
    """Persist compact subclass-walker facts for repository-wide grouping."""

    item_type = SubclassTraversalSite
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(
        lambda source_module, syntax_index: _native_subclass_traversal_sites(
            source_module,
            syntax_index,
        )
    )

    @staticmethod
    def _seed(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[str, str] | None:
        for statement in statements_without_docstring(node.body):
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                continue
            root_expression = (
                HELPER_SYNTAX_PROJECTION_AUTHORITY.subclasses_root_expression(
                    statement.value
                )
            )
            if root_expression is not None:
                return statement.targets[0].id, root_expression
        return None

    @classmethod
    def site(
        cls,
        module: ParsedModule,
        qualname: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SubclassTraversalSite | None:
        seed = cls._seed(node)
        if seed is None:
            return None
        current_name = SubclassLoopRule.first_match_anywhere(node, seed[0])
        if current_name is None:
            return None
        result_name = _returned_sequence_name(node)
        if result_name is None:
            return None
        append_arguments = HELPER_SUPPORT_PROJECTION_AUTHORITY.result_append_args(
            node, result_name
        )
        if len(append_arguments) != 1:
            return None
        return SubclassTraversalSite(
            file_path=module.file_path,
            line=node.lineno,
            symbol=qualname,
            root_expression=seed[1],
            materialization_kind=SubclassMaterializationKind.from_append_argument(
                append_arguments[0]
            ),
            registry_attribute_names=_registry_attribute_names(node),
            filter_names=sorted_tuple(
                set(SubclassFilterNameRule.matches_anywhere(node, current_name))
            ),
        )

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[SubclassTraversalSite]:
        return [
            site
            for qualname, function in _iter_named_functions(parsed_module)
            if (
                site := cls.site(
                    parsed_module,
                    qualname,
                    cast(ast.FunctionDef | ast.AsyncFunctionDef, function),
                )
            )
            is not None
        ]


def _native_subclass_traversal_sites(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[SubclassTraversalSite] | None:
    """Project subclass walkers from native-selected function fragments."""

    if not syntax_index.is_complete:
        return None
    parsed_module = source_module.parsed_module(
        ast.Module(body=[], type_ignores=[]),
    )
    sites: list[SubclassTraversalSite] = []
    try:
        for function_node in sorted(
            syntax_index.common_captures().get("function", ()),
            key=lambda node: (node.start_byte, -node.end_byte),
        ):
            if b"__subclasses__" not in syntax_index.source_for(function_node):
                continue
            function = syntax_index.function_for(function_node)
            site = SubclassTraversalSiteFamily.site(
                parsed_module,
                syntax_index.class_qualified_function_name(function_node),
                function,
            )
            if site is not None:
                sites.append(site)
        return sites
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _source_type_name_for_constructor(node: ast.FunctionDef) -> str | None:
    if len(node.args.args) < 3:
        return None
    source_arg = node.args.args[2]
    if source_arg.annotation is None:
        return source_arg.arg
    return ast.unparse(source_arg.annotation)


def _alternate_constructor_family_groups(
    module: ParsedModule,
) -> tuple[AlternateConstructorFamilyGroup, ...]:
    groups: list[AlternateConstructorFamilyGroup] = []
    for node in _walk_nodes(module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        constructor_methods: list[tuple[ast.FunctionDef, ast.Call, str]] = []
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not statement.name.startswith("from_") or not _is_classmethod(statement):
                continue
            return_call = HELPER_SUPPORT_PROJECTION_AUTHORITY.constructor_return_call(
                statement
            )
            if return_call is None:
                continue
            source_type_name = _source_type_name_for_constructor(statement)
            if source_type_name is None:
                continue
            constructor_methods.append((statement, return_call, source_type_name))
        if len(constructor_methods) < 3:
            continue
        keyword_sets = [
            {keyword.arg for keyword in call.keywords if keyword.arg is not None}
            for _, call, _ in constructor_methods
        ]
        shared_keyword_names = sorted_tuple(
            (str(item) for item in set.intersection(*keyword_sets))
        )
        if len(shared_keyword_names) < 4:
            continue
        groups.append(
            AlternateConstructorFamilyGroup(
                file_path=module.file_path,
                class_name=node.name,
                method_names=tuple(
                    (method.name for method, _, _ in constructor_methods)
                ),
                line_numbers=tuple(
                    (method.lineno for method, _, _ in constructor_methods)
                ),
                keyword_names=shared_keyword_names,
                source_type_names=tuple(
                    (source_type_name for _, _, source_type_name in constructor_methods)
                ),
            )
        )
    return tuple(groups)


_MethodFamilyKey: TypeAlias = tuple[str, tuple[str, ...]]
_MethodFamilyMethodNamesByFamily: TypeAlias = dict[_MethodFamilyKey, set[str]]
_MethodFamilyAxisSpecsByFamily: TypeAlias = dict[_MethodFamilyKey, set[str]]
_MethodFamilyAxisRow: TypeAlias = tuple[str, tuple[str, ...]]
_MethodFamilyAxisRowsByFamily: TypeAlias = dict[
    _MethodFamilyKey, set[_MethodFamilyAxisRow]
]
_MethodFamilyLatticeEdgesByFamily: TypeAlias = dict[
    _MethodFamilyKey, set[tuple[tuple[str, ...], tuple[str, ...]]]
]
_MethodFamilyMethodPlanKey: TypeAlias = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class _MethodFamilyMethodGroupProfile(ResidueHookNamesCarrier):
    shared_statement_count: int
    varying_coordinates: tuple[CompactMethodSemanticCoordinate, ...]
    compression_certificate: CompressionCertificate


@dataclass(frozen=True)
class _MethodFamilyMethodPlan(ClassFamilyWitnessCarrier):
    base_symbol: str
    method_name: str
    profile: _MethodFamilyMethodGroupProfile
    line_numbers: tuple[int, ...]
    line_count: int


@dataclass(frozen=True)
class _MethodFamilyResidueEvidence(ResidueHookNamesCarrier):
    leaf_residue_names: tuple[str, ...]
    residue_declaration_count: int
    shared_to_residue_ratio: float


@dataclass(frozen=True)
class _MethodFamilyMethodSurface:
    base_name: str
    method_names: tuple[str, ...]


@dataclass(frozen=True)
class _MethodFamilyEvidence(
    _MethodFamilyMethodSurface,
    ResidueHookNamesCarrier,
    MethodFamilyRelationSpecCarrier,
    MethodFamilyLatticeMetricsCarrier,
    MethodFamilyResidueEvidenceCarrier,
):
    class_names: tuple[str, ...]


_MethodFamilyCandidateT = TypeVar("_MethodFamilyCandidateT")
_MethodFamilyCandidateBuilder: TypeAlias = Callable[
    [tuple[_MethodFamilyMethodPlan, ...], _MethodFamilyEvidence],
    _MethodFamilyCandidateT | None,
]


def _method_family_residue_names(
    method_name: str,
    varying_coordinates: tuple[CompactMethodSemanticCoordinate, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    classvar_names: list[str] = []
    property_hook_names: list[str] = []
    behavior_hook_names: list[str] = []
    for index, (_, kind, _) in enumerate(varying_coordinates, start=1):
        if kind == "constant":
            classvar_names.append(f"{method_name}_constant_{index}".upper())
        elif kind == "self_attr":
            property_hook_names.append(f"{method_name}_property_{index}")
        elif kind == "call":
            behavior_hook_names.append(f"_{method_name}_operation_{index}")
        elif kind in {"attribute", "name"}:
            property_hook_names.append(f"{method_name}_value_{index}")
        else:
            behavior_hook_names.append(f"_{method_name}_hook_{index}")
    return (
        tuple(dict.fromkeys(classvar_names)),
        tuple(dict.fromkeys(property_hook_names)),
        tuple(dict.fromkeys(behavior_hook_names)),
    )


def _method_family_certificate(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
) -> CompressionCertificate | None:
    class_names = method_plans[0].class_names
    manual_object_count = sum(
        (
            len(class_names) * method_plan.profile.shared_statement_count
            for method_plan in method_plans
        )
    )
    residue_names = sorted_tuple(
        {
            residue_name
            for method_plan in method_plans
            for residue_name in (
                *method_plan.profile.classvar_names,
                *method_plan.profile.property_hook_names,
                *method_plan.profile.behavior_hook_names,
            )
        }
    )
    certificate = CompressionCertificate.from_object_family(
        manual_object_count=manual_object_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("abc_base", "family_template"),
            per_axis_objects=("residue_declaration",),
        ),
        semantic_axes=(*method_plans[0].class_names, *residue_names),
        residual_object_count=len(class_names) * len(residue_names),
        provenance_object_count=1,
        independent_source_count=len(set(method_plans[0].file_paths)),
    )
    return certificate if certificate.pays_rent else None


def _method_family_residue_axis_catalog_certificate(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    residue_kind_names: tuple[str, ...],
) -> CompressionCertificate | None:
    axis_names = tuple(
        (
            f"{index}:{residue_kind}"
            for index, residue_kind in enumerate(residue_kind_names)
        )
    )
    certificate = factorization_axis_catalog_certificate(
        (
            FactorizationRow.from_mapping(
                method_plan.method_name,
                {
                    axis_name: residue_kind
                    for axis_name, residue_kind in zip(axis_names, residue_kind_names)
                },
                source_name="|".join(sorted_tuple(frozenset(method_plan.file_paths))),
            )
            for method_plan in method_plans
        ),
        shared_objects=("residue_axis_catalog",),
        per_axis_objects=("residue_axis_row",),
    )
    return certificate if certificate.pays_rent else None


def _method_family_base_is_more_specific(
    candidate_base_name: str,
    incumbent_base_name: str,
    class_index: ClassFamilyIndex,
) -> bool:
    return incumbent_base_name in class_index.ancestor_symbols(candidate_base_name)


def _method_family_more_specific_method_plans(
    method_plans: Iterable[_MethodFamilyMethodPlan],
    class_index: ClassFamilyIndex,
) -> tuple[_MethodFamilyMethodPlan, ...]:
    plans_by_key: dict[_MethodFamilyMethodPlanKey, _MethodFamilyMethodPlan] = {}
    for method_plan in method_plans:
        key = (method_plan.method_name, method_plan.class_names)
        incumbent = plans_by_key.get(key)
        if incumbent is None or _method_family_base_is_more_specific(
            method_plan.base_symbol, incumbent.base_symbol, class_index
        ):
            plans_by_key[key] = method_plan
    return tuple(plans_by_key.values())


def _method_family_candidate_from_method_plan_path(
    file_path: str,
    method_plan: _MethodFamilyMethodPlan,
    family_plan: _MethodFamilyEvidence,
) -> SemanticOverlapMethodCandidate:
    return SemanticOverlapMethodCandidate(
        file_path=file_path,
        line=min(method_plan.line_numbers),
        base_name=method_plan.base_name,
        method_name=method_plan.method_name,
        class_names=method_plan.class_names,
        file_paths=method_plan.file_paths,
        line_numbers=method_plan.line_numbers,
        shared_statement_count=method_plan.profile.shared_statement_count,
        varying_coordinate_count=len(method_plan.profile.varying_coordinates),
        classvar_names=method_plan.profile.classvar_names,
        property_hook_names=method_plan.profile.property_hook_names,
        behavior_hook_names=method_plan.profile.behavior_hook_names,
        family_method_names=family_plan.method_names,
        leaf_residue_names=family_plan.leaf_residue_names,
        residue_declaration_count=family_plan.residue_declaration_count,
        shared_to_residue_ratio=family_plan.shared_to_residue_ratio,
        strict_subset_family_specs=family_plan.strict_subset_family_specs,
        partial_overlap_family_specs=family_plan.partial_overlap_family_specs,
        lattice_node_count=family_plan.lattice_node_count,
        lattice_edge_count=family_plan.lattice_edge_count,
        line_count=method_plan.line_count,
        compression_certificate=method_plan.profile.compression_certificate,
    )


def _method_family_candidate(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plan: _MethodFamilyEvidence,
) -> SemanticOverlapMethodFamilyCandidate | None:
    if len(method_plans) < 2:
        return None
    certificate = _method_family_certificate(method_plans)
    if certificate is None:
        return None
    file_paths = tuple(
        (
            file_path
            for method_plan in method_plans
            for file_path in method_plan.file_paths
        )
    )
    line_numbers = tuple(
        (
            line_number
            for method_plan in method_plans
            for line_number in method_plan.line_numbers
        )
    )
    method_symbols = tuple(
        (
            f"{class_name}.{method_plan.method_name}"
            for method_plan in method_plans
            for class_name in method_plan.class_names
        )
    )
    residue_count = sum(
        (
            len(method_plan.profile.classvar_names)
            + len(method_plan.profile.property_hook_names)
            + len(method_plan.profile.behavior_hook_names)
            for method_plan in method_plans
        )
    )
    return SemanticOverlapMethodFamilyCandidate(
        file_path=file_paths[0],
        line=min(line_numbers),
        base_name=family_plan.base_name,
        class_names=family_plan.class_names,
        method_names=family_plan.method_names,
        file_paths=file_paths,
        line_numbers=line_numbers,
        method_symbols=method_symbols,
        shared_statement_count=sum(
            (method_plan.profile.shared_statement_count for method_plan in method_plans)
        ),
        residue_declaration_count=residue_count,
        classvar_names=family_plan.classvar_names,
        property_hook_names=family_plan.property_hook_names,
        behavior_hook_names=family_plan.behavior_hook_names,
        leaf_residue_names=family_plan.leaf_residue_names,
        shared_to_residue_ratio=family_plan.shared_to_residue_ratio,
        strict_subset_family_specs=family_plan.strict_subset_family_specs,
        partial_overlap_family_specs=family_plan.partial_overlap_family_specs,
        lattice_node_count=family_plan.lattice_node_count,
        lattice_edge_count=family_plan.lattice_edge_count,
        line_count=sum((method_plan.line_count for method_plan in method_plans)),
        compression_certificate=certificate,
    )


def _method_family_residue_kind_names(
    method_plan: _MethodFamilyMethodPlan,
) -> tuple[str, ...]:
    return tuple((kind for _, kind, _ in method_plan.profile.varying_coordinates))


def _method_family_residue_axis_catalog_candidate(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plan: _MethodFamilyEvidence,
) -> SemanticOverlapResidueAxisCandidate | None:
    residue_context = (
        Maybe.of(method_plans)
        .filter(lambda plans: len(plans) >= 2)
        .map(
            lambda plans: {
                _method_family_residue_kind_names(method_plan) for method_plan in plans
            }
        )
        .filter(lambda signatures: len(signatures) == 1)
        .map(lambda signatures: next(iter(signatures)))
        .filter(bool)
        .combine(
            lambda residue_kind_names: _method_family_residue_axis_catalog_certificate(
                method_plans, residue_kind_names
            ),
            lambda residue_kind_names, certificate: (residue_kind_names, certificate),
        )
        .unwrap_or_none()
    )
    if residue_context is None:
        return None
    residue_kind_names, certificate = residue_context
    file_paths = tuple(
        (
            file_path
            for method_plan in method_plans
            for file_path in method_plan.file_paths
        )
    )
    line_numbers = tuple(
        (
            line_number
            for method_plan in method_plans
            for line_number in method_plan.line_numbers
        )
    )
    method_symbols = tuple(
        (
            f"{class_name}.{method_plan.method_name}"
            for method_plan in method_plans
            for class_name in method_plan.class_names
        )
    )
    return SemanticOverlapResidueAxisCandidate(
        file_path=file_paths[0],
        line=min(line_numbers),
        base_name=family_plan.base_name,
        class_names=family_plan.class_names,
        method_names=family_plan.method_names,
        residue_kind_names=residue_kind_names,
        file_paths=file_paths,
        line_numbers=line_numbers,
        method_symbols=method_symbols,
        residue_site_count=len(method_plans) * len(residue_kind_names),
        line_count=sum((method_plan.line_count for method_plan in method_plans)),
        compression_certificate=certificate,
    )


def _method_family_member_spec(method_name: str, class_names: tuple[str, ...]) -> str:
    return f"{method_name}[{','.join(class_names)}]"


def _method_family_group_spec(
    method_names: tuple[str, ...], class_names: tuple[str, ...]
) -> str:
    return f"{'+'.join(method_names)}[{','.join(class_names)}]"


def _method_family_global_certificate(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plans: tuple[_MethodFamilyEvidence, ...],
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=sum(
            (
                len(method_plan.class_names)
                * method_plan.profile.shared_statement_count
                for method_plan in method_plans
            )
        ),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("inheritance_lattice", "abc_base"),
            per_axis_objects=("residue_declaration",),
        ),
        semantic_axes=(
            *(method_plan.method_name for method_plan in method_plans),
            *(
                _method_family_group_spec(
                    family_plan.method_names, family_plan.class_names
                )
                for family_plan in family_plans
            ),
        ),
        residual_object_count=sum(
            (
                len(method_plan.class_names)
                * (
                    len(method_plan.profile.classvar_names)
                    + len(method_plan.profile.property_hook_names)
                    + len(method_plan.profile.behavior_hook_names)
                )
                for method_plan in method_plans
            )
        ),
        provenance_object_count=len(family_plans),
        independent_source_count=len(
            {
                class_name
                for method_plan in method_plans
                for class_name in method_plan.class_names
            }
        ),
    )


def _method_family_sets_have_global_structure(
    family_plans: tuple[_MethodFamilyEvidence, ...],
) -> bool:
    class_sets = tuple(
        (frozenset(family_plan.class_names) for family_plan in family_plans)
    )
    for left_index, left_classes in enumerate(class_sets):
        for right_classes in class_sets[left_index + 1 :]:
            if (left_classes & right_classes) and left_classes != right_classes:
                return True
    return False


def _method_family_global_lattice_edge_count(
    family_plans: tuple[_MethodFamilyEvidence, ...],
) -> int:
    edges: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    class_sets = tuple(
        (frozenset(family_plan.class_names) for family_plan in family_plans)
    )
    for left_index, left_classes in enumerate(class_sets):
        for right_classes in class_sets[left_index + 1 :]:
            intersection = left_classes & right_classes
            if not intersection:
                continue
            intersection_names = sorted_tuple(intersection)
            for class_set in (left_classes, right_classes):
                class_names = sorted_tuple(class_set)
                if intersection_names != class_names:
                    edges.add((intersection_names, class_names))
    return len(edges)


def _method_family_global_inheritance_candidate(
    base_name: str,
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plans: tuple[_MethodFamilyEvidence, ...],
) -> OverlappingInheritanceFamiliesCandidate | None:
    if len(family_plans) < 2:
        return None
    if not _method_family_sets_have_global_structure(family_plans):
        return None
    certificate = _method_family_global_certificate(method_plans, family_plans)
    if not certificate.pays_rent:
        return None
    file_paths = tuple(
        (
            file_path
            for method_plan in method_plans
            for file_path in method_plan.file_paths
        )
    )
    line_numbers = tuple(
        (
            line_number
            for method_plan in method_plans
            for line_number in method_plan.line_numbers
        )
    )
    method_symbols = tuple(
        (
            f"{class_name}.{method_plan.method_name}"
            for method_plan in method_plans
            for class_name in method_plan.class_names
        )
    )
    class_names = sorted_tuple(
        {
            class_name
            for method_plan in method_plans
            for class_name in method_plan.class_names
        }
    )
    return OverlappingInheritanceFamiliesCandidate(
        file_path=file_paths[0],
        line=min(line_numbers),
        base_name=base_name,
        class_names=class_names,
        method_names=sorted_tuple(
            (method_plan.method_name for method_plan in method_plans)
        ),
        family_specs=tuple(
            (
                _method_family_group_spec(
                    family_plan.method_names, family_plan.class_names
                )
                for family_plan in family_plans
            )
        ),
        strict_subset_family_specs=sorted_tuple(
            (
                axis_spec
                for family_plan in family_plans
                for axis_spec in family_plan.strict_subset_family_specs
            )
        ),
        partial_overlap_family_specs=sorted_tuple(
            (
                axis_spec
                for family_plan in family_plans
                for axis_spec in family_plan.partial_overlap_family_specs
            )
        ),
        file_paths=file_paths,
        line_numbers=line_numbers,
        method_symbols=method_symbols,
        shared_statement_count=sum(
            (method_plan.profile.shared_statement_count for method_plan in method_plans)
        ),
        residue_declaration_count=sum(
            (family_plan.residue_declaration_count for family_plan in family_plans)
        ),
        leaf_residue_names=sorted_tuple(
            (
                residue_name
                for family_plan in family_plans
                for residue_name in family_plan.leaf_residue_names
            )
        ),
        lattice_node_count=len(
            {family_plan.class_names for family_plan in family_plans}
            | {
                sorted_tuple(set(left.class_names) & set(right.class_names))
                for left in family_plans
                for right in family_plans
                if set(left.class_names) & set(right.class_names)
            }
        ),
        lattice_edge_count=_method_family_global_lattice_edge_count(family_plans),
        line_count=sum((method_plan.line_count for method_plan in method_plans)),
        compression_certificate=certificate,
    )


def _method_family_residue_evidence(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    *,
    class_names: tuple[str, ...],
    shared_statement_score: int,
) -> _MethodFamilyResidueEvidence:
    classvars = sorted_tuple(
        (
            name
            for method_plan in method_plans
            for name in method_plan.profile.classvar_names
        )
    )
    properties = sorted_tuple(
        (
            name
            for method_plan in method_plans
            for name in method_plan.profile.property_hook_names
        )
    )
    behaviors = sorted_tuple(
        (
            name
            for method_plan in method_plans
            for name in method_plan.profile.behavior_hook_names
        )
    )
    residue_declaration_count = len(class_names) * sum(
        (
            len(method_plan.profile.classvar_names)
            + len(method_plan.profile.property_hook_names)
            + len(method_plan.profile.behavior_hook_names)
            for method_plan in method_plans
        )
    )
    return _MethodFamilyResidueEvidence(
        classvar_names=classvars,
        property_hook_names=properties,
        behavior_hook_names=behaviors,
        leaf_residue_names=sorted_tuple((*classvars, *properties, *behaviors)),
        residue_declaration_count=residue_declaration_count,
        shared_to_residue_ratio=(
            shared_statement_score / max(residue_declaration_count, 1)
        ),
    )


def _method_family_plan(
    family_key: _MethodFamilyKey,
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    subset_axis_specs: tuple[str, ...],
    overlap_axis_specs: tuple[str, ...],
    lattice_node_count: int,
    lattice_edge_count: int,
) -> _MethodFamilyEvidence:
    _, class_names = family_key
    base_name = method_plans[0].base_name
    method_names = sorted_tuple(
        (method_plan.method_name for method_plan in method_plans)
    )
    shared_statement_score = sum(
        (method_plan.profile.shared_statement_count for method_plan in method_plans)
    )
    residue = _method_family_residue_evidence(
        method_plans,
        class_names=class_names,
        shared_statement_score=shared_statement_score,
    )
    return _MethodFamilyEvidence(
        base_name=base_name,
        class_names=class_names,
        method_names=method_names,
        classvar_names=residue.classvar_names,
        property_hook_names=residue.property_hook_names,
        behavior_hook_names=residue.behavior_hook_names,
        leaf_residue_names=residue.leaf_residue_names,
        residue_declaration_count=residue.residue_declaration_count,
        shared_to_residue_ratio=residue.shared_to_residue_ratio,
        strict_subset_family_specs=subset_axis_specs,
        partial_overlap_family_specs=overlap_axis_specs,
        lattice_node_count=lattice_node_count,
        lattice_edge_count=lattice_edge_count,
    )


def _method_family_partitioned_axis_coordinates(
    subset_rows: set[_MethodFamilyAxisRow],
    overlap_method_names: set[str],
    overlap_axis_specs: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nonorthogonal_subset_rows: set[_MethodFamilyAxisRow] = set()
    subset_row_tuple = tuple(subset_rows)
    for left_index, left_row in enumerate(subset_row_tuple):
        left_classes = set(left_row[1])
        for right_row in subset_row_tuple[left_index + 1 :]:
            right_classes = set(right_row[1])
            if not (left_classes & right_classes):
                continue
            if left_classes < right_classes or right_classes < left_classes:
                continue
            nonorthogonal_subset_rows.update((left_row, right_row))

    clean_subset_rows = subset_rows - nonorthogonal_subset_rows
    clean_subset_specs = {
        _method_family_member_spec(method_name, class_names)
        for method_name, class_names in clean_subset_rows
    }
    nonorthogonal_specs = {
        _method_family_member_spec(method_name, class_names)
        for method_name, class_names in nonorthogonal_subset_rows
    }
    return (
        sorted_tuple(clean_subset_specs),
        sorted_tuple(overlap_axis_specs | nonorthogonal_specs),
    )


def _method_family_plans(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
) -> dict[_MethodFamilyKey, _MethodFamilyEvidence]:
    exact_groups: dict[_MethodFamilyKey, list[_MethodFamilyMethodPlan]] = defaultdict(
        list
    )
    overlap_methods_by_family: _MethodFamilyMethodNamesByFamily = defaultdict(set)
    subset_rows_by_family: _MethodFamilyAxisRowsByFamily = defaultdict(set)
    overlap_specs_by_family: _MethodFamilyAxisSpecsByFamily = defaultdict(set)
    lattice_nodes_by_family: dict[_MethodFamilyKey, set[tuple[str, ...]]] = defaultdict(
        set
    )
    lattice_edges_by_family: _MethodFamilyLatticeEdgesByFamily = defaultdict(set)
    for method_plan in method_plans:
        family_key = (method_plan.base_symbol, method_plan.class_names)
        family_classes = set(method_plan.class_names)
        family_class_names = sorted_tuple(family_classes)
        lattice_nodes_by_family[family_key].add(family_class_names)
        exact_groups[family_key].append(method_plan)
        for other_plan in method_plans:
            if other_plan.base_symbol != method_plan.base_symbol:
                continue
            other_classes = set(other_plan.class_names)
            if other_classes == family_classes:
                continue
            class_intersection = family_classes & other_classes
            if not class_intersection:
                continue
            other_class_names = sorted_tuple(other_classes)
            intersection_class_names = sorted_tuple(class_intersection)
            lattice_nodes_by_family[family_key].add(other_class_names)
            lattice_nodes_by_family[family_key].add(intersection_class_names)
            if intersection_class_names != family_class_names:
                lattice_edges_by_family[family_key].add(
                    (intersection_class_names, family_class_names)
                )
            if intersection_class_names != other_class_names:
                lattice_edges_by_family[family_key].add(
                    (intersection_class_names, other_class_names)
                )
            if other_classes < family_classes:
                subset_rows_by_family[family_key].add(
                    (other_plan.method_name, other_class_names)
                )
            elif (
                other_classes != family_classes
                and (not other_classes < family_classes)
                and (not family_classes < other_classes)
            ):
                overlap_methods_by_family[family_key].add(other_plan.method_name)
                overlap_specs_by_family[family_key].add(
                    _method_family_member_spec(
                        other_plan.method_name, other_class_names
                    )
                )
    return {
        family_key: _method_family_plan(
            family_key,
            tuple(group),
            *_method_family_partitioned_axis_coordinates(
                subset_rows_by_family[family_key],
                overlap_methods_by_family[family_key],
                overlap_specs_by_family[family_key],
            ),
            len(lattice_nodes_by_family[family_key]),
            len(lattice_edges_by_family[family_key]),
        )
        for family_key, group in exact_groups.items()
    }


def _compact_method_family_varying_coordinates(
    profiles: tuple[CompactClassMethodSemanticProfile, ...],
) -> tuple[CompactMethodSemanticCoordinate, ...]:
    grouped: dict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)
    representatives: dict[
        tuple[tuple[str, ...], str], CompactMethodSemanticCoordinate
    ] = {}
    for profile in profiles:
        for path, kind, value in profile.coordinates:
            key = (path, kind)
            grouped[key].add(value)
            representatives.setdefault(key, (path, kind, value))
    return tuple(
        representatives[key]
        for key, values in sorted(grouped.items(), key=lambda item: item[0])
        if len(values) >= 2
    )


def _compact_method_family_method_plan(
    base_symbol: str,
    base_name: str,
    method_name: str,
    class_methods: tuple[tuple[CompactIndexedClass, CompactClassMethod], ...],
) -> _MethodFamilyMethodPlan | None:
    if len(class_methods) < 2:
        return None
    methods = tuple(method for _, method in class_methods)
    statement_counts = {method.statement_count for method in methods}
    if len(statement_counts) != 1:
        return None
    shared_statement_count = next(iter(statement_counts))
    if shared_statement_count < 3:
        return None
    profiles = tuple(method.semantic_profile for method in methods)
    skeletons = {profile.skeleton for profile in profiles}
    if len(skeletons) != 1:
        return None
    varying_coordinates = _compact_method_family_varying_coordinates(profiles)
    if not varying_coordinates or len(varying_coordinates) > max(
        4, shared_statement_count * 2
    ):
        return None
    classvar_names, property_hook_names, behavior_hook_names = (
        _method_family_residue_names(method_name, varying_coordinates)
    )
    certificate = ClassFamilyCompressionProfile.from_repeated_method_family(
        class_count=len({indexed_class.symbol for indexed_class, _ in class_methods}),
        shared_statement_count=shared_statement_count,
        hook_count=len(property_hook_names) + len(behavior_hook_names),
        classvar_count=len(classvar_names),
    ).compression_certificate
    if not certificate.pays_rent:
        return None
    return _MethodFamilyMethodPlan(
        base_symbol=base_symbol,
        base_name=base_name,
        method_name=method_name,
        profile=_MethodFamilyMethodGroupProfile(
            shared_statement_count=shared_statement_count,
            varying_coordinates=varying_coordinates,
            classvar_names=classvar_names,
            property_hook_names=property_hook_names,
            behavior_hook_names=behavior_hook_names,
            compression_certificate=certificate,
        ),
        class_names=tuple(
            indexed_class.simple_name for indexed_class, _ in class_methods
        ),
        file_paths=tuple(indexed_class.file_path for indexed_class, _ in class_methods),
        line_numbers=tuple(method.line for method in methods),
        line_count=sum(method.line_count for method in methods),
    )


def _compact_method_family_classes_by_base(
    class_index: CompactClassFamilyIndex,
) -> dict[tuple[str, str], list[CompactIndexedClass]]:
    classes_by_base = {
        (base_symbol, base.simple_name): [
            indexed_class
            for descendant_symbol in descendant_symbols
            if (indexed_class := class_index.class_for(descendant_symbol)) is not None
        ]
        for base_symbol, descendant_symbols in class_index.descendants_by_symbol.items()
        if (base := class_index.class_for(base_symbol)) is not None
        and ClassSymbolResolutionAuthority.establishes_nominal_family(base.simple_name)
    }
    for indexed_class in class_index.classes_by_symbol.values():
        resolved_base_names = {
            base.simple_name
            for base_symbol in indexed_class.resolved_base_symbols
            if (base := class_index.class_for(base_symbol)) is not None
        }
        for base_name in indexed_class.declared_base_names:
            simple_name = base_name.rsplit(".", 1)[-1]
            if (
                ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
                and simple_name not in resolved_base_names
            ):
                classes_by_base.setdefault((base_name, simple_name), []).append(
                    indexed_class
                )
    return classes_by_base


def _compact_method_family_specific_method_plans(
    projections: tuple[CompactModuleClassProjection, ...],
    class_index: CompactClassFamilyIndex,
) -> tuple[_MethodFamilyMethodPlan, ...]:
    methods_by_class: dict[str, dict[str, CompactClassMethod]] = defaultdict(dict)
    for projection in projections:
        for method in projection.class_methods:
            methods_by_class[method.class_symbol][method.method_name] = method
    classes_by_base = _compact_method_family_classes_by_base(class_index)
    method_plans: list[_MethodFamilyMethodPlan] = []
    for (base_symbol, base_name), indexed_classes in classes_by_base.items():
        if len(indexed_classes) < 2:
            continue
        method_names = sorted_tuple(
            {
                method_name
                for indexed_class in indexed_classes
                for method_name in methods_by_class.get(indexed_class.symbol, ())
            }
        )
        for method_name in method_names:
            class_methods = tuple(
                (indexed_class, method)
                for indexed_class in indexed_classes
                if (
                    method := methods_by_class.get(indexed_class.symbol, {}).get(
                        method_name
                    )
                )
                is not None
            )
            method_plan = _compact_method_family_method_plan(
                base_symbol,
                base_name,
                method_name,
                class_methods,
            )
            if method_plan is not None:
                method_plans.append(method_plan)
    return _method_family_more_specific_method_plans(method_plans, class_index)


def _compact_semantic_overlap_method_candidates(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plans: dict[_MethodFamilyKey, _MethodFamilyEvidence],
) -> tuple[SemanticOverlapMethodCandidate, ...]:
    candidates: list[SemanticOverlapMethodCandidate] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for method_plan in method_plans:
        family_plan = family_plans[(method_plan.base_symbol, method_plan.class_names)]
        candidate = _method_family_candidate_from_method_plan_path(
            method_plan.file_paths[0], method_plan, family_plan
        )
        key = (candidate.base_name, candidate.method_name, candidate.class_names)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.base_name,
            candidate.method_name,
        ),
    )


def _compact_method_family_candidates(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plans: dict[_MethodFamilyKey, _MethodFamilyEvidence],
    builder: _MethodFamilyCandidateBuilder[_MethodFamilyCandidateT],
) -> tuple[_MethodFamilyCandidateT, ...]:
    candidates: list[_MethodFamilyCandidateT] = []
    for family_key, family_plan in family_plans.items():
        candidate = builder(
            _method_family_method_plans(method_plans, family_key),
            family_plan,
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.base_name,
            candidate.method_names,
        ),
    )


def _compact_global_inheritance_candidates(
    method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_plans: dict[_MethodFamilyKey, _MethodFamilyEvidence],
) -> tuple[OverlappingInheritanceFamiliesCandidate, ...]:
    method_plans_by_base: dict[str, list[_MethodFamilyMethodPlan]] = defaultdict(list)
    family_plans_by_base: dict[str, list[_MethodFamilyEvidence]] = defaultdict(list)
    for method_plan in method_plans:
        method_plans_by_base[method_plan.base_name].append(method_plan)
    for family_plan in family_plans.values():
        family_plans_by_base[family_plan.base_name].append(family_plan)
    candidates = tuple(
        candidate
        for base_name, base_method_plans in method_plans_by_base.items()
        if (
            candidate := _method_family_global_inheritance_candidate(
                base_name,
                tuple(base_method_plans),
                tuple(family_plans_by_base[base_name]),
            )
        )
        is not None
    )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.base_name,
        ),
    )


@dataclass(frozen=True)
class CompactMethodFamilyContext:
    exact_method_candidates: tuple[ExactMethodRoleComponent, ...]
    exact_ancestor_promotion_candidates: tuple[
        ExactLeafMethodAncestorPromotionComponent,
        ...,
    ]
    method_candidates: tuple[SemanticOverlapMethodCandidate, ...]
    family_candidates: tuple[SemanticOverlapMethodFamilyCandidate, ...]
    global_candidates: tuple[OverlappingInheritanceFamiliesCandidate, ...]
    residue_axis_candidates: tuple[SemanticOverlapResidueAxisCandidate, ...]

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> "CompactMethodFamilyContext":
        if class_index is None:
            class_index = build_compact_class_family_index(projections)
        method_plans = _compact_method_family_specific_method_plans(
            projections, class_index
        )
        family_plans = _method_family_plans(method_plans)
        exact_promotion_builder = (
            ExactLeafMethodAncestorPromotionComponentBuilder.from_projections(
                projections,
                class_index=class_index,
            )
        )
        return cls(
            exact_method_candidates=ExactMethodRoleComponentBuilder(
                exact_promotion_builder
            ).proven_components,
            exact_ancestor_promotion_candidates=(
                exact_promotion_builder.proven_components
            ),
            method_candidates=_compact_semantic_overlap_method_candidates(
                method_plans, family_plans
            ),
            family_candidates=_compact_method_family_candidates(
                method_plans,
                family_plans,
                _method_family_candidate,
            ),
            global_candidates=_compact_global_inheritance_candidates(
                method_plans, family_plans
            ),
            residue_axis_candidates=_compact_method_family_candidates(
                method_plans,
                family_plans,
                _method_family_residue_axis_catalog_candidate,
            ),
        )


def _method_family_method_plans(
    specific_method_plans: tuple[_MethodFamilyMethodPlan, ...],
    family_key: _MethodFamilyKey,
) -> tuple[_MethodFamilyMethodPlan, ...]:
    return tuple(
        (
            method_plan
            for method_plan in specific_method_plans
            if (method_plan.base_symbol, method_plan.class_names) == family_key
        )
    )


def _builder_patch(builders: tuple[BuilderCallShape, ...]) -> str:
    target_file = builders[0].file_path
    callee_name = builders[0].callee_name
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n+@classmethod\n+def from_source(cls, source):\n+    return {callee_name}(...)\n*** End Patch"


def _autoregister_patch(
    registry_name: str,
    class_names: set[str],
    registrations: tuple[RegistrationShape, ...],
) -> str:
    target_file = registrations[0].file_path
    base_name = (
        HELPER_SUPPORT_PROJECTION_AUTHORITY.shared_family_name(sorted(class_names))
        or "RegisteredBase"
    )
    ordered_class_names = sorted_tuple(class_names)
    key_values = tuple(
        (
            key_value
            for class_name in ordered_class_names
            if (
                key_value := _string_constant_expression(
                    next(
                        (
                            registration.key_expression
                            for registration in registrations
                            if registration.registered_class == class_name
                        )
                    )
                )
            )
            is not None
        )
    )
    use_extractor = len(key_values) == len(ordered_class_names) and (
        DISPATCH_ALGEBRA_AUTHORITY.derivable_registry_key_suffix(
            ordered_class_names, key_values
        )
        is not None
    )
    config_block = (
        DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(ordered_class_names)
        if use_extractor
        else DISPATCH_ALGEBRA_AUTHORITY.declared_registry_key_block("registry_key")
    )
    return f"*** Begin Patch\n*** Update File: {target_file}\n@@\n" + (
        "+from metaclass_registry import AutoRegisterMeta\n"
        + ("+import re\n" if use_extractor else "")
        + "+\n"
        + f"+class {base_name}(ABC, metaclass=AutoRegisterMeta):\n"
        + "".join(f"+{line}\n" for line in config_block.splitlines())
        + "+\n"
        + f"+# Replace `{registry_name}` with `{base_name}.__registry__`.\n"
        + "*** End Patch"
    )


@lru_cache(maxsize=None)
def _attribute_branch_evidence(
    module: ParsedModule, attr_name: str
) -> list[SourceLocation]:
    evidence: list[SourceLocation] = []
    for node in _walk_nodes(module.module):
        if isinstance(node, ast.If):
            if _test_compares_attribute(node.test, attr_name):
                evidence.append(
                    SourceLocation(module.file_path, node.lineno, f"if-{attr_name}")
                )
        if isinstance(node, ast.Match):
            subject = node.subject
            if isinstance(subject, ast.Attribute) and subject.attr == attr_name:
                evidence.append(
                    SourceLocation(module.file_path, node.lineno, f"match-{attr_name}")
                )
    return evidence


def _test_compares_attribute(test: ast.AST, attr_name: str) -> bool:
    for node in _walk_nodes(test):
        if isinstance(node, ast.Compare):
            values = [node.left] + list(node.comparators)
            attr_match = any(
                (
                    isinstance(value, ast.Attribute) and value.attr == attr_name
                    for value in values
                )
            )
            literal_match = any(
                (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, (str, int, bool))
                    for value in values
                )
            )
            if attr_match and literal_match:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == _GETATTR_BUILTIN and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and arg.value == attr_name:
                    return True
    return False


def _iter_functions(module: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in _walk_nodes(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


@dataclass(frozen=True)
class _FunctionSignatureView:
    function: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def parameter_names(self) -> set[str]:
        names = {arg.arg for arg in self.function.args.posonlyargs}
        names.update(arg.arg for arg in self.function.args.args)
        names.update(arg.arg for arg in self.function.args.kwonlyargs)
        if self.function.args.vararg is not None:
            names.add(self.function.args.vararg.arg)
        if self.function.args.kwarg is not None:
            names.add(self.function.args.kwarg.arg)
        return names

    @property
    def explicit_parameter_names(self) -> set[str]:
        return self.parameter_names - _IMPLICIT_METHOD_PARAMETER_NAMES

    @property
    def arguments(self) -> tuple[ast.arg, ...]:
        return (
            *self.function.args.posonlyargs,
            *self.function.args.args,
            *self.function.args.kwonlyargs,
        )

    def has_parameter(self, parameter_name: str) -> bool:
        return any((arg.arg == parameter_name for arg in self.function.args.args))


def _is_transport_expression(node: ast.AST, *, allowed_roots: set[str]) -> bool:
    return (
        HELPER_SUPPORT_PROJECTION_AUTHORITY.expression_root_names(node) <= allowed_roots
    )


class TransportProjectionAuthority:
    def attribute_path(self, node: ast.AST) -> tuple[str, ...] | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        return (current.id, *reversed(parts))

    def call_chain_from_outer_call(self, call: ast.Call) -> tuple[ast.Call, ...]:
        chain = [call]
        current = call
        while isinstance(current.func, ast.Attribute) and isinstance(
            current.func.value, ast.Call
        ):
            current = current.func.value
            chain.append(current)
        return tuple(chain)

    def call_chain_delegate_symbol(
        self,
        chain: tuple[ast.Call, ...],
        *,
        class_name: str | None,
    ) -> str:
        inner = chain[-1]
        symbol = HELPER_SUPPORT_PROJECTION_AUTHORITY.wrapper_delegate_symbol(
            inner.func, class_name=class_name
        )
        if symbol is None:
            symbol = ast.unparse(inner.func)
        for call in reversed(chain[:-1]):
            method_name = AstExpressionProjection.terminal_name(call.func)
            if method_name is None:
                method_name = ast.unparse(call.func)
            symbol = f"{symbol}.{method_name}"
        return symbol

    def transported_values(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        chain: tuple[ast.Call, ...],
    ) -> tuple[ast.AST, ...] | None:
        allowed_roots = (
            _FunctionSignatureView(function).parameter_names
            | _IMPLICIT_METHOD_PARAMETER_NAMES
        )
        values = _call_chain_transport_values(chain)
        return (
            values
            if values
            and all(
                (
                    _is_transport_expression(value, allowed_roots=allowed_roots)
                    for value in values
                )
            )
            else None
        )

    def call_chain_match(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        min_depth: int = 2,
        exact_depth: int | None = None,
    ) -> tuple[tuple[ast.Call, ...], tuple[ast.AST, ...]] | None:
        return (
            Maybe.of(single_return_call(statements_without_docstring(function.body)))
            .map(self.call_chain_from_outer_call)
            .project(
                lambda chain: (
                    chain
                    if len(chain) >= min_depth
                    and (exact_depth is None or len(chain) == exact_depth)
                    else None
                )
            )
            .with_projection(lambda chain: self.transported_values(function, chain))
            .unwrap_or_none()
        )

    def field_delegate_match(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[tuple[ast.Call, ...], tuple[ast.AST, ...]] | None:
        call = single_return_call(statements_without_docstring(function.body))
        if call is None or not isinstance(call.func, ast.Attribute):
            return None
        path = self.attribute_path(call.func)
        if (
            path is None
            or len(path) < 3
            or path[0] not in _IMPLICIT_METHOD_PARAMETER_NAMES
        ):
            return None
        values = self.transported_values(function, (call,))
        if values is None:
            return None
        return ((call,), values)

    def forwarding_match(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[tuple[ast.Call, ...], tuple[ast.AST, ...]] | None:
        return self.call_chain_match(function) or self.field_delegate_match(function)


TRANSPORT_PROJECTION_AUTHORITY = TransportProjectionAuthority()


def _call_chain_transport_values(chain: tuple[ast.Call, ...]) -> tuple[ast.AST, ...]:
    values: list[ast.AST] = []
    for call in chain:
        values.extend(call.args)
        values.extend(keyword.value for keyword in call.keywords)
    return tuple(values)


def _pipeline_body_stages(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[PipelineAssemblyStage, ...] | None:
    body = list(function.body)
    if body and is_docstring_statement(body[0]):
        body = body[1:]
    if len(body) < 2:
        return None
    stages: list[PipelineAssemblyStage] = []
    for statement in body:
        stage = _pipeline_stage(statement)
        if stage is None:
            return None
        stages.append(stage)
    if not stages or stages[-1].kind != _PIPELINE_RETURN_STAGE:
        return None
    return tuple(stages)


@dataclass(frozen=True)
class _PipelineStageSource:
    kind: str
    call: ast.Call
    output_arity: int


def _pipeline_stage_source(statement: ast.stmt) -> _PipelineStageSource | None:
    assignment = as_ast(statement, ast.Assign)
    if assignment is not None:
        call = as_ast(assignment.value, ast.Call)
        output_arity = _assignment_target_arity(single_assign_target(assignment))
        if call is None or output_arity is None:
            return None
        return _PipelineStageSource(_PIPELINE_ASSIGN_STAGE, call, output_arity)
    call = return_call(statement)
    if call is None:
        return None
    return _PipelineStageSource(_PIPELINE_RETURN_STAGE, call, 0)


def _pipeline_stage(statement: ast.stmt) -> PipelineAssemblyStage | None:
    source = _pipeline_stage_source(statement)
    callee_name = (
        AstExpressionProjection.terminal_name(source.call.func)
        if source is not None
        else None
    )
    if source is None or callee_name is None:
        return None
    keyword_names = tuple(
        (keyword.arg for keyword in source.call.keywords if keyword.arg is not None)
    )
    return PipelineAssemblyStage(
        kind=source.kind,
        callee_name=callee_name,
        output_arity=source.output_arity,
        arg_count=len(source.call.args) + len(keyword_names),
        keyword_names=keyword_names,
    )


_DECLARATIVE_DETECTOR_BASE_NAMES = DerivedCandidateCollectorMixin.collector_base_names()


def _class_assignment_names(node: ast.ClassDef) -> tuple[str, ...]:
    assignment_names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            assignment_names.extend(
                (
                    name
                    for target in statement.targets
                    for name in (name_id(target),)
                    if name is not None
                )
            )
        elif isinstance(statement, ast.AnnAssign):
            target_name = name_id(statement.target)
            if target_name is not None:
                assignment_names.append(target_name)
        elif not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return ()
    return tuple(assignment_names)


def _declarative_detector_class_candidates(
    module: ParsedModule,
) -> tuple[DeclarativeDetectorClassCandidate, ...]:
    candidates: list[DeclarativeDetectorClassCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        base_parts = tuple(
            part
            for base in node.bases
            for part in (ParameterizedBaseSource.from_node(base),)
            if part is not None
        )
        base_part = single_item(
            tuple(
                part
                for part in base_parts
                if part.base_name in _DECLARATIVE_DETECTOR_BASE_NAMES
            )
        )
        assignment_names = _class_assignment_names(node)
        required_assignment_names = DetectorDeclaration.required_namespace_field_names()
        if base_part is None or not set(required_assignment_names).issubset(
            assignment_names
        ):
            continue
        candidates.append(
            DeclarativeDetectorClassCandidate(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                base_name=base_part.base_name,
                candidate_type_name=base_part.parameter_source,
                assignment_names=assignment_names,
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(candidates)


_STATIC_OBSERVATION_DETECTOR_BASE_NAMES = frozenset({"StaticModulePatternDetector"})
_STATIC_OBSERVATION_CONTROL_FLOW_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
)


def _typed_observation_collection_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str] | None:
    if any(
        (
            isinstance(child, _STATIC_OBSERVATION_CONTROL_FLOW_NODES)
            for child in _walk_nodes(method)
        )
    ):
        return None
    collection_calls: list[tuple[str, str]] = []
    for child in _walk_nodes(method):
        if not isinstance(child, ast.Call):
            continue
        if name_id(child.func) != "_collect_typed_family_items":
            continue
        if len(child.args) < 3 or name_id(child.args[0]) != "module":
            continue
        family_name = name_id(child.args[1])
        observation_type_name = name_id(child.args[2])
        if family_name is not None and observation_type_name is not None:
            collection_calls.append((family_name, observation_type_name))
    return single_item(tuple(collection_calls))


def _source_location_from_line_symbol_call(
    node: ast.AST,
) -> bool:
    if not isinstance(node, ast.Call) or name_id(node.func) != "SourceLocation":
        return False
    if len(node.args) < 3:
        return False
    return all(
        (
            isinstance(argument, ast.Attribute) and argument.attr == expected_attribute
            for argument, expected_attribute in zip(
                node.args[:3], ("file_path", "line", "symbol"), strict=True
            )
        )
    )


def _static_minimum_evidence_count(node: ast.ClassDef) -> int | None:
    minimum_method = CLASS_NODE_AUTHORITY.method_named(node, "_minimum_evidence")
    if minimum_method is None:
        return 1
    returns = tuple(
        child for child in _walk_nodes(minimum_method) if isinstance(child, ast.Return)
    )
    returned = single_item(returns)
    if returned is None:
        return None
    value = constant_value(returned.value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _static_summary_expression(module: ParsedModule, node: ast.ClassDef) -> str | None:
    return (
        Maybe.of(CLASS_NODE_AUTHORITY.method_named(node, "_summary"))
        .map(
            lambda summary_method: tuple(
                child
                for child in _walk_nodes(summary_method)
                if isinstance(child, ast.Return)
            )
        )
        .project(single_item)
        .project(lambda returned: returned.value)
        .project(lambda value: _source_segment(module, value))
        .unwrap_or_none()
    )


def _static_typed_observation_detector_candidates(
    module: ParsedModule,
) -> tuple[StaticTypedObservationDetectorCandidate, ...]:
    candidates: list[StaticTypedObservationDetectorCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or node.end_lineno is None:
            continue
        if not (
            _STATIC_OBSERVATION_DETECTOR_BASE_NAMES
            & set(CLASS_NODE_AUTHORITY.declared_base_names(node))
        ):
            continue
        evidence_method = CLASS_NODE_AUTHORITY.method_named(node, "_module_evidence")
        if evidence_method is None:
            continue
        collection = _typed_observation_collection_call(evidence_method)
        if collection is None:
            continue
        if not any(
            _source_location_from_line_symbol_call(child)
            for child in _walk_nodes(evidence_method)
        ):
            continue
        minimum_evidence = _static_minimum_evidence_count(node)
        summary_expression = _static_summary_expression(module, node)
        if minimum_evidence is None or summary_expression is None:
            continue
        family_name, observation_type_name = collection
        candidates.append(
            StaticTypedObservationDetectorCandidate(
                file_path=module.file_path,
                line=node.lineno,
                class_name=node.name,
                observation_family_name=family_name,
                observation_type_name=observation_type_name,
                minimum_evidence_count=minimum_evidence,
                summary_expression=summary_expression,
                line_count=node.end_lineno - node.lineno + 1,
            )
        )
    return tuple(candidates)


_TYPED_CANDIDATE_DETECTOR_BASE_NAMES = (
    DerivedCandidateCollectorMixin.collector_base_names()
)


def _single_payload_parameter_name(method: ast.FunctionDef) -> str | None:
    positional = (*method.args.posonlyargs, *method.args.args)
    payload_names = tuple(
        (argument.arg for argument in positional if argument.arg not in {"self", "cls"})
    )
    if method.args.vararg is not None or method.args.kwarg is not None:
        return None
    return single_item(payload_names) if len(payload_names) == 1 else None


def _parameter_name_is_reused(
    statements: Sequence[ast.stmt], parameter_name: str
) -> bool:
    return any(
        (
            isinstance(node, ast.Name) and node.id == parameter_name
            for statement in statements
            for node in _walk_nodes(statement)
        )
    )


def _first_named_call_assignment(
    statements: Sequence[ast.stmt],
) -> NamedCallAssignment | None:
    first_statement = single_item(statements[:1])
    assignment = as_ast(first_statement, ast.Assign)
    return named_call_assignment(assignment) if assignment is not None else None


def _call_is_cast_of_parameter(call: ast.Call, parameter_name: str) -> bool:
    return (
        AstExpressionProjection.terminal_name(call.func) == "cast"
        and len(call.args) == 2
        and (name_id(call.args[1]) == parameter_name)
    )


def _typed_candidate_cast_assignment(
    method: ast.FunctionDef,
) -> tuple[str, str, str] | None:
    parameter_name = _single_payload_parameter_name(method)
    body = statements_without_docstring(method.body)
    call_assignment = _first_named_call_assignment(body)
    if parameter_name is None or call_assignment is None:
        return None
    cast_call = call_assignment.call
    if not _call_is_cast_of_parameter(cast_call, parameter_name):
        return None
    if _parameter_name_is_reused(body[1:], parameter_name):
        return None
    return (parameter_name, call_assignment.target_name, ast.unparse(cast_call.args[0]))


def _typed_candidate_cast_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[TypedCandidateCastBoilerplateCandidate, ...]:
    candidates: list[TypedCandidateCastBoilerplateCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        detector_base_name = (
            HELPER_SUPPORT_PROJECTION_AUTHORITY.concrete_detector_base_name(node)
        )
        if detector_base_name is None:
            continue
        for statement in node.body:
            if not (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "_finding_for_candidate"
            ):
                continue
            cast_assignment = _typed_candidate_cast_assignment(statement)
            if cast_assignment is None:
                continue
            parameter_name, local_name, candidate_type_name = cast_assignment
            candidates.append(
                TypedCandidateCastBoilerplateCandidate(
                    file_path=module.file_path,
                    line=statement.lineno,
                    class_name=node.name,
                    method_name=statement.name,
                    parameter_name=parameter_name,
                    local_name=local_name,
                    candidate_type_name=candidate_type_name,
                    detector_base_name=detector_base_name,
                )
            )
    return tuple(candidates)


def _keyword_value_name(keyword: ast.keyword | None) -> str | None:
    return (
        AstExpressionProjection.terminal_name(keyword.value)
        if keyword is not None
        else None
    )


def _keyword_semantic_value(
    keyword: ast.keyword | None,
) -> FindingSpecSemanticValue | None:
    value_name = _keyword_value_name(keyword)
    if value_name is None:
        return None
    return finding_spec_semantic_value_from_import_name(value_name)


def _semantic_keyword_field(
    keyword_name: str | None,
) -> FindingSpecSemanticField | None:
    if keyword_name is None:
        return None
    try:
        return FindingSpecSemanticField(keyword_name)
    except ValueError:
        return None


def _finding_spec_semantic_keywords(
    keywords: Sequence[ast.keyword],
) -> dict[FindingSpecSemanticField, ast.keyword]:
    semantic_keywords: dict[FindingSpecSemanticField, ast.keyword] = {}
    for keyword in keywords:
        field_name = _semantic_keyword_field(keyword.arg)
        if field_name is not None:
            semantic_keywords[field_name] = keyword
    return semantic_keywords


def _recommended_finding_spec_constructor(
    constructor_name: str,
    semantic_keywords: dict[FindingSpecSemanticField, ast.keyword],
) -> str:
    factory = finding_spec_factory_for_constructor_name(constructor_name)
    if factory is None:
        return constructor_name
    target_defaults = FindingSpecSemanticDefaults(
        confidence=cast(
            ConfidenceLevel,
            _keyword_semantic_value(
                semantic_keywords.get(FindingSpecSemanticField.CONFIDENCE)
            )
            or factory.semantic_defaults.confidence,
        ),
        certification=cast(
            CertificationLevel,
            _keyword_semantic_value(
                semantic_keywords.get(FindingSpecSemanticField.CERTIFICATION)
            )
            or factory.semantic_defaults.certification,
        ),
    )
    recommended_factory = finding_spec_factory_for_defaults(target_defaults)
    return (
        recommended_factory.constructor_name
        if recommended_factory is not None
        else constructor_name
    )


def _finding_spec_default_field_candidate(
    module: ParsedModule, node: ast.Call
) -> tuple[FindingSpecDefaultFieldCandidate, ...]:
    constructor_name = AstExpressionProjection.terminal_name(node.func)
    factory = (
        finding_spec_factory_for_constructor_name(constructor_name)
        if constructor_name is not None
        else None
    )
    if factory is None:
        return ()
    semantic_keywords = _finding_spec_semantic_keywords(node.keywords)
    if not semantic_keywords:
        return ()
    recommended_constructor_name = _recommended_finding_spec_constructor(
        constructor_name, semantic_keywords
    )
    recommended_factory = finding_spec_factory_for_constructor_name(
        recommended_constructor_name
    )
    if recommended_factory is None:
        return ()
    recommended_defaults = recommended_factory.semantic_defaults
    redundant_keywords = tuple(
        (
            (field_name, value_name)
            for field_name, keyword in semantic_keywords.items()
            for value in (_keyword_semantic_value(keyword),)
            for value_name in (_keyword_value_name(keyword),)
            if value is not None
            if value == recommended_defaults.value_for_field(field_name)
            if value_name is not None
        )
    )
    if not redundant_keywords:
        return ()
    return (
        FindingSpecDefaultFieldCandidate(
            file_path=module.file_path,
            line=node.lineno,
            constructor_name=constructor_name,
            recommended_constructor_name=recommended_constructor_name,
            redundant_keyword_names=tuple(
                (name.value for name, _ in redundant_keywords)
            ),
            redundant_keyword_values=tuple((value for _, value in redundant_keywords)),
        ),
    )


def _finding_spec_default_field_candidates(
    module: ParsedModule,
) -> tuple[FindingSpecDefaultFieldCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.Call,
        _finding_spec_default_field_candidate,
    )


def _self_finding_spec_build_call(node: ast.AST) -> ast.Call | None:
    call = as_ast(node, ast.Call)
    if call is None or len(call.args) < 1:
        return None
    if not _is_self_finding_spec_build_func(call.func):
        return None
    if not _is_self_detector_id_attribute(call.args[0]):
        return None
    return call


def _is_self_finding_spec_build_func(node: ast.AST) -> bool:
    build_attr = as_ast(node, ast.Attribute)
    spec_attr = as_ast(build_attr.value if build_attr else None, ast.Attribute)
    return (
        build_attr is not None
        and build_attr.attr == "build"
        and (spec_attr is not None)
        and (spec_attr.attr == "finding_spec")
        and (name_id(spec_attr.value) == "self")
    )


def _is_self_detector_id_attribute(node: ast.AST) -> bool:
    detector_id_arg = as_ast(node, ast.Attribute)
    return (
        detector_id_arg is not None
        and detector_id_arg.attr == "detector_id"
        and (name_id(detector_id_arg.value) == "self")
    )


def _class_declares_detector_id(node: ast.ClassDef) -> bool:
    return any(
        (
            isinstance(statement, ast.Assign)
            and any((name_id(target) == "detector_id" for target in statement.targets))
            for statement in node.body
        )
    )


def _finding_spec_build_boilerplate_class_candidates(
    module: ParsedModule, node: ast.ClassDef
) -> tuple[ClassMethodLineWitnessCandidate, ...]:
    if not _class_declares_detector_id(node):
        return ()
    return tuple(
        (
            ClassMethodLineWitnessCandidate(
                file_path=module.file_path,
                line=child.lineno,
                class_name=node.name,
                method_name=statement.name,
            )
            for statement in node.body
            if isinstance(statement, ast.FunctionDef)
            for child in _walk_nodes(statement)
            if _self_finding_spec_build_call(child) is not None
        )
    )


def _finding_spec_build_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[ClassMethodLineWitnessCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.ClassDef,
        _finding_spec_build_boilerplate_class_candidates,
    )


def _self_build_finding_call(node: ast.AST) -> ast.Call | None:
    return (
        Maybe.of(return_call(node))
        .combine(
            lambda call: as_ast(call.func, ast.Attribute),
            lambda call, function: (
                call
                if function.attr == "build_finding"
                and name_id(function.value) == "self"
                else None
            ),
        )
        .unwrap_or_none()
    )


def _direct_build_finding_renderer_candidates(
    module: ParsedModule,
) -> tuple[DirectBuildFindingRendererCandidate, ...]:
    candidates: list[DirectBuildFindingRendererCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_name = HELPER_SUPPORT_PROJECTION_AUTHORITY.concrete_detector_base_name(
            node
        )
        if base_name is None:
            continue
        for statement in node.body:
            if not (
                isinstance(statement, ast.FunctionDef)
                and statement.name == "_finding_for_candidate"
            ):
                continue
            body = statements_without_docstring(statement.body)
            call = (
                _self_build_finding_call(single_item(body)) if len(body) == 1 else None
            )
            if call is None:
                continue
            candidates.append(
                DirectBuildFindingRendererCandidate(
                    file_path=module.file_path,
                    line=statement.lineno,
                    class_name=node.name,
                    method_name=statement.name,
                    base_name=base_name,
                    positional_arg_count=len(call.args),
                    keyword_names=tuple(
                        (keyword.arg for keyword in call.keywords if keyword.arg)
                    ),
                )
            )
    return tuple(candidates)


@lru_cache(maxsize=1)
def _canonical_finding_spec_field_names() -> tuple[str, ...]:
    signature = inspect.signature(FindingSpecFactory.__call__)
    return tuple(
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


def _canonical_finding_spec_builder_candidates(
    module: ParsedModule,
) -> tuple[CanonicalFindingSpecBuilderCandidate, ...]:
    candidates: list[CanonicalFindingSpecBuilderCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                (name_id(target) == "finding_spec" for target in statement.targets)
            ):
                continue
            call = as_ast(statement.value, ast.Call)
            if call is None or call.args:
                continue
            constructor_name = name_id(call.func)
            factory = (
                finding_spec_factory_for_constructor_name(constructor_name)
                if constructor_name is not None
                else None
            )
            if factory is None:
                continue
            keyword_names = tuple(
                keyword.arg for keyword in call.keywords if keyword.arg
            )
            if not set(_canonical_finding_spec_field_names()[:5]).issubset(
                keyword_names
            ):
                continue
            candidates.append(
                CanonicalFindingSpecBuilderCandidate(
                    file_path=module.file_path,
                    line=statement.lineno,
                    class_name=node.name,
                    constructor_name=constructor_name,
                    builder_name=factory.builder_name,
                    keyword_names=keyword_names,
                )
            )
    return tuple(candidates)


def _source_segment(module: ParsedModule, node: ast.expr) -> str:
    return module.source_segments.segment_for_node(node) or ast.unparse(node)


ClassShapeT = TypeVar("ClassShapeT")
BuiltCandidateT = TypeVar("BuiltCandidateT")
ClassShapeProjector = Callable[[ast.ClassDef], ClassShapeT | None]
ClassShapeCandidateFactory = Callable[
    [ParsedModule, ast.ClassDef, ClassShapeT], BuiltCandidateT
]


class HelperSyntaxProjectionAuthority:
    def assignment_target_value(
        self, statement: ast.stmt
    ) -> tuple[ast.AST, ast.AST] | None:
        assignment = as_ast(statement, ast.Assign)
        if assignment is not None:
            target = single_assign_target(assignment)
            return None if target is None else (target, assignment.value)
        annotated_assignment = as_ast(statement, ast.AnnAssign)
        if annotated_assignment is None or annotated_assignment.value is None:
            return None
        return annotated_assignment.target, annotated_assignment.value

    def abstract_method_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        return sorted_tuple(
            (
                method.name
                for method in CLASS_NODE_AUTHORITY.methods(node)
                if _is_abstract_method(method)
            )
        )

    def direct_source_attribute_name(
        self, node: ast.AST, source_name: str
    ) -> str | None:
        attribute = as_ast(node, ast.Attribute)
        if attribute is None or name_id(attribute.value) != source_name:
            return None
        return attribute.attr

    def classvar_assignment_names(self, node: ast.ClassDef) -> tuple[str, ...] | None:
        assigned_names: list[str] = []
        for statement in statements_without_docstring(node.body):
            binding = named_value_binding(statement)
            if (
                binding is None
                or binding.value is None
                or (not _is_simple_classvar_value(binding.value))
            ):
                return None
            assigned_names.append(binding.name)
        return tuple(assigned_names)

    def class_declares_finding_spec(self, node: ast.ClassDef) -> bool:
        return any(
            (
                isinstance(statement, ast.Assign)
                and any(
                    (name_id(target) == "finding_spec" for target in statement.targets)
                )
                for statement in node.body
            )
        )

    def class_shape_candidates(
        self,
        module: ParsedModule,
        shape_projector: ClassShapeProjector[ClassShapeT],
        candidate_factory: ClassShapeCandidateFactory[ClassShapeT, BuiltCandidateT],
    ) -> tuple[BuiltCandidateT, ...]:
        candidates: list[BuiltCandidateT] = []
        for node in _walk_nodes(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            shape = shape_projector(node)
            if shape is None:
                continue
            candidates.append(candidate_factory(module, node, shape))
        return tuple(candidates)

    def annotation_root_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self.annotation_root_name(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.annotation_root_name(node.left) or self.annotation_root_name(
                node.right
            )
        return None

    def class_base_names(self, node: ast.ClassDef) -> tuple[str, ...]:
        return tuple(
            (
                base_name
                for base in node.bases
                if (base_name := AstExpressionProjection.terminal_name(base))
                is not None
            )
        )

    def typed_field_map(self, node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
        typed_fields: dict[str, str] = {}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                if CLASSVAR_ANNOTATION_AUTHORITY.matches(statement.annotation):
                    continue
                typed_fields.setdefault(
                    statement.target.id, ast.unparse(statement.annotation)
                )
                continue
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if statement.name != "__init__":
                continue
            parameter_annotations = _function_parameter_annotation_map(statement)
            for inner in statement.body:
                target: ast.AST | None = None
                value: ast.AST | None = None
                if isinstance(inner, ast.Assign) and len(inner.targets) == 1:
                    target = inner.targets[0]
                    value = inner.value
                elif isinstance(inner, ast.AnnAssign):
                    target = inner.target
                    value = inner.value
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and (target.value.id == "self")
                    and isinstance(value, ast.Name)
                    and (value.id in parameter_annotations)
                ):
                    continue
                typed_fields.setdefault(target.attr, parameter_annotations[value.id])
        return sorted_tuple(typed_fields.items())

    def shared_typed_field_names(
        self, concrete: NominalAuthorityShape, authority: NominalAuthorityShape
    ) -> tuple[str, ...]:
        concrete_types = dict(concrete.field_type_map)
        return tuple(
            (
                name
                for name, annotation_text in authority.field_type_map
                if concrete_types.get(name) == annotation_text
            )
        )

    def simple_attribute_accesses(self, node: ast.AST) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                (current.value.id, current.attr)
                for current in _walk_nodes(node)
                if isinstance(current, ast.Attribute)
                and isinstance(current.value, ast.Name)
                and (current.value.id not in {"self", "cls"})
            )
        )

    def metadata_only_class_assignment_names(
        self, node: ast.ClassDef
    ) -> tuple[str, ...] | None:
        assigned_names: list[str] = []
        for statement in statements_without_docstring(node.body):
            if isinstance(statement, ast.Pass):
                continue
            binding = named_value_binding(statement)
            if (
                binding is None
                or binding.value is None
                or (not _is_declarative_class_value(binding.value))
            ):
                return None
            assigned_names.append(binding.name)
        return tuple(assigned_names)

    def type_name_set(self, node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            return (ast.unparse(node),)
        if isinstance(node, ast.Tuple):
            return sorted_tuple(
                {
                    type_name
                    for element in node.elts
                    for type_name in self.type_name_set(element)
                }
            )
        return ()

    def subclasses_root_expression(self, node: ast.AST) -> str | None:
        subclasses_call = single_named_call_argument(
            node, call_name=BuiltinCallName.LIST, argument_type=ast.Call
        ) or as_ast(node, ast.Call)
        if subclasses_call is None:
            return None
        match = attribute_call_match(
            subclasses_call,
            method_name="__subclasses__",
            owner_type=ast.AST,
            argument_count=0,
        )
        return None if match is None else ast.unparse(match.owner)

    def extends_subclasses_queue(
        self, statement: ast.stmt, queue_name: str, current_name: str
    ) -> bool:
        if (
            not isinstance(statement, ast.Expr)
            or not isinstance(statement.value, ast.Call)
            or (not isinstance(statement.value.func, ast.Attribute))
            or (statement.value.func.attr != "extend")
            or (not isinstance(statement.value.func.value, ast.Name))
            or (statement.value.func.value.id != queue_name)
            or (len(statement.value.args) != 1)
        ):
            return False
        return self.subclasses_root_expression(statement.value.args[0]) == current_name

    def parameter_receiver_attribute_names(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, parameter_name: str
    ) -> tuple[str, ...]:
        return sorted_tuple(
            {
                item.attr
                for item in _walk_nodes(function)
                if isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == parameter_name
            }
        )

    def direct_forwarded_parameter_names(
        self,
        call: ast.Call,
        *,
        parameter_names: set[str],
    ) -> tuple[str, ...] | None:
        forwarded: list[str] = []
        seen: set[str] = set()
        for argument in call.args:
            if isinstance(argument, ast.Name) and argument.id in parameter_names:
                if argument.id not in seen:
                    seen.add(argument.id)
                    forwarded.append(argument.id)
                continue
            return None
        for keyword in call.keywords:
            if keyword.arg is None:
                return None
            if (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameter_names
            ):
                if keyword.value.id not in seen:
                    seen.add(keyword.value.id)
                    forwarded.append(keyword.value.id)
                continue
            return None
        return tuple(forwarded)


HELPER_SYNTAX_PROJECTION_AUTHORITY = HelperSyntaxProjectionAuthority()


def _field_only_frozen_dataclass_candidates(
    module: ParsedModule,
) -> tuple[FieldOnlyFrozenDataclassCandidate, ...]:
    return tuple(
        candidate
        for node in _walk_nodes(module.module)
        if isinstance(node, ast.ClassDef)
        for candidate in (FieldOnlyFrozenDataclassCandidate.from_class(module, node),)
        if candidate is not None
    )


_CONVERSION_FUNCTION_NAME_SEPARATORS = ("_to_", "_from_")


def _conversion_axis_pair(function_name: str) -> tuple[str, str] | None:
    for separator in _CONVERSION_FUNCTION_NAME_SEPARATORS:
        if separator not in function_name:
            continue
        left, right = function_name.split(separator, maxsplit=1)
        if not left or not right:
            continue
        return (left, right) if separator == "_to_" else (right, left)
    return None


def _closed_axis_conversion_matrix_candidates(
    module: ParsedModule,
) -> tuple[ClosedAxisConversionMatrixCandidate, ...]:
    module_stem = module.path.stem.lower()
    module_declares_conversion_domain = any(
        (term in module_stem for term in ("conversion", "converter", "convert"))
    )
    conversion_functions: list[tuple[str, int, int, str, str]] = []
    for node in module.module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (
            module_declares_conversion_domain
            or node.name.startswith(("convert_", "converter_"))
        ):
            continue
        axis_pair = _conversion_axis_pair(node.name)
        if axis_pair is None:
            continue
        source_axis_value, target_axis_value = axis_pair
        conversion_functions.append(
            (
                node.name,
                node.lineno,
                (node.end_lineno or node.lineno) - node.lineno + 1,
                source_axis_value,
                target_axis_value,
            )
        )
    if len(conversion_functions) < 4:
        return ()
    source_values = sorted_tuple({item[3] for item in conversion_functions})
    target_values = sorted_tuple({item[4] for item in conversion_functions})
    if len(source_values) < 2 or len(target_values) < 2:
        return ()
    matrix_capacity = len(source_values) * len(target_values)
    if len(conversion_functions) < min(4, matrix_capacity):
        return ()
    function_names = tuple((item[0] for item in conversion_functions))
    line_numbers = tuple((item[1] for item in conversion_functions))
    return (
        ClosedAxisConversionMatrixCandidate(
            file_path=module.file_path,
            line=min(line_numbers),
            function_names=function_names,
            source_axis_values=source_values,
            target_axis_values=target_values,
            line_numbers=line_numbers,
            line_count=sum((item[2] for item in conversion_functions)),
        ),
    )


def _method_body_fingerprint(method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = statements_without_docstring(method.body)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


_NODE_VISITOR_BASE_NAME = "NodeVisitor"
_VISITOR_METHOD_PREFIX = "visit_"
_VISITOR_STACK_SUFFIX = "_stack"


def _iter_scoped_class_defs(
    statements: Sequence[ast.stmt], scope: tuple[str, ...] = ()
) -> tuple[tuple[str, ast.ClassDef], ...]:
    class_defs: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            qualname = ".".join((*scope, statement.name))
            class_defs.append((qualname, statement))
            class_defs.extend(
                _iter_scoped_class_defs(statement.body, (*scope, statement.name))
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            class_defs.extend(
                _iter_scoped_class_defs(statement.body, (*scope, statement.name))
            )
        else:
            class_defs.extend(
                _iter_scoped_class_defs(
                    tuple(
                        (
                            child
                            for child in ast.iter_child_nodes(statement)
                            if isinstance(child, ast.stmt)
                        )
                    ),
                    scope,
                )
            )
    return tuple(class_defs)


def _inherits_node_visitor(node: ast.ClassDef) -> bool:
    return any(
        (
            (
                (base_chain := AstExpressionProjection.attribute_chain(base))
                is not None
            )
            and base_chain[-1] == _NODE_VISITOR_BASE_NAME
            for base in node.bases
        )
    )


def _self_attribute_name_from_target(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _is_empty_list_value(node: ast.AST | None) -> bool:
    return isinstance(node, ast.List) and len(node.elts) == 0


def _assigned_self_stack_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> tuple[str, ...]:
    if method is None:
        return ()
    stack_names: set[str] = set()
    for statement in statements_without_docstring(method.body):
        if isinstance(statement, ast.AnnAssign) and _is_empty_list_value(
            statement.value
        ):
            target_name = _self_attribute_name_from_target(statement.target)
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and _is_empty_list_value(statement.value)
        ):
            target_name = _self_attribute_name_from_target(statement.targets[0])
        else:
            target_name = None
        if target_name is not None and target_name.endswith(_VISITOR_STACK_SUFFIX):
            stack_names.add(target_name)
    return sorted_tuple(stack_names)


def _self_stack_call_name(call: ast.Call, method_name: str) -> str | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != method_name:
        return None
    stack_expr = call.func.value
    if not isinstance(stack_expr, ast.Attribute):
        return None
    if not isinstance(stack_expr.value, ast.Name) or stack_expr.value.id != "self":
        return None
    return stack_expr.attr


def _is_node_name_argument(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "node"
    )


def _visitor_stack_transition_names(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    stack_names: frozenset[str],
) -> tuple[str, ...]:
    appended: set[str] = set()
    popped: set[str] = set()
    for statement in statements_without_docstring(method.body):
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        append_stack = _self_stack_call_name(call, _APPEND_METHOD_NAME)
        if (
            append_stack in stack_names
            and len(call.args) == 1
            and _is_node_name_argument(call.args[0])
        ):
            appended.add(append_stack)
            continue
        pop_stack = _self_stack_call_name(call, "pop")
        if pop_stack in stack_names and not call.args and not call.keywords:
            popped.add(pop_stack)
    return sorted_tuple(appended & popped)


def _node_visitor_stack_boilerplate_candidates(
    module: ParsedModule,
) -> tuple[NodeVisitorStackBoilerplateCandidate, ...]:
    candidates: list[NodeVisitorStackBoilerplateCandidate] = []
    for qualname, node in _iter_scoped_class_defs(module.module.body):
        if (not _inherits_node_visitor(node)) or CLASS_NODE_AUTHORITY.is_abstract(node):
            continue
        stack_names = _assigned_self_stack_names(
            CLASS_NODE_AUTHORITY.method_named(node, "__init__")
        )
        if not stack_names:
            continue
        transitions_by_method = {
            method.name: _visitor_stack_transition_names(method, frozenset(stack_names))
            for method in CLASS_NODE_AUTHORITY.methods(node)
            if method.name.startswith(_VISITOR_METHOD_PREFIX)
        }
        transition_stack_names = sorted_tuple(
            {
                stack_name
                for stack_names_for_method in transitions_by_method.values()
                for stack_name in stack_names_for_method
            }
        )
        transition_method_names = sorted_tuple(
            (
                method_name
                for method_name, method_stack_names in transitions_by_method.items()
                if method_stack_names
            )
        )
        if len(transition_stack_names) < 2:
            continue
        candidates.append(
            NodeVisitorStackBoilerplateCandidate(
                file_path=module.file_path,
                line=node.lineno,
                qualname=qualname,
                stack_names=transition_stack_names,
                transition_method_names=transition_method_names,
                line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
            )
        )
    return tuple(candidates)


def _enum_metadata_table_cases(module: ast.Module) -> dict[str, tuple[str, int]]:
    tables: dict[str, tuple[str, int]] = {}
    for statement in module.body:
        binding = named_value_binding(statement)
        value = as_ast(None if binding is None else binding.value, ast.Dict)
        if binding is None or value is None:
            continue
        enum_key_names = tuple(
            (
                key.value.id
                for key in value.keys
                if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
            )
        )
        enum_names = set(enum_key_names)
        if len(enum_key_names) == len(value.keys) and len(enum_names) == 1:
            tables[binding.name] = (next(iter(enum_names)), len(value.keys))
    return tables


def _enum_metadata_property_table(method: ast.FunctionDef) -> str | None:
    returned = single_return_value(statements_without_docstring(method.body))
    lookup_source = returned.value if isinstance(returned, ast.Attribute) else returned
    lookup = as_ast(lookup_source, ast.Subscript)
    table_name = name_id(None if lookup is None else lookup.value)
    property_method = any(
        (name_id(decorator) == "property" for decorator in method.decorator_list)
    )
    return (
        table_name
        if property_method and lookup is not None and (name_id(lookup.slice) == "self")
        else None
    )


def _enum_metadata_table_candidates(
    module: ParsedModule,
) -> tuple[EnumMetadataTableCandidate, ...]:
    table_cases = _enum_metadata_table_cases(module.module)
    candidates: list[EnumMetadataTableCandidate] = []
    for statement in module.module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        property_tables: dict[str, list[str]] = defaultdict(list)
        for item in statement.body:
            if isinstance(item, ast.FunctionDef):
                table_name = _enum_metadata_property_table(item)
                if table_name is not None:
                    property_tables[table_name].append(item.name)
        for table_name, property_names in property_tables.items():
            enum_name, case_count = table_cases.get(table_name, (None, 0))
            if enum_name == statement.name:
                candidates.append(
                    EnumMetadataTableCandidate(
                        file_path=module.file_path,
                        line=statement.lineno,
                        class_name=statement.name,
                        table_name=table_name,
                        property_names=tuple(property_names),
                        case_count=case_count,
                    )
                )
    return tuple(candidates)


_TUPLE_INDEX_OPACITY_CARRIER_CALLS = frozenset(
    {"Maybe.of", "project", "map", "filter", "with_projection", "bind_all"}
)


def _numeric_subscript_path(node: ast.AST) -> tuple[str, tuple[int, ...]] | None:
    indexes: list[int] = []
    current = node
    while isinstance(current, ast.Subscript):
        if not isinstance(current.slice, ast.Constant) or not isinstance(
            current.slice.value, int
        ):
            return None
        indexes.append(current.slice.value)
        current = current.value
    root_name = name_id(current)
    if root_name is None:
        return None
    return root_name, tuple(reversed(indexes))


def _tuple_index_semantic_opacity_candidates(
    module: ParsedModule,
) -> tuple[TupleIndexSemanticOpacityCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.named_function_candidates(
        module,
        _tuple_index_semantic_opacity_candidate_for_function,
        sort_key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.function_name,
        ),
    )


def _tuple_index_semantic_opacity_candidate_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
) -> tuple[TupleIndexSemanticOpacityCandidate, ...]:
    body_nodes = walk_function_body_nodes(function)
    carrier_call_names = sorted_tuple(
        {
            call_name
            for node in body_nodes
            if isinstance(node, ast.Call)
            for call_name in (AstExpressionProjection.terminal_name(node.func),)
            if call_name in _TUPLE_INDEX_OPACITY_CARRIER_CALLS
        }
    )
    index_paths = sorted_tuple(
        {
            f"{root}[{']['.join(str(index) for index in indexes)}]"
            for node in body_nodes
            if isinstance(node, ast.Subscript)
            for path in (_numeric_subscript_path(node),)
            if path is not None
            for root, indexes in (path,)
            if root not in {"args", "body", "items"}
            if len(indexes) >= 2 or root in {"pair", "pairs", "call_pair"}
        }
    )
    if not carrier_call_names or not index_paths:
        return ()
    return (
        TupleIndexSemanticOpacityCandidate(
            file_path=module.file_path,
            line=function.lineno,
            function_name=qualname,
            index_expressions=index_paths,
            nested_index_count=len(index_paths),
            carrier_call_names=carrier_call_names,
        ),
    )


def _dataclass_config_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    if not any(
        (
            name_id(decorator) == "dataclass"
            or (
                isinstance(decorator, ast.Call)
                and name_id(decorator.func) == "dataclass"
            )
            for decorator in node.decorator_list
        )
    ):
        return ()
    return tuple(
        (
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        )
    )


def _from_namespace_keyword_names(
    node: ast.ClassDef,
) -> tuple[int, tuple[str, ...]] | None:
    for statement in node.body:
        if (
            not isinstance(statement, ast.FunctionDef)
            or statement.name != "from_namespace"
        ):
            continue
        for call in _walk_nodes(statement):
            if isinstance(call, ast.Call) and name_id(call.func) == "cls":
                keyword_names = tuple(
                    (
                        keyword.arg
                        for keyword in call.keywords
                        if keyword.arg is not None
                    )
                )
                if keyword_names:
                    return (statement.lineno, keyword_names)
    return None


def _argument_spec_field_name(node: ast.AST) -> str | None:
    return (
        Maybe.of(as_ast(node, ast.Call))
        .filter(lambda call: (name_id(call.func) or "").endswith("ArgumentSpec"))
        .map(lambda call: {keyword.arg: keyword.value for keyword in call.keywords})
        .combine(
            lambda keywords: as_ast(keywords.get("flags"), ast.Tuple),
            lambda keywords, flags: (keywords, flags) if flags.elts else None,
        )
        .combine(
            lambda context: constant_value(context[1].elts[0]),
            lambda context, first_flag: (
                (
                    dest_name
                    if isinstance(
                        dest_name := constant_value(context[0].get("dest")), str
                    )
                    else first_flag.removeprefix("--").replace("-", "_")
                )
                if isinstance(first_flag, str) and first_flag.startswith("--")
                else None
            ),
        )
        .unwrap_or_none()
    )


def _cli_argument_spec_fields(
    module: ParsedModule,
) -> NamedStringSequenceSpecs:
    specs: MutableNamedStringSequenceSpecs = []
    for statement in module.module.body:
        binding = named_value_binding(statement)
        if binding is None or not isinstance(binding.value, ast.Tuple):
            continue
        field_names = tuple(
            (
                field_name
                for field_name in (
                    _argument_spec_field_name(element) for element in binding.value.elts
                )
                if field_name is not None
            )
        )
        if field_names:
            specs.append((binding.name, binding.line, field_names))
    return tuple(specs)


def _assignment_target_arity(target: ast.AST) -> int | None:
    if isinstance(target, ast.Name):
        return 1
    if isinstance(target, (ast.Tuple, ast.List)):
        if not target.elts or not all(
            (isinstance(item, ast.Name) for item in target.elts)
        ):
            return None
        return len(target.elts)
    return None


def _result_assembly_pipeline_functions(
    module: ParsedModule,
) -> tuple[ResultAssemblyPipelineFunction, ...]:
    functions: list[ResultAssemblyPipelineFunction] = []
    for qualname, function in _iter_named_functions(module):
        stages = _pipeline_body_stages(function)
        if stages is None:
            continue
        functions.append(
            ResultAssemblyPipelineFunction(
                file_path=module.file_path,
                qualname=qualname,
                lineno=function.lineno,
                stages=stages,
            )
        )
    return sorted_tuple(functions, key=lambda item: (item.lineno, item.qualname))


def _shared_pipeline_tail(
    left: ResultAssemblyPipelineFunction,
    right: ResultAssemblyPipelineFunction,
) -> tuple[PipelineAssemblyStage, ...]:
    shared: list[PipelineAssemblyStage] = []
    left_index = len(left.stages) - 1
    right_index = len(right.stages) - 1
    while left_index >= 0 and right_index >= 0:
        left_stage = left.stages[left_index]
        right_stage = right.stages[right_index]
        if left_stage.shape_key != right_stage.shape_key:
            break
        shared.append(left_stage)
        left_index -= 1
        right_index -= 1
    return tuple(reversed(shared))


def _repeated_result_assembly_pipeline_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[RepeatedResultAssemblyPipelineCandidate, ...]:
    functions = _result_assembly_pipeline_functions(module)
    if len(functions) < 2:
        return ()
    grouped_functions: dict[
        (
            tuple[tuple[object, ...], ...],
            tuple[
                tuple[PipelineAssemblyStage, ...], set[ResultAssemblyPipelineFunction]
            ],
        )
    ] = {}
    for left, right in combinations(functions, 2):
        shared_tail = _shared_pipeline_tail(left, right)
        if len(shared_tail) < config.min_shared_pipeline_stages:
            continue
        if len(shared_tail) >= len(left.stages) or len(shared_tail) >= len(
            right.stages
        ):
            continue
        if shared_tail[-1].kind != _PIPELINE_RETURN_STAGE:
            continue
        distinct_stage_names = {stage.callee_name for stage in shared_tail}
        if len(distinct_stage_names) < config.min_shared_pipeline_stages - 1:
            continue
        key = tuple(stage.shape_key for stage in shared_tail)
        if key not in grouped_functions:
            grouped_functions[key] = (shared_tail, set())
        grouped_functions[key][1].update((left, right))

    candidates = [
        RepeatedResultAssemblyPipelineCandidate(
            file_path=module.file_path,
            shared_tail=shared_tail,
            functions=sorted_tuple(
                grouped, key=lambda item: (item.lineno, item.qualname)
            ),
        )
        for shared_tail, grouped in grouped_functions.values()
        if len(grouped) >= 2
    ]
    filtered_candidates: list[RepeatedResultAssemblyPipelineCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -len(item.shared_tail),
            -len(item.functions),
            item.functions[0].qualname,
        ),
    ):
        candidate_function_names = tuple(
            (function.qualname for function in candidate.functions)
        )
        if any(
            (
                len(existing.shared_tail) >= len(candidate.shared_tail)
                and candidate_function_names
                == tuple((function.qualname for function in existing.functions))
                for existing in filtered_candidates
            )
        ):
            continue
        filtered_candidates.append(candidate)
    return tuple(filtered_candidates)


_SCHEMA_ACCESSOR_FETCH_METHOD_NAMES = frozenset({"required", "optional"})
_SCHEMA_ACCESSOR_RUNTIME_GUARD_CALL_NAMES = frozenset({"isinstance"})
_SCHEMA_ACCESSOR_MIN_METHODS = 4
_SCHEMA_ACCESSOR_COPY_CALL_NAMES = BuiltinCallName.schema_accessor_copy_call_names()
_SCHEMA_ACCESSOR_SELF_NAME = "self"


def _schema_accessor_self_fetch_func(call: ast.Call) -> ast.Attribute | None:
    return (
        Maybe.of(as_ast(call.func, ast.Attribute))
        .filter(
            lambda func: (
                isinstance(func.value, ast.Name)
                and func.value.id == _SCHEMA_ACCESSOR_SELF_NAME
            )
        )
        .unwrap_or_none()
    )


def _schema_accessor_fetch_call(value: ast.AST) -> tuple[str, str, str] | None:
    return (
        Maybe.of(as_ast(value, ast.Call))
        .filter(lambda call: len(call.args) == 1 and not call.keywords)
        .combine(
            _schema_accessor_self_fetch_func,
            lambda call, func: (call, func),
        )
        .filter(lambda context: context[1].attr in _SCHEMA_ACCESSOR_FETCH_METHOD_NAMES)
        .combine(
            lambda context: AstExpressionProjection.attribute_chain(
                context[0].args[0]
            ),
            lambda context, field_chain: (context[1], field_chain),
        )
        .filter(lambda context: len(context[1]) >= 2)
        .map(
            lambda context: (
                context[0].attr,
                ".".join(context[1][:-1]),
                context[1][-1],
            )
        )
        .unwrap_or_none()
    )


def _schema_accessor_fetch_assignment(
    statement: ast.stmt,
) -> tuple[str, str, str, str] | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            return None
        target_name = statement.targets[0].id
        fetch = _schema_accessor_fetch_call(statement.value)
    elif isinstance(statement, ast.AnnAssign) and isinstance(
        statement.target, ast.Name
    ):
        if statement.value is None:
            return None
        target_name = statement.target.id
        fetch = _schema_accessor_fetch_call(statement.value)
    else:
        return None
    if fetch is None:
        return None
    fetch_mode, enum_name, field_name = fetch
    return target_name, fetch_mode, enum_name, field_name


def _node_mentions_name(node: ast.AST, local_name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == local_name
        for item in _walk_nodes(node)
    )


def _compare_mentions_none(node: ast.Compare, local_name: str) -> bool:
    operands = (node.left, *node.comparators)
    return _node_mentions_name(node, local_name) and any(
        isinstance(operand, ast.Constant) and operand.value is None
        for operand in operands
    )


def _schema_accessor_coercion_kinds(
    function: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> tuple[str, ...]:
    kinds: set[str] = set()
    for node in _walk_nodes(function):
        if isinstance(node, ast.Compare) and _compare_mentions_none(node, local_name):
            kinds.add("none_guard")
        if not isinstance(node, ast.Call):
            continue
        call_name = AstExpressionProjection.terminal_name(node.func)
        if (
            call_name in _SCHEMA_ACCESSOR_RUNTIME_GUARD_CALL_NAMES
            and node.args
            and _node_mentions_name(node.args[0], local_name)
        ):
            kinds.add("runtime_type_guard")
        elif call_name in _SCHEMA_ACCESSOR_COPY_CALL_NAMES and any(
            _node_mentions_name(argument, local_name) for argument in node.args
        ):
            kinds.add(f"{call_name}_coercion")
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("from_")
            and any(_node_mentions_name(argument, local_name) for argument in node.args)
        ):
            kinds.add(node.func.attr)
    return tuple(sorted(kinds))


def _schema_accessor_returns_local(
    function: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> bool:
    return any(
        isinstance(node, ast.Return)
        and node.value is not None
        and _node_mentions_name(node.value, local_name)
        for node in _walk_nodes(function)
    )


def _schema_accessor_method_row(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str, str, str, int, int] | None:
    if method.name in _SCHEMA_ACCESSOR_FETCH_METHOD_NAMES:
        return None
    if _is_private_symbol_name(method.name):
        return None
    body = statements_without_docstring(list(method.body))
    fetches = tuple(
        fetch
        for statement in body
        if (fetch := _schema_accessor_fetch_assignment(statement)) is not None
    )
    if len(fetches) != 1:
        return None
    local_name, fetch_mode, enum_name, field_name = fetches[0]
    coercion_kinds = _schema_accessor_coercion_kinds(method, local_name)
    if not coercion_kinds or not _schema_accessor_returns_local(method, local_name):
        return None
    line_count = max(1, (method.end_lineno or method.lineno) - method.lineno + 1)
    return (
        method.name,
        enum_name,
        field_name,
        fetch_mode,
        "+".join(coercion_kinds),
        method.lineno,
        line_count,
    )


def _schema_accessor_axis_system(
    rows: tuple[tuple[str, str, str, str, str, int, int], ...],
) -> FiniteAxisSystem[str, str]:
    return FiniteAxisSystem.from_rows(
        (
            (
                method_name,
                {
                    "method": method_name,
                    "enum": enum_name,
                    "field": field_name,
                    "fetch_mode": fetch_mode,
                    "coercion": coercion_kind,
                },
            )
            for (
                method_name,
                enum_name,
                field_name,
                fetch_mode,
                coercion_kind,
                _line,
                _line_count,
            ) in rows
        )
    )


def _schema_accessor_family_certificate(
    *,
    method_count: int,
    line_count: int,
    field_axes: tuple[str, ...],
    coercion_kind_count: int,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(line_count, method_count * max(coercion_kind_count, 1)),
        replacement_shape=ObjectFamilyShape.from_roles(
            ("payload_projection_schema", "typed_payload_projector"),
            axis=("projection_row",),
        ),
        semantic_axes=field_axes,
        residual_object_count=coercion_kind_count,
    )


def _schema_accessor_family_candidates(
    module: ParsedModule,
) -> tuple[SchemaAccessorFamilyCandidate, ...]:
    candidates: list[SchemaAccessorFamilyCandidate] = []
    for class_node in _typed_ast_nodes(module.module, ast.ClassDef):
        grouped_rows: dict[str, list[tuple[str, str, str, str, str, int, int]]] = (
            defaultdict(list)
        )
        for method in CLASS_NODE_AUTHORITY.methods(class_node):
            row = _schema_accessor_method_row(method)
            if row is not None:
                grouped_rows[row[1]].append(row)
        for enum_name, rows in grouped_rows.items():
            if len(rows) < _SCHEMA_ACCESSOR_MIN_METHODS:
                continue
            ordered_rows = tuple(sorted(rows, key=lambda item: (item[5], item[0])))
            field_names = tuple((row[2] for row in ordered_rows))
            if len(frozenset(field_names)) < _SCHEMA_ACCESSOR_MIN_METHODS:
                continue
            axis_system = _schema_accessor_axis_system(ordered_rows)
            if not axis_system.determines(("field",), "method"):
                continue
            line_count = sum((row[6] for row in ordered_rows))
            coercion_kinds = tuple((row[4] for row in ordered_rows))
            field_axes = tuple(
                (f"{enum_name}.{field_name}" for field_name in field_names)
            )
            certificate = _schema_accessor_family_certificate(
                method_count=len(ordered_rows),
                line_count=line_count,
                field_axes=field_axes,
                coercion_kind_count=len(frozenset(coercion_kinds)),
            )
            if not certificate.pays_rent:
                continue
            candidates.append(
                SchemaAccessorFamilyCandidate(
                    file_path=module.file_path,
                    line=class_node.lineno,
                    class_name=class_node.name,
                    enum_name=enum_name,
                    method_names=tuple((row[0] for row in ordered_rows)),
                    field_names=field_names,
                    requirement_modes=tuple((row[3] for row in ordered_rows)),
                    coercion_kinds=coercion_kinds,
                    line_numbers=tuple((row[5] for row in ordered_rows)),
                    line_count=line_count,
                    compression_certificate=certificate,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.class_name,
                item.enum_name,
            ),
        )
    )


def _indexed_family_wrapper_candidates_for_function(
    module: ParsedModule, node: ast.FunctionDef
) -> Iterable[IndexedFamilyWrapperCandidate]:
    del module
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return
    value = node.body[0].value
    if not isinstance(value, ast.ListComp) or len(value.generators) != 1:
        return
    generator = value.generators[0]
    if not isinstance(generator.target, ast.Name) or generator.target.id != "item":
        return
    if not isinstance(generator.iter, ast.Call):
        return
    collector_name = AstExpressionProjection.terminal_name(generator.iter.func)
    if collector_name not in {
        "_collect_items_from_spec_root",
        "collect_family_items",
    }:
        return
    if collector_name == "_collect_items_from_spec_root":
        if len(generator.iter.args) < 3:
            return
        spec_root_name = AstExpressionProjection.terminal_name(generator.iter.args[0])
        item_type_name = AstExpressionProjection.terminal_name(generator.iter.args[2])
    else:
        if len(generator.iter.args) < 2:
            return
        spec_root_name = AstExpressionProjection.terminal_name(generator.iter.args[1])
        item_type_name = AstExpressionProjection.terminal_name(generator.iter.args[1])
    if spec_root_name is None or item_type_name is None:
        return
    if not _is_instance_filter(generator.ifs, item_type_name):
        return
    yield IndexedFamilyWrapperCandidate(
        function_name=node.name,
        lineno=node.lineno,
        collector_name=collector_name,
        spec_root_name=spec_root_name,
        item_type_name=item_type_name,
    )


def _indexed_family_wrapper_candidates(
    module: ParsedModule,
) -> tuple[IndexedFamilyWrapperCandidate, ...]:
    return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
        module,
        module.module,
        ast.FunctionDef,
        _indexed_family_wrapper_candidates_for_function,
        sort_key=lambda item: item.lineno,
    )


def _is_instance_filter(filters: list[ast.expr], item_type_name: str) -> bool:
    for condition in filters:
        if not isinstance(condition, ast.Call):
            continue
        if AstExpressionProjection.terminal_name(condition.func) != "isinstance":
            continue
        if len(condition.args) != 2:
            continue
        if (
            not isinstance(condition.args[0], ast.Name)
            or condition.args[0].id != "item"
        ):
            continue
        if AstExpressionProjection.terminal_name(condition.args[1]) == item_type_name:
            return True
    return False


def _collect_class_sentinel_attrs(
    module: ast.Module,
) -> dict[str, list[SourceLocation]]:
    grouped: dict[str, list[SourceLocation]] = defaultdict(list)
    for node in _walk_nodes(module):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Constant):
                continue
            if not isinstance(stmt.value.value, (str, int, bool)):
                continue
            grouped[target.id].append(
                SourceLocation("<module>", stmt.lineno, f"{node.name}.{target.id}")
            )
    return grouped


def _predicate_factory_chain_branch_count(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int | None:
    current = as_ast(function.body[0], ast.If) if function.body else None
    branches: list[ast.If] = []
    while current is not None:
        branches.append(current)
        current = as_ast(single_item(current.orelse), ast.If)
    if len(branches) < 2:
        return None
    for branch in branches:
        if not _test_has_call(branch.test):
            return None
        if not any((return_call(statement) is not None for statement in branch.body)):
            return None
    return len(branches)


def _test_has_call(node: ast.AST) -> bool:
    return any((isinstance(child, ast.Call) for child in _walk_nodes(node)))


__all__ = tuple(name for name in globals() if not name.startswith("__"))
