"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

import ast
import copy
import os
import re
import tempfile
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, ClassVar, Generic, TypeAlias, TypeVar

from ..factorization import (
    FactorizationEngine,
    FactorizationLattice,
    FactorizationPlan,
)
from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..codemod import CancelableCompositionSignal, detect_cancelable_composition_signals
from ..source_index import build_source_index

from ..record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)

from ._base import *
from ._helpers import *
from ._helpers import (
    _accessor_wrapper_groups,
    _autoregister_meta_rent_candidates,
    _projection_helper_groups,
    _semantic_inheritance_family_ssot_candidates,
)


class _ReplacementShapeRole:
    PROCESS_STAGE_PLAN = object()
    TEXT_REWRITE_PLAN = object()
    BLOCK_ALGEBRA = object()


SemanticBranchObservation: TypeAlias = tuple[int, str, str]
SemanticBranchChain: TypeAlias = tuple[SemanticBranchObservation, ...]
SemanticBranchChains: TypeAlias = tuple[SemanticBranchChain, ...]
ReturnGuardBranchObservation: TypeAlias = tuple[int, str, str, bool]
ReturnGuardBranchChain: TypeAlias = tuple[ReturnGuardBranchObservation, ...]
ReturnGuardBranchChains: TypeAlias = tuple[ReturnGuardBranchChain, ...]
BranchObservationT = TypeVar("BranchObservationT")
BranchChainPredicate: TypeAlias = Callable[[tuple[BranchObservationT, ...]], bool]
BranchLineNumber: TypeAlias = Callable[[BranchObservationT], int]
ElifBranchCollector: TypeAlias = Callable[
    [ast.stmt],
    tuple[BranchObservationT, ...],
]
SequentialBranchCollector: TypeAlias = Callable[
    [Sequence[ast.stmt], int],
    tuple[BranchObservationT, ...],
]


@dataclass(frozen=True)
class BranchChainCollectionSpec(Generic[BranchObservationT]):
    elif_chain: ElifBranchCollector[BranchObservationT]
    sequential_guard_chain: SequentialBranchCollector[BranchObservationT]
    branch_line_number: BranchLineNumber[BranchObservationT]
    chain_is_active: BranchChainPredicate[BranchObservationT]


def iter_nested_statement_bodies(statement: ast.stmt) -> tuple[Sequence[ast.stmt], ...]:
    nested_bodies: list[Sequence[ast.stmt]] = []
    if isinstance(
        statement,
        (
            ast.AsyncFor,
            ast.AsyncFunctionDef,
            ast.AsyncWith,
            ast.ClassDef,
            ast.For,
            ast.FunctionDef,
            ast.If,
            ast.While,
            ast.With,
        ),
    ):
        nested_bodies.append(statement.body)
    if isinstance(statement, (ast.AsyncFor, ast.For, ast.If, ast.While)):
        nested_bodies.append(statement.orelse)
    if isinstance(statement, ast.Try):
        nested_bodies.append(statement.body)
        nested_bodies.append(statement.orelse)
        nested_bodies.append(statement.finalbody)
        nested_bodies.extend(handler.body for handler in statement.handlers)
    if isinstance(statement, ast.Match):
        nested_bodies.extend(match_case.body for match_case in statement.cases)
    return tuple(nested_bodies)


def branch_observation_first_line(observation: Sequence[object]) -> int:
    line_number = observation[0]
    if not isinstance(line_number, int):
        raise TypeError("branch observation line number must be an int")
    return line_number


def all_branch_chains_active(chain: tuple[BranchObservationT, ...]) -> bool:
    return True


def return_guard_chain_has_literal_default(chain: ReturnGuardBranchChain) -> bool:
    return any(
        (is_literal_default for _line, _test, _result, is_literal_default in chain)
    )


def collect_nested_branch_chains_from_body(
    body: Sequence[ast.stmt],
    spec: BranchChainCollectionSpec[BranchObservationT],
) -> tuple[tuple[BranchObservationT, ...], ...]:
    trimmed_body = tuple(_trim_docstring_body(tuple(body)))
    chains: list[tuple[BranchObservationT, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for index, statement in enumerate(trimmed_body):
        for chain in (
            spec.elif_chain(statement),
            spec.sequential_guard_chain(trimmed_body, index),
        ):
            if not chain or not spec.chain_is_active(chain):
                continue
            line_key = tuple(
                (spec.branch_line_number(observation) for observation in chain)
            )
            if line_key in seen:
                continue
            seen.add(line_key)
            chains.append(chain)

        for nested_body in iter_nested_statement_bodies(statement):
            chains.extend(collect_nested_branch_chains_from_body(nested_body, spec))
    return tuple(chains)


_REPLACEMENT_SHAPE_ROWS = (
    (
        _ReplacementShapeRole.PROCESS_STAGE_PLAN,
        ObjectFamilyShape(
            shared_objects=("process_stage_plan", "stage_runner"),
            per_axis_objects=("stage_step",),
        ),
    ),
    (
        _ReplacementShapeRole.TEXT_REWRITE_PLAN,
        ObjectFamilyShape(
            shared_objects=("text_rewrite_plan", "file_application_surface"),
            per_axis_objects=("file_collection",),
        ),
    ),
    (
        _ReplacementShapeRole.BLOCK_ALGEBRA,
        ObjectFamilyShape(
            shared_objects=("block_algebra", "block_runner"),
            per_source_objects=("context_row",),
        ),
    ),
)


@dataclass(frozen=True)
class ReplacementShapeProjector:
    rows: tuple[tuple[object, ObjectFamilyShape], ...]

    def shape_for(self, role: object) -> ObjectFamilyShape:
        return next(
            (
                replacement_shape
                for candidate_role, replacement_shape in self.rows
                if candidate_role is role
            )
        )


_REPLACEMENT_SHAPE_PROJECTOR = ReplacementShapeProjector(_REPLACEMENT_SHAPE_ROWS)


def _manual_process_step_ladder_compression_certificate(
    candidate: ManualProcessStepLadderCandidate,
) -> CompressionCertificate:
    table_count = len(candidate.step_table_names)
    step_count = max(candidate.minimum_step_count, 1)
    return CompressionCertificate.from_object_family(
        manual_object_count=table_count * (step_count + 1),
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.PROCESS_STAGE_PLAN
        ),
        semantic_axes=tuple((f"step:{index}" for index in range(step_count))),
    )


def _mirrored_file_rewrite_loop_compression_certificate(
    candidate: MirroredFileRewriteLoopCandidate,
) -> CompressionCertificate:
    loop_count = len(candidate.line_numbers)
    return CompressionCertificate.from_object_family(
        manual_object_count=loop_count * 4,
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.TEXT_REWRITE_PLAN
        ),
        semantic_axes=tuple(
            (f"file_collection:{index}" for index in range(loop_count))
        ),
    )


def _algebraic_duplicate_compound_block_compression_certificate(
    candidate: AlgebraicDuplicateCompoundBlockCandidate,
) -> CompressionCertificate:
    source_count = len(candidate.function_names)
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.normal_form_size * source_count,
            source_count * 4,
        ),
        replacement_shape=_REPLACEMENT_SHAPE_PROJECTOR.shape_for(
            _ReplacementShapeRole.BLOCK_ALGEBRA
        ),
        semantic_axes=(candidate.block_kind,),
        independent_source_count=source_count,
    )


def _literal_dispatch_authority_name(axis_expression: str) -> str:
    words = "".join(
        (character if character.isalnum() else "_" for character in axis_expression)
    ).strip("_")
    return f"dispatch_{words or 'case'}"


def _literal_dispatch_case_class_name(literal_case: str, index: int) -> str:
    words = "".join(
        (
            character if character.isalnum() else "_"
            for character in literal_case.strip("'\"")
        )
    )
    return f"{_camel_case(words) or f'Case{index}'}DispatchCase"


def _literal_dispatch_authority_patch(
    observation: LiteralDispatchObservation,
) -> str:
    return f"# Replace the repeated `{observation.axis_expression} == literal` branches with one AutoRegisterMeta-backed case family.\n# Move per-case behavior into `DispatchCase` subclasses keyed by the same axis.\n# Dispatch through `DispatchCase.for_case(...)` / `DispatchCase.__registry__` instead of if/elif or match/case."


class LiteralDispatchFindingFactory:
    def authority_scaffold(self, observation: LiteralDispatchObservation) -> str:
        dispatch_name = _literal_dispatch_authority_name(observation.axis_expression)
        case_classes = tuple(
            (
                _literal_dispatch_case_class_name(case, index)
                for index, case in enumerate(observation.literal_cases, start=1)
            )
        )
        case_class_blocks = "\n\n".join(
            (
                f"class {class_name}(DispatchCase):\n    case = {case}\n\n    def apply(self, *args, **kwargs):\n        ..."
                for class_name, case in zip(case_classes, observation.literal_cases)
            )
        )
        return (
            "from abc import ABC, abstractmethod\n"
            "from typing import ClassVar\n"
            "from metaclass_registry import AutoRegisterMeta\n\n"
            "class DispatchCase(ABC, metaclass=AutoRegisterMeta):\n"
            '    __registry_key__ = "case"\n'
            "    __skip_if_no_key__ = True\n"
            "    case: ClassVar[object] = None\n\n"
            "    @classmethod\n"
            "    def for_case(cls, key):\n"
            "        return cls.__registry__[key]()\n\n"
            "    @abstractmethod\n"
            "    def apply(self, *args, **kwargs): ...\n\n"
            f"{case_class_blocks}\n\n"
            f"def {dispatch_name}(axis_value, *args, **kwargs):\n"
            "    return DispatchCase.for_case(axis_value).apply(*args, **kwargs)"
        )

    def finding(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        observation: LiteralDispatchObservation,
        *,
        case_summary_label: str,
        relation_case_label: str,
    ) -> RefactorFinding:
        return detector.build_finding(
            f"{module.path} dispatches on `{observation.axis_expression}` through {case_summary_label} {observation.literal_cases}.",
            (
                SourceLocation(
                    observation.file_path, observation.line, observation.symbol
                ),
            ),
            relation_context=(
                f"same observed axis `{observation.axis_expression}` is split across {relation_case_label} {observation.literal_cases}"
            ),
            scaffold=self.authority_scaffold(observation),
            codemod_patch=_literal_dispatch_authority_patch(observation),
            metrics=DispatchCountMetrics.from_literal_family(
                observation.axis_expression,
                observation.literal_cases,
            ),
        )

    def findings(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        config: DetectorConfig,
        observation_family: type[object],
        *,
        case_summary_label: str,
        relation_case_label: str,
    ) -> list[RefactorFinding]:
        observations: tuple[LiteralDispatchObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module,
                observation_family,
                LiteralDispatchObservation,
            )
        )
        return [
            self.finding(
                detector,
                module,
                observation,
                case_summary_label=case_summary_label,
                relation_case_label=relation_case_label,
            )
            for observation in observations
            if len(observation.literal_cases) >= config.min_string_cases
        ]


LITERAL_DISPATCH_FINDING_FACTORY = LiteralDispatchFindingFactory()


def _mirrored_validation_call(value: ast.AST) -> tuple[str, str] | None:
    if not isinstance(value, ast.Call) or len(value.args) < 2:
        return None
    literal = value.args[0]
    source = value.args[1]
    if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
        return None
    if not isinstance(source, ast.Name):
        return None
    if literal.value != source.id:
        return None
    return literal.value, ast.unparse(value.func)


def _constructor_name(value: ast.AST) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        owner = _constructor_name(value.value)
        return f"{owner}.{value.attr}" if owner else value.attr
    return ""


def _literal_default_kind(value: ast.AST) -> str | None:
    if isinstance(value, ast.Constant):
        if value.value is None:
            return "none"
        if isinstance(value.value, bool):
            return "bool"
        if isinstance(value.value, (int, float, complex, str, bytes)):
            return type(value.value).__name__
    if isinstance(value, ast.List) and not value.elts:
        return "empty_list"
    if isinstance(value, ast.Tuple) and not value.elts:
        return "empty_tuple"
    if isinstance(value, ast.Dict) and not value.keys:
        return "empty_dict"
    if isinstance(value, ast.Set) and not value.elts:
        return "empty_set"
    return None


def _call_fallback_kind(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        if node.func.id == "getattr" and len(node.args) >= 3:
            return "getattr_default"
        if node.func.id == "next" and len(node.args) >= 2:
            return "next_default"
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "get" and len(node.args) >= 2:
            if _is_class_namespace_get_default(node):
                return None
            return "mapping_get_default"
        if node.func.attr == "setdefault":
            if _is_class_namespace_setdefault(node):
                return None
            return "mapping_setdefault"
    if any((keyword.arg == "default" for keyword in node.keywords)):
        return "keyword_default"
    return None


def _mapping_receiver_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


def _is_class_namespace_get_default(node: ast.Call) -> bool:
    receiver_name = _mapping_receiver_name(node)
    key_name = _constant_string(node.args[0]) if node.args else None
    return (
        receiver_name is not None
        and key_name is not None
        and _is_class_namespace_mapping_key(receiver_name, key_name)
    )


def _is_class_namespace_setdefault(node: ast.Call) -> bool:
    receiver_name = _mapping_receiver_name(node)
    return receiver_name in _CLASS_NAMESPACE_MAPPING_NAMES


def _default_ifexp_kind(node: ast.IfExp) -> str | None:
    if _is_optional_none_projection_ifexp(node):
        return None
    if (kind := _literal_default_kind(node.orelse)) is not None:
        return f"ifexp_else_{kind}"
    if (kind := _literal_default_kind(node.body)) is not None:
        return f"ifexp_body_{kind}"
    return None


def _is_optional_none_projection_ifexp(node: ast.IfExp) -> bool:
    has_none_body = _is_none_literal(node.body)
    has_none_orelse = _is_none_literal(node.orelse)
    if has_none_body == has_none_orelse:
        return False
    return _is_optional_projection_guard(node.test)


def _is_none_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_optional_projection_guard(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and _call_name(node.func) == "isinstance":
        return len(node.args) >= 2
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops) and any(
            _is_none_literal(comparator) for comparator in node.comparators
        )
    return False


def _boolop_default_kind(node: ast.BoolOp) -> str | None:
    if not isinstance(node.op, ast.Or):
        return None
    literal_kinds = tuple(
        kind
        for value in node.values[1:]
        for kind in (_literal_default_kind(value),)
        if kind is not None
    )
    if not literal_kinds:
        return None
    return "or_default_" + "_".join(sorted_tuple(set(literal_kinds)))


def _fallback_owner(
    class_stack: Sequence[str],
    function_stack: Sequence[str],
) -> str:
    owner_parts = (*tuple(class_stack), *tuple(function_stack))
    return ".".join(owner_parts) if owner_parts else "module"


_PRIVATE_OBJECT_BOUNDARY_FIELD_TOKENS = frozenset(
    (
        "callback",
        "callable",
        "executor",
        "function",
        "handler",
        "impl",
        "materializer",
        "predicate",
        "provider",
        "resolver",
        "runtime",
    )
)


def _private_boundary_identifier_tokens(text: str) -> tuple[str, ...]:
    normalized = "".join(
        (character.lower() if character.isalnum() else "_") for character in text
    )
    return tuple((token for token in normalized.split("_") if token))


def _is_exact_object_annotation(annotation: ast.AST) -> bool:
    return (isinstance(annotation, ast.Name) and annotation.id == "object") or (
        isinstance(annotation, ast.Attribute) and annotation.attr == "object"
    )


_OPAQUE_OBJECT_ANNOTATION_NAMES = frozenset(("Any", "object"))
_SMELLY_TYPE_ALIAS_BARE_ROOT_NAMES = frozenset(("ABC", "Enum", "Protocol"))
_SMELLY_TYPE_ALIAS_CALLABLE_ROOT_NAMES = frozenset(("Callable",))
_SMELLY_TYPE_ALIAS_MAPPING_ROOT_NAMES = frozenset(
    ("Dict", "Mapping", "MutableMapping", "dict")
)
_SMELLY_TYPE_ALIAS_STRUCTURAL_NAME_TOKENS = frozenset(
    (
        "by",
        "dict",
        "dicts",
        "group",
        "groups",
        "index",
        "indexes",
        "indices",
        "key",
        "keys",
        "list",
        "lists",
        "map",
        "maps",
        "mapping",
        "mappings",
        "pair",
        "pairs",
        "sequence",
        "sequences",
        "set",
        "sets",
        "spec",
        "specs",
        "table",
        "tables",
        "tuple",
        "tuples",
    )
)


def _opaque_object_annotation_names(annotation: ast.AST) -> tuple[str, ...]:
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id in _OPAQUE_OBJECT_ANNOTATION_NAMES:
                names.append(node.id)
            return
        if isinstance(node, ast.Attribute):
            if node.attr in _OPAQUE_OBJECT_ANNOTATION_NAMES:
                names.append(node.attr)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(annotation)
    return tuple(dict.fromkeys(names))


def _has_opaque_object_annotation(annotation: ast.AST) -> bool:
    return bool(_opaque_object_annotation_names(annotation))


def _annotation_leaf_names(annotation: ast.AST) -> tuple[str, ...]:
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.append(node.id)
            return
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(annotation)
    return tuple(dict.fromkeys(names))


def _is_ellipsis_node(annotation: ast.AST) -> bool:
    return isinstance(annotation, ast.Constant) and annotation.value is Ellipsis


def _callable_has_variadic_parameter_list(annotation: ast.AST) -> bool:
    if not _annotation_has_subscripted_root(
        annotation,
        _SMELLY_TYPE_ALIAS_CALLABLE_ROOT_NAMES,
    ):
        return False
    if not isinstance(annotation, ast.Subscript):
        return False
    slice_node = annotation.slice
    if _is_ellipsis_node(slice_node):
        return True
    if isinstance(slice_node, ast.Tuple) and slice_node.elts:
        return _is_ellipsis_node(slice_node.elts[0])
    return False


def _is_bare_annotation_root(annotation: ast.AST, root_name: str) -> bool:
    return (
        isinstance(annotation, ast.Name)
        and annotation.id == root_name
        or isinstance(annotation, ast.Attribute)
        and annotation.attr == root_name
    )


def _annotation_has_subscripted_root(
    annotation: ast.AST,
    root_names: frozenset[str],
) -> bool:
    if isinstance(annotation, ast.Subscript):
        root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(annotation)
        return root_name in root_names
    return False


def _annotation_contains_subscripted_root(
    annotation: ast.AST,
    root_names: frozenset[str],
) -> bool:
    if _annotation_has_subscripted_root(annotation, root_names):
        return True
    return any(
        _annotation_contains_subscripted_root(child, root_names)
        for child in ast.iter_child_nodes(annotation)
    )


def _annotation_contains_variadic_callable(annotation: ast.AST) -> bool:
    if _callable_has_variadic_parameter_list(annotation):
        return True
    return any(
        _annotation_contains_variadic_callable(child)
        for child in ast.iter_child_nodes(annotation)
    )


def _type_alias_marker_name(annotation: ast.AST) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        return _type_alias_marker_name(annotation.value)
    return None


def _is_type_alias_annotation(annotation: ast.AST) -> bool:
    return _type_alias_marker_name(annotation) == "TypeAlias"


def _split_alias_name_tokens(alias_name: str) -> tuple[str, ...]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", alias_name)
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", spaced).lower()
    return tuple(token for token in normalized.split("_") if token)


def _alias_name_describes_collection_shape(alias_name: str) -> bool:
    tokens = _split_alias_name_tokens(alias_name)
    if not tokens:
        return False
    if any(token in _SMELLY_TYPE_ALIAS_STRUCTURAL_NAME_TOKENS for token in tokens):
        return True
    return any(token.endswith("s") and len(token) > 3 for token in tokens)


def _type_alias_reason_names(alias_name: str, value: ast.AST) -> tuple[str, ...]:
    reason_names: list[str] = []
    leaf_names = frozenset(_annotation_leaf_names(value))
    root_name = HELPER_SYNTAX_PROJECTION_AUTHORITY.annotation_root_name(value)

    if leaf_names & _OPAQUE_OBJECT_ANNOTATION_NAMES:
        reason_names.append("opaque-member")

    for broad_root in sorted(_SMELLY_TYPE_ALIAS_BARE_ROOT_NAMES):
        if _is_bare_annotation_root(value, broad_root):
            reason_names.append("bare-generic-root")
            break

    if _annotation_contains_variadic_callable(value) or (
        _annotation_contains_subscripted_root(
            value,
            _SMELLY_TYPE_ALIAS_CALLABLE_ROOT_NAMES,
        )
        and bool(leaf_names & _OPAQUE_OBJECT_ANNOTATION_NAMES)
    ):
        reason_names.append("opaque-callable")

    if (
        root_name in _SMELLY_TYPE_ALIAS_MAPPING_ROOT_NAMES
        and not _alias_name_describes_collection_shape(alias_name)
    ):
        reason_names.append("semantic-mapping-shell")

    return tuple(dict.fromkeys(reason_names))


def _is_annotation_like_expression(node: ast.AST) -> bool:
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Constant):
        return node.value in (None, Ellipsis)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_annotation_like_expression(element) for element in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_annotation_like_expression(
            node.left
        ) and _is_annotation_like_expression(node.right)
    if isinstance(node, ast.Subscript):
        return _is_annotation_like_expression(
            node.value
        ) and _is_annotation_like_expression(node.slice)
    if isinstance(node, ast.Slice):
        return all(
            (
                bound is None or _is_annotation_like_expression(bound)
                for bound in (node.lower, node.upper, node.step)
            )
        )
    return False


def _is_cast_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "cast"
    return isinstance(node.func, ast.Attribute) and node.func.attr == "cast"


def _is_dataclass_declaration(node: ast.ClassDef) -> bool:
    return any(
        (
            SYNTAX_PROJECTION_AUTHORITY.is_dataclass_decorator(decorator)
            or (
                isinstance(decorator, ast.Call)
                and SYNTAX_PROJECTION_AUTHORITY.is_dataclass_decorator(decorator.func)
            )
        )
        for decorator in node.decorator_list
    )


def _private_object_boundary_fields(
    module: ParsedModule,
) -> dict[str, list[tuple[int, str]]]:
    fields_by_class: dict[str, list[tuple[int, str]]] = {}
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not _is_dataclass_declaration(node):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(
                statement.target,
                ast.Name,
            ):
                continue
            field_name = statement.target.id
            if not field_name.startswith("_"):
                continue
            if not _is_exact_object_annotation(statement.annotation):
                continue
            field_tokens = frozenset(_private_boundary_identifier_tokens(field_name))
            if not (field_tokens & _PRIVATE_OBJECT_BOUNDARY_FIELD_TOKENS):
                continue
            fields_by_class.setdefault(node.name, []).append(
                (int(getattr(statement, "lineno", node.lineno)), field_name)
            )
    return fields_by_class


class PrivateObjectBoundaryFieldDetector(PerModuleIssueDetector):
    detector_id = "private_object_boundary_field"
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Private object-typed boundary field should become a typed authority",
        "A private dataclass field annotated as `object` and named like an executable/runtime boundary hides both ownership and callable shape. That lets local Python closures cross request boundaries without static evidence.",
        "nominal typed authority or protocol field for each executable/runtime boundary",
        "dataclass request boundary stores a private executable/runtime field as `object`",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for class_name, fields in sorted(
            _private_object_boundary_fields(module).items()
        ):
            field_names = tuple(field_name for _line, field_name in fields)
            evidence = tuple(
                SourceLocation(str(module.path), line, f"{class_name}.{field_name}")
                for line, field_name in fields
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{class_name}` stores private runtime boundary field(s) "
                        f"{field_names} as untyped `object`."
                    ),
                    evidence,
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        "class BoundaryRuntime:\n"
                        "    def execute(self, request: BoundaryRequest) -> BoundaryResult: ...\n\n"
                        "@dataclass(frozen=True)\n"
                        "class Request:\n"
                        "    boundary_runtime: BoundaryRuntime"
                    ),
                    codemod_patch=(
                        f"# Replace private object boundary fields on `{class_name}` "
                        "with a named typed authority/protocol field. Do not pass "
                        "private closures through request dataclasses."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(field_names),
                        mapping_name=class_name,
                        field_names=field_names,
                    ),
                )
            )
        return findings


@dataclass(frozen=True)
class OpaqueObjectAnnotationSite:
    owner_name: str
    member_name: str
    role_name: str
    line: int

    @property
    def symbol(self) -> str:
        return f"{self.owner_name}.{self.member_name}"


def _opaque_object_annotation_owner(
    class_stack: Sequence[str],
    function_stack: Sequence[str],
) -> str:
    owner_parts = (*tuple(class_stack), *tuple(function_stack))
    return ".".join(owner_parts) if owner_parts else "module"


def _opaque_object_annotation_sites(
    module: ParsedModule,
) -> dict[str, list[OpaqueObjectAnnotationSite]]:
    sites_by_owner: dict[str, list[OpaqueObjectAnnotationSite]] = defaultdict(list)
    class_stack: list[str] = []
    function_stack: list[str] = []

    def add_site(
        owner_name: str,
        member_name: str,
        role_name: str,
        line: int,
    ) -> None:
        sites_by_owner.setdefault(owner_name, []).append(
            OpaqueObjectAnnotationSite(
                owner_name=owner_name,
                member_name=member_name,
                role_name=role_name,
                line=line,
            )
        )

    def inspect_argument(
        owner_name: str,
        argument: ast.arg | None,
        role_name: str,
    ) -> None:
        if argument is None or argument.arg in {"self", "cls"}:
            return
        if argument.annotation is None or not _has_opaque_object_annotation(
            argument.annotation
        ):
            return
        add_site(owner_name, argument.arg, role_name, int(argument.lineno))

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            owner_name = _opaque_object_annotation_owner(class_stack, function_stack)
            function_owner_name = (
                f"{owner_name}.{node.name}" if owner_name != "module" else node.name
            )
            if node.returns is not None and _has_opaque_object_annotation(node.returns):
                add_site(function_owner_name, "return", "return", int(node.lineno))
            for argument in (
                *tuple(node.args.posonlyargs),
                *tuple(node.args.args),
                *tuple(node.args.kwonlyargs),
            ):
                inspect_argument(function_owner_name, argument, "parameter")
            inspect_argument(function_owner_name, node.args.vararg, "vararg")
            inspect_argument(function_owner_name, node.args.kwarg, "kwarg")
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if (
                function_stack
                or not isinstance(node.target, ast.Name)
                or not _has_opaque_object_annotation(node.annotation)
            ):
                return
            owner_name = _opaque_object_annotation_owner(class_stack, function_stack)
            add_site(owner_name, node.target.id, "field", int(node.lineno))

        def visit_Assign(self, node: ast.Assign) -> None:
            if function_stack:
                self.generic_visit(node)
                return
            if not _is_annotation_like_expression(node.value):
                self.generic_visit(node)
                return
            if not _has_opaque_object_annotation(node.value):
                self.generic_visit(node)
                return
            owner_name = _opaque_object_annotation_owner(class_stack, function_stack)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    add_site(owner_name, target.id, "type_alias", int(node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                _is_cast_call(node)
                and node.args
                and _has_opaque_object_annotation(node.args[0])
            ):
                owner_name = _opaque_object_annotation_owner(
                    class_stack,
                    function_stack,
                )
                add_site(owner_name, "cast", "cast", int(node.lineno))
            self.generic_visit(node)

    Visitor().visit(module.module)
    return sites_by_owner


class OpaqueObjectAnnotationDetector(PerModuleIssueDetector):
    detector_id = "opaque_object_annotation"
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Opaque object-like annotations should become nominal typed contracts",
        "`object` or `Any` inside a boundary annotation gives the same operational permission as an untyped value: callers can pass any shape and failures move to late runtime paths. Boundary code should name the carrier, ABC, or authority it requires.",
        "nominal dataclass/ABC/authority types at every boundary instead of opaque object-like annotations",
        "field, parameter, return, alias, or cast annotations contain `object` or `Any`",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for owner_name, sites in sorted(
            _opaque_object_annotation_sites(module).items()
        ):
            site_names = tuple(site.member_name for site in sites)
            role_names = tuple(sorted_tuple({site.role_name for site in sites}))
            evidence = tuple(
                SourceLocation(str(module.path), site.line, site.symbol)
                for site in sites[:8]
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{owner_name}` exposes opaque object-like annotations for "
                        f"{site_names}; roles={role_names}."
                    ),
                    evidence,
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        "class BoundaryCarrier:\n"
                        "    value: TypedPayload\n\n"
                        "class BoundaryRuntime(ABC):\n"
                        "    @abstractmethod\n"
                        "    def execute(self, request: BoundaryCarrier) -> BoundaryResult:\n"
                        "        raise NotImplementedError"
                    ),
                    codemod_patch=(
                        f"# Replace opaque object-like annotations on `{owner_name}` "
                        "with nominal carrier/ABC/authority types. If the value "
                        "is polymorphic, make the abstract operation explicit on "
                        "a nominal base class."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(sites),
                        mapping_name=owner_name,
                        field_names=site_names,
                    ),
                )
            )
        return findings


@dataclass(frozen=True)
class SmellyTypeAliasSite:
    owner_name: str
    alias_name: str
    value_text: str
    reason_names: tuple[str, ...]
    line: int

    @property
    def symbol(self) -> str:
        if self.owner_name == "module":
            return self.alias_name
        return f"{self.owner_name}.{self.alias_name}"


def _smelly_type_alias_sites(module: ParsedModule) -> tuple[SmellyTypeAliasSite, ...]:
    sites: list[SmellyTypeAliasSite] = []
    class_stack: list[str] = []
    function_stack: list[str] = []

    def owner_name() -> str:
        return ".".join(class_stack) if class_stack else "module"

    def add_alias(alias_name: str, value: ast.AST, line: int) -> None:
        reason_names = _type_alias_reason_names(alias_name, value)
        if not reason_names:
            return
        sites.append(
            SmellyTypeAliasSite(
                owner_name=owner_name(),
                alias_name=alias_name,
                value_text=ast.unparse(value),
                reason_names=reason_names,
                line=line,
            )
        )

    def add_type_stmt_alias(node: ast.AST) -> bool:
        type_alias_type = getattr(ast, "TypeAlias", None)
        if type_alias_type is None or not isinstance(node, type_alias_type):
            return False
        alias_node = getattr(node, "name", None)
        value = getattr(node, "value", None)
        if value is None:
            return True
        if isinstance(alias_node, ast.Name):
            alias_name = alias_node.id
        elif isinstance(alias_node, str):
            alias_name = alias_node
        else:
            return True
        add_alias(alias_name, value, int(getattr(node, "lineno", 0)))
        return True

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if function_stack:
                self.generic_visit(node)
                return
            if (
                isinstance(node.target, ast.Name)
                and node.value is not None
                and _is_type_alias_annotation(node.annotation)
            ):
                add_alias(node.target.id, node.value, int(node.lineno))
                return
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            if function_stack:
                self.generic_visit(node)
                return
            if not _is_annotation_like_expression(node.value):
                self.generic_visit(node)
                return
            for target in node.targets:
                if isinstance(target, ast.Name):
                    add_alias(target.id, node.value, int(node.lineno))
            self.generic_visit(node)

        def visit(self, node: ast.AST) -> None:
            if add_type_stmt_alias(node):
                return
            super().visit(node)

    Visitor().visit(module.module)
    return tuple(sites)


class SmellyTypeAliasDetector(PerModuleIssueDetector):
    detector_id = "smelly_type_alias"
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Type alias erases a semantic boundary",
        "A type alias is useful when it names a repeated structural shape. It becomes debt when the alias only hides an opaque member, a bare generic root, an unbounded callback, or a map-shaped bag that should be a nominal carrier.",
        "nominal carrier/base/enum authority instead of alias-level erasure",
        "type alias declaration expands to an opaque or over-broad structural permission",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for site in _smelly_type_alias_sites(module):
            findings.append(
                self.build_finding(
                    (
                        f"`{site.symbol}` aliases `{site.value_text}` but the "
                        f"alias still carries {site.reason_names}."
                    ),
                    (SourceLocation(str(module.path), site.line, site.symbol),),
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        f"class {site.alias_name.removesuffix('Alias')}:\n"
                        "    ...\n\n"
                        "# Keep raw mapping/callback shapes at parser or adapter "
                        "edges; pass this nominal carrier through the core."
                    ),
                    codemod_patch=(
                        f"# Replace `{site.alias_name} = {site.value_text}` with "
                        "a nominal carrier, closed enum family, or named base "
                        "authority. Preserve structural containers only at IO "
                        "boundaries where they are decoded."
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(site.reason_names),
                        mapping_name=site.symbol,
                        field_names=site.reason_names,
                        source_name="smelly_type_alias",
                    ),
                )
            )
        return findings


@dataclass(frozen=True)
class LiteralSchemaFieldAccess:
    source_expression: str
    key_name: str
    line: int
    access_kind: str


@dataclass(frozen=True)
class LiteralSchemaDispatchOwner:
    qualname: str
    source_expression: str
    key_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    access_count: int


@dataclass(frozen=True)
class LiteralSchemaDispatchCandidate(LineWitnessCandidate):
    function_names: tuple[str, ...]
    source_expressions: tuple[str, ...]
    key_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    access_count: int

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(self.file_path, line, function_name)
            for line, function_name in zip(
                self.line_numbers,
                self.function_names,
                strict=True,
            )
        )


@dataclass(frozen=True)
class StringKeyedFormulaSubclassFamilyCandidate(LineWitnessCandidate):
    base_class_name: str
    key_attr_name: str
    subclass_names: tuple[str, ...]
    key_values: tuple[str, ...]
    method_names: tuple[str, ...]
    expression_snippets: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return self.base_class_name


_STRING_KEYED_FORMULA_ATTR_RE = re.compile(r"^(?:kind|mode|.+_(?:kind|mode))$")
_FORMULA_CALLEE_NAMES = frozenset(
    (
        "abs",
        "all",
        "any",
        "argmax",
        "argmin",
        "array",
        "asarray",
        "clip",
        "concatenate",
        "count_nonzero",
        "flatnonzero",
        "max",
        "mean",
        "min",
        "ones",
        "prod",
        "sum",
        "where",
        "zeros",
    )
)


def _literal_string_key_assignments(node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for statement in node.body:
        targets: tuple[ast.expr, ...]
        value: ast.expr | None
        if isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if _STRING_KEYED_FORMULA_ATTR_RE.match(target.id) is None:
                continue
            rows.append((target.id, value.value))
    return tuple(rows)


def _formula_callee_name(call: ast.Call) -> str | None:
    name = _ast_terminal_name(call.func)
    if name is not None:
        return name
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _function_contains_formula_semantics(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare)):
            return True
        if isinstance(node, ast.Call):
            callee_name = _formula_callee_name(node)
            if callee_name in _FORMULA_CALLEE_NAMES:
                return True
    return False


def _string_keyed_formula_methods(
    node: ast.ClassDef,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if statement.name.startswith("__"):
            continue
        if not _function_contains_formula_semantics(statement):
            continue
        rows.append((statement.name, ast.unparse(statement)))
    return tuple(rows)


def _string_keyed_formula_subclass_family_candidates(
    module: ParsedModule,
) -> tuple[StringKeyedFormulaSubclassFamilyCandidate, ...]:
    classes = {
        node.name: node
        for node in ast.walk(module.module)
        if isinstance(node, ast.ClassDef)
    }
    grouped: dict[
        tuple[str, str],
        list[tuple[ast.ClassDef, str, tuple[tuple[str, str], ...]]],
    ] = defaultdict(list)
    for class_node in classes.values():
        key_assignments = _literal_string_key_assignments(class_node)
        method_rows = _string_keyed_formula_methods(class_node)
        if not key_assignments or not method_rows:
            continue
        for base in class_node.bases:
            base_name = _ast_terminal_name(base)
            if base_name is None:
                continue
            for key_attr_name, key_value in key_assignments:
                grouped[(base_name, key_attr_name)].append(
                    (class_node, key_value, method_rows)
                )
    candidates: list[StringKeyedFormulaSubclassFamilyCandidate] = []
    for (base_name, key_attr_name), rows in grouped.items():
        if len(rows) < 2:
            continue
        method_names = sorted_tuple(
            {
                method_name
                for _class_node, _key_value, method_rows in rows
                for method_name, _method_source in method_rows
            }
        )
        if method_names == ("eval",):
            continue
        base_line = (
            classes.get(base_name).lineno if base_name in classes else rows[0][0].lineno
        )
        candidates.append(
            StringKeyedFormulaSubclassFamilyCandidate(
                file_path=str(module.path),
                line=base_line,
                base_class_name=base_name,
                key_attr_name=key_attr_name,
                subclass_names=tuple(
                    class_node.name for class_node, _key, _methods in rows
                ),
                key_values=tuple(
                    key_value for _class_node, key_value, _methods in rows
                ),
                method_names=method_names,
                expression_snippets=tuple(
                    method_source
                    for _class_node, _key_value, method_rows in rows
                    for _method_name, method_source in method_rows
                )[:4],
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.base_class_name,
            candidate.key_attr_name,
        ),
    )


_SCHEMA_FIELD_ACCESS_HELPER_TOKENS = frozenset(
    (
        "field",
        "key",
        "schema",
    )
)
_SCHEMA_FIELD_ACCESS_INTENT_TOKENS = frozenset(
    (
        "extract",
        "optional",
        "read",
        "required",
        "resolve",
        "validate",
    )
)
_SCHEMA_MAPPING_METHOD_NAMES = frozenset(("get", "pop", "setdefault"))


def _literal_schema_mapping_expression(node: ast.AST) -> str | None:
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return ast.unparse(node)
    return None


def _literal_schema_helper_call_looks_like_field_access(call_name: str) -> bool:
    tokens = frozenset(_runtime_semantic_identifier_tokens(call_name))
    return bool(tokens & _SCHEMA_FIELD_ACCESS_HELPER_TOKENS) and bool(
        tokens & _SCHEMA_FIELD_ACCESS_INTENT_TOKENS
    )


def _literal_schema_field_access_from_call(
    node: ast.Call,
) -> LiteralSchemaFieldAccess | None:
    if isinstance(node.func, ast.Attribute) and node.func.attr in (
        _SCHEMA_MAPPING_METHOD_NAMES
    ):
        if not node.args:
            return None
        key_name = _constant_string(node.args[0])
        source_expression = _literal_schema_mapping_expression(node.func.value)
        if key_name is None or source_expression is None:
            return None
        return LiteralSchemaFieldAccess(
            source_expression=source_expression,
            key_name=key_name,
            line=node.lineno,
            access_kind=f"mapping_{node.func.attr}",
        )

    call_name = _call_name(node.func)
    if (
        call_name is None
        or len(node.args) < 2
        or not _literal_schema_helper_call_looks_like_field_access(call_name)
    ):
        return None
    source_expression = _literal_schema_mapping_expression(node.args[0])
    key_name = _constant_string(node.args[1])
    if source_expression is None or key_name is None:
        return None
    return LiteralSchemaFieldAccess(
        source_expression=source_expression,
        key_name=key_name,
        line=node.lineno,
        access_kind="field_helper",
    )


def _literal_schema_field_access_from_subscript(
    node: ast.Subscript,
) -> LiteralSchemaFieldAccess | None:
    key_name = _constant_string(node.slice)
    source_expression = _literal_schema_mapping_expression(node.value)
    if key_name is None or source_expression is None:
        return None
    return LiteralSchemaFieldAccess(
        source_expression=source_expression,
        key_name=key_name,
        line=node.lineno,
        access_kind="subscript",
    )


def _literal_schema_field_access_from_membership(
    node: ast.Compare,
) -> LiteralSchemaFieldAccess | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    if not isinstance(node.ops[0], (ast.In, ast.NotIn)):
        return None
    key_name = _constant_string(node.left)
    source_expression = _literal_schema_mapping_expression(node.comparators[0])
    if key_name is None or source_expression is None:
        return None
    return LiteralSchemaFieldAccess(
        source_expression=source_expression,
        key_name=key_name,
        line=node.lineno,
        access_kind="membership",
    )


def _literal_schema_dispatch_owners_for_function(
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    config: DetectorConfig,
) -> tuple[LiteralSchemaDispatchOwner, ...]:
    accesses_by_source: dict[str, list[LiteralSchemaFieldAccess]] = defaultdict(list)

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Call(self, node: ast.Call) -> None:
            access = _literal_schema_field_access_from_call(node)
            if access is not None:
                accesses_by_source[access.source_expression].append(access)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            access = _literal_schema_field_access_from_subscript(node)
            if access is not None:
                accesses_by_source[access.source_expression].append(access)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            access = _literal_schema_field_access_from_membership(node)
            if access is not None:
                accesses_by_source[access.source_expression].append(access)
            self.generic_visit(node)

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)

    owners: list[LiteralSchemaDispatchOwner] = []
    for source_expression, accesses in sorted(accesses_by_source.items()):
        key_names = sorted_tuple({access.key_name for access in accesses})
        if len(key_names) < config.min_literal_schema_field_count:
            continue
        line_numbers = tuple(access.line for access in accesses[: len(key_names)])
        owners.append(
            LiteralSchemaDispatchOwner(
                qualname=qualname,
                source_expression=source_expression,
                key_names=key_names,
                line_numbers=line_numbers,
                access_count=len(accesses),
            )
        )
    return tuple(owners)


def _literal_schema_dispatch_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[LiteralSchemaDispatchCandidate, ...]:
    owners_by_signature: dict[
        tuple[str, ...],
        list[LiteralSchemaDispatchOwner],
    ] = defaultdict(list)
    for qualname, function in _iter_named_functions(module):
        for owner in _literal_schema_dispatch_owners_for_function(
            qualname,
            function,
            config,
        ):
            owners_by_signature[owner.key_names].append(owner)

    candidates: list[LiteralSchemaDispatchCandidate] = []
    for key_names, owners in sorted(owners_by_signature.items()):
        if len(owners) < config.min_literal_schema_owner_count:
            continue
        ordered = sorted_tuple(
            owners, key=lambda item: (item.line_numbers[0], item.qualname)
        )
        candidates.append(
            LiteralSchemaDispatchCandidate(
                file_path=str(module.path),
                line=ordered[0].line_numbers[0],
                function_names=tuple(owner.qualname for owner in ordered),
                source_expressions=tuple(owner.source_expression for owner in ordered),
                key_names=key_names,
                line_numbers=tuple(owner.line_numbers[0] for owner in ordered),
                access_count=sum(owner.access_count for owner in ordered),
            )
        )
    return tuple(candidates)


class LiteralSchemaDispatchDetector(
    ConfiguredModuleCollectorCandidateDetector[LiteralSchemaDispatchCandidate]
):
    detector_id = "literal_schema_dispatch"
    candidate_collector = _literal_schema_dispatch_candidates
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated literal schema dispatch should move behind one nominal authority",
        "Multiple owners that read the same mapping schema through string-literal field selectors duplicate the schema boundary. That makes adding or changing a field require coordinated edits and lets operational semantics leak into consumers.",
        "one nominal schema authority owns literal field validation, dependencies, and projection",
        "same mapping-field signature is manually walked by multiple owners",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self,
        candidate: LiteralSchemaDispatchCandidate,
    ) -> RefactorFinding:
        key_summary = ", ".join(candidate.key_names[:6])
        owner_summary = ", ".join(candidate.function_names[:4])
        return self.build_finding(
            (
                f"{len(candidate.function_names)} owners manually read the same "
                f"literal schema fields ({key_summary}) from mapping-like sources: "
                f"{owner_summary}."
            ),
            candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class SchemaFieldSpec:\n"
                "    name: str\n"
                "    required: bool\n\n"
                "class SchemaFieldAuthority:\n"
                "    fields: ClassVar[tuple[SchemaFieldSpec, ...]]\n"
                "    @classmethod\n"
                "    def required_value(cls, payload, field_name): ..."
            ),
            codemod_patch=(
                "# Replace repeated literal mapping-field extraction with one "
                "nominal schema authority.\n"
                "# Consumers should ask the authority for validated values, "
                "dependency fields, and projection coordinates instead of "
                "walking the same string keys locally."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(candidate.function_names),
                field_names=candidate.key_names,
                mapping_name="literal_schema",
                source_name=", ".join(
                    sorted_tuple(set(candidate.source_expressions))[:4]
                ),
            ),
        )


_FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS = frozenset(
    {
        "artifact",
        "default",
        "formal",
        "kernel",
        "lean",
        "manifest",
        "policy",
        "profile",
        "schema",
        "theorem",
    }
)
_FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS = 3
_FORMAL_BOUNDARY_STRING_ID_TOKENS = frozenset(
    {
        "field",
        "fields",
        "id",
        "ids",
        "key",
        "keys",
        "name",
        "names",
        "source",
        "sources",
    }
)


@dataclass(frozen=True)
class FormalBoundaryStringRegistryConstant:
    target_name: str
    value: str
    line: int


def _literal_string_registry_fields(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Dict):
        items = _string_dict_items(node)
        if items is None:
            return ()
        return tuple(items)
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values: list[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value,
                str,
            ):
                return ()
            values.append(element.value)
        return tuple(values)
    return ()


def _formal_boundary_call_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _call_leaf_name(node: ast.Call) -> str:
    return _formal_boundary_call_name(node).rsplit(".", 1)[-1]


_FORMAL_BOUNDARY_SOURCE_SCOPE_CALL_TOKENS = frozenset(
    {
        "boundary",
        "carrier",
        "default",
        "formal",
        "kernel",
        "lean",
        "materialization",
        "payload",
        "policy",
        "profile",
        "scope",
        "source",
    }
)
_FORMAL_BOUNDARY_SOURCE_SCOPE_FUNCTION_TOKENS = frozenset(
    {
        "boundary",
        "carrier",
        "formal",
        "lean",
        "payload",
        "policy",
        "scope",
        "source",
    }
)
_FORMAL_BOUNDARY_SOURCE_SCOPE_MIN_FIELDS = 2


def _is_nominal_source_scope_carrier_constructor_call(node: ast.Call) -> bool:
    leaf_name = _call_leaf_name(node)
    if not leaf_name[:1].isupper():
        return False
    tokens = frozenset(_runtime_semantic_identifier_tokens(leaf_name))
    return bool(tokens & {"carrier", "domain", "payload", "request", "scope"})


def _is_formal_boundary_source_scope_call(node: ast.Call) -> bool:
    if _is_nominal_source_scope_carrier_constructor_call(node):
        return False
    call_tokens = frozenset(_runtime_semantic_identifier_tokens(_call_leaf_name(node)))
    return bool(call_tokens & _FORMAL_BOUNDARY_SOURCE_SCOPE_CALL_TOKENS) and bool(
        {"scope", "source", "payload"} & call_tokens
    )


def _function_name_tokens(function_stack: Sequence[str]) -> frozenset[str]:
    if not function_stack:
        return frozenset()
    return frozenset(_runtime_semantic_identifier_tokens(function_stack[-1]))


def _function_is_formal_boundary_source_scope(
    function_stack: Sequence[str],
) -> bool:
    tokens = _function_name_tokens(function_stack)
    return bool(tokens & _FORMAL_BOUNDARY_SOURCE_SCOPE_FUNCTION_TOKENS) and bool(
        {"scope", "source", "payload"} & tokens
    )


def _explicit_source_scope_keyword_fields(node: ast.Call) -> tuple[str, ...]:
    fields = tuple(
        keyword.arg
        for keyword in node.keywords
        if keyword.arg is not None and keyword.arg not in {"source_scope"}
    )
    return tuple(field for field in fields if field is not None)


def _is_formal_boundary_literal_registry_call(node: ast.Call) -> bool:
    call_name = _formal_boundary_call_name(node).lower()
    return any(
        token in call_name for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
    )


def _formal_boundary_registry_target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            element.id for element in target.elts if isinstance(element, ast.Name)
        )
    return ()


def _formal_boundary_registry_target_tokens(target_name: str) -> frozenset[str]:
    return frozenset(_runtime_semantic_identifier_tokens(target_name))


def _formal_boundary_registry_value_tokens(value: str) -> frozenset[str]:
    return frozenset(_runtime_semantic_identifier_tokens(value))


def _is_formal_boundary_string_registry_constant(
    target_name: str,
    value: str,
) -> bool:
    target_tokens = _formal_boundary_registry_target_tokens(target_name)
    value_tokens = _formal_boundary_registry_value_tokens(value)
    boundary_tokens = target_tokens | value_tokens
    return bool(
        boundary_tokens & _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
    ) and bool((target_tokens | value_tokens) & _FORMAL_BOUNDARY_STRING_ID_TOKENS)


class FormalBoundaryStringRegistryAuthority:
    @staticmethod
    def module_constants(
        module: ParsedModule,
    ) -> tuple[FormalBoundaryStringRegistryConstant, ...]:
        constants: list[FormalBoundaryStringRegistryConstant] = []
        for statement in module.module.body:
            assignment_targets: tuple[ast.AST, ...]
            assignment_value: ast.AST | None
            if isinstance(statement, ast.Assign):
                assignment_targets = tuple(statement.targets)
                assignment_value = statement.value
            elif isinstance(statement, ast.AnnAssign):
                assignment_targets = (statement.target,)
                assignment_value = statement.value
            else:
                continue
            if assignment_value is None:
                continue
            value = _constant_string(assignment_value)
            if value is None:
                continue
            for target in assignment_targets:
                for target_name in _formal_boundary_registry_target_names(target):
                    if _is_formal_boundary_string_registry_constant(target_name, value):
                        constants.append(
                            FormalBoundaryStringRegistryConstant(
                                target_name=target_name,
                                value=value,
                                line=statement.lineno,
                            )
                        )
        return tuple(constants)


class FormalBoundaryLiteralRegistryCallVisitor(ClassFunctionStackNodeVisitor):
    traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
    traverse_function_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body

    def __init__(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        findings: list[RefactorFinding],
    ) -> None:
        super().__init__()
        self.detector = detector
        self.module = module
        self.findings = findings

    def visit_Call(self, node: ast.Call) -> None:
        if not _is_formal_boundary_literal_registry_call(node):
            self.generic_visit(node)
            return
        rows: list[tuple[ast.AST, tuple[str, ...], str]] = []
        for arg in node.args:
            fields = _literal_string_registry_fields(arg)
            if len(fields) >= _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
                rows.append((arg, fields, "positional"))
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            fields = _literal_string_registry_fields(keyword.value)
            if len(fields) >= _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
                rows.append((keyword.value, fields, keyword.arg))
        if not rows:
            self.generic_visit(node)
            return
        call_name = _formal_boundary_call_name(node)
        owner = _owner_symbol(
            tuple(self.class_stack),
            tuple(self.function_stack),
            "formal_boundary_literal_registry",
        )
        for literal_node, fields, argument_role in rows:
            self.findings.append(
                self.detector.build_finding(
                    self.summary(owner, fields, argument_role, call_name),
                    (
                        SourceLocation(
                            str(self.module.path),
                            literal_node.lineno,
                            owner,
                        ),
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=1,
                        field_names=fields,
                        mapping_name="formal_boundary_literal_registry",
                        source_name=call_name,
                    ),
                    scaffold=(
                        "class FormalBoundarySourceAuthority:\n"
                        "    @classmethod\n"
                        "    def source_fields(cls):\n"
                        "        return ExportedFormalProfile.source_fields()\n\n"
                        "    @classmethod\n"
                        "    def source_object(cls, carrier):\n"
                        "        return carrier.project(cls.source_fields())"
                    ),
                    codemod_patch=(
                        "# Replace the local literal field registry with a "
                        "projection derived from the exported formal/profile "
                        "authority. The runtime boundary should receive a "
                        "source object produced by that authority, not a "
                        "handwritten dict/list of field names."
                    ),
                )
            )
        self.generic_visit(node)

    @staticmethod
    def summary(
        owner: str,
        fields: tuple[str, ...],
        argument_role: str,
        call_name: str,
    ) -> str:
        field_summary = ", ".join(fields[:6])
        return (
            f"`{owner}` passes a {len(fields)}-field literal string registry "
            f"({field_summary}) as {argument_role} data to formal-boundary call "
            f"`{call_name}`."
        )


class FormalBoundaryLiteralRegistryMirrorDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Formal-boundary literal registries should be derived",
        "A runtime call into a formal/default/profile/schema/policy boundary should not construct a local string-key registry, and a runtime module should not keep a local catalog of formal-boundary string ids. The field set and ids are proof-relevant boundary data and should be read from the exported formal authority, nominal schema, or typed source carrier.",
        "formal-boundary source/schema fields are derived from one exported authority",
        "literal string registry is passed into or mirrored beside a formal/runtime boundary",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _module_string_registry_findings(
        self,
        module: ParsedModule,
    ) -> list[RefactorFinding]:
        constants = FormalBoundaryStringRegistryAuthority.module_constants(module)
        if len(constants) < _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
            return []
        names = tuple(constant.target_name for constant in constants)
        values = tuple(constant.value for constant in constants)
        name_summary = ", ".join(names[:6])
        value_summary = ", ".join(values[:6])
        return [
            self.build_finding(
                (
                    f"`{module.path}` declares {len(constants)} local "
                    f"formal-boundary string ids ({name_summary}) with values "
                    f"({value_summary})."
                ),
                tuple(
                    SourceLocation(
                        str(module.path), constant.line, constant.target_name
                    )
                    for constant in constants
                ),
                metrics=MappingMetrics.from_field_names(
                    mapping_site_count=1,
                    field_names=values,
                    mapping_name="formal_boundary_string_id_catalog",
                    source_name=str(module.path),
                ),
                scaffold=(
                    "class FormalBoundaryIdAuthority:\n"
                    "    @classmethod\n"
                    "    def profile_id(cls, symbolic_name):\n"
                    "        return ExportedFormalProfileCatalog.id_for(symbolic_name)"
                ),
                codemod_patch=(
                    "# Delete the local formal-boundary string-id catalog and "
                    "derive ids from the exported formal/profile/schema catalog. "
                    "The runtime module should name a nominal profile binding, "
                    "not mirror the external string registry."
                ),
            )
        ]

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = self._module_string_registry_findings(module)
        FormalBoundaryLiteralRegistryCallVisitor(self, module, findings).visit(
            module.module
        )
        return findings


class FormalBoundaryStringlySourceScopeVisitor(ClassFunctionStackNodeVisitor):
    traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
    traverse_function_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body

    def __init__(
        self,
        detector: PerModuleIssueDetector,
        module: ParsedModule,
        findings: list[RefactorFinding],
    ) -> None:
        super().__init__()
        self.detector = detector
        self.module = module
        self.findings = findings

    def visit_Call(self, node: ast.Call) -> None:
        owner = _owner_symbol(
            tuple(self.class_stack),
            tuple(self.function_stack),
            "formal_boundary_stringly_source_scope",
        )
        call_name = _formal_boundary_call_name(node)
        is_source_scope_call = _is_formal_boundary_source_scope_call(node)
        for literal_node, fields, argument_role in self.literal_mapping_rows(node):
            if (
                len(fields) >= _FORMAL_BOUNDARY_SOURCE_SCOPE_MIN_FIELDS
                and is_source_scope_call
            ):
                self.findings.append(
                    self.detector.build_finding(
                        self.literal_mapping_summary(
                            owner,
                            fields,
                            argument_role,
                            call_name,
                        ),
                        (
                            SourceLocation(
                                str(self.module.path),
                                literal_node.lineno,
                                owner,
                            ),
                        ),
                        metrics=MappingMetrics.from_field_names(
                            mapping_site_count=1,
                            field_names=fields,
                            mapping_name="formal_boundary_source_scope_literal_mapping",
                            source_name=call_name,
                        ),
                        scaffold=self.scaffold(),
                        codemod_patch=self.codemod_patch(),
                    )
                )
        fields = _explicit_source_scope_keyword_fields(node)
        if (
            len(fields) >= _FORMAL_BOUNDARY_SOURCE_SCOPE_MIN_FIELDS
            and is_source_scope_call
        ):
            self.findings.append(
                self.detector.build_finding(
                    self.call_summary(
                        owner,
                        fields,
                        call_name,
                    ),
                    (
                        SourceLocation(
                            str(self.module.path),
                            node.lineno,
                            owner,
                        ),
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=1,
                        field_names=fields,
                        mapping_name="formal_boundary_source_scope_kwargs",
                        source_name=call_name,
                    ),
                    scaffold=self.scaffold(),
                    codemod_patch=self.codemod_patch(),
                )
            )
        self.generic_visit(node)

    @staticmethod
    def literal_mapping_rows(
        node: ast.Call,
    ) -> tuple[tuple[ast.AST, tuple[str, ...], str], ...]:
        rows: list[tuple[ast.AST, tuple[str, ...], str]] = []
        for index, arg in enumerate(node.args):
            fields = _literal_string_registry_fields(arg)
            if fields:
                rows.append((arg, fields, f"positional argument {index}"))
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            fields = _literal_string_registry_fields(keyword.value)
            if fields:
                rows.append((keyword.value, fields, keyword.arg))
        return tuple(rows)

    def visit_Return(self, node: ast.Return) -> None:
        fields = _literal_string_registry_fields(node.value) if node.value else ()
        if len(
            fields
        ) >= _FORMAL_BOUNDARY_SOURCE_SCOPE_MIN_FIELDS and _function_is_formal_boundary_source_scope(
            tuple(self.function_stack)
        ):
            owner = _owner_symbol(
                tuple(self.class_stack),
                tuple(self.function_stack),
                "formal_boundary_stringly_source_scope",
            )
            self.findings.append(
                self.detector.build_finding(
                    self.return_summary(owner, fields),
                    (SourceLocation(str(self.module.path), node.lineno, owner),),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=1,
                        field_names=fields,
                        mapping_name="formal_boundary_source_scope_return_dict",
                        source_name=owner,
                    ),
                    scaffold=self.scaffold(),
                    codemod_patch=self.codemod_patch(),
                )
            )
        self.generic_visit(node)

    @staticmethod
    def call_summary(owner: str, fields: tuple[str, ...], call_name: str) -> str:
        preview = ", ".join(fields[:6])
        return (
            f"`{owner}` assembles a formal/source-scope boundary with "
            f"{len(fields)} explicit string-key fields ({preview}) via "
            f"`{call_name}`."
        )

    @staticmethod
    def literal_mapping_summary(
        owner: str,
        fields: tuple[str, ...],
        argument_role: str,
        call_name: str,
    ) -> str:
        preview = ", ".join(fields[:6])
        return (
            f"`{owner}` passes a formal/source-scope boundary as a "
            f"{len(fields)}-field string-key mapping ({preview}) in "
            f"{argument_role} to `{call_name}`."
        )

    @staticmethod
    def return_summary(owner: str, fields: tuple[str, ...]) -> str:
        preview = ", ".join(fields[:6])
        return (
            f"`{owner}` returns a formal/source-scope boundary as a "
            f"{len(fields)}-field string-key mapping ({preview})."
        )

    @staticmethod
    def scaffold() -> str:
        return (
            "@dataclass(frozen=True)\n"
            "class FormalBoundarySourcePayload:\n"
            "    declared_field: object\n\n"
            "class FormalBoundarySourceScopeAuthority:\n"
            "    @classmethod\n"
            "    def source_object(cls, payload: FormalBoundarySourcePayload):\n"
            "        return DeclaredDataclassFieldValuesAuthority.values(payload)"
        )

    @staticmethod
    def codemod_patch() -> str:
        return (
            "# Replace explicit string-key source-scope assembly with a declared "
            "dataclass/nominal carrier. Keep exactly one adapter that converts "
            "declared fields into the formal boundary API; runtime code should "
            "construct the carrier, not spell boundary field names."
        )


class FormalBoundaryStringlySourceScopeDetector(PerModuleIssueDetector):
    detector_id = "formal_boundary_stringly_source_scope"
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Formal-boundary source scopes should use nominal carriers",
        "Runtime code should not assemble proof-relevant source scopes by spelling string-key fields at the call site. The formal boundary should receive a source object projected from a declared dataclass, generated carrier, or exported schema authority so Python cannot drift from the formal field contract.",
        "formal-boundary source scopes are assembled from nominal carriers",
        "source-scope boundary is constructed by explicit string-key mapping fields",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        FormalBoundaryStringlySourceScopeVisitor(self, module, findings).visit(
            module.module
        )
        return findings


_FORMAL_BOUNDARY_EXTERNAL_FILE_SUFFIXES = frozenset(
    {".json", ".lean", ".toml", ".yaml", ".yml"}
)
_FORMAL_BOUNDARY_EXTERNAL_PATH_HINT_TOKENS = frozenset(
    {
        "artifact",
        "bundle",
        "formal",
        "kernel",
        "lean",
        "manifest",
        "policy",
        "profile",
        "schema",
        "theorem",
    }
)
_FORMAL_BOUNDARY_EXTERNAL_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".pytest-tmp",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "benchmark_results",
        "build",
        "diagnostics",
        "dist",
        "htmlcov",
        "node_modules",
        "site-packages",
        "venv",
    }
)
_FORMAL_BOUNDARY_EXTERNAL_MAX_BYTES = 32 * 1024 * 1024
_FORMAL_BOUNDARY_EXTERNAL_MAX_FILES = 256


@dataclass(frozen=True)
class FormalBoundaryExternalStringSite(FormalBoundaryStringRegistryConstant):
    path: Path


FormalBoundaryStringConstantRecord: TypeAlias = tuple[
    ParsedModule,
    FormalBoundaryStringRegistryConstant,
]
FormalBoundaryStringConstantRecords: TypeAlias = tuple[
    FormalBoundaryStringConstantRecord,
    ...,
]
FormalBoundaryStringConstantsByValue: TypeAlias = dict[
    str,
    list[FormalBoundaryStringConstantRecord],
]
FormalBoundaryExternalSitesByValue: TypeAlias = dict[
    str,
    list[FormalBoundaryExternalStringSite],
]


def _module_formal_boundary_string_constants(
    modules: list[ParsedModule],
) -> FormalBoundaryStringConstantRecords:
    constants: list[FormalBoundaryStringConstantRecord] = []
    for module in modules:
        constants.extend(
            (module, constant)
            for constant in FormalBoundaryStringRegistryAuthority.module_constants(
                module
            )
        )
    return tuple(constants)


def _formal_boundary_python_constants_by_value(
    constants: FormalBoundaryStringConstantRecords,
) -> FormalBoundaryStringConstantsByValue:
    grouped: FormalBoundaryStringConstantsByValue = defaultdict(list)
    for module, constant in constants:
        grouped[constant.value].append((module, constant))
    return grouped


def _formal_boundary_nearest_repository_root(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    fallback_root = current.parent if current.parent != current else current
    temp_root = Path(tempfile.gettempdir()).resolve()
    for candidate in (current, *current.parents):
        if candidate.resolve() == temp_root:
            continue
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return fallback_root


def _formal_boundary_scan_root(modules: list[ParsedModule]) -> Path | None:
    if not modules:
        return None
    resolved_paths = tuple(str(module.path.resolve()) for module in modules)
    common_path = Path(os.path.commonpath(resolved_paths))
    if common_path.is_file():
        common_path = common_path.parent
    return _formal_boundary_nearest_repository_root(common_path)


def _formal_boundary_external_path_has_authority_hint(path: Path) -> bool:
    path_tokens = frozenset(
        token
        for part in path.with_suffix("").parts
        for token in _runtime_semantic_identifier_tokens(part)
    )
    return bool(path_tokens & _FORMAL_BOUNDARY_EXTERNAL_PATH_HINT_TOKENS)


def _formal_boundary_external_file_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            (
                dirname
                for dirname in dirnames
                if dirname not in _FORMAL_BOUNDARY_EXTERNAL_IGNORED_DIR_NAMES
                and not dirname.startswith(".")
                and not dirname.startswith("benchmark_results")
                and not dirname.endswith((".egg-info", ".dist-info"))
            )
        )
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.suffix not in _FORMAL_BOUNDARY_EXTERNAL_FILE_SUFFIXES:
                continue
            if not _formal_boundary_external_path_has_authority_hint(path):
                continue
            try:
                if path.stat().st_size > _FORMAL_BOUNDARY_EXTERNAL_MAX_BYTES:
                    continue
            except OSError:
                continue
            paths.append(path)
            if len(paths) >= _FORMAL_BOUNDARY_EXTERNAL_MAX_FILES:
                return tuple(paths)
    return tuple(paths)


def _formal_boundary_external_string_sites(
    path: Path,
    values: tuple[str, ...],
) -> tuple[FormalBoundaryExternalStringSite, ...]:
    if not values:
        return ()
    ordered_values = tuple(sorted(values, key=lambda value: (-len(value), value)))
    pattern = re.compile("|".join(re.escape(value) for value in ordered_values))
    sites: list[FormalBoundaryExternalStringSite] = []
    try:
        with path.open(encoding="utf-8", errors="ignore") as source:
            for line_number, line in enumerate(source, start=1):
                for match in pattern.finditer(line):
                    sites.append(
                        FormalBoundaryExternalStringSite(
                            target_name=match.group(0),
                            value=match.group(0),
                            line=line_number,
                            path=path,
                        )
                    )
    except OSError:
        return ()
    return tuple(sites)


def _formal_boundary_external_sites_by_value(
    path: Path,
    values: tuple[str, ...],
) -> FormalBoundaryExternalSitesByValue:
    grouped: FormalBoundaryExternalSitesByValue = defaultdict(list)
    for site in _formal_boundary_external_string_sites(path, values):
        grouped[site.value].append(site)
    return grouped


def _formal_boundary_python_evidence_for_values(
    constants_by_value: FormalBoundaryStringConstantsByValue,
    values: tuple[str, ...],
) -> tuple[SourceLocation, ...]:
    evidence: list[SourceLocation] = []
    for value in values:
        module, constant = constants_by_value[value][0]
        evidence.append(
            SourceLocation(str(module.path), constant.line, constant.target_name)
        )
    return tuple(evidence)


def _formal_boundary_external_evidence_for_values(
    sites_by_value: FormalBoundaryExternalSitesByValue,
    values: tuple[str, ...],
) -> tuple[SourceLocation, ...]:
    evidence: list[SourceLocation] = []
    for value in values:
        site = sites_by_value[value][0]
        evidence.append(SourceLocation(str(site.path), site.line, value))
    return tuple(evidence)


class FormalBoundaryExternalStringRegistryMirrorAuthority:
    @classmethod
    def findings(
        cls,
        detector: IssueDetector,
        modules: list[ParsedModule],
    ) -> list[RefactorFinding]:
        constants = _module_formal_boundary_string_constants(modules)
        if len(constants) < _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
            return []
        constants_by_value = _formal_boundary_python_constants_by_value(constants)
        values = tuple(sorted(constants_by_value))
        root = _formal_boundary_scan_root(modules)
        if root is None:
            return []
        return [
            finding
            for path in _formal_boundary_external_file_paths(root)
            if (
                finding := cls.finding_for_external_path(
                    detector,
                    constants_by_value,
                    values,
                    path,
                )
            )
            is not None
        ]

    @staticmethod
    def finding_for_external_path(
        detector: IssueDetector,
        constants_by_value: FormalBoundaryStringConstantsByValue,
        values: tuple[str, ...],
        path: Path,
    ) -> RefactorFinding | None:
        sites_by_value = _formal_boundary_external_sites_by_value(path, values)
        shared_values = tuple(sorted(set(sites_by_value) & set(constants_by_value)))
        if len(shared_values) < _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
            return None
        return detector.build_finding(
            FormalBoundaryExternalStringRegistryMirrorAuthority.summary(
                path,
                shared_values,
            ),
            (
                _formal_boundary_python_evidence_for_values(
                    constants_by_value,
                    shared_values[:6],
                )
                + _formal_boundary_external_evidence_for_values(
                    sites_by_value,
                    shared_values[:6],
                )
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=2,
                field_names=shared_values,
                mapping_name="formal_boundary_external_string_registry",
                source_name=str(path),
            ),
            scaffold=(
                "class GeneratedFormalBoundaryIdAuthority:\n"
                "    @classmethod\n"
                "    def id_for(cls, symbolic_name):\n"
                "        return FormalArtifactCatalog.current().id_for(symbolic_name)"
            ),
            codemod_patch=(
                "# Replace the Python-side string-id catalog with a generated "
                "authority loaded from the formal artifact/export. Keep symbolic "
                "names in runtime code and derive external ids from the formal "
                "catalog so Lean/formal and Python cannot drift."
            ),
        )

    @staticmethod
    def summary(path: Path, shared_values: tuple[str, ...]) -> str:
        preview_values = ", ".join(shared_values[:6])
        return (
            f"`{path}` and Python runtime code mirror "
            f"{len(shared_values)} formal-boundary string ids ({preview_values})."
        )


class FormalBoundaryExternalStringRegistryMirrorDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Formal-boundary string registries should not be mirrored across sources",
        "When Python runtime modules and external Lean/formal policy artifacts declare the same proof-relevant string ids, the runtime has a second source of truth. The Python side should consume a generated/typed authority derived from the formal artifact instead of copying the registry values.",
        "formal-boundary string ids have one generated authority shared by runtime and formal artifacts",
        "Python and external formal artifacts mirror the same string-id registry",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        return FormalBoundaryExternalStringRegistryMirrorAuthority.findings(
            self, modules
        )


class UnclassifiedRuntimeFallbackDetector(PerModuleIssueDetector):
    detector_id = "unclassified_runtime_fallback"
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Runtime fallback/default sites should be classified at the formal boundary",
        "Fallback/default operators in runtime code can silently choose behavior when required semantics are missing. They should either be formal-boundary-declared defaults, explicit cache-miss semantics, diagnostic-only behavior, or fail-loud errors.",
        "each runtime fallback is classified as theorem-backed default, cache miss, diagnostic-only, or hard failure",
        "owner contains fallback/default operators that hide missing values locally",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        minimum = 1
        sites_by_owner: dict[str, list[tuple[int, str]]] = {}

        class Visitor(ClassFunctionStackNodeVisitor):
            traverse_class_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )
            traverse_function_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )

            def _record(self, node: ast.AST, fallback_kind: str) -> None:
                owner = _fallback_owner(self.class_stack, self.function_stack)
                sites_by_owner.setdefault(owner, []).append(
                    (int(getattr(node, "lineno", 1)), fallback_kind)
                )

            def visit_Call(self, node: ast.Call) -> None:
                if (fallback_kind := _call_fallback_kind(node)) is not None:
                    self._record(node, fallback_kind)
                self.generic_visit(node)

            def visit_IfExp(self, node: ast.IfExp) -> None:
                if (fallback_kind := _default_ifexp_kind(node)) is not None:
                    self._record(node, fallback_kind)
                self.generic_visit(node)

            def visit_BoolOp(self, node: ast.BoolOp) -> None:
                if (fallback_kind := _boolop_default_kind(node)) is not None:
                    self._record(node, fallback_kind)
                self.generic_visit(node)

        Visitor().visit(module.module)
        findings: list[RefactorFinding] = []
        for owner, sites in sorted(sites_by_owner.items()):
            if len(sites) < minimum:
                continue
            fallback_kinds = sorted_tuple({kind for _line, kind in sites})
            evidence = tuple(
                SourceLocation(str(module.path), line, f"{owner}:{kind}")
                for line, kind in sites[: min(len(sites), 12)]
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{owner}` has {len(sites)} unclassified runtime "
                        f"fallback/default sites: {', '.join(fallback_kinds)}."
                    ),
                    evidence,
                    scaffold=(
                        "@dataclass(frozen=True)\n"
                        "class RuntimeFallbackClassification:\n"
                        "    source: str\n"
                        "    kind: Literal['formal_default', 'cache_miss', "
                        "'diagnostic_only', 'fail_loud']\n"
                        "    theorem: str | None = None"
                    ),
                    codemod_patch=(
                        f"# Classify fallback/default sites in `{owner}`.\n"
                        "# Replace unclassified fallback operators with a "
                        "formal-boundary-declared default lookup, explicit cache miss branch, "
                        "diagnostic-only path, or a fail_loud hard error."
                    ),
                    metrics=DispatchCountMetrics(dispatch_site_count=len(sites)),
                    capability_gap=(
                        "no unclassified runtime fallback/default operator remains"
                    ),
                )
            )
        return findings


@dataclass(frozen=True)
class RuntimeNamespaceBridgeSite:
    line: int
    symbol: str
    bridge_kind: str


def _call_symbol(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_symbol(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _is_globals_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "globals"
        and not node.args
        and not node.keywords
    )


def _is_runtime_bridge_namespace_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_symbol(node.func).endswith(
        "runtime_bridge_namespace"
    )


def _globals_guard_symbol(node: ast.If) -> str | None:
    test = node.test
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.NotIn):
        return None
    if len(test.comparators) != 1 or not _is_globals_call(test.comparators[0]):
        return None
    if isinstance(test.left, ast.Constant) and isinstance(test.left.value, str):
        return test.left.value
    return None


class RuntimeNamespaceBridgeDetector(PerModuleIssueDetector):
    detector_id = "runtime_namespace_bridge"
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Runtime namespace bridges should be replaced with explicit authorities",
        "Copying another module namespace into globals, or conditionally defining names only when globals lacks them, creates a hidden compatibility layer. Split modules should import their dependencies explicitly and publish one authoritative public surface so missing names fail loudly.",
        "explicit import/authority boundary with no globals namespace copying",
        "module mutates globals or guards definitions through a namespace bridge",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        sites: list[RuntimeNamespaceBridgeSite] = []

        class Visitor(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                for alias in node.names:
                    if alias.name == "runtime_bridge_namespace":
                        sites.append(
                            RuntimeNamespaceBridgeSite(
                                line=int(node.lineno),
                                symbol=alias.asname or alias.name,
                                bridge_kind="runtime_bridge_namespace import",
                            )
                        )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and _is_globals_call(node.func.value)
                ):
                    bridge_kind = "globals update"
                    if any(_is_runtime_bridge_namespace_call(arg) for arg in node.args):
                        bridge_kind = "runtime_bridge_namespace globals update"
                    sites.append(
                        RuntimeNamespaceBridgeSite(
                            line=int(node.lineno),
                            symbol=ast.unparse(node),
                            bridge_kind=bridge_kind,
                        )
                    )
                self.generic_visit(node)

            def visit_If(self, node: ast.If) -> None:
                guarded_symbol = _globals_guard_symbol(node)
                if guarded_symbol is not None:
                    sites.append(
                        RuntimeNamespaceBridgeSite(
                            line=int(node.lineno),
                            symbol=guarded_symbol,
                            bridge_kind="guarded globals definition",
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(module.module)
        if not sites:
            return []
        bridge_kinds = sorted_tuple(site.bridge_kind for site in sites)
        evidence = tuple(
            SourceLocation(str(module.path), site.line, site.symbol)
            for site in sites[:12]
        )
        return [
            self.build_finding(
                (
                    f"`{module.path}` has {len(sites)} runtime namespace bridge "
                    f"site(s): {', '.join(bridge_kinds)}."
                ),
                evidence,
                scaffold=(
                    "# Replace namespace bridge imports with explicit imports from the true owner module.\n"
                    "# Delete `globals().update(...)` compatibility transport and publish one public authority/export surface.\n"
                    "# Replace `if name not in globals()` guards with unconditional definitions or fail-loud imports."
                ),
                codemod_patch=(
                    "# Remove runtime namespace copying in this module.\n"
                    "# Add explicit imports for every required dependency, then let missing names raise at import time."
                ),
                capability_gap="no runtime namespace bridge or guarded globals definition remains",
            )
        ]


_RUNTIME_SEMANTIC_BRANCH_AXIS_TOKENS = frozenset(
    (
        "action",
        "basis",
        "budget",
        "certified",
        "family",
        "frontier",
        "kind",
        "formal",
        "materialization",
        "mode",
        "policy",
        "profile",
        "projection",
        "repair",
        "request",
        "residual",
        "runtime",
        "selection",
        "semantic",
        "source",
        "theorem",
    )
)


def _runtime_semantic_identifier_tokens(text: str) -> tuple[str, ...]:
    normalized = "".join(
        (character.lower() if character.isalnum() else "_") for character in text
    )
    return tuple((token for token in normalized.split("_") if token))


def _runtime_semantic_axis_is_interesting(axis_expression: str) -> bool:
    tokens = set(_runtime_semantic_identifier_tokens(axis_expression))
    return bool(tokens & _RUNTIME_SEMANTIC_BRANCH_AXIS_TOKENS)


def _runtime_semantic_branch_test(
    test: ast.AST,
) -> tuple[str, str] | None:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _runtime_semantic_branch_test(test.operand)
        if inner is None:
            return None
        axis_expression, case_expression = inner
        return axis_expression, f"not {case_expression}"

    if isinstance(test, ast.Compare) and len(test.ops) == 1 and test.comparators:
        operator = test.ops[0]
        comparator = test.comparators[0]
        if isinstance(operator, (ast.In, ast.NotIn)):
            axis_expression = ast.unparse(comparator)
            case_expression = ast.unparse(test.left)
            if _runtime_semantic_axis_is_interesting(axis_expression):
                return axis_expression, case_expression
        if isinstance(operator, (ast.Eq, ast.Is)):
            left_expression = ast.unparse(test.left)
            comparator_expression = ast.unparse(comparator)
            if _literal_default_kind(comparator) is not None:
                if _runtime_semantic_axis_is_interesting(left_expression):
                    return left_expression, comparator_expression
            if _literal_default_kind(test.left) is not None:
                if _runtime_semantic_axis_is_interesting(comparator_expression):
                    return comparator_expression, left_expression
            if _runtime_semantic_axis_is_interesting(left_expression):
                return left_expression, comparator_expression

    if isinstance(test, (ast.Name, ast.Attribute, ast.Subscript)):
        axis_expression = ast.unparse(test)
        if _runtime_semantic_axis_is_interesting(axis_expression):
            return axis_expression, "truthy"

    if isinstance(test, ast.BoolOp):
        for value in test.values:
            branch_test = _runtime_semantic_branch_test(value)
            if branch_test is not None:
                return branch_test

    return None


def _runtime_semantic_elif_chain(
    statement: ast.stmt,
) -> SemanticBranchChain:
    if not isinstance(statement, ast.If):
        return ()
    chain: list[SemanticBranchObservation] = []
    current: ast.If | None = statement
    while current is not None:
        branch_test = _runtime_semantic_branch_test(current.test)
        if branch_test is None:
            return ()
        axis_expression, case_expression = branch_test
        chain.append((current.lineno, axis_expression, case_expression))
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        current = None
    return tuple(chain) if len(chain) >= 2 else ()


def _runtime_semantic_sequential_guard_chain(
    body: Sequence[ast.stmt],
    start: int,
) -> SemanticBranchChain:
    chain: list[SemanticBranchObservation] = []
    index = start
    while index < len(body):
        statement = body[index]
        if not isinstance(statement, ast.If) or statement.orelse:
            break
        branch_test = _runtime_semantic_branch_test(statement.test)
        if branch_test is None:
            break
        axis_expression, case_expression = branch_test
        chain.append((statement.lineno, axis_expression, case_expression))
        index += 1
    return tuple(chain) if len(chain) >= 2 else ()


RUNTIME_SEMANTIC_BRANCH_COLLECTION_SPEC = BranchChainCollectionSpec(
    _runtime_semantic_elif_chain,
    _runtime_semantic_sequential_guard_chain,
    branch_observation_first_line,
    all_branch_chains_active,
)


def _runtime_semantic_branch_chains_from_body(
    body: Sequence[ast.stmt],
) -> SemanticBranchChains:
    return collect_nested_branch_chains_from_body(
        body,
        RUNTIME_SEMANTIC_BRANCH_COLLECTION_SPEC,
    )


class RuntimeSemanticBranchChainDetector(PerModuleIssueDetector):
    detector_id = "runtime_semantic_branch_chain"
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Runtime semantic if-chain should move behind a formal policy authority",
        "Branch chains over runtime policy, source, projection, materialization, theorem, or repair axes encode operational semantics in local Python control flow. The formal boundary should own those cases and expose one declared policy/profile result.",
        "single formal policy authority owns each runtime semantic branch axis",
        "same runtime semantic axis is selected through a local if/elif or guard chain",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        minimum = 2
        findings: list[RefactorFinding] = []
        for qualname, function in _iter_named_functions(module):
            for chain in _runtime_semantic_branch_chains_from_body(function.body):
                if len(chain) < minimum:
                    continue
                axis_names = sorted_tuple({axis for _line, axis, _case in chain})
                case_names = tuple((case for _line, _axis, case in chain))
                axis_summary = ", ".join(axis_names[:3])
                case_summary = ", ".join(case_names[:4])
                evidence = tuple(
                    SourceLocation(
                        str(module.path),
                        line,
                        f"{qualname}:{axis_expression}:{case_expression}",
                    )
                    for line, axis_expression, case_expression in chain[:6]
                )
                findings.append(
                    self.build_finding(
                        (
                            f"`{qualname}` keeps a {len(chain)}-branch runtime "
                            f"semantic if-chain over {axis_summary} with cases "
                            f"{case_summary}."
                        ),
                        evidence,
                        scaffold=(
                            "class RuntimeSemanticPolicy(ABC):\n"
                            "    @abstractmethod\n"
                            "    def materialize(self, source): ...\n\n"
                            "# Generate or load the concrete policy from the formal runtime profile;\n"
                            "# Python should consume the selected policy result, not branch on the cases."
                        ),
                        codemod_patch=(
                            f"# Replace the runtime semantic branch chain in `{qualname}` with one "
                            "formal policy/profile authority.\n"
                            "# Move case membership, precedence, and default behavior into the formal "
                            "runtime profile and make this Python site consume the declared result."
                        ),
                        metrics=BranchCountMetrics(branch_site_count=len(chain)),
                        capability_gap=(
                            "runtime semantic branch cases are declared by the formal policy boundary"
                        ),
                    )
                )
        return findings


@dataclass(frozen=True, slots=True)
class RuntimeSubjectFunctionCandidate:
    file_path: str
    line: int
    qualname: str
    subject_expression: str


@dataclass(frozen=True, slots=True)
class IsinstanceFamilyScatterCandidate(RuntimeSubjectFunctionCandidate):
    type_names: tuple[str, ...]
    site_count: int
    line_numbers: tuple[int, ...]
    test_expressions: tuple[str, ...]


_ISINSTANCE_SCATTER_EXCLUDED_TYPE_NAMES = frozenset(
    {
        "ABC",
        "Any",
        "Callable",
        "ClassVar",
        "Iterable",
        "Iterator",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "None",
        "NoneType",
        "Sequence",
        "Set",
        "SimpleNamespace",
        "TypeAlias",
        "bytearray",
        "bool",
        "bytes",
        "dict",
        "float",
        "frozenset",
        "generic",
        "int",
        "list",
        "memoryview",
        "ndarray",
        "object",
        "set",
        "str",
        "tuple",
        "type",
    }
)


def _isinstance_scatter_type_name(type_expr: ast.AST) -> str | None:
    if isinstance(type_expr, ast.Attribute) and isinstance(type_expr.value, ast.Name):
        if type_expr.value.id == "ast":
            return None
    type_name = ast.unparse(type_expr)
    terminal_name = _ast_terminal_name(type_expr)
    if terminal_name in _ISINSTANCE_SCATTER_EXCLUDED_TYPE_NAMES:
        return None
    if type_name.startswith(("ast.", "jax.", "jnp.", "np.", "numpy.")):
        return None
    return type_name


def _isinstance_scatter_type_names(type_expr: ast.AST) -> tuple[str, ...]:
    if isinstance(type_expr, (ast.Tuple, ast.List, ast.Set)):
        names = tuple(
            name
            for element in type_expr.elts
            if (name := _isinstance_scatter_type_name(element)) is not None
        )
    else:
        name = _isinstance_scatter_type_name(type_expr)
        names = () if name is None else (name,)
    return sorted_tuple(set(names))


def _isinstance_family_scatter_candidates(
    module: ParsedModule,
) -> tuple[IsinstanceFamilyScatterCandidate, ...]:
    candidates: list[IsinstanceFamilyScatterCandidate] = []
    for qualname, function in _iter_named_functions(module):
        grouped: dict[str, list[tuple[int, str, tuple[str, ...]]]] = defaultdict(list)
        for subnode in _walk_nodes(function):
            if not (
                isinstance(subnode, ast.Call)
                and len(subnode.args) == 2
                and not subnode.keywords
                and _ast_terminal_name(subnode.func) == "isinstance"
            ):
                continue
            type_names = _isinstance_scatter_type_names(subnode.args[1])
            if not type_names:
                continue
            grouped[ast.unparse(subnode.args[0])].append(
                (subnode.lineno, ast.unparse(subnode), type_names)
            )
        for subject_expression, sites in sorted(grouped.items()):
            unique_type_names = sorted_tuple(
                {
                    type_name
                    for _line, _test, type_names in sites
                    for type_name in type_names
                }
            )
            if len(sites) < 2 or len(unique_type_names) < 2:
                continue
            line_numbers = tuple(line for line, _test, _types in sites)
            candidates.append(
                IsinstanceFamilyScatterCandidate(
                    file_path=str(module.path),
                    line=min(line_numbers),
                    qualname=qualname,
                    subject_expression=subject_expression,
                    type_names=unique_type_names,
                    site_count=len(sites),
                    line_numbers=line_numbers,
                    test_expressions=tuple(test for _line, test, _types in sites),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.file_path,
                item.line,
                item.qualname,
                item.subject_expression,
            ),
        )
    )


class IsinstanceFamilyScatterDetector(PerModuleIssueDetector):
    detector_id = "isinstance_family_scatter"
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Scattered concrete isinstance recovery should become polymorphic behavior",
        "Repeated `isinstance` checks against several concrete members of the same semantic family leave the consumer responsible for decoding family membership. The nominal boundary should expose polymorphic behavior through an ABC/base method or generated adapter family instead of scattering concrete type recovery through the walker.",
        "single polymorphic ABC/base authority owns concrete family evidence projection",
        "one owner repeatedly checks one subject against several concrete runtime types",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _isinstance_family_scatter_candidates(module):
            type_summary = ", ".join(candidate.type_names[:6])
            evidence = tuple(
                SourceLocation(
                    candidate.file_path,
                    line,
                    f"{candidate.qualname}:{candidate.subject_expression}",
                )
                for line in candidate.line_numbers[:8]
            )
            family_name = (
                f"{_camel_case(candidate.subject_expression.rsplit('.', 1)[-1])}"
                "Carrier"
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.qualname}` scatters {candidate.site_count} "
                        f"`isinstance` checks on `{candidate.subject_expression}` "
                        f"across concrete family types {type_summary}."
                    ),
                    evidence,
                    scaffold=(
                        f"class {family_name}(ABC):\n"
                        "    @abstractmethod\n"
                        "    def project_family_value(self, request): ...\n\n"
                        "# Make each concrete family member inherit the ABC and implement the hook.\n"
                        "# If the concrete classes are external, register nominal adapter classes once;\n"
                        "# consumers should call the polymorphic method, not enumerate concrete types."
                    ),
                    codemod_patch=(
                        "# Replace the scattered "
                        f"`isinstance({candidate.subject_expression}, ...)` branches "
                        f"in `{candidate.qualname}` with one polymorphic ABC/base method.\n"
                        "# Move each concrete case body onto the matching subclass or a registered "
                        "adapter; leave at most one fail-loud nominal boundary check."
                    ),
                    metrics=DispatchCountMetrics(
                        dispatch_site_count=candidate.site_count,
                        dispatch_axis=candidate.subject_expression,
                        literal_cases=candidate.type_names,
                    ),
                )
            )
        return findings


@dataclass(frozen=True, slots=True)
class RoleGuardedSurfaceAccessCandidate(RuntimeSubjectFunctionCandidate):
    role_type_name: str
    guard_expression: str
    accessed_members: tuple[str, ...]
    declared_members: tuple[str, ...]


def _declared_class_surface_members(node: ast.ClassDef) -> tuple[str, ...]:
    members: set[str] = set()
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not statement.name.startswith("__"):
                members.add(statement.name)
            continue
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            members.add(statement.target.id)
            continue
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    members.add(target.id)
    return tuple(sorted(members))


def _role_surface_members_by_type_name(
    modules: Sequence[ParsedModule],
) -> dict[str, tuple[str, ...]]:
    surfaces: dict[str, set[str]] = defaultdict(set)
    for module in modules:
        for node in ast.walk(module.module):
            if not isinstance(node, ast.ClassDef):
                continue
            members = _declared_class_surface_members(node)
            if len(members) == 0:
                continue
            surfaces[node.name].update(members)
            surfaces[f"{module.module_name}.{node.name}"].update(members)
    return {
        type_name: tuple(sorted(members))
        for type_name, members in sorted(surfaces.items())
    }


def _isinstance_guard_bindings(
    test: ast.AST,
) -> tuple[tuple[str, str, str], ...]:
    bindings: list[tuple[str, str, str]] = []
    for node in ast.walk(test):
        if not (
            isinstance(node, ast.Call)
            and len(node.args) == 2
            and not node.keywords
            and _ast_terminal_name(node.func) == "isinstance"
        ):
            continue
        type_names = _isinstance_scatter_type_names(node.args[1])
        if len(type_names) == 0:
            continue
        subject_expression = ast.unparse(node.args[0])
        guard_expression = ast.unparse(node)
        bindings.extend(
            (subject_expression, type_name, guard_expression)
            for type_name in type_names
        )
    return tuple(bindings)


def _accessed_declared_members(
    statements: Sequence[ast.stmt],
    *,
    subject_expression: str,
    declared_members: tuple[str, ...],
) -> tuple[str, ...]:
    declared_member_set = frozenset(declared_members)
    accessed: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            del node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            del node

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            del node

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if (
                node.attr in declared_member_set
                and ast.unparse(node.value) == subject_expression
            ):
                accessed.add(node.attr)
            self.generic_visit(node)

    visitor = Visitor()
    for statement in statements:
        visitor.visit(statement)
    return sorted_tuple(accessed)


def _role_guarded_surface_access_candidates_for_function(
    module: ParsedModule,
    qualname: str,
    function: NamedFunctionNode,
    role_surfaces: dict[str, tuple[str, ...]],
) -> Iterable[RoleGuardedSurfaceAccessCandidate]:
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        for (
            subject_expression,
            type_name,
            guard_expression,
        ) in _isinstance_guard_bindings(node.test):
            if type_name not in role_surfaces:
                continue
            declared_members = role_surfaces[type_name]
            accessed_members = _accessed_declared_members(
                node.body,
                subject_expression=subject_expression,
                declared_members=declared_members,
            )
            if len(accessed_members) == 0:
                continue
            yield RoleGuardedSurfaceAccessCandidate(
                file_path=str(module.path),
                line=node.lineno,
                qualname=qualname,
                subject_expression=subject_expression,
                role_type_name=type_name,
                guard_expression=guard_expression,
                accessed_members=accessed_members,
                declared_members=declared_members,
            )


def _role_guarded_surface_access_candidates(
    modules: Sequence[ParsedModule],
) -> tuple[RoleGuardedSurfaceAccessCandidate, ...]:
    role_surfaces = _role_surface_members_by_type_name(modules)
    return tuple(
        sorted(
            (
                candidate
                for module in modules
                for candidate in _collect_named_function_candidates(
                    module,
                    _role_guarded_surface_access_candidates_for_function,
                    role_surfaces,
                )
            ),
            key=lambda item: (
                item.file_path,
                item.line,
                item.qualname,
                item.subject_expression,
                item.role_type_name,
            ),
        )
    )


def _role_guarded_surface_access_summary(
    candidate: RoleGuardedSurfaceAccessCandidate,
) -> str:
    accessed_summary = ", ".join(candidate.accessed_members)
    return (
        f"`{candidate.qualname}` checks `{candidate.guard_expression}` "
        f"and then accesses role-owned member(s) {accessed_summary} "
        f"on `{candidate.subject_expression}`."
    )


def _role_guarded_surface_access_evidence(
    candidate: RoleGuardedSurfaceAccessCandidate,
) -> tuple[SourceLocation, ...]:
    return (
        SourceLocation(
            candidate.file_path,
            candidate.line,
            (
                f"{candidate.qualname}:"
                f"{candidate.subject_expression}:"
                f"{candidate.role_type_name}"
            ),
        ),
    )


def _role_guarded_surface_access_scaffold(
    candidate: RoleGuardedSurfaceAccessCandidate,
) -> str:
    return (
        "@dataclass(frozen=True)\n"
        "class SemanticOperationRequest:\n"
        "    target: object\n"
        "    role_owned_value: object\n\n"
        "# Build this request at the owner/call boundary, or type the callee "
        f"parameter as `{candidate.role_type_name}` if the role itself is required. "
        "Keep inheritance when it is the contract; do not use it as a hidden "
        "capability channel for a generic helper."
    )


def _role_guarded_surface_access_patch(
    candidate: RoleGuardedSurfaceAccessCandidate,
) -> str:
    return (
        f"# Replace the `{candidate.guard_expression}` block in "
        f"`{candidate.qualname}` with one explicit semantic input.\n"
        "# If the operation semantically requires the role, make the callee "
        "role-typed and fail before calling. If it only needs the value currently "
        "pulled from the role, pass that value/request explicitly from the owner."
    )


class RoleGuardedSurfaceAccessDetector(IssueDetector):
    detector_priority = -25
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Runtime role guard leaks role-owned semantics into the caller",
        "An `isinstance` guard proves a nominal role and the guarded block immediately consumes members declared on that same role. Inheritance is appropriate when that role is the declared contract; the smell is using inheritance as an optional side channel that lets a general caller discover and interpret role semantics it should have received explicitly.",
        "role-typed contract when the role is required, or explicit request/value boundary when only the role-owned value is required",
        "caller-side optional role probe followed by role-surface member access",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.RUNTIME_MEMBERSHIP,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        return [
            self.build_finding(
                _role_guarded_surface_access_summary(candidate),
                _role_guarded_surface_access_evidence(candidate),
                scaffold=_role_guarded_surface_access_scaffold(candidate),
                codemod_patch=_role_guarded_surface_access_patch(candidate),
                metrics=DispatchCountMetrics(
                    dispatch_site_count=1,
                    dispatch_axis=candidate.subject_expression,
                    literal_cases=(candidate.role_type_name,),
                ),
            )
            for candidate in _role_guarded_surface_access_candidates(modules)
        ]


_LITERAL_DISCRIMINATOR_AXIS_TOKENS = frozenset(
    (
        "action",
        "backend",
        "case",
        "family",
        "field",
        "format",
        "key",
        "kind",
        "mode",
        "policy",
        "profile",
        "rank",
        "schema",
        "scope",
        "source",
        "state",
        "status",
        "strategy",
        "type",
        "version",
    )
)
_LITERAL_DISCRIMINATOR_OWNER_TOKENS = frozenset(
    (
        "authority",
        "contract",
        "dispatch",
        "normalize",
        "parse",
        "parser",
        "policy",
        "project",
        "require",
        "resolve",
        "schema",
        "validate",
        "validator",
    )
)


@dataclass(frozen=True)
class LiteralDiscriminatorBranchCandidate:
    file_path: str
    line: int
    qualname: str
    axis_expression: str
    literal_cases: tuple[str, ...]
    test_expression: str


def _non_none_literal_case(value: ast.AST) -> str | None:
    if not isinstance(value, ast.Constant):
        return None
    if value.value is None:
        return None
    if not isinstance(value.value, (str, int, bool)):
        return None
    return repr(value.value)


def _literal_case_collection(value: ast.AST) -> tuple[str, ...]:
    if not isinstance(value, (ast.Set, ast.Tuple, ast.List)):
        return ()
    literal_cases = tuple(
        case
        for element in value.elts
        for case in (_non_none_literal_case(element),)
        if case is not None
    )
    if len(literal_cases) != len(value.elts):
        return ()
    return sorted_tuple(literal_cases, key=str)


def _literal_discriminator_axis_is_interesting(
    axis_expression: str, qualname: str
) -> bool:
    axis_tokens = set(_runtime_semantic_identifier_tokens(axis_expression))
    if axis_tokens & _LITERAL_DISCRIMINATOR_AXIS_TOKENS:
        return True
    owner_tokens = set(_runtime_semantic_identifier_tokens(qualname))
    return bool(owner_tokens & _LITERAL_DISCRIMINATOR_OWNER_TOKENS)


def _literal_discriminator_compare(
    test: ast.AST, qualname: str
) -> tuple[str, tuple[str, ...]] | None:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _literal_discriminator_compare(test.operand, qualname)
    if isinstance(test, ast.BoolOp):
        for value in test.values:
            match = _literal_discriminator_compare(value, qualname)
            if match is not None:
                return match
        return None
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not test.comparators:
        return None

    operator = test.ops[0]
    comparator = test.comparators[0]
    if isinstance(operator, (ast.Eq, ast.Is)):
        right_case = _non_none_literal_case(comparator)
        if right_case is not None:
            axis_expression = ast.unparse(test.left)
            if _literal_discriminator_axis_is_interesting(axis_expression, qualname):
                return axis_expression, (right_case,)
        left_case = _non_none_literal_case(test.left)
        if left_case is not None:
            axis_expression = ast.unparse(comparator)
            if _literal_discriminator_axis_is_interesting(axis_expression, qualname):
                return axis_expression, (left_case,)
        return None

    if isinstance(operator, (ast.In, ast.NotIn)):
        literal_cases = _literal_case_collection(comparator)
        if not literal_cases:
            return None
        axis_expression = ast.unparse(test.left)
        if _literal_discriminator_axis_is_interesting(axis_expression, qualname):
            return axis_expression, literal_cases
    return None


def _literal_discriminator_branch_candidates(
    module: ParsedModule,
) -> tuple[LiteralDiscriminatorBranchCandidate, ...]:
    candidates: list[LiteralDiscriminatorBranchCandidate] = []
    for qualname, function in _iter_named_functions(module):
        for node in _walk_nodes(function):
            if not isinstance(node, ast.If):
                continue
            match = _literal_discriminator_compare(node.test, qualname)
            if match is None:
                continue
            axis_expression, literal_cases = match
            candidates.append(
                LiteralDiscriminatorBranchCandidate(
                    file_path=str(module.path),
                    line=node.lineno,
                    qualname=qualname,
                    axis_expression=axis_expression,
                    literal_cases=literal_cases,
                    test_expression=ast.unparse(node.test),
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.file_path,
            item.line,
            item.qualname,
            item.axis_expression,
        ),
    )


class LiteralDiscriminatorBranchDetector(PerModuleIssueDetector):
    detector_id = "literal_discriminator_branch"
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Literal discriminator branch should be a nominal closed-axis authority",
        "A parser, validator, resolver, or authority method that branches directly on a literal discriminator value is locally interpreting a closed semantic axis. The axis should be represented by a nominal case family or generated contract table so missing cases fail loudly.",
        "nominal closed-axis authority owns literal discriminator cases",
        "single runtime branch compares a discriminator axis to literal case values",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in _literal_discriminator_branch_candidates(module):
            case_summary = ", ".join(candidate.literal_cases)
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.qualname}` branches on discriminator "
                        f"`{candidate.axis_expression}` with literal case(s) "
                        f"{case_summary}."
                    ),
                    (
                        SourceLocation(
                            candidate.file_path,
                            candidate.line,
                            (
                                f"{candidate.qualname}:"
                                f"{candidate.axis_expression}:{case_summary}"
                            ),
                        ),
                    ),
                    scaffold=(
                        "class ClosedAxisCase(ABC, metaclass=AutoRegisterMeta):\n"
                        "    __registry_key__ = 'case_key'\n"
                        "    case_key: ClassVar[object]\n"
                        "    @classmethod\n"
                        "    def for_case(cls, key):\n"
                        "        return cls.__registry__[key]\n\n"
                        "# Put allowed cases and per-case invariants in this family or a generated table;\n"
                        "# parser/validator code should dispatch through the authority, not branch on literals."
                    ),
                    codemod_patch=(
                        f"# Replace `{candidate.test_expression}` in "
                        f"`{candidate.qualname}` with a closed-axis authority lookup.\n"
                        "# Move case membership and invariants into nominal case declarations "
                        "or a generated artifact contract, and let unknown cases fail loudly."
                    ),
                    metrics=DispatchCountMetrics.from_literal_family(
                        candidate.axis_expression,
                        candidate.literal_cases,
                    ),
                )
            )
        return findings


declare_candidate_rule_detector(
    StringKeyedFormulaSubclassFamilyCandidate,
    high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "String-keyed formula subclasses should be derived from a typed policy algebra",
        "A subclass family that assigns string `kind`/`mode` keys and implements formulas on the subclasses is a split semantic authority: the key registry owns case identity while method bodies own case semantics. The formulas should be represented by a typed/generated policy algebra or nominal proof-backed carrier so runtime code interprets one schema instead of maintaining per-string behavior.",
        "typed/generated policy algebra owns case formulas with fail-loud validation",
        "subclasses repeat formula semantics behind literal kind/mode keys",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.base_class_name}` has string-keyed subclasses "
        f"{candidate.subclass_names} on `{candidate.key_attr_name}` with formulas "
        f"in methods {candidate.method_names}; keys={candidate.key_values}."
    ),
    scaffold=lambda candidate: (
        "class PolicyExprAuthority:\n"
        "    def evaluate(self, expr: PolicyExpr, sources: SourceValues) -> int: ...\n\n"
        "# Export case-specific formulas as typed data (Enum/dataclass/generated artifact),\n"
        "# then route all cases through one interpreter/authority."
    ),
    codemod_patch=lambda candidate: (
        f"# Replace literal `{candidate.key_attr_name}` subclasses under "
        f"`{candidate.base_class_name}` with a typed formula schema or generated "
        "policy artifact.\n"
        "# Keep runtime behavior in one generic interpreter; derive case formulas "
        "from the typed policy source so missing or unknown cases fail loudly."
    ),
    metrics=lambda candidate: DispatchCountMetrics.from_literal_family(
        candidate.key_attr_name,
        candidate.key_values,
    ),
    compression_certificate=lambda candidate: CompressionCertificate.from_object_family(
        manual_object_count=len(candidate.subclass_names)
        * max(1, len(candidate.method_names)),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("policy_expr_authority",),
            per_axis_objects=("typed_expr_variant",),
        ),
        semantic_axes=candidate.key_values,
    ),
    candidate_collector=_string_keyed_formula_subclass_family_candidates,
)


def _direct_terminal_return(
    body: Sequence[ast.stmt],
) -> ast.Return | None:
    trimmed_body = tuple(_trim_docstring_body(tuple(body)))
    if not trimmed_body:
        return None
    statement = trimmed_body[-1]
    if not isinstance(statement, ast.Return):
        return None
    return statement


def _return_expression_is_literal_default(statement: ast.Return) -> bool:
    return (
        statement.value is not None
        and _literal_default_kind(statement.value) is not None
    )


def _runtime_authority_return_guard_chain_from_elif(
    statement: ast.stmt,
) -> ReturnGuardBranchChain:
    if not isinstance(statement, ast.If):
        return ()
    chain: list[ReturnGuardBranchObservation] = []
    current: ast.If | None = statement
    while current is not None:
        branch_return = _direct_terminal_return(current.body)
        if branch_return is None:
            return ()
        chain.append(
            (
                current.lineno,
                ast.unparse(current.test),
                (
                    ast.unparse(branch_return.value)
                    if branch_return.value is not None
                    else "None"
                ),
                _return_expression_is_literal_default(branch_return),
            )
        )
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        current = None
    return tuple(chain) if len(chain) >= 2 else ()


def _runtime_authority_return_guard_chain_from_sequence(
    body: Sequence[ast.stmt],
    start: int,
) -> ReturnGuardBranchChain:
    chain: list[ReturnGuardBranchObservation] = []
    index = start
    while index < len(body):
        statement = body[index]
        if not isinstance(statement, ast.If) or statement.orelse:
            break
        branch_return = _direct_terminal_return(statement.body)
        if branch_return is None:
            break
        chain.append(
            (
                statement.lineno,
                ast.unparse(statement.test),
                (
                    ast.unparse(branch_return.value)
                    if branch_return.value is not None
                    else "None"
                ),
                _return_expression_is_literal_default(branch_return),
            )
        )
        index += 1
    return tuple(chain) if len(chain) >= 2 else ()


RUNTIME_AUTHORITY_RETURN_GUARD_COLLECTION_SPEC = BranchChainCollectionSpec(
    _runtime_authority_return_guard_chain_from_elif,
    _runtime_authority_return_guard_chain_from_sequence,
    branch_observation_first_line,
    return_guard_chain_has_literal_default,
)


def _runtime_authority_return_guard_chains_from_body(
    body: Sequence[ast.stmt],
) -> ReturnGuardBranchChains:
    return collect_nested_branch_chains_from_body(
        body,
        RUNTIME_AUTHORITY_RETURN_GUARD_COLLECTION_SPEC,
    )


def _runtime_authority_name_is_interesting(text: str) -> bool:
    if _runtime_semantic_axis_is_interesting(text):
        return True
    normalized = text.lower()
    return any((token in normalized for token in _RUNTIME_SEMANTIC_BRANCH_AXIS_TOKENS))


class RuntimeAuthorityBranchSemanticsDetector(PerModuleIssueDetector):
    detector_id = "runtime_authority_branch_semantics"
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Runtime authority return guards should move behind a formal policy authority",
        "An Authority method that chooses between runtime values and missing/default returns is still encoding operational semantics in Python. The formal boundary should own the branch cases and expose one declared policy/profile result.",
        "runtime authority return-guard choices are declared by the formal policy boundary",
        "an Authority method contains return guards with local missing/default semantics",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for qualname, function in _iter_named_functions(module):
            if "." not in qualname:
                continue
            owner_name = qualname.rsplit(".", 1)[0]
            if "Authority" not in owner_name:
                continue
            if not _runtime_authority_name_is_interesting(
                f"{owner_name}.{function.name}"
            ):
                continue
            for chain in _runtime_authority_return_guard_chains_from_body(
                function.body
            ):
                test_summary = ", ".join(
                    (
                        test_expression
                        for _line, test_expression, _result, _default in chain[:3]
                    )
                )
                result_summary = ", ".join(
                    (
                        result_expression
                        for _line, _test, result_expression, _default in chain[:3]
                    )
                )
                evidence = tuple(
                    SourceLocation(
                        str(module.path),
                        line,
                        f"{qualname}:{test_expression}->{result_expression}",
                    )
                    for line, test_expression, result_expression, _default in chain[:6]
                )
                findings.append(
                    self.build_finding(
                        (
                            f"`{qualname}` keeps {len(chain)} runtime authority "
                            f"return guards ({test_summary}) selecting "
                            f"{result_summary}."
                        ),
                        evidence,
                        scaffold=(
                            "class RuntimeAuthorityPolicy(ABC):\n"
                            "    @abstractmethod\n"
                            "    def select(self, source): ...\n\n"
                            "# Generate or load the concrete selector from the formal runtime profile;\n"
                            "# Python should consume the selected value, not branch on missing/default cases."
                        ),
                        codemod_patch=(
                            f"# Replace return guards in `{qualname}` with one formal "
                            "policy/profile authority.\n"
                            "# Move case precedence and missing/default behavior into the formal "
                            "runtime profile and make this authority a thin adapter over that result."
                        ),
                        metrics=BranchCountMetrics(branch_site_count=len(chain)),
                        capability_gap=(
                            "Authority method missing/default branch semantics are formal-boundary owned"
                        ),
                    )
                )
        return findings


_RELATION_COMPARISON_AXIS_TOKENS = frozenset(
    (
        "certificate",
        "certified",
        "case",
        "count",
        "family",
        "index",
        "key",
        "length",
        "original",
        "previous",
        "rank",
        "relation",
        "schema",
        "shape",
        "signature",
        "size",
        "type",
        "version",
    )
)
_RELATION_ARTIFACT_RESULT_TOKENS = frozenset(
    (
        "certificate",
        "carrier",
        "plan",
        "policy",
        "profile",
        "proof",
        "projection",
        "record",
        "result",
        "summary",
        "witness",
    )
)


def _relation_text_has_axis(text: str) -> bool:
    return bool(
        set(_runtime_semantic_identifier_tokens(text))
        & _RELATION_COMPARISON_AXIS_TOKENS
    )


def _relation_result_has_artifact(text: str) -> bool:
    return bool(
        set(_runtime_semantic_identifier_tokens(text))
        & _RELATION_ARTIFACT_RESULT_TOKENS
    )


def _relation_compare_has_axis(test: ast.AST) -> bool:
    if isinstance(test, ast.BoolOp):
        return any(_relation_compare_has_axis(value) for value in test.values)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _relation_compare_has_axis(test.operand)
    if not isinstance(test, ast.Compare):
        return False
    operands = (test.left, *tuple(test.comparators))
    return any(_relation_text_has_axis(ast.unparse(operand)) for operand in operands)


def _relation_artifact_factory_return(value: ast.AST) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    expression = ast.unparse(value)
    if _relation_result_has_artifact(expression):
        return expression
    if isinstance(value.func, ast.Attribute) and value.func.attr.startswith("from_"):
        return expression
    return None


def _load_bearing_relation_branch(
    statement: ast.If,
) -> SemanticBranchObservation | None:
    branch_return = _direct_terminal_return(statement.body)
    if branch_return is None or branch_return.value is None:
        return None
    test_expression = ast.unparse(statement.test)
    if not _relation_compare_has_axis(statement.test):
        return None
    result_expression = _relation_artifact_factory_return(branch_return.value)
    if result_expression is None:
        return None
    return statement.lineno, test_expression, result_expression


def _load_bearing_relation_elif_chain(
    statement: ast.stmt,
) -> SemanticBranchChain:
    if not isinstance(statement, ast.If):
        return ()
    chain: list[SemanticBranchObservation] = []
    current: ast.If | None = statement
    while current is not None:
        branch = _load_bearing_relation_branch(current)
        if branch is None:
            return ()
        chain.append(branch)
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            continue
        current = None
    return tuple(chain) if len(chain) >= 2 else ()


def _load_bearing_relation_sequential_guard_chain(
    body: Sequence[ast.stmt],
    start: int,
) -> SemanticBranchChain:
    chain: list[SemanticBranchObservation] = []
    index = start
    while index < len(body):
        statement = body[index]
        if not isinstance(statement, ast.If) or statement.orelse:
            break
        branch = _load_bearing_relation_branch(statement)
        if branch is None:
            break
        chain.append(branch)
        index += 1
    return tuple(chain) if len(chain) >= 2 else ()


LOAD_BEARING_RELATION_COLLECTION_SPEC = BranchChainCollectionSpec(
    _load_bearing_relation_elif_chain,
    _load_bearing_relation_sequential_guard_chain,
    branch_observation_first_line,
    all_branch_chains_active,
)


def _load_bearing_relation_chains_from_body(
    body: Sequence[ast.stmt],
) -> SemanticBranchChains:
    return collect_nested_branch_chains_from_body(
        body,
        LOAD_BEARING_RELATION_COLLECTION_SPEC,
    )


class LoadBearingRelationBranchDetector(PerModuleIssueDetector):
    detector_id = "load_bearing_relation_branch"
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Load-bearing relation dispatch should be a nominal case family",
        "An Authority method that chooses certificate, summary, or projection outputs through ordered relation/count/domain branches is encoding proof-relevant semantics in branch order. The relation cases should be named nominal classes with exactly-one-case selection.",
        "nominal relation-case algebra owns certificate/domain dispatch",
        "Authority method branches over proof-relevant source-domain relations",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for qualname, function in _iter_named_functions(module):
            if "." not in qualname:
                continue
            owner_name = qualname.rsplit(".", 1)[0]
            if "Authority" not in owner_name:
                continue
            for chain in _load_bearing_relation_chains_from_body(function.body):
                test_summary = ", ".join(
                    (test_expression for _line, test_expression, _result in chain[:3])
                )
                result_summary = ", ".join(
                    (result_expression for _line, _test, result_expression in chain[:3])
                )
                evidence = tuple(
                    SourceLocation(
                        str(module.path),
                        line,
                        f"{qualname}:{test_expression}->{result_expression}",
                    )
                    for line, test_expression, result_expression in chain[:6]
                )
                findings.append(
                    self.build_finding(
                        (
                            f"`{qualname}` keeps {len(chain)} load-bearing "
                            f"relation branches ({test_summary}) selecting "
                            f"{result_summary}."
                        ),
                        evidence,
                        scaffold=(
                            "class RelationCase(ABC, metaclass=AutoRegisterMeta):\n"
                            "    __registry_key__ = 'case_name'\n"
                            "    @abstractmethod\n"
                            "    def matches(self, request): ...\n"
                            "    @abstractmethod\n"
                            "    def certificate(self, request): ...\n\n"
                            "# One authority should require exactly one matching relation case;\n"
                            "# branch order must not carry proof-relevant semantics."
                        ),
                        codemod_patch=(
                            f"# Replace the ordered relation branches in `{qualname}` "
                            "with an AutoRegisterMeta-backed relation-case family.\n"
                            "# Move each source-domain/certificate relation into a named case and "
                            "make the authority require exactly one matching case."
                        ),
                        metrics=BranchCountMetrics(branch_site_count=len(chain)),
                        capability_gap=(
                            "proof-relevant certificate/domain dispatch is a nominal relation-case algebra"
                        ),
                    )
                )
        return findings


_SEMANTIC_CERTIFICATE_FALLBACK_TEST_TOKENS = frozenset(
    (
        "certificate",
        "certified",
        "compatibility",
        "count",
        "family",
        "formal",
        "key",
        "length",
        "policy",
        "proof",
        "reuse",
        "schema",
        "shape",
        "signature",
        "size",
        "theorem",
        "version",
    )
)
_SEMANTIC_CERTIFICATE_FALLBACK_RETURN_TOKENS = frozenset(
    (
        "certificate",
        "current",
        "fallback",
        "previous",
        "reuse",
        "witness",
    )
)


def _semantic_certificate_fallback_test_expression(
    test: ast.AST,
) -> str | None:
    if isinstance(test, ast.BoolOp):
        for value in test.values:
            expression = _semantic_certificate_fallback_test_expression(value)
            if expression is not None:
                return expression
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _semantic_certificate_fallback_test_expression(test.operand)
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return None
    operator = test.ops[0]
    if not isinstance(operator, (ast.NotEq, ast.NotIn)):
        return None
    expression = ast.unparse(test)
    lower_expression = expression.lower()
    tokens = set(_runtime_semantic_identifier_tokens(expression))
    if "signature" in tokens or "signature" in lower_expression:
        return expression
    if "certificate" in tokens or "certificate" in lower_expression:
        return expression
    text_matches = tuple(
        token
        for token in _SEMANTIC_CERTIFICATE_FALLBACK_TEST_TOKENS
        if token in lower_expression
    )
    if len((tokens & _SEMANTIC_CERTIFICATE_FALLBACK_TEST_TOKENS)) >= 3:
        return expression
    if len(text_matches) >= 3:
        return expression
    return None


def _semantic_certificate_fallback_return_expression(
    statement: ast.If,
) -> str | None:
    if any(isinstance(item, ast.Raise) for item in statement.body):
        return None
    branch_return = _direct_terminal_return(statement.body)
    if branch_return is None or branch_return.value is None:
        return None
    if _literal_default_kind(branch_return.value) is not None:
        return None
    expression = ast.unparse(branch_return.value)
    tokens = set(_runtime_semantic_identifier_tokens(expression))
    if not tokens & _SEMANTIC_CERTIFICATE_FALLBACK_RETURN_TOKENS:
        return None
    return expression


class SemanticCertificateFallbackDetector(PerModuleIssueDetector):
    detector_id = "semantic_certificate_fallback"
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic mismatch guards should produce typed certificates",
        "A runtime Authority that compares proof-relevant signatures, certificates, or domain counts and returns an existing object on mismatch is a hidden fallback. The semantic relation should be represented as a typed certificate whose construction either succeeds through a formal rule or fails loudly.",
        "typed formal certificate owns semantic reuse and mismatch behavior",
        "proof-relevant mismatch guard returns a runtime fallback object",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        detector = self

        class Visitor(ClassFunctionStackNodeVisitor):
            traverse_class_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )
            traverse_function_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )

            def visit_If(self, node: ast.If) -> None:
                test_expression = _semantic_certificate_fallback_test_expression(
                    node.test
                )
                return_expression = _semantic_certificate_fallback_return_expression(
                    node
                )
                if test_expression is not None and return_expression is not None:
                    qualname = ".".join(
                        (*tuple(self.class_stack), *tuple(self.function_stack))
                    )
                    owner = qualname or "module"
                    findings.append(
                        detector.build_finding(
                            (
                                f"`{owner}` branches on proof-relevant "
                                f"`{test_expression}` and returns "
                                f"`{return_expression}` instead of requiring a "
                                "typed certificate."
                            ),
                            (
                                SourceLocation(
                                    str(module.path),
                                    node.lineno,
                                    f"{owner}:{test_expression}->{return_expression}",
                                ),
                            ),
                            scaffold=(
                                "@dataclass(frozen=True)\n"
                                "class SemanticReuseCertificate:\n"
                                "    signature: FormalSignature\n"
                                "    payload: tuple[RuntimeBlock, ...]\n\n"
                                "    @classmethod\n"
                                "    def from_blocks(cls, blocks):\n"
                                "        # Validate one formal family here; raise on mismatch.\n"
                                "        ...\n\n"
                                "# Consumers should accept SemanticReuseCertificate, not raw blocks plus fallback branches."
                            ),
                            codemod_patch=(
                                f"# Replace the fallback branch in `{owner}` with "
                                "construction of a typed formal certificate.\n"
                                "# Move the mismatch rule into the certificate constructor and "
                                "raise when no theorem-backed runtime morphism exists."
                            ),
                            capability_gap=(
                                "proof-relevant reuse compatibility is a typed formal certificate"
                            ),
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(module.module)
        return findings


class MirroredConstructorValidationDetector(PerModuleIssueDetector):
    detector_id = "mirrored_constructor_validation"
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Mirrored constructor validators should move into the record schema",
        "A constructor call that fills several fields by calling validators with a string literal copy of the source variable keeps field identity in multiple places. The schema/record field declaration should own the source name and materializer once.",
        "single authoritative record-field schema with source and validator metadata",
        "constructor keyword fields mirror validation source names at the callsite",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        minimum = max(config.min_builder_keywords, 4)
        findings: list[RefactorFinding] = []
        detector = self

        class Visitor(ClassFunctionStackNodeVisitor):
            traverse_class_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )
            traverse_function_body = (
                ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
            )

            def visit_Call(self, node: ast.Call) -> None:
                mirrored = tuple(
                    (
                        keyword.arg,
                        validation_source,
                        validator_name,
                    )
                    for keyword in node.keywords
                    if keyword.arg is not None
                    for validation_source, validator_name in (
                        (_mirrored_validation_call(keyword.value) or (None, None)),
                    )
                    if validation_source is not None and validator_name is not None
                )
                if len(mirrored) >= minimum:
                    constructor = _constructor_name(node.func)
                    owner = ".".join(
                        (
                            *tuple(self.class_stack),
                            *tuple(self.function_stack),
                            constructor or "constructor",
                        )
                    ) or (constructor or "constructor")
                    output_fields = tuple(str(item[0]) for item in mirrored)
                    source_fields = tuple(str(item[1]) for item in mirrored)
                    validators = sorted_tuple({str(item[2]) for item in mirrored})
                    findings.append(
                        detector.build_finding(
                            (
                                f"`{owner}` mirrors {len(mirrored)} constructor "
                                f"validation sources for `{constructor}`; move "
                                "source names and validators onto the record schema."
                            ),
                            (
                                SourceLocation(
                                    str(module.path),
                                    int(getattr(node, "lineno", 1)),
                                    owner,
                                ),
                            ),
                            relation_context=(
                                "one constructor call repeats source-name literals "
                                "beside same-named source variables"
                            ),
                            scaffold=(
                                "@dataclass(frozen=True)\n"
                                "class Record:\n"
                                "    field_name: object = field(\n"
                                "        metadata={'source': 'source_name', "
                                "'materializer': validate_source}\n"
                                "    )\n\n"
                                "def materialize_record(source):\n"
                                "    return Record(**{\n"
                                "        field.name: field.metadata['materializer'](\n"
                                "            field.metadata['source'], "
                                "source[field.metadata['source']]\n"
                                "        )\n"
                                "        for field in dataclasses.fields(Record)\n"
                                "    })"
                            ),
                            codemod_patch=(
                                f"# Collapse mirrored constructor validation for "
                                f"`{constructor}` into dataclass field metadata or "
                                "one authoritative spec row per output field.\n"
                                "# Delete callsite pairs of "
                                "`validator('source_name', source_name)` once the "
                                "record schema materializes itself from a source map."
                            ),
                            metrics=MappingMetrics.from_field_names(
                                mapping_site_count=len(mirrored),
                                mapping_name=constructor,
                                field_names=output_fields,
                                source_name=owner,
                                identity_field_names=source_fields,
                            ),
                            capability_gap=(
                                "one schema-owned source/materializer declaration "
                                "per output field"
                            ),
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(module.module)
        return findings


class RepeatedBuilderCallDetector(IssueDetector):
    detector_id = "repeated_builder_calls"
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated field assignment should become an authoritative builder",
        "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once in an authoritative constructor, classmethod, or shared builder rather than copied across call sites.",
        "single authoritative record-builder mapping for a repeated constructor family",
        "same builder role repeated across sibling functions or methods",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        builders = sorted_tuple(
            (
                builder
                for module in modules
                for builder in _module_builder_call_shapes(module)
            ),
            key=lambda item: (item.file_path, item.lineno, item.symbol),
        )
        findings: list[RefactorFinding] = []
        findings.extend(self._exact_mapping_findings(builders, config))
        findings.extend(self._single_owner_family_findings(builders, config))
        return findings

    def _exact_mapping_findings(
        self,
        builders: tuple[BuilderCallShape, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        grouped: dict[
            (tuple[str, tuple[str, ...], tuple[str, ...]], list[BuilderCallShape])
        ] = defaultdict(list)
        for builder in builders:
            if _is_external_declarative_builder_call(builder):
                continue
            if len(builder.field_names) < config.min_builder_keywords:
                continue
            grouped[
                builder.callee_name, builder.field_names, builder.value_fingerprint
            ].append(builder)
        findings: list[RefactorFinding] = []
        for group in grouped.values():
            ordered = sorted_tuple(
                group, key=lambda item: (item.file_path, item.lineno)
            )
            if len(ordered) < 2 or len({builder.symbol for builder in ordered}) < 2:
                continue
            same_source = all(builder.source_arity == 1 for builder in ordered)
            if len(ordered) < 3 and not same_source:
                continue
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in ordered[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"Call `{ordered[0].callee_name}` repeats the same field-mapping shape across {len(ordered)} sites.",
                    evidence,
                    capability_gap=(
                        "single authoritative data-to-record mapping"
                        if same_source
                        else self.finding_spec.capability_gap
                    ),
                    scaffold=_builder_scaffold(ordered),
                    codemod_patch=_builder_patch(ordered),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(ordered),
                        mapping_name=ordered[0].callee_name,
                        field_names=ordered[0].field_names,
                        source_name=ordered[0].source_name,
                        identity_field_names=ordered[0].identity_field_names,
                    ),
                )
            )
        return findings

    def _single_owner_family_findings(
        self,
        builders: tuple[BuilderCallShape, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        grouped: dict[tuple[str, str], list[BuilderCallShape]] = defaultdict(list)
        for builder in builders:
            if _is_external_declarative_builder_call(builder):
                continue
            if not builder.field_names:
                continue
            grouped[(builder.owner_prefix, builder.callee_name)].append(builder)
        findings: list[RefactorFinding] = []
        minimum_sites = max(config.min_builder_keywords, 4)
        for owner_key, group in grouped.items():
            ordered = sorted_tuple(
                group, key=lambda item: (item.file_path, item.lineno)
            )
            if len(ordered) < minimum_sites:
                continue
            distinct_field_names = sorted_tuple(
                {name for builder in ordered for name in builder.field_names}
            )
            if len(distinct_field_names) < config.min_builder_keywords:
                continue
            if len({builder.field_names for builder in ordered}) < 2:
                continue
            owner_symbols = {builder.symbol for builder in ordered}
            if len(owner_symbols) != 1:
                continue
            owner_symbol, callee_name = owner_key
            evidence = tuple(
                (
                    SourceLocation(builder.file_path, builder.lineno, builder.symbol)
                    for builder in ordered[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"`{owner_symbol}` repeats builder `{callee_name}` across {len(ordered)} declarative sites with field family {distinct_field_names}.",
                    evidence,
                    capability_gap="single authoritative declarative builder table for one owner surface",
                    relation_context="one owner repeats a builder call family with varying declarative payload",
                    scaffold=_single_owner_builder_family_scaffold(callee_name),
                    codemod_patch=_single_owner_builder_family_patch(
                        owner_symbol, callee_name
                    ),
                    metrics=MappingMetrics.from_field_names(
                        mapping_site_count=len(ordered),
                        mapping_name=callee_name,
                        field_names=distinct_field_names,
                        source_name=owner_symbol,
                    ),
                )
            )
        return findings


_DECLARED_FIELD_EXTRACTION_REQUIRED_TOKENS = frozenset(("declared", "type"))
_DECLARED_FIELD_EXTRACTION_PAYLOAD_TOKENS = frozenset(
    ("field", "fields", "value", "values")
)


@dataclass(frozen=True)
class DeclaredFieldExtractionSite:
    """One call-site that manually extracts declared values for a nominal target."""

    file_path: str
    lineno: int
    ordinal: int
    owner_symbol: str
    callee_name: str
    target_type: str
    source_expression: str

    @property
    def object_name(self) -> str:
        return (
            f"{self.file_path}:{self.lineno}:"
            f"{self.ordinal}:"
            f"{self.owner_symbol}:{self.target_type}:{self.source_expression}"
        )

    @property
    def source_location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.owner_symbol)

    @property
    def axis_values(self) -> dict[str, str]:
        return {
            "callee_name": self.callee_name,
            "target_type": self.target_type,
            "source_expression": self.source_expression,
            "owner_symbol": self.owner_symbol,
            "file_path": self.file_path,
        }


class DeclaredFieldExtractionFanoutDetector(IssueDetector):
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Declared-field extraction should become a construction authority",
        "Manual declared-field extraction by nominal type is a transitional "
        "refactor state: call sites unpack a carrier surface instead of routing "
        "construction through one typed materialization authority.",
        "single typed construction/materialization authority for the declared "
        "field family",
        "declared-field extraction is repeated across a finite product of target, "
        "source, and owner axes",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _KEYWORD_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        sites = sorted_tuple(
            (
                site
                for module in modules
                for site in _declared_field_extraction_sites(module)
            ),
            key=lambda item: (item.file_path, item.lineno, item.owner_symbol),
        )
        if len(sites) < config.min_declared_field_extraction_sites:
            return []
        site_by_object_name = {site.object_name: site for site in sites}
        plans = _declared_field_extraction_authority_plans(
            sites,
            minimum_site_count=config.min_declared_field_extraction_sites,
        )
        return [
            self._finding_for_plan(plan, site_by_object_name)
            for plan in plans
            if self._plan_is_authority_boundary(plan)
        ]

    def _finding_for_plan(
        self,
        plan: FactorizationPlan,
        site_by_object_name: dict[str, DeclaredFieldExtractionSite],
    ) -> RefactorFinding:
        sites = tuple(
            site_by_object_name[object_name] for object_name in plan.orbit.object_names
        )
        target_types = sorted_tuple({site.target_type for site in sites})
        source_expressions = sorted_tuple({site.source_expression for site in sites})
        owner_symbols = sorted_tuple({site.owner_symbol for site in sites})
        callee_names = sorted_tuple({site.callee_name for site in sites})
        evidence = tuple(site.source_location for site in sites[:8])
        summary_subject = _declared_field_extraction_summary_subject(plan)
        return self.build_finding(
            (
                f"{summary_subject} manually extracts {len(target_types)} nominal "
                f"target type(s) through {len(sites)} declared-field call site(s)."
            ),
            evidence,
            capability_gap=(
                "one fail-loud typed materialization authority or coercion authority "
                "that derives the declared field mapping instead of spreading "
                "unpacking at call sites"
            ),
            relation_context=plan.normal_form,
            scaffold=_declared_field_extraction_scaffold(
                target_types, source_expressions
            ),
            codemod_patch=_declared_field_extraction_patch(
                sites[0].file_path,
                callee_names,
                target_types,
            ),
            compression_certificate=plan.compression_certificate,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(sites),
                mapping_name="/".join(callee_names),
                field_names=target_types,
                source_name="/".join(owner_symbols),
                identity_field_names=source_expressions,
            ),
        )

    @staticmethod
    def _plan_is_authority_boundary(plan: FactorizationPlan) -> bool:
        shared_axis_names = frozenset(plan.orbit.shared_axis_names)
        return bool(
            "callee_name" in shared_axis_names
            and (
                "target_type" in shared_axis_names
                or "owner_symbol" in shared_axis_names
                or len(plan.orbit.rows) >= 3
            )
        )


def _declared_field_extraction_authority_plans(
    sites: tuple[DeclaredFieldExtractionSite, ...],
    *,
    minimum_site_count: int,
) -> tuple[FactorizationPlan, ...]:
    """Return non-overlapping paid construction-authority plans for extraction sites."""

    engine = FactorizationEngine.from_mappings(
        (site.object_name, site.axis_values) for site in sites
    )
    plans = tuple(
        plan
        for plan in engine.candidate_plans(
            "declared_field_materialization_authority",
            minimum_object_count=minimum_site_count,
        )
        if "callee_name" in plan.orbit.shared_axis_names
    )
    best_nodes = FactorizationLattice.from_plans(plans).best_antichain()
    return tuple(node.plan for node in best_nodes)


def _declared_field_extraction_sites(
    module: ParsedModule,
) -> tuple[DeclaredFieldExtractionSite, ...]:
    sites: list[DeclaredFieldExtractionSite] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        @property
        def owner_symbol(self) -> str:
            if self.function_stack:
                return ".".join((*self.class_stack, *self.function_stack))
            if self.class_stack:
                return ".".join(self.class_stack)
            return "<module>"

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            site = _declared_field_extraction_site(
                module,
                node,
                self.owner_symbol,
                len(sites),
            )
            if site is not None:
                sites.append(site)
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(sites)


def _declared_field_extraction_site(
    module: ParsedModule,
    node: ast.Call,
    owner_symbol: str,
    ordinal: int,
) -> DeclaredFieldExtractionSite | None:
    callee_name = _declared_field_extraction_callee_name(node)
    if callee_name is None or len(node.args) < 2:
        return None
    return DeclaredFieldExtractionSite(
        file_path=str(module.path),
        lineno=node.lineno,
        ordinal=ordinal,
        owner_symbol=owner_symbol,
        callee_name=callee_name,
        target_type=_unparse_expression(node.args[0]),
        source_expression=_unparse_expression(node.args[1]),
    )


def _declared_field_extraction_callee_name(node: ast.Call) -> str | None:
    callee_name = _terminal_call_name(node.func)
    if callee_name is None:
        return None
    tokens = frozenset(_runtime_semantic_identifier_tokens(callee_name))
    if not _DECLARED_FIELD_EXTRACTION_REQUIRED_TOKENS <= tokens:
        return None
    if not (tokens & _DECLARED_FIELD_EXTRACTION_PAYLOAD_TOKENS):
        return None
    return callee_name


def _terminal_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _unparse_expression(node: ast.AST) -> str:
    return ast.unparse(node)


def _declared_field_extraction_summary_subject(plan: FactorizationPlan) -> str:
    shared_axes = dict(plan.orbit.shared_signature)
    if "target_type" in shared_axes:
        return f"`{shared_axes['target_type']}`"
    if "owner_symbol" in shared_axes:
        return f"`{shared_axes['owner_symbol']}`"
    if "callee_name" in shared_axes:
        return f"`{shared_axes['callee_name']}`"
    return "Declared-field extraction"


def _declared_field_extraction_scaffold(
    target_types: tuple[str, ...],
    source_expressions: tuple[str, ...],
) -> str:
    target_preview = ", ".join(target_types[:3]) or "TargetCarrier"
    source_preview = source_expressions[0] if source_expressions else "source"
    return (
        "@dataclass(frozen=True)\n"
        "class DeclaredFieldMaterializationAuthority:\n"
        "    target_types: tuple[type[object], ...]\n\n"
        "    def materialize(\n"
        "        self,\n"
        "        target_type: type[object],\n"
        "        source: object,\n"
        "    ) -> object:\n"
        "        # Fail loud unless target_type is declared by this authority.\n"
        "        ...\n\n"
        f"# Targets: {target_preview}\n"
        f"# Replace call-site unpacking from `{source_preview}` with "
        "authority.materialize(...)."
    )


def _declared_field_extraction_patch(
    target_file: str,
    callee_names: tuple[str, ...],
    target_types: tuple[str, ...],
) -> str:
    callee_preview = ", ".join(callee_names) or "declared-field extractor"
    target_preview = ", ".join(target_types[:4]) or "the nominal target family"
    return (
        f"# In {target_file}, replace repeated `{callee_preview}` unpacking with one "
        "typed materialization authority.\n"
        f"# Authority owns target family: {target_preview}.\n"
        "# Delete call-site **declared-field extraction once construction routes "
        "through the authority."
    )


_EXTERNAL_DECLARATIVE_BUILDER_CALLS = frozenset(
    {
        "add_argument",
    }
)


def _is_external_declarative_builder_call(builder: BuilderCallShape) -> bool:
    """Return whether the call is already owned by an external declaration DSL."""
    return builder.callee_name in _EXTERNAL_DECLARATIVE_BUILDER_CALLS


class RepeatedExportDictDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_export_dicts"
    observation_kind = ObservationKind.EXPORT_DICT
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated projection dict should become an authoritative schema",
        "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative row schema or projection builder instead of many hand-maintained dict literals.",
        "single authoritative projection schema for a repeated record or kwargs family",
        "same string-key projection role repeated across sibling functions or methods",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _PROJECTION_DICT_EXPORT_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, ExportDictShapeFamily, ExportDictShape
            )
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        export_shape = _as_export_shape(shape)
        return len(export_shape.key_names) >= config.min_export_keys

    def _group_key(self, shape: object) -> object:
        export_shape = _as_export_shape(shape)
        return (export_shape.key_names, export_shape.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        export_shapes = sorted_tuple(
            (_as_export_shape(shape) for shape in shapes),
            key=lambda item: (item.file_path, item.lineno),
        )
        if len(export_shapes) < 2:
            return None
        owner_symbols = {shape.symbol for shape in export_shapes}
        if len(owner_symbols) < 2:
            return None
        evidence = tuple(
            (
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in export_shapes[:6]
            )
        )
        return self.build_finding(
            f"String-key projection dict with keys {export_shapes[0].key_names} repeats across {len(export_shapes)} sites.",
            evidence,
            scaffold=_projection_schema_scaffold(export_shapes),
            codemod_patch=_projection_schema_patch(export_shapes),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(export_shapes),
                field_names=export_shapes[0].key_names,
                source_name=export_shapes[0].source_name,
                identity_field_names=export_shapes[0].identity_field_names,
            ),
        )


class ManualClassRegistrationDetector(GroupedShapeIssueDetector):
    finding_spec = certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual class registration should become metaclass-registry AutoRegisterMeta",
        "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. It should move into one authoritative `metaclass-registry` base so abstract-class skipping, uniqueness, and inheritance behavior are enforced in one place.",
        "single authoritative metaclass-registry class-registration algorithm with nominal class identity",
        "same registry key family repeated through manual class-level registration assignments",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_CLASS_LEVEL_POSITION_MANUAL_REGISTRATION_OBSERVATION_TAGS,
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[object]:
        return [
            shape
            for module in modules
            for shape in CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, RegistrationShapeFamily, RegistrationShape
            )
        ]

    def _group_key(self, shape: object) -> object:
        registration = _as_registration_shape(shape)
        return registration.registry_name

    def _finding_from_group(
        self, shapes: tuple[object, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        registrations = sorted_tuple(
            (_as_registration_shape(shape) for shape in shapes),
            key=lambda item: (item.file_path, item.lineno),
        )
        if len(registrations) < config.min_registration_sites:
            return None
        class_names = {item.registered_class for item in registrations}
        if len(class_names) < config.min_registration_sites:
            return None
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.lineno, item.symbol)
                for item in registrations[:6]
            )
        )
        registry_name = registrations[0].registry_name
        return self.build_finding(
            f"Registry `{registry_name}` is populated manually for {len(class_names)} classes across {len(registrations)} sites.",
            evidence,
            scaffold=_autoregister_scaffold(registry_name, class_names),
            codemod_patch=_autoregister_patch(
                registry_name, class_names, registrations
            ),
            metrics=RegistrationMetrics(
                registration_site_count=len(registrations),
                class_count=len(class_names),
                registry_name=registry_name,
                class_names=sorted_tuple(class_names),
                class_key_pairs=tuple(
                    (
                        f"{item.registered_class}={item.key_expression}"
                        for item in registrations
                    )
                ),
            ),
        )


class ManualConcreteSubclassRosterDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        ManualConcreteSubclassRosterCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual concrete-subclass roster should become a metaclass-registry base",
        "The docs treat mutable subclass rosters maintained through __init_subclass__ as framework logic. Abstract filtering, subclass discovery, and family access should live in one reusable `metaclass-registry` base instead of being reimplemented inside each domain family.",
        "single authoritative metaclass-registry concrete-subclass registration hook with reusable family discovery",
        "class family maintains a mutable subclass roster through __init_subclass__ and then queries it manually",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_CLASS_FAMILY_MANUAL_REGISTRATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, roster_candidate: ManualConcreteSubclassRosterCandidate
    ) -> RefactorFinding:
        evidence = [roster_candidate.evidence]
        evidence.extend(
            (
                SourceLocation(
                    roster_candidate.file_path,
                    roster_candidate.line,
                    f"{roster_candidate.class_name}.{consumer_name}",
                )
                for consumer_name in roster_candidate.consumer_names[:3]
            )
        )
        evidence.extend(
            (
                SourceLocation(
                    roster_candidate.file_path, roster_candidate.line, class_name
                )
                for class_name in roster_candidate.concrete_class_names[:2]
            )
        )
        guard_summary = (
            f" guarded by `{roster_candidate.guard_summary}`"
            if roster_candidate.guard_summary
            else ""
        )
        concrete_preview = ", ".join(roster_candidate.concrete_class_names[:3])
        config_block = (
            DISPATCH_ALGEBRA_AUTHORITY.declared_registry_key_block(
                roster_candidate.registration_site.selector_attr_name
            )
            if roster_candidate.registration_site.selector_attr_name is not None
            else DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(
                roster_candidate.concrete_class_names
            )
        )
        scaffold_imports = (
            "from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\n\n"
            if roster_candidate.registration_site.selector_attr_name is None
            else "from abc import ABC\nfrom metaclass_registry import AutoRegisterMeta\n\n"
        )
        return self.build_finding(
            (
                f"`{roster_candidate.class_name}` maintains roster `{roster_candidate.registry_name}` for {len(roster_candidate.concrete_class_names)} concrete subclasses ({concrete_preview}){guard_summary} and consumes it via {roster_candidate.consumer_names}."
            ),
            tuple(evidence[:6]),
            scaffold=(
                scaffold_imports
                + "class AutoRegisteredFamily(ABC, metaclass=AutoRegisterMeta):\n"
                + f"{config_block}\n\n"
                + "registered_types = tuple(AutoRegisteredFamily.__registry__.values())"
            ),
            codemod_patch=(
                f"# Remove manual roster `{roster_candidate.registry_name}` from `{roster_candidate.class_name}`.\n"
                "# Reuse one metaclass-registry base so descendant discovery and abstract filtering are not rewritten per family."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(roster_candidate.concrete_class_names),
                registry_name=roster_candidate.registry_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


class LatentImplementationRosterDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[LatentImplementationRosterCandidate]
):
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual implementation enumeration should derive from the ABC registry",
        "A collection or inline literal whose members mirror concrete implementations of one ABC family is a shadow registry even when it is just strings, class objects, instances, or a dict passed to `update(...)`. Membership should be derived from an AutoRegisterMeta-backed ABC or from a named projection policy over that registry.",
        "AutoRegisterMeta-backed implementation registry with generated projection surfaces",
        "manual collection or inline literal repeats the complete concrete implementation set of an ABC family",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, roster_candidate: LatentImplementationRosterCandidate
    ) -> RefactorFinding:
        key_attr = roster_candidate.key_attr_name or "derived_registry_key"
        projection_suffix = (
            f" with subset policy `{roster_candidate.projection_policy_hint}`; "
            f"missing {roster_candidate.missing_member_names}"
            if roster_candidate.projection_policy_hint is not None
            else ""
        )
        projection_expression = (
            f"tuple({roster_candidate.class_name}.__registry__.keys())"
            if roster_candidate.key_attr_name is not None
            else f"tuple({roster_candidate.class_name}.__registry__.values())"
        )
        registry_block = (
            DISPATCH_ALGEBRA_AUTHORITY.declared_registry_key_block(
                roster_candidate.key_attr_name
            )
            if roster_candidate.key_attr_name is not None
            else DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(
                roster_candidate.concrete_class_names
            )
        )
        return self.build_finding(
            (
                f"`{roster_candidate.roster_name}` is a `{roster_candidate.roster_kind}` roster "
                f"{roster_candidate.roster_member_names} via `{roster_candidate.projection_role}` "
                f"covering {roster_candidate.coverage_ratio:.2f} of concrete `{roster_candidate.class_name}` "
                f"implementations {roster_candidate.concrete_class_names}; derive it from registry key `{key_attr}`"
                f"{projection_suffix}."
            ),
            (roster_candidate.evidence,),
            scaffold=(
                "from abc import ABC\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                f"class {roster_candidate.class_name}(ABC, metaclass=AutoRegisterMeta):\n"
                f"{registry_block}\n\n"
                f"{roster_candidate.roster_name} = {projection_expression}"
            ),
            codemod_patch=(
                f"# Delete manual roster `{roster_candidate.roster_name}`.\n"
                f"# Promote `{roster_candidate.class_name}` to `ABC, metaclass=AutoRegisterMeta` and derive this projection from `__registry__`"
                + (
                    f" through a named `{roster_candidate.projection_policy_hint}` subset policy."
                    if roster_candidate.projection_policy_hint is not None
                    else "."
                )
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(roster_candidate.concrete_class_names),
                registry_name=roster_candidate.roster_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


class SemanticInheritanceFamilySSOTDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        SemanticInheritanceFamilySSOTCandidate
    ]
):
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Semantic inheritance family should have a metaclass membership SSOT",
        "When an inheritance root owns multiple concrete semantic leaves, family membership itself is architectural state. The root should derive membership from subclass declaration through `AutoRegisterMeta` instead of leaving membership implicit in scattered imports, subclass traversal, or downstream rosters.",
        "AutoRegisterMeta-backed ABC as the single source of truth for semantic inheritance membership",
        "behavioral or abstract inheritance family has multiple concrete leaves but no metaclass registration authority",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )
    detector_id = "semantic_inheritance_family_ssot"
    candidate_collector = _semantic_inheritance_family_ssot_candidates

    def _finding_for_candidate(
        self, family_candidate: SemanticInheritanceFamilySSOTCandidate
    ) -> RefactorFinding:
        key_block = DISPATCH_ALGEBRA_AUTHORITY.declared_registry_key_block(
            family_candidate.suggested_key_attr_name
        )
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        key_summary = (
            f"declared key attrs {family_candidate.key_attr_names}"
            if family_candidate.key_attr_names
            else "derive the key from class identity or add a canonical `registry_key`"
        )
        return self.build_finding(
            (
                f"`{family_candidate.class_name}` has {len(family_candidate.concrete_class_names)} concrete semantic leaves "
                f"({concrete_preview}) with methods {family_candidate.semantic_method_names} and abstract hooks "
                f"{family_candidate.abstract_method_names}, but no metaclass membership SSOT; {key_summary}. "
                f"AutoRegisterMeta pays rent by replacing {family_candidate.membership_object_count} membership object(s) "
                f"with {family_candidate.derived_projection_count} derived registry projection(s), margin {family_candidate.rent_margin}."
            ),
            (
                family_candidate.evidence,
                *(
                    SourceLocation(
                        family_candidate.file_path,
                        family_candidate.line,
                        class_name,
                    )
                    for class_name in family_candidate.concrete_class_names[:3]
                ),
            ),
            scaffold=(
                "from abc import ABC\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                f"class Registered{family_candidate.class_name}(ABC, metaclass=AutoRegisterMeta):\n"
                f"{key_block}\n\n"
                "    @classmethod\n"
                "    def registered_types(cls):\n"
                "        return tuple(cls.__registry__.values())"
            ),
            codemod_patch=(
                f"# Make `{family_candidate.class_name}` the class-time membership authority with `AutoRegisterMeta`.\n"
                f"# Keep only canonical key `{family_candidate.suggested_key_attr_name}` and semantic hooks on leaves; derive rosters, selectors, and projections from `cls.__registry__`.\n"
                f"# Rent proof: {family_candidate.membership_object_count} manual membership objects -> {family_candidate.derived_projection_count} derived projections, margin {family_candidate.rent_margin}."
            ),
            compression_certificate=family_candidate.compression_certificate,
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(family_candidate.concrete_class_names),
                registry_name=family_candidate.class_name,
                class_names=family_candidate.concrete_class_names,
                class_key_pairs=tuple(
                    (
                        f"{class_name}.{family_candidate.suggested_key_attr_name}"
                        for class_name in family_candidate.concrete_class_names
                    )
                ),
            ),
        )


class AutoRegisterMetaUnderRentedDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[AutoRegisterMetaRentCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegisterMeta family should prove its rent",
        "A metaclass registry pays rent when it derives a semantic family membership surface: a stable key axis, multiple registered leaves, a behavioral or abstract contract, and some registry projection or consumer. Without those coordinates, the metaclass is mostly signature noise and the same information usually belongs in a typed declaration table, enum, or ordinary ABC.",
        "AutoRegisterMeta-backed family with computed rent evidence over key axis, leaves, behavior, projections, and consumers",
        "class declares AutoRegisterMeta but lacks enough generic rent signals to justify metaclass registration",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )
    detector_id = "autoregister_meta_under_rented"
    candidate_collector = _autoregister_meta_rent_candidates

    def _finding_for_candidate(
        self, rent_candidate: AutoRegisterMetaRentCandidate
    ) -> RefactorFinding:
        key_summary = (
            f"key `{rent_candidate.registry_key_attr_name}`"
            if rent_candidate.registry_key_attr_name is not None
            else (
                f"key extractor `{rent_candidate.key_extractor_name}`"
                if rent_candidate.key_extractor_name is not None
                else "no stable key axis"
            )
        )
        concrete_preview = ", ".join(rent_candidate.concrete_class_names[:4]) or "none"
        return self.build_finding(
            (
                f"`{rent_candidate.class_name}` declares AutoRegisterMeta with {key_summary}, "
                f"{len(rent_candidate.concrete_class_names)} concrete leaf/leaves ({concrete_preview}), "
                f"dynamic factories {rent_candidate.dynamic_factory_symbols}, "
                f"behavior methods {rent_candidate.behavior_method_names}, abstract hooks "
                f"{rent_candidate.abstract_method_names}, projections {rent_candidate.registry_projection_names}, "
                f"and consumers {rent_candidate.consumer_symbols}; missing rent signal(s): "
                f"{rent_candidate.missing_rent_signals}. Rent margin {rent_candidate.rent_margin}."
            ),
            (rent_candidate.evidence,),
            scaffold=(
                "from abc import ABC, abstractmethod\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "class RentedFamily(ABC, metaclass=AutoRegisterMeta):\n"
                '    __registry_key__ = "semantic_key"\n\n'
                "    @classmethod\n"
                "    def for_key(cls, key):\n"
                "        return cls.__registry__[key]\n\n"
                "    @abstractmethod\n"
                "    def run(self, value): ..."
            ),
            codemod_patch=(
                f"# Prove or remove AutoRegisterMeta on `{rent_candidate.class_name}`.\n"
                "# Rent proof must expose a stable key axis, multiple registered leaves, a behavioral contract,\n"
                "# and a registry projection/consumer derived from `cls.__registry__`.\n"
                "# If the family is metadata-only or has no projection surface, replace it with a typed table or ordinary ABC."
            ),
            compression_certificate=rent_candidate.compression_certificate,
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(rent_candidate.concrete_class_names),
                registry_name=rent_candidate.class_name,
                class_names=rent_candidate.concrete_class_names,
            ),
        )


class PredicateSelectedConcreteFamilyDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        PredicateSelectedConcreteFamilyCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Predicate-selected concrete family should collapse into one metaclass-registry selector base",
        "The docs treat repeated scans over `registered_types()` plus `matches_*` predicates as family-selection framework logic. When a root class manually filters registered concrete descendants, enforces exactly one match, and then consumes the chosen subclass, the selection algorithm should live in one reusable `metaclass-registry` family base.",
        "single authoritative metaclass-registry predicate-selected concrete-family substrate",
        "registered concrete subclasses are manually scanned and cardinality-checked inside a family root",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_PREDICATE_CHAIN_REGISTRY_POPULATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: PredicateSelectedConcreteFamilyCandidate
    ) -> RefactorFinding:
        concrete_preview = ", ".join(family_candidate.concrete_class_names[:4])
        evidence = [family_candidate.evidence]
        evidence.extend(
            (
                SourceLocation(
                    family_candidate.file_path, family_candidate.line, class_name
                )
                for class_name in family_candidate.concrete_class_names[:3]
            )
        )
        return self.build_finding(
            (
                f"`{family_candidate.class_name}.{family_candidate.selector_method_name}` scans `registered_types()` and "
                f"predicate `{family_candidate.predicate_method_name}({family_candidate.context_param_name})` across "
                f"{len(family_candidate.concrete_class_names)} concrete leaves ({concrete_preview}) before manually choosing one match."
            ),
            tuple(evidence[:6]),
            scaffold=(
                f'from abc import ABC\nimport re\nfrom metaclass_registry import AutoRegisterMeta\nfrom typing import Generic, Self, TypeVar\n\nContextT = TypeVar("ContextT")\n\nclass PredicateSelectedConcreteFamily(ABC, Generic[ContextT], metaclass=AutoRegisterMeta):\n{DISPATCH_ALGEBRA_AUTHORITY.derived_registry_key_block(family_candidate.concrete_class_names)}\n\n    @classmethod\n    def matches_context(cls, context: ContextT) -> bool:\n        return True\n\n    @classmethod\n    def select_matching_type(cls, context: ContextT) -> type[Self]:\n        matches = tuple(\n            candidate\n            for candidate in cls.__registry__.values()\n            if candidate.matches_context(context)\n        )\n        ...\n'
            ),
            codemod_patch=(
                f"# Move `{family_candidate.class_name}` selection logic into a reusable predicate-selected family base.\n"
                "# Leave only `matches_context(...)` and family-specific error shaping on the root, and stop reimplementing `cls.__registry__.values()` scans."
            ),
        )


class ParallelMirroredLeafFamilyDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[ParallelMirroredLeafFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Parallel mirrored leaf families should derive from one axis-declared family substrate",
        "The docs treat mirrored registered leaf catalogs as framework duplication when the same contract is repeated across two family roots and only one nominal axis really varies. The axis and role table should be authoritative so registration and leaf generation are derived instead of hand-expanded twice.",
        "single authoritative axis-declared family or role-spec table that derives mirrored registered leaves",
        "two registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, mirrored_candidate: ParallelMirroredLeafFamilyCandidate
    ) -> RefactorFinding:
        shared_preview = ", ".join(mirrored_candidate.shared_leaf_family_names[:4])
        contract_preview = ", ".join(mirrored_candidate.contract_method_names)
        class_names = (
            mirrored_candidate.left.root_name,
            mirrored_candidate.right.root_name,
            *(item.symbol for item in mirrored_candidate.left.leaf_evidence),
            *(item.symbol for item in mirrored_candidate.right.leaf_evidence),
        )
        return self.build_finding(
            (
                f"`{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` expose mirrored `{contract_preview}` leaf catalogs "
                f"across {len(mirrored_candidate.shared_leaf_family_names)} shared role families ({shared_preview})."
            ),
            mirrored_candidate.evidence[:6],
            scaffold=(
                "@dataclass(frozen=True)\nclass FamilyRoleSpec:\n    role_name: str\n    axis_impls: tuple[callable, ...]\n\nclass GeneratedLeafFamily(ABC): ...\n# Declare the varying axis once, declare roles once, and derive leaf registration from the spec table."
            ),
            codemod_patch=(
                f"# Replace mirrored roots `{mirrored_candidate.left.root_name}` and `{mirrored_candidate.right.root_name}` with one axis-declared family substrate.\n"
                "# Move shared role names into one spec table and derive concrete leaf registration from that authority."
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=(
                    len(mirrored_candidate.left.leaf_evidence)
                    + len(mirrored_candidate.right.leaf_evidence)
                ),
                registry_name=(
                    f"{mirrored_candidate.left.root_name}/{mirrored_candidate.right.root_name}"
                ),
                class_names=class_names,
            ),
        )


class SentinelAttributeSimulationDetector(CandidateFindingDetector):
    finding_spec = finding_spec_template(
        PatternId.NOMINAL_BOUNDARY,
        "Sentinel attribute is simulating nominal identity",
        "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across multiple classes, the boundary should become a nominal family or another explicit identity handle.",
        "enumerable and enforceable nominal role identity",
        "same class-level sentinel attribute reused as a fake identity boundary",
        _NOMINAL_IDENTITY_ENUMERATION_PROVENANCE_CAPABILITY_TAGS,
        _SENTINEL_ATTRIBUTE_BRANCH_DISPATCH_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        candidates: list[object] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            branch_evidence = _attribute_branch_evidence(module, attr_name)
            if not branch_evidence:
                continue
            generic_name = attr_name.lower() in {"name", "label", "title"}
            if generic_name and len(branch_evidence) < 2:
                continue
            candidates.append((attr_name, tuple(evidence), tuple(branch_evidence)))
        return tuple(candidates)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        attr_name, evidence, branch_evidence = cast(
            tuple[str, tuple[SourceLocation, ...], tuple[SourceLocation, ...]],
            candidate,
        )
        return self.build_finding(
            f"Attribute `{attr_name}` is declared across {len(evidence)} classes and also drives {len(branch_evidence)} branch sites.",
            tuple((evidence + branch_evidence)[:6]),
            metrics=SentinelSimulationMetrics(
                class_count=len(evidence), branch_site_count=len(branch_evidence)
            ),
        )


class PredicateFactoryChainDetector(CandidateFindingDetector):
    finding_spec = finding_spec_template(
        PatternId.DISCRIMINATED_UNION,
        "Predicate chain should become a discriminated union family",
        "The docs say repeated predicate-driven variant selection should become an explicit subclass family with enumeration rather than an open-ended if/elif chain.",
        "exhaustive nominal variant discovery and extension",
        "same factory role repeated as predicate branches inside one function",
        _ENUMERATION_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PREDICATE_CHAIN_FACTORY_DISPATCH_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return tuple(
            (
                (str(module.path), function, branch_count)
                for function in _iter_functions(module.module)
                if (branch_count := _predicate_factory_chain_branch_count(function))
                is not None
            )
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, function, branch_count = cast(
            tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int], candidate
        )
        return self.build_finding(
            f"{function.name} contains a {branch_count}-branch predicate factory chain returning variant constructors.",
            (SourceLocation(file_path, function.lineno, function.name),),
            metrics=BranchCountMetrics(branch_site_count=branch_count),
        )


declare_typed_observation_detector(
    "ConfigAttributeDispatchDetector",
    finding_spec_template(
        PatternId.CONFIG_CONTRACTS,
        "Config dispatch is encoded through fragile attribute probing",
        "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name probing or ad hoc attribute comparisons.",
        "fail-loud polymorphic configuration contracts",
        "same config-family choice expressed through attribute-level probing",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_CONFIG_DISPATCH_OBSERVATION_TAGS,
    ),
    ConfigDispatchObservationFamily,
    ConfigDispatchObservation,
    "{module_path} contains {evidence_count} config-specific attribute probes or comparisons.",
    minimum_evidence_count=2,
)


class ConcreteConfigFieldProbeDetector(
    ConfiguredModuleCollectorCandidateDetector[ConcreteConfigFieldProbeCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CONFIG_CONTRACTS,
        "Concrete config backend is probing fields outside its declared contract",
        "The docs say concrete config-backed implementations should rely on declared config fields, not reflective probing of attributes that are absent from the concrete config type. That usually means the backend is borrowing another family's contract instead of owning its own configuration boundary.",
        "fail-loud concrete config contract for one backend family",
        "one concrete backend probes fields that are not declared by its concrete config type",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_CONFIG_DISPATCH_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, probe_candidate: ConcreteConfigFieldProbeCandidate
    ) -> RefactorFinding:
        missing_fields = ", ".join(probe_candidate.missing_field_names)
        reflective_builtins = "/".join(probe_candidate.probe_builtin_names)
        return self.build_finding(
            (
                f"`{probe_candidate.class_name}.{probe_candidate.method_name}` probes undeclared `{probe_candidate.config_type_name}` "
                f"fields {missing_fields} through `{reflective_builtins}` on `{probe_candidate.config_attr_name}`."
            ),
            (probe_candidate.evidence,),
            scaffold=(
                "class BackendConfig(ABC):\n    @property\n    @abstractmethod\n    def declared_parameter(self) -> object: ..."
            ),
            codemod_patch=(
                f"# Delete reflective field probes against `{probe_candidate.config_type_name}`.\n"
                "# Either move this backend onto its own declared config contract or use fields that the concrete config type actually owns."
            ),
        )


class GeneratedTypeLineageDetector(StaticModulePatternDetector):
    finding_spec = speculative_finding_spec(
        PatternId.TYPE_LINEAGE,
        "Generated types need explicit lineage tracking",
        "The docs say generated and rebuilt types need explicit nominal lineage so normalization, reverse lookup, and provenance remain exact.",
        "exact generated-type lineage and normalization",
        "same module combines runtime type generation with lineage-sensitive registries",
        _TYPE_LINEAGE_PROVENANCE_BIDIRECTIONAL_NORMALIZATION_CAPABILITY_TAGS,
        _RUNTIME_TYPE_GENERATION_LINEAGE_OBSERVATION_TAGS,
    )

    def _module_evidence(
        self, module: ParsedModule, config: DetectorConfig
    ) -> tuple[SourceLocation, ...]:
        generation_observations: tuple[RuntimeTypeGenerationObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module,
                RuntimeTypeGenerationObservationFamily,
                RuntimeTypeGenerationObservation,
            )
        )
        generation_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in generation_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
        lineage_observations: tuple[LineageMappingObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, LineageMappingObservationFamily, LineageMappingObservation
            )
        )
        lineage_sites = [
            SourceLocation(item.file_path, item.line, item.symbol)
            for item in lineage_observations
            if not _is_framework_lineage_symbol(item.symbol)
        ]
        if not generation_sites or not lineage_sites:
            return ()
        return tuple((generation_sites + lineage_sites)[:6])

    def _summary(
        self, module: ParsedModule, evidence: tuple[SourceLocation, ...]
    ) -> str:
        return f"{module.path} generates runtime types and also maintains type-lineage state."


class DualAxisResolutionDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.DUAL_AXIS_RESOLUTION,
        "Nested precedence walk should be a dual-axis resolution primitive",
        "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order contribute to resolution and provenance.",
        "explicit dual-axis precedence with provenance",
        "same function combines context hierarchy and type/MRO hierarchy",
        _DUAL_AXIS_RESOLUTION_PROVENANCE_MRO_ORDERING_CAPABILITY_TAGS,
        _NESTED_PRECEDENCE_WALK_SCOPE_HIERARCHY_MRO_HIERARCHY_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[DualAxisResolutionObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module,
                DualAxisResolutionObservationFamily,
                DualAxisResolutionObservation,
            )
        )
        for observation in observations:
            findings.append(
                self.build_finding(
                    f"{observation.symbol} nests scope-like axis `{observation.outer_axis_name}` with MRO/type-like axis `{observation.inner_axis_name}`.",
                    (
                        SourceLocation(
                            observation.file_path, observation.line, observation.symbol
                        ),
                    ),
                    metrics=ResolutionAxisMetrics(resolution_axis_count=2),
                )
            )
        return findings




@dataclass(frozen=True, slots=True)
class FailSoftFallbackCandidate:
    file_path: str
    line: int
    symbol: str
    fallback_kind: str
    expression: str


class _FailSoftFallbackVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.symbol_stack: list[str] = []
        self.candidates: list[FailSoftFallbackCandidate] = []

    @property
    def symbol(self) -> str:
        return ".".join(self.symbol_stack) if self.symbol_stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self._visit_statement_sequence(node.body)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self._visit_statement_sequence(node.body)
        self.symbol_stack.pop()

    def _visit_statement_sequence(self, statements: list[ast.stmt]) -> None:
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.If):
                self._record_guarded_broadening_return(
                    statement, statements[index + 1 :]
                )
            self.visit(statement)

    def visit_If(self, node: ast.If) -> None:
        self._visit_statement_sequence(node.body)
        self._visit_statement_sequence(node.orelse)
        self.visit(node.test)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if self._handler_returns_soft_fallback(handler):
                self._record(
                    handler,
                    "exception-handler fallback",
                    self._statement_text(handler.body[0]) if handler.body else "",
                )
        self._visit_statement_sequence(node.body)
        for handler in node.handlers:
            self._visit_statement_sequence(handler.body)
        self._visit_statement_sequence(node.orelse)
        self._visit_statement_sequence(node.finalbody)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
            values = node.value.values
            if len(values) >= 2 and self._looks_like_fallback_expr(values[-1]):
                self._record(node, "or-expression fallback", ast.unparse(node.value))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
            values = node.value.values
            if len(values) >= 2 and self._looks_like_fallback_expr(values[-1]):
                self._record(node, "or-assignment fallback", ast.unparse(node.value))
        self.generic_visit(node)

    def _record_guarded_broadening_return(
        self, node: ast.If, following: list[ast.stmt]
    ) -> None:
        narrowed_returns = tuple(
            name
            for statement in node.body
            for name in (self._returned_name(statement),)
            if name and self._looks_narrow_name(name)
        )
        if not narrowed_returns:
            return
        broad_returns = tuple(
            name
            for statement in following[:2]
            for name in (self._returned_name(statement),)
            if name and not self._looks_narrow_name(name)
        )
        if broad_returns:
            self._record(
                node,
                "guarded narrow return followed by broad return",
                f"{narrowed_returns[0]} -> {broad_returns[0]}",
            )

    def _handler_returns_soft_fallback(self, handler: ast.ExceptHandler) -> bool:
        if handler.type is None:
            return False
        handled_names = set(self._exception_type_names(handler.type))
        if not handled_names & {"TypeError", "RuntimeError", "ValueError", "KeyError"}:
            return False
        return any(
            isinstance(statement, ast.Return) and statement.value is not None
            for statement in handler.body
        )

    def _exception_type_names(self, node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            return (node.attr,)
        if isinstance(node, ast.Tuple):
            return tuple(
                name
                for element in node.elts
                for name in self._exception_type_names(element)
            )
        return ()

    def _returned_name(self, statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.Return):
            return None
        return self._expr_name(statement.value)

    def _expr_name(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _looks_narrow_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(
            token in lowered
            for token in ("scoped", "selected", "filtered", "resolved", "narrow")
        )

    def _looks_like_fallback_expr(self, node: ast.AST) -> bool:
        text = ast.unparse(node).lower()
        return any(
            token in text
            for token in ("fallback", "default", "records", "tables", "input_plan")
        )

    def _statement_text(self, statement: ast.stmt) -> str:
        return ast.unparse(statement)

    def _record(self, node: ast.AST, fallback_kind: str, expression: str) -> None:
        self.candidates.append(
            FailSoftFallbackCandidate(
                file_path=self.file_path,
                line=getattr(node, "lineno", 0),
                symbol=self.symbol,
                fallback_kind=fallback_kind,
                expression=expression,
            )
        )


class FailSoftFallbackDetector(PerModuleIssueDetector):
    detector_id = "fail_soft_fallback"
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Fail-soft fallback should become a fail-loud resolution contract",
        "A scoped or compatibility-specific resolution path falls back to a broader path after the code has already observed missing, conflicting, or unhandled evidence. That hides contract mismatches and lets stale or unscoped values stand in for authoritative resolution.",
        "explicit fail-loud resolution contract with provenance-bearing miss/conflict cases",
        "same local resolution path accepts a narrow candidate and then silently broadens to fallback data",
        _FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
        _PARTIAL_VIEW_BRANCH_DISPATCH_OBSERVATION_TAGS,
    )
    detector_priority = 5

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        visitor = _FailSoftFallbackVisitor(str(module.path))
        visitor.visit(module.module)
        findings: list[RefactorFinding] = []
        for candidate in visitor.candidates:
            findings.append(
                self.build_finding(
                    (
                        f"{candidate.symbol} uses {candidate.fallback_kind}: "
                        f"{candidate.expression}"
                    ),
                    (
                        SourceLocation(
                            candidate.file_path,
                            candidate.line,
                            candidate.symbol,
                        ),
                    ),
                )
            )
        return findings


declare_typed_observation_detector(
    "ManualVirtualMembershipDetector",
    finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "Manual class-marker membership should become custom isinstance semantics",
        "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks suggest a custom isinstance/subclass boundary rather than scattered manual probing.",
        "runtime-checkable virtual membership on nominal class identity",
        "same membership question repeated through class-marker probing",
        _VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_MARKER_PROBE_RUNTIME_MEMBERSHIP_OBSERVATION_TAGS,
    ),
    ClassMarkerObservationFamily,
    ClassMarkerObservation,
    "{module_path} performs {evidence_count} class-level marker checks on instances.",
    minimum_evidence_count=2,
)


# fmt: off
materialize_product_record(product_record_spec('_ExternalConcreteTypeIdentityTableCandidate', 'symbol: str; row_pairs: tuple[tuple[str, str, int], ...]', 'LineWitnessCandidate'))
# fmt: on


class ExternalConcreteTypeIdentityTableDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "External concrete type identity table should become capability registration",
        "A table of hardcoded external module/type string identities is recovering runtime membership from concrete implementation names. The nominal boundary should be an explicit capability registration surface owned by the integration layer, not a core table of third-party class names.",
        "extension-owned virtual membership registration boundary",
        "same registry table maps external concrete type identities to capability registration",
        _VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _REGISTRY_POPULATION_RUNTIME_MEMBERSHIP_SEMANTIC_STRING_LITERAL_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _external_concrete_type_identity_table_candidates(
            module, config
        ):
            evidence = tuple(
                (
                    SourceLocation(
                        candidate.file_path,
                        line,
                        f"{candidate.symbol}:{module_name}.{type_name}",
                    )
                    for module_name, type_name, line in candidate.row_pairs[:6]
                )
            )
            row_names = tuple(
                (
                    f"{module_name}.{type_name}"
                    for module_name, type_name, _line in candidate.row_pairs
                )
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.symbol}` hardcodes {len(candidate.row_pairs)} "
                        f"external concrete type identities: {', '.join(row_names[:5])}."
                    ),
                    evidence,
                    scaffold=(
                        "class RuntimeCapability(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'capability_key'\n    __skip_if_no_key__ = True\n    capability_key = None\n\n# Integration modules register concrete external classes with the capability boundary.\n# Core runtime code queries the nominal capability, not module/type strings."
                    ),
                    codemod_patch=(
                        f"# Replace `{candidate.symbol}` with explicit capability registration in the "
                        "owning integration modules; keep core validation against the nominal ABC."
                    ),
                    metrics=RegistrationMetrics(
                        registration_site_count=len(candidate.row_pairs),
                        registry_name=candidate.symbol,
                        class_key_pairs=row_names,
                    ),
                )
            )
        return findings


def _external_concrete_type_identity_table_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[_ExternalConcreteTypeIdentityTableCandidate, ...]:
    candidates: list[_ExternalConcreteTypeIdentityTableCandidate] = []

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            symbol = _assignment_symbol(node.targets)
            if symbol is not None:
                self._visit_table_value(node.value, symbol, node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            symbol = _assignment_symbol((node.target,))
            if symbol is not None and node.value is not None:
                self._visit_table_value(node.value, symbol, node.lineno)
            self.generic_visit(node)

        def _visit_table_value(
            self,
            node: ast.AST,
            symbol: str,
            line: int,
        ) -> None:
            if not _table_context_has_type_identity_signal(symbol, node):
                return
            row_pairs = _external_type_identity_rows(node)
            if len(row_pairs) < config.min_string_cases:
                return
            candidates.append(
                _ExternalConcreteTypeIdentityTableCandidate(
                    file_path=str(module.path),
                    line=line,
                    symbol=symbol,
                    row_pairs=row_pairs,
                )
            )

    Visitor().visit(module.module)
    return tuple(candidates)


def _assignment_symbol(targets: Sequence[ast.AST]) -> str | None:
    names = tuple(_assignment_target_name(target) for target in targets)
    names = tuple(name for name in names if name is not None)
    if len(names) != 1:
        return None
    return names[0]


def _assignment_target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parent = _assignment_target_name(target.value)
        if parent is None:
            return target.attr
        return f"{parent}.{target.attr}"
    return None


def _table_context_has_type_identity_signal(symbol: str, node: ast.AST) -> bool:
    names = [symbol]
    names.extend(
        (
            call_name
            for subnode in _walk_nodes(node)
            if isinstance(subnode, ast.Call)
            and (call_name := _call_name(subnode.func)) is not None
        )
    )
    normalized_names = tuple((name.lower() for name in names))
    return any(
        (
            "identity" in name or "type" in name or "class" in name
            for name in normalized_names
        )
    )


def _external_type_identity_rows(
    node: ast.AST,
) -> tuple[tuple[str, str, int], ...]:
    row_pairs: list[tuple[str, str, int]] = []
    seen_pairs: set[tuple[str, str, int]] = set()

    for table_node in _walk_nodes(node):
        row_nodes: Sequence[ast.AST]
        if isinstance(table_node, (ast.Tuple, ast.List, ast.Set)):
            row_nodes = table_node.elts
        elif isinstance(table_node, ast.Dict):
            row_nodes = tuple((key for key in table_node.keys if key is not None))
        else:
            continue

        local_rows: list[tuple[str, str, int]] = []
        for row_node in row_nodes:
            row_pair = _external_type_identity_pair(row_node)
            if row_pair is None:
                continue
            local_rows.append(row_pair)

        if len(local_rows) < 3:
            continue
        for row_pair in local_rows:
            if row_pair in seen_pairs:
                continue
            seen_pairs.add(row_pair)
            row_pairs.append(row_pair)

    return tuple(row_pairs)


def _external_type_identity_pair(
    row_node: ast.AST,
) -> tuple[str, str, int] | None:
    for subnode in _walk_nodes(row_node):
        if not isinstance(subnode, ast.Call):
            continue
        if len(subnode.args) < 2:
            continue
        module_name = _constant_string(subnode.args[0])
        type_name = _constant_string(subnode.args[1])
        if module_name is None or type_name is None:
            continue
        if _looks_like_external_concrete_type_identity(module_name, type_name):
            return (module_name, type_name, subnode.lineno)
    return None


def _looks_like_external_concrete_type_identity(
    module_name: str,
    type_name: str,
) -> bool:
    if module_name == type_name:
        return False
    if not _IDENTIFIER_PATH_RE.fullmatch(module_name):
        return False
    if not _IDENTIFIER_PATH_RE.fullmatch(type_name):
        return False
    if "." not in module_name and module_name.lower() != module_name:
        return False
    return True


_IDENTIFIER_PATH_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


declare_typed_observation_detector(
    "DynamicInterfaceGenerationDetector",
    speculative_finding_spec(
        PatternId.DYNAMIC_INTERFACE,
        "Dynamic interface generation is present or required",
        "The docs treat dynamically generated empty or near-empty interface types as explicit nominal identity handles when structure alone cannot express membership.",
        "explicit runtime-generated nominal interface identity",
        "same module generates interface-like nominal types at runtime",
        _GENERATED_INTERFACE_IDENTITY_VIRTUAL_MEMBERSHIP_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _RUNTIME_TYPE_GENERATION_INTERFACE_IDENTITY_OBSERVATION_TAGS,
    ),
    InterfaceGenerationObservationFamily,
    InterfaceGenerationObservation,
    "{module_path} contains {evidence_count} runtime-generated interface sites.",
    evidence_limit=6,
)


declare_typed_observation_detector(
    "SentinelTypeMarkerDetector",
    finding_spec_template(
        PatternId.SENTINEL_TYPE_MARKER,
        "Unique sentinel type marker is present or should be used",
        "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when exact capability identity matters more than payload.",
        "exact capability-marker identity independent of structure",
        "same module creates or uses unique nominal sentinel markers",
        _CAPABILITY_MARKER_IDENTITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _SENTINEL_TYPE_CAPABILITY_MARKER_OBSERVATION_TAGS,
    ),
    SentinelTypeObservationFamily,
    SentinelTypeObservation,
    "{module_path} contains {evidence_count} sentinel-type capability marker sites.",
    evidence_limit=6,
)


declare_typed_observation_detector(
    "DynamicMethodInjectionDetector",
    finding_spec_template(
        PatternId.TYPE_NAMESPACE_INJECTION,
        "Dynamic method injection belongs in a type-namespace pattern",
        "The docs say behavior that must affect all current and future instances belongs in a class namespace pattern, not in repeated instance-level patching.",
        "shared type-namespace mutation for a nominal family",
        "same module mutates class behavior through runtime namespace injection",
        _SHARED_TYPE_NAMESPACE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DYNAMIC_METHOD_INJECTION_TYPE_NAMESPACE_OBSERVATION_TAGS,
    ),
    DynamicMethodInjectionObservationFamily,
    DynamicMethodInjectionObservation,
    "{module_path} contains {evidence_count} dynamic type-namespace injection sites.",
    evidence_limit=6,
)


class AttributeProbeDetector(PerModuleIssueDetector):
    detector_id = "attribute_probes"
    finding_spec = finding_spec_template(
        PatternId.ABC_TEMPLATE_METHOD,
        "Semantic role recovered from attribute probing",
        "Repeated hasattr/getattr/AttributeError logic means the code is recovering identity from a partial structural view. The documented fix is to migrate this region toward an ABC contract with direct method calls and fail-loud guarantees.",
        "declared semantic role identity and import-time enforcement",
        "same module-level probing layer across multiple call sites",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
        _ATTRIBUTE_PROBE_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        observations: tuple[AttributeProbeObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, AttributeProbeObservationFamily, AttributeProbeObservation
            )
        )
        observations = tuple(
            (item for item in observations if not _is_framework_attribute_probe(item))
        )
        total = len(observations)
        if total < config.min_attribute_probes:
            return []
        evidence = tuple(
            (
                SourceLocation(item.file_path, item.line, item.symbol)
                for item in observations[:6]
            )
        )
        return [
            self.build_finding(
                f"{module.path} contains {total} attribute-probe sites.",
                evidence,
                metrics=ProbeCountMetrics(probe_site_count=total),
            )
        ]


class InlineLiteralDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Inline literal dispatch should be a registry",
        "When the same observed value is split across several sibling literal branches, the docs say the local rule family should be moved into one authoritative dispatch object instead of repeating inline branch logic. When the cases select behavior, prefer an auto-registered class family over a handwritten enum table.",
        "single authoritative dispatch representation for a closed local rule family, preferably an auto-registered behavior family when the cases are behavioral",
        "same branch role repeated inline inside a module block",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _LITERAL_BRANCH_DISPATCH_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        observations: tuple[LiteralDispatchObservation, ...] = (
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module,
                InlineStringLiteralDispatchObservationFamily,
                LiteralDispatchObservation,
            )
        )
        for observation in observations:
            branch_count = len(observation.branch_lines)
            if branch_count < config.min_attribute_probes:
                continue
            evidence = tuple(
                (
                    SourceLocation(observation.file_path, line, observation.symbol)
                    for line in observation.branch_lines[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"{module.path} repeats literal-case dispatch over `{observation.axis_expression}` across {branch_count} sibling branches with cases {observation.literal_cases}.",
                    evidence,
                    relation_context=f"same branch role repeated inline inside {observation.scope_owner or 'module block'}",
                    metrics=DispatchCountMetrics.from_literal_family(
                        observation.axis_expression, observation.literal_cases
                    ),
                    scaffold=LITERAL_DISPATCH_FINDING_FACTORY.authority_scaffold(
                        observation
                    ),
                    codemod_patch=_literal_dispatch_authority_patch(observation),
                )
            )
        return findings


class StringDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Closed-family dispatch expressed through strings",
        "The docs prefer enum- or type-keyed O(1) dispatch for closed families. Repeated string branches suggest the code is using a weaker representation than the domain requires. If those strings select implementations, the stronger form is an auto-registered family keyed by the stable nominal axis.",
        "closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        "same dispatch role repeated through string comparisons or string-key registries",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings = LITERAL_DISPATCH_FINDING_FACTORY.findings(
            self,
            module,
            config,
            StringLiteralDispatchObservationFamily,
            case_summary_label="cases",
            relation_case_label="literal string cases",
        )
        dict_evidence = _dispatch_dict_locations(module, config.min_string_cases)
        if dict_evidence:
            findings.append(
                self.build_finding(
                    (
                        f"{module.path} contains {len(dict_evidence)} string-key dispatch table site(s) that encode a closed family."
                    ),
                    tuple(dict_evidence[:6]),
                    certification=STRONG_HEURISTIC,
                    relation_context=(
                        "same closed family encoded in string-key dispatch tables rather than one nominal dispatch boundary"
                    ),
                    codemod_patch=(
                        "# Replace handwritten string-key dispatch tables with one authoritative nominal family and dispatch through `Family.for_key(...)` / `Family.__registry__`. # Keep any string-key projection as a derived view of the auto-registered family."
                    ),
                    metrics=DispatchCountMetrics(
                        dispatch_site_count=len(dict_evidence)
                    ),
                )
            )
        return findings


_SUBSTRING_CLASSIFIER_NAME_RE = re.compile(
    r"(^|_)(case|category|field|key|kind|mode|name|role|selector|state|status|tag|type)($|_)"
)


@dataclass(frozen=True)
class SemanticSubstringClassifierCandidate:
    evidence: SourceLocation
    owner: str
    literal: str
    classifier_expression: str
    operation: str


def _expression_source(node: ast.AST) -> str:
    return ast.unparse(node)


def _is_str_conversion(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
    )


_EXCEPTION_VALUE_NAMES = frozenset(("exc", "err", "error", "exception"))


def _is_exception_message_conversion(node: ast.AST) -> bool:
    return (
        _is_str_conversion(node)
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in _EXCEPTION_VALUE_NAMES
    )


def _classifier_leaf_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class ScalarClassifierOperandAuthority:
    @classmethod
    def matches(cls, node: ast.AST) -> bool:
        if _is_exception_message_conversion(node):
            return False
        if _is_str_conversion(node):
            return True
        leaf_name = _classifier_leaf_name(node)
        if leaf_name is not None:
            return bool(_SUBSTRING_CLASSIFIER_NAME_RE.search(leaf_name))
        if isinstance(node, ast.Subscript):
            return cls.matches(node.value)
        return False


def _substring_classifier_candidate(
    module: ParsedModule,
    owner: str,
    node: ast.Compare,
    left: ast.AST,
    right: ast.AST,
) -> SemanticSubstringClassifierCandidate | None:
    literal = _constant_string(left)
    classifier = right
    if literal is None:
        literal = _constant_string(right)
        classifier = left
    if literal is None or len(literal) < 2:
        return None
    if not ScalarClassifierOperandAuthority.matches(classifier):
        return None
    classifier_expression = _expression_source(classifier)
    return SemanticSubstringClassifierCandidate(
        evidence=SourceLocation(str(module.path), node.lineno, classifier_expression),
        owner=owner,
        literal=literal,
        classifier_expression=classifier_expression,
        operation="substring membership",
    )


_STRING_SHAPE_CLASSIFIER_METHODS = frozenset(("endswith", "startswith"))


def _string_shape_classifier_candidate(
    module: ParsedModule,
    owner: str,
    node: ast.Call,
) -> SemanticSubstringClassifierCandidate | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in _STRING_SHAPE_CLASSIFIER_METHODS:
        return None
    if len(node.args) != 1:
        return None
    literal = _constant_string(node.args[0])
    if literal is None or len(literal) < 2:
        return None
    classifier = node.func.value
    if not ScalarClassifierOperandAuthority.matches(classifier):
        return None
    classifier_expression = _expression_source(classifier)
    return SemanticSubstringClassifierCandidate(
        evidence=SourceLocation(str(module.path), node.lineno, classifier_expression),
        owner=owner,
        literal=literal,
        classifier_expression=classifier_expression,
        operation=f"{node.func.attr} method",
    )


def _semantic_substring_classifier_candidates(
    module: ParsedModule,
) -> tuple[SemanticSubstringClassifierCandidate, ...]:
    candidates: list[SemanticSubstringClassifierCandidate] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        traverse_class_body = ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        traverse_function_body = (
            ClassFunctionStackNodeVisitor.traverse_trimmed_node_body
        )

        def visit_Compare(self, node: ast.Compare) -> None:
            operands = (node.left, *node.comparators)
            for operator, left, right in zip(node.ops, operands, operands[1:]):
                if not isinstance(operator, (ast.In, ast.NotIn)):
                    continue
                candidate = _substring_classifier_candidate(
                    module,
                    self.qualname,
                    node,
                    left,
                    right,
                )
                if candidate is not None:
                    candidates.append(candidate)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            candidate = _string_shape_classifier_candidate(
                module,
                self.qualname,
                node,
            )
            if candidate is not None:
                candidates.append(candidate)
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(candidates)


declare_candidate_rule_detector(
    SemanticSubstringClassifierCandidate,
    high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Substring classifiers should become nominal dispatch",
        "Literal substring, prefix, or suffix checks over key/name-like values derive behavior from spelling instead of a declared variant axis. That makes the rule partial and lets unrelated spelling changes alter runtime behavior.",
        "nominal classifier values with exact equality, enum keys, or registered variants",
        "single-site semantic classification is performed through literal string-shape testing",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _STRING_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.owner}` classifies `{candidate.classifier_expression}` "
        f"with {candidate.operation} literal `{candidate.literal}`."
    ),
    scaffold=lambda candidate: (
        "class CaseAxis(Enum):\n"
        "    FIRST = 'first'\n\n"
        "def classify(value: CaseAxis) -> Result:\n"
        "    return CASE_DISPATCH[value]()"
    ),
    codemod_patch=lambda candidate: (
        "# Replace substring membership with an exact nominal classifier. "
        "Parse external text at the boundary once, then dispatch on the "
        "typed case value."
    ),
    metrics=lambda candidate: DispatchCountMetrics(
        dispatch_site_count=1,
        dispatch_axis=candidate.classifier_expression,
        literal_cases=(candidate.literal,),
    ),
    candidate_collector=_semantic_substring_classifier_candidates,
)


class NumericLiteralDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Closed-family dispatch expressed through numeric IDs",
        "The docs treat repeated numeric pattern or mode IDs the same way as magic strings: the domain axis is real but undeclared. Replace the literal-ID branches with a nominal family keyed by a stable axis; if the cases select behavior, prefer an auto-registered family over a handwritten lookup table.",
        "closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        "same dispatch role repeated through numeric literal comparisons",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_CAPABILITY_TAGS,
        _LITERAL_ID_DISPATCH_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        return LITERAL_DISPATCH_FINDING_FACTORY.findings(
            self,
            module,
            config,
            NumericLiteralDispatchObservationFamily,
            case_summary_label="numeric cases",
            relation_case_label="numeric literal cases",
        )


class RepeatedHardcodedStringDetector(CandidateFindingDetector):
    detector_id = "repeated_hardcoded_strings"
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated hardcoded semantic string should become authoritative",
        "The docs treat repeated hardcoded semantic keys as a coherence failure: the key should be declared once as an authoritative constant, enum member, or nominal handle instead of being copied across sites.",
        "single authoritative semantic-key declaration",
        "same semantic key duplicated across decision-bearing or declarative sites",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _SEMANTIC_STRING_LITERAL_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (
                (str(module.path), literal, tuple(sites))
                for literal, sites in _semantic_string_literal_sites(module).items()
                if len(sites) >= config.min_hardcoded_string_sites
            )
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, literal, sites = cast(
            tuple[str, str, tuple[SourceLocation, ...]], candidate
        )
        return self.build_finding(
            f"String literal `{literal}` repeats across {len(sites)} semantic sites in {file_path}.",
            tuple(sites[:6]),
            metrics=MappingMetrics(
                mapping_site_count=len(sites),
                field_count=1,
                mapping_name=literal,
                field_names=(literal,),
            ),
        )


_STATIC_PAYLOAD_WRITE_METHODS = frozenset(
    {"dump", "dumps", "write", "write_text", "write_bytes", "writelines"}
)
_WRITE_MODE_TOKENS = frozenset({"w", "a", "x", "wt", "at", "xt", "wb", "ab", "xb"})


# fmt: off
materialize_product_records((
    product_record_spec('StaticPayloadStats', 'payload_line_count: int; largest_literal_line_count: int; marker_kinds: tuple[str, ...]'),
    product_record_spec('EmbeddedStaticPayloadCandidate', 'function_name: str; line_count: int; static_payload_line_count: int; largest_literal_line_count: int; marker_kinds: tuple[str, ...]; sink_kinds: tuple[str, ...]; call_site_count: int', 'QualnameLineWitnessCandidate'),
))
# fmt: on

_RuntimeFunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
_RuntimeFunctionSequence: TypeAlias = Sequence[_RuntimeFunctionNode]
_SurfaceFunctionItems: TypeAlias = tuple[tuple[str, _RuntimeFunctionNode], ...]


def _function_line_count(function: _RuntimeFunctionNode) -> int:
    end_lineno = (
        function.end_lineno if function.end_lineno is not None else function.lineno
    )
    return end_lineno - function.lineno + 1


@dataclass(frozen=True)
class SurfaceFunctionIndex:
    functions: _SurfaceFunctionItems

    @classmethod
    @lru_cache(maxsize=None)
    def from_module(cls, module_node: ast.Module) -> "SurfaceFunctionIndex":
        functions: list[tuple[str, _RuntimeFunctionNode]] = []

        def visit_body(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
            for statement in body:
                if isinstance(statement, ast.ClassDef):
                    visit_body(
                        _trim_docstring_body(statement.body), (*prefix, statement.name)
                    )
                    continue
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append((".".join((*prefix, statement.name)), statement))

        visit_body(_trim_docstring_body(module_node.body), ())
        return cls(tuple(functions))


@lru_cache(maxsize=None)
def _walk_function_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    stack = list(reversed(_trim_docstring_body(function.body)))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stack.extend(reversed(tuple(ast.iter_child_nodes(node))))
    return tuple(nodes)


def _payload_literal_line_count(value: str) -> int:
    return max(1, len(value.splitlines()))


def _static_payload_marker_kinds(value: str) -> tuple[str, ...]:
    markers: set[str] = set()
    if len(value.strip()) < 80 or _payload_literal_line_count(value) < 2:
        return ()
    if value.count("<") >= 3 and re.search("</?[A-Za-z][\\w:.-]*(\\s|>|/)", value):
        markers.add("markup")
    if value.count("{") + value.count("}") >= 4 and value.count(":") >= 2:
        markers.add("structured_data")
    if (
        value.count("{") + value.count("}") >= 4
        and value.count(";") >= 2
        and re.search("\\b(class|const|function|let|var)\\b", value)
    ):
        markers.add("script_or_stylesheet")
    if re.search("\\b(SELECT|WITH|INSERT|UPDATE|CREATE|FROM|WHERE)\\b", value, re.I):
        markers.add("query_language")
    if re.search("^[A-Za-z0-9_.-]+:\\s+.+$", value, re.M):
        markers.add("keyed_config")
    return sorted_tuple(markers)


def _static_payload_stats(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> StaticPayloadStats:
    literal_values = tuple(
        (
            node.value
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    )
    payload_values = tuple(
        (
            value
            for value in literal_values
            if len(value.strip()) >= 80 and _payload_literal_line_count(value) >= 2
        )
    )
    marker_kinds = sorted_tuple(
        {
            marker
            for value in payload_values
            for marker in _static_payload_marker_kinds(value)
        }
    )
    return StaticPayloadStats(
        payload_line_count=sum(
            (_payload_literal_line_count(value) for value in payload_values)
        ),
        largest_literal_line_count=max(
            (_payload_literal_line_count(value) for value in payload_values), default=0
        ),
        marker_kinds=marker_kinds,
    )


def _is_write_mode_literal(value: ast.AST) -> bool:
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return False
    mode = value.value.replace("+", "")
    return mode in _WRITE_MODE_TOKENS or any(token in mode for token in ("w", "a", "x"))


def _static_payload_sink_kinds(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    sink_kinds: set[str] = set()
    for node in _walk_function_body_nodes(function):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _STATIC_PAYLOAD_WRITE_METHODS
            ):
                sink_kinds.add(node.func.attr)
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                positional_modes = tuple(node.args[1:2])
                keyword_modes = tuple(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "mode"
                    )
                )
                if any(
                    (
                        _is_write_mode_literal(mode)
                        for mode in (*positional_modes, *keyword_modes)
                    )
                ):
                    sink_kinds.add("open-write")
        elif isinstance(node, ast.Return) and isinstance(
            node.value, (ast.Constant, ast.JoinedStr)
        ):
            sink_kinds.add("return-payload")
    return sorted_tuple(sink_kinds)


@dataclass(frozen=True)
class ReferenceCountIndex:
    total_counts: Counter[str]
    function_counts_by_id: dict[int, Counter[str]]

    @staticmethod
    def symbol_counts(
        root: ast.AST,
        *,
        include_node: Callable[[ast.AST], bool] | None = None,
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        for node in _walk_nodes(root):
            if include_node is not None and (not include_node(node)):
                continue
            if isinstance(node, ast.Name):
                counts[node.id] += 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                counts[node.value] += 1
        return counts

    @staticmethod
    def symbol_count(root: ast.AST, symbol_name: str) -> int:
        count = 0
        for node in _walk_nodes(root):
            if isinstance(node, ast.Name):
                if node.id == symbol_name:
                    count += 1
            elif isinstance(node, ast.Attribute):
                if node.attr == symbol_name:
                    count += 1
            elif isinstance(node, ast.Constant) and node.value == symbol_name:
                count += 1
        return count

    @classmethod
    def from_modules(cls, modules: Sequence[ParsedModule]) -> "ReferenceCountIndex":
        total_counts: Counter[str] = Counter()
        for module in modules:
            total_counts.update(cls.symbol_counts(module.module))
        return cls(
            total_counts=total_counts,
            function_counts_by_id={},
        )

    def reference_count_outside_function(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef, symbol_name: str
    ) -> int:
        function_key = id(function)
        if function_key not in self.function_counts_by_id:
            self.function_counts_by_id[function_key] = Counter()
        function_counts = self.function_counts_by_id[function_key]
        if symbol_name not in function_counts:
            function_counts[symbol_name] = self.symbol_count(function, symbol_name)
        return self.total_counts[symbol_name] - function_counts[symbol_name]


def _embedded_static_payload_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
) -> tuple[EmbeddedStaticPayloadCandidate, ...]:
    candidates: list[EmbeddedStaticPayloadCandidate] = []
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        reference_modules or (module,)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if not _is_private_symbol_name(function.name):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_static_payload_function_lines:
            continue
        stats = _static_payload_stats(function)
        if stats.payload_line_count < config.min_static_payload_literal_lines:
            continue
        if not stats.marker_kinds:
            continue
        sink_kinds = _static_payload_sink_kinds(function)
        if not sink_kinds:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        candidates.append(
            EmbeddedStaticPayloadCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                line_count=line_count,
                static_payload_line_count=stats.payload_line_count,
                largest_literal_line_count=stats.largest_literal_line_count,
                marker_kinds=stats.marker_kinds,
                sink_kinds=sink_kinds,
                call_site_count=sum(
                    (
                        isinstance(node, ast.Call)
                        for node in _walk_function_body_nodes(function)
                    )
                ),
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (item.file_path, item.line, item.qualname))
    )


class DeadEmbeddedStaticPayloadDetector(
    ConfiguredModuleCollectorCandidateDetector[EmbeddedStaticPayloadCandidate]
):
    cache_granularity = DetectorCacheGranularity.GLOBAL
    candidate_collector = _embedded_static_payload_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced embedded static-payload emitter should collapse",
        "A private function that is not referenced in its module but still embeds and writes a large static artifact payload is a duplicate derived-view authority. Delete it if it is genuinely dead; if it is reached dynamically, move the payload to a template/resource or generate it from an authoritative schema.",
        "single authoritative template/resource or generated schema for static artifact views",
        "private unreferenced emitter owns a large embedded static payload independently of call flow",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_EXPORT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        private_reference_context = _private_reference_detector_context(tuple(modules))
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _embedded_static_payload_candidates(
                module,
                config,
                reference_index=private_reference_context.reference_index,
            )
        ]

    def _finding_for_candidate(
        self, payload_candidate: EmbeddedStaticPayloadCandidate
    ) -> RefactorFinding:
        marker_summary = ", ".join(payload_candidate.marker_kinds)
        sink_summary = ", ".join(payload_candidate.sink_kinds)
        return self.build_finding(
            (
                f"`{payload_candidate.qualname}` spans {payload_candidate.line_count} lines, embeds "
                f"{payload_candidate.static_payload_line_count} static payload lines ({marker_summary}), "
                f"writes through {sink_summary}, and has no in-module references."
            ),
            (payload_candidate.evidence,),
            scaffold=(
                f"# First verify whether `{payload_candidate.qualname}` is externally or dynamically invoked.\n# If not, delete the emitter and its embedded payload.\n# If it is live, move the payload into a template/resource or generate the artifact from one authoritative schema."
            ),
            codemod_patch=(
                f"# Collapse `{payload_candidate.qualname}` as a dead or duplicate static-payload view.\n"
                "# Keep at most one artifact authority: a template/resource file or a generated schema-backed writer."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=payload_candidate.line_count,
                branch_site_count=0,
                call_site_count=payload_candidate.call_site_count,
                parameter_count=0,
                callee_family_count=max(1, len(payload_candidate.sink_kinds)),
            ),
        )


@dataclass(frozen=True)
class LineCountedWitnessCandidate(LineWitnessCandidate):
    line_count: int


@dataclass(frozen=True)
class LineCountedQualnameCandidate(
    QualnameWitnessNameMixin,
    LineCountedWitnessCandidate,
):
    qualname: str


@dataclass(frozen=True)
class CallCountedQualnameCandidate(LineCountedQualnameCandidate):
    call_site_count: int


@dataclass(frozen=True)
class UnreferencedPrivateFunctionCandidate(CallCountedQualnameCandidate):
    function_name: str


@dataclass(frozen=True)
class DanglingPrivateMethodCandidate(CallCountedQualnameCandidate):
    owner_name: str
    method_name: str


@dataclass(frozen=True)
class PrivateHelperResiduePlan:
    classvar_names: tuple[str, ...]
    property_hook_names: tuple[str, ...]
    behavior_hook_names: tuple[str, ...]
    transported_parameter_names: tuple[str, ...]
    callsite_axis_count: int
    shared_statement_count: int
    residue_normal_form: str


@dataclass(frozen=True)
class PrivateHelperPlacementPlan:
    placement_kind: str
    insertion_owner_name: str
    insertion_detail: str
    residue_plan: PrivateHelperResiduePlan
    caller_owner_names: tuple[str, ...]


@dataclass(frozen=True)
class NonNominalPrivateHelperCandidate(CallCountedQualnameCandidate):
    function_name: str
    parameter_names: tuple[str, ...]
    caller_symbols: tuple[str, ...]
    placement_plan: PrivateHelperPlacementPlan


@dataclass(frozen=True)
class PrivateHelperClusterClassification:
    owner_name: str
    cluster_normal_form: str
    shared_stem: str
    classification_role_tokens: tuple[str, ...]
    return_kinds: tuple[str, ...]
    constructed_type_names: tuple[str, ...]


@dataclass(frozen=True)
class PrivateHelperSemanticClusterCandidate(LineCountedWitnessCandidate):
    helper_names: tuple[str, ...]
    semantic_family: str
    classification: PrivateHelperClusterClassification
    shared_parameter_names: tuple[str, ...]
    shared_call_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]
    line_numbers: tuple[int, ...]
    cluster_size: int
    evidence_locations: ClassVar[ZippedSourceLocationEvidenceProperty] = (
        ZippedSourceLocationEvidenceProperty("line_numbers", "helper_names")
    )


@dataclass(frozen=True)
class PrivateHelperResidueNameTemplate:
    kind: str
    prefix: str
    suffix: str
    uppercase: bool
    parameter_name_is_authority: bool


@dataclass(frozen=True)
class PrivateHelperAuthorityRole:
    authority_role_tokens: tuple[str, ...]
    suffix: str
    drop_tokens: tuple[str, ...]


_RuntimeFunctionsByQualname: TypeAlias = dict[str, _RuntimeFunctionNode]


class _PrivateHelperResidueKind(StrEnum):
    ATTRIBUTE = "attribute"
    CALL = "call"
    CONSTANT = "constant"
    EXPRESSION = "expression"
    NAME = "name"
    SELF_ATTR = "self_attr"
    VALUE = "value"


class _PrivateHelperResidueSink(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[_PrivateHelperResidueKind | None] = None

    @classmethod
    def for_kind(cls, kind: _PrivateHelperResidueKind) -> "_PrivateHelperResidueSink":
        if kind in cls.__registry__:
            sink_class = cls.__registry__[kind]
        else:
            sink_class = _PrivateHelperPropertyResidueSink
        return sink_class()

    @abstractmethod
    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        raise NotImplementedError


class _PrivateHelperConstantResidueSink(_PrivateHelperResidueSink):
    kind = _PrivateHelperResidueKind.CONSTANT

    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        classvar_names.append(residue_name)


class _PrivateHelperCallResidueSink(_PrivateHelperResidueSink):
    kind = _PrivateHelperResidueKind.CALL

    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        behavior_hook_names.append(residue_name)


class _PrivateHelperPropertyResidueSink(_PrivateHelperResidueSink):
    def append_residue(
        self,
        residue_name: str,
        *,
        classvar_names: list[str],
        property_hook_names: list[str],
        behavior_hook_names: list[str],
    ) -> None:
        property_hook_names.append(residue_name)


class DerivedCandidateCollectorContracts:
    def names(self, modules: Sequence[ParsedModule]) -> frozenset[str]:
        return frozenset(
            (
                _candidate_collector_name_from_class_name(node.name)
                for module in modules
                for node in module.module.body
                if isinstance(node, ast.ClassDef)
                and HELPER_SYNTAX_PROJECTION_AUTHORITY.class_declares_finding_spec(node)
            )
        )


DERIVED_CANDIDATE_COLLECTOR_CONTRACTS = DerivedCandidateCollectorContracts()


def _has_external_protocol_shape(
    function: _RuntimeFunctionNode,
) -> bool:
    if function.decorator_list:
        return True
    return function.name.endswith("_")


def _unreferenced_private_function_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
) -> tuple[UnreferencedPrivateFunctionCandidate, ...]:
    candidates: list[UnreferencedPrivateFunctionCandidate] = []
    contract_modules = reference_modules or (module,)
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        contract_modules
    )
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(contract_modules)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        if function.name in derived_candidate_collector_contract_names:
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_unreferenced_private_function_lines:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        candidates.append(
            UnreferencedPrivateFunctionCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                line_count=line_count,
                call_site_count=sum(
                    (
                        isinstance(node, ast.Call)
                        for node in _walk_function_body_nodes(function)
                    )
                ),
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (item.file_path, item.line, item.qualname))
    )


_DETECTOR_OVERRIDE_HOOK_NAMES = frozenset(("_collect_findings", "_findings_for_module"))
_DETECTOR_BASE_NAME_SUFFIXES = (
    "CandidateDetector",
    "IssueDetector",
    "ModuleDetector",
)
ClassBaseNameRows: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ClassBaseNameIndex:
    """Immutable class-base lookup for one AST module."""

    base_names_by_qualname: ClassBaseNameRows

    @classmethod
    @lru_cache(maxsize=None)
    def from_module(cls, module: ast.Module) -> "ClassBaseNameIndex":
        base_names_by_qualname: dict[str, tuple[str, ...]] = {}
        class_stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                class_stack.append(node.name)
                base_names_by_qualname[".".join(class_stack)] = tuple(
                    base_name
                    for base in node.bases
                    for base_name in (_ast_terminal_name(base),)
                    if base_name is not None
                )
                self.generic_visit(node)
                class_stack.pop()

        Visitor().visit(module)
        return cls(tuple(sorted(base_names_by_qualname.items())))

    def base_names(self, owner_name: str) -> tuple[str, ...]:
        for qualname, base_names in self.base_names_by_qualname:
            if qualname == owner_name:
                return base_names
        return ()


def _is_detector_override_hook(
    module: ParsedModule,
    owner_name: str,
    method_name: str,
) -> bool:
    if method_name not in _DETECTOR_OVERRIDE_HOOK_NAMES:
        return False
    base_names = ClassBaseNameIndex.from_module(module.module).base_names(owner_name)
    return any(
        base_name.endswith(_DETECTOR_BASE_NAME_SUFFIXES) for base_name in base_names
    )


def _dangling_private_method_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    reference_index: ReferenceCountIndex | None = None,
) -> tuple[DanglingPrivateMethodCandidate, ...]:
    candidates: list[DanglingPrivateMethodCandidate] = []
    reference_index = reference_index or ReferenceCountIndex.from_modules(
        reference_modules or (module,)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." not in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        owner_name = qualname.rsplit(".", 1)[0]
        if _is_detector_override_hook(module, owner_name, function.name):
            continue
        if _has_external_protocol_shape(function):
            continue
        line_count = _function_line_count(function)
        if line_count < config.min_unreferenced_private_function_lines:
            continue
        if (
            reference_index.reference_count_outside_function(function, function.name)
            > 0
        ):
            continue
        candidates.append(
            DanglingPrivateMethodCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                owner_name=owner_name,
                method_name=function.name,
                line_count=line_count,
                call_site_count=sum(
                    (
                        isinstance(node, ast.Call)
                        for node in _walk_function_body_nodes(function)
                    )
                ),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.qualname)
    )


class FunctionParameterNameProjection:
    def names(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, ...]:
        return tuple(
            (
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            )
        )


FUNCTION_PARAMETER_NAME_PROJECTION = FunctionParameterNameProjection()


def _function_symbol_references(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    return frozenset(
        (
            (
                node.id
                if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else node.value
            )
            for node in _walk_function_body_nodes(function)
            if (
                isinstance(node, ast.Name)
                or isinstance(node, ast.Attribute)
                or (isinstance(node, ast.Constant) and isinstance(node.value, str))
            )
        )
    )


@dataclass(frozen=True)
class PrivateHelperCallGraph:
    caller_symbols_by_name: dict[str, tuple[str, ...]]
    functions_by_qualname: _RuntimeFunctionsByQualname

    @classmethod
    def from_modules(cls, modules: Sequence[ParsedModule]) -> "PrivateHelperCallGraph":
        callers_by_symbol: dict[str, set[str]] = {}
        functions_by_qualname: _RuntimeFunctionsByQualname = {}
        for module in modules:
            for qualname, function in SurfaceFunctionIndex.from_module(
                module.module
            ).functions:
                functions_by_qualname[qualname] = function
                for symbol_name in _function_symbol_references(function):
                    callers_by_symbol.setdefault(symbol_name, set()).add(qualname)
        return cls(
            caller_symbols_by_name={
                symbol_name: sorted_tuple(caller_symbols)
                for symbol_name, caller_symbols in callers_by_symbol.items()
            },
            functions_by_qualname=functions_by_qualname,
        )

    def caller_symbols(self, *, function_name: str, qualname: str) -> tuple[str, ...]:
        return tuple(
            caller_symbol
            for caller_symbol in self.caller_symbols_by_name.get(function_name, ())
            if caller_symbol != qualname
        )

    def caller_functions(
        self, *, function_name: str, qualname: str
    ) -> tuple[_RuntimeFunctionNode, ...]:
        return tuple(
            (
                function
                for caller_symbol in self.caller_symbols(
                    function_name=function_name, qualname=qualname
                )
                if (function := self.functions_by_qualname.get(caller_symbol))
                is not None
            )
        )


@dataclass(frozen=True)
class PrivateReferenceDetectorContext:
    """Shared repo-wide indexes for private-reference detector families."""

    modules: tuple[ParsedModule, ...]

    @cached_property
    def reference_index(self) -> ReferenceCountIndex:
        return ReferenceCountIndex.from_modules(self.modules)

    @cached_property
    def derived_candidate_collector_contract_names(self) -> frozenset[str]:
        return DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(self.modules)

    @cached_property
    def private_helper_call_graph(self) -> PrivateHelperCallGraph:
        return PrivateHelperCallGraph.from_modules(self.modules)

    @cached_property
    def class_index(self) -> ClassFamilyIndex:
        return build_class_family_index(list(self.modules))


@lru_cache(maxsize=4)
def _private_reference_detector_context(
    modules: tuple[ParsedModule, ...],
) -> PrivateReferenceDetectorContext:
    return PrivateReferenceDetectorContext(modules)


def _private_helper_caller_owner_names(
    caller_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    return sorted_tuple(
        (
            caller_symbol.rsplit(".", 1)[0]
            for caller_symbol in caller_symbols
            if "." in caller_symbol
        )
    )


def _private_helper_unique_class_symbol(
    class_index: ClassFamilyIndex, owner_name: str
) -> str | None:
    symbols = tuple(
        symbol
        for symbol, indexed_class in class_index.classes_by_symbol.items()
        if indexed_class.qualname == owner_name
        or indexed_class.simple_name == owner_name.rsplit(".", 1)[-1]
    )
    if len(symbols) == 1:
        return symbols[0]
    return None


def _private_helper_deepest_common_ancestor_symbol(
    class_index: ClassFamilyIndex, class_symbols: tuple[str, ...]
) -> str | None:
    return (
        Maybe.of(class_symbols)
        .filter(bool)
        .map(
            lambda symbols: tuple(
                (
                    frozenset(
                        (
                            class_symbol,
                            *class_index.ancestor_symbols(class_symbol),
                        )
                    )
                    for class_symbol in symbols
                )
            )
        )
        .map(
            lambda ancestor_sets: set.intersection(
                *(set(symbols) for symbols in ancestor_sets)
            )
        )
        .filter(bool)
        .map(
            lambda common_symbols: max(
                common_symbols,
                key=lambda symbol: len(class_index.ancestor_symbols(symbol)),
            )
        )
        .unwrap_or_none()
    )


def _private_helper_call_nodes(
    function: _RuntimeFunctionNode, helper_name: str
) -> tuple[ast.Call, ...]:
    return tuple(
        (
            node
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == helper_name)
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == helper_name
                )
            )
        )
    )


def _private_helper_call_argument_map(
    call: ast.Call, parameter_names: tuple[str, ...]
) -> dict[str, ast.AST]:
    argument_map: dict[str, ast.AST] = {
        parameter_name: argument
        for parameter_name, argument in zip(parameter_names, call.args)
    }
    argument_map.update(
        {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
    )
    return argument_map


def _private_helper_residue_kind(argument: ast.AST) -> _PrivateHelperResidueKind:
    if isinstance(argument, ast.Constant):
        return _PrivateHelperResidueKind.CONSTANT
    if isinstance(argument, ast.Attribute):
        if isinstance(argument.value, ast.Name) and argument.value.id == "self":
            return _PrivateHelperResidueKind.SELF_ATTR
        return _PrivateHelperResidueKind.ATTRIBUTE
    if isinstance(argument, ast.Call):
        return _PrivateHelperResidueKind.CALL
    if isinstance(argument, ast.Name):
        return _PrivateHelperResidueKind.NAME
    return _PrivateHelperResidueKind.EXPRESSION


_PRIVATE_HELPER_VALUE_RESIDUE_TEMPLATE = PrivateHelperResidueNameTemplate(
    kind=_PrivateHelperResidueKind.VALUE,
    prefix="",
    suffix="_value",
    uppercase=False,
    parameter_name_is_authority=False,
)
_PRIVATE_HELPER_RESIDUE_NAME_TEMPLATES = (
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.CONSTANT,
        prefix="",
        suffix="",
        uppercase=True,
        parameter_name_is_authority=False,
    ),
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.CALL,
        prefix="_",
        suffix="_operation",
        uppercase=False,
        parameter_name_is_authority=False,
    ),
    PrivateHelperResidueNameTemplate(
        kind=_PrivateHelperResidueKind.SELF_ATTR,
        prefix="",
        suffix="",
        uppercase=False,
        parameter_name_is_authority=True,
    ),
)


def _private_helper_residue_name(
    function_name: str, parameter_name: str, kind: _PrivateHelperResidueKind
) -> str:
    template = next(
        (
            template
            for template in _PRIVATE_HELPER_RESIDUE_NAME_TEMPLATES
            if template.kind == kind
        ),
        _PRIVATE_HELPER_VALUE_RESIDUE_TEMPLATE,
    )
    if template.parameter_name_is_authority:
        return parameter_name
    base_name = f"{function_name.removeprefix('_')}_{parameter_name}"
    residue_name = f"{template.prefix}{base_name}{template.suffix}"
    if template.uppercase:
        return residue_name.upper()
    return residue_name


def _private_helper_residue_plan(
    *,
    function: _RuntimeFunctionNode,
    parameter_names: tuple[str, ...],
    caller_functions: tuple[_RuntimeFunctionNode, ...],
) -> PrivateHelperResiduePlan:
    call_argument_maps = tuple(
        (
            _private_helper_call_argument_map(call, parameter_names)
            for caller_function in caller_functions
            for call in _private_helper_call_nodes(caller_function, function.name)
        )
    )
    classvar_names: list[str] = []
    property_hook_names: list[str] = []
    behavior_hook_names: list[str] = []
    transported_parameter_names: list[str] = []
    callsite_axis_count = 0
    for parameter_name in parameter_names:
        arguments = tuple(
            argument_map[parameter_name]
            for argument_map in call_argument_maps
            if parameter_name in argument_map
        )
        if not arguments:
            continue
        argument_values = {ast.unparse(argument) for argument in arguments}
        argument_kinds = {
            _private_helper_residue_kind(argument) for argument in arguments
        }
        if argument_kinds == {_PrivateHelperResidueKind.NAME} and argument_values == {
            parameter_name
        }:
            transported_parameter_names.append(parameter_name)
            continue
        if (
            len(argument_values) == 1
            and next(iter(argument_kinds)) == _PrivateHelperResidueKind.NAME
        ):
            transported_parameter_names.append(parameter_name)
            continue
        callsite_axis_count += 1
        kind = next(iter(sorted(argument_kinds)))
        residue_name = _private_helper_residue_name(function.name, parameter_name, kind)
        _PrivateHelperResidueSink.for_kind(kind).append_residue(
            residue_name,
            classvar_names=classvar_names,
            property_hook_names=property_hook_names,
            behavior_hook_names=behavior_hook_names,
        )
    shared_statement_count = len(_trim_docstring_body(list(function.body)))
    leaf_residue_names = sorted_tuple(
        (*classvar_names, *property_hook_names, *behavior_hook_names)
    )
    normal_form = (
        f"HELPER_TEMPLATE({function.name})"
        f" -> input({','.join(sorted_tuple(transported_parameter_names))})"
        f" + residue({','.join(leaf_residue_names)})"
    )
    return PrivateHelperResiduePlan(
        classvar_names=tuple(dict.fromkeys(classvar_names)),
        property_hook_names=tuple(dict.fromkeys(property_hook_names)),
        behavior_hook_names=tuple(dict.fromkeys(behavior_hook_names)),
        transported_parameter_names=tuple(dict.fromkeys(transported_parameter_names)),
        callsite_axis_count=callsite_axis_count,
        shared_statement_count=shared_statement_count,
        residue_normal_form=normal_form,
    )


_PRIVATE_HELPER_AUTHORITY_VERB_TOKENS = frozenset(
    {
        "as",
        "build",
        "candidate",
        "candidates",
        "collect",
        "compute",
        "derive",
        "derived",
        "detect",
        "find",
        "for",
        "from",
        "get",
        "has",
        "is",
        "iter",
        "make",
        "to",
    }
)
_PRIVATE_HELPER_AUTHORITY_ROLES = (
    PrivateHelperAuthorityRole(
        authority_role_tokens=("candidate", "candidates", "collect", "collector"),
        suffix="CandidateCollector",
        drop_tokens=("candidate", "candidates", "collect", "collector"),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("metric", "metrics"),
        suffix="MetricsBuilder",
        drop_tokens=("metric", "metrics"),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("dispatch",),
        suffix="DispatchAuthority",
        drop_tokens=("dispatch",),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("registry", "registered"),
        suffix="RegistryAuthority",
        drop_tokens=("registry", "registered"),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("template", "templates"),
        suffix="TemplateAuthority",
        drop_tokens=("template", "templates"),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("shape", "shapes"),
        suffix="ShapeProjector",
        drop_tokens=("shape", "shapes"),
    ),
    PrivateHelperAuthorityRole(
        authority_role_tokens=("name", "names"),
        suffix="NameProjection",
        drop_tokens=("name", "names"),
    ),
)


def _private_helper_name_tokens(function_name: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in function_name.removeprefix("_").split("_")
        if token and token not in _PRIVATE_HELPER_AUTHORITY_VERB_TOKENS
    )


def _private_helper_pascal_name(tokens: tuple[str, ...], fallback: str) -> str:
    if not tokens:
        return fallback
    return "".join(token.capitalize() for token in tokens)


def _shared_private_helper_stem(
    functions: _RuntimeFunctionSequence,
) -> tuple[str, ...]:
    token_lists = tuple(
        (_private_helper_name_tokens(function.name) for function in functions)
    )
    if not token_lists:
        return ()
    shared: list[str] = []
    for token_column in zip(*token_lists):
        if len(set(token_column)) != 1:
            break
        shared.append(token_column[0])
    return tuple(shared)


_PRIVATE_HELPER_OWNER_RESIDUE_TOKENS = frozenset(
    (
        "api",
        "body",
        "candidate",
        "candidates",
        "expression",
        "for",
        "from",
        "function",
        "names",
        "public",
        "return",
        "returns",
        "strategy",
        "surface",
    )
)


def _dominant_private_helper_role_tokens(
    functions: _RuntimeFunctionSequence,
    stem_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    token_lists = tuple(
        (_private_helper_name_tokens(function.name) for function in functions)
    )
    threshold = max(2, (len(token_lists) + 1) // 2)
    stem_set = frozenset(stem_tokens)
    ordered_tokens = tuple(
        dict.fromkeys((token for tokens in token_lists for token in tokens))
    )
    return tuple(
        token
        for token in ordered_tokens
        if token not in stem_set
        and token not in _PRIVATE_HELPER_OWNER_RESIDUE_TOKENS
        and sum((token in tokens for tokens in token_lists)) >= threshold
    )


def _private_helper_authority_role(
    function_name: str,
) -> PrivateHelperAuthorityRole | None:
    all_tokens = frozenset(function_name.removeprefix("_").split("_"))
    return next(
        (
            role
            for role in _PRIVATE_HELPER_AUTHORITY_ROLES
            if all_tokens & frozenset(role.authority_role_tokens)
        ),
        None,
    )


def _private_helper_derived_authority_name(
    function_name: str,
    *,
    caller_owner_names: tuple[str, ...],
    fallback_suffix: str,
) -> str:
    shared_caller_name = HELPER_SUPPORT_PROJECTION_AUTHORITY.shared_family_name(
        caller_owner_names
    )
    if shared_caller_name is not None:
        return shared_caller_name
    role = _private_helper_authority_role(function_name)
    tokens = _private_helper_name_tokens(function_name)
    if role is not None:
        role_drop_tokens = frozenset(role.drop_tokens)
        subject_tokens = tuple(
            token for token in tokens if token not in role_drop_tokens
        )
        return f"{_private_helper_pascal_name(subject_tokens, 'Semantic')}{role.suffix}"
    return f"{_private_helper_pascal_name(tokens, 'Semantic')}{fallback_suffix}"


def _private_helper_placement_plan(
    modules: Sequence[ParsedModule],
    *,
    function: _RuntimeFunctionNode,
    function_name: str,
    parameter_names: tuple[str, ...],
    caller_symbols: tuple[str, ...],
    caller_functions: tuple[_RuntimeFunctionNode, ...],
    class_index: ClassFamilyIndex | None = None,
) -> PrivateHelperPlacementPlan:
    caller_owner_names = _private_helper_caller_owner_names(caller_symbols)
    residue_plan = _private_helper_residue_plan(
        function=function,
        parameter_names=parameter_names,
        caller_functions=caller_functions,
    )
    if len(caller_owner_names) == len(caller_symbols):
        class_index = class_index or build_class_family_index(list(modules))
        class_symbols = tuple(
            (
                class_symbol
                for owner_name in caller_owner_names
                if (
                    class_symbol := _private_helper_unique_class_symbol(
                        class_index, owner_name
                    )
                )
                is not None
            )
        )
        if len(class_symbols) == len(caller_owner_names):
            common_ancestor_symbol = _private_helper_deepest_common_ancestor_symbol(
                class_index, class_symbols
            )
            if common_ancestor_symbol is not None:
                ancestor = class_index.class_for(common_ancestor_symbol)
                owner_name = (
                    ancestor.simple_name
                    if ancestor is not None
                    else common_ancestor_symbol.rsplit(".", 1)[-1]
                )
                return PrivateHelperPlacementPlan(
                    placement_kind="existing_inheritance_root",
                    insertion_owner_name=owner_name,
                    insertion_detail=(
                        f"Insert `{function_name}` as a concrete/template method on `{owner_name}`; "
                        "thread transported inputs through the template method and keep callsite axes "
                        "as typed classvars/hooks on the leaves."
                    ),
                    residue_plan=residue_plan,
                    caller_owner_names=caller_owner_names,
                )
        if len(caller_owner_names) == 1:
            owner_name = caller_owner_names[0]
            return PrivateHelperPlacementPlan(
                placement_kind="owning_class_method",
                insertion_owner_name=owner_name,
                insertion_detail=(
                    f"Move `{function_name}` onto `{owner_name}` as an owned method; "
                    "promote it to the nearest ABC only if subclasses also consume it."
                ),
                residue_plan=residue_plan,
                caller_owner_names=caller_owner_names,
            )
        owner_name = _private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="FamilyAuthority",
        )
        return PrivateHelperPlacementPlan(
            placement_kind="new_family_mixin_or_abc",
            insertion_owner_name=owner_name,
            insertion_detail=(
                f"Create `{owner_name}` as the nominal family/mixin owner for `{function_name}`; "
                "attach the participating caller classes through inheritance or composition."
            ),
            residue_plan=residue_plan,
            caller_owner_names=caller_owner_names,
        )
    if caller_owner_names:
        owner_name = _private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="BoundaryPolicy",
        )
        return PrivateHelperPlacementPlan(
            placement_kind="boundary_strategy",
            insertion_owner_name=owner_name,
            insertion_detail=(
                f"Mixed class/module callers should route through `{owner_name}` as an explicit policy "
                f"or effect step that owns `{function_name}`."
            ),
            residue_plan=residue_plan,
            caller_owner_names=caller_owner_names,
        )
    return PrivateHelperPlacementPlan(
        placement_kind="module_nominal_authority",
        insertion_owner_name=_private_helper_derived_authority_name(
            function_name,
            caller_owner_names=caller_owner_names,
            fallback_suffix="Authority",
        ),
        insertion_detail=(
            f"Create a typed product/schema/strategy authority for `{function_name}` and inject it "
            "into the module-level callers."
        ),
        residue_plan=residue_plan,
        caller_owner_names=caller_owner_names,
    )


class ProbablyNominalPrivateHelperContractAuthority:
    def owns(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        derived_candidate_collector_contract_names: frozenset[str],
    ) -> bool:
        if _has_external_protocol_shape(function):
            return True
        if function.name in derived_candidate_collector_contract_names:
            return True
        return False


PROBABLY_NOMINAL_PRIVATE_HELPER_CONTRACT_AUTHORITY = (
    ProbablyNominalPrivateHelperContractAuthority()
)


def _private_helper_cluster_family(function_name: str) -> tuple[str, str]:
    return PUBLIC_BARE_SUPPORT_FUNCTION_FAMILY_AUTHORITY.family(
        function_name.lstrip("_")
    )


def _private_helper_cluster_key(function_name: str) -> tuple[str, str, str]:
    semantic_family, recommended_owner = _private_helper_cluster_family(function_name)
    tokens = _private_helper_name_tokens(function_name)
    role_token = tokens[0] if tokens else function_name.removeprefix("_")
    return semantic_family, recommended_owner, role_token


def _private_helper_callee_names(
    function: _RuntimeFunctionNode,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            call_name
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            for call_name in (_call_name(node.func),)
            if call_name is not None and not call_name.startswith("_")
        }
    )


def _private_helper_return_kind(node: ast.AST | None) -> str:
    if node is None:
        return "none"
    if isinstance(node, ast.Call):
        return _call_name(node.func) or "call"
    if isinstance(node, ast.Tuple):
        return "tuple_literal"
    if isinstance(node, ast.List):
        return "list_literal"
    if isinstance(node, ast.Dict):
        return "dict_literal"
    if isinstance(node, ast.Set):
        return "set_literal"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return type(node).__name__
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Attribute):
        return "attribute"
    return type(node).__name__


def _private_helper_return_kinds(
    functions: _RuntimeFunctionSequence,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            _private_helper_return_kind(returned.value)
            for function in functions
            for returned in _walk_function_body_nodes(function)
            if isinstance(returned, ast.Return)
        }
    )


def _private_helper_constructed_type_names(
    functions: _RuntimeFunctionSequence,
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            call_name
            for function in functions
            for node in _walk_function_body_nodes(function)
            if isinstance(node, ast.Call)
            for call_name in (_call_name(node.func),)
            if call_name is not None
            and call_name.endswith(
                (
                    "Candidate",
                    "Finding",
                    "Metrics",
                    "Observation",
                    "Plan",
                    "Profile",
                    "Shape",
                    "Spec",
                    "Witness",
                )
            )
        }
    )


def _private_helper_cluster_normal_form(
    *,
    semantic_tokens: tuple[str, ...],
    return_kinds: tuple[str, ...],
    constructed_type_names: tuple[str, ...],
    shared_call_names: tuple[str, ...],
) -> str:
    stem = "_".join(semantic_tokens)
    if "manifest" in semantic_tokens or {"TypeError", "isinstance"} <= set(
        shared_call_names
    ):
        return "typed_decoder"
    if "pattern" in semantic_tokens:
        return "catalog_schema"
    if "traversal" in semantic_tokens or "subclass" in semantic_tokens:
        return "traversal_profile"
    if "guard" in semantic_tokens or "validator" in semantic_tokens:
        return "candidate_pipeline"
    if "enum" in semantic_tokens and "dispatch" in semantic_tokens:
        return "extractor_family"
    if set(return_kinds) <= {"join", "str", "Constant", "FormattedValue"} or any(
        token in semantic_tokens for token in ("format", "markdown", "render")
    ):
        return "renderer"
    if constructed_type_names:
        return "candidate_builder"
    if stem.endswith("sorted_tuple") or "tuple" in return_kinds:
        return "collection_projection"
    if semantic_tokens and (
        set(return_kinds) <= {"tuple", "tuple_literal", "name", "attribute"}
    ):
        return "syntax_projection"
    return "semantic_authority"


def _private_helper_owner_suffix(normal_form: str) -> str:
    return {
        "candidate_builder": "Builder",
        "candidate_pipeline": "Pipeline",
        "catalog_schema": "Catalog",
        "collection_projection": "Projection",
        "extractor_family": "Extractor",
        "renderer": "Renderer",
        "semantic_authority": "Authority",
        "syntax_projection": "Projection",
        "traversal_profile": "Profile",
        "typed_decoder": "Decoder",
    }[normal_form]


def _private_helper_cluster_classification(
    functions: _RuntimeFunctionSequence,
    *,
    shared_call_names: tuple[str, ...],
) -> PrivateHelperClusterClassification:
    stem_tokens = _shared_private_helper_stem(functions)
    dominant_role_tokens = _dominant_private_helper_role_tokens(functions, stem_tokens)
    semantic_tokens = (*stem_tokens, *dominant_role_tokens)
    return_kinds = _private_helper_return_kinds(functions)
    constructed_type_names = _private_helper_constructed_type_names(functions)
    normal_form = _private_helper_cluster_normal_form(
        semantic_tokens=semantic_tokens,
        return_kinds=return_kinds,
        constructed_type_names=constructed_type_names,
        shared_call_names=shared_call_names,
    )
    owner_stem = _private_helper_pascal_name(semantic_tokens, "Semantic")
    suffix = _private_helper_owner_suffix(normal_form)
    owner_name = owner_stem if owner_stem.endswith(suffix) else f"{owner_stem}{suffix}"
    role_tokens = sorted_tuple(
        {
            token
            for function in functions
            for token in _private_helper_name_tokens(function.name)
            if token not in set(stem_tokens)
        }
    )
    return PrivateHelperClusterClassification(
        owner_name=owner_name,
        cluster_normal_form=normal_form,
        shared_stem="_".join(stem_tokens),
        classification_role_tokens=role_tokens,
        return_kinds=return_kinds,
        constructed_type_names=constructed_type_names,
    )


def _private_helper_cluster_certificate(
    cluster: PrivateHelperSemanticClusterCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=cluster.cluster_size,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("private_helper_owner",),
        ),
        semantic_axes=(
            *cluster.shared_parameter_names,
            *cluster.shared_call_names,
        ),
        residual_object_count=max(
            1,
            (len(cluster.shared_parameter_names) + len(cluster.shared_call_names)) // 2,
        ),
    )


def _private_helper_semantic_cluster_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
    private_helper_call_graph: PrivateHelperCallGraph | None = None,
) -> tuple[PrivateHelperSemanticClusterCandidate, ...]:
    modules = reference_modules or (module,)
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
    )
    private_helper_call_graph = (
        private_helper_call_graph or PrivateHelperCallGraph.from_modules(modules)
    )
    grouped: dict[
        tuple[str, str, str], list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = defaultdict(list)
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if PROBABLY_NOMINAL_PRIVATE_HELPER_CONTRACT_AUTHORITY.owns(
            function,
            derived_candidate_collector_contract_names=(
                derived_candidate_collector_contract_names
            ),
        ):
            continue
        minimum_cluster_line_count = max(
            3, config.min_unreferenced_private_function_lines // 2
        )
        if _function_line_count(function) < minimum_cluster_line_count:
            continue
        grouped[_private_helper_cluster_key(function.name)].append((qualname, function))

    candidates: list[PrivateHelperSemanticClusterCandidate] = []
    for (semantic_family, _, _), helpers in sorted(grouped.items()):
        if len(helpers) < 4:
            continue
        helper_names = sorted_tuple((function.name for _, function in helpers))
        parameter_sets = tuple(
            (
                set(FUNCTION_PARAMETER_NAME_PROJECTION.names(function))
                for _, function in helpers
            )
        )
        shared_parameter_names = sorted_tuple(set.intersection(*parameter_sets))
        call_name_sets = tuple(
            (set(_private_helper_callee_names(function)) for _, function in helpers)
        )
        shared_call_names = (
            sorted_tuple(set.intersection(*call_name_sets)) if call_name_sets else ()
        )
        consumer_symbols = sorted_tuple(
            {
                caller_symbol
                for qualname, function in helpers
                for caller_symbol in private_helper_call_graph.caller_symbols(
                    function_name=function.name, qualname=qualname
                )
            }
        )
        if not (shared_parameter_names or shared_call_names):
            continue
        functions = tuple((function for _, function in helpers))
        classification = _private_helper_cluster_classification(
            functions, shared_call_names=shared_call_names
        )
        line_numbers = tuple((function.lineno for _, function in helpers))
        candidate = PrivateHelperSemanticClusterCandidate(
            file_path=str(module.path),
            line=min(line_numbers),
            helper_names=helper_names,
            semantic_family=semantic_family,
            classification=classification,
            shared_parameter_names=shared_parameter_names,
            shared_call_names=shared_call_names,
            consumer_symbols=consumer_symbols,
            line_numbers=line_numbers,
            line_count=sum((_function_line_count(function) for _, function in helpers)),
            cluster_size=len(helpers),
        )
        if not _private_helper_cluster_certificate(candidate).pays_rent:
            continue
        candidates.append(candidate)
    return sorted_tuple(
        candidates,
        key=lambda item: (item.file_path, item.line, item.semantic_family),
    )


class PrivateHelperEscapeAuthority:
    def escapes_private_scope(self, caller_symbols: tuple[str, ...]) -> bool:
        if len(caller_symbols) >= 2:
            return True
        return self.has_single_public_module_caller(caller_symbols)

    def has_single_public_module_caller(
        self,
        caller_symbols: tuple[str, ...],
    ) -> bool:
        caller_symbol = single_item(caller_symbols)
        return bool(
            caller_symbol is not None
            and "." not in caller_symbol
            and not _is_private_symbol_name(caller_symbol)
        )


PRIVATE_HELPER_ESCAPE_AUTHORITY = PrivateHelperEscapeAuthority()


def _non_nominal_private_helper_candidates(
    module: ParsedModule,
    config: DetectorConfig,
    reference_modules: Sequence[ParsedModule] | None = None,
    derived_candidate_collector_contract_names: frozenset[str] | None = None,
    private_helper_call_graph: PrivateHelperCallGraph | None = None,
    class_index: ClassFamilyIndex | None = None,
) -> tuple[NonNominalPrivateHelperCandidate, ...]:
    modules = reference_modules or (module,)
    derived_candidate_collector_contract_names = (
        derived_candidate_collector_contract_names
        or DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names(modules)
    )
    private_helper_call_graph = (
        private_helper_call_graph or PrivateHelperCallGraph.from_modules(modules)
    )
    candidates: list[NonNominalPrivateHelperCandidate] = []
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." in qualname:
            continue
        if not _is_private_symbol_name(function.name):
            continue
        if PROBABLY_NOMINAL_PRIVATE_HELPER_CONTRACT_AUTHORITY.owns(
            function,
            derived_candidate_collector_contract_names=(
                derived_candidate_collector_contract_names
            ),
        ):
            continue
        caller_symbols = private_helper_call_graph.caller_symbols(
            function_name=function.name, qualname=qualname
        )
        if not PRIVATE_HELPER_ESCAPE_AUTHORITY.escapes_private_scope(caller_symbols):
            continue
        line_count = _function_line_count(function)
        if (
            not PRIVATE_HELPER_ESCAPE_AUTHORITY.has_single_public_module_caller(
                caller_symbols
            )
            and line_count < config.min_unreferenced_private_function_lines
        ):
            continue
        parameter_names = FUNCTION_PARAMETER_NAME_PROJECTION.names(function)
        caller_functions = private_helper_call_graph.caller_functions(
            function_name=function.name, qualname=qualname
        )
        call_site_count = sum(
            (isinstance(node, ast.Call) for node in _walk_function_body_nodes(function))
        )
        candidates.append(
            NonNominalPrivateHelperCandidate(
                file_path=str(module.path),
                line=function.lineno,
                qualname=qualname,
                function_name=function.name,
                parameter_names=parameter_names,
                caller_symbols=caller_symbols,
                placement_plan=_private_helper_placement_plan(
                    modules,
                    function=function,
                    function_name=function.name,
                    parameter_names=parameter_names,
                    caller_symbols=caller_symbols,
                    caller_functions=caller_functions,
                    class_index=class_index,
                ),
                line_count=line_count,
                call_site_count=call_site_count,
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (item.file_path, item.line, item.qualname))
    )


class UnreferencedPrivateFunctionDetector(
    ConfiguredModuleCollectorCandidateDetector[UnreferencedPrivateFunctionCandidate]
):
    cache_granularity = DetectorCacheGranularity.GLOBAL
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced private function should be deleted or made explicit",
        "A private function with no in-module references is not a witnessed local authority. If it is dead, delete it. If it is invoked dynamically or by an external framework, that contract should be made explicit through a registry, callback table, or public facade instead of relying on an invisible edge.",
        "explicit call-graph witness or deletion of dead private implementation surface",
        "private function is present in the implementation surface but absent from local call flow",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        private_reference_context = _private_reference_detector_context(tuple(modules))
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _unreferenced_private_function_candidates(
                module,
                config,
                reference_modules=modules,
                reference_index=private_reference_context.reference_index,
                derived_candidate_collector_contract_names=(
                    private_reference_context.derived_candidate_collector_contract_names
                ),
            )
        ]

    finding_renderer = CandidateFindingRenderer[UnreferencedPrivateFunctionCandidate](
        summary=lambda function_candidate: f"`{function_candidate.qualname}` spans {function_candidate.line_count} lines and has no in-module references.",
        evidence=lambda function_candidate: (function_candidate.evidence,),
        scaffold=lambda function_candidate: f"# Verify whether `{function_candidate.qualname}` is reached through reflection, subclassing, or an external framework.\n# If no such contract exists, delete it.\n# If it is dynamic API, declare that edge through a registry, callback table, or public facade.",
        codemod_patch=lambda function_candidate: f"# Remove `{function_candidate.qualname}` or replace the implicit dynamic edge with an explicit authority.",
        metrics=lambda function_candidate: OrchestrationMetrics(
            function_line_count=function_candidate.line_count,
            branch_site_count=0,
            call_site_count=function_candidate.call_site_count,
            parameter_count=0,
            callee_family_count=1,
        ),
    )


class NonNominalPrivateHelperDetector(
    ConfiguredModuleCollectorCandidateDetector[NonNominalPrivateHelperCandidate]
):
    cache_granularity = DetectorCacheGranularity.GLOBAL
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Escaped private helper should become nominal",
        "A module-level private helper that is called from multiple functions or exposed through a public module function has escaped local lexical residue. The helper should move into a nominal owner such as an ABC method, strategy object, descriptor, product/schema object, or registered effect step.",
        "explicit nominal owner for private helper behavior that escapes local private scope",
        "module-level private helper escapes private lexical ownership without a nominal owner",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        private_reference_context = _private_reference_detector_context(tuple(modules))
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _non_nominal_private_helper_candidates(
                module,
                config,
                reference_modules=modules,
                derived_candidate_collector_contract_names=(
                    private_reference_context.derived_candidate_collector_contract_names
                ),
                private_helper_call_graph=(
                    private_reference_context.private_helper_call_graph
                ),
                class_index=private_reference_context.class_index,
            )
        ]

    finding_renderer = CandidateFindingRenderer[NonNominalPrivateHelperCandidate](
        summary=lambda helper_candidate: (
            f"`{helper_candidate.qualname}` spans {helper_candidate.line_count} lines "
            f"and is called from {len(helper_candidate.caller_symbols)} surfaces "
            f"{helper_candidate.caller_symbols}; parameters {helper_candidate.parameter_names}. "
            f"Placement: {helper_candidate.placement_plan.placement_kind} "
            f"at `{helper_candidate.placement_plan.insertion_owner_name}`."
        ),
        evidence=lambda helper_candidate: (helper_candidate.evidence,),
        scaffold=lambda helper_candidate: (
            f"class {helper_candidate.placement_plan.insertion_owner_name}(ABC):\n"
            f"    def {helper_candidate.function_name.removeprefix('_')}(self, request): ...\n\n"
            f"# {helper_candidate.placement_plan.insertion_detail}\n"
            f"# Normal form: {helper_candidate.placement_plan.residue_plan.residue_normal_form}\n"
            f"# Classvar residue: {helper_candidate.placement_plan.residue_plan.classvar_names}\n"
            f"# Property hook residue: {helper_candidate.placement_plan.residue_plan.property_hook_names}\n"
            f"# Behavior hook residue: {helper_candidate.placement_plan.residue_plan.behavior_hook_names}"
        ),
        codemod_patch=lambda helper_candidate: (
            f"# Move `{helper_candidate.qualname}` into a nominal owner instead of keeping a reused private helper.\n"
            f"# Placement kind: {helper_candidate.placement_plan.placement_kind}\n"
            f"# Insertion owner: `{helper_candidate.placement_plan.insertion_owner_name}`\n"
            f"# {helper_candidate.placement_plan.insertion_detail}\n"
            f"# Normal form: {helper_candidate.placement_plan.residue_plan.residue_normal_form}\n"
            f"# Caller owners: {helper_candidate.placement_plan.caller_owner_names}\n"
            f"# Transported inputs: {helper_candidate.placement_plan.residue_plan.transported_parameter_names}\n"
            f"# Classvars: {helper_candidate.placement_plan.residue_plan.classvar_names}\n"
            f"# Property hooks: {helper_candidate.placement_plan.residue_plan.property_hook_names}\n"
            f"# Behavior hooks: {helper_candidate.placement_plan.residue_plan.behavior_hook_names}"
        ),
        metrics=lambda helper_candidate: OrchestrationMetrics(
            function_line_count=helper_candidate.line_count,
            branch_site_count=0,
            call_site_count=helper_candidate.call_site_count,
            parameter_count=len(helper_candidate.parameter_names),
            callee_family_count=len(helper_candidate.caller_symbols),
        ),
    )


class PrivateHelperSemanticClusterDetector(
    ConfiguredModuleCollectorCandidateDetector[PrivateHelperSemanticClusterCandidate]
):
    cache_granularity = DetectorCacheGranularity.GLOBAL
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Private helper cluster should have a semantic owner",
        "A family of private module helpers with shared parameters, shared callees, or shared consumers is not local residue; it is an unowned semantic algebra. Making the functions private only hides the missing owner. The normal form is a real ABC/template method, effect-step family, descriptor, product/schema algebra, or registered strategy family that owns the invariant once.",
        "nominal owner for clustered private helper semantics",
        "private helpers cluster by semantic family without an owning abstraction",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        private_reference_context = _private_reference_detector_context(tuple(modules))
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _private_helper_semantic_cluster_candidates(
                module,
                config,
                reference_modules=modules,
                derived_candidate_collector_contract_names=(
                    private_reference_context.derived_candidate_collector_contract_names
                ),
                private_helper_call_graph=(
                    private_reference_context.private_helper_call_graph
                ),
            )
        ]

    finding_renderer = CandidateFindingRenderer[PrivateHelperSemanticClusterCandidate](
        summary=lambda cluster: (
            f"{cluster.cluster_size} private helpers {cluster.helper_names} in "
            f"`{cluster.semantic_family}` share stem `{cluster.classification.shared_stem}` "
            f"and normal form `{cluster.classification.cluster_normal_form}`; inferred owner "
            f"`{cluster.classification.owner_name}`. Roles: {cluster.classification.classification_role_tokens}; "
            f"returns: {cluster.classification.return_kinds}; constructs: "
            f"{cluster.classification.constructed_type_names}; consumers: {cluster.consumer_symbols[:6]}. "
            f"Rent proof: {_private_helper_cluster_certificate(cluster).rent_proof_summary}."
        ),
        evidence=lambda cluster: cluster.evidence_locations,
        scaffold=lambda cluster: (
            f"class {cluster.classification.owner_name}(ABC):\n"
            f"    normal_form = {cluster.classification.cluster_normal_form!r}\n"
            f"    role_tokens = {cluster.classification.classification_role_tokens!r}\n"
            "    # Put the shared algorithm in concrete ABC methods.\n"
            "    # Keep only role-specific residue as classvars/properties/hooks.\n"
            "    ..."
        ),
        codemod_patch=lambda cluster: (
            f"# Do not fix {cluster.helper_names} by renaming or wrapping them.\n"
            f"# Factor `{cluster.classification.shared_stem}` into `{cluster.classification.owner_name}` "
            f"as `{cluster.classification.cluster_normal_form}`.\n"
            f"# Role/residue tokens: {cluster.classification.classification_role_tokens}\n"
            f"# Return kinds: {cluster.classification.return_kinds}\n"
            f"# Constructed types: {cluster.classification.constructed_type_names}\n"
            f"# Shared parameters: {cluster.shared_parameter_names}\n"
            f"# Shared callees: {cluster.shared_call_names}\n"
            f"# Rent proof: {_private_helper_cluster_certificate(cluster).rent_proof_summary}\n"
            "# Insert the owner only where it deletes duplicated helper mechanics; otherwise keep investigating the true invariant."
        ),
        compression_certificate=_private_helper_cluster_certificate,
        metrics=lambda cluster: OrchestrationMetrics(
            function_line_count=cluster.line_count,
            branch_site_count=0,
            call_site_count=len(cluster.consumer_symbols),
            parameter_count=len(cluster.shared_parameter_names),
            callee_family_count=max(1, len(cluster.shared_call_names)),
        ),
    )


class DanglingPrivateMethodDetector(
    ConfiguredModuleCollectorCandidateDetector[DanglingPrivateMethodCandidate]
):
    cache_granularity = DetectorCacheGranularity.GLOBAL
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Dangling private method should be deleted or made nominal",
        "A private method that has no visible callsite, override contract, decorator, or framework hook is not a nominal interface. Inside a class it looks owned, but without a witnessed edge it is dead residue or an implicit protocol that should be made explicit through an ABC hook, public facade, strategy object, or registry-backed dispatch surface.",
        "explicit nominal hook or deletion of unreferenced private method residue",
        "private class method has no repository-visible reference outside its own body",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        private_reference_context = _private_reference_detector_context(tuple(modules))
        return [
            self._finding_for_candidate(candidate)
            for module in modules
            for candidate in _dangling_private_method_candidates(
                module,
                config,
                reference_modules=modules,
                reference_index=private_reference_context.reference_index,
            )
        ]

    finding_renderer = CandidateFindingRenderer[DanglingPrivateMethodCandidate](
        summary=lambda method_candidate: (
            f"`{method_candidate.qualname}` spans {method_candidate.line_count} lines "
            "and has no repository-visible method reference."
        ),
        evidence=lambda method_candidate: (method_candidate.evidence,),
        scaffold=lambda method_candidate: (
            f"# Delete `{method_candidate.qualname}` if it is dead.\n"
            "# If subclasses or framework code call it, declare an explicit ABC hook, public facade,\n"
            "# strategy object, or registry dispatch surface that owns the protocol."
        ),
        codemod_patch=lambda method_candidate: (
            f"# Make `{method_candidate.owner_name}.{method_candidate.method_name}` nominal or remove it.\n"
            "# Private method names should not be the only witness for a dynamic protocol."
        ),
        metrics=lambda method_candidate: OrchestrationMetrics(
            function_line_count=method_candidate.line_count,
            branch_site_count=0,
            call_site_count=method_candidate.call_site_count,
            parameter_count=0,
            callee_family_count=1,
        ),
    )


# fmt: off
materialize_product_record(product_record_spec('SiblingSmallMethodTemplateCandidate', 'owner_name: str; statement_count: int; parameter_count: int; witness_name: ClassVar[AliasProperty[str]]', 'MethodEvidenceLocationsCandidate', defaults={'witness_name': AliasProperty('owner_name')}))
# fmt: on


_NORMALIZED_TEMPLATE_STABLE_NAMES = frozenset(
    {
        "False",
        "None",
        "True",
        "cls",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "open",
        "print",
        "range",
        "re",
        "self",
        "set",
        "shutil",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)


def _trimmed_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    return tuple(_trim_docstring_body(function.body))


def _normalized_small_method_template(
    body: tuple[ast.stmt, ...],
) -> tuple[str, ...]:
    class Normalizer(ast.NodeTransformer):
        def visit_arg(self, node: ast.arg) -> ast.arg:
            return ast.copy_location(ast.arg(arg="ARG", annotation=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in _NORMALIZED_TEMPLATE_STABLE_NAMES:
                return node
            return ast.copy_location(ast.Name(id="NAME", ctx=node.ctx), node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="STR"), node)
            if isinstance(node.value, (int, float, complex, bool, type(None))):
                return ast.copy_location(ast.Constant(value="CONST"), node)
            return node

    normalizer = Normalizer()
    return tuple(
        (
            ast.dump(
                ast.fix_missing_locations(
                    cast(ast.stmt, normalizer.visit(copy.deepcopy(statement)))
                ),
                include_attributes=False,
            )
            for statement in body
        )
    )


def _normalized_role_residue_small_method_template(
    body: tuple[ast.stmt, ...],
) -> tuple[str, ...]:
    """Normalize private sibling-helper shape while ignoring role-specific attrs."""

    class Normalizer(ast.NodeTransformer):
        def visit_arg(self, node: ast.arg) -> ast.arg:
            return ast.copy_location(ast.arg(arg="ARG", annotation=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in _NORMALIZED_TEMPLATE_STABLE_NAMES:
                return node
            return ast.copy_location(ast.Name(id="NAME", ctx=node.ctx), node)

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            value = cast(ast.expr, self.visit(node.value))
            return ast.copy_location(
                ast.Attribute(value=value, attr="ATTR", ctx=node.ctx),
                node,
            )

        def visit_If(self, node: ast.If) -> ast.AST:
            return ast.copy_location(
                ast.If(
                    test=ast.Constant(value="ROLE_PRESENCE_TEST"),
                    body=[cast(ast.stmt, self.visit(item)) for item in node.body],
                    orelse=[cast(ast.stmt, self.visit(item)) for item in node.orelse],
                ),
                node,
            )

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="STR"), node)
            if isinstance(node.value, (int, float, complex, bool, type(None))):
                return ast.copy_location(ast.Constant(value="CONST"), node)
            return node

    normalizer = Normalizer()
    return tuple(
        (
            ast.dump(
                ast.fix_missing_locations(
                    cast(ast.stmt, normalizer.visit(copy.deepcopy(statement)))
                ),
                include_attributes=False,
            )
            for statement in body
        )
    )


def _normalized_cross_class_method_template(
    body: tuple[ast.stmt, ...],
) -> tuple[str, ...]:
    class Normalizer(ast.NodeTransformer):
        def visit_arg(self, node: ast.arg) -> ast.arg:
            return ast.copy_location(ast.arg(arg="ARG", annotation=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in _NORMALIZED_TEMPLATE_STABLE_NAMES:
                return node
            return ast.copy_location(ast.Name(id="NAME", ctx=node.ctx), node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value="STR"), node)
            if isinstance(node.value, (int, float, complex, bool, type(None))):
                return ast.copy_location(ast.Constant(value="CONST"), node)
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
            return ast.copy_location(ast.Constant(value="LAMBDA_RESIDUE"), node)

    normalizer = Normalizer()
    return tuple(
        (
            ast.dump(
                ast.fix_missing_locations(
                    cast(ast.stmt, normalizer.visit(copy.deepcopy(statement)))
                ),
                include_attributes=False,
            )
            for statement in body
        )
    )


def _method_name_family_tokens(method_names: tuple[str, ...]) -> tuple[str, ...]:
    token_sets = [
        set(CLASS_NAME_ALGEBRA.ordered_tokens(method_name.strip("_")))
        for method_name in method_names
    ]
    if not token_sets:
        return ()
    shared = set.intersection(*token_sets)
    return sorted_tuple((token for token in shared if len(token) >= 3))


def _sibling_small_method_template_candidates(
    module: ParsedModule,
) -> tuple[SiblingSmallMethodTemplateCandidate, ...]:
    grouped: dict[
        tuple[str, int, tuple[str, ...]],
        list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    ] = defaultdict(list)
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        if "." not in qualname or not _is_private_symbol_name(function.name):
            continue
        if _has_external_protocol_shape(
            function
        ) and not _has_only_nominal_method_decorators(function):
            continue
        body = _trimmed_function_body(function)
        if not 2 <= len(body) <= 6:
            continue
        owner_name = qualname.rsplit(".", 1)[0]
        parameter_count = len(function.args.args) + len(function.args.kwonlyargs)
        key = (
            owner_name,
            parameter_count,
            _normalized_role_residue_small_method_template(body),
        )
        grouped[key].append((qualname, function))

    candidates: list[SiblingSmallMethodTemplateCandidate] = []
    for (owner_name, parameter_count, template), functions in grouped.items():
        if len(functions) < 2:
            continue
        ordered = sorted_tuple(functions, key=lambda item: (item[1].lineno, item[0]))
        method_names = tuple(function.name for _, function in ordered)
        if not _method_name_family_tokens(method_names):
            continue
        line_numbers = tuple(function.lineno for _, function in ordered)
        candidates.append(
            SiblingSmallMethodTemplateCandidate(
                file_path=str(module.path),
                line=line_numbers[0],
                owner_name=owner_name,
                method_names=method_names,
                line_numbers=line_numbers,
                statement_count=len(template),
                parameter_count=parameter_count,
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.owner_name)
    )


class SiblingSmallMethodTemplateDetector(
    ModuleCollectorCandidateDetector[SiblingSmallMethodTemplateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Sibling small method templates should collapse to one parameterized helper",
        "One owner has private sibling methods with the same small execution template and shared name family. Only role names or literal residue vary, so the implementation should name one local authority and pass the role-specific values as data.",
        "one local helper/table for repeated small method templates",
        "same owner repeats a small private method body template across sibling roles",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, template_candidate: SiblingSmallMethodTemplateCandidate
    ) -> RefactorFinding:
        method_summary = ", ".join(template_candidate.method_names)
        return self.build_finding(
            (
                f"`{template_candidate.owner_name}` repeats the same {template_candidate.statement_count}-statement "
                f"private method template across {method_summary}."
            ),
            template_candidate.evidence_locations,
            scaffold=(
                "# Replace the sibling methods with one parameterized local helper that accepts the varying role/literal values.\n# Keep separate methods only when each owns a distinct invariant or external contract."
            ),
            codemod_patch=(
                f"# Collapse sibling template methods {template_candidate.method_names} into one parameterized local helper."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(template_candidate.method_names),
                statement_count=template_candidate.statement_count,
                class_count=1,
                method_symbols=template_candidate.method_names,
            ),
        )


@dataclass(frozen=True)
class CrossClassSmallMethodTemplateCandidate(MethodEvidenceLocationsCandidate):
    owner_name: str
    owner_names: tuple[str, ...]
    method_name: str
    statement_count: int
    parameter_count: int

    @property
    def witness_name(self) -> str:
        return self.method_name


_CLASSLEVEL_METHOD_DECORATORS = frozenset({"classmethod", "staticmethod"})


def _decorator_simple_name(decorator: ast.AST) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        return _decorator_simple_name(decorator.func)
    return None


def _has_only_nominal_method_decorators(function: _RuntimeFunctionNode) -> bool:
    decorator_names = tuple(
        _decorator_simple_name(decorator) for decorator in function.decorator_list
    )
    return all(
        decorator_name in _CLASSLEVEL_METHOD_DECORATORS
        for decorator_name in decorator_names
    )


def _public_method_template_owner(qualname: str, function_name: str) -> str | None:
    if "." not in qualname:
        return None
    if _is_private_symbol_name(function_name) or function_name.startswith("__"):
        return None
    return qualname.rsplit(".", 1)[0]


def _cross_class_small_method_template_candidates(
    module: ParsedModule,
) -> tuple[CrossClassSmallMethodTemplateCandidate, ...]:
    grouped: dict[
        tuple[str, int, tuple[str, ...]],
        list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]],
    ] = defaultdict(list)
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        owner_name = _public_method_template_owner(qualname, function.name)
        if owner_name is None:
            continue
        if not _has_only_nominal_method_decorators(function):
            continue
        body = _trimmed_function_body(function)
        if not 1 <= len(body) <= 8:
            continue
        parameter_count = len(function.args.args) + len(function.args.kwonlyargs)
        key = (
            function.name,
            parameter_count,
            _normalized_cross_class_method_template(body),
        )
        grouped[key].append((owner_name, qualname, function))

    candidates: list[CrossClassSmallMethodTemplateCandidate] = []
    for (method_name, parameter_count, template), functions in grouped.items():
        owner_names = sorted_tuple({owner_name for owner_name, _, _ in functions})
        if len(owner_names) < 2:
            continue
        ordered = sorted_tuple(functions, key=lambda item: (item[2].lineno, item[1]))
        method_names = tuple(qualname for _, qualname, _ in ordered)
        line_numbers = tuple(function.lineno for _, _, function in ordered)
        candidates.append(
            CrossClassSmallMethodTemplateCandidate(
                file_path=str(module.path),
                line=line_numbers[0],
                owner_name=", ".join(owner_names),
                method_names=method_names,
                line_numbers=line_numbers,
                owner_names=owner_names,
                method_name=method_name,
                statement_count=len(template),
                parameter_count=parameter_count,
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.method_name)
    )


class CrossClassSmallMethodTemplateDetector(
    ModuleCollectorCandidateDetector[CrossClassSmallMethodTemplateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Public authority methods repeat across class leaves",
        "Public, classmethod, or staticmethod bodies repeated across distinct class owners are a hidden behavior family. The shared algorithm should live in one ABC/template-method base while leaf classes declare only the nominal residue that really varies.",
        "one inherited authority algorithm with explicit leaf residue",
        "same public method template repeats across class owners",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _cross_class_small_method_template_candidates(module)

    def _finding_for_candidate(
        self, template_candidate: CrossClassSmallMethodTemplateCandidate
    ) -> RefactorFinding:
        owner_summary = ", ".join(template_candidate.owner_names)
        return self.build_finding(
            (
                f"Classes {owner_summary} repeat `{template_candidate.method_name}` "
                f"with the same {template_candidate.statement_count}-statement public method template."
            ),
            template_candidate.evidence_locations,
            scaffold=(
                f"class {template_candidate.method_name.title().replace('_', '')}TemplateBase(ABC):\n"
                f"    def {template_candidate.method_name}(self, *args, **kwargs):\n"
                "        return self._shared_template(*args, **kwargs)\n\n"
                "    @abstractmethod\n"
                "    def _shared_template(self, *args, **kwargs): ..."
            ),
            codemod_patch=(
                f"# Move repeated `{template_candidate.method_name}` mechanics from "
                f"{template_candidate.owner_names} into one inherited ABC/template-method base.\n"
                "# Keep only class-level field names, enum families, or other irreducible residue in leaves."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(template_candidate.method_names),
                statement_count=template_candidate.statement_count,
                class_count=len(template_candidate.owner_names),
                method_symbols=template_candidate.method_names,
            ),
        )


_IDENTIFIER_STOP_TOKENS = frozenset(
    {
        "abc",
        "api",
        "base",
        "class",
        "cls",
        "data",
        "get",
        "impl",
        "item",
        "make",
        "new",
        "object",
        "old",
        "return",
        "self",
        "set",
        "tmp",
        "value",
        "values",
    }
)


@dataclass(frozen=True)
class _NominalAuthorityBypassSeed:
    module: ParsedModule
    scatter: IsinstanceFamilyScatterCandidate
    indexed_classes: tuple[IndexedClass, ...]
    shared_base: IndexedClass


@dataclass(frozen=True)
class _NominalAuthorityBypassCandidate:
    seed: _NominalAuthorityBypassSeed
    repeated_templates: tuple[CrossClassSmallMethodTemplateCandidate, ...]
    wrapper_chains: tuple[WrapperChainCandidate, ...]
    composition_signals: tuple[CancelableCompositionSignal, ...]


@dataclass(frozen=True)
class _VariantMethodSurface:
    file_path: str
    owner_class_name: str
    owner_line: int
    owner_is_abstract: bool
    owner_base_names: tuple[str, ...]
    qualname: str
    method_name: str
    line: int
    statement_count: int
    method_tokens: tuple[str, ...]
    product_parameter_names: tuple[str, ...]
    forwarded_field_names: tuple[str, ...]
    construction_shape: str

    @property
    def evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.qualname)


@dataclass(frozen=True)
class _VariantMethodFamilySeed:
    methods: tuple[_VariantMethodSurface, ...]
    shared_product_parameter_names: tuple[str, ...]
    shared_field_names: tuple[str, ...]
    anchor_tokens: tuple[str, ...]
    variant_tokens: tuple[str, ...]

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(method.qualname for method in self.methods)

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        exemplar = self.methods[0]
        evidence: list[SourceLocation] = [
            SourceLocation(
                exemplar.file_path,
                exemplar.owner_line,
                exemplar.owner_class_name,
            ),
            *(method.evidence for method in self.methods[:5]),
        ]
        return tuple(evidence[:8])


@dataclass(frozen=True)
class _VariantMethodFamilyCandidate:
    seed: _VariantMethodFamilySeed
    wrapper_chains: tuple[WrapperChainCandidate, ...]
    composition_signals: tuple[CancelableCompositionSignal, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = [
            *self.seed.evidence,
            *(
                SourceLocation(signal.file_path, signal.line, signal.qualname)
                for signal in self.composition_signals[:2]
            ),
            *(
                wrapper.evidence
                for chain in self.wrapper_chains[:2]
                for wrapper in chain.wrappers[:1]
            ),
        ]
        return tuple(evidence[:8])


class SemanticTokenAuthority:
    """Normalize candidate relation strings into comparable semantic tokens."""

    @staticmethod
    def identifier_tokens(value: str) -> tuple[str, ...]:
        tokens: list[str] = []
        for chunk in re.split(r"[^0-9A-Za-z]+", value):
            if not chunk:
                continue
            matches = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[0-9]|\b)|[A-Z]?[a-z]+|[0-9]+",
                chunk,
            )
            tokens.extend(match.lower() for match in matches if match)
        return tuple(tokens)

    @staticmethod
    def tokens(*values: str) -> frozenset[str]:
        tokens = {
            token
            for value in values
            for token in SemanticTokenAuthority.identifier_tokens(value)
            if len(token) >= 3 and token not in _IDENTIFIER_STOP_TOKENS
        }
        return frozenset(tokens)


def _indexed_classes_for_type_names(
    module: ParsedModule,
    class_index: ClassFamilyIndex,
    type_names: tuple[str, ...],
) -> tuple[IndexedClass, ...]:
    indexed_classes: list[IndexedClass] = []
    seen_symbols: set[str] = set()
    for type_name in type_names:
        simple_name = type_name.rsplit(".", 1)[-1]
        indexed_class = SYNTAX_PROJECTION_AUTHORITY.indexed_class_for_simple_name(
            module, class_index, simple_name
        )
        if indexed_class is None or indexed_class.symbol in seen_symbols:
            continue
        seen_symbols.add(indexed_class.symbol)
        indexed_classes.append(indexed_class)
    return tuple(indexed_classes)


def _shared_nominal_base_classes(
    class_index: ClassFamilyIndex, indexed_classes: tuple[IndexedClass, ...]
) -> tuple[IndexedClass, ...]:
    if len(indexed_classes) < 2:
        return ()
    checked_symbols = {indexed_class.symbol for indexed_class in indexed_classes}
    common_symbols = set(
        _indexed_ancestor_symbols(class_index, indexed_classes[0].symbol)
    )
    for indexed_class in indexed_classes[1:]:
        common_symbols &= set(
            _indexed_ancestor_symbols(class_index, indexed_class.symbol)
        )
    base_classes = tuple(
        indexed_class
        for symbol in sorted(common_symbols)
        if symbol not in checked_symbols
        if (indexed_class := class_index.class_for(symbol)) is not None
        if not indexed_class.simple_name.startswith("_")
    )
    abstract_bases = tuple(
        indexed_class
        for indexed_class in base_classes
        if CLASS_NODE_AUTHORITY.is_abstract(indexed_class.node)
    )
    return abstract_bases or base_classes


@dataclass(frozen=True)
class CancelableCompositionSignalQuery:
    """Cached source-index query for cancelable product-composition signals."""

    modules: tuple[ParsedModule, ...]

    @lru_cache(maxsize=None)
    def signals(self) -> tuple[CancelableCompositionSignal, ...]:
        if not self.modules:
            return ()
        source_index = build_source_index(list(self.modules), ())
        return detect_cancelable_composition_signals(
            source_index,
            {str(module.path): module.source for module in self.modules},
        )


class RelatedCompositionSignalsAuthority:
    """Select composition signals relevant to one candidate token surface."""

    @staticmethod
    def related(
        signals: tuple[CancelableCompositionSignal, ...],
        *,
        file_path: str,
        token_sources: tuple[str, ...],
        field_names: tuple[str, ...] = (),
    ) -> tuple[CancelableCompositionSignal, ...]:
        source_tokens = SemanticTokenAuthority.tokens(*token_sources)
        field_name_set = set(field_names)
        related = []
        for signal in signals:
            if signal.file_path != file_path:
                continue
            signal_tokens = SemanticTokenAuthority.tokens(
                signal.qualname,
                signal.carrier_name,
                signal.source_name,
                *signal.field_names,
            )
            if (source_tokens & signal_tokens) or len(
                field_name_set & set(signal.field_names)
            ) >= 2:
                related.append(signal)
        return sorted_tuple(
            related,
            key=lambda item: (
                -item.load_bearing_score,
                item.file_path,
                item.line,
                item.qualname,
            ),
        )


class RelatedWrapperChainsAuthority:
    """Select wrapper chains relevant to one candidate token surface."""

    @staticmethod
    def related(
        chains: tuple[WrapperChainCandidate, ...],
        *,
        file_path: str,
        token_sources: tuple[str, ...],
    ) -> tuple[WrapperChainCandidate, ...]:
        source_tokens = SemanticTokenAuthority.tokens(*token_sources)
        related = []
        for chain in chains:
            if chain.file_path != file_path:
                continue
            chain_tokens = SemanticTokenAuthority.tokens(
                chain.leaf_delegate_symbol,
                *(wrapper.qualname for wrapper in chain.wrappers),
                *(
                    attr
                    for wrapper in chain.wrappers
                    for attr in wrapper.projected_attributes
                ),
            )
            if source_tokens & chain_tokens:
                related.append(chain)
        return sorted_tuple(
            related,
            key=lambda item: (
                -len(item.wrappers),
                item.file_path,
                item.wrappers[0].lineno,
            ),
        )


def _templates_related_to_checked_classes(
    templates: tuple[CrossClassSmallMethodTemplateCandidate, ...],
    checked_classes: tuple[IndexedClass, ...],
) -> tuple[CrossClassSmallMethodTemplateCandidate, ...]:
    checked_names = {indexed_class.simple_name for indexed_class in checked_classes}
    related = []
    for template in templates:
        owner_names = {owner.rsplit(".", 1)[-1] for owner in template.owner_names}
        if len(checked_names & owner_names) >= 2:
            related.append(template)
    return sorted_tuple(
        related,
        key=lambda item: (item.file_path, item.line, item.method_name),
    )


def _nominal_authority_bypass_candidates(
    modules: list[ParsedModule],
) -> tuple[_NominalAuthorityBypassCandidate, ...]:
    class_index = build_class_family_index(modules)
    seeds: list[_NominalAuthorityBypassSeed] = []
    for module in modules:
        for scatter in _isinstance_family_scatter_candidates(module):
            checked_classes = _indexed_classes_for_type_names(
                module, class_index, scatter.type_names
            )
            shared_bases = _shared_nominal_base_classes(class_index, checked_classes)
            if not shared_bases:
                continue
            seeds.append(
                _NominalAuthorityBypassSeed(
                    module=module,
                    scatter=scatter,
                    indexed_classes=checked_classes,
                    shared_base=shared_bases[0],
                )
            )
    if not seeds:
        return ()

    templates = tuple(
        template
        for module in modules
        for template in _cross_class_small_method_template_candidates(module)
    )
    wrapper_chains = tuple(
        chain for module in modules for chain in _wrapper_chain_candidates(module)
    )
    composition_signals = CancelableCompositionSignalQuery(tuple(modules)).signals()

    candidates: list[_NominalAuthorityBypassCandidate] = []
    for seed in seeds:
        base_display_name = CLASS_INDEX_PROJECTION.display_name(
            seed.shared_base, class_index
        )
        related_templates = _templates_related_to_checked_classes(
            templates, seed.indexed_classes
        )
        token_sources = (
            seed.scatter.qualname,
            seed.scatter.subject_expression,
            base_display_name,
            *(indexed_class.simple_name for indexed_class in seed.indexed_classes),
            *(template.method_name for template in related_templates),
        )
        candidates.append(
            _NominalAuthorityBypassCandidate(
                seed=seed,
                repeated_templates=related_templates,
                wrapper_chains=RelatedWrapperChainsAuthority.related(
                    wrapper_chains,
                    file_path=seed.scatter.file_path,
                    token_sources=token_sources,
                ),
                composition_signals=RelatedCompositionSignalsAuthority.related(
                    composition_signals,
                    file_path=seed.scatter.file_path,
                    token_sources=token_sources,
                ),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.seed.scatter.file_path,
            item.seed.scatter.line,
            item.seed.scatter.qualname,
            item.seed.shared_base.symbol,
        ),
    )


class ABCPolymorphismBypassedByConcreteDispatchDetector(IssueDetector):
    detector_id = "abc_polymorphism_bypassed_by_concrete_dispatch"
    detector_priority = -20
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "ABC polymorphism bypassed by helper-level concrete dispatch",
        "A helper that branches on concrete carrier classes while those classes already share a nominal base has moved behavior out of the family authority. The root cause is not the local isinstance syntax: it is bypassed polymorphism, often amplified by duplicate leaf methods and cancelable product pack/unpack/forward layers.",
        "shared nominal carrier authority exposes one polymorphic/template-method hook",
        "helper-level concrete runtime dispatch crosses a shared ABC/base family",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS
        + _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        return [
            self._finding_for_candidate(candidate)
            for candidate in _nominal_authority_bypass_candidates(modules)
        ]

    def _finding_for_candidate(
        self, candidate: _NominalAuthorityBypassCandidate
    ) -> RefactorFinding:
        scatter = candidate.seed.scatter
        base_name = candidate.seed.shared_base.simple_name
        checked_type_names = tuple(
            indexed_class.simple_name for indexed_class in candidate.seed.indexed_classes
        )
        branch_summary = ", ".join(scatter.test_expressions[:4])
        template_summary = ", ".join(
            sorted_tuple(
                {template.method_name for template in candidate.repeated_templates}
            )
        )
        composition_summary = ", ".join(
            signal.qualname for signal in candidate.composition_signals[:3]
        )
        wrapper_summary = ", ".join(
            " -> ".join(wrapper.qualname for wrapper in chain.wrappers)
            for chain in candidate.wrapper_chains[:2]
        )
        extra_context = []
        if template_summary:
            extra_context.append(f"duplicate leaf template(s): {template_summary}")
        if composition_summary:
            extra_context.append(
                f"cancelable product composition(s): {composition_summary}"
            )
        if wrapper_summary:
            extra_context.append(f"wrapper chain(s): {wrapper_summary}")
        context_suffix = (
            f" It also intersects {'; '.join(extra_context)}." if extra_context else ""
        )
        evidence = tuple(
            [
                *(
                    SourceLocation(
                        scatter.file_path,
                        line,
                        f"{scatter.qualname}:{scatter.subject_expression}",
                    )
                    for line in scatter.line_numbers[:4]
                ),
                SourceLocation(
                    candidate.seed.shared_base.file_path,
                    candidate.seed.shared_base.line,
                    base_name,
                ),
                *(
                    location
                    for template in candidate.repeated_templates[:2]
                    for location in template.evidence_locations[:2]
                ),
                *(
                    SourceLocation(signal.file_path, signal.line, signal.qualname)
                    for signal in candidate.composition_signals[:2]
                ),
                *(
                    wrapper.evidence
                    for chain in candidate.wrapper_chains[:2]
                    for wrapper in chain.wrappers[:1]
                ),
            ][:8]
        )
        method_symbols = tuple(
            (
                *(
                    f"{scatter.qualname}:{test}"
                    for test in scatter.test_expressions[:4]
                ),
                *(
                    method_name
                    for template in candidate.repeated_templates
                    for method_name in template.method_names
                ),
                *(signal.qualname for signal in candidate.composition_signals),
                *(
                    wrapper.qualname
                    for chain in candidate.wrapper_chains
                    for wrapper in chain.wrappers
                ),
            )
        )
        return self.build_finding(
            (
                "ABC polymorphism bypassed by helper-level concrete dispatch: "
                f"`{scatter.qualname}` checks `{scatter.subject_expression}` with "
                f"{branch_summary} even though {', '.join(checked_type_names)} share "
                f"`{base_name}`.{context_suffix}"
            ),
            evidence,
            scaffold=(
                f"class {base_name}(...):\n"
                "    def construct_from(self, request):\n"
                "        return self._construct_from_nominal_context(request)\n\n"
                "    def _construct_from_nominal_context(self, request): ...\n\n"
                f"# Replace helper branches in `{scatter.qualname}` with one call through "
                f"`{base_name}`. Put concrete construction residue on the leaf classes or "
                "on a nominal request/context object; do not unpack a product carrier through "
                "external helper dispatch."
            ),
            codemod_patch=(
                f"# Collapse `{scatter.qualname}` concrete isinstance branches into a "
                f"`{base_name}` polymorphic/template-method hook.\n"
                "# Move duplicate leaf method templates and cancelable pack/unpack forwarding "
                "into the same nominal authority before deleting the helper-level branch table."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=max(
                    2,
                    scatter.site_count
                    + len(candidate.repeated_templates)
                    + len(candidate.composition_signals)
                    + len(candidate.wrapper_chains),
                ),
                statement_count=max(
                    2,
                    1
                    + sum(
                        template.statement_count
                        for template in candidate.repeated_templates
                    ),
                ),
                class_count=max(2, len(candidate.seed.indexed_classes)),
                method_symbols=method_symbols,
            ),
        )


def _method_parameter_names(function: _RuntimeFunctionNode) -> tuple[str, ...]:
    names = [arg.arg for arg in function.args.posonlyargs]
    names.extend(arg.arg for arg in function.args.args)
    names.extend(arg.arg for arg in function.args.kwonlyargs)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    return tuple(names)


def _product_parameter_fields(
    function: _RuntimeFunctionNode,
) -> dict[str, tuple[str, ...]]:
    parameter_names = set(_method_parameter_names(function))
    fields_by_parameter: dict[str, set[str]] = defaultdict(set)
    for node in _walk_nodes(function):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in parameter_names
        ):
            continue
        fields_by_parameter[node.value.id].add(node.attr)
    return {
        parameter_name: sorted_tuple(fields)
        for parameter_name, fields in fields_by_parameter.items()
        if len(fields) >= 2
    }


def _single_return_call(function: _RuntimeFunctionNode) -> ast.Call | None:
    return_calls = tuple(
        node.value
        for node in _walk_nodes(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    )
    if len(return_calls) != 1:
        return None
    return return_calls[0]


def _construction_shape(function: _RuntimeFunctionNode) -> str | None:
    call = _single_return_call(function)
    if call is None:
        return None
    callee_name = _call_name(call.func)
    if callee_name is None:
        return None
    keyword_names = sorted_tuple(
        keyword.arg for keyword in call.keywords if keyword.arg is not None
    )
    if len(keyword_names) != len(call.keywords):
        return None
    return f"{callee_name}|args={len(call.args)}|kwargs={','.join(keyword_names)}"


def _variant_method_surface(
    module: ParsedModule,
    class_node: ast.ClassDef,
    method: _RuntimeFunctionNode,
) -> _VariantMethodSurface | None:
    if _is_private_symbol_name(method.name) or method.name.startswith("__"):
        return None
    if not _has_only_nominal_method_decorators(method):
        return None
    product_fields = _product_parameter_fields(method)
    if not product_fields:
        return None
    construction_shape = _construction_shape(method)
    if construction_shape is None:
        return None
    method_tokens = SemanticTokenAuthority.identifier_tokens(method.name)
    if len(method_tokens) < 2:
        return None
    forwarded_field_names = sorted_tuple(
        {field for fields in product_fields.values() for field in fields}
    )
    statement_count = len(_trimmed_function_body(method))
    if statement_count > 8:
        return None
    return _VariantMethodSurface(
        file_path=str(module.path),
        owner_class_name=class_node.name,
        owner_line=class_node.lineno,
        owner_is_abstract=CLASS_NODE_AUTHORITY.is_abstract(class_node),
        owner_base_names=CLASS_NODE_AUTHORITY.declared_base_names(class_node),
        qualname=f"{class_node.name}.{method.name}",
        method_name=method.name,
        line=method.lineno,
        statement_count=max(1, statement_count),
        method_tokens=method_tokens,
        product_parameter_names=sorted_tuple(product_fields),
        forwarded_field_names=forwarded_field_names,
        construction_shape=construction_shape,
    )


def _variant_method_surfaces(module: ParsedModule) -> tuple[_VariantMethodSurface, ...]:
    surfaces = []
    for class_node in sorted(
        (node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)),
        key=lambda item: (item.lineno, item.name),
    ):
        if _is_private_symbol_name(class_node.name):
            continue
        for method in CLASS_NODE_AUTHORITY.methods(class_node):
            surface = _variant_method_surface(module, class_node, method)
            if surface is not None:
                surfaces.append(surface)
    return sorted_tuple(
        surfaces, key=lambda item: (item.file_path, item.owner_line, item.line)
    )


def _variant_method_family_seed(
    methods: tuple[_VariantMethodSurface, ...],
) -> _VariantMethodFamilySeed | None:
    if len(methods) < 2:
        return None
    token_sets = [set(method.method_tokens) for method in methods]
    anchor_tokens = sorted_tuple(set.intersection(*token_sets))
    variant_tokens = sorted_tuple(set.union(*token_sets) - set(anchor_tokens))
    if not anchor_tokens or not variant_tokens:
        return None
    if len(anchor_tokens) < 2 and len(methods) < 3:
        return None
    shared_product_parameter_names = sorted_tuple(
        set.intersection(*(set(method.product_parameter_names) for method in methods))
    )
    shared_field_names = sorted_tuple(
        set.intersection(*(set(method.forwarded_field_names) for method in methods))
    )
    if not shared_product_parameter_names or len(shared_field_names) < 2:
        return None
    return _VariantMethodFamilySeed(
        methods=methods,
        shared_product_parameter_names=shared_product_parameter_names,
        shared_field_names=shared_field_names,
        anchor_tokens=anchor_tokens,
        variant_tokens=variant_tokens,
    )


def _variant_method_family_candidate(
    seed: _VariantMethodFamilySeed,
    *,
    wrapper_chains: tuple[WrapperChainCandidate, ...],
    composition_signals: tuple[CancelableCompositionSignal, ...],
) -> _VariantMethodFamilyCandidate:
    exemplar = seed.methods[0]
    token_sources = (
        exemplar.owner_class_name,
        *exemplar.owner_base_names,
        *seed.anchor_tokens,
        *seed.variant_tokens,
        *seed.shared_product_parameter_names,
        *seed.shared_field_names,
        *(method.method_name for method in seed.methods),
    )
    related_compositions = RelatedCompositionSignalsAuthority.related(
        composition_signals,
        file_path=exemplar.file_path,
        token_sources=token_sources,
        field_names=seed.shared_field_names,
    )
    related_wrappers = RelatedWrapperChainsAuthority.related(
        wrapper_chains,
        file_path=exemplar.file_path,
        token_sources=token_sources,
    )
    return _VariantMethodFamilyCandidate(
        seed=seed,
        wrapper_chains=related_wrappers,
        composition_signals=related_compositions,
    )


def _variant_method_family_candidates(
    modules: list[ParsedModule],
) -> tuple[_VariantMethodFamilyCandidate, ...]:
    grouped: dict[
        tuple[str, str, str, tuple[str, ...], str],
        list[_VariantMethodSurface],
    ] = defaultdict(list)
    for module in modules:
        for surface in _variant_method_surfaces(module):
            grouped[
                (
                    surface.file_path,
                    surface.owner_class_name,
                    surface.construction_shape,
                    surface.product_parameter_names,
                    surface.method_tokens[-1],
                )
            ].append(surface)
    seeds = []
    for surfaces in grouped.values():
        ordered = sorted_tuple(surfaces, key=lambda item: (item.line, item.method_name))
        seed = _variant_method_family_seed(ordered)
        if seed is not None:
            seeds.append(seed)
    if not seeds:
        return ()

    wrapper_chains = tuple(
        chain for module in modules for chain in _wrapper_chain_candidates(module)
    )
    composition_signals = CancelableCompositionSignalQuery(tuple(modules)).signals()
    candidates = [
        _variant_method_family_candidate(
            seed,
            wrapper_chains=wrapper_chains,
            composition_signals=composition_signals,
        )
        for seed in seeds
    ]
    return sorted_tuple(
        candidates,
        key=lambda item: (
            item.seed.methods[0].file_path,
            item.seed.methods[0].owner_line,
            item.seed.methods[0].owner_class_name,
            item.seed.methods[0].construction_shape,
        ),
    )


class AlgebraicVariantMethodFamilyDetector(IssueDetector):
    detector_id = "algebraic_variant_method_family"
    detector_priority = -15
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Algebraic variant method family inflates public authority surface",
        "A public authority class that grows sibling methods whose names encode operation variants is exporting the operation algebra in method names. If those methods share a product carrier/request parameter and forward to the same construction shape, the variant should live in a nominal context, request, or product type instead of multiplying public methods.",
        "one algebraic operation over a nominal context/request/product variant",
        "same owner exposes variant-named methods over the same product construction",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS
        + _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        return [
            self._finding_for_candidate(candidate)
            for candidate in _variant_method_family_candidates(modules)
        ]

    def _finding_for_candidate(
        self, candidate: _VariantMethodFamilyCandidate
    ) -> RefactorFinding:
        seed = candidate.seed
        exemplar = seed.methods[0]
        method_summary = ", ".join(method.method_name for method in seed.methods)
        variant_summary = ", ".join(seed.variant_tokens[:8])
        field_summary = ", ".join(seed.shared_field_names[:8])
        parameter_summary = ", ".join(seed.shared_product_parameter_names)
        authority_kind = (
            "ABC/public authority"
            if exemplar.owner_is_abstract
            else "public authority"
        )
        composition_summary = ", ".join(
            signal.qualname for signal in candidate.composition_signals[:3]
        )
        wrapper_summary = ", ".join(
            " -> ".join(wrapper.qualname for wrapper in chain.wrappers)
            for chain in candidate.wrapper_chains[:2]
        )
        extra_context = []
        if composition_summary:
            extra_context.append(
                f"cancelable product composition(s): {composition_summary}"
            )
        if wrapper_summary:
            extra_context.append(f"wrapper chain(s): {wrapper_summary}")
        context_suffix = (
            f" It also intersects {'; '.join(extra_context)}." if extra_context else ""
        )
        return self.build_finding(
            (
                f"`{exemplar.owner_class_name}` inflates its {authority_kind} surface "
                f"with variant-named methods {method_summary}. They share product "
                f"parameter(s) {parameter_summary}, forward fields {field_summary}, "
                f"and return the same construction shape `{exemplar.construction_shape}`; "
                f"operation variants {variant_summary} should be encoded in the domain "
                f"algebra, not method names.{context_suffix}"
            ),
            candidate.evidence,
            scaffold=(
                f"class {exemplar.owner_class_name}(...):\n"
                "    def with_variants(self, request):\n"
                "        match request.operation:\n"
                "            case ...:\n"
                "                return self._construct_variants(request)\n\n"
                "# Collapse the sibling public methods into one algebraic operation.\n"
                "# Put the operation variant in a nominal request/context/product type, or make "
                "the product variant itself carry the operation semantics."
            ),
            codemod_patch=(
                f"# Replace variant method family {seed.method_names} on "
                f"`{exemplar.owner_class_name}` with one nominal request/context operation.\n"
                "# Use source-index anchored rewrites to migrate callers after the request/product "
                "type represents the operation variant explicitly."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=max(
                    2,
                    len(seed.methods)
                    + len(candidate.composition_signals)
                    + len(candidate.wrapper_chains),
                ),
                statement_count=max(
                    method.statement_count for method in seed.methods
                ),
                class_count=1,
                method_symbols=seed.method_names,
            ),
        )


@dataclass(frozen=True)
class MirroredImportFallbackCandidate(LineWitnessCandidate):
    imported_modules: tuple[str, ...]
    imported_name_count: int

    @property
    def witness_name(self) -> str:
        return "mirrored import fallback"


def _import_from_signature(
    statement: ast.stmt,
) -> tuple[str, tuple[tuple[str, str | None], ...], int] | None:
    if not isinstance(statement, ast.ImportFrom) or statement.module is None:
        return None
    return (
        statement.module,
        tuple(((alias.name, alias.asname) for alias in statement.names)),
        statement.level,
    )


def _is_import_error_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return False
    if isinstance(handler.type, ast.Name):
        return handler.type.id == "ImportError"
    return isinstance(handler.type, ast.Tuple) and any(
        isinstance(item, ast.Name) and item.id == "ImportError"
        for item in handler.type.elts
    )


def _mirrored_import_fallback_candidates(
    module: ParsedModule,
) -> tuple[MirroredImportFallbackCandidate, ...]:
    candidates: list[MirroredImportFallbackCandidate] = []
    for statement in _trim_docstring_body(module.module.body):
        if not isinstance(statement, ast.Try) or not statement.handlers:
            continue
        relative_imports = tuple(
            (
                signature
                for body_statement in statement.body
                if (signature := _import_from_signature(body_statement)) is not None
            )
        )
        if not relative_imports or len(relative_imports) != len(statement.body):
            continue
        if not all((level > 0 for _, _, level in relative_imports)):
            continue
        for handler in statement.handlers:
            if not _is_import_error_handler(handler):
                continue
            absolute_imports = tuple(
                (
                    signature
                    for body_statement in handler.body
                    if (signature := _import_from_signature(body_statement)) is not None
                )
            )
            if len(absolute_imports) != len(handler.body):
                continue
            normalized_relative = tuple(
                (module_name, names) for module_name, names, _ in relative_imports
            )
            normalized_absolute = tuple(
                (
                    (module_name, names)
                    for module_name, names, level in absolute_imports
                    if level == 0
                )
            )
            if normalized_relative != normalized_absolute:
                continue
            candidates.append(
                MirroredImportFallbackCandidate(
                    file_path=str(module.path),
                    line=statement.lineno,
                    imported_modules=tuple(
                        (module_name for module_name, _, _ in relative_imports)
                    ),
                    imported_name_count=sum(
                        (len(names) for _, names, _ in relative_imports)
                    ),
                )
            )
            break
    return tuple(candidates)


class MirroredImportFallbackDetector(
    ModuleCollectorCandidateDetector[MirroredImportFallbackCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Mirrored import fallback should collapse to one import authority",
        "A try/except ImportError block that repeats the same imports once relatively and once absolutely keeps two synchronized import surfaces. Prefer one package bootstrap or import adapter so direct-script and package execution share the same import authority.",
        "single import authority for package and direct-script execution",
        "relative and absolute import lists are mirrored across an ImportError fallback",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, import_candidate: MirroredImportFallbackCandidate
    ) -> RefactorFinding:
        module_summary = ", ".join(import_candidate.imported_modules)
        return self.build_finding(
            (
                f"{import_candidate.file_path} mirrors {import_candidate.imported_name_count} imported names "
                f"from {module_summary} across relative and absolute ImportError branches."
            ),
            (import_candidate.evidence,),
            scaffold=(
                "# Establish one package/direct-script import authority before local imports.\n# Then use canonical relative imports once instead of mirroring every import list."
            ),
            codemod_patch=(
                "# Replace mirrored relative/absolute import branches with a package bootstrap or shared import adapter."
            ),
            metrics=MappingMetrics(
                mapping_site_count=2,
                field_count=import_candidate.imported_name_count,
                mapping_name="mirrored import fallback",
                field_names=import_candidate.imported_modules,
            ),
        )


# fmt: off
materialize_product_record(product_record_spec('ConstantBackedDispatchAxisCandidate', 'axis_name: str; constant_prefix: str; constant_names: tuple[str, ...]; witness_name: ClassVar[AliasProperty[str]]', 'FunctionEvidenceLocationsCandidate', defaults={'witness_name': AliasProperty('axis_name')}))
# fmt: on


def _uppercase_constant_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and re.match("^[A-Z][A-Z0-9_]*$", node.id):
        return node.id
    return None


def _constant_name_prefix(name: str) -> str:
    return name.split("_", 1)[0]


def _axis_key(expression: str) -> str:
    return expression.rsplit(".", 1)[-1]


def _constant_names_in_node(node: ast.AST) -> tuple[str, ...]:
    names = {
        name
        for child in _walk_nodes(node)
        if (name := _uppercase_constant_name(child)) is not None
    }
    return sorted_tuple(names)


def _constant_backed_dispatch_tests(
    node: ast.AST,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tests: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            tests.extend(_constant_backed_dispatch_tests(value))
        return tuple(tests)
    if not isinstance(node, ast.Compare):
        return ()
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return ()
    op = node.ops[0]
    comparator = node.comparators[0]
    if isinstance(op, (ast.Eq, ast.NotEq)):
        left_name = _uppercase_constant_name(node.left)
        right_name = _uppercase_constant_name(comparator)
        if right_name is not None:
            tests.append((ast.unparse(node.left), (right_name,)))
        elif left_name is not None:
            tests.append((ast.unparse(comparator), (left_name,)))
    elif isinstance(op, (ast.In, ast.NotIn)):
        constant_names = _constant_names_in_node(comparator)
        if constant_names:
            tests.append((ast.unparse(node.left), constant_names))
    return tuple(tests)


def _constant_backed_dispatch_axis_candidates(
    module: ParsedModule,
    config: DetectorConfig,
) -> tuple[ConstantBackedDispatchAxisCandidate, ...]:
    del config
    grouped: dict[tuple[str, str], list[tuple[str, int, tuple[str, ...]]]] = (
        defaultdict(list)
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, ast.If):
                continue
            for axis_expression, constant_names in _constant_backed_dispatch_tests(
                node.test
            ):
                if not constant_names:
                    continue
                prefix_counts = Counter(
                    _constant_name_prefix(name) for name in constant_names
                )
                constant_prefix, count = prefix_counts.most_common(1)[0]
                if count != len(constant_names):
                    continue
                grouped[_axis_key(axis_expression), constant_prefix].append(
                    (qualname, node.lineno, constant_names)
                )

    candidates: list[ConstantBackedDispatchAxisCandidate] = []
    for (axis_name, constant_prefix), sites in grouped.items():
        constant_names = sorted_tuple({name for _, _, names in sites for name in names})
        function_names = tuple(dict.fromkeys(qualname for qualname, _, _ in sites))
        if len(constant_names) < 4 or len(function_names) < 2:
            continue
        ordered_sites = sorted_tuple(sites, key=lambda item: (item[1], item[0]))
        evidence_by_function: dict[str, int] = {}
        for qualname, line, _ in ordered_sites:
            evidence_by_function.setdefault(qualname, line)
        candidates.append(
            ConstantBackedDispatchAxisCandidate(
                file_path=str(module.path),
                line=ordered_sites[0][1],
                axis_name=axis_name,
                constant_prefix=constant_prefix,
                constant_names=constant_names,
                function_names=tuple(evidence_by_function.keys()),
                line_numbers=tuple(evidence_by_function.values()),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.axis_name)
    )


class ConstantBackedDispatchAxisDetector(
    ConfiguredModuleCollectorCandidateDetector[ConstantBackedDispatchAxisCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Constant-backed action axis should become one typed dispatch authority",
        "A closed behavior axis is declared as uppercase constants and then re-derived through branch ladders. That splits the action family across constants, choices, and dispatch code. Prefer one typed action authority that derives choices, ordering, and execution.",
        "single typed action-family authority deriving choices and dispatch",
        "same constant family drives branch dispatch across multiple functions",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _LITERAL_ID_DISPATCH_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, axis_candidate: ConstantBackedDispatchAxisCandidate
    ) -> RefactorFinding:
        constants = ", ".join(axis_candidate.constant_names[:8])
        functions = ", ".join(axis_candidate.function_names)
        return self.build_finding(
            (
                f"`{axis_candidate.axis_name}` dispatches over constant family `{axis_candidate.constant_prefix}_*` "
                f"({constants}) across {functions}."
            ),
            axis_candidate.evidence_locations,
            scaffold=(
                "class Action(ABC):\n    key: ClassVar[str]\n    @abstractmethod\n    def run(self, context): ...\n\nACTIONS = tuple(Action.__subclasses__())\nCHOICES = tuple(action.key for action in ACTIONS)"
            ),
            codemod_patch=(
                "# Replace constant choices plus branch ladders with one typed action table or auto-registered action family.\n# Derive CLI choices and all dispatch sites from that authority."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                axis_candidate.axis_name,
                axis_candidate.constant_names,
            ),
        )


@dataclass(frozen=True)
class ManualProcessStepLadderCandidate(FunctionEvidenceLocationsCandidate):
    step_table_names: tuple[str, ...]
    minimum_step_count: int

    @property
    def witness_name(self) -> str:
        return "manual process step ladder"


def _assigned_process_step_tables(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[int, int]]:
    tables: dict[str, tuple[int, int]] = {}
    for node in _walk_function_body_nodes(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) < 2:
            continue
        tuple_items = [
            item
            for item in value.elts
            if isinstance(item, (ast.Tuple, ast.List)) and len(item.elts) >= 2
        ]
        if len(tuple_items) < 2:
            continue
        tables[target.id] = (node.lineno, len(tuple_items))
    return tables


def _loop_iter_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "enumerate")
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def _unpacked_target_leaf_count(node: ast.AST) -> int:
    if isinstance(node, ast.Name):
        return 1
    if isinstance(node, (ast.Tuple, ast.List)):
        return sum((_unpacked_target_leaf_count(elt) for elt in node.elts))
    return 0


def _loop_has_process_call(loop: ast.For) -> bool:
    for node in _walk_nodes(loop):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if any((token in callee.lower() for token in ("run", "popen", "subprocess"))):
            return True
    return False


def _manual_process_step_ladder_candidates(
    module: ParsedModule,
) -> tuple[ManualProcessStepLadderCandidate, ...]:
    sites: list[tuple[str, str, int, int]] = []
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        tables = _assigned_process_step_tables(function)
        if not tables:
            continue
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, ast.For):
                continue
            table_name = _loop_iter_name(node.iter)
            if (
                table_name not in tables
                or _unpacked_target_leaf_count(node.target) < 2
                or (not _loop_has_process_call(node))
            ):
                continue
            table_line, step_count = tables[table_name]
            sites.append((qualname, table_name, table_line, step_count))
    if len(sites) < 2:
        return ()
    ordered = sorted_tuple(sites, key=lambda item: (item[2], item[0], item[1]))
    return (
        ManualProcessStepLadderCandidate(
            file_path=str(module.path),
            line=ordered[0][2],
            step_table_names=tuple((table_name for _, table_name, _, _ in ordered)),
            function_names=tuple((qualname for qualname, _, _, _ in ordered)),
            line_numbers=tuple((line for _, _, line, _ in ordered)),
            minimum_step_count=min((step_count for _, _, _, step_count in ordered)),
        ),
    )


class ManualProcessStepLadderDetector(
    ModuleCollectorCandidateDetector[ManualProcessStepLadderCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Manual process-step ladders should become a typed stage plan",
        "Multiple functions declare local command-step tables and execute them through repeated loops. The step schema, execution policy, and failure policy are one staged orchestration authority, not separate local declarations.",
        "single typed process-stage plan deriving command lists and execution loops",
        "local process-step tables are manually executed by repeated loop skeletons",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ladder_candidate: ManualProcessStepLadderCandidate
    ) -> RefactorFinding:
        tables = ", ".join(ladder_candidate.step_table_names)
        functions = ", ".join(ladder_candidate.function_names)
        return self.build_finding(
            (
                f"{ladder_candidate.file_path} repeats local process-step tables {tables} "
                f"and execution loops across {functions}."
            ),
            ladder_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass ProcessStagePlan:\n    steps: tuple[ProcessStep, ...]\n    def run(self, context): ..."
            ),
            codemod_patch=(
                "# Replace local command-step tables and repeated loops with one typed stage plan.\n# Derive command argv, labels, allowed failures, and callbacks from the plan rows."
            ),
            compression_certificate=_manual_process_step_ladder_compression_certificate(
                ladder_candidate
            ),
            metrics=OrchestrationMetrics(
                function_line_count=sum(ladder_candidate.line_numbers) * 0,
                branch_site_count=len(ladder_candidate.step_table_names),
                call_site_count=len(ladder_candidate.step_table_names),
                parameter_count=0,
                callee_family_count=1,
            ),
        )


@dataclass(frozen=True)
class MirroredFileRewriteLoopCandidate(LineWitnessCandidate):
    function_name: str
    line_numbers: tuple[int, ...]

    @property
    def witness_name(self) -> str:
        return "mirrored file rewrite loops"

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, self.function_name)
                for line in self.line_numbers
            )
        )


def _iterates_globbed_files(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in {"glob", "rglob", "iterdir"}


def _loop_has_text_rewrite_signature(loop: ast.For) -> bool:
    has_file_iteration = _iterates_globbed_files(loop.iter)
    has_read = False
    has_write = False
    has_replace = False
    for node in _walk_nodes(loop):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        has_read = has_read or func.attr == "read_text"
        has_write = has_write or func.attr == "write_text"
        has_replace = has_replace or func.attr == "replace"
    return has_file_iteration and has_read and has_write and has_replace


def _mirrored_file_rewrite_loop_candidates(
    module: ParsedModule,
) -> tuple[MirroredFileRewriteLoopCandidate, ...]:
    candidates: list[MirroredFileRewriteLoopCandidate] = []
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        loops = tuple(
            (
                node
                for node in _walk_function_body_nodes(function)
                if isinstance(node, ast.For) and _loop_has_text_rewrite_signature(node)
            )
        )
        if len(loops) < 2:
            continue
        candidates.append(
            MirroredFileRewriteLoopCandidate(
                file_path=str(module.path),
                line=loops[0].lineno,
                function_name=qualname,
                line_numbers=tuple((loop.lineno for loop in loops)),
            )
        )
    return tuple(candidates)


class MirroredFileRewriteLoopDetector(
    ModuleCollectorCandidateDetector[MirroredFileRewriteLoopCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Mirrored file rewrite loops should become a text rewrite plan",
        "Several loops read files, apply the same textual rewrite mechanics, and write changes back. The traversal roots are local variation, but the rewrite algebra and write policy should be one declared plan.",
        "single text rewrite plan with one file-application surface",
        "same read/transform/write loop mirrored over different file collections",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, loop_candidate: MirroredFileRewriteLoopCandidate
    ) -> RefactorFinding:
        lines = ", ".join(str(line) for line in loop_candidate.line_numbers)
        return self.build_finding(
            (
                f"{loop_candidate.file_path} mirrors file rewrite loops in "
                f"{loop_candidate.function_name} at lines {lines}."
            ),
            loop_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass TextRewritePlan:\n    rules: tuple[TextRewriteRule, ...]\n    def apply_to_files(self, files): ..."
            ),
            codemod_patch=(
                "# Replace mirrored read/replace/write loops with one typed rewrite plan.\n# Pass only the varying file collections and display labels at call sites."
            ),
            compression_certificate=_mirrored_file_rewrite_loop_compression_certificate(
                loop_candidate
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(loop_candidate.line_numbers),
                field_count=0,
                mapping_name="text rewrite",
                field_names=(),
                source_name=loop_candidate.function_name,
                identity_field_names=(),
            ),
        )


@dataclass(frozen=True)
class RepeatedLocalRegexBundleCandidate(FunctionEvidenceLocationsCandidate):
    owner_name: str
    regex_literals: tuple[str, ...]

    @property
    def witness_name(self) -> str:
        return "repeated local regex bundle"


def _regex_literal_from_call(node: ast.Call) -> str | None:
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and (func.value.id == "re")
        and (
            func.attr
            in {"compile", "findall", "finditer", "search", "match", "fullmatch", "sub"}
        )
    ):
        return None
    if not node.args:
        return None
    pattern_arg = node.args[0]
    if not (
        isinstance(pattern_arg, ast.Constant) and isinstance(pattern_arg.value, str)
    ):
        return None
    return pattern_arg.value


def _is_substantial_regex_literal(literal: str) -> bool:
    if len(literal) < 12:
        return False
    if not any((token in literal for token in ("\\", "[", "(", "{", "^", "$"))):
        return False
    alpha_count = sum(1 for char in literal if char.isalpha())
    return alpha_count >= 3


def _local_regex_literals_by_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    literals: dict[str, int] = {}
    for node in _walk_function_body_nodes(function):
        if not isinstance(node, ast.Call):
            continue
        literal = _regex_literal_from_call(node)
        if literal is None or not _is_substantial_regex_literal(literal):
            continue
        literals.setdefault(literal, node.lineno)
    return literals


def _function_owner_name(qualname: str) -> str:
    if "." not in qualname:
        return "<module>"
    return qualname.rsplit(".", 1)[0]


def _repeated_local_regex_bundle_candidates(
    module: ParsedModule, config: DetectorConfig
) -> tuple[RepeatedLocalRegexBundleCandidate, ...]:
    functions_by_owner: dict[
        (str, list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, dict[str, int]]])
    ] = defaultdict(list)
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        literals = _local_regex_literals_by_function(function)
        if literals:
            functions_by_owner[_function_owner_name(qualname)].append(
                (qualname, function, literals)
            )

    candidates: list[RepeatedLocalRegexBundleCandidate] = []
    for owner_name, functions in functions_by_owner.items():
        for left_index, (left_name, _left_function, left_literals) in enumerate(
            functions
        ):
            for right_name, _right_function, right_literals in functions[
                left_index + 1 :
            ]:
                shared = sorted_tuple(set(left_literals) & set(right_literals))
                if len(shared) < config.min_repeated_local_regex_literals:
                    continue
                line_numbers = (
                    min((left_literals[literal] for literal in shared)),
                    min((right_literals[literal] for literal in shared)),
                )
                candidates.append(
                    RepeatedLocalRegexBundleCandidate(
                        file_path=str(module.path),
                        line=min(line_numbers),
                        owner_name=owner_name,
                        function_names=(left_name, right_name),
                        regex_literals=shared,
                        line_numbers=line_numbers,
                    )
                )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.function_names,
            candidate.regex_literals,
        ),
    )


class RepeatedLocalRegexBundleDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedLocalRegexBundleCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated local regex bundles should become a typed syntax authority",
        "Sibling functions redeclare the same substantial regex grammar locally. That makes each function a partial syntax authority instead of deriving parsing from one typed grammar object.",
        "single typed syntax authority deriving all repeated regex recognizers",
        "substantial regex literals are redeclared inside sibling functions",
        _AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, regex_candidate: RepeatedLocalRegexBundleCandidate
    ) -> RefactorFinding:
        functions = ", ".join(regex_candidate.function_names)
        return self.build_finding(
            (
                f"{regex_candidate.file_path} repeats {len(regex_candidate.regex_literals)} "
                f"local regex grammar literals across {functions}."
            ),
            regex_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass SyntaxAuthority:\n    recognizers: tuple[Pattern[str], ...]\n    def parse(self, text: str): ..."
            ),
            codemod_patch=(
                "# Move repeated local regex grammar into one typed syntax authority.\n# Derive parser operations from named recognizers instead of redeclaring patterns in each helper."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(regex_candidate.function_names),
                mapping_name="regex syntax authority",
                field_names=regex_candidate.regex_literals,
                source_name=regex_candidate.owner_name,
                identity_field_names=(),
            ),
        )


@dataclass(frozen=True)
class AlgebraicDuplicateCompoundBlockCandidate(FunctionEvidenceLocationsCandidate):
    block_kind: str
    normal_form_size: int

    @property
    def witness_name(self) -> str:
        return "algebraic duplicate compound block"


def _algebraic_ast_key(node: object) -> object:
    if isinstance(node, ast.Name):
        return ("Name", type(node.ctx).__name__)
    if isinstance(node, ast.arg):
        return ("arg",)
    if isinstance(node, ast.Attribute):
        return ("Attribute", _algebraic_ast_key(node.value), "ATTR")
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value).__name__)
    if isinstance(node, ast.keyword):
        return ("keyword", "ARG", _algebraic_ast_key(node.value))
    if isinstance(node, ast.alias):
        return ("alias",)
    if isinstance(node, ast.AST):
        fields = []
        for field_name, value in ast.iter_fields(node):
            if field_name in {
                "lineno",
                "col_offset",
                "end_lineno",
                "end_col_offset",
                "ctx",
                "type_comment",
            }:
                continue
            fields.append((field_name, _algebraic_ast_key(value)))
        return (type(node).__name__, tuple(fields))
    if isinstance(node, list):
        return tuple((_algebraic_ast_key(item) for item in node))
    if isinstance(node, tuple):
        return tuple((_algebraic_ast_key(item) for item in node))
    return type(node).__name__


def _algebraic_normal_form_size(normal_form: object) -> int:
    if isinstance(normal_form, tuple):
        return 1 + sum((_algebraic_normal_form_size(item) for item in normal_form))
    return 1


def _has_nested_compound_statement(node: ast.AST) -> bool:
    for child in _walk_nodes(node):
        if child is node:
            continue
        if isinstance(child, (ast.For, ast.While, ast.If, ast.Try, ast.With)):
            return True
    return False


def _algebraic_duplicate_compound_block_candidates(
    module: ParsedModule,
) -> tuple[AlgebraicDuplicateCompoundBlockCandidate, ...]:
    grouped: dict[(tuple[str, object], list[tuple[str, int, object]])] = defaultdict(
        list
    )
    for qualname, function in SurfaceFunctionIndex.from_module(module.module).functions:
        for node in _walk_function_body_nodes(function):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            if not _has_nested_compound_statement(node):
                continue
            block_kind = type(node).__name__
            normal_form = _algebraic_ast_key(node)
            grouped[block_kind, normal_form].append(
                (qualname, node.lineno, normal_form)
            )

    candidates: list[AlgebraicDuplicateCompoundBlockCandidate] = []
    for (block_kind, normal_form), sites in grouped.items():
        first_site_by_function: dict[str, tuple[int, object]] = {}
        for function_name, line_number, site_normal_form in sorted(
            sites, key=lambda item: (item[0], item[1])
        ):
            first_site_by_function.setdefault(
                function_name, (line_number, site_normal_form)
            )
        if len(first_site_by_function) < 2:
            continue
        ordered_items = sorted_tuple(
            first_site_by_function.items(), key=lambda item: item[1][0]
        )
        candidates.append(
            AlgebraicDuplicateCompoundBlockCandidate(
                file_path=str(module.path),
                line=ordered_items[0][1][0],
                block_kind=block_kind,
                function_names=tuple(
                    (function_name for function_name, _ in ordered_items)
                ),
                line_numbers=tuple((line for _, (line, _) in ordered_items)),
                normal_form_size=_algebraic_normal_form_size(normal_form),
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.block_kind,
            candidate.function_names,
        ),
    )


class AlgebraicDuplicateCompoundBlockDetector(
    ModuleCollectorCandidateDetector[AlgebraicDuplicateCompoundBlockCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Anti-unified compound blocks should become one derived algebra",
        "The repeated blocks have the same quotient-normal-form AST after alpha-renaming local names, literals, and attribute labels. That is a formal witness that the algorithmic structure is duplicated modulo representation choices.",
        "single derived algorithm authority for an anti-unified compound block",
        "compound blocks are equal in the AST quotient algebra modulo names and literals",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_PROVENANCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, block_candidate: AlgebraicDuplicateCompoundBlockCandidate
    ) -> RefactorFinding:
        functions = ", ".join(block_candidate.function_names)
        return self.build_finding(
            (
                f"{block_candidate.file_path} repeats an anti-unified "
                f"{block_candidate.block_kind} block across {functions}."
            ),
            block_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass BlockAlgebra:\n    def run(self, context): ...\n\n# Route each former block through one derived algebra with typed context rows."
            ),
            codemod_patch=(
                "# Extract the repeated quotient-normal-form block into one typed helper or algebra object.\n# Keep variation as context data; derive the shared control structure once."
            ),
            compression_certificate=_algebraic_duplicate_compound_block_compression_certificate(
                block_candidate
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=len(block_candidate.function_names),
                call_site_count=len(block_candidate.function_names),
                parameter_count=0,
                callee_family_count=1,
            ),
        )


class RepeatedProjectionHelperDetector(
    ModuleCollectorCandidateDetector[tuple[ProjectionHelperShape, ...]]
):
    detector_id = "repeated_projection_helpers"
    candidate_collector = _projection_helper_groups
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated projection helper wrappers should become one projector",
        "The docs treat parallel projection helpers as a coherence failure: once several helpers differ only in which semantic attribute they project, the wrapper structure should be centralized in one authoritative projector and the varying projection should become a parameter.",
        "single authoritative projection helper for a repeated semantic wrapper family",
        "same helper wrapper shape repeated across sibling module functions",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _PROJECTION_HELPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[ProjectionHelperShape, ...]
    ) -> RefactorFinding:
        attributes = {shape.projected_attribute for shape in ordered}
        evidence = tuple(
            (
                SourceLocation(shape.file_path, shape.lineno, shape.symbol)
                for shape in ordered[:6]
            )
        )
        return self.build_finding(
            f"Projection helper wrappers {', '.join((shape.function_name for shape in ordered[:4]))} repeat the same wrapper shape while only projecting different attributes.",
            evidence,
            scaffold=_projection_helper_scaffold(list(ordered)),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered), field_count=len(attributes)
            ),
        )


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Parallel guarded wrappers and specs should become a polymorphic family",
        "Parallel wrapper functions plus parallel spec declarations mean the code already has a hidden strategy family, but it is encoded as duplicated procedural glue. The docs prefer moving the shared algorithm into an ABC and letting polymorphic spec classes own the node family differences.",
        "single authoritative polymorphic wrapper/spec family",
        "same node-guarded wrapper skeleton repeated across multiple wrapper/spec pairs",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _SCOPED_SHAPE_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        wrapper_pairs = _guarded_wrapper_spec_pairs(module)
        if len(wrapper_pairs) < 2:
            return []
        evidence_items = [
            SourceLocation(str(module.path), pair.spec_line, pair.spec_name)
            for pair in wrapper_pairs[:6]
        ]
        evidence_items.extend(
            (
                SourceLocation(str(module.path), pair.function_line, pair.function_name)
                for pair in wrapper_pairs[:6]
            )
        )
        evidence = tuple(
            sorted(evidence_items, key=lambda item: (item.line, item.symbol))[:8]
        )
        function_names = ", ".join(pair.function_name for pair in wrapper_pairs)
        spec_names = ", ".join(pair.spec_name for pair in wrapper_pairs)
        node_families = ", ".join(
            sorted({"/".join(pair.node_types) for pair in wrapper_pairs})
        )
        return [
            self.build_finding(
                f"{module.path} encodes guarded wrapper functions {function_names} and specs {spec_names} as parallel wrapper/spec pairs over node families {node_families}.",
                evidence,
                scaffold="class NodeFamilySpec(ABC):\n    node_types: ClassVar[tuple[type[ast.AST], ...]]\n\n    @classmethod\n    def build(cls, parsed_module, observation):\n        node = observation.node\n        if not isinstance(node, cls.node_types):\n            return None\n        return cls.build_for_node(parsed_module, node, observation)",
            )
        ]


class ManualIndexedFamilyExpansionDetector(PerModuleIssueDetector):
    detector_id = "manual_indexed_family"
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Manually expanded indexed family should become one nominal family abstraction",
        "The same collection scaffold is being hand-expanded over a latent family index. The docs prefer one authoritative nominal family abstraction whose members provide only the varying family metadata.",
        "single authoritative indexed family abstraction",
        "same normalized family scaffold repeated across sibling top-level functions",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        groups: dict[str, list[IndexedFamilyWrapperCandidate]] = defaultdict(list)
        for candidate in _indexed_family_wrapper_candidates(module):
            groups[candidate.collector_name].append(candidate)
        findings: list[RefactorFinding] = []
        for candidates in groups.values():
            if len(candidates) < 2:
                continue
            ordered = sorted(candidates, key=lambda item: item.lineno)
            evidence = tuple(
                (
                    SourceLocation(str(module.path), item.lineno, item.function_name)
                    for item in ordered[:6]
                )
            )
            findings.append(
                self.build_finding(
                    f"{module.path} hand-expands indexed family members {', '.join((item.function_name for item in ordered[:4]))} over `{ordered[0].collector_name}`.",
                    evidence,
                    scaffold="Introduce one nominal family abstraction that owns the shared collection scaffold and encode only the varying family index metadata in subclasses or descriptors.",
                )
            )
        return findings


class AccessorWrapperDetector(
    ModuleCollectorCandidateDetector[tuple[AccessorWrapperCandidate, ...]]
):
    candidate_collector = _accessor_wrapper_groups
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Trivial structural accessor wrapper should collapse to attribute/property access",
        "The docs treat one-step observation wrappers as redundant structure: if a method only transports an already-owned attribute or a one-step computed view of it, the authority should remain the attribute itself, with `@property` reserved for genuine computed access.",
        "direct authoritative attribute/property access instead of transport wrappers",
        "same class exposes owned facts through one-step transport wrappers",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[AccessorWrapperCandidate, ...]
    ) -> RefactorFinding:
        class_name = ordered[0].class_name
        evidence = tuple(
            (
                SourceLocation(
                    ordered_item.file_path, ordered_item.lineno, ordered_item.symbol
                )
                for ordered_item in ordered[:6]
            )
        )
        replacement_examples = "\n".join(
            (
                _accessor_replacement_example(ordered_item)
                for ordered_item in ordered[:3]
            )
        )
        observed_attrs = ", ".join(
            sorted({ordered_item.observed_attribute for ordered_item in ordered})
        )
        wrapper_shapes = ", ".join(
            sorted(
                {
                    ordered_item.wrapper_shape.replace("_", " ")
                    for ordered_item in ordered
                }
            )
        )
        return self.build_finding(
            f"Class {class_name} exposes {len(ordered)} structural accessor wrapper(s) over {observed_attrs}.",
            evidence,
            relation_context=f"same class repeats {wrapper_shapes} around owned attributes instead of exposing one authoritative access path",
            scaffold=f"Collapse these transport wrappers to direct dot access when they only expose owned state. If a one-step computed view must remain public, express it as an `@property`.\n\nExample replacements:\n{replacement_examples}",
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len(
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
                mapping_name=f"{class_name} property",
                field_names=sorted_tuple(
                    {ordered_item.observed_attribute for ordered_item in ordered}
                ),
            ),
        )


@dataclass(frozen=True)
class FlattenedProjectionPropertyCandidate(LineWitnessCandidate):
    class_name: str
    property_name: str
    nested_owner: str
    nested_member: str

    @property
    def nested_access(self) -> str:
        return f"{self.nested_owner}.{self.nested_member}"

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.property_name}"

    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("symbol")


def _flattened_projection_properties(
    module: ParsedModule,
) -> tuple[tuple[FlattenedProjectionPropertyCandidate, ...], ...]:
    grouped: dict[str, list[FlattenedProjectionPropertyCandidate]] = defaultdict(list)
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        for statement in class_node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if not any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            if len(statement.args.args) != 1:
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not (
                isinstance(returned, ast.Attribute)
                and isinstance(returned.value, ast.Attribute)
                and isinstance(returned.value.value, ast.Name)
                and (returned.value.value.id == "self")
            ):
                continue
            nested_owner = returned.value.attr
            nested_member = returned.attr
            expected_alias = f"{nested_owner}_{nested_member}"
            if statement.name != expected_alias:
                continue
            grouped[class_node.name].append(
                FlattenedProjectionPropertyCandidate(
                    file_path=str(module.path),
                    class_name=class_node.name,
                    property_name=statement.name,
                    nested_owner=nested_owner,
                    nested_member=nested_member,
                    line=statement.lineno,
                )
            )
    return tuple(
        (
            sorted_tuple(items, key=lambda item: (item.line, item.property_name))
            for _, items in sorted(grouped.items())
            if len(items) >= 2
        )
    )


class FlattenedProjectionPropertyDetector(
    ModuleCollectorCandidateDetector[tuple[FlattenedProjectionPropertyCandidate, ...]]
):
    candidate_collector = _flattened_projection_properties
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Flattened compatibility projection properties should be deleted",
        "After a role-prefixed field bundle is moved into nominal nested records, adding properties such as `source_value -> source.value` preserves the old flattened schema as a shadow API. That is a local minimum: callers should move to the nested role record directly so the new schema is the only authority.",
        "direct nested record access instead of flattened compatibility aliases",
        "class exposes old role-prefixed fields as properties over nested role records",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_KEYWORD_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, ordered: tuple[FlattenedProjectionPropertyCandidate, ...]
    ) -> RefactorFinding:
        class_name = ordered[0].class_name
        evidence = tuple(item.evidence for item in ordered[:8])
        aliases = ", ".join(item.property_name for item in ordered)
        examples = "\n".join(
            (
                f"- replace `obj.{item.property_name}` with `obj.{item.nested_access}`"
                for item in ordered[:5]
            )
        )
        return self.build_finding(
            (
                f"`{class_name}` keeps flattened compatibility properties {aliases} over nested role records."
            ),
            evidence,
            scaffold=(
                "Delete the compatibility properties and update callers to use the nested nominal record directly.\n\n"
                f"{examples}"
            ),
            codemod_patch=(
                f"# Remove flattened projection properties from `{class_name}`.\n"
                "# Rewrite call sites to the nested role-record path shown in the scaffold."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len({item.nested_access for item in ordered}),
                mapping_name=f"{class_name} flattened projection properties",
                field_names=tuple(item.property_name for item in ordered),
            ),
        )


class WrapperChainDetector(ModuleCollectorCandidateDetector[WrapperChainCandidate]):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Transport wrapper chain should collapse to one authoritative view",
        "The docs treat stacked pass-through helpers and projection wrappers as a coherence failure: once the same facts are rewrapped across multiple helper layers, the code should keep one authoritative carrier and derive smaller views directly from it.",
        "direct authoritative projection/view instead of a stacked transport wrapper chain",
        "same fact family is transported through multiple wrapper layers before reaching the real owner",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, chain_candidate: WrapperChainCandidate
    ) -> RefactorFinding:
        wrapper_symbols = tuple(item.qualname for item in chain_candidate.wrappers)
        evidence = tuple(item.evidence for item in chain_candidate.wrappers[:6])
        projected_attributes = sorted_tuple(
            {
                attr
                for item in chain_candidate.wrappers
                for attr in item.projected_attributes
            }
        )
        scaffold = f"Keep one authoritative view/carrier and derive the smaller wrapper views directly from it.\n\nWrapper chain: {' -> '.join(wrapper_symbols)} -> {chain_candidate.leaf_delegate_symbol}"
        if projected_attributes:
            scaffold += f"\nProjected attributes observed in the chain: {', '.join(projected_attributes)}"
        return self.build_finding(
            f"Wrappers {', '.join(wrapper_symbols)} form a stacked transport chain over `{chain_candidate.leaf_delegate_symbol}`.",
            evidence,
            scaffold=scaffold,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(chain_candidate.wrappers),
                statement_count=max(
                    (item.statement_count for item in chain_candidate.wrappers)
                ),
                class_count=len(
                    {
                        (
                            item.qualname.split(".", 1)[0]
                            if "." in item.qualname
                            else "<module>"
                        )
                        for item in chain_candidate.wrappers
                    }
                ),
                method_symbols=wrapper_symbols,
            ),
        )


class TrivialForwardingWrapperDetector(
    ModuleCollectorCandidateDetector[TrivialForwardingWrapperCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Trivial forwarding wrapper should be deleted in favor of the delegate authority",
        "A one-line wrapper that only transports inputs into `for_*().method()` or a similar nested delegate call adds no stable semantics. The docs treat that as zero-information indirection: call the authority directly at the use site instead of naming a transport shell.",
        "direct delegate authority call instead of a trivial forwarding shell",
        "wrapper symbol only transports existing inputs into a nested delegate call chain",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, wrapper_candidate: TrivialForwardingWrapperCandidate
    ) -> RefactorFinding:
        transported_inputs = ", ".join(wrapper_candidate.transported_value_sources[:4])
        input_summary = (
            f" It only transports {transported_inputs}." if transported_inputs else ""
        )
        private_delegate_root = _delegate_root_symbol(wrapper_candidate.delegate_symbol)
        private_delegate_summary = _is_private_symbol_name(private_delegate_root)
        scaffold = (
            f"# Delete `{wrapper_candidate.qualname}` and call `{wrapper_candidate.delegate_symbol}` directly at the use site.\n"
            "# Keep the wrapper only if it owns a new invariant, provenance boundary, or semantic rename."
        )
        codemod_patch = (
            f"# Inline `{wrapper_candidate.qualname}` into its callers.\n"
            f"# Replace the wrapper with direct calls to `{wrapper_candidate.delegate_symbol}`."
        )
        if private_delegate_summary:
            scaffold = (
                f"# `{wrapper_candidate.qualname}` is trivial, but its delegate root `{private_delegate_root}` is private.\n"
                "# Promote a public facade/ABC/policy authority instead of routing callers directly to the private delegate."
            )
            codemod_patch = (
                f"# Do not inline callers of `{wrapper_candidate.qualname}` directly onto private `{private_delegate_root}`.\n"
                "# Promote one public authority that owns the delegate contract, then route callers through that authority."
            )
        return self.build_finding(
            f"`{wrapper_candidate.qualname}` is a {wrapper_candidate.call_depth}-step forwarding wrapper over `{wrapper_candidate.delegate_symbol}`.{input_summary}",
            (wrapper_candidate.evidence,),
            scaffold=scaffold,
            codemod_patch=codemod_patch,
        )


class PublicApiPrivateDelegateShellDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Public API shell over a private delegate should promote a public authority",
        "A public module-level wrapper is carrying an external API contract only because the real implementation authority is hidden behind a private `_X` root. When multiple external call sites depend on that shell, the docs prefer promoting one public facade/ABC/policy authority instead of inlining callers onto the private delegate.",
        "public authoritative facade over a private delegate family",
        "external modules depend on a public forwarding shell because the true authority is private",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_INTERFACE_IDENTITY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_shell_candidates(modules, config):
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            external_module_suffix = (
                f" External dependents include {external_module_summary}."
                if external_module_summary
                else ""
            )
            findings.append(
                self.build_finding(
                    (
                        f"`{candidate.wrapper.qualname}` is a public forwarding shell over private "
                        f"`{candidate.delegate_root_symbol}`, and {len(candidate.external_callsites)} external "
                        f"call site(s) across {len(candidate.external_module_names)} module(s) depend on it."
                        f"{external_module_suffix}"
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicDelegatePolicy(ABC):\n    @classmethod\n    @abstractmethod\n    def for_key(cls, key): ...\n\n    @abstractmethod\n    def execute(self, *args, **kwargs): ...\n\n# Keep the concrete private delegate hidden behind this public authority."
                    ),
                    codemod_patch=(
                        f"# Do not inline callers of `{candidate.wrapper.qualname}` onto private `{candidate.delegate_root_symbol}`.\n"
                        "# Promote one public facade/ABC/policy authority that owns the contract, then route external call sites through it."
                    ),
                )
            )
        return findings


class PublicApiPrivateDelegateFamilyDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Multiple public shells over one private delegate should collapse into a public facade family",
        "When several public wrappers expose one private delegate root, the external API is fragmented across transport shells instead of owned by one public authority. The docs prefer promoting a public facade, ABC, or policy surface rather than keeping multiple pass-through exports over private machinery.",
        "single public facade family over one private delegate root",
        "multiple public wrappers expose one private delegate family to external modules",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _ACCESSOR_WRAPPER_INTERFACE_IDENTITY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _public_api_private_delegate_family_candidates(
            modules, config
        ):
            wrapper_summary = ", ".join(candidate.wrapper_names[:4])
            external_module_summary = ", ".join(candidate.external_module_names[:3])
            findings.append(
                self.build_finding(
                    (
                        f"Public wrappers {wrapper_summary} expose private `{candidate.delegate_root_symbol}` "
                        f"through {len(candidate.external_callsites)} external call site(s) across "
                        f"{len(candidate.external_module_names)} module(s). External dependents include "
                        f"{external_module_summary}."
                    ),
                    candidate.evidence,
                    scaffold=(
                        "class PublicFacadePolicy(ABC):\n    @classmethod\n    @abstractmethod\n    def for_key(cls, key): ...\n\n    @abstractmethod\n    def route(self, *args, **kwargs): ...\n\n# Re-export the contract through this public authority instead of multiple module-level shells."
                    ),
                    codemod_patch=(
                        f"# Collapse wrappers {candidate.wrapper_names} into one public facade over `{candidate.delegate_root_symbol}`.\n"
                        "# Keep the private delegate hidden and route external modules through the promoted public authority."
                    ),
                )
            )
        return findings


class NominalPolicySurfaceDetector(
    ConfiguredModuleCollectorCandidateDetector[NominalPolicySurfaceFamilyCandidate]
):
    candidate_collector = _nominal_policy_surface_family_candidates
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Nominal surface methods should not be thin shells over a policy family",
        "A nominal owner exposes public methods or properties that do nothing except resolve a policy family and forward into it. The docs treat that as split authority: the owner surface should either own the contract directly or expose one explicit policy hook instead of scattering zero-information shells.",
        "single authoritative owner surface or one explicit policy accessor",
        "public owner surface delegates member-for-member into a policy family",
        _NOMINAL_IDENTITY_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _INTERFACE_IDENTITY_CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: NominalPolicySurfaceFamilyCandidate
    ) -> RefactorFinding:
        method_summary = ", ".join(
            method.method_name for method in family_candidate.methods[:4]
        )
        selector_summary = ", ".join(family_candidate.selector_source_exprs[:2])
        method_count = len(family_candidate.methods)
        method_phrase = (
            f"surface methods {method_summary}"
            if method_count > 1
            else f"surface method `{family_candidate.methods[0].method_name}`"
        )
        return self.build_finding(
            (
                f"`{family_candidate.owner_class_name}` exposes {method_phrase} by resolving "
                f"`{family_candidate.policy_root_symbol}.{family_candidate.selector_method_name}` from {selector_summary}."
            ),
            family_candidate.evidence,
            scaffold=(
                "class PolicyBackedSurface(ABC):\n    @property\n    @abstractmethod\n    def _policy(self): ...\n\n    def _resolve_policy(self):\n        return self._policy\n\n# Keep one explicit policy accessor and move repeated surface forwarding behind it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.owner_class_name}` surface shells into one explicit policy accessor or owner-owned contract.\n"
                f"# Do not keep separate pass-through methods over `{family_candidate.policy_root_symbol}` for {method_summary}."
            ),
        )


class SemanticDictBagDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic dict bag should become a nominal dataclass",
        "The docs treat semantic field bags as coherence failures: once a dict carries named semantic fields rather than serialization payload, the data should move into a nominal dataclass family with one authoritative schema and explicit inheritance.",
        "single authoritative nominal schema for semantic field bags",
        "same semantic field family is carried through an ad hoc dict bag instead of a nominal record",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_CAPABILITY_TAGS,
        _SEMANTIC_DICT_BAG_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in _semantic_dict_bag_candidates(module):
            recommendation = candidate.recommendation
            key_list = ", ".join(candidate.key_names)
            summary = f"Semantic dict bag with keys {candidate.key_names} appears at {module.path}:{candidate.line}."
            if recommendation.matched_schema_name is not None:
                summary = f"Semantic dict bag with keys {candidate.key_names} should use `{recommendation.class_name}` instead of an untyped dict at {module.path}:{candidate.line}."
            findings.append(
                self.build_finding(
                    summary,
                    (
                        SourceLocation(
                            str(module.path), candidate.line, candidate.symbol
                        ),
                    ),
                    confidence=(
                        HIGH_CONFIDENCE
                        if recommendation.certification == CERTIFIED
                        else MEDIUM_CONFIDENCE
                    ),
                    relation_context=f"same semantic field family is carried through a {candidate.context_kind.replace('_', ' ')} instead of a nominal record",
                    scaffold=f"{recommendation.rationale}\nBase: {recommendation.base_class_name}\nFields: {key_list}\n\n{recommendation.scaffold}",
                    certification=recommendation.certification,
                )
            )
        return findings


class BidirectionalRegistryDetector(ModuleCollectorCandidateDetector):
    candidate_collector = _mirrored_registry_candidates
    finding_spec = finding_spec_template(
        PatternId.BIDIRECTIONAL_LOOKUP,
        "Bidirectional registry maintained manually",
        "The docs prescribe a single authoritative bidirectional type registry when exact companion normalization and reverse lookup matter. Manual mirrored assignments are drift-prone and should be centralized.",
        "exact bijection and O(1) reverse lookup on nominal keys",
        "same class maintains forward and reverse registry state",
        _BIDIRECTIONAL_NORMALIZATION_EXACT_LOOKUP_PROVENANCE_CAPABILITY_TAGS,
        _MIRRORED_REGISTRY_CLASS_LEVEL_POSITION_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        file_path, class_name, mirrored_pairs = cast(
            tuple[str, str, tuple[tuple[int, str], ...]], candidate
        )
        evidence = tuple(
            (
                SourceLocation(file_path, lineno, f"{class_name}.{label}")
                for lineno, label in mirrored_pairs[:6]
            )
        )
        return self.build_finding(
            f"Class {class_name} appears to maintain mirrored forward/reverse registry assignments.",
            evidence,
            observation_tags=_MIRRORED_REGISTRY_CLASS_LEVEL_POSITION_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
            metrics=RegistrationMetrics(
                registration_site_count=len(mirrored_pairs),
                registry_name=class_name,
                class_key_pairs=tuple(
                    (f"{class_name}.{label}" for _, label in mirrored_pairs)
                ),
            ),
        )
