"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
import copy
import hashlib
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, ClassVar, Generic, TypeAlias, TypeVar

from tree_sitter import Node

from ..ast_tools import (
    BuiltinCallName,
    CollectedFamily,
    CompactModuleIdentity,
    ParsedModule,
    PythonSourcePathPolicy,
    SourceModule,
    collect_family_items,
    module_syntax_index,
    walk_function_body_nodes,
)
from ..native_syntax import NativePythonSyntaxIndex
from ..class_index import (
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactExactTypeGuard,
    CompactIndexedClass,
    CompactManualFamilyRosterObservation,
    CompactManualSubclassRosterRoot,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    LatentRosterMatch,
    LatentRosterObservation,
    build_compact_class_family_index,
)
from ..codemod import (
    CancelableCompositionSignal,
    CancelableCompositionSignalTargetAuthority,
)
from ..collection_algebra import sorted_tuple
from ..deadline import scan_deadline_checkpoint
from ..factorization import (
    FactorizationEngine,
    FactorizationLattice,
    FactorizationPlan,
    ResidueHookNamesCarrier,
)
from ..models import HierarchyCandidateMetrics, RefactorFinding, SourceLocation
from ..patterns import PatternId
from ..semantic_algebra import DispatchAxisExpression, ObjectFamilyShape
from ..semantic_description_length import CompressionCertificate
from ..semantic_identity import SemanticRoleIdentityToken
from ..source_index import build_source_index_artifacts
from ..source_index import (
    STABLE_ID_AUTHORITY,
    AstTargetDigest,
    AstTargetNodeKind,
    SourceIndex,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ._base import *
from ._base import high_confidence_certified_spec
from ._helpers import *
from ._helpers import _projection_helper_groups


class _ReplacementShapeRole:
    PROCESS_STAGE_PLAN = object()
    TEXT_REWRITE_PLAN = object()
    BLOCK_ALGEBRA = object()


SemanticBranchObservation: TypeAlias = tuple[int, str, str]
SemanticBranchChain: TypeAlias = tuple[SemanticBranchObservation, ...]
SemanticBranchChains: TypeAlias = tuple[SemanticBranchChain, ...]
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


def _literal_dispatch_authority_name(dispatch_axis_expression: str) -> str:
    words = "".join(
        (
            character if character.isalnum() else "_"
            for character in dispatch_axis_expression
        )
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
    return f"# Replace the repeated `{observation.dispatch_axis_expression} == literal` branches with one AutoRegisterMeta-backed case family.\n# Move per-case behavior into `DispatchCase` subclasses keyed by the same axis.\n# Dispatch through `DispatchCase.for_case(...)` / `DispatchCase.__registry__` instead of if/elif or match/case."


class LiteralDispatchFindingFactory:
    def authority_scaffold(self, observation: LiteralDispatchObservation) -> str:
        dispatch_name = _literal_dispatch_authority_name(
            observation.dispatch_axis_expression
        )
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
            f"{module.path} dispatches on `{observation.dispatch_axis_expression}` through {case_summary_label} {observation.literal_cases}.",
            (
                SourceLocation(
                    observation.file_path, observation.line, observation.symbol
                ),
            ),
            relation_context=(
                f"same observed axis `{observation.dispatch_axis_expression}` is split across {relation_case_label} {observation.literal_cases}"
            ),
            scaffold=self.authority_scaffold(observation),
            codemod_patch=_literal_dispatch_authority_patch(observation),
            metrics=DispatchCountMetrics.from_literal_family(
                observation.dispatch_axis_expression,
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
                (statement.lineno, field_name)
            )
    return fields_by_class


class PrivateObjectBoundaryFieldDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Private object-typed boundary field should become a typed authority",
        "A private dataclass field annotated as `object` and named like an executable/runtime boundary hides both ownership and callable shape. That lets local Python closures cross request boundaries without static evidence.",
        "nominal typed authority or ABC field for each executable/runtime boundary",
        "dataclass request boundary stores a private executable/runtime field as `object`",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
                        "with a named typed authority/ABC field. Do not pass "
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
_FORMULA_LIBRARY_CALLEE_NAMES = frozenset(
    (
        "argmax",
        "argmin",
        "array",
        "asarray",
        "clip",
        "concatenate",
        "count_nonzero",
        "flatnonzero",
        "mean",
        "ones",
        "prod",
        "where",
        "zeros",
    )
)
_FORMULA_CALLEE_NAMES = (
    _FORMULA_LIBRARY_CALLEE_NAMES | BuiltinCallName.formula_builtin_callee_names()
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
    (
        *SemanticRoleIdentityToken.pluralized_string_identifier_values(),
        "field",
        "fields",
        "source",
        "sources",
    )
)


@dataclass(frozen=True)
class FormalBoundaryStringRegistryConstant:
    target_name: str
    value: str
    line: int


def _is_formal_boundary_literal_registry_call(node: ast.Call) -> bool:
    call_name = ast.unparse(node.func).lower()
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
        statements = tuple(
            statement
            for statement in module.module.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
        )
        constants = FormalBoundaryStringRegistryAuthority.constants_from_statements(
            statements
        )
        calls = tuple(
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.Call)
            and _is_formal_boundary_literal_registry_call(node)
        )
        if not FormalBoundaryStringRegistryAuthority.has_formal_boundary_consumer(
            calls,
            constants,
        ):
            return ()
        return constants

    @staticmethod
    def constants_from_statements(
        statements: Sequence[ast.stmt],
    ) -> tuple[FormalBoundaryStringRegistryConstant, ...]:
        constants: list[FormalBoundaryStringRegistryConstant] = []
        for statement in statements:
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

    @staticmethod
    def has_formal_boundary_consumer(
        calls: Sequence[ast.Call],
        constants: tuple[FormalBoundaryStringRegistryConstant, ...],
    ) -> bool:
        constant_names = frozenset(constant.target_name for constant in constants)
        if not constant_names:
            return False
        return any(
            FormalBoundaryStringRegistryAuthority.call_consumes_constant(
                node,
                constant_names,
            )
            for node in calls
        )

    @staticmethod
    def call_consumes_constant(
        node: ast.Call,
        constant_names: frozenset[str],
    ) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in constant_names
            for child in ast.walk(node)
        )


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


@dataclass(frozen=True)
class FormalBoundaryPythonStringConstant:
    file_path: str
    target_name: str
    value: str
    line: int


class FormalBoundaryPythonStringConstantFamily(
    CollectedFamily[FormalBoundaryPythonStringConstant]
):
    """Persist compact Python-side formal-boundary constant facts."""

    item_type = FormalBoundaryPythonStringConstant
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(
        lambda source_module, syntax_index: (
            _native_formal_boundary_python_string_constants(
                source_module,
                syntax_index,
            )
        )
    )

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[FormalBoundaryPythonStringConstant]:
        del cls
        return [
            FormalBoundaryPythonStringConstant(
                file_path=str(parsed_module.path),
                target_name=constant.target_name,
                value=constant.value,
                line=constant.line,
            )
            for constant in FormalBoundaryStringRegistryAuthority.module_constants(
                parsed_module
            )
        ]


def _native_formal_boundary_python_string_constants(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[FormalBoundaryPythonStringConstant] | None:
    """Collect formal-boundary constants from native-selected fragments."""

    if not syntax_index.is_complete:
        return None
    try:
        statements = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
            if (
                (statement_source := syntax_index.source_for(node).decode("utf-8"))
                and any(
                    token in statement_source.lower()
                    for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
                )
                and any(
                    token in statement_source.lower()
                    for token in _FORMAL_BOUNDARY_STRING_ID_TOKENS
                )
            )
        )
        constants = FormalBoundaryStringRegistryAuthority.constants_from_statements(
            statements
        )
        if not constants:
            return []
        constant_names = frozenset(constant.target_name for constant in constants)
        calls: list[ast.Call] = []
        for call_node in sorted(
            syntax_index.common_captures().get("call", ()),
            key=lambda node: (node.start_byte, -node.end_byte),
        ):
            function = call_node.child_by_field_name("function")
            if function is None:
                continue
            function_source = syntax_index.source_for(function).decode("utf-8").lower()
            if not any(
                token in function_source
                for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
            ):
                continue
            expression = syntax_index.expression_for(call_node)
            if not isinstance(expression, ast.Call):
                return None
            if _is_formal_boundary_literal_registry_call(
                expression
            ) and FormalBoundaryStringRegistryAuthority.call_consumes_constant(
                expression,
                constant_names,
            ):
                calls.append(expression)
                break
        if not FormalBoundaryStringRegistryAuthority.has_formal_boundary_consumer(
            calls,
            constants,
        ):
            return []
        return [
            FormalBoundaryPythonStringConstant(
                file_path=str(source_module.path),
                target_name=constant.target_name,
                value=constant.value,
                line=constant.line,
            )
            for constant in constants
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


FormalBoundaryStringConstantRecord: TypeAlias = FormalBoundaryPythonStringConstant
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
    return tuple(
        constant
        for module in modules
        for constant in collect_family_items(
            module,
            FormalBoundaryPythonStringConstantFamily,
        )
    )


def _formal_boundary_python_constants_by_value(
    constants: FormalBoundaryStringConstantRecords,
) -> FormalBoundaryStringConstantsByValue:
    grouped: FormalBoundaryStringConstantsByValue = defaultdict(list)
    for constant in constants:
        grouped[constant.value].append(constant)
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


def _formal_boundary_scan_root_for_paths(file_paths: Sequence[str]) -> Path | None:
    if not file_paths:
        return None
    resolved_paths = tuple(str(Path(path).resolve()) for path in file_paths)
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
        constant = constants_by_value[value][0]
        evidence.append(
            SourceLocation(constant.file_path, constant.line, constant.target_name)
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
        return cls.findings_from_constants(detector, constants)

    @classmethod
    def findings_from_constants(
        cls,
        detector: IssueDetector,
        constants: FormalBoundaryStringConstantRecords,
    ) -> list[RefactorFinding]:
        if len(constants) < _FORMAL_BOUNDARY_LITERAL_REGISTRY_MIN_FIELDS:
            return []
        constants_by_value = _formal_boundary_python_constants_by_value(constants)
        values = tuple(sorted(constants_by_value))
        root = _formal_boundary_scan_root_for_paths(
            tuple(constant.file_path for constant in constants)
        )
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


class FormalBoundaryExternalStringRegistryMirrorDetector(
    CompactModuleProjectionDetectorMixin[FormalBoundaryPythonStringConstant],
    SemanticMirrorIssueDetector,
):
    module_projection_family = FormalBoundaryPythonStringConstantFamily
    compact_report_context_requires_target_projection = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Formal-boundary string registries should not be mirrored across sources",
        "When Python runtime modules and external Lean/formal policy artifacts declare the same proof-relevant string ids, the runtime has a second source of truth. The Python side should consume a generated/typed authority derived from the formal artifact instead of copying the registry values.",
        "formal-boundary string ids have one generated authority shared by runtime and formal artifacts",
        "Python and external formal artifacts mirror the same string-id registry",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[FormalBoundaryPythonStringConstant, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return (
            FormalBoundaryExternalStringRegistryMirrorAuthority.findings_from_constants(
                self,
                projections,
            )
        )


_GENERATED_BOUNDARY_TOKENS = frozenset({"autogen", "autogenerated", "generated"})


@dataclass(frozen=True)
class GeneratedBoundarySemanticConstantSite:
    file_path: str
    target_name: str
    value: str
    line: int
    generated_boundary: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.target_name, self.value

    def source_location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.target_name)


class GeneratedBoundarySemanticConstantAuthority:
    @classmethod
    def findings(
        cls,
        detector: IssueDetector,
        modules: list[ParsedModule],
    ) -> list[RefactorFinding]:
        sites = tuple(
            site
            for module in modules
            for site in collect_family_items(
                module,
                GeneratedBoundarySemanticConstantSiteFamily,
            )
        )
        return cls.findings_from_sites(detector, sites)

    @classmethod
    def findings_from_sites(
        cls,
        detector: IssueDetector,
        sites: tuple[GeneratedBoundarySemanticConstantSite, ...],
    ) -> list[RefactorFinding]:
        keys = tuple(sorted({site.key for site in sites}))
        return [
            finding
            for key in keys
            if (finding := cls.finding_for_key(detector, key, sites)) is not None
        ]

    @classmethod
    def module_sites(
        cls,
        module: ParsedModule,
    ) -> tuple[GeneratedBoundarySemanticConstantSite, ...]:
        generated_boundary = cls.module_is_generated_boundary(module)
        return cls.sites_from_statements(
            str(module.path),
            module.module.body,
            generated_boundary,
        )

    @classmethod
    def sites_from_statements(
        cls,
        file_path: str,
        statements: Sequence[ast.stmt],
        generated_boundary: bool,
    ) -> tuple[GeneratedBoundarySemanticConstantSite, ...]:
        sites: list[GeneratedBoundarySemanticConstantSite] = []
        for statement in statements:
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
                if not isinstance(target, ast.Name):
                    continue
                if not cls.is_semantic_constant_name(target.id):
                    continue
                sites.append(
                    GeneratedBoundarySemanticConstantSite(
                        file_path=file_path,
                        target_name=target.id,
                        value=value,
                        line=statement.lineno,
                        generated_boundary=generated_boundary,
                    )
                )
        return tuple(sites)

    @staticmethod
    def is_semantic_constant_name(name: str) -> bool:
        return name.isupper() and "_" in name

    @staticmethod
    def module_is_generated_boundary(module: ParsedModule) -> bool:
        return GeneratedBoundarySemanticConstantAuthority.source_is_generated_boundary(
            module.module_name,
            module.path,
            module.source,
        )

    @staticmethod
    def source_is_generated_boundary(
        module_name: str,
        path: Path,
        source: str,
    ) -> bool:
        path_tokens = frozenset(
            token
            for part in (*module_name.split("."), path.stem)
            for token in _runtime_semantic_identifier_tokens(part)
        )
        if path_tokens & _GENERATED_BOUNDARY_TOKENS:
            return True
        return any(
            line.lstrip().startswith("#")
            and bool(
                frozenset(_runtime_semantic_identifier_tokens(line))
                & _GENERATED_BOUNDARY_TOKENS
            )
            for line in source.splitlines()[:8]
        )

    @staticmethod
    def finding_for_key(
        detector: IssueDetector,
        key: tuple[str, str],
        sites: tuple[GeneratedBoundarySemanticConstantSite, ...],
    ) -> RefactorFinding | None:
        matching_sites = tuple(site for site in sites if site.key == key)
        generated_sites = tuple(
            site for site in matching_sites if site.generated_boundary
        )
        runtime_sites = tuple(
            site for site in matching_sites if not site.generated_boundary
        )
        if not generated_sites or not runtime_sites:
            return None
        target_name, value = key
        evidence = (
            generated_sites[0].source_location(),
            runtime_sites[0].source_location(),
        )
        return detector.build_finding(
            (
                f"`{target_name}` mirrors generated semantic constant value "
                f"{value!r} across generated and non-generated Python modules."
            ),
            evidence,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(matching_sites),
                field_names=(target_name, value),
                mapping_name="generated_boundary_semantic_constant",
                source_name=target_name,
            ),
            scaffold=(
                "class GeneratedSemanticConstantAuthority:\n"
                "    @classmethod\n"
                "    def value_for(cls, symbolic_name):\n"
                "        return GeneratedConstantCatalog.current().value_for(symbolic_name)"
            ),
            codemod_patch=(
                "# Delete the handwritten runtime copy of this generated semantic "
                "constant and read the value from the generated catalog or nominal "
                "authority. Runtime code should name the symbolic fact, not mirror "
                "the generated value."
            ),
        )


class GeneratedBoundarySemanticConstantSiteFamily(
    CollectedFamily[GeneratedBoundarySemanticConstantSite]
):
    """Persist compact module facts used by the generated-boundary detector."""

    item_type = GeneratedBoundarySemanticConstantSite
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(
        lambda source_module, syntax_index: (
            _native_generated_boundary_semantic_constant_sites(
                source_module,
                syntax_index,
            )
        )
    )

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[GeneratedBoundarySemanticConstantSite]:
        del cls
        return list(
            GeneratedBoundarySemanticConstantAuthority.module_sites(parsed_module)
        )


def _native_generated_boundary_semantic_constant_sites(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[GeneratedBoundarySemanticConstantSite] | None:
    """Collect top-level semantic constants from native-selected assignments."""

    if not syntax_index.is_complete:
        return None
    try:
        statements = tuple(
            syntax_index.statement_for(node)
            for node in syntax_index.top_level_assignment_statements()
            if _native_assignment_may_declare_semantic_constant(
                syntax_index,
                node,
            )
        )
        generated_boundary = (
            GeneratedBoundarySemanticConstantAuthority.source_is_generated_boundary(
                source_module.module_name,
                source_module.path,
                source_module.source,
            )
        )
        return list(
            GeneratedBoundarySemanticConstantAuthority.sites_from_statements(
                str(source_module.path),
                statements,
                generated_boundary,
            )
        )
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _native_assignment_may_declare_semantic_constant(
    syntax_index: NativePythonSyntaxIndex,
    statement: Node,
) -> bool:
    """Cheap necessary filter for uppercase named assignment targets."""

    pending = list(statement.named_children)
    while pending:
        node = pending.pop()
        if node.type == "assignment":
            target = node.child_by_field_name("left")
            if target is not None and target.type == "identifier":
                name = syntax_index.source_for(target).decode("utf-8")
                if GeneratedBoundarySemanticConstantAuthority.is_semantic_constant_name(
                    name
                ):
                    return True
        pending.extend(node.named_children)
    return False


class GeneratedBoundarySemanticConstantMirrorDetector(
    CompactModuleProjectionDetectorMixin[GeneratedBoundarySemanticConstantSite],
    SemanticMirrorIssueDetector,
):
    module_projection_family = GeneratedBoundarySemanticConstantSiteFamily
    compact_report_context_requires_target_projection = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Generated semantic constants should not be mirrored in runtime code",
        "When generated and non-generated Python modules declare the same uppercase semantic constant with the same value, the generated artifact is no longer the only source of truth. Runtime code should depend on the generated nominal authority or catalog instead of copying the generated fact.",
        "generated semantic constants are read from one generated authority",
        "same semantic constant name and value is declared on both sides of the generated/runtime boundary",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[GeneratedBoundarySemanticConstantSite, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        return GeneratedBoundarySemanticConstantAuthority.findings_from_sites(
            self,
            projections,
        )


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
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Runtime namespace bridges should be replaced with explicit authorities",
        "Copying another module namespace into globals, or conditionally defining names only when globals lacks them, creates a hidden compatibility layer. Split modules should import their dependencies explicitly and publish one authoritative public surface so missing names fail loudly.",
        "explicit import/authority boundary with no globals namespace copying",
        "module mutates globals or guards definitions through a namespace bridge",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
        *SemanticRoleIdentityToken.runtime_semantic_branch_axis_values(),
        "action",
        "basis",
        "budget",
        "certified",
        "frontier",
        "formal",
        "materialization",
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


def _runtime_semantic_axis_is_interesting(dispatch_axis_expression: str) -> bool:
    tokens = set(_runtime_semantic_identifier_tokens(dispatch_axis_expression))
    return bool(tokens & _RUNTIME_SEMANTIC_BRANCH_AXIS_TOKENS)


def _stable_text_digest(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()


declare_candidate_rule_detector(
    StringKeyedFormulaSubclassFamilyCandidate,
    high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "String-keyed formula subclasses should be derived from a typed policy algebra",
        "A subclass family that assigns string `kind`/`mode` keys and implements formulas on the subclasses is a split semantic authority: the key registry owns case identity while method bodies own case semantics. The formulas should be represented by a typed/generated policy algebra or nominal proof-backed carrier so runtime code interprets one schema instead of maintaining per-string behavior.",
        "typed/generated policy algebra owns case formulas with fail-loud validation",
        "subclasses repeat formula semantics behind literal kind/mode keys",
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        (
            ObservationTag.STRING_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
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


_RELATION_COMPARISON_AXIS_TOKENS = frozenset(
    (
        *SemanticRoleIdentityToken.relation_comparison_axis_values(),
        "certificate",
        "certified",
        "case",
        "count",
        "index",
        "length",
        "original",
        "previous",
        "rank",
        "relation",
        "schema",
        "shape",
        "signature",
        "size",
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
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Load-bearing relation dispatch should be a nominal case family",
        "An Authority method that chooses certificate, summary, or projection outputs through ordered relation/count/domain branches is encoding proof-relevant semantics in branch order. The relation cases should be named nominal classes with exactly-one-case selection.",
        "nominal relation-case algebra owns certificate/domain dispatch",
        "Authority method branches over proof-relevant source-domain relations",
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        (
            ObservationTag.STRING_DISPATCH,
            ObservationTag.CLOSED_FAMILY_CASES,
        ),
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
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic mismatch guards should produce typed certificates",
        "A runtime Authority that compares proof-relevant signatures, certificates, or domain counts and returns an existing object on mismatch is a hidden fallback. The semantic relation should be represented as a typed certificate whose construction either succeeds through a formal rule or fails loudly.",
        "typed formal certificate owns semantic reuse and mismatch behavior",
        "proof-relevant mismatch guard returns a runtime fallback object",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
    ssot_authority_boundary = True
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Mirrored constructor validators should move into the record schema",
        "A constructor call that fills several fields by calling validators with a string literal copy of the source variable keeps field identity in multiple places. The schema/record field declaration should own the source name and materializer once.",
        "single authoritative record-field schema with source and validator metadata",
        "constructor keyword fields mirror validation source names at the callsite",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
                                    node.lineno,
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


_MONOLITHIC_CONSTRUCTOR_METHOD_NAMES = frozenset(("__init__", "__post_init__"))
_MONOLITHIC_CONSTRUCTOR_MIN_PREDICATES = 8
_MONOLITHIC_CONSTRUCTOR_MIN_FIELDS = 3
_MONOLITHIC_CONSTRUCTOR_MIN_INVARIANT_KINDS = 4
_INVARIANT_NORMALIZATION_METHOD_NAMES = frozenset(
    ("absolute", "canonicalize", "lower", "normalize", "resolve", "upper")
)


def _flatten_boolean_operator(
    node: ast.AST,
    operator_type: type[ast.boolop],
) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, operator_type):
        return tuple(
            predicate
            for value in node.values
            for predicate in _flatten_boolean_operator(value, operator_type)
        )
    return (node,)


def _failed_constructor_invariant_predicates(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _flatten_boolean_operator(node, ast.Or)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.BoolOp)
        and isinstance(node.operand.op, ast.And)
    ):
        return _flatten_boolean_operator(node.operand, ast.And)
    return ()


def _call_terminal_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _self_attribute_names(node: ast.AST) -> frozenset[str]:
    return frozenset(
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
    )


def _value_reference_names(node: ast.AST) -> frozenset[str]:
    if isinstance(node, ast.Name):
        return (
            frozenset()
            if node.id in BuiltinCallName.non_value_reference_names()
            else frozenset((node.id,))
        )
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return frozenset((f"self.{node.attr}",))
        return _value_reference_names(node.value)
    if isinstance(node, ast.Call):
        return frozenset(
            reference
            for argument in (*tuple(node.args), *tuple(node.keywords))
            for reference in _value_reference_names(
                argument.value if isinstance(argument, ast.keyword) else argument
            )
        )
    if isinstance(node, ast.Constant):
        return frozenset()
    return frozenset(
        reference
        for child in ast.iter_child_nodes(node)
        for reference in _value_reference_names(child)
    )


def _compare_has_distinct_value_authorities(node: ast.Compare) -> bool:
    operands = (node.left, *tuple(node.comparators))
    reference_sets = tuple(_value_reference_names(operand) for operand in operands)
    return any(
        left_references and right_references and left_references != right_references
        for left_references, right_references in zip(reference_sets, reference_sets[1:])
    )


def _constructor_invariant_kinds(
    predicates: tuple[ast.AST, ...],
) -> frozenset[str]:
    kinds: set[str] = set()
    for predicate in predicates:
        descendants = tuple(ast.walk(predicate))
        call_names = frozenset(
            call_name
            for descendant in descendants
            if isinstance(descendant, ast.Call)
            for call_name in (_call_terminal_name(descendant),)
            if call_name is not None
        )
        if call_names & BuiltinCallName.invariant_refinement_call_names():
            kinds.add("runtime representation")
        if (
            call_names & BuiltinCallName.invariant_cardinality_call_names()
            or isinstance(predicate, ast.UnaryOp)
        ):
            kinds.add("cardinality")
        if call_names & BuiltinCallName.invariant_quantifier_call_names():
            kinds.add("quantified members")
        if call_names & _INVARIANT_NORMALIZATION_METHOD_NAMES:
            kinds.add("normalization")
        if any(
            isinstance(
                descendant, (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
            )
            for descendant in descendants
        ):
            kinds.add("derived projection")
        if any(
            isinstance(descendant, ast.Compare)
            and _compare_has_distinct_value_authorities(descendant)
            for descendant in descendants
        ):
            kinds.add("cross-value relation")
    return frozenset(kinds)


def _single_fail_loud_raise(body: Sequence[ast.stmt]) -> ast.Raise | None:
    return body[0] if len(body) == 1 and isinstance(body[0], ast.Raise) else None


class MonolithicConstructorInvariantDetector(PerModuleIssueDetector):
    ssot_authority_boundary = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Monolithic constructor invariant should move into validated nominal values",
        "One constructor failure guard combines representation, normalization, collection, and relational rules across several fields. The record now owns several independent refinement authorities behind one anonymous Boolean and one failure path, making those contracts difficult to reuse, type-check, or diagnose independently.",
        "field-owned validated nominal values with only cross-field residue on the aggregate record",
        "one fail-loud constructor guard combines many heterogeneous predicates across several record fields",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
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
                if (
                    self.current_class_name is None
                    or self.current_function_name
                    not in _MONOLITHIC_CONSTRUCTOR_METHOD_NAMES
                    or _single_fail_loud_raise(node.body) is None
                ):
                    self.generic_visit(node)
                    return
                predicates = _failed_constructor_invariant_predicates(node.test)
                field_names = _self_attribute_names(node.test)
                invariant_kinds = _constructor_invariant_kinds(predicates)
                if (
                    len(predicates) < _MONOLITHIC_CONSTRUCTOR_MIN_PREDICATES
                    or len(field_names) < _MONOLITHIC_CONSTRUCTOR_MIN_FIELDS
                    or len(invariant_kinds)
                    < _MONOLITHIC_CONSTRUCTOR_MIN_INVARIANT_KINDS
                ):
                    self.generic_visit(node)
                    return
                qualname = self.qualname
                field_summary = ", ".join(sorted(field_names))
                kind_summary = ", ".join(sorted(invariant_kinds))
                findings.append(
                    detector.build_finding(
                        (
                            f"`{qualname}` merges {len(predicates)} failed predicates "
                            f"across fields {field_summary} and invariant kinds "
                            f"{kind_summary} into one constructor failure."
                        ),
                        (
                            SourceLocation(
                                str(module.path),
                                node.lineno,
                                qualname,
                            ),
                        ),
                        scaffold=(
                            "@dataclass(frozen=True)\n"
                            "class ValidatedField:\n"
                            "    value: object\n\n"
                            "    @classmethod\n"
                            "    def parse(cls, value):\n"
                            "        # Own representation and local invariants here.\n"
                            "        ...\n\n"
                            "@dataclass(frozen=True)\n"
                            "class AggregateRecord:\n"
                            "    field: ValidatedField\n\n"
                            "    def __post_init__(self):\n"
                            "        # Keep only aggregate cross-field invariants here.\n"
                            "        ..."
                        ),
                        codemod_patch=(
                            f"# Split the heterogeneous guard in `{qualname}` by semantic authority.\n"
                            "# Move representation, normalization, and member rules into validated field types; retain only irreducible cross-field relations on the aggregate.\n"
                            "# Give each remaining failure a typed or specific diagnostic instead of one anonymous Boolean failure."
                        ),
                    )
                )
                self.generic_visit(node)

        Visitor().visit(module.module)
        return findings


def _native_call_terminal_name(
    syntax_index: NativePythonSyntaxIndex,
    call: Node,
) -> str | None:
    function = call.child_by_field_name("function")
    if function is None:
        return None
    if function.type == "identifier":
        return syntax_index.source_for(function).decode("utf-8")
    if function.type == "attribute":
        attribute = function.child_by_field_name("attribute")
        if attribute is not None:
            return syntax_index.source_for(attribute).decode("utf-8")
    return None


def _native_call_may_be_builder(
    syntax_index: NativePythonSyntaxIndex,
    call: Node,
) -> bool:
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return False
    if any(child.type == "keyword_argument" for child in arguments.named_children):
        return True
    terminal_name = _native_call_terminal_name(syntax_index, call)
    return bool(
        arguments.named_children
        and terminal_name is not None
        and terminal_name.startswith(("for_", "from_", "with_"))
    )


def _native_repeated_builder_call_shapes(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[BuilderCallShape] | None:
    """Project normalized builder calls from native-selected expressions."""

    if not syntax_index.is_complete:
        return None
    try:
        captures = syntax_index.common_captures()
        module_class_names = frozenset(
            syntax_index.declared_name(node) for node in captures.get("class", ())
        )
        functions = tuple(
            sorted(
                captures.get("function", ()),
                key=lambda node: (node.start_byte, -node.end_byte),
            )
        )
        calls_by_function: dict[Node, list[Node]] = defaultdict(list)
        for call in captures.get("call", ()):
            if not _native_call_may_be_builder(syntax_index, call):
                continue
            scopes = syntax_index.named_scope_nodes(call)
            if not scopes or scopes[-1].type != "function_definition":
                continue
            function = scopes[-1]
            body = function.child_by_field_name("body")
            if body is None or not (
                body.start_byte <= call.start_byte and call.end_byte <= body.end_byte
            ):
                continue
            if any(
                ancestor.type == "decorator"
                for ancestor in _native_ancestors_until(call, function)
            ):
                continue
            calls_by_function[function].append(call)

        parsed_module = source_module.parsed_module(
            ast.Module(body=[], type_ignores=[]),
        )
        shapes: list[BuilderCallShape] = []
        for function in functions:
            class_name = (
                ".".join(
                    syntax_index.declared_name(scope)
                    for scope in syntax_index.named_scope_nodes(function)
                    if scope.type == "class_definition"
                )
                or None
            )
            function_name = syntax_index.declared_name(function)
            for call in sorted(
                calls_by_function.get(function, ()),
                key=lambda node: (node.start_byte, -node.end_byte),
            ):
                expression = syntax_index.expression_for(call)
                shape = _builder_call_shape(
                    parsed_module,
                    expression,
                    class_name,
                    function_name,
                    module_class_names,
                )
                if shape is not None:
                    shapes.append(shape)
        return sorted(shapes, key=_builder_call_projection_sort_key)
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _native_ancestors_until(node: Node, boundary: Node) -> tuple[Node, ...]:
    ancestors: list[Node] = []
    current = node.parent
    while current is not None and current != boundary:
        ancestors.append(current)
        current = current.parent
    return tuple(ancestors)


def _builder_call_projection_sort_key(
    shape: BuilderCallShape,
) -> tuple[str, int, str, tuple[str, ...], tuple[str, ...]]:
    """Canonical family order independent of parser traversal details."""

    return (
        shape.file_path,
        shape.lineno,
        shape.symbol,
        shape.field_names,
        shape.value_fingerprint,
    )


@dataclass(frozen=True)
class RepeatedBuilderCallProjectionDemand:
    """Group keys capable of producing a finding with report-target evidence."""

    exact_mapping_keys: frozenset[tuple[str, str, tuple[str, ...], tuple[str, ...]]]
    owner_family_keys: frozenset[tuple[str, str, str]]


def _repeated_builder_call_projection_demand(
    target_items: tuple[object, ...],
    config: object,
) -> RepeatedBuilderCallProjectionDemand:
    del config
    target_builders = tuple(
        item for item in target_items if isinstance(item, BuilderCallShape)
    )
    return RepeatedBuilderCallProjectionDemand(
        exact_mapping_keys=frozenset(
            (
                builder.file_path,
                builder.callee_name,
                builder.field_names,
                builder.value_fingerprint,
            )
            for builder in target_builders
        ),
        owner_family_keys=frozenset(
            (builder.file_path, builder.owner_prefix, builder.callee_name)
            for builder in target_builders
        ),
    )


def _repeated_builder_call_is_demanded(
    builder: BuilderCallShape,
    demand: RepeatedBuilderCallProjectionDemand,
) -> bool:
    return bool(
        (
            builder.file_path,
            builder.callee_name,
            builder.field_names,
            builder.value_fingerprint,
        )
        in demand.exact_mapping_keys
        or (builder.file_path, builder.owner_prefix, builder.callee_name)
        in demand.owner_family_keys
    )


def _project_repeated_builder_call_demand(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, RepeatedBuilderCallProjectionDemand):
        return items
    return tuple(
        item
        for item in items
        if isinstance(item, BuilderCallShape)
        and _repeated_builder_call_is_demanded(item, demand)
    )


def _collect_repeated_builder_call_ast_demand(
    module: ParsedModule,
    demand: object,
) -> list[object]:
    if not isinstance(demand, RepeatedBuilderCallProjectionDemand):
        raise TypeError("repeated-builder demand has the wrong authority type")
    callee_names = frozenset(
        callee_name
        for _file_path, callee_name, *_remainder in demand.exact_mapping_keys
    ) | frozenset(
        callee_name for _file_path, _owner_name, callee_name in demand.owner_family_keys
    )
    return list(
        _project_repeated_builder_call_demand(
            tuple(_module_builder_call_shapes(module, callee_names)),
            demand,
        )
    )


def _collect_repeated_builder_call_source_demand(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    builders = _native_repeated_builder_call_shapes(source_module, syntax_index)
    if builders is None:
        return None
    return list(_project_repeated_builder_call_demand(tuple(builders), demand))


class RepeatedBuilderCallShapeProjectionFamily(CollectedFamily[BuilderCallShape]):
    """Persist normalized builder calls for repository-wide grouping."""

    item_type = BuilderCallShape
    cache_payload_max_bytes = 3_000_000
    source_collector = staticmethod(_native_repeated_builder_call_shapes)
    source_demand_collector = staticmethod(_collect_repeated_builder_call_source_demand)
    ast_demand_collector = staticmethod(_collect_repeated_builder_call_ast_demand)
    report_demand_builder = staticmethod(_repeated_builder_call_projection_demand)
    cached_demand_projector = staticmethod(_project_repeated_builder_call_demand)

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[BuilderCallShape]:
        del cls
        return sorted(
            _module_builder_call_shapes(parsed_module),
            key=_builder_call_projection_sort_key,
        )


class RepeatedBuilderCallDetector(
    CompactModuleProjectionDetectorMixin[BuilderCallShape],
    FlattenedModuleCollectorCandidateDetector[BuilderCallShape],
):
    module_projection_family = RepeatedBuilderCallShapeProjectionFamily
    detector_id = "repeated_builder_calls"
    ssot_authority_boundary = True
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated field assignment should become an authoritative builder",
        "The docs say repeated manual field assignment is an SSOT violation: the mapping should be declared once in an authoritative constructor, classmethod, or shared builder rather than copied across call sites.",
        "single authoritative record-builder mapping for a repeated constructor family",
        "same builder role repeated across sibling functions or methods",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    candidate_collector = staticmethod(_module_builder_call_shapes)
    candidate_sort_key = staticmethod(
        lambda item: (item.file_path, item.lineno, item.symbol)
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[BuilderCallShape, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            type(self)._sorted_candidate_items(projections),
            config,
        )

    def _findings_for_candidates(
        self,
        candidates: Sequence[BuilderCallShape],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        builders = tuple(candidates)
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
            (
                tuple[str, str, tuple[str, ...], tuple[str, ...]],
                list[BuilderCallShape],
            )
        ] = defaultdict(list)
        for builder in builders:
            if len(builder.field_names) < config.min_builder_keywords:
                continue
            grouped[
                builder.file_path,
                builder.callee_name,
                builder.field_names,
                builder.value_fingerprint,
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
        grouped: dict[tuple[str, str, str], list[BuilderCallShape]] = defaultdict(list)
        for builder in builders:
            if not builder.field_names:
                continue
            grouped[
                (builder.file_path, builder.owner_prefix, builder.callee_name)
            ].append(builder)
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
            _file_path, owner_symbol, callee_name = owner_key
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


def _declared_field_extraction_sites(
    module: ParsedModule,
) -> tuple[DeclaredFieldExtractionSite, ...]:
    sites: list[DeclaredFieldExtractionSite] = []

    class Visitor(ClassFunctionStackNodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            site = _declared_field_extraction_site(
                module,
                node,
                self.qualname,
                len(sites),
            )
            if site is not None:
                sites.append(site)
            self.generic_visit(node)

    Visitor().visit(module.module)
    return tuple(sites)


def _native_declared_field_extraction_sites(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[DeclaredFieldExtractionSite] | None:
    """Derive declared-field call sites from the shared native call stream."""

    if not syntax_index.is_complete:
        return None
    parsed_module = source_module.parsed_module(
        ast.Module(body=[], type_ignores=[]),
    )
    sites: list[DeclaredFieldExtractionSite] = []
    try:
        calls = sorted(
            syntax_index.common_captures().get("call", ()),
            key=lambda node: (node.start_byte, -node.end_byte),
        )
        for call in calls:
            callee_name = _native_call_terminal_name(syntax_index, call)
            if callee_name is None:
                continue
            tokens = frozenset(_runtime_semantic_identifier_tokens(callee_name))
            if not _DECLARED_FIELD_EXTRACTION_REQUIRED_TOKENS <= tokens or not (
                tokens & _DECLARED_FIELD_EXTRACTION_PAYLOAD_TOKENS
            ):
                continue
            expression = syntax_index.expression_for(call)
            if not isinstance(expression, ast.Call):
                return None
            scopes = syntax_index.named_scope_nodes(call)
            owner_symbol = (
                ".".join(
                    (
                        *(
                            syntax_index.declared_name(scope)
                            for scope in scopes
                            if scope.type == "class_definition"
                        ),
                        *(
                            syntax_index.declared_name(scope)
                            for scope in scopes
                            if scope.type == "function_definition"
                        ),
                    )
                )
                or "<module>"
            )
            site = _declared_field_extraction_site(
                parsed_module,
                expression,
                owner_symbol,
                len(sites),
            )
            if site is not None:
                sites.append(site)
        return sites
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


class DeclaredFieldExtractionSiteFamily(CollectedFamily[DeclaredFieldExtractionSite]):
    """Persist declared-field extraction facts for global factorization."""

    item_type = DeclaredFieldExtractionSite
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(_native_declared_field_extraction_sites)

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[DeclaredFieldExtractionSite]:
        del cls
        return list(_declared_field_extraction_sites(parsed_module))


class DeclaredFieldExtractionFanoutDetector(
    CompactModuleProjectionDetectorMixin[DeclaredFieldExtractionSite],
    FlattenedModuleCollectorCandidateDetector[DeclaredFieldExtractionSite],
):
    module_projection_family = DeclaredFieldExtractionSiteFamily
    compact_report_context_requires_target_projection = True
    ssot_authority_boundary = True
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
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    candidate_collector = staticmethod(_declared_field_extraction_sites)
    candidate_sort_key = staticmethod(
        lambda item: (item.file_path, item.lineno, item.owner_symbol)
    )

    def _findings_from_compact_projections(
        self,
        projections: tuple[DeclaredFieldExtractionSite, ...],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        return self._findings_for_candidates(
            type(self)._sorted_candidate_items(projections),
            config,
        )

    def _findings_for_candidates(
        self,
        candidates: Sequence[DeclaredFieldExtractionSite],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        sites = tuple(candidates)
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


class RepeatedExportDictDetector(
    CompactFiberCollectedShapeIssueDetector[
        ExportDictShape,
        tuple[tuple[str, ...], str],
    ]
):
    module_projection_family = ExportDictShapeFamily
    compact_report_context_requires_target_projection = True
    detector_id = "repeated_export_dicts"
    ssot_authority_boundary = True
    observation_kind = ObservationKind.EXPORT_DICT
    finding_spec = certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated projection dict should become an authoritative schema",
        "The docs say repeated JSON/CSV/export dicts and kwargs/source-value bags should become one authoritative row schema or projection builder instead of many hand-maintained dict literals.",
        "single authoritative projection schema for a repeated record or kwargs family",
        "same string-key projection role repeated across sibling functions or methods",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.PROJECTION_DICT,
            ObservationTag.EXPORT_MAPPING,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[ExportDictShape, ...]:
        return tuple(
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, ExportDictShapeFamily, ExportDictShape
            )
        )

    def _include_shape(self, shape: ExportDictShape, config: DetectorConfig) -> bool:
        return len(shape.key_names) >= config.min_export_keys

    def _group_key(self, shape: ExportDictShape) -> tuple[tuple[str, ...], str]:
        return (shape.key_names, shape.value_fingerprint)

    def _finding_from_group(
        self, shapes: tuple[ExportDictShape, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        export_shapes = sorted_tuple(
            shapes,
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


class ManualClassRegistrationDetector(
    CompactGroupedShapeIssueDetector[RegistrationShape, str]
):
    module_projection_family = RegistrationShapeFamily
    compact_report_context_requires_target_projection = True
    finding_spec = certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual class registration should become metaclass-registry AutoRegisterMeta",
        "The docs say repeated class-level registration boilerplate is a class-level non-orthogonal algorithm. It should move into one authoritative `metaclass-registry` base so abstract-class skipping, uniqueness, and inheritance behavior are enforced in one place.",
        "single authoritative metaclass-registry class-registration algorithm with nominal class identity",
        "same registry key family repeated through manual class-level registration assignments",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_LEVEL_POSITION,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _collect_shapes(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RegistrationShape]:
        return [
            shape
            for module in modules
            for shape in CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, RegistrationShapeFamily, RegistrationShape
            )
        ]

    def _group_key(self, shape: RegistrationShape) -> str:
        return shape.registry_name

    def _finding_from_group(
        self, shapes: tuple[RegistrationShape, ...], config: DetectorConfig
    ) -> RefactorFinding | None:
        registrations = sorted_tuple(
            shapes,
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


def _target_has_manual_subclass_roster_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """A roster finding reports the root that owns the roster, not its leaves."""

    del config
    return any(
        projection.manual_subclass_roster_roots
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


def _target_has_latent_roster(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """A latent-roster finding is anchored at the roster declaration itself."""

    del config
    return any(
        projection.latent_rosters
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
    )


def _target_has_autoregister_meta_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Under-rent evidence is always the class that declares AutoRegisterMeta."""

    del config
    return any(
        indexed_class.declares_autoregister_meta
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
        for indexed_class in projection.classes
    )


def _target_has_predicate_selected_root(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
    config: DetectorConfig,
) -> bool:
    """Predicate-family evidence is located on its selector root."""

    del config
    return any(
        indexed_class.predicate_selected_methods
        and "_registered_types" in indexed_class.assignments_by_name
        for projection in projections_by_family.get(
            CompactModuleClassProjectionFamily, ()
        )
        if isinstance(projection, CompactModuleClassProjection)
        for indexed_class in projection.classes
    )


@dataclass(frozen=True)
class _CompactConcreteFamilyContext:
    class_index: CompactClassFamilyIndex
    class_reference_resolver: CompactClassReferenceResolver
    module_name_by_file_path: tuple[tuple[str, str], ...]
    manual_subclass_roster_roots: tuple[CompactManualSubclassRosterRoot, ...]
    manual_family_rosters: tuple[CompactManualFamilyRosterObservation, ...]
    latent_rosters: tuple[LatentRosterObservation, ...]


def _compact_concrete_family_context(
    projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> _CompactConcreteFamilyContext:
    del config
    if class_index is None:
        class_index = build_compact_class_family_index(projections)
    return _CompactConcreteFamilyContext(
        class_index=class_index,
        class_reference_resolver=CompactClassReferenceResolver.from_index(
            projections,
            class_index,
        ),
        module_name_by_file_path=tuple(
            (projection.file_path, projection.module_name) for projection in projections
        ),
        manual_subclass_roster_roots=tuple(
            root
            for projection in projections
            for root in projection.manual_subclass_roster_roots
        ),
        manual_family_rosters=tuple(
            roster
            for projection in projections
            for roster in projection.manual_family_rosters
        ),
        latent_rosters=tuple(
            roster for projection in projections for roster in projection.latent_rosters
        ),
    )


def _compact_concrete_family_context_from_repository(
    context: object | None,
) -> _CompactConcreteFamilyContext:
    if isinstance(context, _CompactConcreteFamilyContext):
        return context
    repository = CompactClassRepositoryContext.require(context)
    return repository.cached(
        _compact_concrete_family_context,
        lambda: _compact_concrete_family_context(
            repository.projections,
            repository.config,
            class_index=repository.class_index,
        ),
    )


CompactConcreteFamilyCandidateT = TypeVar("CompactConcreteFamilyCandidateT")


class _CompactConcreteFamilyDetectorBase(
    CompactContextCandidateDetector[
        CompactModuleClassProjection,
        _CompactConcreteFamilyContext,
        CompactConcreteFamilyCandidateT,
    ],
    Generic[CompactConcreteFamilyCandidateT],
):
    module_projection_family = CompactModuleClassProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactClassRepositoryContext.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        config: DetectorConfig,
    ) -> _CompactConcreteFamilyContext:
        return _compact_concrete_family_context(projections, config)

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> _CompactConcreteFamilyContext:
        return _compact_concrete_family_context_from_repository(context)


class ManualConcreteSubclassRosterDetector(
    _CompactConcreteFamilyDetectorBase[ManualConcreteSubclassRosterCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_manual_subclass_roster_root
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual concrete-subclass roster should become a metaclass-registry base",
        "The docs treat mutable subclass rosters maintained through __init_subclass__ as framework logic. Abstract filtering, subclass discovery, and family access should live in one reusable `metaclass-registry` base instead of being reimplemented inside each domain family.",
        "single authoritative metaclass-registry concrete-subclass registration hook with reusable family discovery",
        "class family maintains a mutable subclass roster through __init_subclass__ and then queries it manually",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_REGISTRATION,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: _CompactConcreteFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[ManualConcreteSubclassRosterCandidate]:
        return _compact_manual_concrete_subclass_roster_candidates(context, config)

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
    _CompactConcreteFamilyDetectorBase[LatentImplementationRosterCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(_target_has_latent_roster)
    finding_spec = high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Manual implementation enumeration should derive from the ABC registry",
        "A collection or inline literal whose members mirror concrete implementations of one ABC family is a shadow registry even when it is just strings, class objects, instances, or a dict passed to `update(...)`. Membership should be derived from an AutoRegisterMeta-backed ABC or from a named projection policy over that registry.",
        "AutoRegisterMeta-backed implementation registry with generated projection surfaces",
        "manual collection or inline literal repeats the complete concrete implementation set of an ABC family",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.MANUAL_SYNCHRONIZATION,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: _CompactConcreteFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[LatentImplementationRosterCandidate]:
        return _compact_latent_implementation_roster_candidates(
            context,
            config,
        )

    def _finding_for_candidate(
        self, roster_candidate: LatentImplementationRosterCandidate
    ) -> RefactorFinding:
        roster = roster_candidate.roster
        match = roster_candidate.match
        key_attr = roster_candidate.key_attr_name or "derived_registry_key"
        projection_suffix = (
            f" with subset policy `{match.projection_policy_hint}`; "
            f"missing {match.missing_member_names}"
            if match.projection_policy_hint is not None
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
                f"`{roster.roster_name}` is a `{roster.roster_kind}` roster "
                f"{roster.member_names} via `{roster.projection_role}` "
                f"covering {match.coverage_ratio:.2f} of concrete `{roster_candidate.class_name}` "
                f"implementations {roster_candidate.concrete_class_names}; derive it from registry key `{key_attr}`"
                f"{projection_suffix}."
            ),
            (roster_candidate.evidence,),
            scaffold=(
                "from abc import ABC\n"
                "from metaclass_registry import AutoRegisterMeta\n\n"
                f"class {roster_candidate.class_name}(ABC, metaclass=AutoRegisterMeta):\n"
                f"{registry_block}\n\n"
                f"{roster.roster_name} = {projection_expression}"
            ),
            codemod_patch=(
                f"# Delete manual roster `{roster.roster_name}`.\n"
                f"# Promote `{roster_candidate.class_name}` to `ABC, metaclass=AutoRegisterMeta` and derive this projection from `__registry__`"
                + (
                    f" through a named `{match.projection_policy_hint}` subset policy."
                    if match.projection_policy_hint is not None
                    else "."
                )
            ),
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(roster_candidate.concrete_class_names),
                registry_name=roster.roster_name,
                class_names=roster_candidate.concrete_class_names,
            ),
        )


def _compact_class_display_name(
    indexed_class: CompactIndexedClass,
    class_index: CompactClassFamilyIndex,
) -> str:
    if len(class_index.symbols_by_simple_name.get(indexed_class.simple_name, ())) <= 1:
        return indexed_class.simple_name
    return indexed_class.symbol


def _compact_concrete_descendants(
    class_index: CompactClassFamilyIndex,
    indexed_class: CompactIndexedClass,
) -> tuple[CompactIndexedClass, ...]:
    return tuple(
        descendant
        for symbol in class_index.descendant_symbols(indexed_class.symbol)
        if (descendant := class_index.class_for(symbol)) is not None
        if not descendant.is_abstract
    )


def _compact_manual_concrete_subclass_roster_candidates(
    context: _CompactConcreteFamilyContext,
    config: DetectorConfig,
) -> tuple[ManualConcreteSubclassRosterCandidate, ...]:
    class_index = context.class_index
    roots_by_symbol: dict[str, CompactManualSubclassRosterRoot] = {
        root.class_symbol: root for root in context.manual_subclass_roster_roots
    }
    candidates: list[ManualConcreteSubclassRosterCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        root = roots_by_symbol.get(indexed_class.symbol)
        if root is None:
            continue
        descendants = tuple(
            descendant
            for symbol in class_index.descendant_symbols(indexed_class.symbol)
            if (descendant := class_index.class_for(symbol)) is not None
        )
        if len(descendants) < config.min_registration_sites:
            continue
        for compact_site in root.registration_sites:
            consumers = tuple(
                SourceLocation(file_path, line, symbol)
                for registry_name, line, symbol, file_path in root.consumer_locations
                if registry_name == compact_site.registry_name
            )
            if not consumers:
                continue
            if compact_site.selector_attr_name is not None:
                registered_descendants = tuple(
                    descendant
                    for descendant in descendants
                    if compact_site.selector_attr_name
                    in descendant.direct_non_none_assignment_names
                )
            elif compact_site.requires_concrete_subclass:
                registered_descendants = tuple(
                    descendant
                    for descendant in descendants
                    if not descendant.is_abstract
                )
            else:
                registered_descendants = descendants
            if len(registered_descendants) < config.min_registration_sites:
                continue
            candidates.append(
                ManualConcreteSubclassRosterCandidate(
                    file_path=indexed_class.file_path,
                    line=root.init_subclass_line,
                    class_name=_compact_class_display_name(indexed_class, class_index),
                    registration_site=compact_site,
                    consumer_locations=consumers,
                    concrete_class_names=sorted_tuple(
                        _compact_class_display_name(descendant, class_index)
                        for descendant in registered_descendants
                    ),
                )
            )
    return tuple(candidates)


def _compact_descendant_key_values(
    descendants: tuple[CompactIndexedClass, ...], key_attr_name: str
) -> tuple[str, ...]:
    return sorted_tuple(
        {
            value
            for descendant in descendants
            for name, value in descendant.direct_constant_string_assignments
            if name == key_attr_name
        }
    )


def _compact_matched_latent_roster_key_attr(
    roster: LatentRosterObservation,
    key_values_by_attr: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, LatentRosterMatch] | None:
    for key_attr_name, key_values in key_values_by_attr:
        match = roster.match(key_values)
        if match is not None:
            return key_attr_name, match
    return None


def _compact_latent_implementation_roster_candidates(
    context: _CompactConcreteFamilyContext,
    config: DetectorConfig,
) -> tuple[LatentImplementationRosterCandidate, ...]:
    class_index = context.class_index
    candidates: list[LatentImplementationRosterCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if not indexed_class.is_abstract or _compact_family_has_registration_authority(
            class_index, indexed_class
        ):
            continue
        descendants = _compact_concrete_descendants(class_index, indexed_class)
        if len(descendants) < max(2, config.min_registration_sites):
            continue
        concrete_class_names = sorted_tuple(
            _compact_class_display_name(descendant, class_index)
            for descendant in descendants
        )
        concrete_simple_names = sorted_tuple(
            descendant.simple_name for descendant in descendants
        )
        key_values_by_attr = tuple(
            (key_attr_name, key_values)
            for key_attr_name in _compact_semantic_key_attr_names(descendants)
            if len(
                key_values := _compact_descendant_key_values(
                    descendants,
                    key_attr_name,
                )
            )
            >= 2
        )
        for roster in context.latent_rosters:
            key_attr_name: str | None = None
            match = roster.match(concrete_class_names) or roster.match(
                concrete_simple_names
            )
            if match is None:
                key_match = _compact_matched_latent_roster_key_attr(
                    roster,
                    key_values_by_attr,
                )
                if key_match is not None:
                    key_attr_name, match = key_match
            if match is None:
                continue
            candidates.append(
                LatentImplementationRosterCandidate(
                    file_path=roster.file_path,
                    line=roster.line,
                    class_name=_compact_class_display_name(indexed_class, class_index),
                    roster=roster,
                    match=match,
                    concrete_class_names=concrete_class_names,
                    key_attr_name=key_attr_name,
                )
            )
    return tuple(candidates)


def _compact_predicate_selected_concrete_family_candidates(
    context: _CompactConcreteFamilyContext,
    config: DetectorConfig,
) -> tuple[PredicateSelectedConcreteFamilyCandidate, ...]:
    class_index = context.class_index
    candidates: list[PredicateSelectedConcreteFamilyCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if "_registered_types" not in indexed_class.assignments_by_name:
            continue
        descendants = _compact_concrete_descendants(class_index, indexed_class)
        if len(descendants) < config.min_registration_sites:
            continue
        concrete_class_names = sorted_tuple(
            _compact_class_display_name(descendant, class_index)
            for descendant in descendants
        )
        for (
            line,
            selector_method_name,
            predicate_method_name,
            context_param_name,
        ) in indexed_class.predicate_selected_methods:
            candidates.append(
                PredicateSelectedConcreteFamilyCandidate(
                    file_path=indexed_class.file_path,
                    line=line,
                    class_name=_compact_class_display_name(indexed_class, class_index),
                    selector_method_name=selector_method_name,
                    predicate_method_name=predicate_method_name,
                    context_param_name=context_param_name,
                    concrete_class_names=concrete_class_names,
                )
            )
    return tuple(candidates)


def _compact_mirrored_leaf_family_map(
    descendants: tuple[CompactIndexedClass, ...],
    *,
    axis_prefix_tokens: tuple[str, ...],
) -> dict[str, CompactIndexedClass]:
    leaf_map: dict[str, CompactIndexedClass] = {}
    for descendant in descendants:
        tokens = CLASS_NAME_ALGEBRA.ordered_tokens(descendant.simple_name)
        if (
            len(tokens) <= len(axis_prefix_tokens)
            or tokens[: len(axis_prefix_tokens)] != axis_prefix_tokens
        ):
            continue
        family_tokens = tokens[len(axis_prefix_tokens) :]
        if family_tokens:
            leaf_map.setdefault(" ".join(family_tokens), descendant)
    return leaf_map


def _compact_parallel_mirrored_leaf_family_candidates(
    context: _CompactConcreteFamilyContext,
    config: DetectorConfig,
) -> tuple[ParallelMirroredLeafFamilyCandidate, ...]:
    class_index = context.class_index
    min_shared_families = max(3, config.min_registration_sites)
    roots: list[
        tuple[
            CompactIndexedClass,
            tuple[str, ...],
            tuple[CompactIndexedClass, ...],
        ]
    ] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if "_registered_types" not in indexed_class.assignments_by_name:
            continue
        if not indexed_class.abstract_method_names:
            continue
        descendants = _compact_concrete_descendants(class_index, indexed_class)
        if len(descendants) >= min_shared_families:
            roots.append(
                (indexed_class, indexed_class.abstract_method_names, descendants)
            )

    candidates: list[ParallelMirroredLeafFamilyCandidate] = []
    for (left_root, left_methods, left_descendants), (
        right_root,
        right_methods,
        right_descendants,
    ) in combinations(roots, 2):
        shared_methods = sorted_tuple(set(left_methods) & set(right_methods))
        if not shared_methods:
            continue
        left_tokens = CLASS_NAME_ALGEBRA.ordered_tokens(left_root.simple_name)
        right_tokens = CLASS_NAME_ALGEBRA.ordered_tokens(right_root.simple_name)
        shared_root_suffix = _shared_ordered_suffix(left_tokens, right_tokens)
        if not shared_root_suffix:
            continue
        left_axis_prefix = left_tokens[: len(left_tokens) - len(shared_root_suffix)]
        right_axis_prefix = right_tokens[: len(right_tokens) - len(shared_root_suffix)]
        if (
            not left_axis_prefix
            or not right_axis_prefix
            or left_axis_prefix == right_axis_prefix
        ):
            continue
        left_leaf_map = _compact_mirrored_leaf_family_map(
            left_descendants, axis_prefix_tokens=left_axis_prefix
        )
        right_leaf_map = _compact_mirrored_leaf_family_map(
            right_descendants, axis_prefix_tokens=right_axis_prefix
        )
        if not left_leaf_map or not right_leaf_map:
            continue
        shared_leaf_families = sorted_tuple(set(left_leaf_map) & set(right_leaf_map))
        if len(shared_leaf_families) < max(
            min_shared_families, min(len(left_leaf_map), len(right_leaf_map)) // 2
        ):
            continue

        def leaf_evidence(
            leaf_map: dict[str, CompactIndexedClass],
        ) -> tuple[SourceLocation, ...]:
            return tuple(
                SourceLocation(
                    leaf_map[family_name].file_path,
                    leaf_map[family_name].line,
                    _compact_class_display_name(leaf_map[family_name], class_index),
                )
                for family_name in shared_leaf_families
            )

        candidates.append(
            ParallelMirroredLeafFamilyCandidate(
                left=MirroredLeafFamilySide(
                    file_path=left_root.file_path,
                    line=left_root.line,
                    root_name=_compact_class_display_name(left_root, class_index),
                    leaf_evidence=leaf_evidence(left_leaf_map),
                ),
                right=MirroredLeafFamilySide(
                    file_path=right_root.file_path,
                    line=right_root.line,
                    root_name=_compact_class_display_name(right_root, class_index),
                    leaf_evidence=leaf_evidence(right_leaf_map),
                ),
                contract_method_names=shared_methods,
                shared_leaf_family_names=shared_leaf_families,
            )
        )
    return tuple(candidates)


def _compact_family_has_registration_authority(
    class_index: CompactClassFamilyIndex,
    indexed_class: CompactIndexedClass,
) -> bool:
    return any(
        candidate is not None and candidate.is_registration_authority
        for symbol in (
            indexed_class.symbol,
            *class_index.ancestor_symbols(indexed_class.symbol),
        )
        for candidate in (class_index.class_for(symbol),)
    )


def _compact_semantic_key_attr_names(
    descendants: tuple[CompactIndexedClass, ...],
) -> tuple[str, ...]:
    if not descendants:
        return ()
    assignment_name_sets = tuple(
        frozenset(
            name
            for name, value in descendant.direct_assignment_expressions
            if value is not None and _looks_like_semantic_key_attr(name)
        )
        for descendant in descendants
    )
    common_names = set(assignment_name_sets[0])
    for assignment_names in assignment_name_sets[1:]:
        common_names &= set(assignment_names)
    return sorted_tuple(common_names)


def _compact_inherited_autoregister_registry_key_attr_name(
    class_index: CompactClassFamilyIndex,
    indexed_class: CompactIndexedClass,
) -> str | None:
    for symbol in (
        indexed_class.symbol,
        *class_index.ancestor_symbols(indexed_class.symbol),
    ):
        current_class = class_index.class_for(symbol)
        if (
            current_class is not None
            and current_class.autoregister_registry_key_attr_name is not None
        ):
            return current_class.autoregister_registry_key_attr_name
    return None


def _compact_inherited_autoregister_key_extractor_name(
    class_index: CompactClassFamilyIndex,
    indexed_class: CompactIndexedClass,
) -> str | None:
    for symbol in (
        indexed_class.symbol,
        *class_index.ancestor_symbols(indexed_class.symbol),
    ):
        current_class = class_index.class_for(symbol)
        if (
            current_class is not None
            and current_class.autoregister_key_extractor_name is not None
        ):
            return current_class.autoregister_key_extractor_name
    return None


def _compact_autoregister_dynamic_factory_symbols(
    projections: tuple[CompactModuleClassProjection, ...],
    *,
    family_name: str,
    concrete_class_names: tuple[str, ...],
) -> tuple[str, ...]:
    symbol_names = frozenset((family_name, *concrete_class_names))
    return sorted_tuple(
        {
            reference.qualname
            for projection in projections
            for reference in projection.autoregister_function_references
            if reference.calls_autoregister_meta
            and not frozenset(reference.referenced_symbols).isdisjoint(symbol_names)
        }
    )


def _compact_autoregister_behavior_method_names(
    indexed_class: CompactIndexedClass,
    concrete_descendants: tuple[CompactIndexedClass, ...],
) -> tuple[str, ...]:
    registry_projection_names = set(
        indexed_class.autoregister_registry_projection_names
    )
    return sorted_tuple(
        {
            method_name
            for candidate in (indexed_class, *concrete_descendants)
            for method_name in candidate.method_names
            if not method_name.startswith("__")
            and method_name not in registry_projection_names
        }
    )


def _compact_autoregister_consumer_index(
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
        ) in _compact_autoregister_reference_edges(reference_index.encoded_edges):
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


def _compact_autoregister_reference_edges(
    encoded_edges: str,
) -> Iterator[tuple[int, int, int]]:
    for encoded_edge in encoded_edges.split(";"):
        if not encoded_edge:
            continue
        function_index, receiver_index, attribute_index = encoded_edge.split(",")
        yield int(function_index), int(receiver_index), int(attribute_index)


def _compact_autoregister_consumer_symbols(
    consumer_index: dict[tuple[str, str], frozenset[str]],
    *,
    family_name: str,
    lookup_method_names: tuple[str, ...],
) -> tuple[str, ...]:
    consumer_symbols = {
        qualname
        for method_name in lookup_method_names
        for qualname in consumer_index.get((family_name, method_name), ())
        if not qualname.startswith(f"{family_name}.")
    }
    return sorted_tuple(consumer_symbols)


def _compact_autoregister_meta_rent_candidates(
    projections: tuple[CompactModuleClassProjection, ...],
    config: DetectorConfig,
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[AutoRegisterMetaRentCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(projections)
    relevant_consumer_keys = frozenset(
        (family_name, method_name)
        for indexed_class in class_index.classes_by_symbol.values()
        if indexed_class.declares_autoregister_meta
        for family_name in (_compact_class_display_name(indexed_class, class_index),)
        for method_name in indexed_class.autoregister_registry_projection_names
    )
    consumer_index = _compact_autoregister_consumer_index(
        projections, relevant_consumer_keys
    )
    min_leaf_count = max(2, config.min_registration_sites)
    candidates: list[AutoRegisterMetaRentCandidate] = []
    for indexed_class in sorted(
        class_index.classes_by_symbol.values(), key=lambda item: item.symbol
    ):
        if PythonSourcePathPolicy.is_test_path(Path(indexed_class.file_path)):
            continue
        if not indexed_class.declares_autoregister_meta:
            continue
        concrete_descendants = _compact_concrete_descendants(class_index, indexed_class)
        concrete_class_names = tuple(
            _compact_class_display_name(descendant, class_index)
            for descendant in concrete_descendants
        )
        family_name = _compact_class_display_name(indexed_class, class_index)
        dynamic_factory_symbols = _compact_autoregister_dynamic_factory_symbols(
            projections,
            family_name=family_name,
            concrete_class_names=concrete_class_names,
        )
        registry_key_attr_name = _compact_inherited_autoregister_registry_key_attr_name(
            class_index, indexed_class
        )
        key_extractor_name = _compact_inherited_autoregister_key_extractor_name(
            class_index, indexed_class
        )
        registry_projection_names = indexed_class.autoregister_registry_projection_names
        consumer_symbols = _compact_autoregister_consumer_symbols(
            consumer_index,
            family_name=family_name,
            lookup_method_names=registry_projection_names,
        )
        behavior_method_names = _compact_autoregister_behavior_method_names(
            indexed_class, concrete_descendants
        )
        abstract_method_names = indexed_class.abstract_method_names
        missing_rent_signals = _autoregister_missing_rent_signals(
            concrete_class_names=concrete_class_names,
            dynamic_factory_symbols=dynamic_factory_symbols,
            registry_key_attr_name=registry_key_attr_name,
            key_extractor_name=key_extractor_name,
            behavior_method_names=behavior_method_names,
            abstract_method_names=abstract_method_names,
            registry_projection_names=registry_projection_names,
            consumer_symbols=consumer_symbols,
            min_leaf_count=min_leaf_count,
        )
        if missing_rent_signals == ("registered_leaf_axis",) and (
            behavior_method_names
            or abstract_method_names
            or registry_projection_names
            or consumer_symbols
        ):
            continue
        if not missing_rent_signals:
            continue
        membership_object_count = _autoregister_membership_object_count(
            concrete_class_names=concrete_class_names,
            dynamic_factory_symbols=dynamic_factory_symbols,
            behavior_method_names=behavior_method_names,
            abstract_method_names=abstract_method_names,
            registry_projection_names=registry_projection_names,
            consumer_symbols=consumer_symbols,
        )
        certificate = _autoregister_rent_certificate(
            manual_object_count=membership_object_count,
            class_name=family_name,
            registry_axis_name=registry_key_attr_name
            or key_extractor_name
            or "class_identity",
            semantic_axis_names=(
                *behavior_method_names,
                *abstract_method_names,
                *registry_projection_names,
                *dynamic_factory_symbols,
            ),
            residual_object_count=len(concrete_class_names)
            + len(dynamic_factory_symbols),
            independent_source_count=max(
                1, len(concrete_class_names) + len(dynamic_factory_symbols)
            ),
        )
        candidates.append(
            AutoRegisterMetaRentCandidate(
                file_path=indexed_class.file_path,
                line=indexed_class.line,
                class_name=family_name,
                concrete_class_names=concrete_class_names,
                dynamic_factory_symbols=dynamic_factory_symbols,
                registry_key_attr_name=registry_key_attr_name,
                key_extractor_name=key_extractor_name,
                behavior_method_names=behavior_method_names,
                abstract_method_names=abstract_method_names,
                registry_projection_names=registry_projection_names,
                consumer_symbols=consumer_symbols,
                missing_rent_signals=missing_rent_signals,
                membership_object_count=membership_object_count,
                derived_projection_count=_autoregister_derived_projection_count(
                    registry_key_attr_name=registry_key_attr_name,
                    key_extractor_name=key_extractor_name,
                    behavior_method_names=behavior_method_names,
                    abstract_method_names=abstract_method_names,
                    registry_projection_names=registry_projection_names,
                    consumer_symbols=consumer_symbols,
                ),
                rent_margin=certificate.certified_description_length_savings,
                compression_certificate=certificate,
            )
        )
    return tuple(candidates)


class AutoRegisterMetaUnderRentedDetector(
    CompactClassRepositoryCandidateDetector[AutoRegisterMetaRentCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_autoregister_meta_root
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegisterMeta family should prove its rent",
        "A metaclass registry pays rent when it derives a semantic family membership surface: a stable key axis, multiple registered leaves, a behavioral or abstract contract, and some registry projection or consumer. Without those coordinates, the metaclass is mostly signature noise and the same information usually belongs in a typed declaration table, enum, or ordinary ABC.",
        "AutoRegisterMeta-backed family with computed rent evidence over key axis, leaves, behavior, projections, and consumers",
        "class declares AutoRegisterMeta but lacks enough generic rent signals to justify metaclass registration",
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
    detector_id = "autoregister_meta_under_rented"

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[AutoRegisterMetaRentCandidate, ...]:
        return _compact_autoregister_meta_rent_candidates(
            context.projections,
            config,
            class_index=context.class_index,
        )

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
    _CompactConcreteFamilyDetectorBase[PredicateSelectedConcreteFamilyCandidate],
):
    compact_report_context_promotion_predicate = staticmethod(
        _target_has_predicate_selected_root
    )
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Predicate-selected concrete family should collapse into one metaclass-registry selector base",
        "The docs treat repeated scans over `registered_types()` plus `matches_*` predicates as family-selection framework logic. When a root class manually filters registered concrete descendants, enforces exactly one match, and then consumes the chosen subclass, the selection algorithm should live in one reusable `metaclass-registry` family base.",
        "single authoritative metaclass-registry predicate-selected concrete-family substrate",
        "registered concrete subclasses are manually scanned and cardinality-checked inside a family root",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.REGISTRY_POPULATION,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: _CompactConcreteFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[PredicateSelectedConcreteFamilyCandidate]:
        return _compact_predicate_selected_concrete_family_candidates(context, config)

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
    _CompactConcreteFamilyDetectorBase[ParallelMirroredLeafFamilyCandidate],
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Parallel mirrored leaf families should derive from one axis-declared family substrate",
        "The docs treat mirrored registered leaf catalogs as framework duplication when the same contract is repeated across two family roots and only one nominal axis really varies. The axis and role table should be authoritative so registration and leaf generation are derived instead of hand-expanded twice.",
        "single authoritative axis-declared family or role-spec table that derives mirrored registered leaves",
        "two registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: _CompactConcreteFamilyContext,
        config: DetectorConfig,
    ) -> Sequence[ParallelMirroredLeafFamilyCandidate]:
        return _compact_parallel_mirrored_leaf_family_candidates(
            context,
            config,
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


@dataclass(frozen=True)
class SentinelAttributeSimulationCandidate:
    attr_name: str
    evidence: tuple[SourceLocation, ...]
    branch_evidence: tuple[SourceLocation, ...]


class SentinelAttributeSimulationDetector(
    CandidateFindingDetector[SentinelAttributeSimulationCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.NOMINAL_BOUNDARY,
        "Sentinel attribute is simulating nominal identity",
        "The docs say sentinel attributes only simulate identity by convention. When they drive behavior across multiple classes, the boundary should become a nominal family or another explicit identity handle.",
        "enumerable and enforceable nominal role identity",
        "same class-level sentinel attribute reused as a fake identity boundary",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.SENTINEL_ATTRIBUTE,
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[SentinelAttributeSimulationCandidate]:
        sentinel_attrs = _collect_class_sentinel_attrs(module.module)
        candidates: list[SentinelAttributeSimulationCandidate] = []
        for attr_name, evidence in sentinel_attrs.items():
            if len(evidence) < 2:
                continue
            branch_evidence = _attribute_branch_evidence(module, attr_name)
            if not branch_evidence:
                continue
            generic_name = attr_name.lower() in {"name", "label", "title"}
            if generic_name and len(branch_evidence) < 2:
                continue
            candidates.append(
                SentinelAttributeSimulationCandidate(
                    attr_name=attr_name,
                    evidence=tuple(evidence),
                    branch_evidence=tuple(branch_evidence),
                )
            )
        return tuple(candidates)

    def _finding_for_candidate(
        self, candidate: SentinelAttributeSimulationCandidate
    ) -> RefactorFinding:
        evidence = candidate.evidence
        branch_evidence = candidate.branch_evidence
        return self.build_finding(
            f"Attribute `{candidate.attr_name}` is declared across {len(evidence)} classes and also drives {len(branch_evidence)} branch sites.",
            tuple((evidence + branch_evidence)[:6]),
            metrics=SentinelSimulationMetrics(
                class_count=len(evidence), branch_site_count=len(branch_evidence)
            ),
        )


@dataclass(frozen=True)
class PredicateFactoryChainCandidate:
    file_path: str
    function: ast.FunctionDef | ast.AsyncFunctionDef
    branch_count: int


class PredicateFactoryChainDetector(
    CandidateFindingDetector[PredicateFactoryChainCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.DISCRIMINATED_UNION,
        "Predicate chain should become a discriminated union family",
        "The docs say repeated predicate-driven variant selection should become an explicit subclass family with enumeration rather than an open-ended if/elif chain.",
        "exhaustive nominal variant discovery and extension",
        "same factory role repeated as predicate branches inside one function",
        (
            CapabilityTag.ENUMERATION,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.PREDICATE_CHAIN,
            ObservationTag.FACTORY_DISPATCH,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[PredicateFactoryChainCandidate]:
        del config
        return tuple(
            (
                PredicateFactoryChainCandidate(
                    file_path=str(module.path),
                    function=function,
                    branch_count=branch_count,
                )
                for function in _iter_functions(module.module)
                if (branch_count := _predicate_factory_chain_branch_count(function))
                is not None
            )
        )

    def _finding_for_candidate(
        self, candidate: PredicateFactoryChainCandidate
    ) -> RefactorFinding:
        return self.build_finding(
            f"{candidate.function.name} contains a {candidate.branch_count}-branch predicate factory chain returning variant constructors.",
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.function.lineno,
                    candidate.function.name,
                ),
            ),
            metrics=BranchCountMetrics(branch_site_count=candidate.branch_count),
        )


declare_typed_observation_detector(
    "ConfigAttributeDispatchDetector",
    finding_spec_template(
        PatternId.CONFIG_CONTRACTS,
        "Config dispatch is encoded through fragile attribute probing",
        "The docs say polymorphic configuration should dispatch on declared config family identity, not on field-name probing or ad hoc attribute comparisons.",
        "fail-loud polymorphic configuration contracts",
        "same config-family choice expressed through attribute-level probing",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
        ),
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
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.ATTRIBUTE_PROBE,
            ObservationTag.CONFIG_DISPATCH,
            ObservationTag.CLASS_FAMILY,
        ),
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


class DualAxisResolutionDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.DUAL_AXIS_RESOLUTION,
        "Nested precedence walk should be a dual-axis resolution primitive",
        "The docs say scope x type precedence should be modeled explicitly when both context and inheritance order contribute to resolution and provenance.",
        "explicit dual-axis precedence with provenance",
        "same function combines context hierarchy and type/MRO hierarchy",
        (
            CapabilityTag.DUAL_AXIS_RESOLUTION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.NESTED_PRECEDENCE_WALK,
            ObservationTag.SCOPE_HIERARCHY,
            ObservationTag.MRO_HIERARCHY,
        ),
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


class DynamicRuntimePayloadOwnerDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Runtime payload extraction should use the declared owner",
        "Dispatching runtime payload extraction through type(x) lets a composed subclass decide the payload schema. That can leak unrelated inherited fields across a formal boundary; payload extraction should name the declared carrier or authority that owns the field set.",
        "payload source extraction names the declared owner carrier",
        "runtime payload extraction is dispatched through the concrete dynamic type",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )
    ssot_authority_boundary = True

    _payload_extraction_methods: ClassVar[frozenset[str]] = frozenset(
        (
            "runtime_payload_for_instance",
            "runtime_payload_for_declared_fields",
        )
    )

    @classmethod
    def _dynamic_type_payload_method(cls, node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        method_name = node.func.attr
        if method_name not in cls._payload_extraction_methods:
            return None
        owner = node.func.value
        if not isinstance(owner, ast.Call):
            return None
        if not isinstance(owner.func, ast.Name) or owner.func.id != "type":
            return None
        return method_name

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
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

            def owner_symbol(self) -> str:
                parts = (*self.class_stack, *self.function_stack)
                return ".".join(parts) if parts else "<module>"

            def visit_Call(self, node: ast.Call) -> None:
                method_name = (
                    DynamicRuntimePayloadOwnerDetector._dynamic_type_payload_method(
                        node
                    )
                )
                if method_name is not None:
                    owner = self.owner_symbol()
                    findings.append(
                        detector.build_finding(
                            f"{owner} extracts `{method_name}` through `type(...)` at {module.path}:{node.lineno}.",
                            (
                                SourceLocation(
                                    str(module.path),
                                    node.lineno,
                                    f"{owner}.{method_name}",
                                ),
                            ),
                            relation_context=(
                                "payload schema ownership is selected by the "
                                "concrete runtime type instead of the declared "
                                "carrier authority"
                            ),
                            scaffold=(
                                "# Replace `type(state).runtime_payload_for_instance(state)` with\n"
                                "# `DeclaredPayloadOwner.runtime_payload_for_instance(state)` so\n"
                                "# subclass composition cannot widen the Lean-declared field set."
                            ),
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(module.module)
        return findings


declare_typed_observation_detector(
    "ManualVirtualMembershipDetector",
    finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "Manual class-marker membership should become custom isinstance semantics",
        "The docs say explicit runtime interface membership should be class-level and inspectable. Repeated marker checks suggest a custom isinstance/subclass boundary rather than scattered manual probing.",
        "runtime-checkable virtual membership on nominal class identity",
        "same membership question repeated through class-marker probing",
        (
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.CLASS_MARKER_PROBE,
            ObservationTag.RUNTIME_MEMBERSHIP,
        ),
    ),
    ClassMarkerObservationFamily,
    ClassMarkerObservation,
    "{module_path} performs {evidence_count} class-level marker checks on instances.",
    minimum_evidence_count=2,
)


@dataclass(frozen=True)
class ExactTypeGuardInheritanceRetreatCandidate:
    guard: CompactExactTypeGuard
    base_class: CompactIndexedClass
    descendant_classes: tuple[CompactIndexedClass, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(self.guard.file_path, self.guard.line, self.guard.qualname),
            SourceLocation(
                self.base_class.file_path,
                self.base_class.line,
                self.base_class.simple_name,
            ),
            *(
                SourceLocation(
                    descendant.file_path,
                    descendant.line,
                    descendant.simple_name,
                )
                for descendant in self.descendant_classes[:4]
            ),
        )


def _exact_type_guard_candidates_from_compact_projections(
    projections: tuple[CompactModuleClassProjection, ...],
    *,
    class_index: CompactClassFamilyIndex | None = None,
) -> tuple[ExactTypeGuardInheritanceRetreatCandidate, ...]:
    if class_index is None:
        class_index = build_compact_class_family_index(projections)
    resolver = CompactClassReferenceResolver.from_index(projections, class_index)
    candidates: list[ExactTypeGuardInheritanceRetreatCandidate] = []
    for projection in projections:
        for guard in projection.exact_type_guards:
            base_symbol = resolver.symbol_for(
                module_name=projection.module_name,
                reference_parts=guard.type_reference_parts,
            )
            if base_symbol is None:
                continue
            base_class = class_index.class_for(base_symbol)
            if base_class is None or base_class.is_final:
                continue
            descendants = tuple(
                descendant
                for descendant_symbol in class_index.descendant_symbols(base_symbol)
                if (descendant := class_index.class_for(descendant_symbol)) is not None
            )
            if not descendants:
                continue
            candidates.append(
                ExactTypeGuardInheritanceRetreatCandidate(
                    guard=guard,
                    base_class=base_class,
                    descendant_classes=descendants,
                )
            )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.guard.file_path,
            candidate.guard.line,
            candidate.guard.qualname,
            candidate.base_class.symbol,
        ),
    )


class ExactTypeGuardInheritanceRetreatDetector(
    CompactClassRepositoryCandidateDetector[ExactTypeGuardInheritanceRetreatCandidate],
):
    detector_priority = -21
    finding_spec = high_confidence_certified_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Exact-type boundary guard retreats from nominal inheritance",
        "A fail-loud boundary compares `type(value)` with a class that has resolved nominal descendants. The exact comparison rejects valid family members and weakens substitutability; membership belongs to the inheritance graph through `isinstance`.",
        "inheritance-preserving runtime membership at the fail-loud boundary",
        "exact concrete-type validation contradicts a resolved base-to-descendant class-family edge",
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

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ExactTypeGuardInheritanceRetreatCandidate, ...]:
        del config
        return _exact_type_guard_candidates_from_compact_projections(
            context.projections,
            class_index=context.class_index,
        )

    def _finding_for_candidate(
        self,
        candidate: ExactTypeGuardInheritanceRetreatCandidate,
    ) -> RefactorFinding:
        guard = candidate.guard
        descendants = ", ".join(
            descendant.simple_name for descendant in candidate.descendant_classes[:6]
        )
        return self.build_finding(
            (
                f"`{guard.qualname}` enforces `{guard.expression}` "
                f"against base `{candidate.base_class.simple_name}`, but the resolved "
                f"inheritance graph contains descendant(s) {descendants}."
            ),
            candidate.evidence,
            scaffold=guard.structural_membership_expression,
            codemod_patch=(
                f"# Replace `{guard.expression}` with "
                f"`{guard.structural_membership_expression}` at this "
                "boundary; preserve the existing fail-loud branch and let nominal "
                "subclasses satisfy the base contract."
            ),
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=1,
                class_count=1 + len(candidate.descendant_classes),
            ),
        )


@dataclass(frozen=True)
class _ExternalConcreteTypeIdentityTableCandidate(LineWitnessCandidate):
    symbol: str
    row_pairs: tuple[tuple[str, str, int], ...]


class ExternalConcreteTypeIdentityTableDetector(PerModuleIssueDetector):
    finding_spec = finding_spec_template(
        PatternId.VIRTUAL_MEMBERSHIP,
        "External concrete type identity table should become capability registration",
        "A table of hardcoded external module/type string identities is recovering runtime membership from concrete implementation names. The nominal boundary should be an explicit capability registration surface owned by the integration layer, not a core table of third-party class names.",
        "extension-owned virtual membership registration boundary",
        "same registry table maps external concrete type identities to capability registration",
        (
            CapabilityTag.VIRTUAL_MEMBERSHIP,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.RUNTIME_MEMBERSHIP,
            ObservationTag.SEMANTIC_STRING_LITERAL,
        ),
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
    "SentinelTypeMarkerDetector",
    finding_spec_template(
        PatternId.SENTINEL_TYPE_MARKER,
        "Unique sentinel type marker is present or should be used",
        "The docs distinguish sentinel types from sentinel attributes: unique nominal marker objects are appropriate when exact capability identity matters more than payload.",
        "exact capability-marker identity independent of structure",
        "same module creates or uses unique nominal sentinel markers",
        (
            CapabilityTag.CAPABILITY_MARKER_IDENTITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.SENTINEL_TYPE,
            ObservationTag.CAPABILITY_MARKER,
        ),
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
        (
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.DYNAMIC_METHOD_INJECTION,
            ObservationTag.TYPE_NAMESPACE,
        ),
    ),
    DynamicMethodInjectionObservationFamily,
    DynamicMethodInjectionObservation,
    "{module_path} contains {evidence_count} dynamic type-namespace injection sites.",
    evidence_limit=6,
)


class NumericLiteralDispatchDetector(PerModuleIssueDetector):
    finding_spec = certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Closed-family dispatch expressed through numeric IDs",
        "The docs treat repeated numeric pattern or mode IDs the same way as magic strings: the domain axis is real but undeclared. Replace the literal-ID branches with a nominal family keyed by a stable axis; if the cases select behavior, prefer an auto-registered family over a handwritten lookup table.",
        "closed-family dispatch with stable nominal keys and auto-registered type authority for behavioral cases",
        "same dispatch role repeated through numeric literal comparisons",
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
        ),
        (
            ObservationTag.LITERAL_ID_DISPATCH,
            ObservationTag.PARTIAL_VIEW,
        ),
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


_STATIC_PAYLOAD_WRITE_METHODS = frozenset(
    {"dump", "dumps", "write", "write_text", "write_bytes", "writelines"}
)
_WRITE_MODE_TOKENS = frozenset({"w", "a", "x", "wt", "at", "xt", "wb", "ab", "xb"})


@dataclass(frozen=True)
class StaticPayloadStats:
    payload_line_count: int
    largest_literal_line_count: int
    marker_kinds: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddedStaticPayloadCandidate(QualnameLineWitnessCandidate):
    function_name: str
    line_count: int
    static_payload_stats: StaticPayloadStats
    sink_kinds: tuple[str, ...]
    call_site_count: int


_RuntimeFunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
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


def _walk_function_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    return walk_function_body_nodes(function)


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
class PrivateReferenceIndexedFunction:
    """One surface function projected once for every private-reference facet."""

    module_name: str
    qualname: str
    function: _RuntimeFunctionNode
    symbol_references: frozenset[str]
    body_digest: str


@dataclass(frozen=True)
class PrivateReferenceNamedFunction:
    """One named function plus its shared runtime-membership call projection."""

    qualname: str
    function: _RuntimeFunctionNode
    isinstance_calls: tuple[ast.Call, ...]


@dataclass(frozen=True)
class PrivateReferenceModuleIndex:
    """Single-traversal module index shared by private-reference detectors."""

    total_counts: Counter[str]
    function_counts_by_id: dict[int, Counter[str]]
    functions: tuple[PrivateReferenceIndexedFunction, ...]
    named_functions: tuple[PrivateReferenceNamedFunction, ...]

    @classmethod
    def from_module(cls, module: ParsedModule) -> "PrivateReferenceModuleIndex":
        return _private_reference_module_index(
            module.module,
            module.module_name,
            module.semantic_hash,
        )


@lru_cache(maxsize=None)
def _private_reference_module_index(
    module_node: ast.Module,
    module_name: str,
    semantic_hash: str | None,
) -> PrivateReferenceModuleIndex:
    surface_functions = SurfaceFunctionIndex.from_module(module_node).functions
    surface_qualnames_by_id = {
        id(function): qualname for qualname, function in surface_functions
    }
    total_counts: Counter[str] = Counter()
    function_counts_by_id = {
        id(function): Counter() for _, function in surface_functions
    }
    symbol_references_by_function_id = {
        id(function): set() for _, function in surface_functions
    }
    named_functions: list[tuple[int, str]] = []
    named_function_nodes_by_id: dict[int, _RuntimeFunctionNode] = {}
    isinstance_calls_by_function_id: dict[
        int,
        list[tuple[int, int, ast.Call]],
    ] = defaultdict(list)

    def symbol_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    syntax_index = module_syntax_index(module_node)
    named_functions.extend(
        (id(function), qualname) for qualname, function in syntax_index.named_functions
    )
    named_function_nodes_by_id.update(
        (id(function), function) for _, function in syntax_index.named_functions
    )
    scope_function_ids = tuple(
        tuple(
            id(syntax_index.depth_first_nodes[function_index])
            for function_index in scope.function_node_indices
        )
        for scope in syntax_index.scopes
    )
    counted_function_ids_by_scope = tuple(
        tuple(
            function_id
            for function_id in function_ids
            if function_id in surface_qualnames_by_id
        )
        for function_ids in scope_function_ids
    )
    active_named_functions_by_scope = tuple(
        tuple(
            (function_id, syntax_index.depths[function_index])
            for function_id, function_index in zip(
                function_ids,
                scope.function_node_indices,
                strict=True,
            )
        )
        for function_ids, scope in zip(
            scope_function_ids,
            syntax_index.scopes,
            strict=True,
        )
    )
    surface_function_ids_by_index = {
        function_index: function_id
        for function_index, function_id in (
            (function_index, id(syntax_index.depth_first_nodes[function_index]))
            for scope in syntax_index.scopes
            for function_index in scope.function_node_indices[-1:]
        )
        if function_id in surface_qualnames_by_id
    }

    for node_index, node in enumerate(syntax_index.depth_first_nodes):
        node_ordinal = node_index + 1
        if node_ordinal % 2048 == 0:
            scan_deadline_checkpoint("contextual_private_reference_index")
        scope_id = syntax_index.scope_ids[node_index]
        counted_function_ids = counted_function_ids_by_scope[scope_id]
        executable_function_index = syntax_index.executable_function_indices[node_index]
        reference_function_id = surface_function_ids_by_index.get(
            executable_function_index
        )
        name = symbol_name(node)
        if name is not None:
            total_counts[name] += 1
            for function_id in counted_function_ids:
                function_counts_by_id[function_id][name] += 1
            if reference_function_id is not None:
                symbol_references_by_function_id[reference_function_id].add(name)
        active_named_functions = active_named_functions_by_scope[scope_id]
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 2
            and not node.keywords
            and _ast_terminal_name(node.func) == "isinstance"
        ):
            for function_id, function_depth in active_named_functions:
                isinstance_calls_by_function_id[function_id].append(
                    (
                        syntax_index.depths[node_index] - function_depth,
                        node_ordinal,
                        node,
                    )
                )
    module_semantic_digest = semantic_hash or _stable_text_digest(
        ast.dump(module_node, include_attributes=False)
    )
    return PrivateReferenceModuleIndex(
        total_counts=total_counts,
        function_counts_by_id=function_counts_by_id,
        functions=tuple(
            PrivateReferenceIndexedFunction(
                module_name=module_name,
                qualname=qualname,
                function=function,
                symbol_references=frozenset(
                    symbol_references_by_function_id[id(function)]
                ),
                body_digest=_stable_text_digest(
                    f"{module_semantic_digest}\0{qualname}"
                ),
            )
            for qualname, function in surface_functions
        ),
        named_functions=tuple(
            PrivateReferenceNamedFunction(
                qualname=qualname,
                function=named_function_nodes_by_id[function_id],
                isinstance_calls=tuple(
                    call
                    for _, _, call in sorted(
                        isinstance_calls_by_function_id.get(function_id, ()),
                        key=lambda item: (item[0], item[1]),
                    )
                ),
            )
            for function_id, qualname in named_functions
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
class CallCountedQualnameCandidate(
    CallSiteCountMetric,
    LineCountedQualnameCandidate,
):
    """Candidate with repository-visible call-site evidence."""


@dataclass(frozen=True)
class UnreferencedPrivateFunctionCandidate(CallCountedQualnameCandidate):
    function_name: str


@dataclass(frozen=True)
class DanglingPrivateMethodCandidate(CallCountedQualnameCandidate):
    owner_name: str
    method_name: str


class DerivedCandidateCollectorContracts:
    def names(self, modules: Sequence[ParsedModule]) -> frozenset[str]:
        collector_names: set[str] = set()
        for module in modules:
            for node in module.module.body:
                if not (
                    isinstance(node, ast.ClassDef)
                    and HELPER_SYNTAX_PROJECTION_AUTHORITY.class_declares_finding_spec(
                        node
                    )
                ):
                    continue
                collector_name = _candidate_collector_name_from_class_name(node.name)
                if collector_name is not None:
                    collector_names.add(collector_name)
        return frozenset(collector_names)


DERIVED_CANDIDATE_COLLECTOR_CONTRACTS = DerivedCandidateCollectorContracts()


def _has_external_protocol_shape(
    function: _RuntimeFunctionNode,
) -> bool:
    if function.decorator_list:
        return True
    return function.name.endswith("_")


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


@dataclass(frozen=True)
class CompactPrivateFunctionFact:
    """AST-free surface-function facts used by private-reference detectors."""

    file_path: str
    qualname: str
    function_name: str
    line: int
    line_count: int
    call_site_count: int
    own_name_reference_count: int
    owner_name: str | None
    has_external_protocol_shape: bool
    is_detector_override_hook: bool
    static_payload_stats: StaticPayloadStats
    sink_kinds: tuple[str, ...]


@dataclass(frozen=True)
class CompactPrivateReferenceModuleProjection:
    """One module's reference counts and private function candidates."""

    total_counts: tuple[tuple[str, int], ...]
    derived_candidate_collector_contract_names: tuple[str, ...]
    functions: tuple[CompactPrivateFunctionFact, ...]


class CompactPrivateReferenceModuleProjectionFamily(
    CollectedFamily[CompactPrivateReferenceModuleProjection]
):
    item_type = CompactPrivateReferenceModuleProjection
    # Every private-reference finding is anchored by a private function fact.
    # Context can change reference counts, callers, and placement for a target
    # function, but it cannot create a report-scoped candidate when the target
    # contributes no private function.  Avoid building the otherwise eager
    # repository reference graph in that exact empty-target case.
    report_presence_predicate = staticmethod(
        lambda items, config: any(
            projection.functions
            for projection in items
            if isinstance(projection, CompactPrivateReferenceModuleProjection)
        )
    )

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactPrivateReferenceModuleProjection]:
        del cls
        module_index = PrivateReferenceModuleIndex.from_module(parsed_module)
        functions: list[CompactPrivateFunctionFact] = []
        for indexed_function in module_index.functions:
            function = indexed_function.function
            body_nodes = _walk_function_body_nodes(function)
            if not _is_private_symbol_name(function.name):
                continue
            owner_name = (
                indexed_function.qualname.rsplit(".", 1)[0]
                if "." in indexed_function.qualname
                else None
            )
            functions.append(
                CompactPrivateFunctionFact(
                    file_path=str(parsed_module.path),
                    qualname=indexed_function.qualname,
                    function_name=function.name,
                    line=function.lineno,
                    line_count=_function_line_count(function),
                    call_site_count=sum(
                        isinstance(node, ast.Call) for node in body_nodes
                    ),
                    own_name_reference_count=module_index.function_counts_by_id[
                        id(function)
                    ][function.name],
                    owner_name=owner_name,
                    has_external_protocol_shape=_has_external_protocol_shape(function),
                    is_detector_override_hook=(
                        owner_name is not None
                        and _is_detector_override_hook(
                            parsed_module,
                            owner_name,
                            function.name,
                        )
                    ),
                    static_payload_stats=_static_payload_stats(function),
                    sink_kinds=_static_payload_sink_kinds(function),
                )
            )
        return [
            CompactPrivateReferenceModuleProjection(
                total_counts=tuple(
                    sorted(
                        (symbol_name, count)
                        for symbol_name, count in module_index.total_counts.items()
                        if symbol_name.isidentifier()
                        and _is_private_symbol_name(symbol_name)
                    )
                ),
                derived_candidate_collector_contract_names=tuple(
                    sorted(
                        DERIVED_CANDIDATE_COLLECTOR_CONTRACTS.names((parsed_module,))
                    )
                ),
                functions=tuple(functions),
            )
        ]


def _compact_private_reference_total_counts(
    projections: tuple[CompactPrivateReferenceModuleProjection, ...],
) -> Counter[str]:
    total_counts: Counter[str] = Counter()
    for projection in projections:
        total_counts.update(dict(projection.total_counts))
    return total_counts


def _compact_private_function_is_unreferenced(
    function: CompactPrivateFunctionFact,
    total_counts: Counter[str],
) -> bool:
    return total_counts[function.function_name] - function.own_name_reference_count <= 0


def _compact_embedded_static_payload_candidates(
    projections: tuple[CompactPrivateReferenceModuleProjection, ...],
    config: DetectorConfig,
) -> tuple[EmbeddedStaticPayloadCandidate, ...]:
    total_counts = _compact_private_reference_total_counts(projections)
    return tuple(
        EmbeddedStaticPayloadCandidate(
            file_path=function.file_path,
            line=function.line,
            qualname=function.qualname,
            function_name=function.function_name,
            line_count=function.line_count,
            static_payload_stats=function.static_payload_stats,
            sink_kinds=function.sink_kinds,
            call_site_count=function.call_site_count,
        )
        for projection in projections
        for function in projection.functions
        if function.line_count >= config.min_static_payload_function_lines
        if (
            function.static_payload_stats.payload_line_count
            >= config.min_static_payload_literal_lines
        )
        if function.static_payload_stats.marker_kinds
        if function.sink_kinds
        if _compact_private_function_is_unreferenced(function, total_counts)
    )


def _compact_unreferenced_private_function_candidates(
    projections: tuple[CompactPrivateReferenceModuleProjection, ...],
    config: DetectorConfig,
) -> tuple[UnreferencedPrivateFunctionCandidate, ...]:
    total_counts = _compact_private_reference_total_counts(projections)
    contract_names = frozenset(
        name
        for projection in projections
        for name in projection.derived_candidate_collector_contract_names
    )
    return tuple(
        UnreferencedPrivateFunctionCandidate(
            file_path=function.file_path,
            line=function.line,
            qualname=function.qualname,
            function_name=function.function_name,
            line_count=function.line_count,
            call_site_count=function.call_site_count,
        )
        for projection in projections
        for function in projection.functions
        if function.owner_name is None
        if not function.has_external_protocol_shape
        if function.function_name not in contract_names
        if function.line_count >= config.min_unreferenced_private_function_lines
        if _compact_private_function_is_unreferenced(function, total_counts)
    )


def _compact_dangling_private_method_candidates(
    projections: tuple[CompactPrivateReferenceModuleProjection, ...],
    config: DetectorConfig,
) -> tuple[DanglingPrivateMethodCandidate, ...]:
    total_counts = _compact_private_reference_total_counts(projections)
    return tuple(
        DanglingPrivateMethodCandidate(
            file_path=function.file_path,
            line=function.line,
            qualname=function.qualname,
            owner_name=function.owner_name,
            method_name=function.function_name,
            line_count=function.line_count,
            call_site_count=function.call_site_count,
        )
        for projection in projections
        for function in projection.functions
        if function.owner_name is not None
        if not function.is_detector_override_hook
        if not function.has_external_protocol_shape
        if function.line_count >= config.min_unreferenced_private_function_lines
        if _compact_private_function_is_unreferenced(function, total_counts)
    )


class DeadEmbeddedStaticPayloadDetector(
    CompactProjectionCandidateDetector[
        CompactPrivateReferenceModuleProjection,
        EmbeddedStaticPayloadCandidate,
    ],
):
    module_projection_family = CompactPrivateReferenceModuleProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced embedded static-payload emitter should collapse",
        "A private function that is not referenced in its module but still embeds and writes a large static artifact payload is a duplicate derived-view authority. Delete it if it is genuinely dead; if it is reached dynamically, move the payload to a template/resource or generate it from an authoritative schema.",
        "single authoritative template/resource or generated schema for static artifact views",
        "private unreferenced emitter owns a large embedded static payload independently of call flow",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.EXPORT_MAPPING,
        ),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactPrivateReferenceModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[EmbeddedStaticPayloadCandidate]:
        return _compact_embedded_static_payload_candidates(
            projections,
            config,
        )

    def _finding_for_candidate(
        self, payload_candidate: EmbeddedStaticPayloadCandidate
    ) -> RefactorFinding:
        marker_summary = ", ".join(payload_candidate.static_payload_stats.marker_kinds)
        sink_summary = ", ".join(payload_candidate.sink_kinds)
        return self.build_finding(
            (
                f"`{payload_candidate.qualname}` spans {payload_candidate.line_count} lines, embeds "
                f"{payload_candidate.static_payload_stats.payload_line_count} static payload lines ({marker_summary}), "
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


class UnreferencedPrivateFunctionDetector(
    CompactProjectionCandidateDetector[
        CompactPrivateReferenceModuleProjection,
        UnreferencedPrivateFunctionCandidate,
    ],
):
    module_projection_family = CompactPrivateReferenceModuleProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Unreferenced private function should be deleted or made explicit",
        "A private function with no in-module references is not a witnessed local authority. If it is dead, delete it. If it is invoked dynamically or by an external framework, that contract should be made explicit through a registry, callback table, or public facade instead of relying on an invisible edge.",
        "explicit call-graph witness or deletion of dead private implementation surface",
        "private function is present in the implementation surface but absent from local call flow",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactPrivateReferenceModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[UnreferencedPrivateFunctionCandidate]:
        return _compact_unreferenced_private_function_candidates(
            projections,
            config,
        )

    finding_renderer = CandidateFindingRenderer[UnreferencedPrivateFunctionCandidate](
        summary=lambda function_candidate: (
            f"`{function_candidate.qualname}` spans {function_candidate.line_count} lines and has no in-module references."
        ),
        evidence=lambda function_candidate: (function_candidate.evidence,),
        scaffold=lambda function_candidate: (
            f"# Verify whether `{function_candidate.qualname}` is reached through reflection, subclassing, or an external framework.\n# If no such contract exists, delete it.\n# If it is dynamic API, declare that edge through a registry, callback table, or public facade."
        ),
        codemod_patch=lambda function_candidate: (
            f"# Remove `{function_candidate.qualname}` or replace the implicit dynamic edge with an explicit authority."
        ),
        metrics=lambda function_candidate: OrchestrationMetrics(
            function_line_count=function_candidate.line_count,
            branch_site_count=0,
            call_site_count=function_candidate.call_site_count,
            parameter_count=0,
            callee_family_count=1,
        ),
    )


class DanglingPrivateMethodDetector(
    CompactProjectionCandidateDetector[
        CompactPrivateReferenceModuleProjection,
        DanglingPrivateMethodCandidate,
    ],
):
    module_projection_family = CompactPrivateReferenceModuleProjectionFamily
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Dangling private method should be deleted or made nominal",
        "A private method that has no visible callsite, override contract, decorator, or framework hook is not a nominal interface. Inside a class it looks owned, but without a witnessed edge it is dead residue or an implicit protocol that should be made explicit through an ABC hook, public facade, strategy object, or registry-backed dispatch surface.",
        "explicit nominal hook or deletion of unreferenced private method residue",
        "private class method has no repository-visible reference outside its own body",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactPrivateReferenceModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[DanglingPrivateMethodCandidate]:
        return _compact_dangling_private_method_candidates(
            projections,
            config,
        )

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


@dataclass(frozen=True)
class SiblingSmallMethodTemplateCandidate(MethodEvidenceLocationsCandidate):
    owner_name: str
    statement_count: int
    parameter_count: int
    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("owner_name")


_NORMALIZED_TEMPLATE_DOMAIN_STABLE_NAMES = frozenset(
    {
        "False",
        "None",
        "True",
        "cls",
        "re",
        "self",
        "shutil",
    }
)
_NORMALIZED_TEMPLATE_STABLE_NAMES = (
    _NORMALIZED_TEMPLATE_DOMAIN_STABLE_NAMES
    | BuiltinCallName.normalized_template_stable_builtin_names()
)


def _trimmed_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    return tuple(_trim_docstring_body(function.body))


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
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
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

    evidence = SourceLocationEvidenceProperty("file_path", "line", "qualname")


@dataclass(frozen=True)
class CompactAlgebraicVariantModuleProjection(CompactModuleIdentity):
    """AST-free local facts for algebraic variant-family joins."""

    composition_signals: tuple[CancelableCompositionSignal, ...]
    variant_method_surfaces: tuple[_VariantMethodSurface, ...]


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
    composition_signals: tuple[CancelableCompositionSignal, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        evidence: list[SourceLocation] = [
            *self.seed.evidence,
            *(
                SourceLocation(signal.file_path, signal.line, signal.qualname)
                for signal in self.composition_signals[:2]
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


def _cancelable_composition_signals_for_module(
    module: ParsedModule,
) -> tuple[CancelableCompositionSignal, ...]:
    """Project exact composition signals while the module AST is already live."""

    artifacts = build_source_index_artifacts((module,), ())
    source_index = artifacts.source_index
    nodes_by_target_id = artifacts.target_artifacts.node_cache.nodes_by_target_id
    signals: list[CancelableCompositionSignal] = []
    for target in source_index.ast_targets:
        if not target.is_function_like:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        signal = CancelableCompositionSignalTargetAuthority(
            source_index,
            target,
            node,
        ).signal()
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


_NATIVE_ALGEBRAIC_VARIANT_QUERY = """
(attribute) @attribute
(return_statement) @return
"""


def _native_trimmed_function_statements(
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
) -> tuple[Node, ...]:
    body = function_node.child_by_field_name("body")
    if body is None:
        return ()
    statements = tuple(
        child for child in body.named_children if child.type != "comment"
    )
    if not statements or statements[0].type != "expression_statement":
        return statements
    expression_children = statements[0].named_children
    if expression_children and expression_children[0].type in {
        "concatenated_string",
        "string",
    }:
        return statements[1:]
    return statements


def _native_unwrapped_expression(node: Node | None) -> Node | None:
    while (
        node is not None
        and node.type == "parenthesized_expression"
        and len(node.named_children) == 1
    ):
        node = node.named_children[0]
    return node


def _native_function_decorators(
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
    cache: dict[Node, tuple[ast.expr, ...]],
) -> tuple[ast.expr, ...]:
    cached = cache.get(function_node)
    if cached is not None:
        return cached
    decorated = function_node.parent
    if decorated is None or decorated.type != "decorated_definition":
        cache[function_node] = ()
        return ()
    decorator_sources = tuple(
        syntax_index.source_for(child).decode("utf-8")
        for child in decorated.named_children
        if child.type == "decorator"
    )
    if not decorator_sources:
        cache[function_node] = ()
        return ()
    parsed = ast.parse("\n".join((*decorator_sources, "def _native(): pass")))
    function = parsed.body[-1]
    if not isinstance(function, ast.FunctionDef):
        raise TypeError("native decorators did not parse on a function")
    decorators = tuple(function.decorator_list)
    cache[function_node] = decorators
    return decorators


def _native_function_stub(
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
    body: tuple[ast.stmt, ...],
    decorators: tuple[ast.expr, ...] = (),
) -> _RuntimeFunctionNode:
    function_type: type[ast.FunctionDef] | type[ast.AsyncFunctionDef] = (
        ast.AsyncFunctionDef
        if syntax_index.source_for(function_node).lstrip().startswith(b"async ")
        else ast.FunctionDef
    )
    function = function_type(
        name=syntax_index.declared_name(function_node),
        args=syntax_index.arguments_for(function_node),
        body=list(body),
        decorator_list=list(decorators),
        returns=None,
        type_comment=None,
    )
    function.lineno = function_node.start_point.row + 1
    function.end_lineno = function_node.end_point.row + 1
    return function


def _native_composition_body(
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
) -> tuple[ast.stmt, ...] | None:
    body = function_node.child_by_field_name("body")
    statements = (
        ()
        if body is None
        else tuple(child for child in body.named_children if child.type != "comment")
    )
    if len(statements) == 1 and statements[0].type == "return_statement":
        returned = statements[0].named_children
        returned_value = _native_unwrapped_expression(returned[0] if returned else None)
        if returned_value is None or returned_value.type != "call":
            return None
    elif (
        len(statements) == 2
        and statements[0].type == "expression_statement"
        and statements[1].type == "return_statement"
    ):
        assignment_children = statements[0].named_children
        assignment = assignment_children[0] if assignment_children else None
        assigned_value = _native_unwrapped_expression(
            None if assignment is None else assignment.child_by_field_name("right")
        )
        if (
            assignment is None
            or assignment.type != "assignment"
            or assigned_value is None
            or assigned_value.type != "call"
            or not statements[1].named_children
        ):
            return None
    else:
        return None
    return tuple(syntax_index.statement_for(statement) for statement in statements)


def _native_composition_signal(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    function_node: Node,
    function: _RuntimeFunctionNode,
) -> CancelableCompositionSignal | None:
    file_path = source_module.path.as_posix()
    qualname = syntax_index.fully_qualified_function_name(function_node)
    node_kind = (
        AstTargetNodeKind.METHOD
        if any(
            scope.type == "class_definition"
            for scope in syntax_index.named_scope_nodes(function_node)
        )
        else AstTargetNodeKind.FUNCTION
    )
    line = function_node.start_point.row + 1
    end_line = function_node.end_point.row + 1
    target = AstTargetDigest(
        target_id=STABLE_ID_AUTHORITY.ast_target_id(
            file_path=file_path,
            node_kind=node_kind,
            qualname=qualname,
            line=line,
            end_line=end_line,
        ),
        file_id=STABLE_ID_AUTHORITY.file_id(file_path),
        file_path=file_path,
        node_type=node_kind.value,
        name=syntax_index.declared_name(function_node),
        qualname=qualname,
        line=line,
        end_line=end_line,
    )
    return CancelableCompositionSignalTargetAuthority(
        SourceIndex(ast_targets=(target,)),
        target,
        function,
    ).signal()


def _native_composition_signals(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    function_nodes: tuple[Node, ...],
) -> tuple[CancelableCompositionSignal, ...]:
    signals: list[CancelableCompositionSignal] = []
    for function_node in function_nodes:
        composition_body = _native_composition_body(
            syntax_index,
            function_node,
        )
        if composition_body is not None:
            composition_function = _native_function_stub(
                syntax_index,
                function_node,
                composition_body,
            )
            signal = _native_composition_signal(
                source_module,
                syntax_index,
                function_node,
                composition_function,
            )
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


def _native_variant_method_surfaces(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    function_nodes: tuple[Node, ...],
    decorators_by_function: dict[Node, tuple[ast.expr, ...]],
) -> tuple[_VariantMethodSurface, ...]:
    captures = syntax_index.captures(_NATIVE_ALGEBRAIC_VARIANT_QUERY)
    attributes_by_function: dict[Node, list[tuple[str, str]]] = defaultdict(list)
    returned_calls_by_function: dict[Node, list[Node]] = defaultdict(list)
    function_node_set = set(function_nodes)
    for attribute in captures.get("attribute", ()):
        owner = attribute.child_by_field_name("object")
        field_node = attribute.child_by_field_name("attribute")
        if owner is None or owner.type != "identifier" or field_node is None:
            continue
        owner_name = syntax_index.source_for(owner).decode("utf-8")
        field_name = syntax_index.source_for(field_node).decode("utf-8")
        for function_node in syntax_index.enclosing_function_nodes(attribute):
            if function_node in function_node_set:
                attributes_by_function[function_node].append((owner_name, field_name))
    for return_node in captures.get("return", ()):
        returned = return_node.named_children
        returned_value = _native_unwrapped_expression(returned[0] if returned else None)
        if returned_value is None or returned_value.type != "call":
            continue
        for function_node in syntax_index.enclosing_function_nodes(return_node):
            if function_node in function_node_set:
                returned_calls_by_function[function_node].append(returned_value)

    abstract_classes: set[Node] = set()
    for function_node in function_nodes:
        owner = syntax_index.direct_enclosing_class(function_node)
        if owner is None:
            continue
        decorators = _native_function_decorators(
            syntax_index,
            function_node,
            decorators_by_function,
        )
        if any(
            _decorator_simple_name(decorator) == "abstractmethod"
            for decorator in decorators
        ):
            abstract_classes.add(owner)

    class_header_by_node: dict[Node, ast.ClassDef] = {}
    surfaces: list[_VariantMethodSurface] = []
    for function_node in function_nodes:
        class_node = syntax_index.direct_enclosing_class(function_node)
        if class_node is None:
            continue
        if _is_private_symbol_name(syntax_index.declared_name(class_node)):
            continue
        method_name = syntax_index.declared_name(function_node)
        if _is_private_symbol_name(method_name) or method_name.startswith("__"):
            continue
        decorators = _native_function_decorators(
            syntax_index,
            function_node,
            decorators_by_function,
        )
        if not all(
            _decorator_simple_name(decorator) in _CLASSLEVEL_METHOD_DECORATORS
            for decorator in decorators
        ):
            continue
        statement_nodes = _native_trimmed_function_statements(
            syntax_index,
            function_node,
        )
        if len(statement_nodes) > 8:
            continue
        if len(SemanticTokenAuthority.identifier_tokens(method_name)) < 2:
            continue
        arguments = syntax_index.arguments_for(function_node)
        parameter_names = {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
            if argument.arg not in {"cls", "self"}
        }
        fields_by_parameter: dict[str, set[str]] = defaultdict(set)
        for owner_name, field_name in attributes_by_function.get(function_node, ()):
            if owner_name in parameter_names:
                fields_by_parameter[owner_name].add(field_name)
        if not any(len(fields) >= 2 for fields in fields_by_parameter.values()):
            continue
        if len(returned_calls_by_function.get(function_node, ())) != 1:
            continue
        body = tuple(
            syntax_index.statement_for(statement) for statement in statement_nodes
        )
        function = _native_function_stub(
            syntax_index,
            function_node,
            body,
            decorators,
        )
        class_header = class_header_by_node.get(class_node)
        if class_header is None:
            class_header = syntax_index.class_header_for(class_node)
            class_header_by_node[class_node] = class_header
        base_names = CLASS_NODE_AUTHORITY.declared_base_names(class_header)
        surface = _variant_method_surface_from_owner_facts(
            file_path=str(source_module.path),
            owner_class_name=syntax_index.declared_name(class_node),
            owner_line=class_node.start_point.row + 1,
            owner_is_abstract=(
                class_node in abstract_classes
                or bool({"ABC", "ABCMeta"} & set(base_names))
            ),
            owner_base_names=base_names,
            method=function,
        )
        if surface is not None:
            surfaces.append(surface)
    return sorted_tuple(
        surfaces,
        key=lambda item: (item.file_path, item.owner_line, item.line),
    )


def _native_algebraic_variant_projection(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> list[CompactAlgebraicVariantModuleProjection] | None:
    if not syntax_index.is_complete:
        return None
    try:
        function_nodes = syntax_index.common_captures().get("function", ())
        decorators_by_function: dict[Node, tuple[ast.expr, ...]] = {}
        composition_signals = _native_composition_signals(
            source_module,
            syntax_index,
            function_nodes,
        )
        return [
            CompactAlgebraicVariantModuleProjection(
                module_name=source_module.module_name,
                file_path=str(source_module.path),
                composition_signals=composition_signals,
                variant_method_surfaces=_native_variant_method_surfaces(
                    source_module,
                    syntax_index,
                    function_nodes,
                    decorators_by_function,
                ),
            )
        ]
    except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
        return None


class CompactAlgebraicVariantModuleProjectionFamily(
    CollectedFamily[CompactAlgebraicVariantModuleProjection]
):
    """Persist algebraic variant facts for exact repository-wide joins."""

    item_type = CompactAlgebraicVariantModuleProjection
    source_collector = staticmethod(_native_algebraic_variant_projection)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactAlgebraicVariantModuleProjection]:
        del cls
        return [
            CompactAlgebraicVariantModuleProjection(
                module_name=parsed_module.module_name,
                file_path=str(parsed_module.path),
                composition_signals=_cancelable_composition_signals_for_module(
                    parsed_module
                ),
                variant_method_surfaces=_variant_method_surfaces(parsed_module),
            )
        ]


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
    return _variant_method_surface_from_owner_facts(
        file_path=str(module.path),
        owner_class_name=class_node.name,
        owner_line=class_node.lineno,
        owner_is_abstract=CLASS_NODE_AUTHORITY.is_abstract(class_node),
        owner_base_names=CLASS_NODE_AUTHORITY.declared_base_names(class_node),
        method=method,
    )


def _variant_method_surface_from_owner_facts(
    *,
    file_path: str,
    owner_class_name: str,
    owner_line: int,
    owner_is_abstract: bool,
    owner_base_names: tuple[str, ...],
    method: _RuntimeFunctionNode,
) -> _VariantMethodSurface | None:
    """Project one method using the canonical body semantics and owner facts."""

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
        file_path=file_path,
        owner_class_name=owner_class_name,
        owner_line=owner_line,
        owner_is_abstract=owner_is_abstract,
        owner_base_names=owner_base_names,
        qualname=f"{owner_class_name}.{method.name}",
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
    return _VariantMethodFamilyCandidate(
        seed=seed,
        composition_signals=related_compositions,
    )


def _variant_method_family_candidates_from_compact_projections(
    projections: tuple[CompactAlgebraicVariantModuleProjection, ...],
) -> tuple[_VariantMethodFamilyCandidate, ...]:
    grouped: dict[
        tuple[str, str, str, tuple[str, ...], str],
        list[_VariantMethodSurface],
    ] = defaultdict(list)
    for projection in projections:
        for surface in projection.variant_method_surfaces:
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

    composition_signals = tuple(
        signal
        for projection in projections
        for signal in projection.composition_signals
    )
    candidates = [
        _variant_method_family_candidate(
            seed,
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


class AlgebraicVariantMethodFamilyDetector(
    CompactProjectionCandidateDetector[
        CompactAlgebraicVariantModuleProjection,
        _VariantMethodFamilyCandidate,
    ],
):
    module_projection_family = CompactAlgebraicVariantModuleProjectionFamily
    detector_priority = -15
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Algebraic variant method family inflates public authority surface",
        "A public authority class that grows sibling methods whose names encode operation variants is exporting the operation algebra in method names. If those methods share a product carrier/request parameter and forward to the same construction shape, the variant should live in a nominal context, request, or product type instead of multiplying public methods.",
        "one algebraic operation over a nominal context/request/product variant",
        "same owner exposes variant-named methods over the same product construction",
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidates_from_compact_projections(
        self,
        projections: tuple[CompactAlgebraicVariantModuleProjection, ...],
        config: DetectorConfig,
    ) -> Sequence[_VariantMethodFamilyCandidate]:
        del config
        return _variant_method_family_candidates_from_compact_projections(projections)

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
            "ABC/public authority" if exemplar.owner_is_abstract else "public authority"
        )
        composition_summary = ", ".join(
            signal.qualname for signal in candidate.composition_signals[:3]
        )
        extra_context = []
        if composition_summary:
            extra_context.append(
                f"cancelable product composition(s): {composition_summary}"
            )
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
                    2, len(seed.methods) + len(candidate.composition_signals)
                ),
                statement_count=max(method.statement_count for method in seed.methods),
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
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
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


@dataclass(frozen=True)
class ConstantBackedDispatchAxisCandidate(FunctionEvidenceLocationsCandidate):
    axis_name: str
    constant_prefix: str
    constant_names: tuple[str, ...]
    witness_name: ClassVar[AliasProperty[str]] = AliasProperty("axis_name")


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
            for (
                dispatch_axis_expression,
                constant_names,
            ) in _constant_backed_dispatch_tests(node.test):
                if not constant_names:
                    continue
                prefix_counts = Counter(
                    _constant_name_prefix(name) for name in constant_names
                )
                constant_prefix, count = prefix_counts.most_common(1)[0]
                if count != len(constant_names):
                    continue
                grouped[_axis_key(dispatch_axis_expression), constant_prefix].append(
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
        (
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.LITERAL_ID_DISPATCH,
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
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
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
        ),
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
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.DATAFLOW_ROOT,
        ),
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
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        (
            ObservationTag.PROJECTION_HELPER,
            ObservationTag.NORMALIZED_AST,
        ),
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
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.SCOPED_SHAPE_WRAPPER,
            ObservationTag.NORMALIZED_AST,
        ),
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
        (
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
        (
            ObservationTag.NORMALIZED_AST,
            ObservationTag.PARTIAL_VIEW,
        ),
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
        "Properties such as `source_value -> source.value` preserve a flattened shadow schema over an existing nested owner. Callers should use the nested owner directly so the nominal record remains the only authority.",
        "direct nested owner access instead of flattened compatibility aliases",
        "class exposes flattened fields as properties over nested nominal owners",
        (
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.KEYWORD_MAPPING,
            ObservationTag.NORMALIZED_AST,
        ),
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
                f"`{class_name}` keeps flattened compatibility properties {aliases} over nested nominal owners."
            ),
            evidence,
            scaffold=(
                "Delete the compatibility properties and update callers to use the nested nominal owner directly.\n\n"
                f"{examples}"
            ),
            codemod_patch=(
                f"# Remove flattened projection properties from `{class_name}`.\n"
                "# Rewrite call sites to the nested owner path shown in the scaffold."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(ordered),
                field_count=len({item.nested_access for item in ordered}),
                mapping_name=f"{class_name} flattened projection properties",
                field_names=tuple(item.property_name for item in ordered),
            ),
        )


class CompactPublicApiPrivateDelegateModuleProjectionFamily(
    CollectedFamily[PublicApiPrivateDelegateModuleFacts]
):
    item_type = PublicApiPrivateDelegateModuleFacts
    cache_payload_max_bytes = 1_000_000

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[PublicApiPrivateDelegateModuleFacts]:
        del cls
        return [_public_api_private_delegate_module_facts(parsed_module)]


@dataclass(frozen=True)
class CompactPublicApiPrivateDelegateProjectionDemand:
    """Wrapper symbols connected to report-target wrappers or call sites."""

    target_symbols: frozenset[str]


def _public_api_private_delegate_projection_demand(
    target_items: tuple[object, ...],
    config: object,
) -> CompactPublicApiPrivateDelegateProjectionDemand:
    del config
    projections = tuple(
        item
        for item in target_items
        if isinstance(item, PublicApiPrivateDelegateModuleFacts)
    )
    return CompactPublicApiPrivateDelegateProjectionDemand(
        target_symbols=frozenset(
            (
                *(
                    f"{projection.module_name}.{wrapper.qualname}"
                    for projection in projections
                    for wrapper in projection.wrappers
                ),
                *(
                    target
                    for projection in projections
                    for target, _callsites in projection.callsites_by_target
                ),
            )
        )
    )


def _delegate_demand_symbols_match(left: str, right: str) -> bool:
    return bool(
        left == right or left.endswith(f".{right}") or right.endswith(f".{left}")
    )


def _project_public_api_private_delegate_demand(
    items: tuple[object, ...],
    demand: object,
) -> tuple[object, ...]:
    if not isinstance(demand, CompactPublicApiPrivateDelegateProjectionDemand):
        return items
    projections = tuple(
        item for item in items if isinstance(item, PublicApiPrivateDelegateModuleFacts)
    )
    relevant_symbols = set(demand.target_symbols)
    for projection in projections:
        for wrapper in projection.wrappers:
            wrapper_symbol = f"{projection.module_name}.{wrapper.qualname}"
            if any(
                _delegate_demand_symbols_match(wrapper_symbol, target_symbol)
                for target_symbol in demand.target_symbols
            ):
                relevant_symbols.add(wrapper_symbol)
    projected: list[PublicApiPrivateDelegateModuleFacts] = []
    for projection in projections:
        wrappers = tuple(
            wrapper
            for wrapper in projection.wrappers
            if any(
                _delegate_demand_symbols_match(
                    f"{projection.module_name}.{wrapper.qualname}",
                    symbol,
                )
                for symbol in relevant_symbols
            )
        )
        callsites_by_target = tuple(
            (target, callsites)
            for target, callsites in projection.callsites_by_target
            if any(
                _delegate_demand_symbols_match(target, symbol)
                for symbol in relevant_symbols
            )
        )
        if wrappers or callsites_by_target:
            projected.append(
                PublicApiPrivateDelegateModuleFacts(
                    file_path=projection.file_path,
                    module_name=projection.module_name,
                    top_level_symbol_lines=(
                        projection.top_level_symbol_lines if wrappers else ()
                    ),
                    wrappers=wrappers,
                    callsites_by_target=callsites_by_target,
                )
            )
    return tuple(projected)


def _collect_public_api_private_delegate_ast_demand(
    parsed_module: ParsedModule,
    demand: object,
) -> list[object]:
    if isinstance(demand, CompactPublicApiPrivateDelegateProjectionDemand):
        terminal_names = frozenset(
            symbol.rsplit(".", 1)[-1] for symbol in demand.target_symbols
        )
        if not any(name in parsed_module.source for name in terminal_names):
            return []
    return list(
        _project_public_api_private_delegate_demand(
            tuple(
                CompactPublicApiPrivateDelegateModuleProjectionFamily.collect(
                    parsed_module
                )
            ),
            demand,
        )
    )


def _native_attribute_chain(
    syntax_index: NativePythonSyntaxIndex,
    node: Node | None,
) -> tuple[str, ...] | None:
    if node is None:
        return None
    if node.type == "identifier":
        return (syntax_index.source_for(node).decode("utf-8"),)
    if node.type == "parenthesized_expression":
        children = node.named_children
        return (
            _native_attribute_chain(syntax_index, children[0])
            if len(children) == 1
            else None
        )
    if node.type != "attribute":
        return None
    owner = _native_attribute_chain(
        syntax_index,
        node.child_by_field_name("object"),
    )
    attribute = node.child_by_field_name("attribute")
    if owner is None or attribute is None:
        return None
    return (*owner, syntax_index.source_for(attribute).decode("utf-8"))


def _native_import_aliases_for_delegate_demand(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> dict[str, str]:
    import_statements = tuple(
        syntax_index.statement_for(node)
        for node in syntax_index.tree.root_node.named_children
        if node.type in {"import_statement", "import_from_statement"}
    )
    import_module = source_module.parsed_module(
        ast.Module(body=list(import_statements), type_ignores=[]),
    )
    from ..class_index import _module_import_aliases

    return _module_import_aliases(import_module)


def _native_delegate_callsite_symbol(
    syntax_index: NativePythonSyntaxIndex,
    call_node: Node,
) -> str:
    scopes = list(syntax_index.named_scope_nodes(call_node))
    current = call_node.parent
    while current is not None:
        if current.type == "decorator":
            decorated = current.parent
            if decorated is not None and decorated.type == "decorated_definition":
                definition = next(
                    (
                        child
                        for child in decorated.named_children
                        if child.type in {"class_definition", "function_definition"}
                    ),
                    None,
                )
                if definition is not None:
                    scopes.append(definition)
            break
        current = current.parent
    function_name = next(
        (
            syntax_index.declared_name(scope)
            for scope in reversed(scopes)
            if scope.type == "function_definition"
        ),
        "<module>",
    )
    class_name = next(
        (
            syntax_index.declared_name(scope)
            for scope in reversed(scopes)
            if scope.type == "class_definition"
        ),
        None,
    )
    owner = function_name if class_name is None else f"{class_name}.{function_name}"
    return f"{owner}:call"


def _collect_public_api_private_delegate_source_demand(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    if not isinstance(demand, CompactPublicApiPrivateDelegateProjectionDemand):
        raise TypeError("public-delegate demand has the wrong authority type")
    if not syntax_index.is_complete:
        return None
    terminal_names = frozenset(
        symbol.rsplit(".", 1)[-1] for symbol in demand.target_symbols
    )
    if any(
        syntax_index.declared_name(function) in terminal_names
        for function in syntax_index.common_captures().get("function", ())
    ):
        # Wrapper classification depends on complete local inheritance.  Keep
        # the exact AST authority for the small possible-wrapper frontier.
        return None
    import_aliases = _native_import_aliases_for_delegate_demand(
        source_module,
        syntax_index,
    )
    callsites_by_target: dict[str, set[ResolvedExternalCallsite]] = defaultdict(set)
    for call in syntax_index.common_captures().get("call", ()):
        parts = _native_attribute_chain(
            syntax_index,
            call.child_by_field_name("function"),
        )
        if parts is None:
            continue
        first, *rest = parts
        alias_target = import_aliases.get(first)
        if alias_target is None:
            continue
        target = ".".join((alias_target, *rest)) if rest else alias_target
        if not any(
            _delegate_demand_symbols_match(target, symbol)
            for symbol in demand.target_symbols
        ):
            continue
        callsites_by_target[target].add(
            ResolvedExternalCallsite(
                module_name=source_module.module_name,
                location=SourceLocation(
                    str(source_module.path),
                    call.start_point.row + 1,
                    _native_delegate_callsite_symbol(syntax_index, call),
                ),
            )
        )
    if not callsites_by_target:
        return []
    return [
        PublicApiPrivateDelegateModuleFacts(
            file_path=str(source_module.path),
            module_name=source_module.module_name,
            top_level_symbol_lines=(),
            wrappers=(),
            callsites_by_target=tuple(
                (
                    target,
                    sorted_tuple(
                        callsites,
                        key=lambda item: (
                            item.location.file_path,
                            item.location.line,
                            item.location.symbol,
                            item.module_name,
                        ),
                    ),
                )
                for target, callsites in sorted(callsites_by_target.items())
            ),
        )
    ]


CompactPublicApiPrivateDelegateModuleProjectionFamily.report_demand_builder = (
    staticmethod(_public_api_private_delegate_projection_demand)
)
CompactPublicApiPrivateDelegateModuleProjectionFamily.ast_demand_collector = (
    staticmethod(_collect_public_api_private_delegate_ast_demand)
)
CompactPublicApiPrivateDelegateModuleProjectionFamily.source_demand_collector = (
    staticmethod(_collect_public_api_private_delegate_source_demand)
)
CompactPublicApiPrivateDelegateModuleProjectionFamily.cached_demand_projector = (
    staticmethod(_project_public_api_private_delegate_demand)
)


@dataclass(frozen=True)
class CompactPublicApiPrivateDelegateContext:
    shell_candidates: tuple[PublicApiPrivateDelegateShellCandidate, ...]
    family_candidates: tuple[PublicApiPrivateDelegateFamilyCandidate, ...]

    @classmethod
    def from_projections(
        cls,
        projections: tuple[PublicApiPrivateDelegateModuleFacts, ...],
        config: DetectorConfig,
    ) -> "CompactPublicApiPrivateDelegateContext":
        return cls(
            shell_candidates=_public_api_private_delegate_shell_candidates_from_facts(
                projections, config
            ),
            family_candidates=_public_api_private_delegate_family_candidates_from_facts(
                projections, config
            ),
        )


PublicApiPrivateDelegateCandidateT = TypeVar("PublicApiPrivateDelegateCandidateT")


class _CompactPublicApiPrivateDelegateDetectorBase(
    CompactContextCandidateDetector[
        PublicApiPrivateDelegateModuleFacts,
        CompactPublicApiPrivateDelegateContext,
        PublicApiPrivateDelegateCandidateT,
    ],
    Generic[PublicApiPrivateDelegateCandidateT],
):
    module_projection_family = CompactPublicApiPrivateDelegateModuleProjectionFamily
    compact_shared_context_builder = staticmethod(
        CompactPublicApiPrivateDelegateContext.from_projections
    )

    @classmethod
    def _compact_context_from_projections(
        cls,
        projections: tuple[PublicApiPrivateDelegateModuleFacts, ...],
        config: DetectorConfig,
    ) -> CompactPublicApiPrivateDelegateContext:
        return CompactPublicApiPrivateDelegateContext.from_projections(
            projections, config
        )

    @classmethod
    def _compact_context_from_shared(
        cls,
        context: object | None,
    ) -> CompactPublicApiPrivateDelegateContext:
        if not isinstance(context, CompactPublicApiPrivateDelegateContext):
            raise TypeError("compact public/private delegate context is unavailable")
        return context


class PublicApiPrivateDelegateShellDetector(
    _CompactPublicApiPrivateDelegateDetectorBase[
        PublicApiPrivateDelegateShellCandidate
    ],
):
    ssot_authority_boundary = True
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Public API shell over a private delegate should promote a public authority",
        "A public module-level wrapper is carrying an external API contract only because the real implementation authority is hidden behind a private `_X` root. When multiple external call sites depend on that shell, the docs prefer promoting one public facade/ABC/policy authority instead of inlining callers onto the private delegate.",
        "public authoritative facade over a private delegate family",
        "external modules depend on a public forwarding shell because the true authority is private",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactPublicApiPrivateDelegateContext,
        config: DetectorConfig,
    ) -> Sequence[PublicApiPrivateDelegateShellCandidate]:
        del config
        return context.shell_candidates

    def _findings_for_candidates(
        self,
        candidates: Sequence[PublicApiPrivateDelegateShellCandidate],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for candidate in candidates:
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


class PublicApiPrivateDelegateFamilyDetector(
    _CompactPublicApiPrivateDelegateDetectorBase[
        PublicApiPrivateDelegateFamilyCandidate
    ],
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Multiple public shells over one private delegate should collapse into a public facade family",
        "When several public wrappers expose one private delegate root, the external API is fragmented across transport shells instead of owned by one public authority. The docs prefer promoting a public facade, ABC, or policy surface rather than keeping multiple pass-through exports over private machinery.",
        "single public facade family over one private delegate root",
        "multiple public wrappers expose one private delegate family to external modules",
        (
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        (
            ObservationTag.ACCESSOR_WRAPPER,
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactPublicApiPrivateDelegateContext,
        config: DetectorConfig,
    ) -> Sequence[PublicApiPrivateDelegateFamilyCandidate]:
        del config
        return context.family_candidates

    def _findings_for_candidates(
        self,
        candidates: Sequence[PublicApiPrivateDelegateFamilyCandidate],
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        for candidate in candidates:
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
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
        ),
        (
            ObservationTag.INTERFACE_IDENTITY,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
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
