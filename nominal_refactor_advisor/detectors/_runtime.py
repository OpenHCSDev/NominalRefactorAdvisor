"""Runtime and wrapper detector implementations.

This module groups detector classes around builder duplication, runtime
selection, wrapper surfaces, and dynamic dispatch residue.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Callable, Generic, TypeAlias, TypeVar

from tree_sitter import Node

from ..ast_tools import (
    CollectedFamily,
    CompactModuleIdentity,
    ParsedModule,
    PythonSourcePathPolicy,
    SourceModule,
    collect_family_items,
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
from ..models import HierarchyCandidateMetrics, RefactorFinding, SourceLocation
from ..patterns import PatternId
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
                SourceLocation(module.file_path, line, f"{class_name}.{field_name}")
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
        if not constants:
            return ()
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
                file_path=parsed_module.file_path,
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
                file_path=source_module.file_path,
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
            module.file_path,
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
                source_module.file_path,
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
    bridge_kind: RuntimeNamespaceBridgeKind


class RuntimeNamespaceBridgeKind(StrEnum):
    """Exact runtime namespace compatibility mechanisms witnessed in source."""

    RUNTIME_BRIDGE_IMPORT = "runtime_bridge_namespace import"
    GLOBALS_UPDATE = "globals update"
    RUNTIME_BRIDGE_GLOBALS_UPDATE = "runtime_bridge_namespace globals update"
    GUARDED_GLOBALS_DEFINITION = "guarded globals definition"


_GLOBALS_BUILTIN_NAME = "globals"
_RUNTIME_BRIDGE_NAMESPACE_NAME = "runtime_bridge_namespace"


def _runtime_namespace_bridge_source_may_match(source: str) -> bool:
    """Return whether source contains a symbol required by every bridge witness."""

    return (
        _GLOBALS_BUILTIN_NAME in source
        or _RUNTIME_BRIDGE_NAMESPACE_NAME in source
    )


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
        and node.func.id == _GLOBALS_BUILTIN_NAME
        and not node.args
        and not node.keywords
    )


def _is_runtime_bridge_namespace_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_symbol(node.func).endswith(
        _RUNTIME_BRIDGE_NAMESPACE_NAME
    )


def _runtime_namespace_bridge_kind_for_call(
    node: ast.Call,
) -> RuntimeNamespaceBridgeKind | None:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and _is_globals_call(node.func.value)
    ):
        return None
    if any(_is_runtime_bridge_namespace_call(argument) for argument in node.args):
        return RuntimeNamespaceBridgeKind.RUNTIME_BRIDGE_GLOBALS_UPDATE
    if (
        len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name | ast.Attribute)
    ):
        return RuntimeNamespaceBridgeKind.GLOBALS_UPDATE
    return None


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
        if not _runtime_namespace_bridge_source_may_match(module.source):
            return []
        sites: list[RuntimeNamespaceBridgeSite] = []

        class Visitor(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                for alias in node.names:
                    if alias.name == _RUNTIME_BRIDGE_NAMESPACE_NAME:
                        sites.append(
                            RuntimeNamespaceBridgeSite(
                                line=int(node.lineno),
                                symbol=alias.asname or alias.name,
                                bridge_kind=(
                                    RuntimeNamespaceBridgeKind.RUNTIME_BRIDGE_IMPORT
                                ),
                            )
                        )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                bridge_kind = _runtime_namespace_bridge_kind_for_call(node)
                if bridge_kind is not None:
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
                            bridge_kind=(
                                RuntimeNamespaceBridgeKind.GUARDED_GLOBALS_DEFINITION
                            ),
                        )
                    )
                self.generic_visit(node)

        Visitor().visit(module.module)
        if not sites:
            return []
        bridge_kinds = sorted_tuple(site.bridge_kind.value for site in sites)
        evidence = tuple(
            SourceLocation(module.file_path, site.line, site.symbol)
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
                        module.file_path,
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
            if roster.is_public_export_surface:
                continue
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
        "A metaclass registry pays rent when it derives a semantic family membership surface: a stable key axis, multiple registered leaves, and a behavioral or abstract contract. Explicit registry projections and consumers strengthen that proof. Without those coordinates, the metaclass is mostly signature noise and the same information usually belongs in a typed declaration table, enum, or ordinary ABC.",
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
                "# Rent proof must expose a stable key axis, multiple registered leaves, and a behavioral contract.\n"
                "# Prefer an explicit registry projection/consumer derived from `cls.__registry__` when the family is enumerated.\n"
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
                    file_path=module.file_path,
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


_RuntimeFunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef
_SurfaceFunctionItems: TypeAlias = tuple[tuple[str, _RuntimeFunctionNode], ...]


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


def _trimmed_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.stmt, ...]:
    return tuple(_trim_docstring_body(function.body))


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
    file_path = source_module.file_path
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
            file_path=source_module.file_path,
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
                file_path=source_module.file_path,
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
                file_path=parsed_module.file_path,
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
        file_path=module.file_path,
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
                    file_path=module.file_path,
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
    for node in walk_function_body_nodes(function):
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
                        file_path=module.file_path,
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
            SourceLocation(module.file_path, pair.spec_line, pair.spec_name)
            for pair in wrapper_pairs[:6]
        ]
        evidence_items.extend(
            (
                SourceLocation(module.file_path, pair.function_line, pair.function_name)
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
                    SourceLocation(module.file_path, item.lineno, item.function_name)
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
