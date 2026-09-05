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
from typing import Generic, TypeAlias, TypeVar, cast

from tree_sitter import Node

from ._regex_bundle import RepeatedLocalRegexBundleDetector

from ..ast_tools import (
    CollectedFamily,
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
    CompactManualSubclassRosterRoot,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactRepositoryPublicExposureIndex,
    LatentRosterMatch,
    LatentRosterObservation,
    build_compact_class_family_index,
)
from ..exact_method_authority import (
    ParallelMirroredLeafFamilyComponent,
    ParallelMirroredLeafFamilyComponentBuilder,
)
from ..enum_keyed_query import (
    EnumKeyedDerivedMapFacadeComponent,
    EnumKeyedDerivedMapFacadeModuleProjection,
    EnumKeyedDerivedMapFacadeModuleProjectionFamily,
)
from ..codemod import (
    AutoRegisterMetaUnderRentedFindingRecipeEvaluator,
    EnumKeyedDerivedMapFacadeFindingRecipeSynthesizer,
    ManualClassRegistrationFindingRecipeSynthesizer,
    NumericLiteralDispatchFindingRecipeSynthesizer,
    ParallelMirroredLeafFamilyFindingRecipeSynthesizer,
    RepeatedBuilderCallFindingRecipeSynthesizer,
    SemanticMirrorFindingRecipeEvaluator,
    TypeKeyedBehaviorProjectionFindingRecipeSynthesizer,
)
from ..collection_algebra import sorted_tuple
from ..models import (
    AutoRegisterMetaRentMetrics,
    AutoRegisterMetaRentSignal,
    ConstructorOwnedMappingMetrics,
    HierarchyCandidateMetrics,
    RefactorFinding,
    SourceLocation,
)
from ..patterns import PatternId
from ..semantic_identity import (
    InheritanceIdentityAttributeProjection,
    SemanticIdentifierTokenProjection,
    SemanticRoleIdentityToken,
)
from ..taxonomy import CapabilityTag, ObservationTag
from ..type_keyed_behavior import (
    TypeKeyedBehaviorProjectionComponent,
    TypeKeyedBehaviorProjectionComponentBuilder,
)
from ._base import *
from ._base import high_confidence_certified_spec
from ._helpers import *


def _literal_dispatch_authority_patch(
    observation: LiteralDispatchObservation,
) -> str:
    return f"# Replace the repeated `{observation.dispatch_axis_expression} == literal` branches with one AutoRegisterMeta-backed case family.\n# Move per-case behavior into `DispatchCase` subclasses keyed by the same axis.\n# Dispatch through `DispatchCase.for_case(...)` / `DispatchCase.__registry__` instead of if/elif or match/case."


class LiteralDispatchFindingFactory:
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


class FormalBoundaryStringRegistryAuthority:
    @classmethod
    def module_constants(
        cls,
        module: ParsedModule,
    ) -> tuple[FormalBoundaryStringRegistryConstant, ...]:
        statements = tuple(
            statement
            for statement in module.module.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
        )
        constants = cls.constants_from_statements(statements)
        if not constants:
            return ()
        calls = tuple(
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.Call) and cls.is_registry_call(node)
        )
        if not cls.has_formal_boundary_consumer(calls, constants):
            return ()
        return constants

    @classmethod
    def constants_from_statements(
        cls,
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
                for target_name in cls.target_names(target):
                    if cls.is_registry_constant(target_name, value):
                        constants.append(
                            FormalBoundaryStringRegistryConstant(
                                target_name=target_name,
                                value=value,
                                line=statement.lineno,
                            )
                        )
        return tuple(constants)

    @classmethod
    def has_formal_boundary_consumer(
        cls,
        calls: Sequence[ast.Call],
        constants: tuple[FormalBoundaryStringRegistryConstant, ...],
    ) -> bool:
        constant_names = frozenset(constant.target_name for constant in constants)
        if not constant_names:
            return False
        return any(
            cls.call_consumes_constant(node, constant_names)
            for node in calls
        )

    @staticmethod
    def is_registry_call(node: ast.Call) -> bool:
        call_name = ast.unparse(node.func).lower()
        return any(
            token in call_name
            for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
        )

    @staticmethod
    def target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(
                element.id for element in target.elts if isinstance(element, ast.Name)
            )
        return ()

    @staticmethod
    def is_registry_constant(target_name: str, value: str) -> bool:
        boundary_tokens = frozenset(
            (
                *SemanticIdentifierTokenProjection.project(target_name),
                *SemanticIdentifierTokenProjection.project(value),
            )
        )
        return bool(
            boundary_tokens & _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
        ) and bool(boundary_tokens & _FORMAL_BOUNDARY_STRING_ID_TOKENS)

    @staticmethod
    def call_consumes_constant(
        node: ast.Call,
        constant_names: frozenset[str],
    ) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in constant_names
            for child in ast.walk(node)
        )

    @classmethod
    def source_constants(
        cls,
        source_module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
    ) -> list[FormalBoundaryPythonStringConstant] | None:
        """Collect formal-boundary constants without materializing a module AST."""

        if not syntax_index.is_complete:
            return None
        try:
            statements = tuple(
                syntax_index.statement_for(node)
                for node in syntax_index.top_level_assignment_statements()
                if cls.native_assignment_may_declare_constant(syntax_index, node)
            )
            constants = cls.constants_from_statements(statements)
            if not constants:
                return []
            constant_names = frozenset(
                constant.target_name for constant in constants
            )
            calls: list[ast.Call] = []
            for call_node in sorted(
                syntax_index.common_captures().get("call", ()),
                key=lambda node: (node.start_byte, -node.end_byte),
            ):
                function = call_node.child_by_field_name("function")
                if function is None:
                    continue
                function_source = (
                    syntax_index.source_for(function).decode("utf-8").lower()
                )
                if not any(
                    token in function_source
                    for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
                ):
                    continue
                expression = syntax_index.expression_for(call_node)
                if not isinstance(expression, ast.Call):
                    return None
                if cls.is_registry_call(expression) and cls.call_consumes_constant(
                    expression,
                    constant_names,
                ):
                    calls.append(expression)
                    break
            if not cls.has_formal_boundary_consumer(calls, constants):
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

    @staticmethod
    def native_assignment_may_declare_constant(
        syntax_index: NativePythonSyntaxIndex,
        statement: Node,
    ) -> bool:
        statement_source = syntax_index.source_for(statement).decode("utf-8")
        lowered_source = statement_source.lower()
        return bool(statement_source) and any(
            token in lowered_source
            for token in _FORMAL_BOUNDARY_LITERAL_REGISTRY_CALL_TOKENS
        ) and any(
            token in lowered_source for token in _FORMAL_BOUNDARY_STRING_ID_TOKENS
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
        FormalBoundaryStringRegistryAuthority.source_constants
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
        for token in SemanticIdentifierTokenProjection.project(part)
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
        projection_evidence = _formal_boundary_python_evidence_for_values(
            constants_by_value,
            shared_values[:6],
        )
        authority_evidence = _formal_boundary_external_evidence_for_values(
            sites_by_value,
            shared_values[:6],
        )
        return detector.build_finding(
            FormalBoundaryExternalStringRegistryMirrorAuthority.summary(
                path,
                shared_values,
            ),
            projection_evidence + authority_evidence,
            projection_evidence=projection_evidence[0],
            authority_evidence=authority_evidence[0],
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=2,
                field_names=shared_values,
                mapping_name="formal_boundary_external_string_registry",
                source_name=str(path),
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
    SemanticMirrorFindingRecipeEvaluator,
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
    def source_sites(
        cls,
        source_module: SourceModule,
        syntax_index: NativePythonSyntaxIndex,
    ) -> list[GeneratedBoundarySemanticConstantSite] | None:
        """Collect top-level semantic constants without materializing an AST."""

        if not syntax_index.is_complete:
            return None
        try:
            statements = tuple(
                syntax_index.statement_for(node)
                for node in syntax_index.top_level_assignment_statements()
                if cls.native_assignment_may_declare_constant(syntax_index, node)
            )
            return list(
                cls.sites_from_statements(
                    source_module.file_path,
                    statements,
                    cls.source_is_generated_boundary(
                        source_module.module_name,
                        source_module.path,
                        source_module.source,
                    ),
                )
            )
        except (SyntaxError, UnicodeDecodeError, ValueError, TypeError):
            return None

    @classmethod
    def native_assignment_may_declare_constant(
        cls,
        syntax_index: NativePythonSyntaxIndex,
        statement: Node,
    ) -> bool:
        """Return whether native syntax can contain an owned constant site."""

        pending = list(statement.named_children)
        while pending:
            node = pending.pop()
            if node.type == "assignment":
                target = node.child_by_field_name("left")
                if target is not None and target.type == "identifier":
                    name = syntax_index.source_for(target).decode("utf-8")
                    if cls.is_semantic_constant_name(name):
                        return True
            pending.extend(node.named_children)
        return False

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

    @classmethod
    def module_is_generated_boundary(cls, module: ParsedModule) -> bool:
        return cls.source_is_generated_boundary(
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
            for token in SemanticIdentifierTokenProjection.project(part)
        )
        if path_tokens & _GENERATED_BOUNDARY_TOKENS:
            return True
        return any(
            line.lstrip().startswith("#")
            and bool(
                frozenset(SemanticIdentifierTokenProjection.project(line))
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
        authority_evidence = generated_sites[0].source_location()
        projection_evidence = runtime_sites[0].source_location()
        evidence = (
            authority_evidence,
            projection_evidence,
        )
        return detector.build_finding(
            (
                f"`{target_name}` mirrors generated semantic constant value "
                f"{value!r} across generated and non-generated Python modules."
            ),
            evidence,
            projection_evidence=projection_evidence,
            authority_evidence=authority_evidence,
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(matching_sites),
                field_names=(target_name, value),
                mapping_name="generated_boundary_semantic_constant",
                source_name=target_name,
            ),
        )


class GeneratedBoundarySemanticConstantSiteFamily(
    CollectedFamily[GeneratedBoundarySemanticConstantSite]
):
    """Persist compact module facts used by the generated-boundary detector."""

    item_type = GeneratedBoundarySemanticConstantSite
    report_presence_predicate = staticmethod(lambda items, config: bool(items))
    source_collector = staticmethod(
        GeneratedBoundarySemanticConstantAuthority.source_sites
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


class GeneratedBoundarySemanticConstantMirrorDetector(
    CompactModuleProjectionDetectorMixin[GeneratedBoundarySemanticConstantSite],
    SemanticMirrorIssueDetector,
    SemanticMirrorFindingRecipeEvaluator,
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

    return _GLOBALS_BUILTIN_NAME in source or _RUNTIME_BRIDGE_NAMESPACE_NAME in source


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
                capability_gap="no runtime namespace bridge or guarded globals definition remains",
            )
        ]


def _stable_text_digest(value: str) -> str:
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()


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
    """Exact mapping keys capable of producing a repeated-builder finding."""

    exact_mapping_keys: frozenset[tuple[str, str, tuple[str, ...], tuple[str, ...]]]

    @classmethod
    def from_report_targets(
        cls,
        target_items: tuple[object, ...],
    ) -> "RepeatedBuilderCallProjectionDemand":
        """Derive exact repository mapping keys from report-target shapes."""

        return cls(
            exact_mapping_keys=frozenset(
                (
                    builder.file_path,
                    builder.callee_name,
                    builder.field_names,
                    builder.value_fingerprint,
                )
                for builder in target_items
                if isinstance(builder, BuilderCallShape)
            )
        )

    def includes(self, builder: BuilderCallShape) -> bool:
        """Return whether one builder participates in the demanded mapping."""

        return (
            builder.file_path,
            builder.callee_name,
            builder.field_names,
            builder.value_fingerprint,
        ) in self.exact_mapping_keys

    def project(self, items: tuple[object, ...]) -> tuple[BuilderCallShape, ...]:
        """Project cached or freshly collected shapes through this demand."""

        return tuple(
            item
            for item in items
            if isinstance(item, BuilderCallShape) and self.includes(item)
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
    )
    return list(
        demand.project(tuple(_module_builder_call_shapes(module, callee_names)))
    )


def _collect_repeated_builder_call_source_demand(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    if not isinstance(demand, RepeatedBuilderCallProjectionDemand):
        raise TypeError("repeated-builder demand has the wrong authority type")
    builders = _native_repeated_builder_call_shapes(source_module, syntax_index)
    if builders is None:
        return None
    return list(demand.project(tuple(builders)))


class RepeatedBuilderCallShapeProjectionFamily(CollectedFamily[BuilderCallShape]):
    """Persist normalized builder calls for repository-wide grouping."""

    item_type = BuilderCallShape
    cache_payload_max_bytes = 3_000_000
    source_collector = staticmethod(_native_repeated_builder_call_shapes)
    source_demand_collector = staticmethod(_collect_repeated_builder_call_source_demand)
    ast_demand_collector = staticmethod(_collect_repeated_builder_call_ast_demand)

    @classmethod
    def report_demand(
        cls,
        target_items: tuple[object, ...],
        config: object,
    ) -> RepeatedBuilderCallProjectionDemand:
        """Derive the repeated-builder keys needed by the report scope."""

        del cls, config
        return RepeatedBuilderCallProjectionDemand.from_report_targets(target_items)

    @classmethod
    def project_cached_demand(
        cls,
        items: tuple[object, ...],
        demand: object,
    ) -> tuple[BuilderCallShape, ...]:
        """Project cached builder shapes through their exact demand."""

        del cls
        if not isinstance(demand, RepeatedBuilderCallProjectionDemand):
            raise TypeError("repeated-builder demand has the wrong authority type")
        return demand.project(items)

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
    SsotAuthorityBoundaryDetector,
    RepeatedBuilderCallFindingRecipeSynthesizer,
):
    module_projection_family = RepeatedBuilderCallShapeProjectionFamily
    detector_id = "repeated_builder_calls"
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
        return self._exact_mapping_findings(builders, config)

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
                builder.source_location for builder in ordered[:6]
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
                    projection_evidence=ordered[0].source_location,
                    metrics=ConstructorOwnedMappingMetrics.from_field_names(
                        mapping_site_count=len(ordered),
                        mapping_name=ordered[0].callee_name,
                        field_names=ordered[0].field_names,
                        source_name=ordered[0].source_name,
                        identity_field_names=ordered[0].identity_field_names,
                    ),
                )
            )
        return findings


class ManualClassRegistrationDetector(
    CompactGroupedShapeIssueDetector[RegistrationShape, str],
    ManualClassRegistrationFindingRecipeSynthesizer,
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
            metrics=RegistrationMetrics(
                registration_site_count=len(registrations),
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
    latent_rosters: tuple[LatentRosterObservation, ...]
    parallel_mirrored_leaf_family_builder: ParallelMirroredLeafFamilyComponentBuilder


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
        latent_rosters=tuple(
            roster for projection in projections for roster in projection.latent_rosters
        ),
        parallel_mirrored_leaf_family_builder=(
            ParallelMirroredLeafFamilyComponentBuilder.from_projections(
                projections,
                class_index=class_index,
            )
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
        return self.build_finding(
            (
                f"`{roster_candidate.class_name}` maintains roster `{roster_candidate.registry_name}` for {len(roster_candidate.concrete_class_names)} concrete subclasses ({concrete_preview}){guard_summary} and consumes it via {roster_candidate.consumer_names}."
            ),
            tuple(evidence[:6]),
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
        return self.build_finding(
            (
                f"`{roster.roster_name}` is a `{roster.roster_kind}` roster "
                f"{roster.member_names} via `{roster.projection_role}` "
                f"covering {match.coverage_ratio:.2f} of concrete `{roster_candidate.class_name}` "
                f"implementations {roster_candidate.concrete_class_names}; derive it from registry key `{key_attr}`"
                f"{projection_suffix}."
            ),
            (roster_candidate.evidence,),
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
            for key_attr_name in InheritanceIdentityAttributeProjection.common_names(
                tuple(
                    descendant.direct_non_none_assignment_names
                    for descendant in descendants
                )
            )
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


def _compact_parallel_mirrored_leaf_family_candidates(
    context: _CompactConcreteFamilyContext,
    config: DetectorConfig,
) -> tuple[ParallelMirroredLeafFamilyComponent, ...]:
    builder = context.parallel_mirrored_leaf_family_builder
    return builder.proven_components(
        min_shared_roles=max(
            builder.minimum_product_role_count,
            config.min_registration_sites,
        ),
    )


def _type_keyed_behavior_projection_components(
    repository: CompactClassRepositoryContext,
) -> tuple[TypeKeyedBehaviorProjectionComponent, ...]:
    builder = TypeKeyedBehaviorProjectionComponentBuilder.from_projections(
        repository.projections,
        repository.class_index,
    )
    return builder.proven_components()


def _enum_keyed_derived_map_facade_components(
    projections_by_family: dict[type[CollectedFamily], tuple[object, ...]],
) -> tuple[EnumKeyedDerivedMapFacadeComponent, ...]:
    facade_projections = cast(
        tuple[EnumKeyedDerivedMapFacadeModuleProjection, ...],
        projections_by_family[EnumKeyedDerivedMapFacadeModuleProjectionFamily],
    )
    class_projections = cast(
        tuple[CompactModuleClassProjection, ...],
        projections_by_family[CompactModuleClassProjectionFamily],
    )
    exposure_authority = CompactRepositoryPublicExposureIndex(class_projections)
    return sorted_tuple(
        (
            component
            for projection in facade_projections
            for component in projection.components
            if all(
                exposure_authority.star_imports_exclude(
                    projection.module_name,
                    binding_name,
                )
                for binding_name in component.star_import_exclusion_names
            )
        ),
        key=lambda component: (
            component.file_path,
            component.enum_symbol,
            component.map_owner_symbol,
            component.map_method_name,
        ),
    )


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
        if missing_rent_signals == (
            AutoRegisterMetaRentSignal.REGISTERED_LEAF_AXIS,
        ) and (
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
    AutoRegisterMetaUnderRentedFindingRecipeEvaluator,
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
                f"{tuple(signal.value for signal in rent_candidate.missing_rent_signals)}. "
                f"Rent margin {rent_candidate.rent_margin}."
            ),
            (rent_candidate.evidence,),
            compression_certificate=rent_candidate.compression_certificate,
            metrics=AutoRegisterMetaRentMetrics(
                registration_site_count=len(rent_candidate.concrete_class_names),
                registry_name=rent_candidate.class_name,
                class_names=rent_candidate.concrete_class_names,
                missing_signals=rent_candidate.missing_rent_signals,
                rent_margin=rent_candidate.rent_margin,
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
        )


class ParallelMirroredLeafFamilyDetector(
    _CompactConcreteFamilyDetectorBase[ParallelMirroredLeafFamilyComponent],
    ParallelMirroredLeafFamilyFindingRecipeSynthesizer,
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Parallel mirrored leaf families should factor into a multiple-inheritance product",
        "Mirrored registered leaf catalogs repeat the same role behavior across domain roots. The role behavior belongs on one reusable mixin axis, while each domain root retains its nominal identity; concrete products compose those independent authorities through multiple inheritance.",
        "one authoritative role-behavior mixin axis composed with domain roots through multiple inheritance",
        "registered abstract roots own mirrored concrete leaf catalogs over the same contract method family",
        (
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.MRO_ORDERING,
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
    ) -> Sequence[ParallelMirroredLeafFamilyComponent]:
        return _compact_parallel_mirrored_leaf_family_candidates(
            context,
            config,
        )

    def _finding_for_candidate(
        self, mirrored_candidate: ParallelMirroredLeafFamilyComponent
    ) -> RefactorFinding:
        shared_preview = ", ".join(mirrored_candidate.shared_leaf_family_names[:4])
        contract_preview = ", ".join(mirrored_candidate.contract_method_names)
        root_names = tuple(root.simple_name for root in mirrored_candidate.roots)
        class_names = (
            *root_names,
            *(
                indexed_class.simple_name
                for indexed_class in mirrored_candidate.leaf_classes
            ),
        )
        evidence = mirrored_candidate.evidence_locations
        return self.build_finding(
            (
                f"{', '.join(f'`{root_name}`' for root_name in root_names)} expose "
                f"mirrored `{contract_preview}` leaf catalogs "
                f"across {len(mirrored_candidate.shared_leaf_family_names)} shared role families ({shared_preview})."
            ),
            evidence,
            authority_evidence=evidence[0],
            metrics=RegistrationMetrics.from_class_names(
                registration_site_count=len(mirrored_candidate.leaf_classes),
                registry_name="/".join(root_names),
                class_names=class_names,
            ),
        )


class TypeKeyedBehaviorProjectionDetector(
    CompactClassRepositoryCandidateDetector[TypeKeyedBehaviorProjectionComponent],
    SsotAuthorityBoundaryDetector,
    TypeKeyedBehaviorProjectionFindingRecipeSynthesizer,
):
    finding_spec = high_confidence_spec(
        PatternId.TYPE_NAMESPACE_INJECTION,
        "Type-keyed behavior projection should descend to its nominal hierarchy",
        "An external registered class family repeats behavior for an injectively mapped type hierarchy. The mapped types already own the runtime distinction and MRO fallback, so their namespaces should own the behavior directly.",
        "behavior owned once by the mapped nominal type hierarchy and selected through ordinary MRO",
        "AutoRegister leaves duplicate one behavior contract while their registry keys reproduce an existing nominal type hierarchy",
        (
            CapabilityTag.SHARED_TYPE_NAMESPACE,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REGISTRY_POPULATION,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> Sequence[TypeKeyedBehaviorProjectionComponent]:
        del config
        return context.cached(
            TypeKeyedBehaviorProjectionComponentBuilder,
            lambda: _type_keyed_behavior_projection_components(context),
        )

    def _finding_for_candidate(
        self,
        component: TypeKeyedBehaviorProjectionComponent,
    ) -> RefactorFinding:
        method_names = ", ".join(
            f"`{method_name}`" for method_name in component.behavior_method_names
        )
        return self.build_finding(
            (
                f"`{component.projection_root.simple_name}` maps {len(component.bindings)} "
                f"registered leaves onto `{component.target_root.simple_name}` and its "
                f"descendants while repeating {method_names}; the mapped type hierarchy "
                "already supplies the behavior dispatch relation."
            ),
            component.evidence_locations,
            projection_evidence=component.projection_evidence,
            authority_evidence=component.authority_evidence,
        )


class EnumKeyedDerivedMapFacadeDetector(
    CompactMultiProjectionCandidateDetector[EnumKeyedDerivedMapFacadeComponent],
    SsotAuthorityBoundaryDetector,
    EnumKeyedDerivedMapFacadeFindingRecipeSynthesizer,
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Enum-keyed query facade should descend to its nominal key",
        "A class-owned derived map exposes lookup behavior keyed by an enum while callers and a reverse query interpret the enum identity externally.",
        "key-facing query behavior owned by the enum and backed by the existing derived map",
        "a typed derived map, reverse query, and direct indexed consumer prove the displaced query surface",
        (
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_TYPE_NAMESPACE,
        ),
        (
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )
    module_projection_families = (
        EnumKeyedDerivedMapFacadeModuleProjectionFamily,
        CompactModuleClassProjectionFamily,
    )

    def _candidates_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: object | None,
        config: DetectorConfig,
    ) -> Sequence[EnumKeyedDerivedMapFacadeComponent]:
        del context, config
        return _enum_keyed_derived_map_facade_components(projections_by_family)

    def _finding_for_candidate(
        self,
        component: EnumKeyedDerivedMapFacadeComponent,
    ) -> RefactorFinding:
        enum_name = component.enum_symbol.rsplit(".", maxsplit=1)[-1]
        owner_name = component.map_owner_symbol.rsplit(".", maxsplit=1)[-1]
        return self.build_finding(
            (
                f"`{owner_name}.{component.map_method_name}` is keyed by "
                f"`{enum_name}` while `{component.reverse_method_name}` and "
                f"{len(component.consumers)} direct consumer(s) interpret that "
                "enum identity outside its declaration."
            ),
            component.evidence_locations,
            projection_evidence=component.projection_evidence,
            authority_evidence=component.authority_evidence,
        )


class ConcreteConfigFieldProbeDetector(
    ConfiguredModuleCollectorCandidateDetector[ConcreteConfigFieldProbeCandidate]
):
    candidate_collector = staticmethod(_concrete_config_field_probe_candidates)
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
    finding_spec = high_confidence_certified_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Exact-type boundary guard conflicts with nominal descendants",
        "A fail-loud boundary accepts exactly one class while the resolved inheritance graph declares descendants of that class. The source proves two incompatible membership surfaces, but not which declaration is wrong: the boundary may need nominal membership, or the hierarchy may not represent substitutability. The owning declaration must resolve that contract.",
        "one declared boundary-membership contract consistent with the nominal class graph",
        "exact concrete-type validation and a resolved base-to-descendant edge declare incompatible membership sets",
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
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=1,
                class_count=1 + len(candidate.descendant_classes),
            ),
        )


class NumericLiteralDispatchDetector(
    PerModuleIssueDetector,
    NumericLiteralDispatchFindingRecipeSynthesizer,
):
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
    for statement in statements_without_docstring(module.module.body):
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
    candidate_collector = staticmethod(_mirrored_import_fallback_candidates)
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
            metrics=MappingMetrics(
                mapping_site_count=2,
                field_count=import_candidate.imported_name_count,
                mapping_name="mirrored import fallback",
                field_names=import_candidate.imported_modules,
            ),
        )


class ScopedShapeWrapperDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_certified_spec(
        PatternId.SHARED_ALGORITHM_AUTHORITY,
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
            )
        ]
