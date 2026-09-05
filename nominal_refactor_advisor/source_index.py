"""Source-address index for compact, evidence-grounded agent targeting."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar

from .ast_tools import (
    ClassFunctionStackNodeVisitor,
    ParsedModule,
    PythonModulePathAuthority,
)
from .collection_algebra import UniqueIdentityIndexAuthority, sorted_tuple
from .json_reports import (
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)
from .models import (
    RefactorFinding,
    SourceLocation,
    stable_source_location_id,
)
from .python_module_identity import PythonModulePathIdentity

SourceTargetIdentityValueT = TypeVar(
    "SourceTargetIdentityValueT",
    str,
    str | None,
)


@dataclass(frozen=True, kw_only=True)
class SourceTargetIdentity(Generic[SourceTargetIdentityValueT]):
    """Source-index target identity fields shared by selectors and resolved spans."""

    target_id: SourceTargetIdentityValueT
    file_path: SourceTargetIdentityValueT


@dataclass(frozen=True, kw_only=True)
class AstTargetGeometryKey:
    """Stable key joining source-index target geometry to parsed AST nodes."""

    qualname: str
    line: int
    end_line: int


@dataclass(frozen=True, kw_only=True)
class SourceTargetSpan(SourceTargetIdentity[str], AstTargetGeometryKey):
    """Resolved source-index target span shared by codemod analyses."""

    target_id: str
    file_path: str


IndexKeyT = TypeVar("IndexKeyT")
IndexValueT = TypeVar("IndexValueT")
AstTargetNode: TypeAlias = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
AstTargetNodeMap: TypeAlias = dict[str, AstTargetNode]
StableIdPart: TypeAlias = str | int | bool | None | tuple[str, ...]
TupleIndexItems: TypeAlias = dict[IndexKeyT, tuple[IndexValueT, ...]]


@dataclass(frozen=True)
class StableIdAuthority:
    """Stable short identifiers for source-index rows."""

    def build(self, namespace: str, parts: Iterable[StableIdPart]) -> str:
        payload = "|".join((namespace, *(str(part) for part in parts)))
        return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()

    def file_id(self, file_path: str) -> str:
        return self.build("file", (file_path,))

    def ast_target_id(
        self,
        *,
        file_path: str,
        node_kind: "AstTargetNodeKind",
        qualname: str,
        line: int,
        end_line: int,
    ) -> str:
        return self.build(
            "ast-target",
            (file_path, node_kind.value, qualname, line, end_line),
        )


STABLE_ID_AUTHORITY = StableIdAuthority()


class AstTargetNodeKind(StrEnum):
    """Source-index AST target kinds."""

    MODULE = "module", (ast.Module,)
    CLASS = "class", (ast.ClassDef,)
    FUNCTION = "function", (ast.FunctionDef, ast.AsyncFunctionDef)
    METHOD = "method", (ast.FunctionDef, ast.AsyncFunctionDef)

    node_types: tuple[type[ast.AST], ...]

    def __new__(
        cls, value: str, node_types: tuple[type[ast.AST], ...],
    ) -> "AstTargetNodeKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member.node_types = node_types
        return member

    def accepts(self, node: ast.AST) -> bool:
        return isinstance(node, self.node_types)

    @property
    def is_module(self) -> bool:
        return self is AstTargetNodeKind.MODULE

    @property
    def is_class(self) -> bool:
        return self is AstTargetNodeKind.CLASS

    @property
    def is_function(self) -> bool:
        return self is AstTargetNodeKind.FUNCTION

    @property
    def is_method(self) -> bool:
        return self is AstTargetNodeKind.METHOD

    @property
    def is_function_like(self) -> bool:
        return self.is_function or self.is_method


@dataclass(frozen=True)
class SourceFileDigest(DataclassJsonReport):
    """Stable source id for one parsed file."""

    file_id: str
    module_path_identity: PythonModulePathIdentity = json_report_field(included=False)

    @classmethod
    def from_module(cls, module: ParsedModule) -> "SourceFileDigest":
        return cls(
            file_id=STABLE_ID_AUTHORITY.file_id(module.file_path),
            module_path_identity=module.module_path_identity,
        )

    @json_report_property()
    def file_path(self) -> str:
        return self.module_path_identity.file_path

    @json_report_property()
    def module_name(self) -> str:
        return self.module_path_identity.import_name

    @json_report_property()
    def is_package_init(self) -> bool:
        return self.module_path_identity.is_package_init


@dataclass(frozen=True)
class AstTargetDigest(DataclassJsonReport):
    """Stable AST target address for one module, class, function, or method."""

    target_id: str
    file_id: str
    file_path: str
    node_kind: AstTargetNodeKind = json_report_field(field_name="node_type")
    name: str
    qualname: str
    line: int
    end_line: int
    parameters: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    base_names: tuple[str, ...] = ()

    @property
    def is_module(self) -> bool:
        return self.node_kind.is_module

    @property
    def is_class(self) -> bool:
        return self.node_kind.is_class

    @property
    def is_function(self) -> bool:
        return self.node_kind.is_function

    @property
    def is_method(self) -> bool:
        return self.node_kind.is_method

    @property
    def is_function_like(self) -> bool:
        return self.node_kind.is_function_like

    def require_kind(
        self,
        required_kind: AstTargetNodeKind,
        message: str,
    ) -> None:
        """Prove one operation-specific target-kind precondition."""

        if self.node_kind is not required_kind:
            raise ValueError(message)

    def contains_line(self, line: int) -> bool:
        return self.line <= line <= self.end_line

    def contains_span(self, start_line: int, end_line: int) -> bool:
        return self.line <= start_line and self.end_line >= end_line

    def matches_symbol(self, symbol: str) -> bool:
        return symbol in self.lookup_symbols

    @property
    def lookup_symbols(self) -> tuple[str, ...]:
        """Exact declaration names accepted by source-target lookup."""

        return tuple(dict.fromkeys((self.qualname, self.name)))


@dataclass(frozen=True)
class EvidenceDigest(DataclassJsonReport):
    """Stable source-address row for one finding evidence coordinate."""

    evidence_id: str
    file_id: str | None
    file_path: str
    line: int
    symbol: str
    finding_ids: tuple[str, ...]
    target_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceTargetKey:
    """Human-readable source-index target label for one finding."""

    target_id: str
    label: str


@dataclass(frozen=True)
class TupleIndex(
    Mapping[IndexKeyT, tuple[IndexValueT, ...]],
    Generic[IndexKeyT, IndexValueT],
):
    """Deterministic tuple-valued lookup used by source-index authorities."""

    items_by_key: TupleIndexItems

    def __contains__(self, key: IndexKeyT) -> bool:
        return key in self.items_by_key

    def __getitem__(self, key: IndexKeyT) -> tuple[IndexValueT, ...]:
        return self.items_by_key[key]

    def __iter__(self) -> Iterable[IndexKeyT]:
        return iter(self.items_by_key)

    def __len__(self) -> int:
        return len(self.items_by_key)

    def items(self) -> Iterable[tuple[IndexKeyT, tuple[IndexValueT, ...]]]:
        return self.items_by_key.items()

    def values(self) -> Iterable[tuple[IndexValueT, ...]]:
        return self.items_by_key.values()

    def tuple_for_key(self, key: IndexKeyT) -> tuple[IndexValueT, ...]:
        if key not in self.items_by_key:
            return ()
        return self.items_by_key[key]

@dataclass(frozen=True)
class EvidenceTargetRelation:
    """Bidirectional finding-to-target relation derived from evidence rows."""

    target_ids_by_finding_id: TupleIndex[str, str]
    finding_ids_by_target_id: TupleIndex[str, str]

    @classmethod
    def from_evidence(
        cls,
        evidence_rows: Iterable[EvidenceDigest],
    ) -> "EvidenceTargetRelation":
        targets_by_finding = TupleSetIndexBuilder[str, str]()
        findings_by_target = TupleSetIndexBuilder[str, str]()
        for evidence in evidence_rows:
            for finding_id in evidence.finding_ids:
                targets_by_finding.update(finding_id, evidence.target_ids)
            for target_id in evidence.target_ids:
                findings_by_target.update(target_id, evidence.finding_ids)
        return cls(
            target_ids_by_finding_id=targets_by_finding.to_sorted_tuple_index(),
            finding_ids_by_target_id=findings_by_target.to_sorted_tuple_index(),
        )


@dataclass(frozen=True)
class TargetsByFileIndex:
    """Source-index targets grouped by file path."""

    targets_by_file_path: TupleIndex[str, AstTargetDigest]

    @classmethod
    def from_targets(cls, targets: Iterable[AstTargetDigest]) -> "TargetsByFileIndex":
        builder = TupleListIndexBuilder[str, AstTargetDigest]()
        for target in targets:
            builder.append(target.file_path, target)
        return cls(builder.to_tuple_index())

    def __contains__(self, file_path: str) -> bool:
        return file_path in self.targets_by_file_path

    def __getitem__(self, file_path: str) -> tuple[AstTargetDigest, ...]:
        return self.targets_by_file_path[file_path]

    def items(self) -> Iterable[tuple[str, tuple[AstTargetDigest, ...]]]:
        return self.targets_by_file_path.items()

    def contains_file(self, file_path: str) -> bool:
        return file_path in self.targets_by_file_path

    def smallest_enclosing_target(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> AstTargetDigest | None:
        """Return the narrowest indexed declaration containing one source span."""

        candidates = tuple(
            target
            for target in self.targets_by_file_path.tuple_for_key(file_path)
            if target.contains_span(start_line, end_line)
        )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                target.end_line - target.line,
                target.line,
                target.qualname,
            ),
        )

@dataclass
class EvidenceDigestBuilder:
    """Build evidence rows while preserving stable evidence identity."""

    _source_location_index: UniqueIdentityIndexAuthority[
        str, SourceLocation, SourceLocation
    ]
    _finding_ids_by_evidence_id: TupleListIndexBuilder[str, str]

    def __init__(self) -> None:
        self._source_location_index = UniqueIdentityIndexAuthority()
        self._finding_ids_by_evidence_id = TupleListIndexBuilder()

    def append_finding(self, finding: RefactorFinding) -> None:
        for source_location in finding.evidence:
            evidence_id = stable_source_location_id(source_location)
            self._source_location_index.add(
                evidence_id,
                source_location,
                source_location,
            )
            self._finding_ids_by_evidence_id.append(evidence_id, finding.stable_id)

    def build(
        self,
        *,
        file_ids_by_path: dict[str, str],
        target_resolver: "EvidenceTargetResolver",
    ) -> tuple[EvidenceDigest, ...]:
        finding_ids_by_evidence = self._finding_ids_by_evidence_id.to_tuple_index()
        source_locations_by_id = self._source_location_index.values_by_handle()
        return tuple(
            EvidenceDigest(
                evidence_id=evidence_id,
                file_id=_optional_file_id(source_location.file_path, file_ids_by_path),
                file_path=source_location.file_path,
                line=source_location.line,
                symbol=source_location.symbol,
                finding_ids=sorted_tuple(set(finding_ids_by_evidence[evidence_id])),
                target_ids=target_resolver.target_ids_for_evidence(source_location),
            )
            for evidence_id, source_location in sorted(source_locations_by_id.items())
        )


def _optional_file_id(file_path: str, file_ids_by_path: dict[str, str]) -> str | None:
    if file_path not in file_ids_by_path:
        return None
    return file_ids_by_path[file_path]


@dataclass
class TupleListIndexBuilder(Generic[IndexKeyT, IndexValueT]):
    """Build deterministic tuple-valued indexes without inline setdefault loops."""

    _items_by_key: dict[IndexKeyT, list[IndexValueT]]

    def __init__(self) -> None:
        self._items_by_key = {}

    def append(self, key: IndexKeyT, value: IndexValueT) -> None:
        if key not in self._items_by_key:
            self._items_by_key[key] = []
        self._items_by_key[key].append(value)

    def to_tuple_index(self) -> TupleIndex[IndexKeyT, IndexValueT]:
        return TupleIndex(
            {key: tuple(values) for key, values in self._items_by_key.items()}
        )


@dataclass
class TupleSetIndexBuilder(Generic[IndexKeyT, IndexValueT]):
    """Build deterministic tuple-valued indexes from set membership."""

    _items_by_key: dict[IndexKeyT, set[IndexValueT]]

    def __init__(self) -> None:
        self._items_by_key = {}

    def update(self, key: IndexKeyT, values: Iterable[IndexValueT]) -> None:
        if key not in self._items_by_key:
            self._items_by_key[key] = set()
        self._items_by_key[key].update(values)

    def to_sorted_tuple_index(self) -> TupleIndex[IndexKeyT, IndexValueT]:
        return TupleIndex(
            {key: sorted_tuple(values) for key, values in self._items_by_key.items()}
        )


@dataclass(frozen=True)
class SourceIndex(DataclassJsonReport):
    """Bidirectional source-address index derived from parsed code and findings."""

    files: tuple[SourceFileDigest, ...] = ()
    ast_targets: tuple[AstTargetDigest, ...] = ()
    evidence: tuple[EvidenceDigest, ...] = ()

    @cached_property
    def module_path_authority(self) -> PythonModulePathAuthority:
        return PythonModulePathAuthority(
            tuple(source_file.module_path_identity for source_file in self.files)
        )

    @cached_property
    def file_by_id(self) -> dict[str, SourceFileDigest]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            self.files,
            lambda item: item.file_id,
        )

    @cached_property
    def evidence_by_id(self) -> dict[str, EvidenceDigest]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            self.evidence,
            lambda item: item.evidence_id,
        )

    @cached_property
    def target_by_id(self) -> dict[str, AstTargetDigest]:
        return UniqueIdentityIndexAuthority.declarations_by_handle(
            self.ast_targets,
            lambda item: item.target_id,
        )

    def symbol_for_target(self, target: AstTargetDigest) -> str:
        """Derive a target's repository symbol from its owning source file."""

        if target.is_module:
            return target.qualname
        return f"{self.file_by_id[target.file_id].module_name}.{target.qualname}"

    @cached_property
    def target_index_by_file(self) -> TargetsByFileIndex:
        return TargetsByFileIndex.from_targets(self.ast_targets)

    @cached_property
    def targets_by_file(self) -> TargetsByFileIndex:
        return self.target_index_by_file

    @cached_property
    def target_file_paths(self) -> tuple[str, ...]:
        return tuple(file_path for file_path, _targets in self.targets_by_file.items())

    @cached_property
    def targets_by_qualname(self) -> TupleIndex[str, AstTargetDigest]:
        builder = TupleListIndexBuilder[str, AstTargetDigest]()
        for target in self.ast_targets:
            builder.append(target.qualname, target)
        return builder.to_tuple_index()

    @cached_property
    def targets_by_symbol(self) -> TupleIndex[str, AstTargetDigest]:
        builder = TupleListIndexBuilder[str, AstTargetDigest]()
        for target in self.ast_targets:
            for symbol in target.lookup_symbols:
                builder.append(symbol, target)
        return builder.to_tuple_index()

    @cached_property
    def targets_by_repository_symbol(self) -> TupleIndex[str, AstTargetDigest]:
        """Index declarations by their module-qualified source identity."""

        builder = TupleListIndexBuilder[str, AstTargetDigest]()
        for target in self.ast_targets:
            builder.append(self.symbol_for_target(target), target)
        return builder.to_tuple_index()

    def targets_matching_symbol(self, symbol: str) -> tuple[AstTargetDigest, ...]:
        return self.targets_by_symbol.tuple_for_key(symbol)

    def targets_matching_repository_symbol(
        self,
        symbol: str,
    ) -> tuple[AstTargetDigest, ...]:
        return self.targets_by_repository_symbol.tuple_for_key(symbol)

    @cached_property
    def evidence_target_relation(self) -> EvidenceTargetRelation:
        return EvidenceTargetRelation.from_evidence(self.evidence)

    @cached_property
    def target_ids_by_finding_id(self) -> TupleIndex[str, str]:
        return self.evidence_target_relation.target_ids_by_finding_id

    @cached_property
    def finding_ids_by_target_id(self) -> TupleIndex[str, str]:
        return self.evidence_target_relation.finding_ids_by_target_id

    def target_ids_for_finding_ids(self, finding_ids: Iterable[str]) -> tuple[str, ...]:
        target_ids: set[str] = set()
        for finding_id in finding_ids:
            if finding_id in self.target_ids_by_finding_id:
                target_ids.update(self.target_ids_by_finding_id[finding_id])
        return sorted_tuple(target_ids)

    def finding_ids_for_target_id(self, target_id: str) -> tuple[str, ...]:
        if target_id not in self.finding_ids_by_target_id:
            return ()
        return self.finding_ids_by_target_id[target_id]

    def source_target_keys_for_finding(
        self, finding: RefactorFinding
    ) -> tuple[SourceTargetKey, ...]:
        """Return deterministic AST target declarations touched by a finding."""

        keys_by_target_id: dict[str, SourceTargetKey] = {}
        for source_location in finding.evidence:
            evidence_id = stable_source_location_id(source_location)
            if evidence_id not in self.evidence_by_id:
                continue
            evidence = self.evidence_by_id[evidence_id]
            for target_id in evidence.target_ids:
                if target_id not in self.target_by_id:
                    continue
                target = self.target_by_id[target_id]
                if target_id not in keys_by_target_id:
                    keys_by_target_id[target_id] = SourceTargetKey(
                        target_id=target_id,
                        label=f"{target.file_path}:{target.qualname}",
                    )
        return tuple(
            keys_by_target_id[target_id] for target_id in sorted(keys_by_target_id)
        )


@dataclass(frozen=True)
class IndexedSourceAuthority:
    """One source index paired with the exact source texts it indexes."""

    source_index: SourceIndex
    sources_by_file_path: Mapping[str, str]


@dataclass(frozen=True)
class CodemodSourceIndexReport(DataclassJsonReport):
    """JSON-ready target discovery report for codemod DSL authors."""

    source_index: SourceIndex = json_report_field(included=False)

    @json_report_property()
    def target_count(self) -> int:
        return len(self.source_index.ast_targets)

    @json_report_property()
    def file_count(self) -> int:
        return len(self.source_index.files)

    @json_report_property()
    def evidence_count(self) -> int:
        return len(self.source_index.evidence)

    @json_report_property()
    def files(self) -> tuple[SourceFileDigest, ...]:
        return self.source_index.files

    @json_report_property(field_name="targets")
    def ast_targets(self) -> tuple[AstTargetDigest, ...]:
        return self.source_index.ast_targets

    @json_report_property()
    def evidence(self) -> tuple[EvidenceDigest, ...]:
        return self.source_index.evidence

    @json_report_property()
    def target_ids_by_finding_id(self) -> TupleIndex[str, str]:
        return self.source_index.target_ids_by_finding_id

    @json_report_property()
    def finding_ids_by_target_id(self) -> TupleIndex[str, str]:
        return self.source_index.finding_ids_by_target_id


@dataclass(frozen=True)
class AstTargetNodeIndex:
    """Parsed AST nodes addressed by their canonical source-index targets."""

    nodes_by_target_id: AstTargetNodeMap

    @classmethod
    def from_modules(
        cls,
        source_index: SourceIndex,
        modules: Iterable[ParsedModule],
    ) -> "AstTargetNodeIndex":
        """Join stable indexed target addresses to freshly parsed AST nodes."""

        node_index = UniqueIdentityIndexAuthority[
            str, AstTargetDigest, AstTargetNode
        ]()
        target_authority = AstTargetDigestsAuthority()
        for module in modules:
            artifacts = target_authority.artifacts(
                module,
                SourceFileDigest.from_module(module),
            )
            projected_targets_by_id = UniqueIdentityIndexAuthority.declarations_by_handle(
                artifacts.targets,
                lambda target: target.target_id,
            )
            for target_id, node in artifacts.node_index.nodes_by_target_id.items():
                node_index.add(
                    target_id,
                    projected_targets_by_id[target_id],
                    node,
                )
        current_nodes_by_target_id = node_index.values_by_handle()
        stable_target_ids = (
            source_index.target_by_id.keys() & current_nodes_by_target_id.keys()
        )
        return cls(
            {
                target_id: current_nodes_by_target_id[target_id]
                for target_id in stable_target_ids
            }
        )

    @classmethod
    def from_source_mapping(
        cls,
        source_index: SourceIndex,
        sources_by_file_path: Mapping[str, str],
    ) -> "AstTargetNodeIndex":
        """Parse indexed sources and derive their canonical node handles."""

        return cls.from_modules(
            source_index,
            (
                source_index.module_path_authority.source_module(
                    Path(file_path),
                    source,
                ).parse()
                for file_path, source in sorted(sources_by_file_path.items())
            ),
        )

    @property
    def function_nodes_by_target_id(
        self,
    ) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
        return {
            target_id: node
            for target_id, node in self.nodes_by_target_id.items()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }


@dataclass(frozen=True)
class AstTargetBuildArtifacts:
    """AST target rows plus the parsed-node cache for those rows."""

    targets: tuple[AstTargetDigest, ...]
    node_index: AstTargetNodeIndex


@dataclass(frozen=True)
class SourceIndexBuildArtifacts:
    """Complete source-index build output for codemod snapshot reuse."""

    source_index: SourceIndex
    node_index: AstTargetNodeIndex

    def projected_with_module_overlay(
        self,
        projected_modules: tuple[ParsedModule, ...],
        changed_modules: tuple[ParsedModule, ...],
    ) -> "SourceIndexBuildArtifacts":
        """Reindex only changed modules while preserving exact target identity."""

        changed_modules_by_path = (
            UniqueIdentityIndexAuthority.declarations_by_handle(
                changed_modules,
                lambda module: module.file_path,
            )
        )
        projected_modules_by_path = (
            UniqueIdentityIndexAuthority.declarations_by_handle(
                projected_modules,
                lambda module: module.file_path,
            )
        )
        if not changed_modules_by_path.keys() <= projected_modules_by_path.keys():
            raise ValueError("Changed modules must belong to the projected source state")
        files_by_path = UniqueIdentityIndexAuthority.declarations_by_handle(
            self.source_index.files,
            lambda source_file: source_file.file_path,
        )
        target_authority = AstTargetDigestsAuthority()
        projected_files: list[SourceFileDigest] = []
        projected_targets: list[AstTargetDigest] = []
        projected_nodes: AstTargetNodeMap = {}
        for module in projected_modules:
            file_path = module.file_path
            if file_path in changed_modules_by_path:
                file_digest = SourceFileDigest.from_module(module)
                target_artifacts = target_authority.artifacts(module, file_digest)
            else:
                file_digest = files_by_path.get(file_path)
                if file_digest is None:
                    raise ValueError(
                        "Every new projected module must be declared as changed"
                    )
                if file_digest.module_path_identity != module.module_path_identity:
                    raise ValueError(
                        "Unchanged projected modules must preserve module identity"
                    )
                module_targets = self.source_index.targets_by_file[file_path]
                required_node_ids = tuple(
                    target.target_id
                    for target in module_targets
                    if not target.is_module
                )
                if all(
                    target_id in self.node_index.nodes_by_target_id
                    for target_id in required_node_ids
                ):
                    target_artifacts = AstTargetBuildArtifacts(
                        targets=module_targets,
                        node_index=AstTargetNodeIndex(
                            {
                                target_id: self.node_index.nodes_by_target_id[target_id]
                                for target_id in required_node_ids
                            }
                        ),
                    )
                else:
                    target_artifacts = target_authority.artifacts(
                        module,
                        file_digest,
                    )
                    if target_artifacts.targets != module_targets:
                        raise ValueError(
                            "Unchanged source produced different target identities"
                        )
            projected_files.append(file_digest)
            projected_targets.extend(target_artifacts.targets)
            projected_nodes.update(target_artifacts.node_index.nodes_by_target_id)
        return type(self)(
            source_index=SourceIndex(
                files=tuple(projected_files),
                ast_targets=tuple(projected_targets),
            ),
            node_index=AstTargetNodeIndex(projected_nodes),
        )


def iter_statement_definition_nodes(
    statements: Iterable[ast.stmt],
) -> Iterable[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield nested class/function statements without visiting expression trees."""

    for statement in statements:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield statement
            continue
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                yield from iter_statement_definition_nodes((child,))


class _AstTargetDigestVisitor(ClassFunctionStackNodeVisitor):
    def __init__(self, file_id: str, file_path: str) -> None:
        super().__init__()
        self.file_id = file_id
        self.file_path = file_path
        self.targets: list[AstTargetDigest] = []
        self.target_node_index = UniqueIdentityIndexAuthority[
            str, AstTargetDigest, AstTargetNode
        ]()

    def traverse_statements(self, body: list[ast.stmt]) -> None:
        for node in iter_statement_definition_nodes(body):
            self.visit(node)

    def before_visit_class(self, node: ast.ClassDef) -> None:
        self._append_target(node, AstTargetNodeKind.CLASS)

    def before_visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self._append_target(node, self._function_node_kind())

    def traverse_class_body(self, node: ast.ClassDef) -> None:
        self.traverse_statements(node.body)

    def traverse_function_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.traverse_statements(node.body)

    def _function_node_kind(self) -> AstTargetNodeKind:
        if self.class_stack:
            return AstTargetNodeKind.METHOD
        return AstTargetNodeKind.FUNCTION

    def _append_target(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        node_kind: AstTargetNodeKind,
    ) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        line = node.lineno
        end_line = node.end_lineno or line
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = self._parameters(node)
        else:
            parameters = ()
        decorators = _decorator_names(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            base_names = _base_names(node.bases)
        else:
            base_names = ()
        target_id = STABLE_ID_AUTHORITY.ast_target_id(
            file_path=self.file_path,
            node_kind=node_kind,
            qualname=qualname,
            line=line,
            end_line=end_line,
        )
        target = AstTargetDigest(
            target_id=target_id,
            file_id=self.file_id,
            file_path=self.file_path,
            node_kind=node_kind,
            name=node.name,
            qualname=qualname,
            line=line,
            end_line=end_line,
            parameters=parameters,
            decorators=decorators,
            base_names=base_names,
        )
        self.targets.append(target)
        self.target_node_index.add(target_id, target, node)

    @staticmethod
    def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
        args = [arg.arg for arg in node.args.posonlyargs]
        args.extend(arg.arg for arg in node.args.args)
        if node.args.vararg is not None:
            args.append(f"*{node.args.vararg.arg}")
        args.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg is not None:
            args.append(f"**{node.args.kwarg.arg}")
        return tuple(args)


@dataclass(frozen=True)
class SurfaceNameProjection:
    def project(
        self,
        nodes: Iterable[ast.expr],
        *,
        expand_calls: bool = False,
        expand_subscripts: bool = False,
    ) -> tuple[str, ...]:
        names = []
        for node in nodes:
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif expand_calls and isinstance(node, ast.Call):
                names.extend(self.project((node.func,), expand_calls=expand_calls))
            elif expand_subscripts and isinstance(node, ast.Subscript):
                names.extend(
                    self.project((node.value,), expand_subscripts=expand_subscripts)
                )
        return tuple(names)


SURFACE_NAME_PROJECTION = SurfaceNameProjection()


def _decorator_names(decorators: Iterable[ast.expr]) -> tuple[str, ...]:
    return SURFACE_NAME_PROJECTION.project(decorators, expand_calls=True)


def _base_names(bases: Iterable[ast.expr]) -> tuple[str, ...]:
    return SURFACE_NAME_PROJECTION.project(bases, expand_subscripts=True)


@dataclass(frozen=True)
class AstTargetDigestsAuthority:
    """Project parsed modules into module/class/function target digest rows."""

    def artifacts(
        self,
        module: ParsedModule,
        file_digest: SourceFileDigest,
    ) -> AstTargetBuildArtifacts:
        visitor = _AstTargetDigestVisitor(
            file_digest.file_id,
            file_digest.file_path,
        )
        visitor.visit(module.module)
        return AstTargetBuildArtifacts(
            targets=(self.module_target_digest(module, file_digest), *visitor.targets),
            node_index=AstTargetNodeIndex(visitor.target_node_index.values_by_handle()),
        )

    def module_target_digest(
        self,
        module: ParsedModule,
        file_digest: SourceFileDigest,
    ) -> AstTargetDigest:
        node_kind = AstTargetNodeKind.MODULE
        end_line = max(1, len(module.source.splitlines()))
        return AstTargetDigest(
            target_id=STABLE_ID_AUTHORITY.ast_target_id(
                file_path=file_digest.file_path,
                node_kind=node_kind,
                qualname=module.module_name,
                line=1,
                end_line=end_line,
            ),
            file_id=file_digest.file_id,
            file_path=file_digest.file_path,
            node_kind=node_kind,
            name=module.module_name,
            qualname=module.module_name,
            line=1,
            end_line=end_line,
        )


@dataclass(frozen=True)
class EvidenceTargetResolver:
    """Resolve finding evidence coordinates to source-index target ids."""

    targets_by_file: TargetsByFileIndex

    def target_ids_for_evidence(self, evidence: SourceLocation) -> tuple[str, ...]:
        file_targets = self._targets_in_file(evidence.file_path)
        symbol = evidence.subject_symbol
        symbol_matches: list[AstTargetDigest] = []
        non_module_matches: list[AstTargetDigest] = []
        module_matches: list[AstTargetDigest] = []
        line_matches: list[AstTargetDigest] = []
        for target in file_targets:
            if not target.contains_line(evidence.line):
                continue
            line_matches.append(target)
            if target.matches_symbol(symbol):
                symbol_matches.append(target)
            if target.is_module:
                module_matches.append(target)
            else:
                non_module_matches.append(target)
        if symbol_matches:
            return self._target_ids(symbol_matches)
        if non_module_matches:
            return self._target_ids(non_module_matches)
        if module_matches:
            return self._target_ids(module_matches)
        return self._target_ids(line_matches)

    def _targets_in_file(self, file_path: str) -> tuple[AstTargetDigest, ...]:
        if file_path not in self.targets_by_file:
            return ()
        return self.targets_by_file[file_path]

    @staticmethod
    def _target_ids(targets: Iterable[AstTargetDigest]) -> tuple[str, ...]:
        return tuple(target.target_id for target in targets)


@dataclass(frozen=True)
class EvidenceDigestsAuthority:
    """Project findings and source targets into evidence digest rows."""

    file_ids_by_path: dict[str, str]
    targets_by_file: TargetsByFileIndex

    def digests(
        self, findings: Iterable[RefactorFinding]
    ) -> tuple[EvidenceDigest, ...]:
        builder = EvidenceDigestBuilder()
        for finding in findings:
            builder.append_finding(finding)
        return builder.build(
            file_ids_by_path=self.file_ids_by_path,
            target_resolver=EvidenceTargetResolver(self.targets_by_file),
        )


@dataclass(frozen=True)
class SourceIndexBuildAuthority:
    """Build and warm the source-address index from parsed modules and findings."""

    modules: tuple[ParsedModule, ...]
    findings: tuple[RefactorFinding, ...]

    def build(self) -> SourceIndex:
        return self.build_artifacts().source_index

    def build_artifacts(self) -> SourceIndexBuildArtifacts:
        files = self._file_digests()
        target_artifacts = self._target_artifacts(files)
        targets_by_file = TargetsByFileIndex.from_targets(target_artifacts.targets)
        findings_by_id = UniqueIdentityIndexAuthority.declarations_by_handle(
            self.findings,
            lambda finding: finding.stable_id,
        )
        source_index = SourceIndex(
            files=files,
            ast_targets=target_artifacts.targets,
            evidence=EvidenceDigestsAuthority(
                file_ids_by_path=self._file_ids_by_path(files),
                targets_by_file=targets_by_file,
            ).digests(findings_by_id.values()),
        )
        self._warm_lookup_indexes(source_index)
        return SourceIndexBuildArtifacts(
            source_index=source_index,
            node_index=target_artifacts.node_index,
        )

    def _file_digests(self) -> tuple[SourceFileDigest, ...]:
        return tuple(SourceFileDigest.from_module(module) for module in self.modules)

    def _target_artifacts(
        self,
        files: tuple[SourceFileDigest, ...],
    ) -> AstTargetBuildArtifacts:
        authority = AstTargetDigestsAuthority()
        targets: list[AstTargetDigest] = []
        target_node_index = UniqueIdentityIndexAuthority[
            str, AstTargetDigest, AstTargetNode
        ]()
        for module, file_digest in zip(self.modules, files, strict=True):
            artifacts = authority.artifacts(module, file_digest)
            targets.extend(artifacts.targets)
            targets_by_id = UniqueIdentityIndexAuthority.declarations_by_handle(
                artifacts.targets,
                lambda target: target.target_id,
            )
            for target_id, node in artifacts.node_index.nodes_by_target_id.items():
                target_node_index.add(target_id, targets_by_id[target_id], node)
        return AstTargetBuildArtifacts(
            targets=tuple(targets),
            node_index=AstTargetNodeIndex(target_node_index.values_by_handle()),
        )

    @staticmethod
    def _file_ids_by_path(
        files: Iterable[SourceFileDigest],
    ) -> dict[str, str]:
        index = UniqueIdentityIndexAuthority[str, SourceFileDigest, str]()
        for item in files:
            index.add(item.file_path, item, item.file_id)
        return index.values_by_handle()

    @staticmethod
    def _warm_lookup_indexes(source_index: SourceIndex) -> None:
        _ = (
            source_index.file_by_id,
            source_index.evidence_by_id,
            source_index.target_by_id,
            source_index.targets_by_file,
            source_index.target_ids_by_finding_id,
            source_index.finding_ids_by_target_id,
        )


def build_source_index(
    modules: Iterable[ParsedModule], findings: Iterable[RefactorFinding]
) -> SourceIndex:
    """Build a source-address index from parsed modules and emitted findings."""

    return SourceIndexBuildAuthority(
        modules=tuple(modules),
        findings=tuple(findings),
    ).build()


def build_source_index_artifacts(
    modules: Iterable[ParsedModule], findings: Iterable[RefactorFinding]
) -> SourceIndexBuildArtifacts:
    """Build a source index and parsed-node cache from one AST traversal."""

    return SourceIndexBuildAuthority(
        modules=tuple(modules),
        findings=tuple(findings),
    ).build_artifacts()
