"""Source-address index for compact, evidence-grounded agent targeting."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Generic, Iterable, TypeAlias, TypeVar

from .ast_tools import ClassFunctionStackNodeVisitor, ParsedModule
from .collection_algebra import sorted_tuple
from .models import RefactorFinding, SourceLocation, stable_source_location_id

IndexKeyT = TypeVar("IndexKeyT")
IndexValueT = TypeVar("IndexValueT")
AstTargetNode: TypeAlias = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
AstTargetNodeMap: TypeAlias = dict[str, AstTargetNode]
TupleIndexItems: TypeAlias = dict[IndexKeyT, tuple[IndexValueT, ...]]


@dataclass(frozen=True)
class StableIdAuthority:
    """Stable short identifiers for source-index rows."""

    def build(self, namespace: str, parts: Iterable[str | int]) -> str:
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

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"

    @property
    def is_module(self) -> bool:
        return self is AstTargetNodeKind.MODULE

    @property
    def is_class(self) -> bool:
        return self is AstTargetNodeKind.CLASS

    @property
    def is_function_like(self) -> bool:
        return self in _FUNCTION_LIKE_NODE_KINDS


_FUNCTION_LIKE_NODE_KINDS = frozenset(
    (AstTargetNodeKind.FUNCTION, AstTargetNodeKind.METHOD)
)


@dataclass(frozen=True)
class SourceFileDigest:
    """Stable source id for one parsed file."""

    file_id: str
    file_path: str
    module_name: str
    is_package_init: bool


@dataclass(frozen=True)
class AstTargetDigest:
    """Stable AST target address for one module, class, function, or method."""

    target_id: str
    file_id: str
    file_path: str
    node_type: str
    name: str
    qualname: str
    line: int
    end_line: int
    parameters: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    base_names: tuple[str, ...] = ()

    @property
    def node_kind(self) -> AstTargetNodeKind:
        return AstTargetNodeKind(self.node_type)

    @property
    def is_module(self) -> bool:
        return self.node_kind.is_module

    @property
    def is_class(self) -> bool:
        return self.node_kind.is_class

    @property
    def is_function_like(self) -> bool:
        return self.node_kind.is_function_like

    def contains_line(self, line: int) -> bool:
        return self.line <= line <= self.end_line

    def contains_span(self, start_line: int, end_line: int) -> bool:
        return self.line <= start_line and self.end_line >= end_line

    def matches_symbol(self, symbol: str) -> bool:
        return self.qualname == symbol or self.qualname.endswith(f".{symbol}")


@dataclass(frozen=True)
class EvidenceDigest:
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

    def as_pair(self) -> tuple[str, str]:
        return self.target_id, self.label


@dataclass(frozen=True)
class TupleIndex(Generic[IndexKeyT, IndexValueT]):
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

    def to_dict(self) -> TupleIndexItems:
        return dict(self.items_by_key)


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

    def to_dict(self) -> dict[str, tuple[AstTargetDigest, ...]]:
        return self.targets_by_file_path.to_dict()


@dataclass
class EvidenceDigestBuilder:
    """Build evidence rows while preserving stable evidence identity."""

    _source_locations_by_id: dict[str, SourceLocation]
    _finding_ids_by_evidence_id: TupleListIndexBuilder[str, str]

    def __init__(self) -> None:
        self._source_locations_by_id = {}
        self._finding_ids_by_evidence_id = TupleListIndexBuilder()

    def append_finding(self, finding: RefactorFinding) -> None:
        for source_location in finding.evidence:
            evidence_id = stable_source_location_id(source_location)
            if evidence_id not in self._source_locations_by_id:
                self._source_locations_by_id[evidence_id] = source_location
            self._finding_ids_by_evidence_id.append(evidence_id, finding.stable_id)

    def build(
        self,
        *,
        file_ids_by_path: dict[str, str],
        target_resolver: "EvidenceTargetResolver",
    ) -> tuple[EvidenceDigest, ...]:
        finding_ids_by_evidence = self._finding_ids_by_evidence_id.to_tuple_index()
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
            for evidence_id, source_location in sorted(
                self._source_locations_by_id.items()
            )
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
class SourceIndex:
    """Bidirectional source-address index derived from parsed code and findings."""

    files: tuple[SourceFileDigest, ...] = ()
    ast_targets: tuple[AstTargetDigest, ...] = ()
    evidence: tuple[EvidenceDigest, ...] = ()

    @cached_property
    def evidence_by_id(self) -> dict[str, EvidenceDigest]:
        return {item.evidence_id: item for item in self.evidence}

    @cached_property
    def target_by_id(self) -> dict[str, AstTargetDigest]:
        return {item.target_id: item for item in self.ast_targets}

    @cached_property
    def target_index_by_file(self) -> TargetsByFileIndex:
        return TargetsByFileIndex.from_targets(self.ast_targets)

    @cached_property
    def targets_by_file(self) -> TargetsByFileIndex:
        return self.target_index_by_file

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
    ) -> tuple[tuple[str, str], ...]:
        """Return deterministic AST target id/label pairs touched by a finding."""

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
            keys_by_target_id[target_id].as_pair()
            for target_id in sorted(keys_by_target_id)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "files": tuple(item.__dict__ for item in self.files),
            "ast_targets": tuple(item.__dict__ for item in self.ast_targets),
            "evidence": tuple(item.__dict__ for item in self.evidence),
        }


@dataclass(frozen=True)
class AstTargetNodeCache:
    """Parsed AST nodes addressed by source-index target identifiers."""

    nodes_by_target_id: AstTargetNodeMap


@dataclass(frozen=True)
class AstTargetBuildArtifacts:
    """AST target rows plus the parsed-node cache for those rows."""

    targets: tuple[AstTargetDigest, ...]
    node_cache: AstTargetNodeCache


@dataclass(frozen=True)
class SourceIndexBuildArtifacts:
    """Complete source-index build output for codemod snapshot reuse."""

    source_index: SourceIndex
    target_artifacts: AstTargetBuildArtifacts


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
        self.target_node_cache: AstTargetNodeMap = {}

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
        self.targets.append(
            AstTargetDigest(
                target_id=target_id,
                file_id=self.file_id,
                file_path=self.file_path,
                node_type=node_kind.value,
                name=node.name,
                qualname=qualname,
                line=line,
                end_line=end_line,
                parameters=parameters,
                decorators=decorators,
                base_names=base_names,
            )
        )
        self.target_node_cache[target_id] = node

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
class FileDigestAuthority:
    """Project parsed modules into source-file digest rows."""

    def digest(self, module: ParsedModule) -> SourceFileDigest:
        file_path = Path(module.path).as_posix()
        return SourceFileDigest(
            file_id=STABLE_ID_AUTHORITY.file_id(file_path),
            file_path=file_path,
            module_name=module.module_name,
            is_package_init=module.is_package_init,
        )


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
            node_cache=AstTargetNodeCache(visitor.target_node_cache),
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
            node_type=node_kind.value,
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
        symbol = evidence.symbol.rsplit(":", 1)[0]
        symbol_matches = tuple(
            target
            for target in file_targets
            if target.contains_line(evidence.line) and target.matches_symbol(symbol)
        )
        if symbol_matches:
            return self._target_ids(symbol_matches)

        non_module_matches = tuple(
            target
            for target in file_targets
            if not target.is_module and target.contains_line(evidence.line)
        )
        if non_module_matches:
            return self._target_ids(non_module_matches)

        module_matches = tuple(
            target
            for target in file_targets
            if target.is_module and target.contains_line(evidence.line)
        )
        if module_matches:
            return self._target_ids(module_matches)

        line_matches = tuple(
            target for target in file_targets if target.contains_line(evidence.line)
        )
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
        source_index = SourceIndex(
            files=files,
            ast_targets=target_artifacts.targets,
            evidence=EvidenceDigestsAuthority(
                file_ids_by_path=self._file_ids_by_path(files),
                targets_by_file=targets_by_file,
            ).digests(self.findings),
        )
        self._warm_lookup_indexes(source_index)
        return SourceIndexBuildArtifacts(
            source_index=source_index,
            target_artifacts=target_artifacts,
        )

    def _file_digests(self) -> tuple[SourceFileDigest, ...]:
        authority = FileDigestAuthority()
        return tuple(authority.digest(module) for module in self.modules)

    def _target_artifacts(
        self,
        files: tuple[SourceFileDigest, ...],
    ) -> AstTargetBuildArtifacts:
        authority = AstTargetDigestsAuthority()
        targets: list[AstTargetDigest] = []
        target_node_cache: AstTargetNodeMap = {}
        for module, file_digest in zip(self.modules, files, strict=True):
            artifacts = authority.artifacts(module, file_digest)
            targets.extend(artifacts.targets)
            target_node_cache.update(artifacts.node_cache.nodes_by_target_id)
        return AstTargetBuildArtifacts(
            targets=tuple(targets),
            node_cache=AstTargetNodeCache(target_node_cache),
        )

    @staticmethod
    def _file_ids_by_path(
        files: Iterable[SourceFileDigest],
    ) -> dict[str, str]:
        return {item.file_path: item.file_id for item in files}

    @staticmethod
    def _warm_lookup_indexes(source_index: SourceIndex) -> None:
        _ = (
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
