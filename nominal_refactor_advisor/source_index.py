"""Source-address index for compact, evidence-grounded agent targeting."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, TypeAlias

from .ast_tools import ClassFunctionStackNodeVisitor, ParsedModule
from .collection_algebra import sorted_tuple
from .models import RefactorFinding, SourceLocation, stable_source_location_id
from .record_algebra import product_record


def _stable_id(namespace: str, *parts: object) -> str:
    payload = "|".join((namespace, *(str(part) for part in parts)))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


SourceFileDigest = product_record(
    "SourceFileDigest",
    "file_id: str; file_path: str; module_name: str; is_package_init: bool",
    doc="Stable source id for one parsed file.",
)

AstTargetDigest = product_record(
    "AstTargetDigest",
    (
        "target_id: str; file_id: str; file_path: str; node_type: str; "
        "name: str; qualname: str; line: int; end_line: int; "
        "parameters: tuple[str, ...]; decorators: tuple[str, ...]; "
        "base_names: tuple[str, ...]"
    ),
    defaults={"parameters": (), "decorators": (), "base_names": ()},
    doc="Stable AST target address for one class, function, or method.",
)

EvidenceDigest = product_record(
    "EvidenceDigest",
    (
        "evidence_id: str; file_id: str | None; file_path: str; line: int; "
        "symbol: str; finding_ids: tuple[str, ...]; target_ids: tuple[str, ...]"
    ),
    defaults={"target_ids": ()},
    doc="Stable source-address row for one finding evidence coordinate.",
)

TargetIndex: TypeAlias = dict[str, tuple[AstTargetDigest, ...]]
SourceTargetKey: TypeAlias = tuple[str, str]


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
    def targets_by_file(self) -> TargetIndex:
        target_lists_by_file: dict[str, list[AstTargetDigest]] = {}
        for target in self.ast_targets:
            target_lists_by_file.setdefault(target.file_path, []).append(target)
        return {
            file_path: tuple(targets)
            for file_path, targets in target_lists_by_file.items()
        }

    @cached_property
    def target_ids_by_finding_id(self) -> dict[str, tuple[str, ...]]:
        target_ids_by_finding_id: dict[str, set[str]] = {}
        for evidence in self.evidence:
            for finding_id in evidence.finding_ids:
                target_ids_by_finding_id.setdefault(finding_id, set()).update(
                    evidence.target_ids
                )
        return {
            finding_id: sorted_tuple(target_ids)
            for finding_id, target_ids in target_ids_by_finding_id.items()
        }

    @cached_property
    def finding_ids_by_target_id(self) -> dict[str, tuple[str, ...]]:
        finding_ids_by_target_id: dict[str, set[str]] = {}
        for evidence in self.evidence:
            for target_id in evidence.target_ids:
                finding_ids_by_target_id.setdefault(target_id, set()).update(
                    evidence.finding_ids
                )
        return {
            target_id: sorted_tuple(finding_ids)
            for target_id, finding_ids in finding_ids_by_target_id.items()
        }

    def target_ids_for_finding_ids(self, finding_ids: Iterable[str]) -> tuple[str, ...]:
        target_ids: set[str] = set()
        for finding_id in finding_ids:
            target_ids.update(self.target_ids_by_finding_id.get(finding_id, ()))
        return sorted_tuple(target_ids)

    def finding_ids_for_target_id(self, target_id: str) -> tuple[str, ...]:
        return self.finding_ids_by_target_id.get(target_id, ())

    def source_target_keys_for_finding(
        self, finding: RefactorFinding
    ) -> tuple[SourceTargetKey, ...]:
        """Return deterministic AST target id/label pairs touched by a finding."""

        keys_by_target_id: dict[str, SourceTargetKey] = {}
        for source_location in finding.evidence:
            evidence_id = stable_source_location_id(source_location)
            evidence = self.evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            for target_id in evidence.target_ids:
                target = self.target_by_id.get(target_id)
                if target is None:
                    continue
                keys_by_target_id.setdefault(
                    target_id, (target_id, f"{target.file_path}:{target.qualname}")
                )
        return tuple(
            keys_by_target_id[target_id] for target_id in sorted(keys_by_target_id)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "files": tuple(item.__dict__ for item in self.files),
            "ast_targets": tuple(item.__dict__ for item in self.ast_targets),
            "evidence": tuple(item.__dict__ for item in self.evidence),
        }


class _AstTargetDigestVisitor(ClassFunctionStackNodeVisitor):
    def __init__(self, file_id: str, file_path: str) -> None:
        super().__init__()
        self.file_id = file_id
        self.file_path = file_path
        self.targets: list[AstTargetDigest] = []

    def before_visit_class(self, node: ast.ClassDef) -> None:
        self._append_target(node, "class")

    def before_visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self._append_target(node, self._function_node_type())

    def _function_node_type(self) -> str:
        return "method" if self.class_stack else "function"

    def _append_target(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        node_type: str,
    ) -> None:
        qualname = ".".join((*self.class_stack, *self.function_stack, node.name))
        line = node.lineno
        end_line = node.end_lineno or line
        parameters = (
            self._parameters(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else ()
        )
        decorators = _decorator_names(node.decorator_list)
        base_names = _base_names(node.bases) if isinstance(node, ast.ClassDef) else ()
        self.targets.append(
            AstTargetDigest(
                target_id=_stable_id(
                    "ast-target",
                    self.file_path,
                    node_type,
                    qualname,
                    line,
                    end_line,
                ),
                file_id=self.file_id,
                file_path=self.file_path,
                node_type=node_type,
                name=node.name,
                qualname=qualname,
                line=line,
                end_line=end_line,
                parameters=parameters,
                decorators=decorators,
                base_names=base_names,
            )
        )

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


def _file_digest(module: ParsedModule) -> SourceFileDigest:
    file_path = Path(module.path).as_posix()
    return SourceFileDigest(
        file_id=_stable_id("file", file_path),
        file_path=file_path,
        module_name=module.module_name,
        is_package_init=module.is_package_init,
    )


def _ast_target_digests(
    module: ParsedModule, file_id: str
) -> tuple[AstTargetDigest, ...]:
    visitor = _AstTargetDigestVisitor(file_id, Path(module.path).as_posix())
    visitor.visit(module.module)
    return tuple(visitor.targets)


def _target_ids_for_evidence(
    evidence: SourceLocation, targets_by_file: TargetIndex
) -> tuple[str, ...]:
    file_targets = targets_by_file.get(evidence.file_path, ())
    symbol = evidence.symbol.rsplit(":", 1)[0]
    matches = [
        target
        for target in file_targets
        if target.line <= evidence.line <= target.end_line
        and (target.qualname == symbol or target.qualname.endswith(f".{symbol}"))
    ]
    if not matches:
        matches = [
            target
            for target in file_targets
            if target.line <= evidence.line <= target.end_line
        ]
    return tuple(target.target_id for target in matches[:3])


def _evidence_digests(
    findings: Iterable[RefactorFinding],
    file_ids_by_path: dict[str, str],
    targets_by_file: TargetIndex,
) -> tuple[EvidenceDigest, ...]:
    finding_ids_by_evidence: dict[str, list[str]] = {}
    evidence_by_id: dict[str, SourceLocation] = {}
    for finding in findings:
        for source_location in finding.evidence:
            evidence_id = stable_source_location_id(source_location)
            evidence_by_id.setdefault(evidence_id, source_location)
            finding_ids_by_evidence.setdefault(evidence_id, []).append(
                finding.stable_id
            )

    return tuple(
        EvidenceDigest(
            evidence_id=evidence_id,
            file_id=file_ids_by_path.get(source_location.file_path),
            file_path=source_location.file_path,
            line=source_location.line,
            symbol=source_location.symbol,
            finding_ids=sorted_tuple(set(finding_ids_by_evidence[evidence_id])),
            target_ids=_target_ids_for_evidence(source_location, targets_by_file),
        )
        for evidence_id, source_location in sorted(evidence_by_id.items())
    )


def build_source_index(
    modules: Iterable[ParsedModule], findings: Iterable[RefactorFinding]
) -> SourceIndex:
    """Build a source-address index from parsed modules and emitted findings."""

    module_tuple = tuple(modules)
    finding_tuple = tuple(findings)
    files = tuple(_file_digest(module) for module in module_tuple)
    file_ids_by_path = {item.file_path: item.file_id for item in files}
    ast_targets = tuple(
        target
        for module, file_digest in zip(module_tuple, files, strict=True)
        for target in _ast_target_digests(module, file_digest.file_id)
    )
    target_lists_by_file: dict[str, list[AstTargetDigest]] = {}
    for target in ast_targets:
        target_lists_by_file.setdefault(target.file_path, []).append(target)
    targets_by_file: TargetIndex = {
        file_path: tuple(targets) for file_path, targets in target_lists_by_file.items()
    }
    source_index = SourceIndex(
        files=files,
        ast_targets=ast_targets,
        evidence=_evidence_digests(finding_tuple, file_ids_by_path, targets_by_file),
    )
    _ = (
        source_index.evidence_by_id,
        source_index.target_by_id,
        source_index.targets_by_file,
        source_index.target_ids_by_finding_id,
        source_index.finding_ids_by_target_id,
    )
    return source_index
