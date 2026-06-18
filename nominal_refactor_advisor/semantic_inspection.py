"""Agent-facing semantic AST inspection API.

This module projects parsed Python modules, advisor findings, and the existing
source-address index into compact frozen records. It intentionally reuses the
normal parser, detector, and source-index pipeline instead of exposing raw AST
nodes or building a parallel dump format.
"""

from __future__ import annotations

import ast
import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeAlias, cast

from .analysis import analyze_modules
from .ast_tools import ParsedModule, parse_python_module_roots
from .detectors import DetectorConfig
from .models import RefactorFinding, SemanticRecord, stable_source_location_id
from .source_index import (
    AstTargetDigest,
    SourceFileDigest,
    SourceIndex,
    build_source_index,
)

HashPart: TypeAlias = str | int | bool | None | tuple[str, ...]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar | tuple["JsonValue", ...] | list["JsonValue"] | dict[str, "JsonValue"]
)


def _semantic_id(namespace: str, *parts: HashPart) -> str:
    payload = "|".join((namespace, *(str(part) for part in parts)))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=5).hexdigest()


@dataclass(frozen=True)
class SemanticInspectionRecord(SemanticRecord, ABC):
    """Base class for serializable semantic inspection records."""


@dataclass(frozen=True)
class ImportSummary(SemanticInspectionRecord):
    import_id: str
    file_path: str
    module_name: str
    line: int
    import_kind: str
    imported_module: str | None
    imported_names: tuple[str, ...]
    alias_names: tuple[str, ...]
    level: int = 0


@dataclass(frozen=True)
class AssignmentSummary(SemanticInspectionRecord):
    assignment_id: str
    file_path: str
    module_name: str
    line: int
    scope_qualname: str
    target_id: str | None
    target_names: tuple[str, ...]
    value_kind: str


@dataclass(frozen=True)
class CallSummary(SemanticInspectionRecord):
    call_id: str
    file_path: str
    module_name: str
    line: int
    scope_qualname: str
    target_id: str | None
    callee: str
    argument_names: tuple[str, ...]
    keyword_names: tuple[str, ...]


@dataclass(frozen=True)
class FunctionSummary(SemanticInspectionRecord):
    target_id: str
    file_path: str
    module_name: str
    name: str
    qualname: str
    kind: str
    line: int
    end_line: int
    parameters: tuple[str, ...]
    decorators: tuple[str, ...]
    call_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClassSummary(SemanticInspectionRecord):
    target_id: str
    file_path: str
    module_name: str
    name: str
    qualname: str
    line: int
    end_line: int
    base_names: tuple[str, ...]
    decorators: tuple[str, ...]
    is_dataclass: bool
    method_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class DataclassSummary(SemanticInspectionRecord):
    target_id: str
    file_path: str
    module_name: str
    name: str
    qualname: str
    line: int
    field_names: tuple[str, ...]
    method_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModuleSummary(SemanticInspectionRecord):
    file_id: str
    file_path: str
    module_name: str
    is_package_init: bool
    ast_target_ids: tuple[str, ...]
    class_ids: tuple[str, ...]
    function_ids: tuple[str, ...]
    import_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    call_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class FindingSummary(SemanticInspectionRecord):
    finding_id: str
    detector_id: str
    pattern_id: int
    title: str
    summary: str
    confidence: str
    certification: str
    evidence_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceSummary(SemanticInspectionRecord):
    evidence_id: str
    file_id: str | None
    file_path: str
    line: int
    symbol: str
    finding_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticInspectionReport(SemanticInspectionRecord):
    roots: tuple[str, ...]
    modules: tuple[ModuleSummary, ...]
    classes: tuple[ClassSummary, ...]
    functions: tuple[FunctionSummary, ...]
    dataclasses: tuple[DataclassSummary, ...]
    imports: tuple[ImportSummary, ...]
    calls: tuple[CallSummary, ...]
    assignments: tuple[AssignmentSummary, ...]
    findings: tuple[FindingSummary, ...]
    evidence: tuple[EvidenceSummary, ...]
    ast_targets: tuple[AstTargetDigest, ...]
    source_index: SourceIndex

    def to_dict(self) -> dict[str, JsonValue]:
        payload = cast(dict[str, JsonValue], asdict(self))
        payload["ast_targets"] = tuple(
            cast(dict[str, JsonValue], item.__dict__) for item in self.ast_targets
        )
        payload["source_index"] = cast(
            dict[str, JsonValue], self.source_index.to_dict()
        )
        return payload


@dataclass(frozen=True)
class _TargetKey:
    file_path: str
    node_type: str
    qualname: str
    line: int


@dataclass(frozen=True)
class _Scope:
    qualname: str
    target_id: str
    node_type: str


@dataclass(frozen=True)
class _ModuleInspection:
    imports: tuple[ImportSummary, ...]
    calls: tuple[CallSummary, ...]
    assignments: tuple[AssignmentSummary, ...]
    classes: tuple[ClassSummary, ...]
    functions: tuple[FunctionSummary, ...]
    dataclasses: tuple[DataclassSummary, ...]


class SemanticAstInspector(ABC):
    """Abstract semantic inspection boundary for alternate frontends."""

    @abstractmethod
    def inspect_modules(
        self,
        modules: Sequence[ParsedModule],
        findings: Sequence[RefactorFinding] | None = None,
    ) -> SemanticInspectionReport:
        """Inspect already parsed modules with optional precomputed findings."""


@dataclass(frozen=True)
class SourceIndexSemanticAstInspector(SemanticAstInspector):
    """Semantic inspector backed by parsed modules, detectors, and SourceIndex."""

    roots: tuple[Path, ...] = ()
    config: DetectorConfig | None = None

    def inspect_paths(self, roots: tuple[Path, ...]) -> SemanticInspectionReport:
        modules = parse_python_module_roots(roots)
        return SourceIndexSemanticAstInspector(
            roots=roots,
            config=self.config,
        ).inspect_modules(modules)

    def inspect_modules(
        self,
        modules: Sequence[ParsedModule],
        findings: Sequence[RefactorFinding] | None = None,
    ) -> SemanticInspectionReport:
        module_tuple = tuple(modules)
        if findings is None:
            finding_tuple = tuple(analyze_modules(list(module_tuple), self.config))
        else:
            finding_tuple = tuple(findings)
        source_index = build_source_index(module_tuple, finding_tuple)
        projection = _SemanticInspectionProjection(module_tuple, source_index)
        return projection.build_report(
            roots=self.roots,
            findings=finding_tuple,
        )


class _SemanticInspectionProjection:
    def __init__(
        self,
        modules: tuple[ParsedModule, ...],
        source_index: SourceIndex,
    ) -> None:
        self.modules = modules
        self.source_index = source_index
        self.targets_by_key = {
            _TargetKey(
                file_path=target.file_path,
                node_type=target.node_type,
                qualname=target.qualname,
                line=target.line,
            ): target
            for target in source_index.ast_targets
        }

    def build_report(
        self,
        *,
        roots: tuple[Path, ...],
        findings: tuple[RefactorFinding, ...],
    ) -> SemanticInspectionReport:
        inspections = tuple(self._inspect_module(module) for module in self.modules)
        imports = tuple(
            item for inspection in inspections for item in inspection.imports
        )
        calls = tuple(item for inspection in inspections for item in inspection.calls)
        assignments = tuple(
            item for inspection in inspections for item in inspection.assignments
        )
        classes = tuple(
            item for inspection in inspections for item in inspection.classes
        )
        functions = tuple(
            item for inspection in inspections for item in inspection.functions
        )
        dataclasses = tuple(
            item for inspection in inspections for item in inspection.dataclasses
        )
        modules = tuple(
            self._module_summary(
                module,
                imports=imports,
                calls=calls,
                assignments=assignments,
                findings=findings,
            )
            for module in self.modules
        )
        return SemanticInspectionReport(
            roots=tuple(path.as_posix() for path in roots),
            modules=modules,
            classes=classes,
            functions=functions,
            dataclasses=dataclasses,
            imports=imports,
            calls=calls,
            assignments=assignments,
            findings=self._finding_summaries(findings),
            evidence=self._evidence_summaries(),
            ast_targets=self.source_index.ast_targets,
            source_index=self.source_index,
        )

    def _inspect_module(self, module: ParsedModule) -> _ModuleInspection:
        visitor = _ModuleSemanticVisitor(module, self.source_index, self.targets_by_key)
        visitor.visit(module.module)
        return visitor.inspection()

    def _module_summary(
        self,
        module: ParsedModule,
        *,
        imports: tuple[ImportSummary, ...],
        calls: tuple[CallSummary, ...],
        assignments: tuple[AssignmentSummary, ...],
        findings: tuple[RefactorFinding, ...],
    ) -> ModuleSummary:
        file_path = module.path.as_posix()
        file_digest = self._file_digest(file_path)
        if file_path in self.source_index.targets_by_file:
            targets = self.source_index.targets_by_file[file_path]
        else:
            targets = ()
        class_ids = tuple(
            target.target_id for target in targets if target.node_type == "class"
        )
        function_ids = tuple(
            target.target_id for target in targets if target.node_type != "class"
        )
        return ModuleSummary(
            file_id=file_digest.file_id,
            file_path=file_path,
            module_name=module.module_name,
            is_package_init=module.is_package_init,
            ast_target_ids=tuple(target.target_id for target in targets),
            class_ids=class_ids,
            function_ids=function_ids,
            import_ids=tuple(
                item.import_id for item in imports if item.file_path == file_path
            ),
            assignment_ids=tuple(
                item.assignment_id
                for item in assignments
                if item.file_path == file_path
            ),
            call_ids=tuple(
                item.call_id for item in calls if item.file_path == file_path
            ),
            finding_ids=self._finding_ids_for_file(findings, file_path),
        )

    def _file_digest(self, file_path: str) -> SourceFileDigest:
        for file_digest in self.source_index.files:
            if file_digest.file_path == file_path:
                return file_digest
        raise KeyError(f"source index has no file digest for {file_path}")

    def _finding_summaries(
        self, findings: tuple[RefactorFinding, ...]
    ) -> tuple[FindingSummary, ...]:
        return tuple(
            FindingSummary(
                finding_id=finding.stable_id,
                detector_id=finding.detector_id,
                pattern_id=int(finding.pattern_id),
                title=finding.title,
                summary=finding.summary,
                confidence=str(finding.confidence),
                certification=str(finding.certification),
                evidence_ids=tuple(
                    stable_source_location_id(item) for item in finding.evidence
                ),
                target_ids=self.source_index.target_ids_for_finding_ids(
                    (finding.stable_id,)
                ),
            )
            for finding in findings
        )

    def _evidence_summaries(self) -> tuple[EvidenceSummary, ...]:
        return tuple(
            EvidenceSummary(
                evidence_id=item.evidence_id,
                file_id=item.file_id,
                file_path=item.file_path,
                line=item.line,
                symbol=item.symbol,
                finding_ids=item.finding_ids,
                target_ids=item.target_ids,
            )
            for item in self.source_index.evidence
        )

    @staticmethod
    def _finding_ids_for_file(
        findings: tuple[RefactorFinding, ...], file_path: str
    ) -> tuple[str, ...]:
        finding_ids = {
            finding.stable_id
            for finding in findings
            for evidence in finding.evidence
            if evidence.file_path == file_path
        }
        return tuple(sorted(finding_ids))


class _ModuleSemanticVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: ParsedModule,
        source_index: SourceIndex,
        targets_by_key: dict[_TargetKey, AstTargetDigest],
    ) -> None:
        self.module = module
        self.file_path = module.path.as_posix()
        self.source_index = source_index
        self.targets_by_key = targets_by_key
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.scope_stack: list[_Scope] = []
        self.imports: list[ImportSummary] = []
        self.calls: list[CallSummary] = []
        self.assignments: list[AssignmentSummary] = []
        self.classes: list[ClassSummary] = []
        self.functions: list[FunctionSummary] = []
        self.dataclasses: list[DataclassSummary] = []
        self.call_ids_by_target_id: dict[str, list[str]] = {}
        self.assignment_ids_by_target_id: dict[str, list[str]] = {}
        self.method_ids_by_class_target_id: dict[str, list[str]] = {}

    def inspection(self) -> _ModuleInspection:
        return _ModuleInspection(
            imports=tuple(self.imports),
            calls=tuple(self.calls),
            assignments=tuple(self.assignments),
            classes=tuple(self.classes),
            functions=tuple(self.functions),
            dataclasses=tuple(self.dataclasses),
        )

    def visit_Import(self, node: ast.Import) -> None:
        imported_names = tuple(alias.name for alias in node.names)
        self.imports.append(
            ImportSummary(
                import_id=_semantic_id(
                    "import", self.file_path, node.lineno, imported_names
                ),
                file_path=self.file_path,
                module_name=self.module.module_name,
                line=node.lineno,
                import_kind="import",
                imported_module=None,
                imported_names=imported_names,
                alias_names=_alias_names(node.names),
            )
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_names = tuple(alias.name for alias in node.names)
        self.imports.append(
            ImportSummary(
                import_id=_semantic_id(
                    "import-from",
                    self.file_path,
                    node.lineno,
                    node.module,
                    imported_names,
                ),
                file_path=self.file_path,
                module_name=self.module.module_name,
                line=node.lineno,
                import_kind="from_import",
                imported_module=node.module,
                imported_names=imported_names,
                alias_names=_alias_names(node.names),
                level=node.level,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._qualname_for(node.name)
        target = self._target_for("class", qualname, node.lineno)
        class_scope = _Scope(
            qualname=target.qualname,
            target_id=target.target_id,
            node_type=target.node_type,
        )
        decorators = _decorator_names(node.decorator_list)
        is_dataclass = _is_dataclass_decorator(decorators)
        self.classes.append(
            ClassSummary(
                target_id=target.target_id,
                file_path=self.file_path,
                module_name=self.module.module_name,
                name=node.name,
                qualname=target.qualname,
                line=target.line,
                end_line=target.end_line,
                base_names=target.base_names,
                decorators=decorators,
                is_dataclass=is_dataclass,
                method_ids=(),
                assignment_ids=(),
                finding_ids=self.source_index.finding_ids_for_target_id(
                    target.target_id
                ),
            )
        )
        if is_dataclass:
            self.dataclasses.append(
                DataclassSummary(
                    target_id=target.target_id,
                    file_path=self.file_path,
                    module_name=self.module.module_name,
                    name=node.name,
                    qualname=target.qualname,
                    line=target.line,
                    field_names=_class_field_names(node.body),
                    method_ids=(),
                )
            )
        self.class_stack.append(node.name)
        self.scope_stack.append(class_scope)
        try:
            self.generic_visit(node)
        finally:
            self.scope_stack.pop()
            self.class_stack.pop()
        self._refresh_class_record(target.target_id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def visit_Call(self, node: ast.Call) -> None:
        scope = self._current_scope()
        if scope is None:
            target_id = None
            scope_qualname = "<module>"
        else:
            target_id = scope.target_id
            scope_qualname = scope.qualname
        call = CallSummary(
            call_id=_semantic_id(
                "call",
                self.file_path,
                node.lineno,
                scope_qualname,
                _expression_label(node.func),
            ),
            file_path=self.file_path,
            module_name=self.module.module_name,
            line=node.lineno,
            scope_qualname=scope_qualname,
            target_id=target_id,
            callee=_expression_label(node.func),
            argument_names=tuple(_expression_label(arg) for arg in node.args),
            keyword_names=tuple(_keyword_name(keyword) for keyword in node.keywords),
        )
        self.calls.append(call)
        if target_id is not None:
            self._append_target_value(
                self.call_ids_by_target_id,
                target_id,
                call.call_id,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._append_assignment(node, node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            value_kind = "None"
        else:
            value_kind = type(node.value).__name__
        self._append_assignment_with_value_kind(
            node=node,
            target_names=_target_names(node.target),
            value_kind=value_kind,
        )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._append_assignment_with_value_kind(
            node=node,
            target_names=_target_names(node.target),
            value_kind=type(node.value).__name__,
        )
        self.generic_visit(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        qualname = self._qualname_for(node.name)
        if self.class_stack:
            node_type = "method"
        else:
            node_type = "function"
        target = self._target_for(node_type, qualname, node.lineno)
        kind = _function_kind(node_type, is_async=is_async)
        self.functions.append(
            FunctionSummary(
                target_id=target.target_id,
                file_path=self.file_path,
                module_name=self.module.module_name,
                name=node.name,
                qualname=target.qualname,
                kind=kind,
                line=target.line,
                end_line=target.end_line,
                parameters=target.parameters,
                decorators=_decorator_names(node.decorator_list),
                call_ids=(),
                assignment_ids=(),
                finding_ids=self.source_index.finding_ids_for_target_id(
                    target.target_id
                ),
            )
        )
        if self.scope_stack and self.scope_stack[-1].node_type == "class":
            class_target_id = self.scope_stack[-1].target_id
            self._append_target_value(
                self.method_ids_by_class_target_id,
                class_target_id,
                target.target_id,
            )
        self.function_stack.append(node.name)
        self.scope_stack.append(
            _Scope(
                qualname=target.qualname,
                target_id=target.target_id,
                node_type=target.node_type,
            )
        )
        try:
            self.generic_visit(node)
        finally:
            self.scope_stack.pop()
            self.function_stack.pop()
        self._refresh_function_record(target.target_id)

    def _append_assignment(
        self, node: ast.Assign, targets: Iterable[ast.expr], value: ast.expr
    ) -> None:
        target_names = tuple(
            name for target in targets for name in _target_names(target)
        )
        self._append_assignment_with_value_kind(
            node=node,
            target_names=target_names,
            value_kind=type(value).__name__,
        )

    def _append_assignment_with_value_kind(
        self, *, node: ast.AST, target_names: tuple[str, ...], value_kind: str
    ) -> None:
        line = _node_line(node)
        scope = self._current_scope()
        if scope is None:
            target_id = None
            scope_qualname = "<module>"
        else:
            target_id = scope.target_id
            scope_qualname = scope.qualname
        assignment = AssignmentSummary(
            assignment_id=_semantic_id(
                "assignment",
                self.file_path,
                line,
                scope_qualname,
                target_names,
            ),
            file_path=self.file_path,
            module_name=self.module.module_name,
            line=line,
            scope_qualname=scope_qualname,
            target_id=target_id,
            target_names=target_names,
            value_kind=value_kind,
        )
        self.assignments.append(assignment)
        if target_id is not None:
            self._append_target_value(
                self.assignment_ids_by_target_id,
                target_id,
                assignment.assignment_id,
            )

    def _target_for(self, node_type: str, qualname: str, line: int) -> AstTargetDigest:
        key = _TargetKey(
            file_path=self.file_path,
            node_type=node_type,
            qualname=qualname,
            line=line,
        )
        return self.targets_by_key[key]

    def _qualname_for(self, name: str) -> str:
        return ".".join((*self.class_stack, *self.function_stack, name))

    def _current_scope(self) -> _Scope | None:
        if not self.scope_stack:
            return None
        return self.scope_stack[-1]

    def _refresh_function_record(self, target_id: str) -> None:
        function = next(item for item in self.functions if item.target_id == target_id)
        index = self.functions.index(function)
        self.functions[index] = FunctionSummary(
            target_id=function.target_id,
            file_path=function.file_path,
            module_name=function.module_name,
            name=function.name,
            qualname=function.qualname,
            kind=function.kind,
            line=function.line,
            end_line=function.end_line,
            parameters=function.parameters,
            decorators=function.decorators,
            call_ids=self._target_values(self.call_ids_by_target_id, target_id),
            assignment_ids=self._target_values(
                self.assignment_ids_by_target_id,
                target_id,
            ),
            finding_ids=function.finding_ids,
        )

    def _refresh_class_record(self, target_id: str) -> None:
        class_summary = next(
            item for item in self.classes if item.target_id == target_id
        )
        index = self.classes.index(class_summary)
        method_ids = self._target_values(
            self.method_ids_by_class_target_id,
            target_id,
        )
        self.classes[index] = ClassSummary(
            target_id=class_summary.target_id,
            file_path=class_summary.file_path,
            module_name=class_summary.module_name,
            name=class_summary.name,
            qualname=class_summary.qualname,
            line=class_summary.line,
            end_line=class_summary.end_line,
            base_names=class_summary.base_names,
            decorators=class_summary.decorators,
            is_dataclass=class_summary.is_dataclass,
            method_ids=method_ids,
            assignment_ids=self._target_values(
                self.assignment_ids_by_target_id,
                target_id,
            ),
            finding_ids=class_summary.finding_ids,
        )
        if class_summary.is_dataclass:
            dataclass_summary = next(
                item for item in self.dataclasses if item.target_id == target_id
            )
            dataclass_index = self.dataclasses.index(dataclass_summary)
            self.dataclasses[dataclass_index] = DataclassSummary(
                target_id=dataclass_summary.target_id,
                file_path=dataclass_summary.file_path,
                module_name=dataclass_summary.module_name,
                name=dataclass_summary.name,
                qualname=dataclass_summary.qualname,
                line=dataclass_summary.line,
                field_names=dataclass_summary.field_names,
                method_ids=method_ids,
            )

    @staticmethod
    def _target_values(
        target_values: dict[str, list[str]], target_id: str
    ) -> tuple[str, ...]:
        if target_id in target_values:
            return tuple(target_values[target_id])
        return ()

    @staticmethod
    def _append_target_value(
        target_values: dict[str, list[str]], target_id: str, value: str
    ) -> None:
        if target_id in target_values:
            target_values[target_id].append(value)
        else:
            target_values[target_id] = [value]


def _function_kind(node_type: str, *, is_async: bool) -> str:
    if node_type == "method":
        if is_async:
            return "async_method"
        return "method"
    if is_async:
        return "async_function"
    return "function"


def _node_line(node: ast.AST) -> int:
    if isinstance(node, ast.stmt):
        return node.lineno
    raise TypeError(f"expected statement node, got {type(node).__name__}")


def _alias_names(aliases: Iterable[ast.alias]) -> tuple[str, ...]:
    names: list[str] = []
    for alias in aliases:
        if alias.asname is None:
            names.append(alias.name)
        else:
            names.append(alias.asname)
    return tuple(names)


def _decorator_names(decorators: Iterable[ast.expr]) -> tuple[str, ...]:
    return tuple(_expression_label(decorator) for decorator in decorators)


def _is_dataclass_decorator(decorators: tuple[str, ...]) -> bool:
    dataclass_decorator_names = frozenset(("dataclass", "dataclasses.dataclass"))
    return any(name in dataclass_decorator_names for name in decorators)


def _class_field_names(body: Iterable[ast.stmt]) -> tuple[str, ...]:
    names: list[str] = []
    for statement in body:
        if isinstance(statement, ast.AnnAssign):
            names.extend(_target_names(statement.target))
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.extend(_target_names(target))
    return tuple(names)


def _target_names(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (_expression_label(node),)
    if isinstance(node, ast.Subscript):
        return (_expression_label(node.value),)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for item in node.elts for name in _target_names(item))
    return ()


def _keyword_name(keyword: ast.keyword) -> str:
    if keyword.arg is None:
        return "**"
    return keyword.arg


def _expression_label(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_label(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _expression_label(node.func)
    if isinstance(node, ast.Subscript):
        return _expression_label(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Starred):
        return f"*{_expression_label(node.value)}"
    return type(node).__name__


def inspect_path(
    root: Path, config: DetectorConfig | None = None
) -> SemanticInspectionReport:
    """Parse, analyze, index, and semantically inspect one path."""

    return inspect_paths((root,), config=config)


def inspect_paths(
    roots: tuple[Path, ...], config: DetectorConfig | None = None
) -> SemanticInspectionReport:
    """Parse, analyze, index, and semantically inspect file or directory roots."""

    inspector = SourceIndexSemanticAstInspector(roots=roots, config=config)
    modules = parse_python_module_roots(roots)
    return inspector.inspect_modules(modules)


def inspect_modules(
    modules: Sequence[ParsedModule],
    findings: Sequence[RefactorFinding] | None = None,
    config: DetectorConfig | None = None,
) -> SemanticInspectionReport:
    """Inspect parsed modules with optional caller-supplied findings."""

    inspector = SourceIndexSemanticAstInspector(config=config)
    return inspector.inspect_modules(modules, findings=findings)


__all__ = (
    "AssignmentSummary",
    "CallSummary",
    "ClassSummary",
    "DataclassSummary",
    "EvidenceSummary",
    "FindingSummary",
    "FunctionSummary",
    "ImportSummary",
    "ModuleSummary",
    "SemanticAstInspector",
    "SemanticInspectionRecord",
    "SemanticInspectionReport",
    "SourceIndexSemanticAstInspector",
    "inspect_modules",
    "inspect_path",
    "inspect_paths",
)
