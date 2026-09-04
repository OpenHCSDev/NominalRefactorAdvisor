from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Iterable
from dataclasses import dataclass
from metaclass_registry import AutoRegisterMeta
from typing import (
    ClassVar,
    Self,
    cast,
)

from .class_index import ClassFamilyIndex
from .codemod_payload import (
    BooleanPayloadValueCodec,
    DiscriminatedPayloadRecord,
    OptionalStringArrayPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    codemod_payload_field,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import (
    CallSiteDigest,
    CodemodTargetSelection,
    NodeKindArrayPayloadValueCodec,
    RegexPatternSet,
    SourceRewriteTarget,
)
from .codemod_source_edits import SourceTargetEditor
from .collection_algebra import sorted_tuple
from .json_reports import (
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)
from .models import RefactorFinding
from .registry_identity import (
    DEFAULT_REGISTRY_KEY_ATTRIBUTE,
    suffix_trimmed_class_name_registry_key,
)
from .source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
    SourceIndex,
)


@dataclass(frozen=True)
class CodemodTargetSelector(
    DiscriminatedPayloadRecord,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Semantic selector that resolves to source-index target ids."""

    __registry__: ClassVar[dict[str, type["CodemodTargetSelector"]]] = {}
    __registry_key__ = DEFAULT_REGISTRY_KEY_ATTRIBUTE
    __key_extractor__ = staticmethod(suffix_trimmed_class_name_registry_key)
    __skip_if_no_key__ = True
    registry_key_suffix: ClassVar[str] = "Selector"
    registry_key: ClassVar[str]
    discriminator_field_name: ClassVar[str] = "selector"

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        selector_type = cls.__registry__.get(discriminator)
        if selector_type is None or not issubclass(selector_type, cls):
            raise ValueError(f"Unsupported target selector: {discriminator}")
        return cast(type[Self], selector_type)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.registry_key

    def select(self, context: CodemodSelectorContext) -> CodemodTargetSelection:
        return CodemodTargetSelection(self.target_ids(context))

    @abstractmethod
    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class FindingEvidenceTargetSelector(CodemodTargetSelector):
    """Select source-index targets connected to advisor finding evidence."""

    finding_ids: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )

    @classmethod
    def from_findings(
        cls,
        findings: Iterable[RefactorFinding],
    ) -> "FindingEvidenceTargetSelector":
        return cls(tuple(finding.stable_id for finding in findings))

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        return context.source_index.target_ids_for_finding_ids(self.finding_ids)


@dataclass(frozen=True)
class TargetSetExpressionSelector(CodemodTargetSelector):
    """Compose selectors with union, intersection, and exclusion."""

    include: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )
    require: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )
    exclude: tuple[CodemodTargetSelector, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CodemodTargetSelector),
        default=(),
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        if not (self.include or self.require or self.exclude):
            raise ValueError("Target set expression selector cannot be empty")
        selected_target_ids = self.included_target_ids(context)
        for selector in self.require:
            selected_target_ids.intersection_update(selector.target_ids(context))
        for selector in self.exclude:
            selected_target_ids.difference_update(selector.target_ids(context))
        return sorted_tuple(selected_target_ids)

    def included_target_ids(self, context: CodemodSelectorContext) -> set[str]:
        if not self.include:
            return set(context.source_index.target_by_id)
        selected_target_ids: set[str] = set()
        for selector in self.include:
            selected_target_ids.update(selector.target_ids(context))
        return selected_target_ids


@dataclass(frozen=True)
class SourceIndexTargetSelector(CodemodTargetSelector):
    """Select source-index AST targets by kind, path, qualname, or regex."""

    node_kinds: tuple[AstTargetNodeKind, ...] = codemod_payload_field(
        NodeKindArrayPayloadValueCodec(),
        default=(),
    )
    file_paths: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    qualnames: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    file_path_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    name_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    qualname_patterns: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )

    @classmethod
    def for_function_or_method(
        cls,
        file_path: str,
        qualname: str,
    ) -> "SourceIndexTargetSelector":
        return cls(
            node_kinds=(AstTargetNodeKind.FUNCTION, AstTargetNodeKind.METHOD),
            file_paths=(file_path,),
            qualnames=(qualname,),
        )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        node_kinds = frozenset(self.node_kinds)
        file_paths = context.resolve_source_paths(self.file_paths)
        qualnames = frozenset(self.qualnames)
        file_path_patterns = RegexPatternSet.from_patterns(self.file_path_patterns)
        name_patterns = RegexPatternSet.from_patterns(self.name_patterns)
        qualname_patterns = RegexPatternSet.from_patterns(self.qualname_patterns)
        candidate_targets = self.candidate_targets(context, file_paths)
        return sorted_tuple(
            target.target_id
            for target in candidate_targets
            if (not node_kinds or target.node_kind in node_kinds)
            and (not file_paths or target.file_path in file_paths)
            and (not qualnames or target.qualname in qualnames)
            and file_path_patterns.matches(target.file_path)
            and name_patterns.matches(target.name)
            and qualname_patterns.matches(target.qualname)
        )

    @staticmethod
    def candidate_targets(
        context: CodemodSelectorContext,
        file_paths: frozenset[str],
    ) -> tuple[AstTargetDigest, ...]:
        if not file_paths:
            return context.source_index.ast_targets
        targets_by_file = context.source_index.targets_by_file
        return tuple(
            target
            for file_path in sorted(file_paths)
            if targets_by_file.contains_file(file_path)
            for target in targets_by_file[file_path]
        )


@dataclass(frozen=True)
class ClassFamilyTargetSelector(CodemodTargetSelector):
    """Select class targets from class-family symbols and graph closure."""

    class_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )
    include_self: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )
    include_ancestors: bool = codemod_payload_field(
        BooleanPayloadValueCodec(),
        default=False,
    )
    include_descendants: bool = codemod_payload_field(
        BooleanPayloadValueCodec(),
        default=False,
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        class_index = context.required_class_family_index
        symbols: set[str] = set()
        if self.include_self:
            symbols.update(self.class_symbols)
        for symbol in self.class_symbols:
            if self.include_ancestors:
                symbols.update(class_index.ancestor_symbols(symbol))
            if self.include_descendants:
                symbols.update(class_index.descendant_symbols(symbol))
        return self.target_ids_for_symbols(context.source_index, class_index, symbols)

    @staticmethod
    def target_ids_for_symbols(
        source_index: SourceIndex,
        class_index: ClassFamilyIndex,
        symbols: Iterable[str],
    ) -> tuple[str, ...]:
        target_ids = []
        for symbol in symbols:
            indexed_class = class_index.class_for(symbol)
            if indexed_class is None:
                continue
            target = SourceRewriteTarget(
                qualname=indexed_class.qualname,
                file_path=indexed_class.file_path,
            )
            target_id = target.optional_target_id(source_index)
            if target_id is not None:
                target_ids.append(target_id)
        return sorted_tuple(target_ids)


@dataclass(frozen=True)
class InheritanceEdgeTargetSelector(CodemodTargetSelector):
    """Select class targets participating in resolved inheritance edges."""

    parent_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    child_symbols: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )
    include_parents: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )
    include_children: bool = codemod_payload_field(
        BooleanPayloadValueCodec(declared_default=True),
        default=True,
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        class_index = context.required_class_family_index
        selected_symbols: set[str] = set()
        parent_filter = frozenset(self.parent_symbols)
        child_filter = frozenset(self.child_symbols)
        for child_symbol, indexed_class in class_index.classes_by_symbol.items():
            for parent_symbol in indexed_class.resolved_base_symbols:
                if parent_filter and parent_symbol not in parent_filter:
                    continue
                if child_filter and child_symbol not in child_filter:
                    continue
                if self.include_parents:
                    selected_symbols.add(parent_symbol)
                if self.include_children:
                    selected_symbols.add(child_symbol)
        return ClassFamilyTargetSelector.target_ids_for_symbols(
            context.source_index,
            class_index,
            selected_symbols,
        )


@dataclass(frozen=True)
class CallSiteSelector:
    """Select call sites by surface callee name."""

    callee_names: tuple[str, ...]

    def call_sites(self, context: CodemodSelectorContext) -> tuple[CallSiteDigest, ...]:
        allowed_names = frozenset(self.callee_names)
        call_sites = []
        for file_path, source in context.sources_by_file_path.items():
            visitor = _CallSiteSelectorVisitor(
                file_path=file_path,
                source_index=context.source_index,
                allowed_names=allowed_names,
            )
            visitor.visit(ast.parse(source, filename=file_path))
            call_sites.extend(visitor.call_sites)
        return sorted_tuple(
            call_sites,
            key=lambda item: (item.file_path, item.line, item.symbol),
        )


@dataclass(frozen=True)
class CallSiteTargetSelector(CodemodTargetSelector):
    """Select source-index targets that enclose matching call sites."""

    callee_names: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec()
    )

    def target_ids(self, context: CodemodSelectorContext) -> tuple[str, ...]:
        return sorted_tuple(
            {
                site.enclosing_target_id
                for site in CallSiteSelector(self.callee_names).call_sites(context)
                if site.enclosing_target_id is not None
            }
        )


@dataclass(frozen=True)
class CodemodSelectorResolutionReport(DataclassJsonReport):
    """JSON-ready report for a codemod target selector dry run."""

    selector: CodemodTargetSelector
    selected_target_ids: tuple[str, ...]
    selected_targets: tuple[AstTargetDigest, ...]
    missing_target_ids: tuple[str, ...] = ()

    @json_report_property()
    def selected_count(self) -> int:
        return len(self.selected_targets)

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        context: CodemodSelectorContext,
    ) -> "CodemodSelectorResolutionReport":
        selected_target_ids = selector.target_ids(context)
        selected_targets = tuple(
            context.source_index.target_by_id[target_id]
            for target_id in selected_target_ids
            if target_id in context.source_index.target_by_id
        )
        missing_target_ids = tuple(
            target_id
            for target_id in selected_target_ids
            if target_id not in context.source_index.target_by_id
        )
        return cls(
            selector=selector,
            selected_target_ids=selected_target_ids,
            selected_targets=selected_targets,
            missing_target_ids=missing_target_ids,
        )


@dataclass(frozen=True)
class CodemodTargetSourceRecord(DataclassJsonReport):
    """One selected source-index target with its exact source span."""

    target: AstTargetDigest
    source: str

    @classmethod
    def from_context(
        cls,
        target: AstTargetDigest,
        context: CodemodSelectorContext,
    ) -> "CodemodTargetSourceRecord":
        return cls(
            target=target,
            source="".join(
                SourceTargetEditor(context.sources_by_file_path, target).target_lines
            ),
        )

    @json_report_property()
    def line_count(self) -> int:
        return self.target.end_line - self.target.line + 1


@dataclass(frozen=True)
class CodemodTargetSourceReport(DataclassJsonReport):
    """JSON-ready exact source spans for selected codemod targets."""

    selector_resolution: CodemodSelectorResolutionReport = json_report_field(
        included=False
    )
    records: tuple[CodemodTargetSourceRecord, ...] = json_report_field(included=False)

    @json_report_property()
    def selector(self) -> CodemodTargetSelector:
        return self.selector_resolution.selector

    @json_report_property()
    def selected_count(self) -> int:
        return len(self.records)

    @json_report_property()
    def selected_target_ids(self) -> tuple[str, ...]:
        return self.selector_resolution.selected_target_ids

    @json_report_property()
    def missing_target_ids(self) -> tuple[str, ...]:
        return self.selector_resolution.missing_target_ids

    @json_report_property(field_name="targets")
    def target_records(self) -> tuple[CodemodTargetSourceRecord, ...]:
        return self.records

    @classmethod
    def from_selector_context(
        cls,
        selector: CodemodTargetSelector,
        context: CodemodSelectorContext,
    ) -> "CodemodTargetSourceReport":
        selector_resolution = CodemodSelectorResolutionReport.from_selector_context(
            selector,
            context,
        )
        return cls(
            selector_resolution=selector_resolution,
            records=tuple(
                CodemodTargetSourceRecord.from_context(target, context)
                for target in selector_resolution.selected_targets
            ),
        )


class _CallSiteSelectorVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        file_path: str,
        source_index: SourceIndex,
        allowed_names: frozenset[str],
    ) -> None:
        self.file_path = file_path
        self.source_index = source_index
        self.allowed_names = allowed_names
        self.call_sites: list[CallSiteDigest] = []

    def visit_Call(self, node: ast.Call) -> None:
        symbol = self.call_symbol(node)
        if symbol in self.allowed_names:
            self.call_sites.append(
                CallSiteDigest(
                    file_path=self.file_path,
                    line=node.lineno,
                    symbol=symbol,
                    enclosing_target_id=self.enclosing_target_id(node.lineno),
                )
            )
        self.generic_visit(node)

    def enclosing_target_id(self, line: int) -> str | None:
        candidates = [
            target
            for target in self.source_index.ast_targets
            if target.file_path == self.file_path
            and target.contains_line(line)
            and not target.is_module
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (target.end_line - target.line, target.line),
        ).target_id

    @staticmethod
    def call_symbol(node: ast.Call) -> str:
        return _call_surface_name(node.func)


def _call_surface_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_surface_name(node.value)
        if not parent:
            return node.attr
        return f"{parent}.{node.attr}"
    return ""
