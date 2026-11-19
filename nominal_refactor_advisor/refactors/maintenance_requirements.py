"""Positive maintenance requirements evaluated by the existing codemod preflight."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import AstParentIndex
from nominal_refactor_advisor.codemod_call_source import DeclaredCallAuthority
from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import (
    DeclaredCallSelection,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.json_reports import DataclassJsonReport, JsonReport
from nominal_refactor_advisor.source_geometry import SourceByteSpan


@dataclass(frozen=True)
class DeclaredCallRequirementResolution(DataclassJsonReport):
    """One proved consumer/provider call relation."""

    consumer: str
    provider: str
    call_count: int


@dataclass(frozen=True)
class AbsentDeclarationRequirementResolution(DataclassJsonReport):
    """One declaration proved absent from a selected final source scope."""

    scope: str
    absent_qualname: str


@dataclass(frozen=True)
class InheritanceRequirementResolution(DataclassJsonReport):
    """One consumer/provider ancestry answer."""

    consumer: str
    provider: str
    inherited: bool


@dataclass(frozen=True)
class IterationCallRequirementResolution(DataclassJsonReport):
    """Provider calls proved outside repeated source regions."""

    consumer: str
    provider: str
    call_count: int


@dataclass(frozen=True, kw_only=True)
class MaintenanceRequirementOperation(RepositorySourceReprovedOperation, ABC):
    """A no-edit operation whose source proof enters the normal preflight report."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        self.required_resolution(snapshot)
        return ()

    def required_resolution(self, snapshot: CodemodSourceSnapshot) -> JsonReport:
        return snapshot.retained_operation_derivation(
            (self, MaintenanceRequirementOperation.required_resolution),
            lambda: self.requirement_resolution(snapshot),
        )

    def preflight_reports(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            resolution = self.required_reproof(
                lambda: self.required_resolution(context.execution_snapshot())
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Positive maintenance requirement proved",
                detail=resolution,
            ),
        )

    @abstractmethod
    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> JsonReport: ...


@dataclass(frozen=True, kw_only=True)
class RequireDeclaredCallOperation(
    DeclaredCallSelection,
    MaintenanceRequirementOperation,
):
    """Require one consumer to call one provider through resolved declarations."""

    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclaredCallRequirementResolution:
        _consumer_id, consumer = self.target_digest(snapshot)
        provider_id = self.callee.required_target_id(snapshot.source_index)
        provider = snapshot.source_index.target_by_id[provider_id]
        calls = DeclaredCallAuthority(snapshot, consumer, provider).calls(
            self.selection_count
        )
        return DeclaredCallRequirementResolution(
            consumer=consumer.qualname,
            provider=provider.qualname,
            call_count=len(calls),
        )


@dataclass(frozen=True, kw_only=True)
class RequireAbsentDeclarationOperation(MaintenanceRequirementOperation):
    """Require one competing declaration to be absent from a final source module."""

    forbidden: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> AbsentDeclarationRequirementResolution:
        _scope_id, scope = self.target_digest(snapshot)
        if not scope.is_module:
            raise ValueError("Absent-declaration scope must be a module")
        target = self.forbidden.optional_target_id(snapshot.source_index)
        if target is not None:
            declaration = snapshot.source_index.target_by_id[target]
            raise ValueError(
                f"Competing declaration still exists: {declaration.qualname!r}"
            )
        candidates = self.forbidden.candidate_targets(
            snapshot.source_index,
            self.forbidden.optional_file_path(snapshot.source_index),
        )
        if candidates:
            raise ValueError("Competing declaration target is ambiguous")
        return AbsentDeclarationRequirementResolution(
            scope=scope.file_path,
            absent_qualname=self.forbidden.qualname or "",
        )


@dataclass(frozen=True, kw_only=True)
class InheritanceRequirementOperation(MaintenanceRequirementOperation, ABC):
    """One consumer/provider pair interpreted through resolved Python ancestry."""

    provider: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def resolved_inheritance(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> InheritanceRequirementResolution:
        consumer = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        provider = ResolvedClassTarget.from_rewrite_target(snapshot, self.provider)
        inherited = provider.required_symbol(
            snapshot
        ) in snapshot.required_class_family_index.ancestor_symbols(
            consumer.required_symbol(snapshot)
        )
        return InheritanceRequirementResolution(
            consumer=consumer.qualname,
            provider=provider.qualname,
            inherited=inherited,
        )


@dataclass(frozen=True, kw_only=True)
class RequireInheritanceOperation(InheritanceRequirementOperation):
    """Require the consumer to inherit the provider through native ancestry."""

    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> InheritanceRequirementResolution:
        resolution = self.resolved_inheritance(snapshot)
        if not resolution.inherited:
            raise ValueError(
                f"{resolution.consumer!r} does not inherit {resolution.provider!r}"
            )
        return resolution


@dataclass(frozen=True, kw_only=True)
class ForbidInheritanceOperation(InheritanceRequirementOperation):
    """Require the consumer not to inherit the provider."""

    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> InheritanceRequirementResolution:
        resolution = self.resolved_inheritance(snapshot)
        if resolution.inherited:
            raise ValueError(
                f"{resolution.consumer!r} still inherits {resolution.provider!r}"
            )
        return resolution


@dataclass(frozen=True, kw_only=True)
class RequireCallOutsideIterationOperation(
    DeclaredCallSelection,
    MaintenanceRequirementOperation,
):
    """Require resolved provider calls to execute outside repeated source regions."""

    def requirement_resolution(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> IterationCallRequirementResolution:
        consumer_id, consumer = self.target_digest(snapshot)
        provider_id = self.callee.required_target_id(snapshot.source_index)
        provider = snapshot.source_index.target_by_id[provider_id]
        calls = DeclaredCallAuthority(snapshot, consumer, provider).calls(
            self.selection_count
        )
        consumer_node = snapshot.ast_target_nodes_by_id[consumer_id]
        parents = AstParentIndex.of(consumer_node).parent_by_node
        call_nodes = {
            SourceByteSpan.require_node(node): node
            for node in ast.walk(consumer_node)
            if isinstance(node, ast.Call)
        }
        repeated_lines = tuple(
            sorted(
                {
                    loop.lineno
                    for call in calls
                    for node in (call_nodes[call.call.source_span],)
                    for loop in self.repeated_ancestors(node, parents)
                }
            )
        )
        if repeated_lines:
            raise ValueError(
                "Provider call executes inside repeated regions at lines "
                f"{repeated_lines!r}"
            )
        return IterationCallRequirementResolution(
            consumer=consumer.qualname,
            provider=provider.qualname,
            call_count=len(calls),
        )

    @classmethod
    def repeated_ancestors(
        cls,
        node: ast.AST,
        parents: Mapping[ast.AST, ast.AST],
    ) -> tuple[ast.AST, ...]:
        repeated = []
        child = node
        parent = parents.get(child)
        while parent is not None:
            if isinstance(parent, ast.For | ast.AsyncFor):
                if child is not parent.iter:
                    repeated.append(parent)
            elif isinstance(parent, ast.While):
                repeated.append(parent)
            elif isinstance(
                parent,
                ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
            ):
                outer_iter = parent.generators[0].iter
                if node is not outer_iter and node not in ast.walk(outer_iter):
                    repeated.append(parent)
            child = parent
            parent = parents.get(parent)
        return tuple(repeated)
