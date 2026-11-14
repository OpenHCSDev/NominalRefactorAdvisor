from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.class_index import CompactClassFamilyIndex
from nominal_refactor_advisor.codemod_architecture_guards import (
    ArchitectureGuardRule,
    ArchitectureGuardTargetScope,
    ForbiddenAttributeArchitectureGuardConstraint,
    ForbiddenCallArchitectureGuardConstraint,
    ForbiddenImportArchitectureGuardConstraint,
    ForbiddenReferenceArchitectureGuardConstraint,
)
from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_payload import (
    PayloadRecordValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import (
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_semantics import CodemodSourceDependencyScope
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEdit,
    SourceSpanDeletion,
)
from nominal_refactor_advisor.codemod_statement_source import AssignmentDeletionSource
from nominal_refactor_advisor.codemod_target_selectors import CodemodTargetSelector
from nominal_refactor_advisor.registry_identity import (
    INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
)
from nominal_refactor_advisor.source_index import AstTargetNodeKind


@dataclass(frozen=True, kw_only=True)
class AssignmentDeletionOperationABC(SourceReprovedOperation, ABC):
    """Explicit removal of selected assignments, including their evaluations."""

    assignment_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def __post_init__(self) -> None:
        operation_key = self.operation_key()
        if not self.assignment_names:
            raise ValueError(f"{operation_key} requires assignment_names")
        if any(not name or not name.isidentifier() for name in self.assignment_names):
            raise ValueError(f"{operation_key} requires Python identifier names")
        if len(set(self.assignment_names)) != len(self.assignment_names):
            raise ValueError(f"{operation_key} requires unique assignment_names")

    @abstractmethod
    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        authority = self.source_authority(snapshot)
        return authority.physical_edits(
            file_path=authority.file_path,
            replacements=authority.replacements(self.assignment_names),
            rationale=self.rationale
            or f"Remove assignments {self.assignment_names!r} and their evaluations.",
        )


@dataclass(frozen=True, kw_only=True)
class NamedScopeAssignmentDeletionOperationABC(AssignmentDeletionOperationABC, ABC):
    """Resolve a named scope using the leaf's existing AST-kind declaration."""

    @property
    @abstractmethod
    def scope_kind(self) -> AstTargetNodeKind:
        raise NotImplementedError

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        _identifier, target, node = self.target_node_from_context(snapshot)
        if not self.scope_kind.accepts(node):
            raise ValueError(
                f"Target {target.qualname!r} is not a {self.scope_kind.value} definition"
            )
        return AssignmentDeletionSource(
            source=snapshot.sources_by_file_path[target.file_path],
            node=node,
            file_path=target.file_path,
        )


@dataclass(frozen=True, kw_only=True)
class DeleteClassAssignmentsOperation(NamedScopeAssignmentDeletionOperationABC):
    """Explicitly delete complete class-level assignment statements."""

    scope_kind = AstTargetNodeKind.CLASS


@dataclass(frozen=True, kw_only=True)
class DeleteFunctionAssignmentsOperation(NamedScopeAssignmentDeletionOperationABC):
    """Explicitly delete direct function assignments and their evaluations."""

    scope_kind = AstTargetNodeKind.FUNCTION


@dataclass(frozen=True, kw_only=True)
class DeleteInheritedAutoRegisterConfigurationOperation(
    RepositorySourceReprovedOperation
):
    """Delete only configuration currently proved identical to an inherited value."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_id, target, node = self.target_node_from_context(snapshot)
        if not isinstance(node, ast.ClassDef):
            raise ValueError("Inherited AutoRegister configuration requires a class")
        class_index = CompactClassFamilyIndex.from_modules(snapshot.parsed_modules)
        class_symbol = class_index.symbol_for(
            file_path=target.file_path,
            qualname=target.qualname,
        )
        indexed_class = (
            None if class_symbol is None else class_index.class_for(class_symbol)
        )
        if indexed_class is None or not indexed_class.declares_autoregister_meta:
            raise ValueError(
                "Target no longer declares an AutoRegisterMeta family authority"
            )
        repeated_names = class_index.assignments_repeated_from_ancestors(
            indexed_class.symbol,
            INHERITABLE_AUTOREGISTER_CONFIGURATION_ATTRIBUTE_NAMES,
        )
        if not repeated_names:
            raise ValueError(
                "Target has no AutoRegister configuration repeated from an ancestor"
            )
        return DeleteClassAssignmentsOperation(
            target=SourceRewriteTarget(target_id=target.target_id),
            assignment_names=repeated_names,
            rationale=self.rationale,
        ).source_edits(snapshot)


@dataclass(frozen=True, kw_only=True)
class DeleteModuleAssignmentsOperation(AssignmentDeletionOperationABC):
    """Delete named module-level assignment statements."""

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentDeletionSource:
        source_path = self.required_source_path(
            snapshot,
            "delete_module_assignments",
        )
        module = snapshot.module_nodes_by_file_path[source_path]
        return AssignmentDeletionSource(
            source=snapshot.sources_by_file_path[source_path],
            node=module,
            file_path=source_path,
        )


@dataclass(frozen=True, kw_only=True)
class TargetDeletionOperationABC(RefactorRecipeOperation, ABC):
    """Shared target deletion with leaf-owned residual-use policy."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        return (
            SourceSpanDeletion.for_statement(
                context,
                target_digest,
                rationale=self.rationale
                or f"Delete target {target_digest.qualname!r}.",
            ),
        )

    def declared_architecture_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        return self.deletion_guard_rules(context)

    @abstractmethod
    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        """Return the residual-use policy owned by the concrete deletion."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeleteTargetOperation(TargetDeletionOperationABC):
    """Delete one source-index target without adding a residual-use policy."""

    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        del context
        return ()


@dataclass(frozen=True, kw_only=True)
class EraseDeadCompatibilityOperation(TargetDeletionOperationABC):
    """Delete one obsolete target and forbid its residual repository uses."""

    source_dependency_scope: ClassVar[CodemodSourceDependencyScope] = (
        CodemodSourceDependencyScope.REPOSITORY
    )
    forbidden_attribute_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )
    forbidden_call_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec(),
        default=(),
    )

    def deletion_guard_rules(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ArchitectureGuardRule, ...]:
        _target_identifier, target = self.target_digest(context)
        call_names = self.forbidden_call_names or (
            target.qualname.rsplit(".", maxsplit=1)[-1],
        )
        reason = (
            self.rationale
            or "Erase the dead compatibility path and reject residual uses."
        )
        bare_names = target.node_kind.bare_names_reaching(target.name)
        return (
            ArchitectureGuardRule(
                rule_id=f"{target.qualname}-no-residual-compatibility-uses",
                constraints=tuple(
                    constraint
                    for constraint in (
                        ForbiddenAttributeArchitectureGuardConstraint(
                            self.forbidden_attribute_names
                        ),
                        ForbiddenCallArchitectureGuardConstraint(call_names),
                        ForbiddenImportArchitectureGuardConstraint(bare_names),
                    )
                    if constraint.names
                ),
                reason=reason,
            ),
            *(
                ArchitectureGuardRule(
                    rule_id=f"{target.qualname}-no-residual-references-beside-it",
                    constraints=(
                        ForbiddenReferenceArchitectureGuardConstraint(names),
                    ),
                    scopes=(ArchitectureGuardTargetScope(file_path=target.file_path),),
                    reason=reason,
                )
                for names in (bare_names,)
                if names
            ),
        )


@dataclass(frozen=True, kw_only=True)
class SelectedTargetsOperation(RefactorRecipeOperation, ABC):
    """Operation base whose target set comes from a registered selector."""

    selector: CodemodTargetSelector = codemod_payload_field(
        PayloadRecordValueCodec(CodemodTargetSelector)
    )
    selection_count: SelectionCountExpectation = codemod_payload_field(
        SelectionCountPayloadValueCodec(),
        default_factory=SelectionCountExpectation,
    )

    def selected_target_ids(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[str, ...]:
        target_ids = self.selector.target_ids(context)
        self.selection_count.require_actual_count(len(target_ids))
        return target_ids


@dataclass(frozen=True, kw_only=True)
class DeleteSelectedTargetsOperation(SelectedTargetsOperation):
    """Delete every source-index target selected by a registered selector."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        return tuple(
            SourceSpanDeletion.for_statement(
                context,
                context.source_index.target_by_id[target_id],
                rationale=self.rationale,
            )
            for target_id in self.selected_target_ids(context)
        )
