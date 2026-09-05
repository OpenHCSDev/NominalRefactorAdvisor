"""Source-reproved assignment edits across declared Python scopes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from .codemod_operations import SourcePayloadOperation
from .codemod_payload import RequiredStringPayloadValueCodec, codemod_payload_field
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_source_edits import PhysicalSourceEdit
from .codemod_statement_source import AssignmentReplacementSource, AssignmentSource
from .descriptor_algebra import AliasProperty


@dataclass(frozen=True, kw_only=True)
class AssignmentReplacementOperationABC(
    SourceReprovedOperation, SourcePayloadOperation, ABC
):
    """Share authored assignment validation, selection and exact source rewriting."""

    @cached_property
    def assignment_source(self) -> AssignmentSource:
        return AssignmentSource(self.source)

    @property
    @abstractmethod
    def selected_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentReplacementSource:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[PhysicalSourceEdit, ...]:
        authority = self.source_authority(snapshot)
        return authority.physical_edits(
            file_path=authority.file_path,
            replacements=(
                authority.replacement(self.selected_name, self.assignment_source),
            ),
            rationale=self.rationale or f"Replace assignment {self.selected_name!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceModuleAssignmentOperation(AssignmentReplacementOperationABC):
    """Replace the module assignment named by the supplied declaration."""

    selected_name = AliasProperty[str]("assignment_source.name")

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentReplacementSource:
        path = self.required_source_path(snapshot, self.operation_key())
        return AssignmentReplacementSource(
            source=snapshot.sources_by_file_path[path],
            node=snapshot.module_nodes_by_file_path[path],
            file_path=path,
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceScopeAssignmentOperation(AssignmentReplacementOperationABC):
    """Replace one direct assignment in a selected class or function scope."""

    assignment_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    selected_name = AliasProperty[str]("assignment_name")

    def source_authority(
        self, snapshot: CodemodSourceSnapshot
    ) -> AssignmentReplacementSource:
        _, target, node = self.target_node_from_context(snapshot)
        return AssignmentReplacementSource(
            source=snapshot.sources_by_file_path[target.file_path],
            node=node,
            file_path=target.file_path,
        )
