"""Class declaration edits derived through shared member insertion geometry."""

from dataclasses import dataclass

from .codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassMemberInsertion,
    ClassMemberSource,
)
from .codemod_operations import SourcePayloadOperation
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot


@dataclass(frozen=True, kw_only=True)
class InsertClassMemberOperation(SourceReprovedOperation, SourcePayloadOperation):
    """Insert an authored member, deriving its identity and destination indentation."""

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[ClassMemberInsertion, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        authority = ClassBodySourceAuthority(
            node=node, source=snapshot.sources_by_file_path[target.file_path]
        )
        member = ClassMemberSource.from_source(
            self.source, indentation=authority.indentation
        )
        return (
            ClassMemberInsertion(
                target_id=target.target_id,
                members=(member,),
                rationale=self.rationale
                or f"Insert member {member.name!r} in {target.qualname!r}.",
            ),
        )
