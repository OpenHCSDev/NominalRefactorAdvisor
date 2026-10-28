"""Class declaration edits derived through shared member insertion geometry."""

from dataclasses import dataclass
from typing import ClassVar

from .codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassMemberInsertion,
    ClassMemberSource,
)
from .codemod_operations import SourcePayloadOperation
from .codemod_reproof import RepositorySourceReprovedOperation, SourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import ResolvedClassTarget
from .codemod_selector_models import DeclaredCallSelection
from .codemod_field_metadata_source import (
    FieldMetadataKeywordRemoval,
    PrunedFieldMetadataKeywordRemoval,
)
from .codemod_payload import (
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_source_edits import PhysicalSourceEdit
from .source_index import AstTargetNodeKind


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


@dataclass(frozen=True, kw_only=True)
class RemoveFieldMetadataKeywordsOperation(
    DeclaredCallSelection, RepositorySourceReprovedOperation
):
    """Remove inert keyword expressions, retaining an emptied metadata constructor."""

    field_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    keyword_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )
    source_authority: ClassVar[type[FieldMetadataKeywordRemoval]] = (
        FieldMetadataKeywordRemoval
    )

    def __post_init__(self) -> None:
        if not self.field_name.isidentifier():
            raise ValueError("Selected field must be one identifier")
        if not self.keyword_names or len(set(self.keyword_names)) != len(
            self.keyword_names
        ):
            raise ValueError("Metadata keyword removal requires unique keyword names")
        if not all(name.isidentifier() for name in self.keyword_names):
            raise ValueError("Metadata keyword names must be identifiers")

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[PhysicalSourceEdit, ...]:
        owner = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        if "." in owner.qualname:
            raise ValueError(
                "Field metadata editing requires a module-level class; enclosing bindings remain open"
            )
        callee = snapshot.source_index.target_by_id[
            self.callee.required_target_id(snapshot.source_index)
        ]
        callee.require_kind(
            AstTargetNodeKind.CLASS,
            "Field metadata requires a selected constructor class",
        )
        authority = type(self).source_authority(
            source=snapshot.sources_by_file_path[owner.file_path],
            snapshot=snapshot,
            owner=owner,
            callee=callee,
            field_name=self.field_name,
            keyword_names=self.keyword_names,
            selection_count=self.selection_count,
        )
        return authority.physical_edits(
            file_path=owner.file_path,
            replacements=authority.replacements(),
            rationale=self.rationale_text(
                f"Remove selected metadata keywords from {owner.qualname}.{self.field_name}."
            ),
        )


@dataclass(frozen=True, kw_only=True)
class PruneFieldMetadataKeywordsOperation(RemoveFieldMetadataKeywordsOperation):
    """Remove inert keywords and elide metadata whose empty constructor is redundant."""

    source_authority = PrunedFieldMetadataKeywordRemoval
