from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import ClassVar

from .codemod_declaration_source import (
    DeclarationDecoratorsSourceAuthority,
    DeclarationRegionSourceAuthority,
)
from .codemod_payload import (
    EmptyDefaultStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_source_edits import PhysicalSourceEdit
from .descriptor_algebra import AliasProperty


@dataclass(frozen=True, kw_only=True)
class DeclarationMutationOperationABC(SourceReprovedOperation, ABC):
    """Source-proved mutation of one named declaration's source region."""

    source_authority: ClassVar[type[DeclarationRegionSourceAuthority]]

    @property
    @abstractmethod
    def replacement_source(self) -> str:
        """Project the leaf operation's declared replacement payload."""

        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        authority = type(self).source_authority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=(authority.replacement(self.replacement_source),),
            rationale=self.rationale
            or f"{self.operation_key()} on {target.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class DeclarationDecoratorsPayload:
    """Authored decorators shared independently of the declaration target family."""

    decorators_source: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(), default=""
    )

    replacement_source = AliasProperty[str]("decorators_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclarationDecoratorsOperation(
    DeclarationDecoratorsPayload, DeclarationMutationOperationABC
):
    """Replace class or function decorators while retaining the header and suite."""

    source_authority = DeclarationDecoratorsSourceAuthority
