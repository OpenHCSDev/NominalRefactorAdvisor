from __future__ import annotations

import keyword as keyword_module
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEdit,
    SourceTextGeometry,
)
from nominal_refactor_advisor.declaration_authority_rename import (
    DeclarationAuthorityRenameProof,
)


@dataclass(frozen=True, kw_only=True)
class RenameTopLevelBindingAuthorityOperationABC(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Rename one top-level binding across its proved repository consumers."""

    new_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.new_name.isidentifier() or keyword_module.iskeyword(self.new_name):
            raise ValueError("Declaration rename requires a Python identifier")

    @abstractmethod
    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        proof = self.proof(snapshot)
        rationale = self.rationale or (
            f"Rename declaration authority {proof.target.name!r} to {self.new_name!r}."
        )
        return tuple(
            edit
            for module in proof.modules
            if module.has_replacements
            for edit in SourceTextGeometry(module.module.source).physical_edits(
                file_path=module.module.file_path,
                replacements=module.source_replacements(
                    old_name=proof.target.name,
                    new_name=self.new_name,
                ),
                rationale=rationale,
            )
        )


@dataclass(frozen=True, kw_only=True)
class RenameTopLevelDeclarationAuthorityOperation(
    RenameTopLevelBindingAuthorityOperationABC
):
    """Rename one indexed class or function declaration authority."""

    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        _target_id, target, node = self.target_node_from_context(snapshot)
        return DeclarationAuthorityRenameProof.require(
            snapshot.parsed_modules,
            target,
            node,
            new_name=self.new_name,
        )


@dataclass(frozen=True, kw_only=True)
class RenameTopLevelBindingAuthorityOperation(
    RenameTopLevelBindingAuthorityOperationABC
):
    """Rename one unambiguous movable top-level assignment authority."""

    binding_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.binding_name.isidentifier() or keyword_module.iskeyword(
            self.binding_name
        ):
            raise ValueError("Binding rename requires a Python identifier")

    def proof(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> DeclarationAuthorityRenameProof:
        return DeclarationAuthorityRenameProof.require_binding(
            snapshot.parsed_modules,
            source_path=self.required_source_path(
                snapshot,
                self.operation_key(),
            ),
            binding_name=self.binding_name,
            new_name=self.new_name,
        )
