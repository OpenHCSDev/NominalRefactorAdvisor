from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import ClassVar

from .codemod_call_source import (
    DeclaredCallArgumentsRewrite,
    DeclaredCallExpressionRewrite,
    DeclaredCallRewriteABC,
)
from .codemod_declaration_source import (
    FunctionBindingProjectionSourceAuthority,
    FunctionBodyPrefixSourceAuthority,
    FunctionBodySourceAuthority,
    FunctionLocalProjectionSourceAuthority,
    FunctionParameterProjectionSourceAuthority,
    FunctionRegionSourceAuthority,
    FunctionSignatureSourceAuthority,
)
from .codemod_payload import (
    EmptyDefaultStringPayloadValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selector_models import (
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
    SourceRewriteTarget,
)
from .codemod_source_edits import PhysicalSourceEdit
from .descriptor_algebra import AliasProperty
from .product_flow import LexicalValueReference


@dataclass(frozen=True, kw_only=True)
class FunctionMutationOperationABC(SourceReprovedOperation, ABC):
    """Source-proved mutation of one function declaration."""

    source_authority: ClassVar[type[FunctionRegionSourceAuthority]]

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
class ReplaceFunctionSignatureOperation(FunctionMutationOperationABC):
    """Replace function parameters and return annotation, preserving its body."""

    signature_suffix: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    source_authority = FunctionSignatureSourceAuthority
    replacement_source = AliasProperty[str]("signature_suffix")


@dataclass(frozen=True, kw_only=True)
class FunctionBodySourcePayload:
    """Authored statements shared by function-suite mutation operations."""

    body_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    replacement_source = AliasProperty[str]("body_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionBodyOperation(
    FunctionBodySourcePayload, FunctionMutationOperationABC
):
    """Replace a function or method body while preserving its signature."""

    source_authority = FunctionBodySourceAuthority


@dataclass(frozen=True, kw_only=True)
class PrependFunctionBodyOperation(
    FunctionBodySourcePayload, FunctionMutationOperationABC
):
    """Insert statements after a function's docstring, retaining its existing body."""

    source_authority = FunctionBodyPrefixSourceAuthority


@dataclass(frozen=True, kw_only=True)
class DeclaredCallMutationOperationABC(RepositorySourceReprovedOperation, ABC):
    """Re-prove declaration-selected calls before deriving one authored mutation."""

    callee: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    selection_count: SelectionCountExpectation = codemod_payload_field(
        SelectionCountPayloadValueCodec(),
        default_factory=SelectionCountExpectation,
    )

    source_authority: ClassVar[type[DeclaredCallRewriteABC]]

    @property
    @abstractmethod
    def replacement_source(self) -> str:
        """Project the selected operation's authored payload."""
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _identifier, caller = self.target_digest(snapshot)
        callee_id = self.callee.required_target_id(snapshot.source_index)
        authority = type(self).source_authority(
            snapshot,
            caller,
            snapshot.source_index.target_by_id[callee_id],
            self.replacement_source,
            self.selection_count,
        )
        return authority.geometry.physical_edits(
            file_path=caller.file_path,
            replacements=authority.replacements(),
            rationale=self.rationale
            or f"{self.operation_key()} in {caller.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallArgumentsOperation(DeclaredCallMutationOperationABC):
    """Replace arguments of declaration-resolved calls in one selected scope."""

    arguments_source: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(), default=""
    )
    source_authority = DeclaredCallArgumentsRewrite
    replacement_source = AliasProperty[str]("arguments_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallOperation(DeclaredCallMutationOperationABC):
    """Replace resolved calls with an authored expression; equivalence is author-owned."""

    expression_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_authority = DeclaredCallExpressionRewrite
    replacement_source = AliasProperty[str]("expression_source")


@dataclass(frozen=True, kw_only=True)
class FunctionBindingProjectionOperationABC(SourceReprovedOperation, ABC):
    """Rewrite owned reads through the selected lexical binding authority."""

    projection_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_authority: ClassVar[type[FunctionBindingProjectionSourceAuthority]]

    @property
    @abstractmethod
    def binding_name(self) -> str:
        """Name the binding selected by this operation's payload."""
        raise NotImplementedError

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _identifier, target, node = self.target_node_from_context(snapshot)
        authority = type(self).source_authority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        expression = ast.parse(self.projection_source, mode="eval").body
        reference = LexicalValueReference.from_expression(expression)
        if reference is None:
            raise ValueError("Binding projection requires a Name/Attribute access path")
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=authority.replacements_for(self.binding_name, reference),
            rationale=self.rationale
            or f"Project binding {self.binding_name!r} in {target.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionParameterOperation(FunctionBindingProjectionOperationABC):
    """Rewrite parameter reads; signature and caller migrations remain explicit."""

    parameter_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_authority = FunctionParameterProjectionSourceAuthority
    binding_name = AliasProperty[str]("parameter_name")


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionLocalOperation(FunctionBindingProjectionOperationABC):
    """Rewrite a single-assignment local's reads onto a parameter access path."""

    local_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_authority = FunctionLocalProjectionSourceAuthority
    binding_name = AliasProperty[str]("local_name")
