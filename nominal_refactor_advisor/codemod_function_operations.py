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
    DeclaredCallTargetRewrite,
)
from .codemod_declaration_operations import (
    DeclarationDecoratorsPayload as DeclarationDecoratorsPayload,
    DeclarationMutationOperationABC as DeclarationMutationOperationABC,
    ReplaceDeclarationDecoratorsOperation as ReplaceDeclarationDecoratorsOperation,
)
from .codemod_declaration_source import (
    FunctionBindingProjectionSourceAuthority,
    FunctionAliasSourceAuthority,
    FunctionBodyPrefixSourceAuthority,
    FunctionBodySourceAuthority,
    FunctionDecoratorsSourceAuthority,
    FunctionRegionSourceAuthority,
    FunctionSignatureSourceAuthority,
)
from .codemod_payload import (
    EmptyDefaultStringPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
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
from .declaration_dependencies import (
    FunctionBindingABC,
    FunctionLocalBinding,
    FunctionParameterBinding,
)
from .descriptor_algebra import AliasProperty
from .product_flow import BareCallTargetReference
from .value_expression import LexicalValueReference


@dataclass(frozen=True, kw_only=True)
class FunctionMutationOperationABC(DeclarationMutationOperationABC, ABC):
    """Mutation family whose source authorities require a function declaration."""

    source_authority: ClassVar[type[FunctionRegionSourceAuthority]]


@dataclass(frozen=True, kw_only=True)
class AliasFunctionOperation(RepositorySourceReprovedOperation):
    """Replace a function with an alias to a source-proved same-scope implementation."""

    implementation: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[PhysicalSourceEdit, ...]:
        _, target, node = self.target_node_from_context(snapshot)
        implementation_id = self.implementation.required_target_id(
            snapshot.source_index
        )
        implementation = snapshot.source_index.target_by_id[implementation_id]
        implementation_node = snapshot.ast_target_nodes_by_id[implementation_id]
        authority = FunctionAliasSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        if not isinstance(implementation_node, ast.FunctionDef | ast.AsyncFunctionDef):
            raise ValueError("Alias implementation must be a function declaration")
        if (target.file_path, target.qualname.rpartition(".")[0]) != (
            implementation.file_path,
            implementation.qualname.rpartition(".")[0],
        ):
            raise ValueError("Function alias declarations must share a lexical scope")
        target_symbol = snapshot.source_index.symbol_for_target(target)
        scope_symbol = target_symbol.rpartition(".")[0]
        repository = snapshot.product_flow_repository
        context = repository.flow_contexts_by_owner_symbol.get(scope_symbol)
        if context is None:
            raise ValueError("Function alias has no unique enclosing flow authority")
        target_mutations = tuple(
            mutation
            for mutation in context.flow.mutations_by_root_name.get(node.name, ())
            if mutation.line == node.lineno and mutation.kind.is_definition_binding
        )
        if len(target_mutations) != 1:
            raise ValueError("Function alias target has no unique definition binding")
        declaration = repository.resolve_function_target(
            context,
            BareCallTargetReference(implementation_node.name),
            target_mutations[0].position,
        ).declaration
        if (
            declaration is None
            or declaration.identity.symbol
            != snapshot.source_index.symbol_for_target(implementation)
        ):
            raise ValueError(
                "Alias implementation is not the selected preceding definition"
            )
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=(authority.replacement(implementation_node.name),),
            rationale=self.rationale
            or f"Alias {target.qualname!r} to {implementation.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ReplaceFunctionDecoratorsOperation(
    DeclarationDecoratorsPayload, FunctionMutationOperationABC
):
    """Function-only decorator mutation using the shared source and payload authority."""

    source_authority = FunctionDecoratorsSourceAuthority


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
class CallExpressionSourcePayload:
    """Authored expression shared by whole-call and callable-only edits."""

    expression_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    replacement_source = AliasProperty[str]("expression_source")


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallOperation(
    CallExpressionSourcePayload, DeclaredCallMutationOperationABC
):
    """Replace resolved calls with an authored expression; equivalence is author-owned."""

    source_authority = DeclaredCallExpressionRewrite


@dataclass(frozen=True, kw_only=True)
class ReplaceDeclaredCallTargetOperation(
    CallExpressionSourcePayload, DeclaredCallMutationOperationABC
):
    """Replace a resolved call's callable; new behaviour and binding are author-owned."""

    source_authority = DeclaredCallTargetRewrite


@dataclass(frozen=True, kw_only=True)
class FunctionBindingProjectionOperationABC(SourceReprovedOperation, ABC):
    """Rewrite owned reads through the selected lexical binding authority."""

    projection_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type: ClassVar[type[FunctionBindingABC]]

    attribute_path: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(), default=()
    )

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
        authority = FunctionBindingProjectionSourceAuthority(
            node=node,
            source=snapshot.sources_by_file_path[target.file_path],
        )
        expression = ast.parse(self.projection_source, mode="eval").body
        reference = LexicalValueReference.from_expression(expression)
        if reference is None:
            raise ValueError("Binding projection requires a Name/Attribute access path")
        return authority.geometry.physical_edits(
            file_path=target.file_path,
            replacements=authority.replacements_for(
                type(self).binding_type(node, self.binding_name),
                reference,
                attribute_path=self.attribute_path,
            ),
            rationale=self.rationale
            or f"Project binding {self.binding_name!r} in {target.qualname!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionParameterOperation(FunctionBindingProjectionOperationABC):
    """Rewrite parameter reads; signature and caller migrations remain explicit."""

    parameter_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type = FunctionParameterBinding
    binding_name = AliasProperty[str]("parameter_name")


@dataclass(frozen=True, kw_only=True)
class ProjectFunctionLocalOperation(FunctionBindingProjectionOperationABC):
    """Rewrite a single-assignment local's reads onto a parameter access path."""

    local_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    binding_type = FunctionLocalBinding
    binding_name = AliasProperty[str]("local_name")
