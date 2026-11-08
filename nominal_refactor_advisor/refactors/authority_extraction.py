from __future__ import annotations

import ast
from abc import ABC
from dataclasses import dataclass
from functools import cached_property

from ..codemod import RecipeCallReplacement
from ..codemod_imports import ModuleImportInsertionPoint
from ..codemod_payload import (
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStrEnumPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReprovedOperation,
)
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    PhysicalSourceEdit,
    SourceInsertion,
    SourceSpanDeletion,
    SourceTargetEditor,
)
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)


@dataclass(frozen=True, kw_only=True)
class SourceDerivedAuthorityProjectionOperation(
    RepositorySourceReprovedOperation,
    ABC,
):
    """Exact authority/projection pair whose edits derive from current source."""

    projection_target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )


@dataclass(frozen=True, kw_only=True)
class AuthoritySourceOperation(
    SourceReprovedOperation,
    ABC,
):
    """Codemod operation carrying source for a declared authority boundary."""

    authority_kind: SemanticAuthorityKind = codemod_payload_field(
        RequiredStrEnumPayloadValueCodec(SemanticAuthorityKind)
    )
    authority_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not isinstance(self.authority_kind, SemanticAuthorityKind):
            raise TypeError("authority_kind must be a SemanticAuthorityKind")

    @cached_property
    def authority_declaration(self) -> ast.ClassDef:
        """Return the single top-level class owned by the supplied source."""

        try:
            authority_module = ast.parse(
                self.authority_source,
                filename=f"<{self.operation_key()}-authority>",
            )
        except SyntaxError as error:
            raise ValueError(
                f"Authority source is not valid Python: {error}"
            ) from error
        declarations = tuple(
            statement
            for statement in authority_module.body
            if isinstance(statement, ast.ClassDef)
        )
        if len(declarations) != 1:
            raise ValueError(
                "Authority source must declare exactly one top-level class"
            )
        return declarations[0]

    def current_source_authority_claims(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[AuthorityClaim, ...]:
        return (self.required_authority_claim(context),)

    def required_authority_claim(
        self,
        context: CodemodSelectorContext,
    ) -> AuthorityClaim:
        _target_identifier, target = self.target_digest(context)
        source_path = target.file_path
        authority_name = self.authority_declaration.name
        bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            context.module_nodes_by_file_path[source_path].body
        )
        if authority_name in bound_names and target.name != authority_name:
            raise ValueError(
                f"Authority source name {authority_name!r} is already bound"
            )
        return AuthorityClaim(
            claimed_symbol=authority_name,
            authority_kind=self.authority_kind,
            file_path=source_path,
            qualname=authority_name,
        )


@dataclass(frozen=True, kw_only=True)
class ExtractAuthorityOperation(AuthoritySourceOperation):
    """Replace a helper target with a nominal authority and route call sites."""

    call_replacements: tuple[RecipeCallReplacement, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(RecipeCallReplacement),
        default=(),
    )

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        target_identifier = self.target.required_target_id(context.source_index)
        target_digest = context.source_index.target_by_id[target_identifier]
        target_span = SourceSpanDeletion.target_span(context, target_digest)
        self.required_authority_claim(context)
        return (
            SourceInsertion(
                file_path=target_digest.file_path,
                insertion_line=target_span.start_line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or f"Insert authority before {target_digest.qualname!r}.",
            ),
            SourceSpanDeletion.for_target(
                context,
                target_digest,
                rationale=self.rationale
                or f"Delete helper target {target_digest.qualname!r}.",
            ),
            *(
                replacement.line_replacement(
                    context,
                    rationale=self.rationale,
                )
                for replacement in self.call_replacements
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DeclareAuthorityOperation(AuthoritySourceOperation):
    """Insert a declared authority boundary and derive its authority claim."""

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        authority_claim = self.required_authority_claim(context)
        source_path = authority_claim.file_path
        source = context.sources_by_file_path[source_path]
        insertion_line = ModuleImportInsertionPoint(
            source,
            source_path,
            context.module_nodes_by_file_path[source_path],
        ).line_number
        return (
            SourceInsertion(
                file_path=source_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(self.authority_source),
                rationale=self.rationale
                or (f"Declare authority {authority_claim.claimed_symbol!r}."),
            ),
        )
