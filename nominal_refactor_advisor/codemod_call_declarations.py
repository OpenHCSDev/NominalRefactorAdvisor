"""Typed selection and deletion of module-level call declarations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import cast

from .ast_projection import AstExpressionProjection
from .codemod_operations import RefactorRecipeOperation
from .codemod_payload import (
    CodemodPayloadRecord,
    FlattenedPayloadRecordValueCodec,
    OptionalStringArrayPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import (
    SelectionCountExpectation,
    SelectionCountPayloadValueCodec,
)
from .codemod_source_edits import NominalSourceEdit, SourceSpanDeletion
from .source_index import AstTargetNodeKind


@dataclass(frozen=True)
class ModuleCallDeclaration:
    """One module-level call statement with its source-derived identity."""

    statement: ast.Expr

    def __post_init__(self) -> None:
        if not isinstance(self.statement.value, ast.Call):
            raise TypeError("Module call declaration requires a call statement")

    @classmethod
    def from_statement(cls, statement: ast.stmt) -> "ModuleCallDeclaration | None":
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value,
            ast.Call,
        ):
            return None
        return cls(statement=statement)

    @property
    def call(self) -> ast.Call:
        """Return the call expression proved by this declaration."""

        return cast(ast.Call, self.statement.value)

    @property
    def callee_qualname(self) -> str | None:
        return AstExpressionProjection.qualified_name(self.call.func)

    @property
    def positional_argument_qualnames(self) -> tuple[str | None, ...]:
        return tuple(
            AstExpressionProjection.qualified_name(argument)
            for argument in self.call.args
        )


@dataclass(frozen=True)
class ModuleCallDeclarationSelector(CodemodPayloadRecord):
    """Select top-level call declarations by callee and argument prefix."""

    callee_qualname: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )
    positional_argument_prefix: tuple[str, ...] = codemod_payload_field(
        OptionalStringArrayPayloadValueCodec(),
        default=(),
    )

    def declarations(self, module: ast.Module) -> tuple[ModuleCallDeclaration, ...]:
        return tuple(
            declaration
            for statement in module.body
            if (declaration := ModuleCallDeclaration.from_statement(statement))
            is not None
            and self.matches(declaration)
        )

    def matches(self, declaration: ModuleCallDeclaration) -> bool:
        argument_count = len(self.positional_argument_prefix)
        return (
            declaration.callee_qualname == self.callee_qualname
            and declaration.positional_argument_qualnames[:argument_count]
            == self.positional_argument_prefix
        )


@dataclass(frozen=True, kw_only=True)
class DeleteModuleCallDeclarationsOperation(RefactorRecipeOperation):
    """Delete exact module-level declarative factory calls."""

    declaration_selector: ModuleCallDeclarationSelector = codemod_payload_field(
        FlattenedPayloadRecordValueCodec(ModuleCallDeclarationSelector)
    )
    selection_count: SelectionCountExpectation = codemod_payload_field(
        SelectionCountPayloadValueCodec(),
        default_factory=SelectionCountExpectation,
    )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        _target_id, module_target = self.target_digest(context)
        module_target.require_kind(
            AstTargetNodeKind.MODULE,
            "Module call declaration deletion requires a module target",
        )
        file_path = module_target.file_path
        declarations = self.declaration_selector.declarations(
            context.module_nodes_by_file_path[file_path]
        )
        self.selection_count.require_actual_count(len(declarations))
        source = context.sources_by_file_path[file_path]
        return tuple(
            SourceSpanDeletion.for_statement_node(
                file_path=file_path,
                source=source,
                statement=declaration.statement,
                rationale=self.rationale_text(
                    "Delete the selected module-level call declaration."
                ),
            )
            for declaration in declarations
        )
