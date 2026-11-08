from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

from ..codemod_declaration_source import NamedDeclarationSourceAuthority
from ..codemod_imports import (
    ModuleImportInsertionPoint,
    ModuleImportMutation,
)
from ..codemod_import_scopes import ModuleImportScope
from ..codemod_module_declarations import ModuleSymbolTable
from ..codemod_operations import (
    RefactorRecipeOperation,
    SourcePayloadOperation,
)
from ..codemod_payload import (
    EmptyDefaultStringPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_reproof import SourceReprovedOperation
from ..lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import CodemodSelectorContext
from ..codemod_source_edits import (
    PhysicalSourceEdit,
    SourceFileCreation,
    SourceInsertion,
    SourceLineSpan,
    SourceTargetEditor,
)
from ..codemod_spacing import (
    DestinationInsertionSpacing,
    SourceInsertionBoundary,
)


@dataclass(frozen=True, kw_only=True)
class CreateFileOperation(SourcePayloadOperation):
    """Create a Python source file for later operations in the same plan."""

    source: str = codemod_payload_field(EmptyDefaultStringPayloadValueCodec())

    def source_file_creations(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        if self.target.file_path is None:
            raise ValueError("create_file requires file_path")
        return (
            SourceFileCreation.from_operation(
                self,
                requested_path=self.target.file_path,
                source_index=context.source_index,
                source=self.source,
            ),
        )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[SourceFileCreation, ...]:
        return self.source_file_creations(context)


@dataclass(frozen=True, kw_only=True)
class TargetAdjacentInsertionOperationABC(SourceReprovedOperation, ABC):
    """Source-proved insertion adjacent to one indexed declaration."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, node = self.target_node_from_context(snapshot)
        source = snapshot.sources_by_file_path[target.file_path]
        declaration = NamedDeclarationSourceAuthority(node, source)
        insertion_line = self.insertion_line(declaration.declaration_line_span)
        source_lines = declaration.geometry.lines
        boundary = SourceInsertionBoundary.from_declaration_line(
            source_lines[target.line - 1]
        )
        spacing = DestinationInsertionSpacing.from_source(
            source,
            insertion_line,
            inserted_source_is_import_block=False,
            boundary=boundary,
        )
        inserted_source = self.source.strip("\r\n")
        return (
            SourceInsertion(
                file_path=target.file_path,
                insertion_line=insertion_line,
                inserted_lines=SourceTargetEditor.source_lines(
                    f"{spacing.leading_separator}{inserted_source}"
                    f"{spacing.trailing_separator}"
                ),
                leading_boundary=boundary,
                rationale=self.rationale
                or f"Insert source adjacent to {target.qualname!r}.",
            ),
        )

    @abstractmethod
    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        """Return the leaf operation's insertion geometry."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class InsertBeforeTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately before a source-index target."""

    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        return declaration_span.start_line


@dataclass(frozen=True, kw_only=True)
class InsertAfterTargetOperation(TargetAdjacentInsertionOperationABC):
    """Insert source immediately after a source-index target."""

    def insertion_line(self, declaration_span: SourceLineSpan) -> int:
        return declaration_span.end_line + 1


@dataclass(frozen=True, kw_only=True)
class InsertAfterImportsOperation(SourcePayloadOperation):
    """Insert source after a module docstring and leading import block."""

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[PhysicalSourceEdit, ...]:
        source_path = self.required_source_path(
            context,
            "insert_after_imports",
        )
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
                inserted_lines=SourceTargetEditor.source_lines(self.source),
                rationale=self.rationale
                or f"Insert source imports into {source_path!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EnsureImportOperation(RefactorRecipeOperation):
    """Insert import source after leading imports unless it already exists."""

    import_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        return (self.mutation(context),)

    def mutation(self, context: CodemodSelectorContext) -> ModuleImportMutation:
        source_path = self.required_source_path(context, "ensure_import")
        return ModuleImportMutation.from_source(
            file_path=source_path,
            import_source=self.import_source,
            rationale=self.rationale
            or f"Ensure import source exists in {source_path!r}.",
        )


@dataclass(frozen=True, kw_only=True)
class RemoveImportNamesOperation(RefactorRecipeOperation):
    """Remove selected names from a from-import statement."""

    module_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    import_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(
            context,
            "remove_import_names",
        )
        return (
            ModuleImportMutation.remove_names(
                file_path=source_path,
                module_name=self.module_name,
                names=self.import_names,
                rationale=self.rationale
                or f"Remove imports {self.import_names!r} from {self.module_name!r}.",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RemoveImportBindingsOperation(RefactorRecipeOperation):
    """Remove import bindings by bound name once the module no longer reads them.

    Covers every import shape (``import a.b as c``, ``from a import b``) in every
    scope that binds it, runtime or ``if TYPE_CHECKING:``, so a deletion that
    leaves its imports unread can drop them in the next stage.
    """

    bound_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def source_edits(
        self,
        context: CodemodSelectorContext,
    ) -> tuple[ModuleImportMutation, ...]:
        source_path = self.required_source_path(context, self.operation_key())
        still_read = ModuleSymbolTable(
            file_path=source_path,
            source=context.sources_by_file_path[source_path],
            module=context.module_nodes_by_file_path[source_path],
        ).referenced_names_excluding((), self.bound_names)
        if still_read:
            raise ValueError(
                f"{source_path!r} still reads import bindings "
                f"{tuple(sorted(still_read))!r}"
            )
        module = context.module_nodes_by_file_path[source_path]
        return tuple(
            ModuleImportMutation.remove_bound_names(
                file_path=source_path,
                names=names,
                scope=scope,
                rationale=self.rationale
                or f"Remove unread import bindings {names!r}.",
            )
            for scope in ModuleImportScope
            for names in (
                tuple(
                    name
                    for name in self.bound_names
                    if name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(scope.import_statements(module))
                ),
            )
            if names
        )
