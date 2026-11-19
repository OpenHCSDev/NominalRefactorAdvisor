"""Exact changes to declared non-Python owners in one codemod plan."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_payload import (RequiredStringPayloadValueCodec,
                                                      codemod_payload_field)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import (NominalSourceEdit,
                                                           SourceLineSpan,
                                                           SourceSpanReplacement,
                                                           SourceTargetEditor,
                                                           SourceTextSpanReplacement,
                                                           TextFileCreation,
                                                           TextFileDeletion,
                                                           TextFileEdit)
from nominal_refactor_advisor.codemod_semantics import CodemodSourceDependencyScope


@dataclass(frozen=True, kw_only=True)
class CreateTextFileOperation(RefactorRecipeOperation):
    """Create one declared file outside the Python source index."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        del context
        return ()

    def text_file_changes(self, context: CodemodSelectorContext) -> tuple[TextFileCreation, ...]:
        del context
        if self.target.file_path is None:
            raise ValueError("create_text_file requires file_path")
        return (TextFileCreation(operation_type=type(self), file_path=self.target.file_path,
                                 source=self.source, rationale=self.rationale),)


@dataclass(frozen=True, kw_only=True)
class ReplaceModuleSourceOperation(RefactorRecipeOperation):
    """Replace one complete indexed Python module with exact declared source."""

    source: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def source_edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        path = self.required_source_path(context, self.operation_key())
        current = context.sources_by_file_path[path]
        return (SourceSpanReplacement(
            file_path=path,
            span=SourceLineSpan(1, max(1, len(current.splitlines()))),
            replacement_lines=SourceTargetEditor.source_lines(self.source),
            rationale=self.rationale or f"Replace complete module {path!r}.",
        ),)


@dataclass(frozen=True, kw_only=True)
class DeleteTextFileOperation(RefactorRecipeOperation):
    """Delete one exact non-Python owner as part of the same plan."""

    def source_edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        del context
        return ()

    def text_file_changes(self, context: CodemodSelectorContext) -> tuple[TextFileDeletion, ...]:
        del context
        if self.target.file_path is None:
            raise ValueError("delete_text_file requires file_path")
        source = Path(self.target.file_path).read_text()
        return (TextFileDeletion(operation_type=type(self), file_path=self.target.file_path,
                                 source=source, rationale=self.rationale),)


@dataclass(frozen=True, kw_only=True)
class PatchTextFileOperation(RefactorRecipeOperation):
    """Replace one unique exact span in a text file outside the Python source index."""

    old_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_dependency_scope = CodemodSourceDependencyScope.REPOSITORY

    def source_edits(self, context: CodemodSelectorContext) -> tuple[NominalSourceEdit, ...]:
        del context
        return ()

    def text_file_changes(self, context: CodemodSelectorContext) -> tuple[TextFileEdit, ...]:
        del context
        if self.target.file_path is None:
            raise ValueError("patch_text_file requires file_path")
        source = Path(self.target.file_path).read_text()
        count = source.count(self.old_source)
        if count != 1:
            raise ValueError(f"patch_text_file expected one exact span, found {count}")
        start = source.index(self.old_source)
        replacement = SourceTextSpanReplacement.from_offsets(
            start_offset=start,
            end_offset=start + len(self.old_source),
            replacement_source=self.new_source,
        )
        return (TextFileEdit(operation_type=type(self), file_path=self.target.file_path,
                             source=source, replacements=(replacement,),
                             rationale=self.rationale),)
