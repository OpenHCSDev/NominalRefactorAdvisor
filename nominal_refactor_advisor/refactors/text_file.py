"""Exact changes to declared non-Python owners in one codemod plan."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.codemod_operations import RefactorRecipeOperation
from nominal_refactor_advisor.codemod_payload import (RequiredStringPayloadValueCodec,
                                                      codemod_payload_field)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import (NominalSourceEdit,
                                                           SourceTextSpanReplacement,
                                                           TextFileEdit)
from nominal_refactor_advisor.codemod_semantics import CodemodSourceDependencyScope


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
