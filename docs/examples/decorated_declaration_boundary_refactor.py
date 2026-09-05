"""Route adjacent edits through the authored named-declaration span property."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    PrependFunctionBodyOperation,
    RemoveImportNamesOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        PatchTargetOperation(
            target=replace(
                module,
                qualname="TargetAdjacentInsertionOperationABC.source_edits_from_snapshot",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="_target_identifier, target, _node =",
                    new_source="_target_identifier, target, node =",
                ),
                SourceTextReplacement(
                    old_source="insertion_line = self.insertion_line(target)\n        source_lines = source.splitlines(keepends=True)",
                    new_source="declaration = NamedDeclarationSourceAuthority(node, source)\n        insertion_line = self.insertion_line(declaration.declaration_line_span)\n        source_lines = declaration.geometry.lines",
                ),
            ),
        ),
        *(
            ReplaceFunctionSignatureOperation(
                target=replace(module, qualname=f"{owner}.insertion_line"),
                signature_suffix="(self, declaration_span: SourceLineSpan) -> int:",
            )
            for owner in (
                "TargetAdjacentInsertionOperationABC",
                "InsertBeforeTargetOperation",
                "InsertAfterTargetOperation",
            )
        ),
        *(
            ReplaceFunctionBodyOperation(
                target=replace(module, qualname=f"{owner}.insertion_line"),
                body_source=body,
            )
            for owner, body in (
                ("InsertBeforeTargetOperation", "return declaration_span.start_line\n"),
                (
                    "InsertAfterTargetOperation",
                    "return declaration_span.end_line + 1\n",
                ),
            )
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(
                module,
                qualname="CandidateCollectorMigration.candidate_declaration_insertion",
            ),
            assignment_name="insertion_line",
            source=dedent("""\
                insertion_line = (
                    NamedDeclarationSourceAuthority(anchor, self.source).declaration_line_span.start_line
                    if anchor is not None
                    else body.declaration_insert_line + 1
                )
                """),
        ),
        PatchTargetOperation(
            target=replace(
                module, qualname="CandidateCollectorMigration.candidate_method_deletion"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="return SourceNodeSpan(\n            method,\n            SourceNodeDecoratorPolicy.INCLUDE,\n        ).line_span.line_deletion(",
                    new_source="return NamedDeclarationSourceAuthority(\n            method, self.source,\n        ).declaration_line_span.line_deletion(",
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(
                module,
                qualname="DispatchToPolymorphismOperation.family_insertion_replacement",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="insertion_line=SourceNodeSpan(\n                source.dispatch_function.node,\n                SourceNodeDecoratorPolicy.INCLUDE,\n            ).start_line,",
                    new_source="insertion_line=NamedDeclarationSourceAuthority(\n                source.dispatch_function.node,\n                context.sources_by_file_path[target_digest.file_path],\n            ).declaration_line_span.start_line,",
                ),
            ),
        ),
        RemoveImportNamesOperation(
            target=module,
            module_name=".source_geometry",
            import_names=("ClassHeaderSourceSpan",),
        ),
        PrependFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/detectors/_base.py",
                qualname="CandidateCollectorBoilerplateCandidate.collector_call",
            ),
            body_source="if method.decorator_list:\n    return None\n",
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
