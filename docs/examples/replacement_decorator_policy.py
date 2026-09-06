"""Use one decorator policy for replacement validation and source geometry."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

operation = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod.py", qualname="ReplaceTargetOperation"
)
policy = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_source_edits.py",
    qualname="SourceNodeDecoratorPolicy",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=policy,
            source=dedent('''\
                def validate_replacement(
                    self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                ) -> None:
                    """Require authored decorators to belong to the selected source region."""
                    if node.decorator_list and not self.includes_decorators:
                        raise ValueError(
                            "Replacement decorators require a decorator-inclusive source region"
                        )
                '''),
        ),
        InsertClassMemberOperation(
            target=operation,
            source="decorator_policy: ClassVar[SourceNodeDecoratorPolicy] = SourceNodeDecoratorPolicy.EXCLUDE",
        ),
        PatchTargetOperation(
            target=replace(
                operation, qualname=f"{operation.qualname}.replacement_declaration"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="        return replacement_module.body[0]",
                    new_source=(
                        "        self.decorator_policy.validate_replacement(replacement_module.body[0])\n"
                        "        return replacement_module.body[0]"
                    ),
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(
                operation, qualname=f"{operation.qualname}.source_edits_from_snapshot"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="        return (\n            SourceSpanReplacement(",
                    new_source=(
                        "        span = SourceTextGeometry(\n"
                        "            snapshot.sources_by_file_path[target.file_path]\n"
                        "        ).node_line_span(SourceNodeSpan(target_node, self.decorator_policy))\n"
                        "        return (\n            SourceSpanReplacement("
                    ),
                ),
                SourceTextReplacement(
                    old_source="start_line=target.line,",
                    new_source="start_line=span.start_line,",
                ),
                SourceTextReplacement(
                    old_source="end_line=target.end_line,",
                    new_source="end_line=span.end_line,",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
