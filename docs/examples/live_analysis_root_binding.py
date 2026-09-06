"""Bind live root requests before deriving analysis and report scopes."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    PrependFunctionBodyOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/analysis.py")

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source="from .cache_checkout import lexical_absolute_path",
        ),
        PrependFunctionBodyOperation(
            target=replace(module, qualname="AnalysisPathScope.from_requested_roots"),
            body_source=(
                "requested_roots, context_roots = (\n"
                "    tuple(lexical_absolute_path(root) for root in group)\n"
                "    for group in (requested_roots, context_roots)\n"
                ")"
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(
                module, qualname="AnalysisContextRootResolver.context_root_for_file"
            ),
            assignment_name="parent",
            source="parent = lexical_absolute_path(file_path).parent",
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
