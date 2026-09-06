"""Keep continuation proof context internal to the default JSON projection."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceScopeAssignmentOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/codemod_runtime.py",
                qualname="CodemodPlanSequenceContinuationReport",
            ),
            assignment_name="source_index",
            source="source_index: SourceIndex = json_report_field(included=False)",
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
