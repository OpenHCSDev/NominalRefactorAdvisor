"""Retain a resolved object and derive its existing declaration view."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    ReplaceFunctionBodyOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = "resolution.py"
resolution = SourceRewriteTarget(file_path=module, qualname="Call.target_resolution")

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceScopeAssignmentOperation(
            target=SourceRewriteTarget(file_path=module, qualname="Call"),
            assignment_name="callee",
            source="resolved_target: Target",
        ),
        ReplaceFunctionBodyOperation(
            target=resolution,
            body_source="return self.resolved_target",
        ),
        InsertClassMemberOperation(
            target=SourceRewriteTarget(file_path=module, qualname="Call"),
            source=(
                "@property\n"
                "def callee(self) -> str:\n"
                "    return self.resolved_target.declaration\n"
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(file_path=module, qualname="make"),
            body_source="return Call(target)",
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
