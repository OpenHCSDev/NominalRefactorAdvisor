"""Derive value-bearing assignment projections from the shared target authority."""

import json

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = "nominal_refactor_advisor/assignment_projection.py"
projection = "SingleAssignmentAndValueNameProjection"
target = SourceRewriteTarget(file_path=module, qualname=projection)

PLAN = CodemodPlanSequence.from_operations(
    (
        AddClassBaseOperation(
            target=target, base_name="AssignmentStatementNameProjection"
        ),
        DeleteClassAssignmentsOperation(target=target, assignment_names=("statement",)),
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(file_path=module, qualname=f"{projection}.pair"),
            body_source=(
                "if (\n"
                "    isinstance(self.statement, ast.Assign | ast.AnnAssign)\n"
                "    and self.statement.value is not None\n"
                "    and (name := self.direct_name) is not None\n"
                "):\n"
                "    return name, self.statement.value\n"
                "return None\n"
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
