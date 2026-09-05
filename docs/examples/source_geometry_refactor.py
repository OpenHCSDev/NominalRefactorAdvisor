"""Record NRA's migration from a local geometry object to its existing authority."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteFunctionAssignmentsOperation,
    ProjectFunctionLocalOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

target = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_declaration_source.py",
    qualname="ClassBodySourceAuthority.before_first_method_offset",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        ProjectFunctionLocalOperation(
            target=target, local_name="geometry", projection_source="self.geometry"
        ),
        DeleteFunctionAssignmentsOperation(
            target=target, assignment_names=("geometry",)
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
