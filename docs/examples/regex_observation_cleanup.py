"""Remove superseded regex machinery after declaring the source-backed detector."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteModuleAssignmentsOperation,
    DeleteTargetOperation,
    EnsureImportOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

SOURCE_PATH = "nominal_refactor_advisor/detectors/_runtime.py"
PLAN = CodemodPlanSequence.from_operations(
    (
        *(
            DeleteTargetOperation(
                target=SourceRewriteTarget(file_path=SOURCE_PATH, qualname=name)
            )
            for name in (
                "RepeatedLocalRegexBundleDetector",
                "RepeatedLocalRegexBundleCandidate",
                "_regex_literal_from_call",
                "_is_substantial_regex_literal",
                "_local_regex_literals_by_function",
                "_function_owner_name",
                "_repeated_local_regex_bundle_candidates",
            )
        ),
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=SOURCE_PATH),
            import_source="from ._regex_bundle import RepeatedLocalRegexBundleDetector\n",
        ),
        DeleteTargetOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH, qualname="SurfaceFunctionIndex"
            )
        ),
        DeleteModuleAssignmentsOperation(
            target=SourceRewriteTarget(file_path=SOURCE_PATH),
            assignment_names=("_SurfaceFunctionItems", "_RuntimeFunctionNode"),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
