"""Snapshot declared class members before visiting lazily evaluated dependencies."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        PatchTargetOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/implementation_identity.py",
                qualname="_ImplementationDependencyTraversal._visit",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="for declared_value in vars(owner).values():",
                    new_source="for declared_value in tuple(vars(owner).values()):",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
