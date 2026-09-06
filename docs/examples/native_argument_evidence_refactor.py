"""Give reusable argument evidence its generic owner and update all imports."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    MoveSymbolClosureToModuleOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        RenameTopLevelDeclarationAuthorityOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/native_subscription.py",
                qualname="NativeSubscriptionArgument",
            ),
            new_name="NativeArgumentEvidence",
        ),
        MoveSymbolClosureToModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/native_subscription.py"
            ),
            root_symbol_qualnames=("NativeArgumentEvidence",),
            maximum_moved_symbol_count=1,
            destination_path="nominal_refactor_advisor/native_reference.py",
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
