"""Move compile-time function scope ownership beside ordered class scope ownership."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    MoveSymbolClosureToModuleOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        MoveSymbolClosureToModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/declaration_dependencies.py"
            ),
            root_symbol_qualnames=("FunctionBindingProjection",),
            maximum_moved_symbol_count=4,
            destination_path="nominal_refactor_advisor/lexical_scopes.py",
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
