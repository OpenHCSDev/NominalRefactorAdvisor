"""Separate signature binding and value expressions from product-flow collection."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ExtractSymbolClosureToNewModuleOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        ExtractSymbolClosureToNewModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/product_flow.py"
            ),
            root_symbol_qualnames=("CompactFunctionSignature",),
            maximum_moved_symbol_count=24,
            destination_path="nominal_refactor_advisor/call_binding.py",
        ),
        ExtractSymbolClosureToNewModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/call_binding.py"
            ),
            root_symbol_qualnames=("CompactValueExpression",),
            maximum_moved_symbol_count=8,
            destination_path="nominal_refactor_advisor/value_expression.py",
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
