"""Express NRA's shared import visitor implementation as one staged DSL plan."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    AliasFunctionOperation,
    CodemodPlanSequence,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
implementation = replace(module, qualname="_DeclarationCollector.visit_Import")

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionSignatureOperation(
            target=implementation,
            signature_suffix="(self, node: ast.Import | ast.ImportFrom) -> None:",
        ),
        AliasFunctionOperation(
            target=replace(module, qualname="_DeclarationCollector.visit_ImportFrom"),
            implementation=implementation,
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
