"""Retain receiver evaluation through the shared AST flow collector."""

import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/product_flow.py",
                qualname="_CompactFlowCollector._visit_call_target_evaluation",
            ),
            body_source=dedent('''\
                """Retain receiver reads; the call itself owns its terminal lookup."""
                if isinstance(expression, ast.Attribute):
                    self.visit(expression.value)
                elif not isinstance(expression, ast.Name):
                    self.visit(expression)
            '''),
        ),
        DeleteClassAssignmentsOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/product_flow.py",
                qualname="CompactValueOriginViolation",
            ),
            assignment_names=("CONTROL_FLOW_JOIN",),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
