"""Memoize immutable call-declaration projections without changing their logic."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ReplaceFunctionDecoratorsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    ReplaceFunctionDecoratorsOperation(
        target=SourceRewriteTarget(
            file_path="nominal_refactor_advisor/product_flow.py",
            qualname=f"CompactFunctionDeclaration.{projection}",
        ),
        decorators_source="@cached_property",
    )
    for projection in (
        "binding_kind",
        "signature_decorator_hazard",
        "call_signature",
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
