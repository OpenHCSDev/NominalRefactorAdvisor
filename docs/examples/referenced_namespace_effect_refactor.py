"""Historical field-factoring replay, not a plan for the current NRA checkout.

The standalone input specimen lives in test_referenced_namespace_effect_replay.
Current native admission uses captured-source identity without these carriers.
"""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    InsertBeforeTargetOperation,
    ReplaceClassBaseOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_declaration_operations import (
    ReplaceDeclarationDecoratorsOperation,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_namespace.py")
native = replace(module, qualname="NativeClassNamespaceEffect")
subscription = replace(module, qualname="SubscriptionClassNamespaceEffect")

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=native,
            source=dedent("""\
                @dataclass(frozen=True)
                class ReferencedClassNamespaceEffect(ClassNamespaceEffect, ABC):
                    node: ast.expr
                    reference: ScopedNativeReference

                """),
        ),
        ReplaceClassBaseOperation(
            target=native,
            base_name="ClassNamespaceEffect",
            replacement_base_name="ReferencedClassNamespaceEffect",
        ),
        DeleteClassAssignmentsOperation(
            target=native,
            assignment_names=("node", "reference"),
        ),
        ReplaceDeclarationDecoratorsOperation(target=native),
        ReplaceClassBaseOperation(
            target=subscription,
            base_name="ClassNamespaceEffect",
            replacement_base_name="ReferencedClassNamespaceEffect",
        ),
        DeleteClassAssignmentsOperation(
            target=subscription,
            assignment_names=("reference",),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
