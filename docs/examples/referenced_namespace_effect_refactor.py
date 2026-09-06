"""Factor reference-bearing namespace effects without replacing method bodies."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    EnsureImportOperation,
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
        EnsureImportOperation(
            target=module,
            import_source="from .descriptor_algebra import AliasProperty",
        ),
        InsertBeforeTargetOperation(
            target=native,
            source=dedent("""\
                @dataclass(frozen=True)
                class ReferencedClassNamespaceEffect(ClassNamespaceEffect, ABC):
                    node: ast.expr
                    reference: ScopedNativeReference

                    recording_node = AliasProperty[ast.AST]("reference.node")
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
        *(
            DeleteTargetOperation(
                target=replace(module, qualname=f"{owner}.recording_node")
            )
            for owner in (
                "SubscriptionClassNamespaceEffect",
                "DescriptorCallClassNamespaceEffect",
            )
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
