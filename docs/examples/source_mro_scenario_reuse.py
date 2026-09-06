"""Scope lazy C3 projection reuse to one immutable source/substitution scenario."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertClassMemberOperation,
    ProjectFunctionParameterOperation,
    ReplaceDeclaredCallOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceScopeAssignmentOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/source_native_mro.py")
owner = replace(module, qualname="SourceNativeClassMro")
projection = replace(module, qualname="SourceNativeClassMro.for_source_class")

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module, import_source="from dataclasses import field, replace"
        ),
        *(
            InsertClassMemberOperation(target=owner, source=source)
            for source in (
                "substitution: NativeClassBaseSubstitution | None = None",
                "_mro_types: dict[str, type] = field(default_factory=dict, init=False, repr=False, compare=False)",
            )
        ),
        ProjectFunctionParameterOperation(
            target=projection,
            parameter_name="substitution",
            projection_source="self.substitution",
        ),
        ReplaceFunctionSignatureOperation(
            target=projection,
            signature_suffix="(self, source_class: IndexedClass) -> DeclarationMroType[QualifiedDeclaration]:",
        ),
        ReplaceScopeAssignmentOperation(
            target=projection,
            assignment_name="projected",
            source="projected = self._mro_types",
        ),
        ReplaceDeclaredCallOperation(
            target=replace(
                module, qualname="SourceNativeClassMro.require_inherited_method"
            ),
            callee=projection,
            selection_count=SelectionCountExpectation(exact=1),
            expression_source="replace(self, substitution=substitution).for_source_class(substitution.owner)",
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
