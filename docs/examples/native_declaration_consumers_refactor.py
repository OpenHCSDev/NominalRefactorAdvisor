"""Route native identity and source consumers through NativeDeclaration."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    RemoveImportNamesOperation,
    ReplaceFunctionBodyOperation,
    ReplaceScopeAssignmentOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

forwarding = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/positional_forwarding.py"
)
bindings = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_index.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        *(
            EnsureImportOperation(
                target=target,
                import_source="from .native_declarations import NativeDeclaration",
            )
            for target in (forwarding, bindings)
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                forwarding, qualname="PositionalForwardingCall.from_callable"
            ),
            body_source='''"""Project native source through the same positional-call contract."""
node = NativeDeclaration(function).node
if not isinstance(node, ast.FunctionDef):
    return None
return cls.from_function(node)
''',
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(
                bindings,
                qualname="ModuleNominalBindingView.require_native_type_in_class",
            ),
            assignment_name="qualified_name",
            source="qualified_name = NativeDeclaration(declaration).qualified_name",
        ),
        RemoveImportNamesOperation(
            target=forwarding, module_name="textwrap", import_names=("dedent",)
        ),
        PatchTargetOperation(
            target=forwarding,
            replacements=(
                SourceTextReplacement(old_source="import inspect\n", new_source=""),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
