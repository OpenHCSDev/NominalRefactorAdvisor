"""Reuse method-ownership proof when descending a type-keyed projection."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    PatchTargetOperation,
    PrependFunctionBodyOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/projection_descent_codemod.py"
)
method_descent = replace(module, qualname="_TypeKeyedBehaviorMethodDescent")
DERIVE_ARGUMENT_NAMES = ReplaceScopeAssignmentOperation(
    target=replace(
        method_descent,
        qualname=f"{method_descent.qualname}._require_target_module_bindings",
    ),
    assignment_name="parameter_names",
    source="parameter_names = LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(method)",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source=(
                "from nominal_refactor_advisor.class_index "
                "import ClassMethodPromotionSafetyProfile\n"
            ),
        ),
        PatchTargetOperation(
            target=method_descent,
            replacements=(
                SourceTextReplacement(
                    old_source="target_symbol: str",
                    new_source="projection_class: IndexedClass",
                ),
                SourceTextReplacement(
                    old_source="self.target_symbol",
                    new_source="self.target_class.symbol",
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(
                module, qualname="_TypeKeyedBehaviorSourceDerivation._method_insertions"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="target_symbol=target_class.symbol,",
                    new_source="projection_class=projection_class,",
                ),
            ),
        ),
        PrependFunctionBodyOperation(
            target=replace(
                method_descent,
                qualname=f"{method_descent.qualname}.transformed_source",
            ),
            body_source=dedent("""\
                safety = ClassMethodPromotionSafetyProfile.from_method(
                    self.projection_method,
                    LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.source_module.module.body),
                    LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.projection_class.node.body),
                    source_lines=tuple(self.source_module.source.splitlines()),
                )
                if safety.hazards:
                    raise ValueError(
                        f"projected method {self.projection_method.name!r} "
                        f"has ownership dependencies: {', '.join(safety.hazards)}"
                    )
                """),
        ),
        DERIVE_ARGUMENT_NAMES,
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
