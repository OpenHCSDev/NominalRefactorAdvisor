"""Replace duplicated parameter-name recovery with the lexical binding authority."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    EnsureImportOperation,
    RemoveImportNamesOperation,
    ReplaceDeclaredCallOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

scopes = SourceRewriteTarget(file_path="nominal_refactor_advisor/lexical_scopes.py")
dependencies = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/declaration_dependencies.py"
)
helper = replace(scopes, qualname="_argument_names")

PLAN = CodemodPlanSequence.from_operations(
    (
        *(
            EnsureImportOperation(
                target=target,
                import_source="from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY\n",
            )
            for target in (scopes, dependencies)
        ),
        *(
            ReplaceDeclaredCallOperation(
                target=target,
                callee=helper,
                expression_source=f"LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names({node})",
                selection_count=SelectionCountExpectation(exact=1),
            )
            for target, node in (
                (
                    replace(scopes, qualname="FunctionBindingProjection.from_function"),
                    "node",
                ),
                (
                    replace(
                        dependencies,
                        qualname="FunctionParameterBinding.without_binding",
                    ),
                    "self.node",
                ),
            )
        ),
        RemoveImportNamesOperation(
            target=dependencies,
            module_name=".lexical_scopes",
            import_names=("_argument_names",),
        ),
        DeleteTargetOperation(target=helper),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
