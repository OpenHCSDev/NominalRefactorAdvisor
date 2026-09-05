"""Unify lexical scope collection and separate module-path identity through the DSL."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ExtractSymbolClosureToNewModuleOperation,
    MoveSymbolClosureToModuleOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

bindings = "nominal_refactor_advisor/lexical_bindings.py"

PLAN = CodemodPlanSequence.from_operations(
    (
        ExtractSymbolClosureToNewModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/ast_tools.py"
            ),
            root_symbol_qualnames=(
                "LexicalScopeBindingAuthority",
                "LEXICAL_SCOPE_BINDING_AUTHORITY",
            ),
            maximum_moved_symbol_count=8,
            destination_path=bindings,
        ),
        MoveSymbolClosureToModuleOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/declaration_dependencies.py"
            ),
            root_symbol_qualnames=("_CurrentScopeBindingCollector",),
            maximum_moved_symbol_count=8,
            destination_path=bindings,
        ),
        ExtractSymbolClosureToNewModuleOperation(
            target=SourceRewriteTarget(file_path=bindings),
            root_symbol_qualnames=("PythonModulePathIdentity",),
            maximum_moved_symbol_count=4,
            destination_path="nominal_refactor_advisor/python_module_identity.py",
        ),
        RenameTopLevelDeclarationAuthorityOperation(
            target=SourceRewriteTarget(
                file_path=bindings, qualname="_CurrentScopeBindingCollector"
            ),
            new_name="ScopeBindingCollector",
        ),
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path=bindings, qualname="LexicalScopeBindingAuthority.bound_names"
            ),
            body_source=(
                "collector = ScopeBindingCollector()\n"
                "for node in nodes:\n"
                "    collector.visit(node)\n"
                "return frozenset(collector.bound_names)\n"
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
