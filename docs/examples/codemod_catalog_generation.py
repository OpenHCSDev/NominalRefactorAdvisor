"""Expose registered operations through the existing reference generator."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    PatchTargetOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="docs/source/_ext/catalog_generation.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source="from nominal_refactor_advisor.codemod import RefactorRecipeOperation",
        ),
        EnsureImportOperation(
            target=module,
            import_source="from nominal_refactor_advisor.native_declarations import NativeDeclaration",
        ),
        EnsureImportOperation(
            target=module,
            import_source="from nominal_refactor_advisor.source_geometry import read_source_text",
        ),
        InsertBeforeTargetOperation(
            target=replace(module, qualname="_render_pattern_catalog"),
            source=dedent("""\
                def _render_codemod_catalog() -> str:
                    lines = [
                        ".. Generated from RefactorRecipeOperation.__registry__.",
                        ".. Do not edit manually.",
                        "",
                    ]
                    for operation in RefactorRecipeOperation.__registry__.values():
                        declaration = NativeDeclaration(operation)
                        title = operation.__name__
                        lines.extend([
                            title,
                            "-" * len(title),
                            "",
                            f":Declaration: ``{declaration.qualified_name}``",
                            f":Operation key: ``{operation.operation_key()}``",
                            f":Source proof scope: ``{operation.source_dependency_scope.value}``",
                            "",
                            f".. autoclass:: {declaration.qualified_name}",
                            "   :show-inheritance:",
                            "   :no-index:",
                            "",
                        ])
                    return "\\n".join(lines)
                """),
        ),
        PatchTargetOperation(
            target=replace(module, qualname="generate_api_reference_pages"),
            replacements=(
                SourceTextReplacement(
                    old_source="    detector_types = IssueDetector.registered_detector_types()",
                    new_source='    detector_types = IssueDetector.registered_detector_types()\n    _write_if_changed(generated_dir / "codemod_catalog.rst", _render_codemod_catalog())',
                ),
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(module, qualname="_render_detector_reference_page"),
            assignment_name="qualified_name",
            source="qualified_name = NativeDeclaration(detector_type).qualified_name",
        ),
        PatchTargetOperation(
            target=replace(module, qualname="_write_if_changed"),
            replacements=(
                SourceTextReplacement(
                    old_source="path.read_text()",
                    new_source="read_source_text(path)",
                ),
                SourceTextReplacement(
                    old_source="path.write_text(content)",
                    new_source='path.write_text(content, encoding="utf-8", newline="")',
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
