"""Use the shared evaluation-phase policy at a declaration-resolved call."""

import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    ReplaceDeclaredCallArgumentsOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceDeclaredCallArgumentsOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/declaration_authority_rename.py",
                qualname="DeclarationAuthorityModuleReferenceProof.proves_qualified_reference",
            ),
            callee=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/class_index.py",
                qualname="ModuleNominalBindingAuthority.qualified_name_at",
            ),
            arguments_source="""reference,
                line=root_surface.use.binding_phase(
                    root_surface.binding_phase,
                    eager_annotations=self.annotation_mode.annotations_execute_at_declaration,
                ).snapshot_line_for(root_surface.reference),""",
            selection_count=SelectionCountExpectation(exact=1),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
