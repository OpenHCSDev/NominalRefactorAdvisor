"""Derive generated descriptor spelling and proof from its native declaration."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    PrependFunctionBodyOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
candidate = SourceRewriteTarget(file_path="nominal_refactor_advisor/detectors/_base.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source="from .class_index import RepositoryModuleBindingProof",
        ),
        EnsureImportOperation(
            target=candidate,
            import_source="from ..descriptor_algebra import ConstantProperty",
        ),
        InsertBeforeTargetOperation(
            target=replace(
                candidate, qualname="CandidateCollectorBoilerplateCandidate.from_module"
            ),
            source="    collector_descriptor_type = ConstantProperty[type[staticmethod]](staticmethod)\n\n",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                candidate,
                qualname="CandidateCollectorBoilerplateCandidate.collector_declaration_source",
            ),
            body_source="""return (
    f"{self.collector_declaration_name} = "
    f"{self.collector_descriptor_type.__name__}({self.collector_name})"
)
""",
        ),
        ReplaceFunctionSignatureOperation(
            target=replace(module, qualname="CandidateCollectorMigration.source_edits"),
            signature_suffix="(self, context: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:",
        ),
        PrependFunctionBodyOperation(
            target=replace(module, qualname="CandidateCollectorMigration.source_edits"),
            body_source=dedent("""\
            RepositoryModuleBindingProof(context.parsed_modules).require_native_type_in_class(
                context.parsed_module_for_source_path(self.target.file_path),
                self.node, self.candidate.collector_descriptor_type,
            )
            """),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
