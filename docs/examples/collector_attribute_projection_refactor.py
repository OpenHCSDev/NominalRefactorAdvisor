"""Make generated and authored collectors share native attribute lookup."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.detectors._base import DerivedCandidateCollectorMixin
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/detectors/_base.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="DetectorDeclaration.runtime_namespace"),
            body_source=dedent("""\
                return {
                    "__module__": self.module_name,
                    "__firstlineno__": self.source_line,
                    "detector_declaration": self,
                    **{
                        name: ClassAliasProperty(f"{source_path}.{name}")
                        for source_path, names in (
                            ("detector_declaration", self.required_class_shell_field_names()),
                            ("detector_declaration.options", self.optional_class_shell_field_names()),
                        )
                        for name in names
                    },
                }
                """),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                module, qualname="DerivedCandidateCollectorMixin.__init_subclass__"
            ),
            body_source=dedent("""\
                super().__init_subclass__()
                if _has_finding_spec_contract(cls) and (
                    "candidate_collector" not in vars(cls)
                    or cls.candidate_collector is None
                ):
                    raise TypeError(
                        f"{cls.__name__} must own its candidate_collector declaration"
                    )
                """),
        ),
        *(
            PatchTargetOperation(
                target=replace(
                    module, qualname=f"{base.__qualname__}._candidate_items"
                ),
                replacements=(
                    SourceTextReplacement(
                        old_source=".required_candidate_collector()",
                        new_source=".candidate_collector",
                    ),
                ),
            )
            for base in DerivedCandidateCollectorMixin.registered_collector_base_types()
        ),
        PatchTargetOperation(
            target=replace(
                module,
                qualname="SourceModuleCollectorCandidateDetector._findings_for_source",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source=".required_source_candidate_collector()",
                    new_source=".source_candidate_collector",
                ),
            ),
        ),
        *(
            DeleteTargetOperation(target=replace(module, qualname=qualname))
            for qualname in (
                "DerivedCandidateCollectorMixin.required_candidate_collector",
                "SourceModuleCollectorCandidateDetector.required_source_candidate_collector",
            )
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
