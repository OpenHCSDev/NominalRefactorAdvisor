"""Derive capability obligations and member ownership from native ABCs and MRO."""

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
    file_path="nominal_refactor_advisor/detector_capabilities.py"
)
member_proof = replace(module, qualname="NominalContractMemberEvidence.from_mro")

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(target=module, import_source="from abc import ABC\n"),
        *(
            PatchTargetOperation(
                target=replace(module, qualname=qualname),
                replacements=tuple(
                    SourceTextReplacement(
                        old_source=old_source,
                        new_source=old_source.replace("type[object]", "type[ABC]"),
                    )
                    for old_source in annotations
                ),
            )
            for qualname, annotations in (
                ("DetectorContributionRole.__new__", ("contract_type: type[object]",)),
                ("DetectorContributionRole.contract_type", ("-> type[object]",)),
                (
                    member_proof.qualname,
                    (
                        "declaration_type: type[object]",
                        "requirement_type: type[object]",
                    ),
                ),
                ("_declaration_member_source", ("declaration_type: type[object]",)),
            )
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(module, qualname="DetectorContributionRole.evidence_for"),
            assignment_name="abstract_member_names",
            source="abstract_member_names = self.contract_type.__abstractmethods__\n",
        ),
        PrependFunctionBodyOperation(
            target=member_proof,
            body_source=dedent("""\
                if (
                    requirement_type not in declaration_type.__mro__
                    or member_name not in requirement_type.__abstractmethods__
                ):
                    raise TypeError(
                        f"{member_name!r} is not a nominal contract obligation of "
                        f"{declaration_type.__qualname__} through {requirement_type.__qualname__}"
                    )
                """),
        ),
        ReplaceScopeAssignmentOperation(
            target=member_proof,
            assignment_name="implementation_type",
            source=dedent("""\
                implementation_type = next(
                    candidate
                    for candidate in declaration_type.__mro__
                    if member_name in vars(candidate)
                )
                """),
        ),
        PatchTargetOperation(
            target=member_proof,
            replacements=(
                SourceTextReplacement(
                    old_source="if implementation_type is None:",
                    new_source="if member_name in declaration_type.__abstractmethods__:",
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
