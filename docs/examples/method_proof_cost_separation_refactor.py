"""Keep exact-method binding proof independent of source-size cost estimates."""

import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

PLAN = CodemodPlanSequence.from_operations(
    (
        PatchTargetOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/exact_method_authority.py",
                qualname="ExactLeafMethodAncestorPromotionComponentBuilder.assessed_components",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="if component.compression_certificate.pays_rent:\n"
                    "                components.append(component)",
                    new_source="components.append(component)",
                ),
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/exact_method_authority.py",
                qualname="ExactMethodRoleComponentBuilder.proven_components",
            ),
            assignment_name="components",
            source=dedent("""\
                components = tuple(
                    ExactMethodRoleComponent(orbits=closed_orbits)
                    for cohort_orbits in orbits_by_cohort.values()
                    if (
                        closed_orbits := receiver_closed_exact_method_orbits(
                            tuple(sorted(cohort_orbits, key=lambda orbit: orbit.method_name))
                        )
                    )
                )
                """),
        ),
        PatchTargetOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/detectors/_structural.py"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="unrelated classes repeat the same promotion-safe method declaration and the exact-role compression certificate pays rent",
                    new_source="unrelated classes repeat the same promotion-safe method declaration with a closed receiver contract; ownership placement remains a practitioner decision",
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
