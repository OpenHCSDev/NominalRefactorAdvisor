"""Insert a method and its alias together, then edit the new method's body."""

import json
import sys

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    InsertClassMemberOperation,
    RefactorRecipe,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def build_plan(file_path: str) -> CodemodPlanSequence:
    owner = SourceRewriteTarget(file_path=file_path, qualname="Visitor")
    members = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                recipe_id="method-and-alias",
                operations=(
                    InsertClassMemberOperation(
                        target=owner,
                        source="def visit(self): return 'initial'",
                    ),
                    InsertClassMemberOperation(
                        target=owner,
                        source="visit_alias = visit",
                    ),
                ),
            ),
        ),
    )
    update = CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=file_path, qualname="Visitor.visit"
                ),
                body_source="return 'updated'",
            ),
        )
    )
    return CodemodPlanSequence.compose((members, update))


if __name__ == "__main__":
    print(json.dumps(json_report_object(build_plan(sys.argv[1])), indent=2))
