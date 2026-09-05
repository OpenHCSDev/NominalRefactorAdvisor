"""Collapse identical validation forwarders into their existing base implementation."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    ReplaceDeclaredCallTargetOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceDeclaredCallTargetOperation(
            target=replace(
                module,
                qualname="DataclassAuthorityMappingRecipeBuilder.authority_target",
            ),
            callee=replace(
                module,
                qualname="DataclassAuthorityMappingRecipeBuilder.resolved_target_matches_fields",
            ),
            expression_source="self.resolved_target_is_exhaustive_dataclass",
            selection_count=SelectionCountExpectation(exact=1),
        ),
        *(
            DeleteTargetOperation(
                target=replace(
                    module, qualname=f"{owner}.resolved_target_matches_fields"
                )
            )
            for owner in (
                "DataclassAuthorityMappingRecipeBuilder",
                "DataclassPayloadProjectionMappingRecipeBuilder",
                "DataclassFieldNameCollectionProjectionMappingRecipeBuilder",
                "DataclassKeyValueSequenceProjectionMappingRecipeBuilder",
                "DataclassConstructorProjectionMappingRecipeBuilder",
            )
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
