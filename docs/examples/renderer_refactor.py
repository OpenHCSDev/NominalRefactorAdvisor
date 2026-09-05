"""Emit an authored renderer refactor as a normal, reviewable JSON plan."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    DeleteFunctionAssignmentsOperation,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    PrependFunctionBodyOperation,
    ProjectFunctionParameterOperation,
    PromoteClassMembersToAncestorOperation,
    ReplaceDeclaredCallArgumentsOperation,
    ReplaceFunctionSignatureOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
original = replace(
    module, qualname="DirectBuildFindingRendererFindingRecipeSynthesizer"
)
ancestor = replace(module, qualname="RendererSourceAuthority")
renderer = replace(module, qualname=f"{ancestor.qualname}.renderer_source")
caller = replace(module, qualname="run")

EXTRACTION = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=original, source=f"class {ancestor.qualname}:\n    pass"
        ),
        AddClassBaseOperation(target=original, base_name=ancestor.qualname),
        PromoteClassMembersToAncestorOperation(
            target=original,
            destination=ancestor,
            member_names=("renderer_lambda", "renderer_source"),
        ),
        ReplaceFunctionSignatureOperation(
            target=renderer,
            signature_suffix="(cls, candidate: DirectBuildFindingRendererCandidate, call: ast.Call) -> str:",
        ),
    )
)

WITNESS = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source="from nominal_refactor_advisor.codemod import CurrentLineWitness",
        ),
        ReplaceFunctionSignatureOperation(
            target=renderer,
            signature_suffix="(cls, candidate, call, *, witness: CurrentLineWitness[DirectBuildFindingRendererCandidate, ast.FunctionDef]) -> str:",
        ),
        ProjectFunctionParameterOperation(
            target=renderer,
            parameter_name="candidate",
            projection_source="witness.candidate",
        ),
        PrependFunctionBodyOperation(
            target=renderer,
            body_source='call = witness.candidate.build_call(witness.node)\nif call is None:\n    raise ValueError("finding renderer call is no longer derivable")',
        ),
        ReplaceFunctionSignatureOperation(
            target=renderer,
            signature_suffix="(cls, witness: CurrentLineWitness[DirectBuildFindingRendererCandidate, ast.FunctionDef]) -> str:",
        ),
        ReplaceDeclaredCallArgumentsOperation(
            target=caller,
            callee=renderer,
            arguments_source="witness",
            selection_count=SelectionCountExpectation(exact=1),
        ),
        DeleteFunctionAssignmentsOperation(
            target=caller, assignment_names=("candidate", "call")
        ),
    )
)

PLAN = CodemodPlanSequence.compose((EXTRACTION, WITNESS))

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
