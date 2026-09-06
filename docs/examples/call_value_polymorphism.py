"""Make the existing signature binder preserve its caller's value type."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

binding = SourceRewriteTarget(file_path="nominal_refactor_advisor/call_binding.py")
flow = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
repository = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)
call_source = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_call_source.py"
)


def patch(
    target: SourceRewriteTarget, *changes: tuple[str, str]
) -> PatchTargetOperation:
    return PatchTargetOperation(
        target=target,
        replacements=tuple(
            SourceTextReplacement(old_source=old, new_source=new)
            for old, new in changes
        ),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=binding, import_source="from typing import Generic, TypeVar"
        ),
        patch(
            binding,
            (
                "class CompactParameterKind(StrEnum):",
                'CallValueT = TypeVar("CallValueT")\n\n\nclass CompactParameterKind(StrEnum):',
            ),
        ),
        *(
            patch(
                replace(binding, qualname=name),
                (f"class {name}:", f"class {name}(Generic[CallValueT]):"),
                ("CompactValueExpression", "CallValueT"),
            )
            for name in (
                "CompactCallArgument",
                "CompactKeywordArgument",
                "CompactBoundCallArgument",
            )
        ),
        patch(
            replace(binding, qualname="CompactCallBinding"),
            (
                "class CompactCallBinding(ABC):",
                "class CompactCallBinding(ABC, Generic[CallValueT]):",
            ),
            (
                '"""Nominal result of applying a Python signature to one call."""',
                '"""Nominal binding result retaining the supplied value type and objects."""',
            ),
            (
                "CompactBoundCallArgument | None",
                "CompactBoundCallArgument[CallValueT] | None",
            ),
        ),
        patch(
            replace(binding, qualname="ExactCompactCallBinding"),
            (
                "class ExactCompactCallBinding(CompactCallBinding):",
                "class ExactCompactCallBinding(CompactCallBinding[CallValueT]):",
            ),
            (
                "tuple[CompactBoundCallArgument, ...]",
                "tuple[CompactBoundCallArgument[CallValueT], ...]",
            ),
            (
                "CompactBoundCallArgument | None",
                "CompactBoundCallArgument[CallValueT] | None",
            ),
        ),
        patch(
            replace(binding, qualname="ViolatedCompactCallBinding"),
            (
                "class ViolatedCompactCallBinding(CompactCallBinding):",
                "class ViolatedCompactCallBinding(CompactCallBinding[CallValueT]):",
            ),
        ),
        patch(
            replace(binding, qualname="CompactFunctionSignature.bind"),
            (
                "tuple[CompactCallArgument, ...]",
                "tuple[CompactCallArgument[CallValueT], ...]",
            ),
            (
                "tuple[CompactKeywordArgument, ...]",
                "tuple[CompactKeywordArgument[CallValueT], ...]",
            ),
            ("-> CompactCallBinding:", "-> CompactCallBinding[CallValueT]:"),
            (
                "list[tuple[CompactValueExpression, str | None]]",
                "list[tuple[CallValueT, str | None]]",
            ),
        ),
        EnsureImportOperation(
            target=flow, import_source="from .call_binding import CallValueT"
        ),
        patch(
            replace(flow, qualname="CompactCallArguments"),
            (
                "class CompactCallArguments:",
                "class CompactCallArguments(Generic[CallValueT]):",
            ),
            (
                "tuple[CompactCallArgument, ...]",
                "tuple[CompactCallArgument[CallValueT], ...]",
            ),
            (
                "tuple[CompactKeywordArgument, ...]",
                "tuple[CompactKeywordArgument[CallValueT], ...]",
            ),
            (
                "def from_call(cls, node: ast.Call) -> Self:",
                "def from_call(cls, node: ast.Call, project_value: Callable[[ast.expr], CallValueT]) -> Self:",
            ),
            ("CompactValueExpression.project(\n", "project_value(\n"),
            (
                "CompactValueExpression.project(keyword.value)",
                "project_value(keyword.value)",
            ),
            ("tuple[CompactValueExpression, ...]", "tuple[CallValueT, ...]"),
            ('-> "CompactCallBinding":', '-> "CompactCallBinding[CallValueT]":'),
        ),
        patch(
            replace(flow, qualname="CompactFunctionDeclaration.bind_call"),
            (
                "tuple[CompactCallArgument, ...]",
                "tuple[CompactCallArgument[CallValueT], ...]",
            ),
            (
                "tuple[CompactKeywordArgument, ...]",
                "tuple[CompactKeywordArgument[CallValueT], ...]",
            ),
            ("-> CompactCallBinding:", "-> CompactCallBinding[CallValueT]:"),
        ),
        patch(
            replace(flow, qualname="CompactFunctionCall"),
            (
                "arguments: CompactCallArguments\n",
                "arguments: CompactCallArguments[CompactValueExpression]\n",
            ),
            (
                "-> CompactCallBinding:",
                "-> CompactCallBinding[CompactValueExpression]:",
            ),
        ),
        patch(
            replace(flow, qualname="CompactProductConstruction"),
            (
                "tuple[CompactKeywordArgument, ...]",
                "tuple[CompactKeywordArgument[CompactValueExpression], ...]",
            ),
        ),
        patch(
            replace(flow, qualname="_CompactFlowCollector.visit_Call"),
            (
                "CompactCallArguments.from_call(node)",
                "CompactCallArguments[CompactValueExpression].from_call(node, CompactValueExpression.project)",
            ),
        ),
        EnsureImportOperation(
            target=repository, import_source="from .call_binding import CallValueT"
        ),
        patch(
            replace(
                repository, qualname="ResolvedCompactFunctionTarget.bind_arguments"
            ),
            (
                "arguments: CompactCallArguments",
                "arguments: CompactCallArguments[CallValueT]",
            ),
            ("-> CompactCallBinding:", "-> CompactCallBinding[CallValueT]:"),
        ),
        EnsureImportOperation(
            target=repository,
            import_source="from .value_expression import CompactValueExpression",
        ),
        patch(
            replace(repository, qualname="CompactResolvedFunctionCall.binding"),
            (
                "-> CompactCallBinding:",
                "-> CompactCallBinding[CompactValueExpression]:",
            ),
        ),
        EnsureImportOperation(
            target=call_source,
            import_source="from .value_expression import CompactValueExpression",
        ),
        patch(
            replace(call_source, qualname="DeclaredCallArgumentsRewrite.arguments"),
            (
                "-> CompactCallArguments:",
                "-> CompactCallArguments[CompactValueExpression]:",
            ),
            (
                "CompactCallArguments.from_call(expression)",
                "CompactCallArguments[CompactValueExpression].from_call(expression, CompactValueExpression.project)",
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
