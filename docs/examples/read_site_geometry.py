"""Retain full read-site geometry and derive its human-facing line once."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)


def target(module: str, name: str) -> SourceRewriteTarget:
    return SourceRewriteTarget(
        file_path=f"nominal_refactor_advisor/{module}.py", qualname=name
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=target("source_geometry", "SourceByteSpan"),
            source=(
                "@property\n"
                "def start_line(self) -> int:\n"
                '    """One-based source line, derived from the retained span."""\n'
                "    return self.start_line_index + 1"
            ),
        ),
        PatchTargetOperation(
            target=target("product_flow", "CompactCallableReferenceUse"),
            replacements=(
                SourceTextReplacement(
                    old_source="    line: int\n",
                    new_source='    source_span: SourceByteSpan\n\n    line = AliasProperty[int]("source_span.start_line")\n',
                ),
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=target(
                "product_flow", "_CompactFlowCollector._callable_reference_use"
            ),
            body_source=(
                "return CompactCallableReferenceUse(\n"
                "    target=self._call_target(node),\n"
                "    position=self._position(),\n"
                "    source_span=SourceByteSpan.require_node(node),\n"
                ")"
            ),
        ),
        DeleteTargetOperation(
            target=target("product_flow", "CompactFunctionCall.line")
        ),
        InsertClassMemberOperation(
            target=target("product_flow", "CompactFunctionCall"),
            source='line = AliasProperty[int]("source_span.start_line")',
        ),
    )
)
