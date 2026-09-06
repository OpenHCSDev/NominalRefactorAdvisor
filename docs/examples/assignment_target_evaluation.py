"""Retain target evaluation before proving positioned namespace effects."""

import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object


def target(name: str = "") -> SourceRewriteTarget:
    return SourceRewriteTarget(
        file_path="nominal_refactor_advisor/product_flow.py",
        qualname="_CompactFlowCollector" + (f".{name}" if name else ""),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        PatchTargetOperation(
            target=target("_visit_call_target_evaluation"),
            replacements=(
                SourceTextReplacement(
                    old_source="_visit_call_target_evaluation",
                    new_source="_visit_reference_evaluation",
                ),
                SourceTextReplacement(
                    old_source="Retain receiver reads; the call itself owns its terminal lookup.",
                    new_source="Evaluate a reference's receiver and indices before its terminal access.",
                ),
            ),
        ),
        PatchTargetOperation(
            target=target("visit_Call"),
            replacements=(
                SourceTextReplacement(
                    old_source="_visit_call_target_evaluation",
                    new_source="_visit_reference_evaluation",
                ),
            ),
        ),
        InsertClassMemberOperation(
            target=target(),
            source=dedent('''\
                def _record_target_mutation(
                    self, node: ast.expr, kind: CompactMutationKind | None = None,
                ) -> None:
                    """Record the lexical write after its target has been evaluated."""
                    reference = LexicalValueReference.from_expression(node)
                    if reference is not None:
                        self._record_mutation(reference, node, kind)
            '''),
        ),
        ReplaceFunctionBodyOperation(
            target=target("visit_Attribute"),
            body_source=dedent("""\
                self._visit_reference_evaluation(node)
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    self._record_target_mutation(node)
                    return
                self.callable_reference_uses.append(self._callable_reference_use(node))
            """),
        ),
        ReplaceFunctionBodyOperation(
            target=target("visit_AugAssign"),
            body_source=dedent("""\
                self._visit_reference_evaluation(node.target)
                self.callable_reference_uses.append(
                    self._callable_reference_use(node.target)
                )
                self.visit(node.value)
                self._record_target_mutation(
                    node.target, CompactMutationKind.AUGMENTED_ASSIGNMENT,
                )
            """),
        ),
        PatchTargetOperation(
            target=target("visit_AnnAssign"),
            replacements=(
                SourceTextReplacement(
                    old_source="if node.value is None and not self.owner.kind.is_function_scope:\n            return",
                    new_source=dedent("""\
                        if node.value is None:
                            self._visit_reference_evaluation(node.target)
                            if not (
                                isinstance(node.target, ast.Name)
                                and self.owner.kind.is_function_scope
                            ):
                                return
                    """).rstrip().replace("\n", "\n        "),
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
