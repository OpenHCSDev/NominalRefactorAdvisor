"""Consolidate call-result and name lookup onto one flow binding decision."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

flow = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow.py",
    qualname="CompactFunctionFlow",
)
reference = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/value_expression.py",
    qualname="LexicalValueReference",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=reference,
            source=dedent('''\
                def is_prefix_of(self, other: "LexicalValueReference") -> bool:
                    """Whether replacing this reference can replace the other's value."""
                    return (
                        self.root_name == other.root_name
                        and other.attribute_path[:len(self.attribute_path)] == self.attribute_path
                    )
                '''),
        ),
        InsertClassMemberOperation(
            target=flow,
            source=dedent('''\
                def _binding_resolution_for_mutations(
                    self,
                    mutations: tuple[CompactLexicalMutation, ...],
                    use_position: CompactFlowPosition | None,
                ) -> CompactBindingMutationResolution | None:
                    """Select a write once for both lexical and bound-result queries."""
                    if not mutations:
                        return None
                    if any(mutation.position.branch_path for mutation in mutations):
                        return OpenCompactBindingMutation(
                            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                        )
                    if use_position is None:
                        return self.owner.kind.deferred_binding_resolution(mutations)
                    dominating = tuple(
                        mutation for mutation in mutations
                        if mutation.position.dominates(use_position)
                    )
                    if not dominating:
                        return OpenCompactBindingMutation(
                            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                        )
                    return ExactCompactBindingMutation(dominating[-1])
                '''),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(flow, qualname=f"{flow.qualname}.binding_resolution_for"),
            body_source=dedent('''\
                """Select one binding from ordered flow facts; absence permits outer lookup."""
                return self._binding_resolution_for_mutations(
                    self.mutations_by_root_name.get(root_name, ()), use_position
                )
                '''),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(flow, qualname=f"{flow.qualname}.bound_call_result_for"),
            body_source=dedent('''\
                """Return the unique call whose unchanged result reaches one use."""
                selection = self._binding_resolution_for_mutations(
                    tuple(
                        mutation for mutation in self.mutations
                        if mutation.reference.is_prefix_of(reference)
                    ),
                    use_position,
                )
                binding = None if selection is None else selection.mutation
                if binding is None or binding.kind is not CompactMutationKind.ASSIGNMENT:
                    return None
                matching_calls = tuple(
                    call for call in self.calls
                    if call.result.binding == reference
                    and binding.reference == reference
                    and binding.position.branch_path == call.position.branch_path
                    and binding.position.statement_index == call.position.statement_index
                    and call.position.dominates(binding.position)
                )
                return matching_calls[0] if len(matching_calls) == 1 else None
                '''),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
