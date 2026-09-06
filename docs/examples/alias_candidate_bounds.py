"""Join unresolved binding evidence without flattening its candidate bounds."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertBeforeTargetOperation,
    PatchTargetOperation,
    ReplaceTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

REPOSITORY = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)


PLAN = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=replace(REPOSITORY, qualname="_CompactClassMemberResolution"),
            source=dedent('''\
            @dataclass(frozen=True)
            class AlternativeCompactFunctionTargets(CompactCallTargetResolution):
                """Unselected binding alternatives retaining each authority's candidate bound."""

                alternatives: tuple[CompactCallTargetResolution, ...]
                violation: CompactFunctionTargetResolutionViolation

                @cached_property
                def possible_symbols(self) -> tuple[str, ...]:
                    return tuple(dict.fromkeys(
                        symbol for alternative in self.alternatives
                        for symbol in alternative.possible_symbols
                    ))

                def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
                    return frozenset().union(*(
                        alternative.candidate_symbols_within(symbols)
                        for alternative in self.alternatives
                    ))


        '''),
        ),
        ReplaceTargetOperation(
            target=replace(
                REPOSITORY,
                qualname="CompactProductFlowRepository._possible_binding_symbols",
            ),
            replacement_source="""\
    def _possible_binding_symbols(
        self, context: CompactProductFlowContext, reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
    ) -> CompactCallTargetResolution:
        mutations = context.flow.mutations_by_root_name.get(reference.root_name, ())
        local_and_imported = OpenCompactFunctionTarget(
            tuple(dict.fromkeys((
                *(".".join((mutation.imported_origin, *reference.attribute_path))
                  for mutation in mutations if mutation.imported_origin is not None),
                ".".join((context.owner_symbol, *reference.parts)),
            ))),
            violation,
        )
        return AlternativeCompactFunctionTargets(
            (local_and_imported, *(
                alias.source_use.resolve(
                    self, context, attribute_path=reference.attribute_path,
                    pending_bindings=pending_bindings | {(context.owner_symbol, mutation)},
                )
                for mutation in mutations
                if (context.owner_symbol, mutation) not in pending_bindings
                and (alias := context.flow.exact_aliases_by_binding_mutation.get(mutation)) is not None
            )),
            violation,
        )
""",
        ),
        PatchTargetOperation(
            target=replace(
                REPOSITORY,
                qualname="CompactProductFlowRepository._scope_binding_resolution",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="""return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference, pending_bindings),
                selection.target_lookup_violation,
            )""",
                    new_source="""return self._possible_binding_resolution(
                context, reference, selection.target_lookup_violation, pending_bindings,
            )""",
                ),
                SourceTextReplacement(
                    old_source="""return OpenCompactFunctionTarget(
                    self._possible_binding_symbols(
                        context, reference, pending_bindings
                    ),
                    CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                )""",
                    new_source="""return self._possible_binding_resolution(
                    context, reference,
                    CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                    pending_bindings,
                )""",
                ),
                SourceTextReplacement(
                    old_source="""return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference, pending_bindings),
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )""",
                    new_source="""return self._possible_binding_resolution(
                context, reference,
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
                pending_bindings,
            )""",
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(
                REPOSITORY,
                qualname="CompactProductFlowRepository._possible_binding_symbols",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="_possible_binding_symbols",
                    new_source="_possible_binding_resolution",
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
