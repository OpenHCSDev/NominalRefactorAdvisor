"""Guard current-class lookup through shared receiver and candidate contracts."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

FLOW = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
REPOSITORY = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)
CALL_SOURCE = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_call_source.py"
)


def member(module: SourceRewriteTarget, name: str) -> SourceRewriteTarget:
    return replace(module, qualname=name)


PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=member(FLOW, "CompactCallTargetResolverABC"),
            source=dedent('''\
                @abstractmethod
                def _through_receiver_binding(
                    self, context: ResolutionContextT, position: CompactFlowPosition,
                    resolution: TargetResolutionT,
                ) -> TargetResolutionT:
                    """Require the current-class receiver to retain its entry origin."""
                    raise NotImplementedError
            '''),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CurrentClassCallTargetReference"),
            source=dedent("""\
                def resolve(
                    self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                    context: ResolutionContextT, position: CompactFlowPosition,
                ) -> TargetResolutionT:
                    return resolver._through_receiver_binding(
                        context, position,
                        self.resolve_current_class_target(resolver, context, position),
                    )
            """),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CurrentClassCallTargetReference"),
            source=dedent("""\
                def resolve_current_class_target(
                    self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                    context: ResolutionContextT, position: CompactFlowPosition,
                ) -> TargetResolutionT:
                    return resolver._local_function_target_resolution(context, self)
            """),
        ),
        PatchTargetOperation(
            target=member(FLOW, "CurrentClassMemberMethodReference.resolve"),
            replacements=(
                SourceTextReplacement(
                    old_source="def resolve(",
                    new_source="def resolve_current_class_target(",
                ),
            ),
        ),
        InsertClassMemberOperation(
            target=member(REPOSITORY, "CompactProductFlowRepository"),
            source=dedent("""\
                def _through_receiver_binding(
                    self, context: CompactProductFlowContext, position: CompactFlowPosition,
                    resolution: CompactCallTargetResolution,
                ) -> CompactCallTargetResolution:
                    declaration = context.declaration
                    if declaration is None or declaration.nominal_receiver_name is None:
                        return UnboundedCompactFunctionTarget(
                            resolution.possible_symbols,
                            CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                        )
                    reference = LexicalValueReference(declaration.nominal_receiver_name)
                    origin = context.flow.value_origin_for(reference, position)
                    if origin.exact_origin != reference:
                        return UnboundedCompactFunctionTarget(
                            resolution.possible_symbols,
                            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
                        )
                    return resolution
            """),
        ),
        InsertClassMemberOperation(
            target=member(REPOSITORY, "CompactCallTargetResolution"),
            source=dedent('''\
                def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
                    """Project potentially referenced participants through this target's bound."""
                    return symbols.intersection(self.possible_symbols)
            '''),
        ),
        InsertBeforeTargetOperation(
            target=member(REPOSITORY, "_CompactClassMemberResolution"),
            source=dedent('''\
                class UnboundedCompactFunctionTarget(OpenCompactFunctionTarget):
                    """Receiver provenance cannot exclude any participant; observed names remain diagnostic."""

                    def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
                        return symbols


            '''),
        ),
        PatchTargetOperation(
            target=member(REPOSITORY, "OpenCompactFunctionTarget.through_alias"),
            replacements=(
                SourceTextReplacement(
                    old_source="return OpenCompactFunctionTarget(",
                    new_source="return type(self)(",
                ),
            ),
        ),
        PatchTargetOperation(
            target=member(
                REPOSITORY,
                "CompactProductFlowRepository._class_member_method_resolution",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="return OpenCompactFunctionTarget(\n                candidate_symbols,\n                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,",
                    new_source="return UnboundedCompactFunctionTarget(\n                candidate_symbols,\n                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,",
                ),
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=member(
                REPOSITORY, "CompactProductFlowRepository.callable_escapes_for"
            ),
            body_source=dedent("""\
                symbols = frozenset((function_symbol,))
                return tuple(
                    escape for escape in self.callable_escapes
                    if escape.target_resolution.candidate_symbols_within(symbols)
                )
            """),
        ),
        PatchTargetOperation(
            target=member(
                REPOSITORY,
                "CompactProductFlowRepository.callable_component_authority_proof",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="for possible_symbol in resolution.target_resolution.possible_symbols\n            if possible_symbol in participant_symbols",
                    new_source="for possible_symbol in resolution.target_resolution.candidate_symbols_within(participant_symbols)",
                ),
                SourceTextReplacement(
                    old_source=dedent("""\
                        escaping_callable_symbols = participant_symbols.intersection(
                            symbol
                            for escape in self.callable_escapes
                            for symbol in escape.target_resolution.possible_symbols
                        )
                    """).rstrip().replace("\n", "\n        "),
                    new_source=dedent("""\
                        escaping_callable_symbols = {
                            symbol
                            for escape in self.callable_escapes
                            for symbol in escape.target_resolution.candidate_symbols_within(participant_symbols)
                        }
                    """).rstrip().replace("\n", "\n        "),
                ),
            ),
        ),
        PatchTargetOperation(
            target=member(CALL_SOURCE, "DeclaredCallRewriteABC.selected_calls"),
            replacements=(
                SourceTextReplacement(
                    old_source="        callee_symbol = source_index.symbol_for_target(self.callee)",
                    new_source="        callee_symbol = source_index.symbol_for_target(self.callee)\n        callee_symbols = frozenset((callee_symbol,))",
                ),
                SourceTextReplacement(
                    old_source="callee_symbol in resolution.target_resolution.possible_symbols",
                    new_source="resolution.target_resolution.candidate_symbols_within(callee_symbols)",
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
