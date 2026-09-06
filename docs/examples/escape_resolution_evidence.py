"""Keep unresolved callable uses in the shared signature-boundary proof."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

MODULE = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)


def target(name: str) -> SourceRewriteTarget:
    return replace(MODULE, qualname=name)


PLAN = CodemodPlanSequence.from_operations(
    (
        RenameTopLevelDeclarationAuthorityOperation(
            target=target("CompactResolvedCallableEscape"),
            new_name="CompactCallableEscape",
        ),
        ReplaceTargetOperation(
            target=target("CompactCallableEscape"),
            replacement_source=dedent('''\
                class CompactCallableEscape:
                    """One non-call use retaining its complete target-resolution evidence."""

                    context: CompactProductFlowContext
                    use: CompactCallableReferenceUse
                    target_resolution: CompactCallTargetResolution
            '''),
        ),
        ReplaceFunctionBodyOperation(
            target=target("CompactProductFlowRepository.callable_escapes"),
            body_source=dedent("""\
                return tuple(
                    self.resolve_callable_escape(context, use)
                    for context in self.flow_contexts
                    for use in context.flow.callable_reference_uses
                )
            """),
        ),
        ReplaceFunctionSignatureOperation(
            target=target("CompactProductFlowRepository.resolve_callable_escape"),
            signature_suffix="(self, context: CompactProductFlowContext, use: CompactCallableReferenceUse) -> CompactCallableEscape:",
        ),
        ReplaceFunctionBodyOperation(
            target=target("CompactProductFlowRepository.resolve_callable_escape"),
            body_source="return CompactCallableEscape(context, use, use.resolve(self, context))",
        ),
        PatchTargetOperation(
            target=target("CompactProductFlowRepository.callable_escapes_for"),
            replacements=(
                SourceTextReplacement(
                    old_source="escape.declaration.identity.symbol == function_symbol",
                    new_source="function_symbol in escape.target_resolution.possible_symbols",
                ),
            ),
        ),
        PatchTargetOperation(
            target=target(
                "CompactProductFlowRepository.callable_component_authority_proof"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source=dedent("""\
                        escaping_callable_symbols = {
                            participant_symbol
                            for participant_symbol in participant_symbols
                            if self.callable_escapes_for(participant_symbol)
                        }
                    """).rstrip().replace("\n", "\n        "),
                    new_source=dedent("""\
                        escaping_callable_symbols = participant_symbols.intersection(
                            symbol
                            for escape in self.callable_escapes
                            for symbol in escape.target_resolution.possible_symbols
                        )
                    """).rstrip().replace("\n", "\n        "),
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
