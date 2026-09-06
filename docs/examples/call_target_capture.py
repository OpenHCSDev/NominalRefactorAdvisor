"""Capture callable identity before evaluating the call's arguments."""

from dataclasses import replace
import json
from textwrap import dedent, indent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

flow = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
repository = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py",
    qualname="CompactProductFlowRepository",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=flow,
            import_source="from .descriptor_algebra import AliasProperty",
        ),
        InsertClassMemberOperation(
            target=replace(flow, qualname="CompactCallableReferenceUse"),
            source=dedent('''\
                def resolve(
                    self,
                    resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                    context: ResolutionContextT,
                ) -> TargetResolutionT:
                    """Resolve the reference at its evaluation event."""
                    return self.target.resolve(resolver, context, self.position)
                '''),
        ),
        PatchTargetOperation(
            target=replace(flow, qualname="CompactFunctionCall"),
            replacements=(
                SourceTextReplacement(
                    old_source="    target: CompactCallTargetReference\n",
                    new_source="    target_use: CompactCallableReferenceUse\n",
                ),
                SourceTextReplacement(
                    old_source="    source_span: SourceByteSpan\n",
                    new_source=(
                        "    source_span: SourceByteSpan\n\n"
                        '    target = AliasProperty[CompactCallTargetReference]("target_use.target")\n'
                    ),
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(flow, qualname="_CompactFlowCollector"),
            replacements=(
                SourceTextReplacement(
                    old_source=indent(
                        dedent("""\
                        def _record_callable_reference(self, node: ast.expr) -> None:
                            self.callable_reference_uses.append(
                                CompactCallableReferenceUse(
                                    target=self._call_target(node),
                                    position=self._position(),
                                    line=node.lineno,
                                )
                            )
                        """),
                        "    ",
                    ),
                    new_source=indent(
                        dedent("""\
                        def _callable_reference_use(self, node: ast.expr) -> CompactCallableReferenceUse:
                            return CompactCallableReferenceUse(
                                target=self._call_target(node),
                                position=self._position(),
                                line=node.lineno,
                            )
                        """),
                        "    ",
                    ),
                ),
            ),
        ),
        *(
            PatchTargetOperation(
                target=replace(flow, qualname=f"_CompactFlowCollector.{method}"),
                replacements=(
                    SourceTextReplacement(
                        old_source="self._record_callable_reference(node)",
                        new_source=(
                            "self.callable_reference_uses.append("
                            "self._callable_reference_use(node))"
                        ),
                    ),
                ),
            )
            for method in ("visit_Attribute", "visit_Name")
        ),
        PatchTargetOperation(
            target=replace(flow, qualname="_CompactFlowCollector.visit_Call"),
            replacements=(
                SourceTextReplacement(
                    old_source="        self._visit_call_target_evaluation(node.func)\n",
                    new_source=(
                        "        self._visit_call_target_evaluation(node.func)\n"
                        "        target_use = self._callable_reference_use(node.func)\n"
                    ),
                ),
                SourceTextReplacement(
                    old_source="target=self._call_target(node.func),",
                    new_source="target_use=target_use,",
                ),
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                repository, qualname=f"{repository.qualname}.resolve_function_call"
            ),
            body_source=(
                "return call.target_use.resolve(self, context).resolve_call(context, call)"
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                repository,
                qualname=f"{repository.qualname}.resolve_product_construction",
            ),
            body_source=dedent("""\
                return call.target_use.resolve(self, context).resolve_construction(
                    self, context, call
                )
                """),
        ),
        PatchTargetOperation(
            target=replace(
                repository, qualname=f"{repository.qualname}.resolve_callable_escape"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source=(
                        "self.resolve_function_target(\n"
                        "            context,\n"
                        "            use.target,\n"
                        "            use.position,\n"
                        "        ).declaration"
                    ),
                    new_source="use.resolve(self, context).declaration",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
