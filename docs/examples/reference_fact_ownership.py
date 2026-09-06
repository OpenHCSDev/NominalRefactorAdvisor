"""Derive flow read summaries and preserve nominal non-call target facts."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

MODULE = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")


def target(name: str) -> SourceRewriteTarget:
    return replace(MODULE, qualname=name)


def remove(name: str, source: str) -> PatchTargetOperation:
    return PatchTargetOperation(
        target=target(name),
        replacements=(SourceTextReplacement(old_source=source, new_source=""),),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=MODULE, import_source="from itertools import chain"
        ),
        DeleteClassAssignmentsOperation(
            target=target("CompactFunctionFlow"),
            assignment_names=("loaded_value_root_names",),
        ),
        InsertClassMemberOperation(
            target=target("CompactFunctionFlow"),
            source=dedent('''\
                @cached_property
                def loaded_value_root_names(self) -> tuple[str, ...]:
                    """Derive observed names from retained calls and value reads."""
                    return tuple(sorted({
                        reference.root_name
                        for use in chain(
                            self.callable_reference_uses,
                            (call.target_use for call in self.calls),
                        )
                        if (reference := use.target.lexical_reference) is not None
                    }))
            '''),
        ),
        remove(
            "_CompactFlowCollector.__init__",
            "        self.loaded_value_root_names: set[str] = set()\n",
        ),
        remove(
            "_CompactFlowCollector.collect",
            "            loaded_value_root_names=tuple(sorted(self.loaded_value_root_names)),\n",
        ),
        remove(
            "_CompactFlowCollector.visit_Call",
            "        target_reference = LexicalValueReference.from_expression(node.func)\n"
            "        if target_reference is not None:\n"
            "            self.loaded_value_root_names.add(target_reference.root_name)\n",
        ),
        ReplaceFunctionBodyOperation(
            target=target("_CompactFlowCollector.visit_Attribute"),
            body_source=dedent("""\
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    reference = LexicalValueReference.from_expression(node)
                    if reference is not None:
                        self._record_mutation(reference, node)
                    return
                self.visit(node.value)
                self.callable_reference_uses.append(self._callable_reference_use(node))
            """),
        ),
        ReplaceFunctionBodyOperation(
            target=target("_CompactFlowCollector.visit_Name"),
            body_source=dedent("""\
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    self._record_mutation(LexicalValueReference(node.id), node)
                elif isinstance(node.ctx, ast.Load):
                    self.callable_reference_uses.append(self._callable_reference_use(node))
            """),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
