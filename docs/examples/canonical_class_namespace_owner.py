"""Keep prepared namespace admission on the original declaration entry."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)

PATH = "nominal_refactor_advisor/source_execution.py"


def canonical_namespace_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=PATH,
                    qualname="SourceClassBodyEntryABC.require_admitted",
                ),
                body_source="""if initial is not self.initial:
    raise ValueError("Source class belongs to a foreign native admission")
if self.execution.class_entry(self.node) is not self:
    self.require_original_operation()
    raise ValueError("Source class namespace requires its canonical entry")""",
            ),
        )
    )
