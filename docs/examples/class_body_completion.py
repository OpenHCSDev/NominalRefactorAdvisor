"""Join an original final class statement to its canonical completion cut."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertClassMemberOperation,
    SourceRewriteTarget,
)

SOURCE_PATH = "nominal_refactor_advisor/source_execution.py"

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=SOURCE_PATH),
            import_source="from .ast_tools import module_syntax_index",
        ),
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=SOURCE_PATH),
            import_source="from .product_flow import SourceFlowEvaluation",
        ),
        InsertClassMemberOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH, qualname="SourceClassBodyEntryABC"
            ),
            source='''
@property
def final_evaluation(self) -> SourceFlowEvaluation:
    """Join the original final statement, including event-free statements.

    Source completion does not establish native tail execution or construction.
    The retained syntax order prevents a mutated AST from dropping a trailing
    pass or selecting an earlier store as the original completion boundary.
    """
    self.require_original_operation()
    source = self.execution.source
    syntax = module_syntax_index(source.module.module)
    original_body = tuple(
        node for node in syntax.depth_first_nodes
        if isinstance(node, ast.stmt) and syntax.parent_by_node.get(node) is self.node
    )
    if not original_body or len(original_body) != len(self.node.body) or any(
        original is not current
        for original, current in zip(original_body, self.node.body, strict=True)
    ):
        raise ValueError("Class completion requires its original body statements")
    evaluations = tuple(
        evaluation for evaluation in source.evaluation_bounds_by_node.get(original_body[-1], ())
        if evaluation.owner is self.context.flow.owner
    )
    if len(evaluations) != 1:
        raise ValueError("Final statement has no unique original evaluation")
    evaluation = evaluations[0]
    prefix = self.completion_prefix
    remainder = SingleFlowPrefix(self.context, prefix.endpoint.frame, None, evaluation.exit)
    if any(
        operation.owner is evaluation.owner and remainder.contains(operation.event)
        for operation in source.operations
    ) or tuple(self.execution.effects.occurrences(source, remainder)):
        raise ValueError("Later original source operations or effects remain")
    return evaluation
''',
        ),
    )
)
