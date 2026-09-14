"""Project original children once; reuse them in class-completion queries."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/ast_tools.py",
                qualname="ModuleSyntaxIndex",
            ),
            source='''
@cached_property
def children_by_node(self) -> dict[ast.AST, tuple[ast.AST, ...]]:
    """Original child order projected from the unambiguous parent authority."""
    children: dict[ast.AST, list[ast.AST]] = {}
    for child, parent in self.parent_by_node.items():
        children.setdefault(parent, []).append(child)
    return {parent: tuple(nodes) for parent, nodes in children.items()}
''',
        ),
        PatchTargetOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/source_execution.py",
                qualname="SourceClassBodyEntryABC.final_evaluation",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source=(
                        "for node in syntax.depth_first_nodes\n"
                        "            if isinstance(node, ast.stmt)\n"
                        "            and syntax.parent_by_node.get(node) is self.node"
                    ),
                    new_source=(
                        "for node in syntax.children_by_node.get(self.node, ())\n"
                        "            if isinstance(node, ast.stmt)"
                    ),
                ),
            ),
        ),
    )
)
