"""Separate effect selection from the existing ordered lexical traversal."""

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

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_namespace.py")
collector = replace(module, qualname="_ClassNamespaceExecutionCollector")

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=collector,
            source=dedent('''\
                class _ClassNamespaceEffectProjection(ast.NodeVisitor):
                    """Select one node's effects; the scope collector alone traverses children."""

                    def __init__(self, scope: _DeclarationDependencyCollector) -> None:
                        self.scope = scope
                        self.effects_by_node: dict[ast.AST, ClassNamespaceEffect] = {}

                    def _record_effect(
                        self, effect_type: type[ClassNamespaceEffect], node: ast.AST
                    ) -> None:
                        self.effects_by_node.setdefault(
                            node, effect_type.from_scope(node, self.scope.use, self.scope)
                        )

                    def generic_visit(self, node: ast.AST) -> None:
                        # Unknown executable forms require proof rather than implicit trust.
                        if isinstance(node, (ast.expr, ast.stmt)):
                            self._record_effect(InstalledClassNamespaceValue, node)

                    def visit_Name(self, node: ast.Name) -> None:
                        # Lookup and binding are accounted for by the lexical traversal.
                        pass

                    visit_Pass = visit_Delete = visit_Lambda = visit_Tuple = visit_List = visit_Name

                    def visit_Call(self, node: ast.Call) -> None:
                        self._record_effect(DescriptorClassNamespaceEffect, node.func)

                    def visit_Subscript(self, node: ast.Subscript) -> None:
                        self._record_effect(GenericAliasClassNamespaceEffect, node.value)

                    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                        for decorator in node.decorator_list:
                            self._record_effect(DescriptorClassNamespaceEffect, decorator)

                    visit_AsyncFunctionDef = visit_FunctionDef

                    def visit_Assign(self, node: ast.Assign | ast.AnnAssign | ast.NamedExpr | ast.Expr) -> None:
                        if node.value is not None:
                            self._record_effect(InstalledClassNamespaceValue, node.value)

                    visit_AnnAssign = visit_NamedExpr = visit_Expr = visit_Assign

                    def visit_If(self, node: ast.If | ast.IfExp) -> None:
                        self._record_effect(InstalledClassNamespaceValue, node.test)

                    visit_IfExp = visit_If

                    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
                        # Python obtains the outer iterator immediately; its body is deferred.
                        self._record_effect(InstalledClassNamespaceValue, node.generators[0].iter)


                '''),
        ),
        ReplaceTargetOperation(
            target=collector,
            replacement_source=dedent('''\
                class _ClassNamespaceExecutionCollector(_DeclarationDependencyCollector):
                    """Attach effect evidence to the existing ordered, scope-aware traversal."""

                    completed_scope: ClassNamespaceScope

                    def __init__(self, owner: ast.ClassDef) -> None:
                        super().__init__()
                        self.owner = owner
                        self.effect_projection = _ClassNamespaceEffectProjection(self)

                    def visit(self, node: ast.AST) -> None:
                        if (
                            self.owner in self.owner_classes
                            and self.binding_phase is ModuleBindingResolutionPhase.SOURCE_POSITION
                        ):
                            self.effect_projection.visit(node)
                        super().visit(node)

                    @contextmanager
                    def _scope(self, scope: LexicalScopeABC) -> Iterator[None]:
                        with super()._scope(scope):
                            yield
                        if isinstance(scope, ClassNamespaceScope) and scope.node is self.owner:
                            self.completed_scope = scope
                '''),
        ),
        PatchTargetOperation(
            target=replace(
                module, qualname="ClassNamespaceExecutionEvidence.from_class"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="collector.effects_by_node",
                    new_source="collector.effect_projection.effects_by_node",
                ),
            ),
        ),
        PatchTargetOperation(
            target=replace(
                module, qualname="InstalledClassNamespaceValue.require_closed"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="""        if isinstance(self.node, (ast.Call, ast.Lambda)):
            # Calls and defaults have their own definition-time evidence.
""",
                    new_source="""        if isinstance(self.node, (ast.Call, ast.Lambda, ast.Tuple, ast.List)):
            # Calls/defaults have separate evidence. Native sequence construction
            # has no element hashing or class installation hooks.
""",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
