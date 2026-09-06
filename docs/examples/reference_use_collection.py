"""Retain lexical use evidence; resolve callable identity in the repository."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    ReplaceDeclaredCallArgumentsOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
record = replace(module, qualname="_CompactFlowCollector._record_callable_reference")

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionSignatureOperation(
            target=record,
            signature_suffix="(self, node: ast.expr) -> None:",
        ),
        ReplaceDeclaredCallArgumentsOperation(
            target=replace(module, qualname="_CompactFlowCollector.visit_Name"),
            callee=record,
            arguments_source="node",
        ),
        ReplaceFunctionBodyOperation(
            target=record,
            body_source=dedent("""\
                self.callable_reference_uses.append(
                    CompactCallableReferenceUse(
                        target=self._call_target(node),
                        position=self._position(),
                        line=node.lineno,
                    )
                )
                """),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="_CompactFlowCollector.visit_Attribute"),
            body_source=dedent("""\
                reference = LexicalValueReference.from_expression(node)
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    if reference is not None:
                        self._record_mutation(reference, node)
                    return
                self.visit(node.value)
                if reference is not None:
                    self.loaded_value_root_names.add(reference.root_name)
                    self._record_callable_reference(node)
                """),
        ),
        DeleteTargetOperation(
            target=replace(
                module,
                qualname="_CompactFlowCollector._is_potential_callable_reference",
            ),
        ),
        ReplaceFunctionSignatureOperation(
            target=replace(module, qualname="_CompactFlowCollector.__init__"),
            signature_suffix=(
                "(self, *, owner: CompactFlowOwner, "
                "module_identity: PythonModulePathIdentity, "
                "lexical_scope_qualnames: tuple[str, ...], "
                "current_class_qualname: str | None, "
                "current_class_receiver_name: str | None) -> None:"
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="_CompactFlowCollector.__init__"),
            body_source=dedent("""\
                self.owner = owner
                self.module_identity = module_identity
                self.lexical_scope_qualnames = lexical_scope_qualnames
                self.current_class_qualname = current_class_qualname
                self.current_class_receiver_name = current_class_receiver_name
                self.calls: list[CompactFunctionCall] = []
                self.callable_reference_uses: list[CompactCallableReferenceUse] = []
                self.mutations: list[CompactLexicalMutation] = []
                self.exact_value_aliases: list[CompactExactValueAlias] = []
                self.loaded_value_root_names: set[str] = set()
                self.global_binding_names: set[str] = set()
                self.nonlocal_binding_names: set[str] = set()
                self.branch_path: tuple[CompactControlBranch, ...] = ()
                self.statement_index = 0
                self.event_index = 0
                self.call_results: dict[int, CompactCallResult] = {}
                self.mutation_kind = CompactMutationKind.ASSIGNMENT
                """),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="_DeclarationCollector.__init__"),
            body_source=dedent("""\
                self.module_name = module_name
                self.scope_names: list[str] = []
                self.scope_kinds: list[CompactFlowOwnerKind] = []
                self.function_qualnames: list[str] = []
                self.class_qualnames: list[str] = []
                self.function_contexts: list[_FunctionContext] = []
                self.class_contexts: list[_ClassContext] = []
                """),
        ),
        DeleteClassAssignmentsOperation(
            target=replace(module, qualname="_DeclarationCollector"),
            assignment_names=("visit_ImportFrom",),
        ),
        DeleteTargetOperation(
            target=replace(module, qualname="_DeclarationCollector.visit_Import"),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="compact_product_flow_projection"),
            body_source=dedent('''\
                """Project one parsed module into AST-free closed-flow evidence."""

                declarations = _DeclarationCollector(parsed_module.module_name)
                declarations.visit(parsed_module.module)
                flows = [
                    _CompactFlowCollector(
                        owner=CompactFlowOwner(CompactFlowOwnerKind.MODULE, ""),
                        module_identity=parsed_module.module_path_identity,
                        lexical_scope_qualnames=("",),
                        current_class_qualname=None,
                        current_class_receiver_name=None,
                    ).collect(parsed_module.module.body)
                ]
                flows.extend(
                    _CompactFlowCollector(
                        owner=CompactFlowOwner(
                            CompactFlowOwnerKind.CLASS_BODY,
                            context.qualname,
                        ),
                        module_identity=parsed_module.module_path_identity,
                        lexical_scope_qualnames=context.lexical_scope_qualnames,
                        current_class_qualname=context.current_class_qualname,
                        current_class_receiver_name=None,
                    ).collect(context.node.body)
                    for context in declarations.class_contexts
                )
                flows.extend(
                    _CompactFlowCollector(
                        owner=CompactFlowOwner(
                            CompactFlowOwnerKind.FUNCTION,
                            context.declaration.identity.qualname,
                        ),
                        module_identity=parsed_module.module_path_identity,
                        lexical_scope_qualnames=context.lexical_scope_qualnames,
                        current_class_qualname=context.current_class_qualname,
                        current_class_receiver_name=context.declaration.nominal_receiver_name,
                    ).collect(context.node.body)
                    for context in declarations.function_contexts
                )
                return CompactProductFlowModuleProjection(
                    module_name=parsed_module.module_name,
                    file_path=parsed_module.file_path,
                    function_declarations=tuple(
                        context.declaration for context in declarations.function_contexts
                    ),
                    flows=tuple(flows),
                )
                '''),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
