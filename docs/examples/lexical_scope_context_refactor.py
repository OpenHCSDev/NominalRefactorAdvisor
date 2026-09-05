"""Promote lexical lookup, derive provenance, then move the shared authority."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    EnsureImportOperation,
    MoveSymbolClosureToModuleOperation,
    PatchTargetOperation,
    PromoteClassMembersToAncestorOperation,
    RemoveImportNamesOperation,
    ReplaceDeclarationDecoratorsOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/declaration_dependencies.py"
)
collector = replace(module, qualname="_DeclarationDependencyCollector")
context = replace(module, qualname="LexicalScopeContext")

INHERIT_CONTEXT = AddClassBaseOperation(
    target=collector, base_name="LexicalScopeContext"
)
PROMOTE_LOOKUP = PromoteClassMembersToAncestorOperation(
    target=collector,
    destination=context,
    member_names=("_resolve_name", "_active_class_scope"),
)
MOVE_CONTEXT = MoveSymbolClosureToModuleOperation(
    target=module,
    root_symbol_qualnames=("LexicalScopeContext",),
    maximum_moved_symbol_count=4,
    destination_path="nominal_refactor_advisor/lexical_scopes.py",
)
OWNERSHIP_PLAN = CodemodPlanSequence.from_operations(
    (INHERIT_CONTEXT, PROMOTE_LOOKUP, MOVE_CONTEXT)
)

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module, import_source="from contextlib import nullcontext\n"
        ),
        INHERIT_CONTEXT,
        PROMOTE_LOOKUP,
        *(
            ReplaceFunctionBodyOperation(
                target=replace(module, qualname=f"{collector.qualname}.{name}"),
                body_source=dedent(body).strip() + "\n",
            )
            for name, body in (
                (
                    "__init__",
                    """
                    super().__init__()
                    self.names_by_use: dict[DeclarationDependencyUse, set[str]] = {
                        use: set() for use in DeclarationDependencyUse
                    }
                    self.use = DeclarationDependencyUse.EXECUTION
                    self.binding_phase = ModuleBindingResolutionPhase.SOURCE_POSITION
                    self.annotation_count = 0
                    self.direct_name_surfaces: list[ModuleNameReferenceSurface] = []
                    self.stringized_annotation_surfaces: list[StringizedAnnotationSurface] = []
                """,
                ),
                (
                    "visit_Lambda",
                    """
                    self._visit_argument_defaults(node.args)
                    with self._scope(FunctionBindingProjection.from_function(node)):
                        with self._binding_phase(ModuleBindingResolutionPhase.FINAL_MODULE):
                            self.visit(node.body)
                """,
                ),
                (
                    "_visit_function",
                    """
                    for decorator in node.decorator_list:
                        self.visit(decorator)
                    self._visit_argument_defaults(node.args)
                    with self._type_parameter_scope(node):
                        self._visit_type_parameters(node)
                        self._visit_argument_annotations(node.args)
                        if node.returns is not None:
                            self._visit_annotation(node.returns)
                        with self._scope(FunctionBindingProjection.from_function(node)):
                            with self._binding_phase(ModuleBindingResolutionPhase.FINAL_MODULE):
                                self._visit_nodes(node.body)
                    self._record_class_binding((node.name,), LexicalNameResolution.INTERNAL)
                """,
                ),
                (
                    "_visit_class",
                    """
                    for decorator in node.decorator_list:
                        self.visit(decorator)
                    with self._type_parameter_scope(node):
                        for base in node.bases:
                            self.visit(base)
                        for keyword in node.keywords:
                            self.visit(keyword)
                        self._visit_type_parameters(node)
                        with self._scope(ClassNamespaceScope(node=node)):
                            self._visit_nodes(node.body)
                    self._record_class_binding((node.name,), LexicalNameResolution.INTERNAL)
                """,
                ),
                (
                    "_type_parameter_scope",
                    """
                    parameter_names = _type_parameter_names(node)
                    context = self._scope(TypeParameterScope(
                        local_names=parameter_names,
                        global_names=frozenset(),
                        nonlocal_names=frozenset(),
                    )) if parameter_names else nullcontext()
                    with context:
                        yield
                """,
                ),
                (
                    "_visit_comprehension_tail",
                    """
                    local_names = {
                        name
                        for generator in node.generators
                        for name in _store_names(generator.target)
                    }
                    scope = FunctionBindingProjection(
                        local_names=frozenset(local_names),
                        global_names=frozenset(),
                        nonlocal_names=frozenset(),
                    )
                    with self._scope(scope):
                        for condition in first.ifs:
                            self.visit(condition)
                        for generator in remaining:
                            self.visit(generator.iter)
                            for condition in generator.ifs:
                                self.visit(condition)
                        self._visit_nodes(result_expressions)
                """,
                ),
            )
        ),
        MOVE_CONTEXT,
        RemoveImportNamesOperation(
            target=module,
            module_name=".lexical_scopes",
            import_names=("ScopeBindingProjection", "LexicalScopeABC"),
        ),
        ReplaceDeclarationDecoratorsOperation(
            target=replace(
                context, file_path="nominal_refactor_advisor/lexical_scopes.py"
            ),
            decorators_source="@dataclass(eq=False)",
        ),
        *(
            PatchTargetOperation(
                target=replace(module, qualname=f"{collector.qualname}.{name}"),
                replacements=(
                    SourceTextReplacement(
                        old_source="owner_classes=tuple(self.owner_classes)",
                        new_source="owner_classes=self.owner_classes",
                    ),
                ),
            )
            for name in ("_record_reference", "_visit_annotation")
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
