"""Derive both move-owner namespaces before checking class-local capture."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PrependFunctionBodyOperation,
    ProjectFunctionParameterOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

selection = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_selection_context.py"
)
movement = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/class_member_authority_codemod.py"
)
context = replace(movement, qualname="ClassMemberMoveProofContext")
require_move = replace(movement, qualname="ClassMemberMoveSelection.require")

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=selection,
            import_source="from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY",
        ),
        EnsureImportOperation(
            target=movement,
            import_source="from .annotation_semantics import StringizedAnnotationSurface",
        ),
        InsertClassMemberOperation(
            target=replace(selection, qualname="ResolvedClassTarget"),
            source=(
                "@cached_property\n"
                "def bound_names(self) -> frozenset[str]:\n"
                '    """Derive lexical bindings from this class declaration."""\n'
                "    return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.node.body)\n"
            ),
        ),
        DeleteClassAssignmentsOperation(
            target=context, assignment_names=("source_class_bound_names",)
        ),
        InsertClassMemberOperation(
            target=context,
            source=(
                "@cached_property\n"
                "def class_bound_names(self) -> frozenset[str]:\n"
                '    """Names that can capture a header at either move owner."""\n'
                "    return self.source_class.bound_names | self.destination_class.bound_names\n"
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=require_move,
            assignment_name="context",
            source=(
                "context = ClassMemberMoveProofContext(\n"
                "    source_class=source_class,\n"
                "    destination_class=destination_class,\n"
                "    source=source,\n"
                "    module_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body),\n"
                ")"
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=require_move,
            assignment_name="destination_names",
            source="destination_names = destination_class.bound_names",
        ),
        ProjectFunctionParameterOperation(
            target=replace(
                movement, qualname="ClassMethodPromotionStatement.require_safe_move"
            ),
            parameter_name="context",
            attribute_path=("source_class_bound_names",),
            projection_source="context.class_bound_names",
        ),
        PrependFunctionBodyOperation(
            target=replace(
                movement,
                qualname="ClassDeclarationPromotionStatement.require_safe_move",
            ),
            body_source=(
                "annotation_surfaces = (\n"
                "    StringizedAnnotationSurface.from_annotation(self.statement.annotation)\n"
                "    if isinstance(self.statement, ast.AnnAssign) else ()\n"
                ")"
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(
                movement,
                qualname="ClassDeclarationPromotionStatement.require_safe_move",
            ),
            assignment_name="class_local_references",
            source=(
                "class_local_references = frozenset(\n"
                "    node.id\n"
                "    for node in ast.walk(self.statement)\n"
                "    if isinstance(node, ast.Name)\n"
                "    and isinstance(node.ctx, ast.Load)\n"
                "    and node.id in context.class_bound_names\n"
                ") | frozenset(\n"
                "    name\n"
                "    for surface in annotation_surfaces\n"
                "    for name in context.class_bound_names\n"
                "    if surface.reference_count(name)\n"
                ")"
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
