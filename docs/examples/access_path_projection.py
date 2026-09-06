"""Select lexical access paths and remove mirrored projection-authority variants."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    EnsureImportOperation,
    InsertClassMemberOperation,
    ProjectFunctionParameterOperation,
    RemoveClassBaseOperation,
    RemoveImportNamesOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

value = SourceRewriteTarget(file_path="nominal_refactor_advisor/value_expression.py")
source = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_declaration_source.py"
)
operations = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_function_operations.py"
)
exports = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
authority = replace(source, qualname="FunctionBindingProjectionSourceAuthority")
rewrite = replace(source, qualname=f"{authority.qualname}.replacements_for")
operation = replace(operations, qualname="FunctionBindingProjectionOperationABC")
obsolete = (
    "FunctionParameterProjectionSourceAuthority",
    "FunctionLocalProjectionSourceAuthority",
)

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=value, import_source="from collections.abc import Mapping"
        ),
        EnsureImportOperation(
            target=source, import_source="from .ast_tools import AstParentIndex"
        ),
        EnsureImportOperation(
            target=operations,
            import_source=(
                "from .declaration_dependencies import "
                "FunctionBindingABC, FunctionLocalBinding, FunctionParameterBinding"
            ),
        ),
        EnsureImportOperation(
            target=operations,
            import_source="from .codemod_payload import OptionalStringArrayPayloadValueCodec",
        ),
        InsertClassMemberOperation(
            target=replace(value, qualname="LexicalValueReference"),
            source=dedent('''\
                def select_expression(
                    self, root: ast.Name, parent_by_node: Mapping[ast.AST, ast.AST],
                ) -> ast.expr | None:
                    """Select this exact access prefix from an already-owned root."""
                    if root.id != self.root_name:
                        return None
                    expression: ast.expr = root
                    for attribute_name in self.attribute_path:
                        parent = parent_by_node[expression]
                        if not (
                            isinstance(parent, ast.Attribute)
                            and parent.value is expression
                            and parent.attr == attribute_name
                        ):
                            return None
                        expression = parent
                    return expression
                '''),
        ),
        InsertClassMemberOperation(
            target=authority,
            source=dedent('''\
                def selected_reads(
                    self, binding: FunctionBindingABC, attribute_path: tuple[str, ...],
                ) -> tuple[ast.expr, ...]:
                    """Narrow owned roots, rejecting writes and discarded comments."""
                    roots = binding.required_references()
                    if not attribute_path:
                        return roots
                    selector = LexicalValueReference(binding.binding_name, attribute_path)
                    parents = AstParentIndex(self.node).parent_by_node
                    reads = tuple(
                        expression for root in roots
                        if (expression := selector.select_expression(root, parents)) is not None
                    )
                    for read in reads:
                        if not isinstance(read.ctx, ast.Load):
                            raise ValueError("Access projection cannot migrate a direct write or delete")
                        span = SourceTextSpan.from_offsets(self.geometry.required_node_offsets(read))
                        if self.geometry.span_contains_comment(span):
                            raise ValueError("Access projection would discard a comment")
                    return reads
                '''),
        ),
        ReplaceFunctionSignatureOperation(
            target=rewrite,
            signature_suffix=(
                "(self, binding_name: str, reference: LexicalValueReference, *, "
                "binding: FunctionBindingABC, attribute_path: tuple[str, ...] = ()) "
                "-> tuple[SourceTextSpanReplacement, ...]:"
            ),
        ),
        ProjectFunctionParameterOperation(
            target=rewrite,
            parameter_name="binding_name",
            projection_source="binding.binding_name",
        ),
        ReplaceScopeAssignmentOperation(
            target=rewrite,
            assignment_name="reads",
            source="reads = self.selected_reads(binding, attribute_path)",
        ),
        ReplaceFunctionSignatureOperation(
            target=rewrite,
            signature_suffix=(
                "(self, binding: FunctionBindingABC, reference: LexicalValueReference, *, "
                "attribute_path: tuple[str, ...] = ()) "
                "-> tuple[SourceTextSpanReplacement, ...]:"
            ),
        ),
        DeleteTargetOperation(
            target=replace(source, qualname=f"{authority.qualname}.binding_type")
        ),
        RemoveClassBaseOperation(target=authority, base_name="ABC"),
        InsertClassMemberOperation(
            target=operation,
            source=(
                "attribute_path: tuple[str, ...] = "
                "codemod_payload_field(OptionalStringArrayPayloadValueCodec(), default=())"
            ),
        ),
        ReplaceScopeAssignmentOperation(
            target=operation,
            assignment_name="source_authority",
            source="binding_type: ClassVar[type[FunctionBindingABC]]",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                operations, qualname=f"{operation.qualname}.source_edits_from_snapshot"
            ),
            body_source=dedent("""\
                _identifier, target, node = self.target_node_from_context(snapshot)
                authority = FunctionBindingProjectionSourceAuthority(
                    node=node,
                    source=snapshot.sources_by_file_path[target.file_path],
                )
                expression = ast.parse(self.projection_source, mode="eval").body
                reference = LexicalValueReference.from_expression(expression)
                if reference is None:
                    raise ValueError("Binding projection requires a Name/Attribute access path")
                return authority.geometry.physical_edits(
                    file_path=target.file_path,
                    replacements=authority.replacements_for(
                        type(self).binding_type(node, self.binding_name), reference,
                        attribute_path=self.attribute_path,
                    ),
                    rationale=self.rationale
                    or f"Project binding {self.binding_name!r} in {target.qualname!r}.",
                )
                """),
        ),
        *(
            ReplaceScopeAssignmentOperation(
                target=replace(operations, qualname=f"ProjectFunction{kind}Operation"),
                assignment_name="source_authority",
                source=f"binding_type = Function{kind}Binding",
            )
            for kind in ("Parameter", "Local")
        ),
        *(
            DeleteTargetOperation(target=replace(source, qualname=name))
            for name in obsolete
        ),
        *(
            RemoveImportNamesOperation(
                target=module,
                module_name=".codemod_declaration_source",
                import_names=obsolete,
            )
            for module in (operations, exports)
        ),
        RemoveImportNamesOperation(
            target=source,
            module_name=".declaration_dependencies",
            import_names=("FunctionLocalBinding",),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
