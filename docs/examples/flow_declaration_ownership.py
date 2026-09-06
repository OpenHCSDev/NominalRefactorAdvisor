"""Make function declarations own flows and derive their repository views."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

FLOW = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
REPOSITORY = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)


def member(module: SourceRewriteTarget, name: str) -> SourceRewriteTarget:
    return replace(module, qualname=name)


PLAN = CodemodPlanSequence.from_operations(
    (
        DeleteTargetOperation(target=member(FLOW, "CompactFlowOwner")),
        InsertBeforeTargetOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            source=dedent('''\
                class CompactFlowOwner(ABC):
                    """Nominal scope owner, retaining its declaration when it is a function."""

                    kind: CompactFlowOwnerKind
                    qualname: str

                    @property
                    @abstractmethod
                    def declaration(self) -> CompactFunctionDeclaration | None:
                        raise NotImplementedError


                @dataclass(frozen=True)
                class CompactNamespaceFlowOwner(CompactFlowOwner):
                    """Module or class-body scope without a function signature."""

                    kind: CompactFlowOwnerKind
                    qualname: str

                    def __post_init__(self) -> None:
                        if self.kind.is_function_scope:
                            raise ValueError("Function flows must be owned by their declaration")

                    @property
                    def declaration(self) -> None:
                        return None


            '''),
        ),
        PatchTargetOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            replacements=(
                SourceTextReplacement(
                    old_source="class CompactFunctionDeclaration:",
                    new_source="class CompactFunctionDeclaration(CompactFlowOwner):",
                ),
            ),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            source="kind = CompactFlowOwnerKind.FUNCTION",
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            source='qualname = AliasProperty[str]("identity.qualname")',
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            source=dedent("""\
                @property
                def declaration(self) -> CompactFunctionDeclaration:
                    return self
            """),
        ),
        PatchTargetOperation(
            target=member(FLOW, "compact_product_flow_projection"),
            replacements=(
                SourceTextReplacement(
                    old_source='owner=CompactFlowOwner(CompactFlowOwnerKind.MODULE, ""),',
                    new_source='owner=CompactNamespaceFlowOwner(CompactFlowOwnerKind.MODULE, ""),',
                ),
                SourceTextReplacement(
                    old_source="owner=CompactFlowOwner(\n                CompactFlowOwnerKind.CLASS_BODY,",
                    new_source="owner=CompactNamespaceFlowOwner(\n                CompactFlowOwnerKind.CLASS_BODY,",
                ),
                SourceTextReplacement(
                    old_source="owner=CompactFlowOwner(\n                CompactFlowOwnerKind.FUNCTION,\n                context.declaration.identity.qualname,\n            ),",
                    new_source="owner=context.declaration,",
                ),
                SourceTextReplacement(
                    old_source="        function_declarations=tuple(\n            context.declaration for context in declarations.function_contexts\n        ),\n",
                    new_source="",
                ),
            ),
        ),
        DeleteClassAssignmentsOperation(
            target=member(FLOW, "CompactProductFlowModuleProjection"),
            assignment_names=("function_declarations",),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactProductFlowModuleProjection"),
            source=dedent("""\
                @cached_property
                def function_declarations(self) -> tuple[CompactFunctionDeclaration, ...]:
                    return tuple(
                        declaration
                        for flow in self.flows
                        if (declaration := flow.owner.declaration) is not None
                    )
            """),
        ),
        EnsureImportOperation(
            target=REPOSITORY,
            import_source="from .descriptor_algebra import AliasProperty",
        ),
        DeleteClassAssignmentsOperation(
            target=member(REPOSITORY, "CompactProductFlowContext"),
            assignment_names=("declaration",),
        ),
        InsertClassMemberOperation(
            target=member(REPOSITORY, "CompactProductFlowContext"),
            source='declaration = AliasProperty[CompactFunctionDeclaration | None]("flow.owner.declaration")',
        ),
        ReplaceFunctionBodyOperation(
            target=member(REPOSITORY, "CompactProductFlowRepository.flow_contexts"),
            body_source=dedent("""\
                return tuple(
                    CompactProductFlowContext(
                        module_name=projection.module_name,
                        file_path=projection.file_path,
                        flow=flow,
                    )
                    for projection in self.product_projections
                    for flow in projection.flows
                )
            """),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
