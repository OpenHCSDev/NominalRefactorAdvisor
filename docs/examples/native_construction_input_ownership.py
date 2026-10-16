"""Factor actual prepared-input validation into the source construction owner."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)

PATH = "nominal_refactor_advisor/source_execution.py"

PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=SourceRewriteTarget(
                file_path=PATH, qualname="SourceClassBodyEntryABC"
            ),
            source='''def require_construction_inputs(self) -> None:
    """Validate actual prepared values, without claiming constructor effects."""
    tail = self.native_tail
    names = tail.names
    for requirement in NativeCreationBackend.current().class_construction_fields(names):
        requirement.require_value(tail.require_member(requirement.value), self)
    for name in names:
        tail.require_member(name).require_class_installation()
''',
        ),
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path=PATH, qualname="SourceClassEntry.construction_admission"
            ),
            body_source='''"""Complete raw native construction over actual final namespace values."""
self.require_construction_inputs()
''',
        ),
    )
)

COMPILER_RETENTION_PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/codemod_runtime.py",
                qualname="RefactorRecipeOperationCompiler.from_context",
            ),
            body_source="""if isinstance(context, cls):
    return context
snapshot = context.execution_snapshot()
return cls._from_modules_with_indexes(
    snapshot.parsed_modules,
    snapshot.required_class_family_index,
    snapshot._source_index_build_artifacts,
    snapshot.product_flow_repository,
)
""",
        ),
    )
)
