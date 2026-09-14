"""Share original-event availability through the existing source-frame base."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)

SOURCE_PATH = "nominal_refactor_advisor/source_execution.py"

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=SOURCE_PATH),
            import_source="from .product_flow import SourceFlowEvent",
        ),
        InsertClassMemberOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH, qualname="SourceNativeFrameResolver"
            ),
            source='''
def require_event_available_in(
    self, prefix: AdmittedExecutionPrefixABC, event: SourceFlowEvent
) -> None:
    """Require an original event in this frame at its canonical completed cut."""
    endpoint = prefix.endpoint
    if self.execution.required_prefix(endpoint.context, endpoint.position) is not prefix:
        raise ValueError("Source event requires the canonical closed source cut")
    occurrence = prefix.require_event(self.native_frame_context, event)
    if occurrence.frame is not self.native_frame_prefix.endpoint.frame:
        raise ValueError("Source event belongs to a different source frame")
''',
        ),
        ReplaceTargetOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH,
                qualname="SourceDefinitionCapture.require_available_in",
            ),
            replacement_source='''
def require_available_in(self, prefix: AdmittedExecutionPrefixABC) -> None:
    """Authenticate a supplied canonical cut containing this original definition."""
    creation = self.creation
    creation.require_original_operation()
    self.require_closed()
    creation.require_event_available_in(prefix, creation.definition)
''',
        ),
        ReplaceTargetOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH,
                qualname="SourceDefinitionCapture.require_available_at",
            ),
            replacement_source='''
def require_available_at(
    self,
    resolver: CapturedReferenceKernel,
    context: CompactFlowContext,
    position: CompactFlowPosition | None,
) -> AdmittedExecutionPrefixABC:
    """Require the installed original definition before a canonical source cut."""
    creation = self.creation
    if resolver is not creation.execution.kernel:
        raise ValueError("Source definition belongs to another execution admission")
    prefix = creation.execution.required_prefix(context, position)
    self.require_available_in(prefix)
    return prefix
''',
        ),
        PatchTargetOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE_PATH,
                qualname="SourceScalarAssignmentStore.require_installation",
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="""endpoint = prefix.endpoint
        if (
            self.execution.required_prefix(endpoint.context, endpoint.position)
            is not prefix
        ):
            raise ValueError("Scalar installation requires its canonical completed cut")
        occurrence = prefix.require_event(context, self.binding)
        if occurrence.frame is not self.native_frame_prefix.endpoint.frame:
            raise ValueError("Scalar installation belongs to a different source frame")""",
                    new_source="self.require_event_available_in(prefix, self.binding)",
                ),
            ),
        ),
    )
)
