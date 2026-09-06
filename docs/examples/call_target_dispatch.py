"""Move call-target dispatch onto the nominal syntax declarations."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertAfterTargetOperation,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    ReplaceClassBaseOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

syntax = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
repository = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py",
    qualname="CompactProductFlowRepository",
)
target = replace(syntax, qualname="CompactCallTargetReference")

PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=syntax, import_source="from typing import Generic, TypeVar"
        ),
        InsertBeforeTargetOperation(
            target=target,
            source=dedent('''\
                ResolutionContextT = TypeVar("ResolutionContextT")
                TargetResolutionT = TypeVar("TargetResolutionT")


                class CompactCallTargetResolverABC(ABC, Generic[ResolutionContextT, TargetResolutionT]):
                    """Repository obligations selected by nominal call-target syntax."""

                    @abstractmethod
                    def _local_function_target_resolution(
                        self, context: ResolutionContextT, target: CompactCallTargetReference,
                    ) -> TargetResolutionT:
                        """Resolve candidates supplied by a target's local lookup contract."""
                        raise NotImplementedError

                    @abstractmethod
                    def _lexical_function_target_resolution(
                        self, context: ResolutionContextT, reference: LexicalValueReference,
                        position: CompactFlowPosition,
                    ) -> TargetResolutionT:
                        """Resolve a lexical access path through its reaching bindings."""
                        raise NotImplementedError

                    @abstractmethod
                    def _class_member_method_resolution(
                        self, context: ResolutionContextT, target: CurrentClassMemberMethodReference,
                        position: CompactFlowPosition,
                    ) -> TargetResolutionT:
                        """Resolve a method through a declared current-class member."""
                        raise NotImplementedError
                '''),
        ),
        InsertClassMemberOperation(
            target=target,
            source=dedent('''\
                def resolve(
                    self,
                    resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                    context: ResolutionContextT,
                    position: CompactFlowPosition,
                ) -> TargetResolutionT:
                    """Select the target's nominal lookup contract."""
                    return resolver._local_function_target_resolution(context, self)
                '''),
        ),
        InsertAfterTargetOperation(
            target=target,
            source=dedent('''\
                class LexicalCallTargetReference(CompactCallTargetReference, ABC):
                    """Call syntax whose target is supplied by an exact lexical path."""

                    @property
                    @abstractmethod
                    def lexical_reference(self) -> LexicalValueReference:
                        raise NotImplementedError

                    def resolve(
                        self,
                        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                        context: ResolutionContextT,
                        position: CompactFlowPosition,
                    ) -> TargetResolutionT:
                        return resolver._lexical_function_target_resolution(
                            context, self.lexical_reference, position
                        )
                '''),
        ),
        *(
            ReplaceClassBaseOperation(
                target=replace(syntax, qualname=name),
                base_name="CompactCallTargetReference",
                replacement_base_name="LexicalCallTargetReference",
            )
            for name in ("BareCallTargetReference", "QualifiedCallTargetReference")
        ),
        InsertClassMemberOperation(
            target=replace(syntax, qualname="CurrentClassMemberMethodReference"),
            source=dedent("""\
                def resolve(
                    self,
                    resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                    context: ResolutionContextT,
                    position: CompactFlowPosition,
                ) -> TargetResolutionT:
                    return resolver._class_member_method_resolution(context, self, position)
                """),
        ),
        EnsureImportOperation(
            target=repository,
            import_source="from .product_flow import CompactCallTargetResolverABC",
        ),
        AddClassBaseOperation(
            target=repository,
            base_name="CompactCallTargetResolverABC[CompactProductFlowContext, CompactFunctionTargetResolution]",
        ),
        InsertClassMemberOperation(
            target=repository,
            source=dedent('''\
                def _local_function_target_resolution(
                    self, context: CompactProductFlowContext, target: CompactCallTargetReference,
                ) -> CompactFunctionTargetResolution:
                    """Resolve source-local candidates without rediscovering target syntax."""
                    candidates = context.flow.local_candidate_symbols(target, context.module_name)
                    if not candidates:
                        return OpenCompactFunctionTarget(
                            (), CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                        )
                    if len(candidates) != 1:
                        return OpenCompactFunctionTarget(
                            candidates, CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                        )
                    return self._function_resolution_for_symbol(candidates[0]).through_descriptor(
                        target.receiver_access(context.declaration)
                    )
                '''),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                repository, qualname=f"{repository.qualname}.resolve_function_target"
            ),
            body_source='"""Resolve through the target declaration rather than its concrete class."""\nreturn target.resolve(self, context, position)',
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
