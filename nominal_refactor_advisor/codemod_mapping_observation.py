"""Authored mapping observations checked against the complete simulated batch."""

from __future__ import annotations

from dataclasses import dataclass

from .codemod_payload import PayloadRecordArrayValueCodec, codemod_payload_field
from .codemod_preflight import CodemodOperationPreflightReport
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import CodemodPlanDocumentSimulation, CodemodSourceSnapshot
from .codemod_selector_models import SourceRewriteTargetPreflightDetail
from .codemod_semantics import CodemodPreflightStatus
from .codemod_source_correspondence import SourceReadCorrespondence
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
)
from .native_call import CopiedNativeNamespace
from .native_reference import NativeReferenceEnvironment
from .product_flow import SourceFlowOperation
from .source_geometry import SourceByteSpan


@dataclass(frozen=True, kw_only=True)
class RequireDictionaryCopyMappingOperation(SourceReprovedOperation):
    """Require unchanged unordered keys and same-object values at retained copies.

    This is one explicit relation, not full execution equivalence. Source-created
    value identity, dictionary identity and insertion order need separate proofs.
    """

    call_spans: tuple[SourceByteSpan, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceByteSpan)
    )

    def selected_calls(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NativeReferenceEnvironment, tuple[SourceFlowOperation, ...]]:
        _, digest = self.target_digest(snapshot)
        module = snapshot.parsed_module_for_source_path(digest.file_path)
        environment = snapshot.product_flow_repository.native_reference_environment(
            module
        )
        if not self.call_spans:
            raise ValueError("Mapping requirement needs at least one selected copy")
        operations = tuple(
            environment.source.call_operation(span) for span in self.call_spans
        )
        geometry = SourceTextGeometry(module.source)
        start, end = geometry.target_span_offsets(digest)
        for operation in operations:
            call_start, call_end = geometry.required_node_offsets(operation.node)
            if not start <= call_start < call_end <= end:
                raise ValueError(
                    "Selected dictionary copy is outside the declared target"
                )
        return environment, operations

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        self.selected_calls(snapshot)
        return ()

    def simulation_reports(
        self,
        simulation: CodemodPlanDocumentSimulation,
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        self.required_reproof(lambda: self.require_mapping(simulation))
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Complete unordered copied mappings are preserved",
                detail=SourceRewriteTargetPreflightDetail(self.target),
            ),
        )

    def require_mapping(self, simulation: CodemodPlanDocumentSimulation) -> None:
        before, originals = self.selected_calls(
            simulation.after_snapshot_projection.base_snapshot
        )
        snapshot = simulation.required_after_snapshot
        module = snapshot.parsed_module_for_source_path(before.source.module.file_path)
        after = snapshot.product_flow_repository.native_reference_environment(module)
        correspondence = SourceReadCorrespondence(
            simulation, before.source, after.source
        )
        rewritten = correspondence.corresponding_calls(originals)
        for original, current in zip(originals, rewritten, strict=True):
            CopiedNativeNamespace.from_call(before, original).require_same_mapping(
                CopiedNativeNamespace.from_call(after, current)
            )
