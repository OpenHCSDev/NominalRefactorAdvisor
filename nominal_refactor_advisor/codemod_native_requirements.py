"""Source-bound native-use obligations and explicit authored execution contracts.

An accepted invariant restricts the supported executions of a codemod. It is
never a captured runtime object and never changes the capture kernel's answer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

from .ast_tools import ParsedModule
from .captured_reference import OpenCapturedReference
from .codemod_operations import RefactorRecipeOperation
from .codemod_payload import (
    CodemodPayloadRecord,
    EmptyDefaultStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import CodemodOperationPreflightReport
from .codemod_semantics import CodemodPreflightStatus
from .codemod_source_correspondence import SourceReadCorrespondence
from .codemod_source_edits import CodemodSourceRevision
from .descriptor_algebra import AliasProperty
from .json_reports import DataclassJsonReport
from .native_declarations import NativeDeclaration
from .native_reference import NativeReferenceEnvironment
from .product_flow import CompactDefinitionTarget, CompactMutation
from .source_geometry import SourceByteSpan

if TYPE_CHECKING:
    from .product_flow_authority import SourceProductFlowRepository


@dataclass(frozen=True)
class NativeUseReceipt(CodemodPayloadRecord):
    """Wire identity of one operation-derived, exact-source native requirement.

    Native names are labels compared to the operation's own requirements. They
    are never imported or converted into trusted declarations by the payload.
    """

    operation: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    revision: CodemodSourceRevision = codemod_payload_field(
        PayloadRecordValueCodec(CodemodSourceRevision)
    )
    span: SourceByteSpan = codemod_payload_field(
        PayloadRecordValueCodec(SourceByteSpan)
    )
    native_declarations: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def __post_init__(self) -> None:
        if self.revision.source_hash is None:
            raise ValueError(
                "A native-use receipt requires an existing source revision"
            )
        if not self.native_declarations or len(set(self.native_declarations)) != len(
            self.native_declarations
        ):
            raise ValueError(
                "Native-use requirements need distinct declared expectations"
            )


@dataclass(frozen=True)
class DeclaredNativeUseInvariants(CodemodPayloadRecord):
    """Practitioner-declared native identity and behavior at every transformed use.

    Acceptance covers all supported executions of the selected uses, including
    the relied-on implementation behavior, not merely one observed invocation.
    Empty acceptance leaves every unproved native-use requirement unresolved.
    """

    requirements: tuple[NativeUseReceipt, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(NativeUseReceipt), default=()
    )
    rationale: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(), default=""
    )

    def __post_init__(self) -> None:
        if len(set(self.requirements)) != len(self.requirements):
            raise ValueError("Native-use acceptance must not duplicate a requirement")
        if bool(self.requirements) != bool(self.rationale.strip()):
            raise ValueError(
                "Accepted native-use invariants require an explicit rationale"
            )

    @classmethod
    def from_requirements(
        cls, requirements: tuple[NativeUseRequirement, ...], *, rationale: str
    ) -> DeclaredNativeUseInvariants:
        return cls(
            tuple(requirement.receipt for requirement in requirements), rationale
        )

    def require_current(self, requirements: tuple[NativeUseRequirement, ...]) -> None:
        actual = tuple(requirement.receipt for requirement in requirements)
        if len(set(actual)) != len(actual):
            raise ValueError("Current native-use requirements are ambiguous")
        if not set(self.requirements).issubset(actual):
            raise ValueError(
                "Native-use acceptance is stale, foreign or no longer required"
            )

    def resolve(
        self, requirements: tuple[NativeUseRequirement, ...]
    ) -> tuple[NativeUseResolution, ...]:
        """Keep source evidence distinct from accepted supported-execution claims."""
        self.require_current(requirements)
        resolutions = []
        for requirement in requirements:
            evidence = requirement.inspect()
            if evidence.receipt in self.requirements:
                evidence = replace(
                    evidence,
                    provenance=NativeUseProvenance.DECLARED,
                    rationale=self.rationale,
                )
            resolutions.append(evidence)
        return tuple(resolutions)


class NativeUseProvenance(StrEnum):
    PROVED = ("proved", CodemodPreflightStatus.PASSED)
    CAPTURED_IDENTITY = ("captured_identity", CodemodPreflightStatus.FAILED)
    DECLARED = ("declared", CodemodPreflightStatus.PASSED)
    UNRESOLVED = ("unresolved", CodemodPreflightStatus.FAILED)

    def __new__(cls, value: str, status: CodemodPreflightStatus) -> NativeUseProvenance:
        member = str.__new__(cls, value)
        member._value_ = value
        member.preflight_status = status
        return member

    @property
    def is_admitted(self) -> bool:
        return self.preflight_status.is_passed

    def require_admitted(self, receipt: NativeUseReceipt) -> None:
        if not self.is_admitted:
            raise ValueError(
                f"Native use at {receipt.revision.file_path}:{receipt.span.start_line} "
                "requires proof or an explicit supported-execution invariant"
            )


@dataclass(frozen=True)
class NativeUseResolution(DataclassJsonReport):
    receipt: NativeUseReceipt
    provenance: NativeUseProvenance
    rationale: str

    def preflight_report(self) -> CodemodOperationPreflightReport:
        """Project conditional applicability without discarding its provenance."""
        return CodemodOperationPreflightReport(
            operation=self.receipt.operation,
            status=self.provenance.preflight_status,
            message=self.rationale,
            detail=self,
        )

    def require_admitted(self) -> None:
        self.provenance.require_admitted(self.receipt)


@dataclass(frozen=True, eq=False)
class NativeUseRequirement:
    """One original source read and its operation-owned native expectations."""

    operation: type[RefactorRecipeOperation]
    node: ast.expr
    declarations: tuple[NativeDeclaration, ...]
    environment: NativeReferenceEnvironment
    source_state: SourceProductFlowRepository | None = field(default=None, kw_only=True)

    module = AliasProperty[ParsedModule]("environment.source.module")

    @staticmethod
    def after_edits(
        requirements: tuple[NativeUseRequirement, ...],
        correspondence: SourceReadCorrespondence,
        environment: NativeReferenceEnvironment,
        *,
        source_state: SourceProductFlowRepository | None = None,
    ) -> tuple[NativeUseRequirement, ...]:
        """Derive a batch of new obligations without transporting old acceptance."""
        if source_state is None and any(
            use.source_state is not None for use in requirements
        ):
            raise ValueError(
                "Native-use transport requires its projected source-state authority"
            )
        if environment.source is not correspondence.after or any(
            use.environment.source is not correspondence.before for use in requirements
        ):
            raise ValueError(
                "Native-use transport requires its actual source environments"
            )
        nodes = correspondence.corresponding_reads(use.node for use in requirements)
        return tuple(
            replace(use, node=node, environment=environment, source_state=source_state)
            for use, node in zip(requirements, nodes, strict=True)
        )

    def __post_init__(self) -> None:
        if self.node not in self.environment.source.reference_reads_by_node:
            raise ValueError("A native-use requirement needs a canonical original read")
        if self.source_state is not None and not any(
            module is self.environment.source.module
            for module in self.source_state.modules
        ):
            raise ValueError(
                "Native-use source state belongs to different parsed owners"
            )

    @cached_property
    def receipt(self) -> NativeUseReceipt:
        return NativeUseReceipt(
            self.operation.operation_key(),
            CodemodSourceRevision(
                self.module.file_path,
                CodemodSourceRevision.hash_source(self.module.source),
            ),
            SourceByteSpan.require_node(self.node),
            tuple(declaration.qualified_name for declaration in self.declarations),
        )

    def inspect(self) -> NativeUseResolution:
        """Keep captured identity distinct from original-operation behavior proof."""
        capture = self.environment.capture(self.node)
        if isinstance(capture, OpenCapturedReference):
            return NativeUseResolution(
                self.receipt, NativeUseProvenance.UNRESOLVED, capture.violation.value
            )
        # A contradictory known capture cannot be bypassed by an assumption.
        capture.require_native(self.declarations)
        try:
            self.require_behavior()
        except ValueError as error:
            return NativeUseResolution(
                self.receipt,
                NativeUseProvenance.CAPTURED_IDENTITY,
                str(error),
            )
        return NativeUseResolution(
            self.receipt,
            NativeUseProvenance.PROVED,
            "Original operation behavior and result proved under the supplied source entry premises.",
        )

    def require_behavior(self) -> None:
        if self.source_state is None:
            raise ValueError(
                "Native operation proof requires its complete source-state authority"
            )
        for declaration in self.declarations:
            self.source_state.require_native_source_dependency(declaration)
        self.require_operation_behavior()

    def require_operation_behavior(self) -> None:
        """A declaration read alone specifies no relied-on execution behavior."""
        raise ValueError(
            "Identity captured under the supplied native entry premise; "
            "the relied-on implementation behavior remains a separate obligation."
        )


class NativeInvocationUseRequirement(NativeUseRequirement):
    """A callee read consumed by its actual original call authority."""

    def require_operation_behavior(self) -> None:
        read = self.environment.source.reference_reads_by_node[self.node]
        calls = tuple(
            call for call in read.context.flow.calls if call.target_use is read.use
        )
        if len(calls) != 1:
            raise ValueError("Native use has no unique original invocation")
        authority = self.environment.call_authority(read.context, calls[0])
        authority.require_closed()
        authority.result().require_closed()


class NativeDefinitionUseRequirement(NativeUseRequirement):
    """An eager definition input consumed by its actual construction/application."""

    def require_operation_behavior(self) -> None:
        read = self.environment.source.value_reads_by_node.get(self.node)
        if read is None:
            raise ValueError("Definition input has no original evaluated value")
        operations = tuple(
            operation
            for operation in self.environment.source.operations
            if isinstance(operation.event, CompactMutation)
            and isinstance(operation.event.target, CompactDefinitionTarget)
            and any(use is read.use for use in operation.event.target.header_uses)
        )
        if len(operations) != 1:
            raise ValueError("Native use has no unique original definition application")
        self.environment.capture_definition(operations[0].node).require_closed()
