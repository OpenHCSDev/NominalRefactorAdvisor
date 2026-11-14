"""Shared nominal contracts for collapsing flat parameters into a carrier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjection,
    CompactProductAuthority,
)
from nominal_refactor_advisor.fail_closed_proof import ProofObligation
from nominal_refactor_advisor.product_flow import (
    CompactFunctionDeclaration,
    CompactFlowContext,
    CompactProductFlowModuleProjection,
)
from nominal_refactor_advisor.product_flow_authority import (
    CompactCallableComponentAuthorityProof,
    CompactFunctionCallIdentity,
    CompactProductFlowRepository,
    CompactResolvedFunctionCall,
    ProductFlowRepository,
    SourceProductFlowRepository,
)
from nominal_refactor_advisor.value_expression import LexicalValueReference


@dataclass(frozen=True)
class CarrierCollapseFieldBinding:
    """One authority field mapped injectively to a callee parameter."""

    field_name: str
    parameter_name: str
    value_reference: LexicalValueReference


class CarrierCollapseCallEdge(ABC):
    """A complete carrier-to-parameter mapping across one nominal call edge."""

    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[CarrierCollapseFieldBinding, ...]

    @property
    def caller_symbol(self) -> str:
        return self.resolved_call.context.owner_symbol

    @property
    def call_identity(self) -> CompactFunctionCallIdentity:
        return CompactFunctionCallIdentity.from_resolution(self.resolved_call)

    @property
    @abstractmethod
    def carrier_source_participant_symbols(self) -> tuple[str, ...]:
        """Return participants whose remaining call arguments use the carrier."""

        raise NotImplementedError

    @abstractmethod
    def carrier_value_reference(
        self,
        carrier_parameter_names: Mapping[str, str],
    ) -> LexicalValueReference:
        """Derive the carrier expression supplied to this edge's callee."""

        raise NotImplementedError

    @property
    def callee_symbol(self) -> str:
        return self.resolved_call.callee.identity.symbol

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(binding.field_name for binding in self.field_bindings)

    @property
    def field_mapping(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (binding.field_name, binding.parameter_name)
            for binding in self.field_bindings
        )


@dataclass(frozen=True)
class CarrierCollapseParticipant:
    """One function whose flat field parameters can become one carrier."""

    declaration: CompactFunctionDeclaration
    context: CompactFlowContext

    @property
    def symbol(self) -> str:
        return self.declaration.identity.symbol


class CarrierCollapseAuthorityViolation(ProofObligation):
    """One failed obligation every carrier collapse answers for."""

    INCOMPLETE_PRODUCT_CONSUMPTION = (
        "incomplete_product_consumption",
        "a participant does not consume every projected product field",
        lambda proof: bool(proof.incompletely_consuming_participant_symbols),
    )
    CONFLICTING_CALL_MAPPING = (
        "conflicting_call_mapping",
        "the same participant is reached through competing field mappings",
        lambda proof: bool(proof.conflicting_mapping_symbols),
    )
    NON_POSITIVE_BATCH_COMPRESSION = (
        "non_positive_batch_compression",
        "the whole atomic batch does not pay positive compression rent",
        lambda proof: proof.batch_compression_delta <= 0,
    )


@dataclass(frozen=True, kw_only=True)
class CarrierCollapseAuthorityProof(
    CompactCallableComponentAuthorityProof,
    obligations=CarrierCollapseAuthorityViolation,
):
    """What every carrier collapse proves of its participants.

    They are closed to unknown callers, which the proof inherits, and each consumes the
    whole product through one mapping in a batch that pays for itself.
    """

    projected_field_names_by_edge: tuple[tuple[str, ...], ...]
    incompletely_consuming_participant_symbols: tuple[str, ...]
    conflicting_mapping_symbols: tuple[str, ...]
    batch_compression_delta: int


class ClosedCarrierCollapseComponent(ABC):
    """Nominal component contract consumed by the atomic carrier rewriter."""

    authority: CompactProductAuthority
    participants: tuple[CarrierCollapseParticipant, ...]
    proof: CarrierCollapseAuthorityProof

    @property
    @abstractmethod
    def edges(self) -> tuple[CarrierCollapseCallEdge, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def field_mapping_by_participant(
        self,
    ) -> Mapping[str, tuple[tuple[str, str], ...]]:
        raise NotImplementedError

    @abstractmethod
    def require_rewrite_authority(self) -> None:
        """Raise unless the complete current component is proven rewritable."""

        raise NotImplementedError


CarrierComponent = TypeVar("CarrierComponent", bound=ClosedCarrierCollapseComponent)


@dataclass(frozen=True)
class CarrierCollapseBuilder(ABC, Generic[CarrierComponent]):
    """Share repository construction and proven-result admission across families."""

    repository: ProductFlowRepository

    @classmethod
    def from_projections(
        cls,
        product_projections: Sequence[CompactProductFlowModuleProjection],
        class_projections: Sequence[CompactModuleClassProjection],
    ) -> Self:
        return cls(
            CompactProductFlowRepository(
                product_projections=product_projections,
                class_projections=class_projections,
            )
        )

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        return cls(SourceProductFlowRepository.from_modules(modules))

    @abstractmethod
    def assessed_components(self) -> tuple[CarrierComponent, ...]:
        """Return assessments, including hypotheses without rewrite authority."""
        raise NotImplementedError

    def proven_components(self) -> tuple[CarrierComponent, ...]:
        if not self.repository.product_authorities_by_symbol:
            return ()
        return tuple(
            component
            for component in self.assessed_components()
            if component.proof.is_proven
        )
