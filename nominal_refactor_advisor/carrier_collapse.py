"""Shared nominal contracts for collapsing flat parameters into a carrier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from .ast_tools import ParsedModule
from .class_index import CompactModuleClassProjection, CompactProductAuthority
from .product_flow import (
    CompactFunctionDeclaration,
    CompactFlowContext,
    CompactProductFlowModuleProjection,
)
from .product_flow_authority import (
    CompactFunctionCallIdentity,
    CompactResolvedFunctionCall,
    CompactProductFlowRepository,
    ProductFlowRepository,
    SourceProductFlowRepository,
)
from .value_expression import LexicalValueReference


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


class CarrierCollapseAuthorityProof(ABC):
    """Proof predicate shared by carrier-collapse component families."""

    @property
    @abstractmethod
    def is_proven(self) -> bool:
        raise NotImplementedError


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
