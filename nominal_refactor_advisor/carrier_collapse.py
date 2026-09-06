"""Shared nominal contracts for collapsing flat parameters into a carrier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass


from .class_index import CompactProductAuthority
from .product_flow import (
    CompactFunctionDeclaration,
    CompactFlowContext,
)
from .product_flow_authority import (
    CompactFunctionCallIdentity,
    CompactResolvedFunctionCall,
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


class ClosedCarrierCollapseComponent(ABC):
    """Nominal component contract consumed by the atomic carrier rewriter."""

    authority: CompactProductAuthority
    participants: tuple[CarrierCollapseParticipant, ...]

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
