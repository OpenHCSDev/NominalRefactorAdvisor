"""Compact facts for calls that expand one nominal carrier into parameters."""

from __future__ import annotations

from abc import ABC
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Self

from .ast_tools import ParsedModule
from .class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from .product_flow import (
    CompactFlowPosition,
    CompactProductFlowModuleProjection,
    LexicalValueReference,
    compact_product_flow_projection,
)
from .product_flow_authority import (
    CompactProductFlowRepository,
    CompactResolvedFunctionCall,
)


@dataclass(frozen=True)
class CarrierExpansionFieldBinding:
    """One carrier field projected into one callee parameter."""

    field_name: str
    parameter_name: str


@dataclass(frozen=True)
class CarrierExpansionCallEdge(ABC):
    """Shared field-to-parameter semantics for one carrier call edge."""

    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[CarrierExpansionFieldBinding, ...]

    @property
    def caller_symbol(self) -> str:
        return self.resolved_call.context.owner_symbol

    @property
    def callee_symbol(self) -> str:
        return self.resolved_call.callee.identity.symbol

    @property
    def field_mapping(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (binding.field_name, binding.parameter_name)
            for binding in self.field_bindings
        )


@dataclass(frozen=True)
class DeclaredCarrierExpansion(CarrierExpansionCallEdge):
    """One call that expands fields from a declaration-typed carrier value."""

    carrier_class_symbol: str
    carrier_reference: LexicalValueReference


@dataclass(frozen=True)
class ForwardedCarrierExpansion(CarrierExpansionCallEdge):
    """One downstream call forwarding every field through flat parameters."""


@dataclass(frozen=True)
class DeclaredCarrierExpansionComponent:
    """One root carrier expansion and its complete reachable forwarding graph."""

    root: DeclaredCarrierExpansion
    forwarding_edges: tuple[ForwardedCarrierExpansion, ...]

    @property
    def carrier_class_symbol(self) -> str:
        return self.root.carrier_class_symbol

    @property
    def edges(self) -> tuple[CarrierExpansionCallEdge, ...]:
        return (self.root, *self.forwarding_edges)

    @cached_property
    def field_mappings_by_participant(
        self,
    ) -> Mapping[str, tuple[tuple[tuple[str, str], ...], ...]]:
        mappings: dict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
        for edge in self.edges:
            mappings[edge.callee_symbol].add(edge.field_mapping)
        return {
            participant_symbol: tuple(sorted(participant_mappings))
            for participant_symbol, participant_mappings in mappings.items()
        }

    @cached_property
    def field_mapping_by_participant(
        self,
    ) -> Mapping[str, tuple[tuple[str, str], ...]]:
        if any(
            len(participant_mappings) != 1
            for participant_mappings in self.field_mappings_by_participant.values()
        ):
            raise ValueError("carrier expansion has conflicting participant mappings")
        return {
            participant_symbol: participant_mappings[0]
            for participant_symbol, participant_mappings in (
                self.field_mappings_by_participant.items()
            )
        }


@dataclass(frozen=True)
class DeclaredCarrierExpansionBuilder:
    """Derive carrier expansions from resolved calls and bound-result types."""

    repository: CompactProductFlowRepository

    minimum_field_count: ClassVar[int] = 2

    @classmethod
    def from_projections(
        cls,
        product_projections: tuple[CompactProductFlowModuleProjection, ...],
        class_projections: tuple[CompactModuleClassProjection, ...],
    ) -> Self:
        return cls(
            CompactProductFlowRepository(
                product_projections=product_projections,
                class_projections=class_projections,
            )
        )

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        return cls.from_projections(
            tuple(compact_product_flow_projection(module) for module in modules),
            CompactModuleClassProjectionFamily.collect_modules(modules),
        )

    @cached_property
    def expansions(self) -> tuple[DeclaredCarrierExpansion, ...]:
        return tuple(
            expansion
            for call in self.repository.resolved_function_calls
            for expansion in self._call_expansions(call)
        )

    @cached_property
    def outgoing_calls_by_owner_symbol(
        self,
    ) -> Mapping[str, tuple[CompactResolvedFunctionCall, ...]]:
        grouped: dict[str, list[CompactResolvedFunctionCall]] = defaultdict(list)
        for call in self.repository.resolved_function_calls:
            grouped[call.context.owner_symbol].append(call)
        return {
            owner_symbol: tuple(calls) for owner_symbol, calls in grouped.items()
        }

    @cached_property
    def components(self) -> tuple[DeclaredCarrierExpansionComponent, ...]:
        return tuple(self._component(expansion) for expansion in self.expansions)

    def _call_expansions(
        self,
        call: CompactResolvedFunctionCall,
    ) -> tuple[DeclaredCarrierExpansion, ...]:
        call_binding = call.call.bind_to(call.callee)
        if not call_binding.is_exact:
            return ()
        bindings_by_carrier: dict[
            tuple[LexicalValueReference, str],
            list[CarrierExpansionFieldBinding],
        ] = {}
        for parameter in call.callee.call_signature.parameters:
            argument = call_binding.argument_for(parameter.name)
            if argument is None or len(argument.values) != 1:
                continue
            value_reference = argument.values[0].lexical_reference
            if value_reference is None or not value_reference.attribute_path:
                continue
            carrier_reference = LexicalValueReference(
                value_reference.root_name,
                value_reference.attribute_path[:-1],
            )
            carrier_class_symbol = (
                self.repository.declared_bound_value_class_symbol(
                    call.context,
                    carrier_reference,
                    call.call.position,
                )
            )
            if carrier_class_symbol is None:
                continue
            bindings_by_carrier.setdefault(
                (carrier_reference, carrier_class_symbol),
                [],
            ).append(
                CarrierExpansionFieldBinding(
                    field_name=value_reference.terminal_name,
                    parameter_name=parameter.name,
                )
            )
        return tuple(
            DeclaredCarrierExpansion(
                carrier_class_symbol=carrier_class_symbol,
                carrier_reference=carrier_reference,
                resolved_call=call,
                field_bindings=tuple(field_bindings),
            )
            for (
                carrier_reference,
                carrier_class_symbol,
            ), field_bindings in bindings_by_carrier.items()
            if len(field_bindings) >= self.minimum_field_count
            and len({binding.field_name for binding in field_bindings})
            == len(field_bindings)
        )

    def _component(
        self,
        root: DeclaredCarrierExpansion,
    ) -> DeclaredCarrierExpansionComponent:
        pending = deque(((root.callee_symbol, root.field_mapping),))
        visited: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        forwarding_edges: dict[
            tuple[str, CompactFlowPosition, tuple[tuple[str, str], ...]],
            ForwardedCarrierExpansion,
        ] = {}
        while pending:
            caller_symbol, caller_mapping = pending.popleft()
            participant = caller_symbol, caller_mapping
            if participant in visited:
                continue
            visited.add(participant)
            for call in self.outgoing_calls_by_owner_symbol.get(caller_symbol, ()):
                edge = self._forwarded_edge(call, caller_mapping)
                if edge is None:
                    continue
                edge_identity = (
                    edge.resolved_call.context.owner_symbol,
                    edge.resolved_call.call.position,
                    edge.field_mapping,
                )
                forwarding_edges[edge_identity] = edge
                pending.append((edge.callee_symbol, edge.field_mapping))
        return DeclaredCarrierExpansionComponent(
            root=root,
            forwarding_edges=tuple(forwarding_edges.values()),
        )

    def _forwarded_edge(
        self,
        call: CompactResolvedFunctionCall,
        caller_mapping: tuple[tuple[str, str], ...],
    ) -> ForwardedCarrierExpansion | None:
        call_binding = call.call.bind_to(call.callee)
        if not call_binding.is_exact:
            return None
        parameters_by_origin: dict[LexicalValueReference, set[str]] = defaultdict(set)
        for parameter in call.callee.call_signature.parameters:
            argument = call_binding.argument_for(parameter.name)
            if argument is None or len(argument.values) != 1:
                continue
            reference = argument.values[0].lexical_reference
            if reference is None:
                continue
            origin = call.context.flow.value_origin_for(
                reference,
                call.call.position,
            ).exact_origin
            parameters_by_origin[reference].add(parameter.name)
            if origin is not None:
                parameters_by_origin[origin].add(parameter.name)
        field_bindings = []
        for field_name, caller_parameter_name in caller_mapping:
            parameter_names = parameters_by_origin.get(
                LexicalValueReference(caller_parameter_name),
                set(),
            )
            if len(parameter_names) != 1:
                return None
            field_bindings.append(
                CarrierExpansionFieldBinding(
                    field_name=field_name,
                    parameter_name=next(iter(parameter_names)),
                )
            )
        if len({binding.parameter_name for binding in field_bindings}) != len(
            field_bindings
        ):
            return None
        return ForwardedCarrierExpansion(
            resolved_call=call,
            field_bindings=tuple(field_bindings),
        )
