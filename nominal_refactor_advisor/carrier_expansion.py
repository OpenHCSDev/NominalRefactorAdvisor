"""Compact facts for calls that expand one nominal carrier into parameters."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Callable, ClassVar, Self, TypeAlias

from .ast_tools import ParsedModule
from .carrier_collapse import (
    CarrierCollapseCallEdge,
    CarrierCollapseFieldBinding,
    CarrierCollapseParticipant,
    ClosedCarrierCollapseComponent,
)
from .class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactProductAuthority,
)
from .product_flow import (
    CompactFlowPosition,
    CompactProductFlowModuleProjection,
    compact_product_flow_projection,
)
from .product_flow_authority import (
    CompactCallableComponentAuthorityProof,
    CompactFunctionCallIdentity,
    CompactProductFlowRepository,
    CompactResolvedFunctionCall,
)
from .value_expression import LexicalValueReference

@dataclass(frozen=True)
class DeclaredCarrierExpansion(CarrierCollapseCallEdge):
    """One call that expands fields from a declaration-typed carrier value."""

    carrier_class_symbol: str
    carrier_reference: LexicalValueReference
    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[CarrierCollapseFieldBinding, ...]

    @property
    def carrier_source_participant_symbols(self) -> tuple[str, ...]:
        return ()

    def carrier_value_reference(
        self,
        carrier_parameter_names: Mapping[str, str],
    ) -> LexicalValueReference:
        del carrier_parameter_names
        return self.carrier_reference


@dataclass(frozen=True)
class ForwardedCarrierExpansion(CarrierCollapseCallEdge):
    """One downstream call forwarding every field through flat parameters."""

    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[CarrierCollapseFieldBinding, ...]

    @property
    def carrier_source_participant_symbols(self) -> tuple[str, ...]:
        return (self.caller_symbol,)

    def carrier_value_reference(
        self,
        carrier_parameter_names: Mapping[str, str],
    ) -> LexicalValueReference:
        return LexicalValueReference(carrier_parameter_names[self.caller_symbol])


@dataclass(frozen=True)
class DeclaredCarrierExpansionComponent:
    """One maximal carrier expansion and its reachable forwarding graph."""

    root_edges: tuple[DeclaredCarrierExpansion, ...]
    forwarding_edges: tuple[ForwardedCarrierExpansion, ...]

    @property
    def carrier_class_symbol(self) -> str:
        carrier_symbols = {edge.carrier_class_symbol for edge in self.root_edges}
        if len(carrier_symbols) != 1:
            raise ValueError("carrier expansion has no unique nominal carrier")
        return next(iter(carrier_symbols))

    @property
    def edges(self) -> tuple[CarrierCollapseCallEdge, ...]:
        return (*self.root_edges, *self.forwarding_edges)

    @cached_property
    def participant_symbols(self) -> tuple[str, ...]:
        return tuple(sorted({edge.callee_symbol for edge in self.edges}))

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


CarrierExpansionAuthorityPredicate: TypeAlias = Callable[
    ["DeclaredCarrierExpansionAuthorityProof"],
    bool,
]


class DeclaredCarrierExpansionAuthorityViolation(StrEnum):
    """One failed proof obligation for collapsing a carrier expansion graph."""

    UNPROVEN_CARRIER_PRODUCT = (
        "unproven_carrier_product",
        "the carrier does not have one closed nominal product authority",
        lambda proof: proof.carrier_authority is None,
    )
    INCOMPLETE_PRODUCT_PROJECTION = (
        "incomplete_product_projection",
        "an edge does not project the complete carrier product",
        lambda proof: proof.carrier_authority is not None
        and any(
            frozenset(field_names)
            != frozenset(proof.carrier_authority.field_names)
            for field_names in proof.projected_field_names_by_edge
        ),
    )
    AMBIGUOUS_ROOT_CARRIER = (
        "ambiguous_root_carrier",
        "a root call expands more than one candidate carrier",
        lambda proof: bool(proof.ambiguous_root_call_identities),
    )
    INCOMPLETE_PRODUCT_CONSUMPTION = (
        "incomplete_product_consumption",
        "a participant does not consume every projected carrier field",
        lambda proof: bool(proof.incompletely_consuming_participant_symbols),
    )
    CONFLICTING_CALL_MAPPING = (
        "conflicting_call_mapping",
        "the same participant is reached through competing field mappings",
        lambda proof: bool(proof.conflicting_mapping_symbols),
    )
    COMPETING_CARRIER_AUTHORITY = (
        "competing_carrier_authority",
        "a participant belongs to expansion graphs for competing carriers",
        lambda proof: bool(proof.competing_carrier_participant_symbols),
    )
    REBINDING_OR_MUTATION = (
        "rebinding_or_mutation",
        "a projected field parameter is rebound or mutated in the component",
        lambda proof: bool(proof.mutated_parameter_symbols),
    )
    MISSING_PARTICIPANT_DECLARATION = (
        "missing_participant_declaration",
        "a component participant lacks one declaration and execution flow",
        lambda proof: bool(
            proof.callable_component.missing_declaration_symbols
        ),
    )
    UNRESOLVED_CONSUMER = (
        "unresolved_consumer",
        "a possible participant call cannot be resolved nominally",
        lambda proof: bool(proof.callable_component.unresolved_consumer_symbols),
    )
    INCOMPLETE_CALL_FAMILY = (
        "incomplete_call_family",
        "not every incoming call belongs to the atomic component",
        lambda proof: bool(
            proof.callable_component.incomplete_call_family_symbols
        ),
    )
    ESCAPING_CALLABLE_REFERENCE = (
        "escaping_callable_reference",
        "a participant callable escapes direct nominal invocation",
        lambda proof: bool(proof.callable_component.escaping_callable_symbols),
    )
    SIGNATURE_SEMANTICS_HAZARD = (
        "signature_semantics_hazard",
        "a participant has declaration-time or observed signature semantics",
        lambda proof: bool(proof.callable_component.signature_hazard_symbols),
    )
    PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED = (
        "public_or_external_boundary_not_closed",
        "a participant is not a repository-private callable boundary",
        lambda proof: bool(proof.callable_component.open_boundary_symbols),
    )
    INCOMPLETE_METHOD_FAMILY = (
        "incomplete_method_family",
        "an override-related method declaration lies outside the component",
        lambda proof: bool(
            proof.callable_component.incomplete_method_family_symbols
        ),
    )
    NON_POSITIVE_BATCH_COMPRESSION = (
        "non_positive_batch_compression",
        "the whole atomic batch does not pay positive compression rent",
        lambda proof: proof.batch_compression_delta <= 0,
    )

    def __new__(
        cls,
        value: str,
        explanation: str,
        predicate: CarrierExpansionAuthorityPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._explanation = explanation
        member._predicate = predicate
        return member

    @property
    def explanation(self) -> str:
        return self._explanation

    def is_violated_by(self, proof: "DeclaredCarrierExpansionAuthorityProof") -> bool:
        return self._predicate(proof)


@dataclass(frozen=True)
class DeclaredCarrierExpansionAuthorityProof:
    """Fail-closed authority proof for one maximal carrier expansion graph."""

    carrier_authority: CompactProductAuthority | None
    projected_field_names_by_edge: tuple[tuple[str, ...], ...]
    ambiguous_root_call_identities: tuple[CompactFunctionCallIdentity, ...]
    incompletely_consuming_participant_symbols: tuple[str, ...]
    conflicting_mapping_symbols: tuple[str, ...]
    competing_carrier_participant_symbols: tuple[str, ...]
    mutated_parameter_symbols: tuple[str, ...]
    callable_component: CompactCallableComponentAuthorityProof
    batch_compression_delta: int

    @cached_property
    def violations(self) -> tuple[DeclaredCarrierExpansionAuthorityViolation, ...]:
        return tuple(
            violation
            for violation in DeclaredCarrierExpansionAuthorityViolation
            if violation.is_violated_by(self)
        )

    @property
    def is_proven(self) -> bool:
        return not self.violations

    @property
    def rejection_reason(self) -> str:
        return "; ".join(violation.explanation for violation in self.violations)


@dataclass(frozen=True)
class DeclaredCarrierExpansionAssessment(ClosedCarrierCollapseComponent):
    """One derived expansion component paired with its rewrite authority proof."""

    component: DeclaredCarrierExpansionComponent
    proof: DeclaredCarrierExpansionAuthorityProof
    participants: tuple[CarrierCollapseParticipant, ...]

    @property
    def authority(self) -> CompactProductAuthority:
        if self.proof.carrier_authority is None:
            raise ValueError("carrier expansion has no proven product authority")
        return self.proof.carrier_authority

    @property
    def edges(self) -> tuple[CarrierCollapseCallEdge, ...]:
        return self.component.edges

    @property
    def field_mapping_by_participant(
        self,
    ) -> Mapping[str, tuple[tuple[str, str], ...]]:
        if not self.proof.is_proven:
            raise ValueError("carrier expansion mapping requires a proven component")
        return self.component.field_mapping_by_participant

    def require_rewrite_authority(self) -> None:
        if not self.proof.is_proven:
            raise ValueError("carrier expansion rewrite requires a proven component")


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
        seeds = tuple(self._component_seed(expansion) for expansion in self.expansions)
        seed_indices_by_carrier_participant: dict[
            tuple[str, str],
            list[int],
        ] = defaultdict(list)
        for seed_index, seed in enumerate(seeds):
            for participant_symbol in seed.participant_symbols:
                seed_indices_by_carrier_participant[
                    seed.carrier_class_symbol,
                    participant_symbol,
                ].append(seed_index)
        unvisited = set(range(len(seeds)))
        components: list[DeclaredCarrierExpansionComponent] = []
        for seed_index, seed in enumerate(seeds):
            if seed_index not in unvisited:
                continue
            unvisited.remove(seed_index)
            pending = deque((seed_index,))
            component_indices: list[int] = []
            while pending:
                component_index = pending.popleft()
                component_indices.append(component_index)
                component = seeds[component_index]
                connected_indices = {
                    connected_index
                    for participant_symbol in component.participant_symbols
                    for connected_index in seed_indices_by_carrier_participant[
                        component.carrier_class_symbol,
                        participant_symbol,
                    ]
                    if connected_index in unvisited
                }
                unvisited.difference_update(connected_indices)
                pending.extend(sorted(connected_indices))
            components.append(
                self._merge_components(
                    tuple(seeds[index] for index in component_indices)
                )
            )
        return tuple(components)

    def assessed_components(self) -> tuple[DeclaredCarrierExpansionAssessment, ...]:
        carrier_symbols_by_participant: dict[str, set[str]] = defaultdict(set)
        for component in self.components:
            for participant_symbol in component.participant_symbols:
                carrier_symbols_by_participant[participant_symbol].add(
                    component.carrier_class_symbol
                )
        root_expansion_count_by_call: dict[CompactFunctionCallIdentity, int] = (
            defaultdict(int)
        )
        for expansion in self.expansions:
            root_expansion_count_by_call[expansion.call_identity] += 1
        return tuple(
            DeclaredCarrierExpansionAssessment(
                component,
                self._authority_proof(
                    component,
                    carrier_symbols_by_participant,
                    root_expansion_count_by_call,
                ),
                tuple(
                    CarrierCollapseParticipant(
                        declaration=(
                            self.repository.function_declarations_by_symbol[
                                participant_symbol
                            ]
                        ),
                        context=self.repository.flow_contexts_by_owner_symbol[
                            participant_symbol
                        ],
                    )
                    for participant_symbol in component.participant_symbols
                ),
            )
            for component in self.components
        )

    def proven_components(self) -> tuple[DeclaredCarrierExpansionAssessment, ...]:
        return tuple(
            assessment
            for assessment in self.assessed_components()
            if assessment.proof.is_proven
        )

    def _authority_proof(
        self,
        component: DeclaredCarrierExpansionComponent,
        carrier_symbols_by_participant: Mapping[str, set[str]],
        root_expansion_count_by_call: Mapping[CompactFunctionCallIdentity, int],
    ) -> DeclaredCarrierExpansionAuthorityProof:
        exact_field_mapping_by_participant = {
            participant_symbol: participant_mappings[0]
            for participant_symbol, participant_mappings in (
                component.field_mappings_by_participant.items()
            )
            if len(participant_mappings) == 1
        }
        parameter_names_by_participant = {
            participant_symbol: frozenset(
                parameter_name
                for _field_name, parameter_name in (
                    exact_field_mapping_by_participant.get(participant_symbol, ())
                )
            )
            for participant_symbol in component.participant_symbols
        }
        callable_component = self.repository.callable_component_authority_proof(
            parameter_names_by_participant,
            frozenset(edge.call_identity for edge in component.edges),
        )
        incompletely_consuming_participant_symbols = tuple(
            participant_symbol
            for participant_symbol in component.participant_symbols
            if participant_symbol not in exact_field_mapping_by_participant
            or not parameter_names_by_participant[participant_symbol].issubset(
                self.repository.flow_contexts_by_owner_symbol[
                    participant_symbol
                ].flow.loaded_value_root_names
            )
        )
        mutated_parameter_symbols = tuple(
            participant_symbol
            for participant_symbol in component.participant_symbols
            if self.repository.flow_contexts_by_owner_symbol[
                participant_symbol
            ].flow.mutated_roots_within(
                parameter_names_by_participant[participant_symbol]
            )
        )
        field_count = min(len(edge.field_bindings) for edge in component.edges)
        return DeclaredCarrierExpansionAuthorityProof(
            carrier_authority=self.repository.product_authorities_by_symbol.get(
                component.carrier_class_symbol
            ),
            projected_field_names_by_edge=tuple(
                tuple(binding.field_name for binding in edge.field_bindings)
                for edge in component.edges
            ),
            ambiguous_root_call_identities=tuple(
                edge.call_identity
                for edge in component.root_edges
                if root_expansion_count_by_call[edge.call_identity] != 1
            ),
            incompletely_consuming_participant_symbols=(
                incompletely_consuming_participant_symbols
            ),
            conflicting_mapping_symbols=tuple(
                participant_symbol
                for participant_symbol, mappings in (
                    component.field_mappings_by_participant.items()
                )
                if len(mappings) != 1
            ),
            competing_carrier_participant_symbols=tuple(
                participant_symbol
                for participant_symbol in component.participant_symbols
                if len(carrier_symbols_by_participant[participant_symbol]) != 1
            ),
            mutated_parameter_symbols=mutated_parameter_symbols,
            callable_component=callable_component,
            batch_compression_delta=(field_count - 1)
            * (len(component.participant_symbols) + len(component.edges)),
        )

    def _call_expansions(
        self,
        call: CompactResolvedFunctionCall,
    ) -> tuple[DeclaredCarrierExpansion, ...]:
        call_binding = call.binding
        if not call_binding.is_exact:
            return ()
        bindings_by_carrier: dict[
            tuple[LexicalValueReference, str],
            list[CarrierCollapseFieldBinding],
        ] = {}
        for parameter_name, argument in call.bound_value_uses.items():
            value_reference = argument.lexical_reference
            if value_reference is None or not value_reference.attribute_path:
                continue
            carrier_reference = LexicalValueReference(
                value_reference.root_name,
                value_reference.attribute_path[:-1],
            )
            carrier_class_symbol = self.repository.declared_bound_value_class_symbol(
                call.context,
                carrier_reference,
                argument.position,
            )
            if carrier_class_symbol is None:
                continue
            bindings_by_carrier.setdefault(
                (carrier_reference, carrier_class_symbol),
                [],
            ).append(
                CarrierCollapseFieldBinding(
                    field_name=value_reference.terminal_name,
                    parameter_name=parameter_name,
                    value_reference=value_reference,
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

    def _component_seed(
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
            root_edges=(root,),
            forwarding_edges=tuple(forwarding_edges.values()),
        )

    @staticmethod
    def _merge_components(
        components: tuple[DeclaredCarrierExpansionComponent, ...],
    ) -> DeclaredCarrierExpansionComponent:
        root_edges = {
            (
                edge.call_identity,
                edge.carrier_reference,
                edge.carrier_class_symbol,
                edge.field_mapping,
            ): edge
            for component in components
            for edge in component.root_edges
        }
        forwarding_edges = {
            (edge.call_identity, edge.callee_symbol, edge.field_mapping): edge
            for component in components
            for edge in component.forwarding_edges
        }
        return DeclaredCarrierExpansionComponent(
            root_edges=tuple(root_edges.values()),
            forwarding_edges=tuple(forwarding_edges.values()),
        )

    def _forwarded_edge(
        self,
        call: CompactResolvedFunctionCall,
        caller_mapping: tuple[tuple[str, str], ...],
    ) -> ForwardedCarrierExpansion | None:
        call_binding = call.binding
        if not call_binding.is_exact:
            return None
        parameters_by_origin: dict[LexicalValueReference, set[str]] = defaultdict(set)
        for parameter_name, argument in call.bound_value_uses.items():
            reference = argument.lexical_reference
            if reference is None:
                continue
            origin = argument.origin_in(call.context.flow).exact_origin
            parameters_by_origin[reference].add(parameter_name)
            if origin is not None:
                parameters_by_origin[origin].add(parameter_name)
        field_bindings = []
        for field_name, caller_parameter_name in caller_mapping:
            parameter_names = parameters_by_origin.get(
                LexicalValueReference(caller_parameter_name),
                set(),
            )
            if len(parameter_names) != 1:
                return None
            field_bindings.append(
                CarrierCollapseFieldBinding(
                    field_name=field_name,
                    parameter_name=next(iter(parameter_names)),
                    value_reference=LexicalValueReference(
                        caller_parameter_name
                    ),
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
