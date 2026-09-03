"""Closed-component proofs for replacing flat parameter conveyors with products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from collections.abc import Mapping
from typing import Callable, Self, TypeAlias

from .ast_tools import ParsedModule
from .class_index import (
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
)
from .product_flow import (
    CompactFlowPosition,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactLexicalMutation,
    CompactMutationKind,
    CompactValueOriginResolution,
    CompactProductFlowModuleProjection,
    LexicalValueReference,
    compact_product_flow_projection,
)
from .product_flow_authority import (
    CompactCallableComponentAuthorityProof,
    CompactFunctionCallIdentity,
    CompactFunctionCallResolution,
    CompactProductAuthority,
    CompactProductFlowContext,
    CompactProductFlowRepository,
    CompactResolvedFunctionCall,
    CompactResolvedProductConstruction,
)


@dataclass(frozen=True)
class ParameterConveyorFieldBinding:
    """One authority field mapped injectively to a callee parameter."""

    field_name: str
    parameter_name: str
    value_reference: LexicalValueReference


class ParameterConveyorCallEdge(ABC):
    """A complete product mapping across one nominal call edge."""

    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[ParameterConveyorFieldBinding, ...]

    @property
    @abstractmethod
    def authority(self) -> CompactProductAuthority:
        raise NotImplementedError

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
class ParameterConveyorProductProjection:
    """One authority product expressed as allowed lexical origins per field."""

    references_by_field: tuple[tuple[str, frozenset[LexicalValueReference]], ...]

    @classmethod
    def from_field_mapping(
        cls,
        field_mapping: tuple[tuple[str, str], ...],
    ) -> "ParameterConveyorProductProjection":
        return cls(
            tuple(
                (field_name, frozenset((LexicalValueReference(parameter_name),)))
                for field_name, parameter_name in field_mapping
            )
        )

    def is_covered_by(self, origins: frozenset[LexicalValueReference]) -> bool:
        return all(
            references & origins for _field_name, references in self.references_by_field
        )


@dataclass(frozen=True)
class ConstructedProductCallEdge(ParameterConveyorCallEdge):
    construction: CompactResolvedProductConstruction
    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[ParameterConveyorFieldBinding, ...]
    carrier_binding_is_local: bool
    carrier_binding_dominates_call: bool
    intervening_mutation_roots: tuple[str, ...]

    @property
    def authority(self) -> CompactProductAuthority:
        return self.construction.authority

    @property
    def carrier_source_participant_symbols(self) -> tuple[str, ...]:
        return ()

    def carrier_value_reference(
        self,
        carrier_parameter_names: Mapping[str, str],
    ) -> LexicalValueReference:
        del carrier_parameter_names
        return self.construction.construction.result_binding

    @property
    def carrier_binding_is_unobserved(self) -> bool:
        return (
            self.construction.construction.result_binding.root_name
            not in self.construction.context.flow.loaded_value_root_names
        )

    @property
    def source_values_are_stable_local_loads(self) -> bool:
        return all(
            reference is not None and not reference.attribute_path
            for argument in self.construction.construction.field_arguments
            for reference in (argument.value.lexical_reference,)
        )


@dataclass(frozen=True)
class ForwardedProductCallEdge(ParameterConveyorCallEdge):
    caller_symbol: str
    product_authority: CompactProductAuthority
    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[ParameterConveyorFieldBinding, ...]

    @property
    def authority(self) -> CompactProductAuthority:
        return self.product_authority

    @property
    def carrier_source_participant_symbols(self) -> tuple[str, ...]:
        return (self.caller_symbol,)

    def carrier_value_reference(
        self,
        carrier_parameter_names: Mapping[str, str],
    ) -> LexicalValueReference:
        return LexicalValueReference(carrier_parameter_names[self.caller_symbol])


@dataclass(frozen=True)
class ParameterConveyorParticipant:
    """One function whose flat field parameters can become one carrier."""

    declaration: CompactFunctionDeclaration
    context: CompactProductFlowContext

    @property
    def symbol(self) -> str:
        return self.declaration.identity.symbol


ParameterConveyorAuthorityPredicate: TypeAlias = Callable[
    ["ClosedParameterConveyorAuthorityProof"],
    bool,
]


def _has_no_unique_product_authority(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return len(frozenset(proof.authority_symbols)) != 1


def _has_incomplete_product_projection(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    expected = frozenset(proof.authority_field_names)
    return any(
        frozenset(projected_fields) != expected
        for projected_fields in proof.projected_field_names_by_edge
    )


def _has_non_injective_field_binding(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.non_injective_edge_ids)


def _has_ambiguous_root_carrier(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.ambiguous_root_call_ids)


def _has_incomplete_product_consumption(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.incompletely_consuming_participant_symbols)


def _has_unresolved_consumer(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.unresolved_consumer_symbols)


def _has_missing_participant_declaration(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.missing_declaration_symbols)


def _has_open_value_alias_forwarding(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.open_value_alias_call_ids)


def _has_unresolved_complete_product_call(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.unresolved_complete_product_call_ids)


def _has_incomplete_call_family(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.incomplete_call_family_symbols)


def _has_escaping_callable_reference(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.escaping_callable_symbols)


def _has_conflicting_call_mapping(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.conflicting_mapping_symbols)


def _has_non_dominating_carrier_binding(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.non_dominating_root_symbols)


def _has_rebinding_or_mutation(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.mutated_binding_symbols)


def _has_observed_root_carrier(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.observed_root_carrier_symbols)


def _has_repeated_source_evaluation(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.reevaluated_source_expression_symbols)


def _has_signature_semantics_hazard(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.signature_hazard_symbols)


def _has_open_public_boundary(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.open_boundary_symbols)


def _has_incomplete_method_family(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.callable_component.incomplete_method_family_symbols)


def _has_non_positive_batch_compression(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return proof.batch_compression_delta <= 0


class ClosedParameterConveyorAuthorityViolation(StrEnum):
    """One failed proof obligation for an atomic parameter-conveyor rewrite."""

    NO_UNIQUE_NOMINAL_AUTHORITY = (
        "no_unique_nominal_authority",
        "the component does not descend from exactly one nominal product authority",
        _has_no_unique_product_authority,
    )
    INCOMPLETE_PRODUCT_PROJECTION = (
        "incomplete_product_projection",
        "an edge projects only part of the authority product",
        _has_incomplete_product_projection,
    )
    NON_INJECTIVE_FIELD_BINDING = (
        "non_injective_field_binding",
        "two authority fields map to the same callee parameter",
        _has_non_injective_field_binding,
    )
    AMBIGUOUS_ROOT_CARRIER = (
        "ambiguous_root_carrier",
        "a converted root call is dominated by more than one matching carrier",
        _has_ambiguous_root_carrier,
    )
    INCOMPLETE_PRODUCT_CONSUMPTION = (
        "incomplete_product_consumption",
        "a participant does not consume every authority field",
        _has_incomplete_product_consumption,
    )
    UNRESOLVED_CONSUMER = (
        "unresolved_consumer",
        "a same-name consumer cannot be resolved to one nominal declaration",
        _has_unresolved_consumer,
    )
    MISSING_PARTICIPANT_DECLARATION = (
        "missing_participant_declaration",
        "a component participant lacks one declaration and execution flow",
        _has_missing_participant_declaration,
    )
    OPEN_VALUE_ALIAS_FORWARDING = (
        "open_value_alias_forwarding",
        "a complete product may flow through an alias whose origin is not exact",
        _has_open_value_alias_forwarding,
    )
    UNRESOLVED_COMPLETE_PRODUCT_CALL = (
        "unresolved_complete_product_call",
        "a complete product reaches a call outside the closed component",
        _has_unresolved_complete_product_call,
    )
    INCOMPLETE_CALL_FAMILY = (
        "incomplete_call_family",
        "not every incoming call belongs to the atomic component",
        _has_incomplete_call_family,
    )
    ESCAPING_CALLABLE_REFERENCE = (
        "escaping_callable_reference",
        "a participant callable escapes direct nominal invocation",
        _has_escaping_callable_reference,
    )
    CONFLICTING_CALL_MAPPING = (
        "conflicting_call_mapping",
        "the same participant is reached through competing product mappings",
        _has_conflicting_call_mapping,
    )
    NON_DOMINATING_CARRIER_BINDING = (
        "non_dominating_carrier_binding",
        "a root carrier binding does not dominate its converted call",
        _has_non_dominating_carrier_binding,
    )
    REBINDING_OR_MUTATION_BETWEEN_BINDING_AND_USE = (
        "rebinding_or_mutation_between_binding_and_use",
        "a carrier, source value, or field parameter is mutated across the component",
        _has_rebinding_or_mutation,
    )
    OBSERVED_ROOT_CARRIER = (
        "observed_root_carrier",
        "a fresh root carrier is observed outside the converted call",
        _has_observed_root_carrier,
    )
    REPEATED_SOURCE_EVALUATION = (
        "repeated_source_evaluation",
        "a source expression can change when two evaluations collapse to one",
        _has_repeated_source_evaluation,
    )
    SIGNATURE_SEMANTICS_HAZARD = (
        "signature_semantics_hazard",
        "a decorator, parameter declaration, or receiver has non-plain signature semantics",
        _has_signature_semantics_hazard,
    )
    PUBLIC_OR_EXTERNAL_BOUNDARY_NOT_CLOSED = (
        "public_or_external_boundary_not_closed",
        "a participant is not a repository-private callable boundary",
        _has_open_public_boundary,
    )
    INCOMPLETE_METHOD_FAMILY = (
        "incomplete_method_family",
        "an override-related method declaration lies outside the component",
        _has_incomplete_method_family,
    )
    NON_POSITIVE_BATCH_COMPRESSION = (
        "non_positive_batch_compression",
        "the whole atomic batch does not pay positive compression rent",
        _has_non_positive_batch_compression,
    )

    def __new__(
        cls,
        value: str,
        explanation: str,
        predicate: ParameterConveyorAuthorityPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._explanation = explanation
        member._predicate = predicate
        return member

    @property
    def explanation(self) -> str:
        return self._explanation

    def is_violated_by(self, proof: "ClosedParameterConveyorAuthorityProof") -> bool:
        return self._predicate(proof)


@dataclass(frozen=True)
class ClosedParameterConveyorAuthorityProof:
    """Representation-independent proof that one whole component is final."""

    authority_symbols: tuple[str, ...]
    authority_field_names: tuple[str, ...]
    projected_field_names_by_edge: tuple[tuple[str, ...], ...]
    non_injective_edge_ids: tuple[str, ...]
    ambiguous_root_call_ids: tuple[str, ...]
    incompletely_consuming_participant_symbols: tuple[str, ...]
    open_value_alias_call_ids: tuple[str, ...]
    unresolved_complete_product_call_ids: tuple[str, ...]
    conflicting_mapping_symbols: tuple[str, ...]
    non_dominating_root_symbols: tuple[str, ...]
    mutated_binding_symbols: tuple[str, ...]
    observed_root_carrier_symbols: tuple[str, ...]
    reevaluated_source_expression_symbols: tuple[str, ...]
    callable_component: CompactCallableComponentAuthorityProof
    batch_compression_delta: int

    @cached_property
    def violations(self) -> tuple[ClosedParameterConveyorAuthorityViolation, ...]:
        return tuple(
            violation
            for violation in ClosedParameterConveyorAuthorityViolation
            if violation.is_violated_by(self)
        )

    @property
    def is_proven(self) -> bool:
        return not self.violations

    @property
    def rejection_reason(self) -> str:
        return "; ".join(violation.explanation for violation in self.violations)


@dataclass(frozen=True)
class ClosedParameterConveyorComponent:
    """One maximal connected component and its final-authority proof."""

    authority: CompactProductAuthority
    participants: tuple[ParameterConveyorParticipant, ...]
    root_edges: tuple[ConstructedProductCallEdge, ...]
    forwarding_edges: tuple[ForwardedProductCallEdge, ...]
    proof: ClosedParameterConveyorAuthorityProof

    @property
    def participant_symbols(self) -> tuple[str, ...]:
        return tuple(participant.symbol for participant in self.participants)

    @property
    def edges(self) -> tuple[ParameterConveyorCallEdge, ...]:
        return (*self.root_edges, *self.forwarding_edges)

    @cached_property
    def field_mapping_by_participant(
        self,
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        """Derive each participant's unique authority-to-parameter relation."""

        if not self.proof.is_proven:
            raise ValueError(
                "parameter-conveyor field mapping requires a proven component"
            )
        mappings: dict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
        for edge in self.edges:
            mappings[edge.callee_symbol].add(edge.field_mapping)
        if set(mappings) != set(self.participant_symbols) or any(
            len(participant_mappings) != 1 for participant_mappings in mappings.values()
        ):
            raise ValueError(
                "proven parameter-conveyor component has no unique mapping"
            )
        return {
            participant_symbol: next(iter(participant_mappings))
            for participant_symbol, participant_mappings in mappings.items()
        }


@dataclass(frozen=True)
class _ParameterConveyorComponentSeed:
    authority: CompactProductAuthority
    participant_symbols: frozenset[str]
    root_edges: tuple[ConstructedProductCallEdge, ...]
    forwarding_edges: tuple[ForwardedProductCallEdge, ...]


_RootEdgeIdentity: TypeAlias = tuple[
    str,
    CompactFunctionCallIdentity,
    CompactFlowPosition,
]
_ValueProjectionKey: TypeAlias = tuple[str, LexicalValueReference]


@dataclass(frozen=True)
class ClosedParameterConveyorComponentBuilder:
    """Build maximal components before evaluating any executable candidate."""

    repository: CompactProductFlowRepository

    @classmethod
    def from_projections(
        cls,
        product_projections: tuple[CompactProductFlowModuleProjection, ...],
        class_projections: tuple[CompactModuleClassProjection, ...],
    ) -> Self:
        """Join the two declaration-owned fact families into one proof builder."""

        return cls(
            CompactProductFlowRepository(
                product_projections=product_projections,
                class_projections=class_projections,
            )
        )

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        """Collect both proof families from one complete source snapshot."""

        return cls.from_projections(
            tuple(compact_product_flow_projection(module) for module in modules),
            CompactModuleClassProjectionFamily.collect_modules(modules),
        )

    @cached_property
    def simple_bound_arguments_by_call(
        self,
    ) -> dict[CompactFunctionCallIdentity, dict[str, LexicalValueReference]]:
        return {
            self.call_identity(edge): self._simple_bound_arguments(edge)
            for edge in self.repository.resolved_function_calls
        }

    @cached_property
    def bound_argument_origins_by_call(
        self,
    ) -> dict[
        CompactFunctionCallIdentity,
        dict[str, CompactValueOriginResolution],
    ]:
        return {
            self.call_identity(edge): {
                parameter_name: edge.context.flow.value_origin_for(
                    reference,
                    edge.call.position,
                )
                for parameter_name, reference in self.simple_bound_arguments_by_call[
                    self.call_identity(edge)
                ].items()
            }
            for edge in self.repository.resolved_function_calls
        }

    @cached_property
    def calls_by_value_projection(
        self,
    ) -> dict[_ValueProjectionKey, tuple[CompactResolvedFunctionCall, ...]]:
        grouped: dict[_ValueProjectionKey, list[CompactResolvedFunctionCall]] = (
            defaultdict(list)
        )
        for edge in self.repository.resolved_function_calls:
            for parameter_name, reference in self.simple_bound_arguments_by_call.get(
                self.call_identity(edge), {}
            ).items():
                projections = [reference]
                exact_origin = self.bound_argument_origins_by_call[
                    self.call_identity(edge)
                ][parameter_name].exact_origin
                if exact_origin is not None:
                    projections.append(exact_origin)
                for projection in dict.fromkeys(projections):
                    grouped[(edge.context.owner_symbol, projection)].append(edge)
        return {key: tuple(edges) for key, edges in grouped.items()}

    @cached_property
    def root_edges(self) -> tuple[ConstructedProductCallEdge, ...]:
        edges: dict[_RootEdgeIdentity, ConstructedProductCallEdge] = {}
        for construction in self.repository.resolved_product_constructions:
            authority_fields = construction.authority.field_names
            construction_arguments = self._construction_arguments(construction)
            if (
                frozenset(construction_arguments) != frozenset(authority_fields)
                or not authority_fields
            ):
                continue
            first_field = authority_fields[0]
            source_reference = construction_arguments[first_field]
            if source_reference is None:
                continue
            carrier_reference = LexicalValueReference(
                construction.construction.result_binding.root_name,
                (
                    *construction.construction.result_binding.attribute_path,
                    first_field,
                ),
            )
            first_field_references = (
                *self._reference_equivalents(
                    construction.context,
                    source_reference,
                    construction.call.position,
                ),
                carrier_reference,
            )
            candidate_edges = {
                self.call_identity(edge): edge
                for expected_reference in dict.fromkeys(first_field_references)
                for edge in self.calls_by_value_projection.get(
                    (construction.context.owner_symbol, expected_reference),
                    (),
                )
            }
            for identity, call_edge in candidate_edges.items():
                edge = self._constructed_edge(
                    construction,
                    call_edge,
                    construction_arguments,
                )
                if edge is not None:
                    edges[
                        (
                            edge.authority.class_symbol,
                            identity,
                            construction.call.position,
                        )
                    ] = edge
        return tuple(edges.values())

    def assessed_components(self) -> tuple[ClosedParameterConveyorComponent, ...]:
        seeds = self._component_seeds()
        authority_symbols_by_participant: dict[str, set[str]] = defaultdict(set)
        for seed in seeds:
            for participant_symbol in seed.participant_symbols:
                authority_symbols_by_participant[participant_symbol].add(
                    seed.authority.class_symbol
                )
        return tuple(
            self._assessed_component(seed, authority_symbols_by_participant)
            for seed in seeds
        )

    def proven_components(self) -> tuple[ClosedParameterConveyorComponent, ...]:
        return tuple(
            component
            for component in self.assessed_components()
            if component.proof.is_proven
        )

    def _component_seeds(self) -> tuple[_ParameterConveyorComponentSeed, ...]:
        seeds: list[_ParameterConveyorComponentSeed] = []
        roots_by_authority: dict[str, list[ConstructedProductCallEdge]] = defaultdict(
            list
        )
        authority_by_symbol: dict[str, CompactProductAuthority] = {}
        for edge in self.root_edges:
            roots_by_authority[edge.authority.class_symbol].append(edge)
            authority_by_symbol[edge.authority.class_symbol] = edge.authority
        for authority_symbol, root_edges in roots_by_authority.items():
            authority = authority_by_symbol[authority_symbol]
            participant_symbols, forwarding_edges = self._reachable_participants(
                authority,
                tuple(root_edges),
            )
            seeds.append(
                _ParameterConveyorComponentSeed(
                    authority=authority,
                    participant_symbols=participant_symbols,
                    root_edges=tuple(root_edges),
                    forwarding_edges=forwarding_edges,
                )
            )
        return tuple(seeds)

    def _reachable_participants(
        self,
        authority: CompactProductAuthority,
        root_edges: tuple[ConstructedProductCallEdge, ...],
    ) -> tuple[frozenset[str], tuple[ForwardedProductCallEdge, ...]]:
        participants = {edge.callee_symbol for edge in root_edges}
        pending: deque[ParameterConveyorCallEdge] = deque(root_edges)
        visited_mappings: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        edges: dict[
            tuple[CompactFunctionCallIdentity, tuple[tuple[str, str], ...]],
            ForwardedProductCallEdge,
        ] = {}
        while pending:
            incoming_edge = pending.popleft()
            caller_symbol = incoming_edge.callee_symbol
            caller_mapping = incoming_edge.field_mapping
            mapping_identity = caller_symbol, caller_mapping
            if mapping_identity in visited_mappings:
                continue
            visited_mappings.add(mapping_identity)
            caller_parameters_by_field = dict(caller_mapping)
            first_field = authority.field_names[0]
            candidate_edges = self.calls_by_value_projection.get(
                (
                    caller_symbol,
                    LexicalValueReference(caller_parameters_by_field[first_field]),
                ),
                (),
            )
            for call_edge in candidate_edges:
                edge = self._forwarded_edge(
                    caller_symbol,
                    authority,
                    call_edge,
                    caller_parameters_by_field,
                )
                if edge is None:
                    continue
                edge_mapping = edge.field_mapping
                edges[(self.call_identity(call_edge), edge_mapping)] = edge
                if edge.callee_symbol not in participants:
                    participants.add(edge.callee_symbol)
                pending.append(edge)
        return frozenset(participants), tuple(edges.values())

    def _constructed_edge(
        self,
        construction: CompactResolvedProductConstruction,
        call_edge: CompactResolvedFunctionCall,
        construction_arguments: dict[str, LexicalValueReference | None],
    ) -> ConstructedProductCallEdge | None:
        if call_edge.context.owner_symbol != construction.context.owner_symbol:
            return None
        field_bindings = self._field_bindings(
            construction.authority,
            call_edge,
            {
                field_name: frozenset(
                    (
                        *(
                            self._reference_equivalents(
                                construction.context,
                                construction_arguments[field_name],
                                construction.call.position,
                            )
                            if construction_arguments[field_name] is not None
                            else ()
                        ),
                        LexicalValueReference(
                            construction.construction.result_binding.root_name,
                            (
                                *construction.construction.result_binding.attribute_path,
                                field_name,
                            ),
                        ),
                    )
                )
                for field_name in construction.authority.field_names
            },
        )
        if field_bindings is None:
            return None
        construction_position = construction.call.position
        call_position = call_edge.call.position
        protected_roots = {
            construction.construction.result_binding.root_name,
            *(
                reference.root_name
                for reference in construction_arguments.values()
                if reference is not None
            ),
        }
        intervening_mutation_roots = tuple(
            sorted(
                {
                    mutation.reference.root_name
                    for mutation in construction.context.flow.mutations
                    if self._mutation_reaches_protected_root(
                        construction.context,
                        mutation,
                        protected_roots,
                    )
                    and construction_position.dominates(mutation.position)
                    and mutation.position.dominates(call_position)
                    and not (
                        mutation.kind is CompactMutationKind.ASSIGNMENT
                        and mutation.reference
                        == construction.construction.result_binding
                        and mutation.position.branch_path
                        == construction_position.branch_path
                        and mutation.position.statement_index
                        == construction_position.statement_index
                    )
                }
            )
        )
        return ConstructedProductCallEdge(
            construction=construction,
            resolved_call=call_edge,
            field_bindings=field_bindings,
            carrier_binding_is_local=(
                not construction.construction.result_binding.attribute_path
            ),
            carrier_binding_dominates_call=construction_position.dominates(
                call_position
            ),
            intervening_mutation_roots=intervening_mutation_roots,
        )

    def _forwarded_edge(
        self,
        caller_symbol: str,
        authority: CompactProductAuthority,
        call_edge: CompactResolvedFunctionCall,
        caller_parameters_by_field: dict[str, str],
    ) -> ForwardedProductCallEdge | None:
        bindings = self._field_bindings(
            authority,
            call_edge,
            {
                field_name: frozenset(
                    (LexicalValueReference(caller_parameters_by_field[field_name]),)
                )
                for field_name in authority.field_names
            },
        )
        if bindings is None:
            return None
        return ForwardedProductCallEdge(
            caller_symbol=caller_symbol,
            product_authority=authority,
            resolved_call=call_edge,
            field_bindings=bindings,
        )

    def _field_bindings(
        self,
        authority: CompactProductAuthority,
        call_edge: CompactResolvedFunctionCall,
        expected_references_by_field: dict[
            str,
            frozenset[LexicalValueReference],
        ],
    ) -> tuple[ParameterConveyorFieldBinding, ...] | None:
        simple_arguments = self.simple_bound_arguments_by_call.get(
            self.call_identity(call_edge),
            {},
        )
        argument_origins = self.bound_argument_origins_by_call.get(
            self.call_identity(call_edge),
            {},
        )
        bindings: list[ParameterConveyorFieldBinding] = []
        for field_name in authority.field_names:
            matches = tuple(
                (parameter_name, value_reference)
                for parameter_name, value_reference in simple_arguments.items()
                if expected_references_by_field[field_name]
                & frozenset(
                    reference
                    for reference in (
                        value_reference,
                        argument_origins[parameter_name].exact_origin,
                    )
                    if reference is not None
                )
            )
            if len(matches) != 1:
                return None
            parameter_name, value_reference = matches[0]
            bindings.append(
                ParameterConveyorFieldBinding(
                    field_name=field_name,
                    parameter_name=parameter_name,
                    value_reference=value_reference,
                )
            )
        return tuple(bindings)

    @staticmethod
    def _reference_equivalents(
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
    ) -> tuple[LexicalValueReference, ...]:
        exact_origin = context.flow.value_origin_for(
            reference,
            use_position,
        ).exact_origin
        return tuple(
            dict.fromkeys(
                candidate
                for candidate in (reference, exact_origin)
                if candidate is not None
            )
        )

    @staticmethod
    def _mutation_reaches_protected_root(
        context: CompactProductFlowContext,
        mutation: CompactLexicalMutation,
        protected_roots: set[str],
    ) -> bool:
        origin_resolution = context.flow.value_origin_for(
            mutation.reference,
            mutation.position,
        )
        return bool(
            mutation.reference.root_name in protected_roots
            or any(
                origin.root_name in protected_roots
                for origin in origin_resolution.possible_origins
            )
        )

    @staticmethod
    def _construction_arguments(
        construction: CompactResolvedProductConstruction,
    ) -> dict[str, LexicalValueReference | None]:
        return {
            argument.name: argument.value.lexical_reference
            for argument in construction.construction.field_arguments
            if argument.name is not None
        }

    @staticmethod
    def _simple_bound_arguments(
        edge: CompactResolvedFunctionCall,
    ) -> dict[str, LexicalValueReference]:
        binding = edge.call.bind_to(edge.callee)
        if not binding.is_exact:
            return {}
        arguments: dict[str, LexicalValueReference] = {}
        for parameter in edge.callee.call_signature.parameters:
            bound_argument = binding.argument_for(parameter.name)
            if bound_argument is None or len(bound_argument.values) != 1:
                continue
            reference = bound_argument.values[0].lexical_reference
            if reference is not None:
                arguments[parameter.name] = reference
        return arguments

    @staticmethod
    def call_identity(
        edge: CompactResolvedFunctionCall,
    ) -> CompactFunctionCallIdentity:
        return CompactFunctionCallIdentity.from_resolution(edge)

    def _assessed_component(
        self,
        seed: _ParameterConveyorComponentSeed,
        authority_symbols_by_participant: dict[str, set[str]],
    ) -> ClosedParameterConveyorComponent:
        participants = tuple(
            ParameterConveyorParticipant(
                declaration=self.repository.function_declarations_by_symbol[symbol],
                context=self.repository.flow_contexts_by_owner_symbol[symbol],
            )
            for symbol in sorted(seed.participant_symbols)
        )
        edges: tuple[ParameterConveyorCallEdge, ...] = (
            *seed.root_edges,
            *seed.forwarding_edges,
        )
        component_call_ids = frozenset(
            self.call_identity(edge.resolved_call) for edge in edges
        )
        participant_symbols = frozenset(seed.participant_symbols)
        field_mappings_by_participant: dict[
            str,
            set[tuple[tuple[str, str], ...]],
        ] = defaultdict(set)
        for edge in edges:
            field_mappings_by_participant[edge.callee_symbol].add(edge.field_mapping)
        exact_field_mapping_by_participant = {
            participant_symbol: next(iter(mappings))
            for participant_symbol, mappings in field_mappings_by_participant.items()
            if len(mappings) == 1
        }
        callable_component = self.repository.callable_component_authority_proof(
            {
                participant_symbol: frozenset(
                    parameter_name
                    for _field_name, parameter_name in (
                        exact_field_mapping_by_participant.get(participant_symbol, ())
                    )
                )
                for participant_symbol in participant_symbols
            },
            component_call_ids,
        )
        open_alias_call_ids, unresolved_product_call_ids = (
            self._participant_product_call_hazards(
                participants,
                component_call_ids,
                exact_field_mapping_by_participant,
            )
        )
        root_open_alias_ids, unresolved_root_call_ids = self._root_call_hazards(
            seed.authority,
            seed.root_edges,
        )
        open_alias_call_ids.update(root_open_alias_ids)
        unresolved_product_call_ids.update(unresolved_root_call_ids)
        incomplete_consumption_symbols = {
            participant.symbol
            for participant in participants
            if participant.symbol not in exact_field_mapping_by_participant
            or not frozenset(
                parameter_name
                for _field_name, parameter_name in exact_field_mapping_by_participant[
                    participant.symbol
                ]
            ).issubset(participant.context.flow.loaded_value_root_names)
        }
        mutated_binding_symbols = {
            participant.symbol
            for participant in participants
            if any(
                mutation.reference.root_name
                in {
                    parameter_name
                    for _field_name, parameter_name in exact_field_mapping_by_participant.get(
                        participant.symbol,
                        (),
                    )
                }
                for mutation in participant.context.flow.mutations
            )
        }
        mutated_binding_symbols.update(
            edge.construction.context.owner_symbol
            for edge in seed.root_edges
            if edge.intervening_mutation_roots
        )
        authority_symbols = tuple(
            sorted(
                {
                    authority_symbol
                    for participant_symbol in participant_symbols
                    for authority_symbol in authority_symbols_by_participant[
                        participant_symbol
                    ]
                }
            )
        )
        compression_delta = (len(seed.authority.field_names) - 1) * (
            len(participants) + len(edges)
        )
        root_edges_by_call: dict[
            CompactFunctionCallIdentity,
            list[ConstructedProductCallEdge],
        ] = defaultdict(list)
        for root_edge in seed.root_edges:
            root_edges_by_call[self.call_identity(root_edge.resolved_call)].append(
                root_edge
            )
        proof = ClosedParameterConveyorAuthorityProof(
            authority_symbols=authority_symbols,
            authority_field_names=seed.authority.field_names,
            projected_field_names_by_edge=tuple(edge.field_names for edge in edges),
            non_injective_edge_ids=tuple(
                self._edge_display_id(edge)
                for edge in edges
                if len({binding.parameter_name for binding in edge.field_bindings})
                != len(edge.field_bindings)
            ),
            ambiguous_root_call_ids=tuple(
                self._edge_display_id(root_edges[0])
                for root_edges in root_edges_by_call.values()
                if len(root_edges) != 1
            ),
            incompletely_consuming_participant_symbols=tuple(
                sorted(incomplete_consumption_symbols)
            ),
            open_value_alias_call_ids=tuple(sorted(open_alias_call_ids)),
            unresolved_complete_product_call_ids=tuple(
                sorted(unresolved_product_call_ids)
            ),
            conflicting_mapping_symbols=tuple(
                sorted(
                    participant_symbol
                    for participant_symbol in participant_symbols
                    if len(authority_symbols_by_participant[participant_symbol]) > 1
                    or len(field_mappings_by_participant[participant_symbol]) > 1
                )
            ),
            non_dominating_root_symbols=tuple(
                sorted(
                    {
                        edge.construction.context.owner_symbol
                        for edge in seed.root_edges
                        if not edge.carrier_binding_is_local
                        or not edge.carrier_binding_dominates_call
                    }
                )
            ),
            mutated_binding_symbols=tuple(sorted(mutated_binding_symbols)),
            observed_root_carrier_symbols=tuple(
                sorted(
                    {
                        edge.construction.context.owner_symbol
                        for edge in seed.root_edges
                        if not edge.carrier_binding_is_unobserved
                    }
                )
            ),
            reevaluated_source_expression_symbols=tuple(
                sorted(
                    {
                        edge.construction.context.owner_symbol
                        for edge in seed.root_edges
                        if not edge.source_values_are_stable_local_loads
                    }
                )
            ),
            callable_component=callable_component,
            batch_compression_delta=compression_delta,
        )
        return ClosedParameterConveyorComponent(
            authority=seed.authority,
            participants=participants,
            root_edges=seed.root_edges,
            forwarding_edges=seed.forwarding_edges,
            proof=proof,
        )

    def _participant_product_call_hazards(
        self,
        participants: tuple[ParameterConveyorParticipant, ...],
        component_call_ids: frozenset[CompactFunctionCallIdentity],
        field_mapping_by_participant: dict[str, tuple[tuple[str, str], ...]],
    ) -> tuple[set[str], set[str]]:
        participant_symbols = frozenset(
            participant.symbol for participant in participants
        )
        open_alias_call_ids: set[str] = set()
        unresolved_call_ids: set[str] = set()
        for resolution in self.repository.function_call_resolutions:
            if resolution.context.owner_symbol not in participant_symbols:
                continue
            caller_mapping = field_mapping_by_participant.get(
                resolution.context.owner_symbol
            )
            if caller_mapping is None:
                continue
            product_projection = ParameterConveyorProductProjection.from_field_mapping(
                caller_mapping
            )
            identity = self._raw_call_identity(resolution.context, resolution.call)
            if identity in component_call_ids:
                continue
            self._record_call_hazard(
                product_projection,
                resolution,
                open_alias_call_ids,
                unresolved_call_ids,
                source_alias_is_open=False,
            )
        return open_alias_call_ids, unresolved_call_ids

    def _root_call_hazards(
        self,
        authority: CompactProductAuthority,
        exact_root_edges: tuple[ConstructedProductCallEdge, ...],
    ) -> tuple[set[str], set[str]]:
        exact_root_identities = frozenset(
            (
                self.call_identity(edge.resolved_call),
                edge.construction.call.position,
            )
            for edge in exact_root_edges
        )
        open_alias_call_ids: set[str] = set()
        unresolved_call_ids: set[str] = set()
        for construction in self.repository.resolved_product_constructions:
            if construction.authority.class_symbol != authority.class_symbol:
                continue
            construction_arguments = self._construction_arguments(construction)
            if frozenset(construction_arguments) != frozenset(authority.field_names):
                continue
            construction_origin_resolutions = {
                field_name: construction.context.flow.value_origin_for(
                    reference,
                    construction.call.position,
                )
                for field_name in authority.field_names
                if (reference := construction_arguments[field_name]) is not None
            }
            if len(construction_origin_resolutions) != len(authority.field_names):
                continue
            product_projection = ParameterConveyorProductProjection(
                tuple(
                    (
                        field_name,
                        frozenset(
                            (
                                construction_arguments[field_name],
                                *origin_resolution.possible_origins,
                            )
                        ),
                    )
                    for field_name, origin_resolution in construction_origin_resolutions.items()
                )
            )
            construction_alias_is_open = any(
                resolution.exact_origin is None
                for resolution in construction_origin_resolutions.values()
            )
            for resolution in self.repository.function_call_resolutions:
                if (
                    resolution.context.owner_symbol != construction.context.owner_symbol
                    or not construction.call.position.dominates(
                        resolution.call.position
                    )
                ):
                    continue
                identity = self._raw_call_identity(
                    resolution.context,
                    resolution.call,
                )
                if (identity, construction.call.position) in exact_root_identities:
                    continue
                self._record_call_hazard(
                    product_projection,
                    resolution,
                    open_alias_call_ids,
                    unresolved_call_ids,
                    source_alias_is_open=construction_alias_is_open,
                )
        return open_alias_call_ids, unresolved_call_ids

    @staticmethod
    def _record_call_hazard(
        product_projection: ParameterConveyorProductProjection,
        resolution: CompactFunctionCallResolution,
        open_alias_call_ids: set[str],
        unresolved_call_ids: set[str],
        *,
        source_alias_is_open: bool,
    ) -> None:
        if not product_projection.is_covered_by(resolution.possible_argument_origins):
            return
        display_id = (
            ClosedParameterConveyorComponentBuilder._call_resolution_display_id(
                resolution
            )
        )
        if source_alias_is_open or not product_projection.is_covered_by(
            resolution.exact_argument_origins
        ):
            open_alias_call_ids.add(display_id)
        unresolved_call_ids.add(display_id)

    @staticmethod
    def _call_resolution_display_id(
        resolution: CompactFunctionCallResolution,
    ) -> str:
        return f"{resolution.context.file_path}:{resolution.call.line}"

    @staticmethod
    def _edge_display_id(edge: ParameterConveyorCallEdge) -> str:
        call = edge.resolved_call
        return f"{call.context.file_path}:{call.call.line}:{edge.callee_symbol}"

    @staticmethod
    def _raw_call_identity(
        context: CompactProductFlowContext,
        call: CompactFunctionCall,
    ) -> CompactFunctionCallIdentity:
        return CompactFunctionCallIdentity(context.owner_symbol, call.position)
