"""Closed-component proofs for replacing flat parameter conveyors with products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Callable, Self, TypeAlias

from .product_flow import (
    CompactFlowPosition,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactMutationKind,
    LexicalValueReference,
)
from .product_flow_authority import (
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
    def callee_symbol(self) -> str:
        return self.resolved_call.callee.identity.symbol

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(binding.field_name for binding in self.field_bindings)


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


@dataclass(frozen=True)
class ForwardedProductCallEdge(ParameterConveyorCallEdge):
    caller_symbol: str
    product_authority: CompactProductAuthority
    resolved_call: CompactResolvedFunctionCall
    field_bindings: tuple[ParameterConveyorFieldBinding, ...]

    @property
    def authority(self) -> CompactProductAuthority:
        return self.product_authority


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


def _has_incomplete_product_consumption(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.incompletely_consuming_participant_symbols)


def _has_unresolved_consumer(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.unresolved_consumer_symbols)


def _has_dynamic_call_target(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.dynamic_product_call_symbols)


def _has_incomplete_call_family(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.incomplete_call_family_symbols)


def _has_escaping_callable_reference(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.escaping_callable_symbols)


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


def _has_signature_semantics_hazard(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.signature_hazard_symbols)


def _has_open_public_boundary(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.open_boundary_symbols)


def _has_incomplete_method_family(
    proof: "ClosedParameterConveyorAuthorityProof",
) -> bool:
    return bool(proof.incomplete_method_family_symbols)


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
    DYNAMIC_CALL_TARGET = (
        "dynamic_call_target",
        "a participant forwards the complete product through a dynamic call target",
        _has_dynamic_call_target,
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
    SIGNATURE_SEMANTICS_HAZARD = (
        "signature_semantics_hazard",
        "a decorator or malformed implicit receiver makes signature binding uncertain",
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
    participant_symbols: tuple[str, ...]
    projected_field_names_by_edge: tuple[tuple[str, ...], ...]
    non_injective_edge_ids: tuple[str, ...]
    incompletely_consuming_participant_symbols: tuple[str, ...]
    unresolved_consumer_symbols: tuple[str, ...]
    dynamic_product_call_symbols: tuple[str, ...]
    incomplete_call_family_symbols: tuple[str, ...]
    escaping_callable_symbols: tuple[str, ...]
    conflicting_mapping_symbols: tuple[str, ...]
    non_dominating_root_symbols: tuple[str, ...]
    mutated_binding_symbols: tuple[str, ...]
    signature_hazard_symbols: tuple[str, ...]
    open_boundary_symbols: tuple[str, ...]
    incomplete_method_family_symbols: tuple[str, ...]
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


@dataclass(frozen=True)
class _ParameterConveyorComponentSeed:
    authority: CompactProductAuthority
    participant_symbols: frozenset[str]
    root_edges: tuple[ConstructedProductCallEdge, ...]
    forwarding_edges: tuple[ForwardedProductCallEdge, ...]


_CallIdentity: TypeAlias = tuple[str, int, CompactFlowPosition]
_RootEdgeIdentity: TypeAlias = tuple[str, _CallIdentity, CompactFlowPosition]
_FieldProjectionKey: TypeAlias = tuple[str, str, LexicalValueReference]


@dataclass(frozen=True)
class ClosedParameterConveyorComponentBuilder:
    """Build maximal components before evaluating any executable candidate."""

    repository: CompactProductFlowRepository

    @cached_property
    def simple_bound_arguments_by_call(
        self,
    ) -> dict[_CallIdentity, dict[str, LexicalValueReference]]:
        return {
            self.call_identity(edge): self._simple_bound_arguments(edge)
            for edge in self.repository.resolved_function_calls
        }

    @cached_property
    def calls_by_field_projection(
        self,
    ) -> dict[_FieldProjectionKey, tuple[CompactResolvedFunctionCall, ...]]:
        grouped: dict[_FieldProjectionKey, list[CompactResolvedFunctionCall]] = (
            defaultdict(list)
        )
        for edge in self.repository.resolved_function_calls:
            for parameter_name, reference in self.simple_bound_arguments_by_call.get(
                self.call_identity(edge), {}
            ).items():
                grouped[(edge.context.owner_symbol, parameter_name, reference)].append(
                    edge
                )
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
            candidate_edges = {
                self.call_identity(edge): edge
                for expected_reference in (source_reference, carrier_reference)
                for edge in self.calls_by_field_projection.get(
                    (
                        construction.context.owner_symbol,
                        first_field,
                        expected_reference,
                    ),
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
                frozenset(edge.callee_symbol for edge in root_edges),
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
        root_participant_symbols: frozenset[str],
    ) -> tuple[frozenset[str], tuple[ForwardedProductCallEdge, ...]]:
        participants = set(root_participant_symbols)
        pending = deque(root_participant_symbols)
        edges: dict[_CallIdentity, ForwardedProductCallEdge] = {}
        while pending:
            caller_symbol = pending.popleft()
            first_field = authority.field_names[0]
            candidate_edges = self.calls_by_field_projection.get(
                (
                    caller_symbol,
                    first_field,
                    LexicalValueReference(first_field),
                ),
                (),
            )
            for call_edge in candidate_edges:
                edge = self._forwarded_edge(caller_symbol, authority, call_edge)
                if edge is None:
                    continue
                edges[self.call_identity(call_edge)] = edge
                if edge.callee_symbol not in participants:
                    participants.add(edge.callee_symbol)
                    pending.append(edge.callee_symbol)
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
                    reference
                    for reference in (
                        construction_arguments[field_name],
                        LexicalValueReference(
                            construction.construction.result_binding.root_name,
                            (
                                *construction.construction.result_binding.attribute_path,
                                field_name,
                            ),
                        ),
                    )
                    if reference is not None
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
                    if mutation.reference.root_name in protected_roots
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
    ) -> ForwardedProductCallEdge | None:
        bindings = self._field_bindings(
            authority,
            call_edge,
            {
                field_name: frozenset((LexicalValueReference(field_name),))
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
        bindings: list[ParameterConveyorFieldBinding] = []
        for field_name in authority.field_names:
            value_reference = simple_arguments.get(field_name)
            if (
                value_reference is None
                or value_reference not in expected_references_by_field[field_name]
            ):
                return None
            bindings.append(
                ParameterConveyorFieldBinding(
                    field_name=field_name,
                    parameter_name=field_name,
                    value_reference=value_reference,
                )
            )
        return tuple(bindings)

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
    def call_identity(edge: CompactResolvedFunctionCall) -> _CallIdentity:
        return (
            edge.context.owner_symbol,
            edge.call.line,
            edge.call.position,
        )

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
        participant_simple_names = {
            symbol: symbol.rsplit(".", 1)[-1] for symbol in participant_symbols
        }
        resolved_call_ids = frozenset(
            self.call_identity(edge) for edge in self.repository.resolved_function_calls
        )
        unresolved_consumers = {
            participant_symbol
            for context in self.repository.flow_contexts
            for call in context.flow.calls
            if self._raw_call_identity(context, call) not in resolved_call_ids
            for participant_symbol, simple_name in participant_simple_names.items()
            if call.target.terminal_name == simple_name
        }
        incomplete_call_family_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols
            if any(
                self.call_identity(incoming) not in component_call_ids
                for incoming in self.repository.incoming_calls_for(participant_symbol)
            )
        }
        escaping_callable_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols
            if self.repository.callable_escapes_for(participant_symbol)
        }
        dynamic_product_call_symbols = {
            participant.symbol
            for participant in participants
            if self._has_dynamic_complete_product_call(
                participant.context,
                seed.authority.field_names,
            )
        }
        incomplete_consumption_symbols = {
            participant.symbol
            for participant in participants
            if not frozenset(seed.authority.field_names).issubset(
                participant.context.flow.loaded_value_root_names
            )
        }
        mutated_binding_symbols = {
            participant.symbol
            for participant in participants
            if any(
                mutation.reference.root_name in seed.authority.field_names
                for mutation in participant.context.flow.mutations
            )
        }
        mutated_binding_symbols.update(
            edge.construction.context.owner_symbol
            for edge in seed.root_edges
            if edge.intervening_mutation_roots
        )
        incomplete_method_family_symbols = self._incomplete_method_family_symbols(
            participant_symbols
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
        authority_module_name = self.repository.class_index.class_for(
            seed.authority.class_symbol
        ).module_name
        participant_module_names = {
            participant.declaration.identity.module_name for participant in participants
        }
        import_cost = len(participant_module_names - {authority_module_name})
        compression_delta = (len(seed.authority.field_names) - 1) * (
            len(participants) + len(edges)
        ) - import_cost
        proof = ClosedParameterConveyorAuthorityProof(
            authority_symbols=authority_symbols,
            authority_field_names=seed.authority.field_names,
            participant_symbols=tuple(sorted(participant_symbols)),
            projected_field_names_by_edge=tuple(edge.field_names for edge in edges),
            non_injective_edge_ids=tuple(
                self._edge_display_id(edge)
                for edge in edges
                if len({binding.parameter_name for binding in edge.field_bindings})
                != len(edge.field_bindings)
            ),
            incompletely_consuming_participant_symbols=tuple(
                sorted(incomplete_consumption_symbols)
            ),
            unresolved_consumer_symbols=tuple(sorted(unresolved_consumers)),
            dynamic_product_call_symbols=tuple(sorted(dynamic_product_call_symbols)),
            incomplete_call_family_symbols=tuple(
                sorted(incomplete_call_family_symbols)
            ),
            escaping_callable_symbols=tuple(sorted(escaping_callable_symbols)),
            conflicting_mapping_symbols=tuple(
                sorted(
                    participant_symbol
                    for participant_symbol in participant_symbols
                    if len(authority_symbols_by_participant[participant_symbol]) > 1
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
            signature_hazard_symbols=tuple(
                sorted(
                    participant.symbol
                    for participant in participants
                    if participant.declaration.signature_decorator_hazard
                    or participant.declaration.nominal_receiver_name is None
                    and participant.declaration.binding_kind.implicit_parameter_count
                )
            ),
            open_boundary_symbols=tuple(
                sorted(
                    participant.symbol
                    for participant in participants
                    if not self._is_repository_private(participant.declaration)
                )
            ),
            incomplete_method_family_symbols=tuple(
                sorted(incomplete_method_family_symbols)
            ),
            batch_compression_delta=compression_delta,
        )
        return ClosedParameterConveyorComponent(
            authority=seed.authority,
            participants=participants,
            root_edges=seed.root_edges,
            forwarding_edges=seed.forwarding_edges,
            proof=proof,
        )

    def _incomplete_method_family_symbols(
        self,
        participant_symbols: frozenset[str],
    ) -> set[str]:
        incomplete: set[str] = set()
        for participant_symbol in participant_symbols:
            declaration = self.repository.function_declarations_by_symbol[
                participant_symbol
            ]
            if declaration.owner_class_qualname is None:
                continue
            owner_symbol = (
                f"{declaration.identity.module_name}."
                f"{declaration.owner_class_qualname}"
            )
            method_name = declaration.identity.qualname.rsplit(".", 1)[-1]
            related_class_symbols = (
                *self.repository.class_index.ancestor_symbols(owner_symbol),
                *self.repository.class_index.descendant_symbols(owner_symbol),
            )
            related_declarations = {
                f"{class_symbol}.{method_name}"
                for class_symbol in related_class_symbols
                if f"{class_symbol}.{method_name}"
                in self.repository.function_declarations_by_symbol
            }
            if related_declarations - participant_symbols:
                incomplete.add(participant_symbol)
        return incomplete

    @staticmethod
    def _has_dynamic_complete_product_call(
        context: CompactProductFlowContext,
        field_names: tuple[str, ...],
    ) -> bool:
        expected_roots = frozenset(field_names)
        return any(
            call.target.terminal_name is None
            and expected_roots.issubset(
                {
                    reference.root_name
                    for argument in (
                        *(item.value for item in call.positional_arguments),
                        *(item.value for item in call.keyword_arguments),
                    )
                    if (reference := argument.lexical_reference) is not None
                }
            )
            for call in context.flow.calls
        )

    @staticmethod
    def _is_repository_private(declaration: CompactFunctionDeclaration) -> bool:
        name = declaration.identity.qualname.rsplit(".", 1)[-1]
        return name.startswith("_") and not name.startswith("__")

    @staticmethod
    def _edge_display_id(edge: ParameterConveyorCallEdge) -> str:
        call = edge.resolved_call
        return f"{call.context.file_path}:{call.call.line}:{edge.callee_symbol}"

    @staticmethod
    def _raw_call_identity(
        context: CompactProductFlowContext,
        call: CompactFunctionCall,
    ) -> _CallIdentity:
        return context.owner_symbol, call.line, call.position
