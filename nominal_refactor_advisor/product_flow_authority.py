"""Repository joins for product-flow facts and their nominal authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Callable, Self, TypeAlias, cast

from .ast_tools import CollectedFamily, ParsedModule
from .class_index import (
    CompactClassMemberDeclaration,
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactProductAuthority,
    CompactPublicNameExposure,
    CompactRepositoryPublicExposureIndex,
    build_compact_class_family_index,
)
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
)
from .product_flow import (
    CompactCallTargetReference,
    CompactCallableReferenceUse,
    CompactFlowPosition,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactFunctionFlow,
    CompactProductConstruction,
    CompactProductFlowModuleProjection,
    CompactProductFlowModuleProjectionFamily,
    CompactValueOriginResolution,
    CurrentClassMemberMethodReference,
    LexicalValueReference,
    compact_product_flow_projection,
)


@dataclass(frozen=True)
class CompactProductFlowContext:
    """One execution flow joined to its module and optional declaration."""

    module_name: str
    file_path: str
    flow: CompactFunctionFlow
    declaration: CompactFunctionDeclaration | None

    @property
    def owner_symbol(self) -> str:
        if self.flow.owner.kind.is_module_scope:
            return self.module_name
        return f"{self.module_name}.{self.flow.owner.qualname}"


@dataclass(frozen=True)
class CompactFunctionTargetResolution(ABC):
    """Nominal result of resolving one function-valued call target."""

    @property
    @abstractmethod
    def declaration(self) -> CompactFunctionDeclaration | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def possible_symbols(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedCompactFunctionTarget(CompactFunctionTargetResolution):
    """One target proven to have exactly one nominal function declaration."""

    resolved_declaration: CompactFunctionDeclaration

    @property
    def declaration(self) -> CompactFunctionDeclaration:
        return self.resolved_declaration

    @property
    def possible_symbols(self) -> tuple[str, ...]:
        return (self.resolved_declaration.identity.symbol,)


class CompactFunctionTargetResolutionViolation(StrEnum):
    """Typed reasons a call target lacks one closed nominal declaration."""

    DYNAMIC_BINDING = "dynamic_binding"
    MISSING_DECLARATION = "missing_declaration"
    AMBIGUOUS_DECLARATION = "ambiguous_declaration"
    INCOMPLETE_RECEIVER_FAMILY = "incomplete_receiver_family"
    UNSUPPORTED_RECEIVER = "unsupported_receiver"


@dataclass(frozen=True)
class OpenCompactFunctionTarget(CompactFunctionTargetResolution):
    """A target whose possible nominal identity is not uniquely closed."""

    candidate_symbols: tuple[str, ...]
    violation: CompactFunctionTargetResolutionViolation

    @property
    def declaration(self) -> None:
        return None

    @property
    def possible_symbols(self) -> tuple[str, ...]:
        return self.candidate_symbols


@dataclass(frozen=True)
class _CompactClassMemberResolution:
    candidates: tuple[tuple[CompactIndexedClass, CompactClassMemberDeclaration], ...]
    violation: CompactFunctionTargetResolutionViolation | None


class CompactFunctionCallResolution(ABC):
    """Complete call-resolution result retained for every projected call."""

    context: CompactProductFlowContext
    call: CompactFunctionCall

    @property
    @abstractmethod
    def target_resolution(self) -> CompactFunctionTargetResolution:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolved_call(self) -> "CompactResolvedFunctionCall | None":
        raise NotImplementedError

    @cached_property
    def argument_origin_resolutions(
        self,
    ) -> tuple[CompactValueOriginResolution, ...]:
        return tuple(
            self.context.flow.value_origin_for(
                reference,
                self.call.position,
            )
            for value in self.call.arguments.values
            if (reference := value.lexical_reference) is not None
        )

    @cached_property
    def possible_argument_origins(self) -> frozenset[LexicalValueReference]:
        return frozenset(
            origin
            for resolution in self.argument_origin_resolutions
            for origin in resolution.possible_origins
        )

    @cached_property
    def exact_argument_origins(self) -> frozenset[LexicalValueReference]:
        return frozenset(
            exact_origin
            for resolution in self.argument_origin_resolutions
            if (exact_origin := resolution.exact_origin) is not None
        )


@dataclass(frozen=True)
class CompactResolvedFunctionCall(CompactFunctionCallResolution):
    """One call edge whose callee has exactly one nominal declaration."""

    context: CompactProductFlowContext
    call: CompactFunctionCall
    callee: CompactFunctionDeclaration

    @property
    def target_resolution(self) -> ResolvedCompactFunctionTarget:
        return ResolvedCompactFunctionTarget(self.callee)

    @property
    def resolved_call(self) -> "CompactResolvedFunctionCall":
        return self


@dataclass(frozen=True)
class CompactOpenFunctionCall(CompactFunctionCallResolution):
    """One projected call whose target is not nominally closed."""

    context: CompactProductFlowContext
    call: CompactFunctionCall
    open_target_resolution: CompactFunctionTargetResolution

    @property
    def target_resolution(self) -> CompactFunctionTargetResolution:
        return self.open_target_resolution

    @property
    def resolved_call(self) -> None:
        return None


@dataclass(frozen=True)
class CompactResolvedCallableEscape:
    """One non-call callable use joined to its nominal declaration."""

    context: CompactProductFlowContext
    use: CompactCallableReferenceUse
    declaration: CompactFunctionDeclaration


@dataclass(frozen=True)
class CompactFunctionCallIdentity:
    """Repository-unique identity for one function-local call site."""

    caller_symbol: str
    position: CompactFlowPosition

    @classmethod
    def from_resolution(cls, resolution: CompactFunctionCallResolution) -> Self:
        return cls(resolution.context.owner_symbol, resolution.call.position)


@dataclass(frozen=True)
class CompactCallableComponentAuthorityProof:
    """Shared closure proof for an atomic callable-component rewrite."""

    participant_symbols: tuple[str, ...]
    missing_declaration_symbols: tuple[str, ...]
    unresolved_consumer_symbols: tuple[str, ...]
    incomplete_call_family_symbols: tuple[str, ...]
    escaping_callable_symbols: tuple[str, ...]
    signature_hazard_symbols: tuple[str, ...]
    open_boundary_symbols: tuple[str, ...]
    incomplete_method_family_symbols: tuple[str, ...]

    @property
    def is_closed(self) -> bool:
        return not any(
            (
                self.missing_declaration_symbols,
                self.unresolved_consumer_symbols,
                self.incomplete_call_family_symbols,
                self.escaping_callable_symbols,
                self.signature_hazard_symbols,
                self.open_boundary_symbols,
                self.incomplete_method_family_symbols,
            )
        )


@dataclass(frozen=True)
class CompactResolvedProductConstruction:
    """One explicit construction joined to its complete product authority."""

    context: CompactProductFlowContext
    call: CompactFunctionCall
    construction: CompactProductConstruction
    authority: CompactProductAuthority


class CompactProductRuntimeViolation(StrEnum):
    """Repository-flow evidence that opens an otherwise exact product class."""

    CLASS_REBINDING_OR_MEMBER_MUTATION = "class_rebinding_or_member_mutation"
    CLASS_OBJECT_ESCAPE = "class_object_escape"


CompactClassReferencePartCountProjection: TypeAlias = Callable[
    [LexicalValueReference],
    tuple[int, ...],
]


def _exact_class_reference_part_counts(
    reference: LexicalValueReference,
) -> tuple[int, ...]:
    return (len(reference.parts),)


def _member_prefix_class_reference_part_counts(
    reference: LexicalValueReference,
) -> tuple[int, ...]:
    return tuple(range(len(reference.parts), 0, -1))


class CompactProductClassReferenceUse(StrEnum):
    """Class-reference uses with member-prefix resolution owned by each leaf."""

    EXACT_CLASS_OBJECT = "exact_class_object", _exact_class_reference_part_counts
    MUTATION_TARGET = "mutation_target", _member_prefix_class_reference_part_counts

    part_count_projection: CompactClassReferencePartCountProjection

    def __new__(
        cls,
        value: str,
        part_count_projection: CompactClassReferencePartCountProjection,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.part_count_projection = part_count_projection
        return member

    def candidate_part_counts(
        self,
        reference: LexicalValueReference,
    ) -> tuple[int, ...]:
        return self.part_count_projection(reference)


@dataclass(frozen=True)
class CompactProductRuntimeFailure:
    """One runtime-opening product use retained at its exact flow source."""

    authority_symbol: str
    owner_symbol: str
    line: int
    violation: CompactProductRuntimeViolation


@dataclass(frozen=True)
class CompactProductFlowRepository:
    """Derived query authority over product flows and nominal class declarations."""

    product_projections: tuple[CompactProductFlowModuleProjection, ...]
    class_projections: tuple[CompactModuleClassProjection, ...]

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        """Derive both joined fact families from the same source snapshot."""

        return cls(
            tuple(compact_product_flow_projection(module) for module in modules),
            CompactModuleClassProjectionFamily.collect_modules(modules),
        )

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            tuple[object, ...],
        ],
    ) -> Self:
        """Recover the typed product-flow join declared by its fact families."""

        return cls(
            product_projections=cast(
                tuple[CompactProductFlowModuleProjection, ...],
                projections_by_family[CompactProductFlowModuleProjectionFamily],
            ),
            class_projections=cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
        )

    @classmethod
    def require(cls, context: object | None) -> Self:
        if not isinstance(context, cls):
            raise TypeError("compact product-flow repository is unavailable")
        return context

    @cached_property
    def class_index(self) -> CompactClassFamilyIndex:
        return build_compact_class_family_index(self.class_projections)

    @cached_property
    def class_resolver(self) -> CompactClassReferenceResolver:
        return CompactClassReferenceResolver.from_index(
            self.class_projections,
            self.class_index,
        )

    @cached_property
    def public_exposure_index(self) -> CompactRepositoryPublicExposureIndex:
        return CompactRepositoryPublicExposureIndex(self.class_projections)

    @cached_property
    def function_declaration_multiplicity(
        self,
    ) -> IdentityHandleMultiplicityProjection[str, CompactFunctionDeclaration]:
        return UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            (
                declaration
                for projection in self.product_projections
                for declaration in projection.function_declarations
            ),
            lambda declaration: declaration.identity.symbol,
        )

    @cached_property
    def function_declarations_by_symbol(
        self,
    ) -> dict[str, CompactFunctionDeclaration]:
        return self.function_declaration_multiplicity.unambiguous_declarations_by_handle

    @cached_property
    def ambiguous_function_declaration_symbols(self) -> frozenset[str]:
        return self.function_declaration_multiplicity.ambiguous_handles

    @cached_property
    def class_projections_by_module_name(
        self,
    ) -> dict[str, CompactModuleClassProjection]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.class_projections,
            lambda projection: projection.module_name,
        )

    @cached_property
    def flow_contexts(self) -> tuple[CompactProductFlowContext, ...]:
        contexts: list[CompactProductFlowContext] = []
        for projection in self.product_projections:
            declarations_by_qualname = {
                declaration.identity.qualname: declaration
                for declaration in projection.function_declarations
            }
            for flow in projection.flows:
                declaration = (
                    declarations_by_qualname.get(flow.owner.qualname)
                    if flow.owner.kind.is_function_scope
                    else None
                )
                contexts.append(
                    CompactProductFlowContext(
                        module_name=projection.module_name,
                        file_path=projection.file_path,
                        flow=flow,
                        declaration=declaration,
                    )
                )
        return tuple(contexts)

    @cached_property
    def module_flow_contexts(self) -> dict[str, CompactProductFlowContext]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                context
                for context in self.flow_contexts
                if context.flow.owner.kind.is_module_scope
            ),
            lambda context: context.module_name,
        )

    @cached_property
    def flow_contexts_by_owner_symbol(self) -> dict[str, CompactProductFlowContext]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.flow_contexts,
            lambda context: context.owner_symbol,
        )

    @cached_property
    def product_authorities_by_symbol(self) -> dict[str, CompactProductAuthority]:
        return {
            class_symbol: authority
            for class_symbol, resolution in (
                self.class_index.product_authority_resolutions_by_symbol.items()
            )
            if (authority := resolution.authority) is not None
            and len(authority.field_names) >= 2
            and class_symbol not in self.product_runtime_failures_by_authority_symbol
        }

    @cached_property
    def product_runtime_failures_by_authority_symbol(
        self,
    ) -> dict[str, tuple[CompactProductRuntimeFailure, ...]]:
        failures: dict[str, list[CompactProductRuntimeFailure]] = {}
        for context in self.flow_contexts:
            for mutation in context.flow.mutations:
                if mutation.kind.preserves_nominal_identity:
                    continue
                class_symbol = self._class_symbol_for_reference(
                    context,
                    mutation.reference,
                    line=mutation.line,
                    use=CompactProductClassReferenceUse.MUTATION_TARGET,
                )
                if class_symbol is not None:
                    failures.setdefault(class_symbol, []).append(
                        CompactProductRuntimeFailure(
                            class_symbol,
                            context.owner_symbol,
                            mutation.line,
                            CompactProductRuntimeViolation.CLASS_REBINDING_OR_MEMBER_MUTATION,
                        )
                    )
            for alias in context.flow.exact_value_aliases:
                class_symbol = self._class_symbol_for_reference(
                    context,
                    alias.source,
                    line=alias.binding_mutation.line,
                    use=CompactProductClassReferenceUse.EXACT_CLASS_OBJECT,
                )
                if class_symbol is not None:
                    failures.setdefault(class_symbol, []).append(
                        CompactProductRuntimeFailure(
                            class_symbol,
                            context.owner_symbol,
                            alias.binding_mutation.line,
                            CompactProductRuntimeViolation.CLASS_OBJECT_ESCAPE,
                        )
                    )
            for call in context.flow.calls:
                for value in call.arguments.values:
                    reference = value.lexical_reference
                    if reference is None:
                        continue
                    exact_origin = context.flow.value_origin_for(
                        reference,
                        call.position,
                    ).exact_origin
                    class_symbol = self._class_symbol_for_reference(
                        context,
                        exact_origin or reference,
                        line=call.line,
                        use=CompactProductClassReferenceUse.EXACT_CLASS_OBJECT,
                    )
                    if class_symbol is not None:
                        failures.setdefault(class_symbol, []).append(
                            CompactProductRuntimeFailure(
                                class_symbol,
                                context.owner_symbol,
                                call.line,
                                CompactProductRuntimeViolation.CLASS_OBJECT_ESCAPE,
                            )
                        )
        return {
            class_symbol: tuple(dict.fromkeys(class_failures))
            for class_symbol, class_failures in failures.items()
        }

    def _class_symbol_for_reference(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        *,
        line: int,
        use: CompactProductClassReferenceUse,
    ) -> str | None:
        """Resolve one exact class-object reference without crossing a local rebind."""

        if context.declaration is not None and reference.root_name in {
            parameter.name for parameter in context.declaration.signature.parameters
        }:
            return None
        if not context.flow.owner.kind.is_module_scope:
            if reference.root_name in context.flow.nonlocal_binding_names:
                return None
            if reference.root_name not in context.flow.global_binding_names and any(
                mutation.reference.root_name == reference.root_name
                and not mutation.reference.attribute_path
                for mutation in context.flow.mutations
            ):
                return None

        for part_count in use.candidate_part_counts(reference):
            class_symbol = self.class_resolver.symbol_for(
                module_name=context.module_name,
                reference_parts=reference.parts[:part_count],
                allow_unique_unqualified=False,
            )
            if class_symbol is None:
                continue
            indexed_class = self.class_index.class_for(class_symbol)
            if indexed_class is None:
                continue
            if (
                context.flow.owner.kind.is_module_scope
                and indexed_class.module_name == context.module_name
                and indexed_class.line >= line
            ):
                continue
            return class_symbol
        return None

    @cached_property
    def function_call_resolutions(self) -> tuple[CompactFunctionCallResolution, ...]:
        return tuple(
            self.resolve_function_call(context, call)
            for context in self.flow_contexts
            for call in context.flow.calls
        )

    @cached_property
    def resolved_function_calls(self) -> tuple[CompactResolvedFunctionCall, ...]:
        return tuple(
            resolved
            for resolution in self.function_call_resolutions
            if (resolved := resolution.resolved_call) is not None
        )

    @cached_property
    def callable_escapes(self) -> tuple[CompactResolvedCallableEscape, ...]:
        return tuple(
            resolved
            for context in self.flow_contexts
            for use in context.flow.callable_reference_uses
            if (resolved := self.resolve_callable_escape(context, use)) is not None
        )

    @cached_property
    def resolved_product_constructions(
        self,
    ) -> tuple[CompactResolvedProductConstruction, ...]:
        return tuple(
            resolved
            for context in self.flow_contexts
            for call in context.flow.calls
            if (resolved := self.resolve_product_construction(context, call))
            is not None
        )

    def resolve_function_call(
        self,
        context: CompactProductFlowContext,
        call: CompactFunctionCall,
    ) -> CompactFunctionCallResolution:
        target_resolution = self.resolve_function_target(
            context,
            call.target,
            call.position,
        )
        declaration = target_resolution.declaration
        if declaration is None:
            return CompactOpenFunctionCall(context, call, target_resolution)
        return CompactResolvedFunctionCall(context, call, declaration)

    def resolve_callable_escape(
        self,
        context: CompactProductFlowContext,
        use: CompactCallableReferenceUse,
    ) -> CompactResolvedCallableEscape | None:
        declaration = self.resolve_function_target(
            context,
            use.target,
            use.position,
        ).declaration
        if declaration is None:
            return None
        return CompactResolvedCallableEscape(context, use, declaration)

    def resolve_function_target(
        self,
        context: CompactProductFlowContext,
        target: CompactCallTargetReference,
        position: CompactFlowPosition,
    ) -> CompactFunctionTargetResolution:
        if isinstance(target, CurrentClassMemberMethodReference):
            return self._class_member_method_resolution(context, target, position)
        reference = target.lexical_reference
        if reference is None:
            candidates = context.flow.local_candidate_symbols(
                target,
                context.module_name,
            )
            if not candidates:
                return OpenCompactFunctionTarget(
                    (),
                    CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                )
            if len(candidates) != 1:
                return OpenCompactFunctionTarget(
                    candidates,
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                )
            return self._function_resolution_for_symbol(candidates[0])

        return self._lexical_function_target_resolution(
            context,
            reference,
            position,
        )

    def _class_member_method_resolution(
        self,
        context: CompactProductFlowContext,
        target: CurrentClassMemberMethodReference,
        position: CompactFlowPosition,
    ) -> CompactFunctionTargetResolution:
        candidate_symbols = target.local_candidate_symbols(context.module_name, ())
        if target.uses_runtime_class_lookup and self._lexical_binding_exists(
            context,
            LexicalValueReference("type"),
            position,
        ):
            return OpenCompactFunctionTarget(
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )
        owner_symbol = self.class_index.symbol_for(
            file_path=context.file_path,
            qualname=target.owner_class_qualname,
        )
        if owner_symbol is None:
            return OpenCompactFunctionTarget(
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
            )
        member_resolution = self._member_declarations_in_single_lineage(
            owner_symbol,
            target.member_name,
        )
        if member_resolution.violation is not None:
            return OpenCompactFunctionTarget(
                candidate_symbols,
                member_resolution.violation,
            )
        declaring_class, member_declaration = member_resolution.candidates[0]
        reference_parts = member_declaration.annotation_reference_parts
        if reference_parts is None:
            return OpenCompactFunctionTarget(
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        member_type_symbol = self.class_resolver.symbol_for(
            module_name=declaring_class.module_name,
            reference_parts=reference_parts,
            allow_unique_unqualified=False,
        )
        if member_type_symbol is None:
            return OpenCompactFunctionTarget(
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        return self._function_resolution_for_symbol(
            f"{member_type_symbol}.{target.method_name}"
        )

    def _member_declarations_in_single_lineage(
        self,
        owner_symbol: str,
        member_name: str,
    ) -> _CompactClassMemberResolution:
        current_symbol = owner_symbol
        while (current := self.class_index.class_for(current_symbol)) is not None:
            declarations = tuple(
                declaration
                for declaration in current.direct_member_declarations
                if declaration.name == member_name
            )
            if declarations:
                candidates = tuple(
                    (current, declaration) for declaration in declarations
                )
                return _CompactClassMemberResolution(
                    candidates,
                    (
                        None
                        if len(candidates) == 1
                        else CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
                    ),
                )
            if not current.base_resolution_is_complete:
                return _CompactClassMemberResolution(
                    (),
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            if len(current.resolved_base_symbols) > 1:
                return _CompactClassMemberResolution(
                    (),
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                )
            if not current.resolved_base_symbols:
                return _CompactClassMemberResolution(
                    (),
                    CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                )
            current_symbol = current.resolved_base_symbols[0]
        return _CompactClassMemberResolution(
            (),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _lexical_binding_exists(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        position: CompactFlowPosition,
    ) -> bool:
        for scope_qualname in context.flow.lexical_scope_qualnames:
            owner_symbol = (
                f"{context.module_name}.{scope_qualname}"
                if scope_qualname
                else context.module_name
            )
            scope_context = self.flow_contexts_by_owner_symbol.get(owner_symbol)
            if scope_context is None:
                continue
            if self._scope_binding_resolution(
                scope_context,
                reference,
                position
                if scope_context.owner_symbol == context.owner_symbol
                else None,
            ) is not None:
                return True
        return False

    def resolve_product_construction(
        self,
        context: CompactProductFlowContext,
        call: CompactFunctionCall,
    ) -> CompactResolvedProductConstruction | None:
        construction = call.product_construction()
        if construction is None or construction.target.lexical_reference is None:
            return None
        reference = construction.target.lexical_reference
        assert reference is not None
        if self._has_dynamic_local_binding(context, construction.target, call.position):
            return None
        class_symbol = self.class_resolver.symbol_for(
            module_name=context.module_name,
            reference_parts=reference.parts,
            allow_unique_unqualified=False,
        )
        if class_symbol is None:
            return None
        authority = self.product_authorities_by_symbol.get(class_symbol)
        if authority is None:
            return None
        return CompactResolvedProductConstruction(
            context,
            call,
            construction,
            authority,
        )

    def incoming_calls_for(
        self,
        function_symbol: str,
    ) -> tuple[CompactResolvedFunctionCall, ...]:
        return tuple(
            edge
            for edge in self.resolved_function_calls
            if edge.callee.identity.symbol == function_symbol
        )

    def declared_return_class_symbol_for(
        self,
        call: CompactResolvedFunctionCall,
    ) -> str | None:
        """Resolve one callee-owned nominal return annotation as a class."""

        reference_parts = call.callee.return_annotation_reference_parts
        if reference_parts is None:
            return None
        return self.class_resolver.symbol_for(
            module_name=call.callee.identity.module_name,
            reference_parts=reference_parts,
            allow_unique_unqualified=False,
        )

    def declared_bound_value_class_symbol(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition,
    ) -> str | None:
        """Resolve the declared class of one unchanged bound call result."""

        call = context.flow.bound_call_result_for(reference, use_position)
        if call is None:
            return None
        resolved_call = self.resolve_function_call(context, call).resolved_call
        return (
            None
            if resolved_call is None
            else self.declared_return_class_symbol_for(resolved_call)
        )

    def callable_escapes_for(
        self,
        function_symbol: str,
    ) -> tuple[CompactResolvedCallableEscape, ...]:
        return tuple(
            escape
            for escape in self.callable_escapes
            if escape.declaration.identity.symbol == function_symbol
        )

    def callable_boundary_exposure(
        self,
        declaration: CompactFunctionDeclaration,
    ) -> CompactPublicNameExposure:
        qualname = declaration.identity.qualname
        if declaration.owner_class_qualname is not None:
            binding_name = declaration.owner_class_qualname.split(".", 1)[0]
        elif "." not in qualname:
            binding_name = qualname
        else:
            return CompactPublicNameExposure.PRIVATE
        return self.public_exposure_index.exposure_for(
            declaration.identity.module_name,
            binding_name,
        )

    def callable_component_authority_proof(
        self,
        parameter_names_by_participant: Mapping[str, frozenset[str]],
        component_call_identities: frozenset[CompactFunctionCallIdentity],
    ) -> CompactCallableComponentAuthorityProof:
        """Prove the shared callable boundary of one atomic signature rewrite."""

        participant_symbols = frozenset(parameter_names_by_participant)
        missing_declaration_symbols = participant_symbols - (
            self.function_declarations_by_symbol.keys()
            & self.flow_contexts_by_owner_symbol.keys()
        )
        unresolved_consumer_symbols = {
            possible_symbol
            for resolution in self.function_call_resolutions
            if resolution.resolved_call is None
            for possible_symbol in resolution.target_resolution.possible_symbols
            if possible_symbol in participant_symbols
        }
        incomplete_call_family_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols
            if any(
                CompactFunctionCallIdentity.from_resolution(incoming)
                not in component_call_identities
                for incoming in self.incoming_calls_for(participant_symbol)
            )
        }
        escaping_callable_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols
            if self.callable_escapes_for(participant_symbol)
        }
        signature_hazard_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols - missing_declaration_symbols
            if not self._signature_is_closed_for_parameters(
                self.function_declarations_by_symbol[participant_symbol],
                self.flow_contexts_by_owner_symbol[participant_symbol],
                parameter_names_by_participant[participant_symbol],
            )
        }
        open_boundary_symbols = {
            participant_symbol
            for participant_symbol in participant_symbols - missing_declaration_symbols
            if self.callable_boundary_exposure(
                self.function_declarations_by_symbol[participant_symbol]
            ).blocks_closed_boundary
        }
        return CompactCallableComponentAuthorityProof(
            participant_symbols=tuple(sorted(participant_symbols)),
            missing_declaration_symbols=tuple(sorted(missing_declaration_symbols)),
            unresolved_consumer_symbols=tuple(sorted(unresolved_consumer_symbols)),
            incomplete_call_family_symbols=tuple(
                sorted(incomplete_call_family_symbols)
            ),
            escaping_callable_symbols=tuple(sorted(escaping_callable_symbols)),
            signature_hazard_symbols=tuple(sorted(signature_hazard_symbols)),
            open_boundary_symbols=tuple(sorted(open_boundary_symbols)),
            incomplete_method_family_symbols=tuple(
                sorted(self._incomplete_method_family_symbols(participant_symbols))
            ),
        )

    @staticmethod
    def _signature_is_closed_for_parameters(
        declaration: CompactFunctionDeclaration,
        context: CompactProductFlowContext,
        parameter_names: frozenset[str],
    ) -> bool:
        parameters_by_name = {
            parameter.name: parameter
            for parameter in declaration.call_signature.parameters
        }
        return (
            parameter_names <= parameters_by_name.keys()
            and not declaration.signature_decorator_hazard
            and not context.flow.local_signature_is_observed
            and not (
                declaration.nominal_receiver_name is None
                and declaration.binding_kind.implicit_parameter_count
            )
            and all(
                parameters_by_name[parameter_name].is_plain_required
                for parameter_name in parameter_names
            )
        )

    def _incomplete_method_family_symbols(
        self,
        participant_symbols: frozenset[str],
    ) -> set[str]:
        incomplete: set[str] = set()
        for participant_symbol in participant_symbols:
            declaration = self.function_declarations_by_symbol.get(
                participant_symbol
            )
            if declaration is None or declaration.owner_class_qualname is None:
                continue
            owner_symbol = (
                f"{declaration.identity.module_name}."
                f"{declaration.owner_class_qualname}"
            )
            method_name = declaration.identity.qualname.rsplit(".", 1)[-1]
            related_class_symbols = (
                *self.class_index.ancestor_symbols(owner_symbol),
                *self.class_index.descendant_symbols(owner_symbol),
            )
            related_declarations = {
                f"{class_symbol}.{method_name}"
                for class_symbol in related_class_symbols
                if f"{class_symbol}.{method_name}"
                in self.function_declarations_by_symbol
            }
            if related_declarations - participant_symbols:
                incomplete.add(participant_symbol)
        return incomplete

    def _lexical_function_target_resolution(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        position: CompactFlowPosition,
    ) -> CompactFunctionTargetResolution:
        possible_symbols: list[str] = []
        for scope_qualname in context.flow.lexical_scope_qualnames:
            owner_symbol = (
                f"{context.module_name}.{scope_qualname}"
                if scope_qualname
                else context.module_name
            )
            scope_context = self.flow_contexts_by_owner_symbol.get(owner_symbol)
            candidate_symbol = ".".join((owner_symbol, *reference.parts))
            possible_symbols.append(candidate_symbol)
            if scope_context is None:
                continue
            resolution = self._scope_binding_resolution(
                scope_context,
                reference,
                position
                if scope_context.owner_symbol == context.owner_symbol
                else None,
            )
            if resolution is not None:
                return resolution
        return OpenCompactFunctionTarget(
            tuple(dict.fromkeys(possible_symbols)),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _scope_binding_resolution(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
    ) -> CompactFunctionTargetResolution | None:
        root_name = reference.root_name
        class_projection = self.class_projections_by_module_name.get(
            context.module_name
        )
        if (
            context.flow.owner.kind.is_module_scope
            and class_projection is not None
            and class_projection.star_import_origins
            and not self.public_exposure_index.star_imports_exclude(
                context.module_name,
                root_name,
            )
        ):
            star_import_symbols = tuple(
                ".".join(
                    (
                        origin.module_name,
                        root_name,
                        *reference.attribute_path,
                    )
                )
                for origin in class_projection.star_import_origins
                if origin.module_name is not None
            )
            return OpenCompactFunctionTarget(
                tuple(
                    dict.fromkeys(
                        (
                            *star_import_symbols,
                            ".".join((context.owner_symbol, *reference.parts)),
                        )
                    )
                ),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )
        if context.declaration is not None and root_name in {
            parameter.name for parameter in context.declaration.signature.parameters
        }:
            return OpenCompactFunctionTarget(
                (".".join((context.owner_symbol, *reference.parts)),),
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )

        mutations = tuple(
            mutation
            for mutation in context.flow.mutations
            if mutation.reference.root_name == root_name
            and not mutation.reference.attribute_path
        )
        if not mutations:
            return None
        if any(mutation.position.branch_path for mutation in mutations):
            return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference),
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )

        if use_position is not None:
            dominating = tuple(
                mutation
                for mutation in mutations
                if mutation.position.dominates(use_position)
            )
            if not dominating:
                return OpenCompactFunctionTarget(
                    self._possible_binding_symbols(context, reference),
                    CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
                )
            binding = dominating[-1]
        elif context.flow.owner.kind.is_module_scope:
            binding = mutations[-1]
        elif len(mutations) == 1:
            binding = mutations[0]
        else:
            return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )

        if binding.kind.is_import_binding:
            alias_target = (
                None
                if class_projection is None
                else dict(class_projection.import_aliases).get(root_name)
            )
            if alias_target is None:
                return OpenCompactFunctionTarget(
                    self._possible_binding_symbols(context, reference),
                    CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                )
            return self._function_resolution_for_symbol(
                ".".join((alias_target, *reference.attribute_path))
            )

        if not binding.kind.is_definition_binding:
            return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference),
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )

        binding_symbol = f"{context.owner_symbol}.{root_name}"
        if reference.attribute_path:
            if self.class_index.class_for(binding_symbol) is None:
                return OpenCompactFunctionTarget(
                    (".".join((binding_symbol, *reference.attribute_path)),),
                    CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                )
            binding_symbol = ".".join((binding_symbol, *reference.attribute_path))
        return self._function_resolution_for_symbol(binding_symbol)

    def _possible_binding_symbols(
        self,
        context: CompactProductFlowContext,
        reference: LexicalValueReference,
    ) -> tuple[str, ...]:
        local_symbol = ".".join((context.owner_symbol, *reference.parts))
        class_projection = self.class_projections_by_module_name.get(
            context.module_name
        )
        alias_target = (
            None
            if class_projection is None
            else dict(class_projection.import_aliases).get(reference.root_name)
        )
        imported_symbol = (
            None
            if alias_target is None
            else ".".join((alias_target, *reference.attribute_path))
        )
        return tuple(
            dict.fromkeys(
                symbol
                for symbol in (imported_symbol, local_symbol)
                if symbol is not None
            )
        )

    def _function_resolution_for_symbol(
        self,
        symbol: str,
    ) -> CompactFunctionTargetResolution:
        declaration = self.function_declarations_by_symbol.get(symbol)
        if declaration is not None:
            return ResolvedCompactFunctionTarget(declaration)
        if symbol in self.ambiguous_function_declaration_symbols:
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )
        owner_symbol, separator, member_name = symbol.rpartition(".")
        if not separator:
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        owner = self.class_index.class_for(owner_symbol)
        if owner is None:
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        if owner.method_names.count(member_name) > 1:
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )
        return self._inherited_method_resolution(owner_symbol, member_name)

    def _inherited_method_resolution(
        self,
        owner_symbol: str,
        member_name: str,
    ) -> CompactFunctionTargetResolution:
        possible_symbols = [
            f"{class_symbol}.{member_name}"
            for class_symbol in (
                owner_symbol,
                *self.class_index.ancestor_symbols(owner_symbol),
            )
        ]
        current_symbol = owner_symbol
        while True:
            current = self.class_index.class_for(current_symbol)
            if current is None:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            if not current.base_resolution_is_complete:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            if len(current.resolved_base_symbols) > 1:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                )
            if not current.resolved_base_symbols:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                )
            current_symbol = current.resolved_base_symbols[0]
            candidate_symbol = f"{current_symbol}.{member_name}"
            declaration = self.function_declarations_by_symbol.get(candidate_symbol)
            if declaration is not None:
                return ResolvedCompactFunctionTarget(declaration)
            if candidate_symbol in self.ambiguous_function_declaration_symbols:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                )
            ancestor = self.class_index.class_for(current_symbol)
            if ancestor is not None and ancestor.method_names.count(member_name) > 1:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
                )

    def _has_dynamic_local_binding(
        self,
        context: CompactProductFlowContext,
        target: CompactCallTargetReference,
        position: CompactFlowPosition,
    ) -> bool:
        reference = target.lexical_reference
        if reference is None:
            return False
        root_name = reference.root_name
        class_projection = self.class_projections_by_module_name.get(
            context.module_name
        )
        if class_projection is not None and class_projection.star_import_origins:
            return True
        if context.declaration is not None and root_name in {
            parameter.name for parameter in context.declaration.signature.parameters
        }:
            return True
        local_mutations = tuple(
            mutation
            for mutation in context.flow.mutations
            if mutation.reference.root_name == root_name
        )
        if context.flow.owner.kind.is_function_scope:
            if any(
                not mutation.kind.is_definition_binding for mutation in local_mutations
            ):
                return True
            if local_mutations and not any(
                mutation.position.dominates(position) for mutation in local_mutations
            ):
                return True
        module_context = self.module_flow_contexts.get(context.module_name)
        if module_context is None:
            return True
        relevant_module_mutations = tuple(
            mutation
            for mutation in module_context.flow.mutations
            if mutation.reference.root_name == root_name
        )
        if any(mutation.position.branch_path for mutation in relevant_module_mutations):
            return True
        if context.flow.owner.kind.is_module_scope and not any(
            mutation.kind.preserves_nominal_identity
            and mutation.position.dominates(position)
            for mutation in relevant_module_mutations
        ):
            return True
        return any(
            not mutation.kind.preserves_nominal_identity
            for mutation in relevant_module_mutations
        )
