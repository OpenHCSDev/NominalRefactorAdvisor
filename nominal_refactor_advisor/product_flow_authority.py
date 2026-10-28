"""Repository joins for product-flow facts and their nominal authorities."""

from __future__ import annotations

import ast
import builtins
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import (
    ClassVar,
    Self,
    TypeVar,
    cast,
)

from .annotation_semantics import NOMINAL_ANNOTATION_SOURCE_AUTHORITY
from .ast_tools import (
    CollectedFamily,
    ParsedModule,
    ParsedModuleSourceProjection,
    module_syntax_index,
)
from .call_binding import (
    CallValueT,
    CompactCallBinding,
    CompactFunctionSignature,
)
from .captured_reference import (
    CapturedReferenceResolution,
    CapturedReferenceViolation,
    OpenCapturedReference,
)
from .closed_world import ClosedWorldPremise
from .class_index import (
    CompactClassFamilyIndex,
    CompactClassMemberDeclaration,
    CompactClassReferenceResolver,
    CompactIndexedClass,
    CompactModuleClassHeader,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    CompactModuleStarImportOrigin,
    CompactProductAuthority,
    CompactPublicNameExposure,
    CompactRepositoryPublicExposureIndex,
    FunctionNominalParameterBindingAuthority,
    ModuleNominalBindingAuthority,
    RepositoryModuleBindingProof,
    build_compact_class_family_index,
)
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
)
from .descriptor_algebra import AliasProperty
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .native_declarations import NativeDeclaration
from .native_compilation import NativePythonCompilation
from .source_identity import source_path_text
from .product_flow import (
    CompactFunctionBindingKind,
    ReceiverAttributeCallTargetReference,
    BareCallTargetReference,
    CompactBindingResolverABC,
    CompactBindingVisit,
    CompactCallArguments,
    CompactCallTargetReference,
    CompactCallTargetResolverABC,
    CompactCallableReferenceUse,
    CompactDefinitionTarget,
    CompactDescriptorAccess,
    CompactExactValueAlias,
    CompactFlowContext as CompactFlowContext,
    CompactFlowPosition,
    CompactFlowValue,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactFunctionTargetResolutionViolation as CompactFunctionTargetResolutionViolation,
    CompactMutation,
    CompactMutationResolverABC,
    CompactProductConstruction,
    CompactProductFlowModuleProjection,
    CompactProductFlowModuleProjectionFamily,
    CompactValueOriginResolution,
    CompactValueUse,
    CompactVariableAnnotationTarget,
    CurrentClassMemberMethodReference,
    SourceProductFlowProjection,
)
from .source_geometry import SourceByteSpan
from .scan_cache import ScanCache
from .value_expression import LexicalValueReference

# Hooks through which a class or module answers any attribute name.
DYNAMIC_ATTRIBUTE_HOOKS = frozenset(("__getattr__", "__getattribute__"))
# Attributes whose access reaches something other than a function of that name.
INDIRECT_ATTRIBUTES = DYNAMIC_ATTRIBUTE_HOOKS | frozenset(("__call__", "__get__"))


@dataclass(frozen=True)
class CompactCallTargetResolution(ABC):
    """A lexical callable target, with function and construction projections."""

    @property
    def runtime_mutation_violation(self) -> CompactProductRuntimeViolation:
        """An unselected receiver remains an uncertainty, not a confirmed class write."""
        return CompactProductRuntimeViolation.UNRESOLVED_MUTATION_RECEIVER

    def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
        """Project potentially referenced participants through this target's bound."""
        return symbols.intersection(self.possible_symbols)

    @property
    def declaration(self) -> CompactFunctionDeclaration | None:
        """The function declaration, when this target denotes a function."""
        return None

    @property
    @abstractmethod
    def possible_symbols(self) -> tuple[str, ...]:
        raise NotImplementedError

    def through_alias(
        self, alias: CompactExactValueAlias, context: CompactFlowContext
    ) -> CompactCallTargetResolution:
        return self

    def through_descriptor(
        self, access: CompactDescriptorAccess
    ) -> CompactCallTargetResolution:
        return self

    def resolve_call(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> CompactFunctionCallResolution:
        return CompactOpenFunctionCall(context, call, self)

    def resolve_construction(
        self,
        repository: ProductFlowRepository,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CompactResolvedProductConstruction | None:
        return None


@dataclass(frozen=True)
class ResolvedCompactClassTarget(CompactCallTargetResolution):
    """A class definition selected by the ordinary lexical binding resolver."""

    resolved_declaration: CompactIndexedClass

    @property
    def runtime_mutation_violation(self) -> CompactProductRuntimeViolation:
        return CompactProductRuntimeViolation.CLASS_REBINDING_OR_MEMBER_MUTATION

    @property
    def possible_symbols(self) -> tuple[str, ...]:
        return (self.resolved_declaration.symbol,)

    def resolve_construction(
        self,
        repository: ProductFlowRepository,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CompactResolvedProductConstruction | None:
        construction = call.product_construction()
        authority = repository.product_authorities_by_symbol.get(
            self.resolved_declaration.symbol
        )
        if construction is None or authority is None:
            return None
        return CompactResolvedProductConstruction(
            context, call, construction, authority
        )


@dataclass(frozen=True)
class ResolvedCompactFunctionTarget(CompactCallTargetResolution):
    """One target proven to have exactly one nominal function declaration."""

    resolved_declaration: CompactFunctionDeclaration
    descriptor_access: CompactDescriptorAccess = CompactDescriptorAccess.DIRECT

    @property
    def declaration(self) -> CompactFunctionDeclaration:
        return self.resolved_declaration

    @property
    def possible_symbols(self) -> tuple[str, ...]:
        return (self.resolved_declaration.identity.symbol,)

    def through_descriptor(
        self, access: CompactDescriptorAccess
    ) -> CompactCallTargetResolution:
        return replace(self, descriptor_access=access)

    def resolve_call(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> CompactFunctionCallResolution:
        return CompactResolvedFunctionCall(context, call, self)

    @property
    def call_signature(self) -> CompactFunctionSignature:
        signature = self.declaration.signature_for_access(self.descriptor_access)
        if signature is None:
            raise ValueError(
                "Descriptor access does not establish a callable signature"
            )
        return signature

    def bind_arguments(
        self, arguments: CompactCallArguments[CallValueT]
    ) -> CompactCallBinding[CallValueT]:
        return self.declaration.bind_call(
            arguments.positional,
            arguments.keywords,
            access=self.descriptor_access,
        )

    def through_alias(
        self, alias: CompactExactValueAlias, context: CompactFlowContext
    ) -> CompactCallTargetResolution:
        if self.resolved_declaration.preserves_alias_call_binding(
            alias, context.flow.owner, context.module_name
        ):
            return self
        return OpenCompactFunctionTarget(
            self.possible_symbols,
            CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
        )


@dataclass(frozen=True)
class OpenCompactFunctionTarget(CompactCallTargetResolution):
    """A target whose possible nominal identity is not uniquely closed."""

    candidate_symbols: tuple[str, ...]
    violation: CompactFunctionTargetResolutionViolation

    @property
    def possible_symbols(self) -> tuple[str, ...]:
        return self.candidate_symbols

    def through_alias(
        self, alias: CompactExactValueAlias, context: CompactFlowContext
    ) -> CompactCallTargetResolution:
        return replace(
            self,
            candidate_symbols=tuple(
                dict.fromkeys(
                    (
                        f"{context.owner_symbol}.{alias.target.root_name}",
                        *self.possible_symbols,
                    )
                )
            ),
        )


class UnboundedCompactFunctionTarget(OpenCompactFunctionTarget):
    """Receiver provenance cannot exclude any participant; observed names remain diagnostic."""

    def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
        return symbols


@dataclass(frozen=True)
class NamedAttributeCompactFunctionTarget(OpenCompactFunctionTarget):
    """Attribute `attribute_name` of a receiver whose class is not proved.

    Reading attribute `m` of an object yields what is bound as `m` on it, so
    the access reaches only functions declared under that name. A function
    bound under another name is reached only after it escapes as a value, which
    the escape obligation already holds open; a computed fetch (`getattr` with
    a computed name, `vars`, `__dict__`) is the dynamic fetch obligation's, and
    a scan with a class or module `__getattr__` keeps every such access
    unbounded (`ProductFlowRepository.attribute_target`).
    """

    attribute_name: str = ""

    def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
        return frozenset(
            symbol
            for symbol in symbols
            if symbol.rpartition(".")[2] == self.attribute_name
            or symbol in self.candidate_symbols
        )


@dataclass(frozen=True)
class AlternativeCompactFunctionTargets(CompactCallTargetResolution):
    """Unselected binding alternatives retaining each authority's candidate bound."""

    alternatives: tuple[CompactCallTargetResolution, ...]
    violation: CompactFunctionTargetResolutionViolation

    @cached_property
    def possible_symbols(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                symbol
                for alternative in self.alternatives
                for symbol in alternative.possible_symbols
            )
        )

    def candidate_symbols_within(self, symbols: frozenset[str]) -> frozenset[str]:
        return frozenset().union(
            *(
                alternative.candidate_symbols_within(symbols)
                for alternative in self.alternatives
            )
        )


_MemberFoldT = TypeVar("_MemberFoldT")


class _CompactClassMemberResolution(ABC):
    """A member name looked up through a class's MRO: one declaration, or why not."""

    @abstractmethod
    def fold(
        self,
        member: Callable[
            [CompactIndexedClass, CompactClassMemberDeclaration], _MemberFoldT
        ],
        violation: Callable[[CompactFunctionTargetResolutionViolation], _MemberFoldT],
    ) -> _MemberFoldT:
        raise NotImplementedError


@dataclass(frozen=True)
class _ResolvedClassMember(_CompactClassMemberResolution):
    declaring_class: CompactIndexedClass
    declaration: CompactClassMemberDeclaration

    def fold(
        self,
        member: Callable[
            [CompactIndexedClass, CompactClassMemberDeclaration], _MemberFoldT
        ],
        violation: Callable[[CompactFunctionTargetResolutionViolation], _MemberFoldT],
    ) -> _MemberFoldT:
        return member(self.declaring_class, self.declaration)


@dataclass(frozen=True)
class _ClassMemberViolation(_CompactClassMemberResolution):
    violation: CompactFunctionTargetResolutionViolation

    def fold(
        self,
        member: Callable[
            [CompactIndexedClass, CompactClassMemberDeclaration], _MemberFoldT
        ],
        violation: Callable[[CompactFunctionTargetResolutionViolation], _MemberFoldT],
    ) -> _MemberFoldT:
        return violation(self.violation)


class CompactFunctionCallResolution(ABC):
    """Complete call-resolution result retained for every projected call."""

    context: CompactFlowContext
    call: CompactFunctionCall

    @property
    @abstractmethod
    def target_resolution(self) -> CompactCallTargetResolution:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolved_call(self) -> CompactResolvedFunctionCall | None:
        raise NotImplementedError

    @cached_property
    def argument_origin_resolutions(
        self,
    ) -> tuple[CompactValueOriginResolution, ...]:
        return tuple(
            value.origin_in(self.context.flow) for value in self.call.arguments.values
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

    context: CompactFlowContext
    call: CompactFunctionCall
    resolved_target: ResolvedCompactFunctionTarget

    @cached_property
    def bound_value_uses(self) -> dict[str, CompactValueUse]:
        """Single supplied values selected by the existing binding result."""
        return {
            parameter.name: argument.values[0]
            for parameter in self.call_signature.parameters
            if (argument := self.binding.argument_for(parameter.name)) is not None
            and len(argument.values) == 1
        }

    @property
    def callee(self) -> CompactFunctionDeclaration:
        return self.resolved_target.declaration

    @property
    def target_resolution(self) -> ResolvedCompactFunctionTarget:
        return self.resolved_target

    @property
    def call_signature(self) -> CompactFunctionSignature:
        return self.resolved_target.call_signature

    @cached_property
    def binding(self) -> CompactCallBinding[CompactValueUse]:
        return self.resolved_target.bind_arguments(self.call.arguments)

    @property
    def resolved_call(self) -> CompactResolvedFunctionCall:
        return self


@dataclass(frozen=True)
class CompactOpenFunctionCall(CompactFunctionCallResolution):
    """One projected call without a resolved function declaration."""

    context: CompactFlowContext
    call: CompactFunctionCall
    open_target_resolution: CompactCallTargetResolution

    @property
    def target_resolution(self) -> CompactCallTargetResolution:
        return self.open_target_resolution

    @property
    def resolved_call(self) -> None:
        return None


@dataclass(frozen=True)
class CompactCallableEscape:
    """One non-call use retaining its complete target-resolution evidence."""

    context: CompactFlowContext
    use: CompactCallableReferenceUse
    target_resolution: CompactCallTargetResolution


@dataclass(frozen=True)
class CompactDynamicFetch:
    """One expression that can return a function without naming it as an attribute.

    `getattr(x, name)` with a computed name, `vars(x)`, `x.__dict__`,
    `inspect.getmembers(x)` and `operator.attrgetter`/`methodcaller` fetch any
    attribute of their receiver; `getattr(x, "m")` fetches attribute `m`.
    `attribute_name` is that name, or None for any name; `receiver_classes`
    the classes whose functions the receiver can hold, or None when the receiver
    is not proved. Only an attribute can be fetched: a method declared in a
    class body or a module's top-level function. A nested function is reached
    only after it escapes as a value, which the escape obligation holds.
    """

    context: CompactFlowContext
    line: int
    attribute_name: str | None
    receiver_classes: frozenset[str] | None

    def reaches(
        self,
        symbols: frozenset[str],
        attribute_owners: Mapping[str, str],
    ) -> frozenset[str]:
        return frozenset(
            symbol
            for symbol in symbols
            if symbol in attribute_owners
            and (
                self.attribute_name is None
                or symbol.rpartition(".")[2] == self.attribute_name
            )
            and (
                self.receiver_classes is None
                or attribute_owners[symbol] in self.receiver_classes
            )
        )

    @property
    def display_id(self) -> str:
        return f"{self.context.file_path}:{self.line}"


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
    """Derive closure obligations on demand from one repository snapshot.

    Boolean consumers can stop at a failed obligation. Each diagnostic remains
    independently available and retains the complete, sorted symbol set.
    """

    repository: ProductFlowRepository
    participant_parameters: tuple[tuple[str, frozenset[str]], ...]
    component_call_identities: frozenset[CompactFunctionCallIdentity]

    @cached_property
    def participant_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._participant_symbol_set))

    @cached_property
    def _participant_symbol_set(self) -> frozenset[str]:
        return frozenset(symbol for symbol, _ in self.participant_parameters)

    @cached_property
    def missing_declaration_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._participant_symbol_set
                - (
                    self.repository.function_declarations_by_symbol.keys()
                    & self.repository.flow_contexts_by_owner_symbol.keys()
                )
            )
        )

    @cached_property
    def unresolved_consumer_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    symbol
                    for resolution in self.repository.function_call_resolutions
                    if resolution.resolved_call is None
                    for symbol in resolution.target_resolution.candidate_symbols_within(
                        self._participant_symbol_set
                    )
                }
            )
        )

    @cached_property
    def incomplete_call_family_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                symbol
                for symbol in self._participant_symbol_set
                if any(
                    CompactFunctionCallIdentity.from_resolution(incoming)
                    not in self.component_call_identities
                    for incoming in self.repository.incoming_calls_for(symbol)
                )
            )
        )

    @cached_property
    def escaping_callable_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    symbol
                    for escape in self.repository.callable_escapes
                    for symbol in escape.target_resolution.candidate_symbols_within(
                        self._participant_symbol_set
                    )
                }
            )
        )

    @cached_property
    def dynamically_fetched_symbols(self) -> tuple[str, ...]:
        """Participants a computed fetch can return as a value, to be called anywhere."""

        return tuple(
            sorted(
                {
                    symbol
                    for fetch in self.repository.dynamic_fetches
                    for symbol in fetch.reaches(
                        self._participant_symbol_set,
                        self.repository.function_attribute_owners,
                    )
                }
            )
        )

    @cached_property
    def signature_hazard_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                symbol
                for symbol, parameters in self.participant_parameters
                if symbol not in self.missing_declaration_symbols
                and not self.repository._signature_is_closed_for_parameters(
                    self.repository.function_declarations_by_symbol[symbol],
                    self.repository.flow_contexts_by_owner_symbol[symbol],
                    parameters,
                )
            )
        )

    @cached_property
    def open_boundary_symbols(self) -> tuple[str, ...]:
        """Participants callable from outside the scan: public, in a project not declared closed."""

        return tuple(
            sorted(
                symbol
                for symbol in self._exposed_symbols
                if symbol not in self.closed_world_premises
            )
        )

    @cached_property
    def _exposed_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol
            for symbol in sorted(self._participant_symbol_set)
            if symbol not in self.missing_declaration_symbols
            and self.repository.callable_boundary_exposure(
                self.repository.function_declarations_by_symbol[symbol]
            ).blocks_closed_boundary
        )

    @cached_property
    def closed_world_premises(self) -> dict[str, ClosedWorldPremise]:
        """The declaration (file and digest) that closes each exposed participant to outside callers."""

        premises: dict[str, ClosedWorldPremise] = {}
        for symbol in self._exposed_symbols:
            context = self.repository.flow_contexts_by_owner_symbol.get(symbol)
            premise = (
                None
                if context is None
                else ClosedWorldPremise.declared_for(Path(context.file_path))
            )
            if premise is not None:
                premises[symbol] = premise
        return premises

    @cached_property
    def incomplete_method_family_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.repository._incomplete_method_family_symbols(
                    self._participant_symbol_set
                )
            )
        )

    @property
    def is_closed(self) -> bool:
        return not (
            self.missing_declaration_symbols
            or self.unresolved_consumer_symbols
            or self.incomplete_call_family_symbols
            or self.escaping_callable_symbols
            or self.dynamically_fetched_symbols
            or self.signature_hazard_symbols
            or self.open_boundary_symbols
            or self.incomplete_method_family_symbols
        )


@dataclass(frozen=True)
class CompactResolvedProductConstruction:
    """One explicit construction joined to its complete product authority."""

    context: CompactFlowContext
    call: CompactFunctionCall
    construction: CompactProductConstruction
    authority: CompactProductAuthority


class CompactProductRuntimeViolation(StrEnum):
    """Repository-flow evidence that opens an otherwise exact product class."""

    CLASS_REBINDING_OR_MEMBER_MUTATION = "class_rebinding_or_member_mutation"
    CLASS_OBJECT_ESCAPE = "class_object_escape"
    UNRESOLVED_MUTATION_RECEIVER = "unresolved_mutation_receiver"


@dataclass(frozen=True)
class CompactProductRuntimeFailure:
    """One retained source observation, shared by all affected product queries."""

    context: CompactFlowContext
    source_event: CompactMutation | CompactFunctionCall
    target_resolution: CompactCallTargetResolution
    violation: CompactProductRuntimeViolation

    owner_symbol = AliasProperty[str]("context.owner_symbol")
    line = AliasProperty[int]("source_event.line")


@dataclass(frozen=True, eq=False)
class CompactProductRuntimeFailureIndex(
    Mapping[str, tuple[CompactProductRuntimeFailure, ...]]
):
    """Lazy authority queries over shared source observations, without class fanout."""

    authority_candidates: Mapping[str, CompactProductAuthority]
    observations: tuple[CompactProductRuntimeFailure, ...]

    def _failures_for(self, symbol: str) -> Iterator[CompactProductRuntimeFailure]:
        if symbol in self.authority_candidates:
            candidate = frozenset((symbol,))
            for observation in self.observations:
                if observation.target_resolution.candidate_symbols_within(candidate):
                    yield observation

    def __contains__(self, symbol: object) -> bool:
        if symbol not in self.authority_candidates:
            return False
        return next(self._failures_for(cast(str, symbol)), None) is not None

    def __getitem__(self, symbol: str) -> tuple[CompactProductRuntimeFailure, ...]:
        failures = tuple(self._failures_for(symbol))
        if not failures:
            raise KeyError(symbol)
        return failures

    def __iter__(self) -> Iterator[str]:
        return (symbol for symbol in self.authority_candidates if symbol in self)

    def __len__(self) -> int:
        return sum(1 for _symbol in self)


@dataclass(frozen=True)
class ProductFlowRepository(
    CompactCallTargetResolverABC[CompactFlowContext, CompactCallTargetResolution],
    CompactMutationResolverABC[CompactFlowContext, CompactCallTargetResolution],
    CompactBindingResolverABC[CompactCallTargetResolution],
):
    """Derived query authority over product flows and nominal class declarations."""

    def iter_product_projections(self) -> Iterator[CompactProductFlowModuleProjection]:
        """Visit retained projections; source repositories supply their original demand."""
        yield from self.product_projections

    def iter_flow_contexts(self) -> Iterator[CompactFlowContext]:
        """Visit original contexts without materializing unrequested module graphs."""
        for projection in self.iter_product_projections():
            yield from projection.flow_contexts

    def iter_product_runtime_failures(
        self,
    ) -> Iterator[CompactProductRuntimeFailure]:
        """Generate original rejection evidence for eligibility and diagnostics."""
        candidates = self.declared_product_authorities_by_symbol
        if not candidates:
            return
        product_symbols = frozenset(candidates)
        for context in self.iter_flow_contexts():
            for mutation in context.flow.mutations:
                if mutation.kind.introduces_nominal_binding:
                    continue
                resolution = mutation.resolve(self, context)
                if resolution.candidate_symbols_within(product_symbols):
                    yield (
                        CompactProductRuntimeFailure(
                            context,
                            mutation,
                            resolution,
                            resolution.runtime_mutation_violation,
                        )
                    )
            for alias in context.flow.exact_value_aliases:
                declaration = self._class_for_reference(
                    context,
                    alias.source,
                    line=alias.binding_mutation.line,
                )
                if declaration is not None and declaration.symbol in candidates:
                    yield (
                        CompactProductRuntimeFailure(
                            context,
                            alias.binding_mutation,
                            ResolvedCompactClassTarget(declaration),
                            CompactProductRuntimeViolation.CLASS_OBJECT_ESCAPE,
                        )
                    )
            for call in context.flow.calls:
                escaped_symbols: set[str] = set()
                for value in call.arguments.values:
                    reference = value.lexical_reference
                    if reference is None:
                        continue
                    exact_origin = value.origin_in(context.flow).exact_origin
                    declaration = self._class_for_reference(
                        context,
                        exact_origin or reference,
                        line=call.line,
                    )
                    if (
                        declaration is not None
                        and declaration.symbol in candidates
                        and declaration.symbol not in escaped_symbols
                    ):
                        escaped_symbols.add(declaration.symbol)
                        yield (
                            CompactProductRuntimeFailure(
                                context,
                                call,
                                ResolvedCompactClassTarget(declaration),
                                CompactProductRuntimeViolation.CLASS_OBJECT_ESCAPE,
                            )
                        )

    def star_import_origins_for(
        self, module_name: str
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        projection = self.class_projections_by_module_name.get(module_name)
        return () if projection is None else projection.star_import_origins

    def product_projection_for_module(
        self, module_name: str
    ) -> CompactProductFlowModuleProjection | None:
        return self.product_projections_by_module_name.get(module_name)

    def function_declaration_multiplicity_for_symbol(
        self, symbol: str
    ) -> IdentityHandleMultiplicityProjection[str, CompactFunctionDeclaration]:
        return self.function_declaration_multiplicity

    def module_flow_context_for_name(
        self, module_name: str
    ) -> CompactFlowContext | None:
        return self.module_flow_contexts.get(module_name)

    def flow_context_for_symbol(self, symbol: str) -> CompactFlowContext | None:
        """Select one globally unambiguous source scope."""
        return self.flow_contexts_by_owner_symbol.get(symbol)

    def _deleted_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CompactCallTargetResolution:
        return UnboundedCompactFunctionTarget(
            (),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    @abstractmethod
    def captured_value(self, value: CompactFlowValue) -> CapturedReferenceResolution:
        """Require actual source/runtime evidence independently of lexical selection."""
        raise NotImplementedError

    @property
    @abstractmethod
    def product_projections(self) -> Sequence[CompactProductFlowModuleProjection]:
        raise NotImplementedError

    @property
    @abstractmethod
    def class_projections(self) -> Sequence[CompactModuleClassHeader]:
        raise NotImplementedError

    def _definition_body_context(
        self,
        context: CompactFlowContext,
        binding: CompactMutation[CompactDefinitionTarget],
    ) -> CompactFlowContext | None:
        """Select a body only through its unique retained parent definition event."""
        projection = self.product_projection_for_module(context.module_name)
        if projection is None or projection.file_path != context.file_path:
            return None
        source = projection.definition_sources_by_owner.get(binding.target.owner)
        if source is None or source[0] is not context or source[1] is not binding:
            return None
        return projection.flow_contexts_by_owner[binding.target.owner]

    @cached_property
    def product_projections_by_module_name(
        self,
    ) -> dict[str, CompactProductFlowModuleProjection]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.product_projections, lambda projection: projection.module_name
        )

    def _imported_name_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CompactCallTargetResolution:
        origin = binding.imported_origin
        assert origin is not None
        qualified_name = origin.qualified_name
        if qualified_name is None:
            return self._possible_binding_resolution(
                context,
                reference,
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
                pending,
            )
        return self._function_resolution_for_symbol(
            ".".join((qualified_name, *reference.attribute_path)),
            pending,
        )

    def _installed_alias_resolution(
        self,
        resolution: CompactCallTargetResolution,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
    ) -> CompactCallTargetResolution:
        return resolution.through_alias(alias, context)

    def _captured_alias_resolution(
        self,
        alias: CompactExactValueAlias,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CompactCallTargetResolution:
        return alias.source_use.resolve(
            self,
            context,
            pending_bindings=pending,
            attribute_path=reference.attribute_path,
        )

    def _cyclic_binding_resolution(
        self, pending: frozenset[CompactBindingVisit[CompactFlowContext]]
    ) -> CompactCallTargetResolution:
        return OpenCompactFunctionTarget(
            tuple(
                sorted(
                    {
                        f"{visit.context.owner_symbol}.{visit.mutation.target.bound_name}"
                        for visit in pending
                    }
                )
            ),
            CompactFunctionTargetResolutionViolation.CYCLIC_BINDING,
        )

    def _definition_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        binding: CompactMutation[CompactDefinitionTarget],
        pending_bindings: frozenset[CompactBindingVisit[CompactFlowContext]],
    ) -> CompactCallTargetResolution:
        binding_symbol = f"{context.owner_symbol}.{reference.root_name}"
        if reference.attribute_path:
            owner = self.class_index.class_for(binding_symbol)
            if owner is None or owner.line != binding.line:
                return OpenCompactFunctionTarget(
                    (".".join((binding_symbol, *reference.attribute_path)),),
                    CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                )
            if len(reference.attribute_path) == 1:
                return self._class_method_resolution(
                    owner,
                    reference.attribute_path[0],
                    pending_bindings,
                )
            class_context = self._definition_body_context(context, binding)
            if class_context is not None:
                resolution = self._scope_binding_resolution(
                    class_context,
                    LexicalValueReference(
                        reference.attribute_path[0], reference.attribute_path[1:]
                    ),
                    None,
                    pending_bindings,
                )
                if resolution is not None:
                    return resolution
            return OpenCompactFunctionTarget(
                (".".join((binding_symbol, *reference.attribute_path)),),
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        return binding.kind.resolve_definition(self, binding_symbol, binding)

    @cached_property
    def declared_product_authorities_by_symbol(
        self,
    ) -> dict[str, CompactProductAuthority]:
        """Declaration-proved product candidates before runtime obligations are applied."""
        return {
            symbol: authority
            for symbol, resolution in self.class_index.product_authority_resolutions_by_symbol.items()
            if (authority := resolution.authority) is not None
            and len(authority.field_names) >= 2
        }

    def _receiver_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation,
        receiver_use: CompactValueUse,
    ) -> CompactCallTargetResolution:
        """Bound a receiver only through its proved runtime definition result."""
        try:
            captured = self.captured_value(CompactFlowValue(context, receiver_use))
            return captured.resolve_definition(self)
        except ValueError:
            return UnboundedCompactFunctionTarget(
                (),
                CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
            )

    def _non_definition_resolution(self) -> CompactCallTargetResolution:
        return OpenCompactFunctionTarget(
            (),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _annotation_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation[CompactVariableAnnotationTarget],
    ) -> CompactCallTargetResolution:
        """Exclude only a proved compiler-owned annotation namespace write."""

        if context.flow.entry_annotation_namespace_is_active(mutation):
            return self._non_definition_resolution()
        return super()._annotation_mutation_resolution(context, mutation)

    def _binding_mutation_resolution(
        self,
        context: CompactFlowContext,
        mutation: CompactMutation,
        name: str,
    ) -> CompactCallTargetResolution:
        """A namespace write affects its declared name, not the previous object's contents."""
        declaration = self._class_for_reference(
            context,
            LexicalValueReference(name),
            line=mutation.line,
        )
        if declaration is None:
            return OpenCompactFunctionTarget(
                (),
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        return ResolvedCompactClassTarget(declaration)

    def _through_attribute_suffix(
        self,
        resolution: CompactCallTargetResolution,
        attribute_path: tuple[str, ...],
    ) -> CompactCallTargetResolution:
        if not attribute_path:
            return resolution
        return self.attribute_target(
            attribute_path[-1],
            (),
            CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
        )

    SELF_TYPES: ClassVar[frozenset[str]] = frozenset(
        ("Self", "typing.Self", "typing_extensions.Self")
    )

    def _returned_receiver_attribute_resolution(
        self,
        context: CompactFlowContext,
        target: ReceiverAttributeCallTargetReference,
    ) -> CompactCallTargetResolution:
        """An attribute of a call's result, resolved on the class the call proves it returns.

        `Cls.make(...)` returns an instance of exactly `Cls` when `make`, found
        first in `Cls`'s MRO, is a classmethod declared `-> Self`: `cls` is
        `Cls` there. The attribute is then looked up on `Cls` as an instance
        attribute. Anything less certain leaves the receiver unknown.
        """

        unknown = self._through_attribute_suffix(
            self._local_function_target_resolution(context, target),
            (target.attribute_name,),
        )
        owner = self._self_returning_receiver(context, target)
        if owner is None:
            return unknown
        resolution = self._class_method_resolution(owner, target.attribute_name)
        if resolution.declaration is None:
            return unknown
        return resolution.through_descriptor(CompactDescriptorAccess.INSTANCE)

    def _self_returning_receiver(
        self,
        context: CompactFlowContext,
        target: ReceiverAttributeCallTargetReference,
    ) -> CompactIndexedClass | None:
        receiver_call = target.receiver_call
        reference = None if receiver_call is None else receiver_call.lexical_reference
        if reference is None or not reference.attribute_path:
            return None
        owner = self._class_for_reference(
            context,
            LexicalValueReference(reference.root_name, reference.attribute_path[:-1]),
            line=target.receiver_line,
        )
        if owner is None:
            return None
        made = self._class_method_resolution(owner, reference.attribute_path[-1])
        declaration = made.declaration
        if (
            declaration is None
            or declaration.binding_kind is not CompactFunctionBindingKind.CLASS_METHOD
            or declaration.return_annotation_expression not in self.SELF_TYPES
        ):
            return None
        return owner

    @cached_property
    def dynamic_attribute_owners(self) -> tuple[str, ...]:
        """Classes and modules declaring `__getattr__` or `__getattribute__`.

        Such an owner answers any attribute name with any object, so an
        attribute access on an unproved receiver is bounded by nothing while one
        exists in the scan.
        """

        return tuple(
            sorted(
                symbol
                for symbol in self.function_declarations_by_symbol
                if symbol.rpartition(".")[2] in DYNAMIC_ATTRIBUTE_HOOKS
            )
        )

    @staticmethod
    def is_protocol_attribute(attribute_name: str) -> bool:
        """An attribute that returns or calls something not named by the access.

        `f.__call__` is `f` itself, and `__get__`, `__getattr__` and
        `__getattribute__` return whatever their owner chooses; any other
        attribute, dunder or not, is looked up by its name.
        """

        return attribute_name in INDIRECT_ATTRIBUTES

    def attribute_target(
        self,
        attribute_name: str,
        candidate_symbols: tuple[str, ...],
        violation: CompactFunctionTargetResolutionViolation,
    ) -> CompactCallTargetResolution:
        """An access to `attribute_name` on an unproved receiver, bounded by that name when sound."""

        if self.dynamic_attribute_owners or self.is_protocol_attribute(attribute_name):
            return UnboundedCompactFunctionTarget(candidate_symbols, violation)
        return NamedAttributeCompactFunctionTarget(
            candidate_symbols, violation, attribute_name
        )

    def _through_receiver_binding(
        self,
        context: CompactFlowContext,
        position: CompactFlowPosition,
        resolution: CompactCallTargetResolution,
    ) -> CompactCallTargetResolution:
        declaration = context.declaration
        names = {symbol.rpartition(".")[2] for symbol in resolution.possible_symbols}
        if declaration is None or declaration.nominal_receiver_name is None:
            return self._receiver_attribute_target(
                names,
                resolution.possible_symbols,
                CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
            )
        reference = LexicalValueReference(declaration.nominal_receiver_name)
        origin = context.flow.value_origin_for(reference, position)
        if origin.exact_origin != reference:
            return self._receiver_attribute_target(
                names,
                resolution.possible_symbols,
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )
        return resolution

    def _receiver_attribute_target(
        self,
        names: set[str],
        candidate_symbols: tuple[str, ...],
        violation: CompactFunctionTargetResolutionViolation,
    ) -> CompactCallTargetResolution:
        if len(names) != 1:
            return UnboundedCompactFunctionTarget(candidate_symbols, violation)
        (name,) = names
        return self.attribute_target(name, candidate_symbols, violation)

    def _selected_class_resolution(
        self, symbol: str, binding: CompactMutation
    ) -> CompactCallTargetResolution:
        declaration = self.class_index.class_for(symbol)
        if declaration is None:
            return OpenCompactFunctionTarget(
                (symbol,), CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
            )
        if declaration.line != binding.line:
            return OpenCompactFunctionTarget(
                (symbol,), CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        return ResolvedCompactClassTarget(declaration)

    def _local_function_target_resolution(
        self,
        context: CompactFlowContext,
        target: CompactCallTargetReference,
    ) -> CompactCallTargetResolution:
        """Resolve source-local candidates without rediscovering target syntax."""
        candidates = context.flow.local_candidate_symbols(target, context.module_name)
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
        return self._function_resolution_for_symbol(candidates[0]).through_descriptor(
            target.receiver_access(context.declaration)
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
    def class_projections_by_module_name(self) -> dict[str, CompactModuleClassHeader]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.class_projections,
            lambda projection: projection.module_name,
        )

    @cached_property
    def flow_contexts(self) -> tuple[CompactFlowContext, ...]:
        return tuple(self.iter_flow_contexts())

    @cached_property
    def module_flow_contexts(self) -> dict[str, CompactFlowContext]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                context
                for context in self.flow_contexts
                if context.flow.owner.kind.is_module_scope
            ),
            lambda context: context.module_name,
        )

    @cached_property
    def flow_contexts_by_owner_symbol(self) -> dict[str, CompactFlowContext]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.flow_contexts,
            lambda context: context.owner_symbol,
        )

    @cached_property
    @ScanCache.scope()
    def product_authorities_by_symbol(self) -> dict[str, CompactProductAuthority]:
        """Exhaust all evidence for survivors; stop once every candidate is rejected."""
        remaining = dict(self.declared_product_authorities_by_symbol)
        if remaining:
            for failure in self.iter_product_runtime_failures():
                for symbol in failure.target_resolution.candidate_symbols_within(
                    frozenset(remaining)
                ):
                    del remaining[symbol]
                if not remaining:
                    break
        return remaining

    @cached_property
    @ScanCache.scope()
    def product_runtime_failures_by_authority_symbol(
        self,
    ) -> CompactProductRuntimeFailureIndex:
        return CompactProductRuntimeFailureIndex(
            self.declared_product_authorities_by_symbol,
            tuple(self.iter_product_runtime_failures()),
        )

    def _class_for_reference(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        *,
        line: int,
    ) -> CompactIndexedClass | None:
        """Resolve one exact class-object reference without crossing a local rebind."""
        if context.declaration is not None and reference.root_name in {
            parameter.name for parameter in context.declaration.signature.parameters
        }:
            return None
        if not context.flow.owner.kind.is_module_scope:
            if reference.root_name in context.flow.nonlocal_binding_names:
                return None
            if (
                reference.root_name not in context.flow.global_binding_names
                and reference.root_name in context.flow.mutations_by_root_name
            ):
                return None
        symbol = self.class_resolver.symbol_for(
            module_name=context.module_name,
            reference_parts=reference.parts,
            allow_unique_unqualified=False,
        )
        if symbol is None:
            return None
        declaration = self.class_index.class_for(symbol)
        if declaration is None:
            return None
        if (
            context.flow.owner.kind.is_module_scope
            and declaration.module_name == context.module_name
            and declaration.line >= line
        ):
            return None
        return declaration

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

    # Builtins and standard calls that return attributes chosen by a value.
    COMPUTED_FETCH_BUILTINS: ClassVar[frozenset[str]] = frozenset(("getattr", "vars"))
    COMPUTED_FETCH_CALLS: ClassVar[frozenset[str]] = frozenset(
        ("getmembers", "getmembers_static", "attrgetter", "methodcaller")
    )
    NAMESPACE_ATTRIBUTE: ClassVar[str] = "__dict__"

    @cached_property
    def function_attribute_owners(self) -> dict[str, str]:
        """Each function bound as an attribute, with its owner: the class whose body or the module whose top level declares it."""

        owners: dict[str, str] = {}
        for symbol, declaration in self.function_declarations_by_symbol.items():
            module = declaration.identity.module_name
            qualname = declaration.identity.qualname
            owner_qualname, _dot, _name = qualname.rpartition(".")
            if not owner_qualname:
                owners[symbol] = module
            elif owner_qualname == declaration.owner_class_qualname:
                owners[symbol] = f"{module}.{owner_qualname}"
        return owners

    @cached_property
    def dynamic_fetches(self) -> tuple[CompactDynamicFetch, ...]:
        fetches: list[CompactDynamicFetch] = []
        for context in self.flow_contexts:
            for call in context.flow.calls:
                fetch = self._dynamic_fetch_call(context, call)
                if fetch is not None:
                    fetches.append(fetch)
            for use in context.flow.callable_reference_uses:
                reference = use.lexical_reference
                if (
                    reference is not None
                    and reference.attribute_path
                    and reference.attribute_path[-1] == self.NAMESPACE_ATTRIBUTE
                ):
                    fetches.append(
                        CompactDynamicFetch(
                            context,
                            use.line,
                            None,
                            self._receiver_classes(
                                context,
                                LexicalValueReference(
                                    reference.root_name, reference.attribute_path[:-1]
                                ),
                                use.line,
                                use.position,
                            ),
                        )
                    )
        return tuple(fetches)

    def _dynamic_fetch_call(
        self, context: CompactFlowContext, call: CompactFunctionCall
    ) -> CompactDynamicFetch | None:
        target = call.target
        name = target.terminal_name
        if name in self.COMPUTED_FETCH_CALLS:
            return CompactDynamicFetch(context, call.line, None, None)
        if (
            not isinstance(target, BareCallTargetReference)
            or name not in self.COMPUTED_FETCH_BUILTINS
            or self._lexical_binding_exists(
                context, LexicalValueReference(name), call.position
            )
            or not call.arguments.positional
        ):
            return None
        receiver = call.arguments.positional[0].value.lexical_reference
        receiver_classes = (
            None
            if receiver is None
            else self._receiver_classes(context, receiver, call.line, call.position)
        )
        attribute_name = None
        if len(call.arguments.positional) >= 2:
            attribute_name = call.arguments.positional[1].value.value.constant_string
        if self.dynamic_attribute_owners:
            attribute_name = None
        return CompactDynamicFetch(context, call.line, attribute_name, receiver_classes)

    def _receiver_classes(
        self,
        context: CompactFlowContext,
        receiver: LexicalValueReference,
        line: int,
        position: CompactFlowPosition,
    ) -> frozenset[str] | None:
        """The classes whose functions a fetched receiver can hold, or None when unproved.

        A name still holding its entry value is typed by its declaration: the
        method's own receiver (`self`, `cls`) by its class, a parameter by its
        annotation (`X`, `type[X]`, `X | None`, `Optional[X]`). Either can be
        an instance or class object of a subclass, so it holds the functions of
        the class family: the classes, their ancestors and descendants, and the
        descendants' ancestors. The method's own `self.x` is typed by the
        annotation its class declares for `x`, the same way. A local bound to
        one call's unchanged result is typed by the class the call constructs
        or its callee's declared return class, each as its family. A
        class object reference holds its class's and ancestors' functions.
        Anything else is unproved.
        """

        declaration = context.declaration
        if (
            declaration is not None
            and not receiver.attribute_path
            and context.flow.value_origin_for(receiver, position).exact_origin
            == receiver
        ):
            declared = self._declared_receiver_classes(context, receiver.root_name)
            if declared is not None:
                return self._class_families(declared)
        member = self._own_member_classes(context, receiver)
        if member is not None:
            return self._class_families(member)
        if not receiver.attribute_path:
            produced = self._bound_result_classes(context, receiver, position)
            if produced is not None:
                return produced
        found = self._class_for_reference(context, receiver, line=line)
        if found is None:
            return None
        return frozenset(
            (found.symbol, *self.class_index.ancestor_symbols(found.symbol))
        )

    def _own_member_classes(
        self, context: CompactFlowContext, receiver: LexicalValueReference
    ) -> tuple[str, ...] | None:
        """`self.x`: the classes the method's class declares `x`'s annotation to admit."""

        declaration = context.declaration
        if (
            declaration is None
            or declaration.owner_class_qualname is None
            or receiver.root_name != declaration.nominal_receiver_name
            or len(receiver.attribute_path) != 1
        ):
            return None
        owner = self.class_index.symbol_for(
            file_path=context.file_path, qualname=declaration.owner_class_qualname
        )
        if owner is None:
            return None
        return self._member_declarations_in_mro(
            owner, receiver.attribute_path[0]
        ).fold(self._member_annotation_classes, lambda _violation: None)

    def _member_annotation_classes(
        self,
        declaring_class: CompactIndexedClass,
        member: CompactClassMemberDeclaration,
    ) -> frozenset[str] | None:
        parts = member.annotation_reference_parts
        if parts is None:
            return None
        return self._annotated_classes(
            declaring_class.module_name, ast.parse(".".join(parts), mode="eval").body
        )

    def _bound_result_classes(
        self,
        context: CompactFlowContext,
        receiver: LexicalValueReference,
        position: CompactFlowPosition,
    ) -> frozenset[str] | None:
        call = context.flow.bound_call_result_for(receiver, position)
        if call is None:
            return None
        target = self.resolve_function_call(context, call)
        if isinstance(target.target_resolution, ResolvedCompactClassTarget):
            return self._class_families(
                (target.target_resolution.resolved_declaration.symbol,)
            )
        resolved = target.resolved_call
        returned = (
            None
            if resolved is None
            else self.declared_return_class_symbol_for(resolved)
        )
        return None if returned is None else self._class_families((returned,))

    def _declared_receiver_classes(
        self, context: CompactFlowContext, name: str
    ) -> tuple[str, ...] | None:
        declaration = context.declaration
        if declaration is None:
            return None
        if (
            name == declaration.nominal_receiver_name
            and declaration.owner_class_qualname is not None
        ):
            owner = self.class_index.symbol_for(
                file_path=context.file_path, qualname=declaration.owner_class_qualname
            )
            return None if owner is None else (owner,)
        annotation = next(
            (
                parameter.annotation_expression
                for parameter in declaration.signature.parameters
                if parameter.name == name
            ),
            None,
        )
        if annotation is None:
            return None
        try:
            expression = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return None
        return self._annotated_classes(declaration.identity.module_name, expression)

    def _annotated_classes(
        self, module_name: str, expression: ast.expr
    ) -> tuple[str, ...] | None:
        """The scanned classes an annotation admits, or None when one alternative is unresolved."""

        classes: list[str] = []
        for part in self._annotation_alternatives(expression):
            if isinstance(part, ast.Constant) and part.value is None:
                continue
            reference = LexicalValueReference.from_expression(part)
            if reference is None:
                return None
            symbol = self.class_resolver.symbol_for(
                module_name=module_name,
                reference_parts=reference.parts,
                allow_unique_unqualified=False,
            )
            if symbol is not None:
                classes.append(symbol)
                continue
            foreign = self._foreign_class(module_name, reference.parts)
            if foreign is None:
                return None
            # An instance of a class outside the scan holds no function of the
            # scan except one stored on it, which is an escape; only a scanned
            # subclass of it brings scanned methods.
            classes.extend(self.foreign_base_subclasses.get(foreign, ()))
        return tuple(classes)

    OPTIONAL: ClassVar[frozenset[str]] = frozenset(("Optional", "type", "Type"))

    @classmethod
    def _annotation_alternatives(cls, expression: ast.expr) -> list[ast.expr]:
        """The classes an annotation admits: `A | B`, `Optional[A]` and `type[A]` unpacked."""

        match expression:
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return [
                    *cls._annotation_alternatives(left),
                    *cls._annotation_alternatives(right),
                ]
            case ast.Subscript(value=wrapper, slice=inner) if (
                reference := LexicalValueReference.from_expression(wrapper)
            ) is not None and reference.terminal_name in cls.OPTIONAL:
                return cls._annotation_alternatives(inner)
        return [expression]

    def _foreign_class(self, module_name: str, parts: tuple[str, ...]) -> str | None:
        """The qualified name of a class outside the scan the reference names, or None.

        The reference's root must be imported from a module outside the scan,
        or be an unbound builtin name.
        """

        root, *rest = parts
        imported = self.class_resolver.import_aliases_by_module_name.get(
            module_name, {}
        ).get(root)
        if imported is not None:
            qualified = ".".join((imported, *rest))
            return (
                None if qualified.partition(".")[0] in self.scanned_roots else qualified
            )
        if not rest and hasattr(builtins, root) and root not in self.scanned_roots:
            module_bindings = self.module_flow_context_for_name(module_name)
            if (
                module_bindings is None
                or root in module_bindings.flow.mutations_by_root_name
            ):
                return None
            return f"{builtins.__name__}.{root}"
        return None

    @cached_property
    def scanned_roots(self) -> frozenset[str]:
        return frozenset(
            projection.module_name.partition(".")[0]
            for projection in self.class_projections
        )

    @cached_property
    def foreign_base_subclasses(self) -> dict[str, tuple[str, ...]]:
        """Scanned classes by each class outside the scan they directly inherit from."""

        found: dict[str, list[str]] = {}
        for projection in self.class_projections:
            for indexed in projection.classes:
                for base in indexed.base_references:
                    parts = base.resolved_parts
                    qualified = base.fold_binding(
                        lambda _binding, parts=parts: ".".join(parts),
                        lambda parts=parts: (
                            f"{builtins.__name__}.{parts[0]}"
                            if len(parts) == 1 and hasattr(builtins, parts[0])
                            else ".".join(parts)
                        ),
                    )
                    if qualified.partition(".")[0] not in self.scanned_roots:
                        found.setdefault(qualified, []).append(indexed.symbol)
        return {base: tuple(classes) for base, classes in found.items()}

    def _class_families(self, classes: tuple[str, ...]) -> frozenset[str]:
        family: set[str] = set()
        for owner in classes:
            descendants = self.class_index.descendant_symbols(owner)
            family.update((owner, *self.class_index.ancestor_symbols(owner)))
            for descendant in descendants:
                family.update(
                    (descendant, *self.class_index.ancestor_symbols(descendant))
                )
        return frozenset(family)

    @cached_property
    def callable_escapes(self) -> tuple[CompactCallableEscape, ...]:
        return tuple(
            self.resolve_callable_escape(context, use)
            for context in self.flow_contexts
            for use in context.flow.callable_reference_uses
        )

    @cached_property
    def resolved_product_constructions(
        self,
    ) -> tuple[CompactResolvedProductConstruction, ...]:
        if not self.product_authorities_by_symbol:
            return ()
        return tuple(
            resolved
            for call in self.function_call_resolutions
            if (
                resolved := call.target_resolution.resolve_construction(
                    self, call.context, call.call
                )
            )
            is not None
        )

    def resolve_function_call(
        self,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CompactFunctionCallResolution:
        return call.target_use.resolve(self, context).resolve_call(context, call)

    def resolve_callable_escape(
        self, context: CompactFlowContext, use: CompactCallableReferenceUse
    ) -> CompactCallableEscape:
        return CompactCallableEscape(context, use, use.resolve(self, context))

    def resolve_function_target(
        self,
        context: CompactFlowContext,
        target: CompactCallTargetReference,
        position: CompactFlowPosition,
    ) -> CompactCallTargetResolution:
        """Resolve through the target declaration rather than its concrete class."""
        return target.resolve(self, context, position)

    def _class_member_method_resolution(
        self,
        context: CompactFlowContext,
        target: CurrentClassMemberMethodReference,
        position: CompactFlowPosition,
    ) -> CompactCallTargetResolution:
        candidate_symbols = target.local_candidate_symbols(context.module_name, ())
        if target.uses_runtime_class_lookup and self._lexical_binding_exists(
            context,
            LexicalValueReference("type"),
            position,
        ):
            return UnboundedCompactFunctionTarget(
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )
        # Until the member's class is proved, `self.member.method` is attribute
        # `method` of an unproved object, not the current class's own method.
        owner_symbol = self.class_index.symbol_for(
            file_path=context.file_path,
            qualname=target.owner_class_qualname,
        )
        if owner_symbol is None:
            return self.attribute_target(
                target.method_name,
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
            )
        return self._member_declarations_in_mro(
            owner_symbol,
            target.member_name,
        ).fold(
            lambda declaring_class, member_declaration: self._member_method_target(
                target, candidate_symbols, declaring_class, member_declaration
            ),
            lambda violation: self.attribute_target(
                target.method_name, candidate_symbols, violation
            ),
        )

    def _member_method_target(
        self,
        target: CurrentClassMemberMethodReference,
        candidate_symbols: tuple[str, ...],
        declaring_class: CompactIndexedClass,
        member_declaration: CompactClassMemberDeclaration,
    ) -> CompactCallTargetResolution:
        reference_parts = member_declaration.annotation_reference_parts
        if reference_parts is None:
            return self.attribute_target(
                target.method_name,
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        member_type_symbol = self.class_resolver.symbol_for(
            module_name=declaring_class.module_name,
            reference_parts=reference_parts,
            allow_unique_unqualified=False,
        )
        if member_type_symbol is None:
            return self.attribute_target(
                target.method_name,
                candidate_symbols,
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        return self._function_resolution_for_symbol(
            f"{member_type_symbol}.{target.method_name}"
        ).through_descriptor(CompactDescriptorAccess.INSTANCE)

    def _member_declarations_in_mro(
        self,
        owner_symbol: str,
        member_name: str,
    ) -> _CompactClassMemberResolution:
        owner = self.class_index.class_for(owner_symbol)
        if owner is None:
            return _ClassMemberViolation(
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
            )
        # A direct declaration already owns the member, independent of inherited order.
        owners = (owner,)
        if not any(
            member.name == member_name for member in owner.direct_member_declarations
        ):
            mro = self.class_index.mro_authority.resolve(owner_symbol).mro_type
            if mro is None:
                return _ClassMemberViolation(
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            owners = mro.declarations
        for current in owners:
            declarations = tuple(
                declaration
                for declaration in current.direct_member_declarations
                if declaration.name == member_name
            )
            if len(declarations) == 1:
                return _ResolvedClassMember(current, declarations[0])
            if declarations:
                return _ClassMemberViolation(
                    CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
                )
        return _ClassMemberViolation(
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
        )

    def _lexical_binding_exists(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        position: CompactFlowPosition,
    ) -> bool:
        for scope_qualname in context.flow.lexical_scope_qualnames:
            owner_symbol = (
                f"{context.module_name}.{scope_qualname}"
                if scope_qualname
                else context.module_name
            )
            scope_context = self.flow_context_for_symbol(owner_symbol)
            if scope_context is None:
                continue
            if (
                self._scope_binding_resolution(
                    scope_context,
                    reference,
                    (
                        position
                        if scope_context.owner_symbol == context.owner_symbol
                        else None
                    ),
                )
                is not None
            ):
                return True
        return False

    def resolve_product_construction(
        self,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CompactResolvedProductConstruction | None:
        return call.target_use.resolve(self, context).resolve_construction(
            self, context, call
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
        context: CompactFlowContext,
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
    ) -> tuple[CompactCallableEscape, ...]:
        symbols = frozenset((function_symbol,))
        return tuple(
            escape
            for escape in self.callable_escapes
            if escape.target_resolution.candidate_symbols_within(symbols)
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
        """Capture immutable component inputs; derive obligations when queried."""

        return CompactCallableComponentAuthorityProof(
            repository=self,
            participant_parameters=tuple(
                (symbol, frozenset(parameters))
                for symbol, parameters in sorted(parameter_names_by_participant.items())
            ),
            component_call_identities=frozenset(component_call_identities),
        )

    @staticmethod
    def _signature_is_closed_for_parameters(
        declaration: CompactFunctionDeclaration,
        context: CompactFlowContext,
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
            declaration = self.function_declarations_by_symbol.get(participant_symbol)
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
        context: CompactFlowContext,
        reference: LexicalValueReference,
        position: CompactFlowPosition,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CompactCallTargetResolution:
        possible_symbols: list[str] = []
        for scope_qualname in context.flow.lexical_scope_qualnames:
            owner_symbol = (
                f"{context.module_name}.{scope_qualname}"
                if scope_qualname
                else context.module_name
            )
            scope_context = self.flow_context_for_symbol(owner_symbol)
            candidate_symbol = ".".join((owner_symbol, *reference.parts))
            possible_symbols.append(candidate_symbol)
            if scope_context is None:
                continue
            resolution = self._scope_binding_resolution(
                scope_context,
                reference,
                (
                    position
                    if scope_context.owner_symbol == context.owner_symbol
                    else None
                ),
                pending_bindings,
            )
            if resolution is not None:
                return resolution
        return OpenCompactFunctionTarget(
            tuple(dict.fromkeys(possible_symbols)),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _scope_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        use_position: CompactFlowPosition | None,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CompactCallTargetResolution | None:
        root_name = reference.root_name
        origins = self.star_import_origins_for(context.module_name)
        if (
            context.flow.owner.kind.is_module_scope
            and origins
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
                for origin in origins
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

        selection = context.flow.binding_resolution_for(root_name, use_position)
        if selection is None:
            return None
        return selection.resolve_binding(
            self,
            context,
            reference,
            use_position,
            pending_bindings,
        )

    def _possible_binding_resolution(
        self,
        context: CompactFlowContext,
        reference: LexicalValueReference,
        violation: CompactFunctionTargetResolutionViolation,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CompactCallTargetResolution:
        mutations = context.flow.mutations_by_root_name.get(reference.root_name, ())
        local_and_imported = OpenCompactFunctionTarget(
            tuple(
                dict.fromkeys(
                    (
                        *(
                            ".".join((qualified_name, *reference.attribute_path))
                            for mutation in mutations
                            if (origin := mutation.imported_origin) is not None
                            and (qualified_name := origin.qualified_name) is not None
                        ),
                        ".".join((context.owner_symbol, *reference.parts)),
                    )
                )
            ),
            violation,
        )
        # An unproved value's attribute is attribute `name` of an unknown object.
        attribute = (
            (self.attribute_target(reference.attribute_path[-1], (), violation),)
            if reference.attribute_path
            else ()
        )
        return AlternativeCompactFunctionTargets(
            (
                local_and_imported,
                *attribute,
                *(
                    self._captured_alias_resolution(
                        alias,
                        context,
                        reference,
                        None,
                        pending_bindings | {CompactBindingVisit(context, mutation)},
                    )
                    for mutation in mutations
                    if CompactBindingVisit(context, mutation) not in pending_bindings
                    and (alias := context.flow.exact_alias_for(mutation)) is not None
                ),
            ),
            violation,
        )

    def _function_resolution_for_symbol(
        self,
        symbol: str,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CompactCallTargetResolution:
        parts = symbol.split(".")
        for prefix_length in range(len(parts) - 1, 0, -1):
            context = self.module_flow_context_for_name(".".join(parts[:prefix_length]))
            if context is None:
                continue
            resolution = self._scope_binding_resolution(
                context,
                LexicalValueReference(
                    parts[prefix_length], tuple(parts[prefix_length + 1 :])
                ),
                None,
                pending_bindings,
            )
            if resolution is not None:
                return resolution
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
            )
        return OpenCompactFunctionTarget(
            (symbol,),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _declared_function_resolution(
        self,
        symbol: str,
    ) -> CompactCallTargetResolution:
        multiplicity = self.function_declaration_multiplicity_for_symbol(symbol)
        declaration = multiplicity.unambiguous_declarations_by_handle.get(symbol)
        if declaration is not None:
            return ResolvedCompactFunctionTarget(declaration)
        if symbol in multiplicity.ambiguous_handles:
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )
        return OpenCompactFunctionTarget(
            (symbol,),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )

    def _selected_function_resolution(
        self,
        symbol: str,
        binding: CompactMutation,
    ) -> CompactCallTargetResolution:
        """A selected definition must identify this function's source declaration."""

        resolution = self._declared_function_resolution(symbol)
        if (
            resolution.declaration is not None
            and resolution.declaration.line != binding.line
        ):
            return OpenCompactFunctionTarget(
                (symbol,),
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
            )
        return resolution

    def _class_method_resolution(
        self,
        owner: CompactIndexedClass,
        member_name: str,
        pending_bindings: frozenset[
            CompactBindingVisit[CompactFlowContext]
        ] = frozenset(),
    ) -> CompactCallTargetResolution:
        possible_symbols = [
            f"{class_symbol}.{member_name}"
            for class_symbol in (
                owner.symbol,
                *self.class_index.ancestor_symbols(owner.symbol),
            )
        ]
        owners = (owner,)
        context = self.flow_context_for_symbol(owner.symbol)
        if context is None or member_name not in context.flow.mutations_by_root_name:
            mro = self.class_index.mro_authority.resolve(owner.symbol).mro_type
            if mro is None:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            owners = mro.declarations
        for current in owners:
            context = self.flow_context_for_symbol(current.symbol)
            if context is None:
                return OpenCompactFunctionTarget(
                    tuple(possible_symbols),
                    CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY,
                )
            resolution = self._scope_binding_resolution(
                context,
                LexicalValueReference(member_name),
                None,
                pending_bindings,
            )
            if resolution is None:
                continue
            declaration = resolution.declaration
            if declaration is not None and (
                declaration.identity.module_name != current.module_name
                or declaration.owner_class_qualname != current.qualname
            ):
                return OpenCompactFunctionTarget(
                    resolution.possible_symbols,
                    CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                )
            return resolution.through_descriptor(CompactDescriptorAccess.CLASS)
        return OpenCompactFunctionTarget(
            tuple(possible_symbols),
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        )


@dataclass(frozen=True, slots=True)
class CompactProductFlowRepository(ProductFlowRepository):
    """Persistable declaration queries without an inferred runtime activation."""

    product_projections: Sequence[CompactProductFlowModuleProjection]
    class_projections: Sequence[CompactModuleClassProjection]

    def captured_value(self, value: CompactFlowValue) -> CapturedReferenceResolution:
        return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            Sequence[object],
        ],
    ) -> Self:
        """Recover the typed product-flow join declared by its fact families."""

        return cls(
            product_projections=cast(
                Sequence[CompactProductFlowModuleProjection],
                projections_by_family[CompactProductFlowModuleProjectionFamily],
            ),
            class_projections=cast(
                Sequence[CompactModuleClassProjection],
                projections_by_family[CompactModuleClassProjectionFamily],
            ),
        )

    @classmethod
    def from_source_projections(
        cls, sources: tuple[SourceProductFlowProjection, ...]
    ) -> Self:
        """Join the exact observed events without recollecting equal-looking flows."""
        return cls(
            tuple(source.compact for source in sources),
            CompactModuleClassProjectionFamily.collect_modules(
                tuple(source.module for source in sources)
            ),
        )


@dataclass(frozen=True)
class SourceProductFlowRepository(RepositoryModuleBindingProof, ProductFlowRepository):
    """Derive flow queries and runtime capture from the same original source task."""

    def iter_product_projections(self) -> Iterator[CompactProductFlowModuleProjection]:
        """Preserve every original module without projecting unrequested successors."""
        for module in self.modules:
            yield self.source_projection(module).compact

    def _class_source(self, symbol: str) -> tuple[ParsedModule, ast.ClassDef] | None:
        declaration = self.class_index.class_for(symbol)
        if declaration is None or not self.contains_module(declaration.module_name):
            return None
        module = next(
            module
            for module in self.modules
            if module.module_name == declaration.module_name
        )
        classes = [
            node
            for _, node in module_syntax_index(module.module).indexed_nodes_of_type(
                ast.ClassDef
            )
            if node.lineno == declaration.line
        ]
        return (module, classes[0]) if len(classes) == 1 else None

    def _class_sources_for_lookup(
        self, symbol: str
    ) -> Iterator[tuple[CompactIndexedClass, ParsedModule, ast.ClassDef]]:
        for declaration in self.class_index.mro_authority.lookup_prefix(symbol):
            source = self._class_source(declaration.symbol)
            if source is None:
                return
            yield declaration, *source

    def _boolean_annotation(
        self,
        module: ParsedModule,
        annotation: ast.AST | None,
        *,
        line: int,
        class_names: frozenset[str],
    ) -> bool | None:
        if annotation is None:
            return None
        reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(annotation)
        if reference is None:
            return None
        witness = self.reference_or_builtin_witness_at(
            module, reference, line=line, preceding_class_bound_names=class_names
        )
        return None if witness is None else witness.qualified_name == "builtins.bool"

    def _parameter_bindings(
        self, symbol: str, reference: ast.Name
    ) -> tuple[CompactFlowContext, FunctionNominalParameterBindingAuthority] | None:
        """Admit a reference in the exact captured function body, including nested defs."""
        context = self.flow_context_for_symbol(symbol)
        if context is None or not self.contains_module(context.module_name):
            return None
        declaration = context.declaration
        if declaration is None:
            return None
        module = next(
            (
                module
                for module in self.modules
                if module.module_name == context.module_name
            ),
            None,
        )
        if module is None:
            return None
        syntax = module_syntax_index(module.module)
        current: ast.AST = reference
        while True:
            parent = syntax.parent_by_node.get(current)
            if parent is None:
                return None
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = parent
                break
            if isinstance(parent, (ast.Lambda, ast.ClassDef)):
                return None
            current = parent
        if current not in function.body:
            return None
        if SourceByteSpan.require_node(function) != declaration.source_span:
            return None
        return context, FunctionNominalParameterBindingAuthority(
            ModuleNominalBindingAuthority(module), function
        )

    def function_parameter_type_name(
        self, symbol: str, reference: ast.Name
    ) -> str | None:
        """Join a stable parameter or implicit receiver to its nominal owner."""
        admitted = self._parameter_bindings(symbol, reference)
        if admitted is None:
            return None
        context, bindings = admitted
        declaration = context.declaration
        assert declaration is not None
        function = bindings.function
        name = reference.id
        if bindings.comprehension_owner_for_reference(reference) is not None:
            return None
        declared_type = bindings.type_name_for_reference(name)
        if declared_type is not None:
            return declared_type
        # Call-signature promotion safety is not nominal receiver ownership:
        # a property getter still receives its declared owning instance.
        if name != declaration.nominal_receiver_name:
            return None
        if name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body):
            return None
        return f"{context.module_name}.{declaration.owner_class_qualname}"

    def function_class_parameter_type_name(
        self, symbol: str, reference: ast.Name
    ) -> str | None:
        """Keep class objects distinct when selecting class versus instance members."""
        admitted = self._parameter_bindings(symbol, reference)
        if admitted is None:
            return None
        context, bindings = admitted
        if bindings.comprehension_owner_for_reference(reference) is not None:
            return self._comprehension_class_type_name(symbol, reference, bindings)
        declared_type = bindings.class_type_name_for_reference(reference.id)
        if declared_type is not None:
            return declared_type
        declaration = context.declaration
        assert declaration is not None
        if (
            declaration.binding_kind.receiver_access
            is not CompactDescriptorAccess.CLASS
        ):
            return None
        if reference.id != declaration.nominal_receiver_name:
            return None
        return self.function_parameter_type_name(symbol, reference)

    def _comprehension_class_type_name(
        self,
        symbol: str,
        reference: ast.Name,
        bindings: FunctionNominalParameterBindingAuthority,
    ) -> str | None:
        expression = bindings.comprehension_iterable_for_reference(reference)
        if isinstance(expression, ast.Call):
            # Python's builtin dict.fromkeys preserves its iterable's key types.
            # A merely same-named user callable supplies no such evidence.
            call = expression
            if (
                not isinstance(call.func, ast.Attribute)
                or call.func.attr != "fromkeys"
                or not isinstance(call.func.value, ast.Name)
                or len(call.args) not in (1, 2)
                or call.keywords
                or any(isinstance(argument, ast.Starred) for argument in call.args)
            ):
                return None
            receiver = call.func.value
            witness = self.reference_or_builtin_witness_at(
                bindings.module_bindings.parsed_module,
                receiver,
                line=receiver.lineno,
                preceding_class_bound_names=bindings.runtime_bound_names_for_reference(
                    receiver
                ),
            )
            if witness is None or witness.qualified_name != "builtins.dict":
                return None
            expression = call.args[0]
        if isinstance(expression, ast.Name):
            if bindings.comprehension_owner_for_reference(expression) is not None:
                return None
            annotation = bindings.stable_parameter_annotations.get(expression.id)
            return (
                None
                if annotation is None
                else bindings.module_bindings.iterated_class_annotation_name_at(
                    annotation,
                    line=bindings.function.lineno,
                    class_bound_names=bindings.class_bound_names,
                )
            )
        if isinstance(expression, ast.Attribute) and isinstance(
            expression.value, ast.Name
        ):
            receiver_type = self.function_parameter_type_name(symbol, expression.value)
            if receiver_type is not None:
                return self._project_receiver_field(
                    receiver_type,
                    expression.attr,
                    ModuleNominalBindingAuthority.iterated_class_annotation_name_at,
                )
        return None

    def declared_boolean_inversion(
        self, symbol: str, attribute: ast.Attribute
    ) -> bool | None:
        """A bound public bool query or ClassVar[bool], not arbitrary truthiness."""
        receiver = attribute.value
        if isinstance(receiver, ast.Name):
            class_type = self.function_class_parameter_type_name(symbol, receiver)
            if class_type:
                return self.public_boolean_class_member(class_type, attribute.attr)
            instance_type = self.function_parameter_type_name(symbol, receiver)
            return (
                self.public_boolean_property(instance_type, attribute.attr)
                if instance_type
                else None
            )
        if isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Name):
            if self.function_class_parameter_type_name(symbol, receiver.value):
                return None
            instance_type = self.function_parameter_type_name(symbol, receiver.value)
            field_type = (
                self.receiver_field_type_name(instance_type, receiver.attr)
                if instance_type
                else None
            )
            return (
                self.public_boolean_property(field_type, attribute.attr)
                if field_type
                else None
            )
        return None

    def public_boolean_class_member(self, symbol: str, name: str) -> bool | None:
        """ClassVar[bool] is a declared class capability, unlike a raw instance field."""
        if name.startswith("_"):
            return False
        for owner, module, node in self._class_sources_for_lookup(symbol):
            member = next(
                (
                    member
                    for member in owner.direct_member_declarations
                    if member.name == name
                ),
                None,
            )
            if member is None:
                if name in owner.method_names:
                    return False
                continue
            if member.annotation_expression is None:
                return False
            annotation = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.subscript_or_none(
                ast.parse(member.annotation_expression, mode="eval").body
            )
            if annotation is None:
                return False
            class_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
            wrapper = self.reference_or_builtin_witness_at(
                module,
                annotation.value,
                line=member.line,
                preceding_class_bound_names=class_names,
            )
            if wrapper is None:
                return None
            if wrapper.qualified_name != "typing.ClassVar":
                return False
            return self._boolean_annotation(
                module, annotation.slice, line=member.line, class_names=class_names
            )
        return None

    def receiver_field_type_name(self, symbol: str, field_name: str) -> str | None:
        """A nominal receiver-field type, stopping at opaque writes or ancestry."""
        return self._project_receiver_field(
            symbol, field_name, ModuleNominalBindingAuthority.nominal_annotation_name_at
        )

    def _project_receiver_field(
        self, symbol: str, field_name: str, project: Callable[..., str | None]
    ) -> str | None:
        """One field/constructor/ancestry lookup shared by annotation projections."""
        for owner, module, node in self._class_sources_for_lookup(symbol):
            bindings = ModuleNominalBindingAuthority(module)
            for member in owner.direct_member_declarations:
                if member.name == field_name:
                    annotation = next(
                        (
                            statement.annotation
                            for statement in node.body
                            if isinstance(statement, ast.AnnAssign)
                            and isinstance(statement.target, ast.Name)
                            and statement.target.id == field_name
                        ),
                        None,
                    )
                    return (
                        None
                        if annotation is None
                        else project(
                            bindings,
                            annotation,
                            line=member.line,
                            class_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                                node.body
                            ),
                        )
                    )
            if field_name in owner.method_names:
                return None
            constructors = [
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name == "__init__"
            ]
            if len(constructors) > 1:
                return None
            if constructors:
                parameters = FunctionNominalParameterBindingAuthority(
                    bindings, constructors[0]
                )
                fields = parameters.receiver_field_annotations
                if field_name in fields:
                    annotation = fields[field_name]
                    return (
                        None
                        if annotation is None
                        else project(
                            bindings,
                            annotation,
                            line=constructors[0].lineno,
                            class_bound_names=parameters.class_bound_names,
                        )
                    )
        return None

    def public_boolean_property(self, symbol: str, name: str) -> bool | None:
        """True for a bound public property -> bool; None means unresolved lookup.

        Raw fields/private attributes return False. No inspected code is run.
        """
        if name.startswith("_"):
            return False
        for owner, module, node in self._class_sources_for_lookup(symbol):
            if name in owner.direct_declared_member_names:
                return False
            methods = [
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name == name
            ]
            if not methods:
                continue
            if len(methods) != 1:
                return None
            method = methods[0]
            if len(method.decorator_list) != 1:
                return False
            class_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
            descriptor = self.reference_or_builtin_witness_at(
                module,
                method.decorator_list[0],
                line=method.lineno,
                preceding_class_bound_names=class_names,
            )
            if descriptor is None:
                return None
            if descriptor.qualified_name != "builtins.property":
                return False
            return self._boolean_annotation(
                module, method.returns, line=method.lineno, class_names=class_names
            )
        return None

    def require_native_source_dependency(self, declaration: NativeDeclaration) -> None:
        """Reconcile native source with the complete current projected module set."""
        compilation = NativePythonCompilation.from_native_declaration(declaration)
        if compilation is None:
            return  # Native type immutability was proved by its compiler authority.
        candidates = tuple(
            module
            for module in self.modules
            if source_path_text(module.file_path)
            == source_path_text(compilation.file_path)
        )
        if not candidates:
            return  # External native source remains covered by its explicit entry premise.
        if len(candidates) != 1:
            raise ValueError("Native source dependency has ambiguous projected owners")
        module = candidates[0]
        if module.source != compilation.source:
            raise ValueError(
                "Native source dependency differs from the projected source state"
            )

    def projected_with_source_projection(
        self, projection: ParsedModuleSourceProjection
    ) -> Self:
        """Retain module-local proofs only for exact unchanged source owners."""
        self.require_module_owners(projection.modules)
        projected = type(self).from_modules(projection.projected_modules)
        for original, current in zip(
            self.modules, projection.projected_existing_modules, strict=True
        ):
            if current is not original:
                continue
            module_id = id(original)
            source = self._source_projections.get(module_id)
            execution = self._native_executions.get(module_id)
            if source is not None:
                if source.module is not original:
                    raise ValueError(
                        "Retained source projection belongs to a different parsed owner"
                    )
                projected._source_projections[module_id] = source
            if execution is not None:
                if source is None or execution.source is not source:
                    raise ValueError(
                        "Retained native execution belongs to a different source projection"
                    )
                projected._native_executions[module_id] = execution
        return projected

    def require_module_owners(self, modules: tuple[ParsedModule, ...]) -> None:
        """Authenticate the exact parsed owners represented by this repository."""
        if len(modules) != len(self.modules) or any(
            supplied is not owned
            for supplied, owned in zip(modules, self.modules, strict=True)
        ):
            raise ValueError(
                "Product-flow repository belongs to different parsed module owners"
            )

    def product_projection_for_module(
        self, module_name: str
    ) -> CompactProductFlowModuleProjection | None:
        source = self.source_projection_for_module(module_name)
        return None if source is None else source.compact

    def source_projection_for_module(
        self, module_name: str
    ) -> SourceProductFlowProjection | None:
        if not self.contains_module(module_name):
            return None
        (module,) = (
            module for module in self.modules if module.module_name == module_name
        )
        return self.source_projection(module)

    def function_declaration_multiplicity_for_symbol(
        self, symbol: str
    ) -> IdentityHandleMultiplicityProjection[str, CompactFunctionDeclaration]:
        return UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            (
                declaration
                for projection in self.projections_for_symbol(symbol)
                for declaration in projection.function_declarations
                if declaration.identity.symbol == symbol
            ),
            lambda declaration: declaration.identity.symbol,
        )

    def module_flow_context_for_name(
        self, module_name: str
    ) -> CompactFlowContext | None:
        source = self.source_projection_for_module(module_name)
        return None if source is None else source.module_context

    def flow_context_for_symbol(self, symbol: str) -> CompactFlowContext | None:
        candidates = tuple(
            context
            for projection in self.projections_for_symbol(symbol)
            for context in projection.flow_contexts
            if context.owner_symbol == symbol
        )
        return candidates[0] if len(candidates) == 1 else None

    def projections_for_symbol(
        self, symbol: str
    ) -> Iterator[CompactProductFlowModuleProjection]:
        """Visit every possible module owner, including overlapping and duplicate names.

        Owner symbols are module names followed by qualified scope names. A prefix
        selects possible owners, never a unique declaration or an execution proof.
        The selected queries still check all matching original declaration rows.
        """
        for module in self.modules:
            if symbol == module.module_name or symbol.startswith(
                module.module_name + "."
            ):
                yield self.source_projection(module).compact

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        return cls(modules)

    @cached_property
    def sources(self) -> tuple[SourceProductFlowProjection, ...]:
        return tuple(self.source_projection(module) for module in self.modules)

    @cached_property
    def product_projections(self) -> tuple[CompactProductFlowModuleProjection, ...]:
        return tuple(self.iter_product_projections())

    @cached_property
    def class_projections(self) -> tuple[CompactModuleClassHeader, ...]:
        return CompactModuleClassHeader.from_modules(self.modules)

    def captured_value(self, value: CompactFlowValue) -> CapturedReferenceResolution:
        source = self.source_projection_for_module(value.context.module_name)
        if source is None:
            raise ValueError("Runtime capture has no unique original source owner")
        source.source_operation(value.context, value.use)
        return self.native_reference_environment(source.module).kernel.read(value)
