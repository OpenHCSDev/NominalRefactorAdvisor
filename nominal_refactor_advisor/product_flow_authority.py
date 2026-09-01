"""Repository joins for product-flow facts and their nominal authorities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from .class_index import (
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactModuleClassProjection,
    build_compact_class_family_index,
)
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
)
from .product_flow import (
    CompactCallTargetReference,
    CompactCallableReferenceUse,
    CompactFlowOwnerKind,
    CompactFlowPosition,
    CompactFunctionCall,
    CompactFunctionDeclaration,
    CompactFunctionFlow,
    CompactMutationKind,
    CompactProductConstruction,
    CompactProductFlowModuleProjection,
    CompactValueOriginResolution,
    LexicalValueReference,
)
from .semantic_descent import (
    CompactSemanticModuleProjection,
    SemanticClassSupplement,
)


@dataclass(frozen=True)
class CompactProductAuthority:
    """One resolved dataclass schema which can own a parameter product."""

    class_symbol: str
    field_names: tuple[str, ...]
    file_path: str
    line: int


@dataclass(frozen=True)
class CompactProductFlowContext:
    """One execution flow joined to its module and optional declaration."""

    module_name: str
    file_path: str
    flow: CompactFunctionFlow
    declaration: CompactFunctionDeclaration | None

    @property
    def owner_symbol(self) -> str:
        if self.flow.owner.kind is CompactFlowOwnerKind.MODULE:
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
            for value in (
                *(argument.value for argument in self.call.positional_arguments),
                *(argument.value for argument in self.call.keyword_arguments),
            )
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
class CompactResolvedProductConstruction:
    """One explicit construction joined to its complete product authority."""

    context: CompactProductFlowContext
    call: CompactFunctionCall
    construction: CompactProductConstruction
    authority: CompactProductAuthority


@dataclass(frozen=True)
class CompactProductFlowRepository:
    """Derived query authority over product, class, and semantic projections."""

    product_projections: tuple[CompactProductFlowModuleProjection, ...]
    class_projections: tuple[CompactModuleClassProjection, ...]
    semantic_projections: tuple[CompactSemanticModuleProjection, ...]

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
    def semantic_supplements_by_class_symbol(
        self,
    ) -> dict[str, SemanticClassSupplement]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                supplement
                for projection in self.semantic_projections
                for supplement in projection.class_supplements
            ),
            lambda supplement: supplement.class_symbol,
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
                    if flow.owner.kind is CompactFlowOwnerKind.FUNCTION
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
                if context.flow.owner.kind is CompactFlowOwnerKind.MODULE
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
            for class_symbol in self.class_index.classes_by_symbol
            if (authority := self.product_authority_for(class_symbol)) is not None
        }

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

    def product_authority_for(
        self,
        class_symbol: str,
    ) -> CompactProductAuthority | None:
        indexed_class = self.class_index.class_for(class_symbol)
        supplement = self.semantic_supplements_by_class_symbol.get(class_symbol)
        if indexed_class is None or supplement is None or not supplement.is_dataclass:
            return None
        dataclass_lineage = tuple(
            symbol
            for symbol in reversed(self.class_index.ancestor_symbols(class_symbol))
            if (
                ancestor_supplement := self.semantic_supplements_by_class_symbol.get(
                    symbol
                )
            )
            is not None
            and ancestor_supplement.is_dataclass
        )
        if any(
            len(
                tuple(
                    base_symbol
                    for base_symbol in self.class_index.class_for(
                        symbol
                    ).resolved_base_symbols
                    if base_symbol in dataclass_lineage or base_symbol == class_symbol
                )
            )
            > 1
            for symbol in (*dataclass_lineage, class_symbol)
        ):
            return None
        fields_by_name: dict[str, None] = {}
        for symbol in (*dataclass_lineage, class_symbol):
            lineage_supplement = self.semantic_supplements_by_class_symbol.get(symbol)
            if lineage_supplement is None:
                return None
            for field_name, _line in lineage_supplement.annotated_fields:
                fields_by_name.setdefault(field_name, None)
        if len(fields_by_name) < 2:
            return None
        return CompactProductAuthority(
            class_symbol=class_symbol,
            field_names=tuple(fields_by_name),
            file_path=indexed_class.file_path,
            line=indexed_class.line,
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

    def callable_escapes_for(
        self,
        function_symbol: str,
    ) -> tuple[CompactResolvedCallableEscape, ...]:
        return tuple(
            escape
            for escape in self.callable_escapes
            if escape.declaration.identity.symbol == function_symbol
        )

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
            context.flow.owner.kind is CompactFlowOwnerKind.MODULE
            and class_projection is not None
            and class_projection.star_import_origins
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
        elif context.flow.owner.kind is CompactFlowOwnerKind.MODULE:
            binding = mutations[-1]
        elif len(mutations) == 1:
            binding = mutations[0]
        else:
            return OpenCompactFunctionTarget(
                self._possible_binding_symbols(context, reference),
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION,
            )

        if binding.kind is CompactMutationKind.IMPORT:
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

        if binding.kind is not CompactMutationKind.DEFINITION:
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
        if context.declaration is not None and root_name in {
            parameter.name for parameter in context.declaration.signature.parameters
        }:
            return True
        local_mutations = tuple(
            mutation
            for mutation in context.flow.mutations
            if mutation.reference.root_name == root_name
        )
        if context.flow.owner.kind is CompactFlowOwnerKind.FUNCTION:
            if any(
                mutation.kind is not CompactMutationKind.DEFINITION
                for mutation in local_mutations
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
        return any(
            mutation.kind
            not in {CompactMutationKind.DEFINITION, CompactMutationKind.IMPORT}
            for mutation in relevant_module_mutations
        )
