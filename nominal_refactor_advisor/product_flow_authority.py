"""Repository joins for product-flow facts and their nominal authorities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .class_index import (
    CompactClassFamilyIndex,
    CompactClassReferenceResolver,
    CompactModuleClassProjection,
    build_compact_class_family_index,
)
from .collection_algebra import UniqueIdentityIndexAuthority
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
class CompactResolvedFunctionCall:
    """One call edge whose callee has exactly one nominal declaration."""

    context: CompactProductFlowContext
    call: CompactFunctionCall
    callee: CompactFunctionDeclaration


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
    def function_declarations_by_symbol(
        self,
    ) -> dict[str, CompactFunctionDeclaration]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                declaration
                for projection in self.product_projections
                for declaration in projection.function_declarations
            ),
            lambda declaration: declaration.identity.symbol,
        )

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
    def resolved_function_calls(self) -> tuple[CompactResolvedFunctionCall, ...]:
        return tuple(
            resolved
            for context in self.flow_contexts
            for call in context.flow.calls
            if (resolved := self.resolve_function_call(context, call)) is not None
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
    ) -> CompactResolvedFunctionCall | None:
        declaration = self.resolve_function_target(
            context,
            call.target,
            call.position,
        )
        if declaration is None:
            return None
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
        )
        if declaration is None:
            return None
        return CompactResolvedCallableEscape(context, use, declaration)

    def resolve_function_target(
        self,
        context: CompactProductFlowContext,
        target: CompactCallTargetReference,
        position: CompactFlowPosition,
    ) -> CompactFunctionDeclaration | None:
        if self._has_dynamic_local_binding(context, target, position):
            return None
        symbols = self._function_candidate_symbols(context, target)
        return next(
            (
                declaration
                for symbol in symbols
                if (declaration := self._function_declaration_for_symbol(symbol))
                is not None
            ),
            None,
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

    def _function_candidate_symbols(
        self,
        context: CompactProductFlowContext,
        target: CompactCallTargetReference,
    ) -> tuple[str, ...]:
        candidates = list(
            context.flow.local_candidate_symbols(target, context.module_name)
        )
        reference = target.lexical_reference
        if reference is not None:
            class_projection = self.class_projections_by_module_name.get(
                context.module_name
            )
            if class_projection is not None:
                alias_target = dict(class_projection.import_aliases).get(
                    reference.root_name
                )
                if alias_target is not None:
                    alias_candidate = ".".join(
                        (alias_target, *reference.attribute_path)
                    )
                    if self._module_binding_prefers_import(
                        context.module_name,
                        reference.root_name,
                    ):
                        candidates.insert(0, alias_candidate)
                    else:
                        candidates.append(alias_candidate)
            candidates.extend(
                (
                    f"{context.module_name}.{scope_qualname}."
                    f"{'.'.join(reference.parts)}"
                    if scope_qualname
                    else f"{context.module_name}.{'.'.join(reference.parts)}"
                )
                for scope_qualname in context.flow.lexical_scope_qualnames
            )
        return tuple(dict.fromkeys(candidates))

    def _module_binding_prefers_import(
        self,
        module_name: str,
        root_name: str,
    ) -> bool:
        module_context = self.module_flow_contexts.get(module_name)
        if module_context is None:
            return False
        mutations = tuple(
            mutation
            for mutation in module_context.flow.mutations
            if mutation.reference.root_name == root_name
            and not mutation.position.branch_path
        )
        return bool(mutations and mutations[-1].kind is CompactMutationKind.IMPORT)

    def _function_declaration_for_symbol(
        self,
        symbol: str,
    ) -> CompactFunctionDeclaration | None:
        declaration = self.function_declarations_by_symbol.get(symbol)
        if declaration is not None:
            return declaration
        owner_symbol, separator, member_name = symbol.rpartition(".")
        if not separator or self.class_index.class_for(owner_symbol) is None:
            return None
        return next(
            (
                declaration
                for ancestor_symbol in self.class_index.ancestor_symbols(owner_symbol)
                if (
                    declaration := self.function_declarations_by_symbol.get(
                        f"{ancestor_symbol}.{member_name}"
                    )
                )
                is not None
            ),
            None,
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
