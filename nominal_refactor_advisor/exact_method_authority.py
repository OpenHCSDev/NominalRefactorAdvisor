"""Declaration-derived proof components for exact method promotion."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Self

from .ast_tools import ParsedModule
from .class_index import (
    CLASS_METHOD_OWNERSHIP_HOOK_NAMES,
    ClosedLeafMethodAuthorityProof,
    CompactClassFamilyIndex,
    CompactClassMethod,
    CompactIndexedClass,
    CompactModuleClassProjection,
    CompactModuleClassProjectionFamily,
    ClassSymbolResolutionAuthority,
    build_compact_class_family_index,
)
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from .descriptor_algebra import CollectionAttributeProjection
from .models import SourceLocation
from .name_algebra import CLASS_NAME_ALGEBRA
from .semantic_description_length import (
    CompressionCertificate,
    ExistingAuthorityMethodPromotionCompressionProfile,
)


@dataclass(frozen=True)
class CompactExactMethodOrbit:
    """One exact, promotion-safe method declaration shared by a class cohort."""

    file_path: str
    method_name: str
    indexed_classes: tuple[CompactIndexedClass, ...]
    methods: tuple[CompactClassMethod, ...]

    @classmethod
    def from_declarations(
        cls,
        indexed_classes: tuple[CompactIndexedClass, ...],
        methods: tuple[CompactClassMethod, ...],
    ) -> Self | None:
        """Build one orbit only when every declaration proves one exact source."""

        if len(indexed_classes) < 2 or len(indexed_classes) != len(methods):
            return None
        file_paths = frozenset(item.file_path for item in indexed_classes)
        method_names = frozenset(method.method_name for method in methods)
        source_digests = frozenset(
            method.exact_promotion_source_digest for method in methods
        )
        if (
            len(file_paths) != 1
            or len(method_names) != 1
            or len(source_digests) != 1
            or None in source_digests
            or len(frozenset(method.line_count for method in methods)) != 1
        ):
            return None
        return cls(
            file_path=next(iter(file_paths)),
            method_name=next(iter(method_names)),
            indexed_classes=indexed_classes,
            methods=methods,
        )

    @property
    def class_symbols(self) -> tuple[str, ...]:
        return tuple(indexed_class.symbol for indexed_class in self.indexed_classes)


@dataclass(frozen=True)
class ExactMirroredLeafRoleComponent:
    """One role whose domain leaves carry the same movable implementations."""

    role_name: str
    left_class: CompactIndexedClass
    right_class: CompactIndexedClass
    method_orbits: tuple[CompactExactMethodOrbit, ...]


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyComponent:
    """Exact role implementations repeated across two nominal domain axes."""

    left_root: CompactIndexedClass
    right_root: CompactIndexedClass
    contract_method_names: tuple[str, ...]
    roles: tuple[ExactMirroredLeafRoleComponent, ...]

    shared_leaf_family_names = CollectionAttributeProjection[str](
        "roles", "role_name"
    )
    left_leaf_classes = CollectionAttributeProjection[CompactIndexedClass](
        "roles", "left_class"
    )
    right_leaf_classes = CollectionAttributeProjection[CompactIndexedClass](
        "roles", "right_class"
    )

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.left_root.file_path,
                self.left_root.line,
                self.left_root.symbol,
            ),
            SourceLocation(
                self.right_root.file_path,
                self.right_root.line,
                self.right_root.symbol,
            ),
            *(
                SourceLocation(
                    indexed_class.file_path,
                    indexed_class.line,
                    indexed_class.symbol,
                )
                for indexed_class in (
                    *self.left_leaf_classes[:2],
                    *self.right_leaf_classes[:2],
                )
            ),
        )


@dataclass(frozen=True)
class ExactLeafMethodAncestorPromotionComponent:
    """One complete exact-method family and its existing nominal authority."""

    authority: CompactIndexedClass
    orbits: tuple[CompactExactMethodOrbit, ...]
    proof: ClosedLeafMethodAuthorityProof

    @property
    def file_path(self) -> str:
        return self.authority.file_path

    @property
    def line(self) -> int:
        return self.authority.line

    @property
    def authority_symbol(self) -> str:
        return self.authority.symbol

    @property
    def authority_name(self) -> str:
        return self.authority.qualname

    @property
    def authority_line(self) -> int:
        return self.authority.line

    @property
    def method_names(self) -> tuple[str, ...]:
        return self.proof.promoted_method_names

    @property
    def participant_class_symbols(self) -> tuple[str, ...]:
        return self.proof.participant_symbols

    @property
    def participant_class_names(self) -> tuple[str, ...]:
        return tuple(
            indexed_class.qualname for indexed_class in self.orbits[0].indexed_classes
        )

    @cached_property
    def method_symbols(self) -> tuple[str, ...]:
        return tuple(
            f"{indexed_class.qualname}.{orbit.method_name}"
            for orbit in self.orbits
            for indexed_class in orbit.indexed_classes
        )

    @property
    def file_paths(self) -> tuple[str, ...]:
        return (self.file_path,) * len(self.method_symbols)

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(method.line for orbit in self.orbits for method in orbit.methods)

    @property
    def line_count(self) -> int:
        return sum(
            method.line_count for orbit in self.orbits for method in orbit.methods
        )

    @property
    def statement_count(self) -> int:
        return sum(orbit.methods[0].statement_count for orbit in self.orbits)

    @property
    def method_line_count(self) -> int:
        return sum(orbit.methods[0].line_count for orbit in self.orbits)

    @cached_property
    def compression_certificate(self) -> CompressionCertificate:
        return ExistingAuthorityMethodPromotionCompressionProfile(
            class_count=len(self.participant_class_symbols),
            method_line_count=self.method_line_count,
        ).compression_certificate

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.file_path,
                self.authority_line,
                self.authority_symbol,
            ),
            *(
                SourceLocation(file_path, line, method_symbol)
                for file_path, line, method_symbol in zip(
                    self.file_paths,
                    self.line_numbers,
                    self.method_symbols,
                    strict=True,
                )
            ),
        )


@dataclass(frozen=True)
class ExactLeafMethodAncestorPromotionComponentBuilder:
    """Recover exact promotion components from one complete class projection."""

    projections: tuple[CompactModuleClassProjection, ...]
    class_index: CompactClassFamilyIndex

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> Self:
        return cls(
            projections=projections,
            class_index=(
                build_compact_class_family_index(projections)
                if class_index is None
                else class_index
            ),
        )

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        return cls.from_projections(
            tuple(
                CompactModuleClassProjectionFamily.collect(module)[0]
                for module in modules
            )
        )

    @cached_property
    def class_method_multiplicity(
        self,
    ) -> IdentityHandleMultiplicityProjection[tuple[str, str], CompactClassMethod]:
        """Derive unambiguous method declarations for every exact-method consumer."""

        return UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            (
                method
                for projection in self.projections
                for method in projection.class_methods
            ),
            lambda method: (method.class_symbol, method.method_name),
        )

    @cached_property
    def exact_method_orbits(self) -> tuple[CompactExactMethodOrbit, ...]:
        methods_by_role: dict[
            tuple[str, str, str],
            list[tuple[CompactIndexedClass, CompactClassMethod]],
        ] = defaultdict(list)
        methods = (
            self.class_method_multiplicity.unambiguous_declarations_by_handle.values()
        )
        for method in methods:
            source_digest = method.exact_promotion_source_digest
            if source_digest is None:
                continue
            indexed_class = self.class_index.class_for(method.class_symbol)
            if indexed_class is None or "." in indexed_class.qualname:
                continue
            methods_by_role[
                (
                    indexed_class.file_path,
                    method.method_name,
                    source_digest,
                )
            ].append((indexed_class, method))

        orbits = []
        for (
            _file_path,
            _method_name,
            _source_digest,
        ), class_methods in methods_by_role.items():
            if len(class_methods) < 2:
                continue
            ordered = tuple(
                sorted(class_methods, key=lambda item: (item[0].line, item[0].symbol))
            )
            indexed_classes = tuple(item[0] for item in ordered)
            methods = tuple(item[1] for item in ordered)
            orbit = CompactExactMethodOrbit.from_declarations(
                indexed_classes,
                methods,
            )
            if orbit is not None:
                orbits.append(orbit)
        return sorted_tuple(
            orbits,
            key=lambda orbit: (
                orbit.file_path,
                orbit.method_name,
                orbit.class_symbols,
            ),
        )

    @cached_property
    def assessed_components(
        self,
    ) -> tuple[ExactLeafMethodAncestorPromotionComponent, ...]:
        orbits_by_family: dict[
            tuple[str, tuple[str, ...]],
            list[CompactExactMethodOrbit],
        ] = defaultdict(list)
        for orbit in self.exact_method_orbits:
            common_direct_base_symbols = set.intersection(
                *(
                    set(indexed_class.resolved_base_symbols)
                    for indexed_class in orbit.indexed_classes
                )
            )
            if len(common_direct_base_symbols) != 1:
                continue
            authority_symbol = next(iter(common_direct_base_symbols))
            orbits_by_family[(authority_symbol, orbit.class_symbols)].append(orbit)

        components = []
        for (
            authority_symbol,
            _participant_symbols,
        ), family_orbits in orbits_by_family.items():
            orbits = tuple(sorted(family_orbits, key=lambda orbit: orbit.method_name))
            proof = self._authority_proof(authority_symbol, orbits)
            if proof is None:
                continue
            authority = self.class_index.class_for(authority_symbol)
            if authority is None:
                continue
            component = ExactLeafMethodAncestorPromotionComponent(
                authority=authority,
                orbits=orbits,
                proof=proof,
            )
            if component.compression_certificate.pays_rent:
                components.append(component)
        return sorted_tuple(
            components,
            key=lambda component: (
                component.file_path,
                component.authority_symbol,
                component.method_names,
            ),
        )

    @cached_property
    def proven_components(
        self,
    ) -> tuple[ExactLeafMethodAncestorPromotionComponent, ...]:
        return tuple(
            component
            for component in self.assessed_components
            if component.proof.is_proven
        )

    def required_proven_component(
        self,
        authority_symbol: str,
    ) -> ExactLeafMethodAncestorPromotionComponent:
        """Return the unique current component after proving its placement."""

        components = tuple(
            component
            for component in self.assessed_components
            if component.authority_symbol == authority_symbol
        )
        if len(components) != 1:
            raise ValueError(
                f"Authority {authority_symbol!r} has {len(components)} current "
                "exact leaf-method components"
            )
        component = components[0]
        if not component.proof.is_proven:
            raise ValueError(component.proof.rejection_reason)
        return component

    def _authority_proof(
        self,
        authority_symbol: str,
        orbits: tuple[CompactExactMethodOrbit, ...],
    ) -> ClosedLeafMethodAuthorityProof | None:
        authority = self.class_index.class_for(authority_symbol)
        if authority is None or "." in authority.qualname:
            return None
        participant_symbols = orbits[0].class_symbols
        if any(orbit.class_symbols != participant_symbols for orbit in orbits):
            return None
        participants = tuple(
            self.class_index.class_for(symbol) for symbol in participant_symbols
        )
        if any(participant is None for participant in participants):
            return None
        indexed_participants = tuple(
            participant for participant in participants if participant is not None
        )
        if any(
            participant.file_path != authority.file_path
            for participant in indexed_participants
        ):
            return None

        common_direct_base_symbols = tuple(
            sorted(
                set.intersection(
                    *(
                        set(participant.resolved_base_symbols)
                        for participant in indexed_participants
                    )
                )
            )
        )
        common_declared_nominal_base_names = tuple(
            sorted(
                set.intersection(
                    *(
                        {
                            base_name
                            for base_name in participant.declared_base_names
                            if ClassSymbolResolutionAuthority.establishes_nominal_family(
                                base_name
                            )
                        }
                        for participant in indexed_participants
                    )
                )
            )
        )
        authority_lineage_symbols = frozenset(
            (
                authority_symbol,
                *self.class_index.ancestor_symbols(authority_symbol),
            )
        )
        participant_ancestor_symbols = frozenset(
            ancestor_symbol
            for participant_symbol in participant_symbols
            for ancestor_symbol in self.class_index.ancestor_symbols(participant_symbol)
        )
        relevant_symbols = frozenset(
            (*participant_symbols, *participant_ancestor_symbols)
        )
        directly_rewritten_symbols = frozenset((authority_symbol, *participant_symbols))
        promoted_method_names = tuple(orbit.method_name for orbit in orbits)
        receiver_member_names = tuple(
            sorted(
                {
                    member_name
                    for orbit in orbits
                    for method in orbit.methods
                    for member_name in method.receiver_member_names
                }
            )
        )
        return ClosedLeafMethodAuthorityProof(
            authority_symbol=authority_symbol,
            authority_simple_name=authority.simple_name,
            participant_symbols=participant_symbols,
            common_direct_base_symbols=common_direct_base_symbols,
            common_declared_nominal_base_names=common_declared_nominal_base_names,
            authority_direct_child_symbols=self.class_index.children_by_symbol.get(
                authority_symbol,
                (),
            ),
            non_leaf_participant_symbols=tuple(
                symbol
                for symbol in participant_symbols
                if self.class_index.children_by_symbol.get(symbol)
            ),
            incompletely_resolved_symbols=tuple(
                sorted(
                    symbol
                    for symbol in relevant_symbols
                    if (indexed_class := self.class_index.class_for(symbol)) is not None
                    and not indexed_class.base_resolution_is_complete
                )
            ),
            method_ownership_sensitive_symbols=tuple(
                sorted(
                    symbol
                    for symbol in relevant_symbols
                    if (indexed_class := self.class_index.class_for(symbol)) is not None
                    and (
                        indexed_class.class_keyword_names
                        or indexed_class.declares_autoregister_meta
                        or bool(
                            self._declared_member_names(indexed_class)
                            & CLASS_METHOD_OWNERSHIP_HOOK_NAMES
                        )
                        or (
                            symbol in directly_rewritten_symbols
                            and not indexed_class.class_decorators_are_promotion_safe
                        )
                    )
                )
            ),
            authority_lineage_member_names=tuple(
                sorted(
                    {
                        member_name
                        for symbol in authority_lineage_symbols
                        if (indexed_class := self.class_index.class_for(symbol))
                        is not None
                        for member_name in self._declared_member_names(indexed_class)
                    }
                )
            ),
            competing_ancestor_member_names=tuple(
                sorted(
                    {
                        member_name
                        for symbol in (
                            participant_ancestor_symbols - authority_lineage_symbols
                        )
                        if (indexed_class := self.class_index.class_for(symbol))
                        is not None
                        for member_name in self._declared_member_names(indexed_class)
                    }
                )
            ),
            promoted_method_names=promoted_method_names,
            receiver_member_names=receiver_member_names,
        )

    @staticmethod
    def _declared_member_names(
        indexed_class: CompactIndexedClass,
    ) -> frozenset[str]:
        return frozenset(
            (
                *indexed_class.method_names,
                *(name for name, _value in indexed_class.direct_assignment_expressions),
            )
        )


def receiver_closed_exact_method_orbits(
    orbits: tuple[CompactExactMethodOrbit, ...],
) -> tuple[CompactExactMethodOrbit, ...]:
    """Keep the greatest method set with no undeclared receiver requirements."""

    orbit_by_name = {orbit.method_name: orbit for orbit in orbits}
    reverse_dependencies: dict[str, set[str]] = defaultdict(set)
    invalid_names: set[str] = set()
    for orbit in orbits:
        requirements = frozenset(
            member_name
            for method in orbit.methods
            for member_name in method.receiver_member_names
        )
        undeclared_requirements = requirements - orbit_by_name.keys()
        if undeclared_requirements:
            invalid_names.add(orbit.method_name)
        for requirement in requirements & orbit_by_name.keys():
            reverse_dependencies[requirement].add(orbit.method_name)

    pending = deque(invalid_names)
    while pending:
        invalid_name = pending.popleft()
        for dependent_name in reverse_dependencies[invalid_name]:
            if dependent_name in invalid_names:
                continue
            invalid_names.add(dependent_name)
            pending.append(dependent_name)
    return tuple(orbit for orbit in orbits if orbit.method_name not in invalid_names)


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyComponentBuilder:
    """Prove exact reusable role behavior across nominal domain families."""

    projections: tuple[CompactModuleClassProjection, ...]
    class_index: CompactClassFamilyIndex

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> Self:
        return cls(
            projections=projections,
            class_index=(
                build_compact_class_family_index(projections)
                if class_index is None
                else class_index
            ),
        )

    @cached_property
    def exact_method_builder(
        self,
    ) -> ExactLeafMethodAncestorPromotionComponentBuilder:
        return ExactLeafMethodAncestorPromotionComponentBuilder.from_projections(
            self.projections,
            class_index=self.class_index,
        )

    @cached_property
    def methods_by_identity(self) -> dict[tuple[str, str], CompactClassMethod]:
        multiplicity = self.exact_method_builder.class_method_multiplicity
        return multiplicity.unambiguous_declarations_by_handle

    @cached_property
    def roots(self) -> tuple[CompactIndexedClass, ...]:
        return sorted_tuple(
            (
                indexed_class
                for indexed_class in self.class_index.classes_by_symbol.values()
                if "_registered_types" in indexed_class.assignments_by_name
                and indexed_class.abstract_method_names
            ),
            key=lambda indexed_class: indexed_class.symbol,
        )

    def proven_components(
        self,
        *,
        min_shared_roles: int,
    ) -> tuple[ParallelMirroredLeafFamilyComponent, ...]:
        components = tuple(
            component
            for left_root, right_root in combinations(self.roots, 2)
            if (
                component := self._component_for_roots(
                    left_root,
                    right_root,
                    min_shared_roles=min_shared_roles,
                )
            )
            is not None
        )
        return sorted_tuple(
            components,
            key=lambda component: (
                component.left_root.symbol,
                component.right_root.symbol,
            ),
        )

    def _component_for_roots(
        self,
        left_root: CompactIndexedClass,
        right_root: CompactIndexedClass,
        *,
        min_shared_roles: int,
    ) -> ParallelMirroredLeafFamilyComponent | None:
        if (
            left_root.file_path != right_root.file_path
            or "." in left_root.qualname
            or "." in right_root.qualname
        ):
            return None
        shared_contract = sorted_tuple(
            set(left_root.abstract_method_names) & set(right_root.abstract_method_names)
        )
        if not shared_contract:
            return None
        root_tokens = (
            CLASS_NAME_ALGEBRA.ordered_tokens(left_root.simple_name),
            CLASS_NAME_ALGEBRA.ordered_tokens(right_root.simple_name),
        )
        shared_suffix = CLASS_NAME_ALGEBRA.longest_common_token_suffix(
            (left_root.simple_name, right_root.simple_name)
        )
        if not shared_suffix:
            return None
        axis_prefixes = tuple(
            tokens[: len(tokens) - len(shared_suffix)] for tokens in root_tokens
        )
        if not all(axis_prefixes) or axis_prefixes[0] == axis_prefixes[1]:
            return None
        left_roles = self._leaf_roles(left_root, axis_prefixes[0])
        right_roles = self._leaf_roles(right_root, axis_prefixes[1])
        if left_roles is None or right_roles is None:
            return None
        shared_role_names = sorted_tuple(set(left_roles) & set(right_roles))
        required_role_count = max(
            min_shared_roles,
            min(len(left_roles), len(right_roles)) // 2,
        )
        roles = tuple(
            role
            for role_name in shared_role_names
            if (
                role := self._role_component(
                    role_name,
                    left_roles[role_name],
                    right_roles[role_name],
                    shared_contract,
                )
            )
            is not None
        )
        if len(roles) < required_role_count:
            return None
        return ParallelMirroredLeafFamilyComponent(
            left_root=left_root,
            right_root=right_root,
            contract_method_names=shared_contract,
            roles=roles,
        )

    def _leaf_roles(
        self,
        root: CompactIndexedClass,
        axis_prefix_tokens: tuple[str, ...],
    ) -> dict[str, CompactIndexedClass] | None:
        descendants = tuple(
            descendant
            for symbol in self.class_index.descendant_symbols(root.symbol)
            if (descendant := self.class_index.class_for(symbol)) is not None
            if not descendant.is_abstract
            if descendant.file_path == root.file_path
            if "." not in descendant.qualname
            if descendant.base_resolution_is_complete
            if descendant.direct_base_count == 1
            if descendant.resolved_base_symbols == (root.symbol,)
        )
        declarations = tuple(
            (" ".join(role_tokens), descendant)
            for descendant in descendants
            if (tokens := CLASS_NAME_ALGEBRA.ordered_tokens(descendant.simple_name))
            if len(tokens) > len(axis_prefix_tokens)
            if tokens[: len(axis_prefix_tokens)] == axis_prefix_tokens
            if (role_tokens := tokens[len(axis_prefix_tokens) :])
        )
        multiplicity = UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            declarations,
            lambda declaration: declaration[0],
        )
        if multiplicity.ambiguous_handles:
            return None
        return {
            role_name: descendant
            for role_name, descendant in (
                multiplicity.unambiguous_declarations_by_handle.values()
            )
        }

    def _role_component(
        self,
        role_name: str,
        left_class: CompactIndexedClass,
        right_class: CompactIndexedClass,
        contract_method_names: tuple[str, ...],
    ) -> ExactMirroredLeafRoleComponent | None:
        method_orbits = tuple(
            orbit
            for method_name in contract_method_names
            if (
                orbit := self._method_orbit(
                    left_class,
                    right_class,
                    method_name,
                )
            )
            is not None
        )
        if len(method_orbits) != len(contract_method_names):
            return None
        if receiver_closed_exact_method_orbits(method_orbits) != method_orbits:
            return None
        return ExactMirroredLeafRoleComponent(
            role_name=role_name,
            left_class=left_class,
            right_class=right_class,
            method_orbits=method_orbits,
        )

    def _method_orbit(
        self,
        left_class: CompactIndexedClass,
        right_class: CompactIndexedClass,
        method_name: str,
    ) -> CompactExactMethodOrbit | None:
        methods = tuple(
            self.methods_by_identity.get((indexed_class.symbol, method_name))
            for indexed_class in (left_class, right_class)
        )
        if any(method is None for method in methods):
            return None
        exact_methods = tuple(method for method in methods if method is not None)
        return CompactExactMethodOrbit.from_declarations(
            (left_class, right_class),
            exact_methods,
        )
