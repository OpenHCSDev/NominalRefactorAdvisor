"""Declaration-derived proof components for exact method promotion."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import ClassVar, Self

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
from .semantic_algebra import UndirectedGraph, VertexIndexEdge
from .semantic_description_length import (
    CompressionCertificate,
    ExactMethodRoleCompressionProfile,
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
class ExactMethodOrbitComponent:
    """Shared declaration-derived surface of one exact-method orbit cohort."""

    orbits: tuple[CompactExactMethodOrbit, ...]

    def __post_init__(self) -> None:
        if not self.orbits:
            raise ValueError("Exact-method component requires at least one method orbit")
        cohort = (self.orbits[0].file_path, self.orbits[0].class_symbols)
        if any(
            (orbit.file_path, orbit.class_symbols) != cohort for orbit in self.orbits
        ):
            raise ValueError("Exact-method component orbits must share one class cohort")
        method_names = tuple(orbit.method_name for orbit in self.orbits)
        if method_names != tuple(sorted(set(method_names))):
            raise ValueError("Exact-method component orbits must be uniquely ordered")

    @property
    def file_path(self) -> str:
        return self.orbits[0].file_path

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(orbit.method_name for orbit in self.orbits)

    @property
    def participant_class_symbols(self) -> tuple[str, ...]:
        return self.orbits[0].class_symbols

    @property
    def participant_class_names(self) -> tuple[str, ...]:
        return tuple(
            indexed_class.qualname for indexed_class in self.orbits[0].indexed_classes
        )

    @property
    def method_symbols(self) -> tuple[str, ...]:
        return tuple(
            f"{indexed_class.qualname}.{orbit.method_name}"
            for orbit in self.orbits
            for indexed_class in orbit.indexed_classes
        )

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


@dataclass(frozen=True)
class ExactMethodRoleComponent(ExactMethodOrbitComponent):
    """One maximal exact-method role repeated without an existing authority."""

    @property
    def line(self) -> int:
        return min(self.line_numbers)

    @cached_property
    def compression_certificate(self) -> CompressionCertificate:
        return ExactMethodRoleCompressionProfile(
            class_count=len(self.participant_class_names),
            method_line_count=self.method_line_count,
        ).compression_certificate

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(self.file_path, line, method_symbol)
            for line, method_symbol in zip(
                self.line_numbers,
                self.method_symbols,
                strict=True,
            )
        )


@dataclass(frozen=True)
class ExactMirroredLeafRoleComponent:
    """One role whose domain leaves carry the same movable implementations."""

    role_name: str
    classes: tuple[CompactIndexedClass, ...]
    method_orbits: tuple[CompactExactMethodOrbit, ...]

    @property
    def authority_name(self) -> str:
        return CLASS_NAME_ALGEBRA.pascal_identifier(self.role_name)


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyComponent:
    """Exact role implementations repeated across nominal domain axes."""

    roots: tuple[CompactIndexedClass, ...]
    contract_method_names: tuple[str, ...]
    roles: tuple[ExactMirroredLeafRoleComponent, ...]

    root_symbols = CollectionAttributeProjection[str]("roots", "symbol")
    shared_leaf_family_names = CollectionAttributeProjection[str](
        "roles", "role_name"
    )

    def __post_init__(self) -> None:
        if len(self.roots) < 2:
            raise ValueError("Parallel leaf family requires multiple nominal roots")
        if len(frozenset(root.file_path for root in self.roots)) != 1:
            raise ValueError("Parallel leaf-family roots must share one source file")
        if not self.roles:
            raise ValueError("Parallel leaf family requires at least one exact role")
        if any(len(role.classes) != len(self.roots) for role in self.roles):
            raise ValueError(
                "Every parallel role must span the complete root product"
            )

    @property
    def file_path(self) -> str:
        """Return the source file shared by the proven parallel family."""

        return self.roots[0].file_path

    @cached_property
    def leaf_classes(self) -> tuple[CompactIndexedClass, ...]:
        return tuple(
            indexed_class for role in self.roles for indexed_class in role.classes
        )

    @cached_property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(
                root.file_path,
                root.line,
                root.symbol,
            )
            for root in self.roots
        ) + tuple(
            SourceLocation(
                indexed_class.file_path,
                indexed_class.line,
                indexed_class.symbol,
            )
            for indexed_class in self.leaf_classes
        )


@dataclass(frozen=True)
class ExactLeafMethodAncestorPromotionComponent(ExactMethodOrbitComponent):
    """One complete exact-method family and its existing nominal authority."""

    authority: CompactIndexedClass
    proof: ClosedLeafMethodAuthorityProof

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
                SourceLocation(self.file_path, line, method_symbol)
                for line, method_symbol in zip(
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
            CompactModuleClassProjectionFamily.collect_modules(modules)
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
class ExactMethodRoleComponentBuilder:
    """Prove maximal exact-method roles that still need a nominal authority."""

    exact_method_builder: ExactLeafMethodAncestorPromotionComponentBuilder

    @property
    def class_index(self) -> CompactClassFamilyIndex:
        return self.exact_method_builder.class_index

    @cached_property
    def proven_components(self) -> tuple[ExactMethodRoleComponent, ...]:
        orbits_by_cohort: dict[
            tuple[str, tuple[str, ...]],
            list[CompactExactMethodOrbit],
        ] = defaultdict(list)
        for orbit in self.exact_method_builder.exact_method_orbits:
            if self._has_existing_or_unsafe_authority(orbit):
                continue
            orbits_by_cohort[(orbit.file_path, orbit.class_symbols)].append(orbit)

        components = tuple(
            component
            for cohort_orbits in orbits_by_cohort.values()
            if (
                closed_orbits := receiver_closed_exact_method_orbits(
                    tuple(sorted(cohort_orbits, key=lambda orbit: orbit.method_name))
                )
            )
            if (
                component := ExactMethodRoleComponent(orbits=closed_orbits)
            ).compression_certificate.pays_rent
        )
        return sorted_tuple(
            components,
            key=lambda component: (
                component.file_path,
                component.line,
                component.method_names,
                component.participant_class_names,
            ),
        )

    def required_component_for_method(
        self,
        *,
        file_path: str,
        method_qualname: str,
    ) -> ExactMethodRoleComponent:
        components = tuple(
            component
            for component in self.proven_components
            if component.file_path == file_path
            and method_qualname in component.method_symbols
        )
        if len(components) != 1:
            raise ValueError(
                f"Method {method_qualname!r} belongs to {len(components)} current "
                "exact-method role components"
            )
        return components[0]

    def _has_existing_or_unsafe_authority(
        self,
        orbit: CompactExactMethodOrbit,
    ) -> bool:
        indexed_classes = orbit.indexed_classes
        participant_symbols = frozenset(orbit.class_symbols)
        ancestor_sets = tuple(
            frozenset(self.class_index.ancestor_symbols(indexed_class.symbol))
            for indexed_class in indexed_classes
        )
        if any(participant_symbols & ancestors for ancestors in ancestor_sets):
            return True
        if any(
            indexed_class.class_keyword_names
            or indexed_class.declares_autoregister_meta
            or not indexed_class.class_decorators_are_promotion_safe
            or not indexed_class.class_header_is_reconstructible
            for indexed_class in indexed_classes
        ):
            return True
        if any(
            ancestor.class_keyword_names or ancestor.declares_autoregister_meta
            for ancestors in ancestor_sets
            for ancestor_symbol in ancestors
            if (ancestor := self.class_index.class_for(ancestor_symbol)) is not None
        ):
            return True
        if ancestor_sets and set.intersection(*(set(item) for item in ancestor_sets)):
            return True
        shared_declared_nominal_bases = set.intersection(
            *(
                {
                    base_name
                    for base_name in indexed_class.declared_base_names
                    if ClassSymbolResolutionAuthority.establishes_nominal_family(
                        base_name
                    )
                }
                for indexed_class in indexed_classes
            )
        )
        if shared_declared_nominal_bases:
            return True
        return any(
            orbit.method_name in ancestor.method_names
            for ancestor_symbols in ancestor_sets
            for ancestor_symbol in ancestor_symbols
            if (ancestor := self.class_index.class_for(ancestor_symbol)) is not None
        )


@dataclass(frozen=True)
class ParallelMirroredLeafFamilyComponentBuilder:
    """Prove exact reusable role behavior across nominal domain families."""

    minimum_product_role_count: ClassVar[int] = 3
    exact_method_builder: ExactLeafMethodAncestorPromotionComponentBuilder

    @property
    def class_index(self) -> CompactClassFamilyIndex:
        return self.exact_method_builder.class_index

    @classmethod
    def from_projections(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        *,
        class_index: CompactClassFamilyIndex | None = None,
    ) -> Self:
        return cls(
            exact_method_builder=ExactLeafMethodAncestorPromotionComponentBuilder.from_projections(
                projections,
                class_index=class_index,
            ),
        )

    @classmethod
    def from_modules(cls, modules: tuple[ParsedModule, ...]) -> Self:
        return cls.from_projections(
            CompactModuleClassProjectionFamily.collect_modules(modules)
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
        root_graph = self.root_compatibility_graph(
            min_shared_roles=min_shared_roles,
        )
        components = tuple(
            component
            for roots in root_graph.clique_components
            if len(roots) > 1
            if (
                component := self._component_for_roots(
                    roots,
                    min_shared_roles=min_shared_roles,
                )
            )
            is not None
        )
        return sorted_tuple(
            components,
            key=lambda component: component.root_symbols,
        )

    def root_compatibility_graph(
        self,
        *,
        min_shared_roles: int,
    ) -> UndirectedGraph[CompactIndexedClass]:
        """Derive exact pair compatibility before choosing maximal products."""

        return UndirectedGraph(
            vertices=self.roots,
            edges=tuple(
                VertexIndexEdge.from_indices(left_index, right_index)
                for left_index, right_index in combinations(
                    range(len(self.roots)),
                    2,
                )
                if self._component_for_roots(
                    (self.roots[left_index], self.roots[right_index]),
                    min_shared_roles=min_shared_roles,
                )
                is not None
            ),
        )

    def required_proven_component(
        self,
        root_symbol: str,
    ) -> ParallelMirroredLeafFamilyComponent:
        components = tuple(
            component
            for component in self.default_proven_components
            if root_symbol in component.root_symbols
        )
        if len(components) != 1:
            raise ValueError(
                f"Root {root_symbol!r} has {len(components)} current exact "
                "parallel leaf-family components"
            )
        return components[0]

    @cached_property
    def default_proven_components(
        self,
    ) -> tuple[ParallelMirroredLeafFamilyComponent, ...]:
        """Cache the production-threshold proof set for source reproof consumers."""

        return self.proven_components(
            min_shared_roles=self.minimum_product_role_count,
        )

    def _component_for_roots(
        self,
        roots: tuple[CompactIndexedClass, ...],
        *,
        min_shared_roles: int,
    ) -> ParallelMirroredLeafFamilyComponent | None:
        if len(roots) < 2:
            return None
        if len(frozenset(root.file_path for root in roots)) != 1:
            return None
        if any("." in root.qualname for root in roots):
            return None
        shared_contract = sorted_tuple(
            set.intersection(
                *(set(root.abstract_method_names) for root in roots)
            )
        )
        if not shared_contract:
            return None
        root_tokens = tuple(
            CLASS_NAME_ALGEBRA.ordered_tokens(root.simple_name) for root in roots
        )
        shared_suffix = CLASS_NAME_ALGEBRA.longest_common_token_suffix(
            tuple(root.simple_name for root in roots)
        )
        if not shared_suffix:
            return None
        axis_prefixes = tuple(
            tokens[: len(tokens) - len(shared_suffix)] for tokens in root_tokens
        )
        if not all(axis_prefixes) or len(frozenset(axis_prefixes)) != len(roots):
            return None
        role_maps = tuple(
            self._leaf_roles(root, axis_prefix)
            for root, axis_prefix in zip(roots, axis_prefixes, strict=True)
        )
        if any(role_map is None for role_map in role_maps):
            return None
        exact_role_maps = tuple(
            role_map for role_map in role_maps if role_map is not None
        )
        shared_role_names = sorted_tuple(
            set.intersection(*(set(role_map) for role_map in exact_role_maps))
        )
        required_role_count = max(
            min_shared_roles,
            min(len(role_map) for role_map in exact_role_maps) // 2,
        )
        roles = tuple(
            role
            for role_name in shared_role_names
            if (
                role := self._role_component(
                    role_name,
                    tuple(role_map[role_name] for role_map in exact_role_maps),
                    shared_contract,
                )
            )
            is not None
        )
        if len(roles) < required_role_count:
            return None
        if len(frozenset(role.authority_name for role in roles)) != len(roles):
            return None
        return ParallelMirroredLeafFamilyComponent(
            roots=roots,
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
            if descendant.class_header_is_reconstructible
            if descendant.class_decorators_are_promotion_safe
            if not descendant.class_keyword_names
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
        classes: tuple[CompactIndexedClass, ...],
        contract_method_names: tuple[str, ...],
    ) -> ExactMirroredLeafRoleComponent | None:
        method_orbits = tuple(
            orbit
            for method_name in contract_method_names
            if (
                orbit := self._method_orbit(
                    classes,
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
            classes=classes,
            method_orbits=method_orbits,
        )

    def _method_orbit(
        self,
        classes: tuple[CompactIndexedClass, ...],
        method_name: str,
    ) -> CompactExactMethodOrbit | None:
        methods = tuple(
            self.methods_by_identity.get((indexed_class.symbol, method_name))
            for indexed_class in classes
        )
        if any(method is None for method in methods):
            return None
        exact_methods = tuple(method for method in methods if method is not None)
        return CompactExactMethodOrbit.from_declarations(
            classes,
            exact_methods,
        )
