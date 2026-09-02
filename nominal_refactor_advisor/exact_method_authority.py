"""Declaration-derived proof components for exact method promotion."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
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
from .collection_algebra import sorted_tuple
from .models import SourceLocation
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

    @property
    def class_symbols(self) -> tuple[str, ...]:
        return tuple(indexed_class.symbol for indexed_class in self.indexed_classes)


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
    def exact_method_orbits(self) -> tuple[CompactExactMethodOrbit, ...]:
        methods_by_role: dict[
            tuple[str, str, str],
            list[tuple[CompactIndexedClass, CompactClassMethod]],
        ] = defaultdict(list)
        for projection in self.projections:
            for method in projection.class_methods:
                if method.exact_source_digest is None or method.promotion_hazards:
                    continue
                indexed_class = self.class_index.class_for(method.class_symbol)
                if indexed_class is None or "." in indexed_class.qualname:
                    continue
                methods_by_role[
                    (
                        projection.file_path,
                        method.method_name,
                        method.exact_source_digest,
                    )
                ].append((indexed_class, method))

        orbits = []
        for (
            file_path,
            method_name,
            _source_digest,
        ), class_methods in methods_by_role.items():
            if len(class_methods) < 2:
                continue
            ordered = tuple(
                sorted(class_methods, key=lambda item: (item[0].line, item[0].symbol))
            )
            indexed_classes = tuple(item[0] for item in ordered)
            methods = tuple(item[1] for item in ordered)
            if len(frozenset(method.line_count for method in methods)) != 1:
                continue
            orbits.append(
                CompactExactMethodOrbit(
                    file_path=file_path,
                    method_name=method_name,
                    indexed_classes=indexed_classes,
                    methods=methods,
                )
            )
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
