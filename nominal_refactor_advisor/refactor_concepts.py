from __future__ import annotations

from abc import ABC

from .collection_algebra import UniqueIdentityIndexAuthority
from .registry_identity import class_name_registry_key
from .semantic_match import loaded_nominal_descendants


class RefactorConcept(ABC):
    """Nominal refactor semantics inherited by executable declarations."""

    @classmethod
    def concept_key(cls) -> str:
        return class_name_registry_key(cls.__name__.removesuffix("Concept"), cls)

    @classmethod
    def declaration_types(cls) -> tuple[type["RefactorConcept"], ...]:
        """Return pure concept declarations without cataloging execution classes."""

        descendants = frozenset(loaded_nominal_descendants(cls))
        declarations: set[type[RefactorConcept]] = {cls}
        while True:
            discovered = {
                candidate
                for candidate in descendants
                if candidate not in declarations
                and all(base in declarations for base in candidate.__bases__)
            }
            if not discovered:
                break
            declarations.update(discovered)
        declarations_by_key = UniqueIdentityIndexAuthority.declarations_by_handle(
            declarations,
            lambda declaration: declaration.concept_key(),
        )
        return tuple(declarations_by_key[key] for key in sorted(declarations_by_key))

    @classmethod
    def declaration_for_key(cls, key: str) -> type["RefactorConcept"]:
        """Resolve one exact declaration from the declaration-derived key view."""

        declarations_by_key = UniqueIdentityIndexAuthority.declarations_by_handle(
            cls.declaration_types(),
            lambda declaration: declaration.concept_key(),
        )
        try:
            return declarations_by_key[key]
        except KeyError as error:
            raise ValueError(f"Unknown refactor concept {key!r}") from error

    @classmethod
    def leaf_concept_for_declaration(
        cls,
        declaration_type: type["RefactorConcept"],
    ) -> type["RefactorConcept"]:
        concepts = tuple(
            concept
            for concept in cls.declaration_types()
            if issubclass(declaration_type, concept)
        )
        leaves = tuple(
            concept
            for concept in concepts
            if not any(
                other is not concept and issubclass(other, concept)
                for other in concepts
            )
        )
        if len(leaves) != 1:
            raise TypeError(
                f"{declaration_type.__name__} must inherit exactly one leaf "
                "RefactorConcept"
            )
        return leaves[0]


class NominalBoundaryConcept(RefactorConcept):
    """Select SSOT authority-boundary findings for nominal extraction."""


class SemanticCarrierConcept(NominalBoundaryConcept):
    """Replace structurally repeated data movement with nominal ownership."""


class CallMappingAuthorityConcept(NominalBoundaryConcept):
    """Move repeated call argument mapping behind its nominal owner."""


class ConstructorKwargCollapseConcept(
    SemanticCarrierConcept,
    CallMappingAuthorityConcept,
):
    """Collapse repeated constructor keyword projections behind an authority."""


class ConstructorKwargCarrierProjectionConcept(ConstructorKwargCollapseConcept):
    """Derive constructor keywords through a nominal carrier authority."""


class TupleDictReturnNominalizationConcept(SemanticCarrierConcept):
    """Replace anonymous tuple or mapping results with nominal ownership."""


class DataclassPayloadProjectionConcept(TupleDictReturnNominalizationConcept):
    """Derive payload items from a dataclass declaration."""


class DerivedProjectionConcept(NominalBoundaryConcept):
    """Derive a repeated projection from its existing nominal authority."""


class ClassFamilyAuthorityConcept(NominalBoundaryConcept):
    """Establish a class-family authority for shared behavior or collection views."""


class AutoRegisterConcept(ClassFamilyAuthorityConcept):
    """Replace registration mirrors with nominal automatic registration."""


class AutoRegisterClassRegistryConcept(AutoRegisterConcept):
    """Derive a class registry from registered class declarations."""


class AutoRegisterStrategyFamilyConcept(AutoRegisterConcept):
    """Replace closed dispatch with an automatically registered strategy family."""


class AutoRegisterMroOrderingConcept(AutoRegisterConcept):
    """Derive registered-family precedence from a declared MRO composition."""
