from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import cached_property
from operator import attrgetter
from typing import Callable, Self, Sequence, TypeVar, cast, overload

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.class_index import CompactModuleClassProjection
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.models import (
    EvidenceLocated,
    LineWitness,
    MappingMetrics,
    SourceLineReference,
    SourceLocation,
)
from nominal_refactor_advisor.registry_identity import ClassNameKeyed
from nominal_refactor_advisor.semantic_shape_algebra import InjectiveTypeRegistryProof

AttributeValueT = TypeVar("AttributeValueT")


def _attribute_projection(attribute_name: str) -> Callable[[object], AttributeValueT]:
    return cast(Callable[[object], AttributeValueT], attrgetter(attribute_name))


_DEFAULT_FILE_PATH_ATTRIBUTE = "file_path"


@dataclass(frozen=True)
class SourceLocationEvidenceProperty:
    file_attribute_name: str = _DEFAULT_FILE_PATH_ATTRIBUTE
    line_attribute_name: str = "line"
    symbol_attribute_name: str = "symbol"

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(
        self, instance: object, owner: type[object] | None = None
    ) -> SourceLocation: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> SourceLocation | Self:
        del owner
        if instance is None:
            return self
        return SourceLocation(
            _attribute_projection(self.file_attribute_name)(instance),
            _attribute_projection(self.line_attribute_name)(instance),
            _attribute_projection(self.symbol_attribute_name)(instance),
        )


@dataclass(frozen=True)
class LineWitnessCandidate(
    LineWitness,
    EvidenceLocated,
    ABC,
    ClassNameKeyed,
    metaclass=AutoRegisterMeta,
):
    """A candidate that sits at one line; a member with more sites overrides `evidence`."""

    @property
    def witness_name(self) -> str:
        return type(self).__name__

    @cached_property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (self.location,)


@dataclass(frozen=True)
class WitnessCarrierCandidate(LineWitnessCandidate):
    subject_name: str
    name_family: tuple[str, ...]
    witness_name = AliasProperty[str]("subject_name")
    class_name = AliasProperty[str]("subject_name")


@dataclass(frozen=True)
class ClassNameLineNumbersGroup(
    EvidenceLocated, ABC, ClassNameKeyed, metaclass=AutoRegisterMeta
):
    class_names: tuple[str, ...]
    line_numbers: tuple[int, ...]


@dataclass(frozen=True)
class ClassMethodFamilyCandidate(
    EvidenceLocated, ABC, ClassNameKeyed, metaclass=AutoRegisterMeta
):
    file_path: str
    class_name: str
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, f"{self.class_name}.{method_name}")
                for method_name, line in zip(
                    self.method_names, self.line_numbers, strict=True
                )
            )
        )


class KeyedRecordInfrastructureCandidate(EvidenceLocated, ABC):
    """Candidate that owns its keyed-record surface presentation."""

    @property
    @abstractmethod
    def finding_summary(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def finding_metrics(self) -> MappingMetrics:
        raise NotImplementedError


@dataclass(frozen=True)
class KeyedRegistryAxisSurface:
    key_type_name: str
    registry_key_attr_name: str
    lookup_method_names: tuple[str, ...]
    registered_case_names: tuple[str, ...]
    consumer_symbols: tuple[str, ...]


@dataclass(frozen=True)
class KeyedRegistryAxisFact(KeyedRegistryAxisSurface, SourceLineReference):
    """Neutral source observation; unresolved keys are not counterexamples."""

    class_name: str
    missing_maturity_signals: tuple[str, ...]
    injectivity_proof: InjectiveTypeRegistryProof | None

    def require_proof(self) -> InjectiveTypeRegistryProof:
        if self.injectivity_proof is None:
            raise ValueError("Registry key equality remains unproved")
        return self.injectivity_proof

    @property
    def is_mature_injective(self) -> bool:
        proof = self.injectivity_proof
        return (
            proof is not None
            and not self.missing_maturity_signals
            and proof.is_injective
        )


class CollectsItselfFromRegistryFacts(ABC):
    """A candidate type that finds its candidates in the keyed-registry facts the class
    repository indexes. A registry detector over it has no candidate method to write."""

    @classmethod
    @abstractmethod
    def from_registry_facts(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        facts: Sequence[KeyedRegistryAxisFact],
    ) -> tuple[Self, ...]:
        raise NotImplementedError


class KeyedRegistryFactCandidate(CollectsItselfFromRegistryFacts):
    """Nominal refinement of keyed-registry facts selected by one invariant."""

    @classmethod
    @abstractmethod
    def accepts_fact(cls, fact: KeyedRegistryAxisFact) -> bool:
        raise NotImplementedError

    @classmethod
    def from_registry_facts(
        cls,
        projections: Sequence[CompactModuleClassProjection],
        facts: Sequence[KeyedRegistryAxisFact],
    ) -> tuple[Self, ...]:
        del projections
        field_names = tuple(item.name for item in fields(cls))
        return tuple(
            cls(**{name: getattr(fact, name) for name in field_names})
            for fact in facts
            if cls.accepts_fact(fact)
        )
