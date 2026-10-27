"""Census of which consumer functions know the members of a family.

A consumer that names a family's members (a type test, a class pattern, a
case comparison) restates the family outside its declaration. The census counts
those consumers per family, groups them by the exact member set they name, and
separates the ones that also call an owner of that member set from the ones
that hand-roll the question.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Self

from .ast_projection import ExpressionReference
from .ast_tools import ParsedModule, module_syntax_index, parse_python_modules
from .class_index import (
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjectionFamily,
)
from .detectors import DetectorConfig
from .detectors._base import CompactClassRepositoryContext
from .detectors._systemic import (
    declared_cases,
    recovered_enum_sites,
    unique_direct_enum_classes,
)
from .declared_family import ForeignAdapter
from .foreign_family import ForeignFamily
from .json_reports import SemanticRecord, json_report_property
from .observation_families import ForeignTypeTestModuleFamily, TypeArgumentResolver
from .observation_shapes import ForeignTypeTestShape


class CensusFamilyKind(StrEnum):
    """Which declaration a family's members are read from."""

    FOREIGN_MODULE = "foreign_module"
    ENUM = "enum"


@dataclass(frozen=True)
class CensusConsumer(SemanticRecord):
    """One function that names members of a family."""

    file_path: str
    qualname: str
    owners_called: tuple[str, ...]


@dataclass(frozen=True)
class MemberSetGroup(SemanticRecord):
    """Consumers of one family that name exactly the same member set."""

    kind: CensusFamilyKind
    family: str
    members: tuple[str, ...]
    covering_owners: tuple[str, ...]
    owner_using: tuple[CensusConsumer, ...]
    bypassing: tuple[CensusConsumer, ...]

    @json_report_property()
    def consumer_count(self) -> int:
        return len(self.owner_using) + len(self.bypassing)

    @json_report_property()
    def owner_using_count(self) -> int:
        return len(self.owner_using)

    @json_report_property()
    def bypassing_count(self) -> int:
        return len(self.bypassing)


@dataclass(frozen=True)
class OwnerCensus(SemanticRecord):
    """How many consumers name any member an owner covers, and use it."""

    kind: CensusFamilyKind
    owner: str
    covered_members: tuple[str, ...]
    consumers: int
    owner_using: int

    @json_report_property()
    def bypassing(self) -> int:
        return self.consumers - self.owner_using


@dataclass(frozen=True)
class MemberAwarenessCensus(SemanticRecord):
    """Every family's consumers by exact member set, and owner use."""

    scanned_module_count: int
    single_member_consumers: int
    groups: tuple[MemberSetGroup, ...]
    owners: tuple[OwnerCensus, ...]

    @json_report_property()
    def branching_consumers(self) -> int:
        return sum(group.consumer_count for group in self.groups)

    @json_report_property()
    def distinct_member_sets(self) -> int:
        return len({(group.kind, group.members) for group in self.groups})


@dataclass(frozen=True)
class CalledMethod:
    """One ``receiver.method(...)`` call inside a consumer."""

    receiver_chain: tuple[str, ...]
    receiver_qualified_name: str | None
    method_name: str


@dataclass(frozen=True)
class MemberOwner(ABC):
    """A declaration that answers questions about a member set."""

    kind: CensusFamilyKind
    name: str
    covered: frozenset[str]
    method_names: frozenset[str]
    definitions: frozenset[tuple[str, str]]

    def defines(self, module_name: str, symbol: str) -> bool:
        """Whether the symbol sits inside a class that is this owner's own code."""

        return any(
            module_name == owner_module
            and (symbol == qualname or symbol.startswith(f"{qualname}."))
            for owner_module, qualname in self.definitions
        )

    def covers(self, members: frozenset[str]) -> bool:
        return members <= self.covered

    def names_any(self, members: frozenset[str]) -> bool:
        return bool(members & self.covered)

    @abstractmethod
    def called_by(self, call: CalledMethod) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ClassOwner(MemberOwner):
    """A declared family, adapter or projection reached through its class name."""

    aliases: frozenset[str]

    def called_by(self, call: CalledMethod) -> bool:
        return (
            call.method_name in self.method_names
            and call.receiver_qualified_name in self.aliases
        )


@dataclass(frozen=True)
class EnumOwner(MemberOwner):
    """An enum, whose own methods answer questions about its cases."""

    simple_name: str

    def called_by(self, call: CalledMethod) -> bool:
        return (
            call.method_name in self.method_names
            and self.simple_name in call.receiver_chain
        )


@dataclass(frozen=True)
class MemberObservation:
    """One function's naming of members of one family."""

    kind: CensusFamilyKind
    family: str
    file_path: str
    symbol: str
    subject: str
    members: frozenset[str]


@dataclass(frozen=True)
class CensusCorpus:
    """The parsed modules every source reads."""

    modules: tuple[ParsedModule, ...]

    @cached_property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.module_name for module in self.modules)

    @cached_property
    def module_name_by_file(self) -> dict[str, str]:
        return {module.file_path: module.module_name for module in self.modules}

    @cached_property
    def resolvers(self) -> dict[str, TypeArgumentResolver]:
        return {
            module.file_path: TypeArgumentResolver(module) for module in self.modules
        }

    def aliases_of(self, import_path: str) -> frozenset[str]:
        """The import path and every scanned name that re-exports it."""

        aliases = {import_path}
        grew = True
        while grew:
            grew = False
            for module in self.modules:
                snapshot = self.resolvers[module.file_path].snapshot
                for name, binding in snapshot.bindings_by_name.items():
                    alias = f"{module.module_name}.{name}"
                    if binding.qualified_name in aliases and alias not in aliases:
                        aliases.add(alias)
                        grew = True
        return frozenset(aliases)


class MemberCensusSource(ABC):
    """One way a family's members are named, with the owners declared for them."""

    def __init__(self, corpus: CensusCorpus, config: DetectorConfig) -> None:
        self.corpus = corpus
        self.config = config

    @abstractmethod
    def owners(self) -> tuple[MemberOwner, ...]:
        raise NotImplementedError

    @abstractmethod
    def observations(self) -> Iterator[MemberObservation]:
        raise NotImplementedError


def _own_definitions(owner: type) -> frozenset[tuple[str, str]]:
    """The owner and every class below it, as (module, qualname): the decoder code."""

    def below(klass: type) -> Iterator[type]:
        yield klass
        for child in klass.__subclasses__():
            yield from below(child)

    return frozenset((klass.__module__, klass.__qualname__) for klass in below(owner))


def _matched_subject(ancestors: Sequence[ast.AST]) -> str:
    """The value a class pattern tests: the match subject, or a part of it.

    A pattern nested inside another class, sequence or mapping pattern tests an
    attribute or element of the subject, so it is keyed apart from the top-level
    patterns of the same match.
    """

    parts: list[str] = []
    for ancestor in reversed(ancestors):
        match ancestor:
            case ast.MatchClass() | ast.MatchSequence() | ast.MatchMapping():
                parts.append(f"{ancestor.lineno}:{ancestor.col_offset}")
            case ast.Match(subject=subject):
                return ast.dump(subject) + "".join(f"@{part}" for part in parts)
    return "@".join(parts)


def _public_method_names(owner: type) -> frozenset[str]:
    """Names of the public callables a declaration exposes, from its MRO."""

    return frozenset(
        name
        for klass in owner.__mro__
        for name, value in vars(klass).items()
        if not name.startswith("_")
        if isinstance(value, (classmethod, staticmethod, property))
        or callable(value)
    )


class ForeignModuleSource(MemberCensusSource):
    """Classes of foreign modules named by type tests and class patterns."""

    def owners(self) -> tuple[MemberOwner, ...]:
        return tuple(
            ClassOwner(
                kind=CensusFamilyKind.FOREIGN_MODULE,
                name=family.import_path(),
                covered=family.member_names(),
                method_names=_public_method_names(family),
                aliases=self.corpus.aliases_of(family.import_path()),
                definitions=_own_definitions(family),
            )
            for root in (ForeignFamily, ForeignAdapter)
            for family in root.concrete_families()
        )

    def observations(self) -> Iterator[MemberObservation]:
        for module in self.corpus.modules:
            for shape in self._shapes(module):
                yield MemberObservation(
                    kind=CensusFamilyKind.FOREIGN_MODULE,
                    family=shape.owner,
                    file_path=shape.file_path,
                    symbol=shape.symbol,
                    subject=shape.subject,
                    members=frozenset(shape.type_names),
                )

    def _shapes(self, module: ParsedModule) -> Iterator[ForeignTypeTestShape]:
        tested = ForeignTypeTestModuleFamily.collect(module)[0].sites
        yield from (
            shape
            for shape in chain(tested, self._pattern_shapes(module))
            if shape.is_foreign_to(self.corpus.module_names)
        )

    def _pattern_shapes(self, module: ParsedModule) -> Iterator[ForeignTypeTestShape]:
        index = module_syntax_index(module.module)
        resolver = self.corpus.resolvers[module.file_path]
        for node_index, pattern in index.indexed_nodes_of_type(ast.MatchClass):
            ancestors = index.ancestor_nodes(node_index)
            name = resolver.qualified_class_at(pattern.cls, ancestors)
            owner = "" if name is None else name.rpartition(".")[0]
            if name is not None and owner:
                yield ForeignTypeTestShape(
                    file_path=module.file_path,
                    lineno=pattern.lineno,
                    symbol=index.lexical_symbol(node_index),
                    owner=owner,
                    type_names=(name,),
                    subject=_matched_subject(ancestors),
                )


class EnumSource(MemberCensusSource):
    """Enum cases recovered by branches outside the enum's declaration."""

    @cached_property
    def projections(self) -> tuple:
        return CompactModuleClassProjectionFamily.collect_modules(self.corpus.modules)

    @cached_property
    def class_index(self) -> CompactClassFamilyIndex:
        return CompactClassRepositoryContext.from_projections(
            self.projections, self.config
        ).class_index

    @cached_property
    def enum_classes(self) -> dict[str, CompactIndexedClass]:
        return unique_direct_enum_classes(self.class_index)

    def owners(self) -> tuple[MemberOwner, ...]:
        return tuple(
            EnumOwner(
                kind=CensusFamilyKind.ENUM,
                name=enum_class.symbol,
                covered=frozenset(declared_cases(enum_class)),
                method_names=frozenset(enum_class.method_names),
                simple_name=enum_class.simple_name,
                definitions=frozenset(),
            )
            for enum_class in self.enum_classes.values()
        )

    def observations(self) -> Iterator[MemberObservation]:
        for projection in self.projections:
            for function in projection.closed_axis_branch_functions:
                for enum_name, site in recovered_enum_sites(
                    self.class_index, self.enum_classes, function
                ).items():
                    enum_class = self.enum_classes[enum_name]
                    inside_declaration = (
                        function.file_path == enum_class.file_path
                        and function.qualname.startswith(f"{enum_class.qualname}.")
                    )
                    if site.case_names and not inside_declaration:
                        yield MemberObservation(
                            kind=CensusFamilyKind.ENUM,
                            family=enum_class.symbol,
                            file_path=function.file_path,
                            symbol=function.qualname,
                            subject="",
                            members=frozenset(site.case_names),
                        )


@dataclass(frozen=True)
class ModuleCalls:
    """Method calls of one module by the symbol that makes them."""

    module: ParsedModule
    resolver: TypeArgumentResolver
    method_names: frozenset[str]

    @cached_property
    def by_symbol(self) -> Mapping[str, tuple[CalledMethod, ...]]:
        index = module_syntax_index(self.module.module)
        calls: dict[str, list[CalledMethod]] = defaultdict(list)
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            match call.func:
                case ast.Attribute(value=receiver, attr=method_name) if (
                    method_name in self.method_names
                ):
                    receiver_chain = ExpressionReference.decoded(
                        receiver
                    ).attribute_chain
                    if receiver_chain is not None:
                        calls[index.lexical_symbol(node_index)].append(
                            CalledMethod(
                                receiver_chain=receiver_chain,
                                receiver_qualified_name=self.resolver.qualified_class_at(
                                    receiver, index.ancestor_nodes(node_index)
                                ),
                                method_name=method_name,
                            )
                        )
        return {symbol: tuple(found) for symbol, found in calls.items()}


@dataclass(frozen=True)
class Consumer(MemberObservation):
    """A function and everything it names of one family."""

    @property
    def function(self) -> tuple[CensusFamilyKind, str, str, str]:
        return self.kind, self.family, self.file_path, self.symbol

    def record(self, owners_called: Iterable[str]) -> CensusConsumer:
        return CensusConsumer(
            file_path=self.file_path,
            qualname=self.symbol,
            owners_called=tuple(sorted(set(owners_called))),
        )


@dataclass(frozen=True)
class MemberAwarenessCensusBuilder:
    """Fold observations into groups, then decide owner use from declared owners."""

    corpus: CensusCorpus
    owners: tuple[MemberOwner, ...]
    observations: tuple[MemberObservation, ...]

    @classmethod
    def from_sources(cls, corpus: CensusCorpus, config: DetectorConfig) -> Self:
        sources = tuple(
            source(corpus, config) for source in MemberCensusSource.__subclasses__()
        )
        return cls(
            corpus=corpus,
            owners=tuple(chain.from_iterable(source.owners() for source in sources)),
            observations=tuple(
                chain.from_iterable(source.observations() for source in sources)
            ),
        )

    @cached_property
    def consumers(self) -> tuple[Consumer, ...]:
        members: dict[tuple[CensusFamilyKind, str, str, str, str], set[str]] = (
            defaultdict(set)
        )
        for observation in self.observations:
            members[
                (
                    observation.kind,
                    observation.family,
                    observation.file_path,
                    observation.symbol,
                    observation.subject,
                )
            ] |= observation.members
        return tuple(
            Consumer(kind, family, file_path, symbol, subject, frozenset(found))
            for (kind, family, file_path, symbol, subject), found in sorted(
                members.items()
            )
        )

    @cached_property
    def calls(self) -> dict[str, ModuleCalls]:
        method_names = frozenset().union(*(owner.method_names for owner in self.owners))
        return {
            module.file_path: ModuleCalls(
                module, self.corpus.resolvers[module.file_path], method_names
            )
            for module in self.corpus.modules
        }

    def owners_called(
        self, consumer: Consumer, owners: Iterable[MemberOwner]
    ) -> tuple[str, ...]:
        calls = self.calls[consumer.file_path].by_symbol.get(consumer.symbol, ())
        return tuple(
            owner.name for owner in owners if any(owner.called_by(call) for call in calls)
        )

    def defined_by(self, consumer: Consumer, owner: MemberOwner) -> bool:
        return owner.defines(
            self.corpus.module_name_by_file[consumer.file_path], consumer.symbol
        )

    def covering(self, consumer: Consumer) -> tuple[MemberOwner, ...]:
        return tuple(
            owner
            for owner in self.owners
            if owner.kind is consumer.kind and owner.covers(consumer.members)
        )

    def groups(self) -> tuple[MemberSetGroup, ...]:
        grouped: dict[
            tuple[CensusFamilyKind, str, frozenset[str]], list[Consumer]
        ] = defaultdict(list)
        for consumer in self.consumers:
            if len(consumer.members) >= 2:
                grouped[(consumer.kind, consumer.family, consumer.members)].append(
                    consumer
                )
        groups: list[MemberSetGroup] = []
        for (kind, family, members), candidates in grouped.items():
            covering = self.covering(candidates[0])
            consumers = [
                consumer
                for consumer in {
                    consumer.function: consumer for consumer in candidates
                }.values()
                if not any(self.defined_by(consumer, owner) for owner in covering)
            ]
            if not consumers:
                continue
            records = [
                (consumer, self.owners_called(consumer, covering))
                for consumer in consumers
            ]
            groups.append(
                MemberSetGroup(
                    kind=kind,
                    family=family,
                    members=tuple(sorted(members)),
                    covering_owners=tuple(owner.name for owner in covering),
                    owner_using=tuple(
                        consumer.record(called) for consumer, called in records if called
                    ),
                    bypassing=tuple(
                        consumer.record(called)
                        for consumer, called in records
                        if not called
                    ),
                )
            )
        return tuple(
            sorted(
                groups,
                key=lambda group: (
                    -group.consumer_count,
                    group.kind.value,
                    group.family,
                    group.members,
                ),
            )
        )

    def owner_census(self) -> tuple[OwnerCensus, ...]:
        census: list[OwnerCensus] = []
        for owner in self.owners:
            naming = tuple(
                {
                    consumer.function: consumer
                    for consumer in self.consumers
                    if consumer.kind is owner.kind
                    and owner.names_any(consumer.members)
                    and not self.defined_by(consumer, owner)
                }.values()
            )
            if naming:
                census.append(
                    OwnerCensus(
                        kind=owner.kind,
                        owner=owner.name,
                        covered_members=tuple(sorted(owner.covered)),
                        consumers=len(naming),
                        owner_using=sum(
                            bool(self.owners_called(consumer, (owner,)))
                            for consumer in naming
                        ),
                    )
                )
        return tuple(sorted(census, key=lambda item: (-item.consumers, item.owner)))

    def report(self) -> MemberAwarenessCensus:
        return MemberAwarenessCensus(
            scanned_module_count=len(self.corpus.modules),
            single_member_consumers=len(
                {consumer.function for consumer in self.consumers}
                - {
                    consumer.function
                    for consumer in self.consumers
                    if len(consumer.members) >= 2
                }
            ),
            groups=self.groups(),
            owners=self.owner_census(),
        )


def build_member_awareness_census(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    parse_workers: int = 1,
) -> MemberAwarenessCensus:
    """Count the consumers that name each family's members under ``root``."""

    corpus = CensusCorpus(
        tuple(
            parse_python_modules(
                root,
                cache_dir=cache_dir,
                use_parse_cache=use_parse_cache,
                parse_workers=parse_workers,
            )
        )
    )
    return MemberAwarenessCensusBuilder.from_sources(
        corpus, config or DetectorConfig()
    ).report()
