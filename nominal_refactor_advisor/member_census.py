"""Census of which consumer functions know the members of a family, against
what inheritance derives for them.

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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Self

from nominal_refactor_advisor.amortization import (
    InheritanceAmortization,
    InheritanceAmortizationCache,
    IntrospectionAmortization,
    IntrospectionSource,
    MethodCall,
    NoTrace,
    _inheritance_amortization_incremental,
    inheritance_amortization_for_calls,
)
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleFact,
    parse_python_modules,
)
from nominal_refactor_advisor.cache_paths import ParseCacheDirectory
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactIndexedClass,
    CompactModuleClassProjectionFamily,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._base import CompactClassRepositoryContext
from nominal_refactor_advisor.detectors._systemic import (
    declared_cases,
    recovered_enum_sites,
    unique_direct_enum_classes,
)
from nominal_refactor_advisor.declared_family import ForeignAdapter
from nominal_refactor_advisor.foreign_family import ForeignFamily
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.json_reports import SemanticRecord, json_report_property
from nominal_refactor_advisor.observation_families import (
    TypeArgumentResolver,
    TypeTestShape,
    type_test_shapes,
)
from nominal_refactor_advisor.observation_shapes import DeclaredClassScope
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.type_checker_session import (
    SourceTexts,
    TypeCheckerSession,
)


class CensusFamilyKind(StrEnum):
    """Which declaration a family's members are read from."""

    FOREIGN_MODULE = "foreign_module"
    ENUM = "enum"
    CLASS = "class"


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
class FamilyAddCost(SemanticRecord):
    """What declaring one more member of a family costs outside its declaration: every
    function that names two or more of its members without asking an owner has to
    learn the new one. Correct maintenance is zero."""

    kind: CensusFamilyKind
    family: str
    consumers: tuple[CensusConsumer, ...]

    @json_report_property()
    def edits(self) -> int:
        return len(self.consumers)


@dataclass(frozen=True)
class MemberAwarenessCensus(SemanticRecord):
    """Every family's consumers by exact member set, and owner use."""

    scanned_module_count: int
    single_member_consumers: int
    groups: tuple[MemberSetGroup, ...]
    owners: tuple[OwnerCensus, ...]
    add_member_costs: tuple[FamilyAddCost, ...]
    inherited: InheritanceAmortization
    introspected: IntrospectionAmortization

    @json_report_property()
    def derived_total(self) -> int:
        """What the declarations derive: inherited pairs plus generic field uses."""
        return self.inherited.derived_pairs + self.introspected.derived_field_uses

    @json_report_property()
    def add_member_edits(self) -> int:
        """Edits outside their declarations that one new member of every family costs."""
        return sum(cost.edits for cost in self.add_member_costs)

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
class _ModuleTypeArgumentResolverFact(ParsedModuleFact):
    """Cached ``TypeArgumentResolver`` per ``ParsedModule``."""

    resolver: TypeArgumentResolver

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> _ModuleTypeArgumentResolverFact:
        return cls(TypeArgumentResolver(parsed_module))


@ScanCache.cached
def _cached_module_calls() -> dict[tuple[int, frozenset[str]], ModuleCalls]:
    return {}


@dataclass(frozen=True)
class _CensusBaseline:
    config: DetectorConfig
    modules_by_path: dict[str, ParsedModule]
    module_names: frozenset[str]
    class_index: CompactClassFamilyIndex
    reexports_by_file: dict[str, tuple[tuple[str, str], ...]]
    reexports_by_target: dict[str, tuple[str, ...]]
    foreign_owners: tuple[MemberOwner, ...]
    enum_classes: dict[str, CompactIndexedClass]
    enum_owners: tuple[MemberOwner, ...]
    own_owners_by_symbol: dict[str, MemberOwner]
    foreign_obs_by_file: dict[str, tuple[MemberObservation, ...]]
    enum_obs_by_file: dict[str, tuple[MemberObservation, ...]]
    enum_receiver_owners_by_file: dict[str, frozenset[str]]
    own_obs_by_file: dict[str, tuple[MemberObservation, ...]]
    own_tested_symbols_by_file: dict[str, frozenset[str]]


@ScanCache.cached
def _base_census_slot() -> list[_CensusBaseline | None]:
    return [None]


def _file_reexports(corpus: CensusCorpus, module: ParsedModule) -> tuple[tuple[str, str], ...]:
    snapshot = corpus.resolvers[module.file_path].snapshot
    prefix = f"{module.module_name}."
    return tuple(
        (binding.qualified_name, f"{prefix}{name}")
        for name, binding in snapshot.bindings_by_name.items()
    )


@dataclass(frozen=True)
class CensusCorpus:
    """The parsed modules every source reads, and their one class index."""

    modules: tuple[ParsedModule, ...]
    config: DetectorConfig
    precomputed_projections: tuple | None = None
    precomputed_class_index: CompactClassFamilyIndex | None = None
    amortization_cache_dir: Path | None = None

    @cached_property
    def projections(self) -> tuple:
        if self.precomputed_projections is not None:
            return self.precomputed_projections
        return CompactModuleClassProjectionFamily.collect_modules(self.modules)

    @cached_property
    def class_index(self) -> CompactClassFamilyIndex:
        if self.precomputed_class_index is not None:
            return self.precomputed_class_index
        return CompactClassRepositoryContext.from_projections(
            self.projections, self.config
        ).class_index

    @cached_property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.module_name for module in self.modules)

    @cached_property
    def declared_class_scope(self) -> DeclaredClassScope:
        return DeclaredClassScope(self.module_names)

    @cached_property
    def module_name_by_file(self) -> dict[str, str]:
        return {module.file_path: module.module_name for module in self.modules}

    @cached_property
    def resolvers(self) -> dict[str, TypeArgumentResolver]:
        return {
            module.file_path: module.fact(_ModuleTypeArgumentResolverFact).resolver
            for module in self.modules
        }

    @cached_property
    def _reexports_by_target(self) -> dict[str, tuple[str, ...]]:
        """Reverse re-export adjacency built once per corpus."""

        reexports: dict[str, set[str]] = defaultdict(set)
        for module in self.modules:
            snapshot = self.resolvers[module.file_path].snapshot
            prefix = f"{module.module_name}."
            for name, binding in snapshot.bindings_by_name.items():
                reexports[binding.qualified_name].add(f"{prefix}{name}")
        return {target: tuple(aliases) for target, aliases in reexports.items()}

    def aliases_of(self, import_path: str) -> frozenset[str]:
        """The import path and every scanned name that re-exports it."""

        aliases = {import_path}
        frontier = [import_path]
        reexports = self._reexports_by_target
        while frontier:
            target = frontier.pop()
            for alias in reexports.get(target, ()):
                if alias not in aliases:
                    aliases.add(alias)
                    frontier.append(alias)
        return frozenset(aliases)


class MemberCensusSource(ABC):
    """One way a family's members are named, with the owners declared for them."""

    def __init__(self, corpus: CensusCorpus) -> None:
        self.corpus = corpus

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


@dataclass(frozen=True)
class _ModuleTypeTestShapesFact(ParsedModuleFact):
    """Cached ``type_test_shapes`` extraction per ``ParsedModule``."""

    shapes: tuple[TypeTestShape, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> _ModuleTypeTestShapesFact:
        return cls(tuple(type_test_shapes(parsed_module)))


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
        scope = self.corpus.declared_class_scope
        for module in self.corpus.modules:
            for shape in module.fact(_ModuleTypeTestShapesFact).shapes:
                match tuple(shape.owners):
                    case (owner,) if owner and shape.is_foreign_to(scope):
                        yield MemberObservation(
                            kind=CensusFamilyKind.FOREIGN_MODULE,
                            family=owner,
                            file_path=shape.file_path,
                            symbol=shape.symbol,
                            subject=shape.subject,
                            members=frozenset(shape.type_names),
                        )


class EnumSource(MemberCensusSource):
    """Enum cases recovered by branches outside the enum's declaration."""

    @property
    def class_index(self) -> CompactClassFamilyIndex:
        return self.corpus.class_index

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
        for projection in self.corpus.projections:
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


class OwnClassSource(MemberCensusSource):
    """The scanned project's own class families, named by type tests and class
    patterns. A family is the nearest class every tested class inherits from."""

    @property
    def class_index(self) -> CompactClassFamilyIndex:
        return self.corpus.class_index

    def lineage(self, symbol: str) -> tuple[str, ...]:
        """The class and its ancestors, nearest first."""

        return (symbol, *self.class_index.ancestor_symbols(symbol))

    def family_of(self, members: frozenset[str]) -> str:
        """The nearest class every member is or inherits from; none for one class alone."""

        if len(members) < 2:
            return ""
        first, *rest = sorted(members)
        shared = set.intersection(*(set(self.lineage(member)) for member in rest))
        return next((symbol for symbol in self.lineage(first) if symbol in shared), "")

    def owners(self) -> tuple[MemberOwner, ...]:
        return tuple(
            ClassOwner(
                kind=CensusFamilyKind.CLASS,
                name=symbol,
                covered=frozenset((symbol, *descendants)),
                method_names=frozenset(self.class_index.classes_by_symbol[symbol].method_names),
                aliases=self.corpus.aliases_of(symbol),
                definitions=frozenset(
                    (self.corpus.module_name_by_file[declared.file_path], declared.qualname)
                    for declared in map(self.class_index.classes_by_symbol.__getitem__,
                                        (symbol, *descendants))
                ),
            )
            for symbol, descendants in self.class_index.descendants_by_symbol.items()
            if len(descendants) >= 2 and symbol in self.class_index.classes_by_symbol
        )

    def observations(self) -> Iterator[MemberObservation]:
        """One observation per function and tested value: the classes it tests across all
        its sites, under the nearest family they share."""

        tested: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for module in self.corpus.modules:
            for shape in module.fact(_ModuleTypeTestShapesFact).shapes:
                if set(shape.type_names) <= self.class_index.classes_by_symbol.keys():
                    tested[shape.file_path, shape.symbol, shape.subject].update(shape.type_names)
        for (file_path, symbol, subject), names in sorted(tested.items()):
            members = frozenset(names)
            if family := self.family_of(members):
                yield MemberObservation(
                    kind=CensusFamilyKind.CLASS,
                    family=family,
                    file_path=file_path,
                    symbol=symbol,
                    subject=subject,
                    members=members,
                )


@dataclass(frozen=True)
class ModuleCalls:
    """Method calls of one module by the symbol that makes them."""

    module: ParsedModule
    resolver: TypeArgumentResolver
    method_names: frozenset[str]

    @cached_property
    def by_symbol(self) -> Mapping[str, tuple[CalledMethod, ...]]:
        index = self.module.syntax_index
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
    def from_sources(cls, corpus: CensusCorpus) -> Self:
        source_types = tuple(MemberCensusSource.__subclasses__())
        standard_sources = (
            len(source_types) == 3
            and set(source_types) == {ForeignModuleSource, EnumSource, OwnClassSource}
        )
        if standard_sources and ScanCache.active() is not None:
            slot = _base_census_slot()
            base = slot[0]
            if base is not None and base.config == corpus.config:
                changed_modules = tuple(
                    m for m in corpus.modules if base.modules_by_path.get(m.file_path) is not m
                )
                cur_paths = {m.file_path for m in corpus.modules}
                removed_paths = frozenset(base.modules_by_path.keys() - cur_paths)
                if len(changed_modules) + len(removed_paths) <= 64:
                    changed_reexport_targets: set[str] = set()
                    reexports_by_target: dict[str, tuple[str, ...]] | None = None
                    removed_reexport_pairs: list[tuple[str, str]] = []
                    added_reexport_pairs: list[tuple[str, str]] = []
                    for rpath in removed_paths:
                        for target, alias in base.reexports_by_file.get(rpath, ()):
                            removed_reexport_pairs.append((target, alias))
                    for mod in changed_modules:
                        old_pairs = base.reexports_by_file.get(mod.file_path, ())
                        new_pairs = _file_reexports(corpus, mod)
                        if old_pairs != new_pairs:
                            old_set = set(old_pairs)
                            new_set = set(new_pairs)
                            for pair in old_set - new_set:
                                removed_reexport_pairs.append(pair)
                            for pair in new_set - old_set:
                                added_reexport_pairs.append(pair)
                    if not removed_reexport_pairs and not added_reexport_pairs:
                        reexports_by_target = base.reexports_by_target
                        corpus.__dict__["_reexports_by_target"] = reexports_by_target
                    else:
                        updated_reexports: dict[str, set[str]] = {}
                        for target, alias in removed_reexport_pairs:
                            changed_reexport_targets.add(target)
                            if target not in updated_reexports:
                                updated_reexports[target] = set(
                                    base.reexports_by_target.get(target, ())
                                )
                            updated_reexports[target].discard(alias)
                        for target, alias in added_reexport_pairs:
                            changed_reexport_targets.add(target)
                            if target not in updated_reexports:
                                updated_reexports[target] = set(
                                    base.reexports_by_target.get(target, ())
                                )
                            updated_reexports[target].add(alias)
                        reexports_by_target = dict(base.reexports_by_target)
                        for target, aliases_set in updated_reexports.items():
                            if aliases_set:
                                reexports_by_target[target] = tuple(aliases_set)
                            else:
                                reexports_by_target.pop(target, None)
                        corpus.__dict__["_reexports_by_target"] = reexports_by_target
                    # Every symbol whose aliases_of() chain reaches a changed target:
                    alias_affected_symbols: set[str] = set(changed_reexport_targets)
                    if changed_reexport_targets:
                        reverse_edges: defaultdict[str, list[str]] = defaultdict(list)
                        for target, aliases_t in base.reexports_by_target.items():
                            for alias in aliases_t:
                                reverse_edges[alias].append(target)
                        for target, alias in added_reexport_pairs:
                            reverse_edges[alias].append(target)
                        frontier_a = list(changed_reexport_targets)
                        while frontier_a:
                            node = frontier_a.pop()
                            for pred in reverse_edges.get(node, ()):
                                if pred not in alias_affected_symbols:
                                    alias_affected_symbols.add(pred)
                                    frontier_a.append(pred)

                    cur_module_names = frozenset(corpus.module_names)
                    foreign_src = ForeignModuleSource(corpus)
                    enum_src = EnumSource(corpus)
                    own_src = OwnClassSource(corpus)

                    if not alias_affected_symbols or all(
                        owner.name not in alias_affected_symbols
                        for owner in base.foreign_owners
                    ):
                        foreign_owners = base.foreign_owners
                    else:
                        foreign_owners = foreign_src.owners()

                    base_index = base.class_index
                    cur_index = corpus.class_index
                    cur_enum_classes = unique_direct_enum_classes(cur_index)
                    same_enums = len(cur_enum_classes) == len(base.enum_classes) and all(
                        base.enum_classes.get(k) is v
                        for k, v in cur_enum_classes.items()
                    )
                    enum_owners = (
                        base.enum_owners if same_enums else enum_src.owners()
                    )
                    base_classes = base_index.classes_by_symbol
                    cur_classes = cur_index.classes_by_symbol
                    added_or_removed = base_classes.keys() ^ cur_classes.keys()
                    direct_base_changed: set[str] = set(added_or_removed)
                    direct_meta_changed: set[str] = set(added_or_removed)
                    direct_ann_changed: set[str] = set(added_or_removed)
                    for sym in base_classes.keys() & cur_classes.keys():
                        bc = base_classes[sym]
                        cc = cur_classes[sym]
                        if bc.resolved_base_symbols != cc.resolved_base_symbols:
                            direct_base_changed.add(sym)
                        if (
                            bc.method_names != cc.method_names
                            or bc.file_path != cc.file_path
                            or bc.qualname != cc.qualname
                        ):
                            direct_meta_changed.add(sym)
                        if (
                            bc.direct_annotation_writes_by_name
                            != cc.direct_annotation_writes_by_name
                        ):
                            direct_ann_changed.add(sym)

                    dirty_owners: set[str] = (
                        set(direct_base_changed | direct_meta_changed)
                        | (alias_affected_symbols & (base_classes.keys() | cur_classes.keys()))
                    )
                    for sym in tuple(direct_base_changed | direct_meta_changed):
                        dirty_owners.update(base_index.ancestor_symbols(sym))
                        dirty_owners.update(cur_index.ancestor_symbols(sym))
                    own_owner_list: list[MemberOwner] = [
                        owner
                        for sym, owner in base.own_owners_by_symbol.items()
                        if sym not in dirty_owners
                    ]
                    for sym in dirty_owners:
                        if sym not in cur_classes:
                            continue
                        descendants = cur_index.descendant_symbols(sym)
                        if len(descendants) >= 2:
                            own_owner_list.append(
                                ClassOwner(
                                    kind=CensusFamilyKind.CLASS,
                                    name=sym,
                                    covered=frozenset((sym, *descendants)),
                                    method_names=frozenset(cur_classes[sym].method_names),
                                    aliases=corpus.aliases_of(sym),
                                    definitions=frozenset(
                                        (
                                            corpus.module_name_by_file[declared.file_path],
                                            declared.qualname,
                                        )
                                        for declared in map(
                                            cur_classes.__getitem__,
                                            (sym, *descendants),
                                        )
                                    ),
                                )
                            )
                    own_owners = tuple(own_owner_list)

                    # Observations: Foreign + Own incremental per-file, Enum via source
                    if cur_module_names == base.module_names:
                        changed_paths = frozenset(m.file_path for m in changed_modules)
                        scope = corpus.declared_class_scope
                        foreign_obs: list[MemberObservation] = []
                        for mod in corpus.modules:
                            if mod.file_path not in changed_paths:
                                foreign_obs.extend(
                                    base.foreign_obs_by_file.get(mod.file_path, ())
                                )
                            else:
                                for shape in mod.fact(_ModuleTypeTestShapesFact).shapes:
                                    match tuple(shape.owners):
                                        case (owner,) if owner and shape.is_foreign_to(scope):
                                            foreign_obs.append(
                                                MemberObservation(
                                                    kind=CensusFamilyKind.FOREIGN_MODULE,
                                                    family=owner,
                                                    file_path=shape.file_path,
                                                    symbol=shape.symbol,
                                                    subject=shape.subject,
                                                    members=frozenset(shape.type_names),
                                                )
                                            )
                    else:
                        foreign_obs = list(foreign_src.observations())

                    ancestry_changed: set[str] = set(direct_base_changed)
                    for sym in direct_base_changed:
                        ancestry_changed.update(base_index.descendant_symbols(sym))
                        ancestry_changed.update(cur_index.descendant_symbols(sym))
                    dirty_own_symbols = frozenset(ancestry_changed | added_or_removed)
                    changed_paths = frozenset(m.file_path for m in changed_modules)
                    own_obs: list[MemberObservation] = []
                    recompute_own_modules: list[ParsedModule] = []
                    for mod in corpus.modules:
                        if (
                            mod.file_path not in changed_paths
                            and base.own_tested_symbols_by_file.get(
                                mod.file_path, frozenset()
                            ).isdisjoint(dirty_own_symbols)
                        ):
                            own_obs.extend(base.own_obs_by_file.get(mod.file_path, ()))
                        else:
                            recompute_own_modules.append(mod)
                    if recompute_own_modules:
                        tested: dict[tuple[str, str, str], set[str]] = defaultdict(set)
                        for mod in recompute_own_modules:
                            for shape in mod.fact(_ModuleTypeTestShapesFact).shapes:
                                if set(shape.type_names) <= cur_classes.keys():
                                    tested[
                                        shape.file_path, shape.symbol, shape.subject
                                    ].update(shape.type_names)
                        for (file_path, symbol, subject), names in sorted(tested.items()):
                            members = frozenset(names)
                            if family := own_src.family_of(members):
                                own_obs.append(
                                    MemberObservation(
                                        kind=CensusFamilyKind.CLASS,
                                        family=family,
                                        file_path=file_path,
                                        symbol=symbol,
                                        subject=subject,
                                        members=members,
                                    )
                                )

                    if same_enums:
                        dirty_enum_owners: set[str] = set(ancestry_changed | direct_ann_changed)
                        for sym in direct_ann_changed:
                            dirty_enum_owners.update(base_index.descendant_symbols(sym))
                            dirty_enum_owners.update(cur_index.descendant_symbols(sym))
                        enum_obs: list[MemberObservation] = []
                        for projection in corpus.projections:
                            if (
                                projection.file_path not in changed_paths
                                and base.enum_receiver_owners_by_file.get(
                                    projection.file_path, frozenset()
                                ).isdisjoint(dirty_enum_owners)
                            ):
                                enum_obs.extend(
                                    base.enum_obs_by_file.get(projection.file_path, ())
                                )
                            else:
                                for function in projection.closed_axis_branch_functions:
                                    for enum_name, site in recovered_enum_sites(
                                        cur_index, cur_enum_classes, function
                                    ).items():
                                        enum_class = cur_enum_classes[enum_name]
                                        inside_declaration = (
                                            function.file_path == enum_class.file_path
                                            and function.qualname.startswith(
                                                f"{enum_class.qualname}."
                                            )
                                        )
                                        if site.case_names and not inside_declaration:
                                            enum_obs.append(
                                                MemberObservation(
                                                    kind=CensusFamilyKind.ENUM,
                                                    family=enum_class.symbol,
                                                    file_path=function.file_path,
                                                    symbol=function.qualname,
                                                    subject="",
                                                    members=frozenset(site.case_names),
                                                )
                                            )
                    else:
                        enum_obs = list(enum_src.observations())

                    return cls(
                        corpus=corpus,
                        owners=(*foreign_owners, *enum_owners, *own_owners),
                        observations=(
                            *foreign_obs,
                            *enum_obs,
                            *own_obs,
                        ),
                    )

        sources = tuple(source_cls(corpus) for source_cls in source_types)
        owners_by_type = {type(source): source.owners() for source in sources}
        obs_by_type = {type(source): tuple(source.observations()) for source in sources}
        builder = cls(
            corpus=corpus,
            owners=tuple(chain.from_iterable(owners_by_type.values())),
            observations=tuple(chain.from_iterable(obs_by_type.values())),
        )
        if (
            standard_sources
            and ScanCache.active() is not None
            and len(corpus.modules) >= 20
            and _base_census_slot()[0] is None
        ):
            foreign_obs_by_file: dict[str, list[MemberObservation]] = defaultdict(list)
            for obs in obs_by_type[ForeignModuleSource]:
                foreign_obs_by_file[obs.file_path].append(obs)
            enum_obs_by_file: dict[str, list[MemberObservation]] = defaultdict(list)
            for obs in obs_by_type[EnumSource]:
                enum_obs_by_file[obs.file_path].append(obs)
            enum_receiver_owners_by_file: dict[str, frozenset[str]] = {}
            for projection in corpus.projections:
                owners_in_file: set[str] = set()
                for function in projection.closed_axis_branch_functions:
                    if not function.receiver_field_reads:
                        continue
                    owner_qualname, _dot, _method = function.qualname.rpartition(".")
                    owner = corpus.class_index.symbol_for(
                        file_path=function.file_path, qualname=owner_qualname
                    )
                    if owner is not None:
                        owners_in_file.add(owner)
                if owners_in_file:
                    enum_receiver_owners_by_file[projection.file_path] = frozenset(
                        owners_in_file
                    )
            own_obs_by_file: dict[str, list[MemberObservation]] = defaultdict(list)
            for obs in obs_by_type[OwnClassSource]:
                own_obs_by_file[obs.file_path].append(obs)
            own_tested_by_file: dict[str, frozenset[str]] = {}
            for mod in corpus.modules:
                syms: set[str] = set()
                for shape in mod.fact(_ModuleTypeTestShapesFact).shapes:
                    syms.update(shape.type_names)
                if syms:
                    own_tested_by_file[mod.file_path] = frozenset(syms)
            _base_census_slot()[0] = _CensusBaseline(
                config=corpus.config,
                modules_by_path={m.file_path: m for m in corpus.modules},
                module_names=frozenset(corpus.module_names),
                class_index=corpus.class_index,
                reexports_by_file={
                    m.file_path: _file_reexports(corpus, m) for m in corpus.modules
                },
                reexports_by_target=dict(corpus._reexports_by_target),
                foreign_owners=owners_by_type[ForeignModuleSource],
                enum_classes=unique_direct_enum_classes(corpus.class_index),
                enum_owners=owners_by_type[EnumSource],
                own_owners_by_symbol={
                    owner.name: owner for owner in owners_by_type[OwnClassSource]
                },
                foreign_obs_by_file={
                    path: tuple(items) for path, items in foreign_obs_by_file.items()
                },
                enum_obs_by_file={
                    path: tuple(items) for path, items in enum_obs_by_file.items()
                },
                enum_receiver_owners_by_file=enum_receiver_owners_by_file,
                own_obs_by_file={
                    path: tuple(items) for path, items in own_obs_by_file.items()
                },
                own_tested_symbols_by_file=own_tested_by_file,
            )
        return builder

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
    def _owner_method_names(self) -> frozenset[str]:
        return frozenset().union(*(owner.method_names for owner in self.owners))

    @cached_property
    def _modules_by_file_path(self) -> dict[str, ParsedModule]:
        return {module.file_path: module for module in self.corpus.modules}

    @cached_property
    def _calls_by_file_path(self) -> dict[str, ModuleCalls]:
        return {}

    @cached_property
    def calls(self) -> dict[str, ModuleCalls]:
        method_names = self._owner_method_names
        for module in self.corpus.modules:
            if module.file_path not in self._calls_by_file_path:
                self._calls_by_file_path[module.file_path] = ModuleCalls(
                    module, self.corpus.resolvers[module.file_path], method_names
                )
        return self._calls_by_file_path

    def _module_calls(self, file_path: str) -> ModuleCalls:
        cached = self._calls_by_file_path.get(file_path)
        if cached is None:
            module = self._modules_by_file_path[file_path]
            method_names = self._owner_method_names
            if ScanCache.active() is not None:
                shared_cache = _cached_module_calls()
                key = (id(module), method_names)
                cached = shared_cache.get(key)
                if cached is None:
                    cached = ModuleCalls(
                        module, self.corpus.resolvers[file_path], method_names
                    )
                    shared_cache[key] = cached
            else:
                cached = ModuleCalls(
                    module, self.corpus.resolvers[file_path], method_names
                )
            self._calls_by_file_path[file_path] = cached
        return cached

    def owners_called(
        self, consumer: Consumer, owners: Iterable[MemberOwner]
    ) -> tuple[str, ...]:
        calls = self._module_calls(consumer.file_path).by_symbol.get(consumer.symbol, ())
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

    def add_member_costs(self, groups: tuple[MemberSetGroup, ...]) -> tuple[FamilyAddCost, ...]:
        """Per family, the distinct functions that hand-name two or more of its members."""

        bypassing: dict[tuple[CensusFamilyKind, str], dict[tuple[str, str], CensusConsumer]] = (
            defaultdict(dict)
        )
        for group in groups:
            for consumer in group.bypassing:
                bypassing[group.kind, group.family][consumer.file_path, consumer.qualname] = consumer
        return tuple(sorted(
            (FamilyAddCost(kind, family, tuple(found[key] for key in sorted(found)))
             for (kind, family), found in bypassing.items()),
            key=lambda cost: (-cost.edits, cost.kind.value, cost.family),
        ))

    def report(
        self,
        introspection: IntrospectionSource,
        *,
        include_owners: bool = True,
    ) -> MemberAwarenessCensus:
        groups = self.groups()
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
            groups=groups,
            owners=self.owner_census() if include_owners else (),
            add_member_costs=self.add_member_costs(groups),
            inherited=self.inherited(),
            introspected=introspection.amortization(),
        )

    def inherited(self) -> InheritanceAmortization:
        if TypeCheckerSession.has_active_incremental() and self.corpus.modules:
            with TypeCheckerSession.scoped_session(
                SourceTexts(self.corpus.modules),
                (),
            ) as session:
                if session.file_version(self.corpus.modules[0].file_path) is not None:
                    return _inheritance_amortization_incremental(
                        self.corpus.modules, self.corpus.class_index, session
                    )
        disk_cache = (
            None
            if self.corpus.amortization_cache_dir is None
            else InheritanceAmortizationCache(self.corpus.amortization_cache_dir)
        )
        if disk_cache is not None:
            cached = disk_cache.load(self.corpus.modules)
            if cached is not None:
                return cached
        calls = MethodCall.in_modules(self.corpus.modules)
        with TypeCheckerSession.scoped_session(
            SourceTexts(self.corpus.modules),
            (call.span for call in calls),
        ) as session:
            result = inheritance_amortization_for_calls(
                self.corpus.modules, calls, self.corpus.class_index, session
            )
        if disk_cache is not None:
            disk_cache.store(self.corpus.modules, result)
        return result


@ScanCache.scope()
def build_member_awareness_census(
    root: Path,
    *,
    config: DetectorConfig | None = None,
    cache_dir: Path | None = None,
    use_parse_cache: bool = True,
    introspection: IntrospectionSource = NoTrace(),
) -> MemberAwarenessCensus:
    """Count the consumers that name each family's members under ``root``."""

    cache_paths = ParseCacheDirectory(
        parse_cache_dir=cache_dir,
        use_parse_cache=use_parse_cache,
    )
    with InstalledModule.cached_if_unset(cache_paths.collected_family_cache_dir):
        corpus = CensusCorpus(
            tuple(
                parse_python_modules(
                    root,
                    cache_dir=cache_dir,
                    use_parse_cache=use_parse_cache,
                )
            ),
            config or DetectorConfig(),
            amortization_cache_dir=cache_paths.inheritance_amortization_cache_dir,
        )
        return MemberAwarenessCensusBuilder.from_sources(corpus).report(introspection)
