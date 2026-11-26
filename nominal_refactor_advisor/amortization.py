"""What inheritance derives: the (class, method) pairs nobody wrote.

A class that inherits a method instead of defining it gets that implementation
from its parent's one declaration. Remove the parent's method and each such
pair has to be written by hand: the ablation counts them. A pair counts only
when a call whose receiver the type checker resolves to a scanned class reaches
it, so a fat root nobody calls through earns nothing. The report names how
many calls have no such receiver, the coverage this leaves out.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from pathlib import Path

from nra_introspection_trace import IntrospectionTrace

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleFact
from nominal_refactor_advisor.class_index import CompactClassFamilyIndex
from nominal_refactor_advisor.implementation_identity import (
    ImplementationSource,
    declaration_implementation_module_names,
)
from nominal_refactor_advisor.json_reports import SemanticRecord, json_report_property
from nominal_refactor_advisor.probe_owner import ReceiverSpan, ScannedClasses
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.type_checker_session import (
    InferredReceiverType,
    ReceiverType,
    TypeCheckerSession,
)


@dataclass(frozen=True)
class InheritedMethodOwner(SemanticRecord):
    """One class whose methods are inherited by classes typed calls reach."""

    owner: str
    derived_pairs: int
    members: int
    consumers: int


@dataclass(frozen=True)
class InheritanceAmortization(SemanticRecord):
    """Pairs inheritance derives that typed calls reach, by the class that declares them."""

    scanned_line_count: int
    type_checker: str
    attribute_calls: int
    receiver_typed_calls: int
    calls_reaching_inherited: int
    owners: tuple[InheritedMethodOwner, ...]

    @json_report_property()
    def derived_pairs(self) -> int:
        """(class, method) pairs that would be hand-written without their parent."""
        return sum(owner.derived_pairs for owner in self.owners)

    @json_report_property()
    def derived_pairs_per_kloc(self) -> float:
        return 1000 * self.derived_pairs / max(self.scanned_line_count, 1)


@ScanCache.cached
def _amortization_implementation_digest() -> str:
    """Digest of amortization + type-checker implementation modules and mypy version."""

    try:
        mypy_version = importlib.metadata.version("mypy")
    except importlib.metadata.PackageNotFoundError:
        mypy_version = "unavailable"
    sources = tuple(
        ImplementationSource.from_module_name(module_name)
        for module_name in declaration_implementation_module_names(
            (InheritanceAmortization, TypeCheckerSession)
        )
    )
    payload = repr(
        (
            "inheritance-amortization-cache-v1",
            sys.version_info[:3],
            sys.executable,
            mypy_version,
            sources,
        )
    ).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class InheritanceAmortizationCache:
    """Disk cache for whole-corpus ``InheritanceAmortization`` keyed by exact source text."""

    cache_dir: Path

    def key_for(self, modules: tuple[ParsedModule, ...]) -> str:
        digest = hashlib.blake2s(digest_size=20)
        digest.update(_amortization_implementation_digest().encode("ascii"))
        for module in sorted(modules, key=lambda item: (item.file_path, item.module_name)):
            raw_hash = hashlib.blake2s(
                module.source.encode("utf-8"), digest_size=16
            ).digest()
            digest.update(module.file_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(module.module_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(raw_hash)
        return digest.hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def load(self, modules: tuple[ParsedModule, ...]) -> InheritanceAmortization | None:
        key = self.key_for(modules)
        entry_path = self._entry_path(key)
        try:
            with entry_path.open("rb") as handle:
                stored_key, value = pickle.load(handle)
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError, ValueError, TypeError):
            return None
        if stored_key != key or not isinstance(value, InheritanceAmortization):
            return None
        return value

    def store(
        self,
        modules: tuple[ParsedModule, ...],
        amortization: InheritanceAmortization,
    ) -> None:
        key = self.key_for(modules)
        entry_path = self._entry_path(key)
        tmp_path = self.cache_dir / f".{key}.{os.getpid()}.tmp"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("wb") as handle:
                pickle.dump((key, amortization), handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, entry_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


@dataclass(frozen=True)
class InheritedMethods:
    """Which declaring class answers a method a class does not define itself."""

    index: CompactClassFamilyIndex
    _owner_cache: dict[tuple[str, str], str | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _reached_cache: dict[str, tuple[str, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _ancestors_cache: dict[str, tuple[str, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @cached_property
    def own_by_symbol(self) -> dict[str, frozenset[str]]:
        """Each class's own non-dunder methods."""
        return {
            symbol: frozenset(
                name
                for name in declared.method_names
                if not (name.startswith("__") and name.endswith("__"))
            )
            for symbol, declared in self.index.classes_by_symbol.items()
        }

    @cached_property
    def all_declared_methods(self) -> frozenset[str]:
        """Every non-dunder method name declared by any scanned class."""
        if not self.own_by_symbol:
            return frozenset()
        return frozenset().union(*self.own_by_symbol.values())

    def own(self, symbol: str) -> frozenset[str]:
        return self.own_by_symbol.get(symbol, frozenset())

    def owner_of(self, symbol: str, method: str) -> str | None:
        if method not in self.all_declared_methods:
            return None
        key = (symbol, method)
        if key in self._owner_cache:
            return self._owner_cache[key]
        if method in self.own(symbol):
            owner: str | None = None
        else:
            ancestors = self._ancestors_cache.get(symbol)
            if ancestors is None:
                ancestors = self.index.ancestor_symbols(symbol)
                self._ancestors_cache[symbol] = ancestors
            own_by_symbol = self.own_by_symbol
            owner = next(
                (
                    ancestor
                    for ancestor in ancestors
                    if method in own_by_symbol.get(ancestor, ())
                ),
                None,
            )
        self._owner_cache[key] = owner
        return owner

    def reached(self, symbol: str) -> tuple[str, ...]:
        cached = self._reached_cache.get(symbol)
        if cached is None:
            cached = (symbol, *self.index.descendant_symbols(symbol))
            self._reached_cache[symbol] = cached
        return cached


@dataclass(frozen=True)
class _ModuleMethodCallsFact(ParsedModuleFact):
    """Cached ``MethodCall`` extraction per ``ParsedModule``."""

    calls: tuple[MethodCall, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> _ModuleMethodCallsFact:
        index = parsed_module.syntax_index
        file_path = parsed_module.file_path
        module_name = parsed_module.module_name
        calls: list[MethodCall] = []
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            match call.func:
                case ast.Attribute(value=receiver, attr=method):
                    calls.append(
                        MethodCall(
                            ReceiverSpan.of(file_path, receiver),
                            method,
                            f"{module_name}.{index.lexical_symbol(node_index)}",
                        )
                    )
        return cls(tuple(calls))


@dataclass(frozen=True)
class MethodCall:
    """One ``receiver.method(...)`` call: where its receiver is, and which function makes it."""

    span: ReceiverSpan
    method: str
    consumer: str

    @classmethod
    def in_module(cls, module: ParsedModule) -> Iterator[MethodCall]:
        return iter(module.fact(_ModuleMethodCallsFact).calls)

    @classmethod
    def in_modules(cls, modules: Iterable[ParsedModule]) -> tuple[MethodCall, ...]:
        return tuple(
            chain.from_iterable(
                module.fact(_ModuleMethodCallsFact).calls for module in modules
            )
        )


def inheritance_amortization(
    modules: Iterable[ParsedModule],
    index: CompactClassFamilyIndex,
    session: TypeCheckerSession,
) -> InheritanceAmortization:
    """Count the inherited pairs that calls with resolved receivers reach across ``modules``."""

    resolved_modules = tuple(modules)
    return inheritance_amortization_for_calls(
        resolved_modules,
        MethodCall.in_modules(resolved_modules),
        index,
        session,
    )


@ScanCache.cached
def _amortization_file_cache() -> dict[
    tuple[int, str],
    tuple[
        int,
        int,
        int,
        int,
        tuple[tuple[ReceiverType, str, int, frozenset[str]], ...],
    ],
]:
    """Per-file aggregated (ReceiverType, method) -> (count, consumers) cache."""
    return {}


@dataclass(frozen=True)
class _AmortizationFileAggregate:
    module_id: int
    version: int
    line_count: int
    call_count: int
    typed: int
    reaching: int
    pairs_by_owner: Mapping[str, frozenset[tuple[str, str]]]
    consumers_by_owner: Mapping[str, frozenset[str]]
    candidate_symbols: frozenset[str]
    summary: tuple[tuple[ReceiverType, str, int, frozenset[str]], ...] = ()


@dataclass(frozen=True)
class _AmortizationBaseline:
    session_id: int
    index: CompactClassFamilyIndex
    own_by_symbol: dict[str, frozenset[str]]
    symbols: frozenset[str]
    per_file: dict[str, _AmortizationFileAggregate]
    owner_cache: Mapping[tuple[str, str], str | None] = field(default_factory=dict)
    total_lines: int = 0
    total_calls: int = 0
    typed: int = 0
    reaching: int = 0
    owners_by_name: Mapping[str, InheritedMethodOwner] = field(default_factory=dict)
    call_key_cache: Mapping[
        tuple[tuple[str, ...], str],
        tuple[tuple[str, ...], tuple[tuple[str, frozenset[tuple[str, str]]], ...]],
    ] = field(default_factory=dict)
    pair_counts_by_owner: Mapping[str, Mapping[tuple[str, str], int]] = field(
        default_factory=dict
    )
    consumer_counts_by_owner: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )
    member_pair_counts_by_owner: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )


@ScanCache.cached
def _base_amortization_slot() -> list[_AmortizationBaseline | None]:
    return [None]


def _inheritance_amortization_incremental(
    modules: tuple[ParsedModule, ...],
    index: CompactClassFamilyIndex,
    session: TypeCheckerSession,
) -> InheritanceAmortization:
    cache = _amortization_file_cache()
    session_id = id(session)
    base_slot = _base_amortization_slot()
    base = base_slot[0]
    if base is not None and base.session_id != session_id:
        base = None

    methods = InheritedMethods(index)
    cur_own = methods.own_by_symbol
    cur_symbols = frozenset(index.classes_by_symbol)
    affected_symbols: frozenset[str] | None = None
    down_changed: set[str] = set()
    if base is not None:
        base_index = base.index
        base_own = base.own_by_symbol
        base_classes = base_index.classes_by_symbol
        cur_classes = index.classes_by_symbol
        added_or_removed = base.symbols ^ cur_symbols
        direct_own_changed: set[str] = set(added_or_removed)
        direct_base_changed: set[str] = set(added_or_removed)
        for sym in base.symbols & cur_symbols:
            if base_own.get(sym) != cur_own.get(sym):
                direct_own_changed.add(sym)
            if (
                base_classes[sym].resolved_base_symbols
                != cur_classes[sym].resolved_base_symbols
            ):
                direct_base_changed.add(sym)
        seeds = direct_own_changed | direct_base_changed
        down_changed = set(seeds)
        for sym in seeds:
            down_changed.update(base_index.descendant_symbols(sym))
            down_changed.update(index.descendant_symbols(sym))
        descendants_changed: set[str] = set()
        for sym in direct_base_changed:
            descendants_changed.update(base_index.ancestor_symbols(sym))
            descendants_changed.update(index.ancestor_symbols(sym))
        affected: set[str] = set(down_changed) | descendants_changed
        for sym in down_changed:
            affected.update(base_index.ancestor_symbols(sym))
            affected.update(index.ancestor_symbols(sym))
        affected_symbols = frozenset(affected)

    recompute_indices: list[tuple[int, ParsedModule, int]] = []
    file_entries: dict[
        int, tuple[int, int, tuple[tuple[ReceiverType, str, int, frozenset[str]], ...]]
    ] = {}
    missed_modules: list[tuple[int, ParsedModule, int, tuple[MethodCall, ...]]] = []
    unchanged_paths: set[str] = set()

    for idx, module in enumerate(modules):
        version = session.file_version(module.file_path)
        assert version is not None
        if base is not None and affected_symbols is not None:
            agg = base.per_file.get(module.file_path)
            if (
                agg is not None
                and agg.module_id == id(module)
                and agg.version == version
                and agg.candidate_symbols.isdisjoint(affected_symbols)
            ):
                unchanged_paths.add(module.file_path)
                continue
        recompute_indices.append((idx, module, version))
        cached = cache.get((session_id, module.file_path))
        if cached is not None and cached[0] == id(module) and cached[1] == version:
            file_entries[idx] = (cached[2], cached[3], cached[4])
        else:
            missed_modules.append(
                (idx, module, version, module.fact(_ModuleMethodCallsFact).calls)
            )

    if missed_modules:
        missed_spans = tuple(
            call.span for _idx, _mod, _ver, mod_calls in missed_modules for call in mod_calls
        )
        missed_answers = session.receiver_types(missed_spans) if missed_spans else {}
        for idx, module, version, mod_calls in missed_modules:
            grouped_counts: dict[tuple[ReceiverType, str], int] = defaultdict(int)
            grouped_consumers: dict[tuple[ReceiverType, str], set[str]] = defaultdict(set)
            for call in mod_calls:
                key = (missed_answers[call.span], call.method)
                grouped_counts[key] += 1
                grouped_consumers[key].add(call.consumer)
            summary = tuple(
                (answer, method_name, count, frozenset(grouped_consumers[(answer, method_name)]))
                for (answer, method_name), count in grouped_counts.items()
            )
            line_count = len(module.source.splitlines())
            call_count = len(mod_calls)
            cache[(session_id, module.file_path)] = (
                id(module),
                version,
                line_count,
                call_count,
                summary,
            )
            file_entries[idx] = (line_count, call_count, summary)

    use_owner_delta = (
        base is not None
        and bool(base.owners_by_name)
        and len(unchanged_paths) > len(modules) // 2
    )
    pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    consumers: defaultdict[str, set[str]] = defaultdict(set)
    pair_deltas: defaultdict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    consumer_deltas: defaultdict[str, dict[str, int]] = defaultdict(dict)
    recomputed_paths: set[str] = {module.file_path for _idx, module, _ver in recompute_indices}
    if use_owner_delta:
        assert base is not None
        total_lines = base.total_lines
        total_calls = base.total_calls
        typed = base.typed
        reaching = base.reaching
        for path, old_agg in base.per_file.items():
            if path not in unchanged_paths:
                total_lines -= old_agg.line_count
                total_calls -= old_agg.call_count
                typed -= old_agg.typed
                reaching -= old_agg.reaching
                if path not in recomputed_paths:
                    for owner, old_p in old_agg.pairs_by_owner.items():
                        od = pair_deltas[owner]
                        for pair in old_p:
                            od[pair] = od.get(pair, 0) - 1
                    for owner, old_c in old_agg.consumers_by_owner.items():
                        cd = consumer_deltas[owner]
                        for consumer in old_c:
                            cd[consumer] = cd.get(consumer, 0) - 1
    else:
        total_lines = total_calls = typed = reaching = 0
        if base is not None:
            for path in unchanged_paths:
                agg = base.per_file[path]
                total_lines += agg.line_count
                total_calls += agg.call_count
                typed += agg.typed
                reaching += agg.reaching
                for owner, owner_pairs in agg.pairs_by_owner.items():
                    pairs[owner].update(owner_pairs)
                for owner, owner_consumers in agg.consumers_by_owner.items():
                    consumers[owner].update(owner_consumers)

    classes = ScannedClasses(index.classes_by_symbol)
    symbols_by_answer: dict[ReceiverType, tuple[str, ...]] = {}
    owners_by_call_key: dict[
        tuple[tuple[str, ...], str],
        tuple[tuple[str, ...], tuple[tuple[str, frozenset[tuple[str, str]]], ...]],
    ] = {}
    record_base = base is None and len(modules) >= 20
    per_file_aggs: dict[str, _AmortizationFileAggregate] = {}
    base_pair_counts_builder: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    base_consumer_counts_builder: defaultdict[str, Counter[str]] = defaultdict(Counter)

    all_declared_methods = methods.all_declared_methods
    base_owner_cache = base.owner_cache if base is not None else None
    base_call_key_cache = base.call_key_cache if base is not None else None
    seen_recompute_call_keys: set[tuple[tuple[str, ...], str]] = set()
    for idx, module, version in recompute_indices:
        line_count, call_count, summary = file_entries[idx]
        if (
            use_owner_delta
            and base is not None
            and affected_symbols is not None
        ):
            old_agg = base.per_file.get(module.file_path)
            if (
                old_agg is not None
                and old_agg.line_count == line_count
                and old_agg.call_count == call_count
                and old_agg.summary == summary
                and old_agg.candidate_symbols.isdisjoint(affected_symbols)
            ):
                total_lines += old_agg.line_count
                total_calls += old_agg.call_count
                typed += old_agg.typed
                reaching += old_agg.reaching
                continue
        total_lines += line_count
        total_calls += call_count
        file_typed = 0
        file_reaching = 0
        file_pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
        file_consumers: defaultdict[str, set[str]] = defaultdict(set)
        candidate_symbols: set[str] = set()
        seen_file_call_keys: set[tuple[tuple[str, ...], str]] = set()
        for answer, method_name, count, entry_consumers in summary:
            if record_base and isinstance(answer, InferredReceiverType) and len(answer.classes) == 1:
                candidate_symbols.add(answer.classes[0])
            symbols = symbols_by_answer.get(answer)
            if symbols is None:
                symbols = answer.owner(classes).class_symbols()
                symbols_by_answer[answer] = symbols
            if not symbols:
                continue
            file_typed += count
            if method_name not in all_declared_methods:
                continue
            call_key = (symbols, method_name)
            cached_call = owners_by_call_key.get(call_key)
            if cached_call is None:
                if (
                    base_call_key_cache is not None
                    and affected_symbols is not None
                    and affected_symbols.isdisjoint(symbols)
                ):
                    cached_call = base_call_key_cache.get(call_key)
                if cached_call is None:
                    grouped_pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
                    for symbol in symbols:
                        for member in methods.reached(symbol):
                            if (
                                base_owner_cache is not None
                                and member not in down_changed
                                and (member, method_name) in base_owner_cache
                            ):
                                owner = base_owner_cache[(member, method_name)]
                            else:
                                owner = methods.owner_of(member, method_name)
                            if owner is not None:
                                grouped_pairs[owner].add((member, method_name))
                    cached_call = (
                        tuple(grouped_pairs.keys()),
                        tuple(
                            (owner, frozenset(owner_pairs))
                            for owner, owner_pairs in grouped_pairs.items()
                        ),
                    )
                owners_by_call_key[call_key] = cached_call
            hit_owners, call_pairs_by_owner = cached_call
            if hit_owners:
                file_reaching += count
                if record_base:
                    if call_key not in seen_file_call_keys:
                        seen_file_call_keys.add(call_key)
                        for owner, owner_pair_set in call_pairs_by_owner:
                            pairs[owner].update(owner_pair_set)
                            file_pairs[owner].update(owner_pair_set)
                    for owner in hit_owners:
                        consumers[owner].update(entry_consumers)
                        file_consumers[owner].update(entry_consumers)
                elif use_owner_delta:
                    if call_key not in seen_file_call_keys:
                        seen_file_call_keys.add(call_key)
                        for owner, owner_pair_set in call_pairs_by_owner:
                            file_pairs[owner].update(owner_pair_set)
                    for owner in hit_owners:
                        file_consumers[owner].update(entry_consumers)
                else:
                    if call_key not in seen_recompute_call_keys:
                        seen_recompute_call_keys.add(call_key)
                        for owner, owner_pair_set in call_pairs_by_owner:
                            pairs[owner].update(owner_pair_set)
                    for owner in hit_owners:
                        consumers[owner].update(entry_consumers)
        typed += file_typed
        reaching += file_reaching
        if record_base:
            frozen_pairs: dict[str, frozenset[tuple[str, str]]] = {}
            frozen_consumers: dict[str, frozenset[str]] = {}
            for owner, owner_pairs in file_pairs.items():
                fp = frozenset(owner_pairs)
                frozen_pairs[owner] = fp
                base_pair_counts_builder[owner].update(fp)
            for owner, owner_consumers in file_consumers.items():
                fc = frozenset(owner_consumers)
                frozen_consumers[owner] = fc
                base_consumer_counts_builder[owner].update(fc)
            per_file_aggs[module.file_path] = _AmortizationFileAggregate(
                module_id=id(module),
                version=version,
                line_count=line_count,
                call_count=call_count,
                typed=file_typed,
                reaching=file_reaching,
                pairs_by_owner=frozen_pairs,
                consumers_by_owner=frozen_consumers,
                candidate_symbols=frozenset(candidate_symbols),
                summary=summary,
            )
        elif use_owner_delta:
            assert base is not None
            old_agg = base.per_file.get(module.file_path)
            old_p_map = old_agg.pairs_by_owner if old_agg is not None else {}
            old_c_map = old_agg.consumers_by_owner if old_agg is not None else {}
            for owner in file_pairs.keys() | old_p_map.keys():
                old_p = old_p_map.get(owner)
                new_p = file_pairs.get(owner)
                if old_p is None:
                    assert new_p is not None
                    od = pair_deltas[owner]
                    for pair in new_p:
                        od[pair] = od.get(pair, 0) + 1
                elif new_p is None:
                    od = pair_deltas[owner]
                    for pair in old_p:
                        od[pair] = od.get(pair, 0) - 1
                elif new_p != old_p:
                    od = pair_deltas[owner]
                    for pair in old_p - new_p:
                        od[pair] = od.get(pair, 0) - 1
                    for pair in new_p - old_p:
                        od[pair] = od.get(pair, 0) + 1
            for owner in file_consumers.keys() | old_c_map.keys():
                old_c = old_c_map.get(owner)
                new_c = file_consumers.get(owner)
                if old_c is None:
                    assert new_c is not None
                    cd = consumer_deltas[owner]
                    for consumer in new_c:
                        cd[consumer] = cd.get(consumer, 0) + 1
                elif new_c is None:
                    cd = consumer_deltas[owner]
                    for consumer in old_c:
                        cd[consumer] = cd.get(consumer, 0) - 1
                elif new_c != old_c:
                    cd = consumer_deltas[owner]
                    for consumer in old_c - new_c:
                        cd[consumer] = cd.get(consumer, 0) - 1
                    for consumer in new_c - old_c:
                        cd[consumer] = cd.get(consumer, 0) + 1

    if use_owner_delta:
        assert base is not None
        actually_dirty_owners = pair_deltas.keys() | consumer_deltas.keys()
        owners_list: list[InheritedMethodOwner] = [
            item
            for owner_name, item in base.owners_by_name.items()
            if owner_name not in actually_dirty_owners
        ]
        empty_pairs_map: Mapping[tuple[str, str], int] = {}
        empty_str_map: Mapping[str, int] = {}
        for owner_name in actually_dirty_owners:
            base_owner = base.owners_by_name.get(owner_name)
            derived_count = 0 if base_owner is None else base_owner.derived_pairs
            member_count = 0 if base_owner is None else base_owner.members
            consumer_count = 0 if base_owner is None else base_owner.consumers
            p_deltas = pair_deltas.get(owner_name)
            if p_deltas:
                base_p = base.pair_counts_by_owner.get(owner_name, empty_pairs_map)
                base_m = base.member_pair_counts_by_owner.get(owner_name, empty_str_map)
                member_delta: dict[str, int] = {}
                for pair, delta in p_deltas.items():
                    if not delta:
                        continue
                    old_n = base_p.get(pair, 0)
                    new_n = old_n + delta
                    if old_n == 0 and new_n > 0:
                        derived_count += 1
                        member_delta[pair[0]] = member_delta.get(pair[0], 0) + 1
                    elif old_n > 0 and new_n == 0:
                        derived_count -= 1
                        member_delta[pair[0]] = member_delta.get(pair[0], 0) - 1
                for member, mdelta in member_delta.items():
                    if not mdelta:
                        continue
                    old_m = base_m.get(member, 0)
                    new_m = old_m + mdelta
                    if old_m == 0 and new_m > 0:
                        member_count += 1
                    elif old_m > 0 and new_m == 0:
                        member_count -= 1
            c_deltas = consumer_deltas.get(owner_name)
            if c_deltas:
                base_c = base.consumer_counts_by_owner.get(owner_name, empty_str_map)
                for consumer, delta in c_deltas.items():
                    if not delta:
                        continue
                    old_c = base_c.get(consumer, 0)
                    new_c = old_c + delta
                    if old_c == 0 and new_c > 0:
                        consumer_count += 1
                    elif old_c > 0 and new_c == 0:
                        consumer_count -= 1
            if derived_count > 0:
                owners_list.append(
                    InheritedMethodOwner(
                        owner=owner_name,
                        derived_pairs=derived_count,
                        members=member_count,
                        consumers=consumer_count,
                    )
                )
        owners_tuple = tuple(
            sorted(owners_list, key=lambda item: (-item.derived_pairs, item.owner))
        )
    else:
        owners_tuple = tuple(
            sorted(
                (
                    InheritedMethodOwner(
                        owner=owner,
                        derived_pairs=len(found),
                        members=len({member for member, _ in found}),
                        consumers=len(consumers[owner]),
                    )
                    for owner, found in pairs.items()
                ),
                key=lambda item: (-item.derived_pairs, item.owner),
            )
        )

    if record_base:
        member_pair_counts: dict[str, dict[str, int]] = {}
        for owner, p_cnt in base_pair_counts_builder.items():
            mc: dict[str, int] = {}
            for member, _method_name in p_cnt:
                mc[member] = mc.get(member, 0) + 1
            member_pair_counts[owner] = mc
        base_slot[0] = _AmortizationBaseline(
            session_id=session_id,
            index=index,
            own_by_symbol=cur_own,
            symbols=cur_symbols,
            per_file=per_file_aggs,
            owner_cache=dict(methods._owner_cache),
            total_lines=total_lines,
            total_calls=total_calls,
            typed=typed,
            reaching=reaching,
            owners_by_name={item.owner: item for item in owners_tuple},
            call_key_cache=owners_by_call_key,
            pair_counts_by_owner={
                owner: dict(cnt) for owner, cnt in base_pair_counts_builder.items()
            },
            consumer_counts_by_owner={
                owner: dict(cnt) for owner, cnt in base_consumer_counts_builder.items()
            },
            member_pair_counts_by_owner=member_pair_counts,
        )

    return InheritanceAmortization(
        scanned_line_count=total_lines,
        type_checker=session.status,
        attribute_calls=total_calls,
        receiver_typed_calls=typed,
        calls_reaching_inherited=reaching,
        owners=owners_tuple,
    )


def inheritance_amortization_for_calls(
    modules: tuple[ParsedModule, ...],
    calls: tuple[MethodCall, ...],
    index: CompactClassFamilyIndex,
    session: TypeCheckerSession,
) -> InheritanceAmortization:
    """Count inherited pairs reached by already-collected method calls."""

    if (
        ScanCache.active() is not None
        and modules
        and session.file_version(modules[0].file_path) is not None
        and len(calls)
        == sum(len(module.fact(_ModuleMethodCallsFact).calls) for module in modules)
    ):
        return _inheritance_amortization_incremental(modules, index, session)

    answers = session.receiver_types(call.span for call in calls)
    classes = ScannedClasses(index.classes_by_symbol)
    methods = InheritedMethods(index)
    pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    consumers: defaultdict[str, set[str]] = defaultdict(set)
    symbols_by_answer: dict[ReceiverType, tuple[str, ...]] = {}
    owners_by_call_key: dict[tuple[tuple[str, ...], str], tuple[str, ...]] = {}
    typed = reaching = 0
    for call in calls:
        answer = answers[call.span]
        symbols = symbols_by_answer.get(answer)
        if symbols is None:
            symbols = answer.owner(classes).class_symbols()
            symbols_by_answer[answer] = symbols
        if not symbols:
            continue
        typed += 1
        method_name = call.method
        call_key = (symbols, method_name)
        hit_owners = owners_by_call_key.get(call_key)
        if hit_owners is None:
            seen_owners: dict[str, None] = {}
            for symbol in symbols:
                for member in methods.reached(symbol):
                    owner = methods.owner_of(member, method_name)
                    if owner is not None:
                        pairs[owner].add((member, method_name))
                        seen_owners[owner] = None
            hit_owners = tuple(seen_owners)
            owners_by_call_key[call_key] = hit_owners
        if hit_owners:
            reaching += 1
            consumer = call.consumer
            for owner in hit_owners:
                consumers[owner].add(consumer)
    return InheritanceAmortization(
        scanned_line_count=sum(len(module.source.splitlines()) for module in modules),
        type_checker=session.status,
        attribute_calls=len(calls),
        receiver_typed_calls=typed,
        calls_reaching_inherited=reaching,
        owners=tuple(
            sorted(
                (
                    InheritedMethodOwner(
                        owner=owner,
                        derived_pairs=len(found),
                        members=len({member for member, _ in found}),
                        consumers=len(consumers[owner]),
                    )
                    for owner, found in pairs.items()
                ),
                key=lambda item: (-item.derived_pairs, item.owner),
            )
        ),
    )


@dataclass(frozen=True)
class GenericIntrospector(SemanticRecord):
    """A function a recorded run saw introspecting two or more classes."""

    consumer: str
    classes: int
    fields: int


@dataclass(frozen=True)
class IntrospectionAmortization(SemanticRecord):
    """Field uses generic introspection derives: without the declaration-driven
    consumer, each (field, consumer) pair is a hand-written statement."""

    trace: str
    consumers: int
    classes: int
    generic: tuple[GenericIntrospector, ...]

    @json_report_property()
    def derived_field_uses(self) -> int:
        return sum(item.fields for item in self.generic)


class IntrospectionSource(ABC):
    """Where the run-time introspection term comes from."""

    @abstractmethod
    def amortization(self) -> IntrospectionAmortization:
        raise NotImplementedError

    @staticmethod
    def from_argument(path: Path | None) -> IntrospectionSource:
        """The CLI option decoded once: a recorded trace, or none given."""
        match path:
            case Path():
                return RecordedTrace(path)
        return NoTrace()


@dataclass(frozen=True)
class NoTrace(IntrospectionSource):
    """No run was recorded, so the term is unmeasured, not zero."""

    def amortization(self) -> IntrospectionAmortization:
        return IntrospectionAmortization(
            trace="not measured: record a run with `pytest -p nra_introspection_trace` "
            "and pass --introspection-trace",
            consumers=0,
            classes=0,
            generic=(),
        )


@dataclass(frozen=True)
class RecordedTrace(IntrospectionSource):
    """A trace file written by the nra_introspection_trace pytest plugin."""

    path: Path

    def amortization(self) -> IntrospectionAmortization:
        trace = IntrospectionTrace.read(self.path)
        fields = {count.introspected: count.fields for count in trace.field_counts}
        by_consumer: defaultdict[str, set[str]] = defaultdict(set)
        for use in trace.uses:
            by_consumer[use.consumer].add(use.introspected)
        return IntrospectionAmortization(
            trace=f"recorded in {self.path} for {', '.join(trace.prefixes)}",
            consumers=len(by_consumer),
            classes=len({use.introspected for use in trace.uses}),
            generic=tuple(
                sorted(
                    (
                        GenericIntrospector(
                            consumer,
                            len(introspected),
                            sum(fields.get(name, 0) for name in introspected),
                        )
                        for consumer, introspected in by_consumer.items()
                        if len(introspected) >= 2
                    ),
                    key=lambda item: (-item.fields, item.consumer),
                )
            ),
        )
