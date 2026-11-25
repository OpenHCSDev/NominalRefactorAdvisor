"""One mypy build per scan, whose exported types answer every receiver it is asked about.

mypy is an optional extra (`nominal-refactor-advisor[types]`). Without it, or
when the build fails, every receiver gets a `FailedReceiverType` naming why,
so the report says the owner is unresolved and why; nothing is skipped
silently. The build runs in process once; each receiver is then a lookup in
its type map, keyed by the receiver's exact source span.
"""

from __future__ import annotations

import ast
import fcntl
import gc
import importlib
import importlib.util
import io
import os
import pickle
import signal
import stat
import struct
import sys
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

from nominal_refactor_advisor.json_reports import DataclassJsonReport
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.probe_owner import (
    FailedReceiverType,
    InferredReceiverType,
    ReceiverSpan,
    ReceiverType,
)

if TYPE_CHECKING:
    from mypy.build import BuildResult, State
    from mypy.errors import ErrorInfo, Errors
    from mypy.fscache import FileSystemCache
    from mypy.modulefinder import BuildSource
    from mypy.nodes import Expression, MypyFile
    from mypy.options import Options
    from mypy.server.update import FineGrainedBuildManager

    from nominal_refactor_advisor.ast_tools import ParsedModule
    from mypy.types import Type

TYPES_EXTRA = "pip install 'nominal-refactor-advisor[types]'"


@dataclass(frozen=True)
class TypeCheckerTiming(DataclassJsonReport):
    """Where a scan's type-checker time went."""

    build_seconds: float = 0.0
    lookup_seconds: float = 0.0
    lookups: int = 0


class TypeCheckerTimingLedger:
    """Collects the type-checker timing of the scan that opened it."""

    _active: ClassVar[ContextVar[TypeCheckerTimingLedger]]

    def __init__(self) -> None:
        self.timing = TypeCheckerTiming()

    def record(self, timing: TypeCheckerTiming) -> None:
        self.timing = TypeCheckerTiming(
            build_seconds=round(self.timing.build_seconds + timing.build_seconds, 3),
            lookup_seconds=round(self.timing.lookup_seconds + timing.lookup_seconds, 3),
            lookups=self.timing.lookups + timing.lookups,
        )

    @classmethod
    def active(cls) -> TypeCheckerTimingLedger:
        return cls._active.get()

    @classmethod
    @contextmanager
    def scope(cls) -> Iterator[TypeCheckerTimingLedger]:
        ledger = cls()
        token = cls._active.set(ledger)
        try:
            yield ledger
        finally:
            cls._active.reset(token)


# Outside a scan's scope the timing is recorded into a ledger nobody reads.
TypeCheckerTimingLedger._active = ContextVar(
    "type_checker_timing_ledger", default=TypeCheckerTimingLedger()
)


class TypeCheckerSources(ABC):
    """What one build reads: files on disk, or parsed modules' own text."""

    @abstractmethod
    def build_sources(self, options: Options) -> list[BuildSource]:
        raise NotImplementedError

    def pathed(self, options: Options) -> tuple[tuple[str, BuildSource], ...]:
        """Every source with the file it reads; mypy's optional path is refused here, once."""

        pathed = []
        for source in self.build_sources(options):
            if source.path is None:
                raise ValueError(f"mypy source `{source.module}` names no file")
            pathed.append((source.path, source))
        return tuple(pathed)


@dataclass(frozen=True)
class SourceFiles(TypeCheckerSources):
    """Files as they are on disk; mypy names their modules from the package layout."""

    paths: frozenset[str]

    def build_sources(self, options: Options) -> list[BuildSource]:
        return importlib.import_module("mypy.find_sources").create_source_list(
            sorted(self.paths), options
        )


@dataclass(frozen=True)
class SourceTexts(TypeCheckerSources):
    """Parsed modules' own text under their own names: a search state that exists
    only in memory types the same as one written to disk."""

    modules: tuple[ParsedModule, ...]

    def build_sources(self, options: Options) -> list[BuildSource]:
        build_source = importlib.import_module("mypy.modulefinder").BuildSource
        return [build_source(path, name, text=text) for path, name, text in self.state]

    @property
    def state(self) -> tuple[tuple[str, str, str], ...]:
        """Everything a build of these modules reads: each one's file, name and text."""

        return tuple((module.file_path, module.module_name, module.source) for module in self.modules)


def _writes_receiver_attribute(node: ast.AST, receiver_names: frozenset[str]) -> bool:
    match node:
        case ast.Attribute(
            value=ast.Name(id=root),
            ctx=ast.Store() | ast.Del(),
        ) if root in receiver_names:
            return True
    return False


def _has_exported_body_effect(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    receiver_names: frozenset[str],
) -> bool:
    """Whether a function body can affect types outside itself (receiver attribute
    definitions, globals, nested classes, or `yield` making an `async def` an async
    generator)."""

    is_async = isinstance(fn, ast.AsyncFunctionDef)
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Global | ast.ClassDef):
            return True
        if is_async and isinstance(node, ast.Yield | ast.YieldFrom):
            return True
        if receiver_names and _writes_receiver_attribute(node, receiver_names):
            return True
        for field_name in node._fields:
            child = getattr(node, field_name, None)
            if isinstance(child, ast.AST):
                stack.append(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ast.AST):
                        stack.append(item)
    return False


def _is_ellipsis_body(body: list[ast.stmt]) -> bool:
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )


def _statement_start_line(stmt: ast.stmt) -> int:
    if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        if stmt.decorator_list:
            return stmt.decorator_list[0].lineno
    return stmt.lineno


def _elide_unrequested_annotated_bodies(
    text: str,
    requested_ranges: tuple[tuple[int, int], ...],
) -> str:
    """Replace annotated function bodies that hold no requested span and have no
    exported body effect with `...`, preserving line numbers when spans exist in the
    file. Mypy reads the declared signature without type-checking unrequested bodies."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []

    def visit_block(stmts: list[ast.stmt], in_class: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                visit_block(stmt.body, True)
                continue
            if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if stmt.returns is None or not stmt.body or _is_ellipsis_body(stmt.body):
                continue
            fn_start = _statement_start_line(stmt)
            fn_end = stmt.end_lineno or stmt.body[-1].end_lineno or stmt.body[-1].lineno
            if any(
                start <= fn_end and end >= fn_start
                for start, end in requested_ranges
            ):
                continue
            receiver_names: frozenset[str] = frozenset()
            if in_class:
                positional = (*stmt.args.posonlyargs, *stmt.args.args)
                if positional:
                    receiver_names = frozenset((positional[0].arg,))
            if _has_exported_body_effect(stmt, receiver_names):
                continue
            first = stmt.body[0]
            first_anchor: ast.stmt | ast.expr = (
                first.decorator_list[0]
                if isinstance(
                    first, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                )
                and first.decorator_list
                else first
            )
            start_idx = first_anchor.lineno - 1
            if start_idx <= stmt.lineno - 1:
                continue
            first_line = lines[start_idx]
            leading = first_line[: first_anchor.col_offset]
            if leading.strip() not in ("", "@"):
                continue
            indent = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
            last = stmt.body[-1]
            end_idx = last.end_lineno or last.lineno
            pad = "\n" * max(0, end_idx - start_idx - 1) if requested_ranges else ""
            replacements.append((start_idx, end_idx, f"{indent}...\n{pad}"))

    visit_block(tree.body, False)
    if not replacements:
        return text
    output: list[str] = []
    cursor = 0
    for start_idx, end_idx, replacement in sorted(replacements):
        if start_idx < cursor:
            continue
        output.extend(lines[cursor:start_idx])
        output.append(replacement)
        cursor = end_idx
    output.extend(lines[cursor:])
    return "".join(output)


class ExpressionExtractionScope(ABC):
    """Which built expressions a mypy run checks in untyped defs and exports."""

    @abstractmethod
    def configure_options(
        self,
        options: Options,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_sources(
        self,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> list[BuildSource]:
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        result: BuildResult,
        options: Options,
        unique_paths: frozenset[str],
    ) -> Mapping[ReceiverSpan, InferredReceiverType]:
        raise NotImplementedError

    @staticmethod
    def _inferred_type(
        found: Type,
        options: Options,
        format_type_bare: object,
        mypy_types: ModuleType | None = None,
    ) -> InferredReceiverType:
        types = (
            importlib.import_module("mypy.types")
            if mypy_types is None
            else mypy_types
        )
        return InferredReceiverType(
            format_type_bare(found, options, verbosity=2),  # type: ignore[operator]
            _classes_of(found, types),
            _instances_only(found, types),
            _families_of(found, types),
        )


class LazyInferredTypes(Mapping[ReceiverSpan, InferredReceiverType]):
    """Format mypy types on first lookup so unqueried spans cost zero formatting."""

    def __init__(
        self,
        raw: dict[ReceiverSpan, Type],
        options: Options,
        format_type_bare: object,
        mypy_types: ModuleType,
    ) -> None:
        self._raw = raw
        self._options = options
        self._format_type_bare = format_type_bare
        self._mypy_types = mypy_types
        self._formatted: dict[ReceiverSpan, InferredReceiverType] = {}

    def __getitem__(self, span: ReceiverSpan) -> InferredReceiverType:
        cached = self._formatted.get(span)
        if cached is None:
            raw_type = self._raw[span]
            cached = ExpressionExtractionScope._inferred_type(
                raw_type,
                self._options,
                self._format_type_bare,
                self._mypy_types,
            )
            self._formatted[span] = cached
        return cached

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def __iter__(self) -> Iterator[ReceiverSpan]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


@dataclass(frozen=True)
class AllBuiltExpressions(ExpressionExtractionScope):
    """Check untyped defs and export every typed expression across all built files."""

    def configure_options(
        self,
        options: Options,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> None:
        return None

    def prepare_sources(
        self,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> list[BuildSource]:
        return [source for _path, source in unique]

    def extract(
        self,
        result: BuildResult,
        options: Options,
        unique_paths: frozenset[str],
    ) -> Mapping[ReceiverSpan, InferredReceiverType]:
        format_type_bare = importlib.import_module("mypy.messages").format_type_bare
        mypy_types = importlib.import_module("mypy.types")
        subexpressions = importlib.import_module("mypy.server.subexpr").get_subexpressions
        raw: dict[ReceiverSpan, Type] = {}
        for state in result.graph.values():
            if state.xpath not in unique_paths or state.tree is None:
                continue
            for expression in subexpressions(state.tree):
                found = result.types.get(expression)
                if found is not None:
                    span = ReceiverSpan(
                        state.xpath,
                        expression.line,
                        expression.column,
                        expression.end_line,
                        expression.end_column,
                    )
                    raw[span] = found
        return LazyInferredTypes(raw, options, format_type_bare, mypy_types)


@dataclass(frozen=True)
class RequestedSpanExpressions(ExpressionExtractionScope):
    """Check untyped defs only in files holding requested spans, elide unrequested
    annotated function bodies so mypy skips checking them, and export only those spans."""

    file_paths: frozenset[str]
    coords_by_file: Mapping[str, frozenset[tuple[int, int, int, int]]]
    line_ranges_by_file: Mapping[str, tuple[tuple[int, int], ...]]

    @classmethod
    def of(cls, spans: Iterable[ReceiverSpan]) -> RequestedSpanExpressions:
        grouped: defaultdict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
        ranges: defaultdict[str, set[tuple[int, int]]] = defaultdict(set)
        for span in spans:
            grouped[span.file_path].add(
                (span.line, span.column, span.end_line, span.end_column)
            )
            ranges[span.file_path].add((span.line, span.end_line))
        return cls(
            frozenset(grouped),
            {path: frozenset(coords) for path, coords in grouped.items()},
            {path: tuple(sorted(line_ranges)) for path, line_ranges in ranges.items()},
        )

    def configure_options(
        self,
        options: Options,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> None:
        options.check_untyped_defs = False
        options.ignore_errors = True
        for path, source in unique:
            if path in self.file_paths:
                options.per_module_options[source.module] = {"check_untyped_defs": True}

    def prepare_sources(
        self,
        unique: Sequence[tuple[str, BuildSource]],
    ) -> list[BuildSource]:
        build_source = importlib.import_module("mypy.modulefinder").BuildSource
        prepared: list[BuildSource] = []
        for path, source in unique:
            raw = source.text
            if raw is None:
                try:
                    raw = Path(path).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    prepared.append(source)
                    continue
            elided = _elide_unrequested_annotated_bodies(
                raw, self.line_ranges_by_file.get(path, ())
            )
            prepared.append(
                source
                if elided is raw
                else build_source(source.path, source.module, text=elided)
            )
        return prepared

    def extract(
        self,
        result: BuildResult,
        options: Options,
        unique_paths: frozenset[str],
    ) -> Mapping[ReceiverSpan, InferredReceiverType]:
        format_type_bare = importlib.import_module("mypy.messages").format_type_bare
        mypy_types = importlib.import_module("mypy.types")
        subexpressions = importlib.import_module("mypy.server.subexpr").get_subexpressions
        raw: dict[ReceiverSpan, Type] = {}
        for state in result.graph.values():
            if state.xpath not in unique_paths or state.tree is None:
                continue
            needed = self.coords_by_file.get(state.xpath)
            if not needed:
                continue
            for expression in subexpressions(state.tree):
                coords = (
                    expression.line,
                    expression.column,
                    expression.end_line,
                    expression.end_column,
                )
                if coords not in needed:
                    continue
                found = result.types.get(expression)
                if found is not None:
                    span = ReceiverSpan(state.xpath, *coords)
                    raw[span] = found
        return LazyInferredTypes(raw, options, format_type_bare, mypy_types)


@contextmanager
def _quiet_mypy() -> Iterator[None]:
    """Silence stray mypy stdout/stderr diagnostics during internal builds."""

    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        yield


class TypeCheckerSession(ABC):
    """The scan's type checker: one finished build, or the reason there is none."""

    @abstractmethod
    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> str:
        """What the type checker could answer: how much it typed, or why nothing."""
        raise NotImplementedError

    def file_version(self, file_path: str) -> int | None:
        """Per-file monotonic version for incremental sessions, or None when unversioned."""
        del file_path
        return None

    @staticmethod
    def _active_incremental() -> FineGrainedTypeChecker | None:
        if ScanCache.active() is None:
            return None
        return _incremental_type_checker_slot()[0]

    @staticmethod
    def has_active_incremental() -> bool:
        """Whether an in-memory incremental type checker is active in this scope."""
        return TypeCheckerSession._active_incremental() is not None

    @staticmethod
    def start_speculative_sync(sources: SourceTexts) -> None:
        """Start an asynchronous copy-on-write mypy delta sync in a child process
        so subsequent ``started(sources)`` can merge its results if unchanged."""

        existing = TypeCheckerSession._active_incremental()
        if existing is not None:
            existing.start_speculative_sync(sources)

    @staticmethod
    def send_speculative_followup(sources: SourceTexts) -> None:
        """Send a follow-up stage delta to an already-running speculative mypy sync
        so it finishes syncing and extracting before settlement arrives."""

        existing = TypeCheckerSession._active_incremental()
        if existing is not None:
            existing.send_speculative_followup(sources)

    @staticmethod
    def cancel_speculative_sync() -> None:
        """Abort and reap any in-flight speculative mypy sync owned by this process."""

        existing = TypeCheckerSession._active_incremental()
        if existing is not None:
            existing.cancel_speculative_sync()

    @staticmethod
    def seed_incremental(sources: SourceTexts) -> TypeCheckerSession:
        """Seed or sync an in-memory fine-grained mypy engine inside the active
        ``ScanCache.scope()`` so subsequent queries and forked workers share it COW."""

        if ScanCache.active() is None:
            return TypeCheckerSession.started(sources)
        slot = _incremental_type_checker_slot()
        existing = slot[0]
        if existing is not None:
            synced = existing.sync_corpus(sources)
            if synced is not None:
                return synced
        built = FineGrainedTypeChecker.seeded(sources)
        slot[0] = built if isinstance(built, FineGrainedTypeChecker) else None
        return built

    @staticmethod
    def started(sources: TypeCheckerSources) -> TypeCheckerSession:
        """Build these sources once with exported types, or name why that failed:
        a session for receivers asked about later, so it keeps every file's types."""

        if isinstance(sources, SourceTexts):
            existing = TypeCheckerSession._active_incremental()
            if existing is not None:
                synced = existing.sync_corpus(sources)
                if synced is not None:
                    return synced
                if existing._failed:
                    _incremental_type_checker_slot()[0] = None
            if ScanCache.active() is not None and len(sources.modules) >= 16:
                seeded = FineGrainedTypeChecker.seeded(sources)
                if isinstance(seeded, FineGrainedTypeChecker):
                    _incremental_type_checker_slot()[0] = seeded
                    return seeded
        return TypeCheckerSession._built(sources, AllBuiltExpressions())

    @staticmethod
    def answers(
        sources: TypeCheckerSources, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        """Build these sources once and answer `spans`. Only the files that hold one
        check untyped defs and only the requested spans are formatted and exported."""

        asked = tuple(spans)
        if isinstance(sources, SourceTexts):
            existing = TypeCheckerSession._active_incremental()
            if existing is not None:
                synced = existing.sync_overlay(sources)
                if synced is not None:
                    return synced.receiver_types(asked)
                if existing._failed:
                    _incremental_type_checker_slot()[0] = None
        session = TypeCheckerSession._built(sources, RequestedSpanExpressions.of(asked))
        try:
            return session.receiver_types(asked)
        finally:
            session.stop()

    @staticmethod
    def _built(
        sources: TypeCheckerSources,
        scope: ExpressionExtractionScope,
    ) -> TypeCheckerSession:
        """One build of `sources`, configured and exported according to `scope`."""

        if importlib.util.find_spec("mypy") is None:
            return UnavailableTypeChecker(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        errors = importlib.import_module("mypy.errors")
        options = TypeCheckerSession.options()
        options.export_types = True
        options.preserve_asts = True
        unique, shared = TypeCheckerSession.unique_sources(sources, options)
        scope.configure_options(options, unique)
        build_sources = scope.prepare_sources(unique)
        started = perf_counter()
        try:
            with _quiet_mypy():
                result = build.build(build_sources, options)
        except errors.CompileError as error:
            return UnavailableTypeChecker(
                "the mypy build failed: " + "\n".join(error.messages)[-500:]
            )
        except (Exception, SystemExit) as error:  # noqa: BLE001 - mypy ICE
            return UnavailableTypeChecker(
                f"the mypy build hit an internal error: {type(error).__name__}: {error}"
            )
        session = BuiltTypeChecker.of(
            result,
            options,
            frozenset(path for path, _source in unique),
            shared,
            scope,
        )
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(build_seconds=perf_counter() - started)
        )
        return session

    @staticmethod
    def options() -> Options:
        """The build every scan's type check shares."""

        options = importlib.import_module("mypy.options").Options()
        options.incremental = False
        options.strict_equality = True
        # Unannotated functions are most of the receivers a scan asks about.
        options.check_untyped_defs = True
        # Every scanned file is a source; imports outside the scan stay unanalyzed.
        options.follow_imports = "skip"
        options.ignore_missing_imports = True
        options.python_version = sys.version_info[:2]
        # Raise mypy internal errors as Python exceptions instead of calling
        # sys.exit(2) and aborting an entire scan or plan synthesis run.
        options.raise_exceptions = True
        return options

    @staticmethod
    def unique_sources(
        sources: TypeCheckerSources, options: Options
    ) -> tuple[list[tuple[str, BuildSource]], dict[str, str]]:
        """The sources mypy can build together, and why each other one is left out:
        mypy refuses the whole build when two files claim one module name."""

        sources_by_module: defaultdict[str, list[tuple[str, BuildSource]]] = defaultdict(list)
        for path, source in sources.pathed(options):
            sources_by_module[source.module].append((path, source))
        shared = {
            path: f"module `{module}` is shared by {len(found)} scanned files"
            for module, found in sources_by_module.items()
            if len(found) > 1
            for path, _source in found
        }
        return [only for only, *rest in sources_by_module.values() if not rest], shared

    @staticmethod
    @contextmanager
    def session(sources: TypeCheckerSources) -> Iterator[TypeCheckerSession]:
        session = TypeCheckerSession.started(sources)
        try:
            yield session
        finally:
            session.stop()

    @staticmethod
    @contextmanager
    def scoped_session(
        sources: TypeCheckerSources, spans: Iterable[ReceiverSpan]
    ) -> Iterator[TypeCheckerSession]:
        """Build these sources once, formatting and exporting only `spans`."""

        if isinstance(sources, SourceTexts):
            existing = TypeCheckerSession._active_incremental()
            if existing is not None:
                synced = existing.sync_corpus(sources)
                if synced is not None:
                    yield synced
                    return
        session = TypeCheckerSession._built(sources, RequestedSpanExpressions.of(spans))
        try:
            yield session
        finally:
            session.stop()


@dataclass(frozen=True)
class UnavailableTypeChecker(TypeCheckerSession):
    """No build: every receiver's type is this failure."""

    reason: str

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        return {span: FailedReceiverType(self.reason) for span in spans}

    def stop(self) -> None:
        """Nothing was started."""

    @property
    def status(self) -> str:
        return f"no type checker: {self.reason}"


@dataclass
class DeferredTypeChecker(TypeCheckerSession):
    """One build of these sources, started by the first question asked of it, so a caller
    with several checks pays for the build once and only when a receiver needs it."""

    sources: TypeCheckerSources
    _started: TypeCheckerSession | None = None

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        if self._started is None:
            self._started = TypeCheckerSession.started(self.sources)
        return self._started.receiver_types(spans)

    def stop(self) -> None:
        if self._started is not None:
            self._started.stop()
            self._started = None

    @property
    def status(self) -> str:
        return "not started" if self._started is None else self._started.status


def _classes_of(found: Type, mypy_types: ModuleType | None = None) -> tuple[str, ...]:
    """The classes a mypy type is: the decoder at the type checker's boundary."""

    types = importlib.import_module("mypy.types") if mypy_types is None else mypy_types
    proper = types.get_proper_type(found)
    match proper:
        case types.Instance():
            return (proper.type.fullname,)
        case types.TypeType():
            return _classes_of(proper.item, types)
        case types.TypeVarType():
            return _classes_of(proper.upper_bound, types)
        case types.UnionType():
            return tuple(
                chain.from_iterable(
                    _classes_of(item, types)
                    for item in proper.items
                    if not isinstance(types.get_proper_type(item), types.NoneType)
                )
            )
    return ()


def _instances_only(found: Type, mypy_types: ModuleType | None = None) -> bool:
    """Whether every value of a mypy type is an instance that is not a class. A
    class is itself an instance of `type`, `object`, its metaclass and any
    protocol it satisfies, so none of those, nor an unknown type, proves it."""

    types = importlib.import_module("mypy.types") if mypy_types is None else mypy_types
    proper = types.get_proper_type(found)
    match proper:
        case types.Instance():
            info = proper.type
            return not (
                info.is_protocol
                or info.fullname == "builtins.object"
                or info.has_base("builtins.type")
            )
        case types.TupleType() | types.TypedDictType() | types.LiteralType() | types.NoneType():
            return True
        case types.TypeVarType():
            return _instances_only(proper.upper_bound, types)
        case types.UnionType():
            return all(_instances_only(item, types) for item in proper.items)
    return False


def _families_of(found: Type, mypy_types: ModuleType | None = None) -> tuple[str, ...]:
    """The classes `C` when every value of a mypy type is a class `C` or a subclass:
    `type[C]` for a class that is no protocol, or a union of them; otherwise none."""

    types = importlib.import_module("mypy.types") if mypy_types is None else mypy_types
    proper = types.get_proper_type(found)
    match proper:
        case types.TypeType():
            item = types.get_proper_type(proper.item)
            if isinstance(item, types.Instance) and not item.type.is_protocol:
                return (item.type.fullname,)
        case types.UnionType():
            families = tuple(_families_of(item, types) for item in proper.items)
            if all(families):
                return tuple(chain.from_iterable(families))
    return ()


@dataclass(frozen=True)
class BuiltTypeChecker(TypeCheckerSession):
    """A finished build's types for the scanned files, keyed by expression span."""

    inferred: Mapping[ReceiverSpan, InferredReceiverType]
    shared: Mapping[str, str]

    @classmethod
    def of(
        cls,
        result: BuildResult,
        options: Options,
        paths: frozenset[str],
        shared: Mapping[str, str],
        scope: ExpressionExtractionScope = AllBuiltExpressions(),
    ) -> BuiltTypeChecker:
        """The selected typed expressions of the scanned files, with the classes each is."""

        return cls(scope.extract(result, options, paths), shared)

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        started = perf_counter()
        answers = {span: self.answer(span) for span in spans}
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(lookup_seconds=perf_counter() - started, lookups=len(answers))
        )
        return answers

    def answer(self, span: ReceiverSpan) -> ReceiverType:
        if span in self.inferred:
            return self.inferred[span]
        if span.file_path in self.shared:
            return FailedReceiverType(self.shared[span.file_path])
        return FailedReceiverType(f"the build exported no type at {span.location}")

    def stop(self) -> None:
        """The build holds no process or files."""

    @property
    def status(self) -> str:
        return (
            f"mypy typed {len(self.inferred)} expression(s); "
            f"{len(self.shared)} file(s) left out for a module name another file shares"
        )


@ScanCache.cached
def _incremental_type_checker_slot() -> list[FineGrainedTypeChecker | None]:
    """The active fine-grained mypy engine for this ``ScanCache.scope()``, if seeded."""

    return [None]


class FineGrainedTypeChecker(TypeCheckerSession):
    """One live in-memory mypy ``FineGrainedBuildManager`` updated on source deltas.

    Seeded once before ``RefactorSearch`` forks so every worker inherits the built
    module graph copy-on-write and updates only the files a candidate move changed,
    without touching disk or rebuilding untouched modules.
    """

    def __init__(
        self,
        manager: FineGrainedBuildManager,
        fscache: FileSystemCache,
        options: Options,
        format_type_bare: object,
        mypy_types: ModuleType,
        subexpressions: Callable[[MypyFile], list[Expression]],
        module_prefix: Callable[[dict[str, State], str], str | None],
        hash_digest: Callable[[bytes], str],
        text_by_module: dict[str, str],
        path_by_module: dict[str, str],
        shared: dict[str, str],
        mtime: int,
    ) -> None:
        self._manager = manager
        self._fscache = fscache
        self._options = options
        self._format_type_bare = format_type_bare
        self._mypy_types = mypy_types
        self._subexpressions = subexpressions
        self._module_prefix = module_prefix
        self._hash_digest = hash_digest
        self._text_by_module = text_by_module
        self._path_by_module = path_by_module
        self._state_by_path: dict[str, State] = {
            state.xpath: state for state in manager.graph.values()
        }
        self._shared = shared
        self._raw_by_file: dict[str, dict[ReceiverSpan, Type]] = {}
        self._formatted_by_file: dict[str, dict[ReceiverSpan, InferredReceiverType]] = {}
        self._file_versions: dict[str, int] = {}
        self._mtime = mtime
        self._deferred_split_delta: list[tuple[str, str, str]] = []
        self._split_touched_modules: set[str] = set()
        self._split_typed_coords_by_file: dict[
            str, frozenset[tuple[int, int, int, int]]
        ] = {}
        self._pending_presync: (
            tuple[
                int,
                int | None,
                int,
                int,
                dict[str, tuple[str, str]],
                dict[str, tuple[str, str]] | None,
            ]
            | None
        ) = None
        self._member_expr_cls = importlib.import_module("mypy.nodes").MemberExpr
        self._failed: bool = False
        self._type_errors: FineGrainedTypeErrors | None = None

    def shared_type_errors(self) -> FineGrainedTypeErrors | None:
        """The lockstep ``FineGrainedTypeErrors`` view backed by this engine, if active."""
        if self._failed:
            return None
        return self._type_errors

    def file_version(self, file_path: str) -> int | None:
        return self._file_versions.get(file_path, 0)

    @staticmethod
    def _ensure_virtual_dir(fscache: FileSystemCache, directory: str, mtime: int) -> None:
        norm = os.path.normpath(directory)
        if norm in fscache.listdir_cache:
            return
        pending: list[str] = []
        cursor = norm
        while cursor not in fscache.listdir_cache:
            if os.path.isdir(cursor):
                try:
                    fscache.listdir_cache[cursor] = list(os.listdir(cursor))
                except OSError:
                    fscache.listdir_cache[cursor] = []
                break
            pending.append(cursor)
            parent = os.path.normpath(os.path.dirname(cursor))
            if parent == cursor:
                fscache.listdir_cache[cursor] = []
                break
            cursor = parent
        dir_stat = os.stat_result(
            (stat.S_IFDIR | 0o755, 0, 0, 2, 0, 0, 0, mtime, mtime, mtime)
        )
        for virtual_dir in reversed(pending):
            fscache.stat_or_none_cache[virtual_dir] = dir_stat
            fscache.exists_case_cache[virtual_dir] = True
            fscache.listdir_error_cache.pop(virtual_dir, None)
            fscache.listdir_cache.setdefault(virtual_dir, [])
            parent = os.path.normpath(os.path.dirname(virtual_dir))
            child = os.path.basename(virtual_dir)
            if parent != virtual_dir and parent in fscache.listdir_cache:
                entries = fscache.listdir_cache[parent]
                if child and child not in entries:
                    entries.append(child)

    @classmethod
    def _put_memory_file(
        cls,
        fscache: FileSystemCache,
        hash_digest: Callable[[bytes], str],
        path: str,
        text: str,
        mtime: int,
        register_in_dir: bool,
    ) -> None:
        data = text.encode("utf-8")
        fscache.read_cache[path] = data
        fscache.hash_cache[path] = hash_digest(data)
        fscache.read_error_cache.pop(path, None)
        fscache.stat_or_none_cache[path] = os.stat_result(
            (stat.S_IFREG | 0o644, 0, 0, 1, 0, 0, len(data), mtime, mtime, mtime)
        )
        fscache.isfile_case_cache[path] = True
        fscache.exists_case_cache[path] = True
        parent = os.path.normpath(os.path.dirname(path))
        if register_in_dir or parent in fscache.listdir_cache:
            cls._ensure_virtual_dir(fscache, parent, mtime)
            name = os.path.basename(path)
            entries = fscache.listdir_cache[parent]
            if name and name not in entries:
                entries.append(name)

    @staticmethod
    def _remove_memory_file(
        fscache: FileSystemCache,
        manager: FineGrainedBuildManager,
        module_name: str,
        path: str,
    ) -> None:
        fscache.read_cache.pop(path, None)
        fscache.hash_cache.pop(path, None)
        fscache.read_error_cache.pop(path, None)
        fscache.stat_or_none_cache[path] = None
        fscache.isfile_case_cache[path] = False
        fscache.exists_case_cache[path] = False
        parent = os.path.normpath(os.path.dirname(path))
        if parent in fscache.listdir_cache:
            name = os.path.basename(path)
            entries = fscache.listdir_cache[parent]
            if name in entries:
                entries.remove(name)
        manager.manager.ast_cache.pop(module_name, None)

    def _delete_memory_file(self, module_name: str, path: str) -> None:
        self._remove_memory_file(self._fscache, self._manager, module_name, path)

    @classmethod
    def seeded(cls, sources: SourceTexts) -> TypeCheckerSession:
        """Seed one live ``FineGrainedBuildManager`` over ``sources`` in memory."""

        if importlib.util.find_spec("mypy") is None:
            return UnavailableTypeChecker(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        errors = importlib.import_module("mypy.errors")
        fscache_mod = importlib.import_module("mypy.fscache")
        modulefinder = importlib.import_module("mypy.modulefinder")
        update_mod = importlib.import_module("mypy.server.update")
        subexpr_mod = importlib.import_module("mypy.server.subexpr")
        util_mod = importlib.import_module("mypy.util")
        messages_mod = importlib.import_module("mypy.messages")
        mypy_types = importlib.import_module("mypy.types")

        options = TypeCheckerSession.options()
        options.incremental = True
        options.fine_grained_incremental = True
        options.use_fine_grained_cache = False
        options.local_partial_types = True
        options.cache_dir = os.devnull
        options.export_types = True
        options.preserve_asts = True
        options.ignore_errors = False
        # Force reads through FileSystemCache instead of mypy 2.x's native parser
        # opening unmodified files directly from disk.
        options.native_parser = False

        unique, shared = TypeCheckerSession.unique_sources(sources, options)
        fscache: FileSystemCache = fscache_mod.FileSystemCache()
        hash_digest: Callable[[bytes], str] = util_mod.hash_digest
        build_source = modulefinder.BuildSource
        text_by_module: dict[str, str] = {}
        path_by_module: dict[str, str] = {}
        build_sources: list[BuildSource] = []
        mtime = 1
        for path, source in unique:
            text = source.text or ""
            mtime += 1
            cls._put_memory_file(
                fscache,
                hash_digest,
                path,
                text,
                mtime,
                register_in_dir=not os.path.exists(path),
            )
            text_by_module[source.module] = text
            path_by_module[source.module] = path
            build_sources.append(build_source(path, source.module, None))

        started = perf_counter()
        try:
            with _quiet_mypy():
                result = build.build(build_sources, options, fscache=fscache)
                manager: FineGrainedBuildManager = update_mod.FineGrainedBuildManager(
                    result
                )
        except errors.CompileError as error:
            return UnavailableTypeChecker(
                "the mypy build failed: " + "\n".join(error.messages)[-500:]
            )
        except (Exception, SystemExit) as error:  # noqa: BLE001 - mypy ICE
            return UnavailableTypeChecker(
                f"the mypy build hit an internal error: {type(error).__name__}: {error}"
            )
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(build_seconds=perf_counter() - started)
        )
        checker = cls(
            manager=manager,
            fscache=fscache,
            options=options,
            format_type_bare=messages_mod.format_type_bare,
            mypy_types=mypy_types,
            subexpressions=subexpr_mod.get_subexpressions,
            module_prefix=update_mod.module_prefix,
            hash_digest=hash_digest,
            text_by_module=text_by_module,
            path_by_module=path_by_module,
            shared=shared,
            mtime=mtime,
        )
        if sys.platform.startswith("linux") and len(path_by_module) >= 16:
            checker._type_errors = FineGrainedTypeErrors(
                manager=manager,
                fscache=fscache,
                options=options,
                module_prefix=update_mod.module_prefix,
                hash_digest=hash_digest,
                text_by_module=text_by_module,
                path_by_module=path_by_module,
                errors_by_file_target=FineGrainedTypeErrors._extract_errors_by_file_target(
                    manager.manager.errors
                ),
                mtime=mtime,
                source_checker=checker,
            )
        return checker

    def _invalidate_shared_type_errors(self) -> None:
        shared = self._type_errors
        self._type_errors = None
        if shared is not None and ScanCache.active() is not None:
            errors_slot = _incremental_type_errors_slot()
            if errors_slot[0] is shared:
                errors_slot[0] = None

    def _apply_delta_and_touched(
        self,
        changed: list[tuple[str, str, str]],
        removed: list[tuple[str, str]],
        shared: dict[str, str],
    ) -> set[str] | None:
        if not changed and not removed:
            self._shared = shared
            return set()
        started = perf_counter()
        shared_errors = self._type_errors
        for module_name, path in removed:
            self._delete_memory_file(module_name, path)
            self._text_by_module.pop(module_name, None)
            self._path_by_module.pop(module_name, None)
            self._state_by_path.pop(path, None)
            self._raw_by_file.pop(path, None)
            self._formatted_by_file.pop(path, None)
            self._split_typed_coords_by_file.pop(path, None)
            self._file_versions[path] = self._file_versions.get(path, 0) + 1
            if shared_errors is not None:
                shared_errors._set_file_targets(path, None)
        for module_name, path, text in changed:
            old_path = self._path_by_module.get(module_name)
            if old_path is not None and old_path != path:
                self._delete_memory_file(module_name, old_path)
                self._state_by_path.pop(old_path, None)
                self._raw_by_file.pop(old_path, None)
                self._formatted_by_file.pop(old_path, None)
                self._split_typed_coords_by_file.pop(old_path, None)
                self._file_versions[old_path] = self._file_versions.get(old_path, 0) + 1
                if shared_errors is not None:
                    shared_errors._set_file_targets(old_path, None)
            self._mtime += 1
            is_new = module_name not in self._path_by_module or old_path != path
            self._put_memory_file(
                self._fscache,
                self._hash_digest,
                path,
                text,
                self._mtime,
                register_in_dir=is_new or not os.path.exists(path),
            )
            self._manager.manager.ast_cache.pop(module_name, None)
            self._text_by_module[module_name] = text
            self._path_by_module[module_name] = path
        self._manager.previous_targets_with_errors = set()
        try:
            with _quiet_mypy():
                self._manager.update(
                    [(module_name, path) for module_name, path, _text in changed],
                    removed,
                )
        except (Exception, SystemExit):  # noqa: BLE001 - mypy fine-grained ICE
            self._failed = True
            self._invalidate_shared_type_errors()
            return None
        if self._manager.blocking_error is not None or self._manager.stale:
            self._failed = True
            self._invalidate_shared_type_errors()
            return None
        if shared_errors is not None:
            shared_errors._mtime = self._mtime
            shared_errors._record_updated_errors(changed)
        touched_modules: set[str] = (
            {module_name for module_name, _path, _text in changed}
            | {module_name for module_name, _path in removed}
            | set(self._manager.updated_modules)
        )
        for target in self._manager.processed_targets:
            prefix = self._module_prefix(self._manager.graph, target)
            if prefix is not None:
                touched_modules.add(prefix)
        for module_name in touched_modules:
            state = self._manager.graph.get(module_name)
            touched_path = (
                state.xpath
                if state is not None
                else self._path_by_module.get(module_name)
            )
            if touched_path is not None:
                self._raw_by_file.pop(touched_path, None)
                self._formatted_by_file.pop(touched_path, None)
                self._split_typed_coords_by_file.pop(touched_path, None)
                self._file_versions[touched_path] = (
                    self._file_versions.get(touched_path, 0) + 1
                )
                if state is not None:
                    self._state_by_path[touched_path] = state
                else:
                    self._state_by_path.pop(touched_path, None)
        self._shared = shared
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(build_seconds=perf_counter() - started)
        )
        return touched_modules

    def _apply_delta(
        self,
        changed: list[tuple[str, str, str]],
        removed: list[tuple[str, str]],
        shared: dict[str, str],
    ) -> bool:
        return self._apply_delta_and_touched(changed, removed, shared) is not None

    def _flush_deferred_split_delta(self) -> bool:
        if not self._deferred_split_delta:
            return True
        pending = self._deferred_split_delta
        self._deferred_split_delta = []
        self._split_touched_modules.clear()
        self._split_typed_coords_by_file.clear()
        return self._apply_delta(pending, [], self._shared)

    def _extract_split_file_answers(
        self,
        touched_modules: set[str],
    ) -> tuple[
        dict[str, dict[ReceiverSpan, InferredReceiverType]],
        dict[str, frozenset[tuple[int, int, int, int]]],
    ]:
        formatted_by_file: dict[str, dict[ReceiverSpan, InferredReceiverType]] = {}
        typed_coords_by_file: dict[str, frozenset[tuple[int, int, int, int]]] = {}
        all_types = self._manager.manager.all_types
        member_expr_cls = self._member_expr_cls
        format_type_bare = self._format_type_bare
        mypy_types = self._mypy_types
        options = self._options
        for module_name in touched_modules:
            state = self._manager.graph.get(module_name)
            if state is None or state.tree is None:
                continue
            path = state.xpath
            exprs = self._subexpressions(state.tree)
            member_receivers = {
                expr.expr for expr in exprs if isinstance(expr, member_expr_cls)
            }
            member_receiver_coords: set[tuple[int, int, int, int]] = set()
            last_by_coords: dict[tuple[int, int, int, int], Type] = {}
            for expression in exprs:
                found = all_types.get(expression)
                if (
                    found is None
                    or expression.end_line is None
                    or expression.end_column is None
                ):
                    continue
                coords = (
                    expression.line,
                    expression.column,
                    expression.end_line,
                    expression.end_column,
                )
                last_by_coords[coords] = found
                if expression in member_receivers:
                    member_receiver_coords.add(coords)
            formatted: dict[ReceiverSpan, InferredReceiverType] = {}
            for coords in member_receiver_coords:
                formatted[ReceiverSpan(path, *coords)] = (
                    ExpressionExtractionScope._inferred_type(
                        last_by_coords[coords],
                        options,
                        format_type_bare,
                        mypy_types,
                    )
                )
            formatted_by_file[path] = formatted
            typed_coords_by_file[path] = frozenset(last_by_coords)
        return formatted_by_file, typed_coords_by_file

    @staticmethod
    def _changed_top_level_symbols(
        old_text: str, new_text: str
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """Return `(changed_declared_symbols, added_referenced_names, added_import_modules)`
        via AST comparison of top-level statements."""
        try:
            old_mod = ast.parse(old_text)
            new_mod = ast.parse(new_text)
        except SyntaxError:
            return frozenset(), frozenset(), frozenset()

        def _stmt_names(stmt: ast.stmt) -> tuple[str, ...]:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                return (stmt.name,)
            if isinstance(stmt, ast.Assign):
                return tuple(
                    t.id for t in stmt.targets if isinstance(t, ast.Name)
                )
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                return (stmt.target.id,)
            return ()

        old_by_name: dict[str, list[str]] = defaultdict(list)
        old_imports: set[tuple[str | None, tuple[str, ...]]] = set()
        for stmt in old_mod.body:
            if isinstance(stmt, ast.ImportFrom):
                old_imports.add(
                    (stmt.module, tuple(alias.name for alias in stmt.names))
                )
                continue
            if isinstance(stmt, ast.Import):
                old_imports.add((None, tuple(alias.name for alias in stmt.names)))
                continue
            dumped = ast.dump(stmt)
            for name in _stmt_names(stmt):
                old_by_name[name].append(dumped)

        new_by_name: dict[str, list[str]] = defaultdict(list)
        new_nodes_by_name: dict[str, list[ast.stmt]] = defaultdict(list)
        added_refs: set[str] = set()
        added_import_mods: set[str] = set()
        for stmt in new_mod.body:
            if isinstance(stmt, ast.ImportFrom):
                key = (stmt.module, tuple(alias.name for alias in stmt.names))
                if key not in old_imports:
                    if stmt.module:
                        added_import_mods.add(stmt.module)
                    for alias in stmt.names:
                        added_refs.add(alias.name)
                continue
            if isinstance(stmt, ast.Import):
                key = (None, tuple(alias.name for alias in stmt.names))
                if key not in old_imports:
                    for alias in stmt.names:
                        added_import_mods.add(alias.name)
                continue
            dumped = ast.dump(stmt)
            for name in _stmt_names(stmt):
                new_by_name[name].append(dumped)
                new_nodes_by_name[name].append(stmt)

        changed_symbols: set[str] = set()
        for name in set(old_by_name) | set(new_by_name):
            if old_by_name.get(name) != new_by_name.get(name):
                changed_symbols.add(name)
                for stmt in new_nodes_by_name.get(name, ()):
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Name):
                            added_refs.add(node.id)
                        elif isinstance(node, ast.Attribute):
                            added_refs.add(node.attr)
        # Subtract identifiers already present in the old version of those changed statements.
        old_nodes_by_name: dict[str, list[ast.stmt]] = defaultdict(list)
        for stmt in old_mod.body:
            for name in _stmt_names(stmt):
                if name in changed_symbols:
                    old_nodes_by_name[name].append(stmt)
        old_refs: set[str] = set()
        for stmts in old_nodes_by_name.values():
            for stmt in stmts:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name):
                        old_refs.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        old_refs.add(node.attr)
        return (
            frozenset(changed_symbols),
            frozenset(added_refs - old_refs),
            frozenset(added_import_mods),
        )

    def _partition_independent_delta(
        self,
        changed: list[tuple[str, str, str]],
    ) -> tuple[
        list[tuple[str, str, str]],
        list[tuple[str, str, str]],
        dict[str, frozenset[str]],
        dict[str, frozenset[str]],
    ] | None:
        """Partition ``changed`` into two balanced groups weighted by fine-grained
        mypy trigger fanout and module size, keeping coupled modules together."""
        if len(changed) < 4:
            return None
        deps_map = self._manager.deps
        graph = self._manager.graph
        changed_by_mod = {mod: (mod, path, text) for mod, path, text in changed}
        changed_names = set(changed_by_mod)
        changed_syms_by_mod: dict[str, frozenset[str]] = {}
        added_refs_by_mod: dict[str, frozenset[str]] = {}
        added_imports_by_mod: dict[str, frozenset[str]] = {}
        fanout_by_mod: dict[str, int] = {}

        parent = {mod: mod for mod in changed_names}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        mods_by_changed_sym: dict[str, list[str]] = defaultdict(list)
        for mod, _path, new_text in changed:
            old_text = self._text_by_module.get(mod, "")
            changed_syms, added_refs, added_imports = self._changed_top_level_symbols(
                old_text, new_text
            )
            changed_syms_by_mod[mod] = changed_syms
            added_refs_by_mod[mod] = added_refs
            added_imports_by_mod[mod] = added_imports
            for sym in changed_syms:
                mods_by_changed_sym[sym].append(mod)
            hit_modules: set[str] = set()
            target_work = 0
            for sym in changed_syms:
                targets = deps_map.get(f"<{mod}.{sym}>")
                if not targets:
                    continue
                target_work += len(targets)
                for target in targets:
                    target_work += len(deps_map.get(f"<{target}>", ()))
                    prefix = self._module_prefix(graph, target)
                    if prefix is not None:
                        hit_modules.add(prefix)
            # Weight by 1-hop + 2-hop fine-grained mypy target fanout so heavy
            # hubs (e.g. a widely imported wire module) sit in their own bin.
            fanout_by_mod[mod] = max(
                1, target_work * 4 + len(hit_modules) * 8 + len(new_text) // 20000
            )
            for hit in hit_modules:
                if hit in changed_names and hit != mod:
                    union(mod, hit)

        # Union modules that co-edit the same symbol (e.g. class moves/extractions),
        # or add imports/references to another changed module's edited symbols.
        for mods in mods_by_changed_sym.values():
            first = mods[0]
            for other in mods[1:]:
                union(first, other)
        for mod in changed_names:
            for imp_mod in added_imports_by_mod[mod]:
                if imp_mod in changed_names and imp_mod != mod:
                    union(mod, imp_mod)
            refs = added_refs_by_mod[mod]
            if refs:
                for ref in refs:
                    for other in mods_by_changed_sym.get(ref, ()):
                        if other != mod:
                            union(mod, other)

        comps: dict[str, list[tuple[str, str, str]]] = {}
        for mod, item in changed_by_mod.items():
            comps.setdefault(find(mod), []).append(item)
        if len(comps) < 2:
            return None
        weighted_comps = sorted(
            (
                (sum(fanout_by_mod[item[0]] for item in comp), comp)
                for comp in comps.values()
            ),
            key=lambda pair: -pair[0],
        )
        group_a: list[tuple[str, str, str]] = []
        group_b: list[tuple[str, str, str]] = []
        weight_a = 0
        weight_b = 0
        for comp_weight, comp in weighted_comps:
            if weight_a <= weight_b:
                group_a.extend(comp)
                weight_a += comp_weight
            else:
                group_b.extend(comp)
                weight_b += comp_weight
        if not group_a or not group_b:
            return None
        # Keep the larger-module-count group in group_a (the main speculative
        # child where stage-1 follow-ups run) and isolate the compact heavy hub
        # in group_b so follow-ups rarely collide with group_b.
        if len(group_a) < len(group_b):
            group_a, group_b = group_b, group_a
        return group_a, group_b, changed_syms_by_mod, added_refs_by_mod

    def _run_speculative_child_body(
        self,
        changed: list[tuple[str, str, str]],
        shared: dict[str, str],
        req_r: int,
        res_w: int,
    ) -> tuple[bool, object]:
        """Execute stage-0 (optionally 2-way split-parallel when independent)
        and optional stage-1 follow-up inside the speculative child."""
        split = self._partition_independent_delta(changed)
        split_b_result: tuple[
            set[str],
            set[str],
            frozenset[str],
            dict[str, dict[ReceiverSpan, InferredReceiverType]],
            dict[str, frozenset[tuple[int, int, int, int]]],
        ] | None = None
        group_b: list[tuple[str, str, str]] = []
        changed_syms_by_mod: dict[str, frozenset[str]] = {}
        added_refs_by_mod: dict[str, frozenset[str]] = {}
        pre_deps_by_mod: dict[str, frozenset[str]] = {}

        if split is not None:
            group_a, group_b, changed_syms_by_mod, added_refs_by_mod = split
            for mod, _p, _t in changed:
                st = self._manager.graph.get(mod)
                pre_deps_by_mod[mod] = (
                    frozenset(st.dependencies) if st is not None else frozenset()
                )
            sub_r, sub_w = os.pipe()
            try:
                fcntl.fcntl(sub_w, 1031, 1 << 20)  # F_SETPIPE_SZ
            except OSError:
                pass
            sub_pid = os.fork()
            if sub_pid == 0:
                os.close(sub_r)
                os.close(req_r)
                os.close(res_w)
                gc.disable()
                try:
                    try:
                        touched_b = self._apply_delta_and_touched(group_b, [], shared)
                        if touched_b is None:
                            sub_payload = pickle.dumps((False, None))
                        else:
                            new_deps_b: set[str] = set()
                            for mod, _p, _t in group_b:
                                st = self._manager.graph.get(mod)
                                if st is not None:
                                    new_deps_b.update(
                                        set(st.dependencies) - pre_deps_by_mod[mod]
                                    )
                            fmt_b, coords_b = self._extract_split_file_answers(
                                touched_b
                            )
                            sub_payload = pickle.dumps(
                                (True, (touched_b, new_deps_b, fmt_b, coords_b))
                            )
                    except BaseException:  # noqa: BLE001
                        sub_payload = pickle.dumps((False, None))
                    with os.fdopen(sub_w, "wb") as out:
                        out.write(sub_payload)
                finally:
                    os._exit(0)
            os.close(sub_w)
            touched_a = self._apply_delta_and_touched(group_a, [], shared)
            sub_chunks: list[bytes] = []
            while chunk := os.read(sub_r, 1 << 20):
                sub_chunks.append(chunk)
            os.close(sub_r)
            os.waitpid(sub_pid, 0)
            sub_raw = b"".join(sub_chunks)
            if sub_raw and touched_a is not None:
                sub_ok, sub_val = pickle.loads(sub_raw)
                if sub_ok and sub_val is not None:
                    touched_b, new_deps_b, fmt_b, coords_b = sub_val
                    new_deps_a: set[str] = set()
                    for mod, _p, _t in group_a:
                        st = self._manager.graph.get(mod)
                        if st is not None:
                            new_deps_a.update(
                                set(st.dependencies) - pre_deps_by_mod[mod]
                            )
                    all_b_syms = frozenset().union(
                        *(changed_syms_by_mod[mod] for mod, _p, _t in group_b)
                    )
                    all_a_syms = frozenset().union(
                        *(changed_syms_by_mod[mod] for mod, _p, _t in group_a)
                    )

                    def _edits_cross_reference(
                        edited_group: list[tuple[str, str, str]],
                        other_touched: set[str],
                        other_syms: frozenset[str],
                    ) -> bool:
                        if not other_syms:
                            return False
                        for mod, _p, _t in edited_group:
                            added = added_refs_by_mod.get(mod, frozenset())
                            if not added or added.isdisjoint(other_syms):
                                continue
                            if not pre_deps_by_mod.get(mod, frozenset()).isdisjoint(
                                other_touched
                            ):
                                return True
                        return False

                    if (
                        touched_a.isdisjoint(touched_b)
                        and new_deps_a.isdisjoint(touched_b)
                        and new_deps_b.isdisjoint(touched_a)
                        and not _edits_cross_reference(group_a, touched_b, all_b_syms)
                        and not _edits_cross_reference(group_b, touched_a, all_a_syms)
                    ):
                        stage0_touched: set[str] | None = touched_a
                        split_b_result = (
                            touched_b,
                            new_deps_b,
                            all_b_syms,
                            fmt_b,
                            coords_b,
                        )
                    else:
                        # Sound fallback: apply group_b sequentially on top of group_a.
                        more = self._apply_delta_and_touched(group_b, [], shared)
                        stage0_touched = (
                            None if more is None else (touched_a | more)
                        )
                else:
                    more = self._apply_delta_and_touched(group_b, [], shared)
                    stage0_touched = None if more is None else (touched_a | more)
            else:
                stage0_touched = None
        else:
            stage0_touched = self._apply_delta_and_touched(changed, [], shared)

        hdr = bytearray()
        while len(hdr) < 4:
            chunk = os.read(req_r, 4 - len(hdr))
            if not chunk:
                break
            hdr += chunk
        if len(hdr) < 4:
            os.close(req_r)
            return (False, None)
        (req_len,) = struct.unpack("!I", hdr)
        req_buf = bytearray()
        while len(req_buf) < req_len:
            chunk = os.read(req_r, min(1 << 20, req_len - len(req_buf)))
            if not chunk:
                break
            req_buf += chunk
        os.close(req_r)
        if len(req_buf) < req_len:
            return (False, None)
        followup_changed, followup_removed = pickle.loads(bytes(req_buf))
        if stage0_touched is None:
            return (False, None)

        if split_b_result is not None:
            touched_b, new_deps_b, all_b_syms, fmt_b, coords_b = split_b_result
            if not followup_changed and not followup_removed:
                fmt_a, coords_a = self._extract_split_file_answers(stage0_touched)
                return (
                    True,
                    (
                        stage0_touched | touched_b,
                        {**fmt_b, **fmt_a},
                        {**coords_b, **coords_a},
                    ),
                )
            if (
                not followup_removed
                and {m for m, _p, _t in followup_changed}.isdisjoint(touched_b)
            ):
                followup_pre_deps = {
                    mod: (
                        frozenset(st.dependencies)
                        if (st := self._manager.graph.get(mod)) is not None
                        else frozenset()
                    )
                    for mod, _p, _t in followup_changed
                }
                followup_added = {
                    mod: self._changed_top_level_symbols(
                        self._text_by_module.get(mod, ""), new_text
                    )[1]
                    for mod, _p, new_text in followup_changed
                }
                followup_shared = dict(shared)
                for _mod, ch_path, ch_text in followup_changed:
                    followup_shared[ch_path] = ch_text
                stage1_touched = self._apply_delta_and_touched(
                    followup_changed, [], followup_shared
                )
                if stage1_touched is not None:
                    followup_new_deps: set[str] = set()
                    followup_cross = False
                    for mod, _p, _t in followup_changed:
                        st = self._manager.graph.get(mod)
                        pre = followup_pre_deps[mod]
                        if st is not None:
                            followup_new_deps.update(set(st.dependencies) - pre)
                        added = followup_added[mod]
                        if (
                            added
                            and not added.isdisjoint(all_b_syms)
                            and not pre.isdisjoint(touched_b)
                        ):
                            followup_cross = True
                    if (
                        stage1_touched.isdisjoint(touched_b)
                        and followup_new_deps.isdisjoint(touched_b)
                        and new_deps_b.isdisjoint(stage1_touched)
                        and not followup_cross
                    ):
                        all_a = stage0_touched | stage1_touched
                        fmt_a, coords_a = self._extract_split_file_answers(all_a)
                        return (
                            True,
                            (
                                all_a | touched_b,
                                {**fmt_b, **fmt_a},
                                {**coords_b, **coords_a},
                            ),
                        )
            # Follow-up interacted with group_b: finish sequentially from group_a + group_b + followup.
            followup_shared = dict(shared)
            for _mod, rem_path in followup_removed:
                followup_shared.pop(rem_path, None)
            for _mod, ch_path, ch_text in followup_changed:
                followup_shared[ch_path] = ch_text
            merged_by_mod = {mod: (mod, path, text) for mod, path, text in group_b}
            for mod, ch_path, ch_text in followup_changed:
                merged_by_mod[mod] = (mod, ch_path, ch_text)
            for mod, _rem_path in followup_removed:
                merged_by_mod.pop(mod, None)
            stage1_touched = self._apply_delta_and_touched(
                list(merged_by_mod.values()), followup_removed, followup_shared
            )
            if stage1_touched is None:
                return (False, None)
            all_touched = stage0_touched | stage1_touched
            formatted_by_file, typed_coords_by_file = (
                self._extract_split_file_answers(all_touched)
            )
            return (True, (all_touched, formatted_by_file, typed_coords_by_file))

        if followup_changed or followup_removed:
            followup_shared = dict(shared)
            for _mod, rem_path in followup_removed:
                followup_shared.pop(rem_path, None)
            for _mod, ch_path, ch_text in followup_changed:
                followup_shared[ch_path] = ch_text
            stage1_touched = self._apply_delta_and_touched(
                followup_changed, followup_removed, followup_shared
            )
        else:
            stage1_touched = set()
        if stage1_touched is None:
            return (False, None)
        all_touched = stage0_touched | stage1_touched
        formatted_by_file, typed_coords_by_file = self._extract_split_file_answers(
            all_touched
        )
        return (True, (all_touched, formatted_by_file, typed_coords_by_file))

    def start_speculative_sync(self, sources: SourceTexts) -> None:
        """Fork one child to typecheck ``sources`` delta against this live engine
        in the background and wait for any follow-up delta on a second pipe, so
        subsequent ``sync_corpus`` transitions ``stage0 -> final`` sequentially
        inside the already-warm child."""

        if (
            not sys.platform.startswith("linux")
            or self._deferred_split_delta
            or self._pending_presync is not None
        ):
            return
        unique, shared = TypeCheckerSession.unique_sources(sources, self._options)
        next_by_module = {
            source.module: (path, source.text or "") for path, source in unique
        }
        changed = [
            (module_name, path, text)
            for module_name, (path, text) in next_by_module.items()
            if self._path_by_module.get(module_name) != path
            or self._text_by_module.get(module_name) != text
        ]
        removed = [
            (module_name, path)
            for module_name, path in self._path_by_module.items()
            if module_name not in next_by_module
        ]
        if removed or len(changed) < 3 or len(changed) > max(
            32, len(self._path_by_module) // 2
        ):
            return
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        req_r, req_w = os.pipe()
        res_r, res_w = os.pipe()
        for pipe_fd in (req_w, res_w):
            try:
                fcntl.fcntl(pipe_fd, 1031, 1 << 20)  # F_SETPIPE_SZ
            except OSError:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(req_w)
            os.close(res_r)
            gc.disable()
            try:
                os.setpgrp()
            except OSError:
                pass
            try:
                try:
                    ok_val = self._run_speculative_child_body(
                        changed, shared, req_r, res_w
                    )
                    payload = pickle.dumps(ok_val)
                except BaseException:  # noqa: BLE001
                    payload = pickle.dumps((False, traceback.format_exc()))
                with os.fdopen(res_w, "wb") as out:
                    out.write(payload)
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(0)
        os.close(req_r)
        os.close(res_w)
        if gc_was_enabled:
            gc.enable()
        self._pending_presync = (
            os.getpid(),
            req_w,
            res_r,
            pid,
            next_by_module,
            None,
        )

    def send_speculative_followup(self, sources: SourceTexts) -> None:
        pending = self._pending_presync
        if pending is None:
            return
        owner_pid, req_w, res_r, pid, stage0_by_mod, sent_final_by_mod = pending
        if owner_pid != os.getpid() or req_w is None or sent_final_by_mod is not None:
            return
        unique, shared = TypeCheckerSession.unique_sources(sources, self._options)
        final_by_mod = {
            source.module: (path, source.text or "") for path, source in unique
        }
        followup_changed = [
            (module_name, path, text)
            for module_name, (path, text) in final_by_mod.items()
            if stage0_by_mod.get(module_name) != (path, text)
        ]
        followup_removed = [
            (module_name, path)
            for module_name, (path, _text) in stage0_by_mod.items()
            if module_name not in final_by_mod
        ]
        if followup_removed:
            return
        req_payload = pickle.dumps((followup_changed, followup_removed))
        framed = struct.pack("!I", len(req_payload)) + req_payload
        if len(framed) > (1 << 19):
            return
        try:
            with os.fdopen(req_w, "wb") as out:
                out.write(framed)
        except OSError:
            self.cancel_speculative_sync()
            return
        self._pending_presync = (
            owner_pid,
            None,
            res_r,
            pid,
            stage0_by_mod,
            final_by_mod,
        )

    @staticmethod
    def _terminate_presync_pid(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    def cancel_speculative_sync(self) -> None:
        pending = self._pending_presync
        if pending is None:
            return
        self._pending_presync = None
        owner_pid, req_w, res_r, pid, _stage0_by_mod, _sent_final = pending
        if owner_pid != os.getpid():
            return
        for fd in (req_w, res_r):
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        self._terminate_presync_pid(pid)

    def _try_consume_speculative_sync(
        self,
        changed: list[tuple[str, str, str]],
        removed: list[tuple[str, str]],
        next_by_module: dict[str, tuple[str, str]],
        shared: dict[str, str],
    ) -> TypeCheckerSession | None:
        pending = self._pending_presync
        if pending is None:
            return None
        self._pending_presync = None
        owner_pid, req_w, res_r, pid, stage0_by_mod, sent_final_by_mod = pending
        if owner_pid != os.getpid():
            return None
        if removed or (
            sent_final_by_mod is not None and sent_final_by_mod != next_by_module
        ):
            for fd in (req_w, res_r):
                if fd is None:
                    continue
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._terminate_presync_pid(pid)
            return None
        try:
            if sent_final_by_mod is None:
                assert req_w is not None
                followup_changed = [
                    (module_name, path, text)
                    for module_name, (path, text) in next_by_module.items()
                    if stage0_by_mod.get(module_name) != (path, text)
                ]
                followup_removed = [
                    (module_name, path)
                    for module_name, (path, _text) in stage0_by_mod.items()
                    if module_name not in next_by_module
                ]
                req_payload = pickle.dumps((followup_changed, followup_removed))
                framed = struct.pack("!I", len(req_payload)) + req_payload
                with os.fdopen(req_w, "wb") as out:
                    out.write(framed)
            chunks: list[bytes] = []
            while chunk := os.read(res_r, 1 << 20):
                chunks.append(chunk)
        except BaseException:
            try:
                os.close(res_r)
            except OSError:
                pass
            self._terminate_presync_pid(pid)
            raise
        os.close(res_r)
        ScanCache.defer_child_reap(pid)
        raw = b"".join(chunks)
        if raw:
            ok, value = pickle.loads(raw)
            if ok and value is not None:
                all_touched, child_formatted, child_typed_coords = value
                for path, formatted_map in child_formatted.items():
                    self._formatted_by_file[path] = formatted_map
                    self._raw_by_file.pop(path, None)
                    self._file_versions[path] = (
                        self._file_versions.get(path, 0) + 1
                    )
                for path, coords in child_typed_coords.items():
                    self._split_typed_coords_by_file[path] = coords
                for module_name, path, text in changed:
                    self._text_by_module[module_name] = text
                    self._path_by_module[module_name] = path
                self._deferred_split_delta = list(changed)
                self._split_touched_modules = set(all_touched)
                self._shared = shared
                return self
        return None

    def sync_corpus(self, sources: SourceTexts) -> TypeCheckerSession | None:
        """Update this live engine to match a full corpus ``sources``.

        Returns ``None`` if too many modules changed (different corpus) or mypy
        encountered a blocking error so the caller falls back to a fresh build.
        """

        unique, shared = TypeCheckerSession.unique_sources(sources, self._options)
        next_by_module = {
            source.module: (path, source.text or "") for path, source in unique
        }
        changed = [
            (module_name, path, text)
            for module_name, (path, text) in next_by_module.items()
            if self._path_by_module.get(module_name) != path
            or self._text_by_module.get(module_name) != text
        ]
        removed = [
            (module_name, path)
            for module_name, path in self._path_by_module.items()
            if module_name not in next_by_module
        ]
        if not changed and not removed:
            self._shared = shared
            return self
        if self._pending_presync is not None and not self._deferred_split_delta:
            consumed = self._try_consume_speculative_sync(
                changed, removed, next_by_module, shared
            )
            if consumed is not None:
                return consumed
        if self._deferred_split_delta:
            if (
                not removed
                and len(changed) == 1
                and changed[0][0] not in self._split_touched_modules
            ):
                touched = self._apply_delta_and_touched(changed, [], shared)
                if touched is None:
                    return None
                if touched.isdisjoint(self._split_touched_modules):
                    return self
                if not self._flush_deferred_split_delta():
                    return None
                if not self._apply_delta(changed, [], shared):
                    return None
                return self
            if not self._flush_deferred_split_delta():
                return None
        if len(changed) + len(removed) > max(32, len(self._path_by_module) // 2):
            return None
        if not self._apply_delta(changed, removed, shared):
            return None
        return self

    def sync_overlay(self, sources: SourceTexts) -> TypeCheckerSession | None:
        """Update only files present in a subset ``sources`` without removing
        untouched modules from the live corpus."""

        unique, shared = TypeCheckerSession.unique_sources(sources, self._options)
        changed: list[tuple[str, str, str]] = []
        for path, source in unique:
            if path in self._shared:
                continue
            text = source.text or ""
            if (
                self._path_by_module.get(source.module) != path
                or self._text_by_module.get(source.module) != text
            ):
                changed.append((source.module, path, text))
        merged_shared = {**self._shared, **shared} if shared else self._shared
        if not changed:
            self._shared = merged_shared
            return self
        if self._deferred_split_delta and not self._flush_deferred_split_delta():
            return None
        if not self._apply_delta(changed, [], merged_shared):
            return None
        return self

    def _file_raw_types(self, file_path: str) -> dict[ReceiverSpan, Type]:
        cached = self._raw_by_file.get(file_path)
        if cached is not None:
            return cached
        state = self._state_by_path.get(file_path)
        if state is None or state.tree is None:
            empty: dict[ReceiverSpan, Type] = {}
            self._raw_by_file[file_path] = empty
            return empty
        all_types = self._manager.manager.all_types
        raw: dict[ReceiverSpan, Type] = {}
        for expression in self._subexpressions(state.tree):
            found = all_types.get(expression)
            if (
                found is not None
                and expression.end_line is not None
                and expression.end_column is not None
            ):
                raw[
                    ReceiverSpan(
                        state.xpath,
                        expression.line,
                        expression.column,
                        expression.end_line,
                        expression.end_column,
                    )
                ] = found
        self._raw_by_file[file_path] = raw
        return raw

    def answer(self, span: ReceiverSpan) -> ReceiverType:
        formatted = self._formatted_by_file.get(span.file_path)
        if formatted is not None:
            hit = formatted.get(span)
            if hit is not None:
                return hit
        split_coords = self._split_typed_coords_by_file.get(span.file_path)
        if split_coords is not None:
            coords = (span.line, span.column, span.end_line, span.end_column)
            if coords not in split_coords:
                if span.file_path in self._shared:
                    return FailedReceiverType(self._shared[span.file_path])
                return FailedReceiverType(
                    f"the build exported no type at {span.location}"
                )
            self._flush_deferred_split_delta()
        raw = self._file_raw_types(span.file_path)
        found = raw.get(span)
        if found is not None:
            inferred = ExpressionExtractionScope._inferred_type(
                found,
                self._options,
                self._format_type_bare,
                self._mypy_types,
            )
            self._formatted_by_file.setdefault(span.file_path, {})[span] = inferred
            return inferred
        if span.file_path in self._shared:
            return FailedReceiverType(self._shared[span.file_path])
        return FailedReceiverType(f"the build exported no type at {span.location}")

    def receiver_types(
        self, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        started = perf_counter()
        answers = {span: self.answer(span) for span in spans}
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(lookup_seconds=perf_counter() - started, lookups=len(answers))
        )
        return answers

    def stop(self) -> None:
        """The live engine persists until the enclosing ``ScanCache.scope()`` exits."""

    @property
    def status(self) -> str:
        return (
            f"mypy fine-grained ({len(self._path_by_module)} module(s); "
            f"{len(self._shared)} file(s) left out for a module name another file shares)"
        )


__all__ = (
    "BuiltTypeChecker",
    "FineGrainedTypeChecker",
    "SourceFiles",
    "SourceTexts",
    "TypeCheckerSession",
    "TypeCheckerSources",
    "TypeCheckerTiming",
    "TypeCheckerTimingLedger",
    "UnavailableTypeChecker",
)


@ScanCache.cached
def _type_errors_by_state() -> dict[tuple[tuple[str, str, str], ...], TypeErrors]:
    """The scan's whole-tree type errors, by the source state they were built from."""

    return {}


@ScanCache.cached
def _incremental_type_errors_slot() -> list[FineGrainedTypeErrors | None]:
    """The active fine-grained mypy error engine for this ``ScanCache.scope()``."""

    return [None]


class FineGrainedTypeErrors:
    """In-memory fine-grained mypy error checker for repeated candidate proofs.

    Mypy's ``FineGrainedBuildManager.update()`` normally re-triggers every
    target that previously reported an error (hundreds of untouched files on a
    real repo). Retaining errors per ``(file_path, target)`` lets each candidate
    update recheck only the changed modules and their triggered dependents and
    merge only the reprocessed targets in memory while keeping whole-tree
    ``TypeErrors`` exact."""

    def __init__(
        self,
        *,
        manager: FineGrainedBuildManager,
        fscache: FileSystemCache,
        options: Options,
        module_prefix: Callable[[dict[str, State], str], str | None],
        hash_digest: Callable[[bytes], str],
        text_by_module: dict[str, str],
        path_by_module: dict[str, str],
        errors_by_file_target: dict[str, dict[str, tuple[str, ...]]],
        mtime: int,
        source_checker: FineGrainedTypeChecker | None = None,
    ) -> None:
        self._manager = manager
        self._fscache = fscache
        self._options = options
        self._module_prefix = module_prefix
        self._hash_digest = hash_digest
        self._text_by_module = text_by_module
        self._path_by_module = path_by_module
        self._errors_by_file_target = errors_by_file_target
        self._errors_by_file: dict[str, tuple[str, ...]] = {
            path: self._flatten_targets(targets)
            for path, targets in errors_by_file_target.items()
            if targets
        }
        self._mtime = mtime
        self._source_checker = source_checker
        self._failed: bool = False

    @staticmethod
    def _flatten_targets(targets: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
        return tuple(
            message for messages in targets.values() for message in messages
        )

    @staticmethod
    def _extract_errors_by_file_target(
        errors_obj: Errors,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        extracted: dict[str, dict[str, tuple[str, ...]]] = {}
        for path, infos in errors_obj.error_info_map.items():
            visible = [info for info in infos if not info.hidden]
            if not visible:
                continue
            deduped = errors_obj.remove_duplicates(errors_obj.sort_messages(visible))
            by_target_infos: defaultdict[str, list[ErrorInfo]] = defaultdict(list)
            for info in deduped:
                by_target_infos[info.target or ""].append(info)
            target_messages: dict[str, tuple[str, ...]] = {}
            for target, target_infos in by_target_infos.items():
                lines = errors_obj.format_messages(
                    path, errors_obj.render_messages(path, target_infos)
                )
                parsed = TypeErrors._parse_error_lines(lines)
                flat = tuple(
                    message for msgs in parsed.values() for message in msgs
                )
                if flat:
                    target_messages[target] = flat
            if target_messages:
                extracted[path] = target_messages
        return extracted

    def _set_file_targets(
        self, path: str, targets: dict[str, tuple[str, ...]] | None
    ) -> None:
        if targets:
            self._errors_by_file_target[path] = targets
            self._errors_by_file[path] = self._flatten_targets(targets)
        else:
            self._errors_by_file_target.pop(path, None)
            self._errors_by_file.pop(path, None)

    def _delete_memory_file(self, module_name: str, path: str) -> None:
        FineGrainedTypeChecker._remove_memory_file(
            self._fscache, self._manager, module_name, path
        )

    def _targets_by_module(self, targets: Iterable[str]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        graph = self._manager.graph
        for target in targets:
            prefix = self._module_prefix(graph, target)
            if prefix is not None:
                grouped[prefix].add(target)
        return grouped

    @classmethod
    def seeded(
        cls, sources: SourceTexts
    ) -> tuple[FineGrainedTypeErrors, TypeErrors] | None:
        if importlib.util.find_spec("mypy") is None:
            raise RuntimeError(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        errors = importlib.import_module("mypy.errors")
        fscache_mod = importlib.import_module("mypy.fscache")
        modulefinder = importlib.import_module("mypy.modulefinder")
        update_mod = importlib.import_module("mypy.server.update")
        util_mod = importlib.import_module("mypy.util")

        options = TypeCheckerSession.options()
        options.incremental = True
        options.fine_grained_incremental = True
        options.use_fine_grained_cache = False
        options.local_partial_types = True
        options.cache_dir = os.devnull
        options.export_types = False
        options.preserve_asts = True
        options.ignore_errors = False
        options.native_parser = False

        unique, _shared = TypeCheckerSession.unique_sources(sources, options)
        fscache: FileSystemCache = fscache_mod.FileSystemCache()
        hash_digest: Callable[[bytes], str] = util_mod.hash_digest
        build_source = modulefinder.BuildSource
        text_by_module: dict[str, str] = {}
        path_by_module: dict[str, str] = {}
        build_sources: list[BuildSource] = []
        mtime = 1
        for path, source in unique:
            text = source.text or ""
            mtime += 1
            FineGrainedTypeChecker._put_memory_file(
                fscache,
                hash_digest,
                path,
                text,
                mtime,
                register_in_dir=not os.path.exists(path),
            )
            text_by_module[source.module] = text
            path_by_module[source.module] = path
            build_sources.append(build_source(path, source.module, None))

        try:
            with _quiet_mypy():
                result = build.build(build_sources, options, fscache=fscache)
                manager: FineGrainedBuildManager = update_mod.FineGrainedBuildManager(
                    result
                )
        except (errors.CompileError, Exception, SystemExit):  # noqa: BLE001
            return None

        errors_by_file_target = cls._extract_errors_by_file_target(
            manager.manager.errors
        )
        module_prefix = update_mod.module_prefix
        manager.previous_targets_with_errors = set()
        engine = cls(
            manager=manager,
            fscache=fscache,
            options=options,
            module_prefix=module_prefix,
            hash_digest=hash_digest,
            text_by_module=text_by_module,
            path_by_module=path_by_module,
            errors_by_file_target=errors_by_file_target,
            mtime=mtime,
        )
        return engine, TypeErrors(dict(engine._errors_by_file))

    def errors_for(self, sources: SourceTexts) -> TypeErrors | None:
        if self._source_checker is not None:
            if self._source_checker._failed or not self._source_checker._flush_deferred_split_delta():
                self._failed = True
                return None
            self._mtime = self._source_checker._mtime
        unique, _shared = TypeCheckerSession.unique_sources(sources, self._options)
        next_by_module = {
            source.module: (path, source.text or "") for path, source in unique
        }
        changed = [
            (module_name, path, text)
            for module_name, (path, text) in next_by_module.items()
            if self._path_by_module.get(module_name) != path
            or self._text_by_module.get(module_name) != text
        ]
        removed = [
            (module_name, path)
            for module_name, path in self._path_by_module.items()
            if module_name not in next_by_module
        ]
        if not changed and not removed:
            return TypeErrors(dict(self._errors_by_file))
        if len(changed) + len(removed) > max(32, len(self._path_by_module) // 2):
            return None
        if sys.platform.startswith("linux") and len(self._path_by_module) >= 16:
            return self._errors_for_forked(changed, removed)
        return self._errors_for_in_place(changed, removed)

    def _errors_for_forked(
        self,
        changed: list[tuple[str, str, str]],
        removed: list[tuple[str, str]],
    ) -> TypeErrors | None:
        """Evaluate one speculative delta in a COW child so the parent engine
        stays at its clean baseline (no revert work and no ICE corruption)."""

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        reader, writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            gc.disable()
            try:
                try:
                    self._options.export_types = False
                    self._manager.manager.options.export_types = False
                    result = self._errors_for_in_place(changed, removed)
                    payload = pickle.dumps((True, result))
                except BaseException:  # noqa: BLE001
                    payload = pickle.dumps((False, None))
                with os.fdopen(writer, "wb") as out:
                    out.write(payload)
            finally:
                os._exit(0)
        os.close(writer)
        chunks: list[bytes] = []
        try:
            while chunk := os.read(reader, 1 << 20):
                chunks.append(chunk)
        finally:
            os.close(reader)
            ScanCache.defer_child_reap(pid)
            if gc_was_enabled:
                gc.enable()
        if not chunks:
            return None
        ok, result = pickle.loads(b"".join(chunks))
        return result if ok else None

    def _errors_for_in_place(
        self,
        changed: list[tuple[str, str, str]],
        removed: list[tuple[str, str]],
    ) -> TypeErrors | None:
        for module_name, path in removed:
            self._delete_memory_file(module_name, path)
            self._text_by_module.pop(module_name, None)
            self._path_by_module.pop(module_name, None)
            self._set_file_targets(path, None)
        for module_name, path, text in changed:
            old_path = self._path_by_module.get(module_name)
            if old_path is not None and old_path != path:
                self._delete_memory_file(module_name, old_path)
                self._set_file_targets(old_path, None)
            self._mtime += 1
            is_new = module_name not in self._path_by_module or old_path != path
            FineGrainedTypeChecker._put_memory_file(
                self._fscache,
                self._hash_digest,
                path,
                text,
                self._mtime,
                register_in_dir=is_new or not os.path.exists(path),
            )
            self._manager.manager.ast_cache.pop(module_name, None)
            self._text_by_module[module_name] = text
            self._path_by_module[module_name] = path

        self._manager.previous_targets_with_errors = set()
        try:
            with _quiet_mypy():
                self._manager.update(
                    [(module_name, path) for module_name, path, _text in changed],
                    removed,
                )
        except (Exception, SystemExit):  # noqa: BLE001 - mypy fine-grained ICE
            self._failed = True
            return None
        if self._manager.blocking_error is not None or self._manager.stale:
            self._failed = True
            return None

        self._record_updated_errors(changed)
        return TypeErrors(dict(self._errors_by_file))

    def _record_updated_errors(
        self,
        changed: list[tuple[str, str, str]],
    ) -> None:
        """Merge reprocessed target errors after ``self._manager.update``."""

        new_by_file_target = self._extract_errors_by_file_target(
            self._manager.manager.errors
        )
        proc_targets_by_mod = self._targets_by_module(self._manager.processed_targets)
        self._manager.previous_targets_with_errors = set()

        whole_rechecked = {module_name for module_name, _path, _text in changed} | set(
            self._manager.updated_modules
        )
        partial_rechecked = set(proc_targets_by_mod) - whole_rechecked

        handled_paths: set[str] = set()
        for module_name in whole_rechecked:
            path = self._path_by_module.get(module_name)
            if path is None:
                continue
            handled_paths.add(path)
            self._set_file_targets(path, new_by_file_target.get(path))

        for module_name in partial_rechecked:
            path = self._path_by_module.get(module_name)
            if path is None:
                continue
            handled_paths.add(path)
            existing_targets = dict(self._errors_by_file_target.get(path, {}))
            new_targets = new_by_file_target.get(path, {})
            for target in proc_targets_by_mod.get(module_name, ()):
                if target in new_targets:
                    existing_targets[target] = new_targets[target]
                else:
                    existing_targets.pop(target, None)
            for target, msgs in new_targets.items():
                existing_targets[target] = msgs
            self._set_file_targets(path, existing_targets)

        for path, new_targets in new_by_file_target.items():
            if path in handled_paths:
                continue
            existing_targets = dict(self._errors_by_file_target.get(path, {}))
            existing_targets.update(new_targets)
            self._set_file_targets(path, existing_targets)


@dataclass(frozen=True)
class TypeErrors:
    """The errors one whole-tree build reports, by file and message: a change is
    proved type-safe when the build of its result reports none the build of its
    source does not. Lines are left out, since an edit moves the lines below it."""

    by_file: Mapping[str, tuple[str, ...]]

    @staticmethod
    def _parse_error_lines(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
        found: defaultdict[str, list[str]] = defaultdict(list)
        for line in lines:
            path, sep, rest = line.partition(":")
            _line, _sep, rest = rest.partition(":")
            kind, _sep, message = rest.strip().partition(": ")
            if sep and kind == "error":
                found[path].append(f"{_line}: {message}")
        return {path: tuple(messages) for path, messages in found.items()}

    @classmethod
    def of(cls, sources: SourceTexts) -> TypeErrors:
        """The errors of these sources' build, built once per source state in a scan.

        Inside a ``ScanCache.scope()``, an in-memory ``FineGrainedTypeErrors``
        engine is seeded on first use and updated incrementally for subsequent
        candidate states instead of running a cold whole-tree mypy build per
        candidate.
        """

        built = _type_errors_by_state()
        state = sources.state
        if state in built:
            return built[state]
        if ScanCache.active() is not None:
            slot = _incremental_type_errors_slot()
            existing = slot[0]
            if existing is not None:
                incremental = existing.errors_for(sources)
                if incremental is not None:
                    built[state] = incremental
                    return incremental
                if existing._failed:
                    slot[0] = None
            if slot[0] is None:
                checker = TypeCheckerSession._active_incremental()
                shared_engine = (
                    checker.shared_type_errors() if checker is not None else None
                )
                if shared_engine is not None:
                    slot[0] = shared_engine
                    incremental = shared_engine.errors_for(sources)
                    if incremental is not None:
                        built[state] = incremental
                        return incremental
                else:
                    seeded = FineGrainedTypeErrors.seeded(sources)
                    if seeded is not None:
                        engine, initial = seeded
                        slot[0] = engine
                        built[state] = initial
                        return initial
        built[state] = cls._built(sources)
        return built[state]

    @classmethod
    def _built(cls, sources: TypeCheckerSources) -> TypeErrors:
        if importlib.util.find_spec("mypy") is None:
            raise RuntimeError(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        options = TypeCheckerSession.options()
        unique, _shared = TypeCheckerSession.unique_sources(sources, options)
        with _quiet_mypy():
            result = build.build([source for _path, source in unique], options)
        return cls(cls._parse_error_lines(result.errors))

    def new_since(self, before: TypeErrors) -> tuple[str, ...]:
        """Each error this build reports that `before` does not, as `path:line: message`."""

        new: list[str] = []
        for path, messages in self.by_file.items():
            earlier = Counter(message.partition(": ")[2] for message in before.by_file.get(path, ()))
            for message in messages:
                text = message.partition(": ")[2]
                if earlier[text]:
                    earlier[text] -= 1
                else:
                    new.append(f"{path}:{message}")
        return tuple(new)
