"""One mypy build per scan, whose exported types answer every receiver it is asked about.

mypy is an optional extra (`nominal-refactor-advisor[types]`). Without it, or
when the build fails, every receiver gets a `FailedReceiverType` naming why,
so the report says the owner is unresolved and why; nothing is skipped
silently. The build runs in process once; each receiver is then a lookup in
its type map, keyed by the receiver's exact source span.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter
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
    from mypy.build import BuildResult
    from mypy.modulefinder import BuildSource
    from mypy.options import Options

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
    ) -> dict[ReceiverSpan, InferredReceiverType]:
        raise NotImplementedError

    @staticmethod
    def _inferred_type(
        found: Type,
        options: Options,
        format_type_bare: object,
    ) -> InferredReceiverType:
        return InferredReceiverType(
            format_type_bare(found, options, verbosity=2),  # type: ignore[operator]
            _classes_of(found),
            _instances_only(found),
            _families_of(found),
        )


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
    ) -> dict[ReceiverSpan, InferredReceiverType]:
        format_type_bare = importlib.import_module("mypy.messages").format_type_bare
        subexpressions = importlib.import_module("mypy.server.subexpr").get_subexpressions
        inferred: dict[ReceiverSpan, InferredReceiverType] = {}
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
                    inferred[span] = self._inferred_type(found, options, format_type_bare)
        return inferred


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
    ) -> dict[ReceiverSpan, InferredReceiverType]:
        format_type_bare = importlib.import_module("mypy.messages").format_type_bare
        subexpressions = importlib.import_module("mypy.server.subexpr").get_subexpressions
        inferred: dict[ReceiverSpan, InferredReceiverType] = {}
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
                    inferred[span] = self._inferred_type(found, options, format_type_bare)
        return inferred


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

    @staticmethod
    def started(sources: TypeCheckerSources) -> TypeCheckerSession:
        """Build these sources once with exported types, or name why that failed:
        a session for receivers asked about later, so it keeps every file's types."""

        return TypeCheckerSession._built(sources, AllBuiltExpressions())

    @staticmethod
    def answers(
        sources: TypeCheckerSources, spans: Iterable[ReceiverSpan]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        """Build these sources once and answer `spans`. Only the files that hold one
        check untyped defs and only the requested spans are formatted and exported."""

        asked = tuple(spans)
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
            result = build.build(build_sources, options)
        except errors.CompileError as error:
            return UnavailableTypeChecker(
                "the mypy build failed: " + "\n".join(error.messages)[-500:]
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


def _classes_of(found: Type) -> tuple[str, ...]:
    """The classes a mypy type is: the decoder at the type checker's boundary."""

    types = importlib.import_module("mypy.types")
    proper = types.get_proper_type(found)
    match proper:
        case types.Instance():
            return (proper.type.fullname,)
        case types.TypeType():
            return _classes_of(proper.item)
        case types.TypeVarType():
            return _classes_of(proper.upper_bound)
        case types.UnionType():
            return tuple(
                chain.from_iterable(
                    _classes_of(item)
                    for item in proper.items
                    if not isinstance(types.get_proper_type(item), types.NoneType)
                )
            )
    return ()


def _instances_only(found: Type) -> bool:
    """Whether every value of a mypy type is an instance that is not a class. A
    class is itself an instance of `type`, `object`, its metaclass and any
    protocol it satisfies, so none of those, nor an unknown type, proves it."""

    types = importlib.import_module("mypy.types")
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
            return _instances_only(proper.upper_bound)
        case types.UnionType():
            return all(_instances_only(item) for item in proper.items)
    return False


def _families_of(found: Type) -> tuple[str, ...]:
    """The classes `C` when every value of a mypy type is a class `C` or a subclass:
    `type[C]` for a class that is no protocol, or a union of them; otherwise none."""

    types = importlib.import_module("mypy.types")
    proper = types.get_proper_type(found)
    match proper:
        case types.TypeType():
            item = types.get_proper_type(proper.item)
            if isinstance(item, types.Instance) and not item.type.is_protocol:
                return (item.type.fullname,)
        case types.UnionType():
            families = tuple(_families_of(item) for item in proper.items)
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


__all__ = (
    "BuiltTypeChecker",
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


@dataclass(frozen=True)
class TypeErrors:
    """The errors one whole-tree build reports, by file and message: a change is
    proved type-safe when the build of its result reports none the build of its
    source does not. Lines are left out, since an edit moves the lines below it."""

    by_file: Mapping[str, tuple[str, ...]]

    @classmethod
    def of(cls, sources: SourceTexts) -> TypeErrors:
        """The errors of these sources' build, built once per source state in a scan.

        A goal search proves one recipe against one state several times (its evaluation,
        its simulation, its frontier), and each proof rebuilt the state it projected:
        one concept's plan on Toad ran ten whole-tree builds, about 8 s each.
        """

        built = _type_errors_by_state()
        state = sources.state
        if state not in built:
            built[state] = cls._built(sources)
        return built[state]

    @classmethod
    def _built(cls, sources: TypeCheckerSources) -> TypeErrors:
        if importlib.util.find_spec("mypy") is None:
            raise RuntimeError(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        options = TypeCheckerSession.options()
        unique, _shared = TypeCheckerSession.unique_sources(sources, options)
        result = build.build([source for _path, source in unique], options)
        found: defaultdict[str, list[str]] = defaultdict(list)
        for line in result.errors:
            path, sep, rest = line.partition(":")
            _line, _sep, rest = rest.partition(":")
            kind, _sep, message = rest.strip().partition(": ")
            if sep and kind == "error":
                found[path].append(f"{_line}: {message}")
        return cls({path: tuple(messages) for path, messages in found.items()})

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
