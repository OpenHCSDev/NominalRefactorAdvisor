"""One mypy build per scan, whose exported types answer every receiver it is asked about.

mypy is an optional extra (`nominal-refactor-advisor[types]`). Without it, or
when the build fails, every receiver gets a `FailedReceiverType` naming why,
so the report says the owner is unresolved and why; nothing is skipped
silently. The build runs in process once; each receiver is then a lookup in
its type map, keyed by the receiver's exact source span.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import chain
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import pickle
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar

from .cache_checkout import CheckoutRelocation, checkout_identity_text
from .cache_paths import ActiveScanCache, replace_atomically
from .json_reports import DataclassJsonReport
from .probe_owner import FailedReceiverType, InferredReceiverType, ReceiverSpan, ReceiverType

if TYPE_CHECKING:
    from mypy.build import BuildResult
    from mypy.modulefinder import BuildSource
    from mypy.options import Options

    from .ast_tools import ParsedModule
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

    def receiver_types(
        self, spans: tuple[ReceiverSpan, ...]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        """Build these sources once and answer every span."""

        with TypeCheckerSession.session(self) as session:
            return session.receiver_types(spans)

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

    ANSWER_SCHEMA = "type-checker-answers-v1"
    ANSWER_IMPLEMENTATION_MODULES = (
        "nominal_refactor_advisor.type_checker_session",
        "nominal_refactor_advisor.probe_owner",
    )

    def build_sources(self, options: Options) -> list[BuildSource]:
        return importlib.import_module("mypy.find_sources").create_source_list(
            sorted(self.paths), options
        )

    def receiver_types(
        self, spans: tuple[ReceiverSpan, ...]
    ) -> Mapping[ReceiverSpan, ReceiverType]:
        """Files on disk answer as their content does, so the answers are kept
        in the scan's cache under what they depend on: each file's content
        and place in its checkout, the spans, mypy's and Python's versions and
        the type checker's own source. A scan after an NRA change that leaves
        those alone reads them instead of building again (14-19 s cold)."""

        directory = ActiveScanCache.directory()
        if directory is None or importlib.util.find_spec("mypy") is None:
            return super().receiver_types(spans)
        relocation = CheckoutRelocation.containing(min(self.paths, default="."))
        key = self.answer_key(spans)
        portable_path = directory / "type-checker-answers" / f"{key}.pickle"
        bound_path = portable_path.with_name(
            f"{key}-{_checkout_digest(relocation)}.pickle"
        )
        for path in (portable_path, bound_path):
            stored = _load_answers(path, relocation, key)
            if stored is not None:
                return stored
        answers = super().receiver_types(spans)
        payload = (key, dict(answers))
        if relocation is None:
            data, portable = pickle.dumps(payload, pickle.HIGHEST_PROTOCOL), False
        else:
            data, portable = relocation.portable_dumps(payload)
        try:
            portable_path.parent.mkdir(parents=True, exist_ok=True)
            replace_atomically(portable_path if portable else bound_path, data)
        except OSError:
            pass
        return answers

    def answer_key(self, spans: tuple[ReceiverSpan, ...]) -> str:
        files = sorted(
            (checkout_identity_text(path), _content_hash(Path(path)))
            for path in self.paths
        )
        span_keys = sorted(
            (
                checkout_identity_text(span.file_path),
                span.line,
                span.column,
                span.end_line,
                span.end_column,
            )
            for span in spans
        )
        implementation = tuple(
            _content_hash(Path(importlib.import_module(name).__file__ or name))
            for name in self.ANSWER_IMPLEMENTATION_MODULES
        )
        return hashlib.blake2s(
            repr(
                (
                    self.ANSWER_SCHEMA,
                    importlib.import_module("mypy.version").__version__,
                    sys.version_info[:2],
                    files,
                    span_keys,
                    implementation,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()


def _content_hash(path: Path) -> str:
    try:
        return hashlib.blake2s(path.read_bytes(), digest_size=16).hexdigest()
    except OSError:
        return "unreadable"


def _checkout_digest(relocation: CheckoutRelocation | None) -> str:
    root = "unrooted" if relocation is None else relocation.root_texts[0]
    return hashlib.blake2s(root.encode("utf-8"), digest_size=8).hexdigest()


def _load_answers(
    path: Path, relocation: CheckoutRelocation | None, key: str
) -> Mapping[ReceiverSpan, ReceiverType] | None:
    try:
        data = path.read_bytes()
        payload = (
            pickle.loads(data)
            if relocation is None
            else relocation.load(io.BytesIO(data))
        )
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError):
        return None
    if not isinstance(payload, tuple) or len(payload) != 2 or payload[0] != key:
        return None
    return payload[1]


@dataclass(frozen=True)
class SourceTexts(TypeCheckerSources):
    """Parsed modules' own text under their own names: a search state that exists
    only in memory types the same as one written to disk."""

    modules: tuple[ParsedModule, ...]

    def build_sources(self, options: Options) -> list[BuildSource]:
        build_source = importlib.import_module("mypy.modulefinder").BuildSource
        return [
            build_source(module.file_path, module.module_name, text=module.source)
            for module in self.modules
        ]


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
        """Build these sources once with exported types, or name why that failed."""

        if importlib.util.find_spec("mypy") is None:
            return UnavailableTypeChecker(f"mypy is not installed ({TYPES_EXTRA})")
        build = importlib.import_module("mypy.build")
        errors = importlib.import_module("mypy.errors")
        options = importlib.import_module("mypy.options").Options()
        options.export_types = True
        options.preserve_asts = True
        options.incremental = False
        options.strict_equality = True
        # Unannotated functions are most of the receivers a scan asks about.
        options.check_untyped_defs = True
        # Every scanned file is a source; imports outside the scan stay unanalyzed.
        options.follow_imports = "skip"
        options.ignore_missing_imports = True
        options.python_version = sys.version_info[:2]
        sources_by_module: defaultdict[str, list[tuple[str, BuildSource]]] = defaultdict(list)
        for path, source in sources.pathed(options):
            sources_by_module[source.module].append((path, source))
        # mypy refuses the whole build when two files claim one module name.
        shared = {
            path: f"module `{module}` is shared by {len(found)} scanned files"
            for module, found in sources_by_module.items()
            if len(found) > 1
            for path, _source in found
        }
        unique = [only for only, *rest in sources_by_module.values() if not rest]
        built = [source for _path, source in unique]
        started = perf_counter()
        try:
            result = build.build(built, options)
        except errors.CompileError as error:
            return UnavailableTypeChecker(
                "the mypy build failed: " + "\n".join(error.messages)[-500:]
            )
        session = BuiltTypeChecker.of(
            result, options, frozenset(path for path, _source in unique), shared
        )
        TypeCheckerTimingLedger.active().record(
            TypeCheckerTiming(build_seconds=perf_counter() - started)
        )
        return session

    @staticmethod
    @contextmanager
    def session(sources: TypeCheckerSources) -> Iterator[TypeCheckerSession]:
        session = TypeCheckerSession.started(sources)
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
    ) -> BuiltTypeChecker:
        """Every typed expression of the scanned files, with the classes it is."""

        format_type_bare = importlib.import_module("mypy.messages").format_type_bare
        subexpressions = importlib.import_module("mypy.server.subexpr").get_subexpressions
        inferred: dict[ReceiverSpan, InferredReceiverType] = {}
        for state in result.graph.values():
            if state.xpath not in paths or state.tree is None:
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
                    inferred[span] = InferredReceiverType(
                        format_type_bare(found, options, verbosity=2),
                        _classes_of(found),
                    )
        return cls(inferred, shared)

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
