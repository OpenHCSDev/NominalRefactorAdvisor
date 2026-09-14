"""Applied batch: give declaration caches scan ownership, independently of ASTs."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CreateFileOperation,
    EnsureImportOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
    ReplaceDeclarationDecoratorsOperation,
)

SCAN_CACHE_SOURCE = '''"""Bounded memoization for declarations stable during one analysis invocation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import ClassVar, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ScanCache:
    """Own declaration caches until the outer invocation (or worker) ends.

    Per-module AST cleanup cannot invalidate this storage. Outside an explicit
    scope, decorated functions recompute, so interactive callers cannot retain
    identities from a previous implementation. Nested analysis shares its
    invocation; each process-pool worker starts an independent scope.
    """

    _functions: dict[Callable[..., object], Callable[..., object]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active: ClassVar[ContextVar[ScanCache | None]] = ContextVar(
        "nra_scan_cache", default=None
    )

    def memoized(self, function: Callable[P, T]) -> Callable[P, T]:
        if function not in self._functions:
            self._functions[function] = lru_cache(maxsize=None)(function)
        # Each key retains only a memoized wrapper of that same callable.
        return cast(Callable[P, T], self._functions[function])

    @classmethod
    def cached(cls, function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def invoke(*args: P.args, **kwargs: P.kwargs) -> T:
            active = cls._active.get()
            implementation = function if active is None else active.memoized(function)
            return implementation(*args, **kwargs)

        return invoke

    @classmethod
    @contextmanager
    def scope(cls) -> Iterator[None]:
        if cls._active.get() is not None:
            yield
            return
        token = cls._active.set(cls())
        try:
            yield
        finally:
            cls._active.reset(token)

    @classmethod
    def initialize_worker(cls) -> None:
        """Begin process-local storage whose lifetime is the analysis pool."""
        cls._active.set(cls())
'''


def patch(path, qualname, old, new):
    return PatchTargetOperation(
        target=SourceRewriteTarget(file_path=path, qualname=qualname),
        replacements=(SourceTextReplacement(old_source=old, new_source=new),),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        CreateFileOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/scan_cache.py"
            ),
            source=SCAN_CACHE_SOURCE,
        ),
        *(
            EnsureImportOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/" + module + ".py"
                ),
                import_source="from .scan_cache import ScanCache",
            )
            for module in ("implementation_identity", "ast_tools", "analysis", "cli")
        ),
        *(
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/" + module + ".py",
                    qualname=name,
                ),
                decorators_source=("@classmethod\n" if module == "ast_tools" else "")
                + "@ScanCache.cached",
            )
            for module, name in (
                ("implementation_identity", "_declaration_implementation_module_names"),
                ("implementation_identity", "_source_signature"),
                ("ast_tools", "CollectedFamily.item_schema_signature"),
                ("ast_tools", "CollectedFamily.implementation_identity"),
            )
        ),
        *(
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/" + module + ".py",
                    qualname=name,
                ),
                decorators_source="@ScanCache.scope()",
            )
            for module, name in (
                ("analysis", "build_compact_projection_shard"),
                ("analysis", "analyze_compact_roots_with_cache"),
                ("cli", "main"),
            )
        ),
        patch(
            "nominal_refactor_advisor/analysis.py",
            "analyze_compact_roots_with_cache",
            "mp_context=_analysis_process_pool_mp_context(),",
            "mp_context=_analysis_process_pool_mp_context(),\n            initializer=ScanCache.initialize_worker,",
        ),
    )
)
