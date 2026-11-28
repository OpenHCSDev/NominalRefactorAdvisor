"""Detectors for silent input/exception swallowers, embedded foreign code literals,
and cross-module cloned `DeclaredFamily` roots.

Every detector here flags a concrete, high-precision structural defect that lets
missing inputs, programming bugs, inline non-Python applications, or parallel
registries pass static checks silently:

- `EmbeddedCodeLiteralDetector` (`BOUND-5`): non-docstring multiline string literals
  embedding substantial HTML/CSS/JS, JavaScript, shell, or SQL DDL source code.
- `SilentMissingPathFallbackDetector`: positive `.is_file()` / `.exists()` / `.is_dir()`
  guards that drop entries from a declared path collection or silently substitute an
  empty container or `Default*` / `Fallback*` family member.
- `SilentExceptionFallbackDetector` (`BOUND-7` EAFP bypass + broad silent swallow):
  `except AttributeError` (with optional `TypeError`/`IndexError`/`KeyError`/`ValueError`)
  or broad `except Exception`/`BaseException`/bare `except` whose handler body silently
  swallows the error into a constant, empty container, `None`, `pass`, or `continue`.
- `ClonedDeclaredFamilyDetector` (`MEMB-1` / `TIME-1`): two or more unrelated modules
  declaring distinct `DeclaredFamily` root classes with the same name.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.detectors._base import (
    CompactClassRepositoryCandidateDetector,
    CompactClassRepositoryContext,
    DetectorConfig,
    PerModuleIssueDetector,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


@dataclass(frozen=True)
class ForeignCodeLanguage:
    """One non-Python language recognized inside large Python string literals."""

    label: str
    markers: tuple[str, ...]
    strong_markers: tuple[str, ...] = ()

    def matches(self, lowered: str) -> bool:
        if any(marker in lowered for marker in self.strong_markers):
            return True
        return sum(1 for marker in self.markers if marker in lowered) >= 2


_FOREIGN_CODE_LANGUAGES: tuple[ForeignCodeLanguage, ...] = (
    ForeignCodeLanguage(
        label="HTML/CSS/JS",
        markers=(
            "<html",
            "<script",
            "<style",
            "document.queryselector",
            "document.getelementbyid",
            "addeventlistener(",
            "innerhtml",
            "window.",
        ),
        strong_markers=("<!doctype html",),
    ),
    ForeignCodeLanguage(
        label="JavaScript",
        markers=(
            "await import(",
            "process.stdout",
            "json.stringify(",
            "export default",
            "const ",
            "function ",
        ),
    ),
    ForeignCodeLanguage(
        label="Shell",
        markers=(
            "#!/bin/bash",
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "set -eo pipefail",
        ),
    ),
    ForeignCodeLanguage(
        label="SQL",
        markers=(
            "create table",
            "alter table",
            "primary key",
            "foreign key",
        ),
    ),
)

_EMBEDDED_CODE_MIN_LINES = 20
_EMBEDDED_CODE_MIN_CHARS = 1500

_EXISTENCE_METHODS = frozenset({"is_file", "exists", "is_dir"})
_FILESYSTEM_DISCOVERY_SOURCES = frozenset(
    {"iterdir", "glob", "rglob", "parents", "scandir", "walk"}
)
_EMPTY_CONTAINER_CALLS = frozenset({"set", "frozenset", "list", "tuple", "dict"})
_DEFAULT_CONSTRUCTOR_PREFIXES = ("Default", "Fallback")

_BROAD_EXCEPTION_NAMES = frozenset({"Exception", "BaseException"})
_EAFP_ATTRIBUTE_COMPANIONS = frozenset(
    {"AttributeError", "TypeError", "IndexError", "KeyError", "ValueError"}
)


def _docstring_node_ids(tree: ast.Module) -> frozenset[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return frozenset(ids)


def _string_literal_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
        return "".join(parts) if parts else None
    return None


def _embedded_code_language(text: str) -> tuple[str, int, int] | None:
    line_count = text.count("\n") + 1
    char_count = len(text)
    if line_count < _EMBEDDED_CODE_MIN_LINES and char_count < _EMBEDDED_CODE_MIN_CHARS:
        return None
    lowered = text.lower()
    for language in _FOREIGN_CODE_LANGUAGES:
        if language.matches(lowered):
            return language.label, line_count, char_count
    return None


def _positive_existence_call(node: ast.AST) -> str | None:
    """Return a positive (non-negated) `.is_file()` / `.exists()` / `.is_dir()` call
    inside a boolean guard expression, ignoring negated checks (`if not p.exists()`)."""

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _EXISTENCE_METHODS
        and not node.args
        and not node.keywords
    ):
        return ast.unparse(node)
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            hit = _positive_existence_call(value)
            if hit is not None:
                return hit
    return None


def _is_filesystem_discovery_iterable(iterable: ast.expr) -> bool:
    for node in ast.walk(iterable):
        if isinstance(node, ast.Attribute) and node.attr in _FILESYSTEM_DISCOVERY_SOURCES:
            return True
        if isinstance(node, ast.Name) and node.id in {"os", "glob"}:
            return True
    return False


def _is_empty_or_sentinel_fallback(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant) and expr.value in (None, "", False, 0):
        return True
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return len(expr.elts) == 0
    if isinstance(expr, ast.Dict):
        return len(expr.keys) == 0
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id in _EMPTY_CONTAINER_CALLS
        and not expr.args
        and not expr.keywords
    ):
        return True
    return False


def _default_fallback_constructor_name(expr: ast.expr) -> str | None:
    """Match `DefaultFoo(...)` or `DefaultFoo(...).method(...)`."""

    current: ast.expr = expr
    while isinstance(current, ast.Call):
        func = current.func
        if isinstance(func, ast.Name) and func.id.startswith(
            _DEFAULT_CONSTRUCTOR_PREFIXES
        ):
            return func.id
        if isinstance(func, ast.Attribute):
            if func.attr.startswith(_DEFAULT_CONSTRUCTOR_PREFIXES):
                return func.attr
            current = func.value
            continue
        break
    return None


def _caught_exception_names(handler: ast.ExceptHandler) -> tuple[str, ...]:
    if handler.type is None:
        return ("bare except",)
    nodes = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else (handler.type,)
    )
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    return tuple(names)


def _is_silent_fallback_statement(
    statement: ast.stmt, caught_name: str | None
) -> bool:
    if isinstance(statement, (ast.Pass, ast.Continue)):
        return True
    if isinstance(statement, ast.Return):
        value = statement.value
        if value is None:
            return True
        if isinstance(value, ast.Name):
            return value.id != caught_name
        return isinstance(value, ast.Constant) or _is_empty_or_sentinel_fallback(value)
    if isinstance(statement, ast.Assign):
        value = statement.value
        if isinstance(value, ast.Name):
            return value.id != caught_name
        return isinstance(value, ast.Constant) or _is_empty_or_sentinel_fallback(value)
    return False


def _silent_exception_description(handler: ast.ExceptHandler) -> str | None:
    if len(handler.body) != 1:
        return None
    if not _is_silent_fallback_statement(handler.body[0], handler.name):
        return None
    names = _caught_exception_names(handler)
    if not names:
        return None
    rendered_action = ast.unparse(handler.body[0])
    name_set = set(names)
    if "bare except" in name_set or bool(name_set & _BROAD_EXCEPTION_NAMES):
        return f"broad `except ({', '.join(names)}): {rendered_action}`"
    if "AttributeError" in name_set and name_set <= _EAFP_ATTRIBUTE_COMPANIONS:
        return f"EAFP `except ({', '.join(names)}): {rendered_action}`"
    return None


class EmbeddedCodeLiteralDetector(PerModuleIssueDetector):
    """Detect substantial foreign code (HTML/CSS/JS, JS, shell, SQL) embedded in strings."""

    finding_spec = finding_spec_template(
        PatternId.EMBEDDED_CODE_LITERAL,
        "Embedded foreign code literal belongs in an owned source file (BOUND-5)",
        "Non-Python code inside a large Python string literal is invisible to linters, syntax checkers, and static analysis, and duplicates Python domain labels/contracts by hand.",
        "a checked asset file (`.html`, `.mjs`, `.sh`, `.sql`) loaded from package data",
        "non-docstring multiline string literal with >= 20 lines or >= 1500 characters containing foreign code markers",
        (CapabilityTag.PROVENANCE,),
        (ObservationTag.SEMANTIC_STRING_LITERAL,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        index = module.syntax_index
        docstrings = _docstring_node_ids(module.module)
        joined_child_ids = {
            id(part)
            for _node_index, joined in index.indexed_nodes_of_type(ast.JoinedStr)
            for part in joined.values
        }
        findings: list[RefactorFinding] = []
        for node_index, node in sorted(
            (
                *index.indexed_nodes_of_type(ast.Constant),
                *index.indexed_nodes_of_type(ast.JoinedStr),
            )
        ):
            if id(node) in docstrings or id(node) in joined_child_ids:
                continue
            text = _string_literal_text(node)
            if text is None:
                continue
            detected = _embedded_code_language(text)
            if detected is None:
                continue
            language, line_count, char_count = detected
            symbol = index.lexical_symbol(node_index)
            findings.append(
                self.build_finding(
                    f"`{symbol}` embeds {line_count} lines ({char_count:,} chars) of "
                    f"{language} code in a Python string literal (BOUND-5); move it to "
                    "an owned source asset.",
                    (SourceLocation(module.file_path, node.lineno, symbol),),
                )
            )
        return findings


class SilentMissingPathFallbackDetector(PerModuleIssueDetector):
    """Detect existence guards that silently drop missing paths or substitute defaults."""

    finding_spec = finding_spec_template(
        PatternId.SILENT_MISSING_PATH_FALLBACK,
        "Path existence guard silently drops missing inputs or falls back to a default",
        "Filtering a declared path list by `.is_file()`/`.exists()`/`.is_dir()`, or returning an empty container / `Default*` fallback when a path is absent, hides missing artifacts and typos behind exit code 0.",
        "fail-loud path validation at the boundary, raising `FileNotFoundError` when a declared input is absent",
        "comprehension filtering a non-discovery iterable by a positive existence check, or ternary/`or` falling back to empty/`Default*`",
        (CapabilityTag.FAIL_LOUD_CONTRACTS,),
        (ObservationTag.BRANCH_DISPATCH,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        index = module.syntax_index
        findings: list[RefactorFinding] = []
        for node_index, node in sorted(
            (
                *index.indexed_nodes_of_type(ast.ListComp),
                *index.indexed_nodes_of_type(ast.SetComp),
                *index.indexed_nodes_of_type(ast.GeneratorExp),
            )
        ):
            for generator in node.generators:
                if _is_filesystem_discovery_iterable(generator.iter):
                    continue
                existence = next(
                    (
                        hit
                        for condition in generator.ifs
                        if (hit := _positive_existence_call(condition)) is not None
                    ),
                    None,
                )
                if existence is not None:
                    symbol = index.lexical_symbol(node_index)
                    findings.append(
                        self.build_finding(
                            f"`{symbol}` filters a path collection with `{existence}`, "
                            "silently dropping missing inputs instead of failing loud.",
                            (SourceLocation(module.file_path, node.lineno, symbol),),
                        )
                    )
                    break
        for node_index, ifexp in index.indexed_nodes_of_type(ast.IfExp):
            existence = _positive_existence_call(ifexp.test)
            if existence is None:
                continue
            symbol = index.lexical_symbol(node_index)
            default_ctor = _default_fallback_constructor_name(ifexp.orelse)
            if default_ctor is not None:
                findings.append(
                    self.build_finding(
                        f"`{symbol}` guards `{existence}` with silent fallback to "
                        f"`{default_ctor}`, masking a missing input path.",
                        (SourceLocation(module.file_path, ifexp.lineno, symbol),),
                    )
                )
            elif isinstance(ifexp.body, ast.Call) and _is_empty_or_sentinel_fallback(
                ifexp.orelse
            ):
                findings.append(
                    self.build_finding(
                        f"`{symbol}` wraps a call in `if {existence} else "
                        f"{ast.unparse(ifexp.orelse)}`, silently swallowing missing input.",
                        (SourceLocation(module.file_path, ifexp.lineno, symbol),),
                    )
                )
        for node_index, boolop in index.indexed_nodes_of_type(ast.BoolOp):
            if not isinstance(boolop.op, ast.Or) or len(boolop.values) < 2:
                continue
            default_ctor = _default_fallback_constructor_name(boolop.values[-1])
            if default_ctor is not None:
                symbol = index.lexical_symbol(node_index)
                findings.append(
                    self.build_finding(
                        f"`{symbol}` falls back via `or` to `{default_ctor}`, silently "
                        "substituting default data when the requested input is empty.",
                        (SourceLocation(module.file_path, boolop.lineno, symbol),),
                    )
                )
        return findings


class SilentExceptionFallbackDetector(PerModuleIssueDetector):
    """Detect EAFP `AttributeError` (`BOUND-7` bypass) and broad silent `except` swallowers."""

    finding_spec = finding_spec_template(
        PatternId.SILENT_EXCEPTION_FALLBACK,
        "Exception handler silently swallows attribute or runtime errors into a default",
        "Catching `AttributeError` to return/assign a default is `getattr(..., default)` (BOUND-7) in an EAFP disguise, and broad `except Exception` silent returns mask real bugs behind degraded defaults.",
        "a declared nominal contract on the receiver or fail-loud error propagation",
        "single-statement silent `except` handler catching `AttributeError` (non-I/O) or broad `Exception`/`BaseException`/bare `except`",
        (CapabilityTag.FAIL_LOUD_CONTRACTS,),
        (ObservationTag.ATTRIBUTE_PROBE,),
    )

    def _findings_for_module(
        self, module: ParsedModule, config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        index = module.syntax_index
        findings: list[RefactorFinding] = []
        for node_index, handler in index.indexed_nodes_of_type(ast.ExceptHandler):
            description = _silent_exception_description(handler)
            if description is None:
                continue
            symbol = index.lexical_symbol(node_index)
            findings.append(
                self.build_finding(
                    f"`{symbol}` silently swallows failures via {description}; declare "
                    "the capability or let the error fail loud.",
                    (SourceLocation(module.file_path, handler.lineno, symbol),),
                )
            )
        return findings


@dataclass(frozen=True)
class ClonedDeclaredFamilyCandidate:
    """Two or more unrelated `DeclaredFamily` roots sharing one family name."""

    family_name: str
    roots: tuple[CompactIndexedClass, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return tuple(
            SourceLocation(root.file_path, root.line, root.qualname)
            for root in self.roots
        )


class ClonedDeclaredFamilyDetector(
    CompactClassRepositoryCandidateDetector[ClonedDeclaredFamilyCandidate]
):
    """Detect parallel `DeclaredFamily` roots with the same name across modules."""

    finding_spec = finding_spec_template(
        PatternId.CLONED_DECLARED_FAMILY,
        "Parallel DeclaredFamily roots with the same name split one family across modules",
        "Declaring a second `DeclaredFamily` root with the same name in another module creates a drifting copy of its members and metadata (MEMB-1 / TIME-1). Import the owning family instead.",
        "one `DeclaredFamily` root imported by every consumer module",
        "two or more unrelated classes directly inheriting `DeclaredFamily` with the same class name",
        (CapabilityTag.NOMINAL_IDENTITY, CapabilityTag.ENUMERATION),
        (ObservationTag.CLASS_FAMILY,),
    )

    def _candidates_from_compact_context(
        self,
        context: CompactClassRepositoryContext,
        config: DetectorConfig,
    ) -> tuple[ClonedDeclaredFamilyCandidate, ...]:
        del config
        index = context.class_index
        roots_by_name: dict[str, list[CompactIndexedClass]] = defaultdict(list)
        for indexed in index.classes_by_symbol.values():
            if any(
                base.rsplit(".", 1)[-1] == "DeclaredFamily"
                for base in indexed.declared_base_names
            ):
                roots_by_name[indexed.simple_name].append(indexed)
        candidates: list[ClonedDeclaredFamilyCandidate] = []
        for family_name, group in sorted(roots_by_name.items()):
            if len(group) < 2:
                continue
            independent = tuple(
                sorted(
                    (
                        root
                        for root in group
                        if not any(
                            other.symbol in index.ancestor_symbols(root.symbol)
                            for other in group
                            if other.symbol != root.symbol
                        )
                    ),
                    key=lambda root: (root.file_path, root.line),
                )
            )
            if len(independent) >= 2:
                candidates.append(
                    ClonedDeclaredFamilyCandidate(
                        family_name=family_name,
                        roots=independent,
                    )
                )
        return tuple(candidates)

    def _finding_for_candidate(
        self, candidate: ClonedDeclaredFamilyCandidate
    ) -> RefactorFinding:
        locations = ", ".join(
            f"`{root.symbol}` ({root.file_path}:{root.line})"
            for root in candidate.roots
        )
        return self.build_finding(
            f"`{candidate.family_name}` is declared as {len(candidate.roots)} "
            f"independent `DeclaredFamily` roots ({locations}): consolidate onto one "
            "owning declaration.",
            candidate.evidence,
        )


__all__ = (
    "ClonedDeclaredFamilyCandidate",
    "ClonedDeclaredFamilyDetector",
    "EmbeddedCodeLiteralDetector",
    "SilentExceptionFallbackDetector",
    "SilentMissingPathFallbackDetector",
)
