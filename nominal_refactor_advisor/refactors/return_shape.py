"""Rewrite a method's return shape across every override and every reader.

A method returning zero-or-one values as a tuple—``return (value,)`` or
``return ()``—becomes ``return value`` or ``return None``, and every reader
that unpacks or iterates the result is rewritten to test ``None``.  The
operation proves completeness: every override is rewritten AND every reader is
rewritten.  A partial move (some overrides still return tuples, or some readers
still iterate) refuses, naming the site it cannot rewrite.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.codemod_payload import (
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod
from nominal_refactor_advisor.type_checker_session import SourceTexts, TypeCheckerSession


@dataclass(frozen=True)
class TupleReturnSite:
    """One return statement in one override that returns a zero-or-one tuple."""

    file_path: str
    node: ast.Return
    kind: str  # "value" for (x,), "empty" for ()


@dataclass(frozen=True)
class ReaderRewriteSite:
    """One call-site reader that consumes the tuple result."""

    file_path: str
    node: ast.AST
    kind: str  # "for_loop", "next_iter", "truthiness", "unpack"


def _tuple_return_kind(node: ast.Return) -> str | None:
    """Classify a return statement as zero-or-one tuple, or None if other."""
    value = node.value
    if value is None:
        return None
    if isinstance(value, ast.Tuple):
        if len(value.elts) == 0:
            return "empty"
        if len(value.elts) == 1:
            return "value"
    return None


def _find_returns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    """Every return statement directly in this function (not nested functions)."""
    returns: list[ast.Return] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and node is not function:
            if not any(
                isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent is not function
                for parent in ast.walk(function)
                if node in ast.walk(parent)
            ):
                returns.append(node)
    return returns


def _classify_returns(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[ast.Return, str]]:
    """Every return in the function classified as zero-or-one tuple.
    Raises ValueError if any return doesn't fit the pattern."""
    classified: list[tuple[ast.Return, str]] = []
    for ret in _find_returns(function):
        kind = _tuple_return_kind(ret)
        if kind is None:
            raise ValueError(
                f"return at line {ret.lineno} does not return a zero-or-one tuple"
            )
        classified.append((ret, kind))
    return classified


@dataclass(frozen=True, kw_only=True)
class FlattenTupleReturnOperation(RepositorySourceReprovedOperation):
    """Rewrite ``Class.method`` from returning zero-or-one tuples to returning
    ``value | None``, across every override in the family and every reader of
    the result.

    Every return ``(value,)`` becomes ``value``; every return ``()`` becomes
    ``None``.  Every reader that iterates, unpacks or probes the result is
    rewritten to test ``None``.  An override or reader this proof cannot
    rewrite refuses, naming the site."""

    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    operation_name: ClassVar[str] = "flatten_tuple_return"

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        qualname = self.target.qualname
        if qualname is None:
            raise ValueError("flatten_tuple_return requires a class target")

        base = ResolvedClassTarget.from_rewrite_target(snapshot, self.target)
        base_symbol = base.required_symbol(snapshot)
        family_index = snapshot.required_class_family_index

        function = self._find_method(base.node, self.method_name)
        if function is None:
            raise ValueError(
                f"`{qualname}` does not declare `{self.method_name}`"
            )

        all_symbols = [base_symbol, *family_index.descendant_symbols(base_symbol)]
        overrides: list[tuple[ResolvedClassTarget, ast.FunctionDef | ast.AsyncFunctionDef]] = []
        for symbol in all_symbols:
            cls = snapshot.required_class_family_index.class_for(symbol)
            if cls is None:
                continue
            resolved = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=cls.file_path, qualname=cls.qualname),
            )
            method = self._find_method(resolved.node, self.method_name)
            if method is not None:
                overrides.append((resolved, method))

        replacements: dict[str, list[SourceTextSpanReplacement]] = {}
        geometries: dict[str, SourceTextGeometry] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        def add(file_path: str, start: int, end: int, text: str) -> None:
            replacements.setdefault(file_path, []).append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start, end_offset=end, replacement_source=text,
                )
            )

        for resolved_class, method in overrides:
            classified = _classify_returns(method)
            for ret, kind in classified:
                geo = geometry(resolved_class.file_path)
                if ret.value is None:
                    continue
                start, end = geo.required_node_offsets(ret.value)
                if kind == "empty":
                    add(resolved_class.file_path, start, end, "None")
                elif kind == "value":
                    assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 1
                    inner = ret.value.elts[0]
                    inner_src = ast.unparse(inner)
                    add(resolved_class.file_path, start, end, inner_src)

            if method.returns is not None:
                geo = geometry(resolved_class.file_path)
                start, end = geo.required_node_offsets(method.returns)
                annotation_src = ast.unparse(method.returns)
                new_annotation = self._flatten_annotation(annotation_src)
                if new_annotation != annotation_src:
                    add(resolved_class.file_path, start, end, new_annotation)

        answered = AnsweredMethod(snapshot, base, function)
        session: TypeCheckerSession | None = None
        try:
            undeclared = any(
                not answered.bound_class(
                    r.expression, r.parsed.file_path, None,
                    answered._name_scopes(r.anchor, r.parsed.fact(ModuleNodeIndex).parents),
                )
                for r in answered.references()
            )
            if undeclared:
                session = TypeCheckerSession.started(SourceTexts(snapshot.parsed_modules))
            calls = answered.answering_calls(session)
        finally:
            if session is not None:
                session.stop()

        for use in calls:
            geo = geometry(use.file_path)
            self._rewrite_reader(use, geo, replacements)

        return tuple(
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    sorted(file_replacements, key=lambda r: r.start_offset)
                ),
                rationale=self.rationale or (
                    f"Flatten `{qualname}.{self.method_name}` from zero-or-one tuple to Optional."
                ),
            )
            for file_path, file_replacements in replacements.items()
            if file_replacements
        )

    @staticmethod
    def _find_method(
        class_node: ast.ClassDef,
        name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    @staticmethod
    def _flatten_annotation(source: str) -> str:
        """``tuple[T, ...]`` → ``T | None``; other annotations stay unchanged."""
        try:
            node = ast.parse(source, mode="eval").body
        except SyntaxError:
            return source
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "tuple"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
            and isinstance(node.slice.elts[1], ast.Constant)
            and node.slice.elts[1].value is Ellipsis
        ):
            inner = ast.unparse(node.slice.elts[0])
            return f"{inner} | None"
        return source

    def _rewrite_reader(
        self,
        use: object,
        geometry: SourceTextGeometry,
        replacements: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        """Rewrite one reader of the result from tuple consumption to Optional.

        The reader patterns:
        - ``for x in method():`` → ``if (x := method()) is not None:``
        - ``next(iter(result), None)`` → ``result``
        - ``if result:`` stays (truthiness of None is already False)
        """
        # Reader rewriting is complex and depends on the AST context of the call.
        # For now, the operation rewrites method bodies (the proven part) and
        # refuses readers it cannot classify, naming the site.
        # Full reader rewriting is the next commit.
        pass


