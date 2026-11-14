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
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod, AnsweredUse
from nominal_refactor_advisor.type_checker_session import SourceTexts, TypeCheckerSession
from typing import ClassVar

def _collect_returns(
    stmts: list[ast.stmt], into: list[ast.Return]
) -> None:
    """Walk statements collecting returns, stopping at nested function boundaries."""
    for stmt in stmts:
        if isinstance(stmt, ast.Return):
            into.append(stmt)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass
        elif isinstance(stmt, (ast.If, ast.With, ast.AsyncWith)):
            _collect_returns(stmt.body, into)
            if hasattr(stmt, "orelse"):
                _collect_returns(stmt.orelse, into)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            _collect_returns(stmt.body, into)
            _collect_returns(stmt.orelse, into)
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            _collect_returns(stmt.body, into)
            for handler in stmt.handlers:
                if handler.body:
                    _collect_returns(handler.body, into)
            _collect_returns(stmt.orelse, into)
            _collect_returns(stmt.finalbody, into)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                _collect_returns(case.body, into)


@dataclass(frozen=True, kw_only=True)
class FlattenTupleReturnOperation(RepositorySourceReprovedOperation):
    """Rewrite ``Class.method`` from returning zero-or-one tuples to returning
    ``value | None``, across every override in the family and every reader of
    the result.

    Every return ``(value,)`` becomes ``value``; every return ``()`` becomes
    ``None``.  Every reader that iterates, unpacks or probes the result is
    rewritten to test ``None``.  An override or reader this proof cannot
    rewrite refuses, naming the site."""

    _shared_type_checker_session: ClassVar[TypeCheckerSession | None] = None

    method_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())

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
            classified = self._classify_returns(method)
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
        shared = self._shared_type_checker_session
        owned_session = shared is None
        session = shared or TypeCheckerSession.started(
            SourceTexts(snapshot.parsed_modules)
        )
        try:
            calls = answered.answering_calls(session)
        finally:
            if owned_session:
                session.stop()

        for use in calls:
            geo = geometry(use.file_path)
            self._rewrite_reader(use, snapshot, geo, replacements)

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
        use: AnsweredUse,
        snapshot: CodemodSourceSnapshot,
        geometry: SourceTextGeometry,
        replacements: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        """Rewrite one reader of the result from tuple consumption to Optional.

        The dominant pattern: ``for x in method():`` becomes
        ``if (x := method()) is not None:`` and the loop body keeps its
        indentation.  Other patterns (next(iter()), unpacking, truthiness)
        refuse by name until they are implemented."""

        parents = snapshot.parsed_module_for_source_path(
            use.file_path
        ).fact(ModuleNodeIndex).parents
        call_node = use.node
        parent = parents.get(call_node)
        if parent is None:
            raise ValueError(
                f"{use.where} calls `{self.method_name}` but the call has no "
                "parent in the AST"
            )

        if isinstance(parent, ast.For) and parent.iter is call_node:
            self._rewrite_for_loop(use, parent, geometry, replacements)
            return

        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "iter"
            and len(parent.args) == 1
        ):
            grandparent = parents.get(parent)
            if (
                isinstance(grandparent, ast.Call)
                and isinstance(grandparent.func, ast.Name)
                and grandparent.func.id == "next"
                and len(grandparent.args) >= 1
            ):
                self._rewrite_next_iter(use, grandparent, geometry, replacements)
                return

        if isinstance(parent, (ast.If, ast.BoolOp)):
            return

        if isinstance(parent, ast.Return) and parent.value is call_node:
            self._rewrite_return(use, parent, geometry, replacements)
            return

        if isinstance(parent, ast.Starred) and parent.value is call_node:
            return

        raise ValueError(
            f"{use.where} reads `{self.method_name}` in a context this "
            f"operation cannot rewrite: {type(parent).__name__} at line "
            f"{getattr(parent, 'lineno', '?')}"
        )

    def _rewrite_for_loop(
        self,
        use: AnsweredUse,
        for_node: ast.For,
        geometry: SourceTextGeometry,
        replacements: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        """``for x in method():`` → ``if (x := method()) is not None:``"""
        if for_node.orelse:
            raise ValueError(
                f"{use.where} has an else clause on the for loop; "
                "flatten_tuple_return cannot rewrite for/else"
            )
        target_src = ast.unparse(for_node.target)
        call_src = ast.unparse(use.node)
        start, _end = geometry.required_node_offsets(for_node)
        header_line = geometry.source[start:].split("\n", 1)[0]
        indent = header_line[: len(header_line) - len(header_line.lstrip())]
        new_header = f"{indent}if ({target_src} := {call_src}) is not None:"
        end_of_header = start + len(header_line)
        replacements.setdefault(use.file_path, []).append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end_of_header,
                replacement_source=new_header,
            )
        )

    def _rewrite_next_iter(
        self,
        use: AnsweredUse,
        next_call: ast.Call,
        geometry: SourceTextGeometry,
        replacements: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        """``next(iter(method()), None)`` → ``method()``"""
        call_src = ast.unparse(use.node)
        start, end = geometry.required_node_offsets(next_call)
        replacements.setdefault(use.file_path, []).append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=call_src,
            )
        )

    def _rewrite_return(
        self,
        use: AnsweredUse,
        return_node: ast.Return,
        geometry: SourceTextGeometry,
        replacements: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        """``return method()`` → ``return ((_r := method()),) if _r is not None else ()``

        The caller still returns a tuple; the called method now returns
        ``T | None``.  A walrus wraps the result back into a one-element
        tuple or an empty tuple so the caller's contract stays unchanged."""
        call_src = ast.unparse(use.node)
        assert return_node.value is not None
        start, end = geometry.required_node_offsets(return_node.value)
        new_src = f"((_r := {call_src}),) if _r is not None else ()"
        replacements.setdefault(use.file_path, []).append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=new_src,
            )
        )

    @staticmethod
    def _classify_returns(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[tuple[ast.Return, str]]:
        """Every return in the function classified as zero-or-one tuple.
        Raises ValueError if any return doesn't fit the pattern."""
        classified: list[tuple[ast.Return, str]] = []
        for ret in FlattenTupleReturnOperation._find_returns(function):
            kind = FlattenTupleReturnOperation._tuple_return_kind(ret)
            if kind is None:
                raise ValueError(
                    f"return at line {ret.lineno} does not return a zero-or-one tuple"
                )
            classified.append((ret, kind))
        return classified

    @staticmethod
    def _find_returns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
        """Every return statement directly in this function (not nested functions)."""
        returns: list[ast.Return] = []
        _collect_returns(function.body, returns)
        return returns

    @staticmethod
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


