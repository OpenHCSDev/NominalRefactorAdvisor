"""Exact extraction, replacement and removal of statements within Python scopes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

from .assignment_projection import (
    NamedAssignmentSelection,
    AssignmentStatementNameProjection,
)
from .ast_tools import is_docstring_statement
from .codemod_source_edits import (
    SourceNodeSpan,
    SourceNodeDecoratorPolicy,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .source_geometry import SourceByteSpan


@dataclass(frozen=True)
class StatementScopeSource(SourceTextGeometry):
    """One module, class or function body and its original source geometry."""

    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    file_path: str


@dataclass(frozen=True)
class StatementDeletionSource(StatementScopeSource):
    """Remove selected statements and their separators, not neighbouring code."""

    def replacements_for_statements(
        self, statements: tuple[ast.stmt, ...]
    ) -> tuple[SourceTextSpanReplacement, ...]:
        selected = frozenset(statements)
        body = self.node.body
        if len(selected) != len(statements) or not selected.issubset(body):
            raise ValueError("Statement deletion requires unique direct scope members")
        replacements = []
        for is_selected, group in groupby(
            enumerate(body), key=lambda item: item[1] in selected
        ):
            if not is_selected:
                continue
            run = tuple(group)
            first_index, first = run[0]
            last_index, last = run[-1]
            first_span = SourceNodeSpan(first, SourceNodeDecoratorPolicy.INCLUDE)
            first_line = self.node_start_line(first_span)
            start = self.line_offsets[first_line - 1] + SourceByteSpan.character_column(
                self.lines[first_line - 1], first.col_offset
            )
            end = self.required_node_offsets(last)[1]
            replacement = ""
            if (
                len(selected) == len(body) and not isinstance(self.node, ast.Module)
            ) or (
                first_index == 0
                and last_index + 1 < len(body)
                and is_docstring_statement(body[last_index + 1])
            ):
                replacement = "pass"
            elif (
                last_index + 1 < len(body)
                and body[last_index + 1].lineno == last.end_lineno
            ):
                end = self.required_node_offsets(body[last_index + 1])[0]
            elif first_index > 0 and body[first_index - 1].end_lineno == first.lineno:
                start = self.required_node_offsets(body[first_index - 1])[1]
            else:
                line_start, _ = self.node_span_offsets(first_span)
                _, line_end = self.node_span_offsets(SourceNodeSpan(last))
                if self.source[line_start:start].strip():
                    raise ValueError(
                        "Statement deletion cannot own the preceding source on this line"
                    )
                start, end = line_start, line_end
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=end,
                    replacement_source=replacement,
                )
            )
        return tuple(replacements)


@dataclass(frozen=True)
class AssignmentDeletionSource(StatementDeletionSource):
    """Select complete named assignments before applying shared deletion geometry."""

    def replacements(
        self, names: tuple[str, ...]
    ) -> tuple[SourceTextSpanReplacement, ...]:
        return self.replacements_for_statements(
            NamedAssignmentSelection(names).statements(self.node.body)
        )


@dataclass(frozen=True)
class AssignmentSource(SourceTextGeometry):
    """One authored direct-name assignment, including an annotation-only field."""

    @cached_property
    def statement(self) -> ast.Assign | ast.AnnAssign:
        try:
            module = ast.parse(self.source)
        except SyntaxError as error:
            raise ValueError(
                f"Assignment source is not valid Python: {error}"
            ) from error
        if len(module.body) != 1:
            raise ValueError("Assignment source must contain exactly one statement")
        statement = module.body[0]
        if (
            not isinstance(statement, ast.Assign | ast.AnnAssign)
            or AssignmentStatementNameProjection(statement).direct_name is None
        ):
            raise ValueError("Statement is not a single direct-name assignment")
        start, end = self.required_node_offsets(statement)
        if self.source[:start].strip() or self.source[end:].strip():
            raise ValueError("Assignment source contains text outside its statement")
        return statement

    @property
    def name(self) -> str:
        return AssignmentStatementNameProjection(self.statement).names[0]

    def replacement_source(self, indentation: str) -> str:
        start, end = self.required_node_offsets(self.statement)
        return (
            SourceTextGeometry(self.source[start:end])
            .indented_source(indentation)
            .removeprefix(indentation)
        )


@dataclass(frozen=True)
class AssignmentReplacementSource(StatementScopeSource):
    """Replace one complete assignment without owning its neighbours or suffix."""

    def replacement(
        self, name: str, replacement: AssignmentSource
    ) -> SourceTextSpanReplacement:
        (statement,) = NamedAssignmentSelection((name,)).statements(self.node.body)
        span = SourceTextSpan(*self.required_node_offsets(statement))
        if self.span_contains_comment(span):
            raise ValueError("Assignment replacement would discard comments")
        line = self.lines[statement.lineno - 1]
        indentation = line[: len(line) - len(line.lstrip())]
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source=replacement.replacement_source(indentation),
        )


@dataclass(frozen=True)
class StatementSource(SourceTextGeometry):
    """Project one statement into a destination indentation without its neighbours."""

    node: ast.stmt

    def member_source(self, indentation: str) -> str:
        span = SourceNodeSpan(self.node, SourceNodeDecoratorPolicy.INCLUDE)
        first_line = self.node_start_line(span)
        original_line = self.lines[first_line - 1]
        original_indentation = original_line[
            : len(original_line) - len(original_line.lstrip())
        ]
        start = self.line_offsets[first_line - 1] + SourceByteSpan.character_column(
            original_line, self.node.col_offset
        )
        end = self.required_node_offsets(self.node)[1]
        _, line_end = self.node_span_offsets(SourceNodeSpan(self.node))
        suffix = self.source[end:line_end]
        if not suffix.strip() or suffix.lstrip(" \t;").startswith("#"):
            end = line_end
        member = self.source[start:end]
        continuation_lines = self.literal_continuation_lines(self.node)
        rendered = "".join(
            (
                line
                if number in continuation_lines or not line.strip()
                else indentation
                + (
                    line
                    if number == first_line
                    else line.removeprefix(original_indentation)
                )
            )
            for number, line in enumerate(
                member.splitlines(keepends=True), start=first_line
            )
        )
        newline = "\r\n" if self.lines[span.end_line - 1].endswith("\r\n") else "\n"
        return rendered if rendered.endswith(("\r", "\n")) else rendered + newline
