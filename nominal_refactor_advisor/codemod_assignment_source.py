"""Exact statement geometry for explicit assignment removal in Python scopes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

from .assignment_projection import NamedAssignmentSelection
from .ast_tools import is_docstring_statement
from .codemod_source_edits import (
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)


@dataclass(frozen=True)
class AssignmentDeletionSource:
    """Remove selected statements and their separators, not neighbouring code."""

    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    source: str
    file_path: str

    @cached_property
    def geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(self.source)

    def replacements(
        self, names: tuple[str, ...]
    ) -> tuple[SourceTextSpanReplacement, ...]:
        selected = frozenset(NamedAssignmentSelection(names).statements(self.node.body))
        body = self.node.body
        replacements = []
        for is_selected, group in groupby(
            enumerate(body), key=lambda item: item[1] in selected
        ):
            if not is_selected:
                continue
            run = tuple(group)
            first_index, first = run[0]
            last_index, last = run[-1]
            start = self.geometry.required_node_offsets(first)[0]
            end = self.geometry.required_node_offsets(last)[1]
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
                end = self.geometry.required_node_offsets(body[last_index + 1])[0]
            elif first_index > 0 and body[first_index - 1].end_lineno == first.lineno:
                start = self.geometry.required_node_offsets(body[first_index - 1])[1]
            else:
                line_start, _ = self.geometry.node_span_offsets(SourceNodeSpan(first))
                _, line_end = self.geometry.node_span_offsets(SourceNodeSpan(last))
                if self.source[line_start:start].strip():
                    raise ValueError(
                        "Assignment deletion cannot own the preceding source on this line"
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
