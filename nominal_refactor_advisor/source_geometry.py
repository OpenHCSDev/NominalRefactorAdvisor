"""Exact source-text projections shared by analysis and codemod consumers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class SourceByteSpan:
    """Validated UTF-8 byte span over one parsed source buffer."""

    start_line_index: int
    end_line_index: int
    start_byte: int
    end_byte: int

    @classmethod
    def from_node(
        cls,
        node: ast.expr | ast.stmt,
    ) -> SourceByteSpan | None:
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        return cls(
            start_line_index=node.lineno - 1,
            end_line_index=node.end_lineno - 1,
            start_byte=node.col_offset,
            end_byte=node.end_col_offset,
        )

    @classmethod
    def require_node(cls, node: ast.expr | ast.stmt) -> SourceByteSpan:
        """Return one complete parser-provided span or fail closed."""

        span = cls.from_node(node)
        if span is None:
            raise ValueError("AST node has no complete UTF-8 source span")
        return span

    def fits_lines(self, lines: tuple[str, ...]) -> bool:
        return (
            self.start_line_index >= 0
            and self.end_line_index >= self.start_line_index
            and self.end_line_index < len(lines)
        )

    @property
    def single_line(self) -> bool:
        return self.start_line_index == self.end_line_index

    def segment(self, lines: tuple[str, ...]) -> str:
        if self.single_line:
            return self.line_segment(
                lines[self.start_line_index],
                start_byte=self.start_byte,
                end_byte=self.end_byte,
            )
        return "".join(
            (
                self.line_segment(
                    lines[self.start_line_index],
                    start_byte=self.start_byte,
                    end_byte=None,
                ),
                *lines[self.start_line_index + 1 : self.end_line_index],
                self.line_segment(
                    lines[self.end_line_index],
                    start_byte=0,
                    end_byte=self.end_byte,
                ),
            )
        )

    def character_offsets(
        self,
        lines: tuple[str, ...],
        line_offsets: tuple[int, ...],
    ) -> tuple[int, int]:
        """Project this AST byte span into Python string character offsets."""

        return (
            line_offsets[self.start_line_index]
            + self.character_column(
                lines[self.start_line_index],
                self.start_byte,
            ),
            line_offsets[self.end_line_index]
            + self.character_column(
                lines[self.end_line_index],
                self.end_byte,
            ),
        )

    @staticmethod
    def line_segment(
        line: str,
        *,
        start_byte: int,
        end_byte: int | None,
    ) -> str:
        return line.encode("utf-8")[start_byte:end_byte].decode("utf-8")

    @staticmethod
    def character_column(line: str, byte_offset: int) -> int:
        return len(line.encode("utf-8")[:byte_offset].decode("utf-8"))


@dataclass(frozen=True)
class SourceLineSegmentAuthority:
    """Project parsed AST spans into exact source text from one line index."""

    source: str

    @cached_property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.source.splitlines(keepends=True))

    def segment_for_node(self, node: ast.expr | ast.stmt) -> str | None:
        span = SourceByteSpan.from_node(node)
        if span is None or not span.fits_lines(self.lines):
            return None
        return span.segment(self.lines)
