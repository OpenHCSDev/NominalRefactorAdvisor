"""Exact source-text projections shared by analysis and codemod consumers."""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True)
class ClassHeaderSourceSpan:
    """Exact source span and reconstruction safety for one class header."""

    node: ast.ClassDef
    source_lines: tuple[str, ...]

    @classmethod
    def from_source(cls, node: ast.ClassDef, source: str) -> "ClassHeaderSourceSpan":
        return cls(node=node, source_lines=tuple(source.splitlines(keepends=True)))

    @property
    def start_line(self) -> int:
        return self.node.lineno

    @property
    def end_line(self) -> int:
        return self.end_position[0]

    @cached_property
    def candidate_source(self) -> str:
        """Bound header inspection to the first body statement's starting line."""

        first_statement_line = min(
            self.statement_start_line(statement) for statement in self.node.body
        )
        return "".join(
            self.source_lines[self.start_line - 1 : first_statement_line]
        )

    @cached_property
    def outer_tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        """Tokenise only the candidate header lines and stop at the suite colon."""

        tokens = []
        for token in unenclosed_python_tokens(
            tokenize.generate_tokens(io.StringIO(self.candidate_source).readline)
        ):
            tokens.append(token)
            if token.type == tokenize.OP and token.string == ":":
                return tuple(tokens)
        raise ValueError(f"Cannot resolve class header colon for {self.node.name!r}")

    @property
    def end_position(self) -> tuple[int, int]:
        line, column = self.outer_tokens[-1].end
        return self.start_line + line - 1, column

    @cached_property
    def declaration_prefix(self) -> str:
        boundary = next(
            token
            for token in self.outer_tokens
            if token.type == tokenize.OP and token.string in {"(", ":"}
        )
        line, column = boundary.start
        return (
            "".join(self.source_lines[self.start_line - 1 : self.start_line + line - 2])
            + self.source_lines[self.start_line + line - 2][:column]
        ).strip()

    @staticmethod
    def statement_start_line(statement: ast.stmt) -> int:
        if not isinstance(
            statement,
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            return statement.lineno
        decorator_lines = tuple(
            decorator.lineno
            for decorator in statement.decorator_list
            if decorator.lineno
        )
        return min((*decorator_lines, statement.lineno))

    @property
    def source(self) -> str:
        line, column = self.end_position
        return (
            "".join(self.source_lines[self.start_line - 1 : line - 1])
            + self.source_lines[line - 1][:column]
        )

    @cached_property
    def contains_comment(self) -> bool:
        if "#" not in self.candidate_source:
            return False
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.source).readline)
            return any(token.type == tokenize.COMMENT for token in tokens)
        except tokenize.TokenError:
            return True

    @property
    def is_reconstructible(self) -> bool:
        return not self.contains_comment


def read_source_text(path: Path, *, encoding: str = "utf-8") -> str:
    """Decode exact source bytes without newline translation."""

    return path.read_bytes().decode(encoding)


def unenclosed_python_tokens(
    tokens: Iterable[tokenize.TokenInfo],
) -> Iterator[tokenize.TokenInfo]:
    """Yield tokens outside brackets, including each opening delimiter."""

    depth = 0
    for token in tokens:
        if depth == 0:
            yield token
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1


@dataclass(frozen=True)
class SourceByteSpan:
    """Validated UTF-8 byte span over one parsed source buffer."""

    start_line_index: int
    end_line_index: int
    start_byte: int
    end_byte: int

    @property
    def start_line(self) -> int:
        """One-based source line, derived from the retained span."""
        return self.start_line_index + 1

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


@dataclass(frozen=True)
class SourceCommentLineIndex:
    """One token-derived comment-line projection for a complete source file."""

    comment_lines: frozenset[int]
    is_complete: bool

    @classmethod
    def from_source(cls, source: str) -> "SourceCommentLineIndex":
        try:
            return cls(
                comment_lines=frozenset(
                    token.start[0]
                    for token in tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.COMMENT
                ),
                is_complete=True,
            )
        except (IndentationError, tokenize.TokenError):
            return cls(frozenset(), is_complete=False)

    def intersects(self, node: ast.expr | ast.stmt) -> bool:
        """Return true when comments occur within an AST node's line span."""

        end_line = node.end_lineno or node.lineno
        return not self.is_complete or any(
            node.lineno <= line <= end_line for line in self.comment_lines
        )
