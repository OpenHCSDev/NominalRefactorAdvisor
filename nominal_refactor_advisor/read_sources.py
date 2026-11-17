"""The text a read was parsed from, placed in the file that holds it.

A module is its own source. Python source a string literal holds, which a
child interpreter runs (`python -c`, a probe script), is another: its own
imports resolve its names, and each of its characters maps back to an offset
in the holding file, so a read in it is counted and can be rewritten where it
stands.
"""

from __future__ import annotations

import ast
import io
import textwrap
import tokenize
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Self, cast

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleFact
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.source_geometry import SourceByteSpan, SourceTextLayout


class ReadSource(ABC):
    """One parse a read was found in, and where its text sits in its file."""

    file_path: str
    parsed_module: ParsedModule

    @cached_property
    def layout(self) -> SourceTextLayout:
        return SourceTextLayout.of(self.parsed_module.source)

    def offsets_in_parse(self, node: ast.AST) -> tuple[int, int]:
        return SourceByteSpan.require_node(node).character_offsets(
            self.layout.lines, self.layout.line_offsets
        )

    @abstractmethod
    def file_offset(self, offset: int) -> int:
        """Where a character of this parse's text sits in the holding file."""

        raise NotImplementedError

    @abstractmethod
    def line_of(self, node: ast.AST) -> int:
        """The holding file's line a node starts on."""

        raise NotImplementedError

    def node_offsets(self, node: ast.AST) -> tuple[int, int]:
        start, end = self.offsets_in_parse(node)
        return self.file_offset(start), self.file_offset(end)


@dataclass(frozen=True)
class ModuleSource(ReadSource):
    """A scanned module, read where it stands."""

    parsed_module: ParsedModule

    @property
    def file_path(self) -> str:
        return self.parsed_module.file_path

    def file_offset(self, offset: int) -> int:
        return offset

    def line_of(self, node: ast.AST) -> int:
        return node.lineno


@dataclass(frozen=True)
class EmbeddedSource(ReadSource):
    """Python source a string literal holds, parsed as the module a child
    interpreter runs it as, under a name no importable module has.

    `positions` gives each character's offset in the holding file, then the
    offset past the last; it is empty when the literal's characters do not
    decode to its value one by one, and then any edit refuses.
    """

    holder: ParsedModule
    parsed_module: ParsedModule
    literal: ast.Constant
    positions: tuple[int, ...]

    @property
    def file_path(self) -> str:
        return self.holder.file_path

    def file_offset(self, offset: int) -> int:
        if not self.positions:
            raise NeedsInput(
                f"{self.file_path}:{self.literal.lineno} holds Python source whose characters "
                "NRA cannot place back in the file", NeededInput.OPERATION
            )
        return self.positions[offset]

    def line_at(self, offset: int) -> int:
        """The holding file's line a character of this parse's text sits on."""

        if not self.positions:
            return self.literal.lineno
        return self.holder.source.count("\n", 0, self.positions[offset]) + 1

    def line_of(self, node: ast.AST) -> int:
        return self.line_at(self.offsets_in_parse(node)[0])

    def holder_line(self, line: int) -> int:
        """The holding file's line this parse's `line` starts on."""

        return self.line_at(self.layout.line_offsets[line - 1])


@dataclass(frozen=True)
class EmbeddedSources(ParsedModuleFact):
    """Every string literal in a module that holds Python source importing
    something, as it is run: verbatim, or through `textwrap.dedent`."""

    sources: tuple[EmbeddedSource, ...]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        index = parsed_module.syntax_index
        nodes, parents = index.depth_first_nodes, index.parent_indices
        literals = (
            node
            for position in index.node_indices_by_type.get(ast.Constant, ())
            for node in (nodes[position],)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "import" in node.value
            # A part of an f-string is a fragment of text the program assembles.
            and not (parents[position] >= 0 and isinstance(nodes[parents[position]], ast.JoinedStr))
        )
        return cls(tuple(source for literal in literals for source in cls.parsed(parsed_module, literal)))

    @staticmethod
    def parsed(holder: ParsedModule, literal: ast.Constant) -> Iterator[EmbeddedSource]:
        value = cast(str, literal.value)
        for text in dict.fromkeys((value, textwrap.dedent(value))):
            try:
                module = ast.parse(text)
            except SyntaxError:
                continue
            if not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(module)):
                return
            layout = SourceTextLayout.of(holder.source)
            span = SourceByteSpan.require_node(literal).character_offsets(layout.lines, layout.line_offsets)
            characters = _literal_positions(holder.source, span, value)
            yield EmbeddedSource(
                holder=holder,
                parsed_module=ParsedModule(
                    path=holder.path,
                    module_name=f"{holder.module_name}:script@{literal.lineno}:{literal.col_offset}",
                    module=module,
                    source=text,
                ),
                literal=literal,
                positions=_aligned(text, value, characters) if characters else (),
            )
            return


_SIMPLE_ESCAPES = {
    "\\": "\\", "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f",
    "n": "\n", "r": "\r", "t": "\t", "v": "\v",
}


def _literal_positions(source: str, span: tuple[int, int], value: str) -> tuple[int, ...]:
    """Each character of a string literal's value with its offset in `source`,
    then the offset past the last; empty unless the characters decode to `value`."""

    start, end = span
    decoded: list[str] = []
    positions: list[int] = []
    segment = source[start:end]
    line_starts = [0, *(index + 1 for index, character in enumerate(segment) if character == "\n")]
    try:
        tokens = tuple(tokenize.generate_tokens(io.StringIO(segment).readline))
    except (tokenize.TokenError, SyntaxError):
        return ()
    body_end = start
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        text = token.string
        prefix = text[: len(text) - len(text.lstrip("rRuUbBfF"))].lower()
        if "b" in prefix or "f" in prefix:
            return ()
        opening = text[len(prefix) : len(prefix) + 3]
        quote = opening if opening in ('"""', "'''") else text[len(prefix)]
        body = text[len(prefix) + len(quote) : len(text) - len(quote)]
        body_start = start + line_starts[token.start[0] - 1] + token.start[1] + len(prefix) + len(quote)
        body_end = body_start + len(body)
        index = 0
        while index < len(body):
            character = body[index]
            following = body[index + 1] if index + 1 < len(body) else ""
            if character != "\\" or "r" in prefix:
                decoded.append(character)
                positions.append(body_start + index)
                index += 1
            elif following == "\n":
                index += 2
            elif following in _SIMPLE_ESCAPES:
                decoded.append(_SIMPLE_ESCAPES[following])
                positions.append(body_start + index)
                index += 2
            elif following and following in "01234567":
                window = body[index + 1 : index + 4]
                digits = len(window) - len(window.lstrip("01234567"))
                decoded.append(chr(int(body[index + 1 : index + 1 + digits], 8)))
                positions.append(body_start + index)
                index += 1 + digits
            elif following and following in "xuU":
                width = {"x": 2, "u": 4, "U": 8}[following]
                decoded.append(chr(int(body[index + 2 : index + 2 + width], 16)))
                positions.append(body_start + index)
                index += 2 + width
            elif following == "N":
                close = body.index("}", index)
                decoded.append(unicodedata.lookup(body[index + 3 : close]))
                positions.append(body_start + index)
                index = close + 1
            else:
                decoded.append(character)
                positions.append(body_start + index)
                index += 1
    if "".join(decoded) != value:
        return ()
    return (*positions, body_end)


def _aligned(text: str, value: str, positions: tuple[int, ...]) -> tuple[int, ...]:
    """`positions` for `text`, which is `value` with whitespace removed (what
    `textwrap.dedent` does); each kept character keeps its own offset."""

    aligned: list[int] = []
    index = 0
    for character in text:
        while value[index] != character:
            index += 1
        aligned.append(positions[index])
        index += 1
    return (*aligned, positions[-1] if index == len(value) else positions[index])
