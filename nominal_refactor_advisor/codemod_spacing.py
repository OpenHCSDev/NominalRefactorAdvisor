"""Spacing primitives for composable codemod source insertions."""

from dataclasses import dataclass
from enum import StrEnum

from nominal_refactor_advisor.fielded_enum import FieldedStrEnum


class SourceInsertionLayer(StrEnum):
    """Where an insertion's lines stand among the others at its line: a module's
    imports head it, so they lead every body insertion sharing their line,
    whichever edit the plan produced first."""

    IMPORTS = "imports"
    BODY = "body"

    @property
    def rank(self) -> int:
        return tuple(type(self)).index(self)


class SourceInsertionBoundary(FieldedStrEnum):
    """Typed leading separation owned by the inserted declaration."""

    PRESERVE = ("preserve", None)
    ONE_BLANK_LINE = ("one_blank_line", 1)
    TWO_BLANK_LINES = ("two_blank_lines", 2)

    _blank_line_count: int | None

    @property
    def required_blank_line_count(self) -> int:
        if self._blank_line_count is None:
            raise ValueError("Preserved insertion boundaries have no required spacing")
        return self._blank_line_count

    @classmethod
    def from_declaration_line(cls, source_line: str) -> "SourceInsertionBoundary":
        """Derive module-level versus nested spacing from declaration geometry."""

        indentation = source_line[: len(source_line) - len(source_line.lstrip())]
        if indentation:
            return cls.ONE_BLANK_LINE
        return cls.TWO_BLANK_LINES

    def coalesce_lines(
        self,
        preceding_lines: tuple[str, ...],
        inserted_lines: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Join same-anchor insertions through this insertion's boundary."""

        if self is SourceInsertionBoundary.PRESERVE:
            return (*preceding_lines, *inserted_lines)
        preceding_content = _without_trailing_blank_lines(preceding_lines)
        inserted_content = _without_leading_blank_lines(inserted_lines)
        if not preceding_content or not inserted_content:
            return (*preceding_content, *inserted_content)
        return (
            *preceding_content,
            *("\n" for _ in range(self.required_blank_line_count)),
            *inserted_content,
        )


def _without_leading_blank_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    first_content_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        len(lines),
    )
    return lines[first_content_index:]


def _without_trailing_blank_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    trailing_blank_line_count = next(
        (index for index, line in enumerate(reversed(lines)) if line.strip()),
        len(lines),
    )
    if trailing_blank_line_count == 0:
        return lines
    return lines[:-trailing_blank_line_count]


@dataclass(frozen=True)
class DestinationInsertionSpacing:
    """Whitespace policy for inserting moved symbols into a destination module."""

    has_preceding_content: bool
    preceding_blank_line_count: int
    has_following_content: bool
    following_blank_line_count: int
    following_content_is_definition: bool
    """Whether the first statement after the insertion is a `def`, `class` or a
    decorator: an import block keeps two blank lines before one, one before any
    other statement."""
    inserted_source_is_import_block: bool
    boundary: SourceInsertionBoundary

    @classmethod
    def from_source(
        cls,
        source: str,
        insertion_line: int,
        *,
        inserted_source_is_import_block: bool,
        boundary: SourceInsertionBoundary = SourceInsertionBoundary.TWO_BLANK_LINES,
    ) -> "DestinationInsertionSpacing":
        lines = source.splitlines(keepends=True)
        preceding_lines = lines[: insertion_line - 1]
        following_lines = lines[insertion_line - 1 :]
        preceding_blank_line_count = next(
            (
                index
                for index, line in enumerate(reversed(preceding_lines))
                if line.strip()
            ),
            len(preceding_lines),
        )
        following_blank_line_count = next(
            (index for index, line in enumerate(following_lines) if line.strip()),
            len(following_lines),
        )

        following_content = next((line.lstrip() for line in following_lines if line.strip()), "")
        return cls(
            following_content_is_definition=following_content.startswith(
                ("def ", "async def ", "class ", "@")
            ),
            has_preceding_content=any(line.strip() for line in preceding_lines),
            preceding_blank_line_count=preceding_blank_line_count,
            has_following_content=any(line.strip() for line in following_lines),
            following_blank_line_count=following_blank_line_count,
            inserted_source_is_import_block=inserted_source_is_import_block,
            boundary=boundary,
        )

    @property
    def leading_separator(self) -> str:
        if not self.has_preceding_content:
            return ""
        return "\n" * max(
            0,
            self.boundary.required_blank_line_count - self.preceding_blank_line_count,
        )

    @property
    def trailing_separator(self) -> str:
        missing_blank_lines = max(
            0,
            self.boundary.required_blank_line_count - self.following_blank_line_count,
        )
        if self.inserted_source_is_import_block:
            if not self.has_following_content:
                return ""
            required = 2 if self.following_content_is_definition else 1
            return "\n" * max(0, required - self.following_blank_line_count)
        if not self.has_following_content:
            return "\n"
        return "\n" * (1 + missing_blank_lines)
