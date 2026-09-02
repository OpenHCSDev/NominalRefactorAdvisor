"""Spacing primitives for codemod source moves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationInsertionSpacing:
    """Whitespace policy for inserting moved symbols into a destination module."""

    previous_line: str
    current_line: str
    has_import_block: bool

    @classmethod
    def from_source(
        cls,
        source: str,
        insertion_line: int,
        *,
        has_import_block: bool,
    ) -> "DestinationInsertionSpacing":
        lines = source.splitlines(keepends=True)

        def line_at(line_number: int) -> str:
            return lines[line_number - 1] if 1 <= line_number <= len(lines) else ""

        return cls(
            previous_line=line_at(insertion_line - 1),
            current_line=line_at(insertion_line),
            has_import_block=has_import_block,
        )

    @property
    def leading_separator(self) -> str:
        if self.previous_line.strip():
            return "\n"
        return ""

    @property
    def trailing_separator(self) -> str:
        if not self.current_line:
            return "\n\n"
        if self.current_line.strip():
            return "\n\n"
        return "\n"

    @property
    def import_separator(self) -> str:
        if self.has_import_block:
            return "\n"
        return ""
