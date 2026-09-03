"""Spacing primitives for codemod source moves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationInsertionSpacing:
    """Whitespace policy for inserting moved symbols into a destination module."""

    has_preceding_content: bool
    preceding_blank_line_count: int
    has_following_content: bool
    following_blank_line_count: int
    inserted_source_is_import_block: bool

    @classmethod
    def from_source(
        cls,
        source: str,
        insertion_line: int,
        *,
        inserted_source_is_import_block: bool,
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

        return cls(
            has_preceding_content=any(line.strip() for line in preceding_lines),
            preceding_blank_line_count=preceding_blank_line_count,
            has_following_content=any(line.strip() for line in following_lines),
            following_blank_line_count=following_blank_line_count,
            inserted_source_is_import_block=inserted_source_is_import_block,
        )

    @property
    def leading_separator(self) -> str:
        if not self.has_preceding_content:
            return ""
        return "\n" * max(0, 2 - self.preceding_blank_line_count)

    @property
    def leading_separator_after_pending_imports(self) -> str:
        """Preserve separators consumed by an import inserted at the same anchor."""

        return "\n" * min(2, self.following_blank_line_count)

    @property
    def trailing_separator(self) -> str:
        missing_blank_lines = max(0, 2 - self.following_blank_line_count)
        if self.inserted_source_is_import_block:
            return "\n" * missing_blank_lines
        if not self.has_following_content:
            return "\n"
        return "\n" * (1 + missing_blank_lines)
