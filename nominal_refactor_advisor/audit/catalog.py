"""The pattern catalog's index, derived from the rung files' own headings.

Each rung file declares its title (`# Title: question`) and its entries (`## ID Title`).
README.md's file list and index are this module's output, never a hand edit:
`python -m nominal_refactor_advisor.audit.catalog` rewrites them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

CATALOG = Path(str(files("nominal_refactor_advisor.audit") / "patterns"))


@dataclass(frozen=True)
class Entry:
    pattern: str
    title: str

    def line(self) -> str:
        return f"- {self.pattern} {self.title}"


@dataclass(frozen=True)
class Rung:
    """One rung file; its reading position is its place in `Catalog.rungs`."""

    path: Path
    TITLE: ClassVar[re.Pattern[str]] = re.compile(r"\A# ([^:\n]+)")
    ENTRY: ClassVar[re.Pattern[str]] = re.compile(r"^## ([A-Z]+-\d+) (.+)$", re.M)

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def title(self) -> str:
        match self.TITLE.match(self.text):
            case re.Match() as found:
                return found[1].removesuffix(" rung")
        raise ValueError(f"{self.path} does not open with its '# Title' line")

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(Entry(found[1], found[2]) for found in self.ENTRY.finditer(self.text))

    @property
    def link(self) -> str:
        return f"[{self.path.name}]({self.path.name})"

    def section(self) -> str:
        return "\n".join((f"**{self.title}** ({self.link})", *(entry.line() for entry in self.entries)))


@dataclass(frozen=True)
class Catalog:
    root: Path = CATALOG
    rungs: ClassVar[tuple[str, ...]] = (
        "implementation.md", "membership.md", "identity.md",
        "boundaries.md", "over-time.md", "agent-defaults.md",
    )
    FILES: ClassVar[re.Pattern[str]] = re.compile(r"^- Files: .*$", re.M)
    INDEX: ClassVar[str] = "## Index of patterns\n"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    def members(self) -> tuple[Rung, ...]:
        return tuple(Rung(self.root / name) for name in self.rungs)

    def files_line(self) -> str:
        return "- Files: " + " · ".join(rung.link for rung in self.members())

    def index(self) -> str:
        return self.INDEX + "\n" + "\n\n".join(rung.section() for rung in self.members()) + "\n"

    def rendered(self) -> str:
        head = self.readme.read_text(encoding="utf-8").partition(self.INDEX)[0]
        return self.FILES.sub(self.files_line(), head) + self.index()

    def write(self) -> None:
        self.readme.write_text(self.rendered(), encoding="utf-8")


if __name__ == "__main__":
    Catalog().write()
