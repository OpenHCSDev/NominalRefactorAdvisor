"""The family root a repository declares for the classes CasesFamily folds an enum into.

The root is the target repository's own declaration, read once from the nearest
``pyproject.toml`` above the enum's module::

    [tool.nominal-refactor-advisor.family-root]
    root = "package.module:Root"      # the class every folded family inherits
    value = "value"                   # the attribute each member class carries
    decoder = "of"                    # the root classmethod that decodes a value
    derived-value = "package.module:words"   # optional, see below

Without ``derived-value``, a member assigns its value as a class attribute. With
it, the root derives each member's value from its class name through that
function, and a member whose old enum value differs passes it as the class
keyword named by ``value``. A repository that declares nothing gets a handback.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from nominal_refactor_advisor.scan_cache import ScanCache

TABLE = "[tool.nominal-refactor-advisor.family-root]"
BLACK_DEFAULT_LINE_LENGTH = 88


@ScanCache.cached
def _nearest_pyproject_for_parent(parent: Path) -> Path | None:
    resolved = parent.resolve()
    for directory in (resolved, *resolved.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


@ScanCache.cached
def _pyproject_table(pyproject: Path) -> dict[str, object]:
    return tomllib.loads(pyproject.read_text())


class ValueCarrier(ABC):
    """How a member class carries the value its enum member had."""

    @abstractmethod
    def member(
        self, attribute: str, class_name: str, value: str, source: str
    ) -> tuple[list[str], list[str]]:
        """The member's header keywords and body lines for ``value``, whose literal
        is spelled ``source``."""


class AssignedValue(ValueCarrier):
    """The member assigns its value as a class attribute."""

    def member(
        self, attribute: str, class_name: str, value: str, source: str
    ) -> tuple[list[str], list[str]]:
        return [], [f"{attribute} = {source}"]


@dataclass(frozen=True)
class DerivedValue(ValueCarrier):
    """The root derives the value from the class name; a differing one is a keyword."""

    derived: Callable[[str], object]

    def member(
        self, attribute: str, class_name: str, value: str, source: str
    ) -> tuple[list[str], list[str]]:
        if self.derived(class_name) == value:
            return [], []
        return [f"{attribute}={source}"], []


@dataclass(frozen=True)
class DerivedSpelling:
    """A family whose members' spellings the repository's own function derives.

    Declared once per family mechanism in the repository's table::

        [tool.nominal-refactor-advisor]
        derived-spellings = [{ family = "package.module:Family",
                               rule = "package.module:words",
                               affix = "affix", spelling = "declared_name" }]

    A root below `family` is a class that inherits it directly or passes the
    `affix` class keyword. A member below a root spells itself by its own
    `spelling` keyword or class constant, else by `rule` applied to its class
    name with the root's affix removed. NRA calls `rule`; it never restates it.
    """

    family: str
    rule: Callable[[str], object]
    affix: str
    spelling: str

    def of(self, name: str, affix: str) -> str:
        """The spelling `rule` derives for a member class named `name`."""

        return str(self.rule(name.removesuffix(affix)))


@dataclass(frozen=True)
class DeclaredRoot:
    """One repository's family root: where it lives and how members carry values."""

    module: str
    name: str
    value: str
    decoder: str
    carrier: ValueCarrier
    line_length: int

    @classmethod
    def declared_for(cls, path: Path) -> DeclaredRoot:
        """The declaration in the nearest ``pyproject.toml`` above ``path``."""

        pyproject = cls.nearest_pyproject(path)
        if pyproject is None:
            raise ValueError(
                f"handback: {path}: declare a family root: no pyproject.toml"
                f" above it to hold {TABLE}"
            )
        declared = tomllib.loads(pyproject.read_text())
        match declared:
            case {"tool": {"nominal-refactor-advisor": {"family-root": dict(table)}}}:
                return cls.decoded(pyproject, table, cls.line_length_of(declared))
        raise ValueError(f"handback: {pyproject}: declare a family root in {TABLE}")

    @staticmethod
    def nearest_pyproject(path: Path) -> Path | None:
        """The closest ``pyproject.toml`` in ``path``'s ancestor directories."""

        if ScanCache.active() is not None and path.is_absolute():
            return _nearest_pyproject_for_parent(path.parent)
        return next(
            (
                directory / "pyproject.toml"
                for directory in path.resolve().parents
                if (directory / "pyproject.toml").is_file()
            ),
            None,
        )

    @classmethod
    def line_length_for(cls, path: Path) -> int:
        """The line length the formatter declared for ``path``'s repository."""

        pyproject = cls.nearest_pyproject(path)
        if pyproject is None:
            return BLACK_DEFAULT_LINE_LENGTH
        table = (
            _pyproject_table(pyproject)
            if ScanCache.active() is not None
            else tomllib.loads(pyproject.read_text())
        )
        return cls.line_length_of(table)

    @classmethod
    def durable_bases_for(cls, path: Path) -> frozenset[str]:
        """The roots of stored state ``path``'s repository declares, as qualified
        class names: rows of a class below one outlive the code that wrote them, so
        a move that changes their keys is a data migration::

            [tool.nominal-refactor-advisor]
            durable-bases = ["package.module:Root", ...]
        """

        return cls.declared_names_for(path, "durable-bases")

    @classmethod
    def composition_roots_for(cls, path: Path) -> frozenset[str]:
        """The composition roots ``path``'s repository declares, as qualified
        class names: each builds its services from the services it hands them,
        and those constructor arguments are the repository's layering::

            [tool.nominal-refactor-advisor]
            composition-roots = ["package.module:Root", ...]
        """

        return cls.declared_names_for(path, "composition-roots")

    @classmethod
    def keyword_binders_for(cls, path: Path) -> frozenset[str]:
        """The functions ``path``'s repository declares build a class handed to
        them only by field name, or never (an encoder, a decoder, a query that
        binds its filters), as qualified names::

            [tool.nominal-refactor-advisor]
            keyword-binders = ["package.module:Codec.decode", ...]
        """

        return cls.declared_names_for(path, "keyword-binders")

    @classmethod
    def declared_names_for(cls, path: Path, key: str) -> frozenset[str]:
        """The qualified names listed under ``key`` in ``path``'s repository's table."""

        pyproject = cls.nearest_pyproject(path)
        if pyproject is None:
            return frozenset()
        table_data = (
            _pyproject_table(pyproject)
            if ScanCache.active() is not None
            else tomllib.loads(pyproject.read_text())
        )
        match table_data:
            case {"tool": {"nominal-refactor-advisor": dict(table)}}:
                names = table.get(key)
                if isinstance(names, list):
                    return frozenset(str(name).replace(":", ".") for name in names)
        return frozenset()

    @classmethod
    def derived_spellings_for(cls, path: Path) -> tuple[DerivedSpelling, ...]:
        """The families ``path``'s repository declares derive their members' spellings."""

        pyproject = cls.nearest_pyproject(path)
        if pyproject is None:
            return ()
        return _derived_spellings_of(pyproject)

    @classmethod
    def first_party_for(cls, path: Path) -> frozenset[str]:
        """The top-level modules ruff's import sort counts as first party for
        ``path``'s repository: everything directly under each ``[tool.ruff] src``
        directory (ruff's default is the project root), plus ``known-first-party``."""

        pyproject = cls.nearest_pyproject(path)
        if pyproject is None:
            return frozenset()
        return _first_party_of(pyproject)

    @staticmethod
    def line_length_of(declared: dict[str, object]) -> int:
        """The line length the repository's formatter declares; Black's default else."""

        match declared:
            case {"tool": {"ruff": {"line-length": int(length)}}}:
                return length
            case {"tool": {"black": {"line-length": int(length)}}}:
                return length
        return BLACK_DEFAULT_LINE_LENGTH

    @classmethod
    def decoded(
        cls, pyproject: Path, table: dict[str, object], line_length: int
    ) -> DeclaredRoot:
        match table:
            case {
                "root": str(root),
                "value": str(value),
                "decoder": str(decoder),
                **rest,
            } if set(rest) <= {"derived-value"}:
                module, name = cls.reference(pyproject, root)
                if "." in name:
                    raise ValueError(
                        f"{pyproject}: {TABLE} root {root!r} must be a top-level class"
                    )
                cls.require_root(pyproject, root, value, decoder)
                return cls(
                    module,
                    name,
                    value,
                    decoder,
                    cls.carrier_of(pyproject, rest.get("derived-value")),
                    line_length,
                )
        raise ValueError(
            f"{pyproject}: {TABLE} takes root, value and decoder strings and an"
            f" optional derived-value; found {sorted(table)}"
        )

    @classmethod
    def carrier_of(cls, pyproject: Path, derived: object) -> ValueCarrier:
        match derived:
            case None:
                return AssignedValue()
            case str(reference):
                found = cls.imported(pyproject, "derived-value", reference)
                if not callable(found):
                    raise ValueError(
                        f"{pyproject}: {TABLE} derived-value {reference!r} is not callable"
                    )
                return DerivedValue(found)
        raise ValueError(f"{pyproject}: {TABLE} derived-value must be a string")

    @classmethod
    def require_root(
        cls, pyproject: Path, reference: str, value: str, decoder: str
    ) -> None:
        """The root is a class carrying ``decoder`` as a classmethod, and ``value``
        is declared on its MRO or its metaclass's: the two names every rewritten
        read relies on."""

        root = cls.imported(pyproject, "root", reference)
        if not isinstance(root, type):
            raise ValueError(f"{pyproject}: {TABLE} root {reference!r} is not a class")
        if not any(
            value in vars(declaring) or value in inspect.get_annotations(declaring)
            for declaring in (*root.__mro__, *type(root).__mro__)
        ):
            raise ValueError(
                f"{pyproject}: {TABLE} value {value!r} is not declared on {reference}"
            )
        if not isinstance(
            next(
                (
                    vars(declaring)[decoder]
                    for declaring in root.__mro__
                    if decoder in vars(declaring)
                ),
                None,
            ),
            classmethod,
        ):
            raise ValueError(
                f"{pyproject}: {TABLE} decoder {decoder!r} is not a classmethod"
                f" of {reference}"
            )

    @classmethod
    def imported(cls, pyproject: Path, key: str, reference: str) -> object:
        module, qualname = cls.reference(pyproject, reference)
        try:
            found: object = importlib.import_module(module)
            for part in qualname.split("."):
                found = getattr(found, part)
        except (ImportError, AttributeError) as missing:
            raise ValueError(
                f"{pyproject}: {TABLE} {key} {reference!r} is not importable here:"
                f" {missing}"
            ) from None
        return found

    @staticmethod
    def reference(pyproject: Path, reference: str) -> tuple[str, str]:
        module, separator, qualname = reference.partition(":")
        if not (module and separator and qualname):
            raise ValueError(
                f"{pyproject}: {TABLE} expects module:qualname, got {reference!r}"
            )
        return module, qualname


@contextmanager
def _repository_importable(pyproject: Path) -> Iterator[None]:
    """The repository's own sources first on the import path: the rule a scanned
    repository declares is its source's, installed or not."""

    roots = [
        str(directory)
        for directory in (pyproject.parent / "src", pyproject.parent)
        if directory.is_dir()
    ]
    sys.path[:0] = roots
    try:
        yield
    finally:
        del sys.path[: len(roots)]


DERIVED_SPELLINGS = "derived-spellings"


@cache
def _derived_spellings_of(pyproject: Path) -> tuple[DerivedSpelling, ...]:
    match tomllib.loads(pyproject.read_text()):
        case {"tool": {"nominal-refactor-advisor": {"derived-spellings": list(entries)}}}:
            return tuple(_derived_spelling(pyproject, entry) for entry in entries)
    return ()


def _derived_spelling(pyproject: Path, entry: object) -> DerivedSpelling:
    match entry:
        case {
            "family": str(family),
            "rule": str(rule),
            "affix": str(affix),
            "spelling": str(spelling),
        }:
            with _repository_importable(pyproject):
                found = DeclaredRoot.imported(pyproject, DERIVED_SPELLINGS, rule)
            if not callable(found):
                raise ValueError(
                    f"{pyproject}: {DERIVED_SPELLINGS} rule {rule!r} is not callable"
                )
            module, name = DeclaredRoot.reference(pyproject, family)
            return DerivedSpelling(f"{module}.{name}", found, affix, spelling)
    raise ValueError(
        f"{pyproject}: each {DERIVED_SPELLINGS} entry takes family, rule, affix and"
        f" spelling strings; found {entry!r}"
    )


@cache
def _first_party_of(pyproject: Path) -> frozenset[str]:
    declared = tomllib.loads(pyproject.read_text())
    ruff = declared.get("tool", {}).get("ruff", {})
    isort = {**ruff.get("isort", {}), **ruff.get("lint", {}).get("isort", {})}
    names = set(isort.get("known-first-party", ()))
    for source_dir in ruff.get("src", ["."]):
        directory = pyproject.parent / source_dir
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir() and entry.name.isidentifier():
                names.add(entry.name)
            elif entry.suffix == ".py" and entry.stem.isidentifier():
                names.add(entry.stem)
    return frozenset(names)
