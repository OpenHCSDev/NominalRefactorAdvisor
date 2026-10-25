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
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TABLE = "[tool.nominal-refactor-advisor.family-root]"
BLACK_DEFAULT_LINE_LENGTH = 88


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

        pyproject = next(
            (
                directory / "pyproject.toml"
                for directory in path.resolve().parents
                if (directory / "pyproject.toml").is_file()
            ),
            None,
        )
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
