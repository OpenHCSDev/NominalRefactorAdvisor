"""Package dataclasses are built by keyword.

Moving a field into a parent reorders the constructor; a call that passes two
or more values by position then binds them to other fields, silently when
they share a type (``ModuleImportInsertionPoint(source, file_path)`` did).
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import nominal_refactor_advisor

ROOT = Path(nominal_refactor_advisor.__file__).resolve().parent.parent


def _field_ordered_dataclass_names() -> dict[str, int]:
    """Positional width of each uniquely named package dataclass whose positional
    parameters are exactly its positional init fields."""

    for info in pkgutil.walk_packages(
        nominal_refactor_advisor.__path__, f"{nominal_refactor_advisor.__name__}."
    ):
        importlib.import_module(info.name)
    classes = [
        cls
        for module_name, module in sorted(sys.modules.items())
        if module_name.startswith(nominal_refactor_advisor.__name__)
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module_name and dataclasses.is_dataclass(cls)
    ]
    counts = Counter(cls.__name__ for cls in classes)
    return dict(
        (cls.__name__, len(positional))
        for cls in classes
        if counts[cls.__name__] == 1
        for positional in (
            [
                parameter.name
                for parameter in inspect.signature(cls).parameters.values()
                if parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ],
        )
        if positional
        == [
            field.name
            for field in dataclasses.fields(cls)
            if field.init and not field.kw_only
        ]
    )


def _trees(path: Path) -> Iterator[tuple[int, ast.AST]]:
    """The file's tree, then every string constant that is a program importing
    the package. Fixture programs NRA only reads declare their own classes.

    A subprocess test carries its program as a string, out of reach of the
    file's own tree; the 2026-11-16 kw-only `module` hoist broke three of them.
    """

    tree = ast.parse(path.read_text())
    yield 0, tree
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and nominal_refactor_advisor.__name__ in node.value
        ):
            try:
                yield node.lineno - 1, ast.parse(node.value)
            except SyntaxError:
                continue


def test_repository_dataclasses_are_constructed_by_keyword() -> None:
    widths = _field_ordered_dataclass_names()
    positional = sorted(
        f"{path.relative_to(ROOT)}:{offset + node.lineno} {name}(...) passes {len(node.args)} by position"
        for directory in ("nominal_refactor_advisor", "tests")
        for path in (ROOT / directory).rglob("*.py")
        if "fixtures" not in path.relative_to(ROOT).parts
        for offset, tree in _trees(path)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and len(node.args) >= 2
        and not any(isinstance(argument, ast.Starred) for argument in node.args)
        for name in (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None,
        )
        if 2 <= len(node.args) <= widths.get(name, 0)
    )
    assert not positional, "\n".join(positional)
