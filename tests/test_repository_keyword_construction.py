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
from pathlib import Path

import nominal_refactor_advisor

ROOT = Path(nominal_refactor_advisor.__file__).resolve().parent.parent


def _field_ordered_dataclass_names() -> frozenset[str]:
    """Names of package dataclasses whose constructor is their init fields, by unique name."""

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
    return frozenset(
        cls.__name__
        for cls in classes
        if counts[cls.__name__] == 1
        and [
            parameter.name
            for parameter in inspect.signature(cls).parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        == [field.name for field in dataclasses.fields(cls) if field.init]
    )


def test_repository_dataclasses_are_constructed_by_keyword() -> None:
    names = _field_ordered_dataclass_names()
    positional = sorted(
        f"{path.relative_to(ROOT)}:{node.lineno} {name}(...) passes {len(node.args)} by position"
        for directory in ("nominal_refactor_advisor", "tests")
        for path in (ROOT / directory).rglob("*.py")
        if "fixtures" not in path.relative_to(ROOT).parts
        for node in ast.walk(ast.parse(path.read_text()))
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
        if name in names
    )
    assert not positional, "\n".join(positional)
