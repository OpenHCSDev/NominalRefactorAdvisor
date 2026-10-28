"""The class-body index: which classes make the same declarations (#68).

Two declarations are the same when their kind, name, syntax and the binding of
every free name agree; a name bound only by a star import is settled by the
repository's export contracts.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_body import ClassBodyIndex
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _index(*modules: tuple[str, str]) -> ClassBodyIndex:
    return ClassBodyIndex.from_projections(
        CompactModuleClassProjectionFamily.collect_modules(
            tuple(_module(name, source) for name, source in modules)
        )
    )


RECORDS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Task:\n"
    "    id: int\n    name: str\n\n"
    "    def label(self) -> str:\n        '''Task label.'''\n        return f'{self.id}:{self.name}'\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Job:\n"
    "    id: int\n    name: str\n\n"
    "    def label(self) -> str:\n        return f'{self.id}:{self.name}'\n"
)


def test_equal_fields_and_methods_are_shared_and_docstrings_do_not_count():
    overlap = _index(("pkg.records", RECORDS)).overlap(
        "pkg.records.Job", "pkg.records.Task"
    )
    assert [shared.name for shared in overlap.shared] == ["id", "name", "label"]
    assert overlap.first_is_contained and overlap.second_is_contained


def test_a_name_bound_to_a_different_declaration_diverges():
    index = _index(
        ("pkg.a", "from pkg.units import Meters as Length\n\nclass A:\n    size: Length\n"),
        ("pkg.b", "from pkg.units import Feet as Length\n\nclass B:\n    size: Length\n"),
    )
    overlap = index.overlap("pkg.a.A", "pkg.b.B")
    assert not overlap.shared
    assert [divergent.name for divergent in overlap.divergent] == ["size"]


def test_a_method_body_reads_the_module_binding_defined_after_the_class():
    source = (
        "class A:\n    def run(self):\n        return helper()\n\n\n"
        "class B:\n    def run(self):\n        return helper()\n\n\n"
        "def helper():\n    return 1\n"
    )
    overlap = _index(("pkg.late", source)).overlap("pkg.late.A", "pkg.late.B")
    assert [shared.name for shared in overlap.shared] == ["run"]


def test_a_star_import_that_does_not_export_a_name_keeps_the_local_binding():
    index = _index(
        ("pkg.base", "__all__ = ('Other',)\n\nclass Other:\n    pass\n"),
        (
            "pkg.users",
            "from typing import ClassVar\nfrom pkg.base import *\n\n"
            "class A:\n    tag: ClassVar[str] = 'a'\n\n"
            "class B:\n    tag: ClassVar[str] = 'a'\n",
        ),
    )
    overlap = index.overlap("pkg.users.A", "pkg.users.B")
    assert [shared.name for shared in overlap.shared] == ["tag"]


def test_a_star_import_with_unknown_exports_leaves_its_names_unbound():
    index = _index(
        ("pkg.base", "__all__ = tuple(globals())\n"),
        (
            "pkg.users",
            "from typing import ClassVar\nfrom pkg.base import *\n\n"
            "class A:\n    tag: ClassVar[str] = 'a'\n\n"
            "class B:\n    tag: ClassVar[str] = 'a'\n",
        ),
    )
    overlap = index.overlap("pkg.users.A", "pkg.users.B")
    assert not overlap.shared
    assert [divergent.name for divergent in overlap.divergent] == ["tag"]


def test_overlaps_pair_classes_by_any_shared_declaration():
    index = _index(("pkg.records", RECORDS + "\n\nclass Unrelated:\n    other: int\n"))
    assert [
        (overlap.first.qualname, overlap.second.qualname) for overlap in index.overlaps
    ] == [("Job", "Task")]
