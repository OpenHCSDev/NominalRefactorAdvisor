"""A module's class declarations are a fact of its parse: a later scan of the
same parse, a goal run's next state for instance, reads them without
redeclaring the module."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassHeader
from nominal_refactor_advisor.scan_cache import ScanCache


def test_a_second_scan_of_the_same_parses_reuses_each_modules_declarations(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/a.py").write_text("class A:\n    def run(self):\n        return 1\n")
    (tmp_path / "pkg/b.py").write_text("from pkg.a import A\n\n\nclass B(A):\n    pass\n")
    modules = tuple(parse_python_modules(tmp_path))

    with ScanCache.scope():
        first = {header.module_name: header for header in CompactModuleClassHeader.from_modules(modules)}
    with ScanCache.scope():
        again = tuple(CompactModuleClassHeader.from_module(module) for module in modules)

    assert {header.module_name for header in again} == set(first)
    assert all(
        CompactModuleClassHeader.from_module(module) is header
        for module, header in zip(modules, again, strict=True)
    )
