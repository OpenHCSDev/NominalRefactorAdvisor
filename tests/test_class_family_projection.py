"""A projected snapshot's class-family index is a fresh build's, from facts unchanged parses already hold."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModuleSet, parse_python_modules
from nominal_refactor_advisor.class_families.parsed_index import (
    ModuleIndexedClasses,
    build_class_family_index,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot


def _write(root: Path, files: dict[str, str]) -> None:
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


BEFORE = {
    "pkg/__init__.py": "",
    "pkg/sub/__init__.py": "",
    "pkg/sub/base.py": "",
    "pkg/base.py": "class Base:\n    pass\n\n\nclass Other:\n    pass\n",
    "pkg/shim.py": "from .base import Base\n",
    "pkg/child.py": "from .shim import Base\n\n\nclass Child(Base):\n    pass\n",
    "pkg/direct.py": "from pkg.base import Base\n\n\nclass Direct(Base):\n    pass\n",
    "pkg/bystander.py": "from .base import Other\n\n\nclass Bystander(Other):\n    pass\n",
}
RELOCATED = {
    "pkg/base.py": "class Other:\n    pass\n",
    "pkg/sub/base.py": "class Base:\n    pass\n",
    "pkg/shim.py": "from .sub.base import Base\n",
}


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    _write(root, BEFORE)
    return CodemodSourceSnapshot.from_modules(
        ParsedModuleSet.of(parse_python_modules(root, use_parse_cache=False))
    )


def _bystander(snapshot: CodemodSourceSnapshot):
    return next(
        module
        for module in snapshot.parsed_modules
        if module.module_name == "pkg.bystander"
    )


def test_relocated_class_projects_the_fresh_build_index(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    bystander_records = _bystander(snapshot).fact(ModuleIndexedClasses)

    projected = snapshot.with_virtual_sources(
        {(tmp_path / path).as_posix(): source for path, source in RELOCATED.items()}
    )

    assert (
        projected.required_class_family_index.classes_by_symbol
        == build_class_family_index(projected.parsed_modules).classes_by_symbol
    )
    assert "pkg.base.Base" in snapshot.required_class_family_index.ancestor_symbols(
        "pkg.direct.Direct"
    )
    assert "pkg.base.Base" not in projected.required_class_family_index.ancestor_symbols(
        "pkg.direct.Direct"
    )
    assert _bystander(projected).fact(ModuleIndexedClasses) is bystander_records


def test_deleted_module_projects_the_fresh_build_index(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    bystander_records = _bystander(snapshot).fact(ModuleIndexedClasses)

    projected = snapshot.without_source_paths(
        ((tmp_path / "pkg/direct.py").as_posix(),)
    )

    assert "pkg.direct.Direct" not in projected.required_class_family_index.classes_by_symbol
    assert (
        projected.required_class_family_index.classes_by_symbol
        == build_class_family_index(projected.parsed_modules).classes_by_symbol
    )
    assert projected.source_index == CodemodSourceSnapshot.from_modules(
        ParsedModuleSet.of(projected.parsed_modules)
    ).source_index
    assert _bystander(projected).fact(ModuleIndexedClasses) is bystander_records
