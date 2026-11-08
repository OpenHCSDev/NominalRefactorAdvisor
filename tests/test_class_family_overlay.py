"""A projected class-family index is the full rebuild's, rebuilt only where it can change."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_families import parsed_index
from nominal_refactor_advisor.class_families.parsed_index import (
    build_class_family_index,
)


def _write(root: Path, files: dict[str, str]) -> None:
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


BEFORE = {
    "pkg/__init__.py": "",
    "pkg/sub/__init__.py": "",
    "pkg/base.py": "class Base:\n    pass\n\n\nclass Other:\n    pass\n",
    "pkg/shim.py": "from .base import Base\n",
    "pkg/child.py": "from .shim import Base\n\n\nclass Child(Base):\n    pass\n",
    "pkg/direct.py": "from pkg.base import Base\n\n\nclass Direct(Base):\n    pass\n",
    "pkg/bystander.py": "from .base import Other\n\n\nclass Bystander(Other):\n    pass\n",
}
AFTER = {
    "pkg/base.py": "class Other:\n    pass\n",
    "pkg/sub/base.py": "class Base:\n    pass\n",
    "pkg/shim.py": "from .sub.base import Base\n",
}


def test_relocated_symbol_overlay_equals_full_rebuild(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, BEFORE)
    before = tuple(parse_python_modules(tmp_path))
    _write(tmp_path, AFTER)
    changed_paths = {(tmp_path / path).as_posix() for path in AFTER}
    changed = tuple(
        module
        for module in parse_python_modules(tmp_path)
        if module.file_path in changed_paths
    )
    # A projection carries every unchanged module over as the same parse.
    after = (
        *(module for module in before if module.file_path not in changed_paths),
        *changed,
    )

    base_index = build_class_family_index(list(before))
    rebuilt: list[str] = []
    builder = parsed_index.ClassFamilyIndexBuilder
    original = builder.module_class_records

    def recorded(self):
        rebuilt.extend(module.module_name for module in self.modules)
        return original(self)

    monkeypatch.setattr(builder, "module_class_records", recorded)
    overlay = base_index.projected_with_module_overlay(
        after, changed
    )
    overlaid_modules = sorted(rebuilt)
    rebuilt.clear()
    full = build_class_family_index(list(after))

    assert overlay.classes_by_symbol == full.classes_by_symbol
    assert "pkg.bystander" not in overlaid_modules
    assert {"pkg.child", "pkg.direct"} <= set(overlaid_modules)
