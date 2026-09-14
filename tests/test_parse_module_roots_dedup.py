"""Overlapping parse roots retain first-root identity without repeated parsing."""

from collections import Counter
from pathlib import Path

import pytest

from nominal_refactor_advisor import ast_tools


def _write_source(path: Path, source: str = "class Owner: pass\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="")


def test_overlapping_roots_parse_each_admitted_source_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    _write_source(package / "__init__.py", "")
    _write_source(package / "alpha.py")
    _write_source(package / "nested" / "__init__.py", "")
    _write_source(package / "nested" / "beta.py")

    parsed_paths: list[Path] = []
    original_parse = ast_tools._parse_source_module

    def counted_parse(path: Path, *, context, source_semantic_hash=None):
        parsed_paths.append(path.resolve())
        return original_parse(
            path,
            context=context,
            source_semantic_hash=source_semantic_hash,
        )

    monkeypatch.setattr(ast_tools, "_parse_source_module", counted_parse)
    modules = ast_tools.parse_python_module_roots(
        (package, package / "nested", package / "alpha.py"),
        use_parse_cache=False,
        parse_workers=1,
    )

    assert [module.path for module in modules] == [
        package / "__init__.py",
        package / "alpha.py",
        package / "nested" / "__init__.py",
        package / "nested" / "beta.py",
    ]
    assert [module.module_name for module in modules] == [
        "package",
        "package.alpha",
        "package.nested",
        "package.nested.beta",
    ]
    assert Counter(parsed_paths) == Counter(module.path.resolve() for module in modules)


def test_duplicate_symlink_keeps_first_admitted_path_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "real.py"
    alias = tmp_path / "alias.py"
    _write_source(source)
    try:
        alias.symlink_to(source)
    except OSError as error:
        pytest.skip(f"source symlink creation unavailable: {error}")

    parsed_paths: list[Path] = []
    original_parse = ast_tools._parse_source_module

    def counted_parse(path: Path, *, context, source_semantic_hash=None):
        parsed_paths.append(path)
        return original_parse(
            path,
            context=context,
            source_semantic_hash=source_semantic_hash,
        )

    monkeypatch.setattr(ast_tools, "_parse_source_module", counted_parse)
    modules = ast_tools.parse_python_module_roots(
        (alias, source),
        use_parse_cache=False,
        parse_workers=1,
    )

    assert len(modules) == 1
    assert modules[0].path == alias
    assert modules[0].module_name == "real"
    assert parsed_paths == [alias]


def test_duplicate_file_root_keeps_first_parser_root_module_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "package" / "shared.py"
    _write_source(source)

    modules = ast_tools.parse_python_module_roots(
        (tmp_path, source),
        use_parse_cache=False,
        parse_workers=1,
    )

    assert len(modules) == 1
    assert modules[0].path == source
    assert modules[0].module_name == "package.shared"
