"""A rewritten from-import keeps the layout of the statement it rewrites."""

import ast

from nominal_refactor_advisor.codemod_imports import ImportFromSource, ImportLayout


def _statement(source: str) -> ast.ImportFrom:
    statement = ast.parse(source).body[0]
    assert isinstance(statement, ast.ImportFrom)
    return statement


def _render(statement: ast.ImportFrom, layout: ImportLayout, module: str = "pkg.b") -> str:
    return ImportFromSource(module_name=module, aliases=tuple(statement.names), layout=layout).source


def test_a_one_line_import_stays_one_line_while_it_fits(tmp_path) -> None:
    statement = _statement("from pkg.a import A, B\n")
    layout = ImportLayout.of(statement, str(tmp_path / "m.py"))
    assert _render(statement, layout) == "from pkg.b import A, B\n"


def test_a_one_line_import_past_the_declared_limit_becomes_a_column(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nline-length = 20\n')
    statement = _statement("from pkg.a import Alpha, Beta\n")
    layout = ImportLayout.of(statement, str(tmp_path / "m.py"))
    assert _render(statement, layout) == "from pkg.b import (\n    Alpha,\n    Beta,\n)\n"


def test_a_parenthesised_import_stays_a_column(tmp_path) -> None:
    statement = _statement("from pkg.a import (\n    A,\n    B,\n)\n")
    layout = ImportLayout.of(statement, str(tmp_path / "m.py"))
    assert _render(statement, layout) == "from pkg.b import (\n    A,\n    B,\n)\n"


def test_an_import_rendered_again_as_written_is_never_wrapped() -> None:
    names = ", ".join(f"Name{index}" for index in range(30))
    statement = _statement(f"from pkg.a import {names}\n")
    assert _render(statement, ImportLayout.as_written(statement)) == f"from pkg.b import {names}\n"
