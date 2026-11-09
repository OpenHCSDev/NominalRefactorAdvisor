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


def test_a_requested_import_merged_past_the_limit_becomes_a_column(tmp_path) -> None:
    """A merge can push a one-line import past the file's limit; it is then wrapped,
    never left as an overlong line."""

    from nominal_refactor_advisor.codemod_imports import RequestedImportStatement
    from nominal_refactor_advisor.lexical_bindings import ImportAlias

    (tmp_path / "pyproject.toml").write_text('[tool.ruff]\nline-length = 40\n')
    requested = RequestedImportStatement(_statement("from pkg.a import Alpha\n")).with_aliases(
        ImportAlias.from_alias(ast.alias(name)) for name in ("Alpha", "Beta", "Gamma", "Delta")
    )
    rendered = requested.source_in((tmp_path / "m.py").as_posix())
    assert rendered == "from pkg.a import (\n    Alpha,\n    Beta,\n    Gamma,\n    Delta,\n)\n"
    assert all(len(line) <= 40 for line in rendered.splitlines())


def test_an_import_inside_a_string_stays_one_line() -> None:
    names = ", ".join(f"Name{index}" for index in range(30))
    statement = _statement(f"from pkg.a import {names}\n")
    assert _render(statement, ImportLayout.in_string()) == f"from pkg.b import {names}\n"
