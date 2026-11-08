"""Module identity projections reuse the parse's node inventory without changing answers."""

import ast
from pathlib import Path
from unittest.mock import patch

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.class_identity_observations import (
    DeclaredIdentityFamily,
    NameBindings,
    QualifiedNames,
)


def subject(tmp_path: Path, source: str) -> ParsedModule:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / "subject.py"
    target.write_text(source, encoding="utf-8")
    module = next(
        parsed for parsed in parse_python_modules(tmp_path, use_parse_cache=False)
        if Path(parsed.file_path) == target
    )
    assert module.module_name == "pkg.subject"
    index = module.syntax_index
    assert index.breadth_first_nodes == tuple(ast.walk(module.module))
    module.fact(QualifiedNames)
    return module


def test_identity_projections_preserve_binding_import_and_rename_answers(tmp_path: Path) -> None:
    module = subject(
        tmp_path,
        "from .vendor import Original as Unique, Shadowed\n"
        "import package.module as mod\n"
        "class First: pass\n"
        "class Twin: pass\n"
        "class Twin: pass\n"
        "def f(Shadowed: Unique) -> Unique:\n"
        "    import inner\n"
        "    from .nested import Child\n"
        "    class Hidden: pass\n"
        "    local: Unique\n"
        "    return Shadowed\n"
        "First.__name__ = 'Renamed'\n"
        "setattr(mod.Factory, '__name__', 'Other')\n",
    )
    with patch("ast.walk", wraps=ast.walk) as walks:
        bindings = module.fact(NameBindings)
        (declarations,) = DeclaredIdentityFamily.collect(module)

    assert bindings.bound == (
        ("First", 2, "pkg.subject.First"),
        ("Unique", 0, "pkg.vendor.Original"),
    )
    assert bindings.named == frozenset({
        "Unique", "Shadowed", "mod", "First", "Twin", "f",
        "Child", "Hidden", "local", "setattr",
    })
    # bound_names excludes unaliased module spellings; imported below retains them.
    assert not bindings.starred
    # declared_name retains the annotation's bare spelling, unlike qualified imports.
    assert bindings.typed == frozenset({"Unique"})
    assert bindings.reference(2, "First") == ("", "")
    assert bindings.reference(3, "First") == ("pkg.subject.First", "")
    assert bindings.reference(3, "Shadowed") == ("", "")
    assert bindings.reference(3, "Missing") == ("", "3")
    assert declarations.renamed == ("package.module.Factory", "pkg.subject.First")
    assert declarations.imported == (
        ("inner", ()),
        ("package.module", ()),
        ("pkg.nested", ("Child",)),
        ("pkg.vendor", ("Original", "Shadowed")),
    )
    assert [(item.name, item.line, item.top_level) for item in declarations.classes] == [
        ("First", 3, True), ("Twin", 4, True), ("Twin", 5, True), ("Hidden", 9, False),
    ]
    assert sum(call.args[0] is module.module for call in walks.call_args_list) == 0


def test_star_imports_keep_their_unknown_binding_contract(tmp_path: Path) -> None:
    module = subject(tmp_path, "from .vendor import *\nclass Named: pass\n")
    with patch("ast.walk", wraps=ast.walk) as walks:
        bindings = module.fact(NameBindings)
        (declarations,) = DeclaredIdentityFamily.collect(module)

    assert bindings.starred
    assert bindings.named == frozenset({"Named"})
    assert bindings.typed == frozenset()
    assert bindings.reference(2, "Named") == ("", "")
    assert bindings.reference(2, "Missing") == ("", "")
    assert declarations.imported == (("pkg.vendor", ("*",)),)
    assert declarations.renamed == ()
    assert sum(call.args[0] is module.module for call in walks.call_args_list) == 0
