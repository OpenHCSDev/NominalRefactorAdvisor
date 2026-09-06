"""Import requests retain their source declaration and selected binding semantics."""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from nominal_refactor_advisor.codemod_imports import (
    ImportAliasRequirement as CodemodImportAliasRequirement,
    ImportFromModuleName as CodemodImportFromModuleName,
)
from nominal_refactor_advisor.lexical_bindings import (
    FromImportDeclaration,
    ImportAliasRequirement,
    ImportBoundNameProjection,
    ImportDeclarationABC,
    ImportFromModuleName,
    ModuleImportDeclaration,
)
from nominal_refactor_advisor.python_module_identity import PythonModulePathIdentity


def _projection(source: str) -> ImportBoundNameProjection:
    statement = ast.parse(source).body[0]
    assert isinstance(statement, (ast.Import, ast.ImportFrom))
    return ImportBoundNameProjection(statement)


def _identity(
    name: str = "pkg.mod", *, is_package_init: bool = False
) -> PythonModulePathIdentity:
    parts = name.split(".")
    path = (
        Path(*parts, "__init__.py")
        if is_package_init
        else Path(*parts[:-1], f"{parts[-1]}.py")
    )
    return PythonModulePathIdentity(
        path=path, import_name=name, is_package_init=is_package_init
    )


def _assert_ast_free(value: object) -> None:
    assert not isinstance(value, ast.AST)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_ast_free(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_ast_free(item)


@pytest.mark.parametrize(
    "source,declaration_type,local_name,requested_module,qualified",
    (
        ("import pkg.child", ModuleImportDeclaration, "pkg", "pkg.child", "pkg"),
        (
            "import pkg.child as pkg",
            ModuleImportDeclaration,
            "pkg",
            "pkg.child",
            "pkg.child",
        ),
        (
            "import pkg.child as child",
            ModuleImportDeclaration,
            "child",
            "pkg.child",
            "pkg.child",
        ),
        (
            "from pkg import child",
            FromImportDeclaration,
            "child",
            "pkg",
            "pkg.child",
        ),
        (
            "from pkg import child as child",
            FromImportDeclaration,
            "child",
            "pkg",
            "pkg.child",
        ),
        (
            "from pkg import child as selected",
            FromImportDeclaration,
            "selected",
            "pkg",
            "pkg.child",
        ),
    ),
)
def test_binding_and_request_derive_from_the_selected_declaration(
    source: str,
    declaration_type: type[ImportDeclarationABC],
    local_name: str,
    requested_module: str,
    qualified: str,
) -> None:
    projection = _projection(source)
    declaration = projection.declaration
    identity = _identity()
    (origin,) = projection.origins(identity)

    assert isinstance(declaration, declaration_type)
    assert projection.declaration is declaration
    assert origin.declaration is declaration
    assert origin.alias is declaration.aliases[0]
    assert origin.module_identity is identity
    assert origin.bound_name == local_name
    assert origin.requested_module_name == requested_module
    assert origin.qualified_name == qualified
    assert projection.names() == (local_name,)
    assert projection.name_sources() == ((local_name, origin.source),)
    assert declaration.origins(identity) == (origin,)
    assert ast.dump(ast.parse(declaration.source)) == ast.dump(ast.parse(source))
    assert ast.dump(ast.parse(origin.source)) == ast.dump(ast.parse(source))
    _assert_ast_free(declaration)
    _assert_ast_free(origin)


@pytest.mark.parametrize(
    "source,module_name,is_package,requested_module,qualified",
    (
        ("from . import child", "pkg.mod", False, "pkg", "pkg.child"),
        ("from . import child", "pkg", True, "pkg", "pkg.child"),
        (
            "from .library import child",
            "pkg.mod",
            False,
            "pkg.library",
            "pkg.library.child",
        ),
        (
            "from ..library import child",
            "pkg.nested.mod",
            False,
            "pkg.library",
            "pkg.library.child",
        ),
        ("from .. import child", "pkg.mod", False, None, None),
        ("from .. import child", "pkg", True, None, None),
        ("from . import child", "standalone", False, None, None),
        ("from ...library import child", "pkg.mod", False, None, None),
    ),
)
def test_relative_request_retains_source_even_when_resolution_is_unavailable(
    source: str,
    module_name: str,
    is_package: bool,
    requested_module: str | None,
    qualified: str | None,
) -> None:
    projection = _projection(source)
    identity = _identity(module_name, is_package_init=is_package)
    (origin,) = projection.origins(identity)

    assert isinstance(origin.declaration, FromImportDeclaration)
    assert origin.module_identity is identity
    assert origin.requested_module_name == requested_module
    assert origin.qualified_name == qualified
    assert origin.bound_name == "child"
    assert origin.declaration.module_name.source == source.split()[1]
    assert ast.dump(ast.parse(origin.source)) == ast.dump(ast.parse(source))
    assert projection.names() == ("child",)


@pytest.mark.parametrize(
    "source,names,requests",
    (
        (
            "import pkg.first as same, pkg.second as same",
            ("same", "same"),
            ("pkg.first", "pkg.second"),
        ),
        (
            "from pkg import first as same, second as same",
            ("same", "same"),
            ("pkg", "pkg"),
        ),
    ),
)
def test_alias_order_and_shared_declaration_survive_duplicate_local_names(
    source: str, names: tuple[str, ...], requests: tuple[str, ...]
) -> None:
    projection = _projection(source)
    identity = _identity()
    declaration = projection.declaration
    origins = projection.origins(identity)

    assert len(origins) == len(declaration.aliases) == 2
    assert projection.names() == names
    assert tuple(origin.requested_module_name for origin in origins) == requests
    assert all(origin.declaration is declaration for origin in origins)
    assert all(origin.module_identity is identity for origin in origins)
    assert all(
        origin.alias is alias
        for origin, alias in zip(origins, declaration.aliases, strict=True)
    )
    assert projection.name_sources() == tuple(
        (origin.bound_name, origin.source) for origin in origins
    )
    assert ast.dump(ast.parse(declaration.source)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("source", ("from pkg import *", "from . import *"))
def test_star_request_is_retained_without_inventing_explicit_bindings(
    source: str,
) -> None:
    projection = _projection(source)
    declaration = projection.declaration

    assert isinstance(declaration, FromImportDeclaration)
    assert len(declaration.aliases) == 1
    assert declaration.aliases[0].name == "*"
    assert projection.names() == ()
    assert projection.origins(_identity()) == ()
    assert declaration.origins(_identity()) == ()
    assert projection.name_sources() == ()
    assert ast.dump(ast.parse(declaration.source)) == ast.dump(ast.parse(source))
    _assert_ast_free(declaration)


@pytest.mark.parametrize("alias_index", (-1, 2))
def test_selected_alias_rejects_indices_outside_its_declaration(
    alias_index: int,
) -> None:
    projection = _projection("from pkg import child, other")
    origin, _ = projection.origins(_identity())

    with pytest.raises(ValueError):
        replace(origin, alias_index=alias_index)


def test_selected_alias_is_derived_from_its_source_occurrence() -> None:
    projection = _projection("from pkg import child, other")
    declaration = projection.declaration
    first, second = projection.origins(_identity())

    assert first.alias_index == 0
    assert second.alias_index == 1
    assert first.alias is declaration.aliases[0]
    assert second.alias is declaration.aliases[1]
    selected = replace(first, alias_index=1)
    assert selected.alias is second.alias
    assert selected.bound_name == "other"
    assert selected.qualified_name == "pkg.other"


def test_compact_import_pickle_retains_shared_evidence_and_no_ast() -> None:
    projection = _projection("from .library import first as local, second")
    declaration = projection.declaration
    origins = projection.origins(_identity())
    payload = declaration, origins
    _assert_ast_free(payload)

    restored_declaration, restored_origins = pickle.loads(pickle.dumps(payload))

    assert (restored_declaration, restored_origins) == payload
    assert all(
        origin.declaration is restored_declaration for origin in restored_origins
    )
    assert all(
        origin.alias is alias
        for origin, alias in zip(
            restored_origins, restored_declaration.aliases, strict=True
        )
    )
    assert restored_origins[0].module_identity is restored_origins[1].module_identity
    _assert_ast_free((restored_declaration, restored_origins))


def test_codemod_import_syntax_reexports_the_same_lower_declarations() -> None:
    assert CodemodImportAliasRequirement is ImportAliasRequirement
    assert CodemodImportFromModuleName is ImportFromModuleName


@pytest.fixture
def runtime_package(tmp_path: Path) -> Path:
    package = tmp_path / "nra_import_fixture"
    package.mkdir()
    (package / "__init__.py").write_text(
        "class Marker: pass\nchild = Marker()\n", encoding="utf-8"
    )
    (package / "child.py").write_text("value = 7\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "source,bound_name,expected_type,expected_module,child_loaded",
    (
        (
            "import nra_import_fixture.child",
            "nra_import_fixture",
            "module",
            "nra_import_fixture",
            True,
        ),
        (
            "import nra_import_fixture.child as nra_import_fixture",
            "nra_import_fixture",
            "module",
            "nra_import_fixture.child",
            True,
        ),
        (
            "from nra_import_fixture import child",
            "child",
            "Marker",
            None,
            False,
        ),
        (
            "import nra_import_fixture.child as child",
            "child",
            "module",
            "nra_import_fixture.child",
            True,
        ),
    ),
)
def test_python_runtime_distinguishes_requested_module_and_bound_object(
    runtime_package: Path,
    source: str,
    bound_name: str,
    expected_type: str,
    expected_module: str | None,
    child_loaded: bool,
) -> None:
    program = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "namespace = {}\n"
        "exec(sys.argv[2], namespace)\n"
        "value = namespace[sys.argv[3]]\n"
        "print(json.dumps([type(value).__name__, getattr(value, '__name__', None), "
        "'nra_import_fixture.child' in sys.modules]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(runtime_package), source, bound_name],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(completed.stdout) == [
        expected_type,
        expected_module,
        child_loaded,
    ]

    projection = _projection(source)
    (origin,) = projection.origins(_identity())
    assert origin.bound_name == bound_name
    if isinstance(projection.declaration, ModuleImportDeclaration):
        assert origin.requested_module_name == "nra_import_fixture.child"
        assert origin.qualified_name == expected_module
    else:
        assert isinstance(projection.declaration, FromImportDeclaration)
        assert origin.requested_module_name == "nra_import_fixture"
        # This is a diagnostic path, not a claim that the captured member is a module.
        assert origin.qualified_name == "nra_import_fixture.child"


def test_beyond_package_request_matches_python_import_failure() -> None:
    source = "from .. import child"
    (origin,) = _projection(source).origins(_identity("pkg.mod"))
    assert origin.requested_module_name is None

    with pytest.raises(ImportError):
        exec(source, {"__name__": "pkg.mod", "__package__": "pkg"})
