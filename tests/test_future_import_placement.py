"""Ordinary module imports do not acquire future-directive placement."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    RefactorRecipeOperationCompiler,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_imports import (
    ImportSourceGroup,
    RequestedImportStatement,
    ModuleImportMutation,
)
from nominal_refactor_advisor.lexical_bindings import ImportBoundNameProjection
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_source_edits import SourceTextGeometry


@pytest.mark.parametrize(
    "source, group",
    (
        ("import __future__", ImportSourceGroup.STANDARD_LIBRARY),
        ("import __future__ as features", ImportSourceGroup.STANDARD_LIBRARY),
        ("from __future__ import annotations", ImportSourceGroup.FUTURE),
        ("from __future__ import annotations as policy", ImportSourceGroup.FUTURE),
        ("from .__future__ import annotations", ImportSourceGroup.RELATIVE),
    ),
)
def test_import_declaration_owns_future_placement(
    source: str, group: ImportSourceGroup
) -> None:
    statement = ast.parse(source).body[0]
    declaration = ImportBoundNameProjection(statement).declaration
    (request,) = RequestedImportStatement.from_statement(statement)
    assert declaration.is_future_import is (group is ImportSourceGroup.FUTURE)
    assert request.is_future_import is declaration.is_future_import
    assert request.source_group is group


@pytest.mark.parametrize(
    "source, references, groups",
    (
        (
            "import ast, pathlib as paths",
            ("ast", "pathlib"),
            (ImportSourceGroup.STANDARD_LIBRARY,),
        ),
        (
            "import ast, nra_uninstalled_dependency",
            ("ast", "nra_uninstalled_dependency"),
            (ImportSourceGroup.STANDARD_LIBRARY, ImportSourceGroup.THIRD_PARTY),
        ),
        (
            "from ..package import first, second",
            ("..package",),
            (ImportSourceGroup.RELATIVE,),
        ),
        ("from . import first", (".",), (ImportSourceGroup.RELATIVE,)),
        (
            "from __future__ import division, annotations as mode",
            ("__future__",),
            (ImportSourceGroup.FUTURE,),
        ),
    ),
)
def test_module_reference_projection_retains_nominal_import_kind(
    source: str, references: tuple[str, ...], groups: tuple[ImportSourceGroup, ...]
) -> None:
    declaration = ImportBoundNameProjection(ast.parse(source).body[0]).declaration
    assert (
        tuple(reference.source for reference in declaration.module_references)
        == references
    )
    assert ImportSourceGroup.from_declaration(declaration) == groups


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    "preamble",
    (
        "",
        '"""Kept documentation."""\n\n',
        '# kept header\n"""Kept documentation."""\n\n# compiler policy\n',
    ),
)
@pytest.mark.parametrize(
    "initial, requested",
    (
        ("from __future__ import annotations", "import __future__"),
        ("from __future__ import annotations", "import __future__ as features"),
        ("import __future__", "from __future__ import annotations"),
    ),
)
def test_ensure_import_preserves_native_future_placement(
    newline: str,
    preamble: str,
    initial: str,
    requested: str,
) -> None:
    source = (preamble + initial + "\n\nVALUE = 1\n").replace("\n", newline)
    module = ParsedModule(
        path=Path("future_placement.py"),
        module_name="future_placement",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    plan = CodemodPlanSequence.from_operations(
        (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=module.file_path),
                import_source=requested,
            ),
        )
    )
    result = plan.simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert result.is_clean
    updated = result.final_snapshot.sources_by_file_path[module.file_path]
    tree = ast.parse(updated)
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert isinstance(imports[0], ast.ImportFrom)
    assert isinstance(imports[1], ast.Import)
    compile(updated, module.file_path, "exec", dont_inherit=True)
    if preamble:
        assert ast.get_docstring(tree) == "Kept documentation."
    if "# kept header" in source:
        assert "# kept header" in updated
        assert "# compiler policy" in updated
    if newline == "\r\n":
        assert "\n" not in updated.replace("\r\n", "")


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    "initial, requested, scope",
    (
        (
            "from typing import Any\n",
            "from typing import Callable",
            ModuleImportScope.RUNTIME,
        ),
        ("VALUE = 1\n", "from pathlib import Path", ModuleImportScope.TYPE_CHECKING),
        (
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pathlib import Path\n",
            "from collections import Counter",
            ModuleImportScope.TYPE_CHECKING,
        ),
    ),
)
def test_synthesized_imports_use_destination_line_endings(
    newline: str, initial: str, requested: str, scope: ModuleImportScope
) -> None:
    source = initial.replace("\n", newline)
    module = ParsedModule(
        path=Path("line_endings.py"),
        module_name="line_endings",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,))
    mutation = ModuleImportMutation.from_source(
        file_path=module.file_path,
        import_source=requested,
        scope=scope,
    )
    edits = mutation.resolved_edits(snapshot)
    compiler = RefactorRecipeOperationCompiler.from_context(snapshot)
    result = snapshot.simulate_rewrites(
        compiler._planned_rewrites_from_physical_edits(edits)
    )
    updated = result.rewritten_sources[module.file_path]
    compile(updated, module.file_path, "exec", dont_inherit=True)
    assert all(line.endswith(newline) for line in updated.splitlines(keepends=True))
    if newline == "\r\n":
        assert "\n" not in updated.replace("\r\n", "")


@pytest.mark.parametrize(
    "source, expected",
    (("", "\n"), ("value = 1", "\n"), ("# header\r\nvalue = 1\n", "\r\n")),
)
def test_generated_line_geometry_uses_first_destination_ending(
    source: str, expected: str
) -> None:
    geometry = SourceTextGeometry(source)
    assert geometry.line_ending == expected
    assert geometry.generated_lines("import ast\n\nimport sys") == (
        "import ast" + expected,
        expected,
        "import sys" + expected,
    )
    assert geometry.generated_lines("") == ()
