"""A syntactic binding list is not the namespace consumed by computed __all__."""

import ast
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.class_index import module_public_export_contract
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    ModuleMoveDependencyReport,
    ModuleMoveObstacleKind,
    MoveSymbolClosureToModuleOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)


@pytest.mark.parametrize(
    "installation",
    (
        "globals()['Injected'] = 1\n",
        "globals().update(Injected=1)\n",
        "exec('Injected = 1')\n",
        "def install(): globals()['Injected'] = 1\ninstall()\n",
    ),
)
def test_computed_exports_do_not_prove_absence_from_lexical_bindings(installation):
    source = installation + (
        "__all__ = tuple(name for name in globals() if not name.startswith('__'))\n"
    )
    native = ModuleType("computed_exports")
    exec(source, vars(native))
    assert "Injected" in native.__all__
    module = ParsedModule(
        Path("computed_exports.py"),
        "computed_exports",
        False,
        ast.parse(source),
        source,
    )
    policy = module_public_export_contract(module)
    assert policy.exposure_for("Injected").blocks_closed_boundary


def test_symbol_move_does_not_restore_an_import_overwritten_by_computed_star(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "provider.py").write_text(
        "def decorate(cls):\n    cls.marker = 'star'\n    return cls\n"
        "globals()['dataclass'] = decorate\n"
        "__all__ = tuple(name for name in globals() if not name.startswith('__'))\n",
        encoding="utf-8",
    )
    source_path = package / "source.py"
    source_path.write_text(
        "from dataclasses import dataclass\nfrom .provider import *\n"
        "@dataclass\nclass Payload:\n    value: int\n",
        encoding="utf-8",
    )
    destination = package / "destination.py"
    destination.write_text("", encoding="utf-8")

    def native_output():
        return subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from pkg.source import Payload; print('marker' in vars(Payload))",
            ],
            cwd=tmp_path,
            text=True,
        ).strip()

    before = native_output()
    assert before == "True"
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = MoveSymbolClosureToModuleOperation(
        target=SourceRewriteTarget(file_path=source_path.as_posix()),
        root_symbol_qualnames=("Payload",),
        maximum_moved_symbol_count=32,
        destination_path=destination.as_posix(),
    )
    try:
        simulation = (
            RefactorRecipe("computed-star-binding")
            .with_operation(operation)
            .simulate(snapshot)
        )
    except CodemodOperationPreflightError as error:
        report = error.report.detail
        assert isinstance(report, ModuleMoveDependencyReport)
        assert report.obstacle_details(
            ModuleMoveObstacleKind.AMBIGUOUS_IMPORT_DEPENDENCY
        ) == ("dataclass",)
        assert destination.read_text(encoding="utf-8") == ""
        assert native_output() == before
        return
    simulation.apply()
    assert (
        native_output() == before
    ), "An admitted move changed the actual decorator binding"
