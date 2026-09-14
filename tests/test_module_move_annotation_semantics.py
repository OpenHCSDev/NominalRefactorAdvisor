"""Postponed syntax does not erase runtime annotation-consumer obligations."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    MoveSymbolsToModuleOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_import_scopes import ModuleImportScope
from nominal_refactor_advisor.codemod_module_move_reports import ModuleMoveObstacleKind
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError

CASES = {
    "init_var": (
        "from dataclasses import InitVar\nAlias = InitVar[int]\n",
        "@dataclass\nclass Carrier:\n    value: Alias\n"
        "    def __post_init__(self, value):\n        self.seen = value\n",
        "instance.seen",
    ),
    "class_var": (
        "from typing import ClassVar\nAlias = ClassVar[int]\n",
        "@dataclass\nclass Carrier:\n    value: Alias = 35\n",
        "instance.value",
    ),
}


def _package(root: Path, sources: dict[str, str]) -> None:
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name, source in sources.items():
        (package / name).write_text(source)


def _runtime_result(root: Path, first: str, value_expression: str) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import dataclasses, importlib, json\n"
            f"importlib.import_module('pkg.{first}')\n"
            "from pkg.source import Carrier\n"
            "result = {'fields': [field.name for field in dataclasses.fields(Carrier)], "
            "'annotations': Carrier.__annotations__}\n"
            "try:\n"
            "    instance = Carrier(99)\n"
            f"    result['value'] = {value_expression}\n"
            "except TypeError as error:\n"
            "    result['error'] = str(error)\n"
            "print(json.dumps(result))\n",
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("alias_kind", CASES)
@pytest.mark.parametrize("first", ("source", "destination"))
def test_guarding_original_runtime_alias_changes_dataclass_creation(
    tmp_path: Path, alias_kind: str, first: str
) -> None:
    alias_source, carrier, value_expression = CASES[alias_kind]
    prelude = "from __future__ import annotations\nfrom dataclasses import dataclass\n"
    original = tmp_path / "original"
    rewritten = tmp_path / "rewritten"
    _package(original, {"source.py": prelude + alias_source + carrier})
    _package(
        rewritten,
        {
            "source.py": alias_source + "from .destination import Carrier\n",
            "destination.py": prelude
            + "from typing import TYPE_CHECKING\n"
            + "if TYPE_CHECKING:\n    from .source import Alias\n"
            + carrier,
        },
    )

    before = _runtime_result(original, "source", value_expression)
    after = _runtime_result(rewritten, first, value_expression)

    assert before["annotations"] == after["annotations"] == {"value": "Alias"}
    assert before["fields"] == []
    assert after["fields"] == ["value"]
    if alias_kind == "init_var":
        assert before["value"] == 99
        assert "missing 1 required positional argument: 'value'" in after["error"]
    else:
        assert "positional argument" in before["error"]
        assert after["value"] == 99


def _move(root: Path):
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(root, use_parse_cache=False, parse_workers=1)
    )
    operation = MoveSymbolsToModuleOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/source.py").as_posix()),
        symbol_qualnames=("Carrier",),
        destination_path=(root / "pkg/destination.py").as_posix(),
    )
    document = CodemodPlanDocument(
        recipes=(RefactorRecipe("move-carrier", operations=(operation,)),)
    )
    return snapshot, operation, document


@pytest.mark.parametrize("alias_kind", CASES)
def test_destination_guarded_alias_does_not_discharge_source_local_runtime_binding(
    tmp_path: Path, alias_kind: str
) -> None:
    alias_source, carrier, _ = CASES[alias_kind]
    prelude = "from __future__ import annotations\nfrom dataclasses import dataclass\n"
    _package(
        tmp_path,
        {
            "source.py": prelude + alias_source + carrier,
            "destination.py": "from __future__ import annotations\n"
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from .source import Alias\n",
        },
    )
    snapshot, operation, document = _move(tmp_path)

    report = operation.dependency_report(snapshot)

    assert report.annotation_evaluation_is_preserved
    assert report.moved_annotation_count == 1
    assert report.obstacle_details(ModuleMoveObstacleKind.SOURCE_LOCAL_DEPENDENCY) == (
        "Alias",
    )
    assert not report.is_clean
    with pytest.raises(
        CodemodOperationPreflightError, match="source-local dependencies"
    ):
        document.simulate(snapshot)


def test_declared_source_guarded_alias_stays_guarded_in_actual_move(
    tmp_path: Path,
) -> None:
    _package(
        tmp_path,
        {
            "shared.py": "from dataclasses import InitVar\nAlias = InitVar[int]\n",
            "source.py": "from __future__ import annotations\n"
            "from dataclasses import dataclass\nfrom typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from .shared import Alias\n"
            "@dataclass\nclass Carrier:\n    value: Alias\n",
            "destination.py": "from __future__ import annotations\n",
        },
    )
    snapshot, operation, document = _move(tmp_path)
    before = _runtime_result(tmp_path, "source", "instance.value")

    dependencies = operation.dependency_report(snapshot)
    alias_dependency = next(
        dependency
        for dependency in dependencies.import_dependencies
        if dependency.name == "Alias"
    )
    assert alias_dependency.scope is ModuleImportScope.TYPE_CHECKING
    simulation = document.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    for first in ("source", "destination"):
        after = _runtime_result(tmp_path, first, "instance.value")
        assert (
            before
            == after
            == {
                "fields": ["value"],
                "annotations": {"value": "Alias"},
                "value": 99,
            }
        )
