from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    RenameLocalClassAuthorityOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _write_fixture(root: Path, *, trailing_source: str = "") -> Path:
    _write_module(root, "pkg/__init__.py", "")
    return _write_module(
        root,
        "pkg/family.py",
        "class Legacy:\n"
        "    def __init__(self, value: int) -> None:\n"
        "        self.value = value\n\n\n"
        "class Child(Legacy):\n"
        "    pass\n\n\n"
        "def build(value: Legacy) -> 'Legacy | None':\n"
        "    return Legacy(value.value + 1)\n\n\n"
        "def shadowed(Legacy):\n"
        "    return Legacy\n\n\n"
        "class Consumer:\n"
        "    before = Legacy\n"
        "    Legacy = 5\n"
        "    after = Legacy\n"
        f"{trailing_source}",
    )


def _operation(root: Path) -> RenameLocalClassAuthorityOperation:
    return RenameLocalClassAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/family.py").as_posix(),
            qualname="Legacy",
        ),
        new_name="Canonical",
    )


def test_renames_local_class_authority_without_touching_shadowed_names(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))

    result = (
        RefactorRecipe(recipe_id="rename-local-authority")
        .with_operation(replayed)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]
    payload = json_report_object(operation)

    assert result.is_clean is True
    assert payload == {
        "operation": "rename_local_class_authority",
        "file_path": module_path.as_posix(),
        "target_qualname": "Legacy",
        "rationale": "",
        "new_name": "Canonical",
    }
    assert "class Canonical:" in rewritten
    assert "class Child(Canonical):" in rewritten
    assert "def build(value: Canonical) -> 'Canonical | None':" in rewritten
    assert "return Canonical(value.value + 1)" in rewritten
    assert "def shadowed(Legacy):\n    return Legacy" in rewritten
    assert (
        "class Consumer:\n"
        "    before = Canonical\n"
        "    Legacy = 5\n"
        "    after = Legacy"
    ) in rewritten

    module_path.write_text(rewritten, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from pkg.family import Canonical, Child, Consumer, build, "
            "shadowed; result = build(Canonical(2)); "
            "print(json.dumps([result.value, isinstance(result, Canonical), "
            "Child(4).value, shadowed(6), Consumer.before is Canonical, "
            "Consumer.after]))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [3, True, 4, 6, True, 5]


def test_rejects_imported_repository_consumer(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.family import Legacy\n\n"
        "VALUE = Legacy(1)\n"
        "Legacy = object\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="imported repository consumer",
    ):
        RefactorRecipe(recipe_id="external-consumer").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_used_star_import_consumer(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.family import *\n\nVALUE = Legacy(1)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="star-import consumer",
    ):
        RefactorRecipe(recipe_id="star-consumer").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_renames_qualified_self_module_reference(tmp_path: Path) -> None:
    module_path = _write_fixture(
        tmp_path,
        trailing_source=(
            "\nimport pkg.family as family\n\n"
            "QUALIFIED = family.Legacy\n\n"
            "def shadowed_qualified(family):\n"
            "    return family.Legacy\n"
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = RefactorRecipe(recipe_id="qualified-reference").with_operation(
        _operation(tmp_path)
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]

    assert "QUALIFIED = family.Canonical" in rewritten
    assert "return family.Legacy" in rewritten


def test_rejects_string_name_surface(tmp_path: Path) -> None:
    _write_fixture(tmp_path, trailing_source='\nSERIALIZED = "Legacy"\n')
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="string name surface",
    ):
        RefactorRecipe(recipe_id="dynamic-name").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_class_local_stringized_annotation_shadow(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        trailing_source=(
            "\nclass ShadowedAnnotation:\n"
            "    Legacy = int\n"
            "    value: 'Legacy'\n"
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="string name surface",
    ):
        RefactorRecipe(recipe_id="shadowed-annotation").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_replacement_name_collision(tmp_path: Path) -> None:
    _write_fixture(tmp_path, trailing_source="\nclass Canonical:\n    pass\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="Canonical.*already bound",
    ):
        RefactorRecipe(recipe_id="colliding-name").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)
