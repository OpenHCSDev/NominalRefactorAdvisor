from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    ModuleAnnotationEvaluationMode,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    RenameTopLevelBindingAuthorityOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="")
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


def _operation(root: Path) -> RenameTopLevelDeclarationAuthorityOperation:
    return RenameTopLevelDeclarationAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/family.py").as_posix(),
            qualname="Legacy",
        ),
        new_name="Canonical",
    )


def test_renames_class_authority_without_touching_shadowed_names(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))

    result = (
        RefactorRecipe(recipe_id="rename-class-authority")
        .with_operation(replayed)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]
    payload = json_report_object(operation)

    assert result.is_clean is True
    assert payload == {
        "operation": "rename_top_level_declaration_authority",
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

    module_path.write_text(rewritten, encoding="utf-8", newline="")
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


def test_renames_function_authority_across_repository_consumers(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    declaration_path = _write_module(
        tmp_path,
        "pkg/functions.py",
        "__all__ = ('legacy',)\n\n"
        "def decorate(function):\n"
        "    return function\n\n\n"
        "@decorate\n"
        "def legacy(\n"
        "    value: int,\n"
        ") -> int:\n"
        "    if value <= 0:\n"
        "        return 0\n"
        "    return legacy(value - 1) + 1\n\n\n"
        "def shadowed(legacy):\n"
        "    return legacy\n",
    )
    reexport_path = _write_module(
        tmp_path,
        "pkg/api.py",
        "from pkg.functions import legacy as legacy\n\n__all__ = ('legacy',)\n",
    )
    consumer_path = _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.api import legacy\n"
        "import pkg.api as api\n\n"
        "RESULT = (legacy(3), api.legacy(2))\n",
    )
    operation = RenameTopLevelDeclarationAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=declaration_path.as_posix(),
            qualname="legacy",
        ),
        new_name="canonical",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = (
        RefactorRecipe(recipe_id="rename-function-authority")
        .with_operation(operation)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources

    assert "def canonical(\n" in rewritten[declaration_path.as_posix()]
    assert "return canonical(value - 1) + 1" in rewritten[declaration_path.as_posix()]
    assert (
        "def shadowed(legacy):\n    return legacy"
        in rewritten[declaration_path.as_posix()]
    )
    assert (
        "from pkg.functions import canonical as canonical"
        in rewritten[reexport_path.as_posix()]
    )
    assert "from pkg.api import canonical" in rewritten[consumer_path.as_posix()]
    assert (
        "RESULT = (canonical(3), api.canonical(2))"
        in rewritten[consumer_path.as_posix()]
    )

    for file_path, source in rewritten.items():
        Path(file_path).write_text(source, encoding="utf-8", newline="")
    completed = subprocess.run(
        [sys.executable, "-c", "from pkg.consumer import RESULT; print(RESULT)"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "(3, 2)"


def test_renames_async_function_declaration(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/functions.py",
        "async def legacy(value: int) -> int:\n"
        "    return value\n\n\n"
        "REFERENCE = legacy\n",
    )
    operation = RenameTopLevelDeclarationAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=module_path.as_posix(),
            qualname="legacy",
        ),
        new_name="canonical",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = (
        RefactorRecipe(recipe_id="rename-async-function")
        .with_operation(operation)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]

    assert "async def canonical(value: int) -> int:" in rewritten
    assert "REFERENCE = canonical" in rewritten


def test_renames_assignment_authority_across_repository_consumers(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    declaration_path = _write_module(
        tmp_path,
        "pkg/types.py",
        "from __future__ import annotations\n\n"
        "import ast\n\n"
        "__all__ = ('_TargetNode', 'identity')\n\n"
        "_TargetNode = ast.ClassDef | ast.FunctionDef\n\n\n"
        "def identity(value: '_TargetNode') -> _TargetNode:\n"
        "    return value\n",
    )
    reexport_path = _write_module(
        tmp_path,
        "pkg/api.py",
        "from .types import _TargetNode as _TargetNode\n\n"
        "__all__ = ('_TargetNode',)\n",
    )
    consumer_path = _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from .api import _TargetNode\n"
        "import pkg.api as api\n\n"
        "ALIASES = (_TargetNode, api._TargetNode)\n",
    )
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=declaration_path.as_posix()),
        binding_name="_TargetNode",
        new_name="AstTargetNode",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))

    result = RefactorRecipe("rename-assignment-authority").with_operation(
        replayed
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources

    assert json_report_object(operation) == {
        "operation": "rename_top_level_binding_authority",
        "file_path": declaration_path.as_posix(),
        "rationale": "",
        "new_name": "AstTargetNode",
        "binding_name": "_TargetNode",
    }
    assert "AstTargetNode = ast.ClassDef | ast.FunctionDef" in rewritten[
        declaration_path.as_posix()
    ]
    assert "__all__ = ('AstTargetNode', 'identity')" in rewritten[
        declaration_path.as_posix()
    ]
    assert "value: 'AstTargetNode'" in rewritten[declaration_path.as_posix()]
    assert "-> AstTargetNode:" in rewritten[declaration_path.as_posix()]
    assert "from .types import AstTargetNode as AstTargetNode" in rewritten[
        reexport_path.as_posix()
    ]
    assert "from .api import AstTargetNode" in rewritten[consumer_path.as_posix()]
    assert "api.AstTargetNode" in rewritten[consumer_path.as_posix()]
    for file_path, source in rewritten.items():
        Path(file_path).write_text(source, encoding="utf-8", newline="")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ast; from pkg.consumer import ALIASES; "
            "assert ALIASES == (ast.ClassDef | ast.FunctionDef,) * 2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_renames_annotated_assignment_authority(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/types.py",
        "LegacyType: object = int\n\nREFERENCE = LegacyType\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        binding_name="LegacyType",
        new_name="CanonicalType",
    )

    result = RefactorRecipe("rename-annotated-assignment").with_operation(
        operation
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]

    assert "CanonicalType: object = int" in rewritten
    assert "REFERENCE = CanonicalType" in rewritten


def test_renames_forward_annotations_from_final_assignment_authority(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/types.py",
        "from __future__ import annotations\n\n"
        "def identity(value: 'LegacyType') -> LegacyType:\n"
        "    return LegacyType(value)\n\n\n"
        "LegacyType = int\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        binding_name="LegacyType",
        new_name="CanonicalType",
    )

    result = RefactorRecipe("rename-forward-annotations").with_operation(
        operation
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]

    assert "value: 'CanonicalType'" in rewritten
    assert "-> CanonicalType:" in rewritten
    assert "return CanonicalType(value)" in rewritten
    assert "CanonicalType = int" in rewritten


def test_binding_rename_respects_runtime_forward_annotation_mode(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/types.py",
        "def identity(value: LegacyType) -> int:\n"
        "    return value\n\n\n"
        "LegacyType = int\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        binding_name="LegacyType",
        new_name="CanonicalType",
    )

    command = [
        sys.executable, "-c",
        "import runpy, sys; namespace = runpy.run_path(sys.argv[1]); "
        "print(namespace['identity'].__annotations__['value'] is int)",
        str(module_path),
    ]
    before = subprocess.run(command, capture_output=True, text=True)
    recipe = RefactorRecipe("rename-forward-annotation").with_operation(operation)
    if ModuleAnnotationEvaluationMode.runtime_default().annotations_execute_at_declaration:
        assert before.returncode != 0
        assert "NameError" in before.stderr
        with pytest.raises(
            CodemodOperationPreflightError,
            match="unresolved eager annotation reference",
        ):
            recipe.simulate(snapshot)
        return
    assert before.returncode == 0, before.stderr
    assert before.stdout.strip() == "True"
    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    after = subprocess.run(command, capture_output=True, text=True, check=True)
    assert after.stdout == before.stdout
    assert "value: CanonicalType" in module_path.read_text()


def test_binding_rename_rejects_ambiguous_assignment_authority(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/types.py",
        "LegacyType = AliasType = int\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelBindingAuthorityOperation(
        target=SourceRewriteTarget(file_path=module_path.as_posix()),
        binding_name="LegacyType",
        new_name="CanonicalType",
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="one unambiguous movable top-level declaration",
    ):
        RefactorRecipe("reject-ambiguous-assignment").with_operation(
            operation
        ).simulate(snapshot)


def test_chains_declaration_renames_against_each_prior_source_state(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/functions.py",
        "def legacy(value: int) -> int:\n"
        "    return value\n\n\n"
        "REFERENCE = legacy\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    def rename_stage(old_name: str, new_name: str) -> CodemodPlanDocument:
        return CodemodPlanDocument(
            recipes=(
                RefactorRecipe(recipe_id=f"rename-{old_name}").with_operation(
                    RenameTopLevelDeclarationAuthorityOperation(
                        target=SourceRewriteTarget(
                            file_path=module_path.as_posix(),
                            qualname=old_name,
                        ),
                        new_name=new_name,
                    )
                ),
            )
        )

    sequence = CodemodPlanSequence(
        documents=(
            rename_stage("legacy", "intermediate"),
            rename_stage("intermediate", "canonical"),
        )
    )
    replayed = CodemodPlanSequence.from_json_value(json_report_object(sequence))

    result = replayed.simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]

    assert result.is_clean is True
    assert result.stage_count == 2
    assert "def canonical(value: int) -> int:" in rewritten
    assert "REFERENCE = canonical" in rewritten
    assert "legacy" not in rewritten
    assert "intermediate" not in rewritten


def test_rejects_nested_declaration_target(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    module_path = _write_module(
        tmp_path,
        "pkg/functions.py",
        "class Container:\n"
        "    def legacy(self) -> int:\n"
        "        return 1\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelDeclarationAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=module_path.as_posix(),
            qualname="Container.legacy",
        ),
        new_name="canonical",
    )

    with pytest.raises(CodemodOperationPreflightError, match="top-level target"):
        RefactorRecipe(recipe_id="reject-nested-declaration").with_operation(
            operation
        ).simulate(snapshot)


def test_renames_repository_imports_aliases_and_annotations(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    consumer_path = _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from .family import (\n"
        "    Legacy,\n"
        "    Legacy as KeptAlias,\n"
        ")\n\n"
        "__all__ = ('Legacy', 'consume')\n\n"
        "def consume(value: 'Legacy') -> Legacy:\n"
        "    return KeptAlias(value.value + 1)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = RefactorRecipe(recipe_id="repository-consumer").with_operation(
        _operation(tmp_path)
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[consumer_path.as_posix()]

    assert "    Canonical," in rewritten
    assert "    Canonical as KeptAlias," in rewritten
    assert "__all__ = ('Canonical', 'consume')" in rewritten
    assert "def consume(value: 'Canonical') -> Canonical:" in rewritten

    for file_path, source in result.simulation.rewritten_sources.items():
        Path(file_path).write_text(source, encoding="utf-8", newline="")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pkg.consumer import Canonical, consume; "
            "print(consume(Canonical(2)).value)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "3"


@pytest.mark.parametrize("postponed", (False, True))
@pytest.mark.parametrize("source_first", (False, True))
@pytest.mark.parametrize(
    ("declaration", "annotation_owner"),
    (
        ("def consume(value: selected.Legacy): pass\n", "consumer.consume"),
        ("async def consume(value: selected.Legacy): pass\n", "consumer.consume"),
        ("class Consumer:\n    value: selected.Legacy\n", "consumer.Consumer"),
        ("value: selected.Legacy\n", "consumer"),
    ),
)
def test_qualified_annotation_rename_uses_its_native_evaluation_phase(
    tmp_path: Path,
    postponed: bool,
    source_first: bool,
    declaration: str,
    annotation_owner: str,
) -> None:
    _write_fixture(tmp_path)
    _write_module(tmp_path, "pkg/other.py", "class Legacy: pass\n")
    first, last = ("family", "other") if source_first else ("other", "family")
    consumer_path = _write_module(
        tmp_path,
        "pkg/consumer.py",
        ("from __future__ import annotations\n" if postponed else "")
        + f"import pkg.{first} as selected\n"
        + declaration
        + f"import pkg.{last} as selected\n",
    )
    command = [
        sys.executable,
        "-c",
        "from typing import get_type_hints; from pkg import consumer; "
        f"print(get_type_hints({annotation_owner})['value'].__module__)",
    ]
    before = subprocess.check_output(command, cwd=tmp_path)
    plan = CodemodPlanSequence.from_operations((_operation(tmp_path),))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(tmp_path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(plan)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert subprocess.check_output(command, cwd=tmp_path) == before
    renamed = before.strip() == b"pkg.family"
    assert (
        "value: selected.Canonical" in consumer_path.read_text(encoding="utf-8")
    ) is renamed


def test_rejects_rebound_repository_import(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.family import Legacy\n\n"
        "VALUE = Legacy(1)\n"
        "Legacy = object\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match="binding.*is rebound"):
        RefactorRecipe(recipe_id="rebound-consumer").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


@pytest.mark.parametrize(
    ("trailing_source", "message"),
    (
        ("Canonical = object\n", "Canonical.*collides"),
        ("__all__ = exported_names\n", "unresolved export policy"),
    ),
)
def test_rejects_unproved_import_binding_rename(
    tmp_path: Path,
    trailing_source: str,
    message: str,
) -> None:
    _write_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "from pkg.family import Legacy\n\n"
        "VALUE = Legacy(1)\n"
        f"{trailing_source}",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match=message):
        RefactorRecipe(recipe_id="unproved-consumer").with_operation(
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
        match="affected star-import boundary",
    ):
        RefactorRecipe(recipe_id="star-consumer").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_star_import_that_would_gain_new_public_name(tmp_path: Path) -> None:
    _write_module(tmp_path, "pkg/__init__.py", "")
    family_path = _write_module(
        tmp_path,
        "pkg/family.py",
        "class _Legacy:\n"
        "    pass\n",
    )
    _write_module(tmp_path, "pkg/consumer.py", "from pkg.family import *\n")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = RenameTopLevelDeclarationAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=family_path.as_posix(),
            qualname="_Legacy",
        ),
        new_name="Canonical",
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="affected star-import boundary",
    ):
        RefactorRecipe(recipe_id="new-star-export").with_operation(operation).simulate(
            snapshot
        )


def test_rejects_nested_import_consumer(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_module(
        tmp_path,
        "pkg/consumer.py",
        "def authority():\n"
        "    from pkg.family import Legacy\n\n"
        "    return Legacy\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match="nested import"):
        RefactorRecipe(recipe_id="nested-consumer").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_renames_transitive_repository_consumers(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    reexport_path = _write_module(
        tmp_path,
        "pkg/reexport.py",
        "from pkg.family import Legacy as Legacy\n\n__all__ = ['Legacy']\n",
    )
    direct_path = _write_module(
        tmp_path,
        "pkg/direct_consumer.py",
        "from pkg.reexport import Legacy\n\nVALUE = Legacy(1)\n",
    )
    qualified_path = _write_module(
        tmp_path,
        "pkg/qualified_consumer.py",
        "import pkg.reexport as exported\n\nVALUE = exported.Legacy(2)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = RefactorRecipe(recipe_id="transitive-consumer").with_operation(
        _operation(tmp_path)
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources

    assert "from pkg.family import Canonical as Canonical" in rewritten[
        reexport_path.as_posix()
    ]
    assert "__all__ = ['Canonical']" in rewritten[reexport_path.as_posix()]
    assert "from pkg.reexport import Canonical" in rewritten[direct_path.as_posix()]
    assert "VALUE = Canonical(1)" in rewritten[direct_path.as_posix()]
    assert "VALUE = exported.Canonical(2)" in rewritten[qualified_path.as_posix()]

    for file_path, source in rewritten.items():
        Path(file_path).write_text(source, encoding="utf-8", newline="")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pkg.direct_consumer import VALUE as direct; "
            "from pkg.qualified_consumer import VALUE as qualified; "
            "print(direct.value, qualified.value)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "1 2"


def test_preserves_unrelated_same_spelled_authority_and_export(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    unrelated_path = _write_module(
        tmp_path,
        "pkg/unrelated.py",
        "__all__ = ['Legacy']\n\n"
        "class Legacy:\n"
        "    pass\n\n"
        "VALUE = Legacy\n",
    )
    consumer_path = _write_module(
        tmp_path,
        "pkg/unrelated_consumer.py",
        "import pkg.unrelated as unrelated\n\nVALUE = unrelated.Legacy\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = RefactorRecipe(recipe_id="unrelated-authority").with_operation(
        _operation(tmp_path)
    ).simulate(snapshot)

    assert unrelated_path.as_posix() not in result.simulation.rewritten_sources
    assert consumer_path.as_posix() not in result.simulation.rewritten_sources


def test_preserves_explicit_reexport_alias(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    api_path = _write_module(
        tmp_path,
        "pkg/api.py",
        "from pkg.family import Legacy as PublicAuthority\n\n"
        "__all__ = ['PublicAuthority']\n",
    )
    consumer_path = _write_module(
        tmp_path,
        "pkg/api_consumer.py",
        "from pkg.api import PublicAuthority\n\nVALUE = PublicAuthority(3)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    result = RefactorRecipe(recipe_id="stable-public-alias").with_operation(
        _operation(tmp_path)
    ).simulate(snapshot)
    rewritten = result.simulation.rewritten_sources

    assert "from pkg.family import Canonical as PublicAuthority" in rewritten[
        api_path.as_posix()
    ]
    assert consumer_path.as_posix() not in rewritten


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
