from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    CollapseIntermediateClassAuthorityOperation,
    PromoteClassMembersToAncestorOperation,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _write_fixture(root: Path, *, method_body: str = "return cls.label") -> Path:
    _write_module(root, "pkg/__init__.py", "")
    return _write_module(
        root,
        "pkg/family.py",
        "from abc import ABC\n"
        "from typing import ClassVar\n\n"
        "class Authority(ABC):\n"
        "    label: ClassVar[str] = 'shared'\n\n"
        "    def retained(self) -> str:\n"
        "        return self.label\n\n\n"
        "class Intermediate(Authority, ABC):\n"
        "    payload: ClassVar[int] = 3\n\n"
        "    @classmethod\n"
        "    def describe(cls) -> str:\n"
        f"        {method_body}\n\n\n"
        "class Leaf(Intermediate):\n"
        "    pass\n",
    )


def _operation(root: Path) -> PromoteClassMembersToAncestorOperation:
    return PromoteClassMembersToAncestorOperation(
        target=SourceRewriteTarget(
            file_path=(root / "pkg/family.py").as_posix(),
            qualname="Intermediate",
        ),
        destination=SourceRewriteTarget(
            file_path=(root / "pkg/family.py").as_posix(),
            qualname="Authority",
        ),
        member_names=("payload", "describe"),
    )


def _runtime_output(root: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from pkg.family import Authority, Intermediate, Leaf; "
            "print(json.dumps([Authority.describe(), Intermediate.payload, "
            "Leaf.payload]))",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_promotes_selected_members_to_existing_ancestor_as_one_operation(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = _operation(tmp_path)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))

    result = (
        RefactorRecipe(recipe_id="promote-intermediate-members")
        .with_operation(replayed)
        .simulate(snapshot)
    )
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]
    payload = json_report_object(operation)

    assert result.is_clean is True
    assert payload["operation"] == "promote_class_members_to_ancestor"
    assert payload["target_qualname"] == "Intermediate"
    assert payload["destination"]["target_qualname"] == "Authority"
    assert payload["member_names"] == ("payload", "describe")
    assert "class Authority(ABC):\n    label: ClassVar[str] = 'shared'\n\n    payload: ClassVar[int] = 3" in rewritten
    assert "    @classmethod\n    def describe(cls) -> str:" in rewritten
    assert "class Intermediate(Authority, ABC):\n    pass" in rewritten

    module_path.write_text(rewritten, encoding="utf-8")
    assert json.loads(_runtime_output(tmp_path)) == ["shared", 3, 3]


def test_chains_member_promotion_with_intermediate_authority_collapse(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    promotion = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(recipe_id="promote-intermediate-members").with_operation(
                _operation(tmp_path)
            ),
        ),
    )
    collapse = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(recipe_id="collapse-empty-intermediary").with_operation(
                CollapseIntermediateClassAuthorityOperation(
                    target=_operation(tmp_path).target,
                    replacement_base=_operation(tmp_path).destination,
                )
            ),
        ),
    )

    sequence = CodemodPlanSequence(documents=(promotion, collapse))
    replayed = CodemodPlanSequence.from_json_value(json_report_object(sequence))
    result = replayed.simulate(snapshot)
    rewritten = result.simulation.rewritten_sources[module_path.as_posix()]
    payload = json_report_object(result.sequence)

    assert result.is_clean is True
    assert tuple(
        stage["recipes"][0]["operations"][0]["operation"]
        for stage in payload["stages"]
    ) == (
        "promote_class_members_to_ancestor",
        "collapse_intermediate_class_authority",
    )
    assert all(
        "child_classes" not in stage["recipes"][0]["operations"][0]
        for stage in payload["stages"]
    )
    assert "class Intermediate" not in rewritten
    assert "class Leaf(Authority):" in rewritten

    module_path.write_text(rewritten, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from pkg.family import Authority, Leaf; "
            "print(json.dumps([Authority.describe(), Leaf.payload]))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == ["shared", 3]


def test_rejects_intermediate_collapse_before_members_are_promoted(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="still owns behavior or state",
    ):
        RefactorRecipe(recipe_id="premature-collapse").with_operation(
            CollapseIntermediateClassAuthorityOperation(
                target=_operation(tmp_path).target,
                replacement_base=_operation(tmp_path).destination,
            )
        ).simulate(snapshot)


def test_rejects_destination_outside_source_ancestry(tmp_path: Path) -> None:
    module_path = _write_fixture(tmp_path)
    module_path.write_text(
        module_path.read_text(encoding="utf-8")
        + "\n\nclass Unrelated(ABC):\n    pass\n",
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = PromoteClassMembersToAncestorOperation(
        target=_operation(tmp_path).target,
        destination=SourceRewriteTarget(
            file_path=module_path.as_posix(),
            qualname="Unrelated",
        ),
        member_names=("describe",),
    )

    with pytest.raises(
        CodemodOperationPreflightError,
        match="is not an ancestor",
    ):
        RefactorRecipe(recipe_id="invalid-owner").with_operation(operation).simulate(
            snapshot
        )


def test_rejects_owner_sensitive_method_promotion(tmp_path: Path) -> None:
    _write_fixture(tmp_path, method_body="return super().retained(cls())")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="super_reference",
    ):
        RefactorRecipe(recipe_id="unsafe-method").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)


def test_rejects_class_local_declaration_dependency(tmp_path: Path) -> None:
    module_path = _write_fixture(tmp_path)
    module_path.write_text(
        module_path.read_text(encoding="utf-8").replace(
            "    payload: ClassVar[int] = 3",
            "    default_payload = 3\n"
            "    payload: ClassVar[int] = default_payload",
        ),
        encoding="utf-8",
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="class-local references.*default_payload",
    ):
        RefactorRecipe(recipe_id="unsafe-field").with_operation(
            _operation(tmp_path)
        ).simulate(snapshot)
