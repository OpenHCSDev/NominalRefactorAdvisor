from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor import (
    PromoteExactDataclassFieldsToExistingAuthorityOperation,
)
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
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


def _fixture_source(
    *,
    authority_members: str = "",
    intervening_source: str = "",
    authority_first: bool = False,
) -> str:
    imports = (
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
    )
    authority = (
        "@dataclass(frozen=True)\n"
        "class ProjectionIdentity:\n"
        "    \"\"\"Stable identity shared by projections.\"\"\"\n\n"
        "    module_name: str\n"
        "    file_path: str\n"
        f"{authority_members}"
    )
    participants = (
        "@dataclass(frozen=True)\n"
        "class AlphaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    alpha_value: int\n\n"
        "    def label(self) -> str:\n"
        "        return f'{self.module_name}:{self.file_path}'\n\n\n"
        "@dataclass(frozen=True)\n"
        "class BetaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    beta_value: float\n"
    )
    if authority_first:
        return f"{imports}{authority}\n\n{participants}"
    return f"{imports}{participants}\n\n{intervening_source}{authority}"


def _write_fixture(root: Path, *, source: str | None = None) -> Path:
    _write_module(root, "pkg/__init__.py", "")
    return _write_module(
        root,
        "pkg/models.py",
        _fixture_source() if source is None else source,
    )


def _operation(
    module_path: Path,
) -> PromoteExactDataclassFieldsToExistingAuthorityOperation:
    return PromoteExactDataclassFieldsToExistingAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=module_path.as_posix(),
            qualname="ProjectionIdentity",
        ),
        evidence_field_name="file_path",
    )


def _runtime_output(root: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; "
            "from dataclasses import fields; "
            "from inspect import signature; "
            "from pkg.models import AlphaProjection, BetaProjection, ProjectionIdentity; "
            "types = [AlphaProjection, BetaProjection, ProjectionIdentity]; "
            "values = [AlphaProjection('pkg', 'a.py', 3), "
            "BetaProjection('pkg', 'b.py', 2.5), ProjectionIdentity('pkg', 'p.py')]; "
            "print(json.dumps({'fields': [[f.name for f in fields(t)] for t in types], "
            "'match_args': [t.__match_args__ for t in types], "
            "'signatures': [str(signature(t)) for t in types], "
            "'repr': [repr(value) for value in values], "
            "'label': values[0].label()}, sort_keys=True))",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_operation_reuses_and_relocates_existing_field_authority(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    before = _runtime_output(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))
    operation = _operation(module_path)
    recipe = RefactorRecipe(recipe_id="reuse-projection-identity").with_operation(
        operation
    )

    payload = json_report_object(operation)
    simulation = recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert payload["operation"] == (
        "promote_exact_dataclass_fields_to_existing_authority"
    )
    assert payload["evidence_field_name"] == "file_path"
    assert "base_name" not in payload
    assert "class_names" not in payload
    assert "field_names" not in payload
    assert type(RefactorRecipeOperation.from_dict(payload)) is type(operation)
    assert simulation.is_clean is True
    assert rewritten.count("class ProjectionIdentity:") == 1
    assert rewritten.index("class ProjectionIdentity:") < rewritten.index(
        "class AlphaProjection(ProjectionIdentity):"
    )
    assert "class BetaProjection(ProjectionIdentity):" in rewritten
    assert rewritten.count("module_name: str") == 1
    assert rewritten.count("file_path: str") == 1
    assert "Stable identity shared by projections." in rewritten

    simulation.apply()
    assert _runtime_output(tmp_path) == before
    after_snapshot = CodemodSourceSnapshot.from_modules(
        tuple(parse_python_modules(tmp_path))
    )
    with pytest.raises(
        CodemodOperationPreflightError,
        match="belongs to 0 current exact dataclass field authority components",
    ):
        recipe.simulate(after_snapshot, backend=CodemodBackend.AST_SPAN)


def test_operation_keeps_an_existing_preceding_authority_in_place(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source(authority_first=True),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    simulation = RefactorRecipe(recipe_id="reuse-preceding-authority").with_operation(
        _operation(module_path)
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert rewritten.count("class ProjectionIdentity:") == 1
    assert rewritten.startswith("from __future__ import annotations")
    assert rewritten.index("class ProjectionIdentity:") < rewritten.index(
        "class AlphaProjection(ProjectionIdentity):"
    )


def test_operation_rejects_an_authority_with_additional_behavior(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source(
            authority_members=(
                "\n    def endpoint(self) -> str:\n"
                "        return self.file_path\n"
            )
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="behavior-free outside its fields",
    ):
        RefactorRecipe(recipe_id="behavioral-authority").with_operation(
            _operation(module_path)
        ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


def test_operation_rejects_an_authority_with_additional_fields(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source(authority_members="    identity_version: int\n"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="must own exactly the repeated fields",
    ):
        RefactorRecipe(recipe_id="overwide-authority").with_operation(
            _operation(module_path)
        ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


def test_operation_rejects_a_name_observed_before_relocation(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source(intervening_source="ALIAS = ProjectionIdentity\n\n\n"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="referenced before its current declaration",
    ):
        RefactorRecipe(recipe_id="observed-relocation").with_operation(
            _operation(module_path)
        ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


def test_operation_allows_a_deferred_name_reference_before_relocation(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source(
            intervening_source=(
                "def build_identity() -> ProjectionIdentity:\n"
                "    return ProjectionIdentity('pkg', 'built.py')\n\n\n"
            )
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    simulation = RefactorRecipe(recipe_id="deferred-relocation").with_operation(
        _operation(module_path)
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert simulation.is_clean is True
    assert rewritten.index("class ProjectionIdentity:") < rewritten.index(
        "class AlphaProjection(ProjectionIdentity):"
    )
    assert "return ProjectionIdentity('pkg', 'built.py')" in rewritten
