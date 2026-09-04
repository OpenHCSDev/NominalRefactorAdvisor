from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodOperationPreflightError,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    FactorExactDataclassFieldAuthorityOperation,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.semantic_descent import SemanticAuthorityKind


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _fixture_source() -> str:
    return (
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
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
        "    beta_value: float\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DeltaProjection:\n"
        "    \"\"\"Projection with behavior after the promoted fields.\"\"\"\n\n"
        "    module_name: str\n"
        "    file_path: str\n\n"
        "    def endpoint(self) -> str:\n"
        "        return self.file_path\n\n\n"
        "@dataclass(frozen=True)\n"
        "class EpsilonProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n\n\n"
        "@dataclass(frozen=True)\n"
        "class ZetaProjection:\n"
        "    \"\"\"Projection documented without additional behavior.\"\"\"\n\n"
        "    module_name: str\n"
        "    file_path: str\n\n\n"
        "@dataclass(frozen=True)\n"
        "class GammaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    gamma_value: bytes\n"
    )


def _write_fixture(root: Path, *, source: str | None = None) -> Path:
    _write_module(root, "pkg/__init__.py", "")
    return _write_module(
        root,
        "pkg/models.py",
        _fixture_source() if source is None else source,
    )


def _operation(module_path: Path) -> FactorExactDataclassFieldAuthorityOperation:
    return FactorExactDataclassFieldAuthorityOperation(
        target=SourceRewriteTarget(
            file_path=module_path.as_posix(),
            qualname="AlphaProjection",
        ),
        evidence_field_name="file_path",
        base_name="ProjectionIdentity",
    )


def _runtime_output(root: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; "
            "from dataclasses import fields; "
            "from inspect import signature; "
            "from pkg.models import AlphaProjection, BetaProjection, GammaProjection; "
            "values = [AlphaProjection('pkg', 'a.py', 3), "
            "BetaProjection(module_name='pkg', file_path='b.py', beta_value=2.5), "
            "GammaProjection('pkg', 'c.py', b'x')]; "
            "print(json.dumps({'fields': [[f.name for f in fields(type(v))] for v in values], "
            "'match_args': [type(v).__match_args__ for v in values], "
            "'signatures': [str(signature(type(v))) for v in values], "
            "'repr': [repr(v) for v in values], "
            "'hashable': [isinstance(hash(v), int) for v in values], "
            "'equal': values[0] == AlphaProjection('pkg', 'a.py', 3), "
            "'label': values[0].label()}, sort_keys=True))",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _lattice_source() -> str:
    return (
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class AlphaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    qualname: str\n"
        "    alpha_value: int\n\n\n"
        "@dataclass(frozen=True)\n"
        "class BetaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    qualname: str\n"
        "    beta_value: float\n\n\n"
        "@dataclass(frozen=True)\n"
        "class GammaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    gamma_value: bytes\n\n\n"
        "@dataclass(frozen=True)\n"
        "class DeltaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    delta_value: bool\n"
    )


def _lattice_runtime_output(root: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; "
            "from dataclasses import fields; "
            "from inspect import signature; "
            "from pkg.models import AlphaProjection, BetaProjection, GammaProjection, DeltaProjection; "
            "types = [AlphaProjection, BetaProjection, GammaProjection, DeltaProjection]; "
            "print(json.dumps({'fields': [[f.name for f in fields(t)] for t in types], "
            "'match_args': [t.__match_args__ for t in types], "
            "'signatures': [str(signature(t)) for t in types]}, sort_keys=True))",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_operation_reproves_field_component_without_serializing_rosters(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path)
    before = _runtime_output(tmp_path)
    modules = tuple(parse_python_modules(tmp_path))
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    operation = _operation(module_path)
    recipe = RefactorRecipe(recipe_id="factor-projection-identity").with_operation(
        operation
    )

    payload = json_report_object(operation)
    claims = recipe.declared_authority_claims(snapshot)
    simulation = recipe.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert payload["operation"] == "factor_exact_dataclass_field_authority"
    assert payload["evidence_field_name"] == "file_path"
    assert payload["base_name"] == "ProjectionIdentity"
    assert "class_names" not in payload
    assert "field_names" not in payload
    assert type(RefactorRecipeOperation.from_dict(payload)) is (
        FactorExactDataclassFieldAuthorityOperation
    )
    assert len(claims) == 1
    assert claims[0].claimed_symbol == "ProjectionIdentity"
    assert claims[0].authority_kind is SemanticAuthorityKind.CLASS_FAMILY
    assert simulation.is_clean is True
    assert rewritten.count("module_name: str") == 1
    assert rewritten.count("file_path: str") == 1
    assert "class ProjectionIdentity:" in rewritten
    assert "    file_path: str\n\n\n@dataclass(frozen=True)" in rewritten
    assert (
        '    """Projection with behavior after the promoted fields."""\n\n'
        "    def endpoint"
    ) in rewritten
    assert "class EpsilonProjection(ProjectionIdentity):\n    pass\n\n\n@dataclass" in (
        rewritten
    )
    assert (
        "class ZetaProjection(ProjectionIdentity):\n"
        '    """Projection documented without additional behavior."""\n\n\n'
        "@dataclass"
    ) in rewritten
    assert all(
        f"class {class_name}(ProjectionIdentity):" in rewritten
        for class_name in (
            "AlphaProjection",
            "BetaProjection",
            "DeltaProjection",
            "EpsilonProjection",
            "ZetaProjection",
            "GammaProjection",
        )
    )

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


def test_staged_operations_factor_nested_field_authorities_without_local_minimum(
    tmp_path: Path,
) -> None:
    module_path = _write_fixture(tmp_path, source=_lattice_source())
    before = _lattice_runtime_output(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    def stage(
        recipe_id: str,
        class_name: str,
        evidence_field_name: str,
        base_name: str,
    ) -> CodemodPlanDocument:
        return CodemodPlanDocument(
            recipes=(
                RefactorRecipe(recipe_id=recipe_id).with_operation(
                    FactorExactDataclassFieldAuthorityOperation(
                        target=SourceRewriteTarget(
                            file_path=module_path.as_posix(),
                            qualname=class_name,
                        ),
                        evidence_field_name=evidence_field_name,
                        base_name=base_name,
                    )
                ),
            )
        )

    sequence = CodemodPlanSequence(
        documents=(
            stage(
                "factor-qualified-identity",
                "AlphaProjection",
                "qualname",
                "QualifiedProjectionIdentity",
            ),
            stage(
                "factor-projection-identity",
                "GammaProjection",
                "file_path",
                "ProjectionIdentity",
            ),
        )
    )

    simulation = sequence.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]

    assert simulation.is_clean is True
    assert rewritten.count("module_name: str") == 1
    assert rewritten.count("file_path: str") == 1
    assert rewritten.count("qualname: str") == 1
    assert "class ProjectionIdentity:" in rewritten
    assert "class QualifiedProjectionIdentity(ProjectionIdentity):" in rewritten
    assert "class AlphaProjection(QualifiedProjectionIdentity):" in rewritten
    assert "class BetaProjection(QualifiedProjectionIdentity):" in rewritten
    assert "class GammaProjection(ProjectionIdentity):" in rewritten
    assert "class DeltaProjection(ProjectionIdentity):" in rewritten

    simulation.apply()
    assert _lattice_runtime_output(tmp_path) == before


def test_operation_rejects_a_drifted_target_component(tmp_path: Path) -> None:
    module_path = _write_fixture(
        tmp_path,
        source=_fixture_source().replace(
            "    file_path: str\n",
            "    source_path: str\n",
            1,
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="belongs to 0 current exact dataclass field authority components",
    ):
        RefactorRecipe(recipe_id="drifted-field-authority").with_operation(
            _operation(module_path)
        ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


def test_operation_rejects_generated_authority_name_collision(tmp_path: Path) -> None:
    source = _fixture_source().replace(
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass\n\nProjectionIdentity = object()\n",
    )
    module_path = _write_fixture(tmp_path, source=source)
    snapshot = CodemodSourceSnapshot.from_modules(tuple(parse_python_modules(tmp_path)))

    with pytest.raises(
        CodemodOperationPreflightError,
        match="authority name 'ProjectionIdentity' is already bound",
    ):
        RefactorRecipe(recipe_id="colliding-field-authority").with_operation(
            _operation(module_path)
        ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)


def test_cli_repository_reproof_skips_unrequested_finding_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nominal_refactor_advisor.cli as cli

    module_path = _write_fixture(tmp_path)
    operation = _operation(module_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "recipe_id": "factor-projection-identity",
                        "operations": [json_report_object(operation)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def reject_analysis(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("exact recipe execution must not analyze findings")

    monkeypatch.setattr(cli, "analyze_modules_with_cache", reject_analysis)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nominal-refactor-advisor",
            tmp_path.as_posix(),
            "--no-cache",
            "--no-structural-overlap",
            "--codemod-plan",
            plan_path.as_posix(),
            "--codemod-simulate",
            "--json",
        ],
    )

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["parse_validation"]["parse_valid"] is True
    assert payload["applied_rewrite_count"] == 1
    assert payload["applied"] is False
    assert payload["plan_sequence_simulation"]["is_clean"] is True
