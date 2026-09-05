"""Authored factoring uses binding proofs, independent of source-size estimates."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ExtractSymbolsToNewModuleOperation,
    FactorExactMethodRoleOperation,
    PromoteExactLeafMethodsToAncestorOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _apply_plan(root: Path, plan: CodemodPlanSequence) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(root),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(plan)),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_two_property_leaves_promote_through_the_proved_existing_authority(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "sample.py"
    source = "class Root:\n    value = 7\n\n" + "\n".join(
        f"class {name}(Root):\n"
        "    @property\n"
        "    def result(self):\n"
        "        return self.value\n"
        for name in ("Alpha", "Beta")
    )
    path.write_text(source.replace("\n", newline), encoding="utf-8", newline="")
    command = [
        sys.executable,
        "-c",
        "from sample import Alpha, Beta; print(Alpha().result, Beta().result)",
    ]
    before = subprocess.check_output(command, cwd=tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    component = snapshot.exact_leaf_method_component_builder.required_proven_component(
        "sample.Root"
    )
    assert component.proof.is_proven
    assert not component.compression_certificate.pays_rent
    _apply_plan(
        tmp_path,
        CodemodPlanSequence.from_operations(
            (
                PromoteExactLeafMethodsToAncestorOperation(
                    target=SourceRewriteTarget(
                        file_path=path.as_posix(), qualname="Root"
                    )
                ),
            )
        ),
    )
    assert subprocess.check_output(command, cwd=tmp_path) == before
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from sample import Root, Alpha, Beta; "
            "assert Alpha.result is Beta.result is Root.result",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert path.read_text(encoding="utf-8").count("def result") == 1


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_small_authored_role_can_be_factored_and_extracted_in_one_cli_plan(
    tmp_path: Path, newline: str
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    path = package / "sample.py"
    destination = package / "roles.py"
    source = "\n".join(
        f"class {name}:\n" "    def normalize(self, value): return value.strip()\n"
        for name in ("Alpha", "Beta")
    )
    path.write_text(source.replace("\n", newline), encoding="utf-8", newline="")
    command = [
        sys.executable,
        "-c",
        "from pkg.sample import Alpha, Beta; "
        "print(Alpha().normalize(' A '), Beta().normalize(' B '))",
    ]
    before = subprocess.check_output(command, cwd=tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    component = (
        snapshot.exact_method_role_component_builder.required_component_for_method(
            file_path=path.as_posix(), method_qualname="Alpha.normalize"
        )
    )
    assert not component.compression_certificate.pays_rent
    plan = CodemodPlanSequence.from_operations(
        (
            FactorExactMethodRoleOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Alpha.normalize"
                ),
                base_name="NormalizationRole",
            ),
            ExtractSymbolsToNewModuleOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                symbol_qualnames=("NormalizationRole",),
                destination_path=destination.as_posix(),
            ),
        )
    )
    _apply_plan(tmp_path, plan)
    assert subprocess.check_output(command, cwd=tmp_path) == before
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pkg.sample import Alpha, Beta; "
            "from pkg.roles import NormalizationRole; "
            "assert Alpha.normalize is Beta.normalize is NormalizationRole.normalize",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert "def normalize" not in path.read_text(encoding="utf-8")
    assert destination.read_text(encoding="utf-8").count("def normalize") == 1
