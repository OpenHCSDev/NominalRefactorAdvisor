"""A scan budget follows actual scan work, not every CLI invocation."""

import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor import cli
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    InsertClassMemberOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object


def recipe(path: Path) -> dict:
    return {
        "recipes": [
            {
                "recipe_id": "exact",
                "operations": [
                    {
                        "operation": "ensure_import",
                        "file_path": str(path),
                        "import_source": "import math\n",
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "flag", ("--codemod-preflight", "--codemod-simulate", "--codemod-apply")
)
def test_exact_recipe_consumes_stdin_once_without_scan_deadline(
    tmp_path, monkeypatch, flag
):
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 1\n")
    document = json.dumps(recipe(source))
    monkeypatch.setattr(sys, "stdin", io.StringIO(document))
    invocation = cli.CliArguments.from_argv(
        (
            str(source),
            "--codemod-plan",
            "-",
            flag,
            "--scan-budget-seconds",
            "0.001",
        )
    )
    assert invocation.scan_deadline_request is None
    sequence = invocation.codemod_plan_sequence
    assert sequence.has_recipes
    assert invocation.codemod_plan_sequence is sequence
    assert invocation.scan_deadline_request is None
    assert sys.stdin.tell() == len(document)


def test_projecting_findings_keeps_the_actual_scan_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps(recipe(tmp_path / "sample.py")))
    )
    invocation = cli.CliArguments.from_argv(
        (
            "--codemod-plan",
            "-",
            "--codemod-simulate",
            "--codemod-project-findings",
            "--scan-budget-seconds=1.25",
            "--json",
        )
    )
    request = invocation.scan_deadline_request
    assert request == cli.CliScanDeadlineRequest(1.25, True)


def test_scan_default_and_abbreviated_flags_use_the_actual_parser():
    invocation = cli.CliArguments.from_argv(
        ("sample.py", "--scan-budget-sec=1.75", "--json")
    )
    assert invocation.scan_deadline_request == cli.CliScanDeadlineRequest(1.75, True)
    default = cli.CliArguments.from_argv(("sample.py",))
    assert (
        default.scan_deadline_request.budget_seconds == default.args.scan_budget_seconds
    )


@pytest.mark.parametrize("owner", ("document", "recipe"))
@pytest.mark.parametrize(
    "flag", ("--codemod-preflight", "--codemod-simulate", "--codemod-apply")
)
def test_architecture_guard_retains_scan_deadline_at_each_plan_level(
    tmp_path, monkeypatch, owner, flag
):
    document = recipe(tmp_path / "sample.py")
    guarded = document if owner == "document" else document["recipes"][0]
    guarded["architecture_guards"] = [
        {
            "rule_id": "owned-boundary",
            "constraints": [
                {"constraint": "forbidden_attributes", "names": ["legacy_value"]}
            ],
        }
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(document)))
    invocation = cli.CliArguments.from_argv(
        ("--codemod-plan", "-", flag, "--scan-budget-seconds=1.25", "--json")
    )
    assert invocation.codemod_plan_sequence.has_architecture_guards
    assert invocation.scan_deadline_request == cli.CliScanDeadlineRequest(1.25, True)


@pytest.mark.parametrize(
    "argv",
    (
        ("--detector-capabilities",),
        ("--calibrate", "cases.json"),
        ("--predict-scan",),
        ("--prove-economics",),
        ("--codemod-validate-plan", "--codemod-plan", "-"),
    ),
)
def test_early_command_does_not_read_plan_or_acquire_standard_deadline(
    argv, monkeypatch
):
    def unexpected_read(*args):
        raise AssertionError("Command owns its own input consumption")

    monkeypatch.setattr(cli, "load_codemod_plan_sequence", unexpected_read)
    invocation = cli.CliArguments.from_argv((*argv, "--scan-budget-seconds=0.001"))
    assert invocation.scan_deadline_request is None


def test_cli_parses_once_for_real_early_command(monkeypatch, capsys):
    original = argparse.ArgumentParser.parse_args
    calls = []

    def record(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", record)
    monkeypatch.setattr(
        sys, "argv", ["nra", "--codemod-validate-plan", "--codemod-plan", "-"]
    )
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps(recipe(Path("sample.py"))))
    )
    assert cli.main() == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["recipes"][0]["recipe_id"] == "exact"


@pytest.mark.parametrize("apply", (False, True))
def test_actual_recipe_process_is_not_interrupted_by_scan_budget(tmp_path, apply):
    source = tmp_path / "sample.py"
    original = "VALUE = 1\n"
    source.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(source),
            "--codemod-plan",
            "-",
            "--codemod-apply" if apply else "--codemod-simulate",
            "--scan-budget-seconds=0.000001",
            "--no-cache",
            "--json",
        ],
        input=json.dumps(recipe(source)),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["applied"] is apply
    if apply:
        assert ast.dump(ast.parse(source.read_text())) == ast.dump(
            ast.parse("import math\n" + original)
        )
    else:
        assert source.read_text() == original


@pytest.mark.parametrize("apply", (False, True))
def test_actual_cli_chains_edits_against_the_preceding_stage(tmp_path, apply):
    source = tmp_path / "sample.py"
    original = "class Example:\n    pass\n"
    source.write_text(original)
    sequence = CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(file_path=str(source), qualname="Example"),
                source="def evaluate(self, value):\n    return value\n",
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=str(source), qualname="Example.evaluate"
                ),
                replacements=(
                    SourceTextReplacement(
                        old_source="return value", new_source="return value * 2"
                    ),
                ),
            ),
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(source),
            "--codemod-plan",
            "-",
            "--codemod-apply" if apply else "--codemod-simulate",
            "--scan-budget-seconds=0.000001",
            "--no-cache",
            "--json",
        ],
        input=json.dumps(json_report_object(sequence)),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["applied"] is apply
    report = payload["plan_sequence_simulation"]
    assert report["stage_count"] == 2
    assert all(
        not stage["preflight_report"]["preflight_failed"] for stage in report["stages"]
    )
    if apply:
        tree = ast.parse(source.read_text())
        method = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )
        expected = ast.parse("def evaluate(self, value):\n    return value * 2\n").body[
            0
        ]
        assert ast.dump(method) == ast.dump(expected)
    else:
        assert source.read_text() == original
