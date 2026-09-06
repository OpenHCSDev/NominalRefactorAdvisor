"""Access projections select lexical owners and preserve surrounding syntax."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ProjectFunctionLocalOperation,
    ProjectFunctionParameterOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize("local", (False, True), ids=("parameter", "local"))
@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_cli_projects_only_selected_owned_accesses(
    tmp_path: Path, local: bool, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from types import SimpleNamespace as Box\n"
        "import json\n"
        "def render(original, replacement):\n"
        + ("    candidate = original\n" if local else "")
        + "    # original.old stays in this comment\n"
        + "    shadow = lambda original: original.old\n"
        + f"    result = {'candidate' if local else 'original'}.old + original.other\n"
        + "    return result, shadow(Box(old=11)), 'original.old'\n"
        + "print(json.dumps(render(Box(old=7, other=3), Box(new=7))))\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)])
    target = SourceRewriteTarget(file_path=str(path), qualname="render")
    operation = (
        ProjectFunctionLocalOperation(
            target=target,
            local_name="candidate",
            attribute_path=("old",),
            projection_source="replacement.new",
        )
        if local
        else ProjectFunctionParameterOperation(
            target=target,
            parameter_name="original",
            attribute_path=("old",),
            projection_source="replacement.new",
        )
    )
    sequence = CodemodPlanSequence.from_operations((operation,))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(sequence)),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    rewritten = path.read_bytes().decode("utf-8")
    assert "result = replacement.new + original.other" in rewritten
    assert "lambda original: original.old" in rewritten
    assert "# original.old stays in this comment" in rewritten
    assert "'original.old'" in rewritten
    assert rewritten.count("\r\n") == (source.count("\r\n"))
    assert subprocess.check_output([sys.executable, str(path)]) == before


@pytest.mark.parametrize(
    "body,error",
    (
        ("return (lambda replacement: original.old)(None)", "captured"),
        ("original.old = 3\n    return original.old", "write"),
        ("original.old += 3\n    return original.old", "write"),
        ("del original.old\n    return original.old", "write"),
        ("return original.other", "no owned reads"),
        ("return (original  # owned comment\n            ).old", "comment"),
    ),
)
def test_access_projection_rejects_unproved_edits_without_mutation(
    tmp_path: Path, body: str, error: str
) -> None:
    path = tmp_path / "probe.py"
    source = f"def render(original, replacement):\n    {body}\n"
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ProjectFunctionParameterOperation(
                target=SourceRewriteTarget(file_path=str(path), qualname="render"),
                parameter_name="original",
                attribute_path=("old",),
                projection_source="replacement.new",
            ),
        )
    )
    with pytest.raises(ValueError, match=error):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode("utf-8")


@pytest.mark.parametrize("projection", ("replacement", "original.current"))
def test_nested_access_projection_preserves_suffixes_and_lexical_scopes(
    tmp_path: Path, projection: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from types import SimpleNamespace as Box\n"
        "import json\n"
        "def render(original, replacement):\n"
        "    def closure():\n"
        "        return original.child.old.value\n"
        "    shadowed = [original.child.old.value for original in []]\n"
        "    return (closure(), shadowed, original.child.other, "
        "f'\u03bc{original.child.old.value}')\n"
        "replacement = Box(value=7)\n"
        "original = Box(child=Box(old=replacement, other=3), current=replacement)\n"
        "print(json.dumps(render(original, replacement)))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    expected = subprocess.check_output([sys.executable, str(path)])
    plan = CodemodPlanSequence.from_operations(
        (
            ProjectFunctionParameterOperation(
                target=SourceRewriteTarget(file_path=str(path), qualname="render"),
                parameter_name="original",
                attribute_path=("child", "old"),
                projection_source=projection,
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    rewritten = path.read_bytes().decode("utf-8")
    assert f"return {projection}.value" in rewritten
    assert f"f'\u03bc{{{projection}.value}}'" in rewritten
    assert "[original.child.old.value for original in []]" in rewritten
    assert subprocess.check_output([sys.executable, str(path)]) == expected


def test_owned_prefix_of_a_deeper_write_is_a_read(tmp_path: Path) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from types import SimpleNamespace as Box\n"
        "def render(original, replacement):\n"
        "    original.old.value = 13\n"
        "    return original.old.value\n"
        "replacement = Box(value=7)\n"
        "print(render(Box(old=replacement), replacement))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    expected = subprocess.check_output([sys.executable, str(path)])
    plan = CodemodPlanSequence.from_operations(
        (
            ProjectFunctionParameterOperation(
                target=SourceRewriteTarget(file_path=str(path), qualname="render"),
                parameter_name="original",
                attribute_path=("old",),
                projection_source="replacement",
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert "replacement.value = 13" in path.read_text()
    assert subprocess.check_output([sys.executable, str(path)]) == expected
