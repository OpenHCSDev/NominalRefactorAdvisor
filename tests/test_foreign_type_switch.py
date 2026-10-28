"""isinstance tests on one foreign module's classes, scattered across modules."""

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.patterns import PatternId


def _findings(tmp_path: Path, sources: dict[str, str]):
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)
    return [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == "foreign_type_switch"
    ]


def test_ast_tests_with_a_narrower_set_diverge_from_the_wider_one(tmp_path: Path):
    findings = _findings(
        tmp_path,
        {
            "a": "import ast\ndef f(n):\n    return isinstance(n, ast.FunctionDef)\n",
            "b": "import ast\ndef f(n):\n    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))\n",
            "c": (
                "from ast import FunctionDef, AsyncFunctionDef\n"
                "KINDS = FunctionDef | AsyncFunctionDef\n"
                "def f(n):\n    return isinstance(n, KINDS)\n"
            ),
        },
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.pattern_id is PatternId.FOREIGN_TYPE_SWITCH_LEAD
    assert len(finding.evidence) == 3
    assert "`ast`" in finding.summary
    assert "3 isinstance/issubclass sites in 3 module(s)" in finding.summary
    assert "`ast.AsyncFunctionDef | ast.FunctionDef` x2" in finding.summary
    assert (
        "1 sites test a class without its same-shape twin (possible missed case): "
        "`ast.FunctionDef` without `ast.AsyncFunctionDef`"
    ) in finding.summary


def test_repo_declared_class_is_not_foreign(tmp_path: Path):
    sources = {"own": "class Node:\n    pass\n"} | {
        f"m{index}": "from own import Node\ndef f(n):\n    return isinstance(n, Node)\n"
        for index in range(3)
    }
    assert _findings(tmp_path, sources) == []


def test_unresolvable_names_are_not_counted(tmp_path: Path):
    source = "def f(n, kind):\n    return isinstance(n, kind) or isinstance(n, Mystery)\n"
    assert _findings(tmp_path, {f"m{index}": source for index in range(3)}) == []


def test_builtins_are_a_family(tmp_path: Path):
    findings = _findings(
        tmp_path,
        {
            f"m{index}": "def f(x):\n    return isinstance(x, dict)\n"
            for index in range(3)
        },
    )
    assert len(findings) == 1
    assert "`builtins`" in findings[0].summary
    assert "possible missed case" not in findings[0].summary


def test_fewer_sites_than_the_declared_count_make_no_finding(tmp_path: Path):
    source = "import ast\ndef f(n):\n    return isinstance(n, ast.Call)\n"
    assert _findings(tmp_path, {f"m{index}": source for index in range(2)}) == []


def test_a_local_rebinding_of_the_module_name_is_not_foreign(tmp_path: Path):
    source = "def f(n, ast):\n    return isinstance(n, ast.Call)\n"
    assert _findings(tmp_path, {f"m{index}": source for index in range(3)}) == []


def test_a_narrower_set_of_unlike_classes_is_not_a_missed_case(tmp_path: Path):
    findings = _findings(
        tmp_path,
        {
            "m0": "import ast\ndef f(n):\n    return isinstance(n, ast.Name)\n",
            "m1": "import ast\ndef f(n):\n    return isinstance(n, ast.Name)\n",
            "m2": "import ast\ndef f(n):\n    return isinstance(n, (ast.Name, ast.Attribute))\n",
        },
    )
    assert len(findings) == 1
    assert "possible missed case" not in findings[0].summary
