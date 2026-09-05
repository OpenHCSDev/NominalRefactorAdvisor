"""Authored base mutations retain MRO position and exact class-suite boundaries."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodOperationPreflightError,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    RefactorRecipeOperation,
    RemoveClassBaseOperation,
    ReplaceClassBaseOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevisionError


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    "header", ("class Worker(Old, Tail):", "class Worker(\n    Old,\n    Tail,\n):")
)
def test_replacement_preserves_inline_suite_and_mro(
    tmp_path: Path, newline: str, header: str
) -> None:
    path = tmp_path / "family.py"
    source = (
        "class Old: pass\nclass New: pass\nclass Tail: pass\n"
        + header
        + " value = 'café'; marker = 7 # retain\n"
        "print([base.__name__ for base in Worker.__mro__])\n"
    ).replace("\n", newline)
    path.write_bytes(source.encode())
    target = SourceRewriteTarget(file_path=path.as_posix(), qualname="Worker")
    operation = ReplaceClassBaseOperation(
        target=target, base_name="Old", replacement_base_name="New"
    )
    restored = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    assert restored == operation
    result = CodemodPlanSequence.from_operations((restored,)).simulate(
        CodemodSourceSnapshot.from_source_mapping({path.as_posix(): source})
    )
    assert result.is_clean
    rewritten = result.simulation.rewritten_sources[path.as_posix()]
    assert rewritten.endswith(
        " value = 'café'; marker = 7 # retain"
        + newline
        + "print([base.__name__ for base in Worker.__mro__])"
        + newline
    )
    result.apply()
    output = subprocess.check_output([sys.executable, str(path)], text=True)
    assert ast.literal_eval(output) == ["Worker", "New", "Tail", "object"]


@pytest.mark.parametrize(
    "operation_type,base_name,expected",
    (
        (AddClassBaseOperation, "Tail", "Old, Tail"),
        (RemoveClassBaseOperation, "Old", ""),
    ),
)
def test_add_remove_preserve_nested_inline_suites(
    operation_type, base_name, expected
) -> None:
    source = "class Outer:\n\tclass Worker(Old): value = 'kept' # trailing\n"
    operation = operation_type(
        target=SourceRewriteTarget(file_path="probe.py", qualname="Outer.Worker"),
        base_name=base_name,
    )
    result = CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_source_mapping({"probe.py": source})
    )
    bases = f"({expected})" if expected else ""
    assert result.simulation.rewritten_sources["probe.py"] == (
        f"class Outer:\n\tclass Worker{bases}: value = 'kept' # trailing\n"
    )


def test_replacement_preserves_decorators_keywords_and_following_comments() -> None:
    source = (
        "@decorate\nclass Worker(Old, metaclass=Meta): # reason\n    # body\n    pass\n"
    )
    operation = ReplaceClassBaseOperation(
        target=SourceRewriteTarget(file_path="probe.py", qualname="Worker"),
        base_name="Old",
        replacement_base_name="New",
    )
    result = CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_source_mapping({"probe.py": source})
    )
    assert result.simulation.rewritten_sources["probe.py"] == source.replace(
        "Old", "New"
    )


@pytest.mark.parametrize(
    "source,old,new,message",
    (
        ("class Worker(Old, New): pass\n", "Old", "New", "already contains"),
        ("class Worker(Old): pass\n", "Absent", "New", "requires one base"),
        ("class Worker(Old, # reason\n): pass\n", "Old", "New", "discard comments"),
    ),
)
def test_replacement_rejects_ambiguous_or_lossy_headers(
    source, old, new, message
) -> None:
    operation = ReplaceClassBaseOperation(
        target=SourceRewriteTarget(file_path="probe.py", qualname="Worker"),
        base_name=old,
        replacement_base_name=new,
    )
    with pytest.raises(CodemodOperationPreflightError, match=message):
        CodemodPlanSequence.from_operations((operation,)).simulate(
            CodemodSourceSnapshot.from_source_mapping({"probe.py": source})
        )


def test_parser_cache_and_cli_preserve_exact_source_newlines(tmp_path: Path) -> None:
    path = tmp_path / "family.py"
    source = "class Old: pass\r\nclass New: pass\r\nclass Worker(Old): value = 7\r\n"
    path.write_bytes(source.encode())
    for _ in range(2):
        modules = parse_python_modules(path, cache_dir=tmp_path / "cache")
        assert modules[0].source == source
    snapshot = CodemodSourceSnapshot.from_modules(modules)
    operation = ReplaceClassBaseOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Worker"),
        base_name="Old",
        replacement_base_name="New",
    )
    sequence = CodemodPlanSequence.from_operations((operation,))
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(path),
            "--codemod-plan",
            "-",
            "--codemod-simulate",
            "--json",
        ],
        input=json.dumps(json_report_object(sequence)),
        text=True,
        capture_output=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["plan_sequence_simulation"]["is_clean"]
    result = sequence.simulate(snapshot)
    path.write_bytes(source.replace("\r\n", "\n").encode())
    with pytest.raises(CodemodSourceRevisionError, match="changed after simulation"):
        result.apply()
    path.write_bytes(source.encode())
    result.apply()
    assert path.read_bytes() == source.replace("Worker(Old)", "Worker(New)").encode()


def test_header_geometry_skips_colons_inside_base_expressions() -> None:
    source = "class Worker(factory(lambda: {'base': Old}), metaclass=type): value = 7\n"
    operation = ReplaceClassBaseOperation(
        target=SourceRewriteTarget(file_path="probe.py", qualname="Worker"),
        base_name="factory(lambda: {'base': Old})",
        replacement_base_name="New",
    )
    result = CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_source_mapping({"probe.py": source})
    )
    assert result.simulation.rewritten_sources["probe.py"] == (
        "class Worker(New, metaclass=type): value = 7\n"
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="native generic class syntax")
def test_base_replacement_retains_native_type_parameters(tmp_path: Path) -> None:
    source = "class Old: pass\nclass New: pass\nclass Worker[T: int](Old): value = 7\n"
    path = tmp_path / "generic.py"
    path.write_text(source, newline="")
    operation = ReplaceClassBaseOperation(
        target=SourceRewriteTarget(file_path=path.as_posix(), qualname="Worker"),
        base_name="Old",
        replacement_base_name="New",
    )
    result = CodemodPlanSequence.from_operations((operation,)).simulate(
        CodemodSourceSnapshot.from_source_mapping({path.as_posix(): source})
    )
    result.apply()
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import generic; import json; print(json.dumps([generic.Worker.__bases__[0].__name__, generic.Worker.__type_params__[0].__name__]))",
        ],
        cwd=tmp_path,
        text=True,
    )
    assert json.loads(output) == ["New", "T"]
