"""Adjacent edits retain native decorator ownership, not merely valid syntax."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeriveCandidateCollectorOperation,
    DispatchToPolymorphismOperation,
    InsertAfterTargetOperation,
    InsertBeforeTargetOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize(
    "operation_type", (InsertBeforeTargetOperation, InsertAfterTargetOperation)
)
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("parenthesized", (False, True))
@pytest.mark.parametrize(
    "qualname,scope,declaration,inserted",
    (
        ("Existing", "", "class Existing: pass\n", "class Added: pass\n"),
        ("existing", "", "def existing(): return 42\n", "def added(): return 1\n"),
        (
            "existing",
            "",
            "async def existing(): return 42\n",
            "def added(): return 1\n",
        ),
        (
            "Owner.existing",
            "class Owner:\n",
            "    def existing(self): return 42\n",
            "    def added(self): return 1\n",
        ),
    ),
)
def test_cli_preserves_decorator_evaluation_and_owner(
    tmp_path: Path,
    operation_type,
    newline: str,
    parenthesized: bool,
    qualname: str,
    scope: str,
    declaration: str,
    inserted: str,
) -> None:
    path = tmp_path / "probe.py"
    indentation = "    " if scope else ""
    decorator = (
        "@(\n    tag('outer')\n)\n" if parenthesized else "@tag('outer')\n"
    ) + "# between decorators\n@tag('inner')\n"
    decorated_source = (
        "".join(indentation + line for line in decorator.splitlines(keepends=True))
        + declaration
    )
    source = (
        "events = []\n"
        "def tag(label):\n"
        "    def decorate(declaration):\n"
        "        events.append((label, declaration.__qualname__))\n"
        "        return declaration\n"
        "    return decorate\n" + scope + decorated_source + "print(events)\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            operation_type(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname=qualname
                ),
                source=inserted.replace("\n", newline),
            ),
        )
    )
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
        input=json.dumps(json_report_object(plan)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    assert decorated_source.replace("\n", newline) in path.read_bytes().decode()


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_generated_dispatch_family_keeps_parenthesized_decorator_on_function(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "events = []\n"
        "def traced(function):\n"
        "    events.append(function.__name__)\n"
        "    return function\n"
        "@(\n    traced\n)\n"
        "def render(kind, value):\n"
        "    if kind == 'left': return value + 1\n"
        "    elif kind == 'right': return value + 2\n"
        "    raise ValueError(kind)\n"
        "print(render('left', 3), render('right', 3), events)\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            DispatchToPolymorphismOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="render"
                ),
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize("decorator", ("@changed", "@(\n        changed\n    )"))
def test_collector_migration_does_not_erase_decorated_forwarder_behaviour(
    tmp_path: Path, decorator: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from typing import Generic, TypeVar\n"
        "T = TypeVar('T')\n"
        "class CrossModuleCandidateDetector(Generic[T]): pass\n"
        "class CrossModuleCollectorCandidateDetector(CrossModuleCandidateDetector[T]):\n"
        "    def _candidate_items(self, modules, config): return self.candidate_collector(modules)\n"
        "def collect(modules): return ('original',)\n"
        "def changed(function):\n"
        "    def wrapper(self, modules, config): return ('decorated',)\n"
        "    return wrapper\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        f"    {decorator}\n"
        "    def _candidate_items(self, modules, config):\n"
        "        del config\n"
        "        return collect(modules)\n"
        "print(Owner()._candidate_items([], None))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    assert before.strip() == "('decorated',)"
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="witnesses"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize("inserted", ("MARKER = object()\n", "class Added: pass\n"))
def test_inserting_before_dataclass_cannot_move_its_generated_constructor(
    tmp_path: Path, inserted: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from dataclasses import dataclass, is_dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Existing:\n    value: int\n"
        "print(Existing(42), is_dataclass(Existing))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            InsertBeforeTargetOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Existing"
                ),
                source=inserted,
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_collector_declaration_preserves_decorated_findings_anchor(
    tmp_path: Path, newline: str, native_collector_module: ParsedModule
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from nominal_refactor_advisor.detectors._base import CrossModuleCandidateDetector\n"
        "events = []\n"
        "def collect(modules): return ('original',)\n"
        "def traced(function):\n"
        "    events.append(function.__name__)\n"
        "    return function\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config):\n"
        "        del config\n"
        "        return collect(modules)\n"
        "    @(\n        traced\n    )\n"
        "    def _collect_findings(self): return 'findings'\n"
        "print(Owner()._candidate_items([], None), Owner()._collect_findings(), events)\n"
    ).replace("\n", newline)
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                ),
            ),
        )
    )
    simulation = plan.simulate(
        CodemodSourceSnapshot.from_modules(
            (*parse_python_modules(tmp_path), native_collector_module)
        )
    )
    assert simulation.is_clean
    simulation.apply()
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
