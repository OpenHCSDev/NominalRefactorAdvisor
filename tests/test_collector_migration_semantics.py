"""Collector migration must preserve native forwarding and binding semantics."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeriveCandidateCollectorOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


@pytest.mark.parametrize(
    "declarations,class_prefix,signature,body,after,invocation",
    (
        (
            "def collect(modules, *, mode='default'): return mode\n",
            "",
            "self, modules, config",
            "return collect(modules, mode='chosen')",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules, **options): return options\n",
            "",
            "self, modules, config",
            "return collect(modules, **{'mode': 'chosen'})",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'global'\n"
            "class Namespace:\n    @staticmethod\n    def collect(modules): return 'qualified'\n",
            "",
            "self, modules, config",
            "return Namespace.collect(modules)",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'global'\n",
            "    collect = lambda modules: 'class'\n"
            "    def _collect_findings(self): pass\n",
            "self, modules, config",
            "return collect(modules)",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'early'\n",
            "",
            "self, modules, config",
            "return collect(modules)",
            "def collect(modules): return 'late'\n",
            "Owner()._candidate_items([], None)",
        ),
        (
            "",
            "",
            "self, modules, config",
            "return collect(modules)",
            "def collect(modules): return 'late'\n",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'default'\n",
            "",
            "self, modules, config=None",
            "return collect(modules)",
            "",
            "Owner()._candidate_items([])",
        ),
        (
            "def collect(modules): return 'original'\n",
            "",
            "self, modules, config",
            "del config, modules\n        return collect(modules)",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'keyword contract'\n",
            "",
            "self, modules, settings",
            "return collect(modules)",
            "",
            "Owner()._candidate_items([], settings=None)",
        ),
        (
            "def collect(modules): return 'original'\n"
            "def staticmethod(function): return lambda *args: 'wrong descriptor'\n",
            "",
            "self, modules, config",
            "return collect(modules)",
            "",
            "Owner()._candidate_items([], None)",
        ),
        (
            "def collect(modules): return 'original'\n",
            "    staticmethod = lambda function: lambda *args: 'wrong descriptor'\n"
            "    def _collect_findings(self): pass\n",
            "self, modules, config",
            "return collect(modules)",
            "",
            "Owner()._candidate_items([], None)",
        ),
    ),
    ids=(
        "keyword",
        "unpacking",
        "qualified",
        "class-shadow",
        "rebound",
        "late",
        "default",
        "extra-delete",
        "parameter-rename",
        "module-descriptor-shadow",
        "class-descriptor-shadow",
    ),
)
def test_migration_preserves_native_outcome_or_refuses_without_writing(
    tmp_path: Path,
    declarations: str,
    class_prefix: str,
    signature: str,
    body: str,
    after: str,
    invocation: str,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "from typing import Generic, TypeVar\n"
        "T = TypeVar('T')\n"
        "class CrossModuleCandidateDetector(Generic[T]): pass\n"
        "class CrossModuleCollectorCandidateDetector(CrossModuleCandidateDetector[T]):\n"
        "    def _candidate_items(self, modules, config): return self.candidate_collector(modules)\n"
        + declarations
        + "class Owner(CrossModuleCandidateDetector[int]):\n"
        + class_prefix
        + f"    def _candidate_items({signature}):\n        {body}\n"
        + after
        + f"try:\n    print({invocation})\nexcept Exception as error:\n    print(type(error).__name__)\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert before.returncode == 0, before.stderr
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
    try:
        simulation = plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    except ValueError:
        assert path.read_bytes() == source.encode()
        return
    assert simulation.is_clean
    simulation.apply()
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == before.stdout


@pytest.mark.parametrize("configured", (False, True))
@pytest.mark.parametrize(
    "descriptor_import",
    ("", "from builtins import staticmethod\n", "from provider import *\n"),
)
def test_cli_preserves_stable_imported_collector_alias(
    tmp_path: Path, configured: bool, descriptor_import: str
) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text(
        "__all__ = ('collect',)\n"
        "def collect(modules, config=None): return (modules, config)\n",
        encoding="utf-8",
        newline="",
    )
    path = tmp_path / "probe.py"
    collector_base = (
        "ConfiguredCrossModuleCollectorCandidateDetector"
        if configured
        else "CrossModuleCollectorCandidateDetector"
    )
    arguments = "modules, config" if configured else "modules"
    source = (
        descriptor_import + "from typing import Generic, TypeVar\n"
        "from provider import collect as acquire\n"
        "T = TypeVar('T')\n"
        "class CrossModuleCandidateDetector(Generic[T]): pass\n"
        f"class {collector_base}(CrossModuleCandidateDetector[T]):\n"
        f"    def _candidate_items(self, modules, config): return self.candidate_collector({arguments})\n"
        "class Owner(CrossModuleCandidateDetector[int]):\n"
        "    def _candidate_items(self, modules, config):\n"
        f"        return acquire({arguments})\n"
        "print(Owner()._candidate_items([1], config=42))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            DeriveCandidateCollectorOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner._candidate_items"
                )
            ),
        )
    )
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
    assert json.loads(result.stdout)["applied"]
    assert (
        "def _candidate_items(self, modules, config):\n        return acquire"
        not in path.read_text()
    )
    assert subprocess.check_output([sys.executable, str(path)], text=True) == before
