from __future__ import annotations

import json
from pathlib import Path

from nominal_refactor_advisor import SemanticInspectionReport, inspect_paths
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.models import FindingSpec, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.semantic_inspection import inspect_modules


def _write_module(root: Path, relative_path: str, source: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _finding_spec() -> FindingSpec:
    return FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Collapse repeated class family",
        why="Repeated behavior has one grammar.",
        capability_gap="certified grammar compression",
        relation_context="same orbit under renaming",
    )


def test_semantic_inspection_summarizes_ast_and_source_index_targets(
    tmp_path: Path,
) -> None:
    module_path = _write_module(
        tmp_path,
        "pkg/mod.py",
        "from dataclasses import dataclass\n"
        "import os as operating\n"
        "from collections import defaultdict as dd\n"
        "\n"
        "@dataclass\n"
        "class Payload:\n"
        "    name: str\n"
        "    count: int = 0\n"
        "\n"
        "    def build(self, suffix):\n"
        "        target = f'{self.name}{suffix}'\n"
        "        return dd(list)\n"
        "\n"
        "def helper(raw):\n"
        "    payload = Payload(raw)\n"
        "    result = payload.build('x')\n"
        "    return result\n",
    )
    modules = parse_python_modules(tmp_path)
    finding = _finding_spec().build(
        "semantic_test_detector",
        "Payload.build should move behind a nominal interface",
        (SourceLocation(module_path.as_posix(), 11, "Payload.build"),),
    )

    report = inspect_modules(modules, findings=(finding,))

    assert isinstance(report, SemanticInspectionReport)
    assert len(report.modules) == 1
    assert report.modules[0].file_path == module_path.as_posix()
    assert report.modules[0].ast_target_ids

    payload_class = next(item for item in report.classes if item.name == "Payload")
    payload_dataclass = next(
        item for item in report.dataclasses if item.name == "Payload"
    )
    build_method = next(item for item in report.functions if item.name == "build")
    helper_function = next(item for item in report.functions if item.name == "helper")

    assert payload_class.is_dataclass
    assert payload_dataclass.field_names == ("name", "count")
    assert payload_dataclass.method_ids == (build_method.target_id,)
    assert build_method.kind == "method"
    assert build_method.parameters == ("self", "suffix")
    assert helper_function.kind == "function"

    import_names = {name for item in report.imports for name in item.alias_names}
    assert {"dataclass", "operating", "dd"} <= import_names

    assignment_names = {
        name for item in report.assignments for name in item.target_names
    }
    assert {"name", "count", "target", "payload", "result"} <= assignment_names
    assert any(item.callee == "dd" for item in report.calls)
    assert any(item.callee == "Payload" for item in report.calls)
    assert any(item.callee == "payload.build" for item in report.calls)

    finding_summary = report.findings[0]
    evidence_summary = report.evidence[0]
    assert finding_summary.finding_id == finding.stable_id
    assert build_method.target_id in finding_summary.target_ids
    assert evidence_summary.target_ids == (build_method.target_id,)
    assert report.source_index.target_by_id[build_method.target_id].qualname == (
        "Payload.build"
    )

    json.dumps(report.to_dict())


def test_inspect_paths_loads_and_analyzes_roots(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        "pkg/mod.py",
        "class Loader:\n" "    def run(self):\n" "        return make_value()\n",
    )

    report = inspect_paths((module_path,))

    assert report.roots == (module_path.as_posix(),)
    assert report.modules[0].module_name == "mod"
    assert any(item.qualname == "Loader.run" for item in report.functions)
