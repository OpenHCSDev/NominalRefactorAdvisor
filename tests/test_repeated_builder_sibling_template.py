"""Tests for Pattern 14 sibling receipt-constructor template-method hoisting."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod_runtime import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodSourceSnapshot,
    FindingRecipeSynthesisStatus,
)
from nominal_refactor_advisor.json_reports import json_report_object

REPEATED_BUILDER_CALLS_DETECTOR_ID = "repeated_builder_calls"


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_sibling_receipt_template_hoists_abstract_base_and_early_returns(
    tmp_path: Path,
) -> None:
    """Multiple sibling overrides returning the same receipt dataclass with invariant
    `check=type(self)` (including early-return exits) lift into a base template method
    calling an `@abstractmethod` hook, with deterministic replay and zero remaining findings."""

    source = (
        "from abc import ABC, abstractmethod\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Wave5CheckReceipt:\n"
        "    check: type\n"
        "    passed: bool\n"
        "    detail: str\n\n\n"
        "class Wave5InvariantCheck(ABC):\n"
        "    @abstractmethod\n"
        "    def evaluate(self, strict: bool = False) -> Wave5CheckReceipt:\n"
        "        \"\"\"Run this check and wrap its result in a typed receipt.\"\"\"\n\n\n"
        "class AlphaCheck(Wave5InvariantCheck):\n"
        "    def evaluate(self, strict: bool = False) -> Wave5CheckReceipt:\n"
        "        if strict:\n"
        "            return Wave5CheckReceipt(check=type(self), passed=False, detail='strict-fail')\n"
        "        ok = True\n"
        "        return Wave5CheckReceipt(check=type(self), passed=ok, detail='alpha-ok')\n\n\n"
        "class BetaCheck(Wave5InvariantCheck):\n"
        "    def evaluate(self, strict: bool = False) -> Wave5CheckReceipt:\n"
        "        ok = not strict\n"
        "        return Wave5CheckReceipt(check=type(self), passed=ok, detail='beta')\n\n\n"
        "class GammaCheck(Wave5InvariantCheck):\n"
        "    def evaluate(self, strict: bool = False) -> Wave5CheckReceipt:\n"
        "        return Wave5CheckReceipt(check=type(self), passed=True, detail='gamma')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/checks.py": source})
    modules = parse_python_modules(
        tmp_path,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    assert len(findings) == 1
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(REPEATED_BUILDER_CALLS_DETECTOR_ID,),
    )
    assert plan.records[0].status.value == "executable_candidate"
    assert plan.records[0].evaluation_declaration_name == (
        "SiblingReceiptTemplateDerivation"
    )
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[
        (tmp_path / "pkg/checks.py").as_posix()
    ]
    replay = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert (
        replay.simulation.rewritten_sources[(tmp_path / "pkg/checks.py").as_posix()]
        == rewritten
    )

    preflight = plan.document.preflight_snapshot(snapshot)
    resolution = preflight.reports[0].detail.resolutions[0]
    assert resolution.claim.claimed_symbol == "Wave5InvariantCheck"
    assert resolution.status.value == "resolved"

    assert rewritten.count("Wave5CheckReceipt(") == 1
    assert "passed, detail = self._evaluate_step(strict)" in rewritten
    assert "def _evaluate_step(self, strict: bool=False):" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "checks.py", "exec"), ns)
    Receipt = ns["Wave5CheckReceipt"]
    assert ns["AlphaCheck"]().evaluate(False) == Receipt(ns["AlphaCheck"], True, "alpha-ok")  # type: ignore[operator]
    assert ns["AlphaCheck"]().evaluate(True) == Receipt(ns["AlphaCheck"], False, "strict-fail")  # type: ignore[operator]
    assert ns["BetaCheck"]().evaluate(False) == Receipt(ns["BetaCheck"], True, "beta")  # type: ignore[operator]
    assert ns["BetaCheck"]().evaluate(True) == Receipt(ns["BetaCheck"], False, "beta")  # type: ignore[operator]
    assert ns["GammaCheck"]().evaluate(True) == Receipt(ns["GammaCheck"], True, "gamma")  # type: ignore[operator]

    simulation.document_simulation.apply()
    assert not any(
        finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
        for finding in analyze_modules(
            parse_python_modules(
                tmp_path,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            )
        )
    )


def test_sibling_receipt_template_preserves_concrete_base_and_classmethod_single_slot(
    tmp_path: Path,
) -> None:
    """A concrete base returning the same receipt keeps its own body as the concrete
    default hook (preserving direct callers and non-overriding descendants), and a
    `@classmethod` family with one varying slot returns that value directly without a tuple."""

    source = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class StatusReceipt:\n"
        "    owner: type\n"
        "    code: str\n"
        "    ok: bool\n\n\n"
        "class BaseCheck:\n"
        "    def evaluate(self) -> StatusReceipt:\n"
        "        return StatusReceipt(owner=type(self), code='base', ok=True)\n\n\n"
        "class InheritingCheck(BaseCheck):\n"
        "    pass\n\n\n"
        "class FastCheck(BaseCheck):\n"
        "    def evaluate(self) -> StatusReceipt:\n"
        "        return StatusReceipt(owner=type(self), code='fast', ok=True)\n\n\n"
        "class StrictCheck(BaseCheck):\n"
        "    def evaluate(self) -> StatusReceipt:\n"
        "        return StatusReceipt(owner=type(self), code='strict', ok=True)\n\n\n"
        "def run_direct() -> StatusReceipt:\n"
        "    return BaseCheck().evaluate()\n\n\n"
        "@dataclass(frozen=True)\n"
        "class TagRecord:\n"
        "    owner: type\n"
        "    label: str\n"
        "    version: int\n\n\n"
        "class BaseTag:\n"
        "    pass\n\n\n"
        "class TagA(BaseTag):\n"
        "    @classmethod\n"
        "    def build(cls, suffix: str) -> TagRecord:\n"
        "        return TagRecord(owner=cls, label=f'a:{suffix}', version=1)\n\n\n"
        "class TagB(BaseTag):\n"
        "    @classmethod\n"
        "    def build(cls, suffix: str) -> TagRecord:\n"
        "        return TagRecord(owner=cls, label=f'b:{suffix}', version=1)\n\n\n"
        "class TagC(BaseTag):\n"
        "    @classmethod\n"
        "    def build(cls, suffix: str) -> TagRecord:\n"
        "        return TagRecord(owner=cls, label=f'c:{suffix}', version=1)\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/receipts.py": source})
    modules = parse_python_modules(
        tmp_path,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    assert len(findings) == 2
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(REPEATED_BUILDER_CALLS_DETECTOR_ID,),
    )
    assert all(rec.status.value == "executable_candidate" for rec in plan.records)
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    rewritten = simulation.simulation.rewritten_sources[
        (tmp_path / "pkg/receipts.py").as_posix()
    ]
    assert rewritten.count("StatusReceipt(") == 1
    assert rewritten.count("TagRecord(") == 1
    assert "code = self._evaluate_step()" in rewritten
    assert "label = cls._build_step(suffix)" in rewritten

    ns: dict[str, object] = {}
    exec(compile(rewritten, "receipts.py", "exec"), ns)
    StatusReceipt = ns["StatusReceipt"]
    TagRecord = ns["TagRecord"]
    assert ns["run_direct"]() == StatusReceipt(ns["BaseCheck"], "base", True)  # type: ignore[operator]
    assert ns["InheritingCheck"]().evaluate() == StatusReceipt(ns["InheritingCheck"], "base", True)  # type: ignore[operator]
    assert ns["FastCheck"]().evaluate() == StatusReceipt(ns["FastCheck"], "fast", True)  # type: ignore[operator]
    assert ns["StrictCheck"]().evaluate() == StatusReceipt(ns["StrictCheck"], "strict", True)  # type: ignore[operator]
    assert ns["TagA"].build("1") == TagRecord(ns["TagA"], "a:1", 1)  # type: ignore[operator]
    assert ns["TagB"].build("2") == TagRecord(ns["TagB"], "b:2", 1)  # type: ignore[operator]
    assert ns["TagC"].build("3") == TagRecord(ns["TagC"], "c:3", 1)  # type: ignore[operator]


def test_sibling_receipt_template_rejects_non_overriding_concrete_descendant(
    tmp_path: Path,
) -> None:
    """When a concrete base does NOT itself construct the receipt and a concrete
    descendant inherits the base method without overriding it, hoisting stays refused."""

    source = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Receipt:\n"
        "    check: type\n"
        "    passed: bool\n"
        "    detail: str\n\n\n"
        "class BaseCheck:\n"
        "    def evaluate(self) -> Receipt | None:\n"
        "        return None\n\n\n"
        "class InheritingCheck(BaseCheck):\n"
        "    pass\n\n\n"
        "class CheckA(BaseCheck):\n"
        "    def evaluate(self) -> Receipt | None:\n"
        "        return Receipt(check=type(self), passed=True, detail='a')\n\n\n"
        "class CheckB(BaseCheck):\n"
        "    def evaluate(self) -> Receipt | None:\n"
        "        return Receipt(check=type(self), passed=False, detail='b')\n\n\n"
        "class CheckC(BaseCheck):\n"
        "    def evaluate(self) -> Receipt | None:\n"
        "        return Receipt(check=type(self), passed=True, detail='c')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/unsafe.py": source})
    modules = parse_python_modules(
        tmp_path,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    findings = tuple(
        finding
        for finding in analyze_modules(modules)
        if finding.detector_id == REPEATED_BUILDER_CALLS_DETECTOR_ID
    )
    assert len(findings) == 1
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)
    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=(REPEATED_BUILDER_CALLS_DETECTOR_ID,),
    )
    assert plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
