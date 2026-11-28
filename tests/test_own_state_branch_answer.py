"""End-to-end tests for Pattern 41 (`own_state_branch_answer`) subtype-split codemod."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import codemod_plan_from_findings
from nominal_refactor_advisor.codemod_runtime import (
    CodemodBackend,
    CodemodPlanDocument,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import (
    ConstantSubtypeAnswerDetector,
    OwnStateBranchAnswerDetector,
    ThreadedAbsenceDetector,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    ConstantAnswerPushConsumer,
)
from nominal_refactor_advisor.refactors.own_state_branch_answer import (
    OwnStateBranchAnswerProof,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def _write_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return tmp_path


def _scan(root: Path):
    modules = parse_python_modules(
        root,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    snap = CodemodSourceSnapshot.from_modules(modules)
    with ScanCache.scope():
        findings = OwnStateBranchAnswerDetector().detect(modules, DetectorConfig())
    return modules, snap, findings


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_own_state_external_consumer_static_and_dynamic_ctors_and_replay(tmp_path: Path) -> None:
    """`Review.intervenes` (`verdict == 'INTERVENE'`) splits into `InterveningReview(Review)`,
    pushes `Outcome.of`, deletes `intervenes`, and preserves runtime parity across static & dynamic ctors."""

    repo = _write_repo(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/codriver.py": """from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Review:
    verdict: str | None
    reason: str
    text: str

    @classmethod
    def read(cls, label: str, reason: str, text: str) -> Review:
        if not label:
            return cls(None, "", text)
        return cls(label, reason, text)

    @property
    def intervenes(self) -> bool:
        return self.verdict == "INTERVENE"


@dataclass(frozen=True)
class Outcome:
    blocks: bool
    reason: str

    @classmethod
    def of(cls, review: Review) -> Outcome:
        if not review.intervenes:
            return cls(False, "")
        return cls(True, review.reason or review.text)


def make_direct() -> Review:
    return Review("INTERVENE", "bad", "full")
""",
        },
    )

    _, snap, findings = _scan(repo)
    assert len(findings) == 1
    finding = findings[0]

    plan_no_pick = codemod_plan_from_findings(findings, selector_context=snap)
    assert plan_no_pick.candidate_count == 0
    assert plan_no_pick.report.records[0].evaluation.needs == (NeededInput.OWNER_PICK,)

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="Review.intervenes",
                because="Intervening reviews own outcome construction.",
                name="InterveningReview",
                judgement="Review whose verdict intervenes on the proposed change.",
                hoisted_name="to_outcome",
                hoisted_judgement="Build the outcome for this review.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report

    replay = CodemodPlanDocument.from_json_value(
        json_report_object(plan.document)
    ).simulate(snap, backend=CodemodBackend.AST_SPAN)
    rewritten = sim.simulation.rewritten_sources[str(repo / "pkg/codriver.py")]
    assert replay.simulation.rewritten_sources[str(repo / "pkg/codriver.py")] == rewritten
    sim.apply()

    after_src = (repo / "pkg/codriver.py").read_text(encoding="utf-8")
    assert "class InterveningReview(Review):" in after_src
    assert "def classified(" in after_src
    assert "def intervenes(" not in after_src
    assert 'return cls(None, "", text)' in after_src
    assert "return cls.classified(label, reason, text)" in after_src
    assert 'return InterveningReview("INTERVENE", "bad", "full")' in after_src

    mod = _load_module("_test_p41_codriver", repo / "pkg/codriver.py")
    try:
        r_none = mod.Review.read("", "ignored", "raw")
        r_ok = mod.Review.read("APPROVE", "lgtm", "raw")
        r_int = mod.Review.read("INTERVENE", "stop", "raw")
        r_dir = mod.make_direct()
        assert type(r_none) is mod.Review
        assert type(r_ok) is mod.Review
        assert isinstance(r_int, mod.InterveningReview)
        assert isinstance(r_dir, mod.InterveningReview)
        assert mod.Outcome.of(r_none) == mod.Outcome(False, "")
        assert mod.Outcome.of(r_ok) == mod.Outcome(False, "")
        assert mod.Outcome.of(r_int) == mod.Outcome(True, "stop")
        assert mod.Outcome.of(r_dir) == mod.Outcome(True, "bad")
    finally:
        sys.modules.pop("_test_p41_codriver", None)

    after_mods, _, after_p41 = _scan(repo)
    assert after_p41 == []
    with ScanCache.scope():
        assert ConstantSubtypeAnswerDetector().detect(after_mods, DetectorConfig()) == []
        assert ThreadedAbsenceDetector().detect(after_mods, DetectorConfig()) == []


def test_own_state_internal_whole_method_and_in_place_override(tmp_path: Path) -> None:
    """Whole-function method on `Record` (`Authority.describe`) moves in-place without needing `hoisted_name`."""

    repo = _write_repo(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/auth.py": """from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Authority:
    registry_key_attribute: str | None

    @property
    def keyed_metaclass_family(self) -> bool:
        return self.registry_key_attribute is not None

    def describe(self) -> int:
        if self.keyed_metaclass_family:
            return 1
        return 0


def build(key: str | None) -> Authority:
    return Authority(key)
""",
        },
    )

    _, snap, findings = _scan(repo)
    assert len(findings) == 1
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=findings[0].stable_id,
                owner="Authority.keyed_metaclass_family",
                because="Keyed authorities own keyed description.",
                name="KeyedAuthority",
                judgement="Authority backed by a registry key attribute.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    sim.apply()

    after_src = (repo / "pkg/auth.py").read_text(encoding="utf-8")
    assert "class KeyedAuthority(Authority):" in after_src
    assert "keyed_metaclass_family" not in after_src

    mod = _load_module("_test_p41_auth", repo / "pkg/auth.py")
    try:
        plain = mod.build(None)
        keyed = mod.build("_key")
        assert type(plain) is mod.Authority
        assert isinstance(keyed, mod.KeyedAuthority)
        assert plain.describe() == 0
        assert keyed.describe() == 1
    finally:
        sys.modules.pop("_test_p41_auth", None)

    after_mods, _, after_p41 = _scan(repo)
    assert after_p41 == []
    with ScanCache.scope():
        assert ThreadedAbsenceDetector().detect(after_mods, DetectorConfig()) == []


def test_own_state_comprehension_consumer_and_constantized_data_read(tmp_path: Path) -> None:
    """When non-branching data reads (`summary()`) remain, `passed` is constantized (`False`/`True`)
    while `sum(1 for o in self.outcomes if o.passed)` pushes onto `SlotOutcome` / `PassedSlotOutcome`."""

    repo = _write_repo(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/arena.py": """from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoomVisit:
    passed: bool


@dataclass(frozen=True)
class SlotOutcome:
    slot_id: str
    visit: RoomVisit | None = None

    @property
    def passed(self) -> bool:
        return self.visit is not None and self.visit.passed

    def summary(self) -> dict[str, object]:
        return {"slot_id": self.slot_id, "passed": self.passed}


@dataclass(frozen=True)
class BatchReport:
    outcomes: tuple[SlotOutcome, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)


def run_batch(v1: RoomVisit, v2: RoomVisit | None) -> BatchReport:
    return BatchReport((
        SlotOutcome("a", None),
        SlotOutcome("b", v1),
        SlotOutcome("c", v2),
    ))
""",
        },
    )

    _, snap, findings = _scan(repo)
    assert len(findings) == 1
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=findings[0].stable_id,
                owner="SlotOutcome.passed",
                because="Passed slots carry their passed count directly.",
                name="PassedSlotOutcome",
                judgement="Slot outcome whose room visit passed.",
                hoisted_name="passed_count_contribution",
                hoisted_judgement="Return 1 when this slot outcome passed, else 0.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    sim.apply()

    after_src = (repo / "pkg/arena.py").read_text(encoding="utf-8")
    assert "class PassedSlotOutcome(SlotOutcome):" in after_src
    assert 'SlotOutcome("a", None)' in after_src
    assert 'SlotOutcome.classified("b", v1)' in after_src

    mod = _load_module("_test_p41_arena", repo / "pkg/arena.py")
    try:
        rep = mod.run_batch(mod.RoomVisit(True), mod.RoomVisit(False))
        assert rep.passed_count == 1
        assert [o.summary()["passed"] for o in rep.outcomes] == [False, True, False]
        assert isinstance(rep.outcomes[1], mod.PassedSlotOutcome)
        assert type(rep.outcomes[0]) is mod.SlotOutcome
        assert type(rep.outcomes[2]) is mod.SlotOutcome
    finally:
        sys.modules.pop("_test_p41_arena", None)

    after_mods, _, after_p41 = _scan(repo)
    assert after_p41 == []
    with ScanCache.scope():
        assert ConstantSubtypeAnswerDetector().detect(after_mods, DetectorConfig()) == []
        assert ThreadedAbsenceDetector().detect(after_mods, DetectorConfig()) == []


def test_own_state_constant_false_ancestor_and_cross_module_caller(tmp_path: Path) -> None:
    """`CountPrinted(CheckPrinted)` where `CheckPrinted.is_zero` returns constant `False`
    pushes the default method onto `CheckPrinted`, overrides on `ZeroCountPrinted(CountPrinted)`,
    and deletes `is_zero` from both."""

    repo = _write_repo(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/printed.py": """from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class CheckPrinted(ABC):
    @property
    def is_zero(self) -> bool:
        return False


@dataclass(frozen=True)
class CountPrinted(CheckPrinted):
    count: int

    @property
    def is_zero(self) -> bool:
        return self.count == 0

    def distance(self, expected: int) -> CheckPrinted:
        return CountPrinted(abs(self.count - expected))


def exact_zero() -> CountPrinted:
    return CountPrinted(0)
""",
            "pkg/orders.py": """from __future__ import annotations

from pkg.printed import CheckPrinted


def verdict(printed: CheckPrinted) -> str:
    if not printed.is_zero:
        return "nonzero"
    return "zero"
""",
        },
    )

    _, snap, findings = _scan(repo)
    assert len(findings) == 1
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=findings[0].stable_id,
                owner="CountPrinted.is_zero",
                because="Zero counts own the zero verdict.",
                name="ZeroCountPrinted",
                judgement="CountPrinted whose count is zero.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    sim.apply()

    printed_after = (repo / "pkg/printed.py").read_text(encoding="utf-8")
    orders_after = (repo / "pkg/orders.py").read_text(encoding="utf-8")
    assert "class ZeroCountPrinted(CountPrinted):" in printed_after
    assert "return ZeroCountPrinted(0)" in printed_after
    assert "def is_zero(" not in printed_after
    assert "printed.verdict()" in orders_after

    sys.path.insert(0, str(repo))
    try:
        printed_mod = _load_module("pkg.printed", repo / "pkg/printed.py")
        orders_mod = _load_module("pkg.orders", repo / "pkg/orders.py")
        z = printed_mod.exact_zero()
        nz = printed_mod.CountPrinted(3).distance(1)
        z2 = printed_mod.CountPrinted(3).distance(3)
        assert isinstance(z, printed_mod.ZeroCountPrinted)
        assert isinstance(z2, printed_mod.ZeroCountPrinted)
        assert type(nz) is printed_mod.CountPrinted
        assert orders_mod.verdict(z) == "zero"
        assert orders_mod.verdict(z2) == "zero"
        assert orders_mod.verdict(nz) == "nonzero"
    finally:
        sys.path.remove(str(repo))
        sys.modules.pop("pkg.printed", None)
        sys.modules.pop("pkg.orders", None)

    after_mods, _, after_p41 = _scan(repo)
    assert after_p41 == []
    with ScanCache.scope():
        assert ConstantSubtypeAnswerDetector().detect(after_mods, DetectorConfig()) == []


def test_own_state_negative_soundness_refusals(tmp_path: Path) -> None:
    """Refuse records with existing subclasses, value references (`decode(Record)`),
    and `dataclasses.replace`."""

    # 1. Existing subclass (`desc > 0`)
    repo1 = _write_repo(
        tmp_path / "r1",
        {
            "m.py": """from dataclasses import dataclass

@dataclass(frozen=True)
class Base:
    x: int | None
    @property
    def present(self) -> bool:
        return self.x is not None

@dataclass(frozen=True)
class Child(Base):
    y: int = 1

def use(b: Base) -> int:
    if b.present:
        return 1
    return 0

def mk() -> Base:
    return Base(1)
"""
        },
    )
    _, snap1, _ = _scan(repo1)
    with pytest.raises(NeedsInput, match="already has subclasses"):
        OwnStateBranchAnswerProof.proved(
            snap1,
            SourceRewriteTarget(file_path=str(repo1 / "m.py"), qualname="Base"),
            "present",
            "PresentBase",
            "Present base.",
            (
                ConstantAnswerPushConsumer(
                    target=SourceRewriteTarget(file_path=str(repo1 / "m.py"), qualname="use"),
                    test_line=15,
                    method_name="use",
                ),
            ),
        )

    # 2. Referenced as a value (`decode(Rec, raw)`)
    repo2 = _write_repo(
        tmp_path / "r2",
        {
            "m.py": """from dataclasses import dataclass

@dataclass(frozen=True)
class Rec:
    x: int | None
    @property
    def present(self) -> bool:
        return self.x is not None

def decode(cls, raw):
    return cls(raw)

def use(r: Rec) -> int:
    if r.present:
        return 1
    return 0

def mk(raw):
    return decode(Rec, raw)
"""
        },
    )
    _, snap2, _ = _scan(repo2)
    with pytest.raises(NeedsInput, match="references `Rec` as a value"):
        OwnStateBranchAnswerProof.proved(
            snap2,
            SourceRewriteTarget(file_path=str(repo2 / "m.py"), qualname="Rec"),
            "present",
            "PresentRec",
            "Present rec.",
            (
                ConstantAnswerPushConsumer(
                    target=SourceRewriteTarget(file_path=str(repo2 / "m.py"), qualname="use"),
                    test_line=14,
                    method_name="use",
                ),
            ),
        )

    # 3. `dataclasses.replace` inside record
    repo3 = _write_repo(
        tmp_path / "r3",
        {
            "m.py": """from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Rec:
    x: int | None
    @property
    def present(self) -> bool:
        return self.x is not None
    def cleared(self):
        return replace(self, x=None)

def use(r: Rec) -> int:
    if r.present:
        return 1
    return 0

def mk():
    return Rec(1)
"""
        },
    )
    _, snap3, _ = _scan(repo3)
    with pytest.raises(NeedsInput, match="dataclasses.replace"):
        OwnStateBranchAnswerProof.proved(
            snap3,
            SourceRewriteTarget(file_path=str(repo3 / "m.py"), qualname="Rec"),
            "present",
            "PresentRec",
            "Present rec.",
            (
                ConstantAnswerPushConsumer(
                    target=SourceRewriteTarget(file_path=str(repo3 / "m.py"), qualname="use"),
                    test_line=13,
                    method_name="use",
                ),
            ),
        )


def test_own_state_in_place_ternary_return_does_not_double_return(tmp_path: Path) -> None:
    """In-place ternary push on `OwnStateBranchAnswer` uses `DispatchSite.render_in_place_default`
    rather than duplicating `return` into `return return ...`."""
    repo = _write_repo(
        tmp_path / "ternary_inplace",
        {
            "pkg/__init__.py": "",
            "pkg/m.py": """from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    body: str | None

    @property
    def has_body(self) -> bool:
        return self.body is not None

    def preview(self, fallback: str) -> str:
        return self.body if self.has_body else fallback


def build() -> tuple[Note, Note]:
    return Note("hi"), Note(None)
"""
        },
    )
    _, snap, findings = _scan(repo)
    assert len(findings) == 1
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=findings[0].stable_id,
                owner="Note.has_body",
                because="Note with a body is its own subtype",
                name="NoteWithBody",
                judgement="Note carrying a body.",
            ),
        ),
    )
    picked, report = picks.applied(findings)
    assert not report.unmatched
    plan = codemod_plan_from_findings(picked, selector_context=snap)
    assert plan.candidate_count == 1
    sim = plan.document.simulate(snap)
    assert sim.is_clean, sim.preflight_report
    rewritten = sim.simulation.rewritten_sources[str(repo / "pkg/m.py")]
    assert "return return" not in rewritten
    sim.apply()
    mod = _load_module("_test_p41_ternary_inplace", repo / "pkg/m.py")
    try:
        with_body, empty = mod.build()
        assert with_body.preview("none") == "hi"
        assert empty.preview("none") == "none"
    finally:
        sys.modules.pop("_test_p41_ternary_inplace", None)
    _, _, after_p41 = _scan(repo)
    assert after_p41 == []

