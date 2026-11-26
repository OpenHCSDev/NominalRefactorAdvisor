"""Pushing CONSTANT_SUBTYPE_ANSWER branches onto polymorphic methods of their family."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import ConstantSubtypeAnswerDetector
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    PushConstantSubtypeAnswerOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def _findings(root: Path):
    with ScanCache.scope():
        return ConstantSubtypeAnswerDetector().detect(
            parse_python_modules(
                root,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=False),
            ),
            DetectorConfig(),
        )


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )


def _apply(root: Path, recipe) -> None:
    simulation = recipe.simulate(_snapshot(root))
    assert simulation.is_clean
    simulation.apply()


def test_pushes_in_place_base_method_and_deletes_unread_constant_answer(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/process.py": (
                "class ProcessState:\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return False\n\n"
                "    def require_owner_startable(self) -> None:\n"
                "        if not self.allows_owner_start():\n"
                "            raise RuntimeError('already running')\n\n\n"
                "class NoProcess(ProcessState):\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return True\n\n\n"
                "class ProcessIdentity(ProcessState):\n"
                "    def allows_owner_start(self) -> bool:\n"
                "        return False\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (op,) = recipe.operations
    assert isinstance(op, PushConstantSubtypeAnswerOperation)
    assert len(op.consumers) == 1
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "process.py").read_text()
    assert "allows_owner_start" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["NoProcess"]().require_owner_startable()  # type: ignore[operator]
    for cls_name in ("ProcessState", "ProcessIdentity"):
        try:
            ns[cls_name]().require_owner_startable()  # type: ignore[operator]
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError on {cls_name}")
    assert _findings(tmp_path) == []


def test_pushes_multi_consumer_classmethod_and_ternary_and_deletes_answer(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/verdicts.py": (
                "class HeadVerdict:\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return False\n\n"
                "    @classmethod\n"
                "    def require_ok(cls) -> None:\n"
                "        if not cls.allows_integration():\n"
                "            raise ValueError('blocked')\n\n\n"
                "class ReviewedHeadVerdict(HeadVerdict):\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return True\n\n\n"
                "class SubBlockedVerdict(ReviewedHeadVerdict):\n"
                "    @classmethod\n"
                "    def allows_integration(cls) -> bool:\n"
                "        return False\n\n\n"
                "def accepted_sources(verdict: type[HeadVerdict], req: str) -> tuple[str, ...]:\n"
                "    return (req,) if verdict.allows_integration() else ()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (op,) = recipe.operations
    assert isinstance(op, PushConstantSubtypeAnswerOperation)
    assert len(op.consumers) == 2
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "verdicts.py").read_text()
    assert "allows_integration" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["ReviewedHeadVerdict"].require_ok()  # type: ignore[attr-defined]
    for cls_name in ("HeadVerdict", "SubBlockedVerdict"):
        try:
            ns[cls_name].require_ok()  # type: ignore[attr-defined]
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError on {cls_name}")
    assert ns["accepted_sources"](ns["ReviewedHeadVerdict"], "r1") == ("r1",)  # type: ignore[operator]
    assert ns["accepted_sources"](ns["HeadVerdict"], "r1") == ()  # type: ignore[operator]
    assert ns["accepted_sources"](ns["SubBlockedVerdict"], "r1") == ()  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_embedded_loop_continue_with_owner_pick_and_keeps_answer_if_other_reader_exists(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/members.py": (
                "from typing import ClassVar\n\n\n"
                "class DestinationMember:\n"
                "    callers_import_destination: ClassVar[bool] = False\n\n"
                "    def __init__(self, name: str) -> None:\n"
                "        self.name = name\n\n\n"
                "class DestinationClassMethod(DestinationMember):\n"
                "    callers_import_destination = True\n\n\n"
                "def collect_imports(members: list[DestinationMember], out: list[str]) -> None:\n"
                "    for member in members:\n"
                "        if not member.callers_import_destination:\n"
                "            continue\n"
                "        out.append(member.name)\n\n\n"
                "def describe(member: DestinationMember) -> bool:\n"
                "    return member.callers_import_destination\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert unpicked.recipe is None
    assert NeededInput.OWNER_PICK in unpicked.needs

    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="collect_imports",
                because="destination members own whether callers import them",
                name="record_required_import",
                judgement="Record this member's name when callers must import its destination.",
            ),
        ),
    )
    picked, report = picks.applied((finding,))
    assert [o.outcome for o in report.outcomes] == ["picked"]
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "members.py").read_text()
    assert "member.record_required_import(out)" in after
    assert "@classmethod" not in after
    # Non-branch reader `describe` still reads `callers_import_destination`, so it stays declared.
    assert "callers_import_destination" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    out: list[str] = []
    plain = ns["DestinationMember"]("a")  # type: ignore[operator]
    cm = ns["DestinationClassMethod"]("b")  # type: ignore[operator]
    ns["collect_imports"]([plain, cm], out)  # type: ignore[operator]
    assert out == ["b"]
    assert ns["describe"](plain) is False  # type: ignore[operator]
    assert ns["describe"](cm) is True  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_refuses_in_place_push_when_branch_calls_super(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/base.py": (
                "class Root:\n"
                "    def check(self) -> int:\n"
                "        return 1\n\n\n"
                "class Mid(Root):\n"
                "    allow = False\n\n"
                "    def check(self) -> int:\n"
                "        if self.allow:\n"
                "            return super().check() + 1\n"
                "        return 0\n\n\n"
                "class Leaf(Mid):\n"
                "    allow = True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert NeededInput.SOURCE_CHANGE in evaluation.needs


def test_pushes_intermediate_subclass_in_place_and_uninitialized_annotation_base(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/strategy.py": (
                "class Strategy:\n"
                "    allows_projection: bool\n\n"
                "    @classmethod\n"
                "    def require_valid(cls, requested: bool) -> None:\n"
                "        if requested and not cls.allows_projection:\n"
                "            raise ValueError('needs simulate')\n\n\n"
                "class RunStrategy(Strategy):\n"
                "    allows_projection = False\n\n\n"
                "class SimulateStrategy(Strategy):\n"
                "    allows_projection = True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "strategy.py").read_text()
    assert "allows_projection" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    ns["SimulateStrategy"].require_valid(True)  # type: ignore[attr-defined]
    ns["RunStrategy"].require_valid(False)  # type: ignore[attr-defined]
    try:
        ns["RunStrategy"].require_valid(True)  # type: ignore[attr-defined]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on RunStrategy(requested=True)")
    assert _findings(tmp_path) == []


def test_pushes_or_else_guard_and_compound_ternary_with_owner_pick(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/channel.py": (
                "class ChannelSelection:\n"
                "    def requires_catalog(self) -> bool:\n"
                "        return False\n\n\n"
                "class NamedChannel(ChannelSelection):\n"
                "    def requires_catalog(self) -> bool:\n"
                "        return True\n\n\n"
                "def load_catalog(channel: ChannelSelection, fallback: bool) -> str:\n"
                "    return 'loaded' if channel.requires_catalog() or fallback else 'none'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)

    after = (tmp_path / "pkg" / "channel.py").read_text()
    assert "requires_catalog" not in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["load_catalog"](ns["NamedChannel"](), False) == "loaded"  # type: ignore[operator]
    assert ns["load_catalog"](ns["ChannelSelection"](), True) == "loaded"  # type: ignore[operator]
    assert ns["load_catalog"](ns["ChannelSelection"](), False) == "none"  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_prefix_guarded_or_early_return_and_non_first_compound_ternary(tmp_path: Path):
    """`if PREFIX or not subject.flag(): return fallback; ...` and
    `a if (p1 and p2 and subject.flag) else b` push cleanly while preserving
    PREFIX short-circuiting and deleting the dead constant answer."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/roles.py": (
                "from typing import ClassVar\n\n\n"
                "class ThreadRole:\n"
                "    def hears_each_reply(self) -> bool:\n"
                "        return False\n\n\n"
                "class PersonRole(ThreadRole):\n"
                "    def hears_each_reply(self) -> bool:\n"
                "        return True\n\n\n"
                "def format_route(reply_route: str, role: ThreadRole, target: str) -> str:\n"
                "    if not reply_route or not role.hears_each_reply():\n"
                "        return ''\n"
                "    return reply_route + ':' + target\n\n\n"
                "class ViewportRequest:\n"
                "    animates_arrivals: ClassVar[bool] = False\n\n\n"
                "class LiveViewportRequest(ViewportRequest):\n"
                "    animates_arrivals: ClassVar[bool] = True\n\n\n"
                "def select_keys(arrivals: bool, drawn: bool, request: ViewportRequest, keys: list[str]) -> list[str] | None:\n"
                "    chosen = list(keys) if (not arrivals and drawn and request.animates_arrivals) else None\n"
                "    return chosen\n"
            ),
        },
    )

    # 1. `format_route` (`if not reply_route or not role.hears_each_reply(): return ''`)
    for member_name, consumer, method_name in (
        ("hears_each_reply", "format_route", "format_reply_route"),
        ("animates_arrivals", "select_keys", "animated_keys"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its own branch behavior",
                    name=method_name,
                    judgement="Push constant answer branch onto the family.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "roles.py").read_text()
    assert "hears_each_reply" not in after
    assert "animates_arrivals" not in after
    assert "if not reply_route:" in after
    assert "return role.format_reply_route(" in after
    assert "request.animated_keys(" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["format_route"]("", ns["PersonRole"](), "t") == ""  # type: ignore[operator]
    assert ns["format_route"]("r", ns["ThreadRole"](), "t") == ""  # type: ignore[operator]
    assert ns["format_route"]("r", ns["PersonRole"](), "t") == "r:t"  # type: ignore[operator]
    assert ns["select_keys"](False, True, ns["LiveViewportRequest"](), ["a"]) == ["a"]  # type: ignore[operator]
    assert ns["select_keys"](True, True, ns["LiveViewportRequest"](), ["a"]) is None  # type: ignore[operator]
    assert ns["select_keys"](False, True, ns["ViewportRequest"](), ["a"]) is None  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_pushes_sum_and_comprehension_filter_answers_with_parity_and_deletes_flags(
    tmp_path: Path,
):
    """`sum(1 for ... if x.flag)` (including inside an f-string) and filtered
    `GeneratorExp`/`ListComp`/`frozenset(...)` comprehensions (including prefix-guarded
    `and`, captured loop keys, and negated `@classmethod` filters) push cleanly onto
    the family without introducing `ZERO_ONE_TUPLE` and delete the dead constant flags."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/filters.py": (
                "from dataclasses import dataclass\n"
                "from typing import ClassVar\n\n\n"
                "class ShapeModel:\n"
                "    bypasses: ClassVar[bool] = False\n\n\n"
                "class BypassedModel(ShapeModel):\n"
                "    bypasses: ClassVar[bool] = True\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Finding:\n"
                "    model: ShapeModel\n\n\n"
                "def summary_note(findings: list[Finding]) -> str:\n"
                "    return f'bypass={sum(1 for f in findings if f.model.bypasses)}'\n\n\n"
                "class State:\n"
                "    def attention(self) -> bool:\n"
                "        return False\n\n\n"
                "class AlertState(State):\n"
                "    def __init__(self, label: str) -> None:\n"
                "        self.label = label\n\n"
                "    def attention(self) -> bool:\n"
                "        return True\n\n\n"
                "def ping_labels(states: list[State]) -> tuple[str, ...]:\n"
                "    return tuple(f'ping:{state.label}' for state in states if state.attention())\n\n\n"
                "class Status:\n"
                "    def is_working(self) -> bool:\n"
                "        return False\n\n\n"
                "class WorkingStatus(Status):\n"
                "    def is_working(self) -> bool:\n"
                "        return True\n\n\n"
                "@dataclass(frozen=True)\n"
                "class Row:\n"
                "    active: bool\n"
                "    status: Status\n\n\n"
                "def working_names(rows: dict[str, Row]) -> frozenset[str]:\n"
                "    return frozenset(\n"
                "        name for name, row in rows.items() if row.active and row.status.is_working()\n"
                "    )\n\n\n"
                "class Detector:\n"
                "    @classmethod\n"
                "    def prefers_in_process(cls) -> bool:\n"
                "        return False\n\n\n"
                "class LocalDetector(Detector):\n"
                "    @classmethod\n"
                "    def prefers_in_process(cls) -> bool:\n"
                "        return True\n\n\n"
                "def pooled_detectors(detectors: list[type[Detector]]) -> list[type[Detector]]:\n"
                "    return [dt for dt in detectors if not dt.prefers_in_process()]\n\n\n"
                "class Attempt:\n"
                "    def __init__(self, owner: str) -> None:\n"
                "        self.owner = owner\n\n"
                "    def counts_as_started(self) -> bool:\n"
                "        return False\n\n\n"
                "class StartedAttempt(Attempt):\n"
                "    def counts_as_started(self) -> bool:\n"
                "        return True\n\n\n"
                "def started_for(rows: dict[str, Attempt], owners: set[str]) -> frozenset[str]:\n"
                "    return frozenset(\n"
                "        key for key, row in rows.items() if row.counts_as_started() and row.owner in owners\n"
                "    )\n"
            ),
        },
    )

    for member_name, consumer, method_name in (
        ("bypasses", "summary_note", "bypass_count"),
        ("attention", "ping_labels", "attention_pings"),
        ("is_working", "working_names", "working_name_entries"),
        ("prefers_in_process", "pooled_detectors", "pooled_entries"),
        ("counts_as_started", "started_for", "started_key_entries"),
    ):
        findings = _findings(tmp_path)
        target = next(f for f in findings if f.metrics.member_name == member_name)
        unpicked = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            target, _snapshot(tmp_path)
        )
        assert unpicked.recipe is None
        assert NeededInput.OWNER_PICK in unpicked.needs, (member_name, unpicked.rejection_reason)
        picks = OwnerPicks(
            "picks.json",
            "",
            (
                OwnerPick(
                    finding=target.stable_id,
                    owner=consumer,
                    because="family owns its own comprehension projection",
                    name=method_name,
                    judgement="Project member contribution for comprehension.",
                ),
            ),
        )
        picked, _ = picks.applied((target,))
        ev = ConstantSubtypeAnswerDetector().evaluate_recipe_for_finding(
            picked[0], _snapshot(tmp_path)
        )
        _apply(tmp_path, ev.executable_recipe)

    after = (tmp_path / "pkg" / "filters.py").read_text()
    for deleted in ("bypasses", "attention(", "is_working", "prefers_in_process"):
        assert deleted not in after
    assert "sum((f.model.bypass_count() for f in findings))" in after or "sum(f.model.bypass_count() for f in findings)" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    Finding = ns["Finding"]
    ShapeModel = ns["ShapeModel"]
    BypassedModel = ns["BypassedModel"]
    assert ns["summary_note"]([Finding(ShapeModel()), Finding(BypassedModel()), Finding(BypassedModel())]) == "bypass=2"  # type: ignore[operator]
    assert ns["ping_labels"]([ns["State"](), ns["AlertState"]("a"), ns["AlertState"]("b")]) == ("ping:a", "ping:b")  # type: ignore[operator]
    Row = ns["Row"]
    assert ns["working_names"](  # type: ignore[operator]
        {
            "a": Row(True, ns["WorkingStatus"]()),
            "b": Row(False, ns["WorkingStatus"]()),
            "c": Row(True, ns["Status"]()),
        }
    ) == frozenset({"a"})
    assert ns["pooled_detectors"]([ns["Detector"], ns["LocalDetector"]]) == [ns["Detector"]]  # type: ignore[operator]
    Attempt = ns["Attempt"]
    StartedAttempt = ns["StartedAttempt"]
    assert ns["started_for"](  # type: ignore[operator]
        {"k1": StartedAttempt("a"), "k2": StartedAttempt("b"), "k3": Attempt("a")},
        {"a"},
    ) == frozenset({"k1"})
    assert _findings(tmp_path) == []

