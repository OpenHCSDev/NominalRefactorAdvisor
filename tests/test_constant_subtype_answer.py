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
