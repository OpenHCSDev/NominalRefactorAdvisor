"""A string key selecting behaviour at several sites is a nominal family kept by hand
(paper1 Thm 3.2): its members are the literal groups no site tells apart."""

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.detectors._string_families import (
    StringFamilyMetrics,
    StringKeyedFamilyDetector,
)


def _findings(tmp_path: Path, source: str):
    (tmp_path / "actions.py").write_text(source)
    detector_id = StringKeyedFamilyDetector.effective_detector_id()
    return [finding for finding in analyze_path(tmp_path) if finding.detector_id == detector_id]


ACTIONS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass\nclass Action:\n    kind: str\n    value: int = 0\n\n\n"
    "def apply(action, state):\n"
    "    if action.kind == 'clear':\n        state.clear()\n"
    "    elif action.kind == 'set':\n        state.add(action.value)\n"
    "    elif action.kind == 'pause':\n        state.hold()\n"
    "    else:\n        raise ValueError(action.kind)\n\n\n"
    "def label(action):\n"
    "    match action.kind:\n"
    "        case 'clear':\n            return 'Clear'\n"
    "        case 'set':\n            return 'Set'\n"
    "        case 'pause':\n            return 'Pause'\n"
    "    raise ValueError(action.kind)\n\n\n"
    "def cost(action):\n"
    "    if action.kind == 'clear':\n        return 0\n"
    "    elif action.kind in ('set', 'pause'):\n        return 1\n"
    "    raise ValueError(action.kind)\n"
)


def test_one_key_at_three_sites_is_a_family_with_one_member_per_literal(tmp_path: Path):
    (finding,) = _findings(tmp_path, ACTIONS)
    metrics = finding.metrics
    assert isinstance(metrics, StringFamilyMetrics)
    assert metrics.key == "kind"
    assert metrics.methods == ("apply", "label", "cost")
    # `cost` answers 'set' and 'pause' alike, but `apply` and `label` tell them apart.
    assert metrics.members == (("clear",), ("pause",), ("set",))
    assert metrics.minimum == 3 and metrics.exact
    assert "3 member(s), the minimum" in finding.summary
    assert [location.line for location in finding.evidence] == [11, 22, 33]


SPELLED_TWICE = ACTIONS.replace(
    "    elif action.kind == 'pause':\n        state.hold()\n",
    "    elif action.kind in ('pause', 'halt'):\n        state.hold()\n",
).replace(
    "        case 'pause':\n            return 'Pause'\n",
    "        case 'pause' | 'halt':\n            return 'Pause'\n",
).replace("    elif action.kind in ('set', 'pause'):\n", "    elif action.kind in ('set', 'pause', 'halt'):\n")


def test_two_spellings_no_site_tells_apart_are_one_member(tmp_path: Path):
    (finding,) = _findings(tmp_path, SPELLED_TWICE)
    assert finding.metrics.members == (("clear",), ("halt", "pause"), ("set",))
    assert finding.metrics.minimum == 3
    assert "1 member(s) are spelled more than one way" in finding.summary


DECODERS = (
    "class Clear:\n    pass\n\n\nclass Set:\n    pass\n\n\n"
    "def decode(payload):\n"
    "    if payload['type'] == 'clear':\n        return Clear()\n"
    "    elif payload['type'] == 'set':\n        return Set()\n"
    "    raise ValueError(payload)\n\n\n"
    "def from_wire(payload):\n"
    "    match payload.get('type'):\n"
    "        case 'clear':\n            return Clear\n"
    "        case 'set':\n            return Set\n"
    "    raise ValueError(payload)\n"
)


def test_decoders_keep_their_strings(tmp_path: Path):
    """Reading a foreign tag is the one place strings belong: no finding."""

    assert _findings(tmp_path, DECODERS) == []


DISAGREEING = (
    "def colour(level):\n"
    "    if level == 'warn':\n        return 'yellow'\n"
    "    elif level == 'error':\n        return 'yellow'\n"
    "    elif level == 'info':\n        return 'grey'\n"
    "    raise ValueError(level)\n\n\n"
    "def pages(level):\n"
    "    if level == 'warn':\n        return False\n"
    "    elif level in ('error', 'info'):\n        return level == 'error'\n"
    "    raise ValueError(level)\n"
)


def test_a_literal_two_sites_disagree_on_is_its_own_member(tmp_path: Path):
    """`colour` alone would merge 'warn' and 'error'; `pages` tells them apart, so the
    confusability graph gains that edge and the family needs three members, not two."""

    (finding,) = _findings(tmp_path, DISAGREEING)
    assert finding.metrics.members == (("error",), ("info",), ("warn",))
    assert finding.metrics.minimum == finding.metrics.lower_bound == 3


def test_one_function_alone_is_not_a_cross_site_family(tmp_path: Path):
    assert _findings(tmp_path, DISAGREEING.split("\n\n\ndef pages")[0] + "\n") == []


SPLIT = (
    "def opens(event):\n"
    "    if event.kind == 'open':\n        return True\n"
    "    return False\n\n\n"
    "def closes(event):\n"
    "    if event.kind == 'close':\n        return True\n"
    "    return False\n\n\n"
    "def moved(event):\n"
    "    if event.kind == 'move':\n        return event.target\n"
    "    return None\n"
)


def test_one_literal_per_function_is_still_one_family(tmp_path: Path):
    """Each function tests one literal; together they spell three members of one key
    in one module, which no single function's dispatch shows."""

    (finding,) = _findings(tmp_path, SPLIT)
    assert finding.metrics.methods == ("opens", "closes", "moved")
    assert finding.metrics.members == (("close",), ("move",), ("open",))


def test_a_nested_function_s_test_is_its_own_site(tmp_path: Path):
    """The outer function does not also own the inner one's comparison."""

    nested = (
        "def serve(frames):\n"
        "    def answer(frame):\n"
        "        if frame.method == 'initialize':\n            return 1\n"
        "        return 0\n"
        "    return answer\n\n\n"
        "def other(frame):\n"
        "    if frame.method == 'resume':\n        return 2\n"
        "    return 0\n"
    )
    (finding,) = _findings(tmp_path, nested)
    assert finding.metrics.methods == ("answer", "other")
    assert len(finding.evidence) == 2


def _declared(*spellings: str) -> str:
    return "from typing import ClassVar\n\n\nclass Step:\n    kind: ClassVar[str]\n\n\n" + "".join(
        f"class {spelling.title()}(Step):\n    kind = {spelling!r}\n\n\n" for spelling in spellings
    )


def test_a_family_the_scan_declares_belongs_to_restated_identity(tmp_path: Path):
    (tmp_path / "steps.py").write_text(_declared("clear", "set", "pause"))
    assert _findings(tmp_path, ACTIONS) == []


def test_a_partly_declared_family_names_the_family_to_extend(tmp_path: Path):
    (tmp_path / "steps.py").write_text(_declared("clear", "set"))
    (finding,) = _findings(tmp_path, ACTIONS)
    assert "declared below `Step`" in finding.summary
    assert "the minimum: 'clear', 'pause', 'set'." in finding.summary
    assert "add a member for each of 'pause'" in finding.summary
    assert {location.file_path.rpartition("/")[2] for location in finding.evidence} == {
        "actions.py",
        "steps.py",
    }


def test_only_the_declared_family_spelling_a_literal_is_the_one_to_extend(tmp_path: Path):
    shapes = "class Shape:\n    kind: ClassVar[str]\n\n\n" + "".join(
        f"class {name.title()}(Shape):\n    kind = {name!r}\n\n\n" for name in ("circle", "square")
    )
    (tmp_path / "steps.py").write_text(_declared("clear", "set") + shapes)
    (finding,) = _findings(tmp_path, ACTIONS)
    assert "declared below `Step`," in finding.summary
    assert {location.symbol for location in finding.evidence if location.file_path.endswith("steps.py")} == {
        "Clear",
        "Set",
    }


NEGATED = (
    "def apply(action, state):\n"
    "    if action.kind == 'set':\n        state.add(action.value)\n"
    "    elif action.kind == 'pause':\n        state.hold()\n\n\n"
    "def skip(action):\n"
    "    if action.kind != 'clear':\n        return action.value\n"
    "    return 0\n"
)


def test_a_negated_test_sends_its_literal_to_the_else(tmp_path: Path):
    (finding,) = _findings(tmp_path, NEGATED)
    metrics = finding.metrics
    assert metrics.methods == ("apply", "skip")
    # `skip` answers 'set' and 'pause' alike and 'clear' apart; `apply` splits the other two.
    assert metrics.members == (("clear",), ("pause",), ("set",))
