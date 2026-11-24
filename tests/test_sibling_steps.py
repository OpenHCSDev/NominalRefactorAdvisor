"""Classes overriding one inherited method differently and taking the same step, found."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_steps import (
    SiblingOverrideStepDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.sibling_override_steps import (
    HoistSiblingOverrideStepOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache

HOST = (
    "from abc import abstractmethod\n\n\n"
    "class App:\n    def history_displayed(self) -> None:\n        pass\n\n\n"
    "class Host:\n"
    "    @abstractmethod\n    def history_painted(self) -> None:\n        \"\"\"A publication painted the history.\"\"\"\n\n"
    "    @abstractmethod\n    def label(self) -> str: ...\n\n"
    "    def closed(self) -> None:\n        pass\n\n\n"
)
TEXT = (
    "class TextView(Host):\n"
    "    def __init__(self, app, submissions):\n        self.app, self.submissions = app, submissions\n\n"
    "    def history_painted(self) -> None:\n"
    "        self.retire_splash()\n        self.app.history_displayed()\n        self.submissions.settle()\n\n"
    "    def retire_splash(self) -> None:\n        pass\n\n"
    "    def label(self) -> str:\n        return 'text'\n\n\n"
)
QT = (
    "class QtView(Host):\n"
    "    def __init__(self, app, submissions):\n        self.app, self.submissions = app, submissions\n\n"
    "    def history_painted(self) -> None:\n        self.submissions.settle()\n\n"
    "    def opened(self) -> None:\n        self.app.show()\n\n"
    "    def label(self) -> str:\n        return 'qt'\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _findings(root: Path):
    with ScanCache.scope():
        return SiblingOverrideStepDetector().detect(
            parse_python_modules(
                root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )


def test_sibling_override_step_continues_the_stable_sequence():
    assert PatternId.SIBLING_OVERRIDE_STEP == 59


def test_a_step_every_override_takes_and_one_a_class_leaves_out_are_one_finding(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT})
    (finding,) = _findings(tmp_path)
    assert tuple(evidence.symbol for evidence in finding.evidence) == ("TextView.history_painted", "QtView.history_painted")
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "`QtView` alone does not call `self.app.history_displayed()`, and it reads `self.app`" in finding.summary
    # A call on the receiver itself is the class's own helper, not a collaborator's step.
    assert "retire_splash" not in finding.summary
    # Overrides with nothing in common (`label`) are each class's own answer.
    assert "label" not in finding.summary


def test_a_member_the_other_class_never_reads_is_not_an_omitted_step(tmp_path: Path):
    qt = QT.replace("    def opened(self) -> None:\n        self.app.show()\n\n", "").replace(
        "        self.app, self.submissions = app, submissions\n\n    def history_painted(self) -> None:\n        self.submissions.settle()",
        "        self.submissions = submissions\n\n    def history_painted(self) -> None:\n        self.submissions.settle()",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + qt})
    (finding,) = _findings(tmp_path)
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "alone does not call" not in finding.summary


def test_overrides_with_no_step_in_common_are_not_found(tmp_path: Path):
    qt = QT.replace("self.submissions.settle()", "self.repaint()")
    text = TEXT.replace("        self.app.history_displayed()\n        self.submissions.settle()\n", "")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    assert not _findings(tmp_path)


def test_overrides_that_are_one_declaration_are_the_sibling_override_finding(tmp_path: Path):
    same = "    def history_painted(self) -> None:\n        self.submissions.settle()\n\n"
    text = TEXT.replace(
        "    def history_painted(self) -> None:\n"
        "        self.retire_splash()\n        self.app.history_displayed()\n        self.submissions.settle()\n\n",
        same,
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + QT})
    assert not _findings(tmp_path)


def test_a_class_overriding_through_another_is_compared_once(tmp_path: Path):
    wire = (
        "\n\nclass QtWireView(QtView):\n"
        "    def history_painted(self) -> None:\n        self.submissions.settle()\n        self.repaint()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT + wire})
    (finding,) = _findings(tmp_path)
    assert tuple(evidence.symbol for evidence in finding.evidence) == ("TextView.history_painted", "QtView.history_painted")


def test_overrides_that_each_call_something_else_on_one_member_are_each_class_own_spelling(tmp_path: Path):
    text = TEXT.replace("        self.app.history_displayed()\n", "        self.app.history_displayed()\n        self.bar.update('x')\n")
    qt = QT.replace("        self.submissions.settle()\n\n    def opened", "        self.submissions.settle()\n        self.bar.setText('x')\n        self.app.show()\n\n    def opened")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    (finding,) = _findings(tmp_path)
    assert "every one of them calls `self.submissions.settle()`" in finding.summary
    assert "alone does not call" not in finding.summary


def test_a_call_into_a_library_is_not_a_step_another_class_omits(tmp_path: Path):
    """`history_displayed` is declared by a scanned class; `setVisible` by none."""
    qt = QT.replace("        self.submissions.settle()\n\n    def opened", "        self.submissions.settle()\n        self.panel.setVisible(True)\n\n    def opened")
    text = TEXT.replace("    def label(self) -> str:\n        return 'text'", "    def shown(self) -> None:\n        self.panel.refresh()\n\n    def label(self) -> str:\n        return 'text'")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + text + qt})
    (finding,) = _findings(tmp_path)
    assert "`QtView` alone does not call `self.app.history_displayed()`" in finding.summary
    assert "setVisible" not in finding.summary


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )


def _apply_finding(root: Path):
    (finding,) = _findings(root)
    snapshot = _snapshot(root)
    evaluation = SiblingOverrideStepDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    return finding, snapshot, evaluation


def test_hoisting_trailing_step_creates_template_method_and_deletes_empty_sibling(
    tmp_path: Path,
):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": HOST + TEXT + QT})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, HoistSiblingOverrideStepOperation)
    assert operation.member_name == "history_painted"
    assert operation.hook_name == "_history_painted_step"

    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/views.py")]
    assert "def history_painted(self) -> None:" in rewritten
    assert "self._history_painted_step()" in rewritten
    assert "self.submissions.settle()" in rewritten
    assert "def _history_painted_step(self) -> None:\n        pass" in rewritten
    assert "def _history_painted_step(self) -> None:\n        self.retire_splash()" in rewritten
    # QtView's entire body was hoisted, so its override is deleted completely.
    qt_section = rewritten.split("class QtView(Host):")[1]
    assert "history_painted" not in qt_section

    # Execute the rewritten module and confirm runtime behavior matches identically.
    namespace: dict[str, object] = {}
    exec(compile(rewritten, "views.py", "exec"), namespace)
    calls: list[str] = []

    class FakeApp:
        def history_displayed(self) -> None:
            calls.append("displayed")

    class FakeSubmissions:
        def settle(self) -> None:
            calls.append("settle")

    text_view = namespace["TextView"](FakeApp(), FakeSubmissions())  # type: ignore[operator]
    qt_view = namespace["QtView"](FakeApp(), FakeSubmissions())  # type: ignore[operator]
    text_view.history_painted()
    assert calls == ["displayed", "settle"]
    calls.clear()
    qt_view.history_painted()
    assert calls == ["settle"]

    (tmp_path / "pkg/views.py").write_text(rewritten)
    assert _findings(tmp_path) == []


def test_hoisting_leading_early_return_guard_preserves_abstract_hook_and_return_value(
    tmp_path: Path,
):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Demand(ABC):\n"
        "    def accepts(self, end: int) -> bool:\n"
        "        return end > 0\n\n"
        "    @abstractmethod\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        \"\"\"Rows to read.\"\"\"\n\n\n"
        "class MovingDemand(Demand):\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        if not self.accepts(end):\n"
        "            return 0\n"
        "        return min(end, limit)\n\n\n"
        "class TargetDemand(Demand):\n"
        "    def read_rows(self, end: int, limit: int = 10) -> int:\n"
        "        if not self.accepts(end):\n"
        "            return 0\n"
        "        return end + limit\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/demands.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    recipe = evaluation.executable_recipe
    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/demands.py")]
    assert "return self._read_rows_step(end, limit)" in rewritten
    assert "@abstractmethod\n    def _read_rows_step(self, end: int, limit: int = 10) -> int:" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "demands.py", "exec"), namespace)
    moving = namespace["MovingDemand"]()  # type: ignore[operator]
    target = namespace["TargetDemand"]()  # type: ignore[operator]
    assert moving.read_rows(0) == 0
    assert moving.read_rows(4, 10) == 4
    assert target.read_rows(0) == 0
    assert target.read_rows(4, 10) == 14


def test_hoisting_moves_required_imports_across_modules_and_cleans_obsolete_imports(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/constants.py": "def factor() -> int:\n    return 3\n",
            "pkg/base.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class Base(ABC):\n"
                "    def scale(self, x: int) -> None:\n"
                "        self.value = x\n\n"
                "    @abstractmethod\n"
                "    def run(self) -> int:\n"
                "        ...\n"
            ),
            "pkg/one.py": (
                "from pkg.base import Base\n"
                "from pkg.constants import factor\n\n\n"
                "class One(Base):\n"
                "    def run(self) -> int:\n"
                "        self.scale(factor())\n"
                "        return 1\n"
            ),
            "pkg/two.py": (
                "from pkg.base import Base\n"
                "from pkg.constants import factor\n\n\n"
                "class Two(Base):\n"
                "    def run(self) -> int:\n"
                "        self.scale(factor())\n"
                "        return 2\n"
            ),
        },
    )
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten_base = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/base.py")]
    rewritten_one = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/one.py")]
    rewritten_two = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/two.py")]
    assert "from pkg.constants import factor" in rewritten_base
    assert "self.scale(factor())" in rewritten_base
    assert "factor" not in rewritten_one
    assert "factor" not in rewritten_two


def test_hoisting_supports_classmethod_and_property_descriptors(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Loader(ABC):\n"
        "    @classmethod\n"
        "    def validate(cls, raw: str) -> None:\n"
        "        if not raw:\n"
        "            raise ValueError('empty')\n\n"
        "    @classmethod\n"
        "    @abstractmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        ...\n\n\n"
        "class JsonLoader(Loader):\n"
        "    @classmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        cls.validate(raw)\n"
        "        return 'json:' + raw\n\n\n"
        "class TomlLoader(Loader):\n"
        "    @classmethod\n"
        "    def load(cls, raw: str) -> str:\n"
        "        cls.validate(raw)\n"
        "        return 'toml:' + raw\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/loaders.py": source})
    _finding, snapshot, evaluation = _apply_finding(tmp_path)
    simulation = evaluation.executable_recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    rewritten = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/loaders.py")]
    assert "return cls._load_step(raw)" in rewritten
    assert "@classmethod\n    @abstractmethod\n    def _load_step(cls, raw: str) -> str:" in rewritten

    namespace: dict[str, object] = {}
    exec(compile(rewritten, "loaders.py", "exec"), namespace)
    assert namespace["JsonLoader"].load("a") == "json:a"  # type: ignore[attr-defined]
    assert namespace["TomlLoader"].load("b") == "toml:b"  # type: ignore[attr-defined]

    prop_source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Panel(ABC):\n"
        "    def ready(self) -> bool:\n"
        "        return self.active\n\n"
        "    @property\n"
        "    @abstractmethod\n"
        "    def title(self) -> str:\n"
        "        ...\n\n\n"
        "class FirstPanel(Panel):\n"
        "    active = True\n\n"
        "    @property\n"
        "    def title(self) -> str:\n"
        "        if not self.ready():\n"
        "            return ''\n"
        "        return 'first'\n\n\n"
        "class SecondPanel(Panel):\n"
        "    active = False\n\n"
        "    @property\n"
        "    def title(self) -> str:\n"
        "        if not self.ready():\n"
        "            return ''\n"
        "        return 'second'\n"
    )
    prop_root = tmp_path / "prop"
    _write(prop_root, {"pkg/__init__.py": "", "pkg/panels.py": prop_source})
    _finding_p, snap_p, eval_p = _apply_finding(prop_root)
    sim_p = eval_p.executable_recipe.simulate(snap_p)
    assert sim_p.is_clean, sim_p.preflight_report
    rewritten_p = sim_p.simulation.rewritten_sources[str(prop_root / "pkg/panels.py")]
    assert "return self._title_step()" in rewritten_p
    ns_p: dict[str, object] = {}
    exec(compile(rewritten_p, "panels.py", "exec"), ns_p)
    assert ns_p["FirstPanel"]().title == "first"  # type: ignore[operator]
    assert ns_p["SecondPanel"]().title == ""  # type: ignore[operator]


def test_hoisting_stops_when_leading_step_binds_local_read_by_middle(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Provider(ABC):\n"
        "    def make_matcher(self, q: str) -> str:\n"
        "        return q.lower()\n\n"
        "    @abstractmethod\n"
        "    def search(self, query: str) -> str:\n"
        "        ...\n\n\n"
        "class First(Provider):\n"
        "    def search(self, query: str) -> str:\n"
        "        matcher = self.make_matcher(query)\n"
        "        return matcher + '!'\n\n\n"
        "class Second(Provider):\n"
        "    def search(self, query: str) -> str:\n"
        "        matcher = self.make_matcher(query)\n"
        "        return matcher + '?'\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/providers.py": source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "not independent leading or trailing statements" in evaluation.rejection_reason
    assert NeededInput.SOURCE_CHANGE in evaluation.needs


def test_hoisting_stops_when_middle_returns_early_before_trailing_step(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Base(ABC):\n"
        "    def settle(self) -> None:\n"
        "        pass\n\n"
        "    @abstractmethod\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        ...\n\n\n"
        "class One(Base):\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        if skip:\n"
        "            return\n"
        "        self.settle()\n\n\n"
        "class Two(Base):\n"
        "    def finish(self, skip: bool) -> None:\n"
        "        self.settle()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/finish.py": source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "not independent leading or trailing statements" in evaluation.rejection_reason


def test_hoisting_stops_when_descendant_calls_super_or_inherits_without_overriding(
    tmp_path: Path,
):
    super_source = (
        HOST
        + TEXT
        + QT
        + "\n\nclass ExtraQt(QtView):\n"
        "    def history_painted(self) -> None:\n"
        "        super().history_painted()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": super_source})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "calls `super().history_painted()`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs

    unoverridden = (
        "class Base:\n"
        "    def notify(self) -> None:\n"
        "        pass\n\n"
        "    def run(self) -> None:\n"
        "        pass\n\n\n"
        "class One(Base):\n"
        "    def run(self) -> None:\n"
        "        self.notify()\n"
        "        self.a = 1\n\n\n"
        "class Two(Base):\n"
        "    def run(self) -> None:\n"
        "        self.notify()\n"
        "        self.b = 2\n\n\n"
        "class Bystander(Base):\n"
        "    pass\n"
    )
    other_root = tmp_path / "other"
    _write(other_root, {"pkg/__init__.py": "", "pkg/run.py": unoverridden})
    _finding2, _snapshot2, evaluation2 = _apply_finding(other_root)
    assert "`Bystander` inherits `Base.run` without overriding it" in evaluation2.rejection_reason
    assert NeededInput.PROOF in evaluation2.needs


def test_hoisting_stops_on_dunder_method_and_hook_collision(tmp_path: Path):
    colliding = HOST.replace(
        "    def closed(self) -> None:\n        pass",
        "    def _history_painted_step(self) -> None:\n        pass",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": colliding + TEXT + QT})
    _finding, _snapshot, evaluation = _apply_finding(tmp_path)
    assert "already binds hook name `_history_painted_step`" in evaluation.rejection_reason
    assert NeededInput.SOURCE_CHANGE in evaluation.needs
