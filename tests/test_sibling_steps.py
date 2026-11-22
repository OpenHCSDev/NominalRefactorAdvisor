"""Classes overriding one inherited method differently and taking the same step, found."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_steps import (
    SiblingOverrideStepDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

HOST = (
    "from abc import abstractmethod\n\n\n"
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
