"""Classes overriding one inherited member with the same declaration, found."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_overrides import (
    SiblingOverrideDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

VIEWS = (
    "from abc import abstractmethod\n\n\n"
    "def reading() -> bool:\n    return False\n\n\n"
    "class StatusOwner:\n"
    "    @abstractmethod\n    def loading(self) -> bool:\n        \"\"\"Whether history is loading.\"\"\"\n\n"
    "    def label(self) -> str:\n        return 'view'\n\n\n"
    "class TextConversation(StatusOwner):\n"
    "    def loading(self) -> bool:\n        \"\"\"Only wire histories page in.\"\"\"\n        return reading()\n\n"
    "    def label(self) -> str:\n        return 'conversation'\n\n\n"
    "class QtConversation(StatusOwner):\n"
    "    def loading(self) -> bool:\n        \"\"\"As Textual\'s conversation.\"\"\"\n        return reading()\n\n"
    "    def label(self) -> str:\n        return 'conversation'\n\n\n"
    "class WireView(StatusOwner):\n"
    "    def loading(self) -> bool:\n        return True\n\n\n"
    "class QtWireView(QtConversation):\n"
    "    def loading(self) -> bool:\n        return reading()\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _findings(root: Path):
    with ScanCache.scope():
        return SiblingOverrideDetector().detect(
            parse_python_modules(
                root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )


def test_sibling_override_continues_the_stable_sequence():
    assert PatternId.SIBLING_OVERRIDE == 58


def test_two_classes_answering_an_inherited_member_alike_are_one_finding_per_member(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": VIEWS})
    findings = sorted(_findings(tmp_path), key=lambda finding: finding.primary_evidence.symbol)
    assert [tuple(evidence.symbol for evidence in finding.evidence) for finding in findings] == [
        ("TextConversation.label", "QtConversation.label"),
        ("TextConversation.loading", "QtConversation.loading"),
    ]
    label, loading = findings
    # Docstrings differ and the copies are still one declaration; the abstract hook has no
    # answer of its own, and the class answering otherwise is counted, not named.
    assert "`StatusOwner.loading` is an abstract hook" in loading.summary
    assert "1 other class(es) override it with another answer" in loading.summary
    assert "`StatusOwner.label` answers otherwise" in label.summary
    # `QtWireView.loading` restates what it inherits from `QtConversation`: lookup already
    # reaches that answer, which is `RestatedOverrideDetector`'s finding, not a sibling's.
    assert not any("QtWireView" in finding.summary for finding in findings)


def test_the_same_text_bound_to_different_names_is_two_answers(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/base.py": "class Base:\n    def answer(self) -> int:\n        return 0\n",
            "pkg/one.py": (
                "from pkg.base import Base\n\n\ndef value() -> int:\n    return 1\n\n\n"
                "class One(Base):\n    def answer(self) -> int:\n        return value()\n"
            ),
            "pkg/two.py": (
                "from pkg.base import Base\n\n\ndef value() -> int:\n    return 2\n\n\n"
                "class Two(Base):\n    def answer(self) -> int:\n        return value()\n"
            ),
        },
    )
    assert _findings(tmp_path) == []


def test_classes_over_a_base_with_no_source_are_found_and_their_order_is_said_unproved(tmp_path: Path):
    """A widget over a C-extension base has no proved order; two frontends' widgets are
    the classes this is for, so they are reported with that said, not dropped."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/shared.py": (
                "from abc import abstractmethod\n\n\n"
                "class Scope:\n    @abstractmethod\n    def scope(self) -> str: ...\n\n\n"
                "class Bound(Scope):\n    @abstractmethod\n    def scope(self) -> str: ...\n"
            ),
            "pkg/text.py": (
                "from pkg.shared import Bound\n\n\n"
                "class TextView(Bound):\n    def scope(self) -> str:\n        return 'thread'\n"
            ),
            "pkg/qt.py": (
                "from _ctypes import Structure\n\nfrom pkg.shared import Scope\n\n\n"
                "class QtView(Structure, Scope):\n    def scope(self) -> str:\n        return 'thread'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    assert [evidence.symbol for evidence in finding.evidence] == ["QtView.scope", "TextView.scope"]
    # `TextView` reaches `Bound.scope` first and `QtView` only `Scope.scope`: the declaration
    # both override is `Scope`'s, the nearest one they share.
    assert "each override `Scope.scope` with the same declaration" in finding.summary
    assert "the lookup order of `QtView` is not proved" in finding.summary


def test_a_shared_value_is_a_table_row_unless_a_class_exists_to_hold_it(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/roles.py": (
                "class Role:\n    bold = False\n\n\n"
                "class Strong(Role):\n    bold = True\n\n\n"
                "class Mention(Role):\n    bold = True\n\n\n"
                "class Plain(Role):\n    bold = False\n"
            ),
            "pkg/forms.py": (
                "from typing import ClassVar\n\n\n"
                "class Export:\n    pass\n\n\nclass Import:\n    pass\n\n\n"
                "class Form:\n    kind: ClassVar[type]\n\n\n"
                "class ExportForm(Form):\n    heading = 'Export'\n\n\n"
                "class ImportForm(Form):\n    heading = 'Import'\n\n\n"
                "class TextExport(ExportForm):\n    kind = Export\n\n\n"
                "class QtExport(ExportForm):\n    kind = Export\n\n\n"
                "class TextImport(ImportForm):\n    kind = Import\n\n\n"
                "class QtImport(ImportForm):\n    kind = Import\n"
            ),
        },
    )
    findings = sorted(_findings(tmp_path), key=lambda finding: finding.primary_evidence.symbol)
    # `bold = True` on two roles is two rows of the roles' value table: nothing holds it.
    assert [tuple(evidence.symbol for evidence in finding.evidence) for finding in findings] == [
        ("TextExport.kind", "QtExport.kind"),
        ("TextImport.kind", "QtImport.kind"),
    ]
    assert "`ExportForm` already exists to hold it: every one of them inherits it and no class answering otherwise does" in findings[0].summary
    assert "`ImportForm` already exists to hold it" in findings[1].summary
