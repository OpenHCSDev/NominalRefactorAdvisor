"""Classes overriding one inherited member with the same declaration, found."""

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_overrides import (
    SiblingOverrideDetector,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.owner_picks import OwnerPick, OwnerPicks
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.sibling_overrides import (
    ExtractSiblingOverrideBaseOperation,
    LiftSiblingOverrideOperation,
)
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


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
    )


def _apply(root: Path, recipe) -> None:
    snapshot = _snapshot(root)
    simulation = CodemodPlanSequence.from_operations(recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()


def test_lifts_sibling_override_into_existing_intermediate_home_and_clears_finding(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/forms.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class View(ABC):\n"
                "    @abstractmethod\n"
                "    def label(self) -> str: ...\n\n\n"
                "class Conversation(View, ABC):\n"
                "    pass\n\n\n"
                "class TextConversation(Conversation):\n"
                "    def label(self) -> str:\n"
                "        return 'conversation'\n\n\n"
                "class QtConversation(Conversation):\n"
                "    def label(self) -> str:\n"
                "        return 'conversation'\n\n\n"
                "class WireView(View):\n"
                "    def label(self) -> str:\n"
                "        return 'wire'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, LiftSiblingOverrideOperation)
    assert operation.target.qualname == "Conversation"
    assert operation.member_name == "label"
    _apply(tmp_path, recipe)
    after = (tmp_path / "pkg" / "forms.py").read_text()
    assert "class Conversation(View, ABC):" in after
    assert "    def label(self) -> str:\n        return 'conversation'\n" in after
    assert "class TextConversation(Conversation):\n    pass\n" in after
    assert "class QtConversation(Conversation):\n    pass\n" in after
    assert _findings(tmp_path) == []


def test_lifts_sibling_override_replacing_abstractmethod_stub_on_declaring_home(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/pointers.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class OwnerPointer(ABC):\n"
                "    @abstractmethod\n"
                "    def is_pointed(self) -> bool: ...\n\n\n"
                "class PointedExecution(OwnerPointer):\n"
                "    def is_pointed(self) -> bool:\n"
                "        return True\n\n\n"
                "class PointedAttempt(OwnerPointer):\n"
                "    def is_pointed(self) -> bool:\n"
                "        return True\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)
    after = (tmp_path / "pkg" / "pointers.py").read_text()
    assert "abstractmethod" not in after
    assert "class OwnerPointer(ABC):\n    def is_pointed(self) -> bool:\n        return True\n" in after
    assert _findings(tmp_path) == []


def test_lifts_class_attribute_preserving_classvar_annotation_and_cross_file_imports(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/kinds.py": "class Export:\n    pass\n",
            "pkg/base.py": (
                "from typing import ClassVar\n\n\n"
                "class Form:\n"
                "    kind: ClassVar[type]\n"
            ),
            "pkg/a.py": (
                "from pkg.base import Form\n"
                "from pkg.kinds import Export\n\n\n"
                "class TextExport(Form):\n"
                "    kind = Export\n"
            ),
            "pkg/b.py": (
                "from pkg.base import Form\n"
                "from pkg.kinds import Export\n\n\n"
                "class QtExport(Form):\n"
                "    kind = Export\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)
    base_after = (tmp_path / "pkg" / "base.py").read_text()
    assert "from pkg.kinds import Export" in base_after
    assert "kind: ClassVar[type] = Export" in base_after
    assert "from pkg.kinds import Export" not in (tmp_path / "pkg" / "a.py").read_text()
    assert "from pkg.kinds import Export" not in (tmp_path / "pkg" / "b.py").read_text()
    assert _findings(tmp_path) == []


def test_refuses_lift_when_non_answering_descendant_inherits_from_home_without_override(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/views.py": (
                "class Base:\n"
                "    def label(self) -> str:\n"
                "        return 'base'\n\n\n"
                "class Mid(Base):\n"
                "    pass\n\n\n"
                "class A(Mid):\n"
                "    def label(self) -> str:\n"
                "        return 'mid'\n\n\n"
                "class B(Mid):\n"
                "    def label(self) -> str:\n"
                "        return 'mid'\n\n\n"
                "class InheritsBase(Mid):\n"
                "    pass\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "InheritsBase" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_refuses_lift_when_property_would_shadow_sibling_dataclass_field(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/frames.py": (
                "from abc import ABC\n"
                "from dataclasses import dataclass\n\n\n"
                "class Frame(ABC):\n"
                "    compilation: str\n\n\n"
                "@dataclass(frozen=True)\n"
                "class ModuleFrame(Frame):\n"
                "    compilation: str\n\n\n"
                "class SourceFrame(Frame):\n"
                "    @property\n"
                "    def compilation(self) -> str:\n"
                "        return 'src'\n\n\n"
                "class GeneratedFrame(Frame):\n"
                "    @property\n"
                "    def compilation(self) -> str:\n"
                "        return 'src'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "ModuleFrame" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_refuses_lift_when_concrete_home_is_instantiated_directly(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/views.py": (
                "class Base:\n"
                "    def label(self) -> str:\n"
                "        return 'base'\n\n\n"
                "class Mid(Base):\n"
                "    pass\n\n\n"
                "class A(Mid):\n"
                "    def label(self) -> str:\n"
                "        return 'mid'\n\n\n"
                "class B(Mid):\n"
                "    def label(self) -> str:\n"
                "        return 'mid'\n\n\n"
                "def use() -> str:\n"
                "    return Mid().label()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "Mid" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_stops_for_owner_pick_when_no_home_exists(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": VIEWS})
    findings = sorted(_findings(tmp_path), key=lambda finding: finding.primary_evidence.symbol)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        findings[0], _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "no existing class beneath" in evaluation.rejection_reason
    assert NeededInput.OWNER_PICK in evaluation.needs


def test_lifts_past_abstract_intermediate_and_module_or_instance_dict_writes(
    tmp_path: Path,
):
    """Still-abstract intermediate subclasses, `vars(imported_module)`, and mutating
    `obj.__dict__.setdefault(...)` (impossible on a class `mappingproxy`) do not block
    lifting a sibling override."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/helper.py": (
                "import sys\n\n\n"
                "SYS_NAMES = list(vars(sys))\n\n\n"
                "def cache_slot(obj: object, name: str, value: int) -> int:\n"
                "    return obj.__dict__.setdefault(name, value)\n"
            ),
            "pkg/edges.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class Edge(ABC):\n"
                "    @abstractmethod\n"
                "    def sources(self) -> tuple[str, ...]: ...\n\n"
                "    @abstractmethod\n"
                "    def kind(self) -> str: ...\n\n\n"
                "class ConveyorEdge(Edge, ABC):\n"
                "    pass\n\n\n"
                "class ForwardEdge(ConveyorEdge):\n"
                "    def sources(self) -> tuple[str, ...]:\n"
                "        return ('a',)\n\n"
                "    def kind(self) -> str:\n"
                "        return 'forward'\n\n\n"
                "class ExpansionEdge(Edge):\n"
                "    def sources(self) -> tuple[str, ...]:\n"
                "        return ('a',)\n\n"
                "    def kind(self) -> str:\n"
                "        return 'expansion'\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    _apply(tmp_path, recipe)
    after = (tmp_path / "pkg" / "edges.py").read_text()
    assert "    def sources(self) -> tuple[str, ...]:\n        return ('a',)\n" in after
    assert _findings(tmp_path) == []


def test_lifts_over_unused_concrete_default_on_declaring_and_cleans_obsolete_imports(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/old_helper.py": "def legacy_default() -> str:\n    return 'old'\n",
            "pkg/new_helper.py": "def shared_answer() -> str:\n    return 'shared'\n",
            "pkg/nodes.py": (
                "from pkg.old_helper import legacy_default\n"
                "from pkg.new_helper import shared_answer\n\n\n"
                "class ContextNode:\n"
                "    def search_description(self) -> str:\n"
                "        return legacy_default()\n\n\n"
                "class FileNode(ContextNode):\n"
                "    def search_description(self) -> str:\n"
                "        return shared_answer()\n\n\n"
                "class SymbolNode(ContextNode):\n"
                "    def search_description(self) -> str:\n"
                "        return shared_answer()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, LiftSiblingOverrideOperation)
    assert operation.target.qualname == "ContextNode"
    _apply(tmp_path, recipe)
    after = (tmp_path / "pkg" / "nodes.py").read_text()
    assert "legacy_default" not in after
    assert "class ContextNode:\n    def search_description(self) -> str:\n        return shared_answer()\n" in after
    assert "class FileNode(ContextNode):\n    pass\n" in after
    assert "class SymbolNode(ContextNode):\n    pass\n" in after
    assert _findings(tmp_path) == []


def test_refuses_concrete_default_lift_when_descendant_calls_super_or_declaring_used(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/nodes.py": (
                "class Node:\n"
                "    def describe(self) -> str:\n"
                "        return 'base'\n\n\n"
                "class Left(Node):\n"
                "    def describe(self) -> str:\n"
                "        return 'same'\n\n\n"
                "class Right(Node):\n"
                "    def describe(self) -> str:\n"
                "        return 'same'\n\n"
                "    def original(self) -> str:\n"
                "        return super().describe()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "super().describe" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_extracts_same_module_intermediate_base_with_owner_pick_and_clears_finding(
    tmp_path: Path,
):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": VIEWS})
    findings = sorted(_findings(tmp_path), key=lambda f: f.primary_evidence.symbol)
    label_finding = findings[0]
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=label_finding.stable_id,
                owner="StatusOwner",
                because="both conversation views share the conversation label",
                name="ConversationStatusOwner",
                judgement="Conversation views sharing the conversation label.",
            ),
        ),
    )
    picked, report = picks.applied((label_finding,))
    assert [outcome.outcome for outcome in report.outcomes] == ["picked"]
    assert report.unmatched == ()

    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    recipe = evaluation.executable_recipe
    (operation,) = recipe.operations
    assert isinstance(operation, ExtractSiblingOverrideBaseOperation)
    assert operation.base_name == "ConversationStatusOwner"
    _apply(tmp_path, recipe)
    after = (tmp_path / "pkg" / "views.py").read_text()
    assert "class ConversationStatusOwner(StatusOwner):" in after
    assert "class TextConversation(ConversationStatusOwner):" in after
    assert "class QtConversation(ConversationStatusOwner):" in after
    assert "class WireView(StatusOwner):" in after

    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["TextConversation"]().label() == "conversation"  # type: ignore[operator]
    assert ns["QtConversation"]().label() == "conversation"  # type: ignore[operator]
    assert ns["WireView"]().label() == "view"  # type: ignore[operator]
    remaining = _findings(tmp_path)
    assert all(f.primary_evidence.symbol != "TextConversation.label" for f in remaining)


def test_extracts_cross_module_intermediate_base_and_migrates_imports(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/helpers.py": "def mode_name() -> str:\n    return 'interactive'\n",
            "pkg/driver.py": (
                "from abc import ABC, abstractmethod\n\n\n"
                "class Driver(ABC):\n"
                "    @abstractmethod\n"
                "    def mode(self) -> str: ...\n\n\n"
                "class HeadlessDriver(Driver):\n"
                "    def mode(self) -> str:\n"
                "        return 'headless'\n"
            ),
            "pkg/linux.py": (
                "from pkg.driver import Driver\n"
                "from pkg.helpers import mode_name\n\n\n"
                "class LinuxDriver(Driver):\n"
                "    def mode(self) -> str:\n"
                "        return mode_name()\n"
            ),
            "pkg/windows.py": (
                "from pkg.driver import Driver\n"
                "from pkg.helpers import mode_name\n\n\n"
                "class WindowsDriver(Driver):\n"
                "    def mode(self) -> str:\n"
                "        return mode_name()\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="Driver",
                because="terminal drivers share interactive mode",
                name="InteractiveDriver",
                judgement="Drivers that operate in interactive terminal mode.",
            ),
        ),
    )
    picked, report = picks.applied((finding,))
    assert [outcome.outcome for outcome in report.outcomes] == ["picked"]
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)
    driver_after = (tmp_path / "pkg" / "driver.py").read_text()
    linux_after = (tmp_path / "pkg" / "linux.py").read_text()
    windows_after = (tmp_path / "pkg" / "windows.py").read_text()
    assert "from pkg.helpers import mode_name" in driver_after
    assert "class InteractiveDriver(Driver):" in driver_after
    assert "from pkg.driver import InteractiveDriver" in linux_after
    assert "class LinuxDriver(InteractiveDriver):" in linux_after
    assert "mode_name" not in linux_after
    assert "from pkg.driver import InteractiveDriver" in windows_after
    assert "class WindowsDriver(InteractiveDriver):" in windows_after
    assert "mode_name" not in windows_after
    assert _findings(tmp_path) == []


def test_extracts_indirect_sibling_intermediate_base_with_prepend_mi_and_valid_c3(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/modes.py": (
                "class Mode:\n"
                "    def accepts_escape(self) -> bool:\n"
                "        return False\n\n\n"
                "class InsertBase(Mode):\n"
                "    def kind(self) -> str:\n"
                "        return 'insert'\n\n\n"
                "class VisualBase(Mode):\n"
                "    def kind(self) -> str:\n"
                "        return 'visual'\n\n\n"
                "class VimInsert(InsertBase):\n"
                "    def accepts_escape(self) -> bool:\n"
                "        return True\n\n\n"
                "class VimVisual(VisualBase):\n"
                "    def accepts_escape(self) -> bool:\n"
                "        return True\n\n\n"
                "class PlainInsert(InsertBase):\n"
                "    pass\n"
            ),
        },
    )
    (finding,) = _findings(tmp_path)
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=finding.stable_id,
                owner="Mode",
                because="vim sub-modes exit on escape",
                name="EscapableMode",
                judgement="Modes that accept escape.",
            ),
        ),
    )
    picked, _ = picks.applied((finding,))
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    _apply(tmp_path, evaluation.executable_recipe)
    after = (tmp_path / "pkg" / "modes.py").read_text()
    assert "class EscapableMode(Mode):" in after
    assert "class VimInsert(EscapableMode, InsertBase):" in after
    assert "class VimVisual(EscapableMode, VisualBase):" in after
    ns: dict[str, object] = {}
    exec(after, ns)
    assert ns["VimInsert"]().accepts_escape() is True  # type: ignore[operator]
    assert ns["VimInsert"]().kind() == "insert"  # type: ignore[operator]
    assert ns["VimVisual"]().accepts_escape() is True  # type: ignore[operator]
    assert ns["VimVisual"]().kind() == "visual"  # type: ignore[operator]
    assert ns["PlainInsert"]().accepts_escape() is False  # type: ignore[operator]
    assert _findings(tmp_path) == []


def test_refuses_extract_when_picked_base_name_collides(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/views.py": VIEWS})
    findings = sorted(_findings(tmp_path), key=lambda f: f.primary_evidence.symbol)
    label_finding = findings[0]
    picks = OwnerPicks(
        "picks.json",
        "",
        (
            OwnerPick(
                finding=label_finding.stable_id,
                owner="StatusOwner",
                because="collides with existing name",
                name="WireView",
                judgement="Existing class name.",
            ),
        ),
    )
    picked, _ = picks.applied((label_finding,))
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        picked[0], _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "already bound" in evaluation.rejection_reason
    assert NeededInput.OWNER_PICK in evaluation.needs


def test_sibling_property_with_return_annotation_on_one_sibling_lifts_keeping_annotation(
    tmp_path: Path,
):
    """Reproduces Toad `ConversationPager.selected_categories`: an unannotated
    `@property` on Qt and an annotated `@property` on Textual are one
    declaration and lift onto their shared ancestor keeping the annotation."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/pager.py": (
                "from __future__ import annotations\n\n\n"
                "class HistoryPager:\n"
                "    @property\n"
                "    def selected_categories(self) -> frozenset[str]:\n"
                "        return frozenset(('all',))\n\n\n"
                "class ConversationPager(HistoryPager):\n"
                "    pass\n"
            ),
            "pkg/qt_history.py": (
                "from pkg.pager import ConversationPager\n\n\n"
                "class QtConversationHistory(ConversationPager):\n"
                "    def __init__(self, visible: frozenset[str]) -> None:\n"
                "        self.visible = visible\n\n"
                "    @property\n"
                "    def selected_categories(self):\n"
                "        return self.visible\n"
            ),
            "pkg/tui_history.py": (
                "from __future__ import annotations\n\n"
                "from pkg.pager import ConversationPager\n\n\n"
                "class ConversationHistory(ConversationPager):\n"
                "    def __init__(self, visible: frozenset[str]) -> None:\n"
                "        self.visible = visible\n\n"
                "    @property\n"
                "    def selected_categories(self) -> frozenset[str]:\n"
                "        return self.visible\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 1
    assert "`ConversationPager` already exists to hold it" in findings[0].summary
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        findings[0], _snapshot(tmp_path)
    )
    assert evaluation.recipe is not None, evaluation.rejection_reason
    simulation = evaluation.recipe.simulate(_snapshot(tmp_path))
    assert simulation.is_clean, simulation.preflight_report
    simulation.apply()
    pager_text = (tmp_path / "pkg/pager.py").read_text()
    assert "def selected_categories(self) -> frozenset[str]:" in pager_text
    assert "selected_categories" not in (tmp_path / "pkg/qt_history.py").read_text()
    assert "selected_categories" not in (tmp_path / "pkg/tui_history.py").read_text()
    assert _findings(tmp_path) == []


def test_sibling_property_with_conflicting_return_annotations_is_found_and_refuses_blind_lift(
    tmp_path: Path,
):
    """Reproduces Toad `QtQuestion.question -> QtQuestion` vs `Question.question -> Question`:
    detected as identical runtime overrides, while lifting refuses stamping one sibling's
    return type onto the other."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/question.py": (
                "from __future__ import annotations\n"
                "from abc import ABC, abstractmethod\n\n\n"
                "class QuestionContext(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def question(self) -> QuestionContext: ...\n\n\n"
                "class QtQuestion(QuestionContext):\n"
                "    @property\n"
                "    def question(self) -> QtQuestion:\n"
                "        return self\n\n\n"
                "class TuiQuestion(QuestionContext):\n"
                "    @property\n"
                "    def question(self) -> TuiQuestion:\n"
                "        return self\n"
            ),
        },
    )
    findings = _findings(tmp_path)
    assert len(findings) == 1
    assert "QuestionContext.question" in findings[0].summary
    evaluation = SiblingOverrideDetector().evaluate_recipe_for_finding(
        findings[0], _snapshot(tmp_path)
    )
    assert evaluation.recipe is None
    assert "annotate `question` differently" in evaluation.rejection_reason
    assert NeededInput.SOURCE_CHANGE in evaluation.needs
