"""Restated facts: second copies of a fact that its owner should hold."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.refactors.target_deletion import EraseDeadCompatibilityOperation
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluator,
)
from nominal_refactor_advisor.refactors.module_move import DissolveModuleOperation
from nominal_refactor_advisor.detectors._restated_facts import (
    AbsentAttributeProbeDetector,
    DeadModuleDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    ForwardingFacadeDetector,
    LegacyPathMarkerDetector,
    NamedAttributeDetector,
    OwnTypeTestDetector,
    ProbeOwnerQuery,
    StringKindDispatchDetector,
)
from nominal_refactor_advisor.detectors._restated_identity import RestatedIdentityDetector
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.dispatch_polymorphism import DispatchToPolymorphismOperation


DETECTORS = (
    StringKindDispatchDetector,
    NamedAttributeDetector,
    AbsentAttributeProbeDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    LegacyPathMarkerDetector,
    DeadModuleDetector,
    ForwardingFacadeDetector,
    RestatedIdentityDetector,
    OwnTypeTestDetector,
)


def _findings(tmp_path: Path, detector: type, sources: dict[str, str]):
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)
    detector_id = detector.effective_detector_id()
    return [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == detector_id
    ]


def _lines(findings) -> list[int]:
    return sorted(
        {location.line for finding in findings for location in finding.evidence}
    )


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_finding_states_fact_and_owner_and_is_certified(detector):
    spec = detector.finding_spec
    assert spec.certification.value == "certified"
    assert spec.confidence.value == "high"
    assert spec.why == detector.restated.why
    assert detector.restated.fact in spec.why
    assert detector.restated.owner in spec.why


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_recipe_exists_exactly_when_an_operation_is_declared(detector):
    operations = detector.restated.operations
    assert issubclass(detector, FindingRecipeEvaluator) is bool(operations)


def test_new_patterns_continue_the_stable_sequence():
    assert [
        PatternId.STRING_KIND_DISPATCH,
        PatternId.ATTRIBUTE_ACCESS_BY_NAME,
        PatternId.EXACT_KEY_SET,
        PatternId.FOREIGN_ABSENCE_PROBE,
        PatternId.LEGACY_PATH_MARKER,
        PatternId.DEAD_MODULE,
        PatternId.FORWARDING_FACADE,
        PatternId.RESTATED_IDENTITY,
    ] == [PatternId(number) for number in range(30, 38)]


# ---- string dispatch on a kind

DISPATCH = (
    "def run(kind):\n"
    "    if kind == 'a':\n        return 1\n"
    "    elif kind == 'b':\n        return 2\n"
    "    elif kind in ('c', 'd'):\n        return 3\n"
    "    return 0\n"
)


def test_one_subject_compared_with_several_strings_is_a_finding(tmp_path: Path):
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": DISPATCH})
    assert finding.pattern_id is PatternId.STRING_KIND_DISPATCH
    assert _lines(finding.evidence and [finding]) == [2, 4, 6]
    assert "`kind` is compared with 4 strings ('a', 'b', 'c', 'd')" in finding.summary
    assert "No codemod" not in finding.summary
    assert "DispatchToPolymorphismOperation" in finding.summary


def test_two_strings_or_two_subjects_are_not_a_family(tmp_path: Path):
    source = (
        "def run(kind, mode):\n"
        "    if kind == 'a':\n        return 1\n"
        "    if kind == 'b':\n        return 2\n"
        "    if mode == 'c':\n        return 3\n"
        "    return mode == 'd'\n"
    )
    assert _findings(tmp_path, StringKindDispatchDetector, {"m": source}) == []


def test_a_nested_function_does_not_pool_its_arms_with_its_parent(tmp_path: Path):
    source = (
        "def outer(kind):\n"
        "    if kind == 'a':\n        return 1\n"
        "    def inner(kind):\n"
        "        return kind == 'b' or kind == 'c'\n"
        "    return inner\n"
    )
    assert _findings(tmp_path, StringKindDispatchDetector, {"m": source}) == []


def test_match_string_patterns_are_arms(tmp_path: Path):
    source = (
        "def run(kind):\n    match kind:\n"
        "        case 'a' | 'b':\n            return 1\n"
        "        case 'c':\n            return 2\n"
    )
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": source})
    assert "3 strings" in finding.summary


def test_literal_return_dispatch_is_wired_to_polymorphism_operation(tmp_path: Path):
    source = (
        "def code(kind, value):\n"
        '    """Render one declared mode."""\n'
        "    if kind == 'a':\n        return value + 1\n"
        "    elif kind == 'b':\n        return value + 2\n"
        "    elif kind == 'c':\n        return value + 3\n"
        "    raise ValueError(kind)\n"
    )
    (finding,) = _findings(tmp_path, StringKindDispatchDetector, {"m": source})
    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path), (finding,)
    )
    evaluation = StringKindDispatchDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    operations = evaluation.executable_recipe.operations
    assert [type(operation) for operation in operations] == [
        DispatchToPolymorphismOperation
    ]


# ---- attribute access by name


def test_attribute_access_by_literal_name_or_with_default_is_a_finding(tmp_path: Path):
    source = (
        "def f(obj):\n"
        "    if hasattr(obj, 'name'):\n        return getattr(obj, 'size', 0)\n"
        "    setattr(obj, 'seen', True)\n    delattr(obj, 'seen')\n"
        "    return getattr(obj, KEY, None)\n"
    )
    (finding,) = _findings(
        tmp_path, NamedAttributeDetector, {"m": "KEY = 'owner'\n\n\n" + source}
    )
    assert finding.pattern_id is PatternId.ATTRIBUTE_ACCESS_BY_NAME
    assert _lines([finding]) == [5, 6, 7, 8, 9]
    assert "No codemod operation performs this factoring yet" in finding.summary


def test_dynamic_names_without_a_default_and_plain_reads_are_not_findings(
    tmp_path: Path,
):
    source = (
        "def f(obj, name):\n"
        "    setattr(obj, name, 1)\n    return getattr(obj, name, None) or obj.size\n"
    )
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


def test_a_probe_of_an_imported_module_is_platform_detection_not_a_type_fact(
    tmp_path: Path,
):
    source = "import os\n\n\ndef f():\n    return hasattr(os, 'pidfd_open')\n"
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


def test_a_shadowed_getattr_is_a_method_not_the_builtin(tmp_path: Path):
    source = "def f(registry):\n    return registry.getattr('size', 0)\n"
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


# Each source is one probe no declaration of the scan's can hold, shaped as the
# bench site it was found at; each is a finding before its exemption existed.
EXEMPT_PROBES = {
    "callable_runtime_name": (
        "def name_of(operation):\n"
        "    return getattr(operation, '__qualname__', type(operation).__qualname__)\n"
    ),
    "inspected_partial": (
        "import inspect\n\n\nclass Worker:\n    def __init__(self, work):\n        self._work = work\n\n"
        "    def run(self):\n"
        "        if inspect.iscoroutinefunction(self._work) or hasattr(self._work, 'func')"
        " and inspect.iscoroutinefunction(self._work.func):\n            return 1\n        return 0\n"
    ),
    "ctypes_object": (
        "import ctypes\n\n\ndef open_pidfd(pid):\n    libc = ctypes.CDLL(None, use_errno=True)\n"
        "    function = getattr(libc, 'pidfd_open', None)\n    return function\n"
    ),
    "builtin_iterator": (
        "def compose(result):\n    iterator = None\n    iterator = iter(result)\n"
        "    return hasattr(iterator, 'throw')\n"
    ),
    "dunder_protocol": (
        "def render(error):\n    if hasattr(error, '__rich__'):\n        return error.__rich__()\n    return str(error)\n"
    ),
    "imported_module": (
        "from importlib import import_module\n\n\ndef tools(chosen):\n"
        "    loaded = (import_module(module) for module in chosen)\n"
        "    return {module.__name__: module for module in loaded if hasattr(module, 'main')}\n"
    ),
    "repr_before_init": (
        "class Node:\n    def __init__(self, name):\n        self._name = name\n\n"
        "    def __rich_repr__(self):\n        if hasattr(self, '_name'):\n            yield 'name', self._name\n"
    ),
    "forgotten_super_init": (
        "class Message:\n    def __init__(self):\n        self._prevent = set()\n\n\n"
        "def post(message: Message):\n    if not hasattr(message, '_prevent'):\n"
        "        raise RuntimeError('did you forget to call super().__init__()?')\n    return True\n"
    ),
    "misuse_warning": (
        "class Widget:\n    def check(self, log):\n        if hasattr(self, 'CSS'):\n"
        "            from os import path\n            if not self.screen:\n"
        "                log.warning('CSS is ignored on a widget')\n"
    ),
    "lambda_assignment": (
        "class Details:\n    delivery = None\n\n\n"
        "def watch(details: Details, subscribe):\n"
        "    subscribe(lambda state: setattr(details, 'delivery', state))\n"
    ),
    "isinstance_fallback": (
        "class Visual:\n    pass\n\n\n"
        "def is_visual(obj: object) -> bool:\n    return isinstance(obj, Visual) or hasattr(obj, 'textualize')\n"
    ),
    "isinstance_early_return": (
        "class Visual:\n    pass\n\n\n"
        "def visualize(obj):\n    if isinstance(obj, Visual):\n        return obj\n"
        "    return getattr(obj, 'visualize', None)\n"
    ),
    "isinstance_else": (
        "def deliver(path_or_file):\n    if isinstance(path_or_file, str):\n        name = path_or_file\n"
        "    else:\n        name = getattr(path_or_file, 'name', None)\n    return name\n"
    ),
}


@pytest.mark.parametrize("name", sorted(EXEMPT_PROBES))
def test_a_probe_no_declaration_of_ours_can_hold_is_not_a_finding(tmp_path: Path, name: str):
    assert _findings(tmp_path, NamedAttributeDetector, {"m": EXEMPT_PROBES[name]}) == []
    assert _findings(tmp_path, AbsentAttributeProbeDetector, {"m": EXEMPT_PROBES[name]}) == []


# Each source is the near miss of one exemption: the probe still restates a type's attributes.
RESTATING_PROBES = {
    "partial_name_uninspected": "def f(work):\n    return hasattr(work, 'func')\n",
    "receiver_is_a_parameter": "def f(libc):\n    return getattr(libc, 'pidfd_open', None)\n",
    "receiver_rebound_from_a_parameter": (
        "import ctypes\n\n\ndef f(other):\n    libc = ctypes.CDLL(None)\n    libc = other\n"
        "    return getattr(libc, 'pidfd_open', None)\n"
    ),
    "not_a_module_loader": (
        "def tools(chosen):\n    loaded = (load(module) for module in chosen)\n"
        "    return [module for module in loaded if hasattr(module, 'main')]\n"
    ),
    "repr_of_another_object": (
        "class Node:\n    def __repr__(self):\n        return str(hasattr(self.parent, 'name'))\n"
    ),
    "branch_that_works": (
        "def f(view):\n    if hasattr(view, 'composer'):\n        view.composer.focus()\n"
    ),
    "setattr_outside_a_lambda": "def f(details, state):\n    setattr(details, 'delivery', state)\n",
    "probe_after_isinstance_said_yes": (
        "class Animatable:\n    def blend(self):\n        return self\n\n\n"
        "def duration(value):\n    if not isinstance(value, Animatable):\n        raise TypeError(value)\n"
        "    if hasattr(value, 'get_distance_to'):\n        return 1\n    return 0\n"
    ),
    "isinstance_of_another_value": (
        "class Visual:\n    pass\n\n\n"
        "def f(obj, other):\n    return isinstance(other, Visual) or hasattr(obj, 'textualize')\n"
    ),
}


@pytest.mark.parametrize("name", sorted(RESTATING_PROBES))
def test_the_near_miss_of_each_exemption_stays_a_finding(tmp_path: Path, name: str):
    (finding,) = _findings(tmp_path, NamedAttributeDetector, {"m": RESTATING_PROBES[name]})
    assert finding.pattern_id is PatternId.ATTRIBUTE_ACCESS_BY_NAME


def test_a_marker_our_own_decorator_sets_on_a_function_stays_a_finding(tmp_path: Path):
    source = (
        "def declares(policy):\n    def mark(function):\n        function._height_dependency = policy\n"
        "        return function\n    return mark\n\n\n"
        "class Layout:\n    def arrange(self):\n        return 0\n\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        cls._dependency = getattr(cls.arrange, '_height_dependency', None)\n"
    )
    (finding,) = _findings(tmp_path, NamedAttributeDetector, {"m": source})
    assert _lines([finding]) == [13]


# ---- attribute probes that never succeed

TURN_OWNERS = (
    "from abc import ABC\n\n\nclass TurnOwner(ABC):\n    activity = ''\n\n\n"
    "class NoTurn(TurnOwner):\n    pass\n\n\n"
    "class AgentTurn(TurnOwner):\n    def __init__(self, activity):\n        self.activity = activity\n\n\n"
)


def test_a_probe_for_a_member_no_class_of_the_family_has_never_succeeds(tmp_path: Path):
    source = TURN_OWNERS + (
        "def label(owner: TurnOwner) -> str:\n"
        "    return owner.describe() if hasattr(owner, 'describe') else 'Thinking'\n"
    )
    (finding,) = _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source})
    assert finding.pattern_id is PatternId.ABSENT_ATTRIBUTE_PROBE
    assert _lines([finding]) == [18]
    assert "`m.AgentTurn`, `m.NoTurn`, `m.TurnOwner`" in finding.summary
    assert _findings(tmp_path, NamedAttributeDetector, {"m": source}) == []


def test_a_name_another_class_stores_on_itself_is_still_absent_from_this_family(
    tmp_path: Path,
):
    source = (
        "class Link:\n    underlined = True\n\n\nclass OpenTarget(Link):\n    target = ''\n\n\n"
        "class Action:\n    def __init__(self, url):\n        self.url = url\n\n\n"
        "def open_link(link: Link):\n"
        "    return getattr(link, 'url', None) or getattr(link, 'href', None)\n"
    )
    (finding,) = _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source})
    assert "'url'" in finding.summary and "'href'" in finding.summary


def test_a_declared_attribute_probed_for_a_method_it_lacks_never_succeeds(tmp_path: Path):
    source = (
        "class History:\n    def page(self):\n        return 0\n\n\n"
        "class View:\n    history: History\n\n"
        "    async def wait_content_ready(self):\n"
        "        if hasattr(self.history, 'wait_content_ready'):\n"
        "            await self.history.wait_content_ready()\n\n"
        "    async def retire(self):\n        retire = getattr(self.history, 'retire', None)\n"
        "        if retire is not None:\n            await retire()\n"
    )
    findings = _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source})
    assert _lines(findings) == [10, 14]


def test_an_identity_probed_for_a_name_it_spells_another_way_never_succeeds(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\nclass SessionIdentity(ABC):\n"
        "    @abstractmethod\n    def thread_name(self) -> str: ...\n\n\n"
        "class NoThreadIdentity(SessionIdentity):\n    def thread_name(self) -> str:\n        return ''\n\n\n"
        "def label(identity: SessionIdentity) -> str:\n"
        "    return str(identity.name) if hasattr(identity, 'name') else ''\n"
    )
    (finding,) = _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source})
    assert _lines([finding]) == [15]


@pytest.mark.parametrize(
    "addition",
    [
        "class ActivityTurn(TurnOwner):\n    def describe(self):\n        return self.activity\n",
        "class ActivityTurn(TurnOwner):\n    def start(self):\n        self.describe = lambda: 'busy'\n",
        "def patch(owner):\n    owner.describe = lambda: 'busy'\n",
        "class ActivityTurn(TurnOwner):\n    def __getattr__(self, name):\n        return name\n",
        "class ActivityTurn(TurnOwner):\n    def fill(self, name, value):\n        setattr(self, name, value)\n",
    ],
    ids=["subclass_defines_it", "subclass_stores_it", "stored_elsewhere", "lookup_hook", "computed_store"],
)
def test_a_probe_some_class_of_the_family_can_answer_stays_a_named_attribute(
    tmp_path: Path, addition: str
):
    source = TURN_OWNERS + addition + (
        "\n\ndef label(owner: TurnOwner) -> str:\n"
        "    return owner.describe() if hasattr(owner, 'describe') else 'Thinking'\n"
    )
    assert _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source}) == []
    assert len(_findings(tmp_path, NamedAttributeDetector, {"m": source})) == 1


@pytest.mark.parametrize(
    "receiver",
    ["row: object", "row", "row: 'Unknown'"],
    ids=["object", "unannotated", "unresolved"],
)
def test_a_receiver_no_declaration_types_as_scanned_classes_is_never_absent(
    tmp_path: Path, receiver: str
):
    source = TURN_OWNERS + f"def summary({receiver}) -> str:\n    return getattr(row, 'summary', '')\n"
    assert _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source}) == []
    assert len(_findings(tmp_path, NamedAttributeDetector, {"m": source})) == 1


def test_a_family_behind_a_base_outside_the_scan_is_never_absent(tmp_path: Path):
    source = (
        "from elsewhere import Family\n\n\nclass Owner(Family):\n    busy = False\n\n\n"
        "def label(owner: Owner):\n    return hasattr(owner, 'describe')\n"
    )
    assert _findings(tmp_path, AbsentAttributeProbeDetector, {"m": source}) == []


# The bench shapes: toad's turn owners and line links are `DeclaredFamily`
# members, whose `FamilyMeta` creation the class index proves keeps their order.
DECLARED_FAMILY = (
    "from abc import ABC, ABCMeta\n"
    "from typing import ClassVar, cast\n\n"
    "from metaclass_registry import AutoRegisterMeta, RegistryConfig\n\n\n"
    "class FamilyMeta(AutoRegisterMeta, ABCMeta):\n"
    "    def __new__(mcs, name, bases, namespace, *, affix=None, discovery_package=None, **class_options):\n"
    "        roots = {base.family_root for base in bases if hasattr(base, 'family_root')}\n"
    "        root = next(iter(roots), None)\n"
    "        is_root = affix is not None or not roots\n"
    "        if is_root:\n"
    "            namespace['_family_affix'] = affix if affix is not None else namespace.get('_family_affix', '')\n"
    "            registry = namespace['__registry__'] = {}\n"
    "            namespace['declared_name'] = None\n"
    "            config = namespace['__registry_config__'] = RegistryConfig(\n"
    "                registry_dict=registry, key_attribute='declared_name', skip_if_no_key=True\n"
    "            )\n"
    "        else:\n"
    "            config = root.__registry_config__\n"
    "            namespace['declared_name'] = class_options.pop('declared_name', namespace.get('declared_name', name))\n"
    "        cls = super().__new__(mcs, name, bases, namespace, registry_config=config, **class_options)\n"
    "        if is_root:\n"
    "            cls.family_root = cls\n"
    "        else:\n"
    "            for family_root in roots:\n"
    "                cls.family_root = family_root\n"
    "        return cast(FamilyMeta, cls)\n\n\n"
    "class FamilyMember(ABC):\n"
    "    declared_name: ClassVar[str]\n"
    "    family_root: ClassVar[type]\n\n\n"
    "class DeclaredFamily(FamilyMember, metaclass=FamilyMeta):\n"
    "    _family_affix: ClassVar[str]\n\n"
    "    def __init_subclass__(cls, *, affix=None, declared_name=None, discovery_package=None):\n"
    "        super().__init_subclass__()\n"
)
CONVERSATION_TURN = (
    "from abc import abstractmethod\nfrom dataclasses import dataclass\n\n"
    "from family import DeclaredFamily\n\n\n"
    "class TurnOwner(DeclaredFamily, affix='Turn'):\n"
    "    activity = ''\n\n"
    "    @property\n    def busy(self) -> bool:\n        return bool(self.activity)\n\n"
    "    @property\n    @abstractmethod\n    def session_state(self) -> str | None: ...\n\n"
    "    def with_activity(self, activity: str) -> 'TurnOwner':\n"
    "        return ActivityTurn(activity) if activity else self\n\n\n"
    "@dataclass(frozen=True)\nclass NoTurn(TurnOwner):\n    session_state = None\n\n\n"
    "@dataclass(frozen=True)\nclass ClientTurn(TurnOwner):\n    session_state = 'idle'\n\n\n"
    "@dataclass(frozen=True)\nclass ActivityTurn(TurnOwner):\n    activity: str\n    session_state = 'busy'\n"
)
COMPOSER_SURROUND = (
    "from conversation_turn import TurnOwner\n\n\n"
    "class ComposerSurround:\n"
    "    def _turn_changed(self, owner: TurnOwner) -> None:\n"
    "        if owner.busy:\n"
    "            label = owner.describe() if hasattr(owner, 'describe') else 'Thinking'\n"
    "            self.label = label\n"
)


def test_the_turn_owner_probe_for_describe_never_succeeds(tmp_path: Path):
    sources = {"family": DECLARED_FAMILY, "conversation_turn": CONVERSATION_TURN, "composer_surround": COMPOSER_SURROUND}
    (finding,) = _findings(tmp_path, AbsentAttributeProbeDetector, sources)
    assert finding.primary_evidence.file_path.endswith("composer_surround.py")
    assert _lines([finding]) == [7]
    assert "'describe'" in finding.summary
    assert "`conversation_turn.TurnOwner`" in finding.summary
    # The probe is this detector's alone; the family's own `hasattr(base, ...)` stays named_attribute's.
    assert [
        f for f in _findings(tmp_path, NamedAttributeDetector, sources)
        if f.primary_evidence.file_path.endswith("composer_surround.py")
    ] == []


LINE_BLOCKS = (
    "from typing import ClassVar\n\nfrom family import DeclaredFamily\n\n\n"
    "class LineLink(DeclaredFamily, affix='Link'):\n"
    "    underlined: ClassVar[bool] = True\n"
    "    href_scheme: ClassVar[str] = 'toad-link:'\n\n\n"
    "class NoLink(LineLink):\n    underlined = False\n\n\n"
    "class OpenTarget(LineLink):\n    target = ''\n"
)
PROJECT_LINK = (
    "from abc import abstractmethod\n\nfrom line_blocks import LineLink\n\n\n"
    "class ProjectLink(LineLink):\n"
    "    @property\n    @abstractmethod\n    def href(self) -> str: ...\n\n\n"
    "class ExternalProjectLink(ProjectLink):\n"
    "    def __init__(self, href: str) -> None:\n        self._href = href\n\n"
    "    @property\n    def href(self) -> str:\n        return self._href\n"
)
WINDOW = (
    "from line_blocks import LineLink\n\n\n"
    "class Window:\n"
    "    def open_line_link(self, link: LineLink) -> None:\n"
    "        url = getattr(link, 'url', None) or getattr(link, 'href', None)\n"
    "        if url:\n            self.opened = str(url)\n"
)


def test_a_link_probe_for_href_a_subclass_declares_is_no_absent_probe(tmp_path: Path):
    sources = {"family": DECLARED_FAMILY, "line_blocks": LINE_BLOCKS, "project_link": PROJECT_LINK, "window": WINDOW}
    (finding,) = _findings(tmp_path, AbsentAttributeProbeDetector, sources)
    assert "'url'" in finding.summary and "'href'" not in finding.summary
    (named,) = [
        f for f in _findings(tmp_path, NamedAttributeDetector, sources)
        if f.primary_evidence.file_path.endswith("window.py")
    ]
    assert "'href'" in named.summary and "'url'" not in named.summary


# ---- exact key sets

RECORD = (
    "from dataclasses import dataclass\n\n\n@dataclass\nclass Row:\n"
    "    id: int\n    name: str\n\n\n"
)


def test_mapping_keys_compared_with_a_literal_set_are_a_finding(tmp_path: Path):
    source = (
        "def check(payload):\n"
        "    if set(payload) != {'id', 'name'}:\n        raise ValueError\n"
        "    return payload.keys() <= {'id', 'name', 'x'} and payload['id']\n"
    )
    findings = _findings(tmp_path, ExactKeySetDetector, {"m": source})
    assert [_lines([finding]) for finding in findings] == [[2], [4]]
    assert findings[0].pattern_id is PatternId.EXACT_KEY_SET
    assert "Owner: The owner is a record class whose fields are these keys." in (
        findings[0].summary
    )


def test_a_dataclass_with_exactly_those_fields_is_the_resolved_owner(tmp_path: Path):
    source = RECORD + (
        "def check(payload):\n    return set(payload) == {'id', 'name'} and payload.get('id')\n"
    )
    (finding,) = _findings(tmp_path, ExactKeySetDetector, {"m": source})
    assert "`Row`" in finding.summary and "already declares exactly these fields" in (
        finding.summary
    )
    assert finding.authority_evidence is not None
    assert finding.authority_evidence.symbol == "Row"


def test_generator_membership_over_a_literal_is_a_finding(tmp_path: Path):
    source = "def ok(m):\n    return all(k in m for k in ('id', 'name')) and m['id']\n"
    assert len(_findings(tmp_path, ExactKeySetDetector, {"m": source})) == 1


def test_the_characters_or_items_of_a_non_mapping_are_not_a_record(tmp_path: Path):
    source = (
        "def f(text, names):\n"
        "    return set(text) & {'(', '['} or set(names) == {'a', 'b'}\n"
    )
    assert _findings(tmp_path, ExactKeySetDetector, {"m": source}) == []


def test_one_key_or_a_non_literal_side_is_not_a_record(tmp_path: Path):
    source = (
        "KEYS = {'id', 'name'}\n"
        "def check(payload, other):\n"
        "    return set(payload) == {'id'} or set(payload) == KEYS or set(payload) == set(other)\n"
    )
    assert _findings(tmp_path, ExactKeySetDetector, {"m": source}) == []


# ---- foreign absence probes


def test_another_objects_attribute_compared_with_none_is_a_finding(tmp_path: Path):
    source = (
        "def f(goal):\n"
        "    if goal.paused_at is None:\n        return 1\n"
        "    return goal.paused_at is not None\n"
    )
    (finding,) = _findings(tmp_path, ForeignAbsenceProbeDetector, {"m": source})
    assert finding.pattern_id is PatternId.FOREIGN_ABSENCE_PROBE
    assert "`goal` is probed for absence 2 time(s)" in finding.summary


def test_an_absence_probe_inside_an_assert_observes_and_is_not_a_finding(tmp_path: Path):
    source = (
        "def checked(goal):\n"
        "    assert goal.paused_at is None\n"
        "    assert goal.owner is not None and goal.ready, goal.paused_at is None\n"
        "    if goal.paused_at is None:\n        return 1\n"
        "    return 0\n"
    )
    (finding,) = _findings(tmp_path, ForeignAbsenceProbeDetector, {"m": source})
    assert "`goal` is probed for absence 2 time(s)" in finding.summary
    assert f"{tmp_path}/m.py:3 `goal.paused_at is None`" in finding.summary
    assert f"{tmp_path}/m.py:4 `goal.paused_at is None`" in finding.summary


def test_each_probe_names_its_line_and_the_declared_owner_or_why_not(tmp_path: Path):
    owner = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\nclass Goal:\n    paused_at: int | None = None\n"
    )
    source = (
        "from owner import Goal\n\n\n"
        "def make() -> Goal:\n    return Goal()\n\n\n"
        "def f(goal: Goal | None, other):\n"
        "    made = make()\n"
        "    if goal.paused_at is None:\n        return other.paused_at is None\n"
        "    return made.paused_at is not None\n"
    )
    findings = _findings(
        tmp_path, ForeignAbsenceProbeDetector, {"owner": owner, "m": source}
    )
    by_receiver = {finding.summary.split("`")[1]: finding.summary for finding in findings}
    assert (
        f"{tmp_path / 'm.py'}:10 `goal.paused_at is None`, owner `owner.Goal` "
        f"({tmp_path / 'owner.py'}:5) declares `paused_at: int | None`"
    ) in by_receiver["goal"]
    assert (
        "owner unresolved: the type checker infers `Any`, not one class"
        in by_receiver["other"]
    )
    # `make` declares its return, so the declarations type `made` as the same
    # class the type checker infers, without asking it.
    assert (
        f"{tmp_path / 'm.py'}:12 `made.paused_at is not None`, owner `owner.Goal` "
        f"({tmp_path / 'owner.py'}:5) declares `paused_at: int | None`. "
    ) in by_receiver["made"]


# A lambda declares no return type, so only the type checker types `made`
# (mypy infers the lambda's body; it types an unannotated def's call as `Any`).
_INFERRED_PROBE = (
    "from owner import Goal\n\n\n"
    "make = lambda: Goal()\n\n\n"
    "def f() -> bool:\n    made = make()\n    return made.paused_at is None\n"
)
_GOAL = "class Goal:\n    paused_at: int | None = None\n"


def test_the_inferred_class_is_the_exact_one_when_two_share_a_name(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        ForeignAbsenceProbeDetector,
        {"owner": _GOAL, "other_owner": _GOAL, "m": _INFERRED_PROBE},
    )
    assert "(the type checker infers `owner.Goal`)" in finding.summary


def test_the_declared_class_is_the_exact_one_when_two_share_a_name(tmp_path: Path):
    declared = _INFERRED_PROBE.replace(
        "make = lambda: Goal()", "def make() -> Goal:\n    return Goal()"
    )
    (finding,) = _findings(
        tmp_path,
        ForeignAbsenceProbeDetector,
        {"owner": _GOAL, "other_owner": _GOAL, "m": declared},
    )
    assert f"owner `owner.Goal` ({tmp_path / 'owner.py'}:1) declares" in finding.summary
    assert "type checker" not in finding.summary


def test_without_mypy_the_owner_says_so(tmp_path: Path, monkeypatch):
    from nominal_refactor_advisor import type_checker_session

    monkeypatch.setattr(
        type_checker_session.importlib.util, "find_spec", lambda name: None
    )
    (finding,) = _findings(
        tmp_path, ForeignAbsenceProbeDetector, {"owner": _GOAL, "m": _INFERRED_PROBE}
    )
    assert "owner unresolved: type checker: mypy is not installed" in finding.summary


def test_the_scan_ledger_records_the_build_and_each_lookup(tmp_path: Path):
    from nominal_refactor_advisor.type_checker_session import TypeCheckerTimingLedger

    with TypeCheckerTimingLedger.scope() as ledger:
        _findings(
            tmp_path,
            ForeignAbsenceProbeDetector,
            {"owner": _GOAL, "m": _INFERRED_PROBE},
        )
    assert ledger.timing.lookups == 1
    assert ledger.timing.build_seconds > 0


def test_own_state_modules_and_truth_tests_are_not_foreign_probes(tmp_path: Path):
    source = (
        "import sys\n\n\nclass Goal:\n"
        "    def mine(self):\n        return self.paused_at is None\n"
        "    def other_name(this, other):\n        return this.paused_at is None\n"
        "    def truthy(self, other):\n        return not other.paused_at\n"
        "\n\ndef f():\n    return sys.stdout is None\n"
    )
    assert _findings(tmp_path, ForeignAbsenceProbeDetector, {"m": source}) == []


def test_probe_owner_query_narrows_to_transitive_imports_of_span_files(tmp_path: Path):
    (tmp_path / "base.py").write_text("class Base:\n    paused_at: int | None = None\n")
    (tmp_path / "owner.py").write_text("from base import Base\n\n\nclass Goal(Base):\n    pass\n")
    (tmp_path / "unrelated.py").write_text("class Unrelated:\n    flag: int = 1\n")
    (tmp_path / "m.py").write_text(_INFERRED_PROBE)
    modules = parse_python_modules(tmp_path)
    groups = ForeignAbsenceProbeDetector.compact_module_projection_groups(modules)
    query = ProbeOwnerQuery.of(groups)
    assert {Path(path).name for path in query.file_paths} == {"m.py", "owner.py", "base.py"}


def test_requested_span_elision_keeps_exported_effects_and_preserves_span_types(
    tmp_path: Path,
):
    from nominal_refactor_advisor.type_checker_session import (
        _elide_unrequested_annotated_bodies,
    )

    owner_source = (
        "from collections.abc import AsyncIterator\n\n\n"
        "class Goal:\n"
        "    def __init__(self) -> None:\n"
        "        self.paused_at: int | None = None\n\n"
        "    def heavy(self) -> int:\n"
        "        x = 1 + 2\n"
        "        return x\n\n"
        "    async def stream(self) -> AsyncIterator[int]:\n"
        "        yield 1\n"
    )
    elided = _elide_unrequested_annotated_bodies(owner_source, ())
    assert "self.paused_at: int | None = None" in elided
    assert "yield 1" in elided
    assert "x = 1 + 2" not in elided

    # Line-preserving elision in a file that holds a requested span keeps exact line numbers.
    consumer_source = (
        "from owner import Goal\n\n\n"
        "def helper() -> Goal:\n"
        "    g = Goal()\n"
        "    return g\n\n\n"
        "def check():\n"
        "    goal = helper()\n"
        "    return goal.paused_at is None\n"
    )
    elided_consumer = _elide_unrequested_annotated_bodies(consumer_source, ((11, 11),))
    assert elided_consumer.splitlines()[10] == "    return goal.paused_at is None"
    assert "g = Goal()" not in elided_consumer

    (finding,) = _findings(
        tmp_path,
        ForeignAbsenceProbeDetector,
        {"owner": owner_source, "m": consumer_source},
    )
    assert "owner `owner.Goal`" in finding.summary
    assert "owner unresolved" not in finding.summary


# ---- legacy markers


def test_a_legacy_function_lists_the_consumers_still_on_it(tmp_path: Path):
    findings = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {
            "old": "def load(path):\n    return path\n\n\ndef legacy_load(path):\n    return load(path)\n",
            "user": "from old import legacy_load\n\n\ndef f():\n    return legacy_load('x')\n",
        },
    )
    (finding,) = findings
    assert finding.pattern_id is PatternId.LEGACY_PATH_MARKER
    assert "still use the identifier `legacy_load`" in finding.summary
    assert "user.py" in finding.summary
    assert [location.symbol for location in finding.evidence][-1] == "uses legacy_load"
    assert "EraseDeadCompatibilityOperation" in finding.summary


def test_deprecation_warnings_docstrings_and_parameters_mark_the_old_path(
    tmp_path: Path,
):
    source = (
        "import warnings\n\n\n"
        "def old(path):\n    warnings.warn('x', DeprecationWarning)\n    return path\n\n\n"
        "def documented():\n    '''Deprecated: use new.'''\n\n\n"
        "def f(path, mode=None, legacy_mode=False):\n    return path\n"
    )
    findings = _findings(tmp_path, LegacyPathMarkerDetector, {"m": source})
    assert sorted(finding.evidence[0].symbol for finding in findings) == [
        "documented",
        "f.legacy_mode",
        "old",
    ]


def test_only_the_summary_line_of_a_docstring_declares_the_path_old(tmp_path: Path):
    source = (
        "def parse(path):\n    '''Read a path.\n\n    Replaces the deprecated reader.\n    '''\n"
        "    return path\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_ordinary_declarations_and_plain_imports_of_marked_names_are_not_markers(
    tmp_path: Path,
):
    source = (
        "from elsewhere import legacy_thing\n\n\n"
        "def load(path):\n    '''Read a path.'''\n    return legacy_thing(path)\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_an_unused_legacy_function_gets_an_executable_erasure(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {
            "m": "def load(path):\n    return path\n\n\n"
            "def legacy_load(path):\n    return load(path)\n"
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    (operation,) = evaluation.executable_recipe.operations
    assert isinstance(operation, EraseDeadCompatibilityOperation)


def test_a_legacy_path_with_live_consumers_stops_until_they_migrate(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {
            "old": "def load(path):\n    return path\n\n\n"
            "def legacy_load(path):\n    return load(path)\n",
            "user": "from old import legacy_load\n\n\ndef f():\n    return legacy_load('x')\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "still has 1 consumer(s)" in evaluation.rejection_reason
    assert "user.py" in evaluation.rejection_reason


def test_a_legacy_path_stops_when_sibling_tools_or_tests_import_it(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    src = tmp_path / "src"
    tools = tmp_path / "tools"
    tests = tmp_path / "tests"
    src.mkdir()
    tools.mkdir()
    tests.mkdir()
    (tools / "carry.py").write_text(
        "from old import TeamUpdateStore\n\n\ndef run():\n    return TeamUpdateStore()\n"
    )
    (tests / "test_old.py").write_text(
        "from old import TeamUpdateStore\n\n\ndef test_it():\n    assert TeamUpdateStore\n"
    )
    (finding,) = _findings(
        src,
        LegacyPathMarkerDetector,
        {
            "old": "class TeamUpdateStore:\n"
            '    """Legacy store."""\n'
            "    pass\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(src))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "outside the scanned path" in evaluation.rejection_reason
    assert "carry.py:1" in evaluation.rejection_reason
    assert "test_old.py:1" in evaluation.rejection_reason


def test_a_legacy_path_ignores_comment_only_mentions_outside_scan(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    src = tmp_path / "src"
    tools = tmp_path / "tools"
    src.mkdir()
    tools.mkdir()
    (tools / "notes.py").write_text(
        '"""Replaced TeamUpdateStore long ago."""\n'
        "# TeamUpdateStore is gone\n"
        "VALUE = 1\n"
    )
    (finding,) = _findings(
        src,
        LegacyPathMarkerDetector,
        {
            "old": "class TeamUpdateStore:\n"
            '    """Legacy store."""\n'
            "    pass\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(src))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    (operation,) = evaluation.executable_recipe.operations
    assert isinstance(operation, EraseDeadCompatibilityOperation)


def test_a_legacy_path_uses_git_root_when_no_pyproject_is_present(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    src = tmp_path / "src"
    tools = tmp_path / "tools"
    src.mkdir()
    tools.mkdir()
    (tools / "migrate.py").write_text(
        "from old import legacy_load\n\n\nRESULT = legacy_load('a')\n"
    )
    (finding,) = _findings(
        src,
        LegacyPathMarkerDetector,
        {
            "old": "def load(path):\n    return path\n\n\n"
            "def legacy_load(path):\n    return load(path)\n"
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(src))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "outside the scanned path" in evaluation.rejection_reason
    assert "migrate.py:1" in evaluation.rejection_reason


def test_a_marked_parameter_is_not_widened_to_its_function(tmp_path: Path):
    (finding,) = _findings(
        tmp_path,
        LegacyPathMarkerDetector,
        {"m": "def f(path, mode=None, legacy_mode=False):\n    return path\n"},
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = LegacyPathMarkerDetector().evaluate_recipe_for_finding(
        finding, snapshot
    )
    assert "is not a function or class declaration" in evaluation.rejection_reason


# ---- dead modules


def _dead(tmp_path: Path, sources: dict[str, str]) -> list[str]:
    return sorted(
        finding.evidence[0].symbol
        for finding in _findings(tmp_path, DeadModuleDetector, sources)
    )


def test_a_module_nothing_imports_is_dead(tmp_path: Path):
    sources = {
        "used": "VALUE = 1\n",
        "user": "import used\n\n\ndef f():\n    return used.VALUE\n",
        "orphan": "def lonely():\n    return 1\n",
    }
    assert _dead(tmp_path, sources) == ["orphan", "user"]


def test_modules_that_never_import_each_other_are_standalone_not_dead(tmp_path: Path):
    sources = {"one": "VALUE = 1\n", "two": "VALUE = 2\n"}
    assert _dead(tmp_path, sources) == []


def test_imported_named_main_guarded_and_declared_modules_are_reached(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[project.scripts]\ncli = "pkg.declared:main"\n'
        '\n[project.entry-points."plugins"]\nextra = "pkg.plugged:load"\n'
    )
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    sources = {
        "declared": "def main():\n    return 0\n",
        "plugged": "def load():\n    return 0\n",
        "guarded": "def main():\n    return 0\n\n\nif __name__ == '__main__':\n    main()\n",
        "loaded": "VALUE = 1\n",
        "loader": "import importlib\nMODULE = importlib.import_module('pkg.loaded')\n",
        "anchor": "import pkg.guarded\n",
    }
    for name, source in sources.items():
        (package / f"{name}.py").write_text(source)
    dead = [
        finding.evidence[0].symbol
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    ]
    assert dead == ["pkg.anchor", "pkg.loader"]


def test_a_loader_that_lists_a_package_reaches_every_module_in_it(tmp_path: Path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "loader.py").write_text(
        "import pkgutil\n\n\ndef load(path):\n    return list(pkgutil.iter_modules(path))\n"
    )
    (package / "verb.py").write_text("VALUE = 1\n")
    (tmp_path / "outside.py").write_text("VALUE = 1\n")
    (tmp_path / "anchor.py").write_text(
        "import outside\n\n\nif __name__ == '__main__':\n    print(outside.VALUE)\n"
    )
    (tmp_path / "orphan.py").write_text("VALUE = 1\n")
    dead = [
        finding.evidence[0].symbol
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    ]
    assert dead == ["orphan"]


def _dead_in(tmp_path: Path, files: dict[str, str]) -> list[str]:
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return sorted(
        finding.evidence[0].symbol
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    )


ANCHORED = {
    "outside.py": "VALUE = 1\n",
    "anchor.py": "import outside\n\n\nif __name__ == '__main__':\n    print(outside.VALUE)\n",
    "orphan.py": "VALUE = 1\n",
}


def test_pytest_collects_the_files_the_repository_config_declares(tmp_path: Path):
    files = {
        "pyproject.toml": (
            '[project]\nname = "x"\n\n[tool.pytest.ini_options]\n'
            'testpaths = ["tests"]\npython_files = ["check_*.py"]\n'
        ),
        "tests/check_one.py": "def test_one():\n    assert True\n",
        "tests/conftest.py": "VALUE = 1\n",
        "tests/test_unlisted.py": "def test_two():\n    assert True\n",
        "elsewhere/check_outside.py": "def test_three():\n    assert True\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["elsewhere.check_outside", "orphan", "tests.test_unlisted"]


def test_pytest_defaults_collect_test_files_and_conftest_but_not_helpers(tmp_path: Path):
    files = {
        "tests/test_one.py": "def test_one():\n    assert True\n",
        "tests/two_test.py": "def test_two():\n    assert True\n",
        "tests/conftest.py": "VALUE = 1\n",
        "tests/helper.py": "VALUE = 1\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan", "tests.helper"]


def test_an_enumerator_over_an_imported_package_path_reaches_its_modules(tmp_path: Path):
    files = {
        "plugins/__init__.py": "",
        "plugins/started.py": "VALUE = 1\n",
        "installer.py": (
            "import pkgutil\nimport plugins\n\n\ndef names():\n"
            "    return [info.name for info in pkgutil.iter_modules(plugins.__path__)]\n\n\n"
            "if __name__ == '__main__':\n    print(names())\n"
        ),
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan"]


def test_registry_config_discovery_package_reaches_its_direct_modules_only_when_flat(
    tmp_path: Path,
):
    files = {
        "pkg/__init__.py": "",
        "pkg/base.py": (
            "from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict, RegistryConfig\n\n\n"
            "class Op(metaclass=AutoRegisterMeta):\n"
            "    __registry_config__ = RegistryConfig(\n"
            "        registry_dict=LazyDiscoveryDict(),\n"
            "        discovery_package=f'{__package__}.ops',\n"
            "    )\n"
        ),
        "pkg/ops/__init__.py": "",
        "pkg/ops/move.py": "from pkg.base import Op\n\n\nclass Move(Op):\n    pass\n",
        "pkg/ops/nested/__init__.py": "",
        "pkg/ops/nested/unreached.py": "VALUE = 1\n",
        "pkg/main.py": "from pkg.base import Op\n\n\nif __name__ == '__main__':\n    print(Op)\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan", "pkg.ops.nested.unreached"]


def test_class_header_discovery_package_reaches_family_modules_without_local_iter_modules(
    tmp_path: Path,
):
    files = {
        "app/__init__.py": "",
        "app/verbs/__init__.py": "",
        "app/verbs/base.py": (
            "class DeclaredFamily:\n"
            "    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__()\n\n\n"
            "class CliCommand(DeclaredFamily, discovery_package='app.verbs'):\n"
            "    pass\n"
        ),
        "app/verbs/status.py": (
            "from app.verbs.base import CliCommand\n\n\n"
            "class StatusCliCommand(CliCommand):\n    pass\n"
        ),
        "app/main.py": "from app.verbs.base import CliCommand\n\n\nif __name__ == '__main__':\n    print(CliCommand)\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan"]


def test_a_path_string_reaches_the_file_or_directory_it_names(tmp_path: Path):
    files = {
        "runner.py": (
            "from pathlib import Path\n\nPROBE = Path(__file__).with_name('probe.py')\n"
            "DATA = Path(__file__).parent / 'fixtures/data'\n\n\n"
            "if __name__ == '__main__':\n    print(PROBE, DATA)\n"
        ),
        "probe.py": "VALUE = 1\n",
        "fixtures/data/one.py": "VALUE = 1\n",
        "missing_user.py": "import outside\nGONE = 'not_there.py'\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["missing_user", "orphan"]


def test_a_path_string_naming_an_ancestor_reaches_nothing(tmp_path: Path):
    files = {
        "sub/runner.py": (
            "ROOT = '../'\n\n\nif __name__ == '__main__':\n    print(ROOT)\n"
        ),
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan"]


def test_an_import_inside_a_code_string_reaches_its_module(tmp_path: Path):
    files = {
        "launcher.py": (
            "STUB = 'import sys\\nfrom stub_protocol import serve\\nserve(sys.argv)\\n'\n\n\n"
            "if __name__ == '__main__':\n    print(STUB)\n"
        ),
        "stub_protocol.py": "def serve(argv):\n    return argv\n",
        **ANCHORED,
    }
    assert _dead_in(tmp_path, files) == ["orphan"]


def test_an_empty_dead_module_dissolves(tmp_path: Path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "used.py").write_text("VALUE = 1\n")
    (package / "main.py").write_text(
        "from pkg.used import VALUE\n\n\nif __name__ == '__main__':\n    print(VALUE)\n"
    )
    (package / "shell.py").write_text('"""Moved into pkg.used."""\n')
    (finding,) = [
        finding
        for finding in analyze_path(tmp_path)
        if finding.detector_id == DeadModuleDetector.effective_detector_id()
    ]
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = DeadModuleDetector().evaluate_recipe_for_finding(finding, snapshot)
    (operation,) = evaluation.executable_recipe.operations
    assert isinstance(operation, DissolveModuleOperation)
    assert operation.target.file_path.endswith("pkg/shell.py")


# ---- forwarding facades

FACADE = (
    "class Inner:\n    pass\n\n\n"
    "class Facade:\n"
    "    def __init__(self, inner: Inner):\n        self._inner = inner\n"
    "    @property\n    def size(self):\n        return self._inner.size\n"
    "    @property\n    def label(self):\n        return self._inner.name\n"
    "    def run(self, a, *rest, mode=None, **extra):\n"
    "        return self._inner.run(a, *rest, mode=mode, **extra)\n"
)


def test_members_that_only_forward_to_a_held_object_are_a_facade(tmp_path: Path):
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": FACADE})
    assert finding.pattern_id is PatternId.FORWARDING_FACADE
    assert "`Facade` holds `self._inner` (of class `Inner`)" in finding.summary
    assert "size -> _inner.size" in finding.summary
    assert "label -> _inner.name" in finding.summary
    assert "run -> _inner.run()" in finding.summary
    assert [location.symbol for location in finding.evidence] == [
        "Facade",
        "Facade.size",
        "Facade.label",
        "Facade.run",
    ]
    assert finding.metrics.forwarded_member_count == 3
    assert finding.metrics.impact_delta.lower_bound_removable_loc == 3
    assert "Recipe: `InlineForwarderOperation`" in finding.summary


def test_a_facade_s_forwards_are_inlined_where_each_proves(tmp_path: Path):
    source = (
        "class Inner:\n"
        "    size = 1\n"
        "    def run(self, a):\n        return a\n\n\n"
        "class Facade:\n"
        "    def __init__(self, inner: Inner):\n        self._inner = inner\n"
        "    @property\n    def size(self):\n        return self._inner.size\n"
        "    def run(self, a):\n        return self._inner.run(a)\n\n\n"
        "def use(facade: Facade):\n    return facade.size, facade.run(1)\n"
    )
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": source})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    evaluation = ForwardingFacadeDetector().evaluate_recipe_for_finding(finding, snapshot)
    recipe = evaluation.executable_recipe
    assert [operation.target.qualname for operation in recipe.operations] == [
        "Facade.size",
        "Facade.run",
    ]
    simulation = recipe.simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    (rewritten,) = simulation.simulation.rewritten_sources.values()
    assert "return facade._inner.size, facade._inner.run(1)\n" in rewritten


def test_a_constructed_held_object_names_its_constructor(tmp_path: Path):
    source = (
        "class Facade:\n"
        "    def __init__(self):\n        self.held = build_inner()\n"
        "    @property\n    def a(self):\n        return self.held.a\n"
        "    @property\n    def b(self):\n        return self.held.b\n"
    )
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": source})
    assert "`build_inner(...)`" in finding.summary


def test_a_member_that_adds_behaviour_is_not_a_forward(tmp_path: Path):
    source = (
        "class Facade:\n"
        "    def __init__(self, inner):\n        self._inner = inner\n"
        "    @property\n    def a(self):\n        return self._inner.a\n"
        "    @property\n    def doubled(self):\n        return self._inner.a * 2\n"
        "    def run(self, a):\n        return self._inner.run(a + 1)\n"
        "    def checked(self, a):\n        if a:\n            return self._inner.run(a)\n"
        "    def reordered(self, a, b):\n        return self._inner.run(b, a)\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_two_forwards_to_two_different_held_objects_are_not_a_facade(tmp_path: Path):
    source = (
        "class Pair:\n"
        "    def __init__(self, left, right):\n        self.left = left\n        self.right = right\n"
        "    @property\n    def a(self):\n        return self.left.a\n"
        "    @property\n    def b(self):\n        return self.right.b\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_forwarding_to_an_attribute_the_class_never_holds_is_not_a_facade(
    tmp_path: Path,
):
    source = (
        "class Mixin:\n"
        "    @property\n    def a(self):\n        return self.inner.a\n"
        "    @property\n    def b(self):\n        return self.inner.b\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_a_legacy_word_with_no_replacement_beside_it_is_not_a_marker(tmp_path: Path):
    source = (
        "class LegacyWord:\n    LEGACY = 'legacy'\n\n\n"
        "def parse_legacy_format(text):\n    return text\n\n\n"
        "def f(path, legacy_mode=False):\n    return path\n"
    )
    assert _findings(tmp_path, LegacyPathMarkerDetector, {"m": source}) == []


def test_an_alias_of_a_replacement_under_a_legacy_name_is_a_marker(tmp_path: Path):
    source = "def load(path):\n    return path\n\n\nlegacy_load = load\n"
    (finding,) = _findings(tmp_path, LegacyPathMarkerDetector, {"m": source})
    assert "(alias, of load)" in finding.summary


def test_a_member_overriding_an_inherited_one_is_dispatch_not_a_forward(tmp_path: Path):
    source = (
        "class Base:\n"
        "    @property\n    def rows(self):\n        return ()\n"
        "    def paths(self):\n        return ()\n\n\n"
        "class Decoded(Base):\n"
        "    def __init__(self, tool):\n        self.tool = tool\n"
        "    @property\n    def rows(self):\n        return self.tool.rows\n"
        "    def paths(self):\n        return self.tool.paths()\n"
        "    @property\n    def name(self):\n        return self.tool.name\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_a_base_member_overridden_two_hops_down_is_dispatch_not_a_forward(tmp_path: Path):
    """Reproduces Toad `WireCut.older()` overridden by `AfterSequence(AfterMessage)`:
    a base method overridden anywhere in its subclass hierarchy is polymorphic default
    dispatch, never a forwarding facade member."""

    source = (
        "class WireCut:\n"
        "    def __init__(self, cursor):\n        self.cursor = cursor\n"
        "    def older(self):\n        return self.cursor.older()\n"
        "    def newer(self):\n        return self.cursor.newer()\n\n\n"
        "class AfterMessage(WireCut):\n"
        "    pass\n\n\n"
        "class AfterSequence(AfterMessage):\n"
        "    def older(self):\n        return ('custom', self.cursor)\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_a_cooperative_mi_mixin_implementing_a_sibling_base_is_not_a_facade(tmp_path: Path):
    source = (
        "from abc import ABC, abstractmethod\n\n\n"
        "class UnengagedDefaults:\n"
        "    @property\n    def verdict(self):\n        return None\n"
        "    @property\n    def execution_id(self):\n        return None\n\n\n"
        "class AssignmentState(UnengagedDefaults, ABC):\n"
        "    @property\n    @abstractmethod\n    def mode(self):\n        raise NotImplementedError\n\n\n"
        "class AssignmentDecision:\n"
        "    decision: object\n"
        "    @property\n    def mode(self):\n        return self.decision.mode\n"
        "    @property\n    def verdict(self):\n        return self.decision.verdict\n"
        "    @property\n    def execution_id(self):\n        return self.decision.execution_id\n\n\n"
        "class BoundAssignment(AssignmentDecision, AssignmentState):\n"
        "    pass\n"
    )
    assert _findings(tmp_path, ForwardingFacadeDetector, {"m": source}) == []


def test_an_annotated_held_class_is_named_without_its_metadata(tmp_path: Path):
    source = (
        "from typing import Annotated\n\n\nclass Facade:\n"
        "    inner: Annotated[Inner, 'wire']\n"
        "    @property\n    def a(self):\n        return self.inner.a\n"
        "    @property\n    def b(self):\n        return self.inner.b\n"
    )
    (finding,) = _findings(tmp_path, ForwardingFacadeDetector, {"m": source})
    assert "(of class `Inner`)" in finding.summary


# ---- restated identity

IDENTITY_FAMILY = (
    "class Outcome:\n    pass\n\n\n"
    "class Complete(Outcome, declared_name='complete'):\n    pass\n\n\n"
    "class Failed(Outcome, declared_name='failed'):\n"
    "    def label(self):\n        return self.declared_name\n"
)


LOOKUP_FAMILIES = (
    "from typing import Self\n\n\n"
    "class Kind:\n"
    "    def __init_subclass__(cls, *, declared_name=None):\n"
    "        super().__init_subclass__()\n\n"
    "    @classmethod\n"
    "    def member_named(cls, name: str) -> type[Self]:\n        raise NotImplementedError\n\n\n"
    "class Open(Kind, declared_name='open'):\n    pass\n\n\n"
    "class Shut(Kind, declared_name='shut'):\n    pass\n\n\n"
    "class Store:\n"
    "    def __init_subclass__(cls, *, declared_name=None):\n"
    "        super().__init_subclass__()\n\n\n"
    "class Missing(Store, declared_name='missing'):\n    pass\n\n\n"
    "class Present(Store, declared_name='present'):\n    pass\n"
)


def test_a_looked_up_spelling_restates_the_receivers_own_member(tmp_path: Path):
    source = LOOKUP_FAMILIES + "\n\ndef use():\n    return Kind.member_named('open')\n"
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'open' (`Open`)" in finding.summary


def test_a_spelling_another_family_declares_is_not_the_receivers_member(tmp_path: Path):
    source = LOOKUP_FAMILIES + "\n\ndef use():\n    return Kind.member_named('missing')\n"
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "the `Kind` member spelled 'missing'" in finding.summary
    assert "`Missing`" not in finding.summary


def test_a_test_of_the_lookup_itself_is_no_site(tmp_path: Path):
    source = LOOKUP_FAMILIES + (
        "\n\nimport pytest\n\n\n"
        "def test_decodes():\n"
        "    assert Kind.member_named('open') is Open\n"
        "    assert Kind.member_named('shut') is not Open\n"
        "    with pytest.raises(ValueError):\n"
        "        Kind.member_named('nope')\n"
    )
    assert _findings(tmp_path, RestatedIdentityDetector, {"m": source}) == []


def test_a_quoted_class_name_and_a_passed_class_name_restate_the_class(tmp_path: Path):
    source = (
        "class TurnItem:\n    pass\n\n\n"
        "def first(items: list['TurnItem']) -> 'TurnItem':\n    return items[0]\n\n\n"
        "def declared():\n    return check('TurnItem', 'Item')\n"
    )
    first, declared = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert first.pattern_id is PatternId.RESTATED_IDENTITY
    assert "first spells class `TurnItem` 2 time(s)" in first.summary
    assert "Declared at `TurnItem` (" in first.summary and "m.py:1)" in first.summary
    assert "declared spells class `TurnItem` 1 time(s)" in declared.summary
    assert "UnquoteResolvedTypeNamesOperation" in first.summary


def test_a_compared_family_spelling_restates_the_member(tmp_path: Path):
    source = IDENTITY_FAMILY + (
        "\n\ndef finished(outcome):\n    return outcome.declared_name == 'complete'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert (
        "`declared_name` spelling of a family member 'complete' (`Complete`)"
        in finding.summary
    )


def test_a_sentinel_class_constant_compared_elsewhere_restates_the_class(
    tmp_path: Path,
):
    source = (
        "class Alpha:\n    sigma = 'alpha'\n\n\n"
        "class Beta:\n    sigma = 'beta'\n\n\n"
        "class Gamma(Alpha):\n    sigma = 'gamma'\n\n\n"
        "class Delta(Alpha):\n    sigma = 'delta'\n\n\n"
        "def choose(obj):\n    if obj.sigma == 'gamma':\n        return 1\n    return 2\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'gamma' (`Gamma`)" in finding.summary


def test_shared_constants_one_word_names_and_foreign_strings_are_not_identities(
    tmp_path: Path,
):
    source = (
        "class Read:\n    pass\n\n\n"
        "class Small(Read):\n    unit = 'kb'\n\n\n"
        "class Large(Read):\n    unit = 'kb'\n\n\n"
        "def f(x, path):\n"
        "    if x.unit == 'kb' or path.suffix == '.jsonl':\n        return send('Read')\n"
        "    return x.status == 'completed'\n"
    )
    assert _findings(tmp_path, RestatedIdentityDetector, {"m": source}) == []


def test_a_member_spelling_passed_as_text_is_flattened_but_its_own_read_is_not(
    tmp_path: Path,
):
    source = (
        IDENTITY_FAMILY
        + "\n\ndef emit(send):\n    return send(Complete.declared_name)\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "read off `Complete` as text" in finding.summary
    assert 15 in _lines([finding])


def test_a_module_name_given_to_an_importer_or_run_flag_restates_the_module(
    tmp_path: Path,
):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "worker.py").write_text("X = 1\n")
    source = (
        "import sys\nfrom importlib import import_module\n\n\n"
        "def load():\n    import_module('pkg.worker')\n"
        "    return (sys.executable, '-m', 'pkg.worker', 'pkg.absent')\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"pkg/run": source})
    assert "load spells module `pkg.worker` 2 time(s)" in finding.summary


def test_a_multiword_spelling_of_a_keyword_family_member_restates_it_anywhere(
    tmp_path: Path,
):
    source = (
        "from typing import ClassVar\n\n\n"
        "class Family:\n"
        "    def __init_subclass__(cls, *, declared_name=None):\n"
        "        super().__init_subclass__()\n\n\n"
        "class ClaimedWrite(Family):\n"
        "    declared_name: ClassVar[str] = 'claimed_write'\n\n\n"
        "def decode(name):\n"
        "    if name != 'claimed_write':\n        raise ValueError(name)\n"
        "    return name == 'main'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'claimed_write' (`ClaimedWrite`)" in finding.summary
    assert "line 14" in finding.summary and "'main'" not in finding.summary


def test_a_classvar_slot_declares_the_spelling_every_member_fills(tmp_path: Path):
    source = (
        "from typing import ClassVar\n\n\n"
        "class ToolCall:\n    declared_name: ClassVar[str]\n\n\n"
        "class SelectedWrite(ToolCall):\n"
        "    declared_name: ClassVar[str] = 'selected_write'\n\n\n"
        "class Row:\n    def __init__(self):\n        self.created_at = 0\n\n\n"
        "def decode(name, row):\n"
        "    index(('selected_write', 'created_at'))\n"
        "    return name != 'selected_write'\n"
    )
    (finding,) = _findings(tmp_path, RestatedIdentityDetector, {"m": source})
    assert "'selected_write' (`SelectedWrite`)" in finding.summary
    assert "1 time(s)" in finding.summary


def test_a_quote_the_position_needs_is_not_a_restatement(tmp_path: Path):
    eager = (
        "from typing import ClassVar, TypeVar\n\n"
        "Row = TypeVar('Row', bound='Table')\n\n\n"
        "class Table:\n"
        "    registry: ClassVar[dict[str, type['Table']]] = {}\n\n"
        "    def copy(self) -> 'Table':\n        return self\n"
    )
    assert _findings(tmp_path, RestatedIdentityDetector, {"m": eager}) == []
    (tmp_path / "m.py").unlink()
    lazy = "from __future__ import annotations\n\n" + eager
    findings = _findings(tmp_path, RestatedIdentityDetector, {"lazy": lazy})
    assert sorted(finding.summary.split(" spells")[0] for finding in findings) == [
        "Table",
        "Table.copy",
    ]


@pytest.mark.parametrize(
    ("requires_python", "found"), ((">=3.14,<3.15", 1), (">=3.11", 0))
)
def test_a_quoted_type_checking_name_is_restated_where_annotations_are_lazy(
    tmp_path: Path, requires_python: str, found: int
):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "p"\nrequires-python = "{requires_python}"\n'
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "threads.py").write_text("class Thread:\n    pass\n")
    source = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from .threads import Thread\n\n\n"
        "def resolve(thread: 'Thread') -> None:\n    return None\n"
    )
    findings = _findings(tmp_path, RestatedIdentityDetector, {"pkg/load": source})
    assert len(findings) == found


def test_editing_requires_python_changes_the_cached_answer(tmp_path: Path):
    """Detectors read the declared Python, so the cache key carries it: lowering
    `requires-python` below 3.14 makes the quote required again (11-08)."""

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "p"\nrequires-python = ">=3.14,<3.15"\n')
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "threads.py").write_text("class Thread:\n    pass\n")
    source = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from .threads import Thread\n\n\n"
        "def resolve(thread: 'Thread') -> None:\n    return None\n"
    )
    sources = {"pkg/load": source}
    assert len(_findings(tmp_path, RestatedIdentityDetector, sources)) == 1

    pyproject.write_text('[project]\nname = "p"\nrequires-python = ">=3.11"\n')

    assert _findings(tmp_path, RestatedIdentityDetector, sources) == []


# ---- type tests on the scan's own classes

OWN_CLASSES = (
    "class Shape:\n    pass\n"
    "class Circle(Shape):\n    pass\n"
    "class Square(Shape):\n    pass\n"
    "DEFAULT_MODE = object()\n"
)
OWN_TESTS = (
    "from shapes import Circle, DEFAULT_MODE, Square\n"
    "def area(shape, mode):\n"
    "    if isinstance(shape, Circle):\n        return 1\n"
    "    if type(shape) is Square:\n        return 2\n"
    "    if mode is DEFAULT_MODE:\n        return 3\n"
    "    if isinstance(shape, int):\n        return 4\n"
    "    if type(shape) is int:\n        return 5\n"
    "    return 0\n"
    "def kind(cls):\n"
    "    if cls is Circle:\n        return 'c'\n"
    "    if cls is not Square:\n        return 'n'\n"
    "    return 's' if cls in (Circle, Square) else '?'\n"
    "def matched(shape):\n"
    "    match shape:\n"
    "        case Circle():\n            return 1\n"
    "    return 0\n"
)


def test_every_form_of_type_test_on_an_own_class_is_a_finding(tmp_path: Path):
    findings = _findings(
        tmp_path, OwnTypeTestDetector, {"shapes": OWN_CLASSES, "use": OWN_TESTS}
    )

    assert _lines(findings) == [3, 5, 15, 17, 19, 22]
    assert {finding.evidence[0].symbol for finding in findings} == {"area", "kind", "matched"}


def test_a_type_test_inside_an_assert_observes_and_is_not_a_finding(tmp_path: Path):
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle, Square\n"
                "def checked(shape):\n"
                "    assert isinstance(shape, Circle)\n"
                "    assert type(shape) is not Square, 'no squares'\n"
                "    return shape\n"
                "def area(shape):\n"
                "    if isinstance(shape, Circle):\n        return 1\n"
                "    return 0\n"
            ),
        },
    )

    assert _lines(findings) == [7]


def test_a_test_no_branch_hangs_on_routes_no_arm_and_is_not_a_finding(tmp_path: Path):
    # A comprehension filter keeps each element as the class it is, and a guard that
    # only raises checks a contract: neither supplies an arm the family could own.
    # A bool answer (`any`, a predicate lambda) and a guard with an else still branch.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle, Square\n"
                "def circles(shapes):\n"
                "    return [shape for shape in shapes if isinstance(shape, Circle)]\n"
                "def first(shapes):\n"
                "    return next(s.area for s in shapes if not isinstance(s, Square) and s)\n"
                "def checked(shape):\n"
                "    if not isinstance(shape, Circle):\n"
                "        raise TypeError(shape)\n"
                "    return shape\n"
                "def has_circle(shapes):\n"
                "    return any(isinstance(shape, Circle) for shape in shapes)\n"
                "def picker():\n"
                "    return lambda shape: isinstance(shape, Circle)\n"
                "def area(shape):\n"
                "    if isinstance(shape, Circle):\n"
                "        raise TypeError(shape)\n"
                "    else:\n"
                "        return 1\n"
            ),
        },
    )

    assert _lines(findings) == [11, 13, 15]


def test_a_predicate_lambda_handed_to_a_call_routes_no_arm(tmp_path: Path):
    # `until(lambda: ...)` waits on the predicate and `filter` keeps by it; neither
    # callee knows the family. A lambda bound to a name is a flag and stays.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle\n"
                "async def opened(ui, app):\n"
                "    await ui.until(lambda: isinstance(app.screen, Circle), why='the circle')\n"
                "    await ui.until(condition=lambda: not isinstance(app.screen, Circle))\n"
                "def circles(shapes):\n"
                "    return list(filter(lambda shape: isinstance(shape, Circle), shapes))\n"
                "def flag():\n"
                "    is_circle = lambda shape: isinstance(shape, Circle)\n"
                "    return is_circle\n"
            ),
        },
    )

    assert _lines(findings) == [8]


def test_a_value_first_proved_a_class_is_decoded_by_its_subclass_test(tmp_path: Path):
    # `isinstance(target, type)` says the code did not know the target was a class:
    # the subclass test after it decodes a foreign value. Without that proof the
    # same test on a class the code holds still restates the family.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle\n"
                "def representation(target):\n"
                "    if isinstance(target, type) and issubclass(target, Circle):\n"
                "        return 1\n"
                "    return 0\n"
                "def kind(cls):\n"
                "    if issubclass(cls, Circle):\n"
                "        return 1\n"
                "    return 0\n"
            ),
        },
    )

    assert _lines(findings) == [7]


def test_a_constant_compared_by_identity_is_not_a_type_test(tmp_path: Path):
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {"shapes": OWN_CLASSES, "use": "from shapes import DEFAULT_MODE\ndef f(mode):\n    return mode is DEFAULT_MODE\n"},
    )

    assert findings == []


def test_a_class_imported_inside_the_function_resolves_through_that_import(tmp_path: Path):
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "def area(shape):\n"
                "    from shapes import Circle\n"
                "    if type(shape) is not Circle:\n"
                "        return 0\n"
                "    return 1\n"
            ),
        },
    )

    assert _lines(findings) == [3]


def test_a_local_import_rebound_in_the_function_still_hides_the_name(tmp_path: Path):
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "def area(shape, other):\n"
                "    from shapes import Circle\n"
                "    Circle = other\n"
                "    return isinstance(shape, Circle)\n"
            ),
        },
    )

    assert findings == []


def test_a_test_on_a_value_annotated_object_is_the_decoder_not_slop(tmp_path: Path):
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle, Shape\n"
                "class Ring(Circle):\n"
                "    def __eq__(self, other: object) -> bool:\n"
                "        if not isinstance(other, Circle):\n"
                "            return NotImplemented\n"
                "        return True\n"
                "def decode(value: object) -> Shape:\n"
                "    if type(value) is Circle:\n"
                "        return value\n"
                "    raise TypeError(value)\n"
                "def area(shape: Shape) -> int:\n"
                "    return 1 if isinstance(shape, Circle) else 0\n"
            ),
        },
    )

    assert _lines(findings) == [12]


def test_a_union_with_a_foreign_member_is_decoded_by_the_test(tmp_path: Path):
    # `Circle | str` admits a value the scan declares no class for, so the test
    # tells which one arrived (BOUND-1). A union of own members, an optional own
    # type, or an unresolved member still restates the family.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from typing import Optional, Union\n"
                "from shapes import Circle, Shape, Square\n"
                "def parse(value: Circle | str) -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
                "def legacy(value: Union[Circle, int, None]) -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
                "def quoted(value: 'Circle | bytes') -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
                "def own(value: Circle | Square) -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
                "def optional(value: Optional[Shape]) -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
                "def unknown(value: Circle | Missing) -> int:\n"
                "    if isinstance(value, Circle):\n        return 1\n"
                "    return 0\n"
            ),
        },
    )

    assert _lines(findings) == [16, 20, 24]


def test_a_rebinding_after_the_test_does_not_reach_it(tmp_path: Path):
    # textual `Content.from_markup(cls, markup: str | Content)` tests `markup`, then
    # rebinds it: the test still reads the argument. A binding before the test, or
    # later in a loop around it, does reach it.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle\n"
                "def after(value: Circle | str):\n"
                "    if isinstance(value, Circle):\n        return value\n"
                "    value = value.strip()\n"
                "    return value\n"
                "def before(value: Circle | str):\n"
                "    value = Circle()\n"
                "    if isinstance(value, Circle):\n        return value\n"
                "def looped(value: Circle | str):\n"
                "    while True:\n"
                "        if isinstance(value, Circle):\n            return value\n"
                "        value = Circle()\n"
            ),
        },
    )

    assert _lines(findings) == [9, 13]


def test_an_element_of_a_foreign_union_iterable_is_decoded_by_the_test(tmp_path: Path):
    # textual `OptionList.add_options(new_options: Iterable[OptionListContent])` with
    # `OptionListContent: TypeAlias = "Option | VisualType | None"`: each element is
    # foreign until tested. Elements of the scan's own classes still restate it.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from typing import Iterable\n"
                "from typing_extensions import TypeAlias\n"
                "from shapes import Circle, Shape\n"
                "Content: TypeAlias = 'Circle | str | None'\n"
                "def aliased(values: Iterable[Content]) -> int:\n"
                "    for value in values:\n"
                "        if isinstance(value, Circle):\n            return 1\n"
                "    return 0\n"
                "def starred(*values: Circle | bytes) -> int:\n"
                "    for value in values:\n"
                "        if isinstance(value, Circle):\n            return 1\n"
                "    return 0\n"
                "def own(values: list[Shape]) -> int:\n"
                "    for value in values:\n"
                "        if isinstance(value, Circle):\n            return 1\n"
                "    return 0\n"
            ),
        },
    )

    assert _lines(findings) == [17]


def test_a_methods_operand_of_its_own_family_is_decoded_like_eq(tmp_path: Path):
    # textual `Update.can_replace(self, message: Message)`: whether one message can
    # replace another depends on both, so the test of the receiver's own class
    # decodes the operand. A test of a sibling, or a plain function's test, restates.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle, Shape, Square\n"
                "class Ring(Circle):\n"
                "    def replaces(self, other: Circle) -> bool:\n"
                "        return isinstance(other, Ring)\n"
                "    def sibling(self, other: Shape) -> bool:\n"
                "        return isinstance(other, Square)\n"
                "def plain(other: Shape) -> bool:\n"
                "    return isinstance(other, Ring)\n"
            ),
        },
    )

    assert _lines(findings) == [6, 8]


def test_a_comparison_methods_operand_is_the_decoder_annotated_or_not(tmp_path: Path):
    # The data model hands `__eq__` any object; `merge` receives what its
    # callers pass, so its test still restates the family.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES,
            "use": (
                "from shapes import Circle\n"
                "class Ring(Circle):\n"
                "    def __eq__(self, other):\n"
                "        if not isinstance(other, Circle):\n"
                "            return NotImplemented\n"
                "        return True\n"
                "    def merge(self, other):\n"
                "        return isinstance(other, Circle)\n"
            ),
        },
    )

    assert _lines(findings) == [8]


def test_identity_on_a_sentinel_class_is_not_a_type_test(tmp_path: Path):
    # `class MISSING: pass` extends no scanned class and no scanned class
    # extends it: `value is MISSING` asks which object the value is, not which
    # family member. The membership test beside it still tests `Circle`.
    findings = _findings(
        tmp_path,
        OwnTypeTestDetector,
        {
            "shapes": OWN_CLASSES + "class MISSING:\n    pass\n",
            "use": (
                "from shapes import MISSING, Circle\n"
                "def get(value=MISSING):\n"
                "    if value is MISSING:\n        return None\n"
                "    if value in (MISSING, Circle):\n        return 0\n"
                "    return value\n"
            ),
        },
    )

    assert _lines(findings) == [5]
