"""Relocated probes: each historical relocation from #82 is flagged where it landed."""

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._relocations import (ConstantSubtypeAnswerDetector, LambdaFoldDetector, OwnStateBranchAnswerDetector, ThreadedAbsenceDetector, ZeroOneTupleDetector)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.refactors.constant_subtype_answer import (
    DeleteConstantSubtypeAnswerOperation,
    PushConstantSubtypeAnswerOperation,
)
from nominal_refactor_advisor.scan_cache import ScanCache

from relocation_excerpts import (
    DECLARED_ANNOTATIONS,
    DECLARES_DATACLASS,
    FOLD_BINDING,
    GRAPH_PARAMETERS,
    KEYED_METACLASS_FAMILY,
    SPELLS_AS_CLAUSE,
)

DETECTORS = (
    ConstantSubtypeAnswerDetector,
    OwnStateBranchAnswerDetector,
    LambdaFoldDetector,
    ZeroOneTupleDetector,
    ThreadedAbsenceDetector,
)


def _findings(detector: type, root: Path):
    with ScanCache.scope():
        modules = parse_python_modules(root, use_parse_cache=False)
        return detector().detect(modules, DetectorConfig())


def _sources(tmp_path: Path, sources: dict[str, str]) -> Path:
    for name, source in sources.items():
        (tmp_path / f"{name}.py").write_text(source)
    return tmp_path


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_finding_states_fact_owner_and_prescription(detector):
    spec = detector.finding_spec
    assert spec.certification.value == "certified"
    assert detector.restated.fact in spec.why
    assert detector.restated.owner in spec.why
    if detector is ConstantSubtypeAnswerDetector:
        assert detector.restated.operations == (
            PushConstantSubtypeAnswerOperation,
            DeleteConstantSubtypeAnswerOperation,
        )
    else:
        assert not detector.restated.operations


def test_relocation_patterns_continue_the_stable_sequence():
    assert [
        PatternId.CONSTANT_SUBTYPE_ANSWER,
        PatternId.OWN_STATE_BRANCH_ANSWER,
        PatternId.LAMBDA_FOLD,
        PatternId.ZERO_ONE_TUPLE,
        PatternId.THREADED_ABSENCE,
    ] == [PatternId(number) for number in range(40, 45)]


# ---- the historical relocations, read from git at the commits that made them


@pytest.mark.parametrize(
    ("relocation", "detector", "flagged"),
    (
        (SPELLS_AS_CLAUSE, OwnStateBranchAnswerDetector, "`ImportAlias.spells_as_clause` (property) probes `self.asname`"),
        (FOLD_BINDING, LambdaFoldDetector, "`fold_binding` folds over its callables"),
        (DECLARED_ANNOTATIONS, ZeroOneTupleDetector, "`declared_annotations` encodes absence as a 0/1 tuple"),
        (GRAPH_PARAMETERS, ThreadedAbsenceDetector, "takes `cached_semantic_descent_graph` as `X | None`"),
        # Read off a local, a loop variable and `self.authority`: the declared
        # receiver typing reaches them (#143).
        (DECLARES_DATACLASS, ConstantSubtypeAnswerDetector, "`declares_dataclass` is answered by a constant in 2 members"),
        (KEYED_METACLASS_FAMILY, OwnStateBranchAnswerDetector, "`AutoRegisterClassAuthority.keyed_metaclass_family` (property) probes `self.registry_key_attribute`"),
    ),
)
def test_each_historical_relocation_is_flagged(tmp_path, relocation, detector, flagged):
    summaries = [finding.summary for finding in _findings(detector, relocation.write(tmp_path))]
    assert any(flagged in summary for summary in summaries), summaries


def test_a_constant_answer_read_off_a_declared_family_receiver_is_flagged(tmp_path):
    source = (
        "class Declaration:\n    declares_dataclass = False\n\n"
        "class DataclassDeclaration(Declaration):\n    declares_dataclass = True\n\n"
        "class PlainDeclaration(Declaration):\n    declares_dataclass = False\n\n"
        "def fields(declaration: Declaration):\n"
        "    if declaration.declares_dataclass:\n        return 1\n\n"
        "def guess(declaration):\n"
        "    if declaration.declares_dataclass:\n        return 2\n"
    )
    (finding,) = _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source}))
    counted, _, uncounted = finding.summary.partition("Not counted")
    assert "1 branch test(s) read the name:" in counted
    assert "m.py:11." in counted and "m.py:15" not in counted
    assert "m.py:15." in uncounted
    assert all(location.line != 15 for location in finding.evidence)


def test_an_own_state_answer_branched_on_through_self_is_flagged(tmp_path):
    source = (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class Authority:\n"
        "    registry_key_attribute: str | None\n\n"
        "    @property\n"
        "    def keyed_metaclass_family(self) -> bool:\n"
        "        return self.registry_key_attribute is not None\n\n"
        "    def describe(self):\n"
        "        if self.keyed_metaclass_family:\n            return 1\n"
    )
    summaries = [
        finding.summary
        for finding in _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source}))
    ]
    assert any(
        "`Authority.keyed_metaclass_family` (property) probes `self.registry_key_attribute`" in summary
        for summary in summaries
    ), summaries


# ---- what each detector leaves alone


def test_a_family_member_that_computes_its_answer_is_not_a_constant_family(tmp_path):
    source = (
        "class Base:\n    def flag(self) -> bool:\n        raise NotImplementedError\n\n"
        "class A(Base):\n    def flag(self) -> bool:\n        return True\n\n"
        "class B(Base):\n    def flag(self) -> bool:\n        return self.x > 1\n\n"
        "def use(item):\n    if item.flag():\n        return 1\n"
    )
    assert _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_mutable_object_or_an_unbranched_bool_is_not_an_own_state_answer(tmp_path):
    source = (
        "from dataclasses import dataclass\n\n"
        "class Live:\n    @property\n    def ready(self) -> bool:\n        return self.count > 0\n\n"
        "@dataclass(frozen=True)\nclass Fixed:\n    value: int | None\n"
        "    @property\n    def present(self) -> bool:\n        return self.value is not None\n\n"
        "def use(live, fixed):\n    if live.ready:\n        return fixed.present\n"
    )
    assert _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_defaulted_optional_parameter_is_not_threaded_absence(tmp_path):
    source = (
        "def f(config: int | None = None):\n"
        "    if config is None:\n        config = 1\n    return config\n"
    )
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_function_returning_only_empty_tuples_is_not_a_0_1_tuple(tmp_path):
    source = "def f(x):\n    if x:\n        return ()\n    return ()\n"
    assert _findings(ZeroOneTupleDetector, _sources(tmp_path, {"m": source})) == []


def test_a_family_constant_no_branch_reads_is_data_not_a_subtype_answer(tmp_path):
    source = (
        "from typing import ClassVar\n\n"
        "class Cursor:\n    descending: ClassVar[bool] = True\n\n"
        "class Later(Cursor):\n    descending = False\n\n"
        "def order(cursor):\n    return sorted((), reverse=cursor.descending)\n"
    )
    assert _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_unrelated_classes_sharing_a_member_name_are_not_one_family(tmp_path):
    source = (
        "class A:\n    def flag(self) -> bool:\n        return True\n\n"
        "class B:\n    def flag(self) -> bool:\n        return False\n\n"
        "def use(item):\n    if item.flag():\n        return 1\n"
    )
    assert _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_frozen_bool_that_calls_out_or_probes_no_own_field_is_not_own_state(tmp_path):
    source = (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\nclass Unit:\n    task: object\n    owner: object\n"
        "    @property\n    def busy(self) -> bool:\n        return not self.task.done()\n"
        "    @property\n    def owner_alive(self) -> bool:\n        return self.owner.process_alive\n\n"
        "def use(unit):\n    if unit.busy or unit.owner_alive:\n        return 1\n"
    )
    assert _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_overrides_each_returning_one_arity_are_a_family_not_a_0_1_tuple(tmp_path):
    source = (
        "class Goal:\n    def lines(self):\n        return (self.text,)\n\n"
        "class NoGoal(Goal):\n    def lines(self):\n        return ()\n"
    )
    assert _findings(ZeroOneTupleDetector, _sources(tmp_path, {"m": source})) == []


def test_choosing_between_the_parameter_and_a_replacement_decides_absence(tmp_path):
    source = "def decode(value: list | None):\n    return [] if value is None else value\n"
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_dunder_protocol_signature_is_not_threaded_absence(tmp_path):
    source = (
        "class Boundary:\n"
        "    def __exit__(self, kind: type | None, error: BaseException | None, trace: object):\n"
        "        if error is None:\n            return False\n        return True\n"
    )
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_decoder_settling_the_optional_into_its_return_is_not_threaded_absence(tmp_path):
    source = (
        "class Model:\n    pass\n\nclass Undeclared(Model):\n    pass\n\n"
        "class Declared(Model):\n    def __init__(self, text):\n        self.text = text\n\n"
        "def from_choice(choice: str | None) -> Model:\n"
        "    return Undeclared() if choice is None else Declared(choice)\n\n"
        "def passes_on(value: str | None) -> str | None:\n"
        "    return None if value is None else value.upper()\n"
    )
    findings = _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source}))
    assert [finding.evidence[0].line for finding in findings] == [15]


def test_two_own_fields_compared_are_a_relation_not_a_kind(tmp_path):
    source = (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\nclass Span:\n    start: int\n    end: int\n"
        "    alias: object\n"
        "    @property\n    def single_line(self) -> bool:\n        return self.start == self.end\n"
        "    @property\n    def renamed(self) -> bool:\n        return self.start == self.alias.start\n\n"
        "def use(span):\n    if span.single_line or span.renamed:\n        return 1\n"
    )
    assert _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_comparing_against_mutable_classvar_is_not_own_state_answer(tmp_path):
    source = (
        "from dataclasses import dataclass\nfrom typing import ClassVar\n\n"
        "class Declarations:\n"
        "    generation: ClassVar[int] = 0\n\n"
        "    @classmethod\n"
        "    def changed(cls) -> None:\n"
        "        cls.generation += 1\n\n"
        "@dataclass(frozen=True)\n"
        "class DerivedFact:\n"
        "    generation: int\n"
        "    value: object\n\n"
        "    def current(self) -> bool:\n"
        "        return self.generation == Declarations.generation\n\n"
        "def use(fact: DerivedFact):\n"
        "    if not fact.current():\n"
        "        return None\n"
        "    return fact.value\n"
    )
    assert _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source})) == []


def test_a_branch_read_links_by_its_receivers_class_never_by_the_bare_name(tmp_path):
    # pkg_b never imports pkg_a: its `.visible` reads are another class's
    # member, whether its receiver is declared (`Cursor`) or not (`cursor`).
    for name, source in {
        "pkg_a/__init__": "",
        "pkg_a/roles": (
            "class Role:\n    visible = False\n\n"
            "class Shown(Role):\n    visible = True\n\n"
            "class Hidden(Role):\n    visible = False\n\n"
            "def paint(role: Role):\n    if role.visible:\n        return 1\n"
        ),
        "pkg_b/__init__": "",
        "pkg_b/viewer": (
            "class Cursor:\n    visible = True\n\n"
            "def draw(cursor):\n    if cursor.visible:\n        return 2\n\n"
            "def blink(cursor: Cursor):\n    if cursor.visible:\n        return 3\n"
        ),
    }.items():
        path = tmp_path / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (finding,) = _findings(ConstantSubtypeAnswerDetector, tmp_path)
    assert "1 branch test(s) read the name:" in finding.summary
    assert "pkg_a/roles.py:" in finding.summary
    assert "viewer.py" not in finding.summary
    assert all("viewer.py" not in location.file_path for location in finding.evidence)


_DECLARATION_FAMILY = (
    "class Declaration:\n    declares_dataclass = False\n\n"
    "class DataclassDeclaration(Declaration):\n    declares_dataclass = True\n\n"
    "class PlainDeclaration(Declaration):\n    declares_dataclass = False\n\n"
)


def _branch_lines(source: str) -> list[int]:
    return [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if "if declaration.declares_dataclass" in line or "if cls.declares_dataclass" in line
    ]


def test_an_own_state_answer_of_an_open_class_branched_on_through_self_is_flagged(tmp_path):
    # `Generic[T]` leaves the class open; `self` in its own method is still it.
    source = (
        "from dataclasses import dataclass\n"
        "from typing import Generic, TypeVar\n\n"
        "T = TypeVar('T')\n\n"
        "@dataclass(frozen=True)\n"
        "class Authority(Generic[T]):\n"
        "    registry_key_attribute: T | None\n\n"
        "    @property\n"
        "    def keyed_metaclass_family(self) -> bool:\n"
        "        return self.registry_key_attribute is not None\n\n"
        "    def describe(self):\n"
        "        if self.keyed_metaclass_family:\n            return 1\n"
    )
    summaries = [
        finding.summary
        for finding in _findings(OwnStateBranchAnswerDetector, _sources(tmp_path, {"m": source}))
    ]
    assert any(
        "`Authority.keyed_metaclass_family` (property) probes `self.registry_key_attribute`" in summary
        and "1 branch test(s) select on it:" in summary
        and "/m.py:15." in summary
        for summary in summaries
    ), summaries


def test_a_metaclass_methods_class_parameter_is_shown_never_counted_or_dropped(tmp_path):
    # `cls` in a metaclass method is a class the metaclass creates, not the metaclass.
    source = _DECLARATION_FAMILY + (
        "class Registering(type):\n"
        "    def __init__(cls, name, bases, namespace):\n"
        "        if cls.declares_dataclass:\n            pass\n\n"
        "def fields(declaration: Declaration):\n"
        "    if declaration.declares_dataclass:\n        return 1\n"
    )
    metaclass_line, counted_line = _branch_lines(source)
    (finding,) = _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source}))
    counted, _, uncounted = finding.summary.partition("Not counted")
    assert "1 branch test(s) read the name:" in counted
    assert f"/m.py:{counted_line}." in counted
    assert f"/m.py:{metaclass_line}." in uncounted


def test_a_parameter_rebound_by_a_capture_is_uncounted_but_a_nested_parameter_is_not_a_rebind(
    tmp_path,
):
    source = _DECLARATION_FAMILY + (
        "def caught(declaration: Declaration):\n"
        "    try:\n        pass\n"
        "    except Exception as declaration:\n        pass\n"
        "    if declaration.declares_dataclass:\n        return 1\n\n"
        "def captured(declaration: Declaration, value):\n"
        "    match value:\n        case int() as declaration:\n            pass\n"
        "    if declaration.declares_dataclass:\n        return 2\n\n"
        "def starred(declaration: Declaration, value):\n"
        "    match value:\n        case [*declaration]:\n            pass\n"
        "    if declaration.declares_dataclass:\n        return 3\n\n"
        "def rest(declaration: Declaration, value):\n"
        "    match value:\n        case {'k': _, **declaration}:\n            pass\n"
        "    if declaration.declares_dataclass:\n        return 4\n\n"
        "def nested(declaration: Declaration):\n"
        "    def inner(declaration):\n        return declaration\n"
        "    if declaration.declares_dataclass:\n        return 5\n"
    )
    *rebound_lines, nested_line = _branch_lines(source)
    (finding,) = _findings(ConstantSubtypeAnswerDetector, _sources(tmp_path, {"m": source}))
    counted, _, uncounted = finding.summary.partition("Not counted")
    assert "1 branch test(s) read the name:" in counted
    assert f"/m.py:{nested_line}." in counted
    assert all(
        any(f"/m.py:{line}{end}" in uncounted for end in ",.") for line in rebound_lines
    ), finding.summary


@pytest.mark.parametrize(
    "decoder",
    (
        # toad SessionIdentity.for_thread: the None branch builds the local member
        "class SessionIdentity:\n    @classmethod\n    def for_thread(cls, thread: str, root: str | None):\n"
        "        if root is None:\n            return LocalThreadIdentity(thread)\n"
        "        return RootedThreadIdentity(thread, root)\n\n"
        "class LocalThreadIdentity(SessionIdentity):\n    def __init__(self, thread):\n        self.thread = thread\n\n"
        "class RootedThreadIdentity(SessionIdentity):\n    def __init__(self, thread, root):\n        self.root = root\n",
        # toad CursorStatus.external: the None branch answers with the hidden member class
        "class CursorStatus:\n    @classmethod\n    def external(cls, value: str | None) -> type['CursorStatus']:\n"
        "        if value is None:\n            return HiddenCursorStatus\n        return cls\n\n"
        "class HiddenCursorStatus(CursorStatus):\n    pass\n",
    ),
)
def test_a_none_branch_answering_with_a_family_member_is_the_decoder(tmp_path, decoder):
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": decoder})) == []


def test_a_none_branch_answering_with_a_class_outside_any_family_still_threads(tmp_path):
    source = (
        "class Counter:\n    pass\n\n"
        "def count(value: str | None):\n    if value is None:\n        return Counter()\n    return value\n"
    )
    findings = _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source}))
    assert [finding.evidence[0].line for finding in findings] == [5]


def test_a_boundary_decoder_is_where_absence_is_decided(tmp_path):
    """agent-comms ClientId.decode: `@boundary_decoder` marks the one decode of a wire value."""
    source = (
        "def boundary_decoder(function):\n    return function\n\n"
        "class Wire:\n    @classmethod\n    @boundary_decoder\n"
        "    def decode(cls, value: str | None) -> 'Wire':\n"
        "        if value is None:\n            return EMPTY\n        return cls()\n\n"
        "EMPTY = Wire()\n"
    )
    assert _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source})) == []


def test_a_reactive_callback_takes_the_value_textual_passes(tmp_path):
    """Textual ListView.validate_index and watch_index: the reactive's own value, by its
    declaration; a `watch_` method for no declared reactive is still a probe."""
    source = (
        "from textual.reactive import reactive\n\n"
        "class ListView:\n    index = reactive[int | None](None, init=False)\n\n"
        "    def validate_index(self, index: int | None) -> int | None:\n"
        "        if index is None:\n            return None\n        return max(index, 0)\n\n"
        "    def watch_index(self, index: int | None) -> None:\n"
        "        if index is not None:\n            self.scroll(index)\n\n"
        "    def watch_other(self, other: int | None) -> None:\n"
        "        if other is not None:\n            self.scroll(other)\n"
    )
    findings = _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source}))
    assert [finding.evidence[0].symbol for finding in findings] == ["ListView.watch_other"]


def test_a_none_threaded_through_private_callees_is_reported_where_it_starts(tmp_path):
    """agent-comms TypedTable._query and Textual App._register_child: the private callee
    probes the `None` its callers' defaults pass down; the absence starts at each default.
    `row_type`, which `_query` reads as `row_type or cls` and never probes, starts nothing."""
    source = (
        "class Table:\n"
        "    @classmethod\n    def rows(cls, where: str, limit: int | None = None, row_type: type | None = None) -> list:\n"
        "        return list(cls._query(where, limit, row_type))\n\n"
        "    @classmethod\n    def first(cls, *, limit: int | None = None, row_type: type | None = None) -> list:\n"
        "        return list(cls._query('', limit=limit, row_type=row_type))\n\n"
        "    @classmethod\n    def _query(cls, where: str, limit: int | None, row_type: type | None) -> list:\n"
        "        selected = row_type or cls\n"
        "        clause = ' LIMIT ?' if limit is not None else ''\n"
        "        return [where + clause, *((limit,) if limit is not None else ())]\n\n"
        "    def set_scroll(self, x: float | None) -> None:\n"
        "        if x is not None:\n            self.x = x\n"
    )
    findings = _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source}))
    assert sorted((f.evidence[0].symbol, f.evidence[0].line) for f in findings) == [
        ("Table.first", 7), ("Table.rows", 3), ("Table.set_scroll", 17),
    ]


def test_a_none_passed_on_from_an_undefaulted_parameter_starts_at_that_parameter(tmp_path):
    """The caller's own `X | None` parameter, passed on unprobed, is where the module's
    absence starts. A reactive callback passing its value on (Textual
    ListView.validate_index to _is_valid_index) decides absence, so nothing is reported."""
    from textwrap import dedent

    source = dedent(
        """\
        from textual.reactive import reactive

        class Clock:
            def tick(self, interval: float | None) -> None:
                self._schedule(interval)

            def _schedule(self, interval: float | None) -> None:
                if interval is not None:
                    self.run(interval)

        class ListView:
            index = reactive[int | None](None)

            def validate_index(self, index: int | None) -> int | None:
                return index if self._is_valid_index(index) else None

            def _is_valid_index(self, index: int | None) -> bool:
                if index is None:
                    return False
                return index >= 0
        """
    )
    findings = _findings(ThreadedAbsenceDetector, _sources(tmp_path, {"m": source}))
    assert [(f.evidence[0].symbol, f.evidence[0].line) for f in findings] == [("Clock.tick", 4)]
