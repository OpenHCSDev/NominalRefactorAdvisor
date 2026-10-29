"""Codemods that write a spelled declaration as the reference it spells, with their replay checks."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.analysis import analyze_path
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    FindingRecipeSynthesisStatus,
    RefactorRecipeOperation,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.codemod_declaration_reference import (
    ReferenceClassNameOperation,
    ReferenceEnumMemberOperation,
    ReferenceImportedModuleOperation,
    UnquoteResolvedTypeNamesOperation,
)
from nominal_refactor_advisor.detectors import RestatedIdentityDetector
from nominal_refactor_advisor.json_reports import json_report_object

DETECTOR_ID = RestatedIdentityDetector.effective_detector_id()


def _write(root: Path, sources: dict[str, str]) -> None:
    for name, source in sources.items():
        path = root / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def _identity_findings(root: Path):
    return [
        finding for finding in analyze_path(root) if finding.detector_id == DETECTOR_ID
    ]


def _replayed(root: Path) -> tuple[list, dict[str, str]]:
    """Synthesize the detector's recipes, simulate them, and return the plan's records and sources."""

    findings = _identity_findings(root)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root), findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)
    simulation = snapshot.simulate_rewrites(
        plan.document.source_rewrite_batch(snapshot)
    )
    return list(plan.records), dict(simulation.rewritten_sources)


QUOTED = (
    "from __future__ import annotations\n\n"
    "from typing import Annotated, Literal\n\n\n"
    "class Thread:\n    pass\n\n\n"
    "class Owner:\n"
    "    def resolve(self, thread: 'Thread | None', mode: Literal['Thread']) -> 'Owner':\n"
    "        return self\n\n"
    "    def label(self, value: Annotated['Thread', 'Thread']) -> list['Owner']:\n"
    "        return [self]\n"
)


def test_quoted_annotations_are_unquoted_by_one_file_recipe_and_replay_clean(
    tmp_path: Path,
):
    _write(tmp_path, {"m": QUOTED})
    records, rewritten = _replayed(tmp_path)
    executable = [
        record
        for record in records
        if record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    ]
    assert len(executable) == 1
    (operation,) = executable[0].candidate_recipes[0].operations
    assert isinstance(operation, UnquoteResolvedTypeNamesOperation)
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(operation))
        == operation
    )
    (source,) = rewritten.values()
    assert (
        "def resolve(self, thread: Thread | None, mode: Literal['Thread']) -> Owner:"
        in source
    )
    assert (
        "def label(self, value: Annotated[Thread, 'Thread']) -> list[Owner]:" in source
    )
    (tmp_path / "m.py").write_text(source)
    assert _identity_findings(tmp_path) == []


def test_a_quote_an_eager_position_needs_is_left_alone(tmp_path: Path):
    source = (
        "class Table:\n"
        "    def copy(self) -> 'Table':\n        return self\n\n\n"
        "def first(rows: list['Table']) -> 'Table':\n    return rows[0]\n"
    )
    _write(tmp_path, {"m": source})
    _records, rewritten = _replayed(tmp_path)
    (rewritten_source,) = rewritten.values()
    assert "def copy(self) -> 'Table':" in rewritten_source
    assert "def first(rows: list[Table]) -> Table:" in rewritten_source


def test_an_imported_module_spelled_for_an_importer_becomes_its_reference(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/worker": "X = 1\n",
            "pkg/other": "Y = 1\n",
            "pkg/run": (
                "import sys\nfrom importlib import import_module\n\n"
                "import pkg.worker\n\n\n"
                "def load():\n"
                "    import_module('pkg.worker')\n"
                "    import_module('pkg.other')\n"
                "    return (sys.executable, '-m', 'pkg.worker')\n"
            ),
        },
    )
    records, rewritten = _replayed(tmp_path)
    operations = [
        operation
        for record in records
        for recipe in record.candidate_recipes
        for operation in recipe.operations
    ]
    assert [type(operation) for operation in operations] == [
        ReferenceImportedModuleOperation
    ]
    (source,) = rewritten.values()
    assert "    pkg.worker\n" in source
    assert "import_module('pkg.other')" in source
    assert "(sys.executable, '-m', pkg.worker.__name__)" in source


def test_a_module_the_file_does_not_import_keeps_its_prescription(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/worker": "X = 1\n",
            "pkg/run": (
                "import sys\n\n\n"
                "def command():\n    return (sys.executable, '-m', 'pkg.worker')\n"
            ),
        },
    )
    (finding,) = _identity_findings(tmp_path)
    records, rewritten = _replayed(tmp_path)
    (record,) = records
    assert record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    assert record.reason == (
        "no site of this finding can be written as its reference where it stands"
    )
    assert rewritten == {}
    assert "a site no recipe rewrites keeps this prescription" in finding.summary


FAMILY = (
    "from typing import ClassVar, Iterable\n\n\n"
    "class Family:\n"
    "    def __init_subclass__(cls, *, declared_name=None):\n"
    "        super().__init_subclass__()\n\n\n"
    "class Step(Family, declared_name='step'):\n    pass\n\n\n"
    "class Retire(Step, declared_name='retire'):\n    pass\n\n\n"
    "class Migrate(Step, declared_name='migrate'):\n    pass\n\n\n"
)


def test_loop_and_zip_targets_take_the_element_type_of_an_annotated_collection(
    tmp_path: Path,
):
    source = FAMILY + (
        "def runs(pending: tuple[Step, ...], results):\n"
        "    return [record(step.declared_name, done) for step, done in zip(pending, results)]\n\n\n"
        "def arms(cases: Iterable[tuple[type[Step], str]]):\n"
        "    for member, arm in cases:\n"
        "        emit(member.declared_name, arm)\n\n\n"
        "def named():\n    return send(Retire.declared_name)\n\n\n"
        "def untyped(things):\n    for thing in things:\n        emit(thing.declared_name)\n"
    )
    _write(tmp_path, {"m": source})
    summaries = {
        finding.summary.split(" spells")[0]: finding.summary
        for finding in _identity_findings(tmp_path)
    }
    assert sorted(summaries) == ["arms", "named", "runs", "untyped"]
    assert "codec boundary: needs judgment" in summaries["runs"]
    assert "codec boundary: needs judgment" in summaries["arms"]
    assert "codec boundary" not in summaries["named"]


def test_the_scan_rewrite_is_one_candidate_with_one_operation_per_file(
    tmp_path: Path,
):
    _write(tmp_path, {"a": QUOTED, "b": QUOTED.replace("Owner", "Holder")})
    records, rewritten = _replayed(tmp_path)
    executable = [
        record
        for record in records
        if record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    ]
    assert len(executable) == 1
    # Every other finding is removed by that one recipe: covered, naming where
    # its carrier stands, never reported as rejected.
    others = [record for record in records if record not in executable]
    assert others
    assert {record.status for record in others} == {
        FindingRecipeSynthesisStatus.COVERED_BY_CANDIDATE
    }
    assert all(
        record.reason.startswith("covered by the type rewrite carried at ")
        for record in others
    )
    operations = executable[0].candidate_recipes[0].operations
    assert sorted(
        Path(operation.target.file_path).name for operation in operations
    ) == [
        "a.py",
        "b.py",
    ]
    assert len(rewritten) == 2


def test_a_dunder_protocol_name_is_not_a_family_spelling(tmp_path: Path):
    source = (
        "class Root:\n    pass\n\n\n"
        "class First(Root):\n    __registry_key__ = 'kind'\n    marker = '__registry_key__'\n\n\n"
        "class Second(Root):\n    __registry_key__ = 'name'\n    marker = '__slots_key__'\n\n\n"
        "def keyed(node):\n"
        "    return node.attr == '__registry_key__' or node.attr == 'kind'\n"
    )
    _write(tmp_path, {"m": source})
    assert _identity_findings(tmp_path) == []


def test_a_class_name_spelled_as_a_value_becomes_its_name_attribute(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/models": (
                "from dataclasses import dataclass\n\n\n"
                "def renaming(cls):\n    cls.__name__ = 'Other'\n    return cls\n\n\n"
                "@dataclass(frozen=True)\nclass SourceLocation:\n    line: int\n\n\n"
                "@renaming\nclass Renamed:\n    pass\n\n\n"
                "class WorkerPool:\n    pass\n"
            ),
            "pkg/run": (
                "from pkg.models import Renamed, SourceLocation\n\n\n"
                "def kind(node, cls):\n"
                "    if cls.__name__ == 'Renamed':\n        return None\n"
                "    if node.func == 'SourceLocation':\n"
                "        return frozenset({'SourceLocation', 'WorkerPool', 'LaterClass'})\n"
                "    return send('WorkerPool')\n\n\n"
                "LATER = frozenset({'LaterClass'})\n\n\n"
                "class LaterClass:\n    pass\n"
            ),
        },
    )
    records, rewritten = _replayed(tmp_path)
    operations = [
        operation
        for record in records
        for recipe in record.candidate_recipes
        for operation in recipe.operations
    ]
    assert [type(operation) for operation in operations] == [
        ReferenceClassNameOperation
    ]
    (source,) = rewritten.values()
    assert "if node.func == SourceLocation.__name__:" in source
    assert (
        "frozenset({SourceLocation.__name__, WorkerPool.__name__, 'LaterClass'})"
        in source
    )
    # WorkerPool is bound after both names the existing statement imports, so
    # it gets a statement of its own after the file's imports; models imports
    # nothing back, so that statement cannot meet this module half-initialised.
    assert (
        "from pkg.models import Renamed, SourceLocation\n"
        "from pkg.models import WorkerPool\n" in source
    )
    assert "send(WorkerPool.__name__)" in source
    # A decorator from the scan may return another object, and a class the
    # file binds only after the statement holding the literal is not bound
    # when it runs: both stay spelled.
    assert "cls.__name__ == 'Renamed'" in source
    assert "LATER = frozenset({'LaterClass'})" in source


def test_the_files_own_package_spelled_for_an_interpreter_is_dunder_package(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/worker": "X = 1\n",
            "pkg/run": (
                "import sys\n\nfrom . import worker\n\n\n"
                "def command():\n    return (sys.executable, '-m', 'pkg')\n"
            ),
        },
    )
    records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    assert "(sys.executable, '-m', __package__)" in source


def test_a_class_its_from_statement_can_also_import_is_added_to_that_statement(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/stores": (
                "class SingleFileStore:\n    pass\n\n\n"
                "class SqliteStore:\n    pass\n\n\n"
                "class MultiFileStore:\n    pass\n"
            ),
            "pkg/plan": (
                "from pkg.stores import (\n    SqliteStore,\n)\n\n\n"
                "def phases():\n"
                "    return (SqliteStore, 'SingleFileStore', 'MultiFileStore')\n"
            ),
            # A star import binds names no source lists, so nothing here is
            # known to be bound or unbound: the file keeps its spellings.
            "pkg/starred": (
                "from pkg.stores import SqliteStore\nfrom pkg.plan import *\n\n\n"
                "def phases():\n    return (SqliteStore, 'SingleFileStore')\n"
            ),
        },
    )
    records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    # SingleFileStore is bound before SqliteStore in its module, so the
    # statement that imports SqliteStore can import it too; MultiFileStore is
    # bound after, so it gets a statement of its own after the imports.
    assert (
        "    SqliteStore, SingleFileStore,\n)\nfrom pkg.stores import MultiFileStore\n"
        in source
    )
    assert "(SqliteStore, SingleFileStore.__name__, MultiFileStore.__name__)" in source


def test_an_unquoted_union_keeps_its_own_text_and_gains_parentheses_only_as_an_operand(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "m": (
                "class Source:\n    pass\n\n\n"
                "class Holder:\n"
                "    source: 'Source | None' = None\n"
                "    pair: tuple['Source | None', int] = (None, 0)\n"
                "    either: None | 'Source | int' = None\n"
            )
        },
    )
    _records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    assert "    source: Source | None = None\n" in source
    assert "    pair: tuple[Source | None, int] = (None, 0)\n" in source
    assert "    either: None | (Source | int) = None\n" in source


def test_a_new_import_from_a_module_that_imports_this_one_back_is_refused(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/owner": "import pkg.user\n\n\nclass LoopedClass:\n    pass\n",
            "pkg/user": "import sys\n\n\ndef named():\n    return send('LoopedClass')\n",
        },
    )
    _records, rewritten = _replayed(tmp_path)
    assert rewritten == {}


def test_a_metaclass_outside_the_scan_is_judged_from_its_installed_source(
    tmp_path: Path, monkeypatch
):
    site = tmp_path / "site"
    site.mkdir()
    (site / "nra_test_keeping_meta.py").write_text(
        "from abc import ABCMeta\n\n\nclass Keeping(ABCMeta):\n"
        "    def __new__(mcs, name, bases, namespace):\n"
        "        return super().__new__(mcs, name, bases, namespace)\n"
    )
    (site / "nra_test_renaming_meta.py").write_text(
        "class Renaming(type):\n"
        "    def __new__(mcs, name, bases, namespace):\n"
        "        namespace['__name__'] = 'Other'\n"
        "        return super().__new__(mcs, name, bases, namespace)\n"
    )
    monkeypatch.syspath_prepend(str(site))
    root = tmp_path / "scan"
    _write(
        root,
        {
            "pkg/__init__": "",
            "pkg/kinds": (
                "from nra_test_keeping_meta import Keeping\n"
                "from nra_test_renaming_meta import Renaming\n"
                "from nra_test_missing_meta import Missing\n\n\n"
                "class KeptKind(metaclass=Keeping):\n    pass\n\n\n"
                "class RenamedKind(metaclass=Renaming):\n    pass\n\n\n"
                "class UnknownKind(metaclass=Missing):\n    pass\n"
            ),
            "pkg/use": (
                "from pkg.kinds import KeptKind, RenamedKind, UnknownKind\n\n\n"
                "def names():\n"
                "    return ('KeptKind', 'RenamedKind', 'UnknownKind')\n"
            ),
        },
    )
    _records, rewritten = _replayed(root)
    (source,) = rewritten.values()
    assert "return (KeptKind.__name__, 'RenamedKind', 'UnknownKind')" in source


def test_a_new_import_may_take_submodules_from_the_running_package(tmp_path: Path):
    # `from . import helper` in the owner names a submodule of the package the
    # user is running under, not a name that package may not have bound yet.
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/helper": "X = 1\n",
            "pkg/owner": "from . import helper\n\n\nclass NamedOwner:\n    pass\n",
            "pkg/user": "import sys\n\n\ndef named():\n    return send('NamedOwner')\n",
        },
    )
    _records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    assert "import sys\nfrom pkg.owner import NamedOwner\n" in source
    assert "send(NamedOwner.__name__)" in source


def test_a_string_in_a_position_typed_as_an_enum_becomes_its_member(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/attrs": (
                "from enum import StrEnum\n\n\n"
                "class RegistryAttribute(StrEnum):\n"
                "    REGISTRY = '__registry__'\n"
                "    REGISTRY_KEY = 'registry_key'\n"
                "    PLAIN = 'plain'\n\n\n"
                "class OtherKey(StrEnum):\n    REGISTRY = '__registry__'\n"
            ),
            "pkg/use": (
                "from typing import ClassVar\n\n"
                "from pkg.attrs import RegistryAttribute\n\n"
                "DEFAULT: RegistryAttribute = '__registry__'\n\n\n"
                "class Keyed:\n"
                "    key: ClassVar[RegistryAttribute] = 'registry_key'\n"
                "    marker = 'registry_key'\n\n\n"
                "def pick(kind: RegistryAttribute = 'registry_key') -> RegistryAttribute:\n"
                "    if kind == 'plain':\n"
                "        return '__registry__'\n"
                "    return kind\n\n\n"
                "def untyped(attrs, name: str = '__registry__'):\n"
                "    return attrs['__registry__'], getattr(attrs, 'registry_key')\n"
            ),
        },
    )
    records, rewritten = _replayed(tmp_path)
    operations = [
        operation
        for record in records
        for recipe in record.candidate_recipes
        for operation in recipe.operations
    ]
    assert [type(operation) for operation in operations] == [
        ReferenceEnumMemberOperation
    ]
    (source,) = rewritten.values()
    # The declaration names the enum, so the member is the one of that enum,
    # even where another enum shares the value, and even a one-word value.
    assert "DEFAULT: RegistryAttribute = RegistryAttribute.REGISTRY\n" in source
    assert "key: ClassVar[RegistryAttribute] = RegistryAttribute.REGISTRY_KEY\n" in source
    assert (
        "def pick(kind: RegistryAttribute = RegistryAttribute.REGISTRY_KEY)"
        in source
    )
    assert "if kind == RegistryAttribute.PLAIN:" in source
    assert "return RegistryAttribute.REGISTRY\n" in source
    # Equal text where nothing types the position as the enum is not the enum.
    assert "    marker = 'registry_key'\n" in source
    assert "def untyped(attrs, name: str = '__registry__'):" in source
    assert "return attrs['__registry__'], getattr(attrs, 'registry_key')" in source


def test_an_installed_enum_named_by_an_annotation_is_referenced_by_that_name(
    tmp_path: Path, monkeypatch
):
    site = tmp_path / "site" / "nra_test_attrs_pkg"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("from .core import Attr, Holder\n")
    (site / "core.py").write_text(
        "from enum import StrEnum\n\n\n"
        "class Attr(StrEnum):\n    REGISTRY = '__registry__'\n\n\n"
        "class Holder:\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(site.parent))
    root = tmp_path / "scan"
    _write(
        root,
        {
            "pkg/__init__": "",
            "pkg/use": (
                "from nra_test_attrs_pkg import Attr as Key\n\n\n"
                "def read(attrs, key: Key = '__registry__'):\n"
                "    return attrs[key]\n"
            ),
        },
    )
    _records, rewritten = _replayed(root)
    (source,) = rewritten.values()
    assert "def read(attrs, key: Key = Key.REGISTRY):" in source


def test_an_enum_whose_members_do_not_behave_as_their_strings_keeps_its_spelling(
    tmp_path: Path,
):
    # A (str, Enum) member is equal to its string but formats as `Mixed.A`; a
    # plain Enum member is not even equal; a StrEnum that overrides `__eq__`,
    # or composes a base that overrides `__format__` ahead of StrEnum in its
    # MRO, answers as that override does. All still restate the member, but
    # writing the member there would change what the program does.
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/kinds": (
                "from enum import Enum, StrEnum\n\n\n"
                "class Mixed(str, Enum):\n    A = 'mixed_a'\n\n\n"
                "class Plain(Enum):\n    A = 'plain_a'\n\n\n"
                "class Folded(StrEnum):\n    A = 'folded_a'\n\n"
                "    def __eq__(self, other):\n"
                "        return str(self).casefold() == str(other).casefold()\n\n"
                "    __hash__ = StrEnum.__hash__\n\n\n"
                "class Shouting:\n"
                "    def __format__(self, spec):\n        return str(self).upper()\n\n\n"
                "class Loud(Shouting, StrEnum):\n    A = 'loud_a'\n"
            ),
            "pkg/use": (
                "from pkg.kinds import Folded, Loud, Mixed, Plain\n\n"
                "MIXED: Mixed = 'mixed_a'\n"
                "PLAIN: Plain = 'plain_a'\n"
                "FOLDED: Folded = 'folded_a'\n"
                "LOUD: Loud = 'loud_a'\n"
            ),
        },
    )
    findings = _identity_findings(tmp_path)
    for member in ("`Mixed.A`", "`Plain.A`", "`Folded.A`", "`Loud.A`"):
        assert any(f"enum member {member}" in finding.summary for finding in findings)
    records, rewritten = _replayed(tmp_path)
    assert not any(record.candidate_recipes for record in records)
    assert rewritten == {}


def test_a_parameter_rebound_before_it_is_compared_is_not_its_declared_enum(
    tmp_path: Path,
):
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/kinds": "from enum import StrEnum\n\n\nclass Kind(StrEnum):\n    A = 'kind_a'\n",
            "pkg/use": (
                "from pkg.kinds import Kind\n\n\n"
                "def rebound(kind: Kind):\n"
                "    kind = str(kind).upper()\n"
                "    return kind == 'kind_a'\n\n\n"
                "def kept(kind: Kind):\n"
                "    return kind == 'kind_a'\n"
            ),
        },
    )
    _records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    assert "    kind = str(kind).upper()\n    return kind == 'kind_a'\n" in source
    assert "def kept(kind: Kind):\n    return kind == Kind.A\n" in source


def test_a_parameter_any_scope_form_rebinds_is_not_its_declared_enum(tmp_path: Path):
    # `except ... as`, a match capture, a `for` target and a nested `nonlocal`
    # all rebind the parameter; a nested function's own parameter of the same
    # name is a different variable. None of these reads is the declared `Kind`.
    rebinding = (
        "def caught(kind: Kind):\n"
        "    try:\n        pass\n    except ValueError as kind:\n        pass\n"
        "    return kind == 'kind_a'\n\n\n"
        "def captured(kind: Kind, value):\n"
        "    match value:\n        case str() as kind:\n            pass\n"
        "    return kind == 'kind_a'\n\n\n"
        "def looped(kind: Kind, values):\n"
        "    for kind in values:\n        pass\n"
        "    return kind == 'kind_a'\n\n\n"
        "def closed(kind: Kind):\n"
        "    def reset():\n        nonlocal kind\n        kind = 'other'\n"
        "    reset()\n"
        "    return kind == 'kind_a'\n\n\n"
        "def shadowed(kind: Kind):\n"
        "    def inner(kind):\n        return kind == 'kind_a'\n"
        "    return inner\n"
    )
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/kinds": "from enum import StrEnum\n\n\nclass Kind(StrEnum):\n    A = 'kind_a'\n",
            "pkg/use": "from pkg.kinds import Kind\n\n\n" + rebinding,
        },
    )
    assert not any(
        "enum member `Kind.A`" in finding.summary
        for finding in _identity_findings(tmp_path)
    )


def test_only_the_typing_qualifiers_and_real_members_name_an_enum(tmp_path: Path):
    # A class of the project's own named `Final` qualifies nothing, and an
    # enum body's `_ignore_` string is reserved by `enum`, no member's value.
    _write(
        tmp_path,
        {
            "pkg/__init__": "",
            "pkg/kinds": (
                "from enum import StrEnum\n\n\n"
                "class Kind(StrEnum):\n    _ignore_ = 'kind_ignored'\n    A = 'kind_a'\n\n\n"
                "class Final:\n    def __class_getitem__(cls, item):\n        return cls\n"
            ),
            "pkg/use": (
                "import typing\n\n"
                "from pkg.kinds import Final, Kind\n\n"
                "OWN: Final[Kind] = 'kind_a'\n"
                "IGNORED: Kind = 'kind_ignored'\n"
                "QUALIFIED: typing.Final[Kind] = 'kind_a'\n"
            ),
        },
    )
    _records, rewritten = _replayed(tmp_path)
    (source,) = rewritten.values()
    assert "OWN: Final[Kind] = 'kind_a'\n" in source
    assert "IGNORED: Kind = 'kind_ignored'\n" in source
    assert "QUALIFIED: typing.Final[Kind] = Kind.A\n" in source
