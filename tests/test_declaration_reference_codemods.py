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
        "def resolve(self, thread: (Thread | None), mode: Literal['Thread']) -> Owner:"
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
    assert record.status is not FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
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
    assert "frozenset({SourceLocation.__name__, 'WorkerPool', 'LaterClass'})" in source
    # A decorator from the scan may return another object; a class the file
    # does not import, or binds only after the statement holding the literal,
    # stays spelled.
    assert "cls.__name__ == 'Renamed'" in source
    assert "send('WorkerPool')" in source
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
    # bound after, so importing it could fail where the statement did not.
    assert "    SqliteStore, SingleFileStore,\n" in source
    assert "(SqliteStore, SingleFileStore.__name__, 'MultiFileStore')" in source
