"""The two owner-choice moves: create a mixin, extend a partial owner (#67)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import pytest

from nominal_refactor_advisor.analysis import analyze_modules
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap import (
    NO_FLOOR,
    ClassOverlapCandidate,
    ClassOverlapMetrics,
    ComposeExisting,
    ExtractMixin,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.refactors.carrier_fields import (
    CarrierFieldProjection,
    ReplaceFieldsWithCarrierOperation,
)
from nominal_refactor_advisor.class_mro import ClassMroAuthority
from nominal_refactor_advisor.refactors.class_creation import declared_names
from nominal_refactor_advisor.refactors.class_member_authority import ClassMemberMoveProofContext
from nominal_refactor_advisor.refactors.class_overlap import (
    CreateMixinOperation,
    ExtendOwnerOperation,
    OverlapClass,
    movable_members,
)
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_architecture_guards import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodWorkflowScan,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.owner_candidates import OwnerPick
from nominal_refactor_advisor.owner_picks import OwnerPicks
from nominal_refactor_advisor.refactor_concepts import ClassBodyCompositionConcept

RECORDS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_second: int = 0\n"
)

HOLDER = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Holder:\n"
    "    alpha: int\n"
    "    beta: str\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Uses(Holder):\n"
    "    extra: int = 0\n\n\n"
    "def build():\n"
    "    return Uses(alpha=1, beta='b')\n"
)

PROBE = (
    "from pkg.records import First, Second\n"
    "print(First(alpha=1, beta='x', own_first=2).gamma(), "
    "Second(alpha=3, beta='y').gamma())\n"
)


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _apply(root: Path, operation: RefactorRecipeOperation) -> dict[str, str]:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    result = RefactorRecipe(recipe_id="owner-move").with_operation(replayed).simulate(snapshot)
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")
    return result.simulation.rewritten_sources


def _run(root: Path, program: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", program], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def test_a_new_mixin_is_declared_and_each_class_then_composes_it(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    before = _run(tmp_path, PROBE)
    target = SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First")

    _apply(
        tmp_path,
        CreateMixinOperation(
            target=target,
            member_names=("alpha", "beta", "gamma"),
            mixin_name="Lettered",
            docstring="A lettered record.",
        ),
    )

    source = (tmp_path / "pkg/records.py").read_text()
    assert '@dataclass(frozen=True)\nclass Lettered:\n    """A lettered record."""\n' in source
    assert _run(tmp_path, PROBE) == before
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    composes = {
        candidate.restating.qualname
        for candidate in snapshot.class_overlap_verdicts.candidates
        if candidate.move is ComposeExisting
        and candidate.owner.qualname == "Lettered"
        and candidate.permitted
    }
    assert composes == {"First", "Second"}


def test_a_mixin_name_already_bound_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = CreateMixinOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First"),
        member_names=("alpha", "beta", "gamma"),
        mixin_name="Second",
        docstring="A lettered record.",
    )

    with pytest.raises(CodemodOperationPreflightError, match="already bound"):
        operation.source_edits(snapshot)


def test_extending_an_owner_its_descendants_construct_without_the_field_is_refused(
    tmp_path: Path,
) -> None:
    records = RECORDS.replace("    alpha: int\n    beta: str\n", "    alpha: int\n    beta: str\n    delta: float\n")
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/holder.py": HOLDER},
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ExtendOwnerOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/holder.py"), qualname="Holder"),
        anchor=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First"),
        member_names=("delta",),
    )

    with pytest.raises(CodemodOperationPreflightError, match="breaks 1 construction"):
        operation.source_edits(snapshot)


def test_an_owner_gains_the_declarations_it_lacks_and_is_then_composed(
    tmp_path: Path,
) -> None:
    records = RECORDS.replace(
        "    alpha: int\n    beta: str\n",
        "    alpha: int\n    beta: str\n    delta: float = 0.0\n",
    )
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": records, "pkg/holder.py": HOLDER},
    )
    before = _run(tmp_path, PROBE + "from pkg.holder import build\nprint(build())\n")

    _apply(
        tmp_path,
        ExtendOwnerOperation(
            target=SourceRewriteTarget(
                file_path=str(tmp_path / "pkg/holder.py"), qualname="Holder"
            ),
            anchor=SourceRewriteTarget(
                file_path=str(tmp_path / "pkg/records.py"), qualname="First"
            ),
            member_names=("delta", "gamma"),
        ),
    )

    after = _run(tmp_path, PROBE + "from pkg.holder import build\nprint(build())\n")
    assert after.splitlines()[0] == before.splitlines()[0]
    assert after.splitlines()[1] == "Uses(alpha=1, beta='b', delta=0.0, extra=0)"
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert {
        candidate.restating.qualname
        for candidate in snapshot.class_overlap_verdicts.candidates
        if candidate.move is ComposeExisting and candidate.owner.qualname == "Holder"
    } >= {"First", "Second"}


def test_a_goal_run_executes_the_picked_owner_and_reports_the_pick(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    before = _run(tmp_path, PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    unpicked = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=True,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks("picks.json", "0", ()),
    ).run()
    assert [outcome.outcome for outcome in unpicked.owner_picks.outcomes] == ["missing"]
    assert unpicked.stage_count == 0

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name="Lettered",
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class First(Lettered):\n    own_first: int = 0\n" in source
    assert "class Second(Lettered):\n    own_second: int = 0\n" in source
    assert _run(tmp_path, PROBE) == before


ABSTRACT_BLOCK = (
    "    @abstractmethod\n"
    "    def gamma(self) -> str: ...\n\n"
    "    def delta(self) -> str:\n"
    "        return self.gamma() + '!'\n\n"
    "    def epsilon(self) -> str:\n"
    "        return self.delta() * 2\n\n"
)

ABSTRACT_RECORDS = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class First(ABC):\n" + ABSTRACT_BLOCK + "    def own_first(self) -> int:\n        return 1\n\n\n"
    "class Second(ABC):\n" + ABSTRACT_BLOCK + "    def own_second(self) -> int:\n        return 2\n"
)

ABSTRACT_ROOTS = "from abc import ABC\n\n\nclass RootA(ABC):\n    pass\n\n\nclass RootB(ABC):\n    pass\n"

ABSTRACT_PROBE = (
    "from pkg.records import First, Second\n"
    "class Concrete(First):\n"
    "    def gamma(self) -> str:\n"
    "        return 'c'\n"
    "print(sorted(First.__abstractmethods__), sorted(Second.__abstractmethods__), Concrete().epsilon())\n"
    "for abstract in (First, Second):\n"
    "    try:\n"
    "        abstract()\n"
    "    except TypeError:\n"
    "        print(abstract.__name__, 'stays abstract')\n"
)


def _run_picked_mixin(root: Path) -> None:
    modules = parse_python_modules(root, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name="Lettered",
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()
    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]


@pytest.mark.parametrize(
    ("files", "first_header", "second_header", "import_line"),
    (
        ({"pkg/records.py": ABSTRACT_RECORDS}, "class First(Lettered):", "class Second(Lettered):", None),
        (
            {
                "pkg/base.py": ABSTRACT_ROOTS,
                "pkg/records.py": ABSTRACT_RECORDS.replace(
                    "from abc import ABC, abstractmethod\n",
                    "from abc import abstractmethod\n\nfrom pkg.base import RootA, RootB\n",
                )
                .replace("class First(ABC)", "class First(RootA)")
                .replace("class Second(ABC)", "class Second(RootB)"),
            },
            "class First(Lettered, RootA):",
            "class Second(Lettered, RootB):",
            "from abc import ABC, abstractmethod\n",
        ),
    ),
    ids=("abc_bound", "abc_imported"),
)
def test_a_new_mixin_taking_an_abstract_method_is_an_abc_that_keeps_it_abstract(
    tmp_path: Path,
    files: dict[str, str],
    first_header: str,
    second_header: str,
    import_line: str | None,
) -> None:
    """`ABCMeta` reads a composer's abstract names from its namespace and its bases'
    `__abstractmethods__`: the mixin is an `ABC`, so each composer keeps them."""

    _write(tmp_path, {"pkg/__init__.py": "", **files})
    before = _run(tmp_path, ABSTRACT_PROBE)
    assert before.splitlines()[0] == "['gamma'] ['gamma'] c!c!"

    _run_picked_mixin(tmp_path)

    source = (tmp_path / "pkg/records.py").read_text()
    assert (
        'class Lettered(ABC):\n    """A lettered record."""\n\n'
        "    @abstractmethod\n    def gamma(self) -> str: ...\n"
    ) in source, source
    assert f"{first_header}\n    def own_first(self) -> int:\n" in source, source
    assert f"{second_header}\n    def own_second(self) -> int:\n" in source, source
    assert import_line is None or source.startswith(import_line), source
    assert _run(tmp_path, ABSTRACT_PROBE) == before


def test_an_abstract_block_one_composer_is_no_abc_of_is_refused_naming_it(tmp_path: Path) -> None:
    records = ABSTRACT_RECORDS.replace("class Second(ABC):", "class Second:")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = CreateMixinOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First"),
        member_names=("gamma", "delta", "epsilon"),
        mixin_name="Lettered",
        docstring="A lettered record.",
    )

    with pytest.raises(
        CodemodOperationPreflightError, match=r"`Second` does not derive from `abc\.ABC`"
    ):
        operation.source_edits(snapshot)


DISPATCHED_RECORDS = (
    "from abc import ABC, abstractmethod\n\n\n"
    "def handles(*classes):\n"
    "    def decorate(handler):\n"
    "        handler.__handled_classes__ = classes\n"
    "        return handler\n"
    "    return decorate\n\n\n"
    "class Dispatch(ABC):\n"
    "    def __init_subclass__(cls, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
    "        methods = {}\n"
    "        for owner in cls.__mro__:\n"
    "            for name, method in vars(owner).items():\n"
    "                methods.setdefault(name, method)\n"
    "        cls._handler_declarations = tuple(\n"
    "            (name, classes)\n"
    "            for name, method in methods.items()\n"
    "            if (classes := getattr(method, '__handled_classes__', ()))\n"
    "        )\n\n\n"
    "class First(Dispatch):\n"
    "    @abstractmethod\n    def is_pointed(self) -> bool: ...\n\n"
    "    @handles(int)\n    def on_int(self, value):\n        return value\n\n"
    "    @handles(str)\n    def on_str(self, value):\n        return value\n\n\n"
    "class Second(Dispatch):\n"
    "    @abstractmethod\n    def is_pointed(self) -> bool: ...\n\n"
    "    @handles(bytes)\n    def on_bytes(self, value):\n        return value\n"
)

DISPATCHED_PROBE = (
    "from pkg.records import First, Second\n"
    "print(First._handler_declarations, Second._handler_declarations)\n"
    "print(sorted(First.__abstractmethods__), sorted(Second.__abstractmethods__))\n"
)


def test_an_abstract_method_leaves_for_a_new_mixin_past_a_hook_walking_the_mro(tmp_path: Path) -> None:
    """The hook's walk meets `is_pointed` before the handlers, and after them
    once composed; `abstractmethod` stores only `__isabstractmethod__` on it, so
    the hook drops it either way and builds the same handler tables."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": DISPATCHED_RECORDS})
    before = _run(tmp_path, DISPATCHED_PROBE)
    records = str(tmp_path / "pkg/records.py")
    _apply(
        tmp_path,
        CreateMixinOperation(
            target=SourceRewriteTarget(file_path=records, qualname="First"),
            member_names=("is_pointed",),
            mixin_name="Pointed",
            docstring="A record that may be pointed.",
            composers=(
                SourceRewriteTarget(file_path=records, qualname="First"),
                SourceRewriteTarget(file_path=records, qualname="Second"),
            ),
        ),
    )
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class First(Pointed, Dispatch):" in source and "class Second(Pointed, Dispatch):" in source, source
    assert _run(tmp_path, DISPATCHED_PROBE) == before
    assert "on_int" in before and "is_pointed" in before.splitlines()[1]


def test_an_abstract_method_moves_only_into_a_new_abc(tmp_path: Path) -> None:
    """`abstractmethod` is admitted only where the receiving class is a new `ABC`:
    an existing class it moved onto would become abstract itself."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": ABSTRACT_RECORDS})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    first = OverlapClass.require(
        snapshot, SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="First")
    )
    context = ClassMemberMoveProofContext.between(snapshot, first.target, first.target)
    gamma = tuple(statement for statement in first.node.body if declared_names(statement) == {"gamma"})

    with pytest.raises(ValueError, match="custom_method_decorator"):
        movable_members(context, first, gamma)
    movable_members(context.into_new_abc(), first, gamma)

def test_a_pick_citing_a_receipt_its_owner_lacks_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    findings = analyze_modules(
        parse_python_modules(tmp_path, use_parse_cache=False), DetectorConfig()
    )
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    picks = OwnerPicks(
        "picks.json",
        "0",
        (
            OwnerPick(
                finding=mixin.stable_id,
                owner="new_mixin:pkg.records",
                because="holds_declaration",
                name="Lettered",
                judgement="a lettered record",
            ),
            OwnerPick(finding="0000000000", owner="new_mixin:pkg.records", because="new_class"),
        ),
    )

    applied, report = picks.applied(findings)

    (outcome,) = report.outcomes
    assert outcome.outcome == "rejected"
    assert "carries no receipt of" in outcome.detail
    assert report.unmatched == ("0000000000",)
    assert all(
        finding.metrics.owner_pick is None
        for finding in applied
        if isinstance(finding.metrics, ClassOverlapMetrics)
    )


def test_a_pick_without_an_owner_declines_the_finding(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": RECORDS})
    findings = analyze_modules(
        parse_python_modules(tmp_path, use_parse_cache=False), DetectorConfig()
    )
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    picks = OwnerPicks(
        "picks.json",
        "0",
        (OwnerPick(finding=mixin.stable_id, owner=None, because="not one concept"),),
    )

    _applied, report = picks.applied(findings)

    assert [(outcome.outcome, outcome.detail) for outcome in report.outcomes] == [
        ("declined", "not one concept")
    ]


REORDERED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    lead: int\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
    "    own_second: int = 0\n"
)

REORDERED_PROBE = (
    "from pkg.records import First, Second\n"
    "print(First(lead=0, alpha=1, beta='x').gamma(), "
    "Second(alpha=3, beta='y', own_second=4).gamma())\n"
)


def _goal_with_pick(root: Path, name: str) -> object:
    modules = parse_python_modules(root, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    return CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name=name,
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()


def test_a_reordering_mixin_is_keyword_only_and_so_is_each_composer(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REORDERED})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "@dataclass(frozen=True, kw_only=True)\nclass Lettered:\n" in source
    assert "@dataclass(frozen=True, kw_only=True)\nclass First(Lettered):\n    lead: int\n" in source
    assert "@dataclass(frozen=True, kw_only=True)\nclass Second(Lettered):\n" in source
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_a_reordered_composer_that_orders_by_field_position_is_not_composed(
    tmp_path: Path,
) -> None:
    ordered = REORDERED.replace(
        "@dataclass(frozen=True)\nclass First:", "@dataclass(frozen=True, order=True)\nclass First:"
    ).replace(
        "@dataclass(frozen=True)\nclass Second:", "@dataclass(frozen=True, order=True)\nclass Second:"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": ordered})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == ordered


def test_a_reordered_composer_whose_field_order_reaches_a_digest_is_not_composed(
    tmp_path: Path,
) -> None:
    digested = "import hashlib\n" + REORDERED.replace(
        "    own_second: int = 0\n",
        "    own_second: int = 0\n\n"
        "    def identity(self) -> str:\n"
        "        return hashlib.blake2b(repr(self).encode()).hexdigest()\n",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": digested})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == digested


ORDER_CODEC = (
    "from dataclasses import fields\n\n\n"
    "class Codec:\n"
    "    @classmethod\n"
    "    def plans(cls, record: type) -> tuple[str, ...]:\n"
    "        return cls._plan(record)\n\n"
    "    @classmethod\n"
    "    def _plan(cls, record: type) -> tuple[str, ...]:\n"
    "        return tuple(item.name for item in fields(record))\n"
)

ORDER_TABLE = (
    "from pkg.codec import Codec\n\n\n"
    "class Table:\n"
    "    @classmethod\n"
    "    def columns(cls) -> str:\n"
    "        return ', '.join(Codec.plans(cls))\n\n\n"
    "class Other(Table):\n"
    "    pass\n"
)


def test_a_reordered_composer_whose_inherited_code_reads_its_field_order_is_not_composed(
    tmp_path: Path,
) -> None:
    """A base's class-level code hands the class to a codec that walks
    `dataclasses.fields` in order, as a table derives its stored column order:
    reordering the composer's fields changes what it builds, so the pick
    refuses, naming the reader, before any construction is keyworded."""

    tabled = (
        REORDERED.replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\n\nfrom pkg.tables import Table\n",
        )
        .replace("class First:", "class First(Table):")
        .replace("class Second:", "class Second(Table):")
        + "\n\ndef build():\n    return First(0, 1, 'x')\n"
    )
    files = {
        "pkg/__init__.py": "",
        "pkg/codec.py": ORDER_CODEC,
        "pkg/tables.py": ORDER_TABLE,
        "pkg/records.py": tabled,
    }
    _write(tmp_path, files)

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert {path: (tmp_path / path).read_text() for path in files} == files
    refusal = json.dumps(json_report_object(report))
    assert "`Table.columns`" in refusal and "`Codec._plan` -> `dataclasses.fields`" in refusal


def test_a_reordered_composer_that_reads_its_own_fields_in_order_is_not_composed(
    tmp_path: Path,
) -> None:
    """A method that takes the instance's fields as a tuple reads their order."""

    rowed = REORDERED.replace(
        "from dataclasses import dataclass\n", "from dataclasses import astuple, dataclass\n"
    ).replace(
        "    own_second: int = 0\n",
        "    own_second: int = 0\n\n"
        "    def row(self) -> tuple[object, ...]:\n"
        "        return astuple(self)\n",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": rowed})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == rowed
    assert "`Second.row`" in json.dumps(json_report_object(report))


def test_a_picks_file_makes_the_picked_findings_the_only_targets(tmp_path: Path) -> None:
    """A compose-existing recipe nobody picked is in the source; the goal leaves it."""

    holder = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n"
        "    w: int\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Located:\n"
        "    x: int\n"
        "    y: int\n"
        "    w: int\n"
        "    z: int = 0\n"
    )
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/records.py": RECORDS, "pkg/points.py": holder},
    )

    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    unpicked = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=True,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(
            modules=modules, findings=analyze_modules(modules, DetectorConfig())
        ),
        migration_type=ClassBodyCompositionConcept,
    ).run()
    assert unpicked.stage_count >= 1

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 1
    assert (tmp_path / "pkg/points.py").read_text() == holder


def test_a_mixin_one_composer_cannot_compose_is_not_created(tmp_path: Path) -> None:
    """Creating the mixin and composing it is one step: a refused composer refuses all."""

    positional = RECORDS + (
        "\n\ndef build():\n    return Second(1, 'b')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": positional})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    records = str(tmp_path / "pkg/records.py")
    operation = CreateMixinOperation(
        target=SourceRewriteTarget(file_path=records, qualname="First"),
        member_names=("alpha", "beta", "gamma"),
        mixin_name="Lettered",
        docstring="A lettered record.",
        composers=(
            SourceRewriteTarget(file_path=records, qualname="First"),
            SourceRewriteTarget(file_path=records, qualname="Second"),
            SourceRewriteTarget(file_path=records, qualname="Missing"),
        ),
    )

    with pytest.raises(CodemodOperationPreflightError):
        operation.source_edits(snapshot)


def test_a_positional_construction_is_keyworded_first_then_the_pick_composes(
    tmp_path: Path,
) -> None:
    """A reordering pick refused only because a composer is built by position takes
    two stages: every construction bound by keyword, then the composition."""
    positional = REORDERED + "\n\ndef build():\n    return First(0, 1, 'x')\n"
    probe = "from pkg.records import build\nprint(build().gamma(), build().lead)\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": positional})
    before = _run(tmp_path, probe)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 2
    assert "return First(lead=0, alpha=1, beta='x')" in source
    assert "class First(Lettered):\n" in source
    assert _run(tmp_path, probe) == before



def test_a_positional_construction_in_a_script_is_keyworded_first_too(
    tmp_path: Path,
) -> None:
    """The only positional construction sits in Python a string holds and a child
    interpreter runs: the scan counts it, so the goal binds it by keyword first."""
    script = (
        "import subprocess, sys\n\n"
        "SCRIPT = (\n"
        '    "from pkg.records import First\\n"\n'
        '    "print(First(0, 1, \'x\').lead)\\n"\n'
        ")\n\n\n"
        "def run():\n"
        "    return subprocess.run([sys.executable, '-c', SCRIPT], check=True, capture_output=True, text=True).stdout\n"
    )
    probe = "from pkg.scripts import run\nprint(run())\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": REORDERED, "pkg/scripts.py": script})
    before = _run(tmp_path, probe)

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 2
    assert "First(lead=0, alpha=1, beta=\'x\').lead" in (tmp_path / "pkg/scripts.py").read_text()
    assert "class First(Lettered):\n" in (tmp_path / "pkg/records.py").read_text()
    assert _run(tmp_path, probe) == before


KEYWORD_ONLY_FIRST = RECORDS.replace(
    "@dataclass(frozen=True)\nclass First:", "@dataclass(frozen=True, kw_only=True)\nclass First:"
)


def test_a_composer_built_by_position_beside_a_keyword_only_one_is_keyworded_first(
    tmp_path: Path,
) -> None:
    """No field moves, but `First` is keyword-only, so the mixin and each composer
    become keyword-only: the goal binds `Second(3, 'y')` by keyword, then composes."""
    positional = KEYWORD_ONLY_FIRST + "\n\ndef build():\n    return Second(3, 'y')\n"
    probe = PROBE + "from pkg.records import build\nprint(build(), build().gamma())\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": positional})
    before = _run(tmp_path, probe)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 2
    assert "return Second(alpha=3, beta='y')" in source, source
    assert "@dataclass(frozen=True, kw_only=True)\nclass Second(Lettered):\n" in source, source
    assert "class First(Lettered):\n" in source, source
    assert _run(tmp_path, probe) == before


def test_a_composer_handed_on_as_a_value_beside_a_keyword_only_one_is_not_composed(
    tmp_path: Path,
) -> None:
    """Whoever calls the value may bind by position: no keyword rewrite reaches it,
    so the direct positional call is not rewritten either."""
    handed_on = KEYWORD_ONLY_FIRST + (
        "\n\ndef build(make=Second):\n    return make(3, 'y')\n\n\n"
        "def direct():\n    return Second(3, 'y')\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": handed_on})

    report = _goal_with_pick(tmp_path, "Lettered")

    assert report.stage_count == 0
    assert (tmp_path / "pkg/records.py").read_text() == handed_on


DOCUMENTED = REORDERED.replace(
    "class Second:\n    alpha: int\n    beta: str\n\n    def gamma(self) -> str:\n        return",
    "class Second:\n    alpha: int\n    beta: str\n    \"\"\"The letter.\"\"\"\n\n"
    "    def gamma(self) -> str:\n        \"\"\"Letter then number.\"\"\"\n        return",
)


def _documented(first: str | None, second: str | None, field: str | None = None) -> str:
    """REORDERED with `gamma` documented by First and Second as given, and First's
    `beta` field documented as `field`."""
    def gamma(text: str | None) -> str:
        doc = "" if text is None else f'        """{text}"""\n'
        return f"    def gamma(self) -> str:\n{doc}        return"
    first_field = "" if field is None else f'    """{field}"""\n'
    first_class, second_class = REORDERED.split("class Second:")
    first_class = first_class.replace("    def gamma(self) -> str:\n        return", gamma(first))
    first_class = first_class.replace("    beta: str\n", f"    beta: str\n{first_field}")
    second_class = second_class.replace("    def gamma(self) -> str:\n        return", gamma(second))
    return f"{first_class}class Second:{second_class}"


def _dropped(report: object) -> str:
    return json.dumps(json_report_object(report))


def test_a_docstring_only_the_anchor_carries_is_dropped(tmp_path: Path) -> None:
    """First (the anchor) documents `gamma` and `beta`; Second documents neither. The
    docstrings describe First's copies only, so the mixin's members carry none and the
    step names each docstring it drops."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": _documented("First's.", None, "First's letter.")})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "class Lettered" in source
    assert "First's" not in source
    assert "    beta: str\n\n    def gamma(self) -> str:\n        return f'{self.beta}{self.alpha}'\n" in source
    assert "Drops the docstring of `First.gamma`" in _dropped(report)
    assert "Drops the docstring of `First.beta`" in _dropped(report)
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_a_docstring_only_another_composer_carries_is_dropped(tmp_path: Path) -> None:
    """The anchor is undocumented and Second documents `gamma` and `beta`: dropped too."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": DOCUMENTED})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "The letter." not in source
    assert "Letter then number." not in source
    assert "class Second(Lettered):\n    own_second: int = 0\n" in source
    assert "Drops the docstring of `Second.gamma`" in _dropped(report)
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_a_docstring_every_composer_carries_is_kept_once(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": _documented("Letter then number.", "Letter then number.")})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert '    def gamma(self) -> str:\n        """Letter then number."""\n        return' in source
    assert source.count("Letter then number.") == 1
    assert "Drops the docstring" not in _dropped(report)
    assert _run(tmp_path, REORDERED_PROBE) == before


def test_docstrings_worded_differently_are_dropped(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": _documented("First's.", "Second's.")})
    before = _run(tmp_path, REORDERED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "First's." not in source and "Second's." not in source
    assert "    def gamma(self) -> str:\n        return f'{self.beta}{self.alpha}'\n" in source
    assert _run(tmp_path, REORDERED_PROBE) == before


PROPERTIED = REORDERED.replace(
    "    def gamma(self) -> str:\n        return f'{self.beta}{self.alpha}'\n",
    "    @property\n    def gamma(self) -> str:\n        \"\"\"DOC\"\"\"\n        return f'{self.beta}{self.alpha}'\n",
)
PROPERTIED_PROBE = (
    "from pkg.records import First, Second\n"
    "print(First(lead=0, alpha=1, beta='x').gamma, Second(alpha=3, beta='y', own_second=4).gamma)\n"
)


def test_a_property_keeps_only_the_docstring_every_composer_carries(tmp_path: Path) -> None:
    first_class, second_class = PROPERTIED.split("class Second:")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": first_class.replace("DOC", "First's.") + "class Second:" + second_class.replace("DOC", "Second's.")})
    before = _run(tmp_path, PROPERTIED_PROBE)

    report = _goal_with_pick(tmp_path, "Lettered")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "    @property\n    def gamma(self) -> str:\n        return f'{self.beta}{self.alpha}'\n" in source
    assert _run(tmp_path, PROPERTIED_PROBE) == before


COPIED = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    gamma: float\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    gamma: float\n"
    "    own_second: int = 0\n\n\n"
    "def summarize(first: First) -> Second:\n"
    "    return Second(alpha=first.alpha, beta=first.beta, gamma=first.gamma)\n\n\n"
    "def describe(second: Second) -> str:\n"
    "    return f'{second.beta}{second.alpha}{second.gamma}'\n"
)


def test_a_hold_source_pick_holds_the_value_it_copied(tmp_path: Path) -> None:
    """A class built by copying one value's block holds that value instead."""
    probe = (
        "from pkg.records import First, describe, summarize\n"
        "print(describe(summarize(First(alpha=1, beta='x', gamma=2.5))))\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": COPIED})
    before = _run(tmp_path, probe)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (copy,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    hold = next(
        owner for owner in copy.metrics.owner_candidates if owner.kind == "hold_source"
    )

    CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=copy.stable_id,
                    owner=hold.candidate_id,
                    because=hold.receipts[0].kind.value,
                    name="first",
                ),
            ),
        ),
    ).run()

    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Second:\n    first: First\n" in source
    assert "return Second(first=first)" in source
    assert _run(tmp_path, probe) == before


def test_holding_the_source_rewrites_a_reader_in_another_file(
    tmp_path: Path,
) -> None:
    """A typed reader of the copier in another module reads through the held source."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/records.py": COPIED,
            "pkg/report.py": "from pkg.records import Second\n\n\ndef label(second: Second) -> str:\n    return second.beta\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceFieldsWithCarrierOperation(
        target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/records.py"), qualname="Second"),
        field_projections=tuple(
            CarrierFieldProjection(source_field=name, carrier_attribute=name)
            for name in ("alpha", "beta", "gamma")
        ),
        carrier_field_declaration="first: First",
    )

    simulation = RefactorRecipe("hold").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    report = simulation.simulation.rewritten_sources[str(tmp_path / "pkg/report.py")]
    assert "return second.first.beta\n" in report



FIELD_RECORDS = (
    "from collections import OrderedDict\n"
    "from dataclasses import dataclass, field\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    items: list = field(default_factory=list, compare=False)\n"
    "    own_first: int = 0\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Second:\n"
    "    alpha: int\n"
    "    beta: str\n"
    "    items: list = field(default_factory=list, compare=False)\n"
    "    own_second: int = 0\n"
)

FIELD_PROBE = (
    "from pkg.records import First, Second\n"
    "a, b = First(alpha=1, beta='x'), Second(alpha=2, beta='y')\n"
    "a.items.append(1)\n"
    "print(a, b, a.items is not First(alpha=1, beta='x').items)\n"
)


HOISTED = (
    'from dataclasses import dataclass\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Action:\n'
    '    problem: str\n'
    '    order: int\n'
    "    request: str = ''\n"
    '\n'
    "    def bound_to(self, request: str) -> 'Action':\n"
    '        return self\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Other(Action):\n'
    '    revision: int = 0\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Create(Action):\n'
    '    repo: str\n'
    '    text: str\n'
    '\n'
    '    def reserve(self) -> str:\n'
    '        return self.repo + self.text\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Phase:\n'
    '    review: float = 1.0\n'
    '\n'
    '    def due(self) -> float:\n'
    '        return self.review\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class Task(Phase):\n'
    '    problem: str\n'
    '    repo: str\n'
    '    text: str\n'
    '    order: int\n'
    '\n'
    '    def checked(self) -> bool:\n'
    '        return self.review > 0\n'
)


HOISTED_PROBE = (
    "from pkg.records import Create, Other, Task\n"
    "print(Create(problem='p', order=1, repo='r', text='t').reserve(), "
    "Other(problem='q', order=2, revision=3).bound_to('x'), "
    "Task(problem='p', repo='r', text='t', order=1).checked())\n"
)


def test_an_inherited_share_is_hoisted_out_of_its_base_then_the_rest_is_extracted(
    tmp_path: Path,
) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": HOISTED})
    before = _run(tmp_path, HOISTED_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="hoisting_mixin:pkg.records",
                    because="hoists_inherited",
                    name="Filed",
                    judgement="the repository and text a problem is filed with",
                    hoisted_name="Numbered",
                    hoisted_judgement="the problem and the order it belongs to",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Numbered:\n" in source and "class Filed:\n" in source
    assert "class Action(Numbered):\n    request: str = ''\n" in source
    assert "class Create(Filed, Action):\n    def reserve" in source
    assert "class Task(Filed, Numbered, Phase):\n    def checked" in source
    assert _run(tmp_path, HOISTED_PROBE) == before



ALONE = HOISTED.replace(
    "class Action:\n    problem: str\n    order: int\n    request: str = ''\n",
    "class Step:\n    request: str = ''\n\n    def bound_to(self, request: str) -> 'Step':\n        return self\n\n\n"
    "@dataclass(frozen=True, kw_only=True)\nclass Action(Step):\n    problem: str\n    order: int\n",
).replace("    def bound_to(self, request: str) -> 'Action':\n        return self\n\n\n", "\n\n")


def test_a_base_declaring_only_the_hoisted_names_is_hoisted_too(tmp_path: Path) -> None:
    """`Action` declares nothing but the names `Task` restates, so the pair
    reads as a duplicate class whose repair composes `Action`. The hoist extracts
    them instead, and its step is proved as that extraction: `Task` must not
    become a `Step`."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": ALONE})
    before = _run(tmp_path, HOISTED_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (mixin,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="hoisting_mixin:pkg.records",
                    because="hoists_inherited",
                    name="Filed",
                    judgement="the repository and text a problem is filed with",
                    hoisted_name="Numbered",
                    hoisted_judgement="the problem and the order it belongs to",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert "class Numbered:\n" in source and "class Filed:\n" in source
    assert "class Action(Numbered, Step):\n    pass\n" in source
    assert "class Task(Filed, Numbered, Phase):\n    def checked" in source
    assert _run(tmp_path, HOISTED_PROBE) == before



def test_an_owner_that_would_hand_over_its_ancestors_members_is_extracted_instead(
    tmp_path: Path,
) -> None:
    """`Task` restates all `Action` declares, but inheriting `Action` would also
    give it `Step`'s field and method, which nothing requires `Task` to have. The
    block needs a mixin of its own, so the composition stays refused and the
    verdict offers that extraction, asking where it lives; without `Step`,
    composing `Action` is sound and stays."""

    def verdict_for(records: str) -> ClassOverlapCandidate:
        _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
        verdicts = CodemodSourceSnapshot.from_modules(
            parse_python_modules(tmp_path, use_parse_cache=False)
        ).class_overlap_verdicts
        verdict = verdicts.verdict(
            verdicts.bodies.overlap("pkg.records.Action", "pkg.records.Task"), NO_FLOOR
        )
        assert verdict is not None
        return verdict

    refused = verdict_for(ALONE)
    assert refused.move is ComposeExisting
    assert {r.kind for r in refused.refusals} == {OverlapReceiptKind.UNWANTED_SUPPLY}
    extracted = refused.offered
    assert extracted is not None
    assert extracted.move is ExtractMixin and extracted.owner_candidates
    assert not [r for r in extracted.receipts if r.kind is OverlapReceiptKind.UNWANTED_SUPPLY]
    composed = verdict_for(ALONE.replace("class Action(Step):", "class Action:"))
    assert composed.move is ComposeExisting and composed.permitted


def test_a_member_inherited_past_an_unordered_base_is_already_had(tmp_path: Path) -> None:
    """`Task` already inherits `Step`, behind a base the scan cannot order: it has
    `Step`'s members whichever ancestor wins their lookup, so composing `Action`
    hands it nothing new."""

    records = ALONE.replace(
        "from dataclasses import dataclass\n", "from dataclasses import dataclass\n\nfrom vendor import External\n", 1
    ).replace("class Task(Phase):", "class Task(External, Step, Phase):")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    verdicts = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path, use_parse_cache=False)
    ).class_overlap_verdicts
    verdict = verdicts.verdict(
        verdicts.bodies.overlap("pkg.records.Action", "pkg.records.Task"), NO_FLOOR
    )
    assert verdict is not None
    assert not [r for r in verdict.receipts if r.kind is OverlapReceiptKind.UNWANTED_SUPPLY]


LAYOUTS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Viewer:\n"
    "    viewer: str\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Basis:\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    shown: int\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class Identity(Viewer):\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    seen: int\n\n\n"
    "@dataclass(frozen=True, eq=False)\n"
    "class Loose:\n"
    "    root: str\n"
    "    project: str\n"
    "    peer: str\n"
    "    note: str = ''\n"
)

LAYOUTS_PROBE = (
    "from pkg.records import Basis, Identity, Loose\n"
    "made = (Basis(root='r', project='p', peer='q', shown=1), Identity(viewer='v', root='r', project='p', peer='q', seen=2),"
    " Loose(root='r', project='p', peer='q'))\n"
    "for value in made:\n"
    "    kind = type(value)\n"
    "    print(repr(value), value == kind(**{k: getattr(value, k) for k in kind.__dataclass_fields__}),"
    " hasattr(value, '__dict__'), kind.__eq__ is object.__eq__, list(kind.__dataclass_fields__))\n"
)


def test_a_mixin_takes_the_declaration_every_composer_can_inherit(tmp_path: Path) -> None:
    """`Basis` and `Identity` are slotted and `Identity` already has a slotted
    base with a field, so a slotted mixin would be a second layout; `Loose` has
    a `__dict__` and no `__eq__`. The mixin lays out nothing, generates no
    `__eq__`, and each class keeps its layout, equality and fields."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": LAYOUTS})
    before = _run(tmp_path, LAYOUTS_PROBE)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    mixin = next(
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.summary.startswith("`Identity` restate")
    )
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=mixin.stable_id,
                    owner="new_mixin:pkg.records",
                    because="new_class",
                    name="Located",
                    judgement="the root, project and peer a record is about",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked", "covered", "covered"]
    source = (tmp_path / "pkg/records.py").read_text()
    assert (
        "@dataclass(eq=False, frozen=True)\nclass Located:\n"
        '    """The root, project and peer a record is about."""\n\n    __slots__ = ()\n\n    root: str\n'
    ) in source, source
    assert "class Identity(Located, Viewer):\n" in source, source
    assert _run(tmp_path, LAYOUTS_PROBE) == before


def _field_mixin(records: str) -> CreateMixinOperation:
    return CreateMixinOperation(
        target=SourceRewriteTarget(file_path=records, qualname="First"),
        member_names=("alpha", "beta", "items"),
        mixin_name="Lettered",
        docstring="A lettered record with its items.",
        composers=(
            SourceRewriteTarget(file_path=records, qualname="First"),
            SourceRewriteTarget(file_path=records, qualname="Second"),
        ),
    )


def test_a_field_default_moves_into_the_new_mixin(tmp_path: Path) -> None:
    """`dataclasses.field` builds no value that learns its class: the decorator
    consumes it, so the block moves and each instance still gets its own list."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FIELD_RECORDS})
    before = _run(tmp_path, FIELD_PROBE)

    _apply(tmp_path, _field_mixin(str(tmp_path / "pkg/records.py")))

    source = (tmp_path / "pkg/records.py").read_text()
    assert source.count("field(default_factory=list, compare=False)") == 1
    assert "class First(Lettered):" in source and "class Second(Lettered):" in source
    assert _run(tmp_path, FIELD_PROBE) == before


def test_a_mixin_step_names_every_input_it_still_needs(tmp_path: Path) -> None:
    """Independent checks all run: a bad class name and a quoted docstring are
    both named in one refusal, so one scan shows everything the pick lacks."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FIELD_RECORDS})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = replace(_field_mixin(str(tmp_path / "pkg/records.py")), mixin_name="class", docstring='say """hi"""')

    with pytest.raises(CodemodOperationPreflightError) as refused:
        operation.source_edits(snapshot)
    assert "one-line docstring" in str(refused.value) and "`class` is no class name" in str(refused.value)


def test_every_raises_one_refusal_with_each_need() -> None:
    def proof() -> None:
        raise NeedsInput("a proof", NeededInput.PROOF)

    def pick() -> None:
        raise NeedsInput("a pick", NeededInput.OWNER_PICK)

    with pytest.raises(NeedsInput) as refused:
        NeedsInput.every(proof, lambda: None, pick)
    assert str(refused.value) == "a proof; a pick"
    assert refused.value.needs == (NeededInput.OWNER_PICK, NeededInput.PROOF)
    NeedsInput.every(lambda: None)


def test_a_moved_value_from_an_unindexed_class_still_refuses(tmp_path: Path) -> None:
    """Only `dataclasses.field` is admitted: any other call may build a field or a
    value that learns its class, so the order stays unproved and the move refuses."""

    records = FIELD_RECORDS.replace(
        "field(default_factory=list, compare=False)", "OrderedDict()"
    ).replace("    items: list", "    items: OrderedDict")
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    with pytest.raises(CodemodOperationPreflightError, match="field order .* is unknown"):
        _field_mixin(str(tmp_path / "pkg/records.py")).source_edits(snapshot)


LETTERED = (
    "    alpha: int\n"
    "    beta: str\n\n"
    "    def gamma(self) -> str:\n"
    "        return f'{self.beta}{self.alpha}'\n\n"
)
LAYERED = {
    "pkg/__init__.py": "",
    "pkg/base.py": "UNIT = 1\n",
    "pkg/low.py": (
        "from dataclasses import dataclass\n\nfrom pkg.base import UNIT\n\n\n"
        "@dataclass(frozen=True)\nclass First:\n" + LETTERED + "    own_first: int = 0\n\n\n"
        "def unit():\n    return UNIT\n"
    ),
    "pkg/high.py": (
        "from dataclasses import dataclass\n\nfrom pkg.base import UNIT\nfrom pkg.low import First\n\n\n"
        "@dataclass(frozen=True)\nclass Second:\n" + LETTERED + "    own_second: int = 0\n\n\n"
        "def pair():\n    return First(alpha=1, beta='x'), Second(alpha=2, beta='y')\n"
    ),
}
LAYERED_PROBE = (
    "from pkg.high import pair\n"
    "print([(value.gamma(), value) for value in pair()])\n"
)


def _mixin_finding(root: Path):
    return next(
        finding
        for finding in analyze_modules(
            parse_python_modules(root, use_parse_cache=False), DetectorConfig()
        )
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ExtractMixin
    )


def test_a_mixin_whose_anchor_module_would_close_a_cycle_lands_in_a_module_both_import(
    tmp_path: Path,
) -> None:
    """`pkg.high` imports `pkg.low`, so a mixin declared beside `Second` would close
    a cycle when `First` composes it. Copied from `Second`, it is declared in
    `pkg.base` instead, which both import and which imports neither; both compose
    it there and the program runs as before."""

    _write(tmp_path, LAYERED)
    before = _run(tmp_path, LAYERED_PROBE)
    finding = _mixin_finding(tmp_path)
    assert "new_mixin:pkg.base" in {
        owner.candidate_id for owner in finding.metrics.owner_candidates
    }
    modules = parse_python_modules(tmp_path, use_parse_cache=False)

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(
            modules=modules, findings=analyze_modules(modules, DetectorConfig())
        ),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=finding.stable_id,
                    owner="new_mixin:pkg.base",
                    because="placed_below",
                    name="Lettered",
                    judgement="a lettered record",
                ),
            ),
        ),
    ).run()

    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    base = (tmp_path / "pkg/base.py").read_text()
    assert "class Lettered:" in base and "import pkg" not in base and "from pkg" not in base
    assert "class First(Lettered):" in (tmp_path / "pkg/low.py").read_text()
    assert "class Second(Lettered):" in (tmp_path / "pkg/high.py").read_text()
    assert _run(tmp_path, LAYERED_PROBE) == before
    assert _run(
        tmp_path,
        "import pkg.low, pkg.high\n"
        "print(pkg.low.First.__mro__[1] is pkg.high.Second.__mro__[1], "
        "pkg.low.First.__mro__[1].__module__)\n",
    ) == "True pkg.base\n"


def _owner_files(package: str) -> dict[str, str]:
    prefix, dotted = (f"{package}/", f"{package}.") if package else ("", "")
    body = (
        "    name: str\n    reason: str\n\n"
        "    def value_or(self, default):\n        return default\n\n"
        "    def label(self):\n        return f'{self.name}: {self.reason}'\n"
    )
    files = {
        f"{prefix}base.py": "UNIT = 1\n",
        f"{prefix}wire.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}base import UNIT\n\n\n"
            "@dataclass(frozen=True)\nclass Omitted:\n" + body + "\n"
            "    def wire(self):\n        return {'omitted': self.name}\n\n\n"
            "def omitted(name):\n    return Omitted(name=name, reason='unit')\n"
        ),
        f"{prefix}tools.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}base import UNIT\n"
            f"from {dotted}wire import omitted\n\n\n"
            "@dataclass(frozen=True)\nclass Unsupplied:\n" + body + "\n\n"
            "def unsupplied(name):\n    return Unsupplied(name=name, reason='missing'), omitted(name)\n"
        ),
    }
    if package:
        files[f"{package}/__init__.py"] = ""
    return files


def _composition(root: Path):
    from nominal_refactor_advisor.refactors.class_overlap import (
        ClassOverlapFindingRecipeSynthesizer,
    )

    (finding,) = (
        finding
        for finding in analyze_modules(
            parse_python_modules(root, use_parse_cache=False), DetectorConfig()
        )
        if isinstance(finding.metrics, ClassOverlapMetrics)
        and finding.metrics.move is ComposeExisting
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
    return ClassOverlapFindingRecipeSynthesizer().evaluate_recipe_for_finding(finding, snapshot), snapshot


def test_an_owner_composing_would_close_a_cycle_moves_down_first(tmp_path: Path) -> None:
    """`Omitted` restates `Unsupplied`, but `pkg.tools`, which declares the owner,
    imports `pkg.wire`: composing it there closes a cycle. `pkg.base`, which both
    import, is merely low enough: the owner reads nothing from it. It moves to
    `pkg.wire`, beside `Omitted`, which composes it there."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    _write(tmp_path, _owner_files("pkg"))
    probe = (
        "from pkg.tools import unsupplied\n"
        "print([(value.label(), value.value_or(3), value) for value in unsupplied('n')])\n"
    )
    before = _run(tmp_path, probe)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    assert (tmp_path / "pkg/base.py").read_text() == "UNIT = 1\n"
    wire = (tmp_path / "pkg/wire.py").read_text()
    assert "class Unsupplied:" in wire and "class Omitted(Unsupplied):" in wire
    assert "class Unsupplied" not in (tmp_path / "pkg/tools.py").read_text()
    assert _run(tmp_path, probe) == before


def test_an_owner_no_module_can_hold_without_a_cycle_refuses_saying_why(tmp_path: Path) -> None:
    """Here `wire` does not import `base`, so no module both import avoids the
    cycle, and top-level modules share no package for a new one: the move
    refuses and says so."""

    files = _owner_files("")
    files["wire.py"] = files["wire.py"].replace("from base import UNIT\n", "")
    _write(tmp_path, files)

    evaluation, _ = _composition(tmp_path)

    assert evaluation.rejection_reason == (
        "`tools` already imports, directly or not, `wire`, so `Unsupplied` moves "
        "first; but no module that `tools`, `wire` all import avoids the cycle, and "
        "they share no package to hold a new one"
    )


def _delivery_files(package: str) -> dict[str, str]:
    """`After` restates `Delivered`, whose module imports `After`'s: composing it
    there closes a cycle. `Delivered` reads `Reference` from `After`'s module and
    its base `Delivery` from its own, so no module below both can hold it."""

    prefix, dotted = (f"{package}/", f"{package}.") if package else ("", "")
    files = {
        f"{prefix}low.py": "UNIT = 1\n",
        f"{prefix}ref.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}low import UNIT\n\n\n"
            "@dataclass(frozen=True)\nclass Reference:\n    seq: int\n\n\n"
            "@dataclass(frozen=True)\nclass After:\n    reference: Reference\n\n"
            "    def refs(self):\n        return (self.reference,)\n\n"
            "    def cursor(self):\n        return self.reference.seq + UNIT\n"
        ),
        f"{prefix}origin.py": (
            f"from dataclasses import dataclass\n\nfrom {dotted}low import UNIT\n"
            f"from {dotted}ref import Reference\n\n\n"
            "class Delivery:\n    def refs(self):\n        raise NotImplementedError\n\n\n"
            "@dataclass(frozen=True)\nclass Delivered(Delivery):\n    reference: Reference\n\n"
            "    def refs(self):\n        return (self.reference,)\n\n\n"
            "class Undelivered(Delivery):\n    def refs(self):\n        return ()\n\n\n"
            "def both():\n    return Delivered(reference=Reference(seq=3)).refs(), Undelivered().refs(), UNIT\n"
        ),
    }
    if package:
        files[f"{package}/__init__.py"] = ""
    return files


def test_a_moved_owner_takes_what_it_reads_to_a_home_that_can_import_the_rest(tmp_path: Path) -> None:
    """`pkg.low` is imported by both and imports neither, but `Delivered` reads
    `Reference` from `pkg.ref`, which `pkg.low` cannot import without a cycle.
    So `Delivered` moves into `pkg.ref` itself, taking its base `Delivery` from
    its own module with it, and `After` composes it there."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    _write(tmp_path, _delivery_files("pkg"))
    probe = (
        "from pkg.origin import both\nfrom pkg.ref import After, Reference\n"
        "after = After(reference=Reference(seq=1))\nprint(both(), after.refs(), after.cursor(), after)\n"
    )
    before = _run(tmp_path, probe)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    ref = (tmp_path / "pkg/ref.py").read_text()
    origin = (tmp_path / "pkg/origin.py").read_text()
    assert "class Delivery:" in ref and "class Delivered(Delivery):" in ref, ref
    assert "class After(Delivered):" in ref, ref
    assert "class Delivered" not in origin and "class Delivery:" not in origin, origin
    assert "class Undelivered(Delivery):" in origin
    assert "from pkg.low" not in ref.replace("from pkg.low import UNIT", "")
    assert _run(tmp_path, probe) == before


def test_a_moved_owner_whose_reads_pin_it_refuses_naming_them(tmp_path: Path) -> None:
    """Top-level modules share no package, so only a module both import could
    hold `Delivered`, and it reads `Reference` from `ref`, one of them."""

    _write(tmp_path, _delivery_files(""))

    evaluation, _ = _composition(tmp_path)

    assert evaluation.rejection_reason == (
        "`origin` already imports, directly or not, `ref`, so `Delivered` moves first; "
        "but `Delivery` and `Delivered` read names from `ref`, which no module below "
        "`origin`, `ref` can import without a cycle"
    )


def test_a_moved_owner_lands_in_no_module_merely_low_enough(
    tmp_path: Path,
) -> None:
    """`pkg.annotated` and `pkg.base` are both low enough, and `pkg.annotated`
    stringizes annotations. The owner reads neither, so it goes beside its
    composer in `pkg.wire`, whose annotations evaluate as `pkg.tools`'s did."""

    from nominal_refactor_advisor.codemod import CodemodPlanSequence

    files = _owner_files("pkg")
    files["pkg/annotated.py"] = "from __future__ import annotations\n\nTWO = 2\n"
    for name in ("pkg/wire.py", "pkg/tools.py"):
        files[name] = files[name].replace(
            "from pkg.base import UNIT\n", "from pkg.base import UNIT\nfrom pkg.annotated import TWO\n"
        )
    _write(tmp_path, files)

    evaluation, snapshot = _composition(tmp_path)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(
        evaluation.required_recipe.operations
    ).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()

    home = (tmp_path / "pkg/wire.py").read_text()
    assert "class Unsupplied:" in home and "from __future__" not in home
    assert (tmp_path / "pkg/base.py").read_text() == "UNIT = 1\n"
    assert "class Unsupplied" not in (tmp_path / "pkg/annotated.py").read_text()


ABSTRACT_READERS = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class Base:\n"
    "    def base_only(self) -> int:\n"
    "        return 1\n\n\n"
    "class Alone(ABC):\n"
    "    @abstractmethod\n"
    "    def gamma(self) -> str: ...\n\n"
    "    @abstractmethod\n"
    "    def alone_only(self) -> int: ...\n\n\n"
    "class Leading(ABC, Base):\n"
    "    @abstractmethod\n"
    "    def gamma(self) -> str: ...\n\n\n"
    "class Trailing(Base, ABC):\n"
    "    @abstractmethod\n"
    "    def gamma(self) -> str: ...\n"
)
ABSTRACT_READERS_PROBE = (
    "from pkg.readers import Alone, Leading, Trailing\n"
    "for c in (Alone, Leading, Trailing):\n"
    "    print(c.__name__, type(c).__name__, sorted(c.__abstractmethods__),\n"
    "          [k.__name__ for k in c.__mro__ if k.__name__ != 'Gammaed'])\n"
)


def test_a_composer_drops_a_direct_base_the_new_mixin_already_supplies(tmp_path: Path) -> None:
    """`Gammaed(ABC)` supplies `ABC`. `Alone` and `Leading` list it where the mixin's
    own order already places it, so it goes from their headers and their C3 order is
    the same; `Trailing` lists it after `Base`, where dropping it would put `ABC`
    before `Base`, so it stays. `Base` is not the mixin's, so it stays everywhere."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/readers.py": ABSTRACT_READERS})
    before = _run(tmp_path, ABSTRACT_READERS_PROBE)

    _apply(
        tmp_path,
        CreateMixinOperation(
            target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/readers.py"), qualname="Alone"),
            member_names=("gamma",),
            mixin_name="Gammaed",
            docstring="Reads gamma.",
            composers=tuple(
                SourceRewriteTarget(file_path=str(tmp_path / "pkg/readers.py"), qualname=name)
                for name in ("Alone", "Leading", "Trailing")
            ),
        ),
    )

    source = (tmp_path / "pkg/readers.py").read_text()
    assert "class Gammaed(ABC):" in source
    assert "class Alone(Gammaed):" in source
    assert "class Leading(Gammaed, Base):" in source
    assert "class Trailing(Gammaed, Base, ABC):" in source
    assert _run(tmp_path, ABSTRACT_READERS_PROBE) == before


GENERIC_READERS = ABSTRACT_READERS.replace("class Leading(ABC, Base):", "class Holder[T]:\n    pass\n\n\nclass Leading[T](Holder[T], ABC):")
LONE_READER = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class Lone(ABC):\n"
    "    @abstractmethod\n"
    "    def gamma(self) -> str: ...\n\n"
    "    @abstractmethod\n"
    "    def lone_only(self) -> int: ...\n"
)


def test_a_generic_composer_keeps_abc_and_a_dropped_base_drops_its_import(tmp_path: Path) -> None:
    """`Leading[T](Holder[T], ABC)` gets `Generic` last from creation; dropping `ABC`
    would put it before `Holder`, so it stays. `Lone` lists only `ABC`, which goes,
    and with it the module's only read of `ABC`."""
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/readers.py": GENERIC_READERS, "pkg/lone.py": LONE_READER},
    )
    probe = ABSTRACT_READERS_PROBE.replace(
        "from pkg.readers import Alone, Leading, Trailing\n",
        "from pkg.readers import Alone, Leading, Trailing\nfrom pkg.lone import Lone\n",
    ).replace("(Alone, Leading, Trailing)", "(Alone, Leading, Trailing, Lone)")
    before = _run(tmp_path, probe)

    _apply(
        tmp_path,
        CreateMixinOperation(
            target=SourceRewriteTarget(file_path=str(tmp_path / "pkg/readers.py"), qualname="Alone"),
            member_names=("gamma",),
            mixin_name="Gammaed",
            docstring="Reads gamma.",
            composers=(
                *(
                    SourceRewriteTarget(file_path=str(tmp_path / "pkg/readers.py"), qualname=name)
                    for name in ("Alone", "Leading")
                ),
                SourceRewriteTarget(file_path=str(tmp_path / "pkg/lone.py"), qualname="Lone"),
            ),
        ),
    )

    readers = (tmp_path / "pkg/readers.py").read_text()
    lone = (tmp_path / "pkg/lone.py").read_text()
    assert "class Leading[T](Gammaed, Holder[T], ABC):" in readers
    assert "class Lone(Gammaed):" in lone
    assert "from abc import abstractmethod\n" in lone
    assert _run(tmp_path, probe) == before


METHOD_BLOCK = (
    "    def otherwise(self, default: int) -> int:\n"
    "        return default\n\n"
    "    def shown(self) -> str:\n"
    "        return f'<{self.otherwise(0)}>'\n\n"
    "    def texts(self) -> tuple[str, ...]:\n"
    "        return ()\n"
)

FROZEN_AND_MUTABLE = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class First:\n"
    "    lead: int\n\n"
    f"{METHOD_BLOCK}\n\n"
    "@dataclass\n"
    "class Second:\n"
    "    count: int\n\n"
    f"{METHOD_BLOCK}"
)

CONSTRUCTION_PROBE = (
    "import dataclasses, inspect\n"
    "from pkg.records import First, Second\n"
    "for record in (First(1), Second(2)):\n"
    "    kind = type(record)\n"
    "    print(kind.__name__, kind.__dataclass_params__, inspect.signature(kind.__init__),\n"
    "          [field.name for field in dataclasses.fields(kind)], hasattr(record, '__dict__'),\n"
    "          record.otherwise(3), record.shown(), record.texts(), record == kind(*dataclasses.astuple(record)), repr(record))\n"
    "    try:\n"
    "        record.__setattr__(dataclasses.fields(kind)[0].name, 9)\n"
    "        print('assigned', record)\n"
    "    except dataclasses.FrozenInstanceError:\n"
    "        print('frozen')\n"
)


def test_a_method_block_a_frozen_and_a_mutable_dataclass_share_is_a_plain_mixin(
    tmp_path: Path,
) -> None:
    """A dataclass collects fields and generated methods only from dataclass bases,
    so a block of methods needs no dataclass: a plain mixin serves both frozennesses,
    where a dataclass mixin could serve neither pair."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": FROZEN_AND_MUTABLE})
    before = _run(tmp_path, CONSTRUCTION_PROBE)

    report = _goal_with_pick(tmp_path, "Shown")

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "\n\n\nclass Shown:\n" in source
    assert "@dataclass(frozen=True)\nclass First(Shown):\n    lead: int\n" in source
    assert "@dataclass\nclass Second(Shown):\n    count: int\n" in source
    assert source.count("def otherwise") == 1
    assert _run(tmp_path, CONSTRUCTION_PROBE) == before


SLOTTED_AND_PLAIN = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True, slots=True)\n"
    "class First:\n"
    "    lead: int\n\n"
    f"{METHOD_BLOCK}\n\n"
    "class Second:\n"
    "    __slots__ = ()\n\n"
    f"{METHOD_BLOCK}"
)


def test_a_slotted_dataclass_composes_a_plain_class_whose_methods_it_restates(
    tmp_path: Path,
) -> None:
    """The plain owner brings no field, no generated method and no `__dict__`, so the
    frozen slotted dataclass keeps how it is built once it composes it."""
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": SLOTTED_AND_PLAIN})
    probe = CONSTRUCTION_PROBE.replace("(First(1), Second(2))", "(First(1),)") + (
        "print(Second().otherwise(4), Second().shown(), Second().texts())\n"
    )
    before = _run(tmp_path, probe)
    modules = parse_python_modules(tmp_path, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
    ).run()

    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "@dataclass(frozen=True, slots=True)\nclass First(Second):\n    lead: int\n" in source
    assert source.count("def otherwise") == 1
    assert _run(tmp_path, probe) == before


def test_a_block_with_a_field_still_needs_one_frozenness(tmp_path: Path) -> None:
    """A field in the block makes the mixin a dataclass, which Python lets only one
    frozenness compose: the finding is the block without the field, whose mixin
    stays plain, and each class keeps its own `note`."""
    records = FROZEN_AND_MUTABLE.replace("    lead: int\n", "    lead: int\n    note: str = ''\n").replace(
        "    count: int\n", "    count: int\n    note: str = ''\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": records})
    modules = parse_python_modules(tmp_path, use_parse_cache=False)

    (finding,) = (
        finding
        for finding in analyze_modules(modules, DetectorConfig())
        if isinstance(finding.metrics, ClassOverlapMetrics)
    )

    (mixin,) = (
        candidate for candidate in finding.metrics.owner_candidates
        if candidate.candidate_id == "new_mixin:pkg.records"
    )
    assert mixin.permitted
    assert finding.metrics.shared_declaration_names == ("otherwise", "shown", "texts")
    assert any(
        "one mixin cannot serve `First` (frozen), `Second` (mutable)" in receipt.detail
        and receipt.detail.endswith("and each class keeps `note`")
        for receipt in finding.metrics.receipts
    )

    before = _run(tmp_path, CONSTRUCTION_PROBE)
    report = _goal_with_pick(tmp_path, "Shown")
    source = (tmp_path / "pkg/records.py").read_text()
    assert report.stage_count == 1
    assert "\n\n\nclass Shown:\n" in source
    assert "@dataclass(frozen=True)\nclass First(Shown):\n    lead: int\n    note: str = ''\n" in source
    assert "@dataclass\nclass Second(Shown):\n    count: int\n    note: str = ''\n" in source
    assert source.count("def otherwise") == 1
    assert _run(tmp_path, CONSTRUCTION_PROBE) == before


SHARED_MARKER = (
    "class State:\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return False\n\n\n"
    "class Open:\n    __slots__ = ()\n\n\n"
    "class Ready(Open, State):\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return True\n\n\n"
    "class Blocked(Open, State):\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return True\n\n\n"
    "def blocked(state: State) -> bool:\n    return type(state) is Blocked\n"
)
SHARED_MARKER_PROBE = (
    "from pkg.todos import Blocked, Ready, State, blocked\n"
    "print(Ready.is_open(), Blocked.is_open(), State.is_open(), blocked(Blocked()), blocked(Ready()), Ready() == Ready(), "
    "[k.__name__ for k in Blocked.__mro__])\n"
)


def _fold_picked_marker(root: Path, source: str, probe: str) -> str:
    """Pick the marker `pkg.todos.Open` as the owner the refused merge's
    extraction offers, run the goal, and return the rewritten source after
    checking one stage ran and the probe prints as before."""

    _write(root, {"pkg/__init__.py": "", "pkg/todos.py": source})
    before = _run(root, probe)
    modules = parse_python_modules(root, use_parse_cache=False)
    findings = analyze_modules(modules, DetectorConfig())
    (offered,) = (
        finding
        for finding in findings
        if isinstance(finding.metrics, ClassOverlapMetrics) and finding.metrics.move is ExtractMixin
    )
    report = CodemodRefactorGoalRunner(
        roots=(root,),
        config=DetectorConfig(),
        dry_run=False,
        guard_suite=ArchitectureGuardSuite(),
        initial_scan=CodemodWorkflowScan(modules=modules, findings=findings),
        migration_type=ClassBodyCompositionConcept,
        owner_picks=OwnerPicks(
            "picks.json",
            "0",
            (
                OwnerPick(
                    finding=offered.stable_id,
                    owner="shared_base:pkg.todos.Open",
                    because="ancestor_of_both",
                    judgement="an open todo is open",
                ),
            ),
        ),
    ).run()
    assert report.stage_count == 1, report.stop_causes
    assert _run(root, probe) == before
    return (root / "pkg/todos.py").read_text()


def test_a_refused_merge_folds_its_block_into_the_base_the_two_alone_share(tmp_path: Path) -> None:
    """The merge of `Ready` and `Blocked` is refused (a type test tells them
    apart), so the verdict offers the extraction; its owner is `Open`, the empty
    base both inherit first. Picking it folds `is_open` there: the fold step
    judges the pair as the fold it makes, not as the refused merge."""

    source = _fold_picked_marker(tmp_path, SHARED_MARKER, SHARED_MARKER_PROBE)
    assert "class Open:\n    __slots__ = ()\n\n    @classmethod\n    def is_open(cls) -> bool:\n        return True\n" in source
    assert source.count("return True") == 1


FAMILY_MARKER = (
    "from abc import ABCMeta\n\n\n"
    "class FamilyMeta(ABCMeta):\n"
    "    def __new__(mcs, name, bases, namespace, *, affix=None, **options):\n"
    "        namespace['_affix'] = affix or ''\n"
    "        cls = super().__new__(mcs, name, bases, namespace, **options)\n"
    "        cls.family_root = cls\n"
    "        return cls\n\n\n"
    "class Family(metaclass=FamilyMeta):\n"
    "    def __init_subclass__(cls, *, affix=None, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return False\n\n\n"
    "class State(Family, affix='State'):\n    pass\n\n\n"
    "class Open:\n    __slots__ = ()\n\n\n"
    "class Ready(Open, State):\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return True\n\n\n"
    "class Blocked(Open, State):\n"
    "    @classmethod\n    def is_open(cls) -> bool:\n        return True\n\n\n"
    "def blocked(state: State) -> bool:\n    return type(state) is Blocked\n"
)
FAMILY_MARKER_PROBE = SHARED_MARKER_PROBE.replace("State, blocked", "State, blocked, Family")


def _family_order(root: Path, source: str, symbol: str) -> tuple[str, ...] | None:
    _write(root, {"pkg/__init__.py": "", "pkg/todos.py": source})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
    mro_type = snapshot.class_overlap_verdicts.class_index.mro_authority.resolve(symbol).mro_type
    return None if mro_type is None else tuple(item.symbol for item in mro_type.declarations)


def test_a_metaclass_that_hands_its_bases_on_keeps_the_declared_order(tmp_path: Path) -> None:
    """`FamilyMeta.__new__` writes a namespace entry, sets an attribute on the
    class it built and takes an `affix=` keyword, but hands its own `bases` to
    `super().__new__` and returns what it built: Python's order is C3's over the
    declared bases, so the order is closed. The strict runtime policy still
    leaves it open; only a proof that reads the creation code closes it."""

    order = _family_order(tmp_path, FAMILY_MARKER, "pkg.todos.Blocked")
    assert order == ("pkg.todos.Blocked", "pkg.todos.Open", "pkg.todos.State", "pkg.todos.Family")
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path, use_parse_cache=False))
    strict = ClassMroAuthority(snapshot.class_overlap_verdicts.class_index.classes_by_symbol)
    assert strict.resolve("pkg.todos.Blocked").mro_type is None


@pytest.mark.parametrize(
    "source",
    (
        FAMILY_MARKER.replace("        namespace['_affix'] = affix or ''\n", "        bases = (*bases, object)\n"),
        FAMILY_MARKER.replace(
            "class FamilyMeta(ABCMeta):\n",
            "class FamilyMeta(ABCMeta):\n    def mro(cls):\n        return super().mro()\n\n",
        ),
        FAMILY_MARKER + "\n\nBlocked.__bases__ = (Open, State)\n",
    ),
    ids=("rebinds-bases", "defines-mro", "stores-bases"),
)
def test_creation_that_may_change_the_order_leaves_it_open(tmp_path: Path, source: str) -> None:
    assert _family_order(tmp_path, source, "pkg.todos.Blocked") is None


def test_a_refused_merge_folds_into_the_marker_under_a_family_metaclass(tmp_path: Path) -> None:
    """The same fold as above with the states in a `DeclaredFamily`-shaped family:
    creation runs code, which a proof reads, so the fold goes ahead."""

    source = _fold_picked_marker(tmp_path, FAMILY_MARKER, FAMILY_MARKER_PROBE)
    assert "class Open:\n    __slots__ = ()\n\n    @classmethod\n    def is_open(cls) -> bool:\n        return True\n" in source
    assert source.count("return True") == 1


BRANCH_PROBE = (
    "import dataclasses\n"
    "from pkg.records import Difference, Growth\n"
    "grown = Growth(problem='p', repository='r', branch='b', growth=1)\n"
    "cut = Difference(repository='r', branch='b', head='h', growth=2)\n"
    "for record in (grown, cut):\n"
    "    print(sorted(dataclasses.asdict(record).items()))\n"
    "print(grown.line(), grown == dataclasses.replace(grown, growth=5), cut == dataclasses.replace(cut, growth=5))\n"
)


def test_a_part_one_class_inherits_is_shared_by_the_ancestor_that_declares_it(
    tmp_path: Path,
) -> None:
    """`growth` differs, so the finding is the agreeing part, `repository` and
    `branch`, which `Growth` inherits from `Rule`: the new mixin is composed by
    `Rule` and `Difference`, `Growth` inherits it, and each keeps its `growth`.
    The fields are keyword-only and nothing reads their order, so the reorder
    the mixin makes changes no value or comparison."""
    from tests.test_class_overlap import BRANCHES

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/records.py": BRANCHES})
    before = _run(tmp_path, BRANCH_PROBE)

    report = _goal_with_pick(tmp_path, "Named")

    source = (tmp_path / "pkg/records.py").read_text()
    assert [outcome.outcome for outcome in report.owner_picks.outcomes] == ["picked"]
    assert '\nclass Named:\n    """A lettered record."""\n\n    repository: str\n    branch: str\n' in source
    assert "class Rule(Named):\n    problem: str\n\n\n" in source
    assert "class Growth(Rule):\n    growth: int = field(compare=False)\n" in source
    assert "class Difference(Named):\n    head: str\n    growth: int\n" in source
    assert _run(tmp_path, BRANCH_PROBE) == before
