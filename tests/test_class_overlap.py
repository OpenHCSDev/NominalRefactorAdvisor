"""Duplicate classes and missing mixins: verdicts, receipts and the identity guard (#68)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_references import ClassReferenceRole
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    build_compact_class_family_index,
    followed_reexports,
)
from tests.dataclass_field_wrappers import FIELD_WRAPPERS
from nominal_refactor_advisor.class_overlap import (
    ClassOverlapCandidate,
    ComposeExisting,
    DeleteAndRedirect,
    PromoteToSharedBase,
    ClassOverlapShape,
    ClassOverlapVerdicts,
    OverlapReceiptKind,
)
from nominal_refactor_advisor.owner_candidates import NewMixinOwner, OwnerReceiptKind


LABELS = Path(__file__).parent / "fixtures" / "class_overlap_labels.json"


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _verdicts(*modules: tuple[str, str]) -> tuple[ClassOverlapCandidate, ...]:
    projections = CompactModuleClassProjectionFamily.collect_modules(
        tuple(_module(name, source) for name, source in modules)
    )
    return ClassOverlapVerdicts(
        tuple(projections), build_compact_class_family_index(projections)
    ).candidates


def _kinds(candidate: ClassOverlapCandidate) -> set[OverlapReceiptKind]:
    return {receipt.kind for receipt in candidate.receipts}


def test_two_protocol_records_with_the_same_fields_are_a_coincidence():
    """agent-comms TextInput (opencode wire model) and TextContent (MCP dataclass).

    Both declare `type: Literal['text'] = 'text'` and `text: str`. As a concept
    the evidence is thin: two declarations only, no consumer uses both, no
    common ancestor, so the concept score stays at even odds. The move guard
    refuses outright: the classes are built differently, and each `type`
    field is a literal tag naming its own wire message.
    """

    (candidate,) = _verdicts(
        (
            "pkg.opencode",
            "from typing import Literal\n\n\n"
            "class WireModel:\n    pass\n\n\n"
            "class TextInput(WireModel):\n"
            "    type: Literal['text'] = 'text'\n    text: str\n\n\n"
            "def prompt(text):\n    return TextInput(text=text)\n",
        ),
        (
            "pkg.connector",
            "from dataclasses import dataclass\nfrom typing import Literal\n\n\n"
            "@dataclass(frozen=True)\n"
            "class TextContent:\n"
            "    text: str\n    type: Literal['text'] = 'text'\n\n\n"
            "def reply(text):\n    return TextContent(text)\n",
        ),
    )
    assert candidate.shape is ClassOverlapShape.DUPLICATE_CLASS
    assert {
        OverlapReceiptKind.FEW_DECLARATIONS,
        OverlapReceiptKind.NO_SHARED_CONSUMER,
        OverlapReceiptKind.SEPARATE_FAMILIES,
    } <= _kinds(candidate)
    assert candidate.confidence <= 0.5
    assert not candidate.permitted
    assert {receipt.kind for receipt in candidate.refusals} == {
        OverlapReceiptKind.DIFFERENT_CONSTRUCTION,
        OverlapReceiptKind.IDENTITY_DECLARED,
    }
    assert candidate.log_odds == sum(receipt.weight for receipt in candidate.receipts)


SURFACE = (
    "from dataclasses import dataclass\nfrom typing import ClassVar\n\n\n"
    "class Zipped:\n    def __init__(self, *names):\n        self.names = names\n\n\n"
    "@dataclass(frozen=True)\nclass LineWitness:\n    file_path: str\n    line: int\n\n\n"
    "@dataclass(frozen=True)\nclass FunctionEvidenceLocationsCandidate(LineWitness):\n"
    "    function_names: tuple[str, ...]\n    line_numbers: tuple[int, ...]\n"
    "    evidence_locations: ClassVar[Zipped] = Zipped('line_numbers', 'function_names')\n\n\n"
    "@dataclass(frozen=True)\nclass FunctionFamilyEvidenceSurface:\n"
    "    function_names: tuple[str, ...]\n    line_numbers: tuple[int, ...]\n"
    "    evidence_locations: ClassVar[Zipped] = Zipped('line_numbers', 'function_names')\n"
)


def test_a_class_restating_another_with_an_extra_base_composes_it():
    (candidate,) = _verdicts(("pkg.surface", SURFACE))
    assert candidate.shape is ClassOverlapShape.DUPLICATE_CLASS
    assert candidate.move is ComposeExisting
    assert candidate.owner.qualname == "FunctionFamilyEvidenceSurface"
    assert candidate.restating.qualname == "FunctionEvidenceLocationsCandidate"
    assert candidate.confidence > 0.5


def test_a_type_test_on_the_owner_refuses_the_move_and_says_where():
    (plain,) = _verdicts(("pkg.surface", SURFACE))
    (tested,) = _verdicts(
        (
            "pkg.surface",
            SURFACE
            + "\n\ndef is_family(value):\n"
            + "    return isinstance(value, FunctionFamilyEvidenceSurface)\n",
        )
    )
    assert plain.permitted
    (refusal,) = tested.refusals
    assert refusal.kind is OverlapReceiptKind.IDENTITY_USED
    assert refusal.evidence[0].symbol == "is_family"
    assert tested.confidence == plain.confidence


def test_a_type_test_on_the_composing_class_leaves_the_composition_permitted():
    (tested,) = _verdicts(
        (
            "pkg.surface",
            SURFACE
            + "\n\ndef is_candidate(value):\n"
            + "    return isinstance(value, FunctionEvidenceLocationsCandidate)\n",
        )
    )
    assert tested.move is ComposeExisting
    assert tested.permitted


@pytest.mark.parametrize(
    ("consumer", "permitted"),
    (
        # The class asked about does not descend from the restating class: its
        # answer is the same before and after the restating class inherits the owner.
        ("def kept():\n    return issubclass(LineWitness, FunctionFamilyEvidenceSurface)\n", True),
        # The restating class itself gains the owner: False becomes True.
        (
            "def flipped():\n"
            "    return issubclass(FunctionEvidenceLocationsCandidate, FunctionFamilyEvidenceSurface)\n",
            False,
        ),
        # A parameter shadows the class name, so the subject is any class at all.
        (
            "def shadowed(LineWitness):\n"
            "    return issubclass(LineWitness, FunctionFamilyEvidenceSurface)\n",
            False,
        ),
    ),
)
def test_a_subclass_test_on_the_owner_refuses_only_when_its_answer_changes(consumer, permitted):
    (candidate,) = _verdicts(("pkg.surface", SURFACE + "\n\n" + consumer))
    assert candidate.move is ComposeExisting
    assert candidate.permitted is permitted
    if not permitted:
        (refusal,) = candidate.refusals
        assert refusal.kind is OverlapReceiptKind.IDENTITY_USED


@pytest.mark.parametrize(
    ("consumer", "permitted"),
    (
        # A generic subscripted in an annotation names the class as a type
        # parameter, as a bare annotation does.
        (
            "class Box[T]:\n    pass\n\n\n"
            "def take(items: Box[FunctionFamilyEvidenceSurface]) -> Box[FunctionFamilyEvidenceSurface]:\n"
            "    held: Box[FunctionFamilyEvidenceSurface] = items\n    return held\n",
            True,
        ),
        # A table keyed by the class, read at run time, answers by its identity.
        (
            "HANDLERS = {}\n\n\ndef handler():\n    return HANDLERS[FunctionFamilyEvidenceSurface]\n",
            False,
        ),
    ),
)
def test_a_subscript_keys_by_the_class_only_outside_a_type(consumer, permitted):
    (candidate,) = _verdicts(("pkg.surface", SURFACE + "\n\n" + consumer))
    assert candidate.move is ComposeExisting
    assert candidate.permitted is permitted


def test_identical_bodies_bases_and_construction_delete_the_restating_class():
    (candidate,) = _verdicts(
        (
            "pkg.records",
            "class Task:\n    def label(self):\n        return self.name\n\n"
            "    def key(self):\n        return (self.name,)\n\n\n"
            "class Job:\n    def label(self):\n        return self.name\n\n"
            "    def key(self):\n        return (self.name,)\n\n\n"
            "def show(task: Task, job: Job):\n    return task.label(), job.label()\n",
        )
    )
    assert candidate.move is DeleteAndRedirect
    assert OverlapReceiptKind.SHARED_CONSUMER in _kinds(candidate)


def test_a_block_two_siblings_alone_share_moves_into_their_base():
    (candidate,) = _verdicts(
        (
            "pkg.proofs",
            "from dataclasses import dataclass\n\n\n"
            "class Proof:\n    pass\n\n\n"
            "@dataclass(frozen=True)\nclass ExpansionProof(Proof):\n"
            "    edges: tuple[str, ...]\n    component: str\n    delta: int\n"
            "    carrier: str\n\n"
            "    def reason(self):\n        return '; '.join(self.edges)\n\n\n"
            "@dataclass(frozen=True)\nclass ConveyorProof(Proof):\n"
            "    edges: tuple[str, ...]\n    component: str\n    delta: int\n"
            "    roots: tuple[str, ...]\n\n"
            "    def reason(self):\n        return '; '.join(self.edges)\n",
        )
    )
    assert candidate.shape is ClassOverlapShape.MISSING_MIXIN
    assert candidate.move is PromoteToSharedBase
    assert candidate.shared_names == ("edges", "component", "delta", "reason")
    assert OverlapReceiptKind.SAME_FAMILY in _kinds(candidate)


def test_enum_members_are_identities_and_never_move():
    candidates = _verdicts(
        (
            "pkg.violations",
            "from enum import StrEnum\n\n\n"
            "class ExpansionViolation(StrEnum):\n"
            "    MISSING = 'missing'\n    ESCAPING = 'escaping'\n    OPEN = 'open'\n\n"
            "    def explanation(self):\n        return self.value\n\n\n"
            "class ConveyorViolation(StrEnum):\n"
            "    MISSING = 'missing'\n    ESCAPING = 'escaping'\n    OPEN = 'open'\n\n"
            "    def explanation(self):\n        return self.value\n",
        )
    )
    assert candidates == ()


def test_one_shared_field_is_no_verdict():
    assert (
        _verdicts(
            (
                "pkg.values",
                "class Flag:\n    value: bool\n\n\nclass Ledger:\n    value: bool\n",
            )
        )
        == ()
    )


def _score(kinds: list[str]) -> float:
    return 1.0 / (1.0 + math.exp(-sum(OverlapReceiptKind(kind).weight for kind in kinds)))


def test_the_receipt_weights_keep_their_error_on_the_labeled_verdicts():
    """Hand-labeled verdicts (same concept or not, judged against the source).

    The confidence score is not fitted: with 50 positives and 8 negatives the
    labels cannot fit eleven weights. They pin the weights instead: changing
    one changes the Brier score and the expected calibration error recorded
    with the labels, and this test fails until the fixture is re-derived and
    the change is justified against it. Labeled pairs NRA no longer reports
    must all be negatives.
    """

    labels = json.loads(LABELS.read_text())
    reported = [row for row in labels["verdicts"] if row["verdict"]]
    assert all(not row["same_concept"] for row in labels["verdicts"] if not row["verdict"])
    brier = sum(
        (_score(row["receipts"]) - row["same_concept"]) ** 2 for row in reported
    ) / len(reported)
    bins: dict[int, list[tuple[float, bool]]] = {}
    for row in reported:
        score = _score(row["receipts"])
        bins.setdefault(min(int(score * 5), 4), []).append((score, row["same_concept"]))
    ece = (
        sum(
            abs(sum(s for s, _ in rows) - sum(y for _, y in rows))
            for rows in bins.values()
        )
        / len(reported)
    )
    assert round(brier, 4) == labels["brier"]
    assert round(ece, 4) == labels["ece"]


def test_a_block_an_existing_owner_declares_is_composed_from_that_owner():
    """Two records restating `file_path`/`line`/`symbol` restate the location
    class other records already inherit, not a new mixin."""

    candidates = _verdicts(
        (
            "pkg.sites",
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\nclass Location:\n"
            "    file_path: str\n    line: int\n    symbol: str\n\n\n"
            "@dataclass(frozen=True)\nclass Evidence(Location):\n    kind: str\n\n\n"
            "@dataclass(frozen=True)\nclass CallSite:\n"
            "    file_path: str\n    line: int\n    symbol: str\n    target: str\n\n\n"
            "@dataclass(frozen=True)\nclass ProofEdge:\n"
            "    file_path: str\n    line: int\n    symbol: str\n    authority: str\n",
        )
    )
    assert {
        (item.owner.qualname, item.restating.qualname, item.move) for item in candidates
        if item.shape is ClassOverlapShape.DUPLICATE_CLASS
    } == {("Location", "CallSite", ComposeExisting), ("Location", "ProofEdge", ComposeExisting)}
    (candidate,) = (
        item for item in candidates if item.shape is ClassOverlapShape.MISSING_MIXIN
    )
    assert candidate.move is ComposeExisting
    assert candidate.owner.qualname == "Location"
    assert {candidate.restating.qualname, candidate.co_restating.qualname} == {
        "CallSite",
        "ProofEdge",
    }
    assert OverlapReceiptKind.EXISTING_OWNER in _kinds(candidate)


SPREAD = (
    "pkg.summaries",
    "class Scope:\n    file_path: str\n    finding_ids: tuple[str, ...]\n\n\n"
    "class ModuleSummary(Scope):\n    module_name: str\n\n\n"
    "class EvidenceSummary:\n"
    "    file_path: str\n    line: int\n    finding_ids: tuple[str, ...]\n    message: str\n\n\n"
    "class FunctionSummary(Scope):\n    line: int\n    name: str\n",
)


def test_a_block_restated_through_an_inherited_mixin_is_still_a_verdict():
    """#96: FunctionSummary inherits file_path and finding_ids from Scope and
    declares line itself; EvidenceSummary declares all three. Own bodies share
    one declaration, so only the MRO shows the restatement. The verdict names
    the inherited part, which the mixin leaves to Scope (857586e)."""

    verdicts = {
        frozenset((candidate.owner.symbol, *candidate.restating_symbols)): candidate
        for candidate in _verdicts(SPREAD)
    }
    verdict = verdicts[
        frozenset(("pkg.summaries.EvidenceSummary", "pkg.summaries.FunctionSummary"))
    ]

    assert verdict.shape is ClassOverlapShape.MISSING_MIXIN
    assert set(verdict.shared_names) == {"file_path", "line", "finding_ids"}
    assert set(verdict.inherited_names) == {"file_path", "finding_ids"}
    assert verdict.permitted


def test_classes_inheriting_one_declaration_from_one_ancestor_restate_nothing():
    verdicts = _verdicts(SPREAD)

    assert not any(
        {candidate.owner.symbol, *candidate.restating_symbols}
        == {"pkg.summaries.ModuleSummary", "pkg.summaries.FunctionSummary"}
        for candidate in verdicts
    )


def test_a_star_settled_builtin_and_a_direct_builtin_are_one_declaration():
    """`file_path: str` under a star import the repository settles to
    `builtins.str` is the same declaration as `file_path: str` beside no star
    import: one resolved binding, two routes to it."""

    verdicts = _verdicts(
        ("pkg.helpers", "__all__ = ['helper']\n\n\ndef helper():\n    return 1\n"),
        (
            "pkg.starred",
            "from pkg.helpers import *\n\n\n"
            "class Starred:\n    file_path: str\n    line: int\n    symbol: str\n    kind: str\n",
        ),
        (
            "pkg.direct",
            "class Direct:\n    file_path: str\n    line: int\n    symbol: str\n    target: int\n",
        ),
    )

    (verdict,) = verdicts
    assert {verdict.owner.qualname, verdict.restating.qualname} == {"Starred", "Direct"}
    assert set(verdict.shared_names) == {"file_path", "line", "symbol"}


def test_a_shared_field_declared_differently_refuses_the_move():
    records = (
        "from dataclasses import dataclass\n"
        "from pkg.fields import PathCodec, codec_field\n\n\n"
        "@dataclass(frozen=True)\nclass Plain:\n"
        "    path: str\n    line: int\n    column: int\n\n\n"
        "@dataclass(frozen=True)\nclass Wrapped:\n"
        "    path: str = codec_field(PathCodec())\n    line: int\n    column: int\n"
    )
    (candidate,) = _verdicts(("pkg.fields", FIELD_WRAPPERS), ("pkg.records", records))
    assert {shared.name for shared in candidate.overlap.shared} == {"path", "line", "column"}
    (refusal,) = candidate.refusals
    assert refusal.kind is OverlapReceiptKind.DECLARATION_DIFFERS
    assert not candidate.permitted


def test_a_classmethod_building_its_own_class_by_position_refuses_the_reorder():
    """`cls(...)` in the class's own classmethod binds its fields by position
    exactly as an outside call does, so a mixin that moves the block ahead of
    the class's own fields would rebind it. The new mixin's guard names it, with
    keywording First as its cure (3bb47ae moved the guard off the verdict)."""

    (candidate,) = _verdicts(
        (
            "pkg.routes",
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\nclass First:\n"
            "    seq: int\n    target: str\n    sender: str\n    lookup: str\n    stamp: float\n\n"
            "    @classmethod\n    def of(cls, message):\n"
            "        return cls(message.seq, message.target, message.sender, message.lookup, 1.0)\n\n\n"
            "@dataclass(frozen=True)\nclass Second:\n"
            "    target: str\n    sender: str\n    lookup: str\n    first_seq: int\n\n"
            "    @classmethod\n    def of(cls, message):\n"
            "        return cls(target=message.target, sender=message.sender,\n"
            "                   lookup=message.lookup, first_seq=message.seq)\n",
        ),
    )

    (mixin,) = (
        owner for owner in candidate.owner_candidates if isinstance(owner, NewMixinOwner)
    )
    (positional,) = (
        receipt
        for receipt in mixin.refusals
        if receipt.kind is OwnerReceiptKind.POSITIONAL_CONSTRUCTION
    )
    assert positional.constructed == ("pkg.routes.First",)
    assert [(site.line, site.symbol) for site in positional.evidence] == [(14, "First.of")]


SITES = (
    "pkg.sites",
    "from dataclasses import dataclass\n\nfrom pkg.summaries import Scope as Area\n\n\n"
    "@dataclass(frozen=True)\nclass Location:\n"
    "    file_path: str\n    line: int\n    symbol: str\n\n\n"
    "@dataclass(frozen=True)\nclass Evidence(Location):\n    kind: str\n\n\n"
    "@dataclass(frozen=True)\nclass CallSite:\n"
    "    file_path: str\n    line: int\n    symbol: str\n    target: str\n\n\n"
    "@dataclass(frozen=True)\nclass ProofEdge:\n"
    "    file_path: str\n    line: int\n    symbol: str\n    authority: str\n\n\n"
    "def edge(site: CallSite) -> ProofEdge:\n"
    "    return ProofEdge(site.file_path, site.line, site.symbol, Area.__name__)\n",
)


def test_verdicts_touching_a_class_are_the_scan_verdicts_naming_it():
    """A move asks only about the classes it names, on every virtual source; the
    narrowed search must find exactly the verdicts and uses the whole scan does,
    inherited pairs and aliased imports included."""

    projections = tuple(
        CompactModuleClassProjectionFamily.collect_modules(
            tuple(_module(name, source) for name, source in (SPREAD, SITES))
        )
    )
    index = build_compact_class_family_index(projections)
    scan = ClassOverlapVerdicts(projections, index)

    def named(candidates, symbol):
        return [
            (candidate.owner.symbol, candidate.restating_symbols, candidate.move)
            for candidate in candidates
            if symbol in (candidate.owner.symbol, *candidate.restating_symbols)
        ]

    assert len(scan.candidates) >= 3
    for symbol in index.classes_by_symbol:
        asked = ClassOverlapVerdicts(projections, index)
        assert named(asked.candidates_touching(frozenset({symbol})), symbol) == named(
            scan.candidates, symbol
        )
    assert {use.consumer for use in scan.uses.uses_for("pkg.summaries.Scope")} >= {"edge"}


def test_a_use_through_a_renamed_re_export_names_the_class_it_binds() -> None:
    """`pkg` re-exports `X` as `Y`, so `Y()` read from `pkg`, by name or through
    `import pkg`, constructs `pkg.a.X`: Python resolves it there, and so must the
    use index, or a redirect or deletion of `X` misses these readers."""

    projections = tuple(
        CompactModuleClassProjectionFamily.collect_modules(
            tuple(
                _module(name, source)
                for name, source in (
                    ("pkg.a", "class X:\n    pass\n"),
                    ("pkg", "from pkg.a import X as Y\n"),
                    ("pkg.b", "from pkg import Y\n\n\ndef by_name():\n    return Y()\n"),
                    ("pkg.c", "import pkg\n\n\ndef through_module():\n    return pkg.Y()\n"),
                )
            )
        )
    )
    scan = ClassOverlapVerdicts(projections, build_compact_class_family_index(projections))

    assert {use.consumer for use in scan.uses.uses_for("pkg.a.X")} == {"by_name", "through_module"}



def test_a_construction_in_a_script_is_a_use_at_its_line_in_the_file() -> None:
    """A string holding Python a child interpreter runs resolves its own imports;
    its call of `X` is a positional construction on the line that holds it."""

    projections = tuple(
        CompactModuleClassProjectionFamily.collect_modules(
            tuple(
                _module(name, source)
                for name, source in (
                    ("pkg.a", "class X:\n    pass\n"),
                    (
                        "pkg.b",
                        "SCRIPT = (\n"
                        '    "from pkg.a import X\\n"\n'
                        '    "X(1, 2)\\n"\n'
                        ")\n",
                    ),
                )
            )
        )
    )
    scan = ClassOverlapVerdicts(projections, build_compact_class_family_index(projections))
    (use,) = scan.uses.uses_for("pkg.a.X")

    assert (use.role, use.reference.line, use.reference.positional_arguments) == (
        ClassReferenceRole.CONSTRUCTION, 3, 2
    )

def test_a_re_export_cycle_stops_at_the_name_it_comes_back_to() -> None:
    aliases = {"pkg": {"Y": "pkg.other.Z"}, "pkg.other": {"Z": "pkg.Y"}}
    assert followed_reexports("pkg.Y", aliases) == "pkg.other.Z"
    assert followed_reexports("pkg.unbound", aliases) == "pkg.unbound"
