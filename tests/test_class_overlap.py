"""Duplicate classes and missing mixins: verdicts, receipts and the identity guard (#68)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.class_overlap import (
    ClassOverlapCandidate,
    ComposeExisting,
    DeleteAndRedirect,
    PromoteToSharedBase,
    ClassOverlapShape,
    ClassOverlapVerdicts,
    OverlapReceiptKind,
)


LABELS = Path(__file__).parent / "fixtures" / "class_overlap_labels.json"


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        is_package_init=False,
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
