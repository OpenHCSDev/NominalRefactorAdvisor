"""Duplicate classes and missing mixins: verdicts, receipts and the identity guard (#68)."""

from __future__ import annotations

import ast
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

    Both declare `type: Literal['text'] = 'text'` and `text: str`. The receipts
    show why that is not one concept: two declarations only, no consumer uses
    both, no common ancestor, different class construction, and each field
    `type` is a literal tag naming its own wire message.
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
        OverlapReceiptKind.DIFFERENT_CONSTRUCTION,
        OverlapReceiptKind.IDENTITY_DECLARED,
    } <= _kinds(candidate)
    assert candidate.probability < 0.05
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
    assert candidate.probability > 0.5


def test_a_type_test_on_either_class_lowers_the_probability_and_says_where():
    (plain,) = _verdicts(("pkg.surface", SURFACE))
    (tested,) = _verdicts(
        (
            "pkg.surface",
            SURFACE
            + "\n\ndef is_family(value):\n"
            + "    return isinstance(value, FunctionFamilyEvidenceSurface)\n",
        )
    )
    used = [
        receipt
        for receipt in tested.receipts
        if receipt.kind is OverlapReceiptKind.IDENTITY_USED
    ]
    assert used and used[0].evidence[0].symbol == "is_family"
    assert tested.probability < plain.probability


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
