"""The two identity-keeping class-overlap moves, applied and run (#68)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_overlap_codemod import (
    ComposeExistingClassOperation,
    FoldSharedDeclarationsIntoBaseOperation,
)
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

SURFACE = (
    "from abc import ABCMeta\n"
    "from dataclasses import dataclass, fields\n"
    "from typing import ClassVar\n\n\n"
    "class Zipped:\n"
    "    def __init__(self, *names):\n        self.names = names\n\n"
    "    def __get__(self, instance, owner=None):\n"
    "        if instance is None:\n            return self\n"
    "        return tuple(zip(*(getattr(instance, name) for name in self.names)))\n\n\n"
    "class Registered(ABCMeta):\n    pass\n\n\n"
    "@dataclass(frozen=True)\n"
    "class LineWitness(metaclass=Registered):\n"
    "    file_path: str\n    line: int\n\n\n"
    "@dataclass(frozen=True)\n"
    "class FunctionEvidenceLocationsCandidate(LineWitness):\n"
    "    function_names: tuple[str, ...]\n    line_numbers: tuple[int, ...]\n"
    "    evidence_locations: ClassVar[Zipped] = Zipped('line_numbers', 'function_names')\n\n\n"
    "def unrelated():\n    return 1\n\n\n"
    "@dataclass(frozen=True)\n"
    "class FunctionFamilyEvidenceSurface:\n"
    "    function_names: tuple[str, ...]\n    line_numbers: tuple[int, ...]\n"
    "    evidence_locations: ClassVar[Zipped] = Zipped('line_numbers', 'function_names')\n"
)

PROOFS = (
    "from abc import ABC, abstractmethod\n"
    "from dataclasses import dataclass\n\n\n"
    "class Proof(ABC):\n"
    "    @property\n    @abstractmethod\n"
    "    def is_proven(self) -> bool:\n        raise NotImplementedError\n"
)

EXPANSION = (
    "from dataclasses import dataclass\n\n"
    "from .base import Proof\n"
    "from .shapes import Edge\n\n\n"
    "@dataclass(frozen=True)\n"
    "class ExpansionProof(Proof):\n"
    "    carrier: str\n"
    "    edges: tuple[Edge, ...]\n    component: str\n    delta: int\n\n"
    "    @property\n    def is_proven(self) -> bool:\n        return not self.carrier\n\n"
    "    def reason(self) -> str:\n        return '; '.join(edge.name for edge in self.edges)\n\n\n"
    "def build(edges):\n"
    "    return ExpansionProof(carrier='', edges=edges, component='c', delta=1)\n"
)

CONVEYOR = (
    "from dataclasses import dataclass\n\n"
    "from .base import Proof\n"
    "from .shapes import Edge\n\n\n"
    "@dataclass(frozen=True)\n"
    "class ConveyorProof(Proof):\n"
    "    roots: tuple[str, ...]\n"
    "    edges: tuple[Edge, ...]\n    component: str\n    delta: int\n\n"
    "    @property\n    def is_proven(self) -> bool:\n        return not self.roots\n\n"
    "    def reason(self) -> str:\n        return '; '.join(edge.name for edge in self.edges)\n\n\n"
    "def build(edges):\n"
    "    return ConveyorProof(roots=(), edges=edges, component='c', delta=2)\n"
)

SHAPES = "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Edge:\n    name: str\n"


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _apply(root: Path, operation: RefactorRecipeOperation) -> dict[str, str]:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    result = RefactorRecipe(recipe_id="class-overlap").with_operation(replayed).simulate(snapshot)
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source, encoding="utf-8")
    return result.simulation.rewritten_sources


def _run(root: Path, program: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


SURFACE_PROBE = (
    "from dataclasses import fields\n"
    "from pkg.surface import FunctionEvidenceLocationsCandidate as C\n"
    "value = C('f.py', 3, ('a', 'b'), (4, 5))\n"
    "print([field.name for field in fields(C)])\n"
    "print(value.evidence_locations, value)\n"
    "print(type(C).__name__, [k.__name__ for k in C.__mro__ if k.__name__ != 'C'])\n"
)


def _compose(root: Path) -> ComposeExistingClassOperation:
    path = (root / "pkg/surface.py").as_posix()
    return ComposeExistingClassOperation(
        target=SourceRewriteTarget(
            file_path=path, qualname="FunctionEvidenceLocationsCandidate"
        ),
        owner=SourceRewriteTarget(file_path=path, qualname="FunctionFamilyEvidenceSurface"),
    )


def test_a_restating_class_composes_the_owner_and_runs_the_same(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/surface.py": SURFACE})
    before = _run(tmp_path, SURFACE_PROBE).splitlines()

    rewritten = _apply(tmp_path, _compose(tmp_path))[(tmp_path / "pkg/surface.py").as_posix()]
    after = _run(tmp_path, SURFACE_PROBE).splitlines()

    assert (
        "class FunctionEvidenceLocationsCandidate(FunctionFamilyEvidenceSurface, LineWitness):\n"
        "    pass\n"
    ) in rewritten
    assert rewritten.index("class FunctionFamilyEvidenceSurface") < rewritten.index(
        "class FunctionEvidenceLocationsCandidate"
    )
    assert rewritten.count("Zipped('line_numbers', 'function_names')") == 1
    assert before[:2] == after[:2]
    assert after[2].split()[1:] == [
        "['FunctionEvidenceLocationsCandidate',",
        "'FunctionFamilyEvidenceSurface',",
        "'LineWitness',",
        "'object']",
    ]


def test_composition_refuses_a_type_test_on_the_owner(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/surface.py": SURFACE
            + "\n\ndef is_family(value):\n"
            "    return isinstance(value, FunctionFamilyEvidenceSurface)\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="refuses the move"):
        _compose(tmp_path).source_edits(snapshot)


def test_composition_refuses_a_field_order_it_would_change(tmp_path: Path) -> None:
    reordered = SURFACE.replace(
        "class FunctionEvidenceLocationsCandidate(LineWitness):\n",
        "class FunctionEvidenceLocationsCandidate(LineWitness):\n    label: str\n",
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/surface.py": reordered})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="reorders the fields"):
        _compose(tmp_path).source_edits(snapshot)


def _proof_files(**overrides: str) -> dict[str, str]:
    return {
        "pkg/__init__.py": "",
        "pkg/base.py": PROOFS,
        "pkg/shapes.py": SHAPES,
        "pkg/expansion.py": EXPANSION,
        "pkg/conveyor.py": CONVEYOR,
        **overrides,
    }


def _fold(root: Path) -> FoldSharedDeclarationsIntoBaseOperation:
    return FoldSharedDeclarationsIntoBaseOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/base.py").as_posix(), qualname="Proof"),
        member_names=("edges", "component", "delta", "reason"),
    )


FOLD_PROBE = (
    "from dataclasses import fields, FrozenInstanceError\n"
    "from pkg.shapes import Edge\n"
    "from pkg import expansion, conveyor\n"
    "for module in (expansion, conveyor):\n"
    "    proof = module.build((Edge('a'), Edge('b')))\n"
    "    print(sorted(field.name for field in fields(proof)), proof.reason(), proof.is_proven)\n"
    "    try:\n        proof.delta = 0\n    except FrozenInstanceError:\n        print('frozen')\n"
)


def test_a_block_every_child_restates_folds_into_their_base(tmp_path: Path) -> None:
    _write(tmp_path, _proof_files())
    before = _run(tmp_path, FOLD_PROBE)

    rewritten = _apply(tmp_path, _fold(tmp_path))
    after = _run(tmp_path, FOLD_PROBE)

    base = rewritten[(tmp_path / "pkg/base.py").as_posix()]
    assert "@dataclass(frozen=True, kw_only=True)\nclass Proof(ABC):\n" in base
    assert "from .shapes import Edge" in base
    for child in ("expansion", "conveyor"):
        source = rewritten[(tmp_path / f"pkg/{child}.py").as_posix()]
        assert "def reason" not in source and "delta: int" not in source
        assert "from .shapes import Edge" not in source
    assert before == after


def test_the_fold_refuses_a_positional_construction_of_its_own_class(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pkg/conveyor.py": CONVEYOR.replace(
                    "    def reason(self) -> str:",
                    "    @classmethod\n    def empty(cls):\n"
                    "        return cls((), (), 'c', 0)\n\n"
                    "    def reason(self) -> str:",
                )
            }
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="positional"):
        _fold(tmp_path).source_edits(snapshot)


def test_the_fold_refuses_a_class_handed_on_as_a_value(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pkg/conveyor.py": CONVEYOR
                + "\n\nFACTORIES = (ConveyorProof,)\n"
            }
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="as a value"):
        _fold(tmp_path).source_edits(snapshot)
