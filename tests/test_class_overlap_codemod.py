"""The two identity-keeping class-overlap moves, applied and run (#68)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import InstalledClassDeclarations
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.refactors.class_overlap import (
    ClassReferenceSites,
    ComposeExistingClassOperation,
    ComposeExistingClassesOperation,
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


SMALL_OWNER = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\nclass Located:\n    file_path: str\n    line: int\n\n\n"
    "@dataclass(frozen=True)\nclass Finding:\n    file_path: str\n    line: int\n    message: str\n"
)
SMALL_PROBE = (
    "from dataclasses import fields\n"
    "from pkg.located import Finding, Located\n"
    "value = Finding('f.py', 3, 'm')\n"
    "print([field.name for field in fields(Finding)], value, issubclass(Finding, Located))\n"
)


def test_a_requested_composition_is_judged_below_the_reporting_floor(tmp_path: Path) -> None:
    """Two shared annotations are too little for the scan to report, but a move
    that asks for the composition gets the pair's verdict and its guards."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": SMALL_OWNER})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert snapshot.class_overlap_verdicts.candidates == ()
    path = (tmp_path / "pkg/located.py").as_posix()
    _apply(
        tmp_path,
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="Finding"),
            owner=SourceRewriteTarget(file_path=path, qualname="Located"),
        ),
    )
    assert _run(tmp_path, SMALL_PROBE).split() == [
        "['file_path',", "'line',", "'message']",
        "Finding(file_path='f.py',", "line=3,", "message='m')", "True",
    ]


BASED_OWNER = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class Snapshot(ABC):\n"
    "    __slots__ = ()\n\n"
    "    @abstractmethod\n"
    "    def owner_identity(self, name: str) -> int: ...\n\n\n"
    "class Reads(ABC):\n"
    "    __slots__ = ()\n\n"
    "    @abstractmethod\n"
    "    def owner_identity(self, name: str) -> int: ...\n\n"
    "    @abstractmethod\n"
    "    def require(self, name: str) -> str: ...\n\n\n"
    "class Live(Reads):\n"
    "    __slots__ = ('names',)\n\n"
    "    def __init__(self) -> None:\n"
    "        self.names = {'a': 1}\n\n"
    "    def owner_identity(self, name: str) -> int:\n"
    "        return self.names[name]\n\n"
    "    def require(self, name: str) -> str:\n"
    "        return name\n"
)
BASED_PROBE = (
    "from pkg.reads import Live, Reads\n"
    "live = Live()\n"
    "print(live.owner_identity('a'), live.require('b'), hasattr(live, '__dict__'),\n"
    "      sorted(Reads.__abstractmethods__))\n"
)


def test_an_owner_whose_base_the_class_already_has_is_composed(tmp_path: Path) -> None:
    """`Snapshot` has a base, but `Reads` already descends from it: C3 puts the
    owner first and keeps every other ancestor in order, each name resolves as
    before or to the owner's equal declaration, and `__slots__` stays, since
    each class's own body declares it."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/reads.py": BASED_OWNER})
    before = _run(tmp_path, BASED_PROBE)
    path = (tmp_path / "pkg/reads.py").as_posix()
    _apply(
        tmp_path,
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="Reads"),
            owner=SourceRewriteTarget(file_path=path, qualname="Snapshot"),
        ),
    )
    source = (tmp_path / "pkg/reads.py").read_text()
    assert "class Reads(Snapshot, ABC):\n    __slots__ = ()\n\n    @abstractmethod\n    def require" in source
    assert source.count("def owner_identity") == 2
    assert _run(tmp_path, BASED_PROBE) == before


def test_a_composition_that_changes_a_name_s_winner_is_refused(tmp_path: Path) -> None:
    """Composing `Labelled` brings its base `Base` ahead of `Mixin` in `Plain`'s
    C3 order, so `Plain.label` would stop being `Mixin`'s."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/labels.py": (
                "class Base:\n    def label(self) -> str:\n        return 'base'\n\n\n"
                "class Mixin:\n    def label(self) -> str:\n        return 'mixin'\n\n\n"
                "class Labelled(Base):\n    def kind(self) -> str:\n        return 'k'\n\n\n"
                "class Plain(Mixin):\n    def kind(self) -> str:\n        return 'k'\n"
            ),
        },
    )
    path = (tmp_path / "pkg/labels.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Plain"),
        owner=SourceRewriteTarget(file_path=path, qualname="Labelled"),
    )
    with pytest.raises(
        CodemodOperationPreflightError,
        match=r"makes `Plain.label` resolve to `pkg.labels.Base` instead of `pkg.labels.Mixin`",
    ):
        operation.source_edits(snapshot)


def test_the_composition_reads_a_subclass_test_by_the_class_it_asks_about(tmp_path: Path) -> None:
    """Finding gaining Located as a base changes `issubclass(Finding, Located)`,
    never `issubclass(Plain, Located)`: only the first stops the move."""

    kept = "from pkg.located import Located, Plain\n\n\ndef kept():\n    return issubclass(Plain, Located)\n"
    flipped = "from pkg.located import Finding, Located\n\n\ndef flipped():\n    return issubclass(Finding, Located)\n"
    owner = SMALL_OWNER + "\n\nclass Plain:\n    pass\n"
    path = (tmp_path / "pkg/located.py").as_posix()
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Finding"),
        owner=SourceRewriteTarget(file_path=path, qualname="Located"),
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": owner, "pkg/checks.py": kept})
    _apply(tmp_path, operation)
    assert _run(tmp_path, SMALL_PROBE + "from pkg.checks import kept\nprint(kept())\n").split()[-2:] == ["True", "False"]

    _write(tmp_path, {"pkg/located.py": owner, "pkg/checks.py": flipped})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"`Located` is told apart by type in 1 place\(s\) \(type_test\)"):
        operation.source_edits(snapshot)


INSTALLED_REPRESENTATION = (
    "from abc import ABC\n\n"
    "from nra_installed_base import Dispatch\n\n\n"
    "class Representation(Dispatch, ABC):\n"
    "    __slots__ = ()\n\n\n"
    "class Omitted(Representation):\n"
    "    def value_or(self, default):\n"
    "        return default\n\n\n"
    "class PiOmitted(Representation):\n"
    "    def value_or(self, default):\n"
    "        return default\n\n"
    "    def is_reported(self) -> bool:\n"
    "        return False\n"
)
INSTALLED_PROBE = (
    "from pkg.omitted import PiOmitted\n"
    "value = PiOmitted()\n"
    "print(value.value_or(3), value.is_reported(), value.handle(), [c.__name__ for c in PiOmitted.__mro__])\n"
)


def _installed_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A package outside the scanned root, importable, re-exporting its class."""

    site = tmp_path / "site"
    (site / "nra_installed_base").mkdir(parents=True)
    (site / "nra_installed_base/__init__.py").write_text("from .dispatch import Dispatch\n")
    (site / "nra_installed_base/dispatch.py").write_text(
        "class Dispatch:\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n\n"
        "    def handle(self) -> str:\n"
        "        return 'dispatch'\n"
    )
    monkeypatch.syspath_prepend(str(site))
    InstalledModule.located_source.cache_clear()
    InstalledModule.parsed.cache_clear()
    InstalledClassDeclarations.module.cache_clear()
    return site


def test_an_owner_whose_ancestry_reaches_an_installed_class_is_composed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Dispatch` lives outside the scan, in installed source NRA reads without
    running: the C3 orders before and after include it, and every name keeps its
    winner, so `PiOmitted` composes `Omitted`."""

    site = _installed_site(tmp_path, monkeypatch)
    project = tmp_path / "project"
    _write(project, {"pkg/__init__.py": "", "pkg/omitted.py": INSTALLED_REPRESENTATION})

    def ran() -> str:
        return subprocess.run(
            [sys.executable, "-c", INSTALLED_PROBE],
            cwd=project,
            env={**os.environ, "PYTHONPATH": str(site)},
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    before = ran()
    path = (project / "pkg/omitted.py").as_posix()
    _apply(
        project,
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="PiOmitted"),
            owner=SourceRewriteTarget(file_path=path, qualname="Omitted"),
        ),
    )
    source = (project / "pkg/omitted.py").read_text()
    assert "class PiOmitted(Omitted, Representation):\n    def is_reported" in source
    assert ran().split("[")[0] == before.split("[")[0]


def test_a_base_with_no_installed_python_source_is_named_in_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sqlite3.Cursor` is implemented in C: no `.py` source declares it, so its
    C3 order is unknown and composing an owner built on it refuses, naming it."""

    _installed_site(tmp_path, monkeypatch)
    _write(
        tmp_path / "project",
        {
            "pkg/__init__.py": "",
            "pkg/cursors.py": (
                "from sqlite3 import Cursor\n\n\n"
                "class Lent(Cursor):\n    def close(self) -> None:\n        pass\n\n\n"
                "class Plain:\n    def close(self) -> None:\n        pass\n"
            ),
        },
    )
    path = (tmp_path / "project/pkg/cursors.py").as_posix()
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path / "project"))
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Plain"),
        owner=SourceRewriteTarget(file_path=path, qualname="Lent"),
    )
    with pytest.raises(
        CodemodOperationPreflightError,
        match=r"`Lent` has base `sqlite3.Cursor`, which neither the scan nor an installed `.py` source declares",
    ):
        operation.source_edits(snapshot)


def _tagged_family(hook: str) -> str:
    return (
        "from dataclasses import dataclass\n\n\n"
        "class Tagged:\n"
        "    def __init_subclass__(cls, *, tag: str = '', **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        f"        {hook}\n\n\n"
        "@dataclass(frozen=True)\nclass Located:\n    file_path: str\n    line: int\n\n\n"
        "@dataclass(frozen=True)\nclass Finding(Tagged, tag='f'):\n"
        "    file_path: str\n    line: int\n    message: str\n"
    )


def test_a_class_created_with_keywords_composes_an_owner_its_hook_cannot_see(tmp_path: Path) -> None:
    """Class keywords leave C3 alone; the hook reads only its keyword and stores it."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _tagged_family("cls.tag = tag")})
    path = (tmp_path / "pkg/located.py").as_posix()
    _apply(
        tmp_path,
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="Finding"),
            owner=SourceRewriteTarget(file_path=path, qualname="Located"),
        ),
    )
    assert _run(tmp_path, SMALL_PROBE + "print(Finding.tag)\n").split()[-2:] == ["True", "f"]


def test_a_creation_hook_that_reads_the_bases_stops_the_composition_at_its_line(tmp_path: Path) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _tagged_family("cls.depth = len(cls.__mro__)")})
    path = (tmp_path / "pkg/located.py").as_posix()
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Finding"),
        owner=SourceRewriteTarget(file_path=path, qualname="Located"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"located.py:7 `cls.__mro__` introspects the new class") as raised:
        operation.source_edits(snapshot)
    assert raised.value.needs == (NeededInput.PROOF,)


def _writers_family(owner_extra: str, composer_extra: str) -> str:
    """A hook keeping the classes whose own body binds `writes`, as agent-comms
    `ProjectionOutputs.writer_declarations` does."""

    return (
        "from dataclasses import dataclass\n\n\n"
        "class Writer:\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        cls.writers = tuple(parent for parent in cls.__mro__ if 'writes' in vars(parent))\n\n\n"
        f"@dataclass(frozen=True)\nclass Located:\n    file_path: str\n    line: int\n{owner_extra}\n\n"
        "@dataclass(frozen=True)\nclass Finding(Writer):\n"
        f"    file_path: str\n    line: int\n{owner_extra}    message: str\n{composer_extra}"
    )


WRITERS_PROBE = (
    "from pkg.located import Finding\n"
    "print([writer.__name__ for writer in Finding.writers])\n"
)


def test_a_hook_selecting_the_classes_binding_a_name_composes_an_owner_that_binds_none(
    tmp_path: Path,
) -> None:
    """`Located` binds no `writes`, so the hook keeps the same classes in the same
    order once `Finding` composes it."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _writers_family("", "    writes = ('table',)\n")})
    before = _run(tmp_path, WRITERS_PROBE)
    path = (tmp_path / "pkg/located.py").as_posix()
    _apply(
        tmp_path,
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="Finding"),
            owner=SourceRewriteTarget(file_path=path, qualname="Located"),
        ),
    )
    assert "class Finding(Located, Writer):" in (tmp_path / "pkg/located.py").read_text()
    assert _run(tmp_path, WRITERS_PROBE) == before == "['Finding']\n"


def test_a_hook_selecting_by_a_name_the_owner_would_take_refuses(tmp_path: Path) -> None:
    """Moving `writes` to `Located` makes the hook select `Located` for `Finding`."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _writers_family("    writes = ('table',)\n", "")})
    path = (tmp_path / "pkg/located.py").as_posix()
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Finding"),
        owner=SourceRewriteTarget(file_path=path, qualname="Located"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"selects \['pkg.located.Finding'\] for `writes`"):
        operation.source_edits(snapshot)


def _dispatch_family(owner_body: str, composer_body: str, after_walk: str = "") -> str:
    """A hook that walks the MRO for each name's first binding, as
    `metaclass_registry.MroDispatch` does, and keeps the marked ones."""

    return (
        "from dataclasses import dataclass\n\n\n"
        "def handles(*classes):\n"
        "    def decorate(handler):\n"
        "        handler.__handled_classes__ = classes\n"
        "        return handler\n"
        "    return decorate\n\n\n"
        "class Dispatch:\n"
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
        "        )\n"
        f"{after_walk}\n\n"
        f"class Located:\n{owner_body}\n\n"
        f"class Finding(Dispatch):\n{composer_body}"
    )


DISPATCH_PROBE = (
    "from pkg.located import Finding, Located\n"
    "print(Finding._handler_declarations, issubclass(Finding, Located))\n"
)
OWNER_BODY = "    def label(self):\n        return 'l'\n"


def _dispatch_composition(path: str) -> ComposeExistingClassOperation:
    return ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Finding"),
        owner=SourceRewriteTarget(file_path=path, qualname="Located"),
    )


def test_a_hook_walking_the_mro_for_marked_methods_composes_the_owner(tmp_path: Path) -> None:
    """Each name keeps its winner and the marked ones keep their order, so the
    hook builds the same handler table."""

    composer = OWNER_BODY + "\n    @handles(int)\n    def on_int(self, value):\n        return value\n"
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _dispatch_family(OWNER_BODY, composer)})
    before = _run(tmp_path, DISPATCH_PROBE).split()
    _apply(tmp_path, _dispatch_composition((tmp_path / "pkg/located.py").as_posix()))
    after = _run(tmp_path, DISPATCH_PROBE).split()
    assert after[:-1] == before[:-1] and after[-1] == "True"
    assert "class Finding(Located, Dispatch):" in (tmp_path / "pkg/located.py").read_text()


@pytest.mark.parametrize(
    ("owner", "composer", "after_walk", "refusal"),
    (
        # What the walk found, counted: the owner adds no name, but NRA reads
        # only `getattr` of a constant attribute of each winner.
        (OWNER_BODY, OWNER_BODY, "        cls.count = len(methods)\n", r"uses what the walk of the MRO found"),
        # The moved handler is met after the composer's own handler once composed.
        (
            "    @handles(str)\n    def on_str(self, value):\n        return value\n",
            "    @handles(str)\n    def on_str(self, value):\n        return value\n\n"
            "    @handles(int)\n    def on_int(self, value):\n        return value\n",
            "",
            r"meets \['on_str', 'on_int'\] in that order, and \['on_int', 'on_str'\] once composed",
        ),
    ),
)
def test_a_hook_walking_the_mro_refuses_what_the_name_check_does_not_cover(
    tmp_path: Path, owner: str, composer: str, after_walk: str, refusal: str
) -> None:
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _dispatch_family(owner, composer, after_walk)})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _dispatch_composition((tmp_path / "pkg/located.py").as_posix()).source_edits(snapshot)


def _declared_file_family(key: str) -> str:
    """A base whose subclass hook asks whether the new class's own body binds `key`."""

    return (
        "class Root:\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        f"        if {key!r} in vars(cls) and not getattr(cls, 'durable', False):\n"
        "            raise TypeError(cls.__qualname__)\n\n\n"
        f"class Located:\n{OWNER_BODY}\n\n"
        "class Finding(Root):\n"
        "    filename = 'f'\n"
        "    durable = True\n\n"
        f"{OWNER_BODY}"
    )


VARS_PROBE = "from pkg.located import Finding\nprint(Finding().label(), Finding.filename, Finding.durable)\n"


def test_a_hook_asking_vars_of_the_class_for_a_kept_name_composes_the_owner(tmp_path: Path) -> None:
    """`'filename' in vars(cls)` reads one name of the new class's own body, and
    composing moves only `label` out of it: the hook answers the same."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _declared_file_family("filename")})
    before = _run(tmp_path, VARS_PROBE)
    _apply(tmp_path, _dispatch_composition((tmp_path / "pkg/located.py").as_posix()))
    assert "class Finding(Located, Root):" in (tmp_path / "pkg/located.py").read_text()
    assert _run(tmp_path, VARS_PROBE) == before


def test_a_hook_asking_vars_of_the_class_for_a_moved_name_stops_the_composition(
    tmp_path: Path,
) -> None:
    """`'label' in vars(cls)` is True before composing and False after: the
    moved name leaves the class's own body."""

    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": _declared_file_family("label")})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"reads the class namespace"):
        _dispatch_composition((tmp_path / "pkg/located.py").as_posix()).source_edits(snapshot)


def test_a_metaclass_iterating_the_namespace_stops_the_composition(tmp_path: Path) -> None:
    """Iterating the namespace reads every name in it, the moved ones included."""

    source = (
        "class Meta(type):\n"
        "    def __new__(mcs, name, bases, namespace):\n"
        "        cls = super().__new__(mcs, name, bases, namespace)\n"
        "        cls.names = [key for key in namespace]\n"
        "        return cls\n\n\n"
        "class Base(metaclass=Meta):\n    pass\n\n\n"
        f"class Located:\n{OWNER_BODY}\n\n"
        f"class Finding(Base):\n{OWNER_BODY}"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/located.py": source})
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"located.py:4 `namespace` iterates the class namespace"):
        _dispatch_composition((tmp_path / "pkg/located.py").as_posix()).source_edits(snapshot)


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


def test_composition_needs_a_type_test_on_the_owner_changed_first(tmp_path: Path) -> None:
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
    with pytest.raises(CodemodOperationPreflightError, match="The move needs") as stopped:
        _compose(tmp_path).source_edits(snapshot)
    assert stopped.value.report.needs == (NeededInput.SOURCE_CHANGE,)


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
    assert "from pkg.shapes import Edge" in base
    for child in ("expansion", "conveyor"):
        source = rewritten[(tmp_path / f"pkg/{child}.py").as_posix()]
        assert "def reason" not in source and "delta: int" not in source
        assert "from pkg.shapes import Edge" not in source
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
    with pytest.raises(CodemodOperationPreflightError, match="by position"):
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


def test_the_fold_keeps_a_keyword_only_class_handed_on_as_a_value(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pkg/conveyor.py": CONVEYOR.replace(
                    "@dataclass(frozen=True)\nclass ConveyorProof",
                    "@dataclass(frozen=True, kw_only=True)\nclass ConveyorProof",
                )
                + "\n\nFACTORIES = (ConveyorProof,)\n"
            }
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


def test_the_fold_reads_a_type_form_of_its_class_as_a_type(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pkg/conveyor.py": CONVEYOR
                + "\n\nKINDS = (type[ConveyorProof],)\n"
            }
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


KEYWORD_BINDER = {
    "pyproject.toml": "[tool.nominal-refactor-advisor]\n"
    'keyword-binders = ["pkg.codec:encode"]\n',
    "pkg/codec.py": "def encode(value):\n    return f'{value.__module__}.{value.__qualname__}'\n",
    "pkg/caller.py": "from pkg.codec import encode\n\n\n"
    "def literal(value):\n    return encode(value) if value is not None else 'NULL'\n\n\n"
    "def build(value):\n    return value((), (), 'c', 0)\n\n\n"
    "def build_named(value, edges):\n    return value(roots=(), edges=edges, component='c', delta=2)\n",
}


def test_the_fold_follows_a_class_into_a_function_that_only_encodes_it(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **KEYWORD_BINDER,
            **{
                "pkg/conveyor.py": CONVEYOR
                + "\n\nfrom pkg.caller import literal\n\nKEY = literal(ConveyorProof)\n"
            },
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


def test_the_fold_follows_a_class_into_a_reader_its_declaring_class_names(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pyproject.toml": "[tool.nominal-refactor-advisor]\n"
                'keyword-binders = ["pkg.rows:Table.select"]\n',
                "pkg/rows.py": "class Table:\n    @classmethod\n"
                "    def select(cls, **key):\n        return [key]\n\n\n"
                "class Rows(Table):\n    pass\n",
                "pkg/conveyor.py": CONVEYOR
                + "\n\nfrom pkg.rows import Rows\n\nFOUND = Rows.select(kind=ConveyorProof)\n",
            },
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


def test_the_fold_follows_a_class_in_a_type_form_into_a_keyword_binder(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pyproject.toml": "[tool.nominal-refactor-advisor]\n"
                'keyword-binders = ["pkg.codec:decode"]\n',
                "pkg/codec.py": "def decode(form, data):\n    return data\n",
                "pkg/conveyor.py": CONVEYOR
                + "\n\nfrom pkg.codec import decode\n\nREAD = decode(tuple[ConveyorProof, ...], [])\n",
            },
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


def test_a_keyword_binder_naming_no_function_refuses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pyproject.toml": "[tool.nominal-refactor-advisor]\n"
                'keyword-binders = ["pkg.codec:missing"]\n',
                "pkg/codec.py": "def encode(value):\n    return str(value)\n",
                "pkg/conveyor.py": CONVEYOR
                + "\n\nfrom pkg.codec import encode\n\nKEY = encode(ConveyorProof)\n",
            },
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="names `pkg.codec.missing`"):
        _fold(tmp_path).source_edits(snapshot)



def test_the_fold_follows_a_class_a_script_hands_to_a_keyword_binder(
    tmp_path: Path,
) -> None:
    """A script a string holds is no repository module: its own imports prove
    what `encode` is, and it star-imports nothing that could rebind it."""

    script = (
        "SCRIPT = \'\'\'\n"
        "from pkg.codec import encode\n"
        "from pkg.conveyor import ConveyorProof\n"
        "for name in ('a',):\n"
        "    assert encode(ConveyorProof)\n"
        "\'\'\'\n"
    )
    _write(tmp_path, _proof_files(**KEYWORD_BINDER, **{"pkg/scripts.py": script}))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)

def test_the_fold_refuses_a_class_handed_to_a_function_that_calls_it(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **KEYWORD_BINDER,
            **{
                "pkg/conveyor.py": CONVEYOR
                + "\n\nfrom pkg.caller import build\n\nMADE = build(ConveyorProof)\n"
            },
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="as a value to `build`"):
        _fold(tmp_path).source_edits(snapshot)


def test_the_fold_follows_a_class_into_a_function_that_builds_it_by_name(
    tmp_path: Path,
) -> None:
    """A parameter called with keywords alone binds every field by name: a fold
    that reorders the fields leaves that call as it was."""

    conveyor = CONVEYOR + (
        "\n\nfrom pkg.caller import build_named\nfrom pkg.shapes import Edge\n\n"
        "MADE = build_named(ConveyorProof, (Edge('a'),))\n"
    )
    _write(tmp_path, _proof_files(**KEYWORD_BINDER, **{"pkg/conveyor.py": conveyor}))
    before = _run(tmp_path, "from pkg.conveyor import MADE\nprint(MADE.reason(), MADE.delta)\n")
    _apply(tmp_path, _fold(tmp_path))
    assert _run(tmp_path, "from pkg.conveyor import MADE\nprint(MADE.reason(), MADE.delta)\n") == before


def test_the_fold_follows_a_class_super_reads(tmp_path: Path) -> None:
    """`super(C, obj)` looks names up along `C`'s MRO and never calls `C`."""

    conveyor = CONVEYOR.replace(
        "    def reason(self) -> str:\n",
        "    def is_bare(self) -> bool:\n"
        "        return super(ConveyorProof, self).__repr__ is not None\n\n"
        "    def reason(self) -> str:\n",
    )
    _write(tmp_path, _proof_files(**{"pkg/conveyor.py": conveyor}))
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    assert _fold(tmp_path).source_edits(snapshot)


def test_the_fold_refuses_a_positional_construction_behind_a_local_import(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        _proof_files(
            **{
                "pkg/caller.py": "def empty():\n"
                "    from pkg.conveyor import ConveyorProof as Proof\n"
                "    return Proof((), (), 'c', 0)\n"
            }
        ),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=r"caller\.py:3 binds 4"):
        _fold(tmp_path).source_edits(snapshot)


SCOPED_USERS = (
    "from pkg.records import Point\n\n\n"
    "def top():\n    return Point(1)\n\n\n"
    "def aliased():\n    from pkg.records import Point as P\n    return P(2)\n\n\n"
    "def shadowed(Point):\n    return Point(3)\n\n\n"
    "def rebound():\n    Point = dict\n    return Point(x=4)\n"
)


def test_class_reads_resolve_through_the_scope_that_binds_them(tmp_path: Path) -> None:
    """A function's own import names the class; its own parameter or local does not."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/records.py": SHAPES.replace("Edge", "Point"),
            "pkg/users.py": SCOPED_USERS,
            "pkg/only_local.py": "def build():\n"
            "    from pkg.records import Point\n    return Point(5)\n",
        },
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))

    sites = ClassReferenceSites(snapshot, frozenset(("pkg.records.Point",))).sites

    assert sorted((Path(site.file_path).name, site.node.lineno) for site in sites) == [
        ("only_local.py", 3),
        ("users.py", 5),
        ("users.py", 10),
    ]


COVERED_OWNERS = (
    "from dataclasses import dataclass\n\n\n"
    "@dataclass(frozen=True)\nclass SourceLineReference:\n    file_path: str\n    line: int\n\n\n"
    "@dataclass(frozen=True)\nclass ClassNameMixin:\n    class_name: str\n\n\n"
    "@dataclass(frozen=True)\nclass ClassLine(ClassNameMixin, SourceLineReference):\n    pass\n"
)
COVERED_FACTS = (
    "from dataclasses import dataclass, fields\n\n\n"
    "@dataclass(frozen=True)\nclass AxisFact:\n"
    "    file_path: str\n    line: int\n    class_name: str\n    signals: tuple[str, ...]\n\n\n"
    "@dataclass(frozen=True)\nclass AuthorityShape:\n"
    "    file_path: str\n    line: int\n    class_name: str\n    method_names: tuple[str, ...]\n"
)
COVERED_PROBE = (
    "from dataclasses import fields\n"
    "from pkg.facts import AxisFact, AuthorityShape\n"
    "for cls, extra in ((AxisFact, ('s',)), (AuthorityShape, ('m',))):\n"
    "    value = cls('f.py', 3, 'C', extra)\n"
    "    print(cls.__name__, [field.name for field in fields(cls)], value)\n"
)


def test_classes_restating_a_block_owned_by_two_classes_compose_both(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/owners.py": COVERED_OWNERS, "pkg/facts.py": COVERED_FACTS},
    )
    before = _run(tmp_path, COVERED_PROBE)
    facts = (tmp_path / "pkg/facts.py").as_posix()
    owners = (tmp_path / "pkg/owners.py").as_posix()

    rewritten = _apply(
        tmp_path,
        ComposeExistingClassesOperation(
            target=SourceRewriteTarget(file_path=facts, qualname="AxisFact"),
            owners=(
                SourceRewriteTarget(file_path=owners, qualname="ClassNameMixin"),
                SourceRewriteTarget(file_path=owners, qualname="SourceLineReference"),
            ),
            composers=(SourceRewriteTarget(file_path=facts, qualname="AuthorityShape"),),
        ),
    )

    assert _run(tmp_path, COVERED_PROBE) == before
    source = rewritten[facts]
    assert "class AxisFact(ClassNameMixin, SourceLineReference):" in source
    assert "class AuthorityShape(ClassNameMixin, SourceLineReference):" in source
    assert "file_path: str" not in source
    assert source.count("from pkg.owners import") + source.count("from .owners import") == 1


def _installed_dispatch_family(hook: str, module: str) -> dict[str, str]:
    """A hook calling an inherited classmethod while a class is created whose MRO
    passes an installed class and `typing.Generic` before the classmethod's owner,
    as agent-comms `WakeClaim` passes `metaclass_registry.MroDispatch`."""

    return {
        "repo/pkg/__init__.py": "",
        "repo/pkg/located.py": (
            "from dataclasses import dataclass\n"
            "from typing import Generic, TypeVar\n\n"
            f"from {module} import Dispatch\n\n"
            "T = TypeVar('T')\n\n\n"
            "class Outputs:\n"
            "    @classmethod\n"
            "    def writer_names(cls):\n"
            "        return tuple(parent for parent in cls.__mro__ if 'writes' in vars(parent))\n\n\n"
            "class Source(Outputs):\n"
            "    def __init_subclass__(cls, **kwargs):\n"
            "        super().__init_subclass__(**kwargs)\n"
            "        cls.writers = cls.writer_names()\n\n\n"
            "class Holder(Generic[T]):\n"
            "    pass\n\n\n"
            "@dataclass(frozen=True)\nclass Located:\n    file_path: str\n    line: int\n\n\n"
            "@dataclass(frozen=True)\nclass Finding(Dispatch, Holder[int], Source):\n"
            "    file_path: str\n    line: int\n    message: str\n    writes = ('table',)\n"
        ),
        f"site/{module}.py": f"class Dispatch:\n{hook}",
    }


def test_a_hook_reaches_its_classmethod_past_an_installed_base_and_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Dispatch` is read from its installed source and binds no `writer_names`;
    `Generic` is a native root. The lookup finds `Outputs.writer_names` and the
    composition keeps what the hook stores."""

    _write(tmp_path, _installed_dispatch_family("    pass\n", "extdispatch_kept"))
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    probe = "from pkg.located import Finding\nprint([w.__name__ for w in Finding.writers], Finding.__mro__[1].__name__)\n"

    def run() -> str:
        return subprocess.run(
            [sys.executable, "-c", probe], cwd=tmp_path / "repo", check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(tmp_path / "site")},
        ).stdout

    before = run()
    path = (tmp_path / "repo/pkg/located.py").as_posix()
    _apply(
        tmp_path / "repo",
        ComposeExistingClassOperation(
            target=SourceRewriteTarget(file_path=path, qualname="Finding"),
            owner=SourceRewriteTarget(file_path=path, qualname="Located"),
        ),
    )
    assert "class Finding(Located, Dispatch, Holder[int], Source):" in Path(path).read_text()
    assert run().split()[0] == before.split()[0] == "['Finding']"


def test_an_installed_base_binding_the_called_name_is_read_as_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the installed class binds `writer_names` itself, the call resolves to
    the installed source, which NRA reads: one handing the class to a function
    it cannot read refuses."""

    _write(tmp_path, _installed_dispatch_family(
        "    @classmethod\n    def writer_names(cls):\n        return opaque(cls)\n", "extdispatch_binding"
    ))
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    path = (tmp_path / "repo/pkg/located.py").as_posix()
    operation = ComposeExistingClassOperation(
        target=SourceRewriteTarget(file_path=path, qualname="Finding"),
        owner=SourceRewriteTarget(file_path=path, qualname="Located"),
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path / "repo"))
    with pytest.raises(CodemodOperationPreflightError, match=r"extdispatch_binding.py:\d+ `opaque\(cls\)`"):
        operation.source_edits(snapshot)
