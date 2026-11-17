"""A member a class declares exactly as the class it would inherit it from, found and deleted.

Each case writes a package, runs it, applies the recipe the finding offers,
checks the exact rewritten source, runs it again and re-scans.
"""

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_overrides import RestatedOverrideDetector
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

ENDS = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class WatchEnd(ABC):\n"
    "    @abstractmethod\n    def line(self) -> str: ...\n\n"
    "    async def settled(self, watch) -> 'WatchEnd':\n"
    "        \"\"\"What is left for the model.\"\"\"\n        return self\n\n\n"
    "class WatchDropped(WatchEnd):\n"
    "    def line(self) -> str:\n        return 'dropped'\n\n"
    "    async def settled(self, watch) -> 'WatchEnd':\n        return watch\n\n\n"
    "class BackToModel(WatchEnd):\n"
    "    exit_code = 1\n\n"
    "    def line(self) -> str:\n        return 'back'\n\n"
    "    async def settled(self, watch) -> 'WatchEnd':\n        return self\n\n\n"
    "class Unopened(WatchDropped):\n"
    "    async def settled(self, watch) -> 'WatchEnd':\n        return self\n"
)
MAIN = (
    "import asyncio\n\nfrom pkg.ends import BackToModel, Unopened\n\n"
    "for end in (BackToModel(), Unopened()):\n"
    "    print(type(end).__name__, asyncio.run(end.settled('w')) is end, end.line(), "
    "sorted(type(end).__abstractmethods__))\n"
)
DELETED = ENDS.replace(
    "    def line(self) -> str:\n        return 'back'\n\n"
    "    async def settled(self, watch) -> 'WatchEnd':\n        return self\n",
    "    def line(self) -> str:\n        return 'back'\n",
)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _findings(root: Path):
    with ScanCache.scope():
        return RestatedOverrideDetector().detect(
            parse_python_modules(
                root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def _evaluation(root: Path, symbol: str):
    finding = next(f for f in _findings(root) if f.primary_evidence.symbol == symbol)
    return RestatedOverrideDetector().evaluate_recipe_for_finding(finding, _snapshot(root))


def test_restated_override_continues_the_stable_sequence():
    assert PatternId.RESTATED_OVERRIDE == 52


def test_a_member_restating_what_it_inherits_is_deleted(tmp_path: Path):
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/ends.py": ENDS, "main.py": MAIN})
    expected = _run(tmp_path)
    findings = _findings(tmp_path)
    # `Unopened.settled` restates `WatchEnd.settled`, but lookup without it reaches
    # `WatchDropped.settled`, which answers otherwise: no finding.
    assert [finding.primary_evidence.symbol for finding in findings] == ["BackToModel"]
    assert "`settled` exactly as `WatchEnd` does" in findings[0].summary
    snapshot = _snapshot(tmp_path)
    evaluation = RestatedOverrideDetector().evaluate_recipe_for_finding(findings[0], snapshot)
    assert evaluation.rejection_reason == ""
    simulation = CodemodPlanSequence.from_operations(evaluation.required_recipe.operations).simulate(snapshot)
    assert simulation.is_clean
    simulation.apply()
    assert (tmp_path / "pkg/ends.py").read_text() == DELETED
    assert _run(tmp_path) == expected
    assert _findings(tmp_path) == []


def test_a_copy_whose_free_name_binds_otherwise_is_no_finding(tmp_path: Path):
    base = "def answer():\n    return 1\n\n\nclass Base:\n    def value(self):\n        return answer()\n"
    child = (
        "from pkg.base import Base\n\n\ndef answer():\n    return 2\n\n\n"
        "class Child(Base):\n    def value(self):\n        return answer()\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/base.py": base, "pkg/child.py": child})
    assert _findings(tmp_path) == []


def test_a_copy_reading_its_class_cell_or_a_private_name_is_no_finding(tmp_path: Path):
    source = (
        "class Base:\n    def value(self):\n        return 1\n\n"
        "    def again(self):\n        return self.__hidden\n\n\n"
        "class Middle(Base):\n    def value(self):\n        return super().value() + 1\n\n"
        "    def again(self):\n        return self.__hidden\n\n\n"
        "class Child(Middle):\n    def value(self):\n        return super().value() + 1\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/cells.py": source})
    assert _findings(tmp_path) == []


def test_a_class_scope_name_in_the_header_is_no_finding(tmp_path: Path):
    source = (
        "class Base:\n    limit = 1\n\n    def value(self, at=limit):\n        return at\n\n\n"
        "class Child(Base):\n    limit = 2\n\n    def value(self, at=limit):\n        return at\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/scoped.py": source})
    assert _findings(tmp_path) == []


def test_a_creation_hook_reading_the_namespace_needs_a_proof(tmp_path: Path):
    source = (
        "class Base:\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n"
        "        cls.declared = tuple(vars(cls))\n\n"
        "    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/hooked.py": source})
    evaluation = _evaluation(tmp_path, "Child")
    assert "`tuple(vars(cls))`" in evaluation.rejection_reason
    assert "while `Child` is created" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_namespace_read_of_another_class_leaves_the_deletion_proved(tmp_path: Path):
    source = (
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "class Other:\n    pass\n\n\n"
        "def other():\n    return list(vars(Other)), Child.__dict__.get('missing')\n\n\n"
        "def any_class(cls):\n    return list(vars(cls))\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    evaluation = _evaluation(tmp_path, "Child")
    assert "pkg/read.py:20" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
    _write(tmp_path, {"pkg/read.py": source.replace("\n\n\ndef any_class(cls):\n    return list(vars(cls))\n", "\n")})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""


def test_an_unmodelled_decorator_needs_a_proof(tmp_path: Path):
    source = (
        "from functools import cache\n\n\n"
        "class Base:\n    @cache\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    @cache\n    def value(self):\n        return 1\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/cached.py": source})
    evaluation = _evaluation(tmp_path, "Child")
    assert "`@cache`" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_an_annotated_declaration_is_no_finding(tmp_path: Path):
    source = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\nclass Base:\n    text: str\n    limit: int = 1\n\n\n"
        "@dataclass\nclass Child(Base):\n    text: str\n    limit: int = 1\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/fields.py": source})
    assert _findings(tmp_path) == []


def test_a_namespace_read_of_an_instance_parameter_sees_no_class(tmp_path: Path):
    source = (
        "import argparse\n\n\n"
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "def given(arguments: argparse.Namespace):\n    return dict(vars(arguments).items())\n\n\n"
        "def parsed(parser: argparse.ArgumentParser):\n"
        "    arguments: argparse.Namespace = parser.parse_args()\n    return vars(arguments)['x']\n\n\n"
        "def classes(kind: type[Base]):\n    return dict(vars(kind).items())\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    evaluation = _evaluation(tmp_path, "Child")
    assert "pkg/read.py:24" in evaluation.rejection_reason
    _write(tmp_path, {"pkg/read.py": source.split("\n\n\ndef classes")[0] + "\n"})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""


def test_a_namespace_read_through_type_of_a_family_sees_only_that_family(tmp_path: Path):
    source = (
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "class Capability:\n    pass\n\n\n"
        "def table(owner: type[Capability]):\n    return dict(vars(owner).items())\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""
    _write(tmp_path, {"pkg/read.py": source.replace("type[Capability]", "type[Base]")})
    evaluation = _evaluation(tmp_path, "Child")
    assert "pkg/read.py:16" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_namespace_read_of_a_derived_key_sees_no_member_name(tmp_path: Path):
    source = (
        "class Fact:\n"
        "    def __init__(self, derive):\n        self.derive = derive\n\n"
        "    def __set_name__(self, owner, name):\n        self.slot = f\"_{name.lstrip('_')}_derived\"\n\n"
        "    def __get__(self, instance, owner):\n"
        "        if self.slot not in owner.__dict__:\n"
        "            setattr(owner, self.slot, self.derive(owner))\n"
        "        return owner.__dict__[self.slot]\n\n\n"
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/facts.py": source})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""
    _write(tmp_path, {"pkg/facts.py": source + "\n\ndef rename(fact):\n    fact.slot = 'value'\n"})
    evaluation = _evaluation(tmp_path, "Child")
    assert "owner.__dict__" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_namespace_only_resolving_annotations_sees_no_unannotated_name(tmp_path: Path):
    source = (
        "import sys\nfrom typing import get_type_hints\n\n\n"
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "def hints(declaration: type, annotations: dict):\n"
        "    return get_type_hints(\n"
        "        declaration, globalns=vars(sys.modules[declaration.__module__]),\n"
        "        localns={'T': int} | dict(vars(declaration)),\n"
        "    )\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/hints.py": source})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""
    _write(tmp_path, {"pkg/hints.py": source + "\n\ndef annotated(x: 'value') -> None:\n    pass\n"})
    evaluation = _evaluation(tmp_path, "Child")
    assert "dict(vars(declaration))" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_namespace_read_the_type_checker_types_as_an_instance_sees_no_class(tmp_path: Path):
    source = (
        "import argparse\n\n\n"
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    arguments = parser.parse_args([])\n"
        "    return vars(arguments)\n\n\n"
        "def held(value: object):\n    return vars(value)\n\n\n"
        "def also(value: object):\n    return value.__dict__\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    evaluation = _evaluation(tmp_path, "Child")
    assert "pkg/read.py:21" in evaluation.rejection_reason
    assert "pkg/read.py:25" in evaluation.rejection_reason
    assert "whose values can be classes" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs
    _write(tmp_path, {"pkg/read.py": source.split("\n\n\ndef held")[0] + "\n"})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""


def test_a_namespace_read_the_type_checker_types_as_another_family_sees_no_member(tmp_path: Path):
    source = (
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "class Capability:\n    pass\n\n\n"
        "def owners() -> tuple[type[Capability], ...]:\n    return (Capability,)\n\n\n"
        "def table():\n    return [dict(vars(owner)) for owner in owners()]\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""
    _write(tmp_path, {"pkg/read.py": source.replace("type[Capability]", "type[Base]").replace("(Capability,)", "(Base,)")})
    evaluation = _evaluation(tmp_path, "Child")
    assert "`dict(vars(owner))`" in evaluation.rejection_reason
    assert "whose values can be that class" in evaluation.rejection_reason
    assert NeededInput.PROOF in evaluation.needs


def test_a_namespace_read_asking_only_its_own_annotations_sees_no_unannotated_member(tmp_path: Path):
    source = (
        "import annotationlib\n\n\n"
        "class Base:\n    def value(self):\n        return 1\n\n\n"
        "class Child(Base):\n    def value(self):\n        return 1\n\n\n"
        "def generated(cls: type):\n"
        "    found = []\n"
        "    for owner in cls.__mro__:\n"
        "        declared = tuple(annotationlib.get_annotations(owner, format=annotationlib.Format.STRING))\n"
        "        for position, name in enumerate(declared):\n"
        "            found.append(vars(owner).get(name))\n"
        "    return found\n"
    )
    _write(tmp_path, {"pkg/__init__.py": "", "pkg/read.py": source})
    assert _evaluation(tmp_path, "Child").rejection_reason == ""
    for other in (
        source.replace("get_annotations(owner,", "get_annotations(Base,"),
        source.replace(
            "        declared = tuple(annotationlib.get_annotations(owner, format=annotationlib.Format.STRING))\n",
            "        if owner is cls:\n            declared = tuple(annotationlib.get_annotations(owner))\n",
        ),
    ):
        _write(tmp_path, {"pkg/read.py": other})
        evaluation = _evaluation(tmp_path, "Child")
        assert "vars(owner).get" in evaluation.rejection_reason, other
        assert NeededInput.PROOF in evaluation.needs
