"""A lazily annotated module becomes stringized only where the strings evaluate to
the same objects, and only once every raw annotation reader has been reviewed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPreflightStatus,
    CodemodSourceSnapshot,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.refactors.annotation_mode import (
    StringizeAnnotationsOperation,
)


def _write(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _operation(root: Path) -> StringizeAnnotationsOperation:
    return StringizeAnnotationsOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/plans.py").as_posix()),
    )


def _preflight(root: Path):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    (report,) = _operation(root).preflight_reports(snapshot)
    return report


_PLANS = (
    '"""Plans."""\n\n'
    "from dataclasses import dataclass\n\n"
    "from pkg.bindings import Binding\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Plan[T]:\n"
    "    binding: Binding\n"
    "    count: int = 0\n\n"
    "    def chosen(self, value: T) -> Plan[T]:\n"
    "        def local(other: Binding) -> None:\n"
    "            seen: Undefined = other\n"
    "        return self\n"
)


def _package(root: Path, plans: str = _PLANS) -> None:
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/bindings.py", "class Binding:\n    pass\n")
    _write(root, "pkg/plans.py", plans)


def test_a_lazy_module_whose_annotations_resolve_at_module_scope_is_stringized(
    tmp_path: Path,
) -> None:
    _package(tmp_path)
    simulation = (
        RefactorRecipe("stringize-plans")
        .with_operation(_operation(tmp_path))
        .simulate(CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path)))
    )

    assert simulation.is_clean
    simulation.apply()
    plans = (tmp_path / "pkg/plans.py").read_text()
    assert plans.startswith('"""Plans."""\n\nfrom __future__ import annotations\n\nfrom dataclasses')
    resolved = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import typing; from pkg.plans import Plan; from pkg.bindings import Binding; "
                "assert Plan.__annotations__['binding'] == 'Binding'; "
                "assert typing.get_type_hints(Plan)['binding'] is Binding; "
                "assert Plan(Binding()).chosen(1).count == 0"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr


def test_an_annotation_reading_an_enclosing_scope_is_refused(tmp_path: Path) -> None:
    _package(
        tmp_path,
        "class Plan:\n"
        "    Kind = int\n\n"
        "    def kind(self) -> Kind:\n"
        "        return 1\n",
    )

    report = _preflight(tmp_path)

    assert report.status is CodemodPreflightStatus.FAILED
    assert "Kind" in report.message


_TOLERANT_READERS = (
    "import annotationlib\n"
    "from inspect import get_annotations\n"
    "from typing import get_type_hints\n\n\n"
    "def names(owner):\n"
    "    return tuple(get_annotations(owner))\n\n\n"
    "def declared(owner, name):\n"
    "    own = get_annotations(owner)\n"
    "    return name in own\n\n\n"
    "def strings(owner):\n"
    "    return get_annotations(owner, format=annotationlib.Format.STRING)\n\n\n"
    "def evaluated(owner):\n"
    "    return get_annotations(owner, eval_str=True)\n\n\n"
    "class Codec:\n"
    "    @staticmethod\n"
    "    def _evaluate(declaration, annotations):\n"
    "        own = type(declaration.__name__, (), {})\n"
    "        own.__annotations__ = annotations\n"
    "        return get_type_hints(own)\n\n"
    "    @classmethod\n"
    "    def types(cls, owner):\n"
    "        return cls._evaluate(\n"
    "            owner, get_annotations(owner, format=annotationlib.Format.FORWARDREF)\n"
    "        )\n\n"
    "    @classmethod\n"
    "    def field(cls, owner, name):\n"
    "        stored = get_annotations(owner, format=annotationlib.Format.FORWARDREF)\n"
    "        return cls._evaluate(owner, {name: stored[name]})[name]\n"
)


def test_reads_that_use_keys_or_reach_get_type_hints_are_proved(tmp_path: Path) -> None:
    _package(tmp_path)
    _write(tmp_path, "pkg/reader.py", _TOLERANT_READERS)

    report = _preflight(tmp_path)

    assert report.status is CodemodPreflightStatus.PASSED, report.message


def test_a_read_that_keeps_the_stored_values_is_refused(tmp_path: Path) -> None:
    _package(tmp_path)
    reader = (
        _TOLERANT_READERS + "\n\n"
        "if True:\n"
        "    class Guarded:\n"
        "        def stored(self, owner):\n"
        "            return owner.__annotations__\n"
    )
    _write(tmp_path, "pkg/reader.py", reader)
    line = reader.splitlines().index("            return owner.__annotations__") + 1

    report = _preflight(tmp_path)

    assert report.status is CodemodPreflightStatus.FAILED
    assert report.detail.unproved_annotation_reads == (f"pkg.reader.Guarded.stored:{line}",)


def test_a_stringized_module_is_refused(tmp_path: Path) -> None:
    _package(tmp_path, "from __future__ import annotations\n\nVALUE: int = 1\n")

    assert _preflight(tmp_path).status is CodemodPreflightStatus.FAILED
