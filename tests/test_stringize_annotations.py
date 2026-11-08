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


def _operation(root: Path, *reviewed: str) -> StringizeAnnotationsOperation:
    return StringizeAnnotationsOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/plans.py").as_posix()),
        reviewed_annotation_readers=reviewed,
    )


def _preflight(root: Path, *reviewed: str):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    (report,) = _operation(root, *reviewed).preflight_reports(snapshot)
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


def test_a_raw_annotation_reader_must_be_reviewed(tmp_path: Path) -> None:
    _package(tmp_path)
    _write(
        tmp_path,
        "pkg/reader.py",
        "from inspect import get_annotations\n\n"
        "def names(owner):\n"
        "    return tuple(get_annotations(owner))\n\n"
        "def strings(owner):\n"
        "    return get_annotations(owner, format=Format.STRING)\n\n"
        "def evaluated(owner):\n"
        "    return get_annotations(owner, eval_str=True)\n\n"
        "def written(owner, annotations):\n"
        "    owner.__annotations__ = annotations\n\n"
        "if True:\n"
        "    class Guarded:\n"
        "        def stored(self, owner):\n"
        "            return owner.__annotations__\n",
    )
    readers = ("pkg.reader.Guarded.stored", "pkg.reader.names")

    refused = _preflight(tmp_path)
    accepted = _preflight(tmp_path, *readers)

    assert refused.status is CodemodPreflightStatus.FAILED
    assert refused.detail.unreviewed_annotation_readers == readers
    assert accepted.status is CodemodPreflightStatus.PASSED


def test_a_stringized_module_is_refused(tmp_path: Path) -> None:
    _package(tmp_path, "from __future__ import annotations\n\nVALUE: int = 1\n")

    assert _preflight(tmp_path).status is CodemodPreflightStatus.FAILED
