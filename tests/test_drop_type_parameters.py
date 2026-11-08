from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.refactors.type_parameters import (
    DroppedTypeParameter,
    DropTypeParametersOperation,
)

BASE = (
    "class Context:\n    pass\n\n\n"
    "class Attached(Context):\n    pass\n\n\n"
    "class Request[Ctx: Context]:\n    thread: str = 'main'\n\n\n"
    "class Result[Ctx: Context](Request[Ctx]):\n    pass\n\n\n"
    "class Served[Ctx: Context](Request[Ctx]):\n"
    "    def bind(self, ctx: Ctx) -> Ctx:\n        return ctx\n"
)
LEAF = (
    "from pkg.base import Attached, Result, Served\n\n\n"
    "class Stop(Result[Attached]):\n    pass\n\n\n"
    "class ServedStop(Served[Attached], Stop):\n    pass\n"
)


def _write(root: Path, sources: dict[str, str]) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    for name, source in sources.items():
        (root / f"pkg/{name}.py").write_text(source)


def _dropped(root: Path, module: str, qualname: str, parameter: str = "Ctx") -> DroppedTypeParameter:
    return DroppedTypeParameter(
        target=SourceRewriteTarget(file_path=(root / f"pkg/{module}.py").as_posix(), qualname=qualname),
        parameter=parameter,
    )


def _simulate(root: Path, *parameters: DroppedTypeParameter):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = DropTypeParametersOperation(parameters=parameters)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="drop").with_operation(replayed).simulate(snapshot)


def _source(root: Path, result, module: str) -> str:
    return result.simulation.rewritten_sources[(root / f"pkg/{module}.py").as_posix()]


@pytest.mark.native_proof
def test_a_chain_passing_its_parameter_on_drops_it_family_wide(tmp_path: Path) -> None:
    _write(tmp_path, {"base": BASE, "leaf": LEAF})
    result = _simulate(
        tmp_path, _dropped(tmp_path, "base", "Request"), _dropped(tmp_path, "base", "Result"),
    )

    assert result.is_clean is True
    base = _source(tmp_path, result, "base")
    assert "class Request:\n" in base
    assert "class Result(Request):\n" in base
    assert "class Served[Ctx: Context](Request):\n" in base
    assert "class Stop(Result):\n" in _source(tmp_path, result, "leaf")
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)
    completed = subprocess.run(
        [sys.executable, "-c", "from pkg.leaf import ServedStop; print(ServedStop().thread)"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert completed.stdout.split() == ["main"]


@pytest.mark.parametrize(
    "sources,dropped,refusal",
    (
        ({"base": BASE.replace("thread: str = 'main'", "context: Ctx | None = None")},
         ("Request",), "Request reads 'Ctx'"),
        ({"base": BASE + "\n\nAlias = Request[Context, Context]\n"},
         ("Request", "Result"), "passes 2 arguments"),
        ({"base": BASE + "\n\ndef read(request: 'Request[Context]') -> None: ...\n"},
         ("Request", "Result"), "String annotations"),
        ({"base": BASE}, ("Context",), "declares no type parameter 'Ctx'"),
    ),
)
def test_dropping_refuses_a_parameter_something_still_reads(
    tmp_path: Path, sources: dict[str, str], dropped: tuple[str, ...], refusal: str,
) -> None:
    _write(tmp_path, sources)

    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _simulate(tmp_path, *(_dropped(tmp_path, "base", qualname) for qualname in dropped))
