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
from nominal_refactor_advisor.refactors.served_subclass import ServeClassMembersOperation

FAMILY = (
    "from abc import ABC, abstractmethod\n\n\n"
    "class Request(ABC):\n"
    "    thread = 'main'\n\n"
    "    def describe(self) -> str:\n"
    "        return self.thread\n\n\n"
    "class Prompt(Request):\n"
    "    text = 'hi'\n\n"
    "    def apply(self) -> str:\n"
    "        return self.result() + '!'\n\n"
    "    def result(self) -> str:\n"
    "        return self.text\n"
)


def _write(root: Path, family: str = FAMILY) -> Path:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    path = root / "pkg/family.py"
    path.write_text(family)
    return path


def _serve(root: Path, header: str = "class ServedPrompt(Prompt)") -> ServeClassMembersOperation:
    return ServeClassMembersOperation(
        target=SourceRewriteTarget(file_path=(root / "pkg/family.py").as_posix(), qualname="Prompt"),
        served_header=header,
        member_names=("apply", "result"),
    )


def _simulate(root: Path, operation: ServeClassMembersOperation):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="serve").with_operation(replayed).simulate(snapshot)


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "header", ("class ServedPrompt(Prompt)", "class ServedPrompt[T: str](Prompt[T])"),
)
def test_serving_members_moves_them_onto_a_subclass_declared_after_the_declaration(
    tmp_path: Path, header: str,
) -> None:
    path = _write(tmp_path, FAMILY.replace("class Prompt(Request):", "class Prompt[T: str = str](Request):"))
    result = _simulate(tmp_path, _serve(tmp_path, header))
    rewritten = result.simulation.rewritten_sources[path.as_posix()]

    assert result.is_clean is True
    assert f"    text = 'hi'\n\n\n{header}:\n    def apply" in rewritten
    path.write_text(rewritten)
    completed = subprocess.run(
        [sys.executable, "-c",
         "from pkg.family import Prompt, ServedPrompt; "
         "print(ServedPrompt().apply(), hasattr(Prompt(), 'apply'), Prompt().describe())"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert completed.stdout.split() == ["hi!", "False", "main"]


@pytest.mark.parametrize(
    "family,refusal",
    (
        (FAMILY.replace("        return self.thread", "        return self.result()"),
         "Request reads 'result'"),
        (FAMILY + "\n\nclass Fresh(Prompt):\n    def result(self) -> str:\n        return super().result()\n",
         "Fresh reads 'result'"),
        (FAMILY.replace("    def describe(self) -> str:\n        return self.thread\n",
                        "    @abstractmethod\n    def apply(self) -> str: ...\n"),
         "Request declares 'apply' abstract"),
        (FAMILY.replace("    def result(self) -> str:\n        return self.text\n",
                        "    def result(self, default: str = text) -> str:\n        return default\n"),
         "'result' reads class-scope 'text', which stays behind"),
        (FAMILY.replace("return self.text", "return self.__text"),
         "'result' uses private name '__text'"),
    ),
)
def test_serving_refuses_a_member_the_plain_family_still_needs(
    tmp_path: Path, family: str, refusal: str,
) -> None:
    _write(tmp_path, family)

    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _simulate(tmp_path, _serve(tmp_path))
