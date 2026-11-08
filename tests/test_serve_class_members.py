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
from nominal_refactor_advisor.refactors.served_subclass import (
    ServeClassMembersOperation,
    ServedClass,
)

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


def _write(root: Path, sources: dict[str, str]) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    for name, source in sources.items():
        (root / f"pkg/{name}.py").write_text(source)


def _served(root: Path, module: str, qualname: str, header: str, *members: str, **header_moves) -> ServedClass:
    return ServedClass(
        target=SourceRewriteTarget(file_path=(root / f"pkg/{module}.py").as_posix(), qualname=qualname),
        served_header=header,
        member_names=members,
        **header_moves,
    )


def _simulate(root: Path, *classes: ServedClass):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = ServeClassMembersOperation(classes=classes)
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="serve").with_operation(replayed).simulate(snapshot)


def _run(root: Path, script: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=root, check=True, capture_output=True, text=True,
    )
    return completed.stdout.split()


def _apply(root: Path, result) -> None:
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "header", ("class ServedPrompt(Prompt)", "class ServedPrompt[T: str](Prompt[T])"),
)
def test_serving_moves_members_onto_a_subclass_declared_after_the_declaration(
    tmp_path: Path, header: str,
) -> None:
    _write(tmp_path, {"family": FAMILY.replace("class Prompt(Request):", "class Prompt[T: str = str](Request):")})
    result = _simulate(tmp_path, _served(tmp_path, "family", "Prompt", header, "apply", "result"))
    rewritten = result.simulation.rewritten_sources[(tmp_path / "pkg/family.py").as_posix()]

    assert result.is_clean is True
    assert f"    text = 'hi'\n\n\n{header}:\n    def apply" in rewritten
    _apply(tmp_path, result)
    assert _run(tmp_path, (
        "from pkg.family import Prompt, ServedPrompt; "
        "print(ServedPrompt().apply(), hasattr(Prompt(), 'apply'), Prompt().describe())"
    )) == ["hi!", "False", "main"]


@pytest.mark.native_proof
def test_a_chain_calling_super_is_served_together_across_modules(tmp_path: Path) -> None:
    _write(tmp_path, {
        "base": (
            "from abc import ABC, abstractmethod\n\n\n"
            "class Request(ABC):\n"
            "    thread = 'main'\n\n"
            "    def apply(self) -> str:\n"
            "        return self.result()\n\n"
            "    @abstractmethod\n"
            "    def result(self) -> str: ...\n"
        ),
        "leaf": (
            "from pkg.base import Request\n\n\n"
            "class Prompt(Request):\n"
            "    def apply(self) -> str:\n"
            "        return super().apply() + '!'\n\n"
            "    def result(self) -> str:\n"
            "        return self.thread\n"
        ),
    })
    alone = (_served(tmp_path, "base", "Request", "class ServedRequest(Request)", "apply", "result"),)
    with pytest.raises(CodemodOperationPreflightError, match="Prompt reads 'apply'"):
        _simulate(tmp_path, *alone)

    result = _simulate(
        tmp_path,
        *alone,
        _served(tmp_path, "leaf", "Prompt", "class ServedPrompt(ServedRequest, Prompt)", "apply", "result"),
    )

    assert result.is_clean is True
    _apply(tmp_path, result)
    assert _run(tmp_path, (
        "from pkg.leaf import Prompt, ServedPrompt; "
        "print(ServedPrompt().apply(), Prompt().thread, hasattr(Prompt, 'apply'))"
    )) == ["main!", "main", "False"]


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
    _write(tmp_path, {"family": family})

    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _simulate(tmp_path, _served(tmp_path, "family", "Prompt", "class ServedPrompt(Prompt)", "apply", "result"))


def test_a_served_header_reads_only_names_bound_where_it_is_declared(tmp_path: Path) -> None:
    _write(tmp_path, {"family": FAMILY})

    with pytest.raises(CodemodOperationPreflightError, match="reads unbound 'Missing'"):
        _simulate(tmp_path, _served(tmp_path, "family", "Prompt", "class ServedPrompt(Missing, Prompt)", "apply"))


@pytest.mark.native_proof
def test_an_abstract_base_moves_to_the_served_header_and_an_intermediate_stays_unnamed(tmp_path: Path) -> None:
    _write(tmp_path, {"family": (
        "from abc import ABC, abstractmethod\n\n\n"
        "class Command(ABC):\n"
        "    @abstractmethod\n"
        "    def apply(self) -> str: ...\n\n\n"
        "class Request(Command):\n"
        "    def apply(self) -> str:\n"
        "        return 'applied'\n"
    )})
    result = _simulate(tmp_path, _served(
        tmp_path, "family", "Request",
        "class ServedRequest(Command, Request, metaclass=type)", "apply",
        moved_bases=("Command",), declaration_keywords=("metaclass=type",),
    ))
    rewritten = result.simulation.rewritten_sources[(tmp_path / "pkg/family.py").as_posix()]

    assert "class Request(metaclass=type):\n    pass" in rewritten
    _apply(tmp_path, result)
    assert _run(tmp_path, (
        "from pkg.family import Request, ServedRequest; "
        "print(ServedRequest().apply(), type(Request()).__name__, hasattr(Request, 'apply'))"
    )) == ["applied", "Request", "False"]


def test_a_moved_base_must_be_declared_and_taken_by_the_served_header(tmp_path: Path) -> None:
    _write(tmp_path, {"family": FAMILY})

    with pytest.raises(CodemodOperationPreflightError, match="does not take base 'Request'"):
        _simulate(tmp_path, _served(
            tmp_path, "family", "Prompt", "class ServedPrompt(Prompt)", "apply", moved_bases=("Request",),
        ))


def test_a_served_header_passes_what_its_lineage_requires_of_subclasses(tmp_path: Path) -> None:
    _write(tmp_path, {"family": FAMILY.replace(
        "class Request(ABC):\n",
        "class Request(ABC):\n    def __init_subclass__(cls, *, action: str = 'none', kind: str, **kwargs):\n"
        "        super().__init_subclass__(**kwargs)\n\n",
    ).replace("class Prompt(Request):", "class Prompt(Request, kind='prompt'):")})

    with pytest.raises(CodemodOperationPreflightError, match="'ServedPrompt' must pass 'kind'"):
        _simulate(tmp_path, _served(tmp_path, "family", "Prompt", "class ServedPrompt(Prompt)", "apply", "result"))
    assert _simulate(tmp_path, _served(
        tmp_path, "family", "Prompt", "class ServedPrompt(Prompt, kind='prompt')", "apply", "result",
    )).is_clean
