"""The installed typeshed stdlib stubs, as declarations NRA reads and never edits.

`ast.AnnAssign.value` is declared `expr | None` once, in typeshed's
`stdlib/ast.pyi`. NRA reads that declaration instead of keeping a second copy:
the installed mypy package carries typeshed's `stdlib` directory, the one copy
every dev/types install already has. A stub module is decoded under its real
module name (`ast`, `importlib.machinery`), with every `sys.version_info` /
`sys.platform` branch resolved once, here, against the running interpreter.

A stub is declaration-only: its only reader is
`declared_receivers.StubDeclarations`, so no scanned module, finding or
codemod ever holds one. Without mypy the capability is absent and named
(`AbsentStubRoot`), never defaulted.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Generic, TypeVar

from .ast_tools import ParsedModule

STUB_SUFFIX = ".pyi"
PACKAGE_STUB = "__init__"
STUB_PROVIDER = "mypy"
STDLIB_STUBS = Path("typeshed") / "stdlib"

AnswerT = TypeVar("AnswerT")


class StubAnswers(ABC, Generic[AnswerT]):
    """What a reader of the stubs makes of each outcome of looking a name up."""

    @abstractmethod
    def declared_in(self, stub: ParsedModule, qualified: str) -> AnswerT:
        """`stub` is the module a qualified name lives in: its longest stubbed prefix."""

    @abstractmethod
    def undeclared(self, qualified: str) -> AnswerT:
        """No stub module declares the name: it is foreign to the stubs."""

    @abstractmethod
    def unstubbed(self, qualified: str, reason: str) -> AnswerT:
        """The name is the stdlib's, but no typeshed is installed to say what it declares."""


class StubRoot(ABC):
    """Where the stdlib's declarations come from, if anywhere."""

    @abstractmethod
    def answer(self, qualified: str, answers: StubAnswers[AnswerT]) -> AnswerT:
        """The answer for the stub module a qualified name lives in, or for its absence."""


@dataclass(frozen=True)
class InstalledStubRoot(StubRoot):
    directory: Path

    def answer(self, qualified: str, answers: StubAnswers[AnswerT]) -> AnswerT:
        parts = qualified.split(".")
        for end in range(len(parts), 0, -1):
            module_name = ".".join(parts[:end])
            if module_name in self.stub_paths:
                return answers.declared_in(self.decoded(module_name), qualified)
        return answers.undeclared(qualified)

    @cached_property
    def stub_paths(self) -> Mapping[str, Path]:
        """Each stub module's name, `pkg/__init__.pyi` naming `pkg`, and its file."""

        paths: dict[str, Path] = {}
        for path in self.directory.rglob(f"*{STUB_SUFFIX}"):
            parts = path.relative_to(self.directory).with_suffix("").parts
            if parts[-1] == PACKAGE_STUB:
                parts = parts[:-1]
            paths[".".join(parts)] = path
        return paths

    @cache
    def decoded(self, module_name: str) -> ParsedModule:
        """The stub module, its branches resolved for the running interpreter."""

        path = self.stub_paths[module_name]
        source = path.read_text(encoding="utf-8")
        module = RunningInterpreterBranches().visit(ast.parse(source, str(path)))
        return ParsedModule(path=path, module_name=module_name, module=module, source=source)


@dataclass(frozen=True)
class AbsentStubRoot(StubRoot):
    """No typeshed is installed: a stdlib name's members are unknown, and say so."""

    def answer(self, qualified: str, answers: StubAnswers[AnswerT]) -> AnswerT:
        root = qualified.partition(".")[0]
        if root in sys.stdlib_module_names:
            return answers.unstubbed(qualified, f"no typeshed stubs installed for `{root}`")
        return answers.undeclared(qualified)


@cache
def installed_stub_root() -> StubRoot:
    """The typeshed `stdlib` the installed mypy package carries, located without importing it."""

    spec = importlib.util.find_spec(STUB_PROVIDER)
    if spec is None or not spec.submodule_search_locations:
        return AbsentStubRoot()
    directory = Path(next(iter(spec.submodule_search_locations))) / STDLIB_STUBS
    if not directory.is_dir():
        return AbsentStubRoot()
    return InstalledStubRoot(directory.resolve())


# --- branch resolution -------------------------------------------------------------


VersionTest = Callable[[tuple[int, ...], tuple[int, ...]], bool]
VERSION_TESTS: dict[type[ast.cmpop], VersionTest] = {
    ast.Lt: lambda running, bound: running < bound,
    ast.LtE: lambda running, bound: running <= bound,
    ast.Gt: lambda running, bound: running > bound,
    ast.GtE: lambda running, bound: running >= bound,
    ast.Eq: lambda running, bound: running == bound,
    ast.NotEq: lambda running, bound: running != bound,
}
PlatformTest = Callable[[str, str], bool]
PLATFORM_TESTS: dict[type[ast.cmpop], PlatformTest] = {
    ast.Eq: lambda running, named: running == named,
    ast.NotEq: lambda running, named: running != named,
}
RUNNING_VERSION: tuple[int, ...] = (
    sys.version_info.major,
    sys.version_info.minor,
    sys.version_info.micro,
)
EITHER = frozenset((True, False))
"""The outcomes of a test on anything but the running interpreter: both."""


class RunningInterpreterBranches(ast.NodeTransformer):
    """Replaces each `if` a stub conditions on `sys.version_info` or
    `sys.platform` by the branch the running interpreter takes, or by `pass`
    when that branch is empty, so every body stays a body; an `if` on anything
    else stays, both branches visited."""

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        outcomes = self.outcomes(node.test)
        if outcomes == EITHER:
            return self.generic_visit(node)
        chosen: list[ast.stmt] = []
        for statement in node.body if True in outcomes else node.orelse:
            visited = self.visit(statement)
            chosen.extend(visited if isinstance(visited, list) else [visited])
        return chosen or [ast.copy_location(ast.Pass(), node)]

    def outcomes(self, test: ast.expr) -> frozenset[bool]:
        """The values `test` can take on the running interpreter: one when it
        tests only the interpreter, both otherwise."""

        match test:
            case ast.BoolOp(op=ast.And(), values=values):
                combined = frozenset((True,))
                for value in values:
                    combined = frozenset(
                        left and right for left in combined for right in self.outcomes(value)
                    )
                return combined
            case ast.BoolOp(op=ast.Or(), values=values):
                combined = frozenset((False,))
                for value in values:
                    combined = frozenset(
                        left or right for left in combined for right in self.outcomes(value)
                    )
                return combined
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                return frozenset(not outcome for outcome in self.outcomes(operand))
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="sys"), attr="version_info"),
                ops=[operator],
                comparators=[ast.Tuple(elts=items)],
            ) if type(operator) in VERSION_TESTS:
                bound = tuple(
                    item.value
                    for item in items
                    if isinstance(item, ast.Constant) and isinstance(item.value, int)
                )
                if len(bound) == len(items):
                    running = RUNNING_VERSION[: len(bound)]
                    return frozenset((VERSION_TESTS[type(operator)](running, bound),))
            case ast.Compare(
                left=ast.Attribute(value=ast.Name(id="sys"), attr="platform"),
                ops=[operator],
                comparators=[ast.Constant(value=str() as named)],
            ) if type(operator) in PLATFORM_TESTS:
                return frozenset((PLATFORM_TESTS[type(operator)](sys.platform, named),))
            case ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="sys"), attr="platform"),
                    attr="startswith",
                ),
                args=[ast.Constant(value=str() as prefix)],
            ):
                return frozenset((sys.platform.startswith(prefix),))
        return EITHER
