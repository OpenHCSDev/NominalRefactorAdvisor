"""The installed typeshed stdlib stubs, as declarations NRA reads and never edits.

`ast.AnnAssign.value` is declared `expr | None` once, in typeshed's
`stdlib/ast.pyi`. NRA reads that declaration instead of keeping a second copy:
the installed mypy package carries typeshed's `stdlib` directory, the one copy
every dev/types install already has. A stub module is decoded under its real
module name (`ast`, `importlib.machinery`), with every `sys.version_info` /
`sys.platform` branch resolved once, here, against the running interpreter.

A stub is declaration-only: `StubRoot.holds` names its files, and the finding
owner refuses evidence in them. Without mypy the capability is absent and
named (`AbsentStubRoot`), never defaulted.
"""

from __future__ import annotations

import ast
import importlib.util
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .ast_tools import ParsedModule

STUB_SUFFIX = ".pyi"
PACKAGE_STUB = "__init__.pyi"
STUB_PROVIDER = "mypy"
STDLIB_STUBS = Path("typeshed") / "stdlib"


class StubRoot(ABC):
    """Where the stdlib's declarations come from, if anywhere."""

    @abstractmethod
    def declaring_module(self, qualified: str) -> ParsedModule | None:
        """The stub module a qualified name lives in: its longest stubbed prefix."""

    @abstractmethod
    def holds(self, file_path: str) -> bool:
        """Whether `file_path` is a declaration-only stub file."""

    @abstractmethod
    def missing(self, qualified: str) -> str | None:
        """Why this root cannot say what a stdlib name declares, or None when it can."""


@dataclass(frozen=True)
class InstalledStubRoot(StubRoot):
    directory: Path

    def declaring_module(self, qualified: str) -> ParsedModule | None:
        parts = qualified.split(".")
        for end in range(len(parts), 0, -1):
            decoded = self.decoded(".".join(parts[:end]))
            if decoded is not None:
                return decoded
        return None

    @cache
    def decoded(self, module_name: str) -> ParsedModule | None:
        """The stub module, its branches resolved for the running interpreter."""

        located = self.directory.joinpath(*module_name.split("."))
        for path in (located.with_suffix(STUB_SUFFIX), located / PACKAGE_STUB):
            if path.is_file():
                source = path.read_text(encoding="utf-8")
                module = RunningInterpreterBranches().visit(ast.parse(source, str(path)))
                return ParsedModule(path, module_name, module, source)
        return None

    def holds(self, file_path: str) -> bool:
        return file_path.endswith(STUB_SUFFIX) and Path(file_path).resolve().is_relative_to(
            self.directory
        )

    def missing(self, qualified: str) -> str | None:
        return None


@dataclass(frozen=True)
class AbsentStubRoot(StubRoot):
    """No typeshed is installed: a stdlib name's members are unknown, and say so."""

    def declaring_module(self, qualified: str) -> ParsedModule | None:
        return None

    def holds(self, file_path: str) -> bool:
        return False

    def missing(self, qualified: str) -> str | None:
        root = qualified.partition(".")[0]
        if root in sys.stdlib_module_names:
            return f"no typeshed stubs installed for `{root}`"
        return None


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


COMPARISONS: dict[type[ast.cmpop], Callable[[object, object], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class RunningInterpreterBranches(ast.NodeTransformer):
    """Replaces each `if` a stub conditions on `sys.version_info` or
    `sys.platform` by the branch the running interpreter takes, or by `pass`
    when that branch is empty, so every body stays a body; an `if` on anything
    else stays, both branches visited."""

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        taken = self.decided(node.test)
        if taken is None:
            return self.generic_visit(node)
        chosen: list[ast.stmt] = []
        for statement in node.body if taken else node.orelse:
            visited = self.visit(statement)
            chosen.extend(visited if isinstance(visited, list) else [visited])
        return chosen or ast.copy_location(ast.Pass(), node)

    def decided(self, test: ast.expr) -> bool | None:
        match test:
            case ast.BoolOp(op=ast.And(), values=values):
                answers = [self.decided(value) for value in values]
                return None if None in answers else all(answers)
            case ast.BoolOp(op=ast.Or(), values=values):
                answers = [self.decided(value) for value in values]
                return None if None in answers else any(answers)
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                answer = self.decided(operand)
                return None if answer is None else not answer
            case ast.Compare(left=left, ops=[op], comparators=[right]) if type(op) in COMPARISONS:
                subject = self.interpreter_value(left)
                compared = self.literal(right)
                if subject is None or compared is None:
                    return None
                if isinstance(subject, tuple) and isinstance(compared, tuple):
                    subject = subject[: len(compared)]
                return COMPARISONS[type(op)](subject, compared)
            case ast.Call(
                func=ast.Attribute(value=receiver, attr="startswith"),
                args=[ast.Constant(value=str() as prefix)],
            ) if isinstance(platform := self.interpreter_value(receiver), str):
                return platform.startswith(prefix)
        return None

    @staticmethod
    def interpreter_value(node: ast.expr) -> tuple[int, ...] | str | None:
        match node:
            case ast.Attribute(value=ast.Name(id="sys"), attr="version_info"):
                return tuple(sys.version_info)[:3]
            case ast.Attribute(value=ast.Name(id="sys"), attr="platform"):
                return sys.platform
        return None

    @staticmethod
    def literal(node: ast.expr) -> tuple[int, ...] | str | None:
        match node:
            case ast.Constant(value=str() as text):
                return text
            case ast.Tuple(elts=items) if all(
                isinstance(item, ast.Constant) and isinstance(item.value, int)
                for item in items
            ):
                return tuple(item.value for item in items if isinstance(item, ast.Constant))
        return None
