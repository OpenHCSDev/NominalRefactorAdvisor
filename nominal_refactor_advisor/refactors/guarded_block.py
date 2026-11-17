"""A block guarded by a type test becomes the tested class's override (IMPL-3)."""

from __future__ import annotations

import ast
import builtins
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from nominal_refactor_advisor.codemod_payload import (
    RequiredIntegerPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import (
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod
from nominal_refactor_advisor.refactors.capability_guard import (
    CapabilityGuardToOverrideOperation,
)

BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True)
class GuardedBlock:
    """`if isinstance(x, C): <body>` (with an optional `else`), the statement of a
    function at one line. `x` is a parameter, the receiver, or an attribute of the
    receiver; its declared class is the family whose dispatch replaces the test."""

    statement: ast.If
    subject: ast.expr
    capability: ast.expr
    on_class: ClassVar[bool] = False
    """An `isinstance` subject is an instance, never a class."""

    @classmethod
    def at(cls, function: ast.FunctionDef | ast.AsyncFunctionDef, line: int) -> GuardedBlock:
        for node in _own_scope(function):
            match node:
                case ast.If(
                    test=ast.Call(func=ast.Name(id="isinstance"), args=[subject, capability], keywords=[])
                ) if node.lineno == line:
                    if isinstance(capability, ast.Tuple):
                        raise ValueError("the test names several classes: each needs its own override")
                    return cls(node, subject, capability)
        raise ValueError(f"line {line} of `{function.name}` holds no `if isinstance(x, C):` block")

    @property
    def subject_name(self) -> str:
        """The subject's own name when it is a plain name, else empty."""

        return self.subject.id if isinstance(self.subject, ast.Name) else ""

    def occurrences(self, statements) -> tuple[ast.expr, ...]:
        """Every spelling of the subject in the statements: the nodes renamed `self`."""

        return tuple(
            node
            for node in _walk(statements)
            if isinstance(node, ast.expr)
            and isinstance(node, type(self.subject))
            and ast.dump(node, include_attributes=False).replace("Store()", "Load()")
            == ast.dump(self.subject, include_attributes=False)
        )

    def reads(self, statements) -> set[str]:
        """The names the statements read, the subject's own spelling aside."""

        inside = {id(node) for occurrence in self.occurrences(statements) for node in ast.walk(occurrence)}
        return {
            node.id
            for node in _walk(statements)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in inside
        }

    @property
    def root(self) -> str:
        """The function name the subject is spelled through: `x`, or `self` of `self.x`."""

        match self.subject:
            case ast.Name(id=name) | ast.Attribute(value=ast.Name(id=name)):
                return name
        raise ValueError(f"`{ast.unparse(self.subject)}` is neither a name nor an attribute of one")


class GuardedBlockToOverrideOperation(CapabilityGuardToOverrideOperation):
    """Turn `if isinstance(x, C): <body>` into `x.method_name(...)`: `C` overrides
    `method_name` with the body, and `base` declares it as the default, empty or
    holding the `else` block. This is the capability guard's move with a body
    instead of a raise, so it inherits that operation's proof: every class the
    subject's declared type can be must dispatch `method_name` to `C`'s override
    exactly when it is a `C`, and to `base` otherwise (paper1 Thm 4.17: lookup
    winners are fixed by the requirement).

    The block moves only when it can run as a method: it reads the subject, the
    function's parameters, module names and builtins, binds nothing the function
    reads after it, and leaves the function by falling through, never by
    `return`, `yield`, `break` or `continue`. Parameters it reads become the
    method's parameters, passed by name. A block that reads the enclosing
    receiver beyond the subject needs a name for it, and is refused.
    """

    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        qualname = self.target.qualname or ""
        class_name, _, function_name = qualname.rpartition(".")
        module = snapshot.module_nodes_by_file_path[source_path]
        owner = (
            ResolvedClassTarget.from_rewrite_target(
                snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name)
            )
            if class_name
            else None
        )
        body = module.body if owner is None else owner.node.body
        function = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            ),
            None,
        )
        if function is None:
            raise ValueError(f"{qualname!r} is not a function declared in {source_path!r}")
        block = GuardedBlock.at(function, self.line)
        base = ResolvedClassTarget.from_rewrite_target(snapshot, self.base)
        base_answer = AnsweredMethod(snapshot, base, function)
        parents = snapshot.parsed_module_for_source_path(source_path).fact(ModuleNodeIndex).parents
        capability = base_answer.class_symbol(source_path, block.capability, parents)
        if capability is None:
            raise ValueError(f"`{ast.unparse(block.capability)}` is not a class the scan declares")
        subject = self.subject_family(snapshot, base_answer, owner, function, block, parents, source_path)
        self.require_dispatch(snapshot, subject, base.required_symbol(snapshot), capability)
        capability_class = ResolvedClassTarget.from_rewrite_target(
            snapshot, self.class_target(snapshot, capability)
        )
        for answering in (base, capability_class):
            if self.method_name in answering.bound_names:
                raise ValueError(f"`{answering.qualname}` already binds {self.method_name!r}")
        arguments = self.arguments(module, function, block)
        source = snapshot.sources_by_file_path[source_path]
        asynchronous = any(isinstance(node, ast.Await) for node in _walk(block.statement.body))
        if asynchronous and not isinstance(function, ast.AsyncFunctionDef):
            raise ValueError("the block awaits outside a coroutine")
        define = "async def" if asynchronous else "def"
        parameters = "".join(f", {name}" for name in arguments)
        call = (
            f"{'await ' if asynchronous else ''}{ast.unparse(block.subject)}."
            f"{self.method_name}({', '.join(f'{name}={name}' for name in arguments)})"
        )
        override = self.method_source(source, block, block.statement.body)
        default = (
            self.method_source(source, block, block.statement.orelse)
            if block.statement.orelse
            else None
        )
        geometries: dict[str, SourceTextGeometry] = {}
        replacements: dict[str, list[tuple[int, int, str]]] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path]))

        def add(file_path: str, start: int, end: int, text: str) -> None:
            replacements.setdefault(file_path, []).append((start, end, text))

        start, end = geometry(source_path).required_node_offsets(block.statement)
        add(source_path, start, end, call)
        signature = f"{define} {self.method_name}(self{parameters}) -> None:"
        add(base.file_path, *self.appended(
            geometry(base.file_path), base.node,
            f"{signature}\n    \"\"\"What a {base.node.name} does where {function_name} asks; "
            f"{capability_class.node.name} overrides it.\"\"\""
            + (f"\n{textwrap.indent(default, '    ')}" if default else ""),
        ))
        add(capability_class.file_path, *self.appended(
            geometry(capability_class.file_path), capability_class.node,
            f"{signature}\n{textwrap.indent(override, '    ')}",
        ))
        return tuple(
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start, end_offset=end, replacement_source=text
                    )
                    for start, end, text in sorted(spans, reverse=True)
                ),
                rationale=self.rationale
                or f"{capability_class.node.name} answers {self.method_name!r}; the test asks it.",
            )
            for file_path, spans in replacements.items()
        )

    def arguments(
        self,
        module: ast.Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        block: GuardedBlock,
    ) -> tuple[str, ...]:
        """The function's parameters the block reads, in their declared order.
        Every other name it reads must be the subject, bound by the block itself,
        a module name or a builtin."""

        statements = (*block.statement.body, *block.statement.orelse)
        if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) for node in _walk(statements)):
            raise ValueError("the block returns or yields: the call site would need its answer")
        if any(isinstance(node, (ast.Break, ast.Continue)) for node in _walk(statements, loops=False)):
            raise ValueError("the block breaks or continues the function's loop")
        bound = _bound(statements)
        later = {
            node.id
            for node in _own_scope(function)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.lineno > (block.statement.end_lineno or block.statement.lineno)
        }
        if escaping := sorted(bound & later):
            raise ValueError(f"the block binds {escaping!r}, which the function reads after it")
        if block.subject_name in bound or any(
            isinstance(node, (ast.Name, ast.Attribute)) and not isinstance(node.ctx, ast.Load)
            for node in block.occurrences(statements)
        ):
            raise ValueError(f"the block rebinds `{ast.unparse(block.subject)}`")
        read = block.reads(statements)
        if block.root != block.subject_name and block.root in read:
            raise ValueError(
                f"the block reads `{block.root}` beyond `{ast.unparse(block.subject)}`: "
                "it needs a parameter name for the enclosing receiver"
            )
        parameters = [
            argument.arg
            for argument in (
                *function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs,
                *(argument for argument in (function.args.vararg, function.args.kwarg) if argument),
            )
        ]
        free = read - bound - _module_names(module) - BUILTIN_NAMES - {block.subject_name}
        if locals_read := sorted(free - set(parameters)):
            raise ValueError(f"the block reads the function's locals {locals_read!r}")
        if "self" in free and block.subject_name != "self":
            raise ValueError("the block reads the enclosing receiver: it needs a parameter name for it")
        return tuple(name for name in parameters if name in free)

    @staticmethod
    def method_source(source: str, block: GuardedBlock, statements: list[ast.stmt]) -> str:
        """The statements' own text, dedented, with the subject spelled `self`."""

        lines = source.splitlines(keepends=True)
        first, last = statements[0], statements[-1]
        start_line, end_line = first.lineno, last.end_lineno or last.lineno
        indent = first.col_offset
        renames = sorted(
            (
                (node.lineno, node.col_offset, node.end_col_offset or node.col_offset)
                for node in block.occurrences(statements)
            ),
            reverse=True,
        )
        text = [lines[number - 1] for number in range(start_line, end_line + 1)]
        for lineno, col, end_col in renames:
            row = text[lineno - start_line]
            encoded = row.encode()
            text[lineno - start_line] = (encoded[:col] + b"self" + encoded[end_col:]).decode()
        return "".join(line[indent:] if line.strip() else line.lstrip(" ") for line in text).rstrip("\n")


def _own_scope(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """The function's nodes, without the bodies of the functions, lambdas and
    classes it declares."""

    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _walk(statements, *, loops: bool = True) -> Iterator[ast.AST]:
    """The statements' nodes in their own scope; without `loops`, also without
    the bodies of loops the statements themselves open."""

    pending: list[ast.AST] = list(statements)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if not loops and isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            pending.extend(node.orelse)
            continue
        pending.extend(ast.iter_child_nodes(node))


def _bound(statements) -> set[str]:
    return {
        node.id for node in _walk(statements) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }


def _module_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for statement in module.body:
        match statement:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names |= {(alias.asname or alias.name).split(".")[0] for alias in aliases}
            case _:
                names |= _bound([statement])
    return names


__all__ = ("GuardedBlock", "GuardedBlockToOverrideOperation")
