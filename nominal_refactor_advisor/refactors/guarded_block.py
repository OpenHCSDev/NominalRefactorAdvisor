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
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import AnsweredMethod
from nominal_refactor_advisor.refactors.capability_guard import (
    CapabilityGuardToOverrideOperation,
)
from nominal_refactor_advisor.refactors.member_move import carried_imports

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
        """Every spelling of the subject in the statements, closures included: the nodes
        renamed `self`."""

        return tuple(
            node
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.expr)
            and isinstance(node, type(self.subject))
            and ast.dump(node, include_attributes=False).replace("Store()", "Load()")
            == ast.dump(self.subject, include_attributes=False)
        )

    def reads(self, statements) -> set[str]:
        """The names the statements read from the function, the subject's own spelling aside."""

        return _free_reads(statements, frozenset(id(node) for node in self.occurrences(statements)))

    @property
    def root(self) -> str:
        """The function name the subject is spelled through: `x`, or `self` of `self.x`."""

        match self.subject:
            case ast.Name(id=name) | ast.Attribute(value=ast.Name(id=name)):
                return name
        raise ValueError(f"`{ast.unparse(self.subject)}` is neither a name nor an attribute of one")


@dataclass(frozen=True, kw_only=True)
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
    receiver beyond the subject needs a name for it, and is refused. A body that
    lands in another module brings the module names it reads, imported the way
    the function's module binds them (move_member's `carried_imports`).
    """

    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    @classmethod
    def at_site(
        cls,
        snapshot: CodemodSourceSnapshot,
        target: SourceRewriteTarget,
        line: int,
        method_name: str,
    ) -> GuardedBlockToOverrideOperation:
        """The operation for the block at `line` of `target`, with `base` derived:
        the class the subject is declared as, so no pick restates it."""

        source_path = target.file_path or ""
        function, owner = cls.function_at(snapshot, target)
        block = GuardedBlock.at(function, line)
        classes = snapshot.required_class_family_index.classes_by_symbol
        reader = owner or next(
            (
                ResolvedClassTarget.from_rewrite_target(
                    snapshot, SourceRewriteTarget(file_path=source_path, qualname=declaration.qualname)
                )
                for declaration in classes.values()
                if declaration.file_path == source_path
            ),
            None,
        )
        if reader is None:
            raise ValueError(f"{source_path!r} declares no class to read its bindings through: it needs a base pick")
        parents = snapshot.parsed_module_for_source_path(source_path).fact(ModuleNodeIndex).parents
        declared = cls.declared_subject(
            snapshot, AnsweredMethod(snapshot, reader, function), owner, function, block, parents, source_path
        )
        return cls(
            target=target,
            base=cls.class_target(snapshot, declared),
            method_name=method_name,
            line=line,
        )

    @staticmethod
    def function_at(
        snapshot: CodemodSourceSnapshot, target: SourceRewriteTarget
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ResolvedClassTarget | None]:
        """The target function and the class it is a method of, if any."""

        source_path = target.file_path or ""
        class_name, _, function_name = (target.qualname or "").rpartition(".")
        owner = (
            ResolvedClassTarget.from_rewrite_target(
                snapshot, SourceRewriteTarget(file_path=source_path, qualname=class_name)
            )
            if class_name
            else None
        )
        body = snapshot.module_nodes_by_file_path[source_path].body if owner is None else owner.node.body
        function = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
            ),
            None,
        )
        if function is None:
            raise ValueError(f"{target.qualname!r} is not a function declared in {source_path!r}")
        return function, owner

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        source_path = self.required_source_path(snapshot, self.operation_key())
        function, owner = self.function_at(snapshot, self.target)
        function_name = function.name
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
        arguments = self.arguments(function, block)
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
        imports = tuple(
            mutation
            for destination, statements in (
                (capability_class.file_path, block.statement.body),
                (base.file_path, block.statement.orelse),
            )
            if destination != source_path and statements
            for mutation in carried_imports(
                snapshot, source_path, destination,
                self.module_reads(function, block, statements),
                self.module_reads(function, block, statements),
                f"{function_name}'s block",
            )
        )
        return imports + tuple(
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
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        block: GuardedBlock,
    ) -> tuple[str, ...]:
        """The function's parameters the block reads, in their declared order.
        Every other name it reads must be the subject, bound by the block alone,
        a module name or a builtin."""

        statements = (*block.statement.body, *block.statement.orelse)
        if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) for node in _walk(statements)):
            raise ValueError("the block returns or yields: the call site would need its answer")
        if any(isinstance(node, (ast.Break, ast.Continue)) for node in _walk(statements, loops=False)):
            raise ValueError("the block breaks or continues the function's loop")
        if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in _walk(statements)):
            raise ValueError("the block declares a global or nonlocal name")
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(statements)
        inside = {id(node) for statement in statements for node in ast.walk(statement)}
        outside = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and id(node) not in inside
        }
        if block.root in bound | _nested_bindings(statements) or any(
            not isinstance(node.ctx, ast.Load) for node in block.occurrences(statements)
        ):
            raise ValueError(f"the block rebinds `{block.root}`")
        if escaping := sorted(bound & outside):
            raise ValueError(f"the block binds {escaping!r}, which the function also uses outside it")
        read = block.reads(statements)
        if read & {"super", "__class__"}:
            raise ValueError("the block calls super(): its class would change")
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
        free = read - bound - {block.subject_name}
        if "self" in free and block.subject_name != "self" and "self" in parameters:
            raise ValueError("the block reads the enclosing receiver: it needs a parameter name for it")
        if locals_read := sorted(free & (LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body) - bound)):
            raise ValueError(f"the block reads the function's locals {locals_read!r}")
        return tuple(name for name in parameters if name in free)

    def module_reads(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        block: GuardedBlock,
        statements: list[ast.stmt],
    ) -> frozenset[str]:
        """The module names the statements read: what their new module must bind the same way."""

        own = (
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
            | {argument.arg for argument in ast.walk(function.args) if isinstance(argument, ast.arg)}
        )
        return frozenset(block.reads(statements) - own - BUILTIN_NAMES - {block.subject_name})

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


def _free_reads(nodes, skipped: frozenset[int] = frozenset()) -> set[str]:
    """The names the nodes read from the scope around them: their own loads and
    augmented targets, and what their functions, lambdas and comprehensions read
    from outside themselves. Nodes in `skipped` are left out with their subtrees."""

    reads: set[str] = set()
    pending: list[ast.AST] = list(nodes)
    while pending:
        node = pending.pop()
        if id(node) in skipped:
            continue
        match node:
            case ast.ClassDef():
                raise ValueError("the block declares a class")
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.Lambda():
                arguments = node.args
                body = node.body if isinstance(node.body, list) else [node.body]
                pending.extend((*arguments.defaults, *(each for each in arguments.kw_defaults if each)))
                if not isinstance(node, ast.Lambda):
                    pending.extend(node.decorator_list)
                own = {argument.arg for argument in ast.walk(arguments) if isinstance(argument, ast.arg)}
                reads |= _free_reads(body, skipped) - own - LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(body)
                continue
            case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                targets = {
                    name.id for generator in node.generators for name in ast.walk(generator.target)
                    if isinstance(name, ast.Name)
                }
                inner = [child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.comprehension)]
                inner.extend(part for generator in node.generators for part in (generator.iter, *generator.ifs))
                reads |= _free_reads(inner, skipped) - targets
                continue
            case ast.Name(id=name, ctx=ast.Load()) | ast.AugAssign(target=ast.Name(id=name)):
                reads.add(name)
        pending.extend(ast.iter_child_nodes(node))
    return reads


def _nested_bindings(statements) -> set[str]:
    """The names the functions, lambdas and comprehensions inside the statements bind."""

    names: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            match node:
                case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.Lambda():
                    names |= {argument.arg for argument in ast.walk(node.args) if isinstance(argument, ast.arg)}
                    body = node.body if isinstance(node.body, list) else [node.body]
                    names |= LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(body)
                case ast.comprehension(target=target):
                    names |= {name.id for name in ast.walk(target) if isinstance(name, ast.Name)}
    return names


__all__ = ("GuardedBlock", "GuardedBlockToOverrideOperation")
