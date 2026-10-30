"""Measures read through the module's own typing bindings: raw string subscripts, written
``typing.Any`` and names imported only for the type checker."""
from __future__ import annotations

import ast
import typing
from dataclasses import dataclass
from typing import Annotated

from metaclass_registry.mro_dispatch import MroDispatch, handles

from .measures import Headline, WholeModule
from .project import Project
from .gate import LocatedSource, OccurrenceMeasure, PerFileMeasure
from .typing_names import TypingImports, TypingReference

type Enclosing = tuple[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...]


class StringSubscript(WholeModule, OccurrenceMeasure, Headline):
    """A container read by a literal string key, ``x["key"]``. A subscript of a typing form,
    such as ``Literal["a"]``, is a type, not a read: the module's typing bindings decide which,
    so the census reads the whole module too."""

    weight = 1

    @classmethod
    def module_lines(cls, tree: ast.Module) -> list[int]:
        imports = TypingImports.of(tree)
        found = []
        for node in ast.walk(tree):
            match node:
                case ast.Subscript(value=container, slice=ast.Constant(value=str())):
                    reference = TypingReference(imports)
                    reference.dispatch_sync(container)
                    found.extend([cls.position.lines(node).start] * reference.raw_access_count())
        return found

    @classmethod
    def lines(cls, repo: Project, source: bytes, filename: str) -> list[int]:
        return cls.module_lines(cls.tree(source, filename))

    @classmethod
    def module_occurrences(cls, module: ast.Module) -> int:
        return len(cls.module_lines(module))


class AnyReference(TypingReference):
    """A reference AnyName asks one question of: does it resolve to ``typing.Any``."""

    @property
    def is_any(self) -> bool:
        return self.value is typing.Any


class AnyInTypeExpression(MroDispatch):
    """Count ``typing.Any`` in one type expression without evaluating it.

    Only type-forming syntax is followed: a subscript, a union, a member list
    and, in an annotation, a quoted forward reference. A call or a comparison
    is a runtime value, so ``annotation is Any`` in introspection code is not a use.
    """

    def __init__(self, imports: dict[str, object], quoted: bool) -> None:
        self.imports = imports
        self.quoted = quoted
        self.references: list[ast.expr] = []

    @property
    def found(self) -> int:
        return len(self.references)

    def resolved(self, node: ast.AST) -> object:
        reference = TypingReference(self.imports)
        reference.dispatch_sync(node)
        return reference.value

    @handles(ast.Name, ast.Attribute)
    def reference(self, node: ast.expr) -> None:
        reference = AnyReference(self.imports)
        reference.dispatch_sync(node)
        if reference.is_any:
            self.references.append(node)

    @handles(ast.Subscript)
    def parameterised(self, node: ast.Subscript) -> None:
        origin = self.resolved(node.value)
        members = [node.slice]
        if origin is typing.Literal:
            members = []  # Literal members are values, not types.
        if origin is Annotated:
            match node.slice:  # The rest is metadata.
                case ast.Tuple(elts=[declared, *_]):
                    members = [declared]
        for member in members:
            self.dispatch_sync(member)

    @handles(ast.Tuple, ast.List)
    def members(self, node: ast.Tuple | ast.List) -> None:
        for member in node.elts:
            self.dispatch_sync(member)

    @handles(ast.BinOp)
    def union(self, node: ast.BinOp) -> None:
        self.dispatch_sync(node.left)
        self.dispatch_sync(node.right)

    @handles(ast.Constant)
    def forward_reference(self, node: ast.Constant) -> None:
        if not self.quoted or not isinstance(node.value, str):
            return
        try:
            quoted = ast.parse(node.value.strip(), mode="eval")
        except (SyntaxError, ValueError):
            return
        self.dispatch_sync(quoted.body)


@dataclass(frozen=True)
class AnyUse:
    """One written ``typing.Any`` and the declaration or reference owning it."""

    line: int
    site: str
    owner: str


class DeclaredTypeSites(MroDispatch):
    """Written Any references, with declaration-owned quoted type parsing."""

    def __init__(self, imports: dict[str, object]) -> None:
        self.imports = imports
        self.uses: list[AnyUse] = []
        self.counted: set[ast.AST] = set()

    def read(self, tree: ast.Module) -> None:
        pending: list[tuple[ast.AST, Enclosing]] = [(tree, ())]
        while pending:
            node, scope = pending.pop()
            self.dispatch_sync(node, scope)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = (*scope, node)
            pending.extend((child, scope) for child in ast.iter_child_nodes(node))

    def declare(self, site: str, expression: ast.expr | None, scope: Enclosing,
                quoted: bool = True) -> None:
        if expression is None:
            return
        counted = AnyInTypeExpression(self.imports, quoted)
        counted.dispatch_sync(expression)
        owner = ".".join(enclosing.name for enclosing in scope)
        self.counted.update(counted.references)
        self.uses.extend([AnyUse(expression.lineno, site, owner)] * counted.found)

    @handles(ast.Name, ast.Attribute)
    def reference(self, node: ast.Name | ast.Attribute, scope: Enclosing) -> None:
        if not isinstance(node.ctx, ast.Load) or node in self.counted:
            return
        reference = AnyReference(self.imports)
        reference.dispatch_sync(node)
        if reference.is_any:
            self.counted.add(node)
            owner = ".".join(enclosing.name for enclosing in scope)
            self.uses.append(AnyUse(node.lineno, "reference", owner))

    @staticmethod
    def record_class(owner: ast.ClassDef) -> bool:
        """Whether the class's annotations are its fields."""
        def last_name(expression: ast.AST) -> str:
            match expression:
                case ast.Call(func=called):
                    expression = called
            return ast.unparse(expression).rpartition(".")[2]
        decorators = {last_name(decorator) for decorator in owner.decorator_list}
        bases = {last_name(base) for base in owner.bases}
        return "dataclass" in decorators or bool(bases & {"NamedTuple", "TypedDict"})

    def assignment_site(self, scope: Enclosing) -> str:
        match scope:
            case (*_, ast.ClassDef() as owner):
                return "dataclass field" if self.record_class(owner) else "class attribute"
        return "variable"

    @handles(ast.arg)
    def parameter(self, node: ast.arg, scope: Enclosing) -> None:
        self.declare("parameter", node.annotation, scope)

    @handles(ast.FunctionDef, ast.AsyncFunctionDef)
    def result(self, node, scope: Enclosing) -> None:
        self.declare("return", node.returns, (*scope, node))

    @handles(ast.AnnAssign)
    def annotated(self, node: ast.AnnAssign, scope: Enclosing) -> None:
        self.declare(self.assignment_site(scope), node.annotation, scope)
        alias = AnyInTypeExpression(self.imports, quoted=True).resolved(node.annotation)
        if alias is typing.TypeAlias:
            self.declare("alias", node.value, scope)

    @handles(ast.Assign)
    def alias(self, node: ast.Assign, scope: Enclosing) -> None:
        # ``Payload = dict[str, Any]`` names a type. A bare member list such as
        # ``(Any, object)`` is a runtime collection, and runtime strings are not
        # forward references.
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            self.declare("alias", node.value, scope, quoted=False)

    @handles(ast.TypeAlias)
    def alias_statement(self, node: ast.TypeAlias, scope: Enclosing) -> None:
        self.declare("alias", node.value, scope)


class AnyName(PerFileMeasure, LocatedSource):
    """Every written Any, including quoted declarations and loaded runtime references."""

    @classmethod
    def uses(cls, source: bytes, filename: str) -> tuple[AnyUse, ...]:
        tree = cls.tree(source, filename)
        sites = DeclaredTypeSites(TypingImports.at_any_depth(tree))
        sites.read(tree)
        return tuple(sorted(sites.uses, key=lambda use: (use.line, use.site, use.owner)))

    @classmethod
    def lines(cls, repo: Project, source: bytes, filename: str) -> list[int]:
        return [use.line for use in cls.uses(source, filename)]


class TypeCheckingImport(PerFileMeasure, LocatedSource):
    """Names imported only while a type checker reads the module: under ``if TYPE_CHECKING:``,
    or in the ``else`` of ``if not TYPE_CHECKING:``. Such a name exists for the checker and not
    at runtime, so the module's real dependency is hidden from every reader that runs it; the
    usual cause is an import cycle the module structure should remove.

    The guard is resolved through the module's own typing bindings (``TYPE_CHECKING``,
    ``typing.TYPE_CHECKING`` and any alias of either), never by spelling. Every guarded name
    counts, a stub-only module such as typeshed's ``_typeshed`` included: the count is the
    source's alone, never the installed packages', and a name that needs the guard is still a
    dependency hidden from the running module. Each guarded name counts once at its import's
    line; growth is refused per file, and the existing guarded imports are measured, not
    remediated."""

    @classmethod
    def lines(cls, repo: Project, source: bytes, filename: str) -> list[int]:
        tree = cls.tree(source, filename)
        imports = TypingImports.at_any_depth(tree)
        guarded: dict[int, ast.Import | ast.ImportFrom] = {}
        for node in ast.walk(tree):
            match node:
                case ast.If(test=test, body=body, orelse=orelse):
                    for statement in cls.checker_branch(test, body, orelse, imports):
                        guarded.update((id(inner), inner) for inner in ast.walk(statement)
                                       if isinstance(inner, (ast.Import, ast.ImportFrom)))
        return sorted(statement.lineno for statement in guarded.values()
                      for _ in statement.names)

    @classmethod
    def checker_branch(cls, test: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt],
                       imports: dict[str, object]) -> list[ast.stmt]:
        """The statements of an ``if`` that only a type checker reads."""
        match test:
            case ast.UnaryOp(op=ast.Not(), operand=operand) if cls.type_checking(operand, imports):
                return orelse
        return body if cls.type_checking(test, imports) else []

    @staticmethod
    def type_checking(test: ast.expr, imports: dict[str, object]) -> bool:
        match test:
            case ast.Name() | ast.Attribute():
                reference = TypingReference(imports)
                reference.dispatch_sync(test)
                return reference.is_type_checking
        return False
