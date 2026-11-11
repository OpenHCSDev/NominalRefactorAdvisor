"""A method of a class family and the calls it answers, each receiver typed by its binding, the declarations or the type checker."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.lexical_bindings import ImportBoundNameProjection
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.probe_owner import (
    InferredReceiverType,
    ProbeOwnerResolver,
    ReceiverSpan,
)
from nominal_refactor_advisor.type_checker_session import (
    SourceTexts,
    TypeCheckerSession,
)


@dataclass(frozen=True)
class AnsweredMethod:
    """A method of a class family, and the calls it answers: every call spelled on
    a receiver of the family, each typed by the class binding, the declarations
    or the type checker. A call on another class is that class's own method; a
    call this proof cannot type refuses."""

    snapshot: CodemodSourceSnapshot
    source_class: ResolvedClassTarget
    method: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def name(self) -> str:
        return self.method.name

    @cached_property
    def symbol(self) -> str:
        return self.source_class.required_symbol(self.snapshot)

    @cached_property
    def descendants(self) -> frozenset[str]:
        return frozenset(self.snapshot.required_class_family_index.descendant_symbols(self.symbol))

    @cached_property
    def family(self) -> frozenset[str]:
        """Every class whose instances answer the method by inheriting it."""

        return self.descendants | {self.symbol}

    @cached_property
    def family_declarations(self) -> dict[tuple[str, str], str]:
        """Each family class by the file and qualname that declare it."""

        classes = self.snapshot.required_class_family_index.classes_by_symbol
        return {
            (classes[symbol].file_path, classes[symbol].qualname): symbol
            for symbol in self.family if symbol in classes
        }

    @cached_property
    def overriding(self) -> tuple[str, ...]:
        """The qualnames of descendants that declare the method again."""

        classes = self.snapshot.required_class_family_index.classes_by_symbol
        return tuple(
            classes[symbol].qualname
            for symbol in self.descendants
            if symbol in classes and self.name in _class_node(self.snapshot, symbol).bound_names_of_body
        )

    @cached_property
    def overridden(self) -> tuple[str, ...]:
        """What the method itself overrides (the class index's answer)."""

        return self.snapshot.required_class_family_index.overridden_declarations(self.symbol, self.name)

    def require_sole_dispatch(self, verb: str) -> None:
        """Refuse when the method takes part in dispatch: a descendant overrides it, or
        it overrides an ancestor. Either way a receiver this proof cannot see reaches it
        through the MRO, so removing it from this class changes what that call does."""

        if self.overriding:
            raise ValueError(f"{self.overriding!r} override {self.name!r}; {verb} breaks their dispatch")
        if self.overridden:
            raise ValueError(
                f"{self.symbol}.{self.name} overrides {self.overridden!r}; {verb} sends every "
                "receiver typed as the base to the base's answer"
            )

    def named_class(self, file_path: str, name: str, scopes: tuple[ast.AST, ...]) -> ClassBinding | None:
        """What a name binds where a call reads it, when that says whether it is the family: a
        family class, another class the scan declares, a module, or an import from outside the
        scan. The innermost of `scopes` (the call's enclosing functions, then its module) that
        binds the name decides; a name it binds any other way is typed as an instance."""

        for scope in scopes:
            binding = _ScopeBindings(scope).binding(name)
            if binding is None:
                continue
            return binding.class_binding(self, file_path, scope)
        return None

    def declared_class(self, file_path: str, qualname: str) -> ClassBinding | None:
        """A class the scan declares at this file and qualname: the family's, or another's."""

        family = self.family_declarations.get((file_path, qualname))
        if family is not None:
            return NamedClass(family)
        other = self.class_declarations.get((file_path, qualname))
        return None if other is None else OtherClass(other)

    @cached_property
    def class_declarations(self) -> dict[tuple[str, str], str]:
        classes = self.snapshot.required_class_family_index.classes_by_symbol
        return {(declaration.file_path, declaration.qualname): symbol for symbol, declaration in classes.items()}

    def answers(self, symbols: tuple[str, ...], where: str) -> bool:
        """Whether a receiver of these classes calls the method: all of the family, or
        none of it (another class's own method of that name)."""

        inside = tuple(symbol for symbol in symbols if symbol in self.family)
        if symbols and len(inside) == len(symbols):
            return True
        if symbols and not inside:
            return False
        raise ValueError(f"{where} calls {self.name!r} on a receiver typed {symbols!r}")

    def bound_class(
        self, receiver: ast.expr, file_path: str, enclosing: str | None, scopes: tuple[ast.AST, ...],
    ) -> ClassBinding | None:
        """The class a call receiver names by its binding, when it names one. A plain
        method's calls are spelled on instances, so only a bare name can name one."""

        if not isinstance(receiver, ast.Name):
            return None
        return self.named_class(file_path, receiver.id, scopes)

    def admits(self, binding: ReceiverBinding, where: str) -> None:
        """Refuse a call whose receiver would change what the method means."""

    def answering_calls(self) -> tuple[MethodCall, ...]:
        """The calls the method answers, each receiver typed by the class binding, the
        declarations or the checker."""

        snapshot = self.snapshot
        typing = snapshot.declared_receiver_typing
        answering: list[MethodCall] = []
        undeclared: dict[ReceiverSpan, MethodReference] = {}
        for parsed in snapshot.parsed_modules:
            if self.name not in parsed.source:
                continue
            parents = parsed.fact(ModuleNodeIndex).parents
            resolver = ProbeOwnerResolver.of(parsed)
            for node in ast.walk(parsed.module):
                where = f"{parsed.file_path}:{getattr(node, 'lineno', 0)}"
                if _dynamic_access(node, self.name):
                    raise ValueError(f"{where} reaches {self.name!r} by a string")
                if node is self.method or not (
                    isinstance(node, ast.Attribute) and node.attr == self.name
                ):
                    continue
                parent = parents.get(node)
                site = MethodReference(
                    parsed.file_path, node,
                    parent if isinstance(parent, ast.Call) and parent.func is node else None,
                )
                enclosing = _enclosing_class(snapshot, parsed.file_path, node, parents)
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (
                    node.value.func.id == "super"
                ):
                    if enclosing in self.family:
                        raise ValueError(f"{where} reaches {self.name!r} through super()")
                    continue
                bound = self.bound_class(
                    node.value, parsed.file_path, enclosing, _name_scopes(node, parents),
                )
                if bound is not None:
                    if bound.reaches_family():
                        answering.append(site.answered(self, bound))
                    continue
                owner = resolver.owner(node.value).resolved(typing)
                spans = owner.type_checker_spans()
                if spans:
                    undeclared[spans[0]] = site
                elif self.answers(tuple(getattr(owner, "class_symbols", tuple)()), where):
                    answering.append(site.answered(self, ReceiverBinding(self.symbol)))
        if undeclared:
            with TypeCheckerSession.session(SourceTexts(snapshot.parsed_modules)) as session:
                answers = session.receiver_types(undeclared)
            for span, site in undeclared.items():
                answer = answers.get(span)
                if not isinstance(answer, InferredReceiverType):
                    raise ValueError(f"{site.where}: the type checker gives no type for the receiver")
                if self.answers(answer.classes, site.where):
                    answering.append(site.answered(self, ReceiverBinding(self.symbol)))
        nested = [
            site for site in answering
            for other in answering
            if other is not site and other.call in ast.walk(site.call)
        ]
        if nested:
            raise ValueError(f"{nested[0].where} nests one answered call inside another")
        return tuple(answering)


@dataclass(frozen=True)
class ReceiverBinding:
    """What a call's receiver says about the class the moved method runs on."""

    symbol: str
    describes: ClassVar[str] = "on an instance"

    def names(self, symbol: str) -> bool:
        """Whether the receiver fixes the runtime class to exactly this one."""

        return False


@dataclass(frozen=True)
class ClassBinding(ReceiverBinding):
    """A receiver that is a class rather than an instance of one."""

    def reaches_family(self) -> bool:
        return True


@dataclass(frozen=True)
class ForeignClass(ClassBinding):
    """A class that is not the family's: its attribute of the method's name is its own."""

    def reaches_family(self) -> bool:
        return False


@dataclass(frozen=True)
class OutsideScan(ForeignClass):
    """A name imported from outside the scan (the standard library, a dependency)."""


@dataclass(frozen=True)
class OtherClass(ForeignClass):
    """Another class the scan declares."""


@dataclass(frozen=True)
class NamedClass(ClassBinding):
    """The class spelled by its binding: the runtime class is exactly it."""

    describes: ClassVar[str] = "on a class named outright"

    def names(self, symbol: str) -> bool:
        return self.symbol == symbol


@dataclass(frozen=True)
class _ClassBody:
    bound_names_of_body: frozenset[str]


def _class_node(snapshot: CodemodSourceSnapshot, symbol: str) -> _ClassBody:
    declaration = snapshot.required_class_family_index.classes_by_symbol[symbol]
    scope: ast.AST = snapshot.module_nodes_by_file_path[declaration.file_path]
    for part in declaration.qualname.split("."):
        scope = next(
            statement for statement in scope.body  # type: ignore[attr-defined]
            if isinstance(statement, ast.ClassDef) and statement.name == part
        )
    assert isinstance(scope, ast.ClassDef)
    return _ClassBody(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(scope.body))


@dataclass(frozen=True)
class OtherModule(ForeignClass):
    """A module bound by `import`: its attribute of the method's name is its own."""


@dataclass(frozen=True)
class NameBinding:
    """How one scope binds a name: by default as a value, typed as an instance."""

    def class_binding(self, moved: AnsweredMethod, file_path: str, scope: ast.AST) -> ClassBinding | None:
        return None


@dataclass(frozen=True)
class ImportedName(NameBinding):
    """`from module import name`: the class it resolves to, or outside the scan."""

    statement: ast.ImportFrom
    alias: ast.alias

    def class_binding(self, moved: AnsweredMethod, file_path: str, scope: ast.AST) -> ClassBinding | None:
        graph = moved.snapshot.module_import_graph
        source_file = graph.source_file_for_path(file_path)
        if source_file is None:
            return None
        module = source_file.module_path_identity.resolve_import_from_module(
            imported_module=self.statement.module, level=self.statement.level,
        )
        provider = None if module is None else graph.unique_source_file_for_module_name(module)
        if provider is None:
            return OutsideScan(str(module))
        return moved.declared_class(provider.file_path, self.alias.name)


@dataclass(frozen=True)
class ImportedModule(NameBinding):
    """`import module`: a module, never a family class."""

    alias: ast.alias

    def class_binding(self, moved: AnsweredMethod, file_path: str, scope: ast.AST) -> ClassBinding | None:
        return OtherModule(self.alias.name)


@dataclass(frozen=True)
class DeclaredClass(NameBinding):
    """`class Name:` in the scope: the class the index declares there, when it is top-level."""

    node: ast.ClassDef

    def class_binding(self, moved: AnsweredMethod, file_path: str, scope: ast.AST) -> ClassBinding | None:
        return moved.declared_class(file_path, self.node.name) if isinstance(scope, ast.Module) else None


_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class _ScopeBindings:
    """The names one scope binds itself, without the scopes nested in it."""

    scope: ast.AST

    def nodes(self) -> list[ast.AST]:
        found: list[ast.AST] = []
        pending = list(ast.iter_child_nodes(self.scope))
        while pending:
            node = pending.pop()
            found.append(node)
            if not isinstance(node, _NESTED_SCOPES):
                pending.extend(ast.iter_child_nodes(node))
        return found

    def binding(self, name: str) -> NameBinding | None:
        last: NameBinding | None = None
        for node in self.nodes():
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if ImportBoundNameProjection(node).alias_bound_name(alias) == name:
                        last = ImportedName(node, alias)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if ImportBoundNameProjection(node).alias_bound_name(alias) == name:
                        last = ImportedModule(alias)
            elif isinstance(node, ast.ClassDef) and node.name == name:
                last = DeclaredClass(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                last = NameBinding()
            elif isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
                last = NameBinding()
            elif isinstance(node, ast.arg) and node.arg == name:
                last = NameBinding()
        return last


def _name_scopes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[ast.AST, ...]:
    """The scopes a name read at `node` resolves through: enclosing functions innermost first,
    a class body only when the read sits directly in it, then the module."""

    scopes: list[ast.AST] = []
    current = parents.get(node)
    direct = True
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scopes.append(current)
            direct = False
        elif isinstance(current, ast.ClassDef):
            if direct:
                scopes.append(current)
            direct = False
        elif isinstance(current, ast.Module):
            scopes.append(current)
        current = parents.get(current)
    return tuple(scopes)


def _enclosing_class(
    snapshot: CodemodSourceSnapshot, file_path: str, node: ast.AST, parents: dict[ast.AST, ast.AST],
) -> str | None:
    """The symbol of the class whose body a node sits in, when the index knows it."""

    names: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            names.append(current.name)
        current = parents.get(current)
    qualname = ".".join(reversed(names))
    return next(
        (
            symbol for symbol, declaration in snapshot.required_class_family_index.classes_by_symbol.items()
            if names and declaration.file_path == file_path and declaration.qualname == qualname
        ),
        None,
    )


@dataclass(frozen=True)
class MethodReference:
    """One attribute reference by the method's name, called or not, before its receiver is typed."""

    file_path: str
    attribute: ast.Attribute
    call: ast.Call | None

    @property
    def where(self) -> str:
        return f"{self.file_path}:{self.attribute.lineno}"

    def answered(self, moved: AnsweredMethod, binding: ReceiverBinding) -> MethodCall:
        """The reference as a call the method answers: one that is not a call would lose
        the method, and the method may refuse the binding."""

        if self.call is None:
            raise ValueError(f"{self.where} uses {moved.name!r} other than by calling it")
        moved.admits(binding, self.where)
        return MethodCall(self.file_path, self.call, self.attribute)


@dataclass(frozen=True)
class MethodCall:
    """One call the moved method answers, and the receiver it is called on."""

    file_path: str
    call: ast.Call
    attribute: ast.Attribute

    @property
    def where(self) -> str:
        return f"{self.file_path}:{self.call.lineno}"


_DYNAMIC_ACCESS = frozenset({"getattr", "hasattr", "setattr", "delattr"})


def _dynamic_access(node: ast.AST, name: str) -> bool:
    """Whether a node reaches an attribute of this name by a string, past any receiver typing."""

    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_ACCESS
        and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and node.args[1].value == name
    )
