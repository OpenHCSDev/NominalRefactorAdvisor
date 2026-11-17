"""What a class's creation reads when a new plain owner joins its bases.

Composing a new owner `M` adds it to a class's bases and moves the owner's
declarations out of the class body. Class keywords never change type's C3 order;
what such a move can change is what the class's creation code reads: its
metaclass chain (`__new__`, `__init__`, `__prepare__`, and the hooks a modelled
metaclass calls) and every `__init_subclass__` in its ancestry. The difference
those functions could see is exactly three facts: `bases` gains `M`; the
namespace loses the moved names; and lookups on the new class can reach the
attributes `M` binds itself.

A metaclass NRA's own dependencies define is modelled (`type`, `ABCMeta`,
`metaclass_registry.AutoRegisterMeta`): it reads registry names and abstract
methods, so the move is invisible to it when no moved name is a registry name and
no moved statement is abstract. Scanned creation code is read: the namespace only
by constant keys that are not moved names; `bases` only by truthiness, by passing
it to `super()`, or by dispatching each base on its metaclass where a plain class
gets a constant; the new class never introspected. Anything else names the exact
expression it could not prove.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import NoReturn
from importlib import import_module

from metaclass_registry.core import AutoRegisterMeta

from nominal_refactor_advisor.class_index import ClassMroAuthority, CompactIndexedClass
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput

MODELLED_METACLASSES = frozenset(
    ("builtins.type", "abc.ABCMeta", "metaclass_registry.core.AutoRegisterMeta")
)
"""Metaclasses whose class-creation reads NRA knows: they read registry names and
abstract methods, never a declaration's other names or a plain base."""

REGISTRY_NAMES = frozenset(
    (
        "__abstractmethods__",
        "__key_extractor__",
        "__registry__",
        "__registry_config__",
        "__registry_dict__",
        "__registry_family__",
        "__registry_key__",
        "__registry_name__",
        "__replaces_registered__",
        "__secondary_registries__",
        "__skip_if_no_key__",
    )
)
"""The names a modelled metaclass reads from a class being created."""

INTROSPECTION = frozenset(
    ("__mro__", "__bases__", "__dict__", "__annotations__", "__annotate__", "mro")
)
"""Attributes that show a class's bases or own body, which the move changes."""

SUPER_CREATION = frozenset(("__new__", "__init__", "__init_subclass__", "__prepare__"))
CREATION_METHODS = frozenset(("__new__", "__init__", "__prepare__"))
DEPTH = 6


def declared_names(statement: ast.stmt) -> frozenset[str]:
    """The class-scope names one declaration statement binds."""

    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset((statement.name,))
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return frozenset((statement.target.id,))
    if isinstance(statement, ast.Assign):
        return frozenset(
            target.id for target in statement.targets if isinstance(target, ast.Name)
        )
    return frozenset()


class Role(Enum):
    """What a name in creation code holds, as far as the move is concerned."""

    NAMESPACE = auto()
    BASES = auto()
    BASE = auto()
    NEW_CLASS = auto()


@dataclass(frozen=True)
class CreationCode:
    """One function that runs while a class is created, with its source."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    location: str
    owner: CompactIndexedClass | None
    """The scanned class declaring it, for resolving `cls.` and `mcs.` calls."""


@dataclass(frozen=True)
class ClassCreationProof:
    """Composing a new plain owner leaves the composer's creation unchanged."""

    snapshot: CodemodSourceSnapshot
    composer: CompactIndexedClass
    ancestry: tuple[CompactIndexedClass, ...]
    owner_name: str
    moved: frozenset[str]
    abstract_moved: frozenset[str]

    @classmethod
    def composing(
        cls,
        snapshot: CodemodSourceSnapshot,
        composer: CompactIndexedClass,
        ancestry: tuple[CompactIndexedClass, ...],
        owners: tuple[tuple[str, ast.ClassDef], ...],
    ) -> ClassCreationProof:
        """The proof for composing these owners: each owner's body leaves the
        composer, and its abstract members with it."""

        statements = tuple(
            statement for _name, node in owners for statement in node.body
        )
        return cls(
            snapshot,
            composer,
            ancestry,
            " + ".join(name for name, _node in owners),
            frozenset(name for statement in statements for name in declared_names(statement)),
            frozenset(
                statement.name
                for statement in statements
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(ast.unparse(decorator).endswith("abstractmethod") for decorator in statement.decorator_list)
            ),
        )

    def require_unchanged(self) -> None:
        reserved = sorted(self.moved & REGISTRY_NAMES)
        if reserved:
            raise NeedsInput(
                f"`{self.owner_name}` would take {reserved}, which `{self.composer.qualname}`'s "
                "creation reads to register it",
                NeededInput.PROOF,
            )
        if self.abstract_moved:
            raise NeedsInput(
                f"`{self.owner_name}` would take abstract {sorted(self.abstract_moved)}, which "
                f"`{self.composer.qualname}`'s creation reads",
                NeededInput.PROOF,
            )
        for indexed in self.ancestry:
            hook = self.method(indexed, "__init_subclass__")
            if hook is not None:
                CreationReads(self, hook, {self.first_parameter(hook.node): Role.NEW_CLASS}).require()
        for metaclass in self.metaclasses():
            for code in self.metaclass_code(metaclass):
                CreationReads(self, code, self.metaclass_roles(code.node)).require()

    @property
    def class_index(self):
        return self.snapshot.class_overlap_verdicts.class_index

    def method(
        self, indexed: CompactIndexedClass, name: str
    ) -> CreationCode | None:
        node = self.class_node(indexed)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == name:
                return CreationCode(statement, f"{indexed.file_path}:{statement.lineno}", indexed)
        return None

    def class_node(self, indexed: CompactIndexedClass) -> ast.ClassDef:
        return ResolvedClassTarget.from_rewrite_target(
            self.snapshot,
            SourceRewriteTarget(file_path=indexed.file_path, qualname=indexed.qualname),
        ).node

    @staticmethod
    def first_parameter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        arguments = (*node.args.posonlyargs, *node.args.args)
        return arguments[0].arg if arguments else ""

    def metaclasses(self) -> Iterator[str]:
        """The qualified metaclass every class in the ancestry names."""

        for indexed in self.ancestry:
            node = self.class_node(indexed)
            for keyword in node.keywords:
                if keyword.arg == "metaclass":
                    yield self.qualified(indexed, keyword.value, node.lineno)

    def qualified(self, indexed: CompactIndexedClass, reference: ast.expr, line: int) -> str:
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            self.snapshot.parsed_module_for_source_path(indexed.file_path), reference, line=line
        )
        if witness is None:
            raise NeedsInput(
                f"{indexed.file_path}:{line} `{ast.unparse(reference)}` is unbound at class creation",
                NeededInput.PROOF,
            )
        return self.canonical(witness.qualified_name)

    def canonical(self, qualified_name: str) -> str:
        """The defining module's spelling of an importable name; a scanned name as is."""

        if self.class_index.class_for(qualified_name) is not None:
            return qualified_name
        module_name, _, attribute = qualified_name.rpartition(".")
        try:
            value = getattr(import_module(module_name or "builtins"), attribute)
        except (ImportError, AttributeError):
            return qualified_name
        return f"{value.__module__}.{value.__qualname__}"

    def metaclass_code(self, symbol: str) -> Iterator[CreationCode]:
        """Scanned creation methods along a metaclass's ancestry; a modelled or
        imported metaclass outside the scan contributes none it has not proved."""

        pending, seen = [symbol], set()
        while pending:
            current = self.canonical(pending.pop())
            if current in seen or current in MODELLED_METACLASSES:
                continue
            seen.add(current)
            indexed = self.class_index.class_for(current)
            if indexed is None:
                raise NeedsInput(
                    f"`{self.composer.qualname}`'s metaclass `{current}` is outside the scan and "
                    "NRA's creation models: NRA needs a proof of what it reads",
                    NeededInput.PROOF,
                )
            hooks = CREATION_METHODS | self.modelled_hooks()
            node = self.class_node(indexed)
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in hooks:
                    yield CreationCode(statement, f"{indexed.file_path}:{statement.lineno}", indexed)
            pending.extend(".".join(reference.resolved_parts) for reference in indexed.base_references)

    @staticmethod
    def modelled_hooks() -> frozenset[str]:
        """Methods a modelled metaclass calls on its subclass while creating a class."""

        return frozenset(
            name for name, value in vars(AutoRegisterMeta).items()
            if callable(getattr(value, "__func__", value)) and not name.startswith("__")
        )

    def metaclass_roles(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Role]:
        """By position for type's creation protocol; by name for a modelled hook."""

        names = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
        if node.name in ("__new__", "__prepare__"):
            return dict(zip(names[2:4], (Role.BASES, Role.NAMESPACE)))
        if node.name == "__init__":
            return dict(zip((names[0], *names[2:4]), (Role.NEW_CLASS, Role.BASES, Role.NAMESPACE)))
        by_name = {"cls": Role.NEW_CLASS, "new_class": Role.NEW_CLASS, "attrs": Role.NAMESPACE,
                   "namespace": Role.NAMESPACE, "bases": Role.BASES}
        static = any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in node.decorator_list
        )
        return {name: by_name[name] for name in names[0 if static else 1:] if name in by_name}

    def new_class_method(self, name: str) -> CreationCode | None:
        """The composer's own resolution of a method its creation calls on it."""

        mro_type = ClassMroAuthority(self.class_index.classes_by_symbol).resolve(
            self.composer.symbol
        ).mro_type
        if mro_type is None:
            raise NeedsInput(f"`{self.composer.qualname}` has an open MRO", NeededInput.PROOF)
        for declaration in mro_type.declarations:
            indexed = self.class_index.class_for(declaration.symbol)
            if indexed is not None and (code := self.method(indexed, name)) is not None:
                return code
        return None


@dataclass
class CreationReads:
    """One creation function's reads of the three facts the move changes."""

    proof: ClassCreationProof
    code: CreationCode
    roles: dict[str, Role]
    depth: int = 0

    def require(self) -> None:
        if self.depth > DEPTH:
            self.refuse(self.code.node, "calls too deep to follow")
        self.parents = {
            child: parent for parent in ast.walk(self.code.node) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(self.code.node):
            match node:
                case ast.For(target=ast.Name(id=target), iter=ast.Name(id=source)) | ast.comprehension(
                    target=ast.Name(id=target), iter=ast.Name(id=source)
                ) if self.roles.get(source) is Role.BASES:
                    self.roles[target] = Role.BASE
                case ast.Assign(targets=[ast.Name(id=target)], value=value) | ast.AnnAssign(
                    target=ast.Name(id=target), value=value
                ) if value is not None and self.is_super_call(value, "__new__"):
                    self.roles[target] = Role.NEW_CLASS
        for node in ast.walk(self.code.node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in self.roles:
                self.require_use(node, self.roles[node.id])

    def refuse(self, node: ast.AST, what: str) -> NoReturn:
        line = getattr(node, "lineno", self.code.node.lineno) - self.code.node.lineno
        file_path, _, start = self.code.location.rpartition(":")
        location = f"{file_path}:{int(start) + line}" if start.isdigit() else self.code.location
        raise NeedsInput(
            f"{location} `{ast.unparse(node)}` {what} while `{self.proof.composer.qualname}` is "
            f"created: NRA needs a proof that the new base `{self.proof.owner_name}` leaves it unchanged",
            NeededInput.PROOF,
        )

    @staticmethod
    def is_super_call(node: ast.AST, *names: str) -> bool:
        match node:
            case ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id="super")), attr=attr)):
                return attr in (names or SUPER_CREATION)
        return False

    def require_use(self, node: ast.Name, role: Role) -> None:
        parent = self.parents.get(node)
        if isinstance(parent, ast.Call) and node in parent.args and self.is_super_call(parent):
            return
        if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
            return
        getattr(self, f"require_{role.name.lower()}")(node, parent)

    def constant_key(self, node: ast.AST | None) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in self.proof.moved
        )

    def require_namespace(self, node: ast.Name, parent: ast.AST | None) -> None:
        match parent:
            case ast.Subscript(value=value, slice=key) if value is node and self.constant_key(key):
                return
            case ast.Compare(left=key, ops=[ast.In() | ast.NotIn()], comparators=[value]) if (
                value is node and self.constant_key(key)
            ):
                return
            case ast.Attribute(attr="get" | "pop" | "setdefault"):
                call = self.parents.get(parent)
                if isinstance(call, ast.Call) and call.args and self.constant_key(call.args[0]):
                    return
        if isinstance(parent, ast.Call) and node in parent.args:
            self.follow(parent, node, Role.NAMESPACE)
            return
        self.refuse(parent or node, "reads the class namespace")

    def require_bases(self, node: ast.Name, parent: ast.AST | None) -> None:
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return
        if isinstance(parent, (ast.If, ast.While, ast.BoolOp, ast.IfExp)):
            if not self.proof.composer.base_references:
                self.refuse(parent, "tests whether the class has bases")
            return
        if isinstance(parent, ast.Call) and node in parent.args:
            self.follow(parent, node, Role.BASES)
            return
        self.refuse(parent or node, "reads the class's bases")

    def require_base(self, node: ast.Name, parent: ast.AST | None) -> None:
        """A base's metaclass, dispatched on: `X.handler_for(type(base))(base)`
        where `X`'s handler for `type` answers a constant."""

        dispatch = parent
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "type":
            handler = self.parents.get(parent)
            dispatch = self.parents.get(handler) if handler is not None else None
        if isinstance(dispatch, ast.Call) and self.constant_plain_dispatch(dispatch):
            return
        self.refuse(parent or node, "reads a base of the class")

    def constant_plain_dispatch(self, call: ast.Call) -> bool:
        match call:
            case ast.Call(
                func=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=table), attr="handler_for"),
                    args=[ast.Call(func=ast.Name(id="type"), args=[ast.Name(id=base)])],
                ),
                args=[ast.Name(id=passed)],
            ) if base == passed and self.roles.get(base) is Role.BASE:
                owner = self.code.owner if table in ("cls", "self") else None
                if owner is None:
                    return False
                node = self.proof.class_node(owner)
                for statement in node.body:
                    if (
                        isinstance(statement, ast.FunctionDef)
                        and any(
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "handles"
                            and [ast.unparse(argument) for argument in decorator.args] == ["type"]
                            for decorator in statement.decorator_list
                        )
                    ):
                        body = [
                            item for item in statement.body
                            if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
                        ]
                        return len(body) == 1 and isinstance(body[0], ast.Return) and (
                            body[0].value is None or self.is_literal(body[0].value)
                        )
        return False

    @staticmethod
    def is_literal(node: ast.expr) -> bool:
        try:
            ast.literal_eval(node)
        except ValueError:
            return False
        return True

    def require_new_class(self, node: ast.Name, parent: ast.AST | None) -> None:
        match parent:
            case ast.Attribute(attr=attr, ctx=ast.Store()):
                return
            case ast.Attribute(attr=attr) if attr in INTROSPECTION:
                self.refuse(parent, "introspects the new class")
            case ast.Attribute(attr=attr):
                call = self.parents.get(parent)
                if isinstance(call, ast.Call) and call.func is parent:
                    code = self.proof.new_class_method(attr)
                    if code is None:
                        self.refuse(call, "calls a method the class does not declare")
                    self.descend(code, {ClassCreationProof.first_parameter(code.node): Role.NEW_CLASS}, call, 1)
                return
            case ast.Return() | ast.Assign() | ast.Compare() | ast.AnnAssign():
                return
            case ast.Call(func=ast.Name(id="cast" | "issubclass" | "isinstance")):
                return
            case ast.Call(func=ast.Name(id="getattr" | "hasattr" | "setattr"), args=[value, key, *_]) if (
                value is node and isinstance(key, ast.Constant) and key.value not in INTROSPECTION
            ):
                return
            case ast.Call() if node in parent.args:
                self.follow(parent, node, Role.NEW_CLASS)
                return
        self.refuse(parent or node, "uses the new class")

    def follow(self, call: ast.Call, node: ast.Name, role: Role) -> None:
        """A role passed to a function: modelled, or read the same way."""

        target = self.callee(call.func)
        if target is True:
            return
        if target is None:
            self.refuse(call, "passes it to a function NRA cannot read")
        assert isinstance(target, CreationCode)
        static = any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in target.node.decorator_list
        )
        self.descend(target, {}, call, 0 if static else 1, passed={call.args.index(node): role})

    def descend(
        self,
        code: CreationCode,
        roles: dict[str, Role],
        call: ast.Call,
        skip: int,
        passed: dict[int, Role] | None = None,
    ) -> None:
        names = [argument.arg for argument in (*code.node.args.posonlyargs, *code.node.args.args)][skip:]
        mapped = dict(roles)
        for index, argument in enumerate(call.args):
            role = (passed or {}).get(index) or (
                self.roles.get(argument.id) if isinstance(argument, ast.Name) else None
            )
            if role is not None and index < len(names):
                mapped[names[index]] = role
        CreationReads(self.proof, code, mapped, self.depth + 1).require()

    def callee(self, func: ast.expr) -> CreationCode | bool | None:
        """The scanned function a call reaches, or True for a modelled one."""

        match func:
            case ast.Attribute(value=ast.Name(id=receiver), attr=attr):
                if receiver in ("cls", "mcs", "self") and self.code.owner is not None:
                    return self.proof.method(self.code.owner, attr)
                owner = self.code.owner
                if owner is None:
                    return None
                qualified = self.proof.qualified(owner, func.value, self.code.node.lineno)
                if qualified in MODELLED_METACLASSES:
                    return True
                indexed = self.proof.class_index.class_for(qualified)
                return self.proof.method(indexed, attr) if indexed is not None else None
        return None

