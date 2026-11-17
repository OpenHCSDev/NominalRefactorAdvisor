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
import types
from collections.abc import Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import ClassVar, NoReturn
from importlib import import_module

from metaclass_registry.core import AutoRegisterMeta

from nominal_refactor_advisor.class_index import (
    ClassMemberAnswersABC,
    CompactAnnotatedMethod,
    CompactAnnotatedProperty,
    CompactCallableMemberDeclaration,
    CompactClassMemberDeclaration,
    CompactIndexedClass,
    CompactUnannotatedCallable,
)
from nominal_refactor_advisor.class_mro import native_root_type
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.installed_source import InstalledModule
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

CONSTRUCTION_NAMES = frozenset(
    (
        "__init__",
        "__new__",
        "__signature__",
        "__annotations__",
        "__annotate__",
        "__dataclass_fields__",
        "__dataclass_params__",
    )
)
"""What decides the parameters a call to a class binds, once its declarations are read."""

SUPER_CREATION = frozenset(("__new__", "__init__", "__init_subclass__", "__prepare__"))
CREATION_METHODS = frozenset(("__new__", "__init__", "__prepare__"))
DEPTH = 6

REFERENCE_STORES: dict[type, frozenset[str]] = {
    list: frozenset(("append", "insert")),
    set: frozenset(("add",)),
    dict: frozenset(("setdefault",)),
}
"""The builtin container methods that keep an argument by reference and run
nothing of it but its type's `__hash__` and `__eq__`: storing a class there
changes nothing a call to it binds. The builtins' surface does not say which
methods call their arguments, so this table declares it."""

BUILTIN_CONTAINERS = {f"builtins.{kind.__name__}": kind for kind in REFERENCE_STORES}


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


def installed_class_node(indexed: CompactIndexedClass) -> ast.ClassDef:
    """A class an installed module declares, from that module's parsed source."""

    parsed = InstalledModule.parsed(indexed.module_name)
    body: list[ast.stmt] = [] if parsed is None else parsed.module.body
    node: ast.ClassDef | None = None
    for name in indexed.qualname.split("."):
        node = next(
            (statement for statement in body if isinstance(statement, ast.ClassDef) and statement.name == name),
            None,
        )
        if node is None:
            raise NeedsInput(
                f"{indexed.file_path}:{indexed.line} `{indexed.qualname}` has no declaration NRA can read",
                NeededInput.PROOF,
            )
        body = node.body
    assert node is not None
    return node


class Role(ABC):
    """What a name in creation code holds, as far as the move is concerned.

    The role and the proof cross: a creation proof and a construction proof
    read the same role differently, so each role answers through its own
    method on the reads that meet it.
    """

    iteration: ClassVar[str]
    """What iterating a value in this role reads, when the role yields nothing."""

    @abstractmethod
    def require(self, reads: CreationReads, node: ast.Name, parent: ast.AST | None) -> None:
        """One read of a name holding this role is one the move leaves unchanged."""

        raise NotImplementedError

    def element(self, reads: CreationReads, loop: ast.For | ast.comprehension) -> Role | None:
        """What iterating a value in this role yields, for the loop's target."""

        return reads.iterated(loop, self)


class NamespaceRole(Role):
    """The namespace the class body filled, without the moved names once composed."""

    iteration = "iterates the class namespace, moved names included"

    def require(self, reads: CreationReads, node: ast.Name, parent: ast.AST | None) -> None:
        reads.require_namespace(node, parent)


class BasesRole(Role):
    """The bases the class statement names; the move adds the owner."""

    iteration = "iterates the class's bases"

    def require(self, reads: CreationReads, node: ast.Name, parent: ast.AST | None) -> None:
        reads.require_bases(node, parent)

    def element(self, reads: CreationReads, loop: ast.For | ast.comprehension) -> Role | None:
        return BASE


class BaseRole(Role):
    """One base of the class."""

    iteration = "iterates a base of the class"

    def require(self, reads: CreationReads, node: ast.Name, parent: ast.AST | None) -> None:
        reads.require_base(node, parent)


class NewClassRole(Role):
    """The class being created, whose lookups can reach what the owner binds."""

    iteration = "iterates the new class"

    def require(self, reads: CreationReads, node: ast.Name, parent: ast.AST | None) -> None:
        reads.require_new_class(node, parent)


NAMESPACE = NamespaceRole()
BASES = BasesRole()
BASE = BaseRole()
NEW_CLASS = NewClassRole()


_IMPLICIT_VALUES = (
    str, int, tuple, dict, frozenset, type(None), types.FunctionType,
    types.GetSetDescriptorType, types.MemberDescriptorType, types.BuiltinFunctionType,
    types.WrapperDescriptorType, types.MethodDescriptorType, types.ClassMethodDescriptorType,
)
"""The types of what class creation binds by itself (`__module__`, `__doc__`,
`__dict__`, slot descriptors, a dataclass's generated methods) and of what a
native base's namespace holds."""


@dataclass(frozen=True)
class MroWalk:
    """What a walk of the new class's `__mro__` visits before and after the move.

    `vars()` of each class lists the names its own body binds, in the order it
    binds them; a walk keeping the first binding of each name finds every
    name's lookup winner, in the order the walk first meets the name.
    """

    before: tuple[str, ...]
    after: tuple[str, ...]
    """The class symbols the walk visits, in C3 order."""
    composer: str
    deleted: frozenset[str]
    """The names the composer's own body stops binding."""

    @classmethod
    def unmoved(cls, ancestry: tuple[CompactIndexedClass, ...]) -> MroWalk:
        order = tuple(indexed.symbol for indexed in ancestry)
        return cls(order, order, order[0] if order else "", frozenset())


@dataclass(frozen=True)
class CreationCode:
    """One function that runs while a class is created, with its source."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    location: str
    owner: CompactIndexedClass | None
    """The scanned class declaring it, for resolving `cls.` and `mcs.` calls."""


@dataclass(frozen=True)
class BuiltinContainer:
    """A value proved to be a builtin container, with the type arguments its
    declaration names, read where `owner`'s module binds names at `line`."""

    kind: type
    arguments: tuple[ast.expr, ...]
    owner: CompactIndexedClass
    line: int


@dataclass(frozen=True)
class ClassCreationProof:
    """Composing a new plain owner leaves the composer's creation unchanged."""

    snapshot: CodemodSourceSnapshot
    composer: CompactIndexedClass
    ancestry: tuple[CompactIndexedClass, ...]
    owner_name: str
    moved: frozenset[str]
    abstract_moved: frozenset[str]
    walk: MroWalk

    @classmethod
    def composing(
        cls,
        snapshot: CodemodSourceSnapshot,
        composer: CompactIndexedClass,
        ancestry: tuple[CompactIndexedClass, ...],
        owners: tuple[tuple[str, ast.ClassDef], ...],
        walk: MroWalk,
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
            walk,
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
        for code, roles in self.creation_code():
            CreationReads(self, code, roles).require()

    @classmethod
    def constructing(
        cls,
        snapshot: CodemodSourceSnapshot,
        composer: CompactIndexedClass,
        ancestry: tuple[CompactIndexedClass, ...],
    ) -> ClassCreationProof:
        """The proof that creating `composer` keeps the parameters its declarations give a call."""

        return cls(
            snapshot, composer, ancestry, composer.qualname, CONSTRUCTION_NAMES, frozenset(),
            MroWalk.unmoved(ancestry),
        )

    def require_construction_kept(self) -> None:
        """Calling the class binds the parameters its declared MRO gives it.

        A decorator may replace the class, a metaclass `__call__` decides what a
        call binds, and creation code may store an `__init__` or annotations;
        anything else creation runs leaves the binding alone.
        """

        for indexed in self.ancestry:
            if not indexed.class_decorators_are_promotion_safe:
                raise NeedsInput(
                    f"{indexed.file_path}:{indexed.line} `{indexed.qualname}` is decorated by code "
                    "that may replace what a call binds: NRA needs a proof of its parameters",
                    NeededInput.PROOF,
                )
        for metaclass in self.metaclasses():
            for indexed in self.metaclass_ancestry(metaclass):
                call = self.method(indexed, "__call__")
                if call is not None:
                    raise NeedsInput(
                        f"{call.location} metaclass `{indexed.qualname}` defines `__call__`, which "
                        f"decides what a call to `{self.composer.qualname}` binds",
                        NeededInput.PROOF,
                    )
        for code, roles in self.creation_code():
            ConstructionReads(self, code, roles).require()

    def creation_code(self) -> Iterator[tuple[CreationCode, dict[str, Role]]]:
        """Each function that runs while the composer is created, with what its names hold."""

        for indexed in self.ancestry:
            hook = self.method(indexed, "__init_subclass__")
            if hook is not None:
                yield hook, {self.first_parameter(hook.node): NEW_CLASS}
        for metaclass in self.metaclasses():
            for code in self.metaclass_code(metaclass):
                yield code, self.metaclass_roles(code.node)

    @property
    def class_index(self):
        return self.snapshot.class_overlap_verdicts.class_index

    def carriers(self, order: tuple[str, ...], composed: bool, attributes: frozenset[str]) -> tuple[str, ...]:
        """The names a first-found walk of `order` meets whose winning value may
        have one of `attributes`, in the order it meets them."""

        found: dict[str, bool] = {}
        for symbol in order:
            if native_root_type(symbol) is not None:
                continue
            for name, statement in self.own_bindings(symbol, composed):
                found.setdefault(name, self.may_carry(statement, attributes))
        return tuple(name for name, carries in found.items() if carries)

    def own_bindings(self, symbol: str, composed: bool) -> Iterator[tuple[str, ast.stmt]]:
        """Each name a scanned class's own body puts in `vars()`, with the
        statement binding it, in binding order; once composed, the composer's
        body no longer binds the names it deletes."""

        indexed = self.snapshot.class_overlap_verdicts.ancestor(symbol)
        if indexed is None:
            raise NeedsInput(
                f"`{symbol}` has no declaration NRA can read, and the walk of "
                f"`{self.composer.qualname}`'s MRO reads its names",
                NeededInput.PROOF,
            )
        for statement in self.class_node(indexed).body:
            names = declared_names(statement)
            if not names and not (
                isinstance(statement, (ast.Pass, ast.ClassDef))
                or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
            ):
                raise NeedsInput(
                    f"{indexed.file_path}:{statement.lineno} `{indexed.qualname}` binds names a "
                    f"statement decides, which the walk of `{self.composer.qualname}`'s MRO reads: "
                    "NRA needs a proof of the order it binds them",
                    NeededInput.PROOF,
                )
            if isinstance(statement, ast.ClassDef):
                names = frozenset((statement.name,))
            if isinstance(statement, ast.AnnAssign) and statement.value is None:
                continue
            for name in names:
                if not (composed and symbol == self.walk.composer and name in self.walk.deleted):
                    yield name, statement

    def binders(self, order: tuple[str, ...], composed: bool, name: str) -> tuple[str, ...]:
        """The classes of `order` whose own `vars()` holds `name`: what a walk
        keeping each class with `name in vars(c)` selects."""

        return tuple(
            symbol
            for symbol in order
            if (
                name in vars(native)
                if (native := native_root_type(symbol)) is not None
                else any(bound == name for bound, _ in self.own_bindings(symbol, composed))
            )
        )

    def require_selection_kept(self, name: str, node: ast.AST, reads: CreationReads) -> None:
        """A walk keeping the classes whose own body binds `name` selects the same
        classes in the same order when the move neither gives `name` to a class
        it adds nor takes it from one it keeps. A dunder name may be bound by
        class creation itself, which no body shows."""

        if name.startswith("__") and name.endswith("__"):
            reads.refuse(node, "selects classes by a name class creation binds by itself")
        before = self.binders(self.walk.before, False, name)
        after = self.binders(self.walk.after, True, name)
        if before != after:
            reads.refuse(node, f"selects {list(before)} for `{name}`, and {list(after)} once composed,")

    @staticmethod
    def may_carry(statement: ast.stmt, attributes: frozenset[str]) -> bool:
        """Whether the value one statement binds may have one of `attributes`:
        a plain function has only a function's; a literal only its own type's."""

        match statement:
            case ast.FunctionDef(decorator_list=[]) | ast.AsyncFunctionDef(decorator_list=[]):
                return any(hasattr(types.FunctionType, attribute) for attribute in attributes)
            case ast.Assign(value=value) | ast.AnnAssign(value=value) if value is not None:
                try:
                    literal = ast.literal_eval(value)
                except ValueError:
                    return True
                return any(hasattr(literal, attribute) for attribute in attributes)
        return True

    def require_walk_kept(self, attributes: frozenset[str], node: ast.AST, reads: CreationReads) -> None:
        """A walk reading only `attributes` of each name's winner answers the same
        when those winners keep their values and the names carrying them keep
        their order. The name check of the composed ancestry keeps every value;
        this keeps the order."""

        if any(hasattr(kind, attribute) for kind in _IMPLICIT_VALUES for attribute in attributes):
            reads.refuse(node, "reads an attribute what class creation binds by itself may have")
        before = self.carriers(self.walk.before, False, attributes)
        after = self.carriers(self.walk.after, True, attributes)
        if before != after:
            reads.refuse(
                node,
                f"meets {list(before)} in that order, and {list(after)} once composed,",
            )

    def method(
        self, indexed: CompactIndexedClass, name: str
    ) -> CreationCode | None:
        return next(self.methods(indexed, frozenset((name,))), None)

    def methods(self, indexed: CompactIndexedClass, names: frozenset[str]) -> Iterator[CreationCode]:
        for statement in self.class_node(indexed).body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in names:
                yield CreationCode(statement, f"{indexed.file_path}:{statement.lineno}", indexed)

    def class_node(self, indexed: CompactIndexedClass) -> ast.ClassDef:
        """Its declaration: the scan's, or else the installed module's source."""

        if self.class_index.class_for(indexed.symbol) is None:
            return installed_class_node(indexed)
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

    def module_end(self, indexed: CompactIndexedClass) -> int:
        """The line after `indexed`'s module: where a name reads its final binding."""

        return len(self.snapshot.parsed_module_for_source_path(indexed.file_path).source.splitlines()) + 1

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
        """Scanned creation methods along a metaclass's ancestry."""

        hooks = CREATION_METHODS | self.modelled_hooks()
        for indexed in self.metaclass_ancestry(symbol):
            yield from self.methods(indexed, hooks)

    def metaclass_ancestry(self, symbol: str) -> Iterator[CompactIndexedClass]:
        """The scanned classes along a metaclass's ancestry; a modelled metaclass
        ends its branch, and one outside the scan refuses."""

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
            yield indexed
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
            return dict(zip(names[2:4], (BASES, NAMESPACE)))
        if node.name == "__init__":
            return dict(zip((names[0], *names[2:4]), (NEW_CLASS, BASES, NAMESPACE)))
        by_name = {"cls": NEW_CLASS, "new_class": NEW_CLASS, "attrs": NAMESPACE,
                   "namespace": NAMESPACE, "bases": BASES}
        static = any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in node.decorator_list
        )
        return {name: by_name[name] for name in names[0 if static else 1:] if name in by_name}


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
                case ast.For(iter=ast.Name(id=source)) | ast.comprehension(iter=ast.Name(id=source)) if (
                    source in self.roles
                ):
                    element = self.roles[source].element(self, node)
                    match node.target:
                        case _ if element is None:
                            pass
                        case ast.Name(id=target):
                            self.roles[target] = element
                        case target:
                            self.refuse(target, "unpacks what it iterates")
                case ast.Assign(targets=[ast.Name(id=target)], value=value) | ast.AnnAssign(
                    target=ast.Name(id=target), value=value
                ) if value is not None and self.is_super_call(value, "__new__"):
                    self.roles[target] = NEW_CLASS
        for node in ast.walk(self.code.node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in self.roles:
                self.require_use(node, self.roles[node.id])

    def refuse(self, node: ast.AST, what: str) -> NoReturn:
        line = getattr(node, "lineno", self.code.node.lineno) - self.code.node.lineno
        file_path, _, start = self.code.location.rpartition(":")
        location = f"{file_path}:{int(start) + line}" if start.isdigit() else self.code.location
        raise NeedsInput(
            f"{location} `{ast.unparse(node)}` {what} while `{self.proof.composer.qualname}` is "
            f"created: {self.unproved}",
            NeededInput.PROOF,
        )

    @property
    def unproved(self) -> str:
        return f"NRA needs a proof that the new base `{self.proof.owner_name}` leaves it unchanged"

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
        role.require(self, node, parent)

    def iterated(self, loop: ast.For | ast.comprehension, role: Role) -> Role | None:
        """Iterating a value whose role yields nothing reads all of it."""

        self.refuse(loop.iter, role.iteration)

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
            self.follow(parent, node, NAMESPACE)
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
            self.follow(parent, node, BASES)
            return
        self.refuse(parent or node, "reads the class's bases")

    def require_base(self, node: ast.Name, parent: ast.AST | None) -> None:
        """A base's metaclass, dispatched on: `X.handler_for(type(base))(base)`
        where `X`'s handler for `type` answers a constant."""

        dispatch = parent
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "type":
            handler = self.parents.get(parent)
            dispatch = self.parents.get(handler) if handler is not None else None
        if isinstance(dispatch, ast.Call) and self.constant_plain_dispatch(dispatch, node.id):
            return
        self.refuse(parent or node, "reads a base of the class")

    def constant_plain_dispatch(self, call: ast.Call, base: str) -> bool:
        """`X.handler_for(type(base))(base)` on the base `base` names."""

        match call:
            case ast.Call(
                func=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=table), attr="handler_for"),
                    args=[ast.Call(func=ast.Name(id="type"), args=[ast.Name(id=dispatched)])],
                ),
                args=[ast.Name(id=passed)],
            ) if dispatched == passed == base:
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
            case ast.Attribute(attr="__mro__") if self.first_found_walk(parent) or self.binding_walk(parent):
                return
            case ast.Attribute(attr=attr) if attr in INTROSPECTION:
                self.refuse(parent, "introspects the new class")
            case ast.Attribute(attr=attr):
                call = self.parents.get(parent)
                if isinstance(call, ast.Call) and call.func is parent:
                    code = self.proof.class_index.declared_member(
                        self.proof.composer.symbol, attr
                    ).answer(NewClassCall(self, call))
                    self.descend(code, {ClassCreationProof.first_parameter(code.node): NEW_CLASS}, call, 1)
                return
            case ast.Return() | ast.Assign() | ast.Compare() | ast.AnnAssign():
                return
            case ast.Call(func=ast.Name(id="cast" | "issubclass" | "isinstance")):
                return
            case ast.Call(func=ast.Name(id="getattr" | "hasattr" | "setattr"), args=[value, key, *_]) if (
                value is node and isinstance(key, ast.Constant) and key.value not in INTROSPECTION
            ):
                return
            case ast.Call(func=ast.Name() as func, args=[value], keywords=[]) if (
                value is node and self.is_builtin(func, "builtins.vars")
            ):
                # `vars(cls)` is the new class's own namespace: a read of it
                # is the namespace's read, which the moved names leave alone
                # unless it asks for one of them or reads all of it.
                self.require_namespace(parent, self.parents.get(parent))
                return
            case ast.Call() if node in parent.args:
                self.follow(parent, node, NEW_CLASS)
                return
        self.refuse(parent or node, "uses the new class")

    def is_builtin(self, func: ast.Name, qualified_name: str) -> bool:
        """Whether `func` names this builtin where the function body runs: its
        module's final binding of the name."""

        owner = self.code.owner
        return owner is not None and (
            self.proof.qualified(owner, func, self.proof.module_end(owner)) == qualified_name
        )

    def first_found_walk(self, attribute: ast.Attribute) -> bool:
        """`for o in cls.__mro__: for n, v in vars(o).items(): found.setdefault(n, v)`:
        each name's lookup winner, in the order the walk meets it. Its other
        reads must be `getattr` of constant attributes of those winners."""

        walk = self.parents.get(attribute)
        match walk:
            case ast.For(
                target=ast.Name(id=visited),
                iter=walked,
                body=[
                    ast.For(
                        target=ast.Tuple(elts=[ast.Name(id=name), ast.Name(id=value)]),
                        iter=ast.Call(
                            func=ast.Attribute(
                                value=ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner)]),
                                attr="items",
                            ),
                            args=[],
                        ),
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(value=ast.Name(id=found), attr="setdefault"),
                                    args=[ast.Name(id=key), ast.Name(id=kept)],
                                )
                            )
                        ],
                        orelse=[],
                    ) as inner
                ],
                orelse=[],
            ) if walked is attribute and owner == visited and key == name and kept == value:
                pass
            case _:
                return False
        assert isinstance(walk, ast.For)
        walked_nodes = frozenset(ast.walk(walk))
        for node in ast.walk(self.code.node):
            if (
                isinstance(node, ast.Name)
                and node.id in (visited, name, value)
                and node not in walked_nodes
                and not self.comprehension_binds(node)
            ):
                self.refuse(node, "reads what the walk of the MRO left behind")
        attributes = self.found_reads(found, walk)
        self.proof.require_walk_kept(attributes, walk, self)
        del inner
        return True

    def binding_walk(self, attribute: ast.Attribute) -> bool:
        """`(c for c in cls.__mro__ if "<name>" in vars(c))`, as any comprehension:
        the classes whose own body binds the name, in C3 order."""

        generator = self.parents.get(attribute)
        comprehension = self.parents.get(generator) if generator is not None else None
        match generator, comprehension:
            case (
                ast.comprehension(
                    target=ast.Name(id=visited),
                    iter=walked,
                    ifs=[
                        ast.Compare(
                            left=ast.Constant(value=str() as name),
                            ops=[ast.In()],
                            comparators=[ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner)], keywords=[])],
                        )
                    ],
                    is_async=0,
                ),
                ast.GeneratorExp(elt=ast.Name(id=kept), generators=[_])
                | ast.ListComp(elt=ast.Name(id=kept), generators=[_])
                | ast.SetComp(elt=ast.Name(id=kept), generators=[_]),
            ) if walked is attribute and owner == visited and kept == visited:
                self.proof.require_selection_kept(name, attribute, self)
                return True
        return False

    def comprehension_binds(self, node: ast.Name) -> bool:
        """Whether a comprehension around `node` binds its name: its own scope's variable."""

        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and any(
                isinstance(bound, ast.Name) and bound.id == node.id
                for generator in current.generators
                for bound in ast.walk(generator.target)
            ):
                return True
            current = self.parents.get(current)
        return False

    def found_reads(self, found: str, walk: ast.For) -> frozenset[str]:
        """The constant attributes read of the winners `found` maps names to:
        it starts empty, the walk fills it, and only `found.items()` reads it,
        each value only through `getattr(value, "<attribute>", default)`."""

        walked_nodes = frozenset(ast.walk(walk))
        attributes: set[str] = set()
        for node in ast.walk(self.code.node):
            if not (isinstance(node, ast.Name) and node.id == found) or node in walked_nodes:
                continue
            parent = self.parents.get(node)
            if isinstance(node.ctx, ast.Store):
                match parent:
                    case ast.Assign(value=ast.Dict(keys=[])) | ast.AnnAssign(value=ast.Dict(keys=[])):
                        continue
                    case ast.Assign(value=ast.Call(func=ast.Name(id="dict"), args=[], keywords=[])):
                        continue
                self.refuse(parent or node, "fills what the walk of the MRO found")
            call = self.parents.get(parent) if parent is not None else None
            loop = self.parents.get(call) if call is not None else None
            match parent, call, loop:
                case (
                    ast.Attribute(attr="items"),
                    ast.Call(args=[], keywords=[]),
                    ast.comprehension(target=ast.Tuple(elts=[ast.Name(), ast.Name(id=value)]))
                    | ast.For(target=ast.Tuple(elts=[ast.Name(), ast.Name(id=value)])),
                ) if loop.iter is call:
                    scope = self.parents.get(loop) if isinstance(loop, ast.comprehension) else loop
                    attributes |= self.value_reads(value, scope)
                    continue
            self.refuse(parent or node, "uses what the walk of the MRO found")
        return frozenset(attributes)

    def value_reads(self, value: str, scope: ast.AST | None) -> set[str]:
        attributes: set[str] = set()
        for node in ast.walk(scope) if scope is not None else ():
            if not (isinstance(node, ast.Name) and node.id == value and isinstance(node.ctx, ast.Load)):
                continue
            match self.parents.get(node):
                case ast.Call(func=ast.Name(id="getattr"), args=[subject, ast.Constant(value=str(attribute)), *_]) if (
                    subject is node and attribute not in INTROSPECTION
                ):
                    attributes.add(attribute)
                case other:
                    self.refuse(other or node, "reads a lookup winner the MRO walk found")
        return attributes

    def follow(self, call: ast.Call, node: ast.Name, role: Role) -> None:
        """A role passed to a function: modelled, or read the same way."""

        if self.stored_by_reference(call, node):
            return
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
        type(self)(self.proof, code, mapped, self.depth + 1).require()

    def stored_by_reference(self, call: ast.Call, node: ast.Name) -> bool:
        """`container.add(value)` on a container proved a builtin one: the call keeps
        a reference and, for a hashing container, hashes it by its metaclass."""

        match call.func:
            case ast.Attribute(value=receiver, attr=attr) if node in call.args and self.code.owner is not None:
                owner = self.code.owner
                container = self.builtin_container(receiver, owner, self.proof.module_end(owner))
                if container is None or attr not in REFERENCE_STORES[container.kind]:
                    return False
                if container.kind is not list:
                    self.require_plain_hash(call)
                return True
        return False

    def require_plain_hash(self, call: ast.Call) -> None:
        """Hashing the class runs its metaclass's `__hash__` and `__eq__`: only
        `type`'s, or a modelled metaclass's, are known to read nothing."""

        for symbol in self.proof.metaclasses():
            for indexed in self.proof.metaclass_ancestry(symbol):
                if self.proof.method(indexed, "__hash__") or self.proof.method(indexed, "__eq__"):
                    self.refuse(call, f"hashes the class with `{indexed.qualname}`'s own `__hash__` or `__eq__`")

    def builtin_container(
        self, expr: ast.expr, owner: CompactIndexedClass, line: int
    ) -> BuiltinContainer | None:
        """The builtin container `expr` evaluates to, read where `owner`'s module
        binds its names at `line`: an empty display or builtin call, a class
        attribute declared with a builtin container type and value, or the value
        `setdefault` returns from such a mapping. None when NRA cannot prove it."""

        match expr:
            case ast.Dict(keys=[]):
                return BuiltinContainer(dict, (), owner, line)
            case ast.List(elts=[]):
                return BuiltinContainer(list, (), owner, line)
            case ast.Call(func=ast.Name() as name, args=[], keywords=[]):
                kind = BUILTIN_CONTAINERS.get(self.proof.qualified(owner, name, line))
                return BuiltinContainer(kind, (), owner, line) if kind is not None else None
            case ast.Attribute(value=ast.Name() as holder, attr=attr):
                indexed = self.proof.class_index.class_for(self.proof.qualified(owner, holder, line))
                return self.declared_container(indexed, attr) if indexed is not None else None
            case ast.Call(func=ast.Attribute(value=receiver, attr="setdefault"), args=[_, default], keywords=[]):
                mapping = self.builtin_container(receiver, owner, line)
                if mapping is None or mapping.kind is not dict or len(mapping.arguments) != 2:
                    return None
                value = self.annotated_container(mapping.arguments[1], mapping.owner, mapping.line)
                stored = self.builtin_container(default, owner, line)
                if value is None or stored is None or stored.kind is not value.kind:
                    return None
                return value
        return None

    def declared_container(self, indexed: CompactIndexedClass, attr: str) -> BuiltinContainer | None:
        """A class attribute whose declared type and declared value are the same
        builtin container. The declared type is the proof of what later
        bindings hold, as a type checker enforces it."""

        for statement in self.proof.class_node(indexed).body:
            match statement:
                case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation, value=value) if (
                    name == attr and value is not None
                ):
                    declared = self.annotated_container(annotation, indexed, statement.lineno)
                    stored = self.builtin_container(value, indexed, statement.lineno)
                    if declared is None or stored is None or stored.kind is not declared.kind:
                        return None
                    return declared
        return None

    def annotated_container(
        self, annotation: ast.expr, owner: CompactIndexedClass, line: int
    ) -> BuiltinContainer | None:
        """The builtin container an annotation names, `ClassVar` unwrapped."""

        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            annotation = ast.parse(annotation.value, mode="eval").body
        head, arguments = (
            (annotation.value, annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice])
            if isinstance(annotation, ast.Subscript)
            else (annotation, [])
        )
        if not isinstance(head, (ast.Name, ast.Attribute)):
            return None
        qualified = self.proof.qualified(owner, head, line)
        if qualified == "typing.ClassVar" and len(arguments) == 1:
            return self.annotated_container(arguments[0], owner, line)
        kind = BUILTIN_CONTAINERS.get(qualified)
        return BuiltinContainer(kind, tuple(arguments), owner, line) if kind is not None else None

    def callee(self, func: ast.expr) -> CreationCode | bool | None:
        """The scanned function a call reaches, or True for a modelled one."""

        match func:
            case ast.Attribute(value=ast.Name(id=receiver), attr=attr):
                if receiver in ("cls", "mcs", "self") and self.code.owner is not None:
                    return self.proof.method(self.code.owner, attr)
                owner = self.code.owner
                if owner is None:
                    return None
                # A function body looks a global up when it runs: the module's final binding.
                qualified = self.proof.qualified(owner, func.value, self.proof.module_end(owner))
                if qualified in MODELLED_METACLASSES:
                    return True
                indexed = self.proof.class_index.class_for(qualified)
                return self.proof.method(indexed, attr) if indexed is not None else None
        return None


@dataclass
class ConstructionReads(CreationReads):
    """One creation function's stores of what a call to the class binds.

    Reading bases, the namespace or the new class changes no parameter; only a
    store of a name in `CONSTRUCTION_NAMES`, or a function NRA cannot read, can.
    """

    @property
    def unproved(self) -> str:
        return "NRA needs a proof that its creation keeps the parameters a call binds"

    def require_bases(self, node: ast.Name, parent: ast.AST | None) -> None:
        return

    def require_base(self, node: ast.Name, parent: ast.AST | None) -> None:
        return

    def iterated(self, loop: ast.For | ast.comprehension, role: Role) -> Role | None:
        """Iterating reads, and reading stores no parameter."""

        return None

    def require_new_class(self, node: ast.Name, parent: ast.AST | None) -> None:
        match parent:
            case ast.Attribute(attr=attr, ctx=ast.Store()) if attr in self.proof.moved:
                self.refuse(parent, "sets what a call binds")
            case ast.Attribute(attr=attr, ctx=ast.Load()) if attr in INTROSPECTION:
                return
            case ast.Call(func=ast.Name(id="setattr"), args=[value, ast.Constant(value=key), *_]) if (
                value is node and key in self.proof.moved
            ):
                self.refuse(parent, "sets what a call binds")
        super().require_new_class(node, parent)


@dataclass(frozen=True)
class NewClassCall(ClassMemberAnswersABC[CreationCode]):
    """A method creation calls on the new class, resolved along its declared MRO.

    The move leaves that resolution unchanged: the new owner binds only names it
    takes from the composer's own body, which came first in the MRO already.
    """

    reads: CreationReads
    call: ast.Call

    def assigned(
        self, owner: CompactIndexedClass, declaration: CompactClassMemberDeclaration
    ) -> CreationCode:
        self.reads.refuse(self.call, f"calls a value `{owner.qualname}` assigns, not a method it defines")

    def annotated_property(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedProperty
    ) -> CreationCode:
        return self.defined(owner, declaration)

    def annotated_method(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedMethod
    ) -> CreationCode:
        return self.defined(owner, declaration)

    def unannotated_callable(
        self, owner: CompactIndexedClass, declaration: CompactUnannotatedCallable
    ) -> CreationCode:
        return self.defined(owner, declaration)

    def defined(
        self, owner: CompactIndexedClass, declaration: CompactCallableMemberDeclaration
    ) -> CreationCode:
        code = self.reads.proof.method(owner, declaration.name)
        if code is None:
            self.reads.refuse(self.call, f"calls `{declaration.name}`, which `{owner.qualname}` binds without a def")
        return code

    def missing(self) -> CreationCode:
        self.reads.refuse(self.call, "calls a method no class in its MRO declares")

    def ambiguous(self) -> CreationCode:
        self.reads.refuse(self.call, "calls a method its first declaring class binds twice")

    def open(self) -> CreationCode:
        authority = self.reads.proof.class_index.declared_mro_authority
        mro = authority.resolve(self.reads.proof.composer.symbol).mro_type
        outside = (
            [entry.__name__ for entry in mro.__mro__ if authority.bases.is_opaque(entry)]
            if mro is not None
            else []
        )
        self.reads.refuse(
            self.call,
            f"calls a method no scanned class declares; the MRO reaches {outside or 'an unresolved base'} "
            "outside the scan",
        )
