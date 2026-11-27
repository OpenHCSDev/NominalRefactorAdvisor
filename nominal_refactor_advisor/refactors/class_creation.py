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
import inspect
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from abc import ABC, abstractmethod
from typing import ClassVar, NoReturn
from importlib import import_module

from metaclass_registry.core import AutoRegisterMeta

from nominal_refactor_advisor.ast_field_decoders import optional_values
from nominal_refactor_advisor.ast_tools import AstParentIndex, ParsedModule
from nominal_refactor_advisor.class_index import (
    ClassMemberAnswersABC,
    CompactAnnotatedMethod,
    CompactAnnotatedProperty,
    CompactCallableMemberDeclaration,
    CompactClassMemberDeclaration,
    CompactIndexedClass,
    CompactUnannotatedCallable,
)
from nominal_refactor_advisor.class_creation_store import MODELLED_METACLASSES, REGISTRY_NAMES
from nominal_refactor_advisor.class_mro import native_root_type
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput



HEADER_BASES = "__orig_bases__"
"""The one name under which a class's own namespace holds its header's bases:
creation sets it when a base is subscripted."""

ORDER_READERS = frozenset(("dataclasses.dataclass", "builtins.dir"))
"""Functions of a class that read the classes of its order and their own bodies,
never the bases its header lists."""

INTROSPECTION = frozenset(
    ("__mro__", "__bases__", "__orig_bases__", "__dict__", "__annotations__", "__annotate__", "mro")
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


@dataclass(frozen=True)
class FunctionDecorator:
    """What a modelled decorator makes of the value it decorates: a value of
    `returns` (None: the same value), with `stores` set on it."""

    returns: type | None
    stores: frozenset[str] = frozenset()


FUNCTION_DECORATORS = {
    "abc.abstractmethod": FunctionDecorator(None, frozenset(("__isabstractmethod__",))),
    "builtins.classmethod": FunctionDecorator(classmethod),
    "builtins.staticmethod": FunctionDecorator(staticmethod),
    "builtins.property": FunctionDecorator(property),
    "functools.cached_property": FunctionDecorator(cached_property),
    "typing.final": FunctionDecorator(None, frozenset(("__final__",))),
    "typing.override": FunctionDecorator(None, frozenset(("__override__",))),
}
"""The decorators whose result NRA knows: the attributes a value they build
from a plain function has are its type's and the ones they store."""


@dataclass(frozen=True)
class FunctionReference:
    """One place a module may refer to a function by its name."""

    node: ast.AST
    parents: AstParentIndex

    def decorates(self) -> bool:
        """`@function`: the decorated value is the call's only argument."""

        match self.parents.get(self.node):
            case ast.FunctionDef(decorator_list=decorators) | ast.AsyncFunctionDef(
                decorator_list=decorators
            ) | ast.ClassDef(decorator_list=decorators):
                return any(decorator is self.node for decorator in decorators)
        return False

    def called_to_decorate(self) -> bool:
        """`@factory(...)`: what the call returns receives the decorated value."""

        match self.parents.get(self.node):
            case ast.Call(func=func) as call if func is self.node:
                return FunctionReference(call, self.parents).decorates()
        return False

    def returned_by(self, factory: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """`return function`, a statement of `factory`'s own body."""

        match self.parents.get(self.node):
            case ast.Return(value=value) as statement if value is self.node:
                return any(own is statement for own in factory.body)
        return False

    def exported(self) -> bool:
        """Whether a constant is an element of the module's `__all__`."""

        sequence = self.parents.get(self.node)
        match sequence, self.parents.get(sequence) if sequence is not None else None:
            case (
                ast.List() | ast.Tuple(),
                ast.Assign(targets=[ast.Name(id="__all__")], value=value)
                | ast.AugAssign(target=ast.Name(id="__all__"), value=value)
                | ast.AnnAssign(target=ast.Name(id="__all__"), value=value),
            ):
                return value is sequence
        return False


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
    """Composing new owners leaves the composer's creation unchanged."""

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
            self.reads(code, roles).require()

    def reads(self, code: CreationCode, roles: dict[str, Role]) -> CreationReads:
        """How one creation function's reads are judged: here, a new owner joins the bases."""

        return CreationReads(self, code, roles)

    @classmethod
    def dropping(
        cls,
        snapshot: CodemodSourceSnapshot,
        composer: CompactIndexedClass,
        ancestry: tuple[CompactIndexedClass, ...],
        dropped: tuple[str, ...],
    ) -> ClassCreationProof:
        """The proof that dropping restated bases leaves the class's creation as it
        is: its order does not change, so only creation code reading `bases` sees it."""

        return BaseDropCreationProof(
            snapshot,
            composer,
            ancestry,
            " + ".join(f"`{symbol.rpartition('.')[2]}`" for symbol in dropped),
            frozenset(),
            frozenset(),
            MroWalk.unmoved(ancestry),
            dropped,
        )

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

    def carriers(
        self, order: tuple[str, ...], composed: bool, attributes: frozenset[str], stored: frozenset[str]
    ) -> tuple[str, ...]:
        """The names a first-found walk of `order` meets whose winning value may
        have one of `attributes`, in the order it meets them; `stored` are those
        scanned code may store on a function a class body declares."""

        found: dict[str, bool] = {}
        for symbol in order:
            if native_root_type(symbol) is not None:
                continue
            indexed = self.walked(symbol)
            for name, statement in self.own_bindings(symbol, composed):
                found.setdefault(name, self.may_carry(indexed, statement, attributes, stored))
        return tuple(name for name, carries in found.items() if carries)

    def walked(self, symbol: str) -> CompactIndexedClass:
        """A class the walk of the composer's MRO visits: the scan's or an installed one."""

        indexed = self.snapshot.class_overlap_verdicts.ancestor(symbol)
        if indexed is None:
            raise NeedsInput(
                f"`{symbol}` has no declaration NRA can read, and the walk of "
                f"`{self.composer.qualname}`'s MRO reads its names",
                NeededInput.PROOF,
            )
        return indexed

    def own_bindings(self, symbol: str, composed: bool) -> Iterator[tuple[str, ast.stmt]]:
        """Each name a scanned class's own body puts in `vars()`, with the
        statement binding it, in binding order; once composed, the composer's
        body no longer binds the names it deletes."""

        indexed = self.walked(symbol)
        for statement in self.runtime_body(indexed, self.class_node(indexed).body):
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
            if isinstance(statement, ast.AnnAssign) and not optional_values(statement):
                continue
            for name in names:
                if not (composed and symbol == self.walk.composer and name in self.walk.deleted):
                    yield name, statement

    def runtime_body(self, indexed: CompactIndexedClass, body: list[ast.stmt]) -> Iterator[ast.stmt]:
        """The statements of a class body that run: `if TYPE_CHECKING:` is false
        whenever the module is imported, so its body declares names for a type
        checker and binds none; what its `else` holds runs in its place."""

        for statement in body:
            if isinstance(statement, ast.If) and self.resolved(indexed, statement.test, statement.lineno) in (
                "typing.TYPE_CHECKING", "typing_extensions.TYPE_CHECKING"
            ):
                yield from self.runtime_body(indexed, statement.orelse)
            else:
                yield statement

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

    def is_subclass_of(self, symbol: str, target_symbol: str) -> bool:
        """Whether `symbol` is `target_symbol` or descends from it along known bases.
        Refuses (via `NeedsInput`) if any reachable ancestor has incomplete base references."""

        if symbol == target_symbol:
            return True
        if native_root_type(symbol) is not None:
            return False
        pending = [symbol]
        visited: set[str] = set()
        while pending:
            current_sym = pending.pop()
            if current_sym in visited:
                continue
            visited.add(current_sym)
            if current_sym == target_symbol:
                return True
            if native_root_type(current_sym) is not None:
                continue
            indexed = self.walked(current_sym)
            if not indexed.base_references_are_complete:
                raise NeedsInput(
                    f"`{indexed.qualname}` has unresolved bases, so NRA cannot prove "
                    f"whether it subclasses `{target_symbol}`",
                    NeededInput.PROOF,
                )
            for reference in indexed.base_references:
                base_sym = reference.qualified_name
                if base_sym == target_symbol:
                    return True
                if native_root_type(base_sym) is None and base_sym not in visited:
                    pending.append(base_sym)
        return False

    def require_subclass_walk_kept(
        self,
        filter_symbol: str,
        read_attrs: frozenset[str],
        node: ast.AST,
        reads: CreationReads,
    ) -> None:
        """A walk over `cls.__mro__` that enters its body only when
        `issubclass(ancestor, filter_symbol)` sees the exact same classes in the
        exact same order when filtering `before` and `after` yields the same tuple
        and none of the attributes read off `ancestor` is a moved name."""

        moved_read = sorted(read_attrs & self.moved)
        if moved_read:
            reads.refuse(node, f"reads moved attribute(s) {moved_read} off MRO subclasses")
        before = tuple(s for s in self.walk.before if self.is_subclass_of(s, filter_symbol))
        after = tuple(s for s in self.walk.after if self.is_subclass_of(s, filter_symbol))
        if before != after:
            reads.refuse(
                node,
                f"visits `{filter_symbol}` subclasses {list(before)}, and {list(after)} once composed,",
            )

    def _statement_is_plain_callable(
        self, indexed: CompactIndexedClass, statement: ast.stmt
    ) -> bool:
        """Whether `statement` binds a plain Python function or modelled decorator
        from `FUNCTION_DECORATORS`, which cannot be an instance of a user-defined
        non-builtin class."""

        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            return False
        return all(
            self.resolved(indexed, decorator, decorator.lineno) in FUNCTION_DECORATORS
            for decorator in statement.decorator_list
        )

    def _type_filtered_bindings(
        self, order: tuple[str, ...], composed: bool, filter_symbol: str
    ) -> tuple[tuple[str, str], ...]:
        """`(symbol, name)` pairs along `order` whose bound value could be an instance
        of `filter_symbol` (non-builtin classes cannot match plain functions/descriptors
        or primitive literals)."""

        is_builtin_filter = filter_symbol.startswith("builtins.")
        matches: list[tuple[str, str]] = []
        for symbol in order:
            if native_root_type(symbol) is not None:
                continue
            indexed = self.walked(symbol)
            for name, statement in self.own_bindings(symbol, composed):
                if not is_builtin_filter:
                    if self._statement_is_plain_callable(indexed, statement):
                        continue
                    match statement:
                        case ast.Assign(value=ast.Constant()) | ast.AnnAssign(value=ast.Constant()):
                            continue
                matches.append((symbol, name))
        return tuple(matches)

    def require_type_filtered_dict_walk_kept(
        self, filter_symbol: str, node: ast.AST, reads: CreationReads
    ) -> None:
        """A reversed MRO walk updating a dict with `{k: v for k, v in base.__dict__.items() if isinstance(v, T)}`
        produces the exact same dict in the exact same order when the sequence of
        `(symbol, name)` bindings that could match `T` is identical before and after."""

        before = self._type_filtered_bindings(tuple(reversed(self.walk.before)), False, filter_symbol)
        after = self._type_filtered_bindings(tuple(reversed(self.walk.after)), True, filter_symbol)
        if before != after:
            reads.refuse(
                node,
                f"collects `{filter_symbol}` bindings {list(before)}, and {list(after)} once composed,",
            )

    def _class_dir_keys(self, symbol: str, composed: bool) -> frozenset[str]:
        """The keys `type.__dir__` collects from `symbol.__dict__`."""

        indexed = self.walked(symbol)
        class_def = self.class_node(indexed)
        keys: set[str] = {"__module__", "__doc__", "__firstlineno__", "__static_attributes__"}
        for name, _ in self.own_bindings(symbol, composed):
            keys.add(name)
        has_annotation = any(
            isinstance(statement, ast.AnnAssign)
            and not (
                composed
                and symbol == self.walk.composer
                and declared_names(statement).issubset(self.walk.deleted)
            )
            for statement in self.runtime_body(indexed, class_def.body)
        )
        if "__slots__" not in keys:
            keys.update(("__dict__", "__weakref__"))
        if has_annotation:
            keys.update(("__annotations__", "__annotate_func__"))
        if any(isinstance(base, ast.Subscript) for base in class_def.bases):
            keys.add("__orig_bases__")
        return frozenset(keys)

    def require_dir_kept(self, node: ast.AST, reads: CreationReads) -> None:
        """`dir(cls)` returns the sorted union of `c.__dict__.keys()` along `cls.__mro__`
        when no metaclass overrides `__dir__`. Keeping all old MRO classes and matching
        `composer`'s pre-move keys against `composer`'s post-move keys plus every plain
        gained base's keys proves `dir(cls)` is identical before and after."""

        for symbol in self.metaclasses():
            for indexed in self.metaclass_ancestry(symbol):
                if self.method(indexed, "__dir__"):
                    reads.refuse(node, f"lists attributes through `{indexed.qualname}`'s own `__dir__`")
        before_set = frozenset(self.walk.before)
        after_set = frozenset(self.walk.after)
        if not before_set.issubset(after_set):
            reads.refuse(node, "drops classes from the MRO that `dir()` reads")
        gained = tuple(symbol for symbol in self.walk.after if symbol not in before_set)
        for symbol in gained:
            indexed = self.walked(symbol)
            class_def = self.class_node(indexed)
            if class_def.decorator_list or class_def.keywords:
                reads.refuse(
                    node, f"gained base `{indexed.qualname}` has decorators or class keywords affecting `dir()`"
                )
            if not indexed.base_references_are_complete or any(
                native_root_type(ref.qualified_name) is None for ref in indexed.base_references
            ):
                reads.refuse(node, f"gained base `{indexed.qualname}` has non-native bases affecting `dir()`")
            for name, statement in self.own_bindings(symbol, composed=True):
                if name in self.moved or self._statement_is_plain_callable(indexed, statement):
                    continue
                match statement:
                    case ast.Assign(value=ast.Constant()) | ast.AnnAssign(value=ast.Constant()):
                        continue
                reads.refuse(node, f"gained base `{indexed.qualname}` binds non-trivial `{name}` affecting `dir()`")
        composer_before = self._class_dir_keys(self.walk.composer, composed=False)
        composer_after = self._class_dir_keys(self.walk.composer, composed=True)
        gained_keys = frozenset().union(*(self._class_dir_keys(symbol, composed=True) for symbol in gained))
        diff = composer_before ^ (composer_after | gained_keys)
        if not diff:
            return
        unchanged = tuple(symbol for symbol in self.walk.before if symbol != self.walk.composer)
        unresolved: list[str] = []
        for key in sorted(diff):
            if key in ("__dict__", "__weakref__"):
                if any(
                    native_root_type(symbol) is None
                    and "__slots__" not in {name for name, _ in self.own_bindings(symbol, composed=False)}
                    for symbol in unchanged
                ):
                    continue
            elif any(
                key in vars(native)
                if (native := native_root_type(symbol)) is not None
                else any(bound == key for bound, _ in self.own_bindings(symbol, composed=False))
                for symbol in unchanged
            ):
                continue
            unresolved.append(key)
        if unresolved:
            reads.refuse(node, f"`dir()` changes attribute set by {unresolved}")

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

    def may_carry(
        self,
        indexed: CompactIndexedClass,
        statement: ast.stmt,
        attributes: frozenset[str],
        stored: frozenset[str],
    ) -> bool:
        """Whether the value one statement of `indexed`'s body binds may have one
        of `attributes`. A function has its type's, those `stored` on functions,
        and what each modelled decorator applied to it makes of it; a literal
        only its own type's. Anything else may have any."""

        match statement:
            case ast.FunctionDef(decorator_list=decorators) | ast.AsyncFunctionDef(decorator_list=decorators):
                kind: type = types.FunctionType
                carried = set(stored)
                for decorator in reversed(decorators):
                    modelled = self.function_decorator(indexed, decorator)
                    if modelled is None:
                        return True
                    kind = modelled.returns or kind
                    carried |= modelled.stores
                return any(hasattr(kind, attribute) or attribute in carried for attribute in attributes)
            case ast.Assign(value=value) | ast.AnnAssign(value=value) if value is not None:
                try:
                    literal = ast.literal_eval(value)
                except ValueError:
                    return True
                return any(hasattr(literal, attribute) for attribute in attributes)
        return True

    def function_decorator(self, indexed: CompactIndexedClass, decorator: ast.expr) -> FunctionDecorator | None:
        """The model of a decorator a scanned class body names, resolved where the
        body runs; none for a call, a name the body binds, or a class outside the scan."""

        if self.class_index.class_for(indexed.symbol) is None:
            return None
        match decorator:
            case ast.Name() | ast.Attribute():
                body = self.class_node(indexed).body
                qualified = self.resolved(
                    indexed,
                    decorator,
                    decorator.lineno,
                    frozenset(name for statement in body for name in declared_names(statement)),
                )
                return None if qualified is None else FUNCTION_DECORATORS.get(qualified)
        return None

    def stored_on_functions(self, attributes: frozenset[str]) -> frozenset[str]:
        """Those of `attributes` scanned code may store on a function a class body
        declares: each `X.<attribute> = ...`, `setattr(X, "<attribute>", ...)` and
        `X.__dict__["<attribute>"] = ...` whose `X` is not the function a decorator
        receives. A store under a name no constant spells is not seen."""

        stored: set[str] = set()
        for module in self.snapshot.parsed_modules:
            if not any(attribute in module.source for attribute in attributes):
                continue
            parents = module.syntax_index.parent_index
            for node in ast.walk(module.module):
                match node:
                    case (
                        ast.Attribute(value=subject, attr=str(attribute), ctx=ast.Store())
                        | ast.Call(
                            func=ast.Name(id="setattr"), args=[subject, ast.Constant(value=str(attribute)), *_]
                        )
                        | ast.Subscript(
                            value=ast.Attribute(value=subject, attr="__dict__")
                            | ast.Call(func=ast.Name(id="vars"), args=[subject]),
                            slice=ast.Constant(value=str(attribute)),
                            ctx=ast.Store(),
                        )
                    ) if attribute in attributes and not self.decorated_parameter(subject, parents):
                        stored.add(attribute)
        return frozenset(stored)

    def decorated_parameter(self, subject: ast.expr, parents: AstParentIndex) -> bool:
        """Whether `subject` is what a decorator receives: the first parameter,
        never rebound, of an undecorated function applied only as `@function`, or
        returned by an undecorated factory applied only as `@factory(...)`, read
        in that function's own body, outside any lambda."""

        function = parents.enclosing_function(subject)
        if function is None:
            return False
        ancestors = parents.ancestors(subject)
        within = (subject, *ancestors[: ancestors.index(function)])
        if any(isinstance(node, ast.Lambda) for node in within) or not any(
            statement is within[-1] for statement in function.body
        ):
            return False
        match subject, function:
            case ast.Name(id=parameter), ast.FunctionDef(decorator_list=[]) | ast.AsyncFunctionDef(
                decorator_list=[]
            ) if parameter == self.first_parameter(function):
                pass
            case _:
                return False
        for node in ast.walk(function):
            match node:
                case (
                    ast.Name(id=name, ctx=ast.Store() | ast.Del())
                    | ast.FunctionDef(name=name)
                    | ast.AsyncFunctionDef(name=name)
                    | ast.ClassDef(name=name)
                    | ast.ExceptHandler(name=str(name))
                    | ast.MatchAs(name=str(name))
                    | ast.MatchStar(name=str(name))
                    | ast.MatchMapping(rest=str(name))
                ) if name == parameter and node is not function:
                    return False
                case ast.alias(name=imported, asname=renamed) if (renamed or imported.partition(".")[0]) == parameter:
                    return False
                case ast.Global(names=names) | ast.Nonlocal(names=names) if parameter in names:
                    return False
        uses = tuple(self.uses(function.name))
        if all(use.decorates() for use in uses):
            return True
        match parents.get(function):
            case ast.FunctionDef(decorator_list=[]) | ast.AsyncFunctionDef(decorator_list=[]) as factory:
                return all(use.returned_by(factory) for use in uses) and all(
                    use.called_to_decorate() for use in self.uses(factory.name)
                )
        return False

    def uses(self, name: str) -> Iterator[FunctionReference]:
        """Each place scanned code may refer to a function by `name`: a name, an
        attribute, a renaming import, or a constant string outside `__all__`."""

        for module in self.snapshot.parsed_modules:
            if name not in module.source:
                continue
            parents = module.syntax_index.parent_index
            for node in ast.walk(module.module):
                reference = FunctionReference(node, parents)
                match node:
                    case ast.Name(id=found) | ast.Attribute(attr=found) if found == name:
                        yield reference
                    case ast.alias(name=found, asname=str(renamed)) if (
                        found.rpartition(".")[2] == name and renamed != name
                    ):
                        yield reference
                    case ast.Constant(value=str(found)) if found == name and not reference.exported():
                        yield reference

    def require_walk_kept(self, attributes: frozenset[str], node: ast.AST, reads: CreationReads) -> None:
        """A walk reading only `attributes` of each name's winner answers the same
        when those winners keep their values and the names whose winners may
        carry one of them keep their order. The name check of the composed
        ancestry keeps every value; this keeps the order."""

        if any(hasattr(kind, attribute) for kind in _IMPLICIT_VALUES for attribute in attributes):
            reads.refuse(node, "reads an attribute what class creation binds by itself may have")
        stored = self.stored_on_functions(attributes)
        before = self.carriers(self.walk.before, False, attributes, stored)
        after = self.carriers(self.walk.after, True, attributes, stored)
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

    def module(self, indexed: CompactIndexedClass) -> ParsedModule:
        """Its module: the scan's, or else the installed module's source, read and
        never run. A class along the ancestry that an installed package declares
        (Textual's `DOMNode`, a metaclass a library ships) has creation code and
        names to resolve as a scanned one has; the snapshot holds only the scan."""

        if self.class_index.class_for(indexed.symbol) is not None:
            return self.snapshot.parsed_module_for_source_path(indexed.file_path)
        parsed = InstalledModule.parsed(indexed.module_name)
        if parsed is None:
            raise NeedsInput(
                f"{indexed.file_path}:{indexed.line} `{indexed.qualname}`'s module has no source NRA can read",
                NeededInput.PROOF,
            )
        return parsed

    def installed_packages(self) -> frozenset[str]:
        """The installed packages that declare a class of the ancestry or of its
        metaclasses' ancestry: the code outside the scan that runs as the class is
        created and may keep it."""

        classes = [*self.ancestry]
        for metaclass in self.metaclasses():
            classes.extend(self.metaclass_ancestry(metaclass))
        return frozenset(
            indexed.module_name.partition(".")[0]
            for indexed in classes
            if self.class_index.class_for(indexed.symbol) is None
        )

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

        return len(self.module(indexed).source.splitlines()) + 1

    def qualified(self, indexed: CompactIndexedClass, reference: ast.expr, line: int) -> str:
        qualified = self.resolved(indexed, reference, line)
        if qualified is None:
            raise NeedsInput(
                f"{indexed.file_path}:{line} `{ast.unparse(reference)}` is unbound at class creation",
                NeededInput.PROOF,
            )
        return qualified

    def resolved(
        self,
        indexed: CompactIndexedClass,
        reference: ast.expr,
        line: int,
        class_bound: frozenset[str] = frozenset(),
    ) -> str | None:
        """What `reference` names in `indexed`'s module at `line`, unless it is
        unbound there or one of `class_bound`, which a class body binds first."""

        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            self.module(indexed),
            reference,
            line=line,
            preceding_class_bound_names=class_bound,
        )
        return None if witness is None else self.canonical(witness.qualified_name)

    def canonical(self, qualified_name: str) -> str:
        """The defining module's spelling of an importable name; a scanned name as is."""

        if self.class_index.class_for(qualified_name) is not None:
            return qualified_name
        module_name, _, attribute = qualified_name.rpartition(".")
        try:
            value = getattr(import_module(module_name or "builtins"), attribute)
            return f"{value.__module__}.{value.__qualname__}"
        except (ImportError, AttributeError):
            # Not importable, or a plain value (`typing.TYPE_CHECKING`): the name as spelled.
            return qualified_name

    def metaclass_code(self, symbol: str) -> Iterator[CreationCode]:
        """The creation methods along a metaclass's ancestry."""

        hooks = CREATION_METHODS | self.modelled_hooks()
        for indexed in self.metaclass_ancestry(symbol):
            yield from self.methods(indexed, hooks)

    def metaclass_ancestry(self, symbol: str) -> Iterator[CompactIndexedClass]:
        """The classes along a metaclass's ancestry, the scan's or an installed
        module's source; a modelled metaclass ends its branch, and one NRA cannot
        read refuses."""

        pending, seen = [symbol], set()
        while pending:
            current = self.canonical(pending.pop())
            if current in seen or current in MODELLED_METACLASSES:
                continue
            seen.add(current)
            indexed = self.snapshot.class_overlap_verdicts.ancestor(current)
            if indexed is None:
                raise NeedsInput(
                    f"`{self.composer.qualname}`'s metaclass `{current}` has no source NRA can read "
                    "and is none of its creation models: NRA needs a proof of what it reads",
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
    texts: dict[str, str] = field(default_factory=dict)
    """The parameters every followed call hands one constant text."""

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
                case ast.For(iter=iterated, target=ast.Name(id=target)) | ast.comprehension(
                    iter=iterated, target=ast.Name(id=target)
                ) if (element := self.walked(iterated)) is not None:
                    self.roles[target] = element
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

    def walked(self, iterable: ast.expr) -> Role | None:
        """What a loop over a value read from a role yields, when these reads follow
        it: nothing here, where each read of a role's name is judged where it stands."""

        return None

    def constant_key(self, node: ast.AST | None) -> bool:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in self.proof.moved
        ):
            return True
        if isinstance(node, ast.Name):
            return self.constant_loop_key(node)
        return False

    def constant_loop_key(self, node: ast.Name) -> bool:
        """Whether `node` is a comprehension or `for`-loop variable ranging strictly
        over literal string keys none of which is moved."""

        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for generator in reversed(current.generators):
                    if isinstance(generator.target, ast.Name) and generator.target.id == node.id:
                        return self._literal_unmoved_strings(generator.iter)
            elif isinstance(current, ast.For):
                if isinstance(current.target, ast.Name) and current.target.id == node.id:
                    rebound = any(
                        isinstance(other, ast.Name)
                        and other.id == node.id
                        and isinstance(other.ctx, (ast.Store, ast.Del))
                        and other is not current.target
                        for other in ast.walk(current)
                    )
                    return not rebound and self._literal_unmoved_strings(current.iter)
            current = self.parents.get(current)
        return False

    def _literal_unmoved_strings(self, expr: ast.expr) -> bool:
        match expr:
            case ast.Tuple(elts=elts) | ast.List(elts=elts) | ast.Set(elts=elts) if elts:
                return all(
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and elt.value not in self.proof.moved
                    for elt in elts
                )
        return False

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

    def dispatch_table(self, call: ast.Call, base: str) -> ast.ClassDef | None:
        """The class `X` of `X.handler_for(type(base))(base)` on the base `base` names."""

        match call:
            case ast.Call(
                func=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=table), attr="handler_for"),
                    args=[ast.Call(func=ast.Name(id="type"), args=[ast.Name(id=dispatched)])],
                ),
                args=[ast.Name(id=passed)],
            ) if dispatched == passed == base:
                owner = self.code.owner if table in ("cls", "self") else None
                return None if owner is None else self.proof.class_node(owner)
        return None

    @staticmethod
    def handlers(table: ast.ClassDef) -> Iterator[tuple[tuple[str, ...], ast.FunctionDef]]:
        """Each `@handles(...)` method of a dispatch table, with what it handles."""

        for statement in table.body:
            if isinstance(statement, ast.FunctionDef):
                for decorator in statement.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "handles"
                    ):
                        yield tuple(ast.unparse(argument) for argument in decorator.args), statement

    def answer(self, handler: ast.FunctionDef) -> ast.expr | None:
        """The literal a handler returns, when it does nothing else."""

        body = [
            item for item in handler.body
            if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
        ]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if value is None:
                return ast.Constant(None)
            if self.is_literal(value):
                return value
        return None

    def plain_answer(self, table: ast.ClassDef) -> ast.expr | None:
        return next(
            (self.answer(handler) for names, handler in self.handlers(table) if names == ("type",)),
            None,
        )

    def constant_plain_dispatch(self, call: ast.Call, base: str) -> bool:
        """`X.handler_for(type(base))(base)` where `X`'s handler for `type` answers a constant."""

        table = self.dispatch_table(call, base)
        return table is not None and self.plain_answer(table) is not None

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
            case ast.Attribute(attr="__mro__") if (
                self.first_found_walk(parent)
                or self.binding_walk(parent)
                or self.subclass_filtered_walk(parent)
                or self.reversed_type_filtered_dict_walk(parent)
            ):
                return
            case ast.Attribute(attr="__dict__", ctx=ast.Load()):
                # `cls.__dict__` is the new class's own namespace, identical to `vars(cls)`.
                self.require_namespace(parent, self.parents.get(parent))
                return
            case ast.Attribute(attr=attr) if attr in INTROSPECTION:
                self.refuse(parent, "introspects the new class")
            case ast.Attribute(attr=attr):
                call = self.parents.get(parent)
                if isinstance(call, ast.Call) and call.func is parent:
                    for code in self.called(call, attr):
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

    def called(self, call: ast.Call, attr: str) -> tuple[CreationCode, ...]:
        """The functions a call of `attr` on the new class may run: the one its
        declared order gives it."""

        return (
            self.proof.class_index.declared_member(self.proof.composer.symbol, attr).answer(
                NewClassCall(self, call)
            ),
        )

    def is_builtin(self, func: ast.Name, qualified_name: str) -> bool:
        """Whether `func` names this builtin where the function body runs: its
        module's final binding of the name, in a scanned module or an installed
        one's source."""

        owner = self.code.owner
        return (
            owner is not None
            and self.proof.resolved(owner, func, self.proof.module_end(owner)) == qualified_name
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
        """`(c for c in cls.__mro__ if "<name>" in vars(c))`, or
        `for c in cls.__mro__ [if ... else ...]: if name in c.__dict__: ...`
        where `name` is a constant string (or constant-passed parameter):
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
                            left=key_expr,
                            ops=[ast.In()],
                            comparators=[
                                ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner)], keywords=[])
                                | ast.Attribute(value=ast.Name(id=owner), attr="__dict__")
                            ],
                        )
                    ],
                    is_async=0,
                ),
                ast.GeneratorExp(elt=ast.Name(id=kept), generators=[_])
                | ast.ListComp(elt=ast.Name(id=kept), generators=[_])
                | ast.SetComp(elt=ast.Name(id=kept), generators=[_]),
            ) if walked is attribute and owner == visited and kept == visited and (name := self.text(key_expr)) is not None:
                self.proof.require_selection_kept(name, attribute, self)
                return True
        walk = generator
        if isinstance(walk, ast.IfExp) and walk.body is attribute:
            walk = self.parents.get(walk)
        match walk:
            case ast.For(
                target=ast.Name(id=visited),
                body=[
                    ast.If(
                        test=ast.Compare(
                            left=key_expr,
                            ops=[ast.In()],
                            comparators=[
                                ast.Attribute(value=ast.Name(id=owner), attr="__dict__")
                                | ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner)], keywords=[])
                            ],
                        ),
                        body=if_body,
                        orelse=[],
                    )
                ],
                orelse=[],
            ) if owner == visited and (name := self.text(key_expr)) is not None:
                if (leaked := self._leaked_loop_read(walk, (visited,))) is not None:
                    self.refuse(leaked, "reads what the binding walk of the MRO left behind")
                for stmt in if_body:
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Name) and node.id == visited:
                            node_parent = self.parents.get(node)
                            match node_parent:
                                case ast.Attribute(value=v, attr="__dict__") if v is node:
                                    sub = self.parents.get(node_parent)
                                    match sub:
                                        case ast.Subscript(value=d, slice=slice_expr) if (
                                            d is node_parent and self.text(slice_expr) == name
                                        ):
                                            continue
                                        case _:
                                            return False
                                case ast.Attribute(value=v, attr=attr) if v is node and attr in INTROSPECTION:
                                    return False
                                case ast.Call(func=ast.Name() as fn, args=[v], keywords=[]) if v is node and self.is_builtin(fn, "builtins.vars"):
                                    return False
                self.proof.require_selection_kept(name, attribute, self)
                return True
        return False

    def subclass_filtered_walk(self, attribute: ast.Attribute) -> bool:
        """`for ancestor in cls.__mro__: if issubclass(ancestor, Filter): ...`
        or `(base for base in cls.__mro__ if issubclass(base, Filter))`:
        only classes descending from `Filter` are kept or enter the loop body."""

        walk = self.parents.get(attribute)
        match walk:
            case ast.Call(func=ast.Name() as rev_fn, args=[arg], keywords=[]) if (
                arg is attribute and self.is_builtin(rev_fn, "builtins.reversed")
            ):
                walk = self.parents.get(walk)
        if isinstance(walk, ast.comprehension):
            comprehension = self.parents.get(walk)
            match walk, comprehension:
                case (
                    ast.comprehension(
                        target=ast.Name(id=visited),
                        ifs=[
                            ast.Call(
                                func=ast.Name() as check_fn,
                                args=[ast.Name(id=tested), filter_expr],
                                keywords=[],
                            )
                        ],
                        is_async=0,
                    ),
                    ast.GeneratorExp(elt=ast.Name(id=kept), generators=[_])
                    | ast.ListComp(elt=ast.Name(id=kept), generators=[_])
                    | ast.SetComp(elt=ast.Name(id=kept), generators=[_]),
                ) if tested == visited == kept and self.is_builtin(check_fn, "builtins.issubclass"):
                    owner = self.code.owner
                    if owner is None:
                        return False
                    filter_symbol = self.proof.resolved(owner, filter_expr, self.proof.module_end(owner))
                    if filter_symbol is None:
                        return False
                    self.proof.require_subclass_walk_kept(filter_symbol, frozenset(), attribute, self)
                    return True
            return False
        match walk:
            case ast.For(
                target=ast.Name(id=visited),
                body=[
                    ast.If(
                        test=ast.Call(
                            func=ast.Name() as check_fn,
                            args=[ast.Name(id=tested), filter_expr],
                            keywords=[],
                        ),
                        body=if_body,
                        orelse=[],
                    )
                ],
                orelse=[],
            ) if tested == visited and self.is_builtin(check_fn, "builtins.issubclass"):
                pass
            case _:
                return False
        assert isinstance(walk, ast.For)
        owner = self.code.owner
        if owner is None:
            return False
        filter_symbol = self.proof.resolved(owner, filter_expr, self.proof.module_end(owner))
        if filter_symbol is None:
            return False
        if (leaked := self._leaked_loop_read(walk, (visited,))) is not None:
            self.refuse(leaked, "reads what the subclass-filtered MRO walk left behind")
        read_attrs: set[str] = set()
        for stmt in if_body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and node.id == visited:
                    node_parent = self.parents.get(node)
                    match node_parent:
                        case ast.Attribute(value=v, attr=attr) if v is node and attr in INTROSPECTION:
                            return False
                        case ast.Attribute(value=v, attr=attr) if v is node:
                            read_attrs.add(attr)
                        case ast.Call(func=ast.Name() as fn, args=[v], keywords=[]) if v is node and self.is_builtin(fn, "builtins.vars"):
                            return False
        self.proof.require_subclass_walk_kept(filter_symbol, frozenset(read_attrs), attribute, self)
        return True

    def reversed_type_filtered_dict_walk(self, attribute: ast.Attribute) -> bool:
        """`for base in reversed(cls.__mro__): d.update({k: v for k, v in base.__dict__.items() if isinstance(v, T)})`."""

        rev_call = self.parents.get(attribute)
        match rev_call:
            case ast.Call(func=ast.Name() as rev_fn, args=[arg], keywords=[]) if (
                arg is attribute and self.is_builtin(rev_fn, "builtins.reversed")
            ):
                pass
            case _:
                return False
        walk = self.parents.get(rev_call)
        match walk:
            case ast.For(
                target=ast.Name(id=visited),
                iter=walked,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(attr="update"),
                            args=[
                                ast.DictComp(
                                    key=ast.Name(id=k_out),
                                    value=ast.Name(id=v_out),
                                    generators=[
                                        ast.comprehension(
                                            target=ast.Tuple(elts=[ast.Name(id=k_in), ast.Name(id=v_in)]),
                                            iter=ast.Call(
                                                func=ast.Attribute(
                                                    value=ast.Attribute(value=ast.Name(id=owner_name), attr="__dict__")
                                                    | ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner_name)], keywords=[]),
                                                    attr="items",
                                                ),
                                                args=[],
                                                keywords=[],
                                            ),
                                            ifs=[
                                                ast.Call(
                                                    func=ast.Name() as inst_fn,
                                                    args=[ast.Name(id=v_test), filter_expr],
                                                    keywords=[],
                                                )
                                            ],
                                            is_async=0,
                                        )
                                    ],
                                )
                            ],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            ) if (
                walked is rev_call
                and owner_name == visited
                and k_out == k_in
                and v_out == v_in == v_test
                and self.is_builtin(inst_fn, "builtins.isinstance")
            ):
                pass
            case _:
                return False
        assert isinstance(walk, ast.For)
        owner = self.code.owner
        if owner is None:
            return False
        filter_symbol = self.proof.resolved(owner, filter_expr, self.proof.module_end(owner))
        if filter_symbol is None:
            return False
        if (leaked := self._leaked_loop_read(walk, (visited,))) is not None:
            self.refuse(leaked, "reads what the reversed MRO walk left behind")
        self.proof.require_type_filtered_dict_walk_kept(filter_symbol, attribute, self)
        return True

    def _leaked_loop_read(self, walk: ast.For, names: tuple[str, ...]) -> ast.Name | None:
        """A `Load` of any loop variable in `names` outside `walk` that is not
        shadowed by an enclosing comprehension or another `for` loop's body."""

        walked_nodes = frozenset(ast.walk(walk))
        for node in ast.walk(self.code.node):
            match node:
                case ast.Name(id=name, ctx=ast.Load()) if name in names and node not in walked_nodes:
                    if not self.comprehension_binds(node) and not self._enclosing_for_body_binds(node, walk):
                        return node
        return None

    def _enclosing_for_body_binds(self, node: ast.Name, ignored_walk: ast.For) -> bool:
        """Whether `node` sits inside the body of a different `for` loop whose
        target binds `node.id` (so `node` reads that loop's target, not `ignored_walk`)."""

        child: ast.AST = node
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, ast.For) and current is not ignored_walk:
                if child in current.body and any(
                    isinstance(bound, ast.Name) and bound.id == node.id
                    for bound in ast.walk(current.target)
                ):
                    return True
            child = current
            current = self.parents.get(current)
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
        match call:
            case ast.Call(args=[arg], keywords=[]) if (
                role is NEW_CLASS
                and arg is node
                and self.code.owner is not None
                and self.is_builtin(call.func, "builtins.dir")
            ):
                self.proof.require_dir_kept(call, self)
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
        handed = [(names[index], argument) for index, argument in enumerate(call.args) if index < len(names)]
        handed.extend((keyword.arg, keyword.value) for keyword in call.keywords if keyword.arg in names)
        texts = {
            name: text for name, argument in handed if (text := self.text(argument)) is not None
        }
        type(self)(self.proof, code, mapped, self.depth + 1, texts).require()

    def text(self, node: ast.AST | None) -> str | None:
        """The one text `node` holds: a constant, or a parameter the call that reached
        this function handed a constant and the function never rebinds."""

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.texts and not any(
            isinstance(other, ast.Name) and other.id == node.id and isinstance(other.ctx, (ast.Store, ast.Del))
            for other in ast.walk(self.code.node)
        ):
            return self.texts[node.id]
        return None

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
            case ast.Name(id=local) if (stored := self.local_container(local)) is not None:
                return self.builtin_container(stored, owner, line)
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

    def local_container(self, name: str) -> ast.expr | None:
        """The value the creation function binds its own local `name` to, when these
        reads follow what a local container keeps: not here, where what is stored in
        one is read again by code these reads do not see."""

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
        """The function a call reaches, in the scan or an installed module's source,
        or True for a modelled one."""

        match func:
            case ast.Attribute(value=ast.Name(id=receiver), attr=attr):
                if receiver in ("cls", "mcs", "self") and self.code.owner is not None:
                    return self.proof.method(self.code.owner, attr)
                owner = self.code.owner
                if owner is None:
                    return None
                # A function body looks a global up when it runs: the module's final binding.
                # A receiver the module does not bind is the function's own local.
                qualified = self.proof.resolved(owner, func.value, self.proof.module_end(owner))
                if qualified is None:
                    return None
                if qualified in MODELLED_METACLASSES:
                    return True
                indexed = self.proof.snapshot.class_overlap_verdicts.ancestor(qualified)
                return self.proof.method(indexed, attr) if indexed is not None else None
        return None


@dataclass(frozen=True)
class BaseDropCreationProof(ClassCreationProof):
    """Dropping restated bases leaves the class's creation unchanged."""

    dropped: tuple[str, ...] = ()
    """The symbols of the bases the drop removes."""

    @cached_property
    def header_is_subscripted(self) -> bool:
        """Whether the class's header subscripts a base: creation then keeps the
        header's bases in the class's own namespace, as `__orig_bases__`."""

        return any(isinstance(base, ast.Subscript) for base in self.class_node(self.composer).bases)

    def reads(self, code: CreationCode, roles: dict[str, Role]) -> CreationReads:
        return BaseDropReads(self, code, roles)


@dataclass
class BaseDropReads(CreationReads):
    """One creation function's reads of a base list a drop shortens: truthiness and
    passing it on stay the same; reading any one base could meet the dropped one."""

    proof: BaseDropCreationProof

    @property
    def unproved(self) -> str:
        return f"NRA needs a proof that dropping {self.proof.owner_name} leaves it unchanged"

    def require_base(self, node: ast.Name, parent: ast.AST | None) -> None:
        """A base's metaclass dispatched on, and the answers flattened: a dropped
        native base reaches the handler for `type`, and an empty answer adds nothing."""

        dispatch = parent
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "type":
            handler = self.parents.get(parent)
            dispatch = self.parents.get(handler) if handler is not None else None
        flattened = isinstance(dispatch, ast.Call) and (
            isinstance(loop := self.parents.get(dispatch), ast.comprehension) and loop.iter is dispatch
        )
        table = self.dispatch_table(dispatch, node.id) if isinstance(dispatch, ast.Call) else None
        if flattened and table is not None and self.dropped_reach_empty_plain_answer(table):
            return
        self.refuse(parent or node, "reads a base of the class, which may be a dropped one")

    def follow(self, call: ast.Call, node: ast.Name, role: Role) -> None:
        """`dataclasses.dataclass(cls, ...)` on the new class builds its fields from
        each class's own annotations along the order, which the drop keeps, and never
        reads the header's bases; `dir(cls)` is the names of every class in the order;
        anything else is followed as before."""

        owner = self.code.owner
        if (
            role is NEW_CLASS
            and owner is not None
            and call.args[:1] == [node]
            and self.proof.resolved(owner, call.func, self.proof.module_end(owner)) in ORDER_READERS
        ):
            return
        super().follow(call, node, role)

    def require_new_class(self, node: ast.Name, parent: ast.AST | None) -> None:
        """The drop keeps the class's order and its own body, so a read of either is
        the read it was: `__mro__` walked class by class, each class judged as the new
        one is (it may be it); `__annotations__`; and its namespace, which holds the
        header's bases only as `__orig_bases__`, set when a base is subscripted."""

        match parent:
            case ast.Attribute(attr="__mro__", ctx=ast.Load()) if self.walked_by(parent) is not None:
                return
            case ast.Attribute(attr="__annotations__" | "__annotate__", ctx=ast.Load()):
                return
            case ast.Attribute(attr="__dict__", ctx=ast.Load()) if (
                not self.proof.header_is_subscripted or self.asks_no_header_bases(parent)
            ):
                return
            case ast.Yield() | ast.Tuple(ctx=ast.Load()) if isinstance(
                parent if isinstance(parent, ast.Yield) else self.parents.get(parent), ast.Yield | ast.Return
            ):
                # Handed back to the caller, alone or in a tuple, as a `return` hands it back.
                return
            case ast.GeneratorExp(elt=kept) | ast.ListComp(elt=kept) | ast.SetComp(elt=kept) if kept is node:
                # Kept in what the comprehension builds: the classes of the order it selects.
                return
            case ast.Call(func=ast.Name(id="setattr"), args=[value, _, _], keywords=[]) if value is node:
                # A store on the class, whatever the name: it reads nothing of it.
                return
            case ast.keyword(value=value) if value is node and isinstance(
                call := self.parents.get(parent), ast.Call
            ) and self.only_stores(call):
                return
            case ast.Call(func=ast.Name(id="getattr" | "hasattr"), args=[value, ast.Name(id=key), *_]) if (
                value is node and self.namespace_key(key)
            ):
                return
            case ast.Call() if node in parent.args and self.only_stores(parent):
                return
            case ast.Call(func=ast.Name() as func, args=[value], keywords=[]) if (
                value is node
                and self.is_builtin(func, "builtins.vars")
                and (not self.proof.header_is_subscripted or self.asks_no_header_bases(parent))
            ):
                return
        super().require_new_class(node, parent)

    def called(self, call: ast.Call, attr: str) -> tuple[CreationCode, ...]:
        """The drop keeps the class's order, its metaclass and every name creation
        binds, so the call runs the function it ran. When the declarations do not
        single it out (a metaclass's own method, called on the class it built; a
        name creation code could have stored), it is one of the functions of that
        name the order and the metaclasses' order define, and each is read."""

        try:
            return super().called(call, attr)
        except NeedsInput:
            defined = [
                code for indexed in self.proof.ancestry if (code := self.proof.method(indexed, attr)) is not None
            ]
            for metaclass in self.proof.metaclasses():
                defined.extend(
                    code
                    for indexed in self.proof.metaclass_ancestry(metaclass)
                    if (code := self.proof.method(indexed, attr)) is not None
                )
            if not defined:
                raise
            return tuple(defined)

    def namespace_key(self, name: str) -> bool:
        """Whether `name` holds only names some class of the order binds in its own
        namespace: the one variable of a loop over `vars(c)` or `c.__dict__`, their
        keys or the names of their items, for a class `c` of the order. A namespace
        holds the header's bases as `__orig_bases__` alone, and without a subscripted
        base that attribute is an ancestor's, which the drop leaves as it is."""

        stores = [
            node for node in ast.walk(self.code.node)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
        ]
        if len(stores) != 1:
            return False
        target: ast.AST = stores[0]
        loop = self.parents.get(target)
        if isinstance(loop, ast.Tuple) and loop.elts and loop.elts[0] is target and len(loop.elts) == 2:
            target, loop, items = loop, self.parents.get(loop), True
        else:
            items = False
        if not isinstance(loop, (ast.For, ast.comprehension)) or loop.target is not target:
            return False
        walked = loop.iter
        match walked:
            case ast.Call(func=ast.Attribute(value=namespace, attr="items"), args=[], keywords=[]) if items:
                pass
            case ast.Call(func=ast.Attribute(value=namespace, attr="keys"), args=[], keywords=[]) if not items:
                pass
            case namespace if not items:
                pass
            case _:
                return False
        match namespace:
            case ast.Call(func=ast.Name(id="vars"), args=[ast.Name(id=owner)], keywords=[]):
                pass
            case ast.Attribute(value=ast.Name(id=owner), attr="__dict__"):
                pass
            case _:
                return False
        if self.roles.get(owner) is not NEW_CLASS:
            return False
        return not self.proof.header_is_subscripted or self.skips_header_bases(loop, "items" if items else "keys")

    def local_container(self, name: str) -> ast.expr | None:
        """The value the creation function binds its own local `name` to, when it
        binds it once and not as a parameter: `found = []`, `seen: set[type] = set()`.
        What such a container keeps is read again as a class, and the one read of a
        class the drop changes is of its bases, which `BasesReaders` judges wherever
        it stands."""

        arguments = self.code.node.args
        if any(
            argument.arg == name
            for argument in (
                *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                *(extra for extra in (arguments.vararg, arguments.kwarg) if extra is not None),
            )
        ):
            return None
        stores = [
            node for node in ast.walk(self.code.node)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
        ]
        if len(stores) != 1:
            return None
        match self.parents.get(stores[0]):
            case ast.Assign(targets=[target], value=value) | ast.AnnAssign(target=target, value=value) if (
                target is stores[0] and value is not None
            ):
                return value
        return None

    def only_stores(self, call: ast.Call) -> bool:
        """Whether the call constructs a value that only keeps what it is handed: a
        class NRA reads, none of whose order writes its own `__new__`, `__init__` or
        `__post_init__` and which names no metaclass, so `object` or the dataclass
        decorator's generated `__init__` assigns each argument to a field. The class
        is the one the function's module names, or the `cls` of a classmethod."""

        owner = self.code.owner
        if owner is None or not isinstance(call.func, ast.Name):
            return False
        constructed: CompactIndexedClass | None
        if call.func.id == ClassCreationProof.first_parameter(self.code.node) and any(
            isinstance(decorator, ast.Name) and decorator.id == "classmethod"
            for decorator in self.code.node.decorator_list
        ):
            constructed = owner
        else:
            qualified = self.proof.resolved(owner, call.func, self.proof.module_end(owner))
            constructed = (
                None if qualified is None else self.proof.snapshot.class_overlap_verdicts.ancestor(qualified)
            )
        if constructed is None:
            return False
        pending, seen = [constructed], set()
        while pending:
            current = pending.pop()
            if current.symbol in seen:
                continue
            seen.add(current.symbol)
            node = self.proof.class_node(current)
            if node.keywords or any(
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name in ("__new__", "__init__", "__post_init__")
                for statement in node.body
            ):
                return False
            for reference in current.base_references:
                qualified_name = ".".join(reference.resolved_parts)
                if native_root_type(qualified_name) is not None:
                    continue
                base = self.proof.snapshot.class_overlap_verdicts.ancestor(qualified_name)
                if base is None:
                    return False
                pending.append(base)
        return True

    def require_namespace(self, node: ast.Name, parent: ast.AST | None) -> None:
        """The namespace the class body filled is the same body's: creation adds the
        header's bases to it, as `__orig_bases__`, only when a base is subscripted."""

        if not self.proof.header_is_subscripted or self.asks_no_header_bases(node):
            return
        super().require_namespace(node, parent)

    def iterated(self, loop: ast.For | ast.comprehension, role: Role) -> Role | None:
        if role is NAMESPACE and (
            not self.proof.header_is_subscripted or self.skips_header_bases(loop, "keys")
        ):
            return None
        return super().iterated(loop, role)

    def asks_no_header_bases(self, namespace: ast.AST) -> bool:
        """Whether this read of a class namespace is the same whatever its
        `__orig_bases__` holds. The header subscripts a base before the drop and
        after it, so the key is there either way and asking whether a key is among
        the names answers the same; a read of a value asks for another key or only
        whether the value is an instance of a class no tuple is; a walk of the
        entries passes that one by (`skips_header_bases`)."""

        parent = self.parents.get(namespace)
        match parent:
            case ast.Subscript(value=value, slice=key) if value is namespace:
                return self.other_key(key) or self.only_tested(parent)
            case ast.Compare(ops=[ast.In() | ast.NotIn()], comparators=[value]) if value is namespace:
                return True
            case ast.Attribute(attr="get"):
                call = self.parents.get(parent)
                return isinstance(call, ast.Call) and call.func is parent and bool(call.args) and self.other_key(
                    call.args[0]
                )
            case ast.Attribute(attr="items" | "values" | "keys" as view):
                call = self.parents.get(parent)
                loop = self.parents.get(call) if call is not None else None
                return (
                    isinstance(call, ast.Call)
                    and call.func is parent
                    and not call.args
                    and not call.keywords
                    and isinstance(loop, (ast.For, ast.comprehension))
                    and loop.iter is call
                    and (self.skips_header_bases(loop, view) or self.finds_no_header_bases(loop))
                )
            case ast.For() | ast.comprehension() if parent.iter is namespace:
                return self.skips_header_bases(parent, "keys")
        return False

    def finds_no_header_bases(self, inner: ast.For | ast.comprehension) -> bool:
        """Whether the walk is the first-found walk of the order (`first_found_walk`),
        whose winners are read only through `getattr(winner, "<name>", default)`, and
        no tuple has any of those names: the winner for `__orig_bases__` answers each
        read with its default, whatever the tuple holds."""

        outer = self.parents.get(inner)
        if not isinstance(inner, ast.For) or not isinstance(outer, ast.For) or outer.body != [inner]:
            return False
        walked = outer.iter
        if not (isinstance(walked, ast.Attribute) and walked.attr == "__mro__") or not self.first_found_walk(walked):
            return False
        match inner.body:
            case [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=found), attr="setdefault")))]:
                return all(not hasattr((), attribute) for attribute in self.found_reads(found, outer))
        return False

    def only_tested(self, read: ast.AST) -> bool:
        """Whether the value read is used for one thing: `isinstance(<read>, C)` for a
        class NRA reads that names no metaclass, which is false of every tuple."""

        match self.parents.get(read):
            case ast.Call(func=ast.Name(id="isinstance"), args=[tested, kind], keywords=[]) if tested is read:
                return self.plain_class(kind)
        return False

    def other_key(self, key: ast.AST | None) -> bool:
        """The key is never `__orig_bases__`: a constant, a parameter handed one, the
        variable of a loop over a display of constants, or a local bound once to a
        text with a constant start or end `__orig_bases__` has not."""

        text = self.text(key)
        if text is not None:
            return text != HEADER_BASES
        if not isinstance(key, ast.Name):
            return False
        stores = [
            node for node in ast.walk(self.code.node)
            if isinstance(node, ast.Name) and node.id == key.id and isinstance(node.ctx, ast.Store)
        ]
        if len(stores) != 1:
            return False
        match self.parents.get(stores[0]):
            case ast.Assign(targets=[target], value=ast.JoinedStr(values=[first, *_] as parts)) if (
                target is stores[0]
            ):
                last = parts[-1]
                prefix = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else ""
                suffix = last.value if isinstance(last, ast.Constant) and isinstance(last.value, str) else ""
                return bool(prefix or suffix) and not (
                    HEADER_BASES.startswith(prefix) and HEADER_BASES.endswith(suffix)
                )
            case ast.For(target=target, iter=ast.Tuple(elts=texts) | ast.List(elts=texts) | ast.Set(elts=texts)) | (
                ast.comprehension(target=target, iter=ast.Tuple(elts=texts) | ast.List(elts=texts) | ast.Set(elts=texts))
            ) if target is stores[0]:
                return all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value != HEADER_BASES
                    for item in texts
                )
        return False

    def skips_header_bases(self, loop: ast.For | ast.comprehension, view: str) -> bool:
        """Whether a walk of a namespace's entries does nothing with the entry
        `__orig_bases__`, whose value is a tuple before the drop and after it.

        Everything the walk does with an entry stands behind one test, the
        comprehension's first condition or the `if` that is the loop's whole body, and
        the test's first term is false for that entry: `isinstance(value, C)` for a
        class NRA reads that names no metaclass, `getattr(value, "<name>", <false>)`
        for a name a tuple has not, `callable(value)`, or `name.startswith("<text>")`
        for a text `__orig_bases__` does not start with."""

        match view, loop.target:
            case "items", ast.Tuple(elts=[ast.Name(id=key), ast.Name(id=value)]):
                pass
            case "values", ast.Name(id=value):
                key = None
            case "keys", ast.Name(id=key):
                value = None
            case _:
                return False
        if isinstance(loop, ast.comprehension):
            guard = loop.ifs[0] if loop.ifs else None
        else:
            match loop.body:
                case [ast.If(test=guard, orelse=[])] if not loop.orelse:
                    pass
                case _:
                    return False
            inside = frozenset(ast.walk(loop))
            if any(
                isinstance(node, ast.Name) and node.id in (key, value) and node not in inside
                for node in ast.walk(self.code.node)
            ):
                return False
        if guard is None:
            return False
        first = guard.values[0] if isinstance(guard, ast.BoolOp) and isinstance(guard.op, ast.And) else guard
        owner = self.code.owner
        match first:
            case ast.Call(func=ast.Name(id="isinstance"), args=[ast.Name(id=tested), kind], keywords=[]) if (
                tested == value
            ):
                return self.plain_class(kind)
            case ast.Call(
                func=ast.Name(id="getattr"),
                args=[ast.Name(id=tested), ast.Constant(value=str() as attribute), ast.Constant(value=absent)],
                keywords=[],
            ) if tested == value:
                return not absent and not hasattr((), attribute)
            case ast.Call(func=ast.Name(id="callable"), args=[ast.Name(id=tested)], keywords=[]) if tested == value:
                return True
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=tested), attr="startswith"), args=[prefix], keywords=[]
            ) if tested == key and owner is not None:
                text = self.text(prefix)
                if text is None and isinstance(prefix, ast.Name):
                    bound = self.local_container(prefix.id)
                    text = bound.value if isinstance(bound, ast.Constant) and isinstance(bound.value, str) else None
                return text is not None and not HEADER_BASES.startswith(text)
        return False

    def plain_class(self, kind: ast.expr) -> bool:
        """Whether `kind` names a class NRA reads whose order names no metaclass:
        `isinstance` then asks its real order, which no tuple is in."""

        owner = self.code.owner
        if owner is None:
            return False
        qualified = self.proof.resolved(owner, kind, self.proof.module_end(owner))
        verdicts = self.proof.snapshot.class_overlap_verdicts
        found = None if qualified is None else verdicts.ancestor(qualified)
        if found is None:
            return False
        pending, seen = [found], set()
        while pending:
            current = pending.pop()
            if current.symbol in seen:
                continue
            seen.add(current.symbol)
            if any(keyword.arg == "metaclass" for keyword in self.proof.class_node(current).keywords):
                return False
            for reference in current.base_references:
                qualified_name = ".".join(reference.resolved_parts)
                if native_root_type(qualified_name) is not None:
                    continue
                base = verdicts.ancestor(qualified_name)
                if base is None:
                    return False
                pending.append(base)
        return True

    def walked_by(self, order: ast.Attribute) -> ast.For | ast.comprehension | None:
        """The loop that walks `<new class>.__mro__` one class at a time: straight,
        `reversed`, or as one arm of a conditional that walks other classes otherwise."""

        walked: ast.AST = order
        around = self.parents.get(walked)
        while True:
            match around:
                case ast.Call(func=ast.Name(id="reversed"), args=[only], keywords=[]) if only is walked:
                    pass
                case ast.IfExp(body=body, orelse=orelse) if walked is body or walked is orelse:
                    pass
                case _:
                    break
            walked, around = around, self.parents.get(around)
        if isinstance(around, (ast.For, ast.comprehension)) and around.iter is walked and isinstance(
            around.target, ast.Name
        ):
            return around
        return None

    def walked(self, iterable: ast.expr) -> Role | None:
        """A class of the new class's order, which may be the new class itself."""

        match iterable:
            case ast.Call(func=ast.Name(id="reversed"), args=[inner], keywords=[]):
                return self.walked(inner)
            case ast.IfExp(body=body, orelse=orelse):
                return self.walked(body) or self.walked(orelse)
            case ast.Attribute(value=ast.Name(id=name), attr="__mro__") if self.roles.get(name) is NEW_CLASS:
                return NEW_CLASS
        return None

    def dropped_reach_empty_plain_answer(self, table: ast.ClassDef) -> bool:
        answer = self.plain_answer(table)
        if answer is None or ast.literal_eval(answer) not in ((), [], None, frozenset(), set()):
            return False
        metaclasses = tuple(
            type(native) for symbol in self.proof.dropped if (native := native_root_type(symbol)) is not None
        )
        if len(metaclasses) != len(self.proof.dropped):
            return False
        ancestors = frozenset(
            f"{ancestor.__module__}.{ancestor.__qualname__}"
            for metaclass in metaclasses
            for ancestor in inspect.getmro(metaclass)
            if ancestor not in (type, object)
        )
        return not any(
            self.handled_symbol(name, handler) in ancestors
            for names, handler in self.handlers(table)
            if names != ("type",)
            for name in names
        )

    def handled_symbol(self, name: str, handler: ast.FunctionDef) -> str:
        """The qualified name a handler's `@handles` argument spells in its module."""

        owner = self.code.owner
        if owner is None:
            return name
        module = self.proof.module(owner)
        witness = self.proof.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            module, ast.parse(name, mode="eval").body, line=handler.lineno
        )
        return name if witness is None else witness.qualified_name


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



