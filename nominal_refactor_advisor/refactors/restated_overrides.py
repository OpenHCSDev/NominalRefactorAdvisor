"""Delete a member that restates the definition its class would inherit anyway.

`class BackToModel(WatchEnd)` declaring `settled` exactly as `WatchEnd` does
keeps a second copy of one answer: without it, lookup reaches `WatchEnd.settled`,
the same declaration bound to the same names. A copy is the same only when
the class-body normal form says so (kind, syntax with the docstring dropped,
and what each free name is bound to) and nothing in it reads the class it sits
in (`ClassBodyNormalForm.reads_its_class`: the class cell, a mangled private
name, a name the class body binds). What can still see the deletion is code
that reads the class namespace: creation code is read by `ClassCreationProof`,
and every other `vars(X)` or `X.__dict__` read must either be of a named class
whose order does not hold the class, or ask for one constant key it is not.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import partial
from typing import ClassVar

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.class_index import PER_CLASS_NAMES, CompactIndexedClass
from nominal_refactor_advisor.class_mro import native_root_type
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.probe_owner import ReceiverSpan
from nominal_refactor_advisor.refactor_concepts import RestatedOverrideConcept
from nominal_refactor_advisor.refactors.class_creation import (
    ClassCreationProof,
    CreationCode,
    CreationReads,
    MroWalk,
    Role,
)
from nominal_refactor_advisor.refactors.class_overlap import (
    OverlapClass,
    RestatedDeclarations,
    obsolete_import_edits,
    require_static_c3,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.class_attribute_reads import ClassAttributeReaders


def inherited_restatements(verdicts: ClassOverlapVerdicts, symbol: str) -> tuple[tuple[str, str], ...]:
    """Each member `symbol` declares exactly as the next class in its C3 order
    declaring that name does, with that class: what lookup reaches once the
    member goes. A member is not one when either copy reads its class, when the
    next binder is native or outside the scan, or when its order is unknown; an
    annotated declaration is not one either, since `__annotations__`, like
    `__slots__`, is read from each class's own body."""

    if symbol not in verdicts.bodies.declarations_by_symbol:
        return ()
    try:
        order = verdicts.linearized(symbol)
    except NeedsInput:
        return ()
    found: list[tuple[str, str]] = []
    for name, mine in verdicts.bodies.declarations_of(symbol).items():
        if name in PER_CLASS_NAMES or mine.declaration.reads_its_class or mine.kind.is_field:
            continue
        inherited = next_binder(verdicts, order[1:], name)
        if inherited is None:
            continue
        theirs = verdicts.bodies.declarations_of(inherited)[name]
        if not theirs.declaration.reads_its_class and mine.same_declaration_as(theirs):
            found.append((name, inherited))
    return tuple(found)


def next_binder(verdicts: ClassOverlapVerdicts, order: tuple[str, ...], name: str) -> str | None:
    """The scanned class of `order` whose own body first declares `name`; none
    when a native or unscanned class comes first, since its namespace is not
    the class-body index's."""

    for symbol in order:
        native = native_root_type(symbol)
        if native is not None:
            if name in vars(native):
                return None
            continue
        if verdicts.class_index.class_for(symbol) is None or symbol not in verdicts.bodies.declarations_by_symbol:
            return None
        if name in verdicts.bodies.declarations_of(symbol):
            return symbol
    return None


@dataclass(frozen=True)
class NamespaceReads(ParsedModuleSetFact):
    """Every read of a class namespace, `vars(X)` or `X.__dict__`, in one state's
    modules, with the node it is read in; an instance's own `self.__dict__` holds
    no class member and is no such read."""

    reads: tuple[tuple[ParsedModule, ast.expr, ast.AST | None, ast.AST | None], ...]
    """Each module, the read, the node it is read in and the node around that.
    A read of a parameter annotated as an instance (`instance_annotation`) reads
    that instance's own `__dict__`, which holds no class member; any other read
    is left for the type checker (`NamespaceReaders.require_unread`)."""

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> NamespaceReads:
        found: list[tuple[ParsedModule, ast.expr, ast.AST | None, ast.AST | None]] = []
        for module in modules:
            reads = tuple(cls.found(module.module))
            if not reads:
                continue
            imported_modules = cls.imported_module_names(module.module)
            parents = {
                child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)
            }
            for read in reads:
                if (
                    cls.reads_an_instance(read, parents)
                    or cls.reads_imported_module(read, parents, imported_modules)
                    or cls.mutates_instance_dict(read, parents)
                ):
                    continue
                parent = parents.get(read)
                found.append((module, read, parent, None if parent is None else parents.get(parent)))
        return cls(tuple(found))

    INSTANCE_DICT_MUTATORS = frozenset(("setdefault", "pop", "popitem", "update", "clear"))

    @classmethod
    def mutates_instance_dict(cls, read: ast.expr, parents: dict[ast.AST, ast.AST]) -> bool:
        """A class's `__dict__` is a read-only `mappingproxy`, which has no mutating
        methods (`setdefault`, `pop`, `update`, `clear`) and forbids item assignment or
        deletion; any mutation of `X.__dict__` therefore targets an instance dictionary."""

        parent = parents.get(read)
        match parent:
            case ast.Subscript(value=target, ctx=ast.Store() | ast.Del()) if target is read:
                return True
            case ast.Attribute(value=target, attr=method) if (
                target is read and method in cls.INSTANCE_DICT_MUTATORS
            ):
                grandparent = parents.get(parent)
                return isinstance(grandparent, ast.Call) and grandparent.func is parent
        return False

    @staticmethod
    def imported_module_names(tree: ast.Module) -> frozenset[str]:
        """Top-level `import mod` / `import mod as alias` names that the module body never rebinds."""

        imported = {
            alias.asname or alias.name.partition(".")[0]
            for statement in tree.body
            if isinstance(statement, ast.Import)
            for alias in statement.names
        }
        if not imported:
            return frozenset()
        rebound = {
            node.id
            for statement in tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        }
        return frozenset(imported - rebound)

    @staticmethod
    def reads_imported_module(
        read: ast.expr,
        parents: dict[ast.AST, ast.AST],
        imported_modules: frozenset[str],
    ) -> bool:
        """Whether `vars(mod)` or `mod.__dict__` reads an unshadowed imported module."""

        match read:
            case ast.Attribute(value=ast.Name(id=name)) | ast.Call(args=[ast.Name(id=name)]) if (
                name in imported_modules
            ):
                pass
            case _:
                return False
        scope = parents.get(read)
        while scope is not None:
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = scope.args
                if any(
                    arg.arg == name
                    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
                ) or (args.vararg is not None and args.vararg.arg == name) or (
                    args.kwarg is not None and args.kwarg.arg == name
                ):
                    return False
                if any(
                    isinstance(node, ast.Name)
                    and node.id == name
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    for node in ast.walk(scope)
                ):
                    return False
            elif isinstance(scope, ast.ClassDef):
                if any(
                    isinstance(node, ast.Name)
                    and node.id == name
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    for statement in scope.body
                    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    for node in ast.walk(statement)
                ):
                    return False
            scope = parents.get(scope)
        return True

    @staticmethod
    def reads_an_instance(read: ast.expr, parents: dict[ast.AST, ast.AST]) -> bool:
        match read:
            case ast.Attribute(value=ast.Name(id=name)) | ast.Call(args=[ast.Name(id=name)]):
                pass
            case _:
                return False
        scope = parents.get(read)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = parents.get(scope)
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        arguments = scope.args
        annotations = [
            argument.annotation
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            if argument.arg == name
        ] + [
            statement.annotation
            for statement in ast.walk(scope)
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ]
        return bool(annotations) and all(
            annotation is not None and NamespaceReads.instance_annotation(annotation) for annotation in annotations
        )

    @staticmethod
    def instance_annotation(annotation: ast.expr) -> bool:
        """An annotation naming a class, not `type[...]`: its values are instances.
        `object` and `Any` admit classes too, so they prove nothing."""

        outer = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        match outer:
            case (
                ast.Name(id="type" | "Type" | "object" | "Any")
                | ast.Attribute(attr="type" | "Type" | "Any")
                | ast.Constant()
            ):
                return False
        return True

    @staticmethod
    def receiver(read: ast.expr) -> tuple[ast.expr, ...]:
        """The expression whose namespace `X.__dict__` or `vars(X)` reads."""

        match read:
            case ast.Attribute(value=receiver) | ast.Call(args=[receiver]):
                return (receiver,)
        return ()

    @staticmethod
    def found(tree: ast.AST) -> Iterator[ast.expr]:
        for node in ast.walk(tree):
            match node:
                case ast.Attribute(value=ast.Name(id="self")) | ast.Call(args=[ast.Name(id="self")]):
                    continue
                case ast.Attribute(value=receiver, attr="__dict__", ctx=ast.Load()) | ast.Call(
                    func=ast.Name(id="vars"), args=[receiver], keywords=[]
                ) if NamespaceReads.names_a_module(receiver):
                    continue
                case ast.Attribute(attr="__dict__", ctx=ast.Load()):
                    yield node
                case ast.Call(func=ast.Name(id="vars"), args=[_], keywords=[]):
                    yield node

    @staticmethod
    def names_a_module(receiver: ast.expr) -> bool:
        """`sys.modules[...]` is a module, whose namespace holds no class member."""

        match receiver:
            case ast.Subscript(value=ast.Attribute(value=ast.Name(id="sys"), attr="modules")):
                return True
        return False


@dataclass(frozen=True)
class AnnotationNames(ParsedModuleSetFact):
    """Every name an annotation in one state's modules spells, string annotations
    included: the only names a namespace passed as `localns` to `get_type_hints`
    can answer."""

    names: frozenset[str]

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> AnnotationNames:
        names: set[str] = set()
        for module in modules:
            for node in ast.walk(module.module):
                annotations: list[ast.expr | None] = []
                match node:
                    case ast.AnnAssign(annotation=annotation):
                        annotations.append(annotation)
                    case ast.arg(annotation=annotation):
                        annotations.append(annotation)
                    case ast.FunctionDef(returns=returns) | ast.AsyncFunctionDef(returns=returns):
                        annotations.append(returns)
                for annotation in annotations:
                    if annotation is not None:
                        names.update(cls.spelled(annotation))
        return cls(frozenset(names))

    @classmethod
    def spelled(cls, annotation: ast.expr) -> Iterator[str]:
        for node in ast.walk(annotation):
            match node:
                case ast.Name(id=name):
                    yield name
                case ast.Constant(value=str() as text):
                    try:
                        parsed = ast.parse(text, mode="eval").body
                    except SyntaxError:
                        yield from ()
                    else:
                        yield from cls.spelled(parsed)


ANNOTATION_SLOTS = frozenset({"__annotations__", "__annotate__"})


@dataclass(frozen=True)
class OwnAnnotationKeys:
    """Which names, at one read of `vars(R)` inside a function, only hold names
    `R`'s own annotations declare. `R` keeps one value around the read: a
    parameter never rebound, or the target of the one loop it is bound by. The
    key is bound only by loops the read sits in, each iterating `R`'s own
    annotations within that value of `R`: `get_annotations(R, ...)` or
    `R.__annotations__`, through `tuple`, `list`, `sorted`, `reversed`, `iter` or
    `enumerate` (the second target), or a name assigned one of those as a
    statement of that region before the read."""

    scope: ast.FunctionDef | ast.AsyncFunctionDef
    parents: dict[ast.AST, ast.AST]
    read: ast.expr
    receiver: str
    region: ast.FunctionDef | ast.AsyncFunctionDef | ast.For | ast.AsyncFor

    WRAPPERS: ClassVar[frozenset[str]] = frozenset({"tuple", "list", "sorted", "reversed", "iter"})

    @classmethod
    def hold(cls, module: ParsedModule, read: ast.expr, key: str) -> bool:
        """Whether `key`, at this read, only holds names its receiver's own
        annotations declare."""

        parents = {child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)}
        scope: ast.AST | None = parents.get(read)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents.get(scope)
        match scope, NamespaceReads.receiver(read):
            case ast.FunctionDef() | ast.AsyncFunctionDef(), (ast.Name(id=receiver),):
                return any(
                    keys.holds(key) for keys in cls(scope, parents, read, receiver, scope).bound_once()
                )
        return False

    def bound_once(self) -> tuple[OwnAnnotationKeys, ...]:
        """This reader with `region` the node within which `R` keeps one value,
        when there is one."""

        stores = self.stores(self.receiver)
        if self.receiver in self.parameters():
            return () if stores else (self,)
        match stores:
            case [store]:
                loop = self.parents.get(store)
                if isinstance(loop, (ast.For, ast.AsyncFor)) and loop.target is store and self.inside(loop.body, self.read):
                    return (replace(self, region=loop),)
        return ()

    def holds(self, key: str) -> bool:
        stores = self.stores(key)
        return bool(stores) and key not in self.parameters() and all(self.loop_binds(store) for store in stores)

    def loop_binds(self, store: ast.Name) -> bool:
        target = self.parents.get(store)
        if isinstance(target, ast.Tuple) and len(target.elts) == 2 and target.elts[1] is store:
            loop = self.parents.get(target)
            if not (isinstance(loop, (ast.For, ast.AsyncFor)) and loop.target is target):
                return False
            match loop.iter:
                case ast.Call(func=ast.Name(id="enumerate"), args=[keys], keywords=[]):
                    pass
                case _:
                    return False
        elif isinstance(target, (ast.For, ast.AsyncFor)) and target.target is store:
            loop, keys = target, target.iter
        else:
            return False
        return self.inside(loop.body, self.read) and self.inside([self.region], loop) and self.annotations(keys, loop)

    def annotations(self, keys: ast.expr, loop: ast.stmt) -> bool:
        match keys:
            case ast.Call(func=ast.Name(id=wrapper), args=[inner], keywords=[]) if wrapper in self.WRAPPERS:
                return self.annotations(inner, loop)
            case ast.Call(func=ast.Name(id="get_annotations") | ast.Attribute(attr="get_annotations"), args=[ast.Name(id=owner), *_]):
                return owner == self.receiver and self.inside(self.region_body(), keys)
            case ast.Attribute(value=ast.Name(id=owner), attr="__annotations__"):
                return owner == self.receiver and self.inside(self.region_body(), keys)
            case ast.Name(id=name) if name not in self.parameters():
                stores = self.stores(name)
                body = self.region_body()
                reading = next((index for index, statement in enumerate(body) if self.inside([statement], loop)), None)
                return bool(stores) and reading is not None and all(
                    isinstance(assigned := self.parents.get(store), ast.Assign)
                    and assigned.targets == [store]
                    and assigned in body[:reading]
                    and self.annotations(assigned.value, loop)
                    for store in stores
                )
        return False

    def region_body(self) -> list[ast.stmt]:
        return self.region.body

    def parameters(self) -> frozenset[str]:
        arguments = self.scope.args
        return frozenset(
            argument.arg
            for argument in (
                *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                *(extra for extra in (arguments.vararg, arguments.kwarg) if extra is not None),
            )
        )

    def stores(self, name: str) -> list[ast.Name]:
        return [
            node
            for node in ast.walk(self.scope)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
        ]

    @staticmethod
    def inside(statements: list[ast.stmt] | list[ast.AST], node: ast.AST) -> bool:
        return any(inner is node for statement in statements for inner in ast.walk(statement))


@dataclass(frozen=True)
class NamespaceReaders(ClassAttributeReaders):
    """Every read of a class namespace in the repository, judged against the
    members a deletion takes from one class."""

    def require_unread(self, indexed: CompactIndexedClass, deleted: frozenset[str]) -> None:
        """Every namespace read the syntax does not settle is asked of the type
        checker, in one batch: a receiver whose every value is an instance that is
        not a class reads that instance's own `__dict__`, and one typed `type[C]`
        reads only `C` or a subclass, never a class that does not subclass `C`."""

        open_reads = {
            ReceiverSpan.of(module.file_path, receiver): (module, read, parent)
            for module, read, parent, around in self.snapshot.module_set.fact(NamespaceReads).reads
            if self.reaches(module, read, indexed)
            and not self.asks_another_key(module, read, parent, around, indexed, deleted)
            and not self.resolves_annotations_only(module, read, deleted)
            for receiver in NamespaceReads.receiver(read)
        }
        if not open_reads:
            return
        answers = self.snapshot.type_checker.receiver_types(open_reads)

        def where(module: ParsedModule, read: ast.expr, parent: ast.AST | None) -> str:
            return (
                f"{module.file_path}:{read.lineno} `{ast.unparse(parent or read)}` reads a class "
                f"namespace, and deleting {', '.join(f'`{name}`' for name in sorted(deleted))} from "
                f"`{indexed.qualname}` could change what it sees"
            )

        NeedsInput.every(
            *(
                partial(
                    answers[span].require_namespace_unseen,
                    where(module, read, parent),
                    partial(self.family_holds, indexed=indexed),
                )
                if self.typed_at_read(module, read)
                else partial(self.require_bound_once, where(module, read, parent), read)
                for span, (module, read, parent) in open_reads.items()
            )
        )

    @staticmethod
    def typed_at_read(module: ParsedModule, read: ast.expr) -> bool:
        """Whether the type checker's type of the receiver holds at the read: a
        name only when its scope binds it once (`bound_once`), since a later
        binding from an untyped value leaves the type of the first in place; any
        other receiver is typed by its declarations."""

        match NamespaceReads.receiver(read):
            case (ast.Name() as name,):
                return module.fact(ModuleNodeIndex).bound_once(name)
        return True

    @staticmethod
    def require_bound_once(where: str, read: ast.expr) -> None:
        raise NeedsInput(
            f"{where}; `{ast.unparse(NamespaceReads.receiver(read)[0])}` is bound more than once "
            "where it is read, so no one type holds every value it has",
            NeededInput.PROOF,
        )

    def asks_another_key(
        self,
        module: ParsedModule,
        read: ast.expr,
        parent: ast.AST | None,
        around: ast.AST | None,
        indexed: CompactIndexedClass,
        deleted: frozenset[str],
    ) -> bool:
        """Whether the read asks only for keys no deleted name can be: one
        constant key that is not deleted (`ns["k"]`, `ns.get("k")`, `"k" in ns`, or
        `"k" in {name for ... in ns}`, which collects the names to ask the same),
        `self.<attribute>` whose every write builds a text with a constant
        prefix or suffix no deleted name has, a local the function binds once to
        such a text (`local_text_avoids`), or a name that only holds what the
        read class's own annotations declare (`OwnAnnotationKeys`) while the
        class losing the members annotates none of them."""

        def other(key: ast.AST | None) -> bool:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value not in deleted
            match key:
                case ast.Attribute(value=ast.Name(id="self"), attr=attribute):
                    return self.keys_avoid(module, read, attribute, deleted)
                case ast.Name(id=name):
                    if self.local_text_avoids(module, read, name, deleted):
                        return True
                    annotated = self.own_annotations(indexed)
                    return (
                        annotated is not None
                        and deleted.isdisjoint(annotated)
                        and OwnAnnotationKeys.hold(module, read, name)
                    )
            return False

        match parent:
            case ast.Subscript(value=value, slice=key) if value is read:
                return other(key)
            case ast.Compare(left=key, ops=[ast.In() | ast.NotIn()], comparators=[value]) if value is read:
                return other(key)
            case ast.Attribute(value=value, attr="get") if value is read:
                return isinstance(around, ast.Call) and around.func is parent and bool(around.args) and other(
                    around.args[0]
                )
            case ast.comprehension(iter=iterated, target=ast.Name(id=element), ifs=[]) if iterated is read:
                # `"k" in {name for c in order for name in vars(c)}`: the names are
                # collected only to ask whether one constant is among them.
                match around:
                    case ast.SetComp(elt=ast.Name(id=kept)) | ast.ListComp(elt=ast.Name(id=kept)) | ast.GeneratorExp(
                        elt=ast.Name(id=kept)
                    ) if kept == element:
                        return any(
                            isinstance(node, ast.Compare)
                            and len(node.ops) == 1
                            and isinstance(node.ops[0], (ast.In, ast.NotIn))
                            and node.comparators[0] is around
                            and other(node.left)
                            for node in ast.walk(module.module)
                        )
        return False

    def own_annotations(self, indexed: CompactIndexedClass) -> frozenset[str] | None:
        """The names `indexed`'s own body annotates, or None when the body also
        spells `__annotations__` or `__annotate__` and so may set others."""

        module = self.snapshot.parsed_module_for_source_path(indexed.file_path)
        for node in ast.walk(module.module):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == indexed.simple_name
                and indexed.line in (node.lineno, *(decorator.lineno for decorator in node.decorator_list))
            ):
                if any(
                    isinstance(inner, ast.Name) and inner.id in ANNOTATION_SLOTS
                    or isinstance(inner, ast.Attribute) and inner.attr in ANNOTATION_SLOTS
                    or isinstance(inner, ast.Constant) and inner.value in ANNOTATION_SLOTS
                    for inner in ast.walk(node)
                ):
                    return None
                return frozenset(
                    statement.target.id
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
                )
        return None

    def resolves_annotations_only(self, module: ParsedModule, read: ast.expr, deleted: frozenset[str]) -> bool:
        """The namespace only becomes `localns` of `get_type_hints`, directly or
        merged through `dict(...)` and `|`, so it answers only names annotations
        spell: no deleted name is one of them."""

        parents = {child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)}
        node: ast.AST = read
        while True:
            parent = parents.get(node)
            match parent:
                case ast.Call(func=ast.Name(id="dict"), args=[only], keywords=[]) if only is node:
                    node = parent
                case ast.BinOp(op=ast.BitOr()):
                    node = parent
                case ast.keyword(arg="localns"):
                    call = parents.get(parent)
                    match call:
                        case ast.Call(func=ast.Name(id="get_type_hints") | ast.Attribute(attr="get_type_hints")):
                            return deleted.isdisjoint(self.snapshot.module_set.fact(AnnotationNames).names)
                    return False
                case _:
                    return False

    def keys_avoid(self, module: ParsedModule, read: ast.expr, attribute: str, deleted: frozenset[str]) -> bool:
        """Every write of `attribute` in the repository is `self.<attribute> = f"..."`
        inside the class that reads it, and each written text starts or ends with a
        constant that no deleted name starts or ends with."""

        owners = [
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.ClassDef) and any(inner is read for inner in ast.walk(node))
        ]
        if not owners:
            return False
        owner = min(owners, key=lambda node: sum(1 for _ in ast.walk(node)))
        owned = {id(node) for node in ast.walk(owner)}
        writes = 0
        for scanned in self.snapshot.module_set:
            for node in ast.walk(scanned.module):
                match node:
                    case ast.Assign(targets=targets, value=value):
                        written = [target for target in targets if isinstance(target, ast.Attribute) and target.attr == attribute]
                    case ast.AugAssign(target=ast.Attribute(attr=name) as target, value=value) if name == attribute:
                        return False
                    case ast.Call(func=ast.Name(id="setattr"), args=[_, ast.Constant(value=name), *_]) if name == attribute:
                        return False
                    case _:
                        continue
                for target in written:
                    if not (
                        scanned is module
                        and id(target) in owned
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and self.text_avoids(value, deleted)
                    ):
                        return False
                    writes += 1
        return writes > 0

    def local_text_avoids(self, module: ParsedModule, read: ast.expr, name: str, deleted: frozenset[str]) -> bool:
        """The key is a local of the function the read stands in, bound there once,
        to a text with a constant prefix or suffix no deleted name has:
        `name = f"action_{declared}"`, then `name in base.__dict__`. A parameter, a
        second binding, or a comprehension around the read binding the name is not
        that local."""

        parents = {child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)}
        scope: ast.AST | None = parents.get(read)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and any(
                isinstance(bound, ast.Name) and bound.id == name
                for generator in scope.generators
                for bound in ast.walk(generator.target)
            ):
                return False
            scope = parents.get(scope)
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        arguments = scope.args
        if any(
            argument.arg == name
            for argument in (
                *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                *(extra for extra in (arguments.vararg, arguments.kwarg) if extra is not None),
            )
        ) or any(
            isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names for node in ast.walk(scope)
        ):
            return False
        stores = [
            node for node in ast.walk(scope)
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
        ]
        if len(stores) != 1:
            return False
        match parents.get(stores[0]):
            case ast.Assign(targets=[target], value=value) if target is stores[0]:
                return self.text_avoids(value, deleted)
        return False

    @staticmethod
    def text_avoids(value: ast.expr, deleted: frozenset[str]) -> bool:
        if not isinstance(value, ast.JoinedStr) or not value.values:
            return False
        first, last = value.values[0], value.values[-1]
        prefix = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else ""
        suffix = last.value if isinstance(last, ast.Constant) and isinstance(last.value, str) else ""
        return bool(prefix or suffix) and not any(
            name.startswith(prefix) and name.endswith(suffix) for name in deleted
        )


@dataclass(frozen=True)
class MemberDeletionCreationProof(ClassCreationProof):
    """Deleting restated members leaves the class's creation unchanged: its order
    stays, and only creation code reading its own namespace can see them go."""

    def reads(self, code: CreationCode, roles: dict[str, Role]) -> CreationReads:
        return MemberDeletionReads(self, code, roles)


@dataclass
class MemberDeletionReads(CreationReads):
    """One creation function's reads, judged against the deleted members."""

    @property
    def unproved(self) -> str:
        return f"NRA needs a proof that deleting {self.proof.owner_name} leaves it unchanged"


@dataclass(frozen=True)
class RestatedOverrideDeletion:
    """One class's members that restate what it inherits, proved deletable."""

    overlap: OverlapClass
    restated: tuple[tuple[RestatedDeclarations, str], ...]
    """Each restatement, with the class whose copy lookup reaches instead."""

    @classmethod
    def proved(cls, snapshot: CodemodSourceSnapshot, overlap: OverlapClass) -> RestatedOverrideDeletion:
        verdicts = snapshot.class_overlap_verdicts
        found = inherited_restatements(verdicts, overlap.symbol)
        if not found:
            raise NeedsInput(
                f"`{overlap.qualname}` restates nothing it would inherit", NeededInput.SOURCE_CHANGE
            )
        names = frozenset(name for name, _inherited in found)
        dunders = sorted(name for name in names if name.startswith("__") and name.endswith("__"))
        if dunders:
            raise NeedsInput(
                f"`{overlap.qualname}` restates {', '.join(f'`{name}`' for name in dunders)}, which "
                "class creation and its decorators treat by name: NRA needs a proof that deleting it "
                "leaves the class unchanged",
                NeededInput.PROOF,
            )
        ancestry = require_static_c3(snapshot, overlap.facts)
        order = verdicts.linearized(overlap.symbol)
        proof = MemberDeletionCreationProof(
            snapshot,
            overlap.facts,
            ancestry,
            " + ".join(f"`{overlap.qualname}.{name}`" for name in sorted(names)),
            names,
            frozenset(),
            MroWalk(order, order, overlap.symbol, names),
        )
        restated = []
        for name, inherited in found:
            binders = proof.binders(order, False, name)
            if binders[:2] != (overlap.symbol, inherited):
                raise NeedsInput(
                    f"`{name}` is bound by {', '.join(f'`{symbol}`' for symbol in binders)} along "
                    f"`{overlap.qualname}`'s order, not first by `{inherited}`",
                    NeededInput.PROOF,
                )
            restatement = RestatedDeclarations.require(
                snapshot, OverlapClass.for_symbol(snapshot, inherited), overlap, (name,)
            )
            for statement in restatement.statements:
                cls.require_once_per_class(proof, overlap, statement)
            restated.append((restatement, inherited))
        cls.require_plain_decorators(proof, overlap)
        proof.require_unchanged()
        NamespaceReaders(snapshot).require_unread(overlap.facts, names)
        return cls(overlap, tuple(restated))

    @staticmethod
    def require_once_per_class(
        proof: ClassCreationProof, overlap: OverlapClass, statement: ast.stmt
    ) -> None:
        """What the class body evaluates for the member does nothing per class:
        it calls nothing, a decorator NRA models aside."""

        evaluated: list[ast.AST] = []
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in statement.decorator_list:
                if proof.function_decorator(overlap.facts, decorator) is None:
                    raise NeedsInput(
                        f"`@{ast.unparse(decorator)}` on `{overlap.qualname}.{statement.name}` runs "
                        "once for each class body: NRA needs a proof that one run serves both",
                        NeededInput.PROOF,
                    )
            evaluated.extend((*statement.args.defaults, *(d for d in statement.args.kw_defaults if d)))
        elif isinstance(statement, ast.Assign):
            evaluated.append(statement.value)
        if any(isinstance(node, ast.Call) for value in evaluated for node in ast.walk(value)):
            raise NeedsInput(
                f"`{overlap.qualname}` builds a value for `{min(_names(statement))}` with a call, once "
                "for each class body: NRA needs a proof that one value serves both",
                NeededInput.PROOF,
            )

    @staticmethod
    def require_plain_decorators(proof: ClassCreationProof, overlap: OverlapClass) -> None:
        """A class decorator may read the namespace it receives; `dataclass` reads
        only annotated names and the methods it generates, neither of which a
        deletion here takes."""

        for decorator in overlap.node.decorator_list:
            called = decorator.func if isinstance(decorator, ast.Call) else decorator
            if proof.qualified(overlap.facts, called, decorator.lineno) != "dataclasses.dataclass":
                raise NeedsInput(
                    f"`{overlap.qualname}` is decorated by `{ast.unparse(decorator)}`, which may read "
                    "its namespace: NRA needs a proof that deleting members leaves it unchanged",
                    NeededInput.PROOF,
                )

    def edits(self, snapshot: CodemodSourceSnapshot, rationale: str) -> tuple[NominalSourceEdit, ...]:
        deletion = StatementDeletionSource(
            source=snapshot.sources_by_file_path[self.overlap.file_path],
            node=self.overlap.node,
            file_path=self.overlap.file_path,
        )
        written = tuple(
            statement for restatement, _inherited in self.restated for statement in restatement.written_statements
        )
        statements = tuple(
            statement for restatement, _inherited in self.restated for statement in restatement.statements
        )
        return (
            *deletion.physical_edits(
                file_path=self.overlap.file_path,
                replacements=deletion.replacements_for_statements(written),
                rationale=rationale,
            ),
            *obsolete_import_edits(snapshot, self.overlap.file_path, statements, rationale),
        )

    def rationale(self) -> str:
        return " ".join(
            f"Deletes `{self.overlap.qualname}.{name}`: `{inherited.rpartition('.')[2]}` declares it the same way."
            for restatement, inherited in self.restated
            for name in restatement.names
        )


def _names(statement: ast.stmt) -> frozenset[str]:
    if isinstance(statement, ast.Assign):
        return frozenset(target.id for target in statement.targets if isinstance(target, ast.Name)) or frozenset(("?",))
    return frozenset((getattr(statement, "name", "?"),))


@dataclass(frozen=True, kw_only=True)
class DeleteRestatedOverridesOperation(RepositorySourceReprovedOperation):
    """Delete every member the target class declares exactly as the class it
    would inherit the member from, then each import only those members read."""

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        deletion = RestatedOverrideDeletion.proved(snapshot, OverlapClass.require(snapshot, self.target))
        return deletion.edits(snapshot, self.rationale_text(deletion.rationale()))


class RestatedOverrideFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    RestatedOverrideConcept,
):
    """One deletion of the members the finding's class restates."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("deleting a restated member needs source context and its class")
        snapshot = context.execution_snapshot()
        indexed = next(
            (
                indexed
                for indexed in snapshot.class_overlap_verdicts.class_index.classes_by_symbol.values()
                if indexed.qualname == evidence.symbol and indexed.file_path == evidence.file_path
            ),
            None,
        )
        if indexed is None:
            return self.rejected_evaluation(f"`{evidence.symbol}` left the source")
        operation = DeleteRestatedOverridesOperation(target=source_rewrite_target(indexed))
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason="The class inherits the same declaration: every lookup reaches an identical copy.",
            ).with_operation(operation)
        )


__all__ = (
    "DeleteRestatedOverridesOperation",
    "MemberDeletionCreationProof",
    "NamespaceReaders",
    "NamespaceReads",
    "RestatedOverrideDeletion",
    "RestatedOverrideFindingRecipeSynthesizer",
    "inherited_restatements",
    "next_binder",
)
