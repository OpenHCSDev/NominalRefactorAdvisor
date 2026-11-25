"""Reads of class attributes (`__bases__`, `__orig_bases__`, `__dict__`, `vars()`)
across the scanned repository and installed ancestor packages."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.source_identity import source_path_is_package_init

BASES_ATTRIBUTES = frozenset(("__bases__", "__orig_bases__"))


@dataclass(frozen=True)
class BaseListReads(ParsedModuleSetFact):
    """Every read of a class's `__bases__` or `__orig_bases__` in one state's
    modules, found once for every drop proved over that state."""

    reads: tuple[tuple[ParsedModule, ast.expr, ast.AST | None], ...]
    """Each module, the read, and the node it is read in."""

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> BaseListReads:
        found: list[tuple[ParsedModule, ast.expr, ast.AST | None]] = []
        for module in modules:
            if not any(attribute in module.source for attribute in BASES_ATTRIBUTES):
                continue
            reads = tuple(cls.found(module.module))
            if not reads:
                continue
            parents = module.syntax_index.parent_index
            found.extend((module, read, parents.get(read)) for read in reads)
        return cls(tuple(found))

    @staticmethod
    def found(tree: ast.AST) -> Iterator[ast.expr]:
        for node in ast.walk(tree):
            match node:
                case ast.Attribute(attr=attr, ctx=ast.Load()) if attr in BASES_ATTRIBUTES:
                    yield node
                case ast.Call(
                    func=ast.Attribute(attr="get"),
                    args=[ast.Constant(value=str() as key), *_],
                ) if key in BASES_ATTRIBUTES:
                    yield node
                case ast.Subscript(slice=ast.Constant(value=str() as key)) if (
                    key in BASES_ATTRIBUTES
                ):
                    yield node


class InstalledPackageBaseReads:
    """Every read of a class's `__bases__` or `__orig_bases__` in one installed
    package's source, read and never run: found once for a run."""

    @staticmethod
    @cache
    def of(package: str) -> tuple[tuple[ParsedModule, ast.expr, ast.AST | None], ...]:
        located = InstalledModule.source_path(package)
        if located is None:
            raise NeedsInput(
                f"`{package}` declares a class of the ancestry and has no source NRA can read: "
                "NRA needs a proof that it reads no class's bases",
                NeededInput.PROOF,
            )
        root = located.parent
        files = (
            sorted(root.rglob("*.py"))
            if source_path_is_package_init(located)
            else [located]
        )
        found: list[tuple[ParsedModule, ast.expr, ast.AST | None]] = []
        for path in files:
            if not any(attribute in path.read_text() for attribute in BASES_ATTRIBUTES):
                continue
            relative = path.relative_to(root).with_suffix("").parts
            parts = relative[:-1] if relative[-1] == "__init__" else relative
            module_name = (
                ".".join((package, *parts))
                if source_path_is_package_init(located)
                else package
            )
            parsed = InstalledModule.parsed(module_name)
            if parsed is None:
                raise NeedsInput(
                    f"{path} names a class's bases and NRA cannot read it as `{module_name}`",
                    NeededInput.PROOF,
                )
            reads = tuple(BaseListReads.found(parsed.module))
            if not reads:
                continue
            parents = {
                child: parent
                for parent in ast.walk(parsed.module)
                for child in ast.iter_child_nodes(parent)
            }
            found.extend((parsed, read, parents.get(read)) for read in reads)
        return tuple(found)


@dataclass(frozen=True)
class ClassAttributeReaders:
    """Reads of one class attribute (`X.<attribute>`, or `vars(X)` for the
    namespace) anywhere in the repository, each judged by the class it reads."""

    snapshot: CodemodSourceSnapshot

    def reaches(
        self, module: ParsedModule, read: ast.expr, indexed: CompactIndexedClass
    ) -> bool:
        """Whether the read can see `indexed`'s own attribute: a read of a named
        class's own entry (`reads_its_own`) sees it only when that class is
        `indexed`, and any other read of a named class only when its order holds
        `indexed`."""

        match read:
            case ast.Attribute(value=receiver) | ast.Call(
                func=ast.Name(id="vars"), args=[receiver]
            ):
                pass
            case _:
                return True
        verdicts = self.snapshot.class_overlap_verdicts
        family = self.annotated_family(
            module, read, receiver
        ) or self.subclass_family(module, read, receiver)
        if family is not None:
            return self.family_holds(family, indexed)
        named = self.named(module, receiver)
        if named is None or verdicts.class_index.class_for(named) is None:
            return True
        if self.reads_its_own(read, named):
            return named == indexed.symbol
        try:
            return indexed.symbol in verdicts.linearized(named)
        except NeedsInput:
            return True

    def reads_its_own(self, read: ast.expr, named: str) -> bool:
        """Whether the read of the named class sees only that class's own entry:
        `X.__bases__`, `X.__dict__` and `vars(X)` always do, and `X.__orig_bases__`
        does when `X`'s header subscripts a base, which sets it in `X`'s namespace;
        otherwise it is the nearest ancestor's."""

        match read:
            case ast.Attribute(attr="__bases__" | "__dict__") | ast.Call():
                return True
            case ast.Attribute(attr="__orig_bases__"):
                indexed = self.snapshot.class_overlap_verdicts.class_index.class_for(
                    named
                )
                if indexed is None:
                    return False
                node = ResolvedClassTarget.from_rewrite_target(
                    self.snapshot,
                    SourceRewriteTarget(
                        file_path=indexed.file_path, qualname=indexed.qualname
                    ),
                ).node
                return any(isinstance(base, ast.Subscript) for base in node.bases)
        return False

    def family_holds(self, family: str, indexed: CompactIndexedClass) -> bool:
        """Whether a value of `type[family]`, `family` or a subclass, can be
        `indexed`: only when `indexed` subclasses `family`, read from its order; a
        family outside the scan or an order that needs input may hold it."""

        verdicts = self.snapshot.class_overlap_verdicts
        if verdicts.class_index.class_for(family) is None:
            return True
        try:
            return family in verdicts.linearized(indexed.symbol)
        except NeedsInput:
            return True

    def annotated_family(
        self, module: ParsedModule, read: ast.expr, receiver: ast.expr
    ) -> str | None:
        """The scanned class `C` when the receiver is a parameter or local every
        annotation of which is `type[C]`: the read sees only `C` or a subclass."""

        if not isinstance(receiver, ast.Name):
            return None
        parents = module.syntax_index.parent_index
        scope: ast.AST | None = parents.get(read)
        while scope is not None and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            scope = parents.get(scope)
        if scope is None:
            return None
        annotations = [
            argument.annotation
            for argument in (
                *scope.args.posonlyargs,
                *scope.args.args,
                *scope.args.kwonlyargs,
            )
            if argument.arg == receiver.id
        ] + [
            statement.annotation
            for statement in ast.walk(scope)
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == receiver.id
        ]
        families = set()
        for annotation in annotations:
            match annotation:
                case ast.Subscript(
                    value=ast.Name(id="type" | "Type")
                    | ast.Attribute(attr="type" | "Type"),
                    slice=held,
                ):
                    family = self.named(module, held)
                    if (
                        family is None
                        or self.snapshot.class_overlap_verdicts.class_index.class_for(
                            family
                        )
                        is None
                    ):
                        return None
                    families.add(family)
                case _:
                    return None
        return families.pop() if len(families) == 1 else None

    def subclass_family(
        self, module: ParsedModule, read: ast.expr, receiver: ast.expr
    ) -> str | None:
        """The scanned class `C` when the receiver is the one variable of a loop over
        `X.__subclasses__()` and `X` is `C`: a class the module names, or the `cls` of
        a classmethod `C` declares. `X.__subclasses__()` lists subclasses of `X`, and
        `cls` is `C` or a subclass, so the read sees only subclasses of `C`."""

        if not isinstance(receiver, ast.Name):
            return None
        parents = module.syntax_index.parent_index
        scope: ast.AST | None = read
        while scope is not None and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            scope = parents.get(scope)
        if scope is None:
            return None
        stores = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Name)
            and node.id == receiver.id
            and isinstance(node.ctx, ast.Store)
        ]
        arguments = scope.args
        if len(stores) != 1 or any(
            argument.arg == receiver.id
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ):
            return None
        match parents.get(stores[0]):
            case (
                ast.For(target=target, iter=walked)
                | ast.comprehension(target=target, iter=walked)
            ) if target is stores[0]:
                pass
            case _:
                return None
        match walked:
            case ast.Call(
                func=ast.Attribute(value=listed, attr="__subclasses__"),
                args=[],
                keywords=[],
            ):
                pass
            case _:
                return None
        owner = parents.get(scope)
        positional = (*arguments.posonlyargs, *arguments.args)
        if (
            isinstance(listed, ast.Name)
            and isinstance(owner, ast.ClassDef)
            and positional
            and positional[0].arg == listed.id
            and any(
                isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                for decorator in scope.decorator_list
            )
            and sum(
                1
                for node in ast.walk(scope)
                if isinstance(node, ast.Name)
                and node.id == listed.id
                and isinstance(node.ctx, ast.Store)
            )
            == 0
        ):
            family = (
                f"{module.module_name}.{owner.name}"
                if isinstance(parents.get(owner), ast.Module)
                else None
            )
        else:
            family = self.named(module, listed)
        if (
            family is None
            or self.snapshot.class_overlap_verdicts.class_index.class_for(family)
            is None
        ):
            return None
        return family

    def named(self, module: ParsedModule, node: ast.expr) -> str | None:
        """The class or function a name or attribute chain spells in `module`: an
        imported or builtin one, or a class the module itself declares."""

        verdicts = self.snapshot.class_overlap_verdicts
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            module, node, line=node.lineno
        )
        if witness is None:
            local = (
                verdicts.ancestor(f"{module.module_name}.{node.id}")
                if isinstance(node, ast.Name)
                else None
            )
            return None if local is None else local.symbol
        indexed = verdicts.ancestor(witness.qualified_name)
        return witness.qualified_name if indexed is None else indexed.symbol


@dataclass(frozen=True)
class BasesReaders(ClassAttributeReaders):
    """Every read of a class's `__bases__` or `__orig_bases__` in the repository."""

    def require_unread(
        self,
        indexed: CompactIndexedClass,
        dropped: tuple[str, ...],
        installed: frozenset[str] = frozenset(),
    ) -> None:
        """Each read either tests a named class none of `dropped` is, or keeps only
        bases that subclass a named class none of `dropped` subclasses.

        The reads are the repository's and those of each `installed` package: the
        packages that declare a class or metaclass of the dropped class's ancestry.
        Their code runs as the class is created and holds it afterwards (a registry,
        a descendants walk), so a read of a class's bases anywhere in them may be a
        read of this class's."""

        verdicts = self.snapshot.class_overlap_verdicts
        orders = {symbol: verdicts.linearized(symbol) for symbol in dropped}
        reads = (
            *self.snapshot.module_set.fact(BaseListReads).reads,
            *(
                read
                for package in sorted(installed)
                for read in InstalledPackageBaseReads.of(package)
            ),
        )
        for module, read, parent in reads:
            if not self.reaches(module, read, indexed):
                continue
            if not self.unaffected(module, read, parent, orders):
                raise NeedsInput(
                    f"{module.file_path}:{read.lineno} `{ast.unparse(parent or read)}` "
                    f"reads bases, and dropping {', '.join(f'`{symbol.rpartition(".")[2]}`' for symbol in dropped)} "
                    f"from `{indexed.qualname}` could change what it sees",
                    NeededInput.PROOF,
                )

    def unaffected(
        self,
        module: ParsedModule,
        read: ast.expr,
        parent: ast.AST | None,
        orders: dict[str, tuple[str, ...]],
    ) -> bool:
        match parent:
            case ast.Compare(
                left=tested, ops=[ast.In() | ast.NotIn()], comparators=[comparator]
            ) if comparator is read:
                named = self.named(module, tested)
                return named is not None and named not in orders
            case ast.comprehension(
                iter=iterated, target=ast.Name(id=element), ifs=tests
            ) if iterated is read:
                return any(
                    self.excludes(module, test, element, orders) for test in tests
                )
        return False

    def excludes(
        self,
        module: ParsedModule,
        test: ast.expr,
        element: str,
        orders: dict[str, tuple[str, ...]],
    ) -> bool:
        """Whether `test` is false for each dropped base: a conjunct that keeps only
        subclasses of a named class no dropped base subclasses, or only generic
        aliases, which a plain base never is."""

        conjuncts = (
            test.values
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            else [test]
        )
        for conjunct in conjuncts:
            match conjunct:
                case ast.Call(
                    func=ast.Name(id="issubclass"),
                    args=[ast.Name(id=subject), owner],
                    keywords=[],
                ) if subject == element:
                    named = self.named(module, owner)
                    if named is not None and all(
                        named not in order for order in orders.values()
                    ):
                        return True
                case ast.Call(
                    func=ast.Name(id="isinstance"),
                    args=[
                        ast.NamedExpr(
                            value=ast.Call(
                                func=origin, args=[ast.Name(id=subject)], keywords=[]
                            )
                        ),
                        ast.Name(id="type"),
                    ],
                    keywords=[],
                ) if (
                    subject == element
                    and self.named(module, origin) == "typing.get_origin"
                ):
                    return True
        return False


__all__ = (
    "BASES_ATTRIBUTES",
    "BaseListReads",
    "BasesReaders",
    "ClassAttributeReaders",
    "InstalledPackageBaseReads",
)
