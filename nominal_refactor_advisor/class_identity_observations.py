"""Declared identities and the strings that spell them again.

A class, a module and a family member each have one declared identity: the
class statement, the module file, and the class keyword or class constant whose
string is distinct across the member's siblings. A string literal equal to one
of them, in a position where the program uses it to pick, compare, import or
type something, is a second copy of that identity.

`DeclaredIdentityFamily` collects what each module declares; `IdentitySpellingFamily`
collects the literals in identity positions. Which literal spells which
identity is a scan-wide join (`DeclaredIdentities`), because the declaration and
the spelling sit in different modules.
"""

from __future__ import annotations

import ast
import builtins
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from pathlib import Path
from typing import ClassVar

from .lexical_bindings import ImportBoundNameProjection
from .name_algebra import CLASS_NAME_ALGEBRA
from .ast_projection import StatementValues
from .ast_projection import CallKeywords
from .ast_tools import (
    BuiltinCall,
    ModuleSyntaxIndex,
    ParsedModule,
    Setattr,
)
from .class_index import module_import_aliases
from .installed_source import InstalledModule
from .quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
    ReferenceRewrite,
    SpelledModules,
)
from .models import SourceLocation
from .observation_families import ShapeFamily
from .restated_fact_observations import (
    RestatingSite,
    RestatingSiteFamily,
    RestatingSiteModule,
)


@dataclass(frozen=True)
class DeclaredClass:
    """One class statement: its name, bases, and the strings it declares about itself.

    `keywords` are the class keywords whose value is a string
    (`class A(Base, declared_name="a")`), and `consumed_keywords` the keywords its
    own `__init_subclass__` accepts: declaration options the class machinery
    consumes. `slots` are class-level names annotated `ClassVar[...]` with no
    value: a spelling every subclass declares. `constants` are class-level string
    constants (`kind = "a"`);
    `instance_attributes` are the names it declares as per-object state (an
    annotated field that is not a ClassVar, or `self.name = ...` in a method);
    `lookups` are its classmethods that return a member class of the family for
    an argument.

    What decides whether the class object's `__name__` is the statement's name:
    `top_level` (a module-level statement, so `module.Name` is its qualified
    name), `decorators` and `origins` (its bases and metaclass), each as the
    qualified name it resolves to in its module, or empty when it resolves to
    nothing the module binds, and `writes_name`, whether anything inside the
    class stores a `__name__` (a body binding, a metaclass or subclass hook
    renaming the classes it makes).
    """

    name: str
    line: int
    base_names: tuple[str, ...]
    keywords: tuple[tuple[str, str], ...]
    consumed_keywords: tuple[str, ...]
    slots: tuple[str, ...]
    constants: tuple[tuple[str, str], ...]
    instance_attributes: tuple[str, ...]
    lookups: tuple[str, ...]
    top_level: bool = False
    decorators: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    writes_name: bool = False


@dataclass(frozen=True)
class FromImport:
    """A top-level `from module import a, b` statement: its index in the module body, the module, the names."""

    index: int
    module: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class ModuleDeclarations:
    """What one module declares: its import name, its classes, what it imports and binds, and which names it renames.

    `imports` are the module's final import bindings with the qualified name
    each resolves to, and `from_imports` its top-level `from` statements;
    `bound_after` the line after which each top-level name is bound
    (`ModuleReferenceScope.bound_after`), and `rebound` the names its module
    scope binds more than once; `renamed` the qualified names whose
    `__name__` the module stores (`X.__name__ = ...`,
    `setattr(X, "__name__", ...)`); `imported` each module any of its import
    statements names, wherever the statement stands, with the names a `from`
    statement takes from it (none for `import module`), and `last_import` the
    index of its last top-level import statement (-1 without one).
    """

    module_name: str
    file_path: str
    classes: tuple[DeclaredClass, ...]
    imports: tuple[tuple[str, str], ...] = ()
    from_imports: tuple[FromImport, ...] = ()
    bound_after: tuple[tuple[str, int], ...] = ()
    rebound: tuple[str, ...] = ()
    renamed: tuple[str, ...] = ()
    imported: tuple[tuple[str, tuple[str, ...]], ...] = ()
    last_import: int = -1


class DeclaredIdentityFamily(ShapeFamily[ModuleDeclarations]):
    item_type = ModuleDeclarations

    SUBCLASS_HOOK = "__init_subclass__"

    @classmethod
    @cache
    def installed(cls, module_name: str) -> ModuleDeclarations | None:
        """What a module outside the scan declares, read from its installed source."""

        parsed = InstalledModule.parsed(module_name)
        if parsed is None:
            return None
        (declared,) = cls.collect(parsed)
        return declared

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ModuleDeclarations]:
        index = parsed_module.syntax_index
        names = QualifiedNames.of(parsed_module)
        top_level = {
            id(statement)
            for statement in parsed_module.module.body
            if isinstance(statement, ast.ClassDef)
        }
        return [
            ModuleDeclarations(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                classes=tuple(
                    cls.declared_class(node, names, id(node) in top_level)
                    for _node_index, node in index.indexed_nodes_of_type(ast.ClassDef)
                ),
                imports=tuple(sorted(names.imports.items())),
                from_imports=tuple(cls.from_imports(parsed_module)),
                bound_after=tuple(
                    sorted(
                        ModuleReferenceScope(parsed_module.module).bound_after.items()
                    )
                ),
                rebound=tuple(
                    sorted(
                        name
                        for name, count in cls.module_scope_bindings(
                            parsed_module.module
                        ).items()
                        if count > 1
                    )
                ),
                renamed=tuple(
                    sorted(
                        {
                            qualified
                            for node in ast.walk(parsed_module.module)
                            for receiver in cls.renamed_receivers(node)
                            if (qualified := names.qualified(receiver))
                        }
                    )
                ),
                imported=tuple(sorted(set(cls.imported(parsed_module)))),
                last_import=max(
                    (
                        index
                        for index, statement in enumerate(parsed_module.module.body)
                        if isinstance(statement, (ast.Import, ast.ImportFrom))
                    ),
                    default=-1,
                ),
            )
        ]

    @staticmethod
    def imported(
        parsed_module: ParsedModule,
    ) -> Iterator[tuple[str, tuple[str, ...]]]:
        """Each module an import statement names, with the names a `from` takes from it."""

        for node in ast.walk(parsed_module.module):
            match node:
                case ast.Import(names=aliases):
                    for alias in aliases:
                        yield alias.name, ()
                case ast.ImportFrom(module=imported, level=level, names=aliases):
                    module = (
                        parsed_module.module_path_identity.resolve_import_from_module(
                            imported_module=imported, level=level
                        )
                    )
                    if module is not None:
                        yield module, tuple(alias.name for alias in aliases)

    NAME = "__name__"
    STAR = "*"

    @classmethod
    def from_imports(cls, parsed_module: ParsedModule) -> Iterator[FromImport]:
        for index, statement in enumerate(parsed_module.module.body):
            match statement:
                case ast.ImportFrom(module=imported, level=level, names=aliases) if all(
                    alias.name != cls.STAR for alias in aliases
                ):
                    module = (
                        parsed_module.module_path_identity.resolve_import_from_module(
                            imported_module=imported, level=level
                        )
                    )
                    if module is not None:
                        yield FromImport(
                            index,
                            module,
                            ImportBoundNameProjection(statement).names(),
                        )

    SCOPES = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    @classmethod
    def module_scope_bindings(cls, module: ast.Module) -> Counter[str]:
        """How many times the module scope binds each name: not inside functions, classes or lambdas."""

        counts: Counter[str] = Counter()
        pending: list[ast.AST] = list(module.body)
        while pending:
            node = pending.pop()
            counts.update(ModuleReferenceScope.bound_names(node))
            if not isinstance(node, cls.SCOPES):
                pending.extend(ast.iter_child_nodes(node))
        return counts

    @classmethod
    def renamed_receivers(cls, node: ast.AST) -> Iterator[ast.expr]:
        """The objects whose `__name__` this node stores."""

        match node:
            case ast.Attribute(value=receiver, attr=attribute, ctx=ast.Store()) if (
                attribute == cls.NAME
            ):
                yield receiver
            case ast.Call():
                match BuiltinCall.decoded(node):
                    case Setattr(
                        object_argument=receiver,
                        name_argument=ast.Constant(value=written_name),
                    ) if written_name == cls.NAME:
                        yield receiver

    @classmethod
    def writes_name(cls, node: ast.ClassDef) -> bool:
        """Whether anything inside the class stores a `__name__`, on any object or namespace."""

        for inner in ast.walk(node):
            if any(True for _receiver in cls.renamed_receivers(inner)):
                return True
            match inner:
                case ast.Name(id=name, ctx=ast.Store()) if name == cls.NAME:
                    return True
                case ast.Subscript(slice=ast.Constant(value=key), ctx=ast.Store()) if (
                    key == cls.NAME
                ):
                    return True
        return False

    @classmethod
    def declared_class(
        cls, node: ast.ClassDef, names: QualifiedNames, top_level: bool
    ) -> DeclaredClass:
        methods = [
            statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return DeclaredClass(
            name=node.name,
            line=node.lineno,
            base_names=tuple(
                name for base in node.bases if (name := cls.base_name(base))
            ),
            keywords=tuple(
                (keyword.name, keyword.value.value)
                for keyword in CallKeywords.of(node.keywords).named
                if isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            consumed_keywords=tuple(
                argument.arg
                for method in methods
                if method.name == cls.SUBCLASS_HOOK
                for argument in (
                    *method.args.posonlyargs[1:],
                    *method.args.args[1:],
                    *method.args.kwonlyargs,
                )
            ),
            slots=tuple(
                statement.target.id
                for statement, _bare in StatementValues.keyed(
                    (statement, statement)
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign)
                ).bare
                if isinstance(statement.target, ast.Name)
                and ClassVar.__name__ in ast.unparse(statement.annotation)
            ),
            constants=tuple(cls.constants(node)),
            instance_attributes=tuple(
                sorted(set(cls.instance_attributes(node, methods)))
            ),
            lookups=tuple(
                method.name for method in methods if cls.is_lookup(method, node.name)
            ),
            top_level=top_level,
            decorators=tuple(
                names.qualified(decorator) for decorator in node.decorator_list
            ),
            origins=tuple(
                names.qualified(origin)
                for origin in (
                    *node.bases,
                    *(
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "metaclass"
                    ),
                )
            ),
            writes_name=cls.writes_name(node),
        )

    @classmethod
    def base_name(cls, base: ast.expr) -> str:
        """The class a base names: `Base`, `module.Base` and `Base[T]` all name `Base`."""

        match base:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
            case ast.Subscript(value=generic):
                return cls.base_name(generic)
        return ""

    @staticmethod
    def constants(node: ast.ClassDef) -> Iterator[tuple[str, str]]:
        for statement in node.body:
            match statement:
                case ast.Assign(
                    targets=[ast.Name(id=name)],
                    value=ast.Constant(value=str() as value),
                ) | ast.AnnAssign(
                    target=ast.Name(id=name), value=ast.Constant(value=str() as value)
                ):
                    yield name, value

    @staticmethod
    def instance_attributes(
        node: ast.ClassDef, methods: Sequence[ast.FunctionDef | ast.AsyncFunctionDef]
    ) -> Iterator[str]:
        for statement in node.body:
            match statement:
                case ast.AnnAssign(
                    target=ast.Name(id=name), annotation=annotation, value=None
                ):
                    if ClassVar.__name__ not in ast.unparse(annotation):
                        yield name
        for method in methods:
            parameters = [*method.args.posonlyargs, *method.args.args]
            if not parameters:
                continue
            own = parameters[0].arg
            for statement in method.body:
                for assigned in ast.walk(statement):
                    match assigned:
                        case ast.Attribute(
                            value=ast.Name(id=receiver), attr=name, ctx=ast.Store()
                        ) if (receiver == own):
                            yield name

    @staticmethod
    def is_lookup(function: ast.FunctionDef | ast.AsyncFunctionDef, owner: str) -> bool:
        """A classmethod taking an argument and returning `type[Self]` or `type[Owner]`."""

        decorated = any(
            isinstance(decorator, ast.Name) and decorator.id == classmethod.__name__
            for decorator in function.decorator_list
        )
        parameters = [*function.args.posonlyargs, *function.args.args]
        match function.returns:
            case ast.Subscript(
                value=ast.Name(id="type"),
                slice=ast.Name(id=returned) | ast.Constant(value=str() as returned),
            ):
                return (
                    decorated and len(parameters) >= 2 and returned in ("Self", owner)
                )
        return False


@dataclass(frozen=True)
class QualifiedNames:
    """What a module-level name or attribute chain names, as a qualified name.

    An imported name resolves through the module's final import bindings, a
    name the module binds itself to `module.name`, and a builtin to
    `builtins.name`; a call (`dataclass(frozen=True)`) or a subscript
    (`Generic[T]`) names what it calls or subscripts.
    """

    module_name: str
    imports: dict[str, str]
    local: frozenset[str]

    @classmethod
    def of(cls, parsed_module: ParsedModule) -> QualifiedNames:
        return cls(
            parsed_module.module_name,
            module_import_aliases(parsed_module),
            frozenset(ModuleReferenceScope(parsed_module.module).bound_after),
        )

    def qualified(self, expression: ast.expr) -> str:
        match expression:
            case ast.Call(func=called):
                return self.qualified(called)
            case ast.Subscript(value=generic):
                return self.qualified(generic)
            case ast.Attribute(value=owner, attr=attribute):
                found = self.qualified(owner)
                return f"{found}.{attribute}" if found else ""
            case ast.Name(id=name):
                if name in self.imports:
                    return self.imports[name]
                if name in self.local:
                    return f"{self.module_name}.{name}"
                if hasattr(builtins, name):
                    return f"{builtins.__name__}.{name}"
        return ""


@dataclass(frozen=True)
class ClassNameOwners:
    """Which qualified names reach a class whose `__name__` is its statement's name.

    `type.__new__` names a class after its statement; a class keeps that name
    unless something renames it: a decorator that returns another object, a
    metaclass or `__init_subclass__` that stores `__name__`, or a store of
    `X.__name__`. A class keeps its name here when nothing in it stores a
    `__name__`, no module read here renames it, every decorator is the
    standard library's, and every base and metaclass is the standard library's
    or a class that keeps its name. A base or metaclass from outside the scan is
    read from its installed source (`InstalledModule`) and judged the same way;
    one whose source cannot be found is not known to keep its name.

    `modules` are the scanned modules and the installed ones read for them;
    `scanned` names the scanned ones, the only ones a rewrite may import from.
    """

    modules: dict[str, ModuleDeclarations]
    scanned: frozenset[str]
    STANDARD: ClassVar[frozenset[str]] = frozenset(
        (*sys.stdlib_module_names, builtins.__name__)
    )
    MAX_REEXPORTS: ClassVar[int] = 16
    MAX_INSTALLED: ClassVar[int] = 64

    @classmethod
    def from_modules(cls, modules: Iterable[ModuleDeclarations]) -> ClassNameOwners:
        loaded = {module.module_name: module for module in modules}
        owners = cls(dict(loaded), frozenset(loaded))
        tried: set[str] = set()
        for _round in range(cls.MAX_REEXPORTS):
            missing = {
                module
                for origin in owners.foreign_origins()
                if (module := owners.missing_module(origin)) is not None
                and module not in tried
            }
            if not missing or len(tried) >= cls.MAX_INSTALLED:
                break
            tried |= missing
            for module_name in missing:
                found = DeclaredIdentityFamily.installed(module_name)
                if found is not None:
                    loaded[module_name] = found
            owners = cls(dict(loaded), owners.scanned)
        return owners

    @cached_property
    def classes(self) -> dict[str, DeclaredClass]:
        """Every top-level class bound once in its module, by qualified name."""

        return {
            f"{module.module_name}.{declared.name}": declared
            for module in self.modules.values()
            for declared in module.classes
            if declared.top_level and declared.name not in module.rebound
        }

    @cached_property
    def scanned_classes(self) -> dict[str, tuple[str, ...]]:
        """The scanned classes of each name, by qualified name."""

        found: dict[str, list[str]] = defaultdict(list)
        for qualified, declared in self.classes.items():
            if qualified.rpartition(".")[0] in self.scanned:
                found[declared.name].append(qualified)
        return {name: tuple(qualified) for name, qualified in found.items()}

    @cached_property
    def imports(self) -> dict[str, dict[str, str]]:
        return {name: dict(module.imports) for name, module in self.modules.items()}

    @cached_property
    def renamed(self) -> frozenset[str]:
        return frozenset(
            filter(
                None,
                (
                    self.canonical(name)
                    for module in self.modules.values()
                    for name in module.renamed
                ),
            )
        )

    @cached_property
    def scanned_roots(self) -> frozenset[str]:
        return frozenset(name.partition(".")[0] for name in self.scanned)

    @cached_property
    def by_path(self) -> dict[str, ModuleDeclarations]:
        return {
            module.file_path: module
            for name, module in self.modules.items()
            if name in self.scanned
        }

    def foreign_origins(self) -> Iterator[str]:
        """Decorators, bases and metaclasses that resolve to no class read here and are not the standard library's."""

        for declared in self.classes.values():
            for origin in (*declared.decorators, *declared.origins):
                if (
                    origin
                    and not self.standard(origin)
                    and self.canonical(origin) is None
                ):
                    yield origin

    def missing_module(self, qualified: str) -> str | None:
        """The first module a qualified name's re-export chain needs that is not read here."""

        for _hop in range(self.MAX_REEXPORTS):
            if qualified in self.classes:
                return None
            module, _dot, name = qualified.rpartition(".")
            if not module:
                return None
            if module not in self.modules:
                return module
            reexported = self.imports[module].get(name)
            if reexported is None:
                return None
            qualified = reexported
        return None

    def canonical(self, qualified: str) -> str | None:
        """The class a qualified name reaches through re-exports, if any."""

        for _hop in range(self.MAX_REEXPORTS):
            if qualified in self.classes:
                return qualified
            module, _dot, name = qualified.rpartition(".")
            reexported = self.imports.get(module, {}).get(name)
            if reexported is None:
                return None
            qualified = reexported
        return None

    def standard(self, qualified: str) -> bool:
        root = qualified.partition(".")[0]
        return root in self.STANDARD and root not in self.scanned_roots

    def keeps_name(
        self, qualified: str, visiting: frozenset[str] = frozenset()
    ) -> bool:
        declared = self.classes.get(qualified)
        if declared is None or qualified in visiting or qualified in self.renamed:
            return False
        if declared.writes_name:
            return False
        if not all(self.standard(decorator) for decorator in declared.decorators):
            return False
        for origin in declared.origins:
            if self.standard(origin):
                continue
            owner = self.canonical(origin)
            if owner is None or not self.keeps_name(owner, visiting | {qualified}):
                return False
        return True

    def references(self, binding: str, literal: str) -> bool:
        """Whether `literal.__name__`, where the name is bound to `binding`, is the literal."""

        owner = self.canonical(binding)
        return (
            owner is not None
            and self.classes[owner].name == literal
            and self.keeps_name(owner)
        )

    def importable(
        self, file_path: str, holding: int, literal: str
    ) -> FromImport | None:
        """A `from` statement of the file that can also import the literal's class, or None.

        The statement runs before the statement holding the literal, imports
        from a module that binds the class once, at top level, before some name
        the statement already imports: whenever the statement succeeds, the
        class is bound in that module, so adding it changes no import order and
        fails nowhere the statement did not.
        """

        module = self.by_path.get(file_path)
        for statement in () if module is None else module.from_imports:
            if statement.index >= holding:
                continue
            source = self.modules.get(statement.module)
            bound = {} if source is None else dict(source.bound_after)
            line = bound.get(literal)
            if (
                line is not None
                and self.references(f"{statement.module}.{literal}", literal)
                and any(bound.get(name, 0) > line for name in statement.names)
            ):
                return statement
        return None

    def new_import(self, file_path: str, holding: int, literal: str) -> str | None:
        """The scanned module a new `from module import Literal` after the file's imports can come from, or None.

        The one scanned class of that name keeps its name, the new statement
        (placed after the file's last top-level import) runs before the
        statement holding the literal, and importing the owner cannot reach
        the file's module or a name of one of its half-run packages
        (`imports_back`): the import binds the class, or fails only where
        importing the owner fails anyway.
        """

        module = self.by_path.get(file_path)
        owners = self.scanned_classes.get(literal, ())
        if module is None or len(owners) != 1:
            return None
        if module.last_import < 0 or holding <= module.last_import:
            return None
        (owner,) = owners
        owner_module = owner.rpartition(".")[0]
        if owner_module == module.module_name or not self.keeps_name(owner):
            return None
        if self.imports_back(owner_module, module.module_name):
            return None
        return owner_module

    def imports_back(self, owner_module: str, module_name: str) -> bool:
        """Whether importing the owner, while `module_name` runs, can meet a module not yet finished.

        When `module_name` runs, its packages have started already and are
        never run again, so the walk does not enter them. It follows every
        other scanned module the owner's imports can run (each named module,
        its packages, and each name a `from` takes that is a scanned submodule)
        and answers yes on reaching `module_name`, or a `from` statement taking
        a name that is no submodule from one of its packages: that half-run
        package may not have bound the name yet.
        """

        parts = module_name.split(".")
        running = frozenset(".".join(parts[:stop]) for stop in range(1, len(parts)))
        seen: set[str] = set()
        pending = [owner_module]
        for current in pending:
            current_parts = current.split(".")
            for stop in range(1, len(current_parts) + 1):
                runs = ".".join(current_parts[:stop])
                if runs == module_name:
                    return True
                if runs in running or runs not in self.scanned or runs in seen:
                    continue
                seen.add(runs)
                for imported, names in self.modules[runs].imported:
                    submodules = [f"{imported}.{name}" for name in names]
                    if imported in running and any(
                        submodule not in self.scanned for submodule in submodules
                    ):
                        return True
                    pending.extend((imported, *submodules))
        return False


class IdentitySpelling(ABC):
    """How a literal in one kind of position spells an identity.

    Each position class declares its `tag`, which the site carries as its first
    detail so the join can find the position again; the class alone knows
    which declared identity its literal can spell.
    """

    tag: ClassVar[str]
    rewrite_kind: ClassVar[type[IdentitySpelling]]
    """The rewrite a rewritable site of this position belongs to, one recipe per scan."""

    @classmethod
    def detail(cls, *values: str) -> tuple[str, ...]:
        return (cls.tag, *(str(value) for value in values))

    @classmethod
    def rewrite(
        cls, detail: Sequence[str], identities: DeclaredIdentities, file_path: str
    ) -> ReferenceRewrite:
        """Where this site can be rewritten as its reference; most positions cannot."""

        return ReferenceRewrite.NONE

    @classmethod
    def spelling(cls, detail: Sequence[str]) -> str:
        """What the site writes: the literal, unless the position says otherwise."""

        return repr(detail[-1])

    @classmethod
    def positions(cls) -> dict[str, type[IdentitySpelling]]:
        return {position.tag: position for position in cls.__subclasses__()}

    @classmethod
    @abstractmethod
    def resolve(
        cls, site: RestatingSite, identities: DeclaredIdentities
    ) -> ResolvedIdentity | None:
        """The identity this site's literal spells, if the scan declares it."""

        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedIdentity:
    """A declared identity: what it is, and where it is declared."""

    identity: str
    declared_at: tuple[SourceLocation, ...]

    def restated_by(self, site: RestatingSite) -> RestatingSite:
        return replace(site, group=self.identity)


class TypeSpelling(IdentitySpelling):
    """A class name inside a quoted annotation, base, TypeVar bound or cast type."""

    tag = "type"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[-1])


class LazyAnnotationSpelling(IdentitySpelling):
    """A class name quoted in an annotation that only an eager interpreter needs quoted.

    The name is not bound when the annotation would run eagerly (a later class,
    an import under `if TYPE_CHECKING:`), so the quote is required unless the
    project's interpreter evaluates annotations lazily (Python 3.14 and later).
    """

    tag = "annotation"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        if not ReferenceRewrite.LAZY_INTERPRETER.rewritable(
            DeclaredInterpreter.of(Path(site.file_path))
        ):
            return None
        return identities.class_named(site.detail[-1])


TypeSpelling.rewrite_kind = TypeSpelling
LazyAnnotationSpelling.rewrite_kind = TypeSpelling


class WordSpelling(IdentitySpelling):
    """A multi-word string passed or returned: a class's name there.

    A one-word class name (Read, Open) is also an ordinary word, so only a
    multi-word spelling is taken as naming a class outside a type position.
    A member's spelling passed along is not taken as naming it: field, column
    and wire names share those spellings by coincidence.
    """

    tag = "word"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        return ClassNameSpelling.rewrite(detail, identities, file_path)

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[-1])


class ComparedWordSpelling(IdentitySpelling):
    """A multi-word string compared or matched: a class's name or a family member's spelling.

    Comparing a value with the spelling asks which declaration it is; a
    one-word spelling (main, completed) is also an ordinary word, so only a
    multi-word one is taken as naming a declaration here.
    """

    tag = "compared word"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        return ClassNameSpelling.rewrite(detail, identities, file_path)

    @classmethod
    def resolve(cls, site, identities):
        literal = site.detail[-1]
        return identities.class_named(literal) or identities.member_named(literal)


class ClassNameSpelling(IdentitySpelling):
    """A string compared with a `__name__` or `__qualname__`: a class name, one word or more.

    Every value position that can hold a class name (this one, a passed or
    returned word, a compared word) carries what the literal's name is where
    the literal runs (`IdentityPositions.class_reference`): the qualified name
    it is bound to, or, where the file binds the name nowhere, the index of the
    top-level statement holding the literal. `Name.__name__` can stand there
    when the binding reaches a scanned class that keeps its name, or when a
    `from` statement before it can import such a class
    (`ClassNameOwners.importable`), or a new `from` statement after the file's
    imports can (`ClassNameOwners.new_import`).
    """

    tag = "class name"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        binding, holding, literal = detail[-3:]
        owners = identities.class_names
        if (binding and owners.references(binding, literal)) or (
            holding
            and (
                owners.importable(file_path, int(holding), literal)
                or owners.new_import(file_path, int(holding), literal)
            )
        ):
            return ReferenceRewrite.ANY_INTERPRETER
        return ReferenceRewrite.NONE

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[-1])


@dataclass(frozen=True)
class Receiver:
    """What a site knows about the object an attribute is read from.

    `name` is the receiver's last segment (a class when the scan declares one of
    that name); `class_object` and `instance` are the class an enclosing
    parameter annotation declares it to be (`type[C]`, or `C`), or empty.
    """

    text: str
    name: str
    class_object: str
    instance: str

    def detail(self) -> tuple[str, ...]:
        return (self.text, self.name, self.class_object, self.instance)

    @classmethod
    def from_detail(cls, detail: Sequence[str]) -> Receiver:
        text, name, class_object, instance = detail
        return cls(text, name, class_object, instance)


class MemberSpelling(IdentitySpelling):
    """A string compared or matched with `x.attribute`, where the attribute is a family's member spelling."""

    tag = "member"

    @classmethod
    def resolve(cls, site, identities):
        _tag, attribute, *receiver, literal = site.detail
        found = identities.member_attribute(attribute, Receiver.from_detail(receiver))
        return None if found is None else found.spelled(literal)


class FlattenedSpelling(IdentitySpelling):
    """A family member's keyword spelling read off a class or object and passed or returned as text."""

    tag = "flattened"

    @classmethod
    def spelling(cls, detail):
        _tag, attribute, text, *_receiver = detail
        return f"{text}.{attribute}"

    @classmethod
    def resolve(cls, site, identities):
        _tag, attribute, *receiver = site.detail
        read = Receiver.from_detail(receiver)
        found = identities.member_attribute(attribute, read)
        if found is None:
            return None
        return found.read_off(read.text, identities.reads_any_member(read, found))


class ModuleSpelling(IdentitySpelling):
    """A dotted module name given to an importer, `sys.modules`, or an interpreter's `-m`."""

    tag = "module"

    @classmethod
    def rewrite(cls, detail, identities, file_path):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        return identities.module_named(site.detail[-1])


ModuleSpelling.rewrite_kind = ModuleSpelling
ClassNameSpelling.rewrite_kind = ClassNameSpelling
WordSpelling.rewrite_kind = ClassNameSpelling
ComparedWordSpelling.rewrite_kind = ClassNameSpelling


class LookupSpelling(IdentitySpelling):
    """A string passed to a family's classmethod that returns the member it names."""

    tag = "lookup"

    @classmethod
    def resolve(cls, site, identities):
        _tag, receiver, method, literal = site.detail
        return identities.member_looked_up(receiver, method, literal)


# A finding names this many declaring sites of one identity.
LISTED_DECLARATIONS = 4


@dataclass(frozen=True)
class MemberSpellings:
    """A class keyword or constant whose explicit values are distinct among the siblings declaring it.

    `members` are the explicit values with their classes; `family_bases` are the
    classes below which it names members: the bases of those siblings, the
    classes whose `__init_subclass__` consumes it, and the classes that declare
    it as a `ClassVar` slot. A keyword may also be derived
    by the class machinery, so any spelling compared with it names a member; a
    class constant names one only by an explicit value.
    """

    attribute: str
    members: tuple[tuple[str, SourceLocation], ...]
    family_bases: frozenset[str]
    declared_by_keyword: bool
    slotted: bool
    declared_at: tuple[SourceLocation, ...]

    @property
    def identity(self) -> str:
        return f"the `{self.attribute}` spelling of a family member"

    CODEC_BOUNDARY = (
        " (codec boundary: needs judgment — the one wire decode may spell the tag)"
    )

    def read_off(self, receiver: str, any_member: bool) -> ResolvedIdentity | None:
        """A keyword spelling read as text; a class constant read is only a value.

        Reading it off one named member restates that member. Reading it off
        whichever member arrives (a receiver typed by the family's base, or not
        typed) is what a codec does at its boundary, so the finding says it
        needs judgment.
        """

        if not self.declared_by_keyword:
            return None
        boundary = self.CODEC_BOUNDARY if any_member else ""
        return ResolvedIdentity(
            f"{self.identity}, read off `{receiver}` as text{boundary}",
            self.declared_at[:LISTED_DECLARATIONS],
        )

    def spelled(self, literal: str) -> ResolvedIdentity | None:
        named = tuple(location for value, location in self.members if value == literal)
        if not named and not self.declared_by_keyword:
            return None
        who = (
            " and ".join(f"`{location.symbol}`" for location in named)
            if named
            else "a member whose spelling the family derives"
        )
        return ResolvedIdentity(
            f"{self.identity} {literal!r} ({who})",
            (named or self.declared_at)[:LISTED_DECLARATIONS],
        )


@dataclass(frozen=True)
class DeclaredIdentities:
    """Every identity the scan declares: classes, modules, member spellings and lookups."""

    classes: dict[str, tuple[SourceLocation, ...]]
    modules: dict[str, SourceLocation]
    members: dict[str, MemberSpellings]
    lookups: dict[str, tuple[str, ...]]
    bases: dict[str, tuple[str, ...]]
    instance_attributes: dict[str, frozenset[str]]
    class_names: ClassNameOwners
    carriers: dict[tuple[str, str, str], tuple[str, ...]] = field(default_factory=dict)
    rewritten_files: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_modules(cls, modules: Iterable[ModuleDeclarations]) -> DeclaredIdentities:
        modules = tuple(modules)
        classes: dict[str, list[SourceLocation]] = defaultdict(list)
        module_locations: dict[str, SourceLocation] = {}
        siblings: dict[tuple[str, str, bool], list[tuple[str, SourceLocation]]] = (
            defaultdict(list)
        )
        consumers: dict[str, list[SourceLocation]] = defaultdict(list)
        slots: dict[str, list[SourceLocation]] = defaultdict(list)
        lookups: dict[str, set[str]] = defaultdict(set)
        bases: dict[str, set[str]] = defaultdict(set)
        instance_attributes: dict[str, set[str]] = defaultdict(set)
        explicit: list[tuple[str, str, SourceLocation]] = []
        for module in modules:
            module_locations[module.module_name] = SourceLocation(
                module.file_path, 1, module.module_name
            )
            for declared in module.classes:
                location = SourceLocation(
                    module.file_path, declared.line, declared.name
                )
                classes[declared.name].append(location)
                lookups[declared.name].update(declared.lookups)
                bases[declared.name].update(declared.base_names)
                instance_attributes[declared.name].update(declared.instance_attributes)
                for keyword in declared.consumed_keywords:
                    consumers[keyword].append(location)
                for slot in declared.slots:
                    slots[slot].append(location)
                for by_keyword, spellings in (
                    (True, declared.keywords),
                    (False, declared.constants),
                ):
                    for attribute, value in spellings:
                        explicit.append((attribute, value, location))
                        for base in declared.base_names:
                            siblings[(attribute, base, by_keyword)].append(
                                (value, location)
                            )
        members: dict[str, list[tuple[str, SourceLocation]]] = defaultdict(list)
        family_bases: dict[str, set[str]] = defaultdict(set)
        keyword_attributes: set[str] = set(consumers)
        for (attribute, base, by_keyword), group in siblings.items():
            values = [value for value, _location in group]
            if (
                len(values) >= 2
                and len(set(values)) == len(values)
                and not cls.protocol_name(attribute)
            ):
                members[attribute].extend(group)
                family_bases[attribute].add(base)
                if by_keyword:
                    keyword_attributes.add(attribute)
        for declaring_by_attribute in (consumers, slots):
            for attribute, declaring in declaring_by_attribute.items():
                if cls.protocol_name(attribute):
                    continue
                family_bases[attribute].update(
                    location.symbol for location in declaring
                )
        return cls(
            classes={name: tuple(found) for name, found in classes.items()},
            modules=module_locations,
            members={
                attribute: MemberSpellings(
                    attribute=attribute,
                    members=tuple(members[attribute]),
                    family_bases=frozenset(found),
                    declared_by_keyword=attribute in keyword_attributes,
                    slotted=attribute in slots,
                    declared_at=(
                        *consumers.get(attribute, ()),
                        *slots.get(attribute, ()),
                        *(location for _value, location in members[attribute]),
                    ),
                )
                for attribute, found in family_bases.items()
            },
            lookups={name: tuple(sorted(found)) for name, found in lookups.items()},
            bases={name: tuple(sorted(found)) for name, found in bases.items()},
            instance_attributes={
                name: frozenset(found) for name, found in instance_attributes.items()
            },
            class_names=ClassNameOwners.from_modules(modules),
        ).with_family_values(explicit)

    def with_family_values(
        self, explicit: Iterable[tuple[str, str, SourceLocation]]
    ) -> DeclaredIdentities:
        """Every explicit value of a keyword or slot spelling on a class of its family names that member.

        A family that consumes a keyword, or declares a slot every subclass fills,
        declares the spelling per member, so a member that declares one alone
        among its siblings still names itself.
        """

        added: dict[str, list[tuple[str, SourceLocation]]] = defaultdict(list)
        for attribute, value, location in explicit:
            spellings = self.members.get(attribute)
            if (
                spellings is not None
                and (spellings.declared_by_keyword or spellings.slotted)
                and (value, location) not in spellings.members
                and self.in_family(location.symbol, spellings)
            ):
                added[attribute].append((value, location))
        return replace(
            self,
            members={
                attribute: replace(
                    spellings, members=(*spellings.members, *added.get(attribute, ()))
                )
                for attribute, spellings in self.members.items()
            },
        )

    @staticmethod
    def protocol_name(name: str) -> bool:
        """A dunder name belongs to Python's or a framework's protocol, not to a family.

        `__registry_key__ = "kind"` configures the registry machinery, and a
        literal `"__registry_key__"` names that protocol slot; neither is a
        member's spelling, whatever family happens to hold the same text.
        """

        return len(name) > 4 and name.startswith("__") and name.endswith("__")

    def member_named(self, literal: str) -> ResolvedIdentity | None:
        """The family member whose explicit spelling the literal is."""

        if self.protocol_name(literal):
            return None
        for spellings in self.members.values():
            if any(value == literal for value, _location in spellings.members):
                return spellings.spelled(literal)
        return None

    def class_named(self, name: str) -> ResolvedIdentity | None:
        declared = self.classes.get(name)
        if declared is None:
            return None
        return ResolvedIdentity(f"class `{name}`", declared[:LISTED_DECLARATIONS])

    def module_named(self, name: str) -> ResolvedIdentity | None:
        declared = self.modules.get(name)
        if declared is None:
            return None
        return ResolvedIdentity(f"module `{name}`", (declared,))

    def ancestors(self, name: str) -> tuple[str, ...]:
        """The class and every scanned class it inherits from, by name."""

        found = {name: None}
        frontier = [name]
        for current in frontier:
            for base in self.bases.get(current, ()):
                if base not in found:
                    found[base] = None
                    frontier.append(base)
        return tuple(found)

    def in_family(self, name: str, spellings: MemberSpellings) -> bool:
        return any(
            ancestor in spellings.family_bases for ancestor in self.ancestors(name)
        )

    def per_object(self, name: str, attribute: str) -> bool:
        """Whether the class or an ancestor declares the attribute as per-object state."""

        return any(
            attribute in self.instance_attributes.get(ancestor, ())
            for ancestor in self.ancestors(name)
        )

    def member_attribute(
        self, attribute: str, receiver: Receiver
    ) -> MemberSpellings | None:
        """The family spelling `receiver.attribute` reads, when the receiver's declared type says so.

        A class named directly, or a parameter annotated `type[C]`, reads the
        class's own spelling; a parameter annotated `C` reads it unless C declares
        the attribute as per-object state; a receiver of no declared type reads
        a family spelling only if no class anywhere declares that name as
        per-object state.
        """

        spellings = self.members.get(attribute)
        if spellings is None:
            return None
        for class_object in (receiver.name, receiver.class_object):
            if class_object in self.classes:
                return spellings if self.in_family(class_object, spellings) else None
        if receiver.instance in self.classes:
            return (
                spellings
                if self.in_family(receiver.instance, spellings)
                and not self.per_object(receiver.instance, attribute)
                else None
            )
        if any(attribute in declared for declared in self.instance_attributes.values()):
            return None
        return spellings if spellings.members else None

    def resolved_class(self, receiver: Receiver) -> str:
        """The scanned class the receiver is, or is an instance of, by its declared type."""

        return next(
            (
                name
                for name in (receiver.name, receiver.class_object, receiver.instance)
                if name in self.classes
            ),
            "",
        )

    def reads_any_member(self, receiver: Receiver, spellings: MemberSpellings) -> bool:
        """Whether the receiver is whichever member arrives rather than one named member."""

        resolved = self.resolved_class(receiver)
        return not resolved or resolved in spellings.family_bases

    def member_looked_up(
        self, receiver: str, method: str, literal: str
    ) -> ResolvedIdentity | None:
        declaring = next(
            (
                ancestor
                for ancestor in self.ancestors(receiver)
                if method in self.lookups.get(ancestor, ())
            ),
            None,
        )
        if declaring is None:
            return None
        for spellings in self.members.values():
            if self.in_family(receiver, spellings) and any(
                value == literal for value, _location in spellings.members
            ):
                return spellings.spelled(literal)
        return ResolvedIdentity(
            f"the `{receiver}` member spelled {literal!r}",
            self.classes[declaring][:LISTED_DECLARATIONS],
        )

    def resolve(self, site: RestatingSite) -> RestatingSite | None:
        position = IdentitySpelling.positions()[site.detail[0]]
        resolved = position.resolve(site, self)
        return None if resolved is None else resolved.restated_by(site)

    def with_carriers(self, sites: Sequence[RestatingSite]) -> DeclaredIdentities:
        """Name, per rewrite, the one finding that carries it for the whole scan, and the files it rewrites.

        A rewrite is one proved operation per file; the scan's rewrite of one kind
        (quoted type names, or spelled modules) is one recipe, so it is one
        candidate: carried by the finding at the scan's first rewritable site of
        that kind (by file, line, then identity).
        """

        positions = IdentitySpelling.positions()
        first: dict[str, RestatingSite] = {}
        files: dict[str, set[str]] = defaultdict(set)
        for site in sites:
            position = positions[site.detail[0]]
            if not position.rewrite(site.detail, self, site.file_path).rewritable(
                DeclaredInterpreter.of(Path(site.file_path))
            ):
                continue
            kind = position.rewrite_kind.tag
            files[kind].add(site.file_path)
            current = first.get(kind)
            if current is None or (site.file_path, site.lineno, site.group) < (
                current.file_path,
                current.lineno,
                current.group,
            ):
                first[kind] = site
        carriers: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for kind, site in sorted(first.items()):
            carriers[(site.file_path, site.symbol, site.group)].append(kind)
        return replace(
            self,
            carriers={key: tuple(found) for key, found in carriers.items()},
            rewritten_files={
                kind: tuple(sorted(found)) for kind, found in files.items()
            },
        )

    def carried_rewrites(
        self, sites: Sequence[RestatingSite]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The rewrites this finding carries, each with the files it rewrites."""

        return tuple(
            (kind, self.rewritten_files[kind])
            for kind in self.carriers.get(
                (sites[0].file_path, sites[0].symbol, sites[0].group), ()
            )
        )

    def declarations_of(
        self, sites: Sequence[RestatingSite]
    ) -> tuple[SourceLocation, ...]:
        """The declaring sites of the identity one group of resolved sites restates."""

        resolved = IdentitySpelling.positions()[sites[0].detail[0]].resolve(
            sites[0], self
        )
        return () if resolved is None else resolved.declared_at


class IdentityPositions:
    """Which expressions of one module are identity positions, and what each says."""

    CLASS_NAME_ATTRIBUTES = frozenset(("__name__", "__qualname__"))
    COLLECTIONS = frozenset(
        (
            "tuple",
            "list",
            "set",
            "frozenset",
            "Sequence",
            "MutableSequence",
            "Iterable",
            "Iterator",
            "Collection",
            "AbstractSet",
            "MutableSet",
        )
    )

    def __init__(self, parsed_module: ParsedModule) -> None:
        self.parsed_module = parsed_module
        self.index: ModuleSyntaxIndex = parsed_module.syntax_index
        self.node_indices = {
            id(node): node_index
            for node_index, node in enumerate(self.index.depth_first_nodes)
        }
        self.references = ModuleReferenceScope(
            parsed_module.module, package=self.package_of(parsed_module)
        )

    @staticmethod
    def package_of(parsed_module: ParsedModule) -> str:
        return parsed_module.module_path_identity.package_name

    @staticmethod
    def site(
        node: ast.expr, detail: tuple[str, ...]
    ) -> tuple[ast.expr, tuple[str, ...]]:
        return node, detail

    def sites(self) -> Iterator[RestatingSite]:
        for node, detail in self.spellings():
            yield RestatingSiteFamily.site(
                self.parsed_module,
                node,
                self.index.lexical_symbol(self.node_indices[id(node)]),
                detail[-1],
                detail,
            )

    @cached_property
    def class_bindings(self) -> dict[str, tuple[int, str]]:
        """Each name a top-level class statement or `from` import binds, and nothing else does.

        The value is the binding statement's index in the module body and the
        qualified name it binds; a name bound anywhere else too (a parameter, a
        local, a second import) is left out, so no scope can shadow it.
        """

        counts = Counter(
            bound
            for node in ast.walk(self.parsed_module.module)
            for bound in ModuleReferenceScope.bound_names(node)
        )
        imports = module_import_aliases(self.parsed_module)
        found: dict[str, tuple[int, str]] = {}
        for index, statement in enumerate(self.parsed_module.module.body):
            match statement:
                case ast.ClassDef(name=name):
                    found[name] = (index, f"{self.parsed_module.module_name}.{name}")
                case ast.ImportFrom(names=aliases):
                    for alias in aliases:
                        bound = ImportBoundNameProjection(statement).alias_bound_name(alias)
                        if bound in imports:
                            found[bound] = (index, imports[bound])
        return {name: bound for name, bound in found.items() if counts[name] == 1}

    @cached_property
    def statement_indices(self) -> dict[int, int]:
        return {
            id(statement): index
            for index, statement in enumerate(self.parsed_module.module.body)
        }

    @cached_property
    def named_anywhere(self) -> frozenset[str]:
        """Every name the module binds or reads anywhere."""

        return frozenset(
            name
            for node in ast.walk(self.parsed_module.module)
            for name in (
                *ModuleReferenceScope.bound_names(node),
                *((node.id,) if isinstance(node, ast.Name) else ()),
            )
        )

    @cached_property
    def star_imported(self) -> bool:
        """Whether a `from module import *` binds names this module cannot list."""

        return any(
            alias.name == DeclaredIdentityFamily.STAR
            for node in ast.walk(self.parsed_module.module)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )

    def class_reference(self, node_index: int, literal: str) -> tuple[str, str]:
        """What the literal's name is where the literal runs: its binding, and where an import could go.

        The first is the qualified name a top-level class statement or import
        binds it to, when that runs before the top-level statement holding the
        literal, so the name is bound whenever the literal is evaluated. The
        second, when the file binds and reads the name nowhere and it is no
        builtin, is
        that holding statement's index: a `from` statement before it could
        import the class. Both are empty otherwise, and in a module with a star
        import, whose bindings no source lists.
        """

        holding = next(
            (
                index
                for ancestor in self.index.ancestor_nodes(node_index)
                if (index := self.statement_indices.get(id(ancestor))) is not None
            ),
            None,
        )
        if holding is None or self.star_imported:
            return "", ""
        bound = self.class_bindings.get(literal)
        if bound is not None:
            return (bound[1] if bound[0] < holding else ""), ""
        if literal in self.named_anywhere or hasattr(builtins, literal):
            return "", ""
        return "", str(holding)

    multiword = staticmethod(CLASS_NAME_ALGEBRA.names_several_words)

    @staticmethod
    def string(node: ast.AST) -> str | None:
        match node:
            case ast.Constant(value=str() as value):
                return value
        return None

    called = staticmethod(ModuleReferenceScope.called)

    def spellings(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        yield from self.type_sites()
        for node_index, node in self.index.indexed_nodes_of_type(ast.Compare):
            yield from self.compare_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Match):
            yield from self.match_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Call):
            yield from self.call_sites(node_index, node)
        for node_index, returned in StatementValues.keyed(
            self.index.indexed_nodes_of_type(ast.Return)
        ).supplied:
            yield from self.passed_sites(node_index, returned.value, "return")
        yield from self.module_sites()

    # Type positions -----------------------------------------------------------

    def type_sites(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        """Class names in quoted type strings: a reference wherever the name would resolve.

        Where it resolves on any interpreter the site is a `type` site; an
        annotation that resolves only when evaluated lazily is an `annotation`
        site, which the join keeps only for projects whose Python is lazy.
        """

        for quoted in self.references.quoted_type_names():
            rewrite = self.references.rewrite_of(quoted)
            for name, root in quoted.names:
                if self.references.resolves_eagerly(
                    root, quoted.node.lineno, quoted.annotation
                ):
                    yield self.site(quoted.node, TypeSpelling.detail(rewrite, name))
                elif quoted.annotation:
                    yield self.site(
                        quoted.node, LazyAnnotationSpelling.detail(rewrite, name)
                    )

    # Comparison and match positions ---------------------------------------------

    def compared(
        self, node_index: int, subject: ast.expr, literal_node: ast.expr
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        literal = self.string(literal_node)
        if literal is None or isinstance(subject, ast.Constant):
            return
        match subject:
            case ast.Attribute(attr=attribute) if (
                attribute in self.CLASS_NAME_ATTRIBUTES
            ):
                yield self.site(
                    literal_node,
                    ClassNameSpelling.detail(
                        *self.class_reference(node_index, literal), literal
                    ),
                )
                return
            case ast.Attribute(value=receiver, attr=attribute):
                yield self.site(
                    literal_node,
                    MemberSpelling.detail(
                        attribute,
                        *self.receiver(node_index, receiver).detail(),
                        literal,
                    ),
                )
        if self.multiword(literal):
            yield self.site(
                literal_node,
                ComparedWordSpelling.detail(
                    *self.class_reference(node_index, literal), literal
                ),
            )

    def compare_sites(
        self, node_index: int, node: ast.Compare
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        left = node.left
        for operator, right in zip(node.ops, node.comparators):
            match operator, right:
                case ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot(), _:
                    yield from self.compared(node_index, left, right)
                    yield from self.compared(node_index, right, left)
                case ast.In() | ast.NotIn(), ast.Set(elts=items) | ast.Tuple(
                    elts=items
                ) | ast.List(elts=items):
                    for item in items:
                        yield from self.compared(node_index, left, item)
            left = right

    def match_sites(
        self, node_index: int, node: ast.Match
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for case in node.cases:
            for literal_node in self.pattern_literals(case.pattern):
                yield from self.compared(node_index, node.subject, literal_node)

    @classmethod
    def pattern_literals(cls, pattern: ast.pattern) -> Iterator[ast.expr]:
        match pattern:
            case ast.MatchValue(value=ast.Constant(value=str()) as literal):
                yield literal
            case ast.MatchOr(patterns=alternatives):
                for alternative in alternatives:
                    yield from cls.pattern_literals(alternative)

    # Argument and return positions -----------------------------------------------

    def call_sites(
        self, node_index: int, node: ast.Call
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            yield from self.passed_sites(node_index, argument, "argument")
        match node:
            case ast.Call(
                func=ast.Attribute(value=receiver, attr=method),
                args=[ast.Constant(value=str() as literal) as literal_node, *_],
            ) if isinstance(receiver, (ast.Name, ast.Attribute)):
                yield self.site(
                    literal_node,
                    LookupSpelling.detail(self.called(receiver), method, literal),
                )

    def passed_sites(
        self, node_index: int, value: ast.expr, position: str
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for passed in self.passed_values(value):
            literal = self.string(passed)
            if literal is not None and self.multiword(literal):
                yield self.site(
                    passed,
                    WordSpelling.detail(
                        position, *self.class_reference(node_index, literal), literal
                    ),
                )
            match passed:
                case ast.Attribute(value=receiver, attr=attribute) if not self.own(
                    node_index, receiver
                ):
                    yield self.site(
                        passed,
                        FlattenedSpelling.detail(
                            attribute, *self.receiver(node_index, receiver).detail()
                        ),
                    )

    def receiver(self, node_index: int, receiver: ast.expr) -> Receiver:
        """The receiver's text, last segment, and the type its binding declares for it."""

        class_object = instance = ""
        if isinstance(receiver, ast.Name):
            declared = self.declared_type(node_index, receiver.id)
            match declared:
                case ast.Subscript(value=ast.Name(id="type"), slice=element):
                    class_object = self.called(element)
                case ast.Name() | ast.Attribute():
                    instance = self.called(declared)
        return Receiver(
            ast.unparse(receiver), self.called(receiver), class_object, instance
        )

    def declared_type(self, node_index: int, name: str) -> ast.expr | None:
        """The type the nearest binding of `name` around this node declares.

        A parameter declares its annotation; a `for` or comprehension target
        declares the element type of what it iterates, when that is a parameter
        annotated as a collection (or a `zip` of such), unpacked by the target.
        Nothing wider: no assignment or flow is followed.
        """

        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            for target, iterated in self.loops(ancestor):
                if name in self.target_names(target):
                    return self.unpacked(
                        target,
                        self.elements(self.node_indices[id(ancestor)], iterated),
                        name,
                    )
            match ancestor:
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    ):
                        if argument.arg == name:
                            return argument.annotation
        return None

    @staticmethod
    def loops(node: ast.AST) -> Iterator[tuple[ast.expr, ast.expr]]:
        match node:
            case ast.For(target=target, iter=iterated) | ast.AsyncFor(
                target=target, iter=iterated
            ):
                yield target, iterated
            case (
                ast.ListComp(generators=generators)
                | ast.SetComp(generators=generators)
                | ast.GeneratorExp(generators=generators)
                | ast.DictComp(generators=generators)
            ):
                for generator in generators:
                    yield generator.target, generator.iter

    @staticmethod
    def target_names(target: ast.expr) -> frozenset[str]:
        return frozenset(
            node.id for node in ast.walk(target) if isinstance(node, ast.Name)
        )

    def elements(self, node_index: int, iterated: ast.expr) -> ast.expr | None:
        """The element type of an iterated parameter, or a tuple of them for `zip`."""

        match iterated:
            case ast.Name(id=name):
                return self.collection_element(self.declared_type(node_index, name))
            case ast.Call(func=ast.Name(id="zip"), args=arguments):
                return ast.Tuple(
                    elts=[
                        self.elements(node_index, argument) or ast.Constant(None)
                        for argument in arguments
                    ]
                )
        return None

    @classmethod
    def collection_element(cls, annotation: ast.expr | None) -> ast.expr | None:
        match annotation:
            case ast.Subscript(
                value=collection,
                slice=ast.Tuple(elts=[element, ast.Constant(value=builtins_ellipsis)]),
            ) if (
                builtins_ellipsis is Ellipsis
                and cls.called(collection) in cls.COLLECTIONS
            ):
                return element
            case ast.Subscript(value=collection, slice=element) if cls.called(
                collection
            ) in cls.COLLECTIONS and not isinstance(element, ast.Tuple):
                return element
        return None

    @classmethod
    def unpacked(
        cls, target: ast.expr, element: ast.expr | None, name: str
    ) -> ast.expr | None:
        match target, element:
            case ast.Name(id=bound), _ if bound == name:
                return element
            case ast.Tuple(elts=targets), ast.Tuple(elts=elements) | ast.Subscript(
                value=ast.Name(id="tuple"), slice=ast.Tuple(elts=elements)
            ) if len(targets) == len(elements):
                for inner, inner_element in zip(targets, elements):
                    if name in cls.target_names(inner):
                        return cls.unpacked(inner, inner_element, name)
        return None

    @classmethod
    def passed_values(cls, value: ast.expr) -> Iterator[ast.expr]:
        """The value itself, or the items of a literal tuple, list, set or boolean chain."""

        match value:
            case (
                ast.Tuple(elts=items)
                | ast.List(elts=items)
                | ast.Set(elts=items)
                | ast.BoolOp(values=items)
            ):
                for item in items:
                    yield from cls.passed_values(item)
            case _:
                yield value

    def own(self, node_index: int, receiver: ast.expr) -> bool:
        """Whether the receiver is the enclosing method's own object or class."""

        if not isinstance(receiver, ast.Name):
            return False
        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            match ancestor:
                case ast.ClassDef(name=name):
                    return receiver.id == name
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    parameters = [*arguments.posonlyargs, *arguments.args]
                    if parameters and parameters[0].arg == receiver.id:
                        return True
        return False

    # Module positions ---------------------------------------------------------

    def module_sites(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for spelled in SpelledModules(self.parsed_module.module):
            yield self.site(
                spelled.literal,
                ModuleSpelling.detail(
                    self.references.module_rewrite(spelled), spelled.name
                ),
            )


class IdentitySpellingFamily(RestatingSiteFamily[RestatingSiteModule]):
    """Literals in positions where a string can stand for a declared identity."""

    item_type = RestatingSiteModule

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        yield from IdentityPositions(parsed_module).sites()
