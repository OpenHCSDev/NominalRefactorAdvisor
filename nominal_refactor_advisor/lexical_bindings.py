"""Shared lexical binding and import-origin declarations."""

from __future__ import annotations

import ast
import __future__
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import (
    ClassVar,
    Generic,
    Self,
    TypeVar,
)

from .assignment_projection import AssignmentTargetNameProjection
from .ast_field_decoders import (
    capture_names,
    declared_annotations,
)
from .python_module_identity import PythonModulePathIdentity


class CompactParameterKind(StrEnum):
    """Python parameter kinds with their binding behavior on each member."""

    POSITIONAL_ONLY = "positional_only", True, False, False
    POSITIONAL_OR_KEYWORD = "positional_or_keyword", True, True, False
    VAR_POSITIONAL = "var_positional", True, False, True
    KEYWORD_ONLY = "keyword_only", False, True, False
    VAR_KEYWORD = "var_keyword", False, True, True

    def __new__(
        cls,
        value: str,
        accepts_positional: bool,
        accepts_keyword: bool,
        variadic: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._accepts_positional = accepts_positional
        member._accepts_keyword = accepts_keyword
        member._variadic = variadic
        return member

    @property
    def accepts_positional(self) -> bool:
        return self._accepts_positional

    @property
    def accepts_keyword(self) -> bool:
        return self._accepts_keyword

    @property
    def variadic(self) -> bool:
        return self._variadic


@dataclass(frozen=True)
class FunctionArgumentSource:
    """Actual argument nodes and kinds, independent of default alignment."""

    argument: ast.arg
    kind: CompactParameterKind

    @classmethod
    def from_arguments(cls, arguments: ast.arguments) -> tuple[Self, ...]:
        parameters = [
            cls(argument, CompactParameterKind.POSITIONAL_ONLY)
            for argument in arguments.posonlyargs
        ]
        parameters.extend(
            cls(argument, CompactParameterKind.POSITIONAL_OR_KEYWORD)
            for argument in arguments.args
        )
        if arguments.vararg is not None:
            parameters.append(
                cls(arguments.vararg, CompactParameterKind.VAR_POSITIONAL)
            )
        parameters.extend(
            cls(argument, CompactParameterKind.KEYWORD_ONLY)
            for argument in arguments.kwonlyargs
        )
        if arguments.kwarg is not None:
            parameters.append(cls(arguments.kwarg, CompactParameterKind.VAR_KEYWORD))
        return tuple(parameters)

    @staticmethod
    def annotation_roots(
        parameters: Iterable[FunctionArgumentSource], returns: ast.expr | None
    ) -> tuple[ast.expr, ...]:
        """Project expressions, preserving the supplied argument-root order."""
        return tuple(
            parameter.argument.annotation
            for parameter in parameters
            if parameter.argument.annotation is not None
        ) + (() if returns is None else (returns,))


DefaultAnswerT = TypeVar("DefaultAnswerT")


class ParameterDefaultQuestion(ABC, Generic[DefaultAnswerT]):
    """A consumer's question about one parameter, answered once per default state."""

    @abstractmethod
    def defaulted(self, source: DefaultedParameterSource) -> DefaultAnswerT:
        raise NotImplementedError

    @abstractmethod
    def undefaulted(self, source: UndefaultedParameterSource) -> DefaultAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionParameterSource(FunctionArgumentSource, ABC):
    """Actual argument/default nodes in signature order, not execution evidence."""

    @classmethod
    def from_arguments(
        cls, arguments: ast.arguments
    ) -> tuple[FunctionParameterSource, ...]:
        positional = (*arguments.posonlyargs, *arguments.args)
        undefaulted = len(positional) - len(arguments.defaults)
        defaults: dict[ast.arg, ast.expr] = dict(
            zip(positional[undefaulted:], arguments.defaults, strict=True)
        )
        defaults.update(
            (argument, default)
            for argument, default in zip(
                arguments.kwonlyargs, arguments.kw_defaults, strict=True
            )
            # The ast boundary: a keyword-only parameter without a default holds None.
            if default is not None
        )
        return tuple(
            DefaultedParameterSource(
                parameter.argument, parameter.kind, defaults[parameter.argument]
            )
            if parameter.argument in defaults
            else UndefaultedParameterSource(parameter.argument, parameter.kind)
            for parameter in FunctionArgumentSource.from_arguments(arguments)
        )

    @abstractmethod
    def answer(self, question: ParameterDefaultQuestion[DefaultAnswerT]) -> DefaultAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class DefaultedParameterSource(FunctionParameterSource):
    """A parameter declared with a default expression."""

    default: ast.expr

    def answer(self, question: ParameterDefaultQuestion[DefaultAnswerT]) -> DefaultAnswerT:
        return question.defaulted(self)


@dataclass(frozen=True)
class UndefaultedParameterSource(FunctionParameterSource):
    """A parameter declared without a default."""

    def answer(self, question: ParameterDefaultQuestion[DefaultAnswerT]) -> DefaultAnswerT:
        return question.undefaulted(self)


class DictionaryEvaluationVisitor(ast.NodeVisitor):
    """Visit each original key before its value, including unpack operands."""

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if key is not None:
                self.visit(key)
            self.visit(value)


class FunctionDefaultVisitor(ast.NodeVisitor):
    """Visit defaults when a function is created, without entering its lambda body."""

    @staticmethod
    def default_roots(arguments: ast.arguments) -> tuple[ast.expr, ...]:
        return tuple(
            default
            for default in (*arguments.defaults, *arguments.kw_defaults)
            if default is not None
        )

    def visit_argument_defaults(self, arguments: ast.arguments) -> None:
        for default in self.default_roots(arguments):
            self.visit(default)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_argument_defaults(node.args)


class FunctionAnnotationVisitor(FunctionDefaultVisitor):
    """Shared annotation inventory; iteration alone asserts no evaluation order."""

    def visit_function_annotations(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        roots = FunctionArgumentSource.annotation_roots(
            FunctionArgumentSource.from_arguments(node.args), node.returns
        )
        self.visit_unordered_annotations(roots)

    def visit_annotation(self, expression: ast.expr) -> None:
        self.visit(expression)

    def visit_ordered_annotations(self, roots: tuple[ast.expr, ...]) -> None:
        for expression in roots:
            self.visit_annotation(expression)

    def visit_unordered_annotations(self, roots: tuple[ast.expr, ...]) -> None:
        # Inventory consumers retain all roots without asserting execution order.
        self.visit_ordered_annotations(roots)


@dataclass(frozen=True)
class ImportAlias:
    """One import alias spelled without an `as` clause."""

    name: str

    @classmethod
    def from_alias(cls, alias: ast.alias) -> ImportAlias:
        """The ast boundary: the authored `as` clause picks the alias type."""

        if alias.name == "*":
            return StarImportAlias(alias.name)
        if alias.asname is None:
            return ImportAlias(alias.name)
        return RenamedImportAlias(alias.name, alias.asname)

    @property
    def source(self) -> str:
        return self.name

    @property
    def ast_alias(self) -> ast.alias:
        return ast.alias(name=self.name)

    @property
    def canonical_key(self) -> tuple[str, str]:
        """Return the source-spelling key for commutative import merging."""

        return self.name, ""

    @property
    def declares_reexport(self) -> bool:
        """`X as X` is the explicit public re-export spelling."""

        return False

    def legacy_marker_spellings(self) -> tuple[str, ...]:
        """Spellings this alias authors for an imported name: none without `as`."""

        return ()

    def renamed(self, new_name: str) -> ImportAlias:
        """The alias importing a renamed declaration under the binding it implies."""

        return ImportAlias(new_name)

    def bound_name_in(self, declaration: ImportDeclarationABC) -> str:
        return declaration._unaliased_bound_name(self)

    def bound_names_in(self, declaration: ImportDeclarationABC) -> tuple[str, ...]:
        """The names this alias binds individually: its one name, none for a star."""

        return (self.bound_name_in(declaration),)

    def module_reference_in(self, declaration: ModuleImportDeclaration) -> str:
        """The dotted module path a module import of this alias reaches by its binding."""

        return declaration._unaliased_bound_name(self)

    @property
    def module_spellings(self) -> tuple[tuple[str, str], ...]:
        """Dotted module paths this module-import alias makes reachable, with their spelling."""

        parts = self.name.split(".")
        return tuple(
            (dotted, dotted)
            for dotted in (".".join(parts[:stop]) for stop in range(1, len(parts) + 1))
        )


@dataclass(frozen=True)
class RenamedImportAlias(ImportAlias):
    """One import alias spelled with an `as` clause."""

    asname: str

    @property
    def source(self) -> str:
        return f"{self.name} as {self.asname}"

    @property
    def ast_alias(self) -> ast.alias:
        return ast.alias(name=self.name, asname=self.asname)

    @property
    def canonical_key(self) -> tuple[str, str]:
        return self.name, self.asname

    @property
    def declares_reexport(self) -> bool:
        return self.asname == self.name

    def legacy_marker_spellings(self) -> tuple[str, ...]:
        return (self.asname,)

    def renamed(self, new_name: str) -> ImportAlias:
        """A foreign `as` binding survives; a re-export follows the new name."""

        return RenamedImportAlias(
            new_name, new_name if self.declares_reexport else self.asname
        )

    def bound_name_in(self, declaration: ImportDeclarationABC) -> str:
        return self.asname

    def module_reference_in(self, declaration: ModuleImportDeclaration) -> str:
        return self.name

    @property
    def module_spellings(self) -> tuple[tuple[str, str], ...]:
        return ((self.name, self.asname),)


@dataclass(frozen=True)
class StarImportAlias(ImportAlias):
    """`from m import *`: it binds whatever `m` exports, no one name of its own."""

    def bound_name_in(self, declaration: ImportDeclarationABC) -> str:
        raise ValueError("A star request has no explicit individual binding")

    def bound_names_in(self, declaration: ImportDeclarationABC) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class ImportFromModuleName:
    """Canonical source spelling for an ImportFrom module."""

    source: str
    package_binding: ClassVar[str] = "__package__"

    def resolve_from_package(self, package: Callable[[], str]) -> str:
        """Native resolution consumes a runtime package only for relative requests."""
        if not self.is_relative:
            return self.module_name
        current = package()
        parts = current.rsplit(".", self.level - 1)
        if not current or len(parts) < self.level:
            raise ValueError("Relative import exceeds its actual package boundary")
        root = parts[0]
        return f"{root}.{self.module_name}" if self.module_name else root

    @property
    def is_relative(self) -> bool:
        return self.source.startswith(".")

    @property
    def root_module_name(self) -> str:
        return self.source.lstrip(".").partition(".")[0]

    @property
    def module_name(self) -> str:
        return self.source.lstrip(".")

    @property
    def level(self) -> int:
        return len(self.source) - len(self.module_name)

    def resolve(self, module_identity: PythonModulePathIdentity) -> str | None:
        return module_identity.resolve_import_from_module(
            imported_module=self.module_name or None,
            level=self.level,
        )

    @classmethod
    def from_node(cls, node: ast.ImportFrom) -> ImportFromModuleName:
        relative_prefix = "." * node.level
        if node.module is None:
            return cls(relative_prefix)
        return cls(f"{relative_prefix}{node.module}")


ImportContextT = TypeVar("ImportContextT")
ImportResolutionT = TypeVar("ImportResolutionT")


class ImportOriginResolverABC(ABC, Generic[ImportContextT, ImportResolutionT]):
    """Interpret an actual import origin with consumer-owned execution evidence."""

    @abstractmethod
    def _module_import_resolution(
        self, origin: ImportedNameOrigin, context: ImportContextT
    ) -> ImportResolutionT:
        raise NotImplementedError

    @abstractmethod
    def _member_import_resolution(
        self, origin: ImportedNameOrigin, context: ImportContextT
    ) -> ImportResolutionT:
        raise NotImplementedError


@dataclass(frozen=True)
class ImportDeclarationABC(ABC):
    """One AST-free import request; binding views derive from its aliases."""

    aliases: tuple[ImportAlias, ...]

    @property
    @abstractmethod
    def module_references(self) -> tuple[ImportFromModuleName, ...]:
        """Project unresolved module references from this declaration's syntax."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_future_import(self) -> bool:
        """Distinguish a language directive from an ordinary module import."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_prefix(self) -> str:
        raise NotImplementedError

    def source_for(self, aliases: tuple[ImportAlias, ...]) -> str:
        return f"{self.source_prefix} {', '.join(alias.source for alias in aliases)}\n"

    @property
    def source(self) -> str:
        return self.source_for(self.aliases)

    @abstractmethod
    def _unaliased_bound_name(self, alias: ImportAlias) -> str:
        raise NotImplementedError

    @abstractmethod
    def requested_module_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def qualified_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        """A source catalogue path, not a claim about the imported runtime object."""
        raise NotImplementedError

    @abstractmethod
    def resolve_origin(
        self,
        resolver: ImportOriginResolverABC[ImportContextT, ImportResolutionT],
        origin: ImportedNameOrigin,
        context: ImportContextT,
    ) -> ImportResolutionT:
        raise NotImplementedError

    def names(self) -> tuple[str, ...]:
        return tuple(
            name for alias in self.aliases for name in alias.bound_names_in(self)
        )

    def origins(
        self, module_identity: PythonModulePathIdentity
    ) -> tuple[ImportedNameOrigin, ...]:
        return tuple(
            ImportedNameOrigin(self, index, module_identity)
            for index, alias in enumerate(self.aliases)
            for _name in alias.bound_names_in(self)
        )

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.source_for((alias,)))
            for alias in self.aliases
            for name in alias.bound_names_in(self)
        )


class ModuleImportDeclaration(ImportDeclarationABC):
    source_prefix = "import"
    is_future_import = False

    @property
    def module_references(self) -> tuple[ImportFromModuleName, ...]:
        return tuple(ImportFromModuleName(alias.name) for alias in self.aliases)

    def resolve_origin(
        self,
        resolver: ImportOriginResolverABC[ImportContextT, ImportResolutionT],
        origin: ImportedNameOrigin,
        context: ImportContextT,
    ) -> ImportResolutionT:
        return resolver._module_import_resolution(origin, context)

    def _unaliased_bound_name(self, alias: ImportAlias) -> str:
        return alias.name.partition(".")[0]

    def requested_module_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return alias.name

    def qualified_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str:
        return alias.module_reference_in(self)


@dataclass(frozen=True)
class FromImportDeclaration(ImportDeclarationABC):
    module_name: ImportFromModuleName

    @property
    def module_references(self) -> tuple[ImportFromModuleName, ...]:
        return (self.module_name,)

    @property
    def is_future_import(self) -> bool:
        return self.module_name.source == __future__.__name__

    @property
    def source_prefix(self) -> str:
        return f"from {self.module_name.source} import"

    def resolve_origin(
        self,
        resolver: ImportOriginResolverABC[ImportContextT, ImportResolutionT],
        origin: ImportedNameOrigin,
        context: ImportContextT,
    ) -> ImportResolutionT:
        return resolver._member_import_resolution(origin, context)

    def _unaliased_bound_name(self, alias: ImportAlias) -> str:
        return alias.name

    def requested_module_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        return self.module_name.resolve(module_identity)

    def qualified_name(
        self,
        alias: ImportAlias,
        module_identity: PythonModulePathIdentity,
    ) -> str | None:
        module_name = self.requested_module_name(alias, module_identity)
        return None if module_name is None else f"{module_name}.{alias.name}"


@dataclass(frozen=True)
class ImportedNameOrigin:
    """A selected source position in the actual import declaration."""

    declaration: ImportDeclarationABC
    alias_index: int
    module_identity: PythonModulePathIdentity

    def resolve(
        self,
        resolver: ImportOriginResolverABC[ImportContextT, ImportResolutionT],
        context: ImportContextT,
    ) -> ImportResolutionT:
        return self.declaration.resolve_origin(resolver, self, context)

    def __post_init__(self) -> None:
        if not 0 <= self.alias_index < len(self.declaration.aliases):
            raise ValueError(
                "Import origin requires an alias position within its declaration"
            )
        self.alias.bound_name_in(self.declaration)

    @property
    def alias(self) -> ImportAlias:
        return self.declaration.aliases[self.alias_index]

    @property
    def bound_name(self) -> str:
        return self.alias.bound_name_in(self.declaration)

    @property
    def qualified_name(self) -> str | None:
        return self.declaration.qualified_name(self.alias, self.module_identity)

    @property
    def requested_module_name(self) -> str | None:
        return self.declaration.requested_module_name(self.alias, self.module_identity)

    @property
    def source(self) -> str:
        return self.declaration.source_for((self.alias,))


@dataclass(frozen=True)
class ImportBoundNameProjection:
    """AST bridge to the declaration owning import syntax and binding rules."""

    statement: ast.Import | ast.ImportFrom

    @cached_property
    def declaration(self) -> ImportDeclarationABC:
        aliases = tuple(
            ImportAlias.from_alias(alias) for alias in self.statement.names
        )
        if isinstance(self.statement, ast.Import):
            return ModuleImportDeclaration(aliases)
        return FromImportDeclaration(
            aliases, ImportFromModuleName.from_node(self.statement)
        )

    def names(self) -> tuple[str, ...]:
        return self.declaration.names()

    def origins(
        self, module_identity: PythonModulePathIdentity
    ) -> tuple[ImportedNameOrigin, ...]:
        return self.declaration.origins(module_identity)

    def name_sources(self) -> tuple[tuple[str, str], ...]:
        return self.declaration.name_sources()

    def alias_bound_name(self, alias: ast.alias) -> str:
        # A star binds no one name; its spelling here stays the empty name.
        return "".join(ImportAlias.from_alias(alias).bound_names_in(self.declaration))


@dataclass
class ScopeBindingCollector(ast.NodeVisitor):
    """Collect compile-time bindings without descending into child scopes."""

    bound_names: set[str] = field(default_factory=set)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.bound_names.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        for declared_annotation in declared_annotations(node):
            self.visit(declared_annotation)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound_names.add(node.name)
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)

    def visit_Import(self, node: ast.Import) -> None:
        self.bound_names.update(ImportBoundNameProjection(node).names())

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bound_names.update(ImportBoundNameProjection(node).names())

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        for capture_name in capture_names(node):
            self.bound_names.add(capture_name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        for capture_name in capture_names(node):
            self.bound_names.add(capture_name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        for capture_name in capture_names(node):
            self.bound_names.add(capture_name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        for capture_name in capture_names(node):
            self.bound_names.add(capture_name)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        _ComprehensionContainingScopeBindingCollector(self.bound_names).generic_visit(
            node
        )

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


@dataclass
class ImportSeparatingScopeBindingCollector(ScopeBindingCollector):
    """The same bindings, with import statements kept apart from every other form."""

    imports: list[ast.Import | ast.ImportFrom] = field(default_factory=list)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)


class _ComprehensionContainingScopeBindingCollector(ScopeBindingCollector):
    """Only walrus targets escape a comprehension; headers retain normal ownership."""

    def visit_Name(self, node: ast.Name) -> None:
        return

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.bound_names.update(_store_names(node.target))
        self.visit(node.value)


def _store_names(target: ast.expr) -> tuple[str, ...]:
    return AssignmentTargetNameProjection(target).names


class LexicalScopeBindingAuthority:
    """Recover names bound by one lexical scope without entering child scopes."""

    @staticmethod
    def bound_names(nodes: Iterable[ast.AST]) -> frozenset[str]:
        collector = ScopeBindingCollector()
        for node in nodes:
            collector.visit(node)
        return frozenset(collector.bound_names)

    @staticmethod
    def import_only_bindings(
        nodes: Iterable[ast.AST],
    ) -> tuple[frozenset[str], dict[str, ast.Import | ast.ImportFrom]]:
        """One scope's names bound by anything but an import, and each name bound
        by exactly one import and nothing else, with that import."""

        collector = ImportSeparatingScopeBindingCollector()
        for node in nodes:
            collector.visit(node)
        by_name: dict[str, list[ast.Import | ast.ImportFrom]] = defaultdict(list)
        for statement in collector.imports:
            for name in ImportBoundNameProjection(statement).names():
                by_name[name].append(statement)
        imported = {
            name: statement
            for name, (statement, *rest) in by_name.items()
            if not rest and name not in collector.bound_names
        }
        return (
            frozenset(collector.bound_names | (by_name.keys() - imported.keys())),
            imported,
        )

    @staticmethod
    def enclosing_comprehensions(
        reference: ast.AST,
        boundary: ast.AST,
        parents: Mapping[ast.AST, ast.AST],
    ) -> tuple[ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp, ...]:
        """Nearest lexical frames, excluding their outer-evaluated first iterable."""
        ancestry = [reference]
        while ancestry[-1] is not boundary:
            parent = parents.get(ancestry[-1])
            if parent is None:
                return ()
            ancestry.append(parent)
        return tuple(
            node
            for node in ancestry
            if isinstance(
                node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            )
            and node.generators[0].iter not in ancestry
        )

    @staticmethod
    def argument_names(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> frozenset[str]:
        arguments = node.args
        return frozenset(
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ) | frozenset(
            argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument
        )


LEXICAL_SCOPE_BINDING_AUTHORITY = LexicalScopeBindingAuthority()
