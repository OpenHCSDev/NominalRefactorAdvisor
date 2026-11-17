"""A composition root's declared layering, and the code that reads above it.

A composition root builds each service from the services it hands it: in
`self.goals = Goals(self.root, self.registry, self.bus)`, `goals` takes
`registry` and `bus`. Those constructor arguments are the repository's own
declaration of which service sits on which, so the layering is derived from
them and never kept by hand. A service needs loaded every module its class's
module loads at top level: its load closure.

A function-local import in code some service S needs loaded, of code only a
service built on S needs loaded, reads above S's layer: the import is hidden
because a top-level one would close a cycle through the layers. The read
declaration is misplaced; its home is the lowest layer whose closure holds
every module that reads it.

    [tool.nominal-refactor-advisor]
    composition-roots = ["package.module:Root", ...]
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.codemod_import_graph import SourceModuleImportGraph
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.hidden_imports import HiddenImport, ImportedBinding, LocalImport
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph

SERVICE_ACCESSORS = frozenset(("cached_property", "property"))
"""Decorators of a zero-argument accessor the root builds a service in."""


@dataclass(frozen=True)
class ComposedService:
    """One service a composition root builds: the attribute holding it, the
    class it is, and the services its construction takes."""

    name: str
    module: str
    qualname: str
    takes: tuple[str, ...]
    position: int


@dataclass(frozen=True)
class LayerRead:
    """A function-local import, in code a service needs loaded, of a
    declaration only a service built on that one needs loaded."""

    file_path: str
    lineno: int
    reader: str
    symbol: str
    loaded: str
    declaration: tuple[str, str | None]
    lower: str
    higher: str

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.lineno}"


@dataclass(frozen=True)
class ModuleBindings:
    """Each scanned module's top-level declarations and the names its top-level
    imports bind, so a name read through a re-export resolves to the module
    declaring it."""

    declared: Mapping[str, frozenset[str]]
    imported: Mapping[str, Mapping[str, tuple[str, str | None]]]
    nodes: Mapping[tuple[str, str], ast.stmt]

    @classmethod
    def of(cls, modules: ParsedModuleSet) -> ModuleBindings:
        declared: dict[str, set[str]] = defaultdict(set)
        imported: dict[str, dict[str, tuple[str, str | None]]] = defaultdict(dict)
        nodes: dict[tuple[str, str], ast.stmt] = {}
        for module in modules:
            name = module.module_name
            for statement in module.module.body:
                match statement:
                    case ast.ClassDef(name=bound) | ast.FunctionDef(name=bound) | ast.AsyncFunctionDef(name=bound):
                        declared[name].add(bound)
                        nodes[(name, bound)] = statement
                    case ast.Assign(targets=targets):
                        for target in targets:
                            if isinstance(target, ast.Name):
                                declared[name].add(target.id)
                                nodes[(name, target.id)] = statement
                    case ast.AnnAssign(target=ast.Name(id=bound)):
                        declared[name].add(bound)
                        nodes[(name, bound)] = statement
                    case ast.ImportFrom():
                        source = module.module_path_identity.resolve_import_from_module(
                            imported_module=statement.module, level=statement.level
                        )
                        for alias in statement.names:
                            imported[name][alias.asname or alias.name] = (source or "", alias.name)
                    case ast.Import():
                        for alias in statement.names:
                            if alias.asname:
                                imported[name][alias.asname] = (alias.name, None)
        return cls(
            {name: frozenset(names) for name, names in declared.items()},
            {name: dict(names) for name, names in imported.items()},
            nodes,
        )

    def owner(self, module: str, name: str) -> tuple[str, str | None]:
        """The declaration `module.name` names: its declaring module and name,
        or the module itself when `name` is a submodule or unresolved."""

        seen: set[tuple[str, str]] = set()
        while (module, name) not in seen:
            seen.add((module, name))
            if name in self.declared.get(module, ()):
                return (module, name)
            bound = self.imported.get(module, {}).get(name)
            if bound is None or bound[1] is None:
                break
            module, name = bound[0], bound[1]
        return (module, None)

    def node(self, declaration: tuple[str, str]) -> ast.stmt | None:
        return self.nodes.get(declaration)


@dataclass(frozen=True)
class CompositionRoot:
    """The services one root class builds, each with its declared dependencies
    and load closure."""

    module: str
    qualname: str
    services: tuple[ComposedService, ...]
    closures: Mapping[str, frozenset[str]]

    @cached_property
    def by_name(self) -> dict[str, ComposedService]:
        return {service.name: service for service in self.services}

    @cached_property
    def built_on(self) -> dict[str, frozenset[str]]:
        """Each service's transitive dependencies: the services below it."""

        below: dict[str, frozenset[str]] = {}

        def of(name: str, pending: frozenset[str]) -> frozenset[str]:
            if name in below:
                return below[name]
            found: set[str] = set()
            for taken in self.by_name[name].takes:
                if taken in self.by_name and taken not in pending:
                    found |= {taken, *of(taken, pending | {name})}
            below[name] = frozenset(found)
            return below[name]

        return {service.name: of(service.name, frozenset()) for service in self.services}

    @cached_property
    def layers(self) -> tuple[ComposedService, ...]:
        """The services lowest first: fewest services below, then built first."""

        return tuple(sorted(self.services, key=lambda service: (len(self.built_on[service.name]), service.position)))

    def higher(self, lower: str, loaded: str) -> str | None:
        """The smallest service built on `lower` that needs `loaded` loaded, or the
        root itself when `loaded` is the root's module."""

        above = [
            service.name
            for service in self.services
            if lower in self.built_on[service.name] and loaded in self.closures[service.name]
        ]
        if above:
            return min(above, key=lambda name: len(self.closures[name]))
        return self.qualname if loaded == self.module else None

    def placement(self, reader: str, loaded: str) -> tuple[str, str] | None:
        """(lower, higher) when code `reader` belongs to reads `loaded` above its layer."""

        for service in self.layers:
            closure = self.closures[service.name]
            if reader in closure and loaded not in closure:
                higher = self.higher(service.name, loaded)
                if higher is not None:
                    return (service.name, higher)
        return None

    def co_declared(self, reader: str, declaration: tuple[str, str | None]) -> tuple[str, str] | None:
        """(lower, higher) when code `reader` belongs to reads a declaration that
        shares its module with the class of a service built on that layer: the
        layer loads the higher service's module to read it."""

        module, name = declaration
        if name is None:
            return None
        for higher in self.services:
            if higher.module != module or higher.qualname == name:
                continue
            for lower in self.layers:
                below = {lower.name, *self.built_on[lower.name]}
                if (
                    lower.name in self.built_on[higher.name]
                    and reader in self.closures[lower.name]
                    and all(self.by_name[service].module != module for service in below)
                ):
                    return (lower.name, higher.name)
        return None

    def home(self, lowers: frozenset[str], readers: frozenset[str]) -> str | None:
        """The lowest service, at or below every layer in `lowers`, whose closure
        holds every reader: moving a declaration there never moves it sideways."""

        return next(
            (
                service.name
                for service in self.layers
                if all(service.name == lower or service.name in self.built_on[lower] for lower in lowers)
                and readers <= self.closures[service.name]
            ),
            None,
        )


@dataclass(frozen=True)
class CompositionRootLayering(ParsedModuleSetFact):
    """Every declared composition root of one module set, and each read of a
    declaration placed above the layer reading it."""

    roots: tuple[CompositionRoot, ...]
    reads: tuple[LayerRead, ...]
    bindings: ModuleBindings

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> CompositionRootLayering:
        parsed = tuple(modules)
        declared = DeclaredRoot.composition_roots_for(Path(parsed[0].file_path)) if parsed else frozenset()
        if not declared:
            return cls((), (), ModuleBindings({}, {}, {}))
        graph = SourceModuleImportGraph.of_modules(parsed)
        imports = ModuleImportGraph(graph.import_edges_by_module)
        bindings = ModuleBindings.of(modules)
        roots = tuple(
            root
            for qualified in sorted(declared)
            if (root := cls.root(qualified, bindings, imports)) is not None
        )
        reads = tuple(read for root in roots for read in cls.reads_of(root, parsed, graph, bindings))
        return cls(roots, reads, bindings)

    @staticmethod
    def root(
        qualified: str, bindings: ModuleBindings, imports: ModuleImportGraph
    ) -> CompositionRoot | None:
        module = imports.module_of(qualified)
        if module is None:
            return None
        qualname = qualified[len(module) + 1 :]
        node = bindings.node((module, qualname))
        if not isinstance(node, ast.ClassDef):
            return None
        services = tuple(ServiceReading(bindings, module, node).services())
        closures = {
            service.name: frozenset(closure(imports, service.module)) for service in services
        }
        return CompositionRoot(module, qualname, services, closures)

    @staticmethod
    def reads_of(
        root: CompositionRoot,
        parsed: tuple[ParsedModule, ...],
        graph: SourceModuleImportGraph,
        bindings: ModuleBindings,
    ) -> Iterator[LayerRead]:
        known = graph.known_module_names
        for module in parsed:
            hidden = {
                id(found.statement): found
                for found in HiddenImport.collected(module, graph)
                if isinstance(found, LocalImport)
            }
            statements = [
                *(statement for statement in module.module.body if isinstance(statement, ast.ImportFrom)),
                *(found.statement for found in hidden.values()),
            ]
            for statement in statements:
                local = hidden.get(id(statement))
                for loaded, declaration in import_reads(ImportedBinding.of(module, statement), bindings, known):
                    placement = (
                        root.placement(module.module_name, loaded) if local is not None else None
                    ) or root.co_declared(module.module_name, declaration)
                    if placement is None:
                        continue
                    yield LayerRead(
                        file_path=module.file_path,
                        lineno=statement.lineno,
                        reader=module.module_name,
                        symbol="" if local is None else enclosing_symbol(local),
                        loaded=loaded,
                        declaration=declaration,
                        lower=placement[0],
                        higher=placement[1],
                    )

    @cached_property
    def by_declaration(self) -> dict[tuple[str, str | None], tuple[LayerRead, ...]]:
        grouped: dict[tuple[str, str | None], list[LayerRead]] = defaultdict(list)
        for read in self.reads:
            grouped[read.declaration].append(read)
        return {declaration: tuple(reads) for declaration, reads in grouped.items()}

    def home(self, declaration: tuple[str, str | None]) -> tuple[CompositionRoot, str] | None:
        """The root and lowest service layer whose closure holds every reader."""

        reads = self.by_declaration.get(declaration, ())
        if not reads:
            return None
        root = next(root for root in self.roots if any(read.lower in root.by_name for read in reads))
        layer = root.home(frozenset(read.lower for read in reads), frozenset(read.reader for read in reads))
        return None if layer is None else (root, layer)

    def layer_of(self, module: str, names: Iterable[str]) -> frozenset[str]:
        """The modules the home layer of `module`'s misplaced declarations among
        `names` loads, where a move of them belongs; empty when none of them is
        misplaced."""

        homes = [home for name in names if (home := self.home((module, name))) is not None]
        return frozenset.intersection(*(root.closures[layer] for root, layer in homes)) if homes else frozenset()


def closure(imports: ModuleImportGraph, start: str) -> set[str]:
    """Every module loading `start` loads, `start` included."""

    seen = {start}
    pending = [start]
    while pending:
        for imported in imports.imports.get(pending.pop(), ()):
            if imported not in seen:
                seen.add(imported)
                pending.append(imported)
    return seen


def enclosing_symbol(hidden: LocalImport) -> str:
    """The dotted class and function names enclosing the import."""

    return ".".join(
        node.name
        for node in reversed(hidden.ancestors)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )


def import_reads(
    imported: tuple[ImportedBinding, ...], bindings: ModuleBindings, known: frozenset[str]
) -> Iterator[tuple[str, tuple[str, str | None]]]:
    """Each module the import loads, with the declaration it reads there: the
    module itself for `import m` or a submodule, else the name's declaring
    module and name."""

    modules: set[str] = set()
    for binding in imported:
        if binding.module not in known:
            continue
        if binding.module not in modules:
            modules.add(binding.module)
            yield binding.module, (binding.module, None)
        if binding.member is None:
            continue
        submodule = f"{binding.module}.{binding.member}"
        if submodule in known:
            if submodule not in modules:
                modules.add(submodule)
                yield submodule, (submodule, None)
            continue
        owner = bindings.owner(binding.module, binding.member)
        if owner[0] != binding.module and owner[0] in known:
            yield owner[0], owner
        elif owner[1] is not None:
            yield binding.module, owner


@dataclass(frozen=True)
class ServiceReading:
    """The services a root class's construction builds, read from its source:
    each `self.name = Class(...)` its `__init__` or a project base's assigns,
    each accessor (`@cached_property`, `@property`) returning one, and each
    `self.method()` followed to the construction it returns."""

    bindings: ModuleBindings
    module: str
    root: ast.ClassDef

    @cached_property
    def chain(self) -> tuple[tuple[str, ast.ClassDef], ...]:
        """The root and its project bases, bases first, as construction runs them."""

        found: list[tuple[str, ast.ClassDef]] = []
        seen: set[tuple[str, str]] = set()

        def visit(module: str, node: ast.ClassDef) -> None:
            if (module, node.name) in seen:
                return
            seen.add((module, node.name))
            for base in node.bases:
                resolved = self.resolve(module, base)
                if resolved is not None:
                    visit(*resolved)
            found.append((module, node))

        visit(self.module, self.root)
        return tuple(found)

    def resolve(self, module: str, expression: ast.expr) -> tuple[str, ast.ClassDef] | None:
        """The project class an expression names, through subscripts and re-exports."""

        while isinstance(expression, ast.Subscript):
            expression = expression.value
        match expression:
            case ast.Name(id=name):
                owner = self.bindings.owner(module, name)
            case ast.Attribute(value=ast.Name(id=bound), attr=name):
                target = self.bindings.imported.get(module, {}).get(bound)
                if target is None or target[1] is not None:
                    return None
                owner = self.bindings.owner(target[0], name)
            case _:
                return None
        if owner[1] is None:
            return None
        node = self.bindings.node((owner[0], owner[1]))
        return (owner[0], node) if isinstance(node, ast.ClassDef) else None

    def methods(self) -> dict[str, tuple[str, ast.FunctionDef]]:
        return {
            statement.name: (module, statement)
            for module, node in self.chain
            for statement in node.body
            if isinstance(statement, ast.FunctionDef)
        }

    def services(self) -> Iterator[ComposedService]:
        methods = self.methods()
        position = 0
        found: dict[str, ComposedService] = {}
        for module, node in self.chain:
            for statement in node.body:
                if not isinstance(statement, ast.FunctionDef):
                    continue
                if statement.name == "__init__":
                    for assignment in ast.walk(statement):
                        if (
                            isinstance(assignment, ast.Assign)
                            and len(assignment.targets) == 1
                            and isinstance(target := assignment.targets[0], ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            built = self.built(module, assignment.value, methods, frozenset())
                            if built is not None:
                                found[target.attr] = ComposedService(target.attr, *built, position)
                                position += 1
                elif any(
                    isinstance(decorator, ast.Name) and decorator.id in SERVICE_ACCESSORS
                    for decorator in statement.decorator_list
                ):
                    built = self.returned(module, statement, methods, frozenset())
                    if built is not None:
                        found[statement.name] = ComposedService(statement.name, *built, position)
                        position += 1
        yield from found.values()

    def built(
        self,
        module: str,
        value: ast.expr,
        methods: Mapping[str, tuple[str, ast.FunctionDef]],
        pending: frozenset[str],
    ) -> tuple[str, str, tuple[str, ...]] | None:
        """(module, class, services taken) when `value` constructs a project class."""

        if not isinstance(value, ast.Call):
            return None
        function = value.func
        match function:
            case ast.Attribute(value=ast.Name(id="self"), attr=name) if name in methods and name not in pending:
                return self.returned(*methods[name], methods, pending | {name})
        owner = function.value if isinstance(function, ast.Attribute) else function
        resolved = self.resolve(module, owner) or self.resolve(module, function)
        if resolved is None:
            return None
        takes = tuple(
            dict.fromkeys(
                node.attr
                for argument in (*value.args, *(keyword.value for keyword in value.keywords))
                for node in ast.walk(argument)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            )
        )
        return (resolved[0], resolved[1].name, takes)

    def returned(
        self,
        module: str,
        function: ast.FunctionDef,
        methods: Mapping[str, tuple[str, ast.FunctionDef]],
        pending: frozenset[str],
    ) -> tuple[str, str, tuple[str, ...]] | None:
        returns = [node.value for node in ast.walk(function) if isinstance(node, ast.Return) and node.value]
        if len(returns) != 1:
            return None
        return self.built(module, returns[0], methods, pending)


__all__ = (
    "ComposedService",
    "CompositionRoot",
    "CompositionRootLayering",
    "LayerRead",
    "ModuleBindings",
)
