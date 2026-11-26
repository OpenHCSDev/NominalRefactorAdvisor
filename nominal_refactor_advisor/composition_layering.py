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
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.hidden_imports import (
    HiddenImport,
    ImportedBinding,
    LocalImport,
    known_import_targets,
    raw_import_targets,
)
from nominal_refactor_advisor.observation_families import ShapeFamily
from nominal_refactor_advisor.owner_candidates import ModuleImportGraph

SERVICE_ACCESSORS = frozenset(("cached_property", "property"))
"""Decorators of a zero-argument accessor the root builds a service in."""


class CompactClassReference(ABC):
    """A module-scoped reference to a class: either `Name` or `bound.Name`."""

    @abstractmethod
    def owner(
        self,
        bindings: ModuleBindings,
        module: str,
    ) -> tuple[str, str | None] | None:
        raise NotImplementedError

    @classmethod
    def from_expr(cls, expression: ast.expr) -> CompactClassReference | None:
        while isinstance(expression, ast.Subscript):
            expression = expression.value
        match expression:
            case ast.Name(id=name):
                return NameClassReference(bound=name)
            case ast.Attribute(value=ast.Name(id=bound), attr=name):
                return QualifiedClassReference(bound=bound, attr=name)
        return None


@dataclass(frozen=True)
class NameClassReference(CompactClassReference):
    """A bare class name reference `Name`."""

    bound: str

    def owner(
        self,
        bindings: ModuleBindings,
        module: str,
    ) -> tuple[str, str | None] | None:
        return bindings.owner(module, self.bound)


@dataclass(frozen=True)
class QualifiedClassReference(CompactClassReference):
    """A module-qualified class reference `bound.attr`."""

    bound: str
    attr: str

    def owner(
        self,
        bindings: ModuleBindings,
        module: str,
    ) -> tuple[str, str | None] | None:
        target = bindings.imported.get(module, {}).get(self.bound)
        if target is None or target[1] is not None:
            return None
        return bindings.owner(target[0], self.attr)


@dataclass(frozen=True)
class CompactCallSpec:
    """A construction or self-method call inside a service root class."""

    self_method: str
    target_refs: tuple[CompactClassReference, ...]
    takes: tuple[str, ...]

    @classmethod
    def from_expr(cls, value: ast.expr) -> CompactCallSpec | None:
        if not isinstance(value, ast.Call):
            return None
        function = value.func
        self_method = (
            function.attr
            if isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "self"
            else ""
        )
        owner = function.value if isinstance(function, ast.Attribute) else function
        refs = tuple(
            ref
            for candidate in (owner, function)
            if (ref := CompactClassReference.from_expr(candidate)) is not None
        )
        takes = tuple(
            dict.fromkeys(
                node.attr
                for argument in (
                    *value.args,
                    *(keyword.value for keyword in value.keywords),
                )
                for node in ast.walk(argument)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            )
        )
        return cls(
            self_method=self_method,
            target_refs=refs,
            takes=takes,
        )


@dataclass(frozen=True)
class CompactInitAssignment:
    """One `self.<attr> = <Call>` assignment inside `__init__`."""

    attr: str
    call: CompactCallSpec


@dataclass(frozen=True)
class CompactMethodSpec:
    """The service-building structure of one method in a top-level class."""

    name: str
    is_accessor: bool
    init_assignments: tuple[CompactInitAssignment, ...]
    returned_call: CompactCallSpec | None

    @classmethod
    def from_function(cls, statement: ast.FunctionDef) -> CompactMethodSpec:
        init_assignments: list[CompactInitAssignment] = []
        if statement.name == "__init__":
            for assignment in ast.walk(statement):
                if (
                    isinstance(assignment, ast.Assign)
                    and len(assignment.targets) == 1
                    and isinstance(target := assignment.targets[0], ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    call = CompactCallSpec.from_expr(assignment.value)
                    if call is not None:
                        init_assignments.append(
                            CompactInitAssignment(attr=target.attr, call=call)
                        )
        is_accessor = any(
            isinstance(decorator, ast.Name) and decorator.id in SERVICE_ACCESSORS
            for decorator in statement.decorator_list
        )
        returns = [
            node.value
            for node in ast.walk(statement)
            if isinstance(node, ast.Return) and node.value
        ]
        returned_call = (
            CompactCallSpec.from_expr(returns[0]) if len(returns) == 1 else None
        )
        return cls(
            name=statement.name,
            is_accessor=is_accessor,
            init_assignments=tuple(init_assignments),
            returned_call=returned_call,
        )

    def returned(self, module, methods, pending, owner):
        if self.returned_call is None:
            return None
        return owner.built(module, self.returned_call, methods, pending)


@dataclass(frozen=True)
class CompactServiceClass:
    """One top-level class declaration, reduced to its base refs and service methods."""

    name: str
    bases: tuple[CompactClassReference, ...]
    methods: tuple[CompactMethodSpec, ...]

    @classmethod
    def from_class(cls, node: ast.ClassDef) -> CompactServiceClass:
        return cls(
            name=node.name,
            bases=tuple(
                ref
                for base in node.bases
                if (ref := CompactClassReference.from_expr(base)) is not None
            ),
            methods=tuple(
                CompactMethodSpec.from_function(statement)
                for statement in node.body
                if isinstance(statement, ast.FunctionDef)
            ),
        )


@dataclass(frozen=True)
class CompactImportReadSite:
    """One top-level `from` import or function-local import that can read a
    declaration from another layer."""

    lineno: int
    is_local: bool
    symbol: str
    bindings: tuple[ImportedBinding, ...]
    candidate_targets: tuple[str, ...]
    attribute_reads: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class CompositionLayeringModuleFacts:
    """AST-free per-module facts for composition-root layering analysis."""

    file_path: str
    module_name: str
    top_level_import_targets: tuple[str, ...]
    declared: tuple[tuple[str, int], ...]
    imported: tuple[tuple[str, str, str | None], ...]
    classes: tuple[CompactServiceClass, ...]
    read_sites: tuple[CompactImportReadSite, ...]


def _pure_attribute_reads(
    scope: ast.AST,
    aliases: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    """For each alias in `aliases`, the sorted attribute names read as
    `alias.Name` when `scope` uses `alias` in no other way."""

    if not aliases:
        return {}
    name_counts: dict[str, int] = defaultdict(int)
    attr_counts: dict[str, int] = defaultdict(int)
    attrs_by_alias: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and node.id in aliases:
            name_counts[node.id] += 1
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            alias = node.value.id
            attr_counts[alias] += 1
            attrs_by_alias[alias].add(node.attr)
    return {
        alias: tuple(sorted(attrs_by_alias[alias]))
        for alias in sorted(aliases)
        if attr_counts[alias] > 0 and attr_counts[alias] == name_counts[alias]
    }


class CompositionLayeringModuleFamily(ShapeFamily[CompositionLayeringModuleFacts]):
    """Compact per-module facts consumed by composition-root layering detection."""

    item_type = CompositionLayeringModuleFacts

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompositionLayeringModuleFacts]:
        declared_lines: dict[str, int] = {}
        imported_bindings: dict[str, tuple[str, str | None]] = {}
        class_nodes: dict[str, ast.ClassDef] = {}
        top_level_import_targets: list[str] = []
        top_level_import_froms: list[ast.ImportFrom] = []

        for statement in parsed_module.module.body:
            match statement:
                case (
                    ast.ClassDef(name=bound)
                    | ast.FunctionDef(name=bound)
                    | ast.AsyncFunctionDef(name=bound)
                ):
                    declared_lines[bound] = statement.lineno
                    if isinstance(statement, ast.ClassDef):
                        class_nodes[bound] = statement
                    else:
                        class_nodes.pop(bound, None)
                case ast.Assign(targets=targets):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            declared_lines[target.id] = statement.lineno
                            class_nodes.pop(target.id, None)
                case ast.AnnAssign(target=ast.Name(id=bound)):
                    declared_lines[bound] = statement.lineno
                    class_nodes.pop(bound, None)
                case ast.ImportFrom():
                    top_level_import_froms.append(statement)
                    top_level_import_targets.extend(
                        raw_import_targets(parsed_module, statement)
                    )
                    source = (
                        parsed_module.module_path_identity.resolve_import_from_module(
                            imported_module=statement.module,
                            level=statement.level,
                        )
                    )
                    for alias in statement.names:
                        imported_bindings[alias.asname or alias.name] = (
                            source or "",
                            alias.name,
                        )
                case ast.Import():
                    top_level_import_targets.extend(
                        raw_import_targets(parsed_module, statement)
                    )
                    for alias in statement.names:
                        if alias.asname:
                            imported_bindings[alias.asname] = (alias.name, None)

        top_bindings = [
            (statement, ImportedBinding.of(parsed_module, statement))
            for statement in top_level_import_froms
        ]
        module_aliases = frozenset(
            binding.bound
            for _statement, bindings in top_bindings
            for binding in bindings
        )
        module_attr_reads = _pure_attribute_reads(parsed_module.module, module_aliases)

        read_sites: list[CompactImportReadSite] = [
            CompactImportReadSite(
                lineno=statement.lineno,
                is_local=False,
                symbol="",
                bindings=bindings,
                candidate_targets=(),
                attribute_reads=tuple(
                    (alias, module_attr_reads[alias])
                    for alias in sorted({b.bound for b in bindings})
                    if alias in module_attr_reads
                ),
            )
            for statement, bindings in top_bindings
        ]

        local_candidates = [
            (hidden, targets)
            for hidden, targets in HiddenImport.collected_candidates(parsed_module)
            if isinstance(hidden, LocalImport)
        ]
        aliases_by_function: dict[int, set[str]] = defaultdict(set)
        function_by_id: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        local_bindings = [
            (local, targets, local.bindings) for local, targets in local_candidates
        ]
        for local, _targets, bindings in local_bindings:
            func_id = id(local.function)
            function_by_id[func_id] = local.function
            for binding in bindings:
                aliases_by_function[func_id].add(binding.bound)
        func_attr_reads = {
            func_id: _pure_attribute_reads(
                function_by_id[func_id], frozenset(aliases)
            )
            for func_id, aliases in aliases_by_function.items()
        }
        for local, candidate_targets, bindings in local_bindings:
            scoped_reads = func_attr_reads[id(local.function)]
            read_sites.append(
                CompactImportReadSite(
                    lineno=local.statement.lineno,
                    is_local=True,
                    symbol=local.symbol,
                    bindings=bindings,
                    candidate_targets=candidate_targets,
                    attribute_reads=tuple(
                        (alias, scoped_reads[alias])
                        for alias in sorted({b.bound for b in bindings})
                        if alias in scoped_reads
                    ),
                )
            )

        return [
            CompositionLayeringModuleFacts(
                file_path=parsed_module.file_path,
                module_name=parsed_module.module_name,
                top_level_import_targets=tuple(top_level_import_targets),
                declared=tuple(sorted(declared_lines.items())),
                imported=tuple(
                    sorted(
                        (bound, target[0], target[1])
                        for bound, target in imported_bindings.items()
                    )
                ),
                classes=tuple(
                    CompactServiceClass.from_class(class_nodes[name])
                    for name in sorted(class_nodes)
                ),
                read_sites=tuple(read_sites),
            )
        ]


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
    """Each scanned module's top-level declarations, top-level imports, and
    compact service classes, so names read through re-exports resolve to their
    declaring module without retaining ASTs."""

    declared: Mapping[str, frozenset[str]]
    imported: Mapping[str, Mapping[str, tuple[str, str | None]]]
    lines: Mapping[tuple[str, str], int]
    classes: Mapping[tuple[str, str], CompactServiceClass]

    @classmethod
    def from_module_facts(
        cls,
        projections: Sequence[CompositionLayeringModuleFacts],
    ) -> ModuleBindings:
        declared: dict[str, frozenset[str]] = {}
        imported: dict[str, dict[str, tuple[str, str | None]]] = {}
        lines: dict[tuple[str, str], int] = {}
        classes: dict[tuple[str, str], CompactServiceClass] = {}
        for projection in projections:
            name = projection.module_name
            declared[name] = frozenset(bound for bound, _lineno in projection.declared)
            for bound, lineno in projection.declared:
                lines[(name, bound)] = lineno
            imported[name] = {
                bound: (source, member)
                for bound, source, member in projection.imported
            }
            for service_class in projection.classes:
                classes[(name, service_class.name)] = service_class
        return cls(declared, imported, lines, classes)

    @classmethod
    def of(cls, modules: ParsedModuleSet) -> ModuleBindings:
        return cls.from_module_facts(
            tuple(
                fact
                for module in modules
                for fact in CompositionLayeringModuleFamily.collect(module)
            )
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

    def lineno(self, declaration: tuple[str, str]) -> int | None:
        return self.lines.get(declaration)

    def service_class(
        self,
        declaration: tuple[str, str],
    ) -> CompactServiceClass | None:
        return self.classes.get(declaration)


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
    def from_module_facts(
        cls,
        projections: Sequence[CompositionLayeringModuleFacts],
    ) -> CompositionRootLayering:
        ordered = tuple(projections)
        declared = (
            DeclaredRoot.composition_roots_for(Path(ordered[0].file_path))
            if ordered
            else frozenset()
        )
        if not declared:
            return cls((), (), ModuleBindings({}, {}, {}, {}))
        known = frozenset(projection.module_name for projection in ordered)
        imports = ModuleImportGraph(
            {
                projection.module_name: known_import_targets(
                    projection.top_level_import_targets, known
                )
                for projection in ordered
            }
        )
        bindings = ModuleBindings.from_module_facts(ordered)
        roots = tuple(
            root
            for qualified in sorted(declared)
            if (root := cls.root(qualified, bindings, imports)) is not None
        )
        reads = tuple(
            read
            for root in roots
            for read in cls.reads_of(root, ordered, known, bindings)
        )
        return cls(roots, reads, bindings)

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> CompositionRootLayering:
        return cls.from_module_facts(
            tuple(
                fact
                for module in modules
                for fact in CompositionLayeringModuleFamily.collect(module)
            )
        )

    @staticmethod
    def root(
        qualified: str, bindings: ModuleBindings, imports: ModuleImportGraph
    ) -> CompositionRoot | None:
        module = imports.module_of(qualified)
        if module is None:
            return None
        qualname = qualified[len(module) + 1 :]
        service_class = bindings.service_class((module, qualname))
        if service_class is None:
            return None
        services = tuple(ServiceReading(bindings, module, service_class).services())
        closures = {service.name: imports.reachable(service.module) for service in services}
        return CompositionRoot(module, qualname, services, closures)

    @staticmethod
    def reads_of(
        root: CompositionRoot,
        projections: tuple[CompositionLayeringModuleFacts, ...],
        known: frozenset[str],
        bindings: ModuleBindings,
    ) -> Iterator[LayerRead]:
        for projection in projections:
            for site in projection.read_sites:
                if site.is_local and not (
                    known_import_targets(site.candidate_targets, known)
                    - {projection.module_name}
                ):
                    continue
                attr_reads = dict(site.attribute_reads)
                for loaded, declaration in import_reads(
                    site.bindings, bindings, known, attr_reads
                ):
                    placement = (
                        root.placement(projection.module_name, loaded)
                        if site.is_local
                        else None
                    ) or root.co_declared(projection.module_name, declaration)
                    if placement is None:
                        continue
                    yield LayerRead(
                        file_path=projection.file_path,
                        lineno=site.lineno,
                        reader=projection.module_name,
                        symbol=site.symbol,
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


def import_reads(
    imported: tuple[ImportedBinding, ...],
    bindings: ModuleBindings,
    known: frozenset[str],
    scoped_attribute_reads: Mapping[str, tuple[str, ...]],
) -> Iterator[tuple[str, tuple[str, str | None]]]:
    """Each module the import loads, with the declaration it reads there: the
    module itself for `import m` or a submodule, else the name's declaring
    module and name. A module bound by name and used in its scope only as
    `alias.Name` also reads each such name, the declarations a move can take."""

    modules: set[str] = set()
    for binding in imported:
        if binding.module not in known:
            continue
        if binding.module not in modules:
            modules.add(binding.module)
            yield binding.module, (binding.module, None)
        if binding.member is None:
            if "." not in binding.module or binding.bound != binding.module.partition(".")[0]:
                yield from attribute_reads(
                    binding.bound, binding.module, bindings, known, scoped_attribute_reads
                )
            continue
        submodule = f"{binding.module}.{binding.member}"
        if submodule in known:
            if submodule not in modules:
                modules.add(submodule)
                yield submodule, (submodule, None)
            yield from attribute_reads(
                binding.bound, submodule, bindings, known, scoped_attribute_reads
            )
            continue
        owner = bindings.owner(binding.module, binding.member)
        if owner[0] != binding.module and owner[0] in known:
            yield owner[0], owner
        elif owner[1] is not None:
            yield binding.module, owner


def attribute_reads(
    alias: str,
    module: str,
    bindings: ModuleBindings,
    known: frozenset[str],
    scoped_attribute_reads: Mapping[str, tuple[str, ...]],
) -> Iterator[tuple[str, tuple[str, str | None]]]:
    """The declarations read as `alias.Name` off `module`, when its scope uses
    `alias` no other way; none when the module object itself is used."""

    for name in scoped_attribute_reads.get(alias, ()):
        owner = bindings.owner(module, name)
        if owner[1] is not None and owner[0] in known and f"{module}.{name}" not in known:
            yield owner[0], owner


@dataclass(frozen=True)
class ServiceReading:
    """The services a root class's construction builds, read from its compact
    class facts: each `self.name = Class(...)` its `__init__` or a project
    base's assigns, each accessor (`@cached_property`, `@property`) returning
    one, and each `self.method()` followed to the construction it returns."""

    bindings: ModuleBindings
    module: str
    root: CompactServiceClass

    @cached_property
    def chain(self) -> tuple[tuple[str, CompactServiceClass], ...]:
        """The root and its project bases, bases first, as construction runs them."""

        found: list[tuple[str, CompactServiceClass]] = []
        seen: set[tuple[str, str]] = set()

        def visit(module: str, service_class: CompactServiceClass) -> None:
            if (module, service_class.name) in seen:
                return
            seen.add((module, service_class.name))
            for base in service_class.bases:
                resolved = self.resolve(module, base)
                if resolved is not None:
                    visit(*resolved)
            found.append((module, service_class))

        visit(self.module, self.root)
        return tuple(found)

    def resolve(
        self,
        module: str,
        reference: CompactClassReference,
    ) -> tuple[str, CompactServiceClass] | None:
        """The project class a compact reference names, through re-exports."""

        owner = reference.owner(self.bindings, module)
        if owner is None or owner[1] is None:
            return None
        service_class = self.bindings.service_class((owner[0], owner[1]))
        return (owner[0], service_class) if service_class is not None else None

    def methods(self) -> dict[str, tuple[str, CompactMethodSpec]]:
        return {
            method.name: (module, method)
            for module, service_class in self.chain
            for method in service_class.methods
        }

    def services(self) -> Iterator[ComposedService]:
        methods = self.methods()
        position = 0
        found: dict[str, ComposedService] = {}
        for module, service_class in self.chain:
            for method in service_class.methods:
                if method.name == "__init__":
                    for assignment in method.init_assignments:
                        built = self.built(module, assignment.call, methods, frozenset())
                        if built is not None:
                            found[assignment.attr] = ComposedService(
                                assignment.attr, *built, position
                            )
                            position += 1
                elif method.is_accessor:
                    built = self.returned(module, method, methods, frozenset())
                    if built is not None:
                        found[method.name] = ComposedService(
                            method.name, *built, position
                        )
                        position += 1
        yield from found.values()

    def built(
        self,
        module: str,
        call: CompactCallSpec,
        methods: Mapping[str, tuple[str, CompactMethodSpec]],
        pending: frozenset[str],
    ) -> tuple[str, str, tuple[str, ...]] | None:
        """(module, class, services taken) when `call` constructs a project class."""

        if call.self_method in methods and call.self_method not in pending:
            return self.returned(
                *methods[call.self_method],
                methods,
                pending | {call.self_method},
            )
        resolved = next(
            (
                found
                for reference in call.target_refs
                if (found := self.resolve(module, reference)) is not None
            ),
            None,
        )
        if resolved is None:
            return None
        return (resolved[0], resolved[1].name, call.takes)

    def returned(
        self,
        module: str,
        method: CompactMethodSpec,
        methods: Mapping[str, tuple[str, CompactMethodSpec]],
        pending: frozenset[str],
    ) -> tuple[str, str, tuple[str, ...]] | None:
        return method.returned(module, methods, pending, self)


__all__ = (
    "CompactCallSpec",
    "CompactClassReference",
    "CompactImportReadSite",
    "CompactInitAssignment",
    "CompactMethodSpec",
    "CompactServiceClass",
    "ComposedService",
    "CompositionLayeringModuleFacts",
    "CompositionLayeringModuleFamily",
    "CompositionRoot",
    "CompositionRootLayering",
    "LayerRead",
    "ModuleBindings",
    "NameClassReference",
    "QualifiedClassReference",
)
