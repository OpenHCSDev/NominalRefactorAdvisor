"""Source edits that move static projection calls onto a declared adapter family."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Self

from .ast_tools import ParsedModule, module_syntax_index
from .class_index import (
    ModuleNominalBindingAuthority,
    CompactNominalBinding,
    ModuleNominalBindingSnapshotPolicy,
)
from .codemod_foreign_adapter_source import AdapterPropertyBodies
from .codemod_foreign_family_source import ForeignFamilyReference, RewrittenLineFit
from .codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .declared_family import ForeignAdapter
from .observation_families import ShapeFamily, TypeArgumentResolver
from .observation_shapes import ForeignFamilySiteModule, StaticProjectionSiteShape


@dataclass(frozen=True)
class StaticProjection:
    """A loaded class of static projections that one adapter family now answers.

    Every public static or class method of the projector is a property of the
    family; the question names are the family's property names, so the
    mapping is name equality and nothing is listed by hand.
    """

    projector: type
    family: type[ForeignAdapter]

    @classmethod
    def declared_for(cls, qualified_name: str) -> Self | None:
        """The projection when the loaded class's public callables all match one family."""

        projector = cls.class_named(qualified_name)
        if projector is None:
            return None
        questions = cls.questions_of(projector)
        families = tuple(
            family
            for family in ForeignAdapter.concrete_families()
            if questions and questions <= frozenset(AdapterPropertyBodies(family).property_names)
        )
        return cls(projector, families[0]) if len(families) == 1 else None

    @staticmethod
    def class_named(qualified_name: str) -> type | None:
        """The loaded class a binding's qualified name reaches, through re-exports."""

        module_name, _, class_name = qualified_name.rpartition(".")
        module = sys.modules.get(module_name)
        declared = None if module is None else vars(module).get(class_name)
        return declared if isinstance(declared, type) else None

    @staticmethod
    def questions_of(projector: type) -> frozenset[str]:
        return frozenset(
            name
            for name, declared in vars(projector).items()
            if not name.startswith("_") and isinstance(declared, staticmethod | classmethod)
        )

    @cached_property
    def questions(self) -> frozenset[str]:
        return self.questions_of(self.projector)


def named_bindings(parsed_module: ParsedModule) -> dict[str, CompactNominalBinding]:
    """Final name bindings, read past star imports; callers prove the star imports."""

    return (
        ModuleNominalBindingAuthority(parsed_module)
        .snapshot_before(
            None, policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION
        )
        .bindings_by_name
    )


def star_imported_modules(parsed_module: ParsedModule) -> tuple[str, ...]:
    """Absolute names of the modules `parsed_module` star-imports."""

    parts = parsed_module.module_name.split(".")
    package = parts if parsed_module.file_path.endswith("__init__.py") else parts[:-1]
    return tuple(
        ".".join((*package[: len(package) - statement.level + 1 if statement.level else 0], *filter(None, [statement.module])))
        for statement in ast.walk(parsed_module.module)
        if isinstance(statement, ast.ImportFrom)
        and any(alias.name == "*" for alias in statement.names)
    )


@dataclass(frozen=True)
class StaticProjectionSite:
    """`Projector.question(argument)`, one positional argument."""

    call: ast.Call
    argument: ast.expr
    question: str
    name: str


@dataclass(frozen=True)
class StaticProjectionSites:
    """Every use in one module of a declared static projection and its import bindings."""

    parsed_module: ParsedModule
    projection: StaticProjection

    @cached_property
    def projector_name(self) -> str:
        return f"{self.projection.projector.__module__}.{self.projection.projector.__qualname__}"

    @cached_property
    def shadowing_names(self) -> frozenset[str]:
        """Names the module rebinds outside imports; a site spelled by one is unproved."""

        names: set[str] = set()
        for node in ast.walk(self.parsed_module.module):
            match node:
                case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                    names.add(name)
                case ast.arg(arg=name):
                    names.add(name)
        return frozenset(names)

    @cached_property
    def bound_names(self) -> frozenset[str]:
        """Names this module binds that reach the projector class.

        A star import cannot silently rebind a name when every star-imported
        module is loaded and either lacks the name or holds the same class.
        """

        return frozenset(
            name
            for name, binding in named_bindings(self.parsed_module).items()
            if name not in self.shadowing_names
            and StaticProjection.class_named(binding.qualified_name)
            is self.projection.projector
            and self.star_imports_keep(name)
        )

    def star_imports_keep(self, name: str) -> bool:
        projector = self.projection.projector
        return all(
            (module := sys.modules.get(star_module)) is not None
            and vars(module).get(name, projector) is projector
            for star_module in star_imported_modules(self.parsed_module)
        )

    @cached_property
    def sites(self) -> tuple[StaticProjectionSite, ...]:
        sites = []
        for node in ast.walk(self.parsed_module.module):
            match node:
                case ast.Call(
                    func=ast.Attribute(value=ast.Name(id=name), attr=question),
                    args=[argument],
                    keywords=[],
                ) if (
                    name in self.bound_names
                    and question in self.projection.questions
                    and not isinstance(argument, ast.Starred)
                ):
                    sites.append(StaticProjectionSite(node, argument, question, name))
        return tuple(sorted(sites, key=lambda site: (site.call.lineno, site.call.col_offset)))

    @cached_property
    def retired_names(self) -> tuple[str, ...]:
        """Bound names that no read keeps alive once every site is rewritten."""

        rewritten = {id(site.call.func.value) for site in self.sites}
        read = {
            node.id
            for node in ast.walk(self.parsed_module.module)
            if isinstance(node, ast.Name) and id(node) not in rewritten
        }
        return tuple(sorted((self.bound_names - read) & self.import_statements.keys()))

    @cached_property
    def import_statements(self) -> dict[str, ast.ImportFrom]:
        """The from-import statement that binds each retired name."""

        return {
            alias.asname or alias.name: statement
            for statement in ast.walk(self.parsed_module.module)
            if isinstance(statement, ast.ImportFrom)
            for alias in statement.names
            if (alias.asname or alias.name) in self.bound_names
        }

    @cached_property
    def unused_import_statements(self) -> tuple[ast.ImportFrom, ...]:
        """Import statements binding the projector where no site needs them."""

        used = {site.name for site in self.sites}
        return tuple(
            dict.fromkeys(
                statement
                for name, statement in self.import_statements.items()
                if name not in used
            )
        )


@dataclass(frozen=True)
class StaticProjectionAuthority:
    """Every static projection call of one module, as source replacements."""

    parsed_module: ParsedModule
    geometry: SourceTextGeometry
    line_length: int
    projection: StaticProjection
    family: ForeignFamilyReference

    @cached_property
    def found(self) -> StaticProjectionSites:
        return StaticProjectionSites(self.parsed_module, self.projection)

    @cached_property
    def fit(self) -> RewrittenLineFit:
        return RewrittenLineFit(self.geometry, self.line_length)

    @cached_property
    def outermost(self) -> tuple[StaticProjectionSite, ...]:
        spans = [self.geometry.required_node_offsets(site.call) for site in self.found.sites]
        return tuple(
            site
            for site, (start, end) in zip(self.found.sites, spans, strict=True)
            if not any(
                other_start <= start and end <= other_end and (other_start, other_end) != (start, end)
                for other_start, other_end in spans
            )
        )

    def rewritten(self, site: StaticProjectionSite) -> str:
        """The site as a family read, with nested sites in its argument rewritten too."""

        argument_start, argument_end = self.geometry.required_node_offsets(site.argument)
        pieces: list[str] = []
        cursor = argument_start
        nested = (
            inner
            for inner in self.found.sites
            if inner is not site
            and argument_start <= self.geometry.required_node_offsets(inner.call)[0]
            and self.geometry.required_node_offsets(inner.call)[1] <= argument_end
        )
        taken_end = argument_start
        for inner in nested:
            start, end = self.geometry.required_node_offsets(inner.call)
            if start < taken_end:
                continue
            pieces += [self.geometry.source[cursor:start], self.rewritten(inner)]
            cursor = taken_end = end
        pieces.append(self.geometry.source[cursor:argument_end])
        return f"{self.family.class_name}.decoded({''.join(pieces)}).{site.question}"

    def replacement(self, site: StaticProjectionSite) -> SourceTextSpanReplacement:
        start, end = self.geometry.required_node_offsets(site.call)
        argument_start, argument_end = self.geometry.required_node_offsets(site.argument)
        for gap in (SourceTextSpan(start, argument_start), SourceTextSpan(argument_end, end)):
            if self.geometry.span_contains_comment(gap):
                raise ValueError("Static projection rewrite would discard a comment")
        candidate = self.rewritten(site)
        if self.geometry.span_contains_comment(SourceTextSpan(start, end)):
            if not self.fit.fits(*self.line_context(start, end, candidate)):
                raise ValueError("Static projection rewrite would exceed the line length")
            text = candidate
        else:
            text = self.fit.fitted(candidate, start, end)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start, end_offset=end, replacement_source=text
        )

    def line_context(self, start: int, end: int, candidate: str) -> tuple[str, str, str]:
        line_start = self.geometry.line_offsets[self.geometry.line_number_for_offset(start) - 1]
        return (
            self.geometry.source[line_start:start],
            candidate,
            self.geometry.source[end:].partition("\n")[0],
        )

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        return tuple(self.replacement(site) for site in self.outermost)

    @cached_property
    def retired_names(self) -> tuple[str, ...]:
        """Projector bindings nothing reads, here or in a module importing them from here."""

        return tuple(
            name
            for name in self.found.retired_names
            if name not in self.imported_from_here
        )

    @cached_property
    def imported_from_here(self) -> frozenset[str]:
        """Retired names another module of the source set still imports from this one."""

        qualified = {
            f"{self.parsed_module.module_name}.{name}": name
            for name in self.found.retired_names
        }
        named = {
            qualified[binding.qualified_name]
            for other in self.other_modules
            for binding in named_bindings(other).values()
            if binding.qualified_name in qualified
        }
        starred = {
            node.id
            for other in self.other_modules
            if self.parsed_module.module_name in star_imported_modules(other)
            for node in ast.walk(other.module)
            if isinstance(node, ast.Name) and node.id in self.found.retired_names
        }
        return frozenset(named | starred)

    other_modules: tuple[ParsedModule, ...] = ()


class StaticProjectionCallModuleFamily(ShapeFamily[ForeignFamilySiteModule]):
    """Uses of a static projection class whose questions an adapter family answers."""

    item_type = ForeignFamilySiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ForeignFamilySiteModule]:
        index = module_syntax_index(parsed_module.module)
        node_indices = {
            statement: node_index
            for node_index, statement in index.indexed_nodes_of_type(ast.Call)
        } | {
            statement: node_index
            for node_index, statement in index.indexed_nodes_of_type(ast.ImportFrom)
        }
        shapes = []
        for projection in cls.loaded_projections(parsed_module):
            found = StaticProjectionSites(parsed_module, projection)
            for node in (
                *(site.call for site in found.sites),
                *found.unused_import_statements,
            ):
                shapes.append(
                    StaticProjectionSiteShape(
                        file_path=parsed_module.file_path,
                        lineno=node.lineno,
                        symbol=index.lexical_symbol(node_indices[node]),
                        family=projection.family.import_path(),
                        projector=found.projector_name,
                    )
                )
        return [
            ForeignFamilySiteModule(
                module_name=parsed_module.module_name,
                sites=tuple(sorted(shapes, key=lambda shape: shape.lineno)),
            )
        ]

    @staticmethod
    def loaded_projections(parsed_module: ParsedModule) -> tuple[StaticProjection, ...]:
        """The projections of the classes this module's own bindings reach."""

        bindings = named_bindings(parsed_module)
        projections = {
            projection.projector: projection
            for projection in map(
                StaticProjection.declared_for,
                sorted({binding.qualified_name for binding in bindings.values()}),
            )
            if projection is not None
        }
        return tuple(projections.values())
