"""Source edits that move static projection calls onto a declared adapter family."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Self

from .ast_tools import ParsedModule, module_syntax_index
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

        module_name, _, class_name = qualified_name.rpartition(".")
        module = sys.modules.get(module_name)
        projector = None if module is None else vars(module).get(class_name)
        if not isinstance(projector, type):
            return None
        questions = cls.questions_of(projector)
        families = tuple(
            family
            for family in ForeignAdapter.concrete_families()
            if questions and questions <= frozenset(AdapterPropertyBodies(family).property_names)
        )
        return cls(projector, families[0]) if len(families) == 1 else None

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


@dataclass(frozen=True)
class StaticProjectionSite:
    """`Projector.question(argument)`, one positional argument."""

    call: ast.Call
    argument: ast.expr
    question: str
    name: str


@dataclass(frozen=True)
class StaticProjectionSites:
    """Every call in one module that a declared static projection can hand to its family."""

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
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                    names.add(name)
        return frozenset(names)

    @cached_property
    def sites(self) -> tuple[StaticProjectionSite, ...]:
        resolver = TypeArgumentResolver(self.parsed_module)
        resolved: dict[str, bool] = {}
        sites = []
        for node in ast.walk(self.parsed_module.module):
            match node:
                case ast.Call(
                    func=ast.Attribute(value=ast.Name(id=name), attr=question),
                    args=[argument],
                    keywords=[],
                ) if question in self.projection.questions and not isinstance(
                    argument, ast.Starred
                ):
                    if name not in resolved:
                        reference = resolver.snapshot.reference_for((name,))
                        resolved[name] = (
                            name not in self.shadowing_names
                            and reference.root_binding is not None
                            and reference.qualified_name == self.projector_name
                        )
                    if resolved[name]:
                        sites.append(StaticProjectionSite(node, argument, question, name))
        return tuple(sorted(sites, key=lambda site: (site.call.lineno, site.call.col_offset)))

    @cached_property
    def remaining_projector_names(self) -> frozenset[str]:
        """Names of the projector still read once every site is rewritten."""

        rewritten = {id(site.call.func.value) for site in self.sites}
        return frozenset(
            node.id
            for node in ast.walk(self.parsed_module.module)
            if isinstance(node, ast.Name)
            and node.id in {site.name for site in self.sites}
            and id(node) not in rewritten
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
        if not self.outermost:
            raise ValueError(
                f"No call of {self.found.projector_name} has a property on "
                f"{self.family.import_path}"
            )
        return tuple(self.replacement(site) for site in self.outermost)

    @property
    def retired_names(self) -> tuple[str, ...]:
        """Import bindings of the projector that no read keeps alive."""

        remaining = self.found.remaining_projector_names
        return tuple(sorted({site.name for site in self.found.sites} - remaining))


class StaticProjectionCallModuleFamily(ShapeFamily[ForeignFamilySiteModule]):
    """Calls of a static projection class whose questions an adapter family answers."""

    item_type = ForeignFamilySiteModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ForeignFamilySiteModule]:
        index = module_syntax_index(parsed_module.module)
        node_indices = {
            call: node_index for node_index, call in index.indexed_nodes_of_type(ast.Call)
        }
        shapes = []
        for projection in cls.loaded_projections(parsed_module):
            found = StaticProjectionSites(parsed_module, projection)
            for site in found.sites:
                shapes.append(
                    StaticProjectionSiteShape(
                        file_path=parsed_module.file_path,
                        lineno=site.call.lineno,
                        symbol=index.lexical_symbol(node_indices[site.call]),
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
        """Projections for the names this module's own bindings resolve to."""

        snapshot = TypeArgumentResolver(parsed_module).snapshot
        references = (
            snapshot.reference_for((node.value.id,))
            for node in ast.walk(parsed_module.module)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        )
        qualified = {
            reference.qualified_name
            for reference in references
            if reference.root_binding is not None
        }
        return tuple(
            projection
            for projection in map(StaticProjection.declared_for, sorted(qualified))
            if projection is not None
        )
