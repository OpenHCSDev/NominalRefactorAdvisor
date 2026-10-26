"""Source edits that route foreign-class filters through one declared family."""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Self

from .ast_tools import BuiltinCallName, ParsedModule, module_syntax_index
from .cases_family import handback
from .codemod_declaration_source import PythonExpressionSourceFormatter
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_source_edits import (
    SourceLineSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .foreign_family import ForeignFamily, FunctionDefinition
from .observation_families import ShapeFamily, TypeArgumentResolver
from .observation_shapes import ForeignFamilyFilterModule, ForeignFamilyFilterShape

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class ForeignFamilyReference:
    """A declared family named by its qualified import path."""

    import_path: str

    @cached_property
    def module_name(self) -> str:
        return self.import_path.rpartition(".")[0]

    @cached_property
    def class_name(self) -> str:
        return self.import_path.rpartition(".")[2]

    @cached_property
    def family(self) -> type[ForeignFamily]:
        return getattr(import_module(self.module_name), self.class_name)

    @property
    def member_names(self) -> frozenset[str]:
        return self.family.member_names()

    def import_source_for(
        self, snapshot: CodemodSourceSnapshot, parsed_module: ParsedModule
    ) -> str:
        """Import in the importing module's own style: relative when it is."""

        family_file = snapshot.module_import_graph.unique_source_file_for_module_name(
            self.module_name
        )
        uses_relative_imports = any(
            statement.level
            for _, statement in module_syntax_index(
                parsed_module.module
            ).indexed_nodes_of_type(ast.ImportFrom)
        )
        reference = (
            snapshot.module_import_graph.import_module_reference(
                importing_file_path=parsed_module.file_path,
                imported_file_path=family_file.file_path,
                imported_name=self.class_name,
            )
            if family_file is not None and uses_relative_imports
            else None
        )
        return f"from {reference or self.module_name} import {self.class_name}"


@dataclass(frozen=True)
class RewrittenLineFit:
    """Keeps a rewritten expression inside the repository's declared line length."""

    geometry: SourceTextGeometry
    line_length: int

    def fitted(self, candidate: str, start: int, end: int) -> str:
        """The candidate, black-formatted when it would overflow its line."""

        line_start = self.geometry.line_offsets[
            self.geometry.line_number_for_offset(start) - 1
        ]
        prefix = self.geometry.source[line_start:start]
        tail = self.geometry.source[end:].partition("\n")[0]
        if self.fits(prefix, candidate, tail):
            return candidate
        indentation = prefix[: len(prefix) - len(prefix.lstrip())]
        # The formatter budgets only the indentation; the rest of the prefix
        # (`for x in `) is charged to the line length instead.
        formatted = PythonExpressionSourceFormatter(
            self.line_length - (len(prefix) - len(indentation))
        ).replacement_source(
            ast.parse(candidate, mode="eval").body, line_prefix=indentation
        )
        return formatted if self.fits(prefix, formatted, tail) else candidate

    def fits(self, prefix: str, text: str, tail: str) -> bool:
        return all(
            len(line) <= self.line_length
            for line in f"{prefix}{text}{tail}".split("\n")
        )


@dataclass(frozen=True)
class FilterSiteABC(ABC):
    """One loop or comprehension that keeps only instances of tested classes."""

    variable: str
    condition: ast.Call
    iterable: ast.expr

    @classmethod
    @abstractmethod
    def in_scope(cls, root: ast.AST) -> tuple[Self, ...]:
        """Every site of this shape under `root`, nested scopes included."""
        raise NotImplementedError

    @abstractmethod
    def replacements(
        self, fit: RewrittenLineFit, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Wrap the iterable in `family.within` and delete the type test."""
        raise NotImplementedError

    @staticmethod
    def tested_variable(call: ast.expr, variable: ast.expr) -> bool:
        match (BuiltinCallName.isinstance_arguments(call), variable):
            case ((ast.Name(id=subject), _), ast.Name(id=name)):
                return subject == name
        return False

    def within_replacement(
        self, fit: RewrittenLineFit, family_name: str, end_offset: int
    ) -> SourceTextSpanReplacement:
        """Replace the iterable through `end_offset` by its family-filtered form."""

        start_offset, iterable_end = fit.geometry.required_node_offsets(self.iterable)
        candidate = (
            f"{family_name}.within({fit.geometry.source[start_offset:iterable_end]})"
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=fit.fitted(candidate, start_offset, end_offset),
        )


@dataclass(frozen=True)
class ComprehensionFilterSite(FilterSiteABC):
    """`for V in ITER if isinstance(V, T)` as a comprehension's first condition."""

    @classmethod
    def in_scope(cls, root: ast.AST) -> tuple[Self, ...]:
        return tuple(
            cls(clause.target.id, clause.ifs[0], clause.iter)
            for clause in ast.walk(root)
            if isinstance(clause, ast.comprehension)
            and clause.ifs
            and isinstance(clause.target, ast.Name)
            and cls.tested_variable(clause.ifs[0], clause.target)
        )

    def replacements(
        self, fit: RewrittenLineFit, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = fit.geometry
        condition_start, condition_end = geometry.required_node_offsets(self.condition)
        _, iterable_end = geometry.required_node_offsets(self.iterable)
        if (
            re.fullmatch(r"\s+if\s+", geometry.source[iterable_end:condition_start])
            is None
        ):
            raise ValueError("Comprehension filter layout is not a bare `if` clause")
        if geometry.span_contains_comment(SourceTextSpan(iterable_end, condition_end)):
            raise ValueError("Comprehension filter rewrite would discard a comment")
        return (self.within_replacement(fit, family_name, condition_end),)


@dataclass(frozen=True)
class ContinueFilterSite(FilterSiteABC):
    """`for V in ITER:` whose first statement is `if not isinstance(V, T): continue`."""

    statement: ast.If

    @classmethod
    def in_scope(cls, root: ast.AST) -> tuple[Self, ...]:
        sites = []
        for loop in ast.walk(root):
            match loop:
                case ast.For(
                    target=ast.Name() as target,
                    iter=iterable,
                    body=[
                        ast.If(
                            test=ast.UnaryOp(op=ast.Not(), operand=ast.Call() as call),
                            body=[ast.Continue()],
                            orelse=[],
                        ) as statement,
                        _,
                        *_,
                    ],
                ) if cls.tested_variable(call, target):
                    sites.append(cls(target.id, call, iterable, statement))
        return tuple(sites)

    def replacements(
        self, fit: RewrittenLineFit, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        geometry = fit.geometry
        statement_span = SourceTextSpan(
            *geometry.line_span_offsets(
                self.statement.lineno,
                self.statement.end_lineno or self.statement.lineno,
            )
        )
        if geometry.span_contains_comment(statement_span):
            raise ValueError("Loop filter rewrite would discard a comment")
        deleted = geometry.statement_deletion_span(
            SourceLineSpan(
                self.statement.lineno,
                self.statement.end_lineno or self.statement.lineno,
            )
        )
        start_offset, end_offset = geometry.line_span_offsets(
            deleted.start_line, deleted.end_line
        )
        _, iterable_end = geometry.required_node_offsets(self.iterable)
        return (
            self.within_replacement(fit, family_name, iterable_end),
            SourceTextSpanReplacement.from_offsets(
                start_offset=start_offset, end_offset=end_offset, replacement_source=""
            ),
        )


@dataclass(frozen=True)
class ForeignFamilyFilterAuthority:
    """Every filter site of one function whose tested classes are a family's."""

    parsed_module: ParsedModule
    geometry: SourceTextGeometry
    line_length: int
    declaration: ast.stmt
    reference: ForeignFamilyReference
    accept_widening: bool

    @cached_property
    def function(self) -> FunctionNode:
        """The function of the indexed module tree that the target node denotes."""

        position = (self.declaration.lineno, self.declaration.col_offset)
        matches = tuple(
            candidate
            for candidate in FunctionDefinition.within(
                ast.walk(self.parsed_module.module)
            )
            if (candidate.lineno, candidate.col_offset) == position
        )
        if len(matches) != 1:
            raise ValueError("Filter routing requires a function declaration target")
        return matches[0]

    @cached_property
    def fit(self) -> RewrittenLineFit:
        return RewrittenLineFit(self.geometry, self.line_length)

    @cached_property
    def resolver(self) -> TypeArgumentResolver:
        return TypeArgumentResolver(self.parsed_module)

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        """Edits for every site testing exactly the family, or a part if accepted.

        A part-of-family site would widen its filter, so it stays untouched
        unless widening is accepted; when only such sites exist the first one
        is handed back, naming the classes the rewrite would add.
        """

        members = self.reference.member_names
        replacements: list[SourceTextSpanReplacement] = []
        refusals: list[ValueError] = []
        for shape in FilterSiteABC.__subclasses__():
            for site in shape.in_scope(self.function):
                tested = self.resolver.qualified_names_of_test(site.condition)
                if tested is None or not set(tested) <= members:
                    continue
                missing = members - set(tested)
                if missing and not self.accept_widening:
                    refusals.append(
                        handback(
                            self.parsed_module,
                            site.condition,
                            f"rewriting to {self.reference.class_name} would widen "
                            f"the filter to include {', '.join(sorted(missing))}; "
                            "set accept_widening to allow it",
                        )
                    )
                    continue
                replacements.extend(
                    site.replacements(self.fit, self.reference.class_name)
                )
        if not replacements:
            raise (
                refusals[0]
                if refusals
                else ValueError(
                    f"No filter site in {self.function.name!r} tests classes of "
                    f"{self.reference.import_path}"
                )
            )
        return tuple(sorted(replacements, key=lambda item: item.start_offset))


class ForeignFamilyFilterModuleFamily(ShapeFamily[ForeignFamilyFilterModule]):
    """Filters testing exactly the members of a declared concrete family."""

    item_type = ForeignFamilyFilterModule

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ForeignFamilyFilterModule]:
        index = module_syntax_index(parsed_module.module)
        node_indices = {
            call: node_index
            for node_index, call in index.indexed_nodes_of_type(ast.Call)
        }
        resolver = TypeArgumentResolver(parsed_module)
        families_by_members = {
            family.member_names(): family
            for family in ForeignFamily.concrete_families()
        }
        shapes = []
        for site_type in FilterSiteABC.__subclasses__():
            for site in site_type.in_scope(parsed_module.module):
                tested = resolver.qualified_names_of_test(site.condition)
                family = families_by_members.get(frozenset(tested or ()))
                if family is not None:
                    shapes.append(
                        ForeignFamilyFilterShape(
                            file_path=parsed_module.file_path,
                            lineno=site.condition.lineno,
                            symbol=index.lexical_symbol(node_indices[site.condition]),
                            family=family.import_path(),
                        )
                    )
        return [
            ForeignFamilyFilterModule(
                module_name=parsed_module.module_name,
                sites=tuple(sorted(shapes, key=lambda shape: shape.lineno)),
            )
        ]
