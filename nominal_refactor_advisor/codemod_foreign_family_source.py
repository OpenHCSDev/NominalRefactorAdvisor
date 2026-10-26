"""Source edits that route foreign-class filters through one declared family."""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Self

from .ast_tools import BuiltinCallName, ParsedModule
from .cases_family import handback
from .codemod_source_edits import (
    SourceLineSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .foreign_family import ForeignFamily, FunctionDefinition
from .observation_families import TypeArgumentResolver

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

    @cached_property
    def member_names(self) -> frozenset[str]:
        return frozenset(
            f"{member.__module__}.{member.__qualname__}"
            for member in self.family.members
        )

    @property
    def import_source(self) -> str:
        return f"from {self.module_name} import {self.class_name}"


@dataclass(frozen=True)
class FilterSiteABC(ABC):
    """One loop or comprehension that keeps only instances of tested classes."""

    variable: str
    condition: ast.Call
    iterable: ast.expr

    @classmethod
    @abstractmethod
    def in_function(cls, function: FunctionNode) -> tuple[Self, ...]:
        """Every site of this shape inside the function, nested scopes included."""
        raise NotImplementedError

    @abstractmethod
    def replacements(
        self, geometry: SourceTextGeometry, family_name: str
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
        self, geometry: SourceTextGeometry, family_name: str, end_offset: int
    ) -> SourceTextSpanReplacement:
        """Replace the iterable through `end_offset` by its family-filtered form."""

        start_offset, iterable_end = geometry.required_node_offsets(self.iterable)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=(
                f"{family_name}.within({geometry.source[start_offset:iterable_end]})"
            ),
        )


@dataclass(frozen=True)
class ComprehensionFilterSite(FilterSiteABC):
    """`for V in ITER if isinstance(V, T)` as a comprehension's first condition."""

    @classmethod
    def in_function(cls, function: FunctionNode) -> tuple[Self, ...]:
        return tuple(
            cls(clause.target.id, clause.ifs[0], clause.iter)
            for clause in ast.walk(function)
            if isinstance(clause, ast.comprehension)
            and clause.ifs
            and isinstance(clause.target, ast.Name)
            and cls.tested_variable(clause.ifs[0], clause.target)
        )

    def replacements(
        self, geometry: SourceTextGeometry, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        condition_start, condition_end = geometry.required_node_offsets(self.condition)
        _, iterable_end = geometry.required_node_offsets(self.iterable)
        if re.fullmatch(r"\s+if\s+", geometry.source[iterable_end:condition_start]) is None:
            raise ValueError("Comprehension filter layout is not a bare `if` clause")
        if geometry.span_contains_comment(SourceTextSpan(iterable_end, condition_end)):
            raise ValueError("Comprehension filter rewrite would discard a comment")
        return (self.within_replacement(geometry, family_name, condition_end),)


@dataclass(frozen=True)
class ContinueFilterSite(FilterSiteABC):
    """`for V in ITER:` whose first statement is `if not isinstance(V, T): continue`."""

    statement: ast.If

    @classmethod
    def in_function(cls, function: FunctionNode) -> tuple[Self, ...]:
        sites = []
        for loop in ast.walk(function):
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
        self, geometry: SourceTextGeometry, family_name: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        statement_span = SourceTextSpan(
            *geometry.line_span_offsets(
                self.statement.lineno, self.statement.end_lineno or self.statement.lineno
            )
        )
        if geometry.span_contains_comment(statement_span):
            raise ValueError("Loop filter rewrite would discard a comment")
        deleted = geometry.statement_deletion_span(
            SourceLineSpan(
                self.statement.lineno, self.statement.end_lineno or self.statement.lineno
            )
        )
        start_offset, end_offset = geometry.line_span_offsets(
            deleted.start_line, deleted.end_line
        )
        _, iterable_end = geometry.required_node_offsets(self.iterable)
        return (
            self.within_replacement(geometry, family_name, iterable_end),
            SourceTextSpanReplacement.from_offsets(
                start_offset=start_offset, end_offset=end_offset, replacement_source=""
            ),
        )


@dataclass(frozen=True)
class ForeignFamilyFilterAuthority:
    """Every filter site of one function whose tested classes are a family's."""

    parsed_module: ParsedModule
    geometry: SourceTextGeometry
    declaration: ast.AST
    reference: ForeignFamilyReference
    accept_widening: bool

    @cached_property
    def function(self) -> FunctionNode:
        """The function of the indexed module tree that the target node denotes."""

        declared = next(iter(FunctionDefinition.within((self.declaration,))), None)
        matches = tuple(
            candidate
            for candidate in FunctionDefinition.within(
                ast.walk(self.parsed_module.module)
            )
            if declared is not None
            and (candidate.name, candidate.lineno, candidate.col_offset)
            == (declared.name, declared.lineno, declared.col_offset)
        )
        if len(matches) != 1:
            raise ValueError("Filter routing requires a function declaration target")
        return matches[0]

    @cached_property
    def resolver(self) -> TypeArgumentResolver:
        return TypeArgumentResolver(self.parsed_module)

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        """Edits for every site testing the family or, if accepted, a part of it."""

        members = self.reference.member_names
        replacements: list[SourceTextSpanReplacement] = []
        for shape in FilterSiteABC.__subclasses__():
            for site in shape.in_function(self.function):
                tested = self.resolver.qualified_names_of_test(site.condition)
                if tested is None or not set(tested) <= members:
                    continue
                missing = members - set(tested)
                if missing and not self.accept_widening:
                    raise handback(
                        self.parsed_module,
                        site.condition,
                        f"rewriting to {self.reference.class_name} would widen the "
                        f"filter to include {', '.join(sorted(missing))}; "
                        "set accept_widening to allow it",
                    )
                replacements.extend(site.replacements(self.geometry, self.reference.class_name))
        if not replacements:
            raise ValueError(
                f"No filter site in {self.function.name!r} tests classes of "
                f"{self.reference.import_path}"
            )
        return tuple(sorted(replacements, key=lambda item: item.start_offset))
