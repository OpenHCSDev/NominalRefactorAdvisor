"""The fields a repository's stored records keep, and the classes each can hold.

A row outlives the code that wrote it. A field annotated with a family keeps
the name of the member it held (`type[Family]`, or an instance tagged by its
class), and an enum field keeps the member's value. So deleting a class or an
enum member that a stored field can hold leaves rows naming a meaning the code
no longer declares: the next schema rebuild or decode fails on them. Which
classes are stored is the repository's declaration (`durable-bases`,
`DeclaredRoot.durable_bases_for`): a record is a class below one of them, or a
class a record's stored field can hold, since its values are written in that
record's rows.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from nominal_refactor_advisor.class_index import CompactClassReferenceResolver
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import ResolvedClassTarget
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


@dataclass(frozen=True)
class StoredRecords:
    """The stored records of one repository, as one source state declares them."""

    snapshot: CodemodSourceSnapshot
    durable_bases: frozenset[str]

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, file_path: str) -> StoredRecords:
        return cls(snapshot, DeclaredRoot.durable_bases_for(Path(file_path)))

    @cached_property
    def resolver(self) -> CompactClassReferenceResolver:
        return CompactClassReferenceResolver.from_index(
            self.snapshot.compact_class_projections, self.snapshot.compact_class_index
        )

    @cached_property
    def records(self) -> tuple[str, ...]:
        """Every scanned class whose values a stored row keeps: each class below a
        declared durable base, and each class a stored class's field can hold, its
        own or inherited, through every field that class holds in turn. A
        `type[...]` field keeps a member's name, not its fields (`held_values`)."""

        index = self.snapshot.compact_class_index
        pending = [
            record
            for base in self.durable_bases
            if base in index.classes_by_symbol
            for record in index.descendant_symbols(base)
        ]
        stored: set[str] = set()
        while pending:
            symbol = pending.pop()
            if symbol in stored or symbol not in index.classes_by_symbol:
                continue
            stored.add(symbol)
            pending.extend(index.descendant_symbols(symbol))
            for declaring in (symbol, *index.ancestor_symbols(symbol)):
                if declaring in index.classes_by_symbol:
                    for _field, annotation in self.annotated_fields(declaring):
                        pending.extend(
                            self.named_classes(declaring, self.held_values(annotation))
                        )
        return tuple(sorted(stored))

    def fields_holding(self, symbols: frozenset[str]) -> tuple[str, ...]:
        """Each `Record.field` whose annotation names one of `symbols`: its rows
        can hold that class, or one of its members."""

        return tuple(
            f"{record}.{field}"
            for record in self.records
            for field, annotation in self.annotated_fields(record)
            if not symbols.isdisjoint(self.named_classes(record, annotation))
        )

    def annotated_fields(self, record: str) -> Iterator[tuple[str, ast.expr]]:
        indexed = self.snapshot.compact_class_index.classes_by_symbol[record]
        node = ResolvedClassTarget.from_rewrite_target(
            self.snapshot,
            SourceRewriteTarget(file_path=indexed.file_path, qualname=indexed.qualname),
        ).node
        for statement in node.body:
            match statement:
                case ast.AnnAssign(target=ast.Name(id=field), annotation=annotation) if (
                    not self.is_class_variable(annotation)
                ):
                    yield field, annotation

    @classmethod
    def held_values(cls, annotation: ast.expr) -> ast.expr:
        """The part of an annotation whose instances a row keeps: a `type[...]`
        field keeps only the member's name, so its classes' fields are not stored."""

        class WithoutClassObjects(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                if StoredRecords.chain(node.value) in (("type",), ("Type",), ("typing", "Type")):
                    return ast.Constant(value=None)
                return self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> ast.AST:
                if not isinstance(node.value, str):
                    return node
                try:
                    parsed = ast.parse(node.value, mode="eval").body
                except SyntaxError:
                    return node
                return self.visit(parsed)

        return WithoutClassObjects().visit(copy.deepcopy(annotation))

    @classmethod
    def is_class_variable(cls, annotation: ast.expr) -> bool:
        """A `ClassVar` annotation declares no stored field."""

        match annotation:
            case ast.Subscript(value=value):
                return cls.is_class_variable(value)
            case ast.Constant(value=str() as text):
                try:
                    return cls.is_class_variable(ast.parse(text, mode="eval").body)
                except SyntaxError:
                    return False
        return cls.chain(annotation) in (("ClassVar",), ("typing", "ClassVar"))

    def named_classes(self, record: str, annotation: ast.expr) -> frozenset[str]:
        """The scanned classes an annotation spells, string annotations included."""

        module_name = self.snapshot.compact_class_index.classes_by_symbol[record].module_name
        return frozenset(
            symbol
            for parts in self.reference_chains(annotation)
            if (
                symbol := self.resolver.symbol_for(
                    module_name=module_name, reference_parts=parts
                )
            )
            is not None
        )

    @classmethod
    def reference_chains(cls, annotation: ast.expr) -> Iterator[tuple[str, ...]]:
        for node in ast.walk(annotation):
            match node:
                case ast.Constant(value=str() as text):
                    try:
                        parsed = ast.parse(text, mode="eval").body
                    except SyntaxError:
                        continue
                    yield from cls.reference_chains(parsed)
                case ast.Name() | ast.Attribute():
                    if (chain := cls.chain(node)) is not None:
                        yield chain

    @staticmethod
    def chain(node: ast.expr) -> tuple[str, ...] | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        return (node.id, *reversed(parts))

    def require_unstored(self, symbols: frozenset[str], deleted: str) -> None:
        """Refuse a deletion whose meaning a stored field can still hold."""

        holding = self.fields_holding(symbols)
        if holding:
            raise NeedsInput(
                f"deleting {deleted} leaves stored rows naming it: {', '.join(holding)} "
                "can hold it (durable-bases), so this is a data migration, not a refactor",
                NeededInput.OPERATION,
            )
