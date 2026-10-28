"""Selected annotated-field metadata edits, retaining original source and bindings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, cast

from .ast_projection import CallKeywords
from .class_index import ModuleNominalBindingAuthority, nominal_reference_root_name
from .codemod_declaration_source import DirectClassDeclarationAuthority
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import ResolvedClassTarget
from .codemod_selector_models import SelectionCountExpectation
from .codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .source_index import AstTargetDigest
from .value_expression import LiteralExpressionEffects


@dataclass(frozen=True)
class FieldMetadataKeywordRemoval(SourceTextGeometry):
    """Remove authored literal keywords from one declaration-selected metadata call.

    Nominal binding and inert discarded expressions are checked. Whether the
    removed metadata constraint is redundant belongs to the authored migration.
    """

    snapshot: CodemodSourceSnapshot
    owner: ResolvedClassTarget
    callee: AstTargetDigest
    field_name: str
    keyword_names: tuple[str, ...]
    selection_count: SelectionCountExpectation

    @cached_property
    def field(self) -> ast.AnnAssign:
        return DirectClassDeclarationAuthority(self, self.owner.node).annotated_field(
            self.field_name
        )

    @cached_property
    def bindings(self) -> ModuleNominalBindingAuthority:
        return ModuleNominalBindingAuthority(
            self.snapshot.parsed_module_for_source_path(self.owner.file_path)
        )

    def stable_name(self, reference: ast.expr) -> str | None:
        root = nominal_reference_root_name(reference)
        if root is None or root in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.owner.node.body
        ):
            return None
        before = self.bindings.qualified_name_at(reference, line=self.owner.node.lineno)
        after = self.bindings.qualified_name_at(reference, line=None)
        return before if before == after else None

    def candidates(self) -> tuple[ast.Call, ...]:
        """Direct field declaration calls, including generated-column initializers.

        Nested executable scopes are deliberately outside this binding proof.
        """
        module = self.snapshot.parsed_module_for_source_path(self.owner.file_path)
        syntax = module.syntax_index
        calls = []
        for index, call in syntax.indexed_nodes_of_type(ast.Call):
            ancestors = syntax.ancestor_nodes(index)
            if self.field not in ancestors:
                continue
            if any(
                isinstance(
                    ancestor,
                    (
                        ast.Lambda,
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                        ast.GeneratorExp,
                    ),
                )
                for ancestor in ancestors
            ):
                continue
            calls.append(call)
        return tuple(calls)

    @cached_property
    def call(self) -> ast.Call:
        symbol = self.snapshot.source_index.symbol_for_target(self.callee)
        calls = tuple(
            call for call in self.candidates() if self.stable_name(call.func) == symbol
        )
        if len(calls) != 1:
            raise ValueError(
                f"Expected one metadata call to {symbol!r}; found {len(calls)}"
            )
        self.selection_count.require_actual_count(len(calls))
        (call,) = calls
        if CallKeywords.of(call.keywords).splats:
            raise ValueError(
                "Expanded keyword arguments leave metadata removal ambiguous"
            )
        return call

    @cached_property
    def discarded(self) -> tuple[ast.keyword, ...]:
        selected = tuple(
            keyword
            for keyword in self.call.keywords
            if keyword.arg in self.keyword_names
        )
        if len(selected) != len(self.keyword_names) or {k.arg for k in selected} != set(
            self.keyword_names
        ):
            raise ValueError("Requested metadata keywords must occur exactly once")
        for keyword in selected:
            LiteralExpressionEffects(keyword.value).require_closed()
        return selected

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        discarded = self.discarded
        if not self.call.args and len(discarded) == len(self.call.keywords):
            return self.empty_call_replacements()
        return self.keyword_replacements()

    def keyword_replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        return self.remove_separated_elements(
            (*self.call.args, *self.call.keywords),
            self.discarded,
            self.call_argument_span(self.call),
        )

    def empty_call_replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        """An empty constructor remains meaningful unless elision was explicitly selected."""
        return self.keyword_replacements()


class PrunedFieldMetadataKeywordRemoval(FieldMetadataKeywordRemoval):
    """The authored contract also permits eliding this constructor when emptied."""

    @cached_property
    def annotation(self) -> ast.Subscript:
        annotation = self.field.annotation
        name = f"{Annotated.__module__}.{Annotated.__qualname__}"
        if (
            not isinstance(annotation, ast.Subscript)
            or self.stable_name(annotation.value) != name
        ):
            raise ValueError(
                "Selected field must directly use the original typing.Annotated"
            )
        if (
            not isinstance(annotation.slice, ast.Tuple)
            or len(annotation.slice.elts) < 2
        ):
            raise ValueError("Selected annotation has no metadata entries")
        return annotation

    @property
    def entries(self) -> tuple[ast.expr, ...]:
        return tuple(cast(ast.Tuple, self.annotation.slice).elts)

    def candidates(self) -> tuple[ast.Call, ...]:
        return tuple(entry for entry in self.entries[1:] if isinstance(entry, ast.Call))

    def empty_call_replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        if len(self.entries) > 2:
            return self.remove_separated_elements(
                self.entries,
                (self.call,),
                SourceTextSpan(*self.required_node_offsets(self.annotation.slice)),
            )
        (value,) = self.entries[:1]
        start, end = self.required_node_offsets(self.annotation)
        value_start, value_end = self.required_node_offsets(value)
        if any(
            self.span_contains_comment(span)
            for span in (
                SourceTextSpan(start, value_start),
                SourceTextSpan(value_end, end),
            )
        ):
            raise ValueError("Annotation elision would remove a comment")
        return (
            SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=self.source[value_start:value_end],
            ),
        )
