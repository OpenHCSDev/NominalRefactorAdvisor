"""Declaration-selected, source-preserving edits of authored calls."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from .codemod_selector_models import SelectionCountExpectation
from .codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .product_flow import CompactCallArguments
from .product_flow_authority import CompactResolvedFunctionCall
from .source_geometry import SourceByteSpan
from .source_index import AstTargetDigest

if TYPE_CHECKING:
    from .codemod_runtime import CodemodSourceSnapshot


@dataclass(frozen=True)
class DeclaredCallRewriteABC(ABC):
    """Share call identity recovery and source protection across authored edits."""

    snapshot: CodemodSourceSnapshot
    caller: AstTargetDigest
    callee: AstTargetDigest
    replacement_source: str
    selection_count: SelectionCountExpectation

    @cached_property
    def geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(
            self.snapshot.sources_by_file_path[self.caller.file_path]
        )

    @cached_property
    def selected_calls(self) -> tuple[CompactResolvedFunctionCall, ...]:
        source_index = self.snapshot.source_index
        caller_symbol = source_index.symbol_for_target(self.caller)
        callee_symbol = source_index.symbol_for_target(self.callee)
        repository = self.snapshot.product_flow_repository
        context = repository.flow_contexts_by_owner_symbol.get(caller_symbol)
        if context is None:
            raise ValueError(
                f"Call scope has no unique flow authority: {caller_symbol!r}"
            )
        calls = tuple(
            repository.resolve_function_call(context, call)
            for call in context.flow.calls
        )
        selected = tuple(
            resolved
            for resolution in calls
            if (resolved := resolution.resolved_call) is not None
            and resolved.callee.identity.symbol == callee_symbol
        )
        if any(
            resolution.resolved_call is None
            and callee_symbol in resolution.target_resolution.possible_symbols
            for resolution in calls
        ):
            raise ValueError(f"Call authority is unresolved for {callee_symbol!r}")
        if not selected:
            raise ValueError(
                f"No resolved calls to {callee_symbol!r} in {caller_symbol!r}"
            )
        self.selection_count.require_actual_count(len(selected))
        return selected

    @abstractmethod
    def replacement_span(self, node: ast.Call) -> SourceTextSpan:
        """Identify the call region owned by this edit."""
        raise NotImplementedError

    @abstractmethod
    def source_for_call(self, call: CompactResolvedFunctionCall) -> str:
        """Validate and render the authored payload for this resolved call."""
        raise NotImplementedError

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        nodes = {
            SourceByteSpan.require_node(node): node
            for node in ast.walk(
                self.snapshot.ast_target_nodes_by_id[self.caller.target_id]
            )
            if isinstance(node, ast.Call)
        }
        replacements = []
        for call in self.selected_calls:
            replacement = self.source_for_call(call)
            span = self.replacement_span(nodes[call.call.source_span])
            if self.geometry.span_contains_comment(span):
                raise ValueError("Call replacement would remove a comment")
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    replacement_source=replacement,
                )
            )
        return tuple(replacements)


@dataclass(frozen=True)
class DeclaredCallArgumentsRewrite(DeclaredCallRewriteABC):
    """Retain the callee and prove authored arguments bind to its declaration."""

    @cached_property
    def arguments(self) -> CompactCallArguments:
        expression = ast.parse(
            f"_nra_call_({self.replacement_source})", mode="eval"
        ).body
        if (
            not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != "_nra_call_"
        ):
            raise ValueError("Replacement must be one call argument list")
        return CompactCallArguments.from_call(expression)

    def replacement_span(self, node: ast.Call) -> SourceTextSpan:
        return self.geometry.call_argument_span(node)

    def source_for_call(self, call: CompactResolvedFunctionCall) -> str:
        binding = call.target_resolution.bind_arguments(self.arguments)
        if not binding.is_exact:
            raise ValueError(
                f"Replacement arguments do not bind to {call.callee.identity.symbol!r}: {binding.violation}"
            )
        return self.replacement_source


@dataclass(frozen=True)
class DeclaredCallExpressionRewrite(DeclaredCallRewriteABC):
    """Replace a proved call site with one authored expression, preserving precedence."""

    @cached_property
    def expression_source(self) -> str:
        source = self.replacement_source.strip()
        ast.parse(source, mode="eval")
        # The newline protects the closing parenthesis from an authored comment.
        rendered = f"({source}\n)" if "\n" in source or "#" in source else f"({source})"
        ast.parse(rendered, mode="eval")
        return rendered

    def replacement_span(self, node: ast.Call) -> SourceTextSpan:
        return SourceTextSpan(*self.geometry.required_node_offsets(node))

    def source_for_call(self, call: CompactResolvedFunctionCall) -> str:
        return self.expression_source
