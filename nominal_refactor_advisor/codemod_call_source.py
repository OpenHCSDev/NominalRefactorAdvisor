"""Declaration-selected, source-preserving edits of authored call arguments."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from .codemod_selector_models import SelectionCountExpectation
from .codemod_source_edits import SourceTextGeometry, SourceTextSpanReplacement
from .product_flow import CompactCallArguments
from .source_geometry import SourceByteSpan
from .source_index import AstTargetDigest

if TYPE_CHECKING:
    from .codemod_runtime import CodemodSourceSnapshot


@dataclass(frozen=True)
class DeclaredCallArgumentsRewrite:
    """Resolve calls to an explicit declaration; argument expressions are authored."""

    snapshot: CodemodSourceSnapshot
    caller: AstTargetDigest
    callee: AstTargetDigest
    arguments_source: str
    selection_count: SelectionCountExpectation

    @cached_property
    def geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(
            self.snapshot.sources_by_file_path[self.caller.file_path]
        )

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        expression = ast.parse(f"_nra_call_({self.arguments_source})", mode="eval").body
        if (
            not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != "_nra_call_"
        ):
            raise ValueError("Replacement must be one call argument list")
        arguments = CompactCallArguments.from_call(expression)
        source_index = self.snapshot.source_index
        caller_symbol = source_index.symbol_for_target(self.caller)
        callee_symbol = source_index.symbol_for_target(self.callee)
        repository = self.snapshot.product_flow_repository
        calls = tuple(
            repository.resolve_function_call(context, call)
            for context in repository.flow_contexts
            if context.owner_symbol == caller_symbol
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
        nodes = {
            SourceByteSpan.require_node(node): node
            for node in ast.walk(
                self.snapshot.module_nodes_by_file_path[self.caller.file_path]
            )
            if isinstance(node, ast.Call)
        }
        replacements = []
        for call in selected:
            binding = arguments.bind_to(call.callee)
            if not binding.is_exact:
                raise ValueError(
                    f"Replacement arguments do not bind to {callee_symbol!r}: {binding.violation}"
                )
            span = self.geometry.call_argument_span(nodes[call.call.source_span])
            if self.geometry.span_contains_comment(span):
                raise ValueError("Call argument replacement would remove a comment")
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    replacement_source=self.arguments_source,
                )
            )
        return tuple(replacements)
