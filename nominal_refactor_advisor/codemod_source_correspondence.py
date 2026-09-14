"""Document-bound correspondence of actual source reads, not execution equivalence."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from .ast_tools import (
    ModuleSyntaxIndex,
    module_syntax_index,
)

from .codemod_runtime import CodemodDocumentSimulationCarrier
from .codemod_selector_models import SourceRewriteTarget

from .codemod_source_edits import (
    SourceNodeDecoratorPolicy,
    SourceNodeSpan,
    SourceRetainedSpanIndex,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpan,
)
from .collection_algebra import UniqueIdentityIndexAuthority
from .product_flow import (
    SourceFlowOperation,
    SourceProductFlowProjection,
)
from .source_geometry import SourceByteSpan


@dataclass(frozen=True)
class SourceDocumentCorrespondence(CodemodDocumentSimulationCarrier):
    """Own original revisions and retained origins for one complete document.

    Both projections own their own nodes, contexts and activation cuts. This
    relation transports source origin only: enclosing scopes, lookup results,
    effect ordering and source-created object identity need separate proofs.
    Changed and generated text has no retained-read correspondence.
    """

    before: SourceProductFlowProjection
    after: SourceProductFlowProjection

    @cached_property
    def retained_span_index(self) -> SourceRetainedSpanIndex:
        """Retain actual module windows in the renderer's declared order once."""
        batch = self.document_simulation.edit_batch
        compiler = batch.compiler
        path = self.before.module.file_path
        target = SourceRewriteTarget(file_path=path)
        digest = compiler.source_index.target_by_id[
            target.required_target_id(compiler.source_index)
        ]
        editor = SourceTargetEditor(compiler.sources_by_file_path, digest)
        windows = editor.ordered_windows(
            window for window in batch.windows if window.physical_edit.file_path == path
        )
        return SourceRetainedSpanIndex(self.before_geometry, windows)

    def __post_init__(self) -> None:
        original = self.before.module
        rewritten = self.after.module
        if original.module_path_identity != rewritten.module_path_identity:
            raise ValueError("Read correspondence requires the same module identity")
        document = self.document_simulation
        if not document.simulation.parse_valid:
            raise ValueError("Read correspondence requires a parsed rendering")
        revision = document.simulation.base_revision_by_file_path[original.file_path]
        if not revision.matches_source(original.source):
            raise ValueError(
                "Read correspondence requires the original document revision"
            )
        expected = document.required_after_snapshot.sources_by_file_path[
            rewritten.file_path
        ]
        if rewritten.source != expected:
            raise ValueError("Read correspondence requires the exact rewritten source")

    @cached_property
    def before_geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(self.before.module.source)

    @cached_property
    def after_geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(self.after.module.source)


class SourceReadCorrespondence(SourceDocumentCorrespondence):
    """Join retained reads and calls without transferring execution admission."""

    def corresponding_calls(
        self,
        operations: Iterable[SourceFlowOperation],
    ) -> tuple[SourceFlowOperation, ...]:
        """Retain whole original calls without transferring execution admission."""
        operations = tuple(operations)
        for operation in operations:
            if (
                not isinstance(operation.node, ast.Call)
                or self.before.call_operation(
                    SourceByteSpan.require_node(operation.node)
                )
                is not operation
            ):
                raise ValueError(
                    "Call correspondence requires an original canonical call"
                )
        spans = self.retained_span_index.project(
            SourceTextSpan(*self.before_geometry.required_node_offsets(operation.node))
            for operation in operations
        )
        corresponding = []
        for span in spans:
            operation = self.after_calls_by_span.get(span)
            if operation is None:
                raise ValueError("Retained text has no unique corresponding call")
            canonical = self.after.call_operation(
                SourceByteSpan.require_node(operation.node)
            )
            if canonical is not operation:
                raise ValueError("Retained call has no original operation association")
            corresponding.append(operation)
        return tuple(corresponding)

    @cached_property
    def after_calls_by_span(self) -> dict[SourceTextSpan, SourceFlowOperation]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.after.call_operations_by_span.values(),
            lambda operation: SourceTextSpan(
                *self.after_geometry.required_node_offsets(operation.node)
            ),
        )

    @cached_property
    def after_reads_by_span(self) -> dict[SourceTextSpan, ast.AST]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            self.after.reference_reads_by_node,
            lambda node: SourceTextSpan(
                *self.after_geometry.required_node_offsets(node)
            ),
        )

    def corresponding_reads(self, nodes: Iterable[ast.expr]) -> tuple[ast.expr, ...]:
        """Project an ordered batch without substituting coordinate lookalikes."""
        nodes = tuple(nodes)
        for node in nodes:
            if node not in self.before.reference_reads_by_node:
                raise ValueError(
                    "Read correspondence requires an original canonical read"
                )
            read = self.before.reference_reads_by_node[node]
            self.before.source_operation(read.context, read.use)
        spans = self.retained_span_index.project(
            (
                SourceTextSpan(*self.before_geometry.required_node_offsets(node))
                for node in nodes
            ),
        )
        corresponding = []
        for original, span in zip(nodes, spans, strict=True):
            node = self.after_reads_by_span.get(span)
            if node is None or type(node) is not type(original):
                raise ValueError("Retained text has no unique corresponding read")
            read = self.after.reference_reads_by_node[node]
            self.after.source_operation(read.context, read.use)
            corresponding.append(cast(ast.expr, node))
        return tuple(corresponding)


class SourceClassSuiteCorrespondence(SourceDocumentCorrespondence):
    """Match complete retained class suites without claiming creation equivalence.

    New statements are explicit caller-owned obligations, not approved semantics.
    Class headers, enclosing scopes and native execution need separate proofs.
    """

    @cached_property
    def after_syntax(self) -> ModuleSyntaxIndex:
        return module_syntax_index(self.after.module.module)

    @cached_property
    def after_class_statements_by_span(self) -> dict[SourceTextSpan, ast.stmt]:
        return UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
            (
                statement
                for statement, parent in self.after_syntax.parent_by_node.items()
                if isinstance(statement, ast.stmt) and isinstance(parent, ast.ClassDef)
            ),
            lambda statement: self.statement_span(self.after_geometry, statement),
        )

    @staticmethod
    def statement_span(
        geometry: SourceTextGeometry, statement: ast.stmt
    ) -> SourceTextSpan:
        return SourceTextSpan(
            *geometry.node_span_offsets(
                SourceNodeSpan(statement, SourceNodeDecoratorPolicy.INCLUDE)
            )
        )

    def corresponding_class(
        self,
        original: ast.ClassDef,
        *,
        added_statements: tuple[ast.stmt, ...] = (),
    ) -> ast.ClassDef:
        """Derive one actual candidate from original statement origins, not names."""
        if not isinstance(original, ast.ClassDef):
            raise TypeError("Class correspondence requires a class declaration")
        self.require_class_suite(self.before, original)
        spans = self.retained_span_index.project(
            self.statement_span(self.before_geometry, statement)
            for statement in original.body
        )
        matched = []
        for statement, span in zip(original.body, spans, strict=True):
            candidate = self.after_class_statements_by_span.get(span)
            if candidate is None or type(candidate) is not type(statement):
                raise ValueError("Retained class statement has no unique candidate")
            matched.append(candidate)
        if not matched:
            raise ValueError("Class correspondence requires a retained suite")
        candidate = self.after_syntax.parent_by_node[matched[0]]
        if not isinstance(candidate, ast.ClassDef) or any(
            self.after_syntax.parent_by_node[statement] is not candidate
            for statement in matched
        ):
            raise ValueError("Retained statements belong to different class suites")
        self.require_class_suite(self.after, candidate)
        self.require_added_statements(candidate, added_statements)
        if tuple(
            statement
            for statement in candidate.body
            if statement not in added_statements
        ) != tuple(matched):
            raise ValueError(
                "Candidate class suite has unaccounted or reordered statements"
            )
        self.require_decorators(original, candidate)
        return candidate

    @staticmethod
    def require_class_suite(
        source: SourceProductFlowProjection, declaration: ast.ClassDef
    ) -> None:
        """Authenticate direct members against their original syntax parents."""
        source.definition_operation(declaration)
        syntax = module_syntax_index(source.module.module)
        if any(
            syntax.parent_by_node.get(node) is not declaration
            for node in (*declaration.body, *declaration.decorator_list)
        ):
            raise ValueError("Class suite contains foreign or reparented source nodes")

    def require_added_statements(
        self, candidate: ast.ClassDef, statements: tuple[ast.stmt, ...]
    ) -> None:
        """Accept only distinct actual direct members of the selected candidate."""
        if len(set(statements)) != len(statements) or any(
            statement not in candidate.body
            or self.after_syntax.parent_by_node.get(statement) is not candidate
            for statement in statements
        ):
            raise ValueError(
                "Added statements require actual distinct candidate owners"
            )

    def require_decorators(
        self, original: ast.ClassDef, candidate: ast.ClassDef
    ) -> None:
        """Body retention cannot conceal changed class-level decorators."""
        original_spans = tuple(
            SourceTextSpan(*self.before_geometry.required_node_offsets(node))
            for node in original.decorator_list
        )
        candidate_spans = tuple(
            SourceTextSpan(*self.after_geometry.required_node_offsets(node))
            for node in candidate.decorator_list
        )
        if self.retained_span_index.project(original_spans) != candidate_spans:
            raise ValueError("Candidate class decorators have changed source origins")
