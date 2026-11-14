"""Native source syntax authority for cold compact projection extraction.

The ordinary Python AST remains the semantic fallback.  This module provides one
shared tree-sitter parse per source shard so compact families can migrate their
lossless source projections incrementally without adding family-local parsers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache

from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree
import tree_sitter_python
from typing import ClassVar


@lru_cache(maxsize=1)
def _python_language() -> Language:
    return Language(tree_sitter_python.language())


@dataclass(frozen=True)
class NativePythonSyntaxIndex:
    """One native concrete-syntax tree with reusable compiled queries."""

    _PYTHON_QUERIES: ClassVar[dict[str, Query]] = {}

    _COMMON_PROJECTION_QUERY = """
    (class_definition) @class
    (function_definition) @function
    (if_statement) @if
    (elif_clause) @elif
    (call) @call
    (dictionary) @dictionary
    (assignment) @assignment
    """

    source_bytes: bytes
    tree: Tree
    _captures_by_query: dict[str, dict[str, tuple[Node, ...]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _statements_by_node: dict[tuple[Node, bool], ast.stmt] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _expressions_by_node: dict[Node, ast.expr] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _arguments_by_node: dict[Node, ast.arguments] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _class_headers_by_node: dict[Node, ast.ClassDef] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _function_headers_by_node: dict[Node, ast.FunctionDef | ast.AsyncFunctionDef] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
            compare=False,
        )
    )

    @classmethod
    def from_source(cls, source: str) -> NativePythonSyntaxIndex:
        source_bytes = source.encode("utf-8")
        return cls(
            source_bytes=source_bytes,
            tree=Parser(_python_language()).parse(source_bytes),
        )

    @property
    def is_complete(self) -> bool:
        """Return whether the native grammar accepted the complete source."""

        return not self.tree.root_node.has_error

    def captures(self, query_source: str) -> dict[str, tuple[Node, ...]]:
        """Run a cached native query and freeze each capture collection."""

        cached = self._captures_by_query.get(query_source)
        if cached is not None:
            return cached
        captures = QueryCursor(self._python_query(query_source)).captures(
            self.tree.root_node
        )
        frozen = {name: tuple(nodes) for name, nodes in captures.items()}
        self._captures_by_query[query_source] = frozen
        return frozen

    def common_captures(self) -> dict[str, tuple[Node, ...]]:
        """Return the shared event stream for compact source projections."""

        return self.captures(self._COMMON_PROJECTION_QUERY)

    def top_level_assignment_statements(self) -> tuple[Node, ...]:
        """Return module-body assignment statements in lexical order.

        Tree-sitter represents chained assignment as nested ``assignment``
        nodes.  Selecting the enclosing expression statement once preserves
        Python AST assignment semantics without a second module-wide walk.
        """

        statements = {
            assignment.parent
            for assignment in self.common_captures().get("assignment", ())
            if assignment.parent is not None
            and assignment.parent.type == "expression_statement"
            and assignment.parent.parent == self.tree.root_node
        }
        return tuple(
            sorted(
                statements,
                key=lambda node: (node.start_byte, -node.end_byte),
            )
        )

    def top_level_declarations(self, capture_name: str) -> tuple[Node, ...]:
        """Return captured class/function declarations owned by the module."""

        declarations = (
            node
            for node in self.common_captures().get(capture_name, ())
            if node.parent == self.tree.root_node
            or (
                node.parent is not None
                and node.parent.type == "decorated_definition"
                and node.parent.parent == self.tree.root_node
            )
        )
        return tuple(
            sorted(
                declarations,
                key=lambda node: (node.start_byte, -node.end_byte),
            )
        )

    def source_for(self, node: Node) -> bytes:
        return self.source_bytes[node.start_byte : node.end_byte]

    def declared_name(self, node: Node) -> str:
        """Return the source name declared by a class or function node."""

        name = node.child_by_field_name("name")
        if name is None:
            raise ValueError(f"{node.type} has no declared name")
        return self.source_for(name).decode("utf-8")

    def named_scope_nodes(self, node: Node) -> tuple[Node, ...]:
        """Return enclosing class/function scopes in lexical order."""

        scopes: list[Node] = []
        current = node.parent
        while current is not None:
            if current.type in {"class_definition", "function_definition"}:
                scopes.append(current)
            current = current.parent
        return tuple(reversed(scopes))

    def statement_for(self, node: Node, *, elif_as_if: bool = False) -> ast.stmt:
        """Parse one native statement fragment with original source lines."""

        cache_key = (node, elif_as_if)
        cached = self._statements_by_node.get(cache_key)
        if cached is not None:
            return cached
        source = self.source_for(node).decode("utf-8")
        if elif_as_if and node.type == "elif_clause":
            source = "if" + source.removeprefix("elif")
        if node.start_point.column:
            # Preserve the original indentation bytes, including indentation
            # inside multiline string values.  A synthetic outer block makes
            # the fragment parseable without destructive text dedenting.
            wrapped = "if True:\n" + " " * node.start_point.column + source
            wrapper = ast.parse(wrapped).body[0]
            if not isinstance(wrapper, ast.If):
                raise TypeError("statement wrapper did not parse as an if block")
            statement = wrapper.body[0]
            ast.increment_lineno(statement, node.start_point.row - 1)
        else:
            statement = ast.parse(source).body[0]
            ast.increment_lineno(statement, node.start_point.row)
        self._statements_by_node[cache_key] = statement
        return statement

    def function_for(
        self,
        node: Node,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Parse one function with its decorators and original source lines."""

        statement_node = (
            node.parent
            if node.parent is not None and node.parent.type == "decorated_definition"
            else node
        )
        statement = self.statement_for(statement_node)
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError(f"{node.type} did not parse as a Python function")
        return statement

    def class_for(self, node: Node) -> ast.ClassDef:
        """Parse one class with its decorators and original source lines."""

        statement_node = (
            node.parent
            if node.parent is not None and node.parent.type == "decorated_definition"
            else node
        )
        statement = self.statement_for(statement_node)
        if not isinstance(statement, ast.ClassDef):
            raise TypeError(f"{node.type} did not parse as a Python class")
        return statement

    def class_header_for(self, node: Node) -> ast.ClassDef:
        """Parse one class header without reparsing its potentially large body."""

        cached = self._class_headers_by_node.get(node)
        if cached is not None:
            return cached
        class_node = self._definition_header_for(node)
        if not isinstance(class_node, ast.ClassDef):
            raise TypeError("class header did not parse as a class")
        class_node.end_lineno = self._definition_end_lineno(node)
        self._class_headers_by_node[node] = class_node
        return class_node

    def function_header_for(
        self,
        node: Node,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Parse one function signature/decorator header with a stub body."""

        cached = self._function_headers_by_node.get(node)
        if cached is not None:
            return cached
        function_node = self._definition_header_for(node)
        if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError("function header did not parse as a function")
        function_node.end_lineno = self._definition_end_lineno(node)
        self._function_headers_by_node[node] = function_node
        return function_node

    def _definition_header_for(self, node: Node) -> ast.stmt:
        body = node.child_by_field_name("body")
        if body is None:
            raise ValueError(f"{node.type} has no body")
        statement_node = (
            node.parent
            if node.parent is not None and node.parent.type == "decorated_definition"
            else node
        )
        header = self.source_bytes[statement_node.start_byte : body.start_byte].decode(
            "utf-8"
        )
        source = f"{header}pass"
        if statement_node.start_point.column:
            source = "if True:\n" + " " * statement_node.start_point.column + source
            wrapper = ast.parse(source).body[0]
            if not isinstance(wrapper, ast.If) or not wrapper.body:
                raise TypeError(
                    "definition header wrapper did not parse as an if block"
                )
            definition = wrapper.body[0]
            ast.increment_lineno(definition, statement_node.start_point.row - 1)
            return definition
        definition = ast.parse(source).body[0]
        ast.increment_lineno(definition, statement_node.start_point.row)
        return definition

    @staticmethod
    def _definition_end_lineno(node: Node) -> int:
        """Match Python AST spans without walking an entire definition body.

        Tree-sitter includes trailing comments in a definition's extent while
        ``ast`` ends at the final syntax token. Follow only the rightmost
        non-comment branch so compact header extraction preserves the canonical
        AST line without paying a body-sized traversal.
        """

        current = node
        while current.children:
            child = next(
                (
                    candidate
                    for candidate in reversed(current.children)
                    if candidate.type != "comment"
                ),
                None,
            )
            if child is None:
                break
            current = child
        return current.end_point.row + 1

    @staticmethod
    def direct_enclosing_class(node: Node) -> Node | None:
        """Return a class when ``node`` is a direct class-body declaration."""

        current = node.parent
        if current is not None and current.type == "decorated_definition":
            current = current.parent
        if current is None or current.type != "block":
            return None
        owner = current.parent
        return owner if owner is not None and owner.type == "class_definition" else None

    def expression_for(self, node: Node) -> ast.expr:
        """Parse one native expression fragment with original source lines."""

        cached = self._expressions_by_node.get(node)
        if cached is not None:
            return cached
        source = self.source_for(node).decode("utf-8")
        # A fluent expression can rely on parentheses owned by an enclosing
        # syntax node, leaving this exact fragment with a leading ``.method``
        # continuation.  Synthetic parentheses restore that lexical context
        # without inserting a line or changing the expression's line numbers.
        expression = ast.parse(f"({source})", mode="eval").body
        ast.increment_lineno(expression, node.start_point.row)
        self._expressions_by_node[node] = expression
        return expression

    @staticmethod
    def _python_query(source: str) -> Query:
        """Return one process-lifetime query authority for a declaration.

        Query objects contain no module AST state.  Keep them outside the generic
        module-memory cache clearing protocol so workers do not recompile every
        declared native query for every source shard.
        """

        query = NativePythonSyntaxIndex._PYTHON_QUERIES.get(source)
        if query is None:
            query = Query(_python_language(), source)
            NativePythonSyntaxIndex._PYTHON_QUERIES[source] = query
        return query
