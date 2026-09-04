"""Source-preserving rendering for Python declaration mutations."""

from __future__ import annotations

import ast
import tokenize
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module as import_module_by_name
from importlib.util import find_spec
from typing import ClassVar

from .ast_tools import AstKeywordSourceProjection, is_docstring_statement
from .class_index import ClassHeaderSourceSpan
from .codemod_source_edits import SourceTextGeometry, SourceTextSpan
from .source_geometry import SourceLineSegmentAuthority


@dataclass(frozen=True)
class DirectClassDeclarationAuthority:
    """Project direct annotated class fields to exact source declarations."""

    source_segments: SourceLineSegmentAuthority
    node: ast.ClassDef

    def declarations_by_name(self) -> dict[str, str]:
        declaration_by_name: dict[str, str] = {}
        for statement in self.node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            source_segment = self.source_segments.segment_for_node(statement)
            if source_segment is None:
                return {}
            declaration_by_name[statement.target.id] = source_segment.strip()
        return declaration_by_name


@dataclass(frozen=True)
class PythonExpressionSourceFormatter:
    """Format expression replacements relative to their source insertion column."""

    line_length: int = 88

    def replacement_source(
        self,
        node: ast.expr,
        *,
        line_prefix: str = "",
    ) -> str:
        expression_source = ast.unparse(node)
        formatted_source = self.black_expression_source(
            expression_source,
            line_prefix=line_prefix,
        )
        return self.prefixed_continuation_source(
            formatted_source or expression_source,
            line_prefix=line_prefix,
        )

    def black_expression_source(
        self,
        expression_source: str,
        *,
        line_prefix: str = "",
    ) -> str | None:
        if find_spec("black") is None:
            return None
        black = import_module_by_name("black")
        mode = black.Mode(
            line_length=max(40, self.line_length - len(line_prefix)),
            target_versions={black.TargetVersion.PY311},
        )
        try:
            formatted = black.format_str(
                f"def _nra_expression():\n    return {expression_source}\n",
                mode=mode,
            )
        except Exception:
            return None
        return self.return_expression_source(formatted)

    @staticmethod
    def return_expression_source(formatted_wrapper_source: str) -> str | None:
        return_prefix = "    return "
        body_prefix = "    "
        lines = formatted_wrapper_source.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(return_prefix):
                continue
            expression_lines = [line.removeprefix(return_prefix)]
            expression_lines.extend(
                continuation_line.removeprefix(body_prefix)
                for continuation_line in lines[index + 1 :]
                if continuation_line.startswith(body_prefix)
            )
            return "\n".join(expression_lines)
        return None

    @staticmethod
    def prefixed_continuation_source(
        source: str,
        *,
        line_prefix: str,
    ) -> str:
        lines = source.splitlines()
        if len(lines) <= 1 or not line_prefix:
            return source
        return "\n".join(
            line if index == 0 else f"{line_prefix}{line}"
            for index, line in enumerate(lines)
        )


@dataclass(frozen=True)
class NamedDeclarationSourceAuthority:
    """Exact source geometry shared by named class and function declarations."""

    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    source: str

    @cached_property
    def name_span(self) -> SourceTextSpan:
        """Return the exact identifier token that declares this source name."""

        geometry = SourceTextGeometry(self.source)
        declaration_tokens = iter(
            token
            for token in geometry.tokens
            if token.start[0] >= self.node.lineno
            and token.start[0] <= (self.node.end_lineno or self.node.lineno)
        )
        declaration_keyword = next(
            (
                token
                for token in declaration_tokens
                if token.type == tokenize.NAME and token.string in {"class", "def"}
            ),
            None,
        )
        name_token = next(declaration_tokens, None)
        if (
            declaration_keyword is None
            or name_token is None
            or name_token.type != tokenize.NAME
            or name_token.string != self.node.name
        ):
            raise ValueError(
                f"Cannot resolve declaration name token for {self.node.name!r}"
            )
        return SourceTextSpan(
            start_offset=geometry.token_position_offset(name_token.start),
            end_offset=geometry.token_position_offset(name_token.end),
        )


@dataclass(frozen=True)
class ClassSourceAuthority(NamedDeclarationSourceAuthority):
    """Class declaration and source text shared by rewrite projections."""

    node: ast.ClassDef


@dataclass(frozen=True)
class ClassHeaderSpanSourceAuthority(ClassSourceAuthority):
    """Rewrite a class header over its full source span."""

    single_line_header_limit: ClassVar[int] = 88

    @cached_property
    def source_span(self) -> ClassHeaderSourceSpan:
        return ClassHeaderSourceSpan.from_source(self.node, self.source)

    @property
    def source_lines(self) -> tuple[str, ...]:
        return self.source_span.source_lines

    @property
    def start_line(self) -> int:
        return self.source_span.start_line

    @property
    def end_line(self) -> int:
        return self.source_span.end_line

    @property
    def indentation(self) -> str:
        if self.node.lineno < 1 or self.node.lineno > len(self.source_lines):
            return ""
        line = self.source_lines[self.node.lineno - 1]
        return line[: len(line) - len(line.lstrip())]

    @property
    def keyword_items(self) -> tuple[str, ...]:
        return tuple(
            AstKeywordSourceProjection(keyword).source()
            for keyword in self.node.keywords
        )

    @property
    def base_items(self) -> tuple[str, ...]:
        return tuple(ast.unparse(base) for base in self.node.bases)

    @property
    def can_rewrite(self) -> bool:
        return (
            self.source_span.is_reconstructible
            and 1 <= self.start_line <= self.end_line <= len(self.source_lines)
            and self.rendered_header_is_parseable
        )

    @property
    def rendered_header_is_parseable(self) -> bool:
        header_source = f"{''.join(self.header_lines(self.base_items, ''))}    pass\n"
        try:
            ast.parse(header_source)
        except SyntaxError:
            return False
        return True

    def with_added_base(self, base_name: str) -> tuple[str, ...]:
        if base_name in self.base_items:
            return self.current_header_lines
        return self.with_base_items((*self.base_items, base_name))

    def with_prepended_base(self, base_name: str) -> tuple[str, ...]:
        if base_name in self.base_items:
            return self.current_header_lines
        return self.with_base_items((base_name, *self.base_items))

    def without_base(self, base_name: str) -> tuple[str, ...]:
        if base_name not in self.base_items:
            return self.current_header_lines
        return self.with_base_items(
            tuple(base for base in self.base_items if base != base_name)
        )

    def with_replaced_base(
        self,
        old_base_name: str,
        new_base_name: str,
    ) -> tuple[str, ...]:
        matching_indexes = tuple(
            index
            for index, base_name in enumerate(self.base_items)
            if base_name == old_base_name
        )
        if len(matching_indexes) != 1:
            raise ValueError(
                f"Class header requires one base {old_base_name!r}; "
                f"found {len(matching_indexes)}"
            )
        replacement_index = matching_indexes[0]
        return self.with_base_items(
            tuple(
                new_base_name if index == replacement_index else base_name
                for index, base_name in enumerate(self.base_items)
            )
        )

    @property
    def current_header_lines(self) -> tuple[str, ...]:
        return self.source_lines[self.start_line - 1 : self.end_line]

    def with_base_items(self, base_items: tuple[str, ...]) -> tuple[str, ...]:
        return self.header_lines(base_items, self.indentation)

    def with_items(
        self,
        base_items: tuple[str, ...],
        keyword_items: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self.header_lines(
            base_items,
            self.indentation,
            keyword_items=keyword_items,
        )

    def header_lines(
        self,
        base_items: tuple[str, ...],
        indentation: str,
        *,
        keyword_items: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        resolved_keyword_items = (
            self.keyword_items if keyword_items is None else keyword_items
        )
        items = (*base_items, *resolved_keyword_items)
        if items:
            header = f"class {self.node.name}({', '.join(items)}):"
        else:
            header = f"class {self.node.name}:"
        if len(f"{indentation}{header}") <= self.single_line_header_limit:
            return (f"{indentation}{header}\n",)
        return (
            f"{indentation}class {self.node.name}(\n",
            *(f"{indentation}    {item},\n" for item in items),
            f"{indentation}):\n",
        )


@dataclass(frozen=True)
class ClassBodySourceAuthority(ClassSourceAuthority):
    """Recover insertion geometry owned by one class body."""

    @property
    def source_lines(self) -> list[str]:
        return self.source.splitlines(keepends=True)

    @property
    def indentation(self) -> str:
        if self.node.body:
            body_line = self.source_lines[self.node.body[0].lineno - 1]
            indentation = body_line[: len(body_line) - len(body_line.lstrip())]
            if indentation:
                return indentation
        return "    "

    @property
    def declaration_insert_line(self) -> int:
        if self.node.body and is_docstring_statement(self.node.body[0]):
            return self.node.body[0].end_lineno or self.node.body[0].lineno
        return self.node.lineno

    @property
    def before_first_method_offset(self) -> int:
        """Return the insertion offset without stealing attached comments."""

        geometry = SourceTextGeometry(self.source)
        first_method = next(
            (
                statement
                for statement in self.node.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            ),
            None,
        )
        if first_method is None:
            return (
                geometry.line_offsets[self.node.end_lineno]
                if self.node.end_lineno is not None
                and self.node.end_lineno < len(geometry.line_offsets)
                else geometry.end_offset
            )
        insertion_line = ClassHeaderSourceSpan.statement_start_line(first_method)
        source_lines = geometry.lines
        method_indent = " " * first_method.col_offset
        while insertion_line > self.node.lineno + 1:
            preceding_line = source_lines[insertion_line - 2]
            if not (
                preceding_line.startswith(method_indent)
                and preceding_line.removeprefix(method_indent).startswith("#")
            ):
                break
            insertion_line -= 1
        return geometry.line_offsets[insertion_line - 1]

    def member_source(self, members: tuple[str, ...]) -> str:
        """Render class members at this point with stable class-body spacing."""

        insertion_offset = self.before_first_method_offset
        prefix = self.source[:insertion_offset]
        suffix = self.source[insertion_offset:]
        if prefix.endswith("\n\n"):
            leading_separator = ""
        elif prefix.endswith("\n"):
            leading_separator = "\n"
        else:
            leading_separator = "\n\n"
        if suffix.startswith("\n\n"):
            trailing_separator = ""
        elif suffix.startswith("\n"):
            trailing_separator = "\n"
        else:
            trailing_separator = "\n\n"
        body = "\n\n".join(member.rstrip("\r\n") for member in members)
        return f"{leading_separator}{body}{trailing_separator}"


@dataclass(frozen=True)
class _SingleLogicalLineSource:
    """Parsed single source line preserving indentation and newline."""

    indent: str
    body: str
    newline: str

    @classmethod
    def parse(cls, original_line: str, role: str) -> "_SingleLogicalLineSource":
        body = original_line.rstrip("\r\n")
        newline = original_line[len(body) :]
        stripped_body = body.lstrip()
        indent = body[: len(body) - len(stripped_body)]
        if "\n" in stripped_body or "\r" in stripped_body:
            raise ValueError(f"{role} operation requires one source line")
        return cls(indent=indent, body=stripped_body, newline=newline)

    def rebuild(self, body: str) -> str:
        return f"{self.indent}{body}{self.newline}"


@dataclass(frozen=True)
class FunctionSignatureSourceAuthority:
    """Rewrite one single-line function signature."""

    original_line: str

    @property
    def declaration_prefix(self) -> str:
        header = self.header.body
        prefix, separator, _suffix = header.partition("(")
        if not separator or not prefix.startswith(("def ", "async def ")):
            raise ValueError(
                "Function signature replacement requires a single-line def"
            )
        return prefix.rstrip()

    @property
    def header(self) -> _SingleLogicalLineSource:
        return _SingleLogicalLineSource.parse(
            self.original_line,
            "function signature",
        )

    def replacement_line(self, signature_suffix: str) -> str:
        line = self.header
        suffix = _SingleLogicalLineSource.parse(
            signature_suffix,
            "function signature suffix",
        ).body.strip()
        if not suffix.startswith("(") or not suffix.endswith(":"):
            raise ValueError(
                "Replacement function signature suffix must start with '(' and "
                "end with ':'"
            )
        replacement_body = f"{self.declaration_prefix}{suffix}"
        try:
            ast.parse(f"{replacement_body}\n    pass\n")
        except SyntaxError as error:
            raise ValueError(
                f"Replacement function signature is not valid Python: {error}"
            ) from error
        return line.rebuild(replacement_body)
