"""Source-preserving rendering for Python declaration mutations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from .class_index import ClassHeaderSourceSpan


@dataclass(frozen=True)
class ClassHeaderSpanSourceAuthority:
    """Rewrite a class header over its full source span."""

    node: ast.ClassDef
    source: str
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
            (
                f"{keyword.arg}={ast.unparse(keyword.value)}"
                if keyword.arg is not None
                else f"**{ast.unparse(keyword.value)}"
            )
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
class ClassSourceAuthority:
    """Class declaration and source text shared by rewrite projections."""

    node: ast.ClassDef
    source: str


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
        if (
            self.node.body
            and isinstance(self.node.body[0], ast.Expr)
            and isinstance(self.node.body[0].value, ast.Constant)
            and isinstance(self.node.body[0].value.value, str)
        ):
            return self.node.body[0].end_lineno or self.node.body[0].lineno
        return self.node.lineno


@dataclass(frozen=True)
class ClassBaseRewriteTarget(ClassSourceAuthority):
    """Class declaration target supported by the class-header rewrite engine."""

    @property
    def supports_base_rewrite(self) -> bool:
        return ClassHeaderSpanSourceAuthority(
            node=self.node,
            source=self.source,
        ).can_rewrite


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
