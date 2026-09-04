"""Source-preserving rendering for Python declaration mutations."""

from __future__ import annotations

import ast
from dataclasses import dataclass


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
