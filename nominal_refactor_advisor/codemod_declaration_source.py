"""Source-preserving rendering for Python declaration mutations."""

from __future__ import annotations

import ast
import copy
import tokenize
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import (
    dataclass,
    replace,
)
from functools import cached_property
from importlib import import_module as import_module_by_name
from importlib.util import find_spec
from typing import (
    ClassVar,
    TYPE_CHECKING,
    cast,
)

from .ast_tools import (
    AstKeywordSourceProjection,
    AstParentIndex,
    FunctionDefinitionNode,
    is_docstring_statement,
)
from .codemod_source_edits import (
    NominalSourceEdit,
    PhysicalSourceEditConflictError,
    SourceLineSpan,
    SourceNodeDecoratorPolicy,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
    _joined_rationales,
)
from .codemod_statement_source import StatementSource
from .declaration_dependencies import (
    FunctionBindingABC,
    FunctionParameterBinding,
)
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .source_geometry import (
    ClassHeaderSourceSpan,
    SourceLineSegmentAuthority,
)
from .value_expression import LexicalValueReference

if TYPE_CHECKING:
    from .codemod_selection_context import CodemodSelectorContext
    from .codemod_source_edits import PhysicalSourceEdit


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

    def __post_init__(self) -> None:
        if not isinstance(
            self.node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            raise ValueError("Named declaration source requires a class or function")

    @cached_property
    def geometry(self) -> SourceTextGeometry:
        return SourceTextGeometry(self.source)

    @cached_property
    def declaration_line_span(self) -> SourceLineSpan:
        """Own the complete declaration, including every decorator marker."""

        return self.geometry.node_line_span(
            SourceNodeSpan(self.node, SourceNodeDecoratorPolicy.INCLUDE)
        )

    @cached_property
    def name_span(self) -> SourceTextSpan:
        """Return the exact identifier token that declares this source name."""

        geometry = self.geometry
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

    def __post_init__(self) -> None:
        if not isinstance(self.node, ast.ClassDef):
            raise ValueError("Class source authority requires a class declaration")


@dataclass(frozen=True)
class ClassHeaderSpanSourceAuthority(ClassSourceAuthority):
    """Rewrite a class header over its full source span."""

    single_line_header_limit: ClassVar[int] = 88

    @cached_property
    def signature_span(self) -> SourceTextSpan:
        """Resolve the header suffix, including generic parameters, through the colon."""

        return SourceTextSpan(
            self.name_span.end_offset,
            self.geometry.token_position_offset(self.source_span.end_position),
        )

    @cached_property
    def declaration_prefix(self) -> str:
        """Retain the exact name and generic parameters before the base list."""

        return self.source_span.declaration_prefix

    def header_replacement(self, lines: tuple[str, ...]) -> SourceTextSpanReplacement:
        """Replace only the header, retaining inline suites and trailing comments."""

        start = self.geometry.required_node_offsets(self.node)[0]
        span = SourceTextSpan(start, self.signature_span.end_offset)
        if not self.source_span.is_reconstructible:
            raise ValueError("Class header replacement would discard comments")
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source="".join(lines).removeprefix(self.indentation).rstrip("\r\n"),
        )

    def source_edits(
        self, lines: tuple[str, ...], *, file_path: str, rationale: str,
    ) -> tuple[PhysicalSourceEdit, ...]:
        """Apply header rendering through the shared exact-boundary contract."""

        if lines == self.current_header_lines:
            return ()
        return self.geometry.physical_edits(
            file_path=file_path, replacements=(self.header_replacement(lines),),
            rationale=rationale,
        )

    @cached_property
    def source_span(self) -> ClassHeaderSourceSpan:
        return ClassHeaderSourceSpan(self.node, self.geometry.lines)

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
        if old_base_name == new_base_name:
            return self.current_header_lines
        if new_base_name in self.base_items:
            raise ValueError(f"Class header already contains base {new_base_name!r}")
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
            header = f"{self.declaration_prefix}({', '.join(items)}):"
        else:
            header = f"{self.declaration_prefix}:"
        if len(f"{indentation}{header}") <= self.single_line_header_limit:
            return (f"{indentation}{header}\n",)
        return (
            f"{indentation}{self.declaration_prefix}(\n",
            *(f"{indentation}    {item},\n" for item in items),
            f"{indentation}):\n",
        )


@dataclass(frozen=True)
class ClassBodySourceAuthority(ClassSourceAuthority):
    """Recover insertion geometry owned by one class body."""

    @property
    def source_lines(self) -> list[str]:
        return self.source.splitlines(keepends=True)

    @cached_property
    def header_span(self) -> ClassHeaderSourceSpan:
        return ClassHeaderSourceSpan(self.node, self.geometry.lines)

    @property
    def has_inline_suite(self) -> bool:
        return self.node.body[0].lineno == self.header_span.end_line

    @property
    def indentation(self) -> str:
        if self.has_inline_suite:
            class_line = self.geometry.lines[self.node.lineno - 1]
            return class_line[: len(class_line) - len(class_line.lstrip())] + "    "
        body_line = self.geometry.lines[self.node.body[0].lineno - 1]
        return body_line[: len(body_line) - len(body_line.lstrip())]

    @property
    def declaration_insert_line(self) -> int:
        if self.node.body and is_docstring_statement(self.node.body[0]):
            return self.node.body[0].end_lineno or self.node.body[0].lineno
        return self.header_span.end_line

    @property
    def before_first_method_offset(self) -> int:
        """Return the insertion offset without stealing attached comments."""

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
                self.geometry.line_offsets[self.node.end_lineno]
                if self.node.end_lineno is not None
                and self.node.end_lineno < len(self.geometry.line_offsets)
                else self.geometry.end_offset
            )
        insertion_line = self.geometry.node_start_line(
            SourceNodeSpan(first_method, SourceNodeDecoratorPolicy.INCLUDE)
        )
        source_lines = self.geometry.lines
        method_indent = self.indentation
        while insertion_line > self.node.lineno + 1:
            preceding_line = source_lines[insertion_line - 2]
            if not (
                preceding_line.startswith(method_indent)
                and preceding_line.removeprefix(method_indent).startswith("#")
            ):
                break
            insertion_line -= 1
        return self.geometry.line_offsets[insertion_line - 1]

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

    def member_insertion_replacement(
        self, members: tuple[str, ...]
    ) -> SourceTextSpanReplacement:
        """Insert members, expanding an inline suite through exact statement geometry."""

        if self.has_inline_suite:
            start = self.geometry.token_position_offset(self.header_span.end_position)
            _, end = self.geometry.node_span_offsets(SourceNodeSpan(self.node))
            existing = "".join(
                StatementSource(source=self.source, node=statement).member_source(
                    self.indentation
                )
                for statement in self.node.body
            )
            newline = (
                "\r\n"
                if self.geometry.lines[self.node.lineno - 1].endswith("\r\n")
                else "\n"
            )
            return SourceTextSpanReplacement.from_offsets(
                start_offset=start,
                end_offset=end,
                replacement_source=newline
                + existing
                + newline.join(member.rstrip("\r\n") for member in members)
                + newline,
            )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=self.before_first_method_offset,
            end_offset=self.before_first_method_offset,
            replacement_source=self.member_source(members),
        )


@dataclass(frozen=True)
class ClassMemberSource:
    """One named class member together with its exact indented source."""

    name: str
    source: str

    @classmethod
    def from_source(cls, source: str, *, indentation: str) -> "ClassMemberSource":
        """Derive the member identity and indented text from one authored declaration."""
        try:
            module = ast.parse(source)
        except SyntaxError as error:
            raise ValueError(
                f"Class member source is not valid Python: {error}"
            ) from error
        if len(module.body) != 1 or not isinstance(
            module.body[0],
            ast.ClassDef
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.Assign
            | ast.AnnAssign,
        ):
            raise ValueError("Class member source must contain one declaration")
        names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        if len(names) != 1:
            raise ValueError("Class member source must bind exactly one member name")
        return cls(
            name=next(iter(names)),
            source=SourceTextGeometry(source).indented_source(indentation),
        )


@dataclass(frozen=True, kw_only=True)
class ClassMemberInsertion(NominalSourceEdit):
    """Coalescible semantic insertion owned by one exact class declaration."""

    target_id: str
    members: tuple[ClassMemberSource, ...]
    member_sequence: ClassVar[
        Callable[[Iterable[ClassMemberSource]], tuple[ClassMemberSource, ...]]
    ] = staticmethod(tuple)

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        insertions_by_target: dict[str, list[ClassMemberInsertion]] = defaultdict(list)
        for peer in peers:
            insertion = cast(ClassMemberInsertion, peer)
            insertions_by_target[insertion.target_id].append(insertion)
        return tuple(
            self._coalesced_same_target(tuple(insertions_by_target[target_id]))
            for target_id in sorted(insertions_by_target)
        )

    @classmethod
    def _coalesced_same_target(
        cls,
        insertions: tuple["ClassMemberInsertion", ...],
    ) -> "ClassMemberInsertion":
        first = insertions[0]
        members_by_name: dict[str, ClassMemberSource] = {}
        for insertion in insertions:
            for member in insertion.members:
                existing = members_by_name.get(member.name)
                if existing is not None and existing.source != member.source:
                    raise PhysicalSourceEditConflictError(
                        f"Class member {member.name!r} has competing derived sources"
                    )
                members_by_name.setdefault(member.name, member)
        return replace(
            first,
            members=cls.member_sequence(members_by_name.values()),
            rationale=_joined_rationales(
                insertion.rationale for insertion in insertions
            ),
            contributors=NominalSourceEdit.merged_contributors(insertions),
            origins=NominalSourceEdit.merged_origins(insertions),
        )

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[PhysicalSourceEdit, ...]:
        target = context.source_index.target_by_id.get(self.target_id)
        node = context.ast_target_nodes_by_id.get(self.target_id)
        if target is None or not isinstance(node, ast.ClassDef):
            raise ValueError(
                f"Class member insertion target {self.target_id!r} is unavailable"
            )
        existing_member_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        collisions = existing_member_names.intersection(
            member.name for member in self.members
        )
        if collisions:
            raise ValueError(
                f"Class {target.qualname!r} already binds members "
                f"{tuple(sorted(collisions))!r}"
            )
        source = context.sources_by_file_path[target.file_path]
        insertion_point = ClassBodySourceAuthority(node=node, source=source)
        return tuple(
            replace(edit, contributors=self.contributors, origins=self.origins)
            for edit in insertion_point.geometry.physical_edits(
                file_path=target.file_path,
                replacements=(
                    insertion_point.member_insertion_replacement(
                        tuple(member.source for member in self.members)
                    ),
                ),
                rationale=self.rationale,
            )
        )


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
class FunctionSourceAuthority(NamedDeclarationSourceAuthority):
    """Source ownership shared by function header and suite rewrites."""

    node: FunctionDefinitionNode

    def __post_init__(self) -> None:
        if not isinstance(self.node, FunctionDefinitionNode):
            raise ValueError("Function source authority requires a function declaration")


@dataclass(frozen=True)
class DeclarationRegionSourceAuthority(NamedDeclarationSourceAuthority, ABC):
    """A named declaration's region replaced by an authored source fragment."""

    @abstractmethod
    def replacement(self, source: str, /) -> SourceTextSpanReplacement:
        """Derive the declaration-owned span and its replacement source."""

        raise NotImplementedError


@dataclass(frozen=True)
class FunctionRegionSourceAuthority(
    FunctionSourceAuthority, DeclarationRegionSourceAuthority, ABC
):
    """A declaration region restricted to functions and methods."""


@dataclass(frozen=True)
class FunctionAliasSourceAuthority(FunctionRegionSourceAuthority):
    """Replace a complete function declaration with a same-scope alias."""

    def replacement(self, implementation_name: str, /) -> SourceTextSpanReplacement:
        reference = LexicalValueReference.from_expression(
            ast.parse(implementation_name, mode="eval").body
        )
        if reference is None or reference.attribute_path:
            raise ValueError("Function alias requires one implementation name")
        node_span = SourceNodeSpan(self.node, SourceNodeDecoratorPolicy.INCLUDE)
        span = SourceTextSpan(*self.geometry.node_span_offsets(node_span))
        if self.geometry.span_contains_comment(span):
            raise ValueError("Function alias would discard comments")
        first = _SingleLogicalLineSource.parse(
            self.geometry.lines[self.geometry.node_start_line(node_span) - 1],
            "Function alias",
        )
        last = _SingleLogicalLineSource.parse(
            self.geometry.lines[node_span.end_line - 1], "Function alias"
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source=f"{first.indent}{self.node.name} = {reference.root_name}{last.newline}",
        )


@dataclass(frozen=True)
class DeclarationDecoratorsSourceAuthority(DeclarationRegionSourceAuthority):
    """Own only the decorator block preceding an unchanged declaration header."""

    def replacement(self, decorators_source: str, /) -> SourceTextSpanReplacement:
        first_line = self.geometry.node_start_line(
            SourceNodeSpan(self.node, SourceNodeDecoratorPolicy.INCLUDE)
        )
        span = SourceTextSpan(
            self.geometry.line_offsets[first_line - 1],
            self.geometry.line_offsets[self.node.lineno - 1],
        )
        if self.geometry.span_contains_comment(span):
            raise ValueError("Declaration decorator replacement would discard comments")
        header = _SingleLogicalLineSource.parse(
            self.geometry.lines[self.node.lineno - 1], "Declaration header"
        )
        prefix = decorators_source
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += header.newline or "\n"
        scaffold = SourceTextGeometry(prefix + "def _decorated(): pass\n")
        module = ast.parse(scaffold.source)
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise ValueError("Replacement must contain only declaration decorators")
        rendered = scaffold.indented_source(header.indent)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source="".join(
                rendered.splitlines(keepends=True)[: module.body[0].lineno - 1]
            ),
        )


@dataclass(frozen=True)
class FunctionDecoratorsSourceAuthority(
    FunctionRegionSourceAuthority, DeclarationDecoratorsSourceAuthority
):
    """Function-only refinement of the shared decorator source region."""


@dataclass(frozen=True)
class FunctionSignatureSourceAuthority(FunctionRegionSourceAuthority):
    """Replace a function signature without rewriting its identity or suite."""

    def replacement(self, signature_suffix: str, /) -> SourceTextSpanReplacement:
        span = self.geometry.function_signature_suffix_span(self.node)
        if self.geometry.span_contains_comment(span):
            raise ValueError("Function signature replacement would discard comments")
        suffix = _SingleLogicalLineSource.parse(
            signature_suffix,
            "function signature suffix",
        ).body.strip()
        if not suffix.startswith("(") or not suffix.endswith(":"):
            raise ValueError(
                "Replacement function signature suffix must start with '(' and "
                "end with ':'"
            )
        declaration_start = self.geometry.required_node_offsets(self.node)[0]
        prefix = self.source[declaration_start : span.start_offset]
        try:
            ast.parse(f"{prefix}{suffix}\n    pass\n")
        except SyntaxError as error:
            raise ValueError(
                f"Replacement function signature is not valid Python: {error}"
            ) from error
        return SourceTextSpanReplacement.from_offsets(
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            replacement_source=suffix,
        )


@dataclass(frozen=True)
class FunctionSuiteLayout:
    """Source-derived extent and formatting of one function suite."""

    span: SourceTextSpan
    indentation: str
    newline: str
    is_inline: bool

    def render(self, source: str) -> str:
        body = SourceTextGeometry(source).indented_source(self.indentation)
        return body if body.endswith(("\n", "\r")) else body + self.newline


@dataclass(frozen=True)
class FunctionSuiteSourceAuthority(FunctionRegionSourceAuthority, ABC):
    """Shared suite geometry for replacing or inserting function statements."""

    @cached_property
    def layout(self) -> FunctionSuiteLayout:
        signature_end = self.geometry.function_signature_suffix_span(self.node).end_offset
        header_line = self.geometry.line_number_for_offset(signature_end - 1)
        first_statement_line = self.geometry.node_start_line(
            SourceNodeSpan(
                self.node.body[0],
                SourceNodeDecoratorPolicy.INCLUDE,
            )
        )
        header_source = self.geometry.lines[header_line - 1]
        newline = "\r\n" if header_source.endswith("\r\n") else "\n"
        is_inline = first_statement_line == header_line
        if is_inline:
            start_offset = signature_end
            indentation = self.geometry.line_indent(signature_end) + "    "
        else:
            start_offset = self.geometry.line_offsets[header_line]
            first_statement_offset = self.geometry.line_offsets[first_statement_line - 1]
            indentation = self.geometry.line_indent(first_statement_offset)
        return FunctionSuiteLayout(
            span=SourceTextSpan(
                start_offset,
                self.geometry.node_span_offsets(SourceNodeSpan(self.node))[1],
            ),
            indentation=indentation, newline=newline, is_inline=is_inline,
        )


@dataclass(frozen=True)
class FunctionBodySourceAuthority(FunctionSuiteSourceAuthority):
    """Own the function suite, including its first nested declaration's decorators."""

    def replacement(self, body_source: str, /) -> SourceTextSpanReplacement:
        return SourceTextSpanReplacement.from_offsets(
            start_offset=self.layout.span.start_offset,
            end_offset=self.layout.span.end_offset,
            replacement_source=(self.layout.newline if self.layout.is_inline else "")
            + self.layout.render(body_source),
        )


@dataclass(frozen=True)
class FunctionBodyPrefixSourceAuthority(FunctionSuiteSourceAuthority):
    """Insert statements after the docstring and before existing executable code."""

    def replacement(self, body_source: str, /) -> SourceTextSpanReplacement:
        layout = self.layout
        docstring = self.node.body[0] if is_docstring_statement(self.node.body[0]) else None
        remaining = self.node.body[1:] if docstring is not None else self.node.body
        body = layout.render(body_source)
        if layout.is_inline:
            end = (
                self.geometry.required_node_offsets(remaining[0])[0]
                if remaining else layout.span.end_offset
            )
            retained_docstring = ""
            if docstring is not None:
                doc_start, doc_end = self.geometry.required_node_offsets(docstring)
                retained_docstring = layout.render(self.source[
                    doc_start : doc_end if remaining else layout.span.end_offset
                ])
            return SourceTextSpanReplacement.from_offsets(
                start_offset=layout.span.start_offset, end_offset=end,
                replacement_source=layout.newline + retained_docstring + body
                + (layout.indentation if remaining else ""),
            )
        if docstring is not None:
            doc_end = self.geometry.required_node_offsets(docstring)[1]
            if remaining and docstring.end_lineno == remaining[0].lineno:
                return SourceTextSpanReplacement.from_offsets(
                    start_offset=doc_end,
                    end_offset=self.geometry.required_node_offsets(remaining[0])[0],
                    replacement_source=layout.newline + body + layout.indentation,
                )
            insertion = self.geometry.node_span_offsets(SourceNodeSpan(docstring))[1]
        else:
            first_statement_line = self.geometry.node_start_line(
                SourceNodeSpan(
                    self.node.body[0],
                    SourceNodeDecoratorPolicy.INCLUDE,
                )
            )
            insertion = self.geometry.line_offsets[first_statement_line - 1]
        separator = "" if self.source[:insertion].endswith(("\n", "\r")) else layout.newline
        return SourceTextSpanReplacement.from_offsets(
            start_offset=insertion, end_offset=insertion,
            replacement_source=separator + body,
        )


@dataclass(frozen=True)
class FunctionBindingProjectionSourceAuthority(FunctionSourceAuthority):
    """Project owned binding reads onto an existing parameter's access path."""

    def selected_reads(
        self,
        binding: FunctionBindingABC,
        attribute_path: tuple[str, ...],
    ) -> tuple[ast.expr, ...]:
        """Narrow owned roots, rejecting writes and discarded comments."""
        roots = binding.required_references()
        if not attribute_path:
            return roots
        selector = LexicalValueReference(binding.binding_name, attribute_path)
        parents = AstParentIndex(self.node).parent_by_node
        reads = tuple(
            expression
            for root in roots
            if (expression := selector.select_expression(root, parents)) is not None
        )
        for read in reads:
            if not isinstance(read.ctx, ast.Load):
                raise ValueError(
                    "Access projection cannot migrate a direct write or delete"
                )
            span = SourceTextSpan.from_offsets(
                self.geometry.required_node_offsets(read)
            )
            if self.geometry.span_contains_comment(span):
                raise ValueError("Access projection would discard a comment")
        return reads

    def replacements_for(
        self,
        binding: FunctionBindingABC,
        reference: LexicalValueReference,
        *,
        attribute_path: tuple[str, ...] = (),
    ) -> tuple[SourceTextSpanReplacement, ...]:
        reads = self.selected_reads(binding, attribute_path)
        if not reads:
            raise ValueError(
                f"Binding {binding.binding_name!r} has no owned reads to project"
            )
        expressions = tuple(
            ast.fix_missing_locations(
                ast.copy_location(reference.as_expression(), read)
            )
            for read in reads
        )
        projected_function = copy.deepcopy(
            self.node,
            {id(read): expression for read, expression in zip(reads, expressions)},
        )
        carrier_reads = frozenset(
            FunctionParameterBinding(
                projected_function,
                reference.root_name,
            ).required_references()
        )
        projected_roots = tuple(
            node
            for expression in expressions
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        )
        if not all(root in carrier_reads for root in projected_roots):
            raise ValueError(
                f"Projection root {reference.root_name!r} is captured by another scope"
            )
        replacement_source = ast.unparse(reference.as_expression())
        return tuple(
            SourceTextSpanReplacement.from_offsets(
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                replacement_source=replacement_source,
            )
            for read in reads
            for span in (
                SourceTextSpan.from_offsets(self.geometry.required_node_offsets(read)),
            )
        )
