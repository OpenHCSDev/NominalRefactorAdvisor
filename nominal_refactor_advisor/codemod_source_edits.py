"""Physical source-edit algebra for codemod execution."""

from __future__ import annotations

import ast
import hashlib
import io
import sys
import tokenize
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    replace,
)
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Self,
    cast,
)

from .codemod_paths import SourceCreationPathAuthority as SourceCreationPathAuthority
from .codemod_payload import (
    CodemodPayloadRecord,
    DataclassPayloadProjection,
    EmptyDefaultStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    RequiredIntegerPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_semantics import RewriteOperation
from .codemod_spacing import SourceInsertionBoundary
from .collection_algebra import sorted_tuple
from .json_reports import (
    DataclassJsonReport,
    json_report_field,
    json_report_property,
)
from .source_geometry import (
    SourceByteSpan,
    SourceLineSegmentAuthority,
)
from .source_index import (
    AstTargetDigest,
    SourceIndex,
    SourceTargetSpan,
)

if TYPE_CHECKING:
    from .codemod_operations import RefactorRecipeOperation
    from .codemod_selection_context import CodemodSelectorContext


class SourceNodeDecoratorPolicy(StrEnum):
    """Whether source node spans include decorators."""

    EXCLUDE = ("exclude", False)
    INCLUDE = ("include", True)

    def __new__(
        cls,
        value: str,
        includes_decorators: bool,
    ) -> "SourceNodeDecoratorPolicy":
        member = str.__new__(cls, value)
        member._value_ = value
        member._includes_decorators = includes_decorators
        return member

    @property
    def includes_decorators(self) -> bool:
        return self._includes_decorators


@dataclass(frozen=True, kw_only=True)
class ReplacementSource:
    replacement_source: str


@dataclass(frozen=True)
class SourceEditOrigin(DataclassPayloadProjection):
    """Operation identity retained until a semantic edit has physical geometry."""

    recipe_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    plan_item_declaration: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )
    plan_item_index: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())

    @property
    def identity(self) -> tuple[object, ...]:
        return self.recipe_id, self.plan_item_declaration, self.plan_item_index

    def contributor_for(
        self,
        source_edit: "PhysicalSourceEdit",
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return SourceRewriteContributor.from_source_edit(
            recipe_id=self.recipe_id,
            plan_item_declaration=self.plan_item_declaration,
            plan_item_index=self.plan_item_index,
            source_edit=source_edit,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def merge(
        cls,
        *origin_groups: Iterable[Self],
    ) -> tuple[Self, ...]:
        origins_by_identity = {
            origin.identity: origin
            for origin_group in origin_groups
            for origin in origin_group
        }
        return tuple(origins_by_identity.values())


@dataclass(frozen=True, kw_only=True)
class SourceRewriteContributor(SourceEditOrigin, CodemodPayloadRecord):
    """Nominal plan-item provenance plus its executable source precondition."""

    file_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    end_line: int = codemod_payload_field(RequiredIntegerPayloadValueCodec())
    source_hash: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @classmethod
    def from_target(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        target: AstTargetDigest,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return cls.from_source_span(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=target.file_path,
            line=target.line,
            end_line=target.end_line,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def from_source_edit(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        source_edit: "PhysicalSourceEdit",
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return cls.from_source_span(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=source_edit.file_path,
            line=source_edit.start_line,
            end_line=source_edit.end_line,
            sources_by_file_path=sources_by_file_path,
        )

    @classmethod
    def from_source_span(
        cls,
        *,
        recipe_id: str,
        plan_item_declaration: str,
        plan_item_index: int,
        file_path: str,
        line: int,
        end_line: int,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        source = sources_by_file_path[file_path]
        return cls(
            recipe_id=recipe_id,
            plan_item_declaration=plan_item_declaration,
            plan_item_index=plan_item_index,
            file_path=file_path,
            line=line,
            end_line=end_line,
            source_hash=CodemodSourceRevision.hash_source(
                SourceLineSpan(line, end_line).source_from(source)
            ),
        )

    def for_target(
        self,
        target: AstTargetDigest,
        sources_by_file_path: Mapping[str, str],
    ) -> "SourceRewriteContributor":
        return type(self).from_target(
            recipe_id=self.recipe_id,
            plan_item_declaration=self.plan_item_declaration,
            plan_item_index=self.plan_item_index,
            target=target,
            sources_by_file_path=sources_by_file_path,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            *super().identity,
            self.file_path,
            self.line,
            self.end_line,
        )

    def require_source(self, sources_by_file_path: Mapping[str, str]) -> None:
        source = sources_by_file_path.get(self.file_path)
        if source is None or self.source_hash != CodemodSourceRevision.hash_source(
            SourceLineSpan(self.line, self.end_line).source_from(source)
        ):
            raise CodemodSourceRevisionError(
                "Compiled source rewrite contributor no longer matches "
                f"{self.file_path}:{self.line}-{self.end_line}: "
                f"{self.recipe_id}/{self.plan_item_declaration}"
                f"[{self.plan_item_index}]"
            )


@dataclass(frozen=True, kw_only=True)
class NominalSourceEdit(ABC):
    """Declaration-owned semantic source edit emitted by recipe operations."""

    rationale: str = ""
    contributors: tuple[SourceRewriteContributor, ...] = ()
    origins: tuple[SourceEditOrigin, ...] = ()

    def with_origin(self, origin: SourceEditOrigin) -> "NominalSourceEdit":
        return replace(
            self,
            origins=SourceEditOrigin.merge(self.origins, (origin,)),
        )

    @abstractmethod
    def coalesced_with_peers(
        self,
        peers: tuple["NominalSourceEdit", ...],
        context: "CodemodSelectorContext",
    ) -> tuple["NominalSourceEdit", ...]:
        """Coalesce edits owned by this exact nominal declaration."""

    @abstractmethod
    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple["PhysicalSourceEdit", ...]:
        """Project this semantic edit into physical source geometry."""

    @classmethod
    def coalesced_by_declaration(
        cls,
        edits: Iterable["NominalSourceEdit"],
        context: "CodemodSelectorContext",
    ) -> tuple["NominalSourceEdit", ...]:
        edits_by_declaration: dict[
            type[NominalSourceEdit],
            list[NominalSourceEdit],
        ] = {}
        for edit in edits:
            edits_by_declaration.setdefault(type(edit), []).append(edit)
        return tuple(
            coalesced
            for declaration_edits in edits_by_declaration.values()
            for coalesced in declaration_edits[0].coalesced_with_peers(
                tuple(declaration_edits),
                context,
            )
        )

    @staticmethod
    def merged_origins(
        edits: Iterable["NominalSourceEdit"],
    ) -> tuple[SourceEditOrigin, ...]:
        return SourceEditOrigin.merge(*(edit.origins for edit in edits))

    @staticmethod
    def merged_contributors(
        edits: Iterable["NominalSourceEdit"],
    ) -> tuple[SourceRewriteContributor, ...]:
        return SourceRewriteContributor.merge(*(edit.contributors for edit in edits))


class PhysicalSourceEditConflictError(ValueError):
    """Physical source edits cannot coexist in one nominal rewrite."""


@dataclass(frozen=True, kw_only=True)
class PhysicalSourceEdit(NominalSourceEdit, ABC):
    """Semantic edit whose absolute source-line geometry is resolved."""

    file_path: str

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple["PhysicalSourceEdit", ...]:
        del context
        return (self,)

    @abstractmethod
    def conflicts_with(self, other: "PhysicalSourceEdit") -> bool:
        """Return whether two physical edits cannot be applied as one rewrite."""

    @abstractmethod
    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        """Accept a span-owned conflict query through nominal dispatch."""

    @abstractmethod
    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        """Accept an insertion-owned conflict query through nominal dispatch."""

    @classmethod
    def require_compatible(
        cls,
        edits: tuple["PhysicalSourceEdit", ...],
    ) -> tuple["PhysicalSourceEdit", ...]:
        for index, first in enumerate(edits):
            for second in edits[index + 1 :]:
                if first.file_path == second.file_path and first.conflicts_with(second):
                    raise PhysicalSourceEditConflictError(
                        "Physical source edits conflict in "
                        f"{first.file_path}:{first.start_line}-{first.end_line} and "
                        f"{second.start_line}-{second.end_line}"
                    )
        return edits


@dataclass(frozen=True, kw_only=True)
class SourceSpanEdit(PhysicalSourceEdit, ABC):
    """Physical edit over one non-empty absolute line span."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line > self.end_line:
            raise ValueError("Source span edits require a non-empty span")

    @classmethod
    def from_replacement_lines(
        cls,
        *,
        file_path: str,
        start_line: int,
        end_line: int,
        replacement_lines: tuple[str, ...],
        rationale: str = "",
        contributors: tuple[SourceRewriteContributor, ...] = (),
        origins: tuple[SourceEditOrigin, ...] = (),
    ) -> "SourceSpanEdit":
        """Classify replacement output once at the physical-edit boundary."""

        if not replacement_lines:
            return SourceSpanDeletion(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                rationale=rationale,
                contributors=contributors,
                origins=origins,
            )
        return SourceSpanReplacement(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            replacement_lines=replacement_lines,
            rationale=rationale,
            contributors=contributors,
            origins=origins,
        )

    def conflicts_with(self, other: PhysicalSourceEdit) -> bool:
        return other.conflicts_with_span(self.start_line, self.end_line)

    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        return self.start_line <= end_line and start_line <= self.end_line

    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        return self.start_line < insertion_line <= self.end_line


@dataclass(frozen=True, kw_only=True)
class SourceSpanReplacement(SourceSpanEdit):
    """Replace one non-empty absolute line span with explicit source lines."""

    replacement_lines: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.replacement_lines:
            raise ValueError(
                "Source span replacements require replacement lines; "
                "use SourceSpanDeletion to remove source"
            )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        replacements_by_span: dict[
            tuple[str, int, int],
            list[SourceSpanReplacement],
        ] = defaultdict(list)
        for peer in peers:
            replacement = cast(SourceSpanReplacement, peer)
            replacements_by_span[
                replacement.file_path,
                replacement.start_line,
                replacement.end_line,
            ].append(replacement)
        return tuple(
            self._coalesced_same_span(tuple(replacements))
            for replacements in replacements_by_span.values()
        )

    @staticmethod
    def _coalesced_same_span(
        replacements: tuple["SourceSpanReplacement", ...],
    ) -> "SourceSpanReplacement":
        first = replacements[0]
        if any(
            replacement.replacement_lines != first.replacement_lines
            for replacement in replacements[1:]
        ):
            raise PhysicalSourceEditConflictError(
                "Conflicting source span replacements target "
                f"{first.file_path}:{first.start_line}-{first.end_line}"
            )
        return replace(
            first,
            rationale=_joined_rationales(
                replacement.rationale for replacement in replacements
            ),
            contributors=NominalSourceEdit.merged_contributors(replacements),
            origins=NominalSourceEdit.merged_origins(replacements),
        )


@dataclass(frozen=True, kw_only=True)
class SourceSpanDeletion(SourceSpanEdit):
    """Delete one non-empty absolute line span."""

    @property
    def replacement_lines(self) -> tuple[str, ...]:
        return ()

    @classmethod
    def target_span(
        cls,
        context: "CodemodSelectorContext",
        target_digest: AstTargetDigest,
    ) -> SourceLineSpan:
        """Derive the complete decorated span owned by one AST target."""

        target_node = context.ast_target_nodes_by_id.get(target_digest.target_id)
        return (
            SourceNodeSpan(
                target_node,
                SourceNodeDecoratorPolicy.INCLUDE,
            ).line_span
            if isinstance(target_node, ast.stmt)
            else SourceLineSpan(
                target_digest.line,
                target_digest.end_line,
            )
        )

    @classmethod
    def for_target(
        cls,
        context: "CodemodSelectorContext",
        target_digest: AstTargetDigest,
        *,
        rationale: str = "",
    ) -> Self:
        """Delete exactly one complete target while preserving its separator."""

        target_span = cls.target_span(context, target_digest)
        return cls(
            file_path=target_digest.file_path,
            start_line=target_span.start_line,
            end_line=target_span.end_line,
            rationale=rationale or f"Delete target {target_digest.qualname!r}.",
        )

    @classmethod
    def for_statement(
        cls,
        context: "CodemodSelectorContext",
        target_digest: AstTargetDigest,
        *,
        rationale: str = "",
    ) -> Self:
        """Delete one complete target and the separator owned by its statement."""

        return cls.for_statement_span(
            file_path=target_digest.file_path,
            source=context.sources_by_file_path[target_digest.file_path],
            statement_span=cls.target_span(context, target_digest),
            rationale=rationale or f"Delete target {target_digest.qualname!r}.",
        )

    @classmethod
    def for_statement_node(
        cls,
        *,
        file_path: str,
        source: str,
        statement: ast.stmt,
        rationale: str = "",
    ) -> Self:
        """Delete one parsed statement and the separator that it owns."""

        return cls.for_statement_span(
            file_path=file_path,
            source=source,
            statement_span=SourceNodeSpan(
                statement,
                SourceNodeDecoratorPolicy.INCLUDE,
            ).line_span,
            rationale=rationale,
        )

    @classmethod
    def for_statement_span(
        cls,
        *,
        file_path: str,
        source: str,
        statement_span: SourceLineSpan,
        rationale: str = "",
    ) -> Self:
        """Delete one statement span and its source-owned separator."""

        deletion_span = SourceTextGeometry(source).statement_deletion_span(
            statement_span
        )
        return cls(
            file_path=file_path,
            start_line=deletion_span.start_line,
            end_line=deletion_span.end_line,
            rationale=rationale,
        )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        deletions = sorted_tuple(
            (cast(SourceSpanDeletion, peer) for peer in peers),
            key=lambda deletion: (
                deletion.file_path,
                deletion.start_line,
                deletion.end_line,
            ),
        )
        coalesced: list[SourceSpanDeletion] = []
        for deletion in deletions:
            if (
                coalesced
                and coalesced[-1].file_path == deletion.file_path
                and deletion.start_line <= coalesced[-1].end_line
            ):
                previous = coalesced[-1]
                coalesced[-1] = replace(
                    previous,
                    end_line=max(previous.end_line, deletion.end_line),
                    rationale=_joined_rationales(
                        (previous.rationale, deletion.rationale)
                    ),
                    contributors=NominalSourceEdit.merged_contributors(
                        (previous, deletion)
                    ),
                    origins=NominalSourceEdit.merged_origins((previous, deletion)),
                )
                continue
            coalesced.append(deletion)
        return tuple(coalesced)


@dataclass(frozen=True, kw_only=True)
class SourceInsertion(PhysicalSourceEdit):
    """Insert source at one absolute line anchor."""

    insertion_line: int
    inserted_lines: tuple[str, ...] = ()
    leading_boundary: SourceInsertionBoundary = SourceInsertionBoundary.PRESERVE

    @property
    def start_line(self) -> int:
        return self.insertion_line

    @property
    def end_line(self) -> int:
        return self.insertion_line - 1

    @property
    def replacement_lines(self) -> tuple[str, ...]:
        return self.inserted_lines

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        insertions_by_anchor: dict[
            tuple[str, int],
            list[SourceInsertion],
        ] = defaultdict(list)
        for peer in peers:
            insertion = cast(SourceInsertion, peer)
            insertions_by_anchor[
                insertion.file_path,
                insertion.insertion_line,
            ].append(insertion)
        return tuple(
            self._coalesced_same_anchor(tuple(insertions))
            for insertions in insertions_by_anchor.values()
        )

    def conflicts_with(self, other: PhysicalSourceEdit) -> bool:
        return other.conflicts_with_insertion(self.insertion_line)

    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        return start_line < self.insertion_line <= end_line

    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        del insertion_line
        return False

    @staticmethod
    def _coalesced_same_anchor(
        insertions: tuple["SourceInsertion", ...],
    ) -> "SourceInsertion":
        first = insertions[0]
        unique_insertions: list[SourceInsertion] = []
        seen_sources: set[tuple[str, ...]] = set()
        for insertion in insertions:
            if insertion.inserted_lines in seen_sources:
                continue
            seen_sources.add(insertion.inserted_lines)
            unique_insertions.append(insertion)
        coalesced_lines = unique_insertions[0].inserted_lines
        for insertion in unique_insertions[1:]:
            coalesced_lines = insertion.leading_boundary.coalesce_lines(
                coalesced_lines,
                insertion.inserted_lines,
            )
        return replace(
            first,
            inserted_lines=coalesced_lines,
            rationale=_joined_rationales(
                insertion.rationale for insertion in insertions
            ),
            contributors=NominalSourceEdit.merged_contributors(insertions),
            origins=NominalSourceEdit.merged_origins(insertions),
        )


@dataclass(frozen=True, kw_only=True)
class SourceFileCreation(NominalSourceEdit):
    """Create one source path with an explicit initial source."""

    operation_type: type["RefactorRecipeOperation"]
    file_path: str
    source: str = ""

    @classmethod
    def from_operation(
        cls,
        operation: "RefactorRecipeOperation",
        *,
        requested_path: str,
        source_index: SourceIndex,
        source: str,
    ) -> "SourceFileCreation":
        file_path = SourceCreationPathAuthority.from_source_index(
            requested_path,
            source_index,
        ).required_path()
        return cls(
            operation_type=type(operation),
            file_path=file_path,
            source=source,
            rationale=operation.rationale_text(f"Create source file {file_path!r}."),
        )

    @property
    def operation_key(self) -> str:
        """Derive report identity from the operation declaration."""

        return self.operation_type.operation_key()

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        creations_by_path: dict[str, list[SourceFileCreation]] = defaultdict(list)
        for peer in peers:
            creation = cast(SourceFileCreation, peer)
            creations_by_path[creation.file_path].append(creation)
        duplicate_paths = tuple(
            sorted(
                file_path
                for file_path, creations in creations_by_path.items()
                if len(creations) > 1
            )
        )
        if duplicate_paths:
            raise ValueError(
                f"Source files require one creation authority: {duplicate_paths!r}"
            )
        return tuple(creations[0] for creations in creations_by_path.values())

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[NominalSourceEdit, ...]:
        virtual_source = context.sources_by_file_path[self.file_path]
        if virtual_source != self.source:
            raise ValueError(
                f"Virtual source for {self.file_path!r} disagrees with its creation"
            )
        return (
            SourceInsertion(
                file_path=self.file_path,
                insertion_line=1,
                inserted_lines=(),
                rationale=self.rationale or f"Create source file {self.file_path!r}.",
                contributors=self.contributors,
                origins=self.origins,
            ),
        )


@dataclass(frozen=True)
class SourceTextSpanReplacement(ReplacementSource):
    """Replacement of one character-offset span inside a source string."""

    start_offset: int
    end_offset: int

    @classmethod
    def from_offsets(
        cls,
        *,
        start_offset: int,
        end_offset: int,
        replacement_source: str,
    ) -> "SourceTextSpanReplacement":
        return cls(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )


@dataclass(frozen=True)
class SourceTextSpan:
    """Character-offset span over one source string."""

    start_offset: int
    end_offset: int

    @classmethod
    def from_offsets(cls, offsets: tuple[int, int]) -> "SourceTextSpan":
        start_offset, end_offset = offsets
        return cls(start_offset=start_offset, end_offset=end_offset)

    def source_text(self, source: str) -> str:
        return source[self.start_offset : self.end_offset]

    def contains_comment(self, source: str) -> bool:
        try:
            return any(
                token.type == tokenize.COMMENT
                for token in tokenize.generate_tokens(
                    io.StringIO(self.source_text(source)).readline
                )
            )
        except (IndentationError, tokenize.TokenError):
            return True

    def replacement(self, source: str, new_source: str) -> "SourceTextReplacement":
        return SourceTextReplacement(
            old_source=self.source_text(source),
            new_source=new_source,
        )


@dataclass(frozen=True)
class SourceTextReplacement(CodemodPayloadRecord):
    """One exact old/new source transformation."""

    old_source: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    new_source: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    def __post_init__(self) -> None:
        if not self.old_source:
            raise ValueError("Exact source replacement requires non-empty old_source")
        if self.old_source == self.new_source:
            raise ValueError("Exact source replacement must change its source")

    def apply_exactly_once(self, source: str, *, subject: str) -> str:
        """Apply this declared transformation only to one exact source surface."""

        match_offset = self.exact_match_offset(source, subject=subject)
        return (
            f"{source[:match_offset]}{self.new_source}"
            f"{source[match_offset + len(self.old_source):]}"
        )

    def exact_match_offset(self, source: str, *, subject: str) -> int:
        """Return the sole match offset or reject an unproved transformation."""

        match_count = 0
        match_offset = -1
        search_offset = 0
        while (candidate_offset := source.find(self.old_source, search_offset)) >= 0:
            if match_count == 0:
                match_offset = candidate_offset
            match_count += 1
            search_offset = candidate_offset + 1
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one match for source text in {subject!r}; "
                f"found {match_count}"
            )
        return match_offset


@dataclass(frozen=True, kw_only=True)
class SourceTextPatch:
    """Non-empty ordered exact transformations over one source surface."""

    replacements: tuple[SourceTextReplacement, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceTextReplacement)
    )

    def __post_init__(self) -> None:
        if not self.replacements:
            raise ValueError("Source text patch requires at least one replacement")

    def apply(self, source: str, *, subject: str) -> str:
        """Apply every exact transformation to the preceding result."""

        replacement_source = source
        for replacement in self.replacements:
            replacement_source = replacement.apply_exactly_once(
                replacement_source,
                subject=subject,
            )
        if replacement_source == source:
            raise ValueError("Source text patch leaves its source unchanged")
        return replacement_source


@dataclass(frozen=True)
class SourceNodeSpan:
    """AST statement span projected into source line coordinates."""

    node: ast.stmt
    decorator_policy: SourceNodeDecoratorPolicy = SourceNodeDecoratorPolicy.EXCLUDE

    @property
    def start_line(self) -> int:
        if self.decorator_policy.includes_decorators and isinstance(
            self.node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            decorator_lines = tuple(
                decorator.lineno for decorator in self.node.decorator_list
            )
            return min((*decorator_lines, self.node.lineno))
        return self.node.lineno

    @property
    def end_line(self) -> int:
        return self.node.end_lineno or self.node.lineno

    @property
    def line_span(self) -> "SourceLineSpan":
        return SourceLineSpan(start_line=self.start_line, end_line=self.end_line)


@dataclass(frozen=True)
class SourceTextGeometry(SourceLineSegmentAuthority):
    """Line and offset geometry for source-index anchored rewrites."""

    literal_node_types: ClassVar[tuple[type[ast.expr], ...]] = (
        ast.Constant, ast.JoinedStr,
        *((ast.TemplateStr,) if sys.version_info >= (3, 14) else ()),
    )

    @cached_property
    def tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        return tuple(tokenize.generate_tokens(io.StringIO(self.source).readline))

    @cached_property
    def line_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for line in self.lines:
            offsets.append(offset)
            offset += len(line)
        if not offsets:
            offsets.append(0)
        return tuple(offsets)

    @cached_property
    def end_offset(self) -> int:
        return sum(len(line) for line in self.lines)

    def token_position_offset(self, position: tuple[int, int]) -> int:
        line, column = position
        if line == len(self.line_offsets) + 1 and column == 0:
            return self.end_offset
        if not 1 <= line <= len(self.line_offsets):
            raise ValueError(f"Token position is outside source geometry: {position!r}")
        return self.line_offsets[line - 1] + column

    def byte_span_offsets(self, span: SourceByteSpan) -> tuple[int, int]:
        return span.character_offsets(self.lines, self.line_offsets)

    def span_contains_comment(self, span: SourceTextSpan) -> bool:
        return any(
            token.type == tokenize.COMMENT
            and span.start_offset
            <= self.token_position_offset(token.start)
            < span.end_offset
            for token in self.tokens
        )

    def indented_source(self, indentation: str) -> str:
        """Indent a Python block while preserving complete literal source spans."""

        module = ast.parse(self.source)
        if not module.body:
            raise ValueError("Replacement source block must contain a statement")
        continuation_lines = frozenset(
            line_number
            for node in ast.walk(module)
            if isinstance(node, self.literal_node_types)
            for line_number in range(
                node.lineno + 1, SourceByteSpan.require_node(node).end_line_index + 2
            )
        )
        return "".join(
            indentation + line
            if line_number not in continuation_lines and line.strip()
            else line
            for line_number, line in enumerate(self.lines, start=1)
        )

    def function_parameter_span(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SourceTextSpan:
        """Resolve the exact source between one function's parameter parentheses."""

        parentheses = self.function_parameter_parentheses(node)
        return SourceTextSpan(parentheses.start_offset + 1, parentheses.end_offset - 1)

    def function_signature_suffix_span(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SourceTextSpan:
        """Resolve parameters and return annotation through the suite colon."""

        parentheses = self.function_parameter_parentheses(node)
        annotation_end = (
            self.required_node_offsets(node.returns)[1]
            if node.returns is not None
            else parentheses.end_offset
        )
        function_end = self.required_node_offsets(node)[1]
        for token in self.tokens:
            start = self.token_position_offset(token.start)
            if (
                annotation_end <= start < function_end
                and token.type == tokenize.OP
                and token.string == ":"
            ):
                return SourceTextSpan(
                    parentheses.start_offset, self.token_position_offset(token.end)
                )
        raise ValueError(f"Cannot resolve signature colon for {node.name!r}")

    def function_parameter_parentheses(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> SourceTextSpan:
        """Resolve the parameter delimiters, after any generic type parameters."""

        function_start, function_end = self.byte_span_offsets(
            SourceByteSpan.require_node(node)
        )
        indexed_tokens = tuple(
            (
                token,
                self.token_position_offset(token.start),
                self.token_position_offset(token.end),
            )
            for token in self.tokens
            if token.type != tokenize.ENDMARKER
        )
        definition_index = next(
            (
                index
                for index, (token, start_offset, _end_offset) in enumerate(
                    indexed_tokens
                )
                if token.type == tokenize.NAME
                and token.string == "def"
                and function_start <= start_offset < function_end
            ),
            None,
        )
        if definition_index is None:
            raise ValueError(f"Cannot resolve parameter span for {node.name!r}")
        opening_offset = None
        depth = 0
        for token, start_offset, end_offset in indexed_tokens[definition_index + 1 :]:
            if end_offset > function_end:
                break
            if token.type != tokenize.OP:
                continue
            if token.string == "(" and depth == 0:
                opening_offset = start_offset
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
                if depth == 0 and opening_offset is not None:
                    return SourceTextSpan(
                        start_offset=opening_offset,
                        end_offset=end_offset,
                    )
        raise ValueError(f"Cannot resolve parameter closing for {node.name!r}")

    def node_span_offsets(self, span: SourceNodeSpan) -> tuple[int, int]:
        return self._line_span_offsets(span.start_line, span.end_line)

    def statement_deletion_span(self, span: "SourceLineSpan") -> "SourceLineSpan":
        """Include the separator owned by one deleted statement."""

        if not 1 <= span.start_line <= span.end_line <= len(self.lines):
            raise ValueError("Statement deletion span is outside source geometry")
        has_following_statement = any(
            line.strip() for line in self.lines[span.end_line :]
        )
        if has_following_statement:
            end_line = span.end_line
            while end_line < len(self.lines) and not self.lines[end_line].strip():
                end_line += 1
            return SourceLineSpan(span.start_line, end_line)

        start_line = span.start_line
        while start_line > 1 and not self.lines[start_line - 2].strip():
            start_line -= 1
        return SourceLineSpan(start_line, len(self.lines))

    def node_offsets(self, node: ast.AST) -> tuple[int, int] | None:
        span = SourceByteSpan.from_node(node)
        if span is None or not span.fits_lines(self.lines):
            return None
        return self.byte_span_offsets(span)

    def required_node_offsets(self, node: ast.AST) -> tuple[int, int]:
        offsets = self.node_offsets(node)
        if offsets is None:
            raise ValueError("AST node lacks source offsets")
        return offsets

    def target_span_offsets(self, target: AstTargetDigest) -> tuple[int, int]:
        start_offset = self.line_offsets[target.line - 1]
        end_offset = (
            self.line_offsets[target.end_line]
            if target.end_line < len(self.line_offsets)
            else self.end_offset
        )
        return start_offset, end_offset

    def target_source_with_replacements(
        self,
        target: AstTargetDigest,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> str:
        return self.source_with_replacements_in_span(
            *self.target_span_offsets(target),
            replacements,
        )

    def line_indent(self, offset: int) -> str:
        line_start = self.source.rfind("\n", 0, offset) + 1
        line_end = self.source.find("\n", offset)
        if line_end == -1:
            line_end = len(self.source)
        line = self.source[line_start:line_end]
        return line[: len(line) - len(line.lstrip())]

    def line_prefix(self, offset: int) -> str:
        line_start = self.source.rfind("\n", 0, offset) + 1
        return self.source[line_start:offset]

    def source_with_replacements_in_span(
        self,
        span_start: int,
        span_end: int,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> str:
        span_source = self.source[span_start:span_end]
        for replacement in reversed(
            self.replacements_in_span(span_start, span_end, replacements)
        ):
            relative_start = replacement.start_offset - span_start
            relative_end = replacement.end_offset - span_start
            span_source = (
                f"{span_source[:relative_start]}"
                f"{replacement.replacement_source}"
                f"{span_source[relative_end:]}"
            )
        return span_source

    def physical_edits(
        self,
        *,
        file_path: str,
        replacements: Iterable[SourceTextSpanReplacement],
        rationale: str = "",
    ) -> tuple[PhysicalSourceEdit, ...]:
        """Project offset edits into the smallest independent line edits."""

        ordered = self.replacements_in_span(0, self.end_offset, replacements)
        line_windows: list[tuple[int, int, list[SourceTextSpanReplacement]]] = []
        insertions: list[SourceInsertion] = []
        for replacement in ordered:
            insertion_line = self._line_start_insertion_line(replacement)
            if insertion_line is not None:
                insertions.append(
                    SourceInsertion(
                        file_path=file_path,
                        insertion_line=insertion_line,
                        inserted_lines=SourceTargetEditor.source_lines(
                            replacement.replacement_source
                        ),
                        rationale=rationale,
                    )
                )
                continue
            start_line = self.line_number_for_offset(replacement.start_offset)
            end_line = self.line_number_for_offset(
                max(replacement.start_offset, replacement.end_offset - 1)
            )
            if line_windows and start_line <= line_windows[-1][1]:
                previous_start, previous_end, previous_replacements = line_windows[-1]
                line_windows[-1] = (
                    previous_start,
                    max(previous_end, end_line),
                    [*previous_replacements, replacement],
                )
                continue
            line_windows.append((start_line, end_line, [replacement]))

        span_replacements = tuple(
            SourceSpanEdit.from_replacement_lines(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                replacement_lines=SourceTargetEditor.source_lines(
                    self.source_with_replacements_in_span(
                        *self._line_span_offsets(start_line, end_line),
                        window_replacements,
                    )
                ),
                rationale=rationale,
            )
            for start_line, end_line, window_replacements in line_windows
        )
        return (*span_replacements, *insertions)

    def _line_start_insertion_line(
        self,
        replacement: SourceTextSpanReplacement,
    ) -> int | None:
        if replacement.start_offset != replacement.end_offset:
            return None
        for line_index, line_offset in enumerate(self.line_offsets):
            if replacement.start_offset == line_offset:
                return line_index + 1
        if replacement.start_offset == self.end_offset:
            return len(self.lines) + 1
        return None

    def line_number_for_offset(self, offset: int) -> int:
        line_number = 1
        for candidate_line, line_offset in enumerate(self.line_offsets, start=1):
            if line_offset > offset:
                break
            line_number = candidate_line
        return line_number

    def replacements_in_span(
        self,
        span_start: int,
        span_end: int,
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """Return one unambiguous replacement per offset span."""

        if not 0 <= span_start <= span_end <= self.end_offset:
            raise ValueError(
                "Replacement target span must fit the source geometry: "
                f"{span_start}:{span_end}"
            )
        replacement_by_span: dict[SourceTextSpan, SourceTextSpanReplacement] = {}
        for replacement in replacements:
            if not (
                span_start
                <= replacement.start_offset
                <= replacement.end_offset
                <= span_end
            ):
                raise ValueError(
                    "Offset replacement must fit its target span: "
                    f"{replacement.start_offset}:{replacement.end_offset} "
                    f"outside {span_start}:{span_end}"
                )
            replacement_span = SourceTextSpan(
                start_offset=replacement.start_offset,
                end_offset=replacement.end_offset,
            )
            existing = replacement_by_span.get(replacement_span)
            if existing is None:
                replacement_by_span[replacement_span] = replacement
                continue
            if existing.replacement_source != replacement.replacement_source:
                raise ValueError(
                    "Offset replacements assign different source to the same span: "
                    f"{replacement.start_offset}:{replacement.end_offset}"
                )

        ordered = sorted_tuple(
            replacement_by_span.values(),
            key=lambda item: (item.start_offset, item.end_offset),
        )
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if second.start_offset > first.end_offset:
                    break
                if self.replacement_spans_overlap(first, second):
                    raise ValueError(
                        "Offset replacement spans overlap: "
                        f"{first.start_offset}:{first.end_offset} and "
                        f"{second.start_offset}:{second.end_offset}"
                    )
        return ordered

    @staticmethod
    def replacement_spans_overlap(
        first: SourceTextSpanReplacement,
        second: SourceTextSpanReplacement,
    ) -> bool:
        if first.start_offset == first.end_offset:
            return second.start_offset < first.start_offset < second.end_offset
        if second.start_offset == second.end_offset:
            return first.start_offset < second.start_offset < first.end_offset
        return (
            first.start_offset < second.end_offset
            and second.start_offset < first.end_offset
        )

    def _line_span_offsets(self, start_line: int, end_line: int) -> tuple[int, int]:
        line_offsets = self.line_offsets
        end_offset = (
            line_offsets[end_line] if end_line < len(line_offsets) else self.end_offset
        )
        return line_offsets[start_line - 1], end_offset


@dataclass(frozen=True)
class SourceTargetEditor:
    """Line-oriented editor for one source-index target span."""

    sources: Mapping[str, str]
    target: AstTargetDigest

    @property
    def file_lines(self) -> list[str]:
        return self.sources[self.target.file_path].splitlines(keepends=True)

    @property
    def target_lines(self) -> list[str]:
        return self.file_lines[self.target.line - 1 : self.target.end_line]

    def replacement_source(
        self,
        replacements: Iterable[PhysicalSourceEdit],
    ) -> str:
        lines = self.target_lines
        ordered_replacements = self._ordered_replacements(replacements)
        for replacement in reversed(ordered_replacements):
            start_index = replacement.start_line - self.target.line
            end_index = replacement.end_line - self.target.line + 1
            lines[start_index:end_index] = list(replacement.replacement_lines)
        return "".join(lines)

    def exact_text_replacement(
        self,
        replacement: SourceTextReplacement,
        *,
        rationale: str = "",
    ) -> SourceSpanEdit:
        target_source = "".join(self.target_lines)
        start_offset = replacement.exact_match_offset(
            target_source, subject=self.target.qualname
        )
        end_offset = start_offset + len(replacement.old_source)
        target_line_offsets = SourceTextGeometry(target_source).line_offsets
        start_index = self._line_index_for_offset(start_offset, target_line_offsets)
        end_index = self._line_index_for_offset(
            max(start_offset, end_offset - 1),
            target_line_offsets,
        )
        span_lines = self.target_lines[start_index : end_index + 1]
        span_source = "".join(span_lines)
        relative_start = start_offset - target_line_offsets[start_index]
        relative_end = end_offset - target_line_offsets[start_index]
        replacement_source = (
            f"{span_source[:relative_start]}{replacement.new_source}"
            f"{span_source[relative_end:]}"
        )
        return SourceSpanEdit.from_replacement_lines(
            file_path=self.target.file_path,
            start_line=self.target.line + start_index,
            end_line=self.target.line + end_index,
            replacement_lines=SourceTargetEditor.source_lines(replacement_source),
            rationale=rationale
            or f"Replace source text inside {self.target.qualname!r}.",
        )

    def exact_text_patch(
        self,
        patch: SourceTextPatch,
        *,
        rationale: str = "",
    ) -> PhysicalSourceEdit:
        """Apply ordered exact transformations as one target-level rewrite."""

        target_source = "".join(self.target_lines)
        replacement_source = patch.apply(target_source, subject=self.target.qualname)
        return self.minimal_replacement_edit(
            replacement_source,
            rationale=rationale
            or f"Patch exact source text inside {self.target.qualname!r}.",
        )

    def minimal_replacement_edit(
        self,
        replacement_source: str,
        *,
        rationale: str = "",
    ) -> PhysicalSourceEdit:
        """Compile changed target source to its smallest enclosing line edit."""

        current_lines = tuple(self.target_lines)
        if replacement_source == "".join(current_lines):
            raise ValueError("Target replacement leaves its source unchanged")
        replacement_lines = self.source_lines(replacement_source)
        prefix_count = 0
        for current_line, replacement_line in zip(current_lines, replacement_lines):
            if current_line != replacement_line:
                break
            prefix_count += 1
        suffix_count = 0
        unmatched_count = min(len(current_lines), len(replacement_lines)) - prefix_count
        while (
            suffix_count < unmatched_count
            and current_lines[-suffix_count - 1] == replacement_lines[-suffix_count - 1]
        ):
            suffix_count += 1
        replacement_end = len(replacement_lines) - suffix_count
        return SourceLineSpan(
            start_line=self.target.line + prefix_count,
            end_line=self.target.line + len(current_lines) - suffix_count - 1,
        ).line_replacement(
            file_path=self.target.file_path,
            replacement_lines=replacement_lines[prefix_count:replacement_end],
            rationale=rationale,
        )

    def _ordered_replacements(
        self,
        replacements: Iterable[PhysicalSourceEdit],
    ) -> tuple[PhysicalSourceEdit, ...]:
        ordered_replacements = sorted_tuple(
            replacements,
            key=lambda item: (item.start_line, item.end_line),
        )
        previous_end = self.target.line - 1
        for replacement in ordered_replacements:
            if replacement.file_path != self.target.file_path:
                raise ValueError(
                    f"Replacement file {replacement.file_path!r} does not match "
                    f"target file {self.target.file_path!r}"
                )
            if (
                replacement.start_line < self.target.line
                or replacement.end_line > self.target.end_line
            ):
                raise ValueError(
                    f"Replacement {replacement.start_line}:{replacement.end_line} "
                    f"is outside target {self.target.qualname!r}"
                )
            if replacement.start_line <= previous_end:
                raise ValueError(
                    f"Overlapping line replacements in {self.target.file_path!r} "
                    f"at line {replacement.start_line}"
                )
            previous_end = replacement.end_line
        return ordered_replacements

    def indentation_for_line(self, line_number: int) -> str:
        line = self.file_lines[line_number - 1]
        return line[: len(line) - len(line.lstrip())]

    @staticmethod
    def source_lines(source: str) -> tuple[str, ...]:
        if source and not source.endswith(("\n", "\r")):
            source = f"{source}\n"
        return tuple(source.splitlines(keepends=True))

    @staticmethod
    def _line_index_for_offset(offset: int, line_offsets: tuple[int, ...]) -> int:
        index = 0
        for candidate_index, line_offset in enumerate(line_offsets):
            if line_offset > offset:
                break
            index = candidate_index
        return index


@dataclass(frozen=True)
class SourceLineSpan:
    start_line: int
    end_line: int

    @classmethod
    def from_offsets(
        cls,
        geometry: SourceTextGeometry,
        start_offset: int,
        end_offset: int,
    ) -> Self:
        return cls(
            start_line=cls.line_number_for_offset(geometry, start_offset),
            end_line=cls.line_number_for_offset(
                geometry,
                max(start_offset, end_offset - 1),
            ),
        )

    @staticmethod
    def line_number_for_offset(
        geometry: SourceTextGeometry,
        offset: int,
    ) -> int:
        line_number = 1
        for index, line_offset in enumerate(geometry.line_offsets):
            if line_offset > offset:
                break
            line_number = index + 1
        return line_number

    def overlaps(self, other: "SourceLineSpan") -> bool:
        return self.start_line <= other.end_line and other.start_line <= self.end_line

    def overlaps_any(self, spans: Iterable["SourceLineSpan"]) -> bool:
        return any(self.overlaps(span) for span in spans)

    def source_from(self, source: str) -> str:
        source_lines = source.splitlines(keepends=True)
        return "".join(source_lines[self.start_line - 1 : self.end_line])

    def line_replacement(
        self,
        *,
        file_path: str,
        replacement_lines: tuple[str, ...],
        rationale: str = "",
    ) -> PhysicalSourceEdit:
        if self.start_line > self.end_line:
            return SourceInsertion(
                file_path=file_path,
                insertion_line=self.start_line,
                inserted_lines=replacement_lines,
                rationale=rationale,
            )
        return SourceSpanEdit.from_replacement_lines(
            file_path=file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            replacement_lines=replacement_lines,
            rationale=rationale,
        )

    def line_deletion(
        self,
        *,
        file_path: str,
        rationale: str = "",
    ) -> SourceSpanDeletion:
        return SourceSpanDeletion(
            file_path=file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            rationale=rationale,
        )


def _joined_rationales(rationales: Iterable[str]) -> str:
    unique_rationales = tuple(dict.fromkeys(item for item in rationales if item))
    return " ".join(unique_rationales)


@dataclass(frozen=True, kw_only=True)
class SourceRewriteDelta(ReplacementSource):
    """Replacement source shared by planned and simulated target rewrites."""

    operation: ClassVar[RewriteOperation] = RewriteOperation.REPLACE_TARGET
    rationale: str = ""
    contributors: tuple[SourceRewriteContributor, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlannedSourceRewrite(SourceRewriteDelta):
    """One planned source rewrite against an AST target digest."""

    target_id: str


@dataclass(frozen=True, kw_only=True)
class SimulatedSourceRewrite(
    SourceTargetSpan,
    SourceRewriteDelta,
    DataclassJsonReport,
):
    """Resolved source span and replacement preview for one planned rewrite."""

    replacement_source: str = json_report_field(included=False)
    original_source: str = json_report_field(included=False)

    @json_report_property(field_name="operation")
    def report_operation(self) -> RewriteOperation:
        return self.operation


@dataclass(frozen=True)
class ResolvedSourceRewrite:
    """Planned rewrite paired with its source-index target geometry."""

    rewrite: PlannedSourceRewrite
    target: AstTargetDigest


class PlannedRewriteConflictError(ValueError):
    """Two non-equivalent planned rewrites claim overlapping source geometry."""

    def __init__(
        self,
        first: ResolvedSourceRewrite,
        second: ResolvedSourceRewrite,
    ) -> None:
        self.first = first
        self.second = second
        super().__init__(
            "Conflicting planned rewrites overlap in "
            f"{first.target.file_path!r}: {first.target.target_id!r} and "
            f"{second.target.target_id!r}"
        )


@dataclass(frozen=True)
class PlannedRewriteSelectionAuthority:
    """Prove a rewrite batch is exact-deduplicated and conflict free."""

    source_index: SourceIndex

    def resolved_rewrites(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[ResolvedSourceRewrite, ...]:
        resolved = tuple(
            ResolvedSourceRewrite(
                rewrite=rewrite,
                target=self.required_target(rewrite),
            )
            for rewrite in self.coalesced_exact_rewrites(rewrites)
        )
        ordered = sorted_tuple(resolved, key=self.resolved_sort_key)
        self.require_disjoint(ordered)
        return ordered

    @staticmethod
    def coalesced_exact_rewrites(
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[PlannedSourceRewrite, ...]:
        rewrites_by_edit: dict[tuple[str, str], PlannedSourceRewrite] = {}
        for rewrite in rewrites:
            edit_key = (
                rewrite.target_id,
                rewrite.replacement_source,
            )
            existing = rewrites_by_edit.get(edit_key)
            if existing is None:
                rewrites_by_edit[edit_key] = rewrite
                continue
            rewrites_by_edit[edit_key] = replace(
                existing,
                rationale=_joined_rationales((existing.rationale, rewrite.rationale)),
                contributors=SourceRewriteContributor.merge(
                    existing.contributors,
                    rewrite.contributors,
                ),
            )
        return tuple(rewrites_by_edit.values())

    def select(
        self,
        rewrites: Iterable[PlannedSourceRewrite],
    ) -> tuple[PlannedSourceRewrite, ...]:
        return tuple(item.rewrite for item in self.resolved_rewrites(rewrites))

    def required_target(self, rewrite: PlannedSourceRewrite) -> AstTargetDigest:
        target = self.source_index.target_by_id.get(rewrite.target_id)
        if target is None:
            raise KeyError(f"Unknown source-index target id: {rewrite.target_id}")
        return target

    @staticmethod
    def resolved_sort_key(
        item: ResolvedSourceRewrite,
    ) -> tuple[str, int, int, str]:
        return (
            item.target.file_path,
            item.target.line,
            -item.target.end_line,
            item.target.qualname,
        )

    @classmethod
    def require_disjoint(
        cls,
        rewrites: tuple[ResolvedSourceRewrite, ...],
    ) -> None:
        previous: ResolvedSourceRewrite | None = None
        for rewrite in rewrites:
            if previous is not None and cls.overlaps(previous.target, rewrite.target):
                raise PlannedRewriteConflictError(previous, rewrite)
            previous = rewrite

    @staticmethod
    def overlaps(first: AstTargetDigest, second: AstTargetDigest) -> bool:
        return (
            first.file_path == second.file_path
            and first.line <= second.end_line
            and second.line <= first.end_line
        )


@dataclass(frozen=True)
class CodemodSourceRevision(DataclassJsonReport):
    """Full-source revision required before one simulated file write."""

    file_path: str
    source_hash: str | None

    @classmethod
    def from_sources(
        cls,
        file_path: str,
        sources_by_file_path: Mapping[str, str],
    ) -> "CodemodSourceRevision":
        source = sources_by_file_path.get(file_path)
        return cls(
            file_path=file_path,
            source_hash=(cls.hash_source(source) if source is not None else None),
        )

    @staticmethod
    def hash_source(source: str) -> str:
        return hashlib.blake2s(
            source.encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    def matches_source(self, source: str | None) -> bool:
        if source is None:
            return self.source_hash is None
        return self.source_hash == self.hash_source(source)

    def require_path_state(
        self,
        path: Path | None = None,
        *,
        encoding: str = "utf-8",
    ) -> None:
        source_path = Path(self.file_path) if path is None else path
        if not source_path.exists():
            current_source = None
        elif source_path.is_file():
            current_source = source_path.read_text(encoding=encoding)
        else:
            raise CodemodSourceRevisionError(
                f"Codemod source path is not a file: {source_path}"
            )
        if not self.matches_source(current_source):
            raise CodemodSourceRevisionError(
                f"Codemod source changed after simulation: {self.file_path}"
            )


class CodemodSourceRevisionError(ValueError):
    """Raised when codemod source no longer matches a required revision."""
