"""Physical source-edit algebra for codemod execution."""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from bisect import (
    bisect_left,
    bisect_right,
)
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import (
    dataclass,
    replace,
)
from enum import StrEnum
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import (
    ClassVar,
    Self,
    TYPE_CHECKING,
    TypeVar,
    cast,
)

from .codemod_paths import SourceCreationPathAuthority as SourceCreationPathAuthority
from .codemod_payload import (
    CodemodPayloadRecord,
    DataclassPayloadProjection,
    EmptyDefaultStringPayloadValueCodec,
    OptionalStringPayloadValueCodec,
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
    read_source_text,
)
from .source_identity import python_source_cache_signature
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

    def validate_replacement(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Require authored decorators to belong to the selected source region."""
        if node.decorator_list and not self.includes_decorators:
            raise ValueError(
                "Replacement decorators require a decorator-inclusive source region"
            )

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


_SourcePeerInput = TypeVar("_SourcePeerInput")


@dataclass(frozen=True, kw_only=True)
class NominalSourceEdit(ABC):
    """Declaration-owned semantic source edit emitted by recipe operations."""

    rationale: str = ""
    contributors: tuple[SourceRewriteContributor, ...] = ()
    origins: tuple[SourceEditOrigin, ...] = ()

    @classmethod
    def projected_peers(
        cls,
        inputs: Iterable[_SourcePeerInput],
        declaration: Callable[[_SourcePeerInput], NominalSourceEdit],
    ) -> Iterator[tuple[_SourcePeerInput, Self]]:
        """Admit each actual input under this exact nominal declaration once."""
        for item in inputs:
            peer = declaration(item)
            if type(peer) is not cls:
                raise ValueError(
                    "Source peer groups require the exact nominal declaration"
                )
            yield item, cast(Self, peer)

    def resolved_peer_windows(
        self, peers: tuple[NominalSourceEdit, ...], context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        """Preserve nominal peer coalescence before resolving each output window."""
        return tuple(
            window
            for edit in self.coalesced_with_peers(peers, context)
            for window in edit.resolved_windows(context)
        )

    @abstractmethod
    def resolved_windows(
        self, context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        """Resolve this declaration into source windows under the supplied revision."""

    @classmethod
    def declaration_groups(
        cls,
        inputs: Iterable[_SourcePeerInput],
        declaration: Callable[[_SourcePeerInput], NominalSourceEdit],
    ) -> tuple[tuple[_SourcePeerInput, ...], ...]:
        """Group actual input occurrences by projected nominal owner in encounter order."""
        groups: dict[type[NominalSourceEdit], list[_SourcePeerInput]] = {}
        for item in inputs:
            groups.setdefault(type(declaration(item)), []).append(item)
        return tuple(tuple(group) for group in groups.values())

    def with_evidence_from(self, peers: Iterable[NominalSourceEdit]) -> Self:
        """Derive merged evidence from this source edit's actual input declarations."""
        peers = tuple(peers)
        return replace(
            self,
            rationale=_joined_rationales(peer.rationale for peer in peers),
            contributors=self.merged_contributors(peers),
            origins=self.merged_origins(peers),
        )

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

    def resolved_edits(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple["PhysicalSourceEdit", ...]:
        """Derive physical output from this declaration's authoritative source windows."""
        return tuple(window.physical_edit for window in self.resolved_windows(context))

    @classmethod
    def coalesced_by_declaration(
        cls,
        edits: Iterable["NominalSourceEdit"],
        context: "CodemodSelectorContext",
    ) -> tuple["NominalSourceEdit", ...]:
        return tuple(
            coalesced
            for group in cls.declaration_groups(edits, lambda edit: edit)
            for coalesced in group[0].coalesced_with_peers(group, context)
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


class SourceEditWindowABC(ABC):
    """Physical output and original-text interiors of one resolved source window."""

    @property
    @abstractmethod
    def physical_edit(self) -> PhysicalSourceEdit:
        """Derive the physical declaration consumed by the existing renderer."""

    def project_retained_interiors(
        self, spans: Iterable[SourceTextSpan]
    ) -> tuple[SourceTextSpan, ...]:
        """Map original absolute interiors to output-window-relative offsets.

        This is source origin only, not syntax or execution equivalence.
        Empty input has no retained-text obligation.
        """
        spans = tuple(spans)
        return self._project_retained_interiors(spans) if spans else ()

    @abstractmethod
    def _project_retained_interiors(
        self, spans: tuple[SourceTextSpan, ...]
    ) -> tuple[SourceTextSpan, ...]:
        """Prove and project a nonempty batch of original source interiors."""


@dataclass(frozen=True, kw_only=True)
class PhysicalSourceEdit(SourceEditWindowABC, NominalSourceEdit, ABC):
    """Semantic edit whose absolute source-line geometry is resolved."""

    file_path: str

    def resolved_peer_windows(
        self, peers: tuple[NominalSourceEdit, ...], context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        return self.coalesced_window_peers(
            cast(tuple[PhysicalSourceEdit, ...], peers), context
        )

    @classmethod
    def coalesced_window_peers(
        cls, windows: tuple[SourceEditWindowABC, ...], context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        """Retain actual window groups through the existing physical merge law."""
        del context
        return tuple(
            CoalescedSourceWindow(group)
            for group in cls.projected_peer_groups(
                windows, lambda window: window.physical_edit
            )
        )

    @classmethod
    @abstractmethod
    def coalesced_group(cls, peers: tuple[Self, ...]) -> Self:
        """Validate and derive one actual physical peer group's merged declaration."""

    @classmethod
    @abstractmethod
    def projected_peer_groups(
        cls,
        inputs: Iterable[_SourcePeerInput],
        declaration: Callable[[_SourcePeerInput], NominalSourceEdit],
    ) -> tuple[tuple[_SourcePeerInput, ...], ...]:
        """Group actual inputs under this physical declaration's coalescence law."""

    @abstractmethod
    def original_span(self, geometry: SourceTextGeometry) -> SourceTextSpan:
        """Project this physical declaration's original character interval."""

    def _project_retained_interiors(
        self, spans: tuple[SourceTextSpan, ...]
    ) -> tuple[SourceTextSpan, ...]:
        raise ValueError(
            "Opaque source window has no unchanged interior correspondence"
        )

    @property
    def physical_edit(self) -> PhysicalSourceEdit:
        return self

    def resolved_windows(
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


@dataclass(frozen=True)
class CoalescedSourceWindow(SourceEditWindowABC):
    """One actual peer group, with derived output and intersected text evidence."""

    windows: tuple[SourceEditWindowABC, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("Coalesced source window requires actual input windows")
        self.physical_edit

    @cached_property
    def physical_edit(self) -> PhysicalSourceEdit:
        peers = tuple(window.physical_edit for window in self.windows)
        return peers[0].coalesced_group(peers)

    def _project_retained_interiors(
        self, spans: tuple[SourceTextSpan, ...]
    ) -> tuple[SourceTextSpan, ...]:
        projected = self.windows[0].project_retained_interiors(spans)
        for window in self.windows[1:]:
            if window.project_retained_interiors(spans) != projected:
                raise ValueError(
                    "Coalesced source windows disagree on retained interior positions"
                )
        return projected


@dataclass(frozen=True, kw_only=True)
class SourceSpanEdit(PhysicalSourceEdit, ABC):
    """Physical edit over one non-empty absolute line span."""

    start_line: int
    end_line: int

    def original_span(self, geometry: SourceTextGeometry) -> SourceTextSpan:
        return SourceTextSpan(
            *geometry.line_span_offsets(self.start_line, self.end_line)
        )

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


class KeyedSourceEdit(NominalSourceEdit, ABC):
    """Actual nominal peers grouped by their declaration-owned source identity."""

    @classmethod
    def projected_peer_groups(
        cls,
        inputs: Iterable[_SourcePeerInput],
        declaration: Callable[[_SourcePeerInput], NominalSourceEdit],
    ) -> tuple[tuple[_SourcePeerInput, ...], ...]:
        """Retain actual inputs under this declaration's single keyed grouping law."""
        groups: dict[Hashable, list[_SourcePeerInput]] = {}
        for item, peer in cls.projected_peers(inputs, declaration):
            groups.setdefault(peer.coalescence_key, []).append(item)
        return tuple(tuple(group) for group in groups.values())

    @property
    @abstractmethod
    def coalescence_key(self) -> Hashable:
        """Identity of the source location whose peers this declaration groups."""

    @classmethod
    def peer_groups(cls, peers: Iterable[Self]) -> tuple[tuple[Self, ...], ...]:
        """Keep actual peers owned by this exact declaration, in encounter order."""
        return cls.projected_peer_groups(peers, lambda peer: peer)


class KeyedSourceEditCoalescence(KeyedSourceEdit, ABC):
    """Retain actual keyed peers and derive their declaration-owned merge."""

    @classmethod
    def _coalesced_owned_group(cls, peers: tuple[Self, ...]) -> Self:
        """Construct one output from an already-validated actual peer group."""
        return cls._coalesced_group(peers).with_evidence_from(peers)

    @staticmethod
    @abstractmethod
    def _coalesced_group(peers: tuple[Self, ...]) -> Self:
        """Derive the declaration payload from one actual peer group."""

    @classmethod
    def coalesced_group(cls, peers: tuple[Self, ...]) -> Self:
        """Merge payload and evidence from one group of this declaration's peers."""
        groups = cls.peer_groups(peers)
        if len(groups) != 1:
            raise ValueError("Coalescence requires one nonempty source peer group")
        return cls._coalesced_owned_group(groups[0])

    def coalesced_with_peers(
        self, peers: tuple[NominalSourceEdit, ...], context: "CodemodSelectorContext"
    ) -> tuple[NominalSourceEdit, ...]:
        del context
        return tuple(
            self._coalesced_owned_group(group)
            for group in self.peer_groups(cast(tuple[Self, ...], peers))
        )


@dataclass(frozen=True, kw_only=True)
class SourceSpanReplacement(KeyedSourceEditCoalescence, SourceSpanEdit):
    """Replace one non-empty absolute line span with explicit source lines."""

    replacement_lines: tuple[str, ...]

    @property
    def coalescence_key(self) -> Hashable:
        return self.file_path, self.start_line, self.end_line

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.replacement_lines:
            raise ValueError(
                "Source span replacements require replacement lines; "
                "use SourceSpanDeletion to remove source"
            )

    @staticmethod
    def _coalesced_group(
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
        return first


@dataclass(frozen=True, kw_only=True)
class SourceSpanDeletion(SourceSpanEdit):
    """Delete one non-empty absolute line span."""

    @classmethod
    def coalesced_group(cls, peers: tuple[Self, ...]) -> Self:
        groups = cls.projected_peer_groups(peers, lambda peer: peer)
        if len(groups) != 1:
            raise ValueError("Coalescence requires one nonempty source peer group")
        merged = groups[0][0]
        for peer in groups[0][1:]:
            merged = replace(
                merged, end_line=max(merged.end_line, peer.end_line)
            ).with_evidence_from((merged, peer))
        return merged

    @classmethod
    def projected_peer_groups(
        cls,
        inputs: Iterable[_SourcePeerInput],
        declaration: Callable[[_SourcePeerInput], NominalSourceEdit],
    ) -> tuple[tuple[_SourcePeerInput, ...], ...]:
        """Retain actual deletion occurrences under the existing sorted overlap union."""
        projected = sorted(
            cls.projected_peers(inputs, declaration),
            key=lambda pair: (pair[1].file_path, pair[1].start_line, pair[1].end_line),
        )
        groups: list[list[_SourcePeerInput]] = []
        last_path = None
        end_line = 0
        for item, peer in projected:
            if groups and peer.file_path == last_path and peer.start_line <= end_line:
                groups[-1].append(item)
                end_line = max(end_line, peer.end_line)
            else:
                groups.append([item])
                last_path, end_line = peer.file_path, peer.end_line
        return tuple(tuple(group) for group in groups)

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
            SourceTextGeometry(
                context.sources_by_file_path[target_digest.file_path]
            ).node_line_span(
                SourceNodeSpan(
                    target_node,
                    SourceNodeDecoratorPolicy.INCLUDE,
                )
            )
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
            statement_span=SourceTextGeometry(source).node_line_span(
                SourceNodeSpan(
                    statement,
                    SourceNodeDecoratorPolicy.INCLUDE,
                )
            ),
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
        return tuple(
            window.physical_edit
            for window in self.resolved_peer_windows(peers, context)
        )


@dataclass(frozen=True, kw_only=True)
class SourceInsertion(KeyedSourceEditCoalescence, PhysicalSourceEdit):
    """Insert source at one absolute line anchor."""

    insertion_line: int
    inserted_lines: tuple[str, ...] = ()
    leading_boundary: SourceInsertionBoundary = SourceInsertionBoundary.PRESERVE

    def original_span(self, geometry: SourceTextGeometry) -> SourceTextSpan:
        offset = geometry.line_anchor_offset(self.insertion_line)
        return SourceTextSpan(offset, offset)

    @property
    def coalescence_key(self) -> Hashable:
        return self.file_path, self.insertion_line

    @property
    def start_line(self) -> int:
        return self.insertion_line

    @property
    def end_line(self) -> int:
        return self.insertion_line - 1

    @property
    def replacement_lines(self) -> tuple[str, ...]:
        return self.inserted_lines

    def conflicts_with(self, other: PhysicalSourceEdit) -> bool:
        return other.conflicts_with_insertion(self.insertion_line)

    def conflicts_with_span(self, start_line: int, end_line: int) -> bool:
        return start_line < self.insertion_line <= end_line

    def conflicts_with_insertion(self, insertion_line: int) -> bool:
        del insertion_line
        return False

    @staticmethod
    def _coalesced_group(
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
                coalesced_lines, insertion.inserted_lines
            )
        return replace(first, inserted_lines=coalesced_lines)


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

    def resolved_windows(
        self,
        context: "CodemodSelectorContext",
    ) -> tuple[PhysicalSourceEdit, ...]:
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
class SourceOffsetSpan:
    """Character interval shared independently of each source object's factory."""

    start_offset: int
    end_offset: int

    def is_within(self, other: SourceOffsetSpan) -> bool:
        return (
            other.start_offset
            <= self.start_offset
            <= self.end_offset
            <= other.end_offset
        )

    def overlaps(self, other: SourceOffsetSpan) -> bool:
        """Retain interior insertions while excluding touching endpoints."""
        return (
            self.start_offset < other.end_offset
            and other.start_offset < self.end_offset
        )


@dataclass(frozen=True)
class SourceTextSpan(SourceOffsetSpan):
    """Character-offset span over one source string."""

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
class SourceTextSpanReplacement(ReplacementSource, SourceOffsetSpan):
    """Replacement of one character-offset span inside a source string."""

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


@dataclass(frozen=True, kw_only=True)
class SourceTextMutation(KeyedSourceEdit):
    """Exact edits of one source revision, lowered together after semantic planning."""

    revision: CodemodSourceRevision
    replacements: tuple[SourceTextSpanReplacement, ...]

    def resolved_windows(
        self, context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        return self.resolved_peer_windows((self,), context)

    def resolved_peer_windows(
        self, peers: tuple[NominalSourceEdit, ...], context: CodemodSelectorContext
    ) -> tuple[SourceEditWindowABC, ...]:
        return tuple(
            window
            for resolution in self.peer_resolutions(
                cast(tuple[Self, ...], peers), context
            )
            for window in resolution.windows
        )

    @property
    def coalescence_key(self) -> Hashable:
        return self.revision.file_path

    @classmethod
    def peer_resolutions(
        cls, peers: Iterable[Self], context: CodemodSelectorContext
    ) -> tuple[ExactSourceEditResolution, ...]:
        """Bind actual nominal peer groups to the source revision they edit."""
        return tuple(
            ExactSourceEditResolution(
                group,
                SourceTextGeometry(
                    context.sources_by_file_path[group[0].revision.file_path]
                ),
            )
            for group in cls.peer_groups(peers)
        )

    def coalesced_with_peers(
        self,
        peers: tuple[NominalSourceEdit, ...],
        context: CodemodSelectorContext,
    ) -> tuple[NominalSourceEdit, ...]:
        return tuple(
            window.physical_edit
            for window in self.resolved_peer_windows(peers, context)
        )


@dataclass(frozen=True)
class ExactSourceEditResolution:
    """Actual same-revision mutation peers and their once-derived exact windows."""

    mutations: tuple[SourceTextMutation, ...]
    geometry: SourceTextGeometry

    def __post_init__(self) -> None:
        if not self.mutations:
            raise ValueError("Exact source resolution requires nonempty mutation peers")
        first = self.mutations[0]
        if not isinstance(first, SourceTextMutation) or any(
            type(peer) is not type(first) for peer in self.mutations
        ):
            raise ValueError(
                "Exact source resolution requires one nominal mutation declaration"
            )
        if any(peer.revision != first.revision for peer in self.mutations):
            raise CodemodSourceRevisionError(
                "Exact edits require one original source revision"
            )
        if not first.revision.matches_source(self.geometry.source):
            raise CodemodSourceRevisionError(
                "Exact edits no longer match their original source revision"
            )

    @cached_property
    def replacement_inputs(
        self,
    ) -> dict[
        SourceTextSpanReplacement,
        tuple[tuple[SourceTextMutation, SourceTextSpanReplacement], ...],
    ]:
        """Retain every actual occurrence, including equal-but-distinct replacements."""
        inputs: dict[
            SourceTextSpanReplacement,
            list[tuple[SourceTextMutation, SourceTextSpanReplacement]],
        ] = defaultdict(list)
        for mutation in self.mutations:
            for replacement in mutation.replacements:
                inputs[replacement].append((mutation, replacement))
        return {
            replacement: tuple(occurrences)
            for replacement, occurrences in inputs.items()
        }

    @cached_property
    def projections(
        self,
    ) -> tuple[tuple[PhysicalSourceEdit, tuple[SourceTextSpanReplacement, ...]], ...]:
        return self.geometry.physical_edit_projections(
            file_path=self.mutations[0].revision.file_path,
            replacements=self.replacement_inputs,
        )

    @cached_property
    def windows(self) -> tuple[ExactSourceWindow, ...]:
        return tuple(
            ExactSourceWindow(self, index) for index in range(len(self.projections))
        )


@dataclass(frozen=True)
class ExactSourceWindow(SourceEditWindowABC):
    """A checked view of one actual projection, not separately authored output."""

    resolution: ExactSourceEditResolution
    projection_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.projection_index < len(self.resolution.projections):
            raise ValueError(
                "Exact source window requires an actual resolution projection"
            )

    @property
    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        return self.resolution.projections[self.projection_index][1]

    @cached_property
    def replacement_inputs(
        self,
    ) -> tuple[tuple[SourceTextMutation, SourceTextSpanReplacement], ...]:
        return tuple(
            occurrence
            for replacement in self.replacements
            for occurrence in self.resolution.replacement_inputs[replacement]
        )

    @cached_property
    def physical_edit(self) -> PhysicalSourceEdit:
        return self.resolution.projections[self.projection_index][0].with_evidence_from(
            mutation for mutation, replacement in self.replacement_inputs
        )

    def _project_retained_interiors(
        self, spans: tuple[SourceTextSpan, ...]
    ) -> tuple[SourceTextSpan, ...]:
        geometry = self.resolution.geometry
        domain = self.physical_edit.original_span(geometry)
        if any(not span.is_within(domain) for span in spans):
            raise ValueError(
                "Retained source interiors must belong to their physical window"
            )
        return tuple(
            SourceTextSpan(
                span.start_offset - domain.start_offset,
                span.end_offset - domain.start_offset,
            )
            for span in geometry.project_unchanged_spans(spans, self.replacements)
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
    """AST envelope; SourceTextGeometry resolves exact decorated source bounds."""

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
        ast.Constant,
        ast.JoinedStr,
        *((ast.TemplateStr,) if sys.version_info >= (3, 14) else ()),
    )

    @cached_property
    def logical_lines(self) -> tuple[str, ...]:
        """Original logical lines; an empty module still owns one empty line."""
        return self.lines or ("",)

    def line_anchor_offset(self, line: int) -> int:
        """Project a physical line anchor, including empty source and end of file."""
        if not 1 <= line <= len(self.line_offsets) + 1:
            raise ValueError("Line anchor is outside source geometry")
        return (
            self.end_offset
            if line == len(self.line_offsets) + 1
            else self.line_offsets[line - 1]
        )

    def project_unchanged_spans(
        self,
        spans: Iterable[SourceTextSpan],
        replacements: Iterable[SourceTextSpanReplacement],
    ) -> tuple[SourceTextSpan, ...]:
        """Transport retained character spans through one exact edit batch.

        Insertions at a span's start precede the retained text; insertions at its
        end follow it. Interior insertions and replacements reject correspondence,
        even when their new text matches. Empty spans carry no retained text.
        This establishes text provenance only, not syntax or execution equivalence.
        """
        domain = SourceTextSpan(0, self.end_offset)
        ordered = tuple(
            replacement
            for replacement in self.replacements_in_span(
                0, self.end_offset, replacements
            )
            if replacement.start_offset != replacement.end_offset
            or replacement.replacement_source
        )
        starts = tuple(replacement.start_offset for replacement in ordered)
        ends = tuple(replacement.end_offset for replacement in ordered)
        deltas = [0]
        for replacement in ordered:
            deltas.append(
                deltas[-1]
                + len(replacement.replacement_source)
                - (replacement.end_offset - replacement.start_offset)
            )
        projected = []
        for span in spans:
            if not span.is_within(domain) or span.start_offset == span.end_offset:
                raise ValueError(
                    "Retained source span must be nonempty and fit its source"
                )
            preceding = bisect_right(ends, span.start_offset)
            following = bisect_left(starts, span.end_offset)
            if preceding != following:
                raise ValueError(
                    "Edited source span has no unchanged-text correspondence"
                )
            delta = deltas[preceding]
            projected.append(
                SourceTextSpan(span.start_offset + delta, span.end_offset + delta)
            )
        return tuple(projected)

    def physical_edit_projections(
        self,
        *,
        file_path: str,
        replacements: Iterable[SourceTextSpanReplacement],
        rationale: str = "",
    ) -> tuple[tuple[PhysicalSourceEdit, tuple[SourceTextSpanReplacement, ...]], ...]:
        """Retain the exact inputs of each physical edit for provenance consumers."""
        ordered = self.replacements_in_span(0, self.end_offset, replacements)
        line_windows: list[tuple[int, int, list[SourceTextSpanReplacement]]] = []
        insertions: list[
            tuple[PhysicalSourceEdit, tuple[SourceTextSpanReplacement, ...]]
        ] = []
        for replacement in ordered:
            insertion_line = self._line_start_insertion_line(replacement)
            if insertion_line is not None:
                insertions.append(
                    (
                        SourceInsertion(
                            file_path=file_path,
                            insertion_line=insertion_line,
                            inserted_lines=tuple(
                                replacement.replacement_source.splitlines(keepends=True)
                            ),
                            rationale=rationale,
                        ),
                        (replacement,),
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
            (
                SourceSpanEdit.from_replacement_lines(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    replacement_lines=tuple(
                        self.source_with_replacements_in_span(
                            *self.line_span_offsets(start_line, end_line),
                            window_replacements,
                        ).splitlines(keepends=True)
                    ),
                    rationale=rationale,
                ),
                tuple(window_replacements),
            )
            for start_line, end_line, window_replacements in line_windows
        )
        return (*span_replacements, *insertions)

    def nominal_edit(
        self,
        *,
        file_path: str,
        replacements: tuple[SourceTextSpanReplacement, ...],
        rationale: str = "",
    ) -> SourceTextMutation:
        """Retain exact spans and their source revision until all peers are known."""
        return SourceTextMutation(
            revision=CodemodSourceRevision(
                file_path, CodemodSourceRevision.hash_source(self.source)
            ),
            replacements=replacements,
            rationale=rationale,
        )

    def iter_tokens(self) -> Iterator[tokenize.TokenInfo]:
        """Read source tokens lazily when only a prefix is required."""
        return tokenize.generate_tokens(io.StringIO(self.source).readline)

    @cached_property
    def tokens(self) -> tuple[tokenize.TokenInfo, ...]:
        return tuple(self.iter_tokens())

    @cached_property
    def line_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for line in self.logical_lines:
            offsets.append(offset)
            offset += len(line)
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

    @cached_property
    def token_start_offsets(self) -> tuple[int, ...]:
        return tuple(self.token_position_offset(token.start) for token in self.tokens)

    def tokens_in_span(self, span: SourceTextSpan) -> tuple[tokenize.TokenInfo, ...]:
        start = bisect_left(self.token_start_offsets, span.start_offset)
        end = bisect_left(self.token_start_offsets, span.end_offset)
        return self.tokens[start:end]

    def span_contains_comment(self, span: SourceTextSpan) -> bool:
        return any(
            token.type == tokenize.COMMENT for token in self.tokens_in_span(span)
        )

    def literal_continuation_lines(self, root: ast.AST) -> frozenset[int]:
        """Lines whose source belongs to literals and must retain its indentation."""

        return frozenset(
            line_number
            for node in ast.walk(root)
            if isinstance(node, self.literal_node_types)
            for line_number in range(
                node.lineno + 1, SourceByteSpan.require_node(node).end_line_index + 2
            )
        )

    def call_argument_span(self, node: ast.Call) -> SourceTextSpan:
        """Locate the final call parentheses, retaining a parenthesised callee."""

        start, end = self.required_node_offsets(node)
        depth = 0
        for token in reversed(self.tokens_in_span(SourceTextSpan(start, end))):
            if token.type != tokenize.OP:
                continue
            if token.string == ")":
                depth += 1
            elif token.string == "(":
                depth -= 1
                if depth == 0:
                    return SourceTextSpan(
                        self.token_position_offset(token.end), end - 1
                    )
        raise ValueError("Call argument parentheses are unavailable")

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
        return self.line_span_offsets(self.node_start_line(span), span.end_line)

    def node_start_line(self, span: SourceNodeSpan) -> int:
        """Recover decorator markers that AST expression positions can omit."""

        node = span.node
        if not (
            span.decorator_policy.includes_decorators
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.decorator_list
        ):
            return node.lineno
        expression_start = self.required_node_offsets(node.decorator_list[0])[0]
        token_index = bisect_left(self.token_start_offsets, expression_start)
        for index in range(token_index - 1, -1, -1):
            token = self.tokens[index]
            if token.exact_type == tokenize.AT:
                return token.start[0]
        raise ValueError("Decorated declaration has no source decorator marker")

    def node_line_span(self, span: SourceNodeSpan) -> SourceLineSpan:
        return SourceLineSpan(self.node_start_line(span), span.end_line)

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
        fragments = []
        cursor = span_start
        for replacement in self.replacements_in_span(
            span_start, span_end, replacements
        ):
            fragments.extend(
                (
                    self.source[cursor : replacement.start_offset],
                    replacement.replacement_source,
                )
            )
            cursor = replacement.end_offset
        fragments.append(self.source[cursor:span_end])
        return "".join(fragments)

    def physical_edits(
        self,
        *,
        file_path: str,
        replacements: Iterable[SourceTextSpanReplacement],
        rationale: str = "",
    ) -> tuple[PhysicalSourceEdit, ...]:
        """Project offset edits into the smallest independent line edits."""
        return tuple(
            edit
            for edit, _window in self.physical_edit_projections(
                file_path=file_path, replacements=replacements, rationale=rationale
            )
        )

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

        target = SourceTextSpan(span_start, span_end)
        if not target.is_within(SourceTextSpan(0, self.end_offset)):
            raise ValueError(
                "Replacement target span must fit the source geometry: "
                f"{span_start}:{span_end}"
            )
        replacement_by_span: dict[SourceTextSpan, SourceTextSpanReplacement] = {}
        for replacement in replacements:
            if not replacement.is_within(target):
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
        for first, second in pairwise(ordered):
            if first.overlaps(second):
                raise ValueError(
                    "Offset replacement spans overlap: "
                    f"{first.start_offset}:{first.end_offset} and "
                    f"{second.start_offset}:{second.end_offset}"
                )
        return ordered

    def line_span_offsets(self, start_line: int, end_line: int) -> tuple[int, int]:
        """Project a nonempty inclusive physical line span through its anchors."""
        if start_line > end_line:
            raise ValueError("Physical line span must be nonempty")
        return self.line_anchor_offset(start_line), self.line_anchor_offset(
            end_line + 1
        )


@dataclass(frozen=True)
class SourceTargetEditor:
    """Line-oriented editor for one source-index target span."""

    sources: Mapping[str, str]
    target: AstTargetDigest

    @property
    def file_lines(self) -> list[str]:
        return list(SourceTextGeometry(self.sources[self.target.file_path]).logical_lines)

    @property
    def target_lines(self) -> list[str]:
        return self.file_lines[self.target.line - 1 : self.target.end_line]

    def replacement_source(
        self,
        replacements: Iterable[SourceEditWindowABC],
    ) -> str:
        lines = self.target_lines
        ordered_windows = self.ordered_windows(replacements)
        for window in reversed(ordered_windows):
            replacement = window.physical_edit
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

    def ordered_windows(
        self,
        windows: Iterable[SourceEditWindowABC],
    ) -> tuple[SourceEditWindowABC, ...]:
        """Order and admit actual windows under this target's physical geometry."""
        ordered_windows = sorted_tuple(
            windows,
            key=lambda window: (
                window.physical_edit.start_line,
                window.physical_edit.end_line,
            ),
        )
        previous_end = self.target.line - 1
        for window in ordered_windows:
            replacement = window.physical_edit
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
        return ordered_windows

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
class CodemodSourceRevision(CodemodPayloadRecord):
    """Full-source revision required by a simulation's read or write context."""

    file_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    source_hash: str | None = codemod_payload_field(OptionalStringPayloadValueCodec())

    @classmethod
    def capture(
        cls,
        sources_by_file_path: Mapping[str, str],
        *,
        required_paths: Iterable[str] = (),
    ) -> tuple["CodemodSourceRevision", ...]:
        """Capture the supplied source context, including absent creation targets."""

        return tuple(
            cls.from_sources(path, sources_by_file_path)
            for path in sorted(set(sources_by_file_path).union(required_paths))
        )

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

    hash_source = staticmethod(python_source_cache_signature)

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
            current_source = read_source_text(source_path, encoding=encoding)
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
