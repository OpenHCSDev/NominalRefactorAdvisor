"""Declaration-only values and wire codecs used by codemod selectors."""

from __future__ import annotations

import re
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from typing import (
    ClassVar,
    Self,
)

from .codemod_paths import SourcePathResolutionAuthority
from .codemod_payload import (
    CodemodPayloadRecord,
    EmptyDefaultStringPayloadValueCodec,
    FlattenedPayloadRecordValueCodec,
    IntegerPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
    OptionalStringPayloadValueCodec,
    PayloadRecordValueCodec,
    PayloadValueCodec,
    codemod_payload_field,
)
from .json_reports import (
    JsonValue,
    json_report_object,
)
from .models import SourceLocation
from .source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
    SourceIndex,
    SourceTargetIdentity,
)


@dataclass(frozen=True)
class SourceRewriteTarget(
    SourceTargetIdentity[str | None],
    CodemodPayloadRecord,
):
    """Source-index target selector for a planned rewrite."""

    target_id: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )
    qualname: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        field_name="target_qualname",
        default=None,
    )
    file_path: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    @classmethod
    def from_semantic_target(cls, target: AstTargetDigest) -> Self:
        """Address a declaration by stable source path and nominal identity."""

        return cls(file_path=target.file_path, qualname=target.qualname)

    def optional_file_path(self, source_index: SourceIndex) -> str | None:
        if self.file_path is None:
            return None
        return SourcePathResolutionAuthority.from_source_index(
            self.file_path,
            source_index,
        ).required_path()

    def required_file_path(self, source_index: SourceIndex) -> str:
        file_path = self.optional_file_path(source_index)
        if file_path is None:
            raise ValueError("Source rewrite target requires file_path")
        return file_path

    def optional_target_id(
        self,
        source_index: SourceIndex,
        *,
        eligible_target_ids: Iterable[str] | None = None,
    ) -> str | None:
        eligible_ids = (
            set(eligible_target_ids) if eligible_target_ids is not None else None
        )
        if self.target_id is not None:
            if self.target_id in source_index.target_by_id and (
                eligible_ids is None or self.target_id in eligible_ids
            ):
                return self.target_id
            return None
        file_path = self.optional_file_path(source_index)
        if self.qualname is None:
            return self._optional_module_target_id(
                source_index,
                eligible_ids,
                file_path,
            )
        matching_target_ids = [
            target.target_id
            for target in self.candidate_targets(source_index, file_path)
            if eligible_ids is None or target.target_id in eligible_ids
        ]
        if len(matching_target_ids) != 1:
            return None
        return matching_target_ids[0]

    def _optional_module_target_id(
        self,
        source_index: SourceIndex,
        eligible_target_ids: set[str] | None,
        file_path: str | None,
    ) -> str | None:
        if file_path is None:
            return None
        matching_target_ids = [
            target.target_id
            for target in source_index.targets_by_file[file_path]
            if target.is_module
            and (eligible_target_ids is None or target.target_id in eligible_target_ids)
        ]
        if len(matching_target_ids) != 1:
            return None
        return matching_target_ids[0]

    def candidate_targets(
        self,
        source_index: SourceIndex,
        file_path: str | None,
    ) -> tuple[AstTargetDigest, ...]:
        if self.qualname is None:
            return ()
        if file_path is not None:
            if file_path not in source_index.targets_by_file:
                return ()
            return tuple(
                target
                for target in source_index.targets_by_file[file_path]
                if target.qualname == self.qualname
            )
        return source_index.targets_by_qualname.tuple_for_key(self.qualname)

    def required_target_id(
        self,
        source_index: SourceIndex,
        *,
        eligible_target_ids: Iterable[str] | None = None,
    ) -> str:
        target_id = self.optional_target_id(
            source_index,
            eligible_target_ids=eligible_target_ids,
        )
        if target_id is not None:
            return target_id
        raise ValueError(
            "Source rewrite target did not resolve to exactly one eligible "
            "source-index target"
        )


@dataclass(frozen=True)
class SourceRewriteTargetPreflightDetail(CodemodPayloadRecord):
    """Typed source target retained by a failed operation preflight."""

    target: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )


@dataclass(frozen=True, kw_only=True)
class SourceRewriteTargetReference(CodemodPayloadRecord):
    """Shared owner for DSL records that reference source-index targets."""

    target: SourceRewriteTarget = codemod_payload_field(
        FlattenedPayloadRecordValueCodec(SourceRewriteTarget),
        default_factory=SourceRewriteTarget,
    )

    def referenced_source_targets(self) -> tuple[SourceRewriteTarget, ...]:
        return self.records_of_type(SourceRewriteTarget)


@dataclass(frozen=True, kw_only=True)
class SourceRewritePlanItem(SourceRewriteTargetReference):
    """Common target and rationale state for source rewrite plan items."""

    rationale: str = codemod_payload_field(
        EmptyDefaultStringPayloadValueCodec(),
        default="",
    )

    def rationale_text(self, default: str) -> str:
        if self.rationale:
            return self.rationale
        return default


@dataclass(frozen=True)
class CodemodTargetSelection:
    """Resolved source-index target ids selected by semantic criteria."""

    target_ids: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.target_ids

    def digests(self, source_index: SourceIndex) -> tuple[AstTargetDigest, ...]:
        return tuple(
            source_index.target_by_id[target_id] for target_id in self.target_ids
        )


@dataclass(frozen=True)
class SelectionCountExpectation(CodemodPayloadRecord):
    """Cardinality contract for selector-backed codemod operations."""

    omit_none_payload_values: ClassVar[bool] = True

    minimum: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        field_name="min",
        default=None,
    )
    maximum: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        field_name="max",
        default=None,
    )
    exact: int | None = codemod_payload_field(
        IntegerPayloadValueCodec(),
        default=None,
    )

    @classmethod
    def from_json_value(cls, value: JsonValue) -> "SelectionCountExpectation":
        expectation = super().from_json_value(value)
        expectation.validate_definition()
        return expectation

    @property
    def is_empty(self) -> bool:
        return self.minimum is None and self.maximum is None and self.exact is None

    def validate_definition(self) -> None:
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("selection_count min cannot exceed max")
        if self.exact is None:
            return
        if self.minimum is not None and self.exact < self.minimum:
            raise ValueError("selection_count exact cannot be less than min")
        if self.maximum is not None and self.exact > self.maximum:
            raise ValueError("selection_count exact cannot exceed max")

    def require_actual_count(self, actual_count: int) -> None:
        self.validate_definition()
        if self.exact is not None and actual_count != self.exact:
            raise ValueError(
                "Selected-target operation expected exactly "
                f"{self.exact} target(s), but selector resolved {actual_count}"
            )
        if self.minimum is not None and actual_count < self.minimum:
            raise ValueError(
                "Selected-target operation expected at least "
                f"{self.minimum} target(s), but selector resolved {actual_count}"
            )
        if self.maximum is not None and actual_count > self.maximum:
            raise ValueError(
                "Selected-target operation expected at most "
                f"{self.maximum} target(s), but selector resolved {actual_count}"
            )


@dataclass(frozen=True)
class NodeKindArrayPayloadValueCodec(OptionalStringArrayPayloadValueCodec):
    """AST target-node kind array payload semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[AstTargetNodeKind, ...]:
        return tuple(
            AstTargetNodeKind(value) for value in super().read(payload, field_name)
        )

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, AstTargetNodeKind) for item in value
        ):
            raise TypeError("node-kind payload codec requires AstTargetNodeKind values")
        return tuple(item.value for item in value)


@dataclass(frozen=True)
class SelectionCountPayloadValueCodec(PayloadValueCodec["SelectionCountExpectation"]):
    """Optional selected-target cardinality contract semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> "SelectionCountExpectation":
        value = payload.get(field_name)
        if value is None:
            return SelectionCountExpectation()
        return SelectionCountExpectation.from_json_value(value)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, SelectionCountExpectation):
            raise TypeError(
                "selection-count payload codec requires SelectionCountExpectation"
            )
        if value.is_empty:
            return None
        return json_report_object(value)


@dataclass(frozen=True)
class RegexPatternSet:
    """Validated regular-expression filter set for source-index selectors."""

    patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def from_patterns(cls, patterns: Iterable[str]) -> "RegexPatternSet":
        try:
            return cls(tuple(re.compile(pattern) for pattern in patterns))
        except re.error as error:
            raise ValueError(f"Invalid selector regex pattern: {error}") from error

    def matches(self, value: str) -> bool:
        if not self.patterns:
            return True
        return any(pattern.search(value) is not None for pattern in self.patterns)


@dataclass(frozen=True)
class CallSiteDigest:
    """Concrete call-site coordinate selected from source text."""

    file_path: str
    line: int
    symbol: str
    enclosing_target_id: str | None = None

    @property
    def source_location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.line, self.symbol)
