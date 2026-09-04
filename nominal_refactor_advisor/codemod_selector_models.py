"""Declaration-only values and wire codecs used by codemod selectors."""

from __future__ import annotations

import re
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from typing import ClassVar

from .codemod_payload import (
    CodemodPayloadRecord,
    IntegerPayloadValueCodec,
    OptionalStringArrayPayloadValueCodec,
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
)


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
