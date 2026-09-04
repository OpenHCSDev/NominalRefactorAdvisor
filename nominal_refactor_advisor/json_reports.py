"""Declaration-owned JSON report projection."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import Counter
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import (
    MISSING,
    dataclass,
    field,
)
from dataclasses import (
    fields as dataclass_fields,
)
from enum import StrEnum
from functools import (
    cached_property,
    lru_cache,
    singledispatchmethod,
)
from typing import (
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from .descriptor_algebra import AliasProperty

JsonScalar: TypeAlias = str | int | float | bool | None


class JsonObject(dict[str, "JsonValue"]):
    """Nominal JSON object payload at codemod and CLI boundaries."""


JsonArray: TypeAlias = tuple["JsonValue", ...] | list["JsonValue"]


JsonValue: TypeAlias = JsonScalar | JsonArray | JsonObject


DataclassFieldValueT = TypeVar("DataclassFieldValueT")


ReportOwnerT = TypeVar("ReportOwnerT")


ReportValueT = TypeVar("ReportValueT")


class JsonReport(ABC):
    """Nominal boundary for declarations that project to JSON objects."""

    @classmethod
    @abstractmethod
    def project_json_object(cls, report: Self) -> JsonObject:
        """Project one instance through its nominal declaration's JSON policy."""

        raise NotImplementedError


class JsonReportValueProjection:
    """Project the closed JSON value algebra through nominal runtime types."""

    @singledispatchmethod
    def project(self, value: object) -> JsonValue:
        raise TypeError(
            f"No JSON report projection is declared for {type(value).__name__}"
        )

    @project.register
    def _project_scalar(self, value: JsonScalar) -> JsonScalar:
        return value

    @project.register
    def _project_enum(self, value: StrEnum) -> str:
        return value.value

    @project.register
    def _project_tuple(self, value: tuple) -> tuple[JsonValue, ...]:
        return tuple(self.project(item) for item in value)

    @project.register
    def _project_list(self, value: list) -> list[JsonValue]:
        return [self.project(item) for item in value]

    @project.register
    def _project_mapping(self, value: Mapping) -> JsonObject:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON report mappings require string keys")
        return JsonObject(
            {cast(str, key): self.project(item) for key, item in value.items()}
        )

    @project.register
    def _project_report(self, value: JsonReport) -> JsonObject:
        return json_report_object(value)


JSON_REPORT_VALUE_PROJECTION = JsonReportValueProjection()


@dataclass(frozen=True)
class JsonReportFieldDeclaration:
    """Output projection semantics attached to one dataclass field."""

    field_name: str | None = None
    included: bool = True
    flattened: bool = False
    omit_none: bool = False


_JSON_REPORT_FIELD_DECLARATION = object()


class JsonReportProperty(property):
    """Property whose output projection is derived through its MRO declaration."""

    def __init__(
        self,
        getter: Callable[[ReportOwnerT], ReportValueT],
        *,
        field_name: str | None = None,
        flattened: bool = False,
        omit_none: bool = False,
    ) -> None:
        super().__init__(getter)
        self.field_name = field_name
        self.flattened = flattened
        self.omit_none = omit_none


class JsonReportCachedProperty(cached_property):
    """Cached property carrying its declaration-owned output projection."""

    def __init__(
        self,
        getter: Callable[[ReportOwnerT], ReportValueT],
        *,
        field_name: str | None = None,
        flattened: bool = False,
        omit_none: bool = False,
    ) -> None:
        super().__init__(getter)
        self.field_name = field_name
        self.flattened = flattened
        self.omit_none = omit_none


@dataclass(frozen=True)
class JsonReportAliasProperty(AliasProperty[ReportValueT]):
    """Report binding that derives its value from another owned attribute."""

    field_name: str | None = None
    flattened: bool = False
    omit_none: bool = False


def json_report_property(
    *,
    field_name: str | None = None,
    flattened: bool = False,
    omit_none: bool = False,
) -> Callable[[Callable[[ReportOwnerT], ReportValueT]], JsonReportProperty]:
    """Declare a computed JSON field on its typed report owner."""

    def declare(
        getter: Callable[[ReportOwnerT], ReportValueT],
    ) -> JsonReportProperty:
        return JsonReportProperty(
            getter,
            field_name=field_name,
            flattened=flattened,
            omit_none=omit_none,
        )

    return declare


def json_report_cached_property(
    *,
    field_name: str | None = None,
    flattened: bool = False,
    omit_none: bool = False,
) -> Callable[[Callable[[ReportOwnerT], ReportValueT]], JsonReportCachedProperty]:
    """Declare a computed JSON field whose runtime value is cached once."""

    def declare(
        getter: Callable[[ReportOwnerT], ReportValueT],
    ) -> JsonReportCachedProperty:
        return JsonReportCachedProperty(
            getter,
            field_name=field_name,
            flattened=flattened,
            omit_none=omit_none,
        )

    return declare


def json_report_alias(
    source_name: str,
    *,
    field_name: str | None = None,
    flattened: bool = False,
    omit_none: bool = False,
) -> JsonReportAliasProperty[ReportValueT]:
    """Declare a JSON field as a derived alias of an owned attribute."""

    return JsonReportAliasProperty(
        source_name=source_name,
        field_name=field_name,
        flattened=flattened,
        omit_none=omit_none,
    )


class DataclassJsonReport(JsonReport, ABC):
    """JSON projection derived from nominal fields and report properties."""

    @classmethod
    @lru_cache(maxsize=None)
    def report_bindings(cls) -> JsonReportBindingSet:
        return JsonReportBindingSet.from_dataclass(cls)

    @classmethod
    def project_json_object(cls, report: Self) -> JsonObject:
        return cls.report_bindings().payload(report)


class SemanticRecord(DataclassJsonReport, ABC):
    """Semantic record whose JSON projection is derived from its declaration."""


def json_report_object(report: JsonReport) -> JsonObject:
    """Erase one typed report only at an explicit JSON-object boundary."""

    return type(report).project_json_object(report)


def declared_dataclass_field(
    metadata: Mapping[object, object],
    *,
    default: DataclassFieldValueT | object = MISSING,
    default_factory: Callable[[], DataclassFieldValueT] | object = MISSING,
    compare: bool = True,
    repr: bool = True,
) -> DataclassFieldValueT:
    if default is not MISSING and default_factory is not MISSING:
        raise TypeError("declared dataclass fields cannot declare two defaults")
    if default is not MISSING:
        return cast(
            DataclassFieldValueT,
            field(
                default=default,
                metadata=metadata,
                compare=compare,
                repr=repr,
            ),
        )
    if default_factory is not MISSING:
        return cast(
            DataclassFieldValueT,
            field(
                default_factory=cast(
                    Callable[[], DataclassFieldValueT],
                    default_factory,
                ),
                metadata=metadata,
                compare=compare,
                repr=repr,
            ),
        )
    return cast(
        DataclassFieldValueT,
        field(metadata=metadata, compare=compare, repr=repr),
    )


def json_report_field(
    *,
    field_name: str | None = None,
    included: bool = True,
    flattened: bool = False,
    omit_none: bool = False,
    default: DataclassFieldValueT | object = MISSING,
    default_factory: Callable[[], DataclassFieldValueT] | object = MISSING,
    compare: bool = True,
    repr: bool = True,
) -> DataclassFieldValueT:
    """Declare output-only JSON semantics on one dataclass field."""

    return declared_dataclass_field(
        {
            _JSON_REPORT_FIELD_DECLARATION: JsonReportFieldDeclaration(
                field_name=field_name,
                included=included,
                flattened=flattened,
                omit_none=omit_none,
            )
        },
        default=default,
        default_factory=default_factory,
        compare=compare,
        repr=repr,
    )


@dataclass(frozen=True)
class JsonReportBinding:
    """One declaration-derived runtime value to JSON field projection."""

    source_name: str
    field_name: str
    flattened: bool = False
    omit_none: bool = False

    def payload_items(self, owner: object) -> tuple[tuple[str, JsonValue], ...]:
        source_value = getattr(owner, self.source_name)
        if self.omit_none and source_value is None:
            return ()
        value = JSON_REPORT_VALUE_PROJECTION.project(source_value)
        if not self.flattened:
            return ((self.field_name, value),)
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Flattened JSON report field {self.source_name!r} must project "
                "to an object"
            )
        return tuple((str(key), item) for key, item in value.items())


class JsonReportBindingSet(tuple[JsonReportBinding, ...]):
    """MRO-derived JSON field bindings for one output-only report."""

    def __new__(
        cls,
        bindings: Iterable[JsonReportBinding] = (),
    ) -> Self:
        binding_tuple = tuple(bindings)
        explicit_names = tuple(
            binding.field_name for binding in binding_tuple if not binding.flattened
        )
        duplicate_names = tuple(
            sorted(name for name, count in Counter(explicit_names).items() if count > 1)
        )
        if duplicate_names:
            raise TypeError(
                f"Duplicate JSON report field declaration(s): {duplicate_names!r}"
            )
        return super().__new__(cls, binding_tuple)

    @classmethod
    def from_dataclass(cls, owner_type: type[object]) -> Self:
        bindings: list[JsonReportBinding] = []
        for record_field in dataclass_fields(owner_type):
            declaration = record_field.metadata.get(_JSON_REPORT_FIELD_DECLARATION)
            if declaration is not None and not isinstance(
                declaration,
                JsonReportFieldDeclaration,
            ):
                raise TypeError(
                    f"Invalid JSON report field declaration on "
                    f"{owner_type.__name__}.{record_field.name}"
                )
            if isinstance(declaration, JsonReportFieldDeclaration):
                if not declaration.included:
                    continue
                field_name = declaration.field_name or record_field.name
                flattened = declaration.flattened
                omit_none = declaration.omit_none
            else:
                field_name = record_field.name
                flattened = False
                omit_none = False
            bindings.append(
                JsonReportBinding(
                    source_name=record_field.name,
                    field_name=field_name,
                    flattened=flattened,
                    omit_none=omit_none,
                )
            )
        declared_property_names: set[str] = set()
        for owner in owner_type.__mro__:
            for member_name, member in owner.__dict__.items():
                if member_name in declared_property_names or not isinstance(
                    member,
                    (
                        JsonReportProperty,
                        JsonReportCachedProperty,
                        JsonReportAliasProperty,
                    ),
                ):
                    continue
                declared_property_names.add(member_name)
                bindings.append(
                    JsonReportBinding(
                        source_name=member_name,
                        field_name=member.field_name or member_name,
                        flattened=member.flattened,
                        omit_none=member.omit_none,
                    )
                )
        return cls(bindings)

    def payload(self, owner: object) -> JsonObject:
        payload = JsonObject()
        for binding in self:
            for field_name, value in binding.payload_items(owner):
                if field_name in payload:
                    raise TypeError(
                        f"Duplicate projected JSON report field {field_name!r}"
                    )
                payload[field_name] = value
        return payload
