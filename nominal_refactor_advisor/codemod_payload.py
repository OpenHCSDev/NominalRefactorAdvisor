"""Declaration-owned JSON payload semantics for the codemod DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, dataclass
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from functools import lru_cache
from typing import (
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
)

from .json_reports import (
    DataclassJsonReport,
    JsonObject,
    JsonValue,
    declared_dataclass_field,
    json_report_object,
)

PayloadOwnerT = TypeVar("PayloadOwnerT")
PayloadValueT = TypeVar("PayloadValueT")
StrEnumT = TypeVar("StrEnumT", bound=StrEnum)
StringPayloadValueT = TypeVar("StringPayloadValueT", bound=str | None)


class DataclassPayloadProjection(DataclassJsonReport, ABC):
    """JSON projection derived completely from nominal dataclass fields."""

    omit_none_payload_values: ClassVar[bool] = False

    @classmethod
    @lru_cache(maxsize=None)
    def payload_bindings(cls) -> PayloadBindingSet[Self, object]:
        return PayloadBindingSet.from_dataclass(cls).require_complete_dataclass_fields(
            cls
        )

    @classmethod
    def from_payload_fields(cls, payload: Mapping[str, JsonValue]) -> Self:
        """Construct this projection through its declaration-owned bindings."""

        return cls(**cls.payload_bindings().constructor_kwargs(payload))

    @classmethod
    def project_json_object(cls, record: Self) -> JsonObject:
        return cls.payload_bindings().payload(
            record,
            omit_none=cls.omit_none_payload_values,
        )


class CodemodPayloadRecord(DataclassPayloadProjection, ABC):
    """Nominal JSON record that owns both decoding and encoding semantics."""

    @classmethod
    def payload_fields(cls, value: JsonValue) -> Mapping[str, JsonValue]:
        """Read this nominal record's object payload."""

        if not isinstance(value, Mapping):
            raise ValueError(f"{cls.__name__} payload must be an object")
        return cast(Mapping[str, JsonValue], value)

    def require_supported_payload_fields(
        self,
        payload: Mapping[str, JsonValue],
    ) -> None:
        """Reject fields absent from this record's declaration-derived projection."""

        supported_fields = set(type(self).payload_bindings().payload_field_names)
        supported_fields.update(json_report_object(self))
        unsupported_fields = tuple(sorted(set(payload) - supported_fields))
        if unsupported_fields:
            raise ValueError(
                f"Unsupported {type(self).__name__} payload field(s): "
                f"{', '.join(repr(field) for field in unsupported_fields)}"
            )

    @classmethod
    def from_json_value(cls, value: JsonValue) -> Self:
        """Decode one record through its declaration-owned field bindings."""

        payload = cls.payload_fields(value)
        record = cls.from_payload_fields(payload)
        record.require_supported_payload_fields(payload)
        return record


class PayloadValueCodec(Generic[PayloadValueT], ABC):
    """Nominal owner of one payload value's wire semantics."""

    @abstractmethod
    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> PayloadValueT:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, value: object) -> JsonValue:
        raise NotImplementedError

    def payload_field_names(self, field_name: str) -> tuple[str, ...]:
        """Return the wire fields projected by this codec."""

        return (field_name,)

    def payload_items(
        self,
        value: object,
        field_name: str,
        *,
        omit_none: bool = False,
    ) -> tuple[tuple[str, JsonValue], ...]:
        """Project one runtime value into its wire fields."""

        serialized = self.serialize(value)
        if omit_none and serialized is None:
            return ()
        return ((field_name, serialized),)


PayloadProjectionT = TypeVar("PayloadProjectionT", bound=DataclassPayloadProjection)


@dataclass(frozen=True)
class FlattenedPayloadRecordValueCodec(
    PayloadValueCodec[PayloadProjectionT],
    Generic[PayloadProjectionT],
):
    """Flatten one nested nominal record into its enclosing object payload."""

    record_type: type[PayloadProjectionT]

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> PayloadProjectionT:
        del field_name
        return self.record_type.from_payload_fields(payload)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, self.record_type):
            raise TypeError(
                f"flattened payload-record codec requires {self.record_type.__name__}"
            )
        return json_report_object(value)

    def payload_field_names(self, field_name: str) -> tuple[str, ...]:
        del field_name
        return self.record_type.payload_bindings().payload_field_names

    def payload_items(
        self,
        value: object,
        field_name: str,
        *,
        omit_none: bool = False,
    ) -> tuple[tuple[str, JsonValue], ...]:
        del field_name
        if not isinstance(value, self.record_type):
            raise TypeError(
                f"flattened payload-record codec requires {self.record_type.__name__}"
            )
        return tuple(
            self.record_type.payload_bindings()
            .payload(value, omit_none=omit_none)
            .items()
        )


class StringPayloadValueCodec(
    PayloadValueCodec[StringPayloadValueT],
    Generic[StringPayloadValueT],
    ABC,
):
    """Shared wire mechanics for the supported nominal string policies."""

    @abstractmethod
    def value_when_missing(self, field_name: str) -> StringPayloadValueT:
        """Return the declared missing-field value or reject its absence."""
        raise NotImplementedError

    def validate_present_value(
        self,
        value: str,
        field_name: str | None,
    ) -> None:
        """Validate one present value under the non-empty string policy."""
        if not value:
            if field_name is None:
                raise ValueError("string payload codec does not permit an empty value")
            raise ValueError(f"Expected non-empty string field {field_name!r}")

    def serialize_missing(self) -> JsonValue:
        """Reject missing values unless a nominal leaf declares them valid."""
        raise TypeError("string payload codec requires a string value")

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> StringPayloadValueT:
        value = payload.get(field_name)
        if value is None:
            return self.value_when_missing(field_name)
        if not isinstance(value, str):
            raise ValueError(f"Expected string field {field_name!r}")
        self.validate_present_value(value, field_name)
        return cast(StringPayloadValueT, value)

    def serialize(self, value: object) -> JsonValue:
        if value is None:
            return self.serialize_missing()
        if not isinstance(value, str):
            raise TypeError("string payload codec requires a string value")
        self.validate_present_value(value, None)
        return value


@dataclass(frozen=True)
class RequiredStringPayloadValueCodec(StringPayloadValueCodec[str]):
    """Require a present, non-empty string payload value."""

    def value_when_missing(self, field_name: str) -> str:
        raise ValueError(f"Expected non-empty string field {field_name!r}")


class DiscriminatedPayloadRecord(CodemodPayloadRecord, ABC):
    """Nominal record family selected by one declaration-owned wire key."""

    discriminator_field_name: ClassVar[str]

    @classmethod
    @abstractmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        """Resolve a registered nominal descendant for one wire key."""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def discriminator_key(cls) -> str:
        """Return this concrete declaration's wire key."""

        raise NotImplementedError

    @classmethod
    def from_json_value(cls, value: JsonValue) -> Self:
        return cls.from_dict(cls.payload_fields(value))

    @classmethod
    def from_dict(cls, payload: Mapping[str, JsonValue]) -> Self:
        discriminator = RequiredStringPayloadValueCodec().read(
            payload,
            cls.discriminator_field_name,
        )
        record_type = cls.record_type_for_discriminator(discriminator)
        record = record_type.from_payload_fields(payload)
        record.require_supported_payload_fields(payload)
        return record

    @classmethod
    def project_json_object(cls, record: Self) -> JsonObject:
        return JsonObject(
            {
                cls.discriminator_field_name: cls.discriminator_key(),
                **cls.payload_bindings().payload(
                    record,
                    omit_none=cls.omit_none_payload_values,
                ),
            }
        )


@dataclass(frozen=True)
class _MissingDefaultStringPayloadValueCodec(
    StringPayloadValueCodec[StringPayloadValueT],
    Generic[StringPayloadValueT],
):
    """Shared missing-value mechanics for declared string policy leaves."""

    missing_value: StringPayloadValueT

    def value_when_missing(self, field_name: str) -> StringPayloadValueT:
        del field_name
        return self.missing_value

    def serialize_missing(self) -> JsonValue:
        return None


@dataclass(frozen=True)
class DefaultedStringPayloadValueCodec(_MissingDefaultStringPayloadValueCodec[str]):
    """Use a declared default when a non-empty string field is absent."""


@dataclass(frozen=True)
class EmptyDefaultStringPayloadValueCodec(_MissingDefaultStringPayloadValueCodec[str]):
    """Accept empty strings and default an absent field to the empty string."""

    missing_value: str = ""

    def validate_present_value(
        self,
        value: str,
        field_name: str | None,
    ) -> None:
        del value, field_name


@dataclass(frozen=True)
class OptionalStringPayloadValueCodec(
    _MissingDefaultStringPayloadValueCodec[str | None]
):
    """Accept empty strings and optionally default an absent field."""

    missing_value: str | None = None

    def validate_present_value(
        self,
        value: str,
        field_name: str | None,
    ) -> None:
        del value, field_name


class _StrEnumPayloadValueCodec(
    PayloadValueCodec[PayloadValueT],
    Generic[StrEnumT, PayloadValueT],
    ABC,
):
    """Shared wire mechanics for typed string-enum payload policies."""

    enum_type: type[StrEnumT]

    @abstractmethod
    def value_when_missing(self) -> PayloadValueT:
        raise NotImplementedError

    def serialize_missing(self) -> JsonValue:
        raise TypeError(f"string-enum payload codec requires {self.enum_type.__name__}")

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> PayloadValueT:
        value = payload.get(field_name)
        if value is None:
            return self.value_when_missing()
        if not isinstance(value, str):
            raise ValueError(f"Expected string enum field {field_name!r}")
        try:
            return cast(PayloadValueT, self.enum_type(value))
        except ValueError as error:
            raise ValueError(f"Unsupported {field_name!r} value: {value!r}") from error

    def serialize(self, value: object) -> JsonValue:
        if value is None:
            return self.serialize_missing()
        if not isinstance(value, self.enum_type):
            raise TypeError(
                f"string-enum payload codec requires {self.enum_type.__name__}"
            )
        return value.value


@dataclass(frozen=True)
class RequiredStrEnumPayloadValueCodec(
    _StrEnumPayloadValueCodec[StrEnumT, StrEnumT],
    Generic[StrEnumT],
):
    """Require one present member of a declared string-enum authority."""

    enum_type: type[StrEnumT]

    def value_when_missing(self) -> StrEnumT:
        raise ValueError(
            f"Expected string enum field for {self.enum_type.__name__}"
        )


@dataclass(frozen=True)
class StrEnumPayloadValueCodec(
    _StrEnumPayloadValueCodec[StrEnumT, StrEnumT],
    Generic[StrEnumT],
):
    """Typed string-enum payload semantics with a declared default."""

    enum_type: type[StrEnumT]
    declared_default: StrEnumT

    def value_when_missing(self) -> StrEnumT:
        return self.declared_default


@dataclass(frozen=True)
class OptionalStrEnumPayloadValueCodec(
    _StrEnumPayloadValueCodec[StrEnumT, StrEnumT | None],
    Generic[StrEnumT],
):
    """Optional typed string-enum payload semantics."""

    enum_type: type[StrEnumT]

    def value_when_missing(self) -> None:
        return None

    def serialize_missing(self) -> None:
        return None


@dataclass(frozen=True)
class StringArrayPayloadValueCodec(PayloadValueCodec[tuple[str, ...]]):
    """Required array-of-string payload semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[str, ...]:
        if field_name not in payload or payload[field_name] is None:
            raise ValueError(f"Expected string array field {field_name!r}")
        value = payload[field_name]
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Expected string array field {field_name!r}")
        return tuple(value)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in value
        ):
            raise TypeError("string-array payload codec requires string values")
        return tuple(value)


@dataclass(frozen=True)
class OptionalStringArrayPayloadValueCodec(StringArrayPayloadValueCodec):
    """Array-of-string payload semantics with an empty missing-field value."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[str, ...]:
        if field_name not in payload or payload[field_name] is None:
            return ()
        return super().read(payload, field_name)


@dataclass(frozen=True)
class BooleanPayloadValueCodec(PayloadValueCodec[bool]):
    """Optional boolean payload semantics with one declared default."""

    declared_default: bool = False

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> bool:
        if field_name not in payload:
            return self.declared_default
        value = payload[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean field {field_name!r}")
        return value

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, bool):
            raise TypeError("boolean payload codec requires a boolean value")
        return value


@dataclass(frozen=True)
class IntegerPayloadValueCodec(PayloadValueCodec[int | None]):
    """Optional non-negative integer payload semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> int | None:
        value = payload.get(field_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Expected non-negative integer field {field_name!r}")
        return value

    def serialize(self, value: object) -> JsonValue:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError("integer payload codec requires a non-negative integer")
        return value


@dataclass(frozen=True)
class RequiredIntegerPayloadValueCodec(IntegerPayloadValueCodec):
    """Require a present non-negative integer payload value."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> int:
        value = super().read(payload, field_name)
        if value is None:
            raise ValueError(f"Expected non-negative integer field {field_name!r}")
        return value


@dataclass(frozen=True)
class ObjectPayloadValueCodec(PayloadValueCodec[Mapping[str, JsonValue]]):
    """Required JSON-object payload semantics."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> Mapping[str, JsonValue]:
        value = payload.get(field_name)
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected object field {field_name!r}")
        return value

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, Mapping):
            raise TypeError("object payload codec requires a mapping value")
        return JsonObject(value)


PayloadRecordT = TypeVar("PayloadRecordT", bound=CodemodPayloadRecord)


@dataclass(frozen=True)
class PayloadRecordValueCodec(
    PayloadValueCodec[PayloadRecordT],
    Generic[PayloadRecordT],
):
    """Required object payload decoded by its nominal record declaration."""

    record_type: type[PayloadRecordT]

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> PayloadRecordT:
        value = ObjectPayloadValueCodec().read(payload, field_name)
        return self.record_type.from_json_value(cast(JsonValue, value))

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, self.record_type):
            raise TypeError(
                f"record payload codec requires {self.record_type.__name__}"
            )
        return json_report_object(value)


@dataclass(frozen=True)
class PayloadRecordArrayValueCodec(
    PayloadValueCodec[tuple[PayloadRecordT, ...]],
    Generic[PayloadRecordT],
):
    """Optional array whose nominal record type owns each JSON object."""

    record_type: type[PayloadRecordT]

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> tuple[PayloadRecordT, ...]:
        value = payload.get(field_name)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected record array field {field_name!r}")
        return tuple(self.record_type.from_json_value(item) for item in value)

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, self.record_type) for item in value
        ):
            raise TypeError(
                f"record-array payload codec requires {self.record_type.__name__} "
                "values"
            )
        return tuple(json_report_object(item) for item in value)


@dataclass(frozen=True)
class PayloadFieldDeclaration(Generic[PayloadValueT]):
    """Wire semantics owned by one dataclass constructor field."""

    codec: PayloadValueCodec[PayloadValueT]
    field_name: str | None = None


_PAYLOAD_FIELD_DECLARATION = object()


def codemod_payload_field(
    codec: PayloadValueCodec[PayloadValueT],
    *,
    field_name: str | None = None,
    default: PayloadValueT | object = MISSING,
    default_factory: Callable[[], PayloadValueT] | object = MISSING,
) -> PayloadValueT:
    """Declare a constructor field and its derived codemod wire projection."""

    return declared_dataclass_field(
        {
            _PAYLOAD_FIELD_DECLARATION: PayloadFieldDeclaration(
                codec=codec,
                field_name=field_name,
            )
        },
        default=default,
        default_factory=default_factory,
    )


@dataclass(frozen=True)
class PayloadBinding(Generic[PayloadOwnerT, PayloadValueT]):
    """Derived JSON-to-constructor binding for one DSL payload field."""

    field_name: str
    constructor_argument_name: str
    codec: PayloadValueCodec[PayloadValueT]

    def constructor_kwargs(
        self,
        payload: Mapping[str, JsonValue],
    ) -> dict[str, PayloadValueT]:
        return {
            self.constructor_argument_name: self.codec.read(payload, self.field_name)
        }

    def payload_items(
        self,
        owner: PayloadOwnerT,
        *,
        omit_none: bool = False,
    ) -> tuple[tuple[str, JsonValue], ...]:
        value = getattr(owner, self.constructor_argument_name)
        return self.codec.payload_items(
            value,
            self.field_name,
            omit_none=omit_none,
        )

    @property
    def payload_field_names(self) -> tuple[str, ...]:
        return self.codec.payload_field_names(self.field_name)


class PayloadBindingSet(
    tuple[PayloadBinding[PayloadOwnerT, PayloadValueT], ...],
    Generic[PayloadOwnerT, PayloadValueT],
):
    """Validated projection of declaration-owned payload fields."""

    def __new__(
        cls,
        bindings: Iterable[PayloadBinding[PayloadOwnerT, PayloadValueT]] = (),
    ) -> Self:
        binding_tuple = tuple(bindings)
        cls.require_unique_binding_names(binding_tuple)
        return super().__new__(cls, binding_tuple)

    @classmethod
    def from_dataclass(
        cls,
        owner_type: type[PayloadOwnerT],
    ) -> Self:
        """Derive wire bindings from the owner's declared dataclass fields."""

        bindings = []
        for record_field in dataclass_fields(owner_type):
            declaration = record_field.metadata.get(_PAYLOAD_FIELD_DECLARATION)
            if declaration is None:
                continue
            if not isinstance(declaration, PayloadFieldDeclaration):
                raise TypeError(
                    f"Invalid payload field declaration on {owner_type.__name__}."
                    f"{record_field.name}"
                )
            bindings.append(
                PayloadBinding(
                    field_name=declaration.field_name or record_field.name,
                    constructor_argument_name=record_field.name,
                    codec=declaration.codec,
                )
            )
        return cls(bindings)

    def require_complete_dataclass_fields(
        self,
        owner_type: type[PayloadOwnerT],
    ) -> Self:
        """Fail when a constructor field has no payload declaration."""

        expected_field_names = frozenset(
            record_field.name for record_field in dataclass_fields(owner_type)
        )
        bound_field_names = frozenset(
            binding.constructor_argument_name for binding in self
        )
        missing_field_names = tuple(sorted(expected_field_names - bound_field_names))
        unexpected_field_names = tuple(sorted(bound_field_names - expected_field_names))
        if missing_field_names or unexpected_field_names:
            raise TypeError(
                f"Incomplete payload field declarations on {owner_type.__name__}: "
                f"missing={missing_field_names!r}, unexpected={unexpected_field_names!r}"
            )
        return self

    def constructor_kwargs(
        self,
        payload: Mapping[str, JsonValue],
    ) -> dict[str, PayloadValueT]:
        constructor_kwargs: dict[str, PayloadValueT] = {}
        for binding in self:
            constructor_kwargs.update(binding.constructor_kwargs(payload))
        return constructor_kwargs

    def payload(
        self,
        owner: PayloadOwnerT,
        *,
        omit_none: bool = False,
    ) -> JsonObject:
        payload = {
            key: value
            for binding in self
            for key, value in binding.payload_items(owner, omit_none=omit_none)
        }
        return JsonObject(payload)

    def has_field_in(self, payload: Mapping[str, JsonValue]) -> bool:
        return any(
            field_name in payload
            for binding in self
            for field_name in binding.payload_field_names
        )

    @property
    def payload_field_names(self) -> tuple[str, ...]:
        """Return every wire field projected by the binding set."""

        return tuple(
            field_name for binding in self for field_name in binding.payload_field_names
        )

    @staticmethod
    def require_unique_binding_names(
        bindings: tuple[
            PayloadBinding[PayloadOwnerT, PayloadValueT],
            ...,
        ],
    ) -> None:
        for name_kind, names in (
            (
                "payload field",
                tuple(
                    field_name
                    for binding in bindings
                    for field_name in binding.payload_field_names
                ),
            ),
            (
                "constructor argument",
                tuple(binding.constructor_argument_name for binding in bindings),
            ),
        ):
            duplicate_names = tuple(
                name for name, count in Counter(names).items() if count > 1
            )
            if duplicate_names:
                raise ValueError(
                    f"Duplicate {name_kind} binding name(s): "
                    f"{', '.join(repr(name) for name in duplicate_names)}"
                )
