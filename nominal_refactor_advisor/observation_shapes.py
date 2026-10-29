"""Collected semantic shapes and structural projection carriers.

These frozen records represent the normalized items emitted by observation specs.
Most detector families reason over these carriers indirectly through their
structural observation projections.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .class_composition import CompositeClassSpec
from .constructor_algebra import (
    ConstructorDerivedField,
    ConstructorVariantCatalog,
    ConstructorVariantSpec,
)
from .descriptor_algebra import AliasProperty
from .export_tools import PublicExportPolicy, derive_public_exports
from .models import SourceLocation
from .observation_graph import (
    ObservationKind,
    StructuralExecutionLevel,
    StructuralObservation,
    StructuralObservationCarrier,
)
from .semantic_algebra import DispatchAxisExpression

if TYPE_CHECKING:
    from .ast_tools import ParsedModule


class LiteralKind(StrEnum):
    """Literal kinds used by literal-dispatch observations."""

    literal_type: type[str] | type[int]

    def __new__(
        cls,
        value: str,
        literal_type: type[str] | type[int],
    ) -> LiteralKind:
        member = str.__new__(cls, value)
        member._value_ = value
        member.literal_type = literal_type
        return member

    STRING = ("string", str)
    NUMERIC = ("numeric", int)


class FieldOriginKind(StrEnum):
    """Origins from which a field observation can be recovered."""

    CLASS_ASSIGNMENT = "class_assignment"
    CLASS_ANNOTATION = "class_annotation"
    DATACLASS_FIELD = "dataclass_field"
    INIT_ASSIGNMENT = "init_assignment"

    @property
    def declares_record_field(self) -> bool:
        """Whether this origin declares a field of a record class (not an assignment or init)."""

        return self in (
            FieldOriginKind.DATACLASS_FIELD,
            FieldOriginKind.CLASS_ANNOTATION,
        )


@dataclass(frozen=True)
class StructuralObservationTemplate(StructuralObservationCarrier, ABC):
    """Base class for shapes that project to structural observations."""

    file_path: str
    OBSERVATION_KIND: ClassVar[ObservationKind]

    @property
    def observation_kind(self) -> ObservationKind:
        return type(self).OBSERVATION_KIND

    @property
    @abstractmethod
    def observation_execution_level(self) -> StructuralExecutionLevel:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_line(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def owner_symbol(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def nominal_witness(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def observed_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def fiber_key(self) -> str:
        raise NotImplementedError

    @property
    def structural_observation(self) -> StructuralObservation:
        return StructuralObservation(
            file_path=self.file_path,
            owner_symbol=self.owner_symbol,
            nominal_witness=self.nominal_witness,
            line=self.observation_line,
            observation_kind=self.observation_kind,
            execution_level=self.observation_execution_level,
            observed_name=self.observed_name,
            fiber_key=self.fiber_key,
        )

    @property
    def source_location(self) -> SourceLocation:
        """Project this structural observation onto its source coordinate."""

        return SourceLocation(
            self.file_path,
            self.observation_line,
            self.owner_symbol,
        )


class ExecutionLevelObservationMixin(ABC):
    execution_level: StructuralExecutionLevel
    observation_execution_level = AliasProperty[StructuralExecutionLevel](
        "execution_level"
    )


class StaticExecutionLevelMixin(ABC):
    OBSERVATION_EXECUTION_LEVEL: ClassVar[StructuralExecutionLevel]

    @property
    def observation_execution_level(self) -> StructuralExecutionLevel:
        return type(self).OBSERVATION_EXECUTION_LEVEL


class FunctionBodyExecutionMixin(StaticExecutionLevelMixin):
    OBSERVATION_EXECUTION_LEVEL = StructuralExecutionLevel.FUNCTION_BODY


class ModuleBodyExecutionMixin(StaticExecutionLevelMixin):
    OBSERVATION_EXECUTION_LEVEL = StructuralExecutionLevel.MODULE_BODY


class LineObservationMixin(ABC):
    line: int
    observation_line = AliasProperty[int]("line")


class LinenoObservationMixin(ABC):
    lineno: int
    observation_line = AliasProperty[int]("lineno")


class SymbolOwnerMixin(ABC):
    symbol: str

    @property
    def owner_symbol(self) -> str:
        return cast(Any, self).symbol


class SymbolNominalWitnessMixin(ABC):
    @property
    def nominal_witness(self) -> str:
        return cast(Any, self).symbol


class LineSymbolObservationMixin(
    LineObservationMixin,
    SymbolOwnerMixin,
    SymbolNominalWitnessMixin,
    ABC,
):
    file_path: str


FunctionBodyLineSymbolObservationMixin = CompositeClassSpec(
    "FunctionBodyLineSymbolObservationMixin",
    (FunctionBodyExecutionMixin, LineSymbolObservationMixin, ABC),
).build(__name__)


ModuleBodyLineSymbolObservationMixin = CompositeClassSpec(
    "ModuleBodyLineSymbolObservationMixin",
    (ModuleBodyExecutionMixin, LineSymbolObservationMixin, ABC),
).build(__name__)


class FunctionBodyLinenoSymbolObservationMixin(
    FunctionBodyExecutionMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    ABC,
):
    file_path: str


@dataclass(frozen=True)
class FunctionBodyCallLikeShape(
    FunctionBodyLinenoSymbolObservationMixin,
    StructuralObservationTemplate,
    ABC,
):
    """Shared function-body call projection fields for call-like observations."""

    class_name: str | None
    function_name: str | None
    lineno: int
    value_fingerprint: tuple[str, ...]
    source_arity: int
    source_name: str | None
    identity_field_names: tuple[str, ...]

    @property
    def owner_prefix(self) -> str:
        owner = self.function_name or "<module>"
        if self.class_name:
            owner = f"{self.class_name}.{owner}"
        return owner

    @property
    def nominal_witness(self) -> str:
        return self.class_name or self.function_name or cast(Any, self).symbol


class FunctionNameOwnerMixin(ABC):
    function_name: str
    owner_symbol = AliasProperty[str]("function_name")


class ClassNameNominalWitnessMixin(ABC):
    class_name: str
    nominal_witness = AliasProperty[str]("class_name")


class FunctionNameNominalWitnessMixin(ABC):
    function_name: str
    nominal_witness = AliasProperty[str]("function_name")


class FunctionNameObservedNameMixin(ABC):
    function_name: str
    observed_name = AliasProperty[str]("function_name")


@dataclass(frozen=True)
class FieldObservation(
    ExecutionLevelObservationMixin,
    LinenoObservationMixin,
    SymbolOwnerMixin,
    ClassNameNominalWitnessMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.FIELD
    class_name: str
    field_name: str
    lineno: int
    execution_level: StructuralExecutionLevel
    origin_kind: FieldOriginKind
    is_dataclass_family: bool
    value_fingerprint: str | None = None
    annotation_text: str | None = None
    annotation_fingerprint: str | None = None

    @property
    def symbol(self) -> str:
        return f"{self.class_name}.{self.field_name}"

    observed_name: ClassVar[AliasProperty[str]] = AliasProperty("field_name")
    fiber_key: ClassVar[AliasProperty[str]] = AliasProperty("field_name")


@dataclass(frozen=True)
class LiteralDispatchObservation(
    ExecutionLevelObservationMixin,
    LineObservationMixin,
    SymbolOwnerMixin,
    DispatchAxisExpression,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.LITERAL_DISPATCH
    line: int
    symbol: str
    axis_fingerprint: str
    literal_cases: tuple[str, ...]
    literal_kind: LiteralKind
    execution_level: StructuralExecutionLevel
    branch_lines: tuple[int, ...] = ()
    scope_owner: str | None = None

    @property
    def nominal_witness(self) -> str:
        return self.scope_owner or self.symbol

    observed_name: ClassVar[AliasProperty[str]] = AliasProperty(
        "dispatch_axis_expression"
    )

    @property
    def fiber_key(self) -> str:
        return f"{self.literal_kind}:{self.axis_fingerprint}"


@dataclass(frozen=True)
class SentinelTypeObservation(
    ModuleBodyLineSymbolObservationMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.SENTINEL_TYPE
    line: int
    symbol: str
    sentinel_name: str

    observed_name: ClassVar[AliasProperty[str]] = AliasProperty("sentinel_name")
    fiber_key: ClassVar[AliasProperty[str]] = AliasProperty("sentinel_name")


@dataclass(frozen=True)
class DynamicMethodInjectionObservation(
    FunctionBodyLineSymbolObservationMixin,
    StructuralObservationTemplate,
):
    OBSERVATION_KIND = ObservationKind.DYNAMIC_METHOD_INJECTION
    line: int
    symbol: str
    mutator_name: str

    observed_name: ClassVar[AliasProperty[str]] = AliasProperty("mutator_name")
    fiber_key: ClassVar[AliasProperty[str]] = AliasProperty("mutator_name")


@dataclass(frozen=True)
class BuilderCallShape(FunctionBodyCallLikeShape):
    OBSERVATION_KIND = ObservationKind.BUILDER_CALL
    callee_name: str
    field_names: tuple[str, ...]

    @property
    def symbol(self) -> str:
        return f"{self.owner_prefix}:{self.callee_name}"

    @property
    def is_nominal_owner_projection(self) -> bool:
        """Whether the nominal record already owns this projection site."""

        return self.class_name == self.callee_name

    observed_name: ClassVar[AliasProperty[str]] = AliasProperty("callee_name")

    @property
    def fiber_key(self) -> str:
        return f"{self.callee_name}:{self.field_names}:{self.value_fingerprint}"


@dataclass(frozen=True)
class ModuleExpressionShape:
    """One maximal expression computed only from imported-module attributes."""

    file_path: str
    lineno: int
    symbol: str
    expression_key: str
    expression_text: str


@dataclass(frozen=True)
class RestatingSite:
    """One place that writes a fact again.

    `group` names what one finding covers inside `symbol` (the dispatch subject,
    the receiver of a probe, the declared name); `detail` is what the site says.
    """

    file_path: str
    lineno: int
    symbol: str
    group: str
    detail: tuple[str, ...]

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.lineno, self.symbol)


@dataclass(frozen=True)
class SubjectReading:
    """What a type test's subject is before the test: a value the code already
    holds as itself, so dispatch on it would answer what the test asks."""

    def restated(self, names: tuple[str, ...]) -> tuple[str, ...]:
        """The tested classes whose membership the test restates."""

        return names


@dataclass(frozen=True)
class ForeignSubject(SubjectReading):
    """A parameter annotated `object`: the value is foreign until this test
    decodes it (`__eq__(self, other: object)`, a codec's input). Nothing in the
    language derives its membership, so the test is the decoder and restates nothing."""

    def restated(self, names: tuple[str, ...]) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class TypeTestShape(RestatingSite):
    """One `isinstance`/`issubclass` site or class pattern whose classes resolved.

    `type_names` (the site's detail) are qualified class names, each reached
    through an import, a module declaration or a builtin; `subject` (its group)
    is the dump of the tested expression, so sites testing the same value can be
    told from sites testing a part of it.
    """

    reading: SubjectReading = SubjectReading()
    type_names = AliasProperty[tuple[str, ...]]("detail")
    subject = AliasProperty[str]("group")

    @property
    def owners(self) -> frozenset[str]:
        """The modules the tested classes are declared in."""

        return frozenset(name.rpartition(".")[0] for name in self.type_names)

    def declared_type_names(self, scope: DeclaredClassScope) -> tuple[str, ...]:
        """The tested classes declared in the scan's modules: the syntax already
        proves each is a class."""

        return tuple(name for name in self.type_names if scope.declares_module_of(name))

    def own_type_names(self, scope: DeclaredClassScope) -> tuple[str, ...]:
        """The declared tested classes whose membership this test restates."""

        return self.reading.restated(self.declared_type_names(scope))

    def is_foreign_to(self, scanned_module_names: Sequence[str]) -> bool:
        """Whether no tested class is declared in the scan."""

        return not self.declared_type_names(DeclaredClassScope(tuple(scanned_module_names)))


@dataclass(frozen=True)
class ClassIdentityTestShape(TypeTestShape):
    """`x is C` or `x in (A, B)`: a type test only when each name is a family class.

    The syntax does not prove the name is a class (`mode is DEFAULT_MODE` is not
    a type test), and a class proves no type test either: identity on a class
    that no scanned class extends and that extends no scanned class (a sentinel
    such as `class MISSING: pass`) asks which object the value is, not which
    family member. So a name counts only when the scan declares it as a class in
    one of its own families (`DeclaredClassScope.family_class_symbols`), and a
    foreign name is never proved one.
    """

    def declared_type_names(self, scope: DeclaredClassScope) -> tuple[str, ...]:
        return tuple(
            name for name in self.type_names if name in scope.family_class_symbols
        )

    def is_foreign_to(self, scanned_module_names: Sequence[str]) -> bool:
        return False


@dataclass(frozen=True)
class DeclaredClassScope:
    """What the scan declares: its module names and, when indexed, the symbols of
    its family classes, those with a scanned base or a scanned subclass."""

    module_names: tuple[str, ...]
    family_class_symbols: frozenset[str] = frozenset()

    def declares_module_of(self, name: str) -> bool:
        """Whether a scanned module is, contains, or sits inside the module of `name`."""

        owner = name.rpartition(".")[0]
        return any(
            owner == scanned
            or owner.startswith(f"{scanned}.")
            or scanned.startswith(f"{owner}.")
            for scanned in self.module_names
        )


@dataclass(frozen=True)
class TypeTestModule:
    """One module's type tests, kept even when empty so a scan knows its modules.
    Whether each tested class is declared in the scan is decided against the whole scan."""

    module_name: str
    sites: tuple[TypeTestShape, ...]


@dataclass(frozen=True)
class ForeignFamilySiteShape:
    """One site whose tested classes are exactly one declared family's members."""

    file_path: str
    lineno: int
    symbol: str
    family: str


@dataclass(frozen=True)
class StaticProjectionSiteShape(ForeignFamilySiteShape):
    """A call of a static projection class that its adapter family answers."""

    projector: str


@dataclass(frozen=True)
class ForeignFamilySiteModule:
    """One module's sites of one family-routing shape, kept even when empty."""

    module_name: str
    sites: tuple[ForeignFamilySiteShape, ...]


@dataclass(frozen=True)
class RegistrationShape:
    """Normalized record for one recovered manual registration site."""

    file_path: str
    lineno: int
    registry_name: str
    registered_class: str
    key_fingerprint: str
    key_expression: str
    constructors: ClassVar[ConstructorVariantCatalog] = ConstructorVariantCatalog(
        (
            ConstructorVariantSpec(
                "from_registration_call",
                (
                    "parsed_module",
                    "node",
                    "registry_name",
                    "registered_class",
                    "key_fingerprint",
                ),
                parameter_fields=(
                    "registry_name",
                    "registered_class",
                    "key_fingerprint",
                ),
                derived_fields=(
                    ConstructorDerivedField(
                        "file_path", lambda bound: bound["parsed_module"].file_path
                    ),
                    ConstructorDerivedField(
                        "lineno", lambda bound: bound["node"].lineno
                    ),
                    ConstructorDerivedField(
                        "key_expression",
                        lambda bound: ast.unparse(
                            bound["node"].args[1]
                            if len(bound["node"].args) >= 2
                            else bound["node"].args[0]
                        ),
                    ),
                ),
            ),
            ConstructorVariantSpec(
                "from_decorator",
                (
                    "parsed_module",
                    "node",
                    "decorator",
                    "registry_name",
                    "key_fingerprint",
                ),
                parameter_fields=("registry_name", "key_fingerprint"),
                derived_fields=(
                    ConstructorDerivedField(
                        "file_path", lambda bound: bound["parsed_module"].file_path
                    ),
                    ConstructorDerivedField(
                        "lineno", lambda bound: bound["node"].lineno
                    ),
                    ConstructorDerivedField(
                        "registered_class", lambda bound: bound["node"].name
                    ),
                    ConstructorDerivedField(
                        "key_expression",
                        lambda bound: ast.unparse(
                            bound["decorator"].args[1]
                            if len(bound["decorator"].args) >= 2
                            else ast.Constant(value=bound["node"].name)
                        ),
                    ),
                ),
            ),
            ConstructorVariantSpec(
                "from_factory_decorator",
                (
                    "parsed_module",
                    "node",
                    "registry_name",
                    "key_expression",
                    "key_fingerprint",
                ),
                parameter_fields=(
                    "registry_name",
                    "key_expression",
                    "key_fingerprint",
                ),
                derived_fields=(
                    ConstructorDerivedField(
                        "file_path", lambda bound: bound["parsed_module"].file_path
                    ),
                    ConstructorDerivedField(
                        "lineno", lambda bound: bound["node"].lineno
                    ),
                    ConstructorDerivedField(
                        "registered_class", lambda bound: bound["node"].name
                    ),
                ),
            ),
        )
    )

    @classmethod
    def from_assignment(
        cls,
        parsed_module: ParsedModule,
        node: ast.Assign,
        registry_name: str,
        key_fingerprint: str,
    ) -> RegistrationShape:
        if not isinstance(node.value, ast.Name):
            raise TypeError("Registration assignment value must be a class name")
        return cls(
            file_path=parsed_module.file_path,
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.value.id,
            key_fingerprint=key_fingerprint,
            key_expression=(
                ast.unparse(node.targets[0].slice)
                if isinstance(node.targets[0], ast.Subscript)
                else "..."
            ),
        )

    from_registration_call, from_decorator, from_factory_decorator = (
        constructors.derived_methods()
    )

    @property
    def symbol(self) -> str:
        return f"{self.registry_name}[...] = {self.registered_class}"


_PUBLIC_EXPORT_POLICY = PublicExportPolicy(
    module_name=__name__,
    include_enums=True,
    exclude_abstract=True,
    root_types=(StructuralObservationCarrier,),
)


__all__ = derive_public_exports(globals(), _PUBLIC_EXPORT_POLICY)
