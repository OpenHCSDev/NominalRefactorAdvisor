"""Repository-wide class-family indexing helpers.

This module builds a lightweight cross-module view of declared classes and
their resolved inheritance edges. The index is intentionally conservative:
it resolves only import patterns and base expressions that can be recovered
reliably from the local AST.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import inspect
import re
import sys
import typing
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import (
    Collection,
    Hashable,
    Mapping,
    Sequence,
)
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from enum import StrEnum
from functools import (
    cache,
    cached_property,
    singledispatchmethod,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    NamedTuple,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from nominal_refactor_advisor.annotation_semantics import (
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
)
from nominal_refactor_advisor.ast_field_decoders import (
    TYPE_PARAMETERS,
    declared_annotations,
    variadic_parameters,
)
from nominal_refactor_advisor.ast_projection import ExpressionReference
from nominal_refactor_advisor.ast_projection import (
    CallKeywords,
    KeyT,
    NoStatementValue,
    StatementValue,
    StatementValueQuestion,
    StatementValues,
    SuppliedValue,
    single_name_targets,
)
from nominal_refactor_advisor.ast_tools import (
    CompactModuleIdentity,
    CollectedFamily,
    LexicalSyntaxScope,
    ParsedModule,
    ParsedModuleFact,
    SourceModule,
    _walk_nodes,
    collect_family_items,
    statements_without_docstring,
)
from nominal_refactor_advisor.class_method_facets import (
    CompactClassMethod,
    _compact_class_methods,
    _is_classmethod,
)
from nominal_refactor_advisor.class_references import (
    ClassReferenceRole,
    CompactClassReference,
)
from nominal_refactor_advisor.class_symbols import (
    ClassSymbolResolutionAuthority,
    _UniqueKnownSymbolSuffixIndex,
    _referenced_bases,
    _terminal_reference_name,
    _unique_known_symbol_by_suffix,
    iter_class_definitions,
)
from nominal_refactor_advisor.class_syntax_facets import (
    CompactClassMemberDeclaration,
    CompactClassSyntaxFacets,
    CompactKeyedTableAxis,
    CompactNamedProjectionSurface,
    CompactSortedKeyCall,
    CompactSourceLocation,
    ModuleClassSyntaxFacets,
    ModuleClassSyntaxFacetsWithoutAutoregister,
    _class_nominal_reference_root_names,
    _class_reference_root_names,
    _compact_base_argument_sources,
    _compact_base_key_type_names,
    _compact_class_member_declarations,
    _compact_keyed_table_axes,
    _compact_named_projection_surfaces,
    _compact_sorted_key_calls,
    _trim_leading_docstring,
)
from nominal_refactor_advisor.collection_algebra import (
    IdentityHandleMultiplicityProjection,
    SequenceViews,
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from nominal_refactor_advisor.class_mro import ClassMroAuthority, DeclaredBases
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.foreign_family import FunctionDefinition
from nominal_refactor_advisor.product_declarations import (
    CompactProductAuthorityFailure,
    CompactProductAuthorityViolation,
    CompactProductDeclarationFailure,
    ProductDeclarationSite,
    _dataclass_decorator_failures,
)
from nominal_refactor_advisor.public_exports import (
    CompactModulePublicExportContract,
    CompactPublicNameExposure,
    module_public_export_contract,
)
from nominal_refactor_advisor.registration_facets import (
    CompactManualSelectorAxis,
    CompactManualSubclassRosterRoot,
    _autoregister_registry_key_attr_name,
    _compact_manual_selector_axes,
    _compact_manual_subclass_roster_roots,
    _compact_predicate_selected_methods,
    _direct_class_assignments,
    _is_registration_authority,
    _string_constant,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.source_entry import ImportedSourceModuleEntryPremise
from nominal_refactor_advisor.value_expression import (
    LexicalValueReference,
    MappingKeyResolverABC,
)
from nominal_refactor_advisor.declaration_dependencies import (
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.descriptor_algebra import AliasProperty
from nominal_refactor_advisor.enum_semantics import (
    PYTHON_ENUM_BASE_AUTHORITY,
    NoEnumDeclaration,
    PythonEnumDeclarationFacet,
)
from nominal_refactor_advisor.export_tools import PYTHON_PUBLIC_EXPORT_ASSIGNMENT
from nominal_refactor_advisor.lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from nominal_refactor_advisor.native_declarations import (
    DataclassRuntimeDeclaration as DataclassRuntimeDeclaration,
    NativeDeclaration,
    QualifiedDeclaration,
)
from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex
from nominal_refactor_advisor.native_reference import NativeReferenceEnvironment
from nominal_refactor_advisor.product_flow import (
    SourceProductFlowProjection,
    source_product_flow_projection,
)
from nominal_refactor_advisor.class_creation_store import CreatedClassStore
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from nominal_refactor_advisor.semantic_algebra import DirectedGraph
from nominal_refactor_advisor.source_geometry import (
    ClassHeaderSourceSpan as ClassHeaderSourceSpan,
)


class ClassRegistryFacts(NamedTuple):
    """What one class statement says about registration, read from it and its module.

    `metaclass_references` and `base_references` are the `metaclass=` keyword
    and the bases, each with the binding its root name has where the class
    is declared. `stores_created_class` is whether, used as a metaclass, its
    creation path stores the class it creates into a mapping;
    `stores_subclass` is whether its `__init_subclass__` stores the subclass
    (`CreatedClassStore`). No class is recognized by its name, and no
    library's attribute names are consulted.
    """

    metaclass_references: tuple[CompactNominalReference, ...] = ()
    base_references: tuple[CompactNominalReference, ...] = ()
    stores_created_class: bool = False
    stores_subclass: bool = False

    @classmethod
    def of_class(
        cls,
        node: ast.ClassDef,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
        module: ast.Module,
    ) -> ClassRegistryFacts:
        creation = CreatedClassStore.of(node, module)
        return cls(
            metaclass_references=tuple(
                module_binding_snapshot.reference_for(parts)
                for keyword in node.keywords
                if keyword.arg == "metaclass"
                if (
                    parts := ExpressionReference.decoded(
                        ClassSymbolResolutionAuthority.reference_node(keyword.value)
                    ).attribute_chain
                )
                is not None
            ),
            base_references=_compact_base_references(node, module_binding_snapshot),
            stores_created_class=creation.stores_created_class(),
            stores_subclass=creation.stores_subclass(),
        )


class NotStandardDataclass(ValueError):
    """A class that is not a standard `dataclasses.dataclass` declaration."""


class CompactDataclassFacet(ABC):
    """A class's dataclass facet: a dataclass-like declaration, or none."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...]

    def with_settled_field_declarations(self, index: FieldDeclarationIndex) -> Self:
        settled = tuple(member.settled(index) for member in self.fields)
        if settled == self.fields:
            return self
        return replace(self, fields=settled)

    @abstractmethod
    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        """This standard dataclass declaration, or NotStandardDataclass."""
        raise NotImplementedError

    @property
    @abstractmethod
    def generates_construction(self) -> bool:
        """Whether a dataclass-like decorator builds the class's `__init__` from
        its fields, so an added field changes how the class is constructed."""
        raise NotImplementedError

    @abstractmethod
    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        """An ancestor's dataclass declaration refuses a heir that adds the decorator."""
        raise NotImplementedError

    @abstractmethod
    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        """What this class contributes as the single direct base of a product."""
        raise NotImplementedError

    @abstractmethod
    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactDataclassDeclaration(CompactDataclassFacet):
    """One dataclass-like declaration, including fail-closed unknown semantics."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...] = ()

    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        if (
            self.runtime_declaration is None
            or not self.runtime_declaration.is_dataclass_decorator
        ):
            raise NotStandardDataclass(reason)
        return self

    @property
    def generates_construction(self) -> bool:
        return True

    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        raise ValueError(f"`{heir_qualname}` inherits dataclass fields")

    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        return composer.compose(
            base_symbol, self, pending_symbols | frozenset((heir.symbol,))
        )

    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        return composer.resolve(class_symbol, indexed_class, self)


@dataclass(frozen=True)
class NoDataclassDeclaration(CompactDataclassFacet):
    """A class no dataclass-like decorator declares: no runtime, no fields."""

    runtime_declaration = None
    fields = ()
    failures = ()

    def require_standard(self, reason: str) -> CompactDataclassDeclaration:
        raise NotStandardDataclass(reason)

    @property
    def generates_construction(self) -> bool:
        return False

    def refuse_inherited_fields(self, heir_qualname: str) -> None:
        return None

    def base_lineage_in(
        self,
        composer: _ProductLineageComposer,
        base_symbol: str,
        heir: CompactIndexedClass,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        return _CompactProductLineage(
            (),
            (
                composer.failure(
                    heir, CompactProductAuthorityViolation.NON_DATACLASS_BASE
                ),
            ),
        )

    def product_resolution_in(
        self,
        composer: _ProductLineageComposer,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
    ) -> CompactProductAuthorityResolution:
        return AbsentCompactProductAuthority(class_symbol)


@dataclass(frozen=True)
class ClassDeclaration(QualifiedDeclaration):
    """Source-form-independent identity shared by repository class indexes."""

    symbol: str
    module_name: str
    qualname: str
    simple_name: str
    file_path: str
    line: int
    declared_base_names: tuple[str, ...]
    resolved_base_symbols: tuple[str, ...] = field(default=(), kw_only=True)
    dataclass_declaration: CompactDataclassFacet = field(
        default_factory=NoDataclassDeclaration,
        kw_only=True,
    )
    class_decorators_are_promotion_safe: bool = field(default=True, kw_only=True)
    registry_facts: ClassRegistryFacts = field(
        default=ClassRegistryFacts(), kw_only=True
    )
    declares_autoregister_meta: bool = field(default=False, kw_only=True)
    """Whether its metaclass registers the classes it creates; set by the index."""

    qualified_name = AliasProperty[str]("symbol")

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> Self:
        if resolved_base_symbols == self.resolved_base_symbols:
            return self
        return replace(self, resolved_base_symbols=resolved_base_symbols)

    def with_registering_metaclass(
        self, authority: RegisteringMetaclassAuthority[Any]
    ) -> Self:
        registers = authority.registers_remembered(self)
        if registers == self.declares_autoregister_meta:
            return self
        return replace(self, declares_autoregister_meta=registers)


@dataclass(frozen=True)
class CompactClassValueConstruction:
    """One class-owned construction of a nominal value declaration."""

    assigned_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    line: int


class CompactNominalReference(ABC):
    """One source reference; its subtype says whether a declaration-time binding roots it."""

    source_parts: tuple[str, ...]

    @staticmethod
    def bound_as(
        parts: tuple[str, ...], binding: CompactNominalBinding | None
    ) -> CompactNominalReference:
        """The binding lookup's answer, read once, picks the reference kind."""

        if binding is None:
            return UnboundNominalReference(parts)
        return BoundNominalReference(parts, binding)

    @property
    @abstractmethod
    def resolved_parts(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def qualified_name(self) -> str:
        return ".".join(self.resolved_parts)

    @property
    def simple_name(self) -> str:
        return self.resolved_parts[-1]

    @abstractmethod
    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        """The bound name, or what the module's star imports settle the spelling to."""
        raise NotImplementedError

    @abstractmethod
    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_bound_name(self, qualified_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        """Whether a binding proves this reference names exactly that builtin."""
        raise NotImplementedError

    @property
    @abstractmethod
    def builtin_qualified_name(self) -> str:
        """The qualified name, reading an unbound builtin spelling as `builtins.X`."""
        raise NotImplementedError

    @abstractmethod
    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        """The known class symbol when the resolved candidate itself is not one."""
        raise NotImplementedError


@dataclass(frozen=True)
class BoundNominalReference(CompactNominalReference):
    source_parts: tuple[str, ...]
    root_binding: CompactNominalBinding

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        return (
            *self.root_binding.qualified_name.split("."),
            *self.source_parts[1:],
        )

    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        bound.append(self)

    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        return self.qualified_name

    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        return other.has_bound_name(self.qualified_name)

    def has_bound_name(self, qualified_name: str) -> bool:
        return self.qualified_name == qualified_name

    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        return self.resolved_parts[0] == "builtins" and self.resolved_parts == expected_parts

    @property
    def builtin_qualified_name(self) -> str:
        return self.qualified_name

    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        if self.root_binding.kind.projects_as_import_alias:
            return unique_symbols_by_suffix.root_relative_match(candidate)
        return None


@dataclass(frozen=True)
class UnboundNominalReference(CompactNominalReference):
    """A reference no binding in scope roots: only its source spelling is known."""

    source_parts: tuple[str, ...]

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        return self.source_parts

    def sort_into(
        self,
        bound: list[BoundNominalReference],
        unbound: list[UnboundNominalReference],
    ) -> None:
        unbound.append(self)

    def settled_qualified_name(
        self, authority: RegisteringMetaclassAuthority[Any], module_name: str
    ) -> str | None:
        return authority.star_settled_name(self, module_name)

    def shares_binding_with(self, other: CompactNominalReference) -> bool:
        return False

    def has_bound_name(self, qualified_name: str) -> bool:
        return False

    def names_builtin(self, expected_parts: tuple[str, ...]) -> bool:
        return False

    @property
    def builtin_qualified_name(self) -> str:
        parts = self.source_parts
        if len(parts) == 1 and hasattr(builtins, parts[0]):
            return f"{builtins.__name__}.{parts[0]}"
        return self.qualified_name

    def known_symbol_beyond(
        self,
        candidate: str,
        module_name: str,
        known_symbols: Collection[str],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        module_local = ".".join((module_name, *self.source_parts))
        return module_local if module_local in known_symbols else None


MemberAnswerT = TypeVar("MemberAnswerT")


@dataclass(frozen=True)
class CompactCallableMemberDeclaration(ABC):
    """One `def` a class body binds: a method or a property, with what it declares it returns.

    A property's return annotation is the type of reading the member; a
    method's is the type of calling it. A def with no return annotation
    declares neither.
    """

    name: str
    line: int

    @abstractmethod
    def answer(
        self,
        owner: CompactIndexedClass,
        answers: ClassMemberAnswersABC[MemberAnswerT],
    ) -> MemberAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class CompactAnnotatedProperty(CompactCallableMemberDeclaration):
    """`@property` or `@cached_property` with a return annotation: reading it has that type."""

    return_annotation_expression: str

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.annotated_property(owner, self)


@dataclass(frozen=True)
class CompactAnnotatedMethod(CompactCallableMemberDeclaration):
    """A method with a return annotation: calling it has that type."""

    return_annotation_expression: str

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.annotated_method(owner, self)


@dataclass(frozen=True)
class CompactUnannotatedCallable(CompactCallableMemberDeclaration):
    """A method or property that declares no return type."""

    def answer(self, owner: CompactIndexedClass, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.unannotated_callable(owner, self)


class ClassMemberAnswersABC(ABC, Generic[MemberAnswerT]):
    """What a consumer makes of each outcome of looking a member up through a class's MRO."""

    @abstractmethod
    def assigned(
        self, owner: CompactIndexedClass, declaration: CompactClassMemberDeclaration
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def annotated_property(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedProperty
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def annotated_method(
        self, owner: CompactIndexedClass, declaration: CompactAnnotatedMethod
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def unannotated_callable(
        self, owner: CompactIndexedClass, declaration: CompactUnannotatedCallable
    ) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def missing(self) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def ambiguous(self) -> MemberAnswerT:
        raise NotImplementedError

    @abstractmethod
    def open(self) -> MemberAnswerT:
        raise NotImplementedError


class ClassMemberLookup(ABC):
    """A member name looked up through a class's MRO: its one binding, or why there is none."""

    @abstractmethod
    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        raise NotImplementedError


@dataclass(frozen=True)
class AssignedClassMember(ClassMemberLookup):
    """The first class in the MRO binding the name assigns or annotates it in its body."""

    owner: CompactIndexedClass
    declaration: CompactClassMemberDeclaration

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.assigned(self.owner, self.declaration)


@dataclass(frozen=True)
class CallableClassMember(ClassMemberLookup):
    """The first class in the MRO binding the name defines it with `def`."""

    owner: CompactIndexedClass
    declaration: CompactCallableMemberDeclaration

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return self.declaration.answer(self.owner, answers)


class MissingClassMember(ClassMemberLookup):
    """No class of the MRO binds the name in its body."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.missing()


class AmbiguousClassMember(ClassMemberLookup):
    """The first class binding the name binds it more than once."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.ambiguous()


class OpenClassMember(ClassMemberLookup):
    """The class does not bind the name and its MRO is open, so no ancestor is known to."""

    def answer(self, answers: ClassMemberAnswersABC[MemberAnswerT]) -> MemberAnswerT:
        return answers.open()


@dataclass(frozen=True)
class CompactAnnotatedFunctionReturn:
    """A module-level `def` and the return type it declares."""

    name: str
    line: int
    return_annotation_expression: str


class CompactDataclassFieldRole(StrEnum):
    """Dataclass field kinds with declaration-owned product admissibility."""

    STORED_INIT = "stored_init", True, True, None, ()
    CLASS_VARIABLE = "class_variable", False, False, None, ("typing.ClassVar",)
    KEYWORD_ONLY_SENTINEL = (
        "keyword_only_sentinel",
        False,
        False,
        None,
        ("dataclasses.KW_ONLY",),
    )
    STORED_NON_INIT = (
        "stored_non_init",
        False,
        True,
        CompactProductAuthorityViolation.NON_INIT_FIELD,
        (),
    )
    INIT_ONLY = (
        "init_only",
        False,
        True,
        CompactProductAuthorityViolation.INIT_ONLY_FIELD,
        ("dataclasses.InitVar",),
    )
    UNRESOLVED = (
        "unresolved",
        False,
        False,
        CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE,
        (),
    )
    DECLARED_BY_CALL = (
        "declared_by_call",
        False,
        False,
        CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE,
        (),
    )
    """A value that calls a function the module cannot judge alone; the
    repository's field declarations settle it to a role."""

    contributes_stored_init_field: bool
    contributes_semantic_field: bool
    authority_violation: CompactProductAuthorityViolation | None
    annotation_qualified_names: tuple[str, ...]

    def __new__(
        cls,
        value: str,
        contributes_stored_init_field: bool,
        contributes_semantic_field: bool,
        authority_violation: CompactProductAuthorityViolation | None,
        annotation_qualified_names: tuple[str, ...],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.contributes_stored_init_field = contributes_stored_init_field
        member.contributes_semantic_field = contributes_semantic_field
        member.authority_violation = authority_violation
        member.annotation_qualified_names = annotation_qualified_names
        return member

    @classmethod
    def for_qualified_annotation(
        cls,
        qualified_name: str | None,
    ) -> Self | None:
        return next(
            (role for role in cls if qualified_name in role.annotation_qualified_names),
            None,
        )

    def declaration_call(
        self,
        statement: ast.AnnAssign,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
        class_aliases: dict[str, str],
        preceding_class_bound_names: frozenset[str],
    ) -> CompactFieldDeclarationCall | None:
        """The call a member declared by a call makes, kept for the repository to judge."""

        if self is not CompactDataclassFieldRole.DECLARED_BY_CALL or not isinstance(
            statement.value, ast.Call
        ):
            return None
        callee = _class_scope_qualified_import_name(
            module_binding_snapshot,
            class_aliases,
            statement.value.func,
            preceding_class_bound_names,
        )
        if callee is None:
            return None
        return CompactFieldDeclarationCall.of_call(callee, statement.value)

    def binding_of_later_fields(
        self, binding: DataclassInitBinding
    ) -> DataclassInitBinding:
        """How the fields after this member bind: a `KW_ONLY` sentinel makes
        them keyword-only; any other member leaves the binding as it was."""

        if self is CompactDataclassFieldRole.KEYWORD_ONLY_SENTINEL:
            return DataclassInitBinding.KEYWORD_ONLY
        return binding

    @property
    def has_known_field_position(self) -> bool:
        """Whether this member's place in the dataclass field order is known.

        A KW_ONLY sentinel turns every later field keyword-only, and an
        unresolved annotation may or may not declare a field.
        """

        return self not in (
            CompactDataclassFieldRole.KEYWORD_ONLY_SENTINEL,
            CompactDataclassFieldRole.UNRESOLVED,
            CompactDataclassFieldRole.DECLARED_BY_CALL,
        )


class DataclassInitBinding(StrEnum):
    """How a dataclass field binds as an `__init__` parameter.

    `kw_only` comes from the field's own `field(kw_only=...)`, else from a
    preceding `KW_ONLY` sentinel, else from the decorator's `kw_only=`. A value
    that is not a literal bool leaves the binding unknown.
    """

    POSITIONAL = "positional"
    KEYWORD_ONLY = "keyword_only"
    UNKNOWN = "unknown"

    @classmethod
    def of_keywords(
        cls, keywords: Iterable[ast.keyword], otherwise: DataclassInitBinding
    ) -> DataclassInitBinding:
        """The binding a call's `kw_only=` keyword declares, or `otherwise`."""

        values = tuple(keyword.value for keyword in keywords if keyword.arg == "kw_only")
        if not values:
            return otherwise
        if len(values) > 1 or not (
            isinstance(values[0], ast.Constant) and isinstance(values[0].value, bool)
        ):
            return cls.UNKNOWN
        return cls.KEYWORD_ONLY if values[0].value else cls.POSITIONAL


class CompactDataclassFieldDeclaration(NamedTuple):
    """One direct annotated member, its exact dataclass role and its binding,
    with the call its value makes while the repository has yet to judge it."""

    name: str
    line: int
    role: CompactDataclassFieldRole
    binding: DataclassInitBinding
    declaration_call: CompactFieldDeclarationCall | None = None

    def settled(self, index: FieldDeclarationIndex) -> CompactDataclassFieldDeclaration:
        """The member once the repository says whether its call declares a field."""

        if self.declaration_call is None:
            return self
        return self._replace(
            role=index.field_role(self.declaration_call), declaration_call=None
        )


@dataclass(frozen=True)
class CompactProductField:
    """One effective dataclass field retaining its nominal declaration source."""

    name: str
    role: CompactDataclassFieldRole
    binding: DataclassInitBinding
    declaring_class_symbol: str
    file_path: str
    line: int


@dataclass(frozen=True, kw_only=True)
class CompactProductAuthority(ProductDeclarationSite):
    """One linearly composed dataclass schema with source-visible fields."""

    effective_fields: tuple[CompactProductField, ...]

    @property
    def fields(self) -> tuple[CompactProductField, ...]:
        return tuple(
            field
            for field in self.effective_fields
            if field.role.contributes_stored_init_field
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def positional_parameter_names(self) -> tuple[str, ...]:
        """The generated `__init__`'s positional parameters, in order: every
        field not keyword-only, in field order. Raises when a field's binding
        is unknown."""

        unknown = tuple(
            field.name
            for field in self.fields
            if field.binding is DataclassInitBinding.UNKNOWN
        )
        if unknown:
            raise ValueError(
                f"`{self.class_symbol}` has field(s) {list(unknown)} whose "
                "keyword-only-ness is not a literal"
            )
        return tuple(
            field.name
            for field in self.fields
            if field.binding is DataclassInitBinding.POSITIONAL
        )


@dataclass(frozen=True)
class CompactProductAuthorityResolution(ABC):
    """Nominal result of composing one class into a product authority."""

    class_symbol: str

    @property
    @abstractmethod
    def authority(self) -> CompactProductAuthority | None:
        raise NotImplementedError


@dataclass(frozen=True)
class AbsentCompactProductAuthority(CompactProductAuthorityResolution):
    """A class declaration which does not claim dataclass product semantics."""

    @property
    def authority(self) -> None:
        return None


@dataclass(frozen=True)
class OpenCompactProductAuthority(CompactProductAuthorityResolution):
    """A dataclass-like declaration with unresolved product obligations."""

    failures: tuple[CompactProductAuthorityFailure, ...]

    @property
    def authority(self) -> None:
        return None


@dataclass(frozen=True)
class ResolvedCompactProductAuthority(CompactProductAuthorityResolution):
    """A dataclass schema proven by a complete linear declaration chain."""

    resolved_authority: CompactProductAuthority

    @property
    def authority(self) -> CompactProductAuthority:
        return self.resolved_authority


@dataclass(frozen=True)
class _CompactProductLineage:
    """Internal source-carrying schema before target-specific admissibility."""

    effective_fields: tuple[CompactProductField, ...]
    failures: tuple[CompactProductAuthorityFailure, ...]


@dataclass(frozen=True)
class NominalReferences:
    """References split into those a declaration-time binding roots and the rest."""

    bound: tuple[BoundNominalReference, ...]
    unbound: tuple[UnboundNominalReference, ...]

    @classmethod
    def of(cls, references: Iterable[CompactNominalReference]) -> NominalReferences:
        bound: list[BoundNominalReference] = []
        unbound: list[UnboundNominalReference] = []
        for reference in references:
            reference.sort_into(bound, unbound)
        return cls(tuple(bound), tuple(unbound))

    @property
    def bound_names(self) -> tuple[str, ...]:
        return tuple(reference.qualified_name for reference in self.bound)


@dataclass(frozen=True)
class CompactClassBindingFacets:
    """Class facts whose authority depends on one module binding snapshot."""

    base_references: tuple[CompactNominalReference, ...]
    base_references_are_complete: bool
    class_decorators_are_promotion_safe: bool
    dataclass_declaration: CompactDataclassFacet
    enum_declaration: PythonEnumDeclarationFacet

    @property
    def product_base_bindings_are_exact(self) -> bool:
        return not NominalReferences.of(
            reference
            for reference in self.base_references
            if ClassSymbolResolutionAuthority.establishes_nominal_family(
                ".".join(reference.source_parts)
            )
        ).unbound


@dataclass(frozen=True)
class CompactStarResolvedFacets:
    """A class's binding-sensitive facts as they are when the module's star
    imports bind none of the names the class reads.

    A module with `from x import *` cannot tell, alone, whether a later name is
    the star import's or its own: its projection keeps the conservative facts.
    These are the facts with every name in `root_names` taken as the module's
    own. The class index uses them when the repository's export contracts
    prove that every star origin excludes every one of those names.
    """

    root_names: frozenset[str]
    facets: CompactClassBindingFacets


@dataclass(frozen=True)
class CompactClassHeader(ClassDeclaration):
    """Class-index surface sufficient for inheritance reconstruction."""

    base_references: tuple[CompactNominalReference, ...]
    base_key_type_names: tuple[str | None, ...] = ()
    """The key type each base reference's subscript names (`Family[Key]`), aligned
    with `base_references`; the family index reads a registry family's key from it."""
    direct_base_count: int = 0
    base_references_are_complete: bool = False
    product_base_bindings_are_exact: bool = False
    is_final: bool = False
    mro_bases_are_static: bool = False
    star_resolution: CompactStarResolvedFacets | None = field(default=None, kw_only=True)
    """The facts past the module's star imports, when it has any; the index
    settles them once the repository's exports are known."""

    @property
    def base_bindings(self) -> NominalReferences:
        return NominalReferences.of(self.base_references)

    @property
    def base_resolution_is_complete(self) -> bool:
        """Return whether every domain-bearing base resolves in the compact graph."""

        return self.base_references_are_complete and len(
            self.resolved_base_symbols
        ) == declared_nominal_base_count(self)


@dataclass(frozen=True)
class CompactIndexedClass(CompactClassHeader):
    """AST-free class declaration used to reconstruct inheritance globally."""

    @property
    def bound_member_names(self) -> frozenset[str]:
        """Every name the class body binds, by assignment or by `def`."""

        return frozenset(
            (
                *(declaration.name for declaration in self.direct_member_declarations),
                *(declaration.name for declaration in self.callable_member_declarations),
            )
        )

    def with_settled_field_declarations(self, index: FieldDeclarationIndex) -> Self:
        """The class once the repository judges the calls its fields' values make."""

        settled = self.dataclass_declaration.with_settled_field_declarations(index)
        if settled is self.dataclass_declaration:
            return self
        return replace(self, dataclass_declaration=settled)

    def resolved_past_star_imports(
        self, exposure: RepositoryPublicExposureAuthority
    ) -> Self:
        """The class with its star-resolved facts when every star origin of its
        module excludes every name they read; otherwise its conservative facts."""

        resolution = self.star_resolution
        if resolution is None:
            return self
        if not all(
            exposure.star_imports_exclude(self.module_name, name)
            for name in resolution.root_names
        ):
            return replace(self, star_resolution=None)
        facets = resolution.facets
        return replace(
            self,
            base_references=facets.base_references,
            base_references_are_complete=facets.base_references_are_complete,
            product_base_bindings_are_exact=facets.product_base_bindings_are_exact,
            class_decorators_are_promotion_safe=(
                facets.class_decorators_are_promotion_safe
            ),
            dataclass_declaration=facets.dataclass_declaration,
            enum_declaration=facets.enum_declaration,
            star_resolution=None,
        )

    def with_registering_metaclass(
        self, authority: RegisteringMetaclassAuthority[Any]
    ) -> Self:
        """Its own metaclass, whether it is a registry family, and its registry family's key type."""

        registers = authority.registers_remembered(self)
        return replace(
            self,
            declares_autoregister_meta=registers,
            is_registration_authority=(
                self.is_registration_authority or authority.is_registry_family(self)
            ),
            keyed_family_key_type_name=authority.registry_family_key_type_name(self),
        )

    @property
    def declares_keyed_base(self) -> bool:
        """Whether a base names a key type (`Family[Key]`): a registry family's key, once the index resolves the base."""

        return any(name is not None for name in self.base_key_type_names)

    @property
    def own_dataclass_field_names(self) -> tuple[str, ...] | None:
        """The class's own dataclass fields in declaration order, ClassVars excluded;
        empty for a plain class, None when a field's position is unknown."""

        declaration = self.dataclass_declaration
        if not all(field.role.has_known_field_position for field in declaration.fields):
            return None
        return tuple(
            field.name
            for field in declaration.fields
            if field.role.contributes_semantic_field
        )

    direct_member_declarations: tuple[CompactClassMemberDeclaration, ...] = ()
    metaclass_names: tuple[str, ...] = ()
    class_keyword_names: tuple[str, ...] = ()
    class_header_is_reconstructible: bool = True
    keyed_family_key_type_name: str | None = None
    """The key type of the first subscripted base that is a registry family; set by the index."""
    end_line: int | None = None
    method_names: tuple[str, ...] = ()
    callable_member_declarations: tuple[CompactCallableMemberDeclaration, ...] = ()
    base_argument_sources: tuple[tuple[str, ...], ...] = ()
    """Each referenced base's subscript arguments (`Base[A, B]`), aligned with its base references."""
    type_parameter_names: tuple[str, ...] = ()
    """The class's own `class C[T]:` type parameters."""
    abstract_method_names: tuple[str, ...] = ()
    is_abstract: bool = False
    is_registration_authority: bool = False
    """Named or configured as a registration authority; the index adds a registering metaclass."""
    autoregister_registry_key_attr_name: str | None = None
    autoregister_registry_projection_names: tuple[str, ...] = ()
    keyed_registry_lookup_method_names: tuple[str, ...] = ()
    keyed_registry_reverse_lookup_method_names: tuple[str, ...] = ()
    predicate_selected_methods: tuple[tuple[int, str, str, str], ...] = ()
    enum_declaration: PythonEnumDeclarationFacet = NoEnumDeclaration()
    body_declarations: tuple[ClassBodyDeclarationDigest, ...] = ()

    @property
    def has_class_creation_hook(self) -> bool:
        """Class creation can invoke hooks declared as methods or assigned values."""

        return not CLASS_METHOD_OWNERSHIP_HOOK_NAMES.isdisjoint(
            (
                *self.method_names,
                *(
                    member.name
                    for member in self.direct_member_declarations
                    if member.expression is not None
                ),
            )
        )

    @property
    def direct_enum_member_names(self) -> tuple[str, ...]:
        """Return public members declared by a direct Python enum owner."""

        if not PYTHON_ENUM_BASE_AUTHORITY.matches_any(self.declared_base_names):
            return ()
        return PYTHON_ENUM_BASE_AUTHORITY.declared_member_names(
            (
                declaration.name,
                declaration.expression is not None,
            )
            for declaration in self.direct_member_declarations
        )

    @property
    def assignments_by_name(self) -> dict[str, str | None]:
        return {
            name: declaration.expression
            for name, declaration in self.direct_value_writes_by_name.items()
        }

    @cached_property
    def direct_value_writes_by_name(self) -> dict[str, CompactClassMemberDeclaration]:
        """Latest direct RHS writes, retaining their original declaration rows."""
        writes: dict[str, CompactClassMemberDeclaration] = {}
        for declaration in self.direct_member_declarations:
            declaration.value.record_write(writes, declaration.name, declaration)
        return writes

    @cached_property
    def direct_annotation_writes_by_name(
        self,
    ) -> dict[str, CompactClassMemberDeclaration]:
        """Annotation writes are independent of subsequent unannotated values."""
        return {
            declaration.name: declaration
            for declaration in self.direct_member_declarations
            if declaration.annotation_expression is not None
        }

    @cached_property
    def direct_declared_member_names(self) -> frozenset[str]:
        return frozenset(
            declaration.name for declaration in self.direct_member_declarations
        )

    def constant_string_assignment(self, name: str) -> str | None:
        declaration = self.direct_value_writes_by_name.get(name)
        return None if declaration is None else declaration.value.constant_string

    @property
    def direct_assignment_expressions(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(self.assignments_by_name.items())

    @property
    def declared_member_lines_by_name(self) -> dict[str, int]:
        lines: dict[str, int] = {}
        for declaration in self.direct_member_declarations:
            lines.setdefault(declaration.name, declaration.line)
        return lines

    @property
    def direct_value_constructions(self) -> tuple[CompactClassValueConstruction, ...]:
        return tuple(
            CompactClassValueConstruction(
                assigned_name=declaration.name,
                constructor_name=declaration.constructor_name,
                keyword_names=declaration.constructor_keyword_names,
                line=declaration.line,
            )
            for declaration in self.direct_value_writes_by_name.values()
            if declaration.constructor_name is not None
        )

    @property
    def direct_constant_string_assignments(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (declaration.name, declaration.value.constant_string)
            for declaration in self.direct_value_writes_by_name.values()
            if declaration.value.constant_string is not None
        )

    @property
    def direct_non_none_assignment_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            declaration.name
            for declaration in self.direct_value_writes_by_name.values()
            if not declaration.value.value_is_none_literal
        )


class StarNameSettlement(ABC):
    """What the star imports seen so far (latest first) do to one module binding."""

    @staticmethod
    def unshadowed(binding: str | None) -> StarNameSettlement:
        """The ast-side lookup, read once: the module's own binding, if it has one."""

        if binding is None:
            return NameUnbound()
        return NameUntouched(binding)

    @abstractmethod
    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        """Fold in an earlier star import; only an untouched name lets it decide."""
        raise NotImplementedError

    @abstractmethod
    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        raise NotImplementedError


class UntouchedSettlement(StarNameSettlement):
    """No star import seen so far exposes the name: an earlier one may still decide."""

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return origin.settlement_in(authority, name, self)


@dataclass(frozen=True)
class NameUntouched(UntouchedSettlement):
    """No star import exposes the name: the module's own binding stands."""

    own_binding: str

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        settled.append((key, self.own_binding))


class NameUnbound(UntouchedSettlement):
    """No star import exposes the name and the module binds it nowhere else."""

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        unsettled.append(key)


class NameUncertain(StarNameSettlement):
    """A star import whose exports are unknown may rebind the name."""

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return self

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        unsettled.append(key)


@dataclass(frozen=True)
class NameSettled(StarNameSettlement):
    """The last star import exposing the name rebinds it to this binding."""

    binding: str

    def or_settled_by(
        self,
        origin: CompactModuleStarImportOrigin,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
    ) -> StarNameSettlement:
        return self

    def sort_into(
        self,
        key: KeyT,
        settled: list[tuple[KeyT, str]],
        unsettled: list[KeyT],
    ) -> None:
        settled.append((key, self.binding))


@dataclass(frozen=True)
class StarSettlements(Generic[KeyT]):
    """Keyed names split into those the star imports settle to a binding and the rest."""

    settled: tuple[tuple[KeyT, str], ...]
    unsettled: tuple[KeyT, ...]

    @classmethod
    def keyed(
        cls, pairs: Iterable[tuple[KeyT, StarNameSettlement]]
    ) -> StarSettlements[KeyT]:
        settled: list[tuple[KeyT, str]] = []
        unsettled: list[KeyT] = []
        for key, settlement in pairs:
            settlement.sort_into(key, settled, unsettled)
        return cls(tuple(settled), tuple(unsettled))


class CompactModuleStarImportOrigin(ABC):
    """One module-scope star import; its subtype says whether its module resolved."""

    @staticmethod
    def resolved_as(module_name: str | None) -> CompactModuleStarImportOrigin:
        """The import resolver's answer, read once, picks the origin kind."""

        if module_name is None:
            return UnresolvedStarImportOrigin()
        return ResolvedStarImportOrigin(module_name)

    @abstractmethod
    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        raise NotImplementedError

    @abstractmethod
    def within(self, module_names: Collection[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        """An unknown origin in a consumer that does not close the name over."""
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedStarImportOrigin(CompactModuleStarImportOrigin):
    module_name: str

    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        return authority.exposure_for(self.module_name, binding_name)

    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        exposure = self.exposure_in(authority, name)
        if exposure.proves_public_exposure:
            origin_projection = authority.projections_by_module_name[self.module_name]
            return NameSettled(
                dict(origin_projection.import_aliases).get(
                    name, f"{self.module_name}.{name}"
                )
            )
        if exposure.introduces_uncertainty:
            return NameUncertain()
        return untouched

    def within(self, module_names: Collection[str]) -> bool:
        return self.module_name in module_names

    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        return False


@dataclass(frozen=True)
class UnresolvedStarImportOrigin(CompactModuleStarImportOrigin):
    """A star import whose module the repository cannot name: it decides nothing."""

    def exposure_in(
        self, authority: RepositoryPublicExposureAuthority, binding_name: str
    ) -> CompactPublicNameExposure:
        return CompactPublicNameExposure.UNRESOLVED

    def settlement_in(
        self,
        authority: CompactRepositoryPublicExposureIndex,
        name: str,
        untouched: UntouchedSettlement,
    ) -> StarNameSettlement:
        return NameUncertain()

    def within(self, module_names: Collection[str]) -> bool:
        return False

    def leaves_open(self, consumer: CompactModuleClassHeader, name: str) -> bool:
        return consumer.public_export_contract.exposure_for(name).blocks_closed_boundary


class RepositoryPublicExposureAuthority(ABC):
    """Repository-owned proof that star imports preserve one local binding."""

    @abstractmethod
    def contains_module(self, module_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        raise NotImplementedError

    def star_imports_exclude(self, module_name: str, binding_name: str) -> bool:
        """Prove every star-import origin excludes the queried binding."""

        return self.contains_module(module_name) and all(
            not origin.exposure_in(self, binding_name).blocks_closed_boundary
            for origin in self.star_import_origins_for(module_name)
        )


@dataclass(frozen=True)
class CompactModuleClassHeader(CompactModuleIdentity):
    """Module namespace and class surface required by the compact family index."""

    import_aliases: tuple[tuple[str, str], ...]
    public_export_contract: CompactModulePublicExportContract
    star_import_origins: tuple[CompactModuleStarImportOrigin, ...]
    classes: tuple[CompactIndexedClass, ...]
    field_declaration_wrappers: tuple[CompactFieldDeclarationWrapper, ...] = ()

    @staticmethod
    @ScanCache.cached
    def from_modules(
        modules: tuple[ParsedModule, ...],
    ) -> tuple[CompactModuleClassHeader, ...]:
        """Join original declarations with the repository's namespace proof."""
        return repository_settled_projections(
            tuple(CompactModuleClassHeader.from_module(module) for module in modules)
        )

    @staticmethod
    def from_module(parsed_module: ParsedModule) -> CompactModuleClassHeader:
        """Acquire complete class declarations without unrelated analysis surfaces:
        a fact of the parse, so every scan and every goal state reads an unchanged
        module's declarations once."""
        return parsed_module.fact(ModuleClassHeader).header

    @classmethod
    def declared_in(
        cls, parsed_module: ParsedModule, *, include_body_facets: bool
    ) -> Self:
        """The module's namespace, classes and field-declaration wrappers, from one
        binding pass: each class reads the bindings before it, each wrapper the
        module's final ones."""
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        binding_snapshots = _module_nominal_binding_snapshots(
            ModuleNominalBindingAuthority(parsed_module),
            tuple(node.lineno for _qualname, node in indexed_class_nodes),
            include_final=True,
        )
        return cls(
            module_name=parsed_module.module_name,
            file_path=parsed_module.file_path,
            import_aliases=tuple(sorted(module_import_aliases(parsed_module).items())),
            public_export_contract=module_public_export_contract(parsed_module),
            star_import_origins=module_star_import_origins(parsed_module),
            classes=_compact_indexed_classes(
                parsed_module,
                indexed_class_nodes,
                binding_snapshots,
                include_body_facets=include_body_facets,
            ),
            field_declaration_wrappers=module_field_declaration_wrappers(
                parsed_module, binding_snapshots[None]
            ),
        )

ClassProjectionT = TypeVar("ClassProjectionT", bound=CompactModuleClassHeader)


@dataclass(frozen=True)
class ModuleClassHeader(ParsedModuleFact):
    """One module's complete class declarations, body facets included. A fact of
    the parse, like ModuleClassHeaderCore: a goal run's states hand unchanged
    parses from stage to stage, so each unchanged module is declared once per run
    instead of once per scan."""

    header: CompactModuleClassHeader

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            CompactModuleClassHeader.declared_in(parsed_module, include_body_facets=True)
        )


@dataclass(frozen=True)
class ModuleClassHeaderCore(ParsedModuleFact):
    """One module's namespace and class headers, the part of its projection a
    class-family index reads. A fact of the parse, so a plan's snapshots, which
    hand unchanged parses from stage to stage, project each unchanged module once."""

    projection: CompactModuleClassProjection

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        """The header-core view of the class projection family, through the
        parse's family batch: a later process over unchanged source reads it."""

        (projection,) = parsed_module.family_batch(
            CompactModuleClassProjectionFamily, HEADER_CORE_DEMAND
        ).items
        return cls(projection)


@dataclass(frozen=True)
class CompactRepositoryPublicExposureIndex(RepositoryPublicExposureAuthority):
    """Derive transitive public exposure from module-owned namespace contracts."""

    module_projections: tuple[CompactModuleClassHeader, ...]

    def star_settlement(
        self, module_name: str, name: str, unshadowed: str | None
    ) -> StarNameSettlement:
        """What a name a star import may rebind is bound to, as the repository's exports decide it.

        `unshadowed` is the module's own binding, star imports aside. No star
        import exposing the name leaves that binding; the last star import
        that exposes it rebinds it to that module's binding; an origin whose
        exports are unknown decides nothing.
        """

        settlement = StarNameSettlement.unshadowed(unshadowed)
        if self.star_imports_exclude(module_name, name):
            return settlement
        if not self.contains_module(module_name):
            return NameUncertain()
        for origin in reversed(self.star_import_origins_for(module_name)):
            settlement = settlement.or_settled_by(origin, self, name)
        return settlement

    @cached_property
    def module_projection_multiplicity(
        self,
    ) -> IdentityHandleMultiplicityProjection[str, CompactModuleClassHeader]:
        return UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            self.module_projections,
            lambda projection: projection.module_name,
        )

    @cached_property
    def projections_by_module_name(self) -> dict[str, CompactModuleClassHeader]:
        return self.module_projection_multiplicity.unambiguous_declarations_by_handle

    def contains_module(self, module_name: str) -> bool:
        return module_name in self.projections_by_module_name

    @cached_property
    def named_reexports_by_target_symbol(
        self,
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for projection in self.module_projections:
            for local_name, target_symbol in projection.import_aliases:
                grouped[target_symbol].append((projection.module_name, local_name))
        return {
            symbol: tuple(dict.fromkeys(reexports))
            for symbol, reexports in grouped.items()
        }

    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        pending = deque(((module_name, binding_name),))
        visited: set[tuple[str, str]] = set()
        unresolved = False
        while pending:
            current_module, current_name = pending.popleft()
            binding = current_module, current_name
            if binding in visited:
                continue
            visited.add(binding)
            projection = self.projections_by_module_name.get(current_module)
            if projection is None:
                unresolved = True
                continue
            exposure = projection.public_export_contract.exposure_for(current_name)
            if exposure.proves_public_exposure:
                return exposure
            if exposure.introduces_uncertainty:
                unresolved = True
            symbol = f"{current_module}.{current_name}"
            pending.extend(self.named_reexports_by_target_symbol.get(symbol, ()))
            if any(
                origin.leaves_open(consumer, current_name)
                for consumer in self.module_projections
                for origin in consumer.star_import_origins
            ):
                unresolved = True
        return (
            CompactPublicNameExposure.UNRESOLVED
            if unresolved
            else CompactPublicNameExposure.PRIVATE
        )

    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        projection = self.projections_by_module_name.get(module_name)
        return () if projection is None else projection.star_import_origins


@dataclass(frozen=True)
class CompactModuleClassProjection(
    CompactClassSyntaxFacets,
    CompactModuleClassHeader,
):
    """One module's class declarations and import aliases, without its AST."""

    sorted_key_calls: tuple[CompactSortedKeyCall, ...] = ()
    keyed_table_axes: tuple[CompactKeyedTableAxis, ...] = ()
    manual_selector_axes: tuple[CompactManualSelectorAxis, ...] = ()
    top_level_definitions: tuple[tuple[str, int], ...] = ()
    annotated_function_returns: tuple[CompactAnnotatedFunctionReturn, ...] = ()
    type_variable_names: tuple[str, ...] = ()
    """Module-level names bound to a `TypeVar`, `ParamSpec` or `TypeVarTuple` call."""
    repeated_keyed_family_roots: tuple[CompactRepeatedKeyedFamilyRoot, ...] = ()
    manual_subclass_roster_roots: tuple[CompactManualSubclassRosterRoot, ...] = ()
    latent_rosters: tuple[LatentRosterObservation, ...] = ()
    named_projection_surfaces: tuple[CompactNamedProjectionSurface, ...] = ()
    class_methods: tuple[CompactClassMethod, ...] = ()
    class_references: tuple[CompactClassReference, ...] = ()

    def header_core(self) -> CompactModuleClassProjection:
        """Project only the class declarations required by the family index."""

        return replace(
            self,
            classes=tuple(
                replace(
                    indexed_class,
                    **self._default_values_outside(
                        CompactIndexedClass,
                        CompactClassHeader,
                    ),
                )
                for indexed_class in self.classes
            ),
            **self._default_values_outside(
                CompactModuleClassProjection,
                CompactModuleClassHeader,
            ),
        )

    @staticmethod
    def _default_values_outside(
        declaration_type: type,
        preserved_type: type,
    ) -> dict[str, Any]:
        """Each field `declaration_type` adds past `preserved_type`, at its declared default.

        The values are each field's own default, read from `fields()`, so each
        has its field's type; `fields()` exposes them only as `Any`.
        """

        preserved_names = frozenset(
            dataclass_field.name for dataclass_field in fields(preserved_type)
        )
        values: dict[str, Any] = {}
        for dataclass_field in fields(declaration_type):
            if dataclass_field.name in preserved_names:
                continue
            if dataclass_field.default is not MISSING:
                values[dataclass_field.name] = dataclass_field.default
            elif dataclass_field.default_factory is not MISSING:
                values[dataclass_field.name] = dataclass_field.default_factory()
            else:
                raise TypeError(
                    f"{declaration_type.__name__}.{dataclass_field.name} has no default"
                )
        return values


class ClassMethodPromotionSafeDecorator(StrEnum):
    """Class decorators proven not to depend on direct method ownership."""

    DATACLASS = ("dataclass", frozenset(("dataclasses",)), True)
    FINAL = ("final", frozenset(("typing", "typing_extensions")), True)

    import_module_names: frozenset[str]
    preserves_product_schema: bool

    def __new__(
        cls,
        value: str,
        import_module_names: frozenset[str],
        preserves_product_schema: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.import_module_names = import_module_names
        member.preserves_product_schema = preserves_product_schema
        return member

    @classmethod
    def for_qualified_name(cls, qualified_name: str | None) -> Self | None:
        return next(
            (
                decorator
                for decorator in cls
                if any(
                    qualified_name == f"{module_name}.{decorator.value}"
                    for module_name in decorator.import_module_names
                )
            ),
            None,
        )

    @classmethod
    def qualified_name_preserves_product_schema(
        cls,
        qualified_name: str | None,
    ) -> bool:
        declaration = cls.for_qualified_name(qualified_name)
        return declaration is not None and declaration.preserves_product_schema


@dataclass(frozen=True)
class LatentRosterMatch:
    """Typed agreement between one roster and one nominal family surface."""

    coverage_ratio: float
    missing_member_names: tuple[str, ...]
    projection_policy_hint: str | None


@dataclass(frozen=True)
class LatentRosterObservation:
    """One source roster, with parsing and family matching owned once."""

    policy_tokens: ClassVar[frozenset[str]] = frozenset(
        {
            "active",
            "all",
            "available",
            "default",
            "enabled",
            "public",
            "selected",
            "supported",
            "test",
            "visible",
        }
    )
    policy_noise_tokens: ClassVar[frozenset[str]] = frozenset(
        {"by", "classes", "formats", "map", "registry", "types"}
    )
    mutation_methods: ClassVar[frozenset[str]] = frozenset({"extend", "update"})

    file_path: str
    roster_name: str
    line: int
    roster_kind: str
    projection_role: str
    member_names: tuple[str, ...]
    line_count: int

    @property
    def is_public_export_surface(self) -> bool:
        """Return whether this roster declares Python's module export contract."""

        return self.roster_name == PYTHON_PUBLIC_EXPORT_ASSIGNMENT

    @classmethod
    def from_module(
        cls,
        parsed_module: ParsedModule,
    ) -> tuple[LatentRosterObservation, ...]:
        observations: list[LatentRosterObservation] = []
        for statement in _trim_leading_docstring(list(parsed_module.module.body)):
            observations.extend(cls.from_statement(parsed_module, statement))
            observations.extend(cls.from_mutation_statement(parsed_module, statement))
            if not isinstance(statement, ast.ClassDef):
                continue
            for class_statement in _trim_leading_docstring(list(statement.body)):
                observations.extend(
                    cls.from_statement(
                        parsed_module,
                        class_statement,
                        roster_prefix=statement.name,
                    )
                )
                observations.extend(
                    cls.from_mutation_statement(parsed_module, class_statement)
                )
        return tuple(observations)

    @classmethod
    def from_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_prefix: str | None = None,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_leading_docstring(list(statement.body))
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                return ()
            return StatementValue.of(body[0]).answer(
                LatentRosterValue(parsed_module, statement, statement.name)
            )
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            return ()
        return tuple(
            observation
            for target in single_name_targets(statement)
            for observation in StatementValue.of(statement).answer(
                LatentRosterValue(
                    parsed_module,
                    statement,
                    (
                        f"{roster_prefix}.{target.id}"
                        if roster_prefix is not None
                        else target.id
                    ),
                )
            )
        )

    @classmethod
    def from_mutation_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
    ) -> tuple[LatentRosterObservation, ...]:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in cls.mutation_methods
        ):
            return ()
        call = statement.value
        return tuple(
            observation.for_mutation(call.func.attr)
            for argument in call.args
            for observation in cls.from_value(
                parsed_module,
                statement,
                roster_name=ast.unparse(call.func.value),
                value=argument,
            )
        )

    @classmethod
    def from_value(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(value, ast.Dict):
            return tuple(
                cls.from_members(
                    parsed_module,
                    statement,
                    roster_name=roster_name,
                    value=value,
                    projection_role=projection_role,
                    member_names=member_names,
                )
                for projection_role, member_names in (
                    ("dict_keys", cls.dict_member_names(value.keys)),
                    ("dict_values", cls.dict_member_names(value.values)),
                )
                if len(member_names) >= 2
            )
        member_names = cls.names_from(value)
        if len(member_names) < 2:
            return ()
        return (
            cls.from_members(
                parsed_module,
                statement,
                roster_name=roster_name,
                value=value,
                projection_role="collection_members",
                member_names=member_names,
            ),
        )

    @classmethod
    def from_members(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
        projection_role: str,
        member_names: tuple[str, ...],
    ) -> LatentRosterObservation:
        return cls(
            file_path=parsed_module.file_path,
            roster_name=roster_name,
            line=statement.lineno,
            roster_kind=type(value).__name__,
            projection_role=projection_role,
            member_names=member_names,
            line_count=(statement.end_lineno or statement.lineno)
            - statement.lineno
            + 1,
        )

    @staticmethod
    def names_from(node: ast.AST) -> tuple[str, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        member_names: set[str] = set()
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                member_names.add(element.value)
                continue
            reference = element.func if isinstance(element, ast.Call) else element
            if (member_name := _terminal_reference_name(reference)) is not None:
                member_names.add(member_name)
        return sorted_tuple(member_names)

    @classmethod
    def dict_member_names(cls, nodes: list[ast.expr | None]) -> tuple[str, ...]:
        return sorted_tuple(
            {
                member_name
                for node in nodes
                if node is not None
                for member_name in cls.names_from(
                    ast.Tuple(elts=[node], ctx=ast.Load())
                )
            }
        )

    def for_mutation(self, mutation_name: str) -> LatentRosterObservation:
        return replace(
            self,
            roster_kind=f"inline_{self.roster_kind}.{mutation_name}",
            projection_role=f"{mutation_name}_{self.projection_role}",
        )

    @property
    def projection_policy_hint(self) -> str | None:
        policy_tokens = tuple(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", self.roster_name)
            if token.lower() not in self.policy_noise_tokens
            and token.lower() in self.policy_tokens
        )
        return "_".join(policy_tokens) or None

    def match(
        self,
        authority_member_names: tuple[str, ...],
    ) -> LatentRosterMatch | None:
        roster_members = set(self.member_names)
        authority_members = set(authority_member_names)
        if not authority_members or not roster_members <= authority_members:
            return None
        missing_names = sorted_tuple(authority_members - roster_members)
        coverage_ratio = len(roster_members) / len(authority_members)
        if not missing_names:
            return LatentRosterMatch(coverage_ratio, (), None)
        if self.projection_policy_hint is None:
            return None
        return LatentRosterMatch(
            coverage_ratio,
            missing_names,
            self.projection_policy_hint,
        )


@dataclass(frozen=True)
class LatentRosterValue(StatementValueQuestion[tuple[LatentRosterObservation, ...]]):
    """The roster a statement's value spells; a bare statement spells none."""

    parsed_module: ParsedModule
    statement: ast.stmt
    roster_name: str

    def supplied(self, value: SuppliedValue) -> tuple[LatentRosterObservation, ...]:
        return LatentRosterObservation.from_value(
            self.parsed_module,
            self.statement,
            roster_name=self.roster_name,
            value=value.value,
        )

    def bare(self, value: NoStatementValue) -> tuple[LatentRosterObservation, ...]:
        return ()


class RegistryLookupStyle(StrEnum):
    TRY_EXCEPT = "try_except"
    MEMBERSHIP_GUARD = "membership_guard"


@dataclass(frozen=True)
class ClsRegistryMembership:
    operator_type: type[ast.cmpop]
    key_expr: str

    @classmethod
    def from_node(cls, node: ast.AST) -> ClsRegistryMembership | None:
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and len(node.comparators) == 1
            and RegistryLookupShape.is_cls_registry_attribute(node.comparators[0])
        ):
            return None
        return cls(type(node.ops[0]), ast.unparse(node.left))


@dataclass(frozen=True)
class RegistryLookupShape:
    """Recognized syntax for a class-owned ``_registry`` lookup."""

    key_expr: str
    error_type_name: str | None
    style: RegistryLookupStyle

    @staticmethod
    def is_cls_registry_attribute(node: ast.AST | None) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "_registry"
            and isinstance(node.value, ast.Name)
            and node.value.id == "cls"
        )

    @classmethod
    def key_expr_from_subscript(cls, node: ast.AST | None) -> str | None:
        if not (
            isinstance(node, ast.Subscript)
            and cls.is_cls_registry_attribute(node.value)
        ):
            return None
        return ast.unparse(node.slice)

    @classmethod
    def references_registry(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        return any(cls.is_cls_registry_attribute(node) for node in ast.walk(method))

    @staticmethod
    def _raise_type_name(node: ast.Raise | None) -> str | None:
        if node is None or node.exc is None:
            return None
        expression = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return _terminal_reference_name(expression)

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> RegistryLookupShape | None:
        body = _trim_method_docstring(method)
        if len(body) == 1 and isinstance(body[0], ast.Try):
            try_node = body[0]
            if (
                not try_node.orelse
                and not try_node.finalbody
                and len(try_node.handlers) == 1
                and _terminal_reference_name(try_node.handlers[0].type) == "KeyError"
                and len(try_node.body) == 1
                and isinstance(try_node.body[0], ast.Return)
                and (key_expr := cls.key_expr_from_subscript(try_node.body[0].value))
                is not None
            ):
                raised = next(
                    (
                        statement
                        for statement in try_node.handlers[0].body
                        if isinstance(statement, ast.Raise)
                    ),
                    None,
                )
                return cls(
                    key_expr,
                    cls._raise_type_name(raised),
                    RegistryLookupStyle.TRY_EXCEPT,
                )
        if len(body) < 2 or not isinstance(body[0], ast.If):
            return None
        guard = body[0]
        returned = body[-1]
        membership = ClsRegistryMembership.from_node(guard.test)
        if not (
            isinstance(returned, ast.Return)
            and membership is not None
            and membership.operator_type is ast.NotIn
            and cls.key_expr_from_subscript(returned.value) == membership.key_expr
        ):
            return None
        raised = next(
            (statement for statement in guard.body if isinstance(statement, ast.Raise)),
            None,
        )
        return cls(
            membership.key_expr,
            cls._raise_type_name(raised),
            RegistryLookupStyle.MEMBERSHIP_GUARD,
        )


class DeclaredFamilyBase(NamedTuple):
    """One base a class declares: the symbol its binding resolves to, shown by its terminal name."""

    symbol: str
    name: str

    @classmethod
    def of(cls, reference: CompactNominalReference) -> Self:
        return cls(reference.builtin_qualified_name, reference.simple_name)

    def is_standard_library(self, scanned_roots: frozenset[str]) -> bool:
        """Whether it resolves into the standard library: a top-level module no scanned module shadows."""

        root = self.symbol.partition(".")[0]
        return root in sys.stdlib_module_names and root not in scanned_roots


@dataclass(frozen=True)
class CompactRepeatedKeyedFamilyRoot(CompactSourceLocation):
    class_name: str
    family_bases: tuple[DeclaredFamilyBase, ...]
    registry_key_attr_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle
    error_type_name: str | None
    abstract_hook_names: tuple[str, ...]


ClassDeclarationT = TypeVar("ClassDeclarationT", bound=ClassDeclaration)


@dataclass(frozen=True)
class ClassDeclarationIndex(Generic[ClassDeclarationT]):
    """One declaration map with lazily derived identity and inheritance views."""

    classes_by_symbol: dict[str, ClassDeclarationT]

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        return frozenset(self.classes_by_symbol)

    @cached_property
    def symbols_by_simple_name(self) -> dict[str, tuple[str, ...]]:
        symbols: dict[str, list[str]] = defaultdict(list)
        for record in self.classes_by_symbol.values():
            symbols[record.simple_name].append(record.symbol)
        return {name: sorted_tuple(values) for name, values in symbols.items()}

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        return {
            name: symbols[0]
            for name, symbols in self.symbols_by_simple_name.items()
            if len(symbols) == 1
        }

    @cached_property
    def symbols_by_file_and_qualname(self) -> dict[tuple[str, str], str]:
        return {
            (record.file_path, record.qualname): record.symbol
            for record in self.classes_by_symbol.values()
        }

    @cached_property
    def inheritance_graph(self) -> DirectedGraph[str]:
        return DirectedGraph(
            {
                symbol: self.classes_by_symbol[symbol].resolved_base_symbols
                for symbol in sorted(self.classes_by_symbol)
            }
        )

    @cached_property
    def children_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return {
            symbol: children
            for symbol, children in self.inheritance_graph.reversed.neighbors.items()
            if children
        }

    @cached_property
    def ancestors_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return self.inheritance_graph.nonempty_reachability_from(
            self.inheritance_graph.neighbors
        )

    @cached_property
    def descendants_by_symbol(self) -> dict[str, tuple[str, ...]]:
        return self.inheritance_graph.reversed.nonempty_reachability_from(
            self.inheritance_graph.neighbors
        )

    def class_for(self, symbol: str) -> ClassDeclarationT | None:
        return self.classes_by_symbol.get(symbol)

    def symbol_for(self, *, file_path: str, qualname: str) -> str | None:
        return self.symbols_by_file_and_qualname.get((file_path, qualname))

    def ancestor_symbols(self, class_symbol: str) -> tuple[str, ...]:
        return self.inheritance_graph.reachable_from(class_symbol)

    def descendant_symbols(self, base_symbol: str) -> tuple[str, ...]:
        if base_symbol not in self.classes_by_symbol:
            return ()
        return self.inheritance_graph.reversed.reachable_from(base_symbol)


@dataclass
class _ProductLineageComposer:
    """Compose dataclass products through their exact direct-base chains, memoized."""

    classes_by_symbol: Mapping[str, CompactIndexedClass]
    lineages: dict[str, _CompactProductLineage] = field(default_factory=dict)

    @staticmethod
    def failure(
        indexed_class: CompactIndexedClass,
        violation: CompactProductAuthorityViolation,
        *,
        line: int | None = None,
    ) -> CompactProductAuthorityFailure:
        return CompactProductAuthorityFailure(
            class_symbol=indexed_class.symbol,
            file_path=indexed_class.file_path,
            line=indexed_class.line if line is None else line,
            violation=violation,
        )

    def compose(
        self,
        class_symbol: str,
        declaration: CompactDataclassDeclaration,
        pending_symbols: frozenset[str],
    ) -> _CompactProductLineage:
        if (lineage := self.lineages.get(class_symbol)) is not None:
            return lineage
        indexed_class = self.classes_by_symbol[class_symbol]
        if class_symbol in pending_symbols:
            return _CompactProductLineage(
                (),
                (
                    self.failure(
                        indexed_class,
                        CompactProductAuthorityViolation.CYCLIC_PRODUCT_LINEAGE,
                    ),
                ),
            )
        failures = tuple(
            self.failure(indexed_class, failure.violation, line=failure.line)
            for failure in declaration.failures
        )
        effective_fields: tuple[CompactProductField, ...] = ()
        direct_base_count = indexed_class.direct_base_count
        if (
            not indexed_class.base_references_are_complete
            or not indexed_class.product_base_bindings_are_exact
            or len(indexed_class.resolved_base_symbols) != direct_base_count
        ):
            failures = (
                *failures,
                self.failure(
                    indexed_class,
                    CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
                ),
            )
        elif direct_base_count > 1:
            failures = (
                *failures,
                self.failure(
                    indexed_class,
                    CompactProductAuthorityViolation.MULTIPLE_PRODUCT_BASES,
                ),
            )
        elif direct_base_count == 1:
            base_symbol = indexed_class.resolved_base_symbols[0]
            base_lineage = self.classes_by_symbol[
                base_symbol
            ].dataclass_declaration.base_lineage_in(
                self, base_symbol, indexed_class, pending_symbols
            )
            effective_fields = base_lineage.effective_fields
            failures = (*failures, *base_lineage.failures)

        fields_by_name = {field.name: field for field in effective_fields}
        for field_declaration in declaration.fields:
            fields_by_name[field_declaration.name] = CompactProductField(
                name=field_declaration.name,
                role=field_declaration.role,
                binding=field_declaration.binding,
                declaring_class_symbol=class_symbol,
                file_path=indexed_class.file_path,
                line=field_declaration.line,
            )
        lineage = _CompactProductLineage(
            effective_fields=tuple(fields_by_name.values()),
            failures=tuple(dict.fromkeys(failures)),
        )
        self.lineages[class_symbol] = lineage
        return lineage

    def resolve(
        self,
        class_symbol: str,
        indexed_class: CompactIndexedClass,
        declaration: CompactDataclassDeclaration,
    ) -> CompactProductAuthorityResolution:
        lineage = self.compose(class_symbol, declaration, frozenset())
        role_failures = tuple(
            CompactProductAuthorityFailure(
                class_symbol=class_symbol,
                file_path=field.file_path,
                line=field.line,
                violation=violation,
            )
            for field in lineage.effective_fields
            if (violation := field.role.authority_violation) is not None
        )
        failures = tuple(dict.fromkeys((*lineage.failures, *role_failures)))
        return (
            OpenCompactProductAuthority(class_symbol, failures)
            if failures
            else ResolvedCompactProductAuthority(
                class_symbol,
                CompactProductAuthority(
                    class_symbol=class_symbol,
                    effective_fields=lineage.effective_fields,
                    file_path=indexed_class.file_path,
                    line=indexed_class.line,
                ),
            )
        )


@dataclass(frozen=True)
class CompactClassFamilyIndex(ClassDeclarationIndex[CompactIndexedClass]):
    """Repository inheritance graph reconstructed from compact declarations."""

    @cached_property
    def mro_authority(self) -> ClassMroAuthority:
        return ClassMroAuthority(self.classes_by_symbol)

    @cached_property
    def declared_mro_authority(self) -> ClassMroAuthority:
        """The MRO the declarations state, for reading declared member types (`DeclaredBases`)."""

        return ClassMroAuthority(self.classes_by_symbol, DeclaredBases())

    def member_in_mro(self, owner_symbol: str, member_name: str) -> ClassMemberLookup:
        """The binding of `member_name` that an instance of `owner_symbol` reads at runtime."""

        return self._member_along(owner_symbol, member_name, self.mro_authority)

    def declared_member(self, owner_symbol: str, member_name: str) -> ClassMemberLookup:
        """The binding of `member_name` the declarations give an instance of `owner_symbol`."""

        return self._member_along(owner_symbol, member_name, self.declared_mro_authority)

    def _member_along(
        self, owner_symbol: str, member_name: str, authority: ClassMroAuthority
    ) -> ClassMemberLookup:
        """The first binding of `member_name` along `authority`'s MRO of `owner_symbol`.

        A class binding the name in its own body owns it, whatever its MRO; any
        other class reads the first binding along its closed MRO, and a lookup
        that reaches a base whose members nothing declares learns nothing past
        it. Assignments, annotations and defs all bind; a class binding the
        name twice, or by both an assignment and a def, is ambiguous.
        """

        owner = self.class_for(owner_symbol)
        if owner is None:
            return MissingClassMember()
        owners: tuple[CompactIndexedClass, ...] = (owner,)
        if not self._binds_member(owner, member_name):
            mro = authority.resolve(owner_symbol).mro_type
            if mro is None:
                return OpenClassMember()
            owners = mro.declarations_before(authority.bases.is_opaque)
        for current in owners:
            assigned = tuple(
                declaration
                for declaration in current.direct_member_declarations
                if declaration.name == member_name
            )
            defined = tuple(
                declaration
                for declaration in current.callable_member_declarations
                if declaration.name == member_name
            )
            match assigned, defined:
                case (), ():
                    continue
                case (declaration,), ():
                    return AssignedClassMember(current, declaration)
                case (), (declaration,):
                    return CallableClassMember(current, declaration)
            return AmbiguousClassMember()
        if not self._binds_member(owner, member_name) and self._reaches_opaque_base(
            owner_symbol, authority
        ):
            return OpenClassMember()
        return MissingClassMember()

    @staticmethod
    def _reaches_opaque_base(owner_symbol: str, authority: ClassMroAuthority) -> bool:
        mro = authority.resolve(owner_symbol).mro_type
        return mro is not None and any(authority.bases.is_opaque(entry) for entry in mro.__mro__)

    @staticmethod
    def _binds_member(owner: CompactIndexedClass, member_name: str) -> bool:
        return member_name in owner.bound_member_names

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> Self:
        """Join complete class declarations without collecting unrelated analyses."""

        return build_compact_class_family_index(
            CompactModuleClassHeader.from_modules(modules)
        )

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            Sequence[object],
        ],
    ) -> Self:
        """Build the class anchor declared by a compact multi-family join."""

        return build_compact_class_family_index(
            CompactModuleClassProjectionFamily.modules_in(projections_by_family)
        )

    @classmethod
    def require(cls, context: object | None) -> Self:
        if not isinstance(context, cls):
            raise TypeError("shared compact class index is unavailable")
        return context

    def assignments_repeated_from_ancestors(
        self,
        class_symbol: str,
        assignment_names: Iterable[str],
    ) -> tuple[str, ...]:
        """Project direct assignments whose exact values already come from a base."""

        indexed_class = self.class_for(class_symbol)
        if indexed_class is None:
            return ()
        ancestors = tuple(
            ancestor
            for ancestor_symbol in self.ancestor_symbols(class_symbol)
            if (ancestor := self.class_for(ancestor_symbol)) is not None
        )
        return tuple(
            assignment_name
            for assignment_name in assignment_names
            if (expression := indexed_class.assignments_by_name.get(assignment_name))
            is not None
            if any(
                ancestor.assignments_by_name.get(assignment_name) == expression
                for ancestor in ancestors
            )
        )

    def direct_or_absent_value_declaration(
        self,
        class_symbol: str,
        member_name: str,
    ) -> CompactClassMemberDeclaration | None:
        """Require a direct value or proved absence throughout the native MRO."""

        owner = self.classes_by_symbol[class_symbol]
        direct = owner.direct_value_writes_by_name.get(member_name)
        if direct is not None:
            return direct
        resolution = self.mro_authority.resolve(class_symbol)
        if resolution.mro_type is None:
            raise ValueError("Class value lookup requires a closed MRO")
        inherited = any(
            member_name in ancestor.direct_value_writes_by_name
            for ancestor in resolution.mro_type.declarations[1:]
        )
        if inherited:
            raise ValueError("Class value is inherited rather than directly declared")
        return None

    @cached_property
    def product_authority_resolutions_by_symbol(
        self,
    ) -> dict[str, CompactProductAuthorityResolution]:
        """Compose every dataclass product through its exact direct-base chain."""

        composer = _ProductLineageComposer(self.classes_by_symbol)
        return {
            class_symbol: indexed_class.dataclass_declaration.product_resolution_in(
                composer, class_symbol, indexed_class
            )
            for class_symbol, indexed_class in self.classes_by_symbol.items()
        }


ClosedLeafMethodAuthorityPredicate: TypeAlias = Callable[
    ["ClosedLeafMethodAuthorityProof"],
    bool,
]
CLASS_METHOD_OWNERSHIP_HOOK_NAMES = frozenset(("__init_subclass__",))
# The metaclasses Python provides: a class reaching one creates classes.
NATIVE_METACLASS_BASES = frozenset(("builtins.type", "abc.ABCMeta"))


class ClosedLeafMethodAuthorityViolation(StrEnum):
    """One failed proof obligation for promoting leaf methods to an ancestor."""

    TOO_FEW_PARTICIPANTS = (
        "too_few_participants",
        "the authority relation requires at least two participating leaves",
        lambda proof: len(proof.participant_symbol_set) < 2,
    )
    AMBIGUOUS_DIRECT_AUTHORITY = (
        "ambiguous_direct_authority",
        "the participants do not have exactly one resolved direct authority",
        lambda proof: (
            frozenset(proof.common_direct_base_symbols)
            != frozenset((proof.authority_symbol,))
        ),
    )
    AMBIGUOUS_DECLARED_AUTHORITY = (
        "ambiguous_declared_authority",
        "the participants do not have exactly one declared nominal base",
        lambda proof: (
            proof.common_declared_nominal_base_simple_names
            != frozenset((proof.authority_simple_name,))
        ),
    )
    INCOMPLETE_DIRECT_FAMILY = (
        "incomplete_direct_family",
        "the participants are not the complete direct-child family",
        lambda proof: (
            frozenset(proof.authority_direct_child_symbols)
            != proof.participant_symbol_set
        ),
    )
    NON_LEAF_PARTICIPANT = (
        "non_leaf_participant",
        "at least one participant still owns a descendant branch",
        lambda proof: bool(proof.non_leaf_participant_symbols),
    )
    INCOMPLETE_BASE_RESOLUTION = (
        "incomplete_base_resolution",
        "a relevant nominal base cannot be resolved from the repository graph",
        lambda proof: bool(proof.incompletely_resolved_symbols),
    )
    METHOD_OWNERSHIP_SENSITIVE_DECLARATION = (
        "method_ownership_sensitive_declaration",
        "a class decorator or metaclass boundary can observe direct method ownership",
        lambda proof: bool(proof.method_ownership_sensitive_symbols),
    )
    EXISTING_AUTHORITY_MEMBER = (
        "existing_authority_member",
        "the authority lineage already binds a promoted member name",
        lambda proof: bool(
            proof.promoted_method_name_set & proof.authority_lineage_member_name_set
        ),
    )
    COMPETING_ANCESTOR_MEMBER = (
        "competing_ancestor_member",
        "another participant ancestor binds a promoted member name",
        lambda proof: bool(
            proof.promoted_method_name_set
            & frozenset(proof.competing_ancestor_member_names)
        ),
    )
    UNDECLARED_RECEIVER_MEMBER = (
        "undeclared_receiver_member",
        "a promoted method requires a receiver member outside the authority contract",
        lambda proof: bool(
            frozenset(proof.receiver_member_names)
            - (proof.authority_lineage_member_name_set | proof.promoted_method_name_set)
        ),
    )

    def __new__(
        cls,
        value: str,
        explanation: str,
        predicate: ClosedLeafMethodAuthorityPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._explanation = explanation
        member._predicate = predicate
        return member

    @property
    def explanation(self) -> str:
        return self._explanation

    def is_violated_by(self, proof: ClosedLeafMethodAuthorityProof) -> bool:
        return self._predicate(proof)


@dataclass(frozen=True)
class ClosedLeafMethodAuthorityProof:
    """Representation-independent proof of one exact method-promotion owner."""

    authority_symbol: str
    authority_simple_name: str
    participant_symbols: tuple[str, ...]
    common_direct_base_symbols: tuple[str, ...]
    common_declared_nominal_base_names: tuple[str, ...]
    authority_direct_child_symbols: tuple[str, ...]
    non_leaf_participant_symbols: tuple[str, ...]
    incompletely_resolved_symbols: tuple[str, ...]
    method_ownership_sensitive_symbols: tuple[str, ...]
    authority_lineage_member_names: tuple[str, ...]
    competing_ancestor_member_names: tuple[str, ...]
    promoted_method_names: tuple[str, ...]
    receiver_member_names: tuple[str, ...]

    @cached_property
    def participant_symbol_set(self) -> frozenset[str]:
        return frozenset(self.participant_symbols)

    @cached_property
    def promoted_method_name_set(self) -> frozenset[str]:
        return frozenset(self.promoted_method_names)

    @cached_property
    def authority_lineage_member_name_set(self) -> frozenset[str]:
        return frozenset(self.authority_lineage_member_names)

    @cached_property
    def common_declared_nominal_base_simple_names(self) -> frozenset[str]:
        return frozenset(
            name.rsplit(".", 1)[-1] for name in self.common_declared_nominal_base_names
        )

    @cached_property
    def violations(self) -> tuple[ClosedLeafMethodAuthorityViolation, ...]:
        return tuple(
            violation
            for violation in ClosedLeafMethodAuthorityViolation
            if violation.is_violated_by(self)
        )

    @property
    def is_proven(self) -> bool:
        return not self.violations

    @property
    def rejection_reason(self) -> str:
        return "; ".join(violation.explanation for violation in self.violations)


def declared_nominal_base_count(declaration: ClassDeclaration) -> int:
    """Count domain-bearing direct bases under the canonical neutral-base policy."""

    return sum(
        ClassSymbolResolutionAuthority.establishes_nominal_family(base_name)
        for base_name in declaration.declared_base_names
    )


class CompactNominalBindingKind(StrEnum):
    """Kinds of exact module bindings and their exported-alias behavior."""

    MODULE_IMPORT = "module_import", True, True
    NAME_IMPORT = "name_import", True, False
    LOCAL_DECLARATION = "local_declaration", False, False
    BUILTIN = "builtin", False, False

    projects_as_import_alias: bool
    binds_module: bool

    def __new__(
        cls, value: str, projects_as_import_alias: bool, binds_module: bool
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.projects_as_import_alias = projects_as_import_alias
        member.binds_module = binds_module
        return member


class ModuleNominalBindingSnapshotPolicy(StrEnum):
    """Typed ambiguity policy for exact proof versus named-import projection."""

    EXACT = "exact", True
    NAMED_IMPORT_PROJECTION = "named_import_projection", False

    invalidates_on_star_import: bool

    def __new__(cls, value: str, invalidates_on_star_import: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.invalidates_on_star_import = invalidates_on_star_import
        return member


@dataclass(frozen=True)
class CompactNominalBinding:
    """One exact nominal origin selected by sequential module binding."""

    qualified_name: str
    kind: CompactNominalBindingKind


DecoratedDefinition: TypeAlias = (
    ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | FunctionDefinition
)


@dataclass(frozen=True)
class ModuleNominalBindingSnapshot:
    """Exact known and unresolved bindings before one module source position."""

    bindings_by_name: dict[str, CompactNominalBinding]
    unresolved_bound_names: frozenset[str] = frozenset()
    star_import_ambiguity: bool = False
    star_import_excluded_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = self.bindings_by_name.keys() & self.unresolved_bound_names
        if overlap:
            raise ValueError(
                "module binding names cannot be both resolved and unresolved: "
                f"{tuple(sorted(overlap))!r}"
            )

    def binding_for(self, name: str) -> CompactNominalBinding | None:
        return self.bindings_by_name.get(name)

    def reference_for(self, parts: tuple[str, ...]) -> CompactNominalReference:
        """Project a name path through its exact module or builtin binding."""

        binding = self.binding_for(parts[0])
        if binding is None and self.resolves_unshadowed_builtin(parts[0]):
            binding = CompactNominalBinding(
                qualified_name=f"builtins.{parts[0]}",
                kind=CompactNominalBindingKind.BUILTIN,
            )
        return CompactNominalReference.bound_as(parts, binding)

    def decorator_references(
        self, node: DecoratedDefinition
    ) -> tuple[CompactNominalReference, ...]:
        """What each decorator of the definition names through these bindings, in
        order; a decorator that is no name path (`@registry[key]`) is unbound."""

        return tuple(self._decorator_reference(decorator) for decorator in node.decorator_list)

    def decorated_by(
        self,
        node: DecoratedDefinition,
        identities: Collection[str],
    ) -> bool:
        """Whether a decorator of the definition binds one of these objects."""

        return any(
            reference.has_bound_name(identity)
            for reference in self.decorator_references(node)
            for identity in identities
        )

    def _decorator_reference(self, decorator: ast.expr) -> CompactNominalReference:
        callee = decorator.func if isinstance(decorator, ast.Call) else decorator
        match ExpressionReference.decoded(callee).attribute_chain:
            case (str(), *_) as parts:
                return self.reference_for(parts)
        return UnboundNominalReference((ast.unparse(decorator),))

    def resolves_unshadowed_builtin(
        self,
        name: str,
        *,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove that one bare name still resolves through Python builtins."""

        return (
            (not self.star_import_ambiguity or name in self.star_import_excluded_names)
            and name not in self.bindings_by_name
            and name not in self.unresolved_bound_names
            and name not in preceding_class_bound_names
            and name in vars(builtins)
        )


@dataclass(frozen=True)
class ModuleNominalBindingAuthority:
    """Resolve nominal module bindings at an exact source position, fail closed."""

    parsed_module: ParsedModule
    declared_assignment_authority_names: frozenset[str] = frozenset()

    def snapshot_before(
        self,
        line: int | None = None,
        *,
        policy: ModuleNominalBindingSnapshotPolicy = (
            ModuleNominalBindingSnapshotPolicy.EXACT
        ),
    ) -> ModuleNominalBindingSnapshot:
        return self.snapshots_before((line,), policy=policy)[line]

    def snapshots_before(
        self,
        lines: Iterable[int | None],
        *,
        policy: ModuleNominalBindingSnapshotPolicy = ModuleNominalBindingSnapshotPolicy.EXACT,
    ) -> dict[int | None, ModuleNominalBindingSnapshot]:
        """Resolve requested declaration positions and final bindings in one pass."""
        requested_lines = tuple(dict.fromkeys(lines))
        return _module_nominal_binding_snapshots(
            self,
            tuple(line for line in requested_lines if line is not None),
            include_final=None in requested_lines,
            policy=policy,
        )

    def qualified_name_at(
        self,
        reference: ast.AST,
        *,
        line: int | None,
        policy: ModuleNominalBindingSnapshotPolicy = (
            ModuleNominalBindingSnapshotPolicy.EXACT
        ),
    ) -> str | None:
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(reference)
        ).attribute_chain
        if parts is None:
            return None
        root_binding = self.snapshot_before(line, policy=policy).binding_for(parts[0])
        if root_binding is None:
            return None
        return ".".join((root_binding.qualified_name, *parts[1:]))

    def nominal_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """Resolve a single nominal annotation through its declaration-time binding."""

        reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(annotation)
        if (
            reference is None
            or nominal_reference_root_name(reference) in class_bound_names
        ):
            return None
        return self.qualified_name_at(reference, line=line)

    def class_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """Resolve type[C] separately from an instance annotation C."""
        envelope = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.subscript_or_none(annotation)
        if envelope is None:
            return None
        if not self._annotation_factory_is(
            envelope.value,
            ("builtins.type",),
            line=line,
            class_bound_names=class_bound_names,
        ):
            return None
        return self.nominal_annotation_name_at(
            envelope.slice, line=line, class_bound_names=class_bound_names
        )

    def _annotation_factory_is(
        self,
        reference: ast.AST,
        factory_names: tuple[str, ...],
        *,
        line: int,
        class_bound_names: frozenset[str],
    ) -> bool:
        """Whether a binding proves the annotation's factory is one of these names."""

        parts = ExpressionReference.decoded(reference).attribute_chain
        if parts is None or parts[0] in class_bound_names:
            return False
        resolved = self.snapshot_before(line).reference_for(parts)
        return any(resolved.has_bound_name(name) for name in factory_names)

    def iterated_class_annotation_name_at(
        self,
        annotation: ast.AST,
        *,
        line: int,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> str | None:
        """One represented class in a declared tuple, not guessed iterable flow."""
        envelope = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.subscript_or_none(annotation)
        if envelope is None:
            return None
        if not self._annotation_factory_is(
            envelope.value,
            ("builtins.tuple", "typing.Tuple"),
            line=line,
            class_bound_names=class_bound_names,
        ):
            return None
        items = (
            envelope.slice.elts
            if isinstance(envelope.slice, ast.Tuple)
            else [envelope.slice]
        )
        if (
            len(items) == 2
            and isinstance(items[-1], ast.Constant)
            and items[-1].value is Ellipsis
        ):
            items = items[:1]
        types = {
            self.class_annotation_name_at(
                item, line=line, class_bound_names=class_bound_names
            )
            for item in items
        }
        return next(iter(types)) if len(types) == 1 else None

    def _direct_nominal_bindings(
        self,
        statement: ast.stmt,
        preceding_bindings: dict[str, CompactNominalBinding],
    ) -> dict[str, CompactNominalBinding]:
        if isinstance(statement, ast.Import | ast.ImportFrom):
            # `import x` always binds a module; `from x import y` may bind anything.
            kind = (
                CompactNominalBindingKind.MODULE_IMPORT
                if isinstance(statement, ast.Import)
                else CompactNominalBindingKind.NAME_IMPORT
            )
            return {
                origin.bound_name: CompactNominalBinding(
                    qualified_name=origin.qualified_name,
                    kind=kind,
                )
                for origin in ImportBoundNameProjection(statement).origins(
                    self.parsed_module.module_path_identity
                )
                if origin.qualified_name is not None
            }
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            return {
                statement.name: CompactNominalBinding(
                    qualified_name=f"{self.parsed_module.module_name}.{statement.name}",
                    kind=CompactNominalBindingKind.LOCAL_DECLARATION,
                )
            }
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            return {}
        local_name = _single_assignment_target_name(statement)
        value = statement.value
        if local_name is None or value is None:
            return {}
        if local_name in self.declared_assignment_authority_names:
            return {
                local_name: CompactNominalBinding(
                    qualified_name=f"{self.parsed_module.module_name}.{local_name}",
                    kind=CompactNominalBindingKind.LOCAL_DECLARATION,
                )
            }
        parts = ExpressionReference.decoded(
            ClassSymbolResolutionAuthority.reference_node(value)
        ).attribute_chain
        if parts is None or (root_binding := preceding_bindings.get(parts[0])) is None:
            return {}
        return {
            local_name: CompactNominalBinding(
                qualified_name=".".join((root_binding.qualified_name, *parts[1:])),
                kind=root_binding.kind,
            )
        }


def nominal_reference_root(reference: ast.AST) -> ast.Name | None:
    """Return the lexical root node whose binding owns a nominal reference."""

    while isinstance(reference, ast.Attribute):
        reference = reference.value
    return reference if isinstance(reference, ast.Name) else None


def nominal_reference_root_name(reference: ast.AST) -> str | None:
    """Return the lexical root name whose binding owns a nominal reference."""

    root = nominal_reference_root(reference)
    return None if root is None else root.id


class ModuleNominalBindingView(ABC):
    """Representation-independent nominal bindings at one module position."""

    source_entry = staticmethod(ImportedSourceModuleEntryPremise.from_source)

    @cached_property
    def _source_projections(self) -> dict[int, SourceProductFlowProjection]:
        return {}

    def source_projection(self, module: ParsedModule) -> SourceProductFlowProjection:
        """Retain one original observation graph per parsed owner in this proof."""
        source = self._source_projections.get(id(module))
        if source is None:
            source = source_product_flow_projection(module)
            self._source_projections[id(module)] = source
        return source

    def native_reference_environment(
        self, module: ParsedModule
    ) -> NativeReferenceEnvironment:
        """Admit execution over this proof's existing original observations."""
        environment = self._native_executions.get(id(module))
        if environment is None:
            source = self.source_projection(module)
            entry = self.source_entry(source)
            if entry.source is not source:
                raise ValueError(
                    "Source entry belongs to a different canonical projection"
                )
            environment = SourceModuleExecution(entry)
            self._native_executions[id(module)] = environment
        return environment

    @cached_property
    def _native_executions(self) -> dict[int, NativeReferenceEnvironment]:
        return {}

    def reference_or_builtin_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> ModuleNominalBindingWitness | None:
        root = nominal_reference_root_name(reference)
        if root in preceding_class_bound_names:
            return None
        witness = self.reference_witness_at(module, reference, line=line)
        if witness is None and isinstance(reference, ast.Name):
            return self.unshadowed_builtin_witness(
                module,
                reference.id,
                line=line,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        return witness

    def require_native_type_in_class(
        self,
        module: ParsedModule,
        owner: ast.ClassDef,
        declaration: type,
    ) -> None:
        """Prove the native type emitted into a class resolves to its declaration."""

        name = declaration.__name__
        class_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(owner.body)
        if name in class_names:
            raise ValueError(f"Class namespace shadows native type {name!r}")
        witness = self.reference_or_builtin_witness_at(
            module,
            ast.Name(id=name, ctx=ast.Load()),
            line=owner.lineno,
            preceding_class_bound_names=class_names,
        )
        qualified_name = NativeDeclaration(declaration).qualified_name
        if witness is None or witness.qualified_name != qualified_name:
            raise ValueError(
                f"Class creation does not prove native type binding {qualified_name!r}"
            )

    @abstractmethod
    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        raise NotImplementedError

    @abstractmethod
    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        raise NotImplementedError


@dataclass(frozen=True)
class ModuleNominalBindingWitness:
    """One resolved declaration and the lexical root requiring star exclusion."""

    qualified_name: str
    root_name: str


class ClassBodyDeclarationKind(StrEnum):
    """What one direct class-body statement declares, the receipt two classes
    sharing it earn, and whether it is a field.

    A supplied value weighs as a value whether or not it is a field's default.
    """

    shared_receipt: OverlapReceiptKind
    is_field: bool

    def __new__(
        cls, value: str, shared_receipt: OverlapReceiptKind, is_field: bool
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.shared_receipt = shared_receipt
        member.is_field = is_field
        return member

    FIELD = "field", OverlapReceiptKind.SHARED_FIELD, True
    DEFAULTED_FIELD = "defaulted_field", OverlapReceiptKind.SHARED_VALUE, True
    VALUE = "value", OverlapReceiptKind.SHARED_VALUE, False
    METHOD = "method", OverlapReceiptKind.SHARED_METHOD, False


class AnnotatedDeclarationKind(StatementValueQuestion[ClassBodyDeclarationKind]):
    """An annotation declares a field; with a value it also supplies the default."""

    def supplied(self, value: SuppliedValue) -> ClassBodyDeclarationKind:
        return ClassBodyDeclarationKind.DEFAULTED_FIELD

    def bare(self, value: NoStatementValue) -> ClassBodyDeclarationKind:
        return ClassBodyDeclarationKind.FIELD


class _ReceiverAliasSubstitution(ast.NodeTransformer):
    """Substitute one proven local receiver alias in a copied method tree."""

    def __init__(self, alias_name: str, receiver_name: str) -> None:
        self.alias_name = alias_name
        self.receiver_name = receiver_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.alias_name:
            return node
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Receiver alias is rebound after its declaration")
        return ast.copy_location(ast.Name(id=self.receiver_name, ctx=node.ctx), node)


class StarDeferredName(NamedTuple):
    """A free name a star import may rebind, with the binding it has if none does.

    `unshadowed` is the qualified name the module binds without its star
    imports (`None` when nothing does); the repository decides which applies.
    """

    name: str
    unshadowed: str | None


IdentityT = TypeVar("IdentityT")


class CompactFieldFact(ABC):
    """The bare field, `name: annotation`, one class-body declaration makes:
    none, the declaration itself, or what a call that declares one names."""

    @abstractmethod
    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        raise NotImplementedError


@dataclass(frozen=True)
class NoFieldFact(CompactFieldFact):
    """A value or a method: it declares no field."""

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        return None


@dataclass(frozen=True)
class CompactBareFieldFact(CompactFieldFact):
    """`name: annotation` with no value: the declaration is its own field."""

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        return identity


@dataclass(frozen=True)
class CompactCalledFieldFact(CompactFieldFact):
    """A field whose value is a call: it declares `bare` when the repository
    proves the call declares a field without a default."""

    call: CompactFieldDeclarationCall
    bare: ClassBodyDeclarationDigest

    def field_identity(
        self,
        identity: IdentityT,
        bare_identity: Callable[[ClassBodyDeclarationDigest], IdentityT | None],
        field_declarations: FieldDeclarationIndex,
    ) -> IdentityT | None:
        if field_declarations.declares_bare_field(self.call) is not True:
            return None
        return bare_identity(self.bare)


class AnnotatedFieldFact(StatementValueQuestion[CompactFieldFact]):
    """An annotation alone is its own field; with a plain value it is a value."""

    def supplied(self, value: SuppliedValue) -> CompactFieldFact:
        return NoFieldFact()

    def bare(self, value: NoStatementValue) -> CompactFieldFact:
        return CompactBareFieldFact()


class ClassBodyDeclarationDigest(NamedTuple):
    """One class-body declaration as the index compares it: no AST.

    `digest` covers the declaration's kind, name, syntax and class-scope
    names; `bindings` what each free name the module binds resolves to, as
    `(name, qualified name)`. `deferred` lists the free names only the
    repository's star-import exposure can resolve, which the index settles
    into the same bindings, and `unbound` the free names nothing binds, which
    make it equal to nothing.
    """

    kind: ClassBodyDeclarationKind
    name: str
    line: int
    digest: str
    bindings: tuple[tuple[str, str], ...]
    deferred: tuple[StarDeferredName, ...]
    unbound: tuple[str, ...]
    field_declaration: CompactFieldFact = NoFieldFact()
    """The bare field, `name: annotation`, the declaration makes, if any."""


@dataclass(frozen=True)
class ClassBodyModuleScope:
    """What a free name in one class body is bound to.

    `declaration` is the module snapshot where the class statement runs and
    `final` the module after import. A name bound at the class statement keeps
    that binding; a name bound only later is read where evaluation is deferred
    (method bodies, postponed annotations), and Python raises at import for any
    other read of it, so code that imports reads the final binding. A name the
    module binds by assignment rather than import or definition is identified by
    the module that binds it.
    """

    module_name: str
    declaration: ModuleNominalBindingSnapshot
    final: ModuleNominalBindingSnapshot
    star_blind_declaration: ModuleNominalBindingSnapshot
    star_blind_final: ModuleNominalBindingSnapshot

    @classmethod
    def for_lines(
        cls, parsed_module: ParsedModule, lines: Iterable[int]
    ) -> dict[int, ClassBodyModuleScope]:
        """One scope per class line, from two passes over the module's bindings."""

        requested = tuple(dict.fromkeys(lines))
        authority = ModuleNominalBindingAuthority(parsed_module)
        exact = authority.snapshots_before((*requested, None))
        star_blind = authority.snapshots_before(
            (*requested, None),
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        return {
            line: cls(
                parsed_module.module_name,
                exact[line],
                exact[None],
                star_blind[line],
                star_blind[None],
            )
            for line in requested
        }

    def witness_for(self, name: str) -> ModuleNominalBindingWitness | None:
        return self._witness_in(name, self.declaration, self.final)

    def deferred_for(self, name: str) -> StarDeferredName | None:
        """A name a star import hides, with its binding if no star import rebinds it."""

        if not self.final.star_import_ambiguity:
            return None
        witness = self._witness_in(
            name, self.star_blind_declaration, self.star_blind_final
        )
        return StarDeferredName(
            name, None if witness is None else witness.qualified_name
        )

    def _witness_in(
        self,
        name: str,
        declaration: ModuleNominalBindingSnapshot,
        final: ModuleNominalBindingSnapshot,
    ) -> ModuleNominalBindingWitness | None:
        for snapshot in (declaration, final):
            if (binding := snapshot.binding_for(name)) is not None:
                return ModuleNominalBindingWitness(binding.qualified_name, name)
        if declaration.resolves_unshadowed_builtin(name):
            return ModuleNominalBindingWitness(f"{builtins.__name__}.{name}", name)
        if name in declaration.unresolved_bound_names or name in final.unresolved_bound_names:
            return ModuleNominalBindingWitness(f"{self.module_name}.{name}", name)
        return None


@dataclass(frozen=True)
class ClassBodyNormalForm:
    """One class-body declaration as Python evaluates it.

    Two declarations are the same declaration when their kind, name, syntax
    (positions and a leading docstring dropped, a method's `x: T = self`
    receiver alias substituted) and the declaration each free name is bound to
    are equal. A free name bound earlier in the class body is recorded by name,
    so it is equal only to the same class-scope name; a free name the module
    snapshot cannot bind leaves the declaration unresolved, and an unresolved
    declaration equals nothing in the class-body index.
    """

    kind: ClassBodyDeclarationKind
    name: str
    line: int
    syntax: str
    global_bindings: tuple[ModuleNominalBindingWitness, ...]
    class_scope_names: tuple[str, ...]
    deferred_names: tuple[StarDeferredName, ...]
    unresolved_names: tuple[str, ...]
    field_declaration: CompactFieldFact = NoFieldFact()

    @property
    def is_resolved(self) -> bool:
        """Every free name is bound inside the module, star imports aside."""

        return not self.unresolved_names and not self.deferred_names

    @cached_property
    def digest(self) -> str:
        return hashlib.blake2s(
            repr(
                (
                    self.kind.value,
                    self.name,
                    self.syntax,
                    tuple(
                        (binding.qualified_name, binding.root_name)
                        for binding in self.global_bindings
                    ),
                    self.class_scope_names,
                )
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()

    @cached_property
    def syntax_digest(self) -> str:
        """The digest of everything but the free names' bindings."""

        return hashlib.blake2s(
            repr((self.kind.value, self.name, self.syntax, self.class_scope_names)).encode(
                "utf-8"
            ),
            digest_size=16,
        ).hexdigest()

    def compact(self) -> ClassBodyDeclarationDigest:
        return ClassBodyDeclarationDigest(
            self.kind,
            self.name,
            self.line,
            self.syntax_digest,
            tuple(
                (binding.root_name, binding.qualified_name)
                for binding in self.global_bindings
            ),
            self.deferred_names,
            self.unresolved_names,
            self.field_declaration,
        )

    @classmethod
    def of_class(
        cls,
        node: ast.ClassDef,
        scope: ClassBodyModuleScope,
    ) -> tuple[ClassBodyNormalForm, ...]:
        """Every direct declaration of one class body, in source order."""

        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        declarations: list[ClassBodyNormalForm] = []
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declarations.append(cls.of_method(statement, scope, class_bound_names))
            elif isinstance(statement, (ast.AnnAssign, ast.Assign)):
                declarations.extend(
                    cls.of_assignment(statement, scope, class_bound_names)
                )
        return tuple(declarations)

    @classmethod
    def of_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str] = frozenset(),
    ) -> ClassBodyNormalForm:
        """A method's header evaluates in the class scope and its body in the module's.

        A free name the class body binds is recorded as a class-scope name and,
        when the module binds it too, with that binding, so equality holds only
        when both scopes agree.
        """

        normalized = cls._normalized_method(method)
        return cls._of_statement(
            ClassBodyDeclarationKind.METHOD,
            method.name,
            method,
            normalized,
            scope,
            class_bound_names,
        )

    @classmethod
    def of_assignment(
        cls,
        statement: ast.AnnAssign | ast.Assign,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> tuple[ClassBodyNormalForm, ...]:
        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name):
                return ()
            kind = StatementValue.of(statement).answer(AnnotatedDeclarationKind())
            if isinstance(statement.value, ast.Call):
                return (
                    cls.of_called_field(statement, statement.value, scope, class_bound_names),
                )
            return (
                replace(
                    cls._of_statement(
                        kind,
                        statement.target.id,
                        statement,
                        statement,
                        scope,
                        class_bound_names - {statement.target.id},
                    ),
                    field_declaration=StatementValue.of(statement).answer(
                        AnnotatedFieldFact()
                    ),
                ),
            )
        names = tuple(
            target.id for target in statement.targets if isinstance(target, ast.Name)
        )
        return tuple(
            cls._of_statement(
                ClassBodyDeclarationKind.VALUE,
                name,
                statement,
                statement,
                scope,
                class_bound_names - {name},
            )
            for name in names
        )

    @classmethod
    def of_called_field(
        cls,
        statement: ast.AnnAssign,
        call: ast.Call,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> ClassBodyNormalForm:
        """A field whose value is a call, with the call and the bare field it
        declares if the call declares a field without a default."""

        name = cast(ast.Name, statement.target).id
        bound_names = class_bound_names - {name}
        bare_statement = ast.copy_location(
            ast.AnnAssign(
                target=statement.target,
                annotation=statement.annotation,
                value=None,
                simple=statement.simple,
            ),
            statement,
        )
        parts = ExpressionReference.decoded(call.func).attribute_chain
        witness = (
            None
            if parts is None or parts[0] in bound_names
            else scope.witness_for(parts[0])
        )
        declaration = cls._of_statement(
            ClassBodyDeclarationKind.DEFAULTED_FIELD,
            name,
            statement,
            statement,
            scope,
            bound_names,
        )
        if witness is None:
            return declaration
        return replace(
            declaration,
            field_declaration=CompactCalledFieldFact(
                CompactFieldDeclarationCall.of_call(
                    ".".join((witness.qualified_name, *cast(tuple[str, ...], parts)[1:])),
                    call,
                ),
                cls._of_statement(
                    ClassBodyDeclarationKind.FIELD,
                    name,
                    bare_statement,
                    bare_statement,
                    scope,
                    bound_names,
                ).compact(),
            ),
        )

    @classmethod
    def _of_statement(
        cls,
        kind: ClassBodyDeclarationKind,
        name: str,
        statement: ast.stmt,
        normalized: ast.stmt,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
    ) -> ClassBodyNormalForm:
        free_names = sorted(
            {
                reference.id
                for reference in ModuleLexicalDependencyProjection.from_module(
                    ast.Module(body=[normalized], type_ignores=[])
                ).external_name_references
            }
        )
        global_bindings: list[ModuleNominalBindingWitness] = []
        class_scope_names: list[str] = []
        deferred_names: list[StarDeferredName] = []
        unresolved_names: list[str] = []
        for free_name in free_names:
            if free_name in class_bound_names:
                class_scope_names.append(free_name)
                if kind is not ClassBodyDeclarationKind.METHOD:
                    continue
            if (witness := scope.witness_for(free_name)) is not None:
                global_bindings.append(witness)
            elif free_name in class_bound_names:
                continue
            elif (deferred := scope.deferred_for(free_name)) is not None:
                deferred_names.append(deferred)
            else:
                unresolved_names.append(free_name)
        return cls(
            kind=kind,
            name=name,
            line=statement.lineno,
            syntax=ast.dump(normalized, include_attributes=False),
            global_bindings=tuple(global_bindings),
            class_scope_names=tuple(class_scope_names),
            deferred_names=tuple(deferred_names),
            unresolved_names=tuple(unresolved_names),
        )

    @classmethod
    def _normalized_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        normalized = copy.copy(method)
        normalized.body = _trim_leading_docstring(list(method.body)) or [ast.Pass()]
        positional = (*normalized.args.posonlyargs, *normalized.args.args)
        receiver_name = positional[0].arg if positional else None
        if receiver_name is None or len(normalized.body) != 2:
            return normalized
        alias_declaration, remaining_statement = normalized.body
        if not (
            isinstance(alias_declaration, ast.AnnAssign)
            and alias_declaration.simple == 1
            and isinstance(alias_declaration.target, ast.Name)
            and isinstance(alias_declaration.value, ast.Name)
            and alias_declaration.value.id == receiver_name
            and alias_declaration.target.id != receiver_name
        ):
            return normalized
        alias_name = alias_declaration.target.id
        alias_references = tuple(
            node
            for node in ast.walk(remaining_statement)
            if isinstance(node, ast.Name) and node.id == alias_name
        )
        if not alias_references or any(
            not isinstance(reference.ctx, ast.Load) for reference in alias_references
        ):
            return normalized
        normalized.body = [
            _ReceiverAliasSubstitution(alias_name, receiver_name).visit(
                copy.deepcopy(remaining_statement)
            )
        ]
        return ast.fix_missing_locations(normalized)


@dataclass(frozen=True)
class NamedImportModuleBindingProjection(ModuleNominalBindingView):
    """Module-local binding view whose star-import obligations remain explicit."""

    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        is_unshadowed = (
            ModuleNominalBindingAuthority(module)
            .snapshot_before(
                line,
                policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
            )
            .resolves_unshadowed_builtin(
                name,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
        return (
            ModuleNominalBindingWitness(f"builtins.{name}", name)
            if is_unshadowed
            else None
        )

    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        root_name = nominal_reference_root_name(reference)
        if root_name is None:
            return None
        qualified_name = ModuleNominalBindingAuthority(module).qualified_name_at(
            reference,
            line=line,
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        if qualified_name is None:
            return None
        return ModuleNominalBindingWitness(qualified_name, root_name)


def _direct_binding_target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Starred):
        return _direct_binding_target_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return tuple(
            name
            for element in target.elts
            for name in _direct_binding_target_names(element)
        )
    return ()


def _direct_statement_bound_names(statement: ast.stmt) -> frozenset[str]:
    """Project common direct bindings without allocating a scope visitor."""

    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return frozenset((statement.name,))
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return frozenset(ImportBoundNameProjection(statement).names())
    if isinstance(statement, ast.Assign):
        return frozenset(
            name
            for target in statement.targets
            for name in _direct_binding_target_names(target)
        )
    if isinstance(statement, ast.AnnAssign | ast.AugAssign):
        return frozenset(_direct_binding_target_names(statement.target))
    if isinstance(statement, ast.Delete):
        return frozenset(
            name
            for target in statement.targets
            for name in _direct_binding_target_names(target)
        )
    return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((statement,))


def _module_nominal_binding_snapshots(
    authority: ModuleNominalBindingAuthority,
    lines: tuple[int, ...],
    *,
    include_final: bool,
    policy: ModuleNominalBindingSnapshotPolicy = (
        ModuleNominalBindingSnapshotPolicy.EXACT
    ),
    star_import_excluded_names: frozenset[str] = frozenset(),
) -> dict[int | None, ModuleNominalBindingSnapshot]:
    parsed_module = authority.parsed_module
    bindings: dict[str, CompactNominalBinding] = {}
    unresolved_bound_names: set[str] = set()
    star_import_ambiguity = False
    statements = iter(parsed_module.module.body)
    statement = next(statements, None)
    snapshots: dict[int | None, ModuleNominalBindingSnapshot] = {}

    def snapshot() -> ModuleNominalBindingSnapshot:
        return ModuleNominalBindingSnapshot(
            bindings_by_name=dict(bindings),
            unresolved_bound_names=frozenset(unresolved_bound_names),
            star_import_ambiguity=star_import_ambiguity,
            star_import_excluded_names=star_import_excluded_names,
        )

    def apply(current: ast.stmt) -> None:
        nonlocal star_import_ambiguity
        if (
            policy.invalidates_on_star_import
            and isinstance(current, ast.ImportFrom)
            and any(alias.name == "*" for alias in current.names)
        ):
            for binding_name in tuple(bindings):
                if binding_name not in star_import_excluded_names:
                    del bindings[binding_name]
            unresolved_bound_names.intersection_update(star_import_excluded_names)
            star_import_ambiguity = True
            return
        direct_bindings = authority._direct_nominal_bindings(current, bindings)
        bound_names = _direct_statement_bound_names(current)
        deleted_names = bound_names if isinstance(current, ast.Delete) else frozenset()
        for bound_name in bound_names:
            bindings.pop(bound_name, None)
            if bound_name in deleted_names:
                unresolved_bound_names.discard(bound_name)
            elif bound_name not in direct_bindings:
                unresolved_bound_names.add(bound_name)
        for bound_name in direct_bindings:
            unresolved_bound_names.discard(bound_name)
        bindings.update(direct_bindings)

    for requested_line in sorted(frozenset(lines)):
        while statement is not None and statement.lineno < requested_line:
            apply(statement)
            statement = next(statements, None)
        snapshots[requested_line] = snapshot()
    if include_final:
        while statement is not None:
            apply(statement)
            statement = next(statements, None)
        snapshots[None] = snapshot()
    return snapshots


@dataclass(frozen=True)
class ModuleImportBindings(ParsedModuleFact):
    """The import aliases one module binds, by local name."""

    bindings: dict[str, CompactNominalBinding]

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> Self:
        return cls(
            {
                local_name: binding
                for local_name, binding in _module_nominal_binding_snapshots(
                    ModuleNominalBindingAuthority(parsed_module),
                    (),
                    include_final=True,
                    policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
                )[None].bindings_by_name.items()
                if binding.kind.projects_as_import_alias
            }
        )


def module_import_aliases(parsed_module: ParsedModule) -> dict[str, str]:
    return {
        local_name: binding.qualified_name
        for local_name, binding in parsed_module.fact(ModuleImportBindings).bindings.items()
    }


def module_import_names(parsed_module: ParsedModule) -> frozenset[str]:
    """Names the module's final bindings resolve to an imported module."""

    return frozenset(
        local_name
        for local_name, binding in parsed_module.fact(ModuleImportBindings).bindings.items()
        if binding.kind.binds_module
    )


def module_star_import_origins(
    parsed_module: ParsedModule,
) -> tuple[CompactModuleStarImportOrigin, ...]:
    origins: list[CompactModuleStarImportOrigin] = []

    class ModuleScopeStarImportCollector(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if any(alias.name == "*" for alias in node.names):
                origins.append(
                    CompactModuleStarImportOrigin.resolved_as(
                        parsed_module.module_path_identity.resolve_import_from_module(
                            imported_module=node.module,
                            level=node.level,
                        )
                    )
                )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    ModuleScopeStarImportCollector().visit(parsed_module.module)
    return tuple(origins)


@dataclass(frozen=True)
class CompactClassProjectionDemand:
    """Target-correlated filters for expensive class-family facets."""

    class_method_names: frozenset[str]
    include_autoregister_references: bool = True
    header_core_only: bool = False


HEADER_CORE_DEMAND = CompactClassProjectionDemand(frozenset(), header_core_only=True)
"""The view `ModuleClassHeaderCore` holds: namespace and class headers only."""


def _native_definition_child(node: object, definition_type: str) -> object | None:
    node_type = getattr(node, "type", None)
    if node_type == definition_type:
        return node
    if node_type != "decorated_definition":
        return None
    return next(
        (
            child
            for child in getattr(node, "named_children", ())
            if child.type == definition_type
        ),
        None,
    )


def _native_sparse_class_header(
    syntax_index: NativePythonSyntaxIndex,
    node: object,
) -> ast.ClassDef:
    class_node = copy.deepcopy(syntax_index.class_header_for(node))
    body_node = node.child_by_field_name("body")
    body: list[ast.stmt] = []
    if body_node is not None:
        for child in body_node.named_children:
            nested = _native_definition_child(child, "class_definition")
            if nested is not None:
                body.append(_native_sparse_class_header(syntax_index, nested))
                continue
            function = _native_definition_child(child, "function_definition")
            if function is not None:
                continue
            if child.type != "expression_statement":
                continue
    if body:
        class_node.body = body
    return class_node


def _native_class_header_module(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
) -> ParsedModule | None:
    if not syntax_index.is_complete:
        return None
    public_export_name = PYTHON_PUBLIC_EXPORT_ASSIGNMENT.encode("utf-8")
    has_public_export_syntax = any(
        syntax_index.source_for(node) == public_export_name
        for node in syntax_index.captures("(identifier) @identifier").get(
            "identifier",
            (),
        )
    )
    has_star_import = any(
        isinstance(statement := syntax_index.statement_for(node), ast.ImportFrom)
        and any(alias.name == "*" for alias in statement.names)
        for node in syntax_index.captures("(import_from_statement) @statement").get(
            "statement", ()
        )
    )
    if has_public_export_syntax or has_star_import:
        return source_module.parse()
    body: list[ast.stmt] = []
    for child in syntax_index.tree.root_node.named_children:
        class_node = _native_definition_child(child, "class_definition")
        if class_node is not None:
            body.append(_native_sparse_class_header(syntax_index, class_node))
        elif child.type in {
            "future_import_statement",
            "import_statement",
            "import_from_statement",
        }:
            body.append(copy.deepcopy(syntax_index.statement_for(child)))
    return source_module.parsed_module(
        ast.Module(body=body, type_ignores=[]),
    )


def _collect_demanded_class_projection_from_source(
    source_module: SourceModule,
    syntax_index: NativePythonSyntaxIndex,
    demand: object,
) -> list[object] | None:
    if not isinstance(demand, CompactClassProjectionDemand):
        raise TypeError("class projection demand has the wrong authority type")
    if not demand.header_core_only:
        return None
    parsed_module = _native_class_header_module(source_module, syntax_index)
    if parsed_module is None:
        return None
    return CompactModuleClassProjectionFamily._collect_header_core(parsed_module)


def _compact_base_references(
    node: ast.ClassDef,
    module_binding_snapshot: ModuleNominalBindingSnapshot,
) -> tuple[CompactNominalReference, ...]:
    return tuple(
        module_binding_snapshot.reference_for(parts)
        for _base, parts in _referenced_bases(node)
    )


def _compact_class_binding_facets(
    node: ast.ClassDef,
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    module_name: str,
    qualname: str,
    *,
    include_body_facets: bool,
) -> CompactClassBindingFacets:
    try:
        enum_declaration = PYTHON_ENUM_BASE_AUTHORITY.declaration_from_node(
            node,
            identity=f"{module_name}.{qualname}",
            qualified_reference=lambda reference, shadowed_names: (
                _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    {},
                    reference,
                    shadowed_names,
                )
            ),
        )
    except ValueError:
        enum_declaration = NoEnumDeclaration()
    base_references = _compact_base_references(node, module_binding_snapshot)
    return CompactClassBindingFacets(
        base_references=base_references,
        base_references_are_complete=len(base_references) == len(node.bases),
        class_decorators_are_promotion_safe=all(
            ClassMethodPromotionSafeDecorator.for_qualified_name(
                _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    {},
                    decorator.func if isinstance(decorator, ast.Call) else decorator,
                    frozenset(),
                )
            )
            is not None
            for decorator in node.decorator_list
        ),
        dataclass_declaration=(
            _dataclass_declaration(module_binding_snapshot, qualname, node)
            if include_body_facets
            else NoDataclassDeclaration()
        ),
        enum_declaration=enum_declaration if include_body_facets else NoEnumDeclaration(),
    )


def _compact_indexed_classes(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    binding_snapshots: Mapping[int | None, ModuleNominalBindingSnapshot],
    *,
    include_body_facets: bool,
) -> tuple[CompactIndexedClass, ...]:
    file_path = parsed_module.file_path
    star_snapshots = (
        _module_nominal_binding_snapshots(
            ModuleNominalBindingAuthority(parsed_module),
            tuple(node.lineno for _qualname, node in indexed_class_nodes),
            include_final=False,
            star_import_excluded_names=_class_nominal_reference_root_names(parsed_module),
        )
        if module_star_import_origins(parsed_module)
        else None
    )
    body_scopes = (
        ClassBodyModuleScope.for_lines(
            parsed_module, (node.lineno for _qualname, node in indexed_class_nodes)
        )
        if include_body_facets
        else {}
    )
    return tuple(
        CompactIndexedClass(
            symbol=f"{parsed_module.module_name}.{qualname}",
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=file_path,
            line=node.lineno,
            declared_base_names=tuple(
                declared_name
                for base in node.bases
                if (
                    declared_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            base_references=binding_facets.base_references,
            mro_bases_are_static=all(
                isinstance(base, ast.Name | ast.Attribute) for base in node.bases
            ),
            direct_base_count=sum(
                ClassSymbolResolutionAuthority.establishes_nominal_family(
                    declared_base_name
                )
                for base in node.bases
                if (
                    declared_base_name := ClassSymbolResolutionAuthority.declared_base_name(
                        base
                    )
                )
                is not None
            ),
            base_references_are_complete=binding_facets.base_references_are_complete,
            product_base_bindings_are_exact=(
                binding_facets.product_base_bindings_are_exact
            ),
            direct_member_declarations=_compact_class_member_declarations(node),
            metaclass_names=tuple(
                terminal_name
                for keyword in node.keywords
                if keyword.arg == "metaclass"
                if (terminal_name := _terminal_reference_name(keyword.value))
                is not None
            ),
            class_keyword_names=tuple(
                keyword.spelled_name for keyword in CallKeywords.of(node.keywords).entries
            ),
            class_decorators_are_promotion_safe=(
                binding_facets.class_decorators_are_promotion_safe
            ),
            class_header_is_reconstructible=ClassHeaderSourceSpan(
                node,
                parsed_module.source_segments.lines,
            ).is_reconstructible,
            base_key_type_names=_compact_base_key_type_names(node),
            base_argument_sources=_compact_base_argument_sources(node),
            type_parameter_names=TYPE_PARAMETERS.names(node),
            is_final=binding_snapshots[node.lineno].decorated_by(node, (FINAL_DECORATOR,)),
            end_line=node.end_lineno,
            method_names=tuple(
                statement.name
                for statement in FunctionDefinition.within(node.body)
            ),
            callable_member_declarations=_compact_callable_member_declarations(
                node, binding_snapshots[node.lineno]
            ),
            abstract_method_names=sorted_tuple(
                statement.name
                for statement in FunctionDefinition.within(node.body)
                if binding_snapshots[node.lineno].decorated_by(
                    statement, (ABSTRACT_METHOD_DECORATOR,)
                )
            ),
            is_abstract=_is_abstract_class(node, binding_snapshots[node.lineno]),
            dataclass_declaration=binding_facets.dataclass_declaration,
            enum_declaration=binding_facets.enum_declaration,
            registry_facts=ClassRegistryFacts.of_class(
                node, module_binding_snapshot, parsed_module.module
            ),
            is_registration_authority=_is_registration_authority(node),
            autoregister_registry_key_attr_name=_autoregister_registry_key_attr_name(
                parsed_module,
                node,
            ),
            autoregister_registry_projection_names=(
                _autoregister_registry_projection_names(node)
                if include_body_facets
                else ()
            ),
            keyed_registry_lookup_method_names=(
                _keyed_registry_lookup_method_names(node) if include_body_facets else ()
            ),
            keyed_registry_reverse_lookup_method_names=(
                _keyed_registry_reverse_lookup_method_names(node)
                if include_body_facets
                else ()
            ),
            predicate_selected_methods=(
                _compact_predicate_selected_methods(node) if include_body_facets else ()
            ),
            body_declarations=(
                tuple(
                    declaration.compact()
                    for declaration in ClassBodyNormalForm.of_class(
                        node, body_scopes[node.lineno]
                    )
                )
                if include_body_facets
                else ()
            ),
            star_resolution=(
                None
                if star_snapshots is None
                else _star_resolved_facets(
                    node,
                    star_snapshots[node.lineno],
                    parsed_module.module_name,
                    qualname,
                    include_body_facets=include_body_facets,
                )
            ),
        )
        for qualname, node in indexed_class_nodes
        for module_binding_snapshot in (binding_snapshots[node.lineno],)
        for binding_facets in (
            _compact_class_binding_facets(
                node,
                module_binding_snapshot,
                parsed_module.module_name,
                qualname,
                include_body_facets=include_body_facets,
            ),
        )
    )


def _star_resolved_facets(
    node: ast.ClassDef,
    star_snapshot: ModuleNominalBindingSnapshot,
    module_name: str,
    qualname: str,
    *,
    include_body_facets: bool,
) -> CompactStarResolvedFacets:
    return CompactStarResolvedFacets(
        root_names=_class_reference_root_names(node),
        facets=_compact_class_binding_facets(
            node,
            star_snapshot,
            module_name,
            qualname,
            include_body_facets=include_body_facets,
        ),
    )


def star_resolved_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every class's star-import-dependent facts from the repository's
    export contracts: the one place those facts are resolved."""

    exposure = CompactRepositoryPublicExposureIndex(projections)
    return tuple(
        projection
        if not projection.star_import_origins
        else replace(
            projection,
            classes=tuple(
                indexed_class.resolved_past_star_imports(exposure)
                for indexed_class in projection.classes
            ),
        )
        for projection in projections
    )


def repository_settled_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every module fact the repository decides: names a star import
    may rebind, then the role of each field whose value calls a function."""

    return field_declaration_settled_projections(star_resolved_projections(projections))


def field_declaration_settled_projections(
    projections: tuple[ClassProjectionT, ...],
) -> tuple[ClassProjectionT, ...]:
    """Settle every field whose value calls a function, from the repository's
    field declarations: the one place those roles are decided."""

    index = FieldDeclarationIndex.of_wrappers(
        wrapper
        for projection in projections
        for wrapper in projection.field_declaration_wrappers
    )
    return tuple(
        replace(
            projection,
            classes=tuple(
                indexed_class.with_settled_field_declarations(index)
                for indexed_class in projection.classes
            ),
        )
        for projection in projections
    )


class CompactModuleClassProjectionFamily(CollectedFamily[CompactModuleClassProjection]):
    """Persist class/import facts needed by the global inheritance graph."""

    item_type = CompactModuleClassProjection
    cache_payload_max_bytes = 3_000_000
    source_demand_collector = staticmethod(
        _collect_demanded_class_projection_from_source
    )

    @classmethod
    def _required_module_projection(
        cls, parsed_module: ParsedModule,
    ) -> CompactModuleClassProjection:
        projections = collect_family_items(parsed_module, cls)
        if len(projections) != 1:
            raise ValueError(
                "Compact class projection family must emit one item per module"
            )
        return projections[0]

    @classmethod
    def report_demand(
        cls,
        target_items: tuple[object, ...],
        config: object,
    ) -> CompactClassProjectionDemand:
        """Derive the exact class facets required by report-target projections."""

        del config
        projections = tuple(
            item for item in target_items if isinstance(item, cls.item_type)
        )
        return CompactClassProjectionDemand(
            class_method_names=frozenset(
                method.method_name
                for projection in projections
                for method in projection.class_methods
            ),
            include_autoregister_references=any(
                projection.named_projection_surfaces
                or any(
                    bool(indexed_class.registry_facts.metaclass_references)
                    or (
                        indexed_class.declares_keyed_base
                        and "registry_key_attr"
                        in indexed_class.direct_declared_member_names
                    )
                    for indexed_class in projection.classes
                )
                for projection in projections
            ),
        )

    @classmethod
    def project_cached_demand(
        cls,
        items: tuple[object, ...],
        demand: object,
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Project cached class facts through their exact report demand."""

        if not isinstance(demand, CompactClassProjectionDemand):
            raise TypeError("class projection demand has the wrong authority type")
        projected = tuple(
            replace(
                item,
                class_methods=tuple(
                    method
                    for method in item.class_methods
                    if method.method_name in demand.class_method_names
                ),
                autoregister_reference_index=(
                    item.autoregister_reference_index
                    if demand.include_autoregister_references
                    else None
                ),
            )
            for item in items
            if isinstance(item, cls.item_type)
        )
        if not demand.header_core_only:
            return projected
        return tuple(item.header_core() for item in projected)

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        return cls._collect(parsed_module, None)

    @classmethod
    def collect_modules(
        cls,
        parsed_modules: Iterable[ParsedModule],
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Collect the single module projection declared by this family."""

        return repository_settled_projections(
            tuple(cls._required_module_projection(module) for module in parsed_modules)
        )

    @classmethod
    def collect_demanded(
        cls,
        parsed_module: ParsedModule,
        demand: object,
    ) -> list[CompactModuleClassProjection] | None:
        if not isinstance(demand, CompactClassProjectionDemand):
            raise TypeError("class projection demand has the wrong authority type")
        if demand.header_core_only:
            return cls._collect_header_core(parsed_module)
        return cls._collect(parsed_module, demand)

    @classmethod
    def header_core_projections(
        cls, parsed_modules: Iterable[ParsedModule]
    ) -> tuple[CompactModuleClassProjection, ...]:
        """Namespace and class headers only: enough to settle imports and exports."""

        return tuple(
            parsed_module.fact(ModuleClassHeaderCore).projection
            for parsed_module in parsed_modules
        )

    @classmethod
    def _collect_header_core(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        del cls
        return [
            CompactModuleClassProjection.declared_in(
                parsed_module, include_body_facets=False
            ).header_core()
        ]

    @classmethod
    def _collect(
        cls,
        parsed_module: ParsedModule,
        demand: CompactClassProjectionDemand | None,
    ) -> list[CompactModuleClassProjection]:
        del cls
        declarations = CompactModuleClassHeader.from_module(parsed_module)
        syntax_facets = parsed_module.fact(
            ModuleClassSyntaxFacets
            if demand is None or demand.include_autoregister_references
            else ModuleClassSyntaxFacetsWithoutAutoregister
        ).facets
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        class_methods = _compact_class_methods(
            parsed_module,
            indexed_class_nodes,
            method_names=(None if demand is None else demand.class_method_names),
        )
        return [
            CompactModuleClassProjection(
                **{
                    declaration.name: getattr(declarations, declaration.name)
                    for declaration in fields(CompactModuleClassHeader)
                },
                sorted_key_calls=_compact_sorted_key_calls(parsed_module),
                keyed_table_axes=_compact_keyed_table_axes(parsed_module),
                closed_axis_branch_functions=syntax_facets.closed_axis_branch_functions,
                manual_selector_axes=_compact_manual_selector_axes(parsed_module),
                top_level_definitions=tuple(
                    (node.name, node.lineno)
                    for node in parsed_module.module.body
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                ),
                type_variable_names=_module_type_variable_names(parsed_module),
                annotated_function_returns=tuple(
                    CompactAnnotatedFunctionReturn(
                        node.name, node.lineno, ast.unparse(returns)
                    )
                    for node in parsed_module.module.body
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    for returns in declared_annotations(node)
                ),
                exact_type_guards=syntax_facets.exact_type_guards,
                autoregister_reference_index=(
                    syntax_facets.autoregister_reference_index
                ),
                repeated_keyed_family_roots=_compact_repeated_keyed_family_roots(
                    parsed_module, declarations
                ),
                manual_subclass_roster_roots=_compact_manual_subclass_roster_roots(
                    parsed_module
                ),
                latent_rosters=LatentRosterObservation.from_module(parsed_module),
                named_projection_surfaces=_compact_named_projection_surfaces(
                    parsed_module
                ),
                class_methods=class_methods,
                class_references=_compact_class_references(parsed_module),
            )
        ]


def _autoregister_registry_projection_names(
    node: ast.ClassDef,
) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in FunctionDefinition.within(node.body)
        if any(
            isinstance(subnode, ast.Attribute)
            and subnode.attr in {"__registry__", "_registry", "registry"}
            for subnode in ast.walk(method)
        )
    )


def _method_references_cls_registry(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return RegistryLookupShape.references_registry(method)


def _keyed_registry_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in FunctionDefinition.within(node.body)
        if _is_classmethod(method) and _method_references_cls_registry(method)
    )


def _keyed_registry_reverse_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in FunctionDefinition.within(node.body)
        if _is_classmethod(method)
        and _method_references_cls_registry(method)
        and any(token in method.name for token in ("class", "type", "reverse"))
    )


def _trim_method_docstring(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(method.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _compact_repeated_keyed_family_roots(
    parsed_module: ParsedModule,
    declarations: CompactModuleClassHeader,
) -> tuple[CompactRepeatedKeyedFamilyRoot, ...]:
    roots: list[CompactRepeatedKeyedFamilyRoot] = []
    classes = {record.qualname: record for record in declarations.classes}
    for node in parsed_module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assignments = _direct_class_assignments(node)
        registry_key_attr_name = _string_constant(assignments.get("registry_key_attr"))
        registry = assignments.get("_registry")
        registry_is_empty = (
            isinstance(registry, ast.Dict) and not registry.keys and not registry.values
        ) or (
            isinstance(registry, ast.Call)
            and isinstance(registry.func, ast.Name)
            and registry.func.id == "dict"
        )
        if registry_key_attr_name is None or not registry_is_empty:
            continue
        lookup_methods = tuple(
            (method, shape)
            for method in FunctionDefinition.within(node.body)
            if _is_classmethod(method)
            and method.name.startswith("for_")
            and (shape := RegistryLookupShape.from_method(method)) is not None
        )
        if len(lookup_methods) != 1:
            continue
        lookup_method, lookup_shape = lookup_methods[0]
        roots.append(
            CompactRepeatedKeyedFamilyRoot(
                file_path=parsed_module.file_path,
                line=node.lineno,
                class_name=node.name,
                family_bases=tuple(
                    DeclaredFamilyBase.of(reference)
                    for reference in classes[node.name].base_references
                ),
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                error_type_name=lookup_shape.error_type_name,
                abstract_hook_names=tuple(
                    method.name
                    for method in FunctionDefinition.within(node.body)
                    if any(
                        _terminal_reference_name(decorator) == "abstractmethod"
                        for decorator in method.decorator_list
                    )
                ),
            )
        )
    return tuple(roots)


def _classmethod_receiver_class(
    node: ast.AST,
    scope: LexicalSyntaxScope,
    nodes: Sequence[ast.AST],
    parent_indices: Sequence[int],
    module_bound_names: frozenset[str],
) -> tuple[str, ...] | None:
    """The class a `@classmethod`'s receiver parameter denotes, read directly in
    that method: its class-name path, which resolves like a module-level name."""

    if not isinstance(node, ast.Name) or not scope.function_node_indices:
        return None
    function_index = scope.function_node_indices[-1]
    function = nodes[function_index]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if not isinstance(nodes[parent_indices[function_index]], ast.ClassDef):
        return None
    if "classmethod" in module_bound_names or not any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in function.decorator_list
    ):
        return None
    positional = (*function.args.posonlyargs, *function.args.args)
    if not positional or positional[0].arg != node.id:
        return None
    if not scope.class_names or scope.class_names[0] not in module_bound_names:
        return None
    return scope.class_names


def _copied_keyword_fields(call: ast.Call) -> tuple[str, ...]:
    """The keywords one call passes as `x.<same name>`, for the value `x` that
    supplies the most of them."""

    by_root: dict[str, list[str]] = defaultdict(list)
    for keyword in call.keywords:
        value = keyword.value
        if (
            keyword.arg is not None
            and isinstance(value, ast.Attribute)
            and value.attr == keyword.arg
            and isinstance(value.value, ast.Name)
        ):
            by_root[value.value.id].append(keyword.arg)
    return max(
        (tuple(sorted(names)) for names in by_root.values()),
        key=lambda names: (len(names), names),
        default=(),
    )


def _compact_class_references(
    parsed_module: ParsedModule,
) -> tuple[CompactClassReference, ...]:
    """Every read of a module-level name chain, by consumer and role.

    Only chains rooted at a name the module itself binds are kept, and a root a
    function rebinds locally is that function's own. The join resolves the
    chains against the class index; most name no class and are dropped there.
    """

    syntax_index = parsed_module.syntax_index
    module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    import_aliases = module_import_aliases(parsed_module)
    locals_by_function_index = {
        function_index: LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
        | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
        for function_index in {
            index
            for scope in syntax_index.scopes
            for index in scope.function_node_indices
        }
        if isinstance(
            function := syntax_index.depth_first_nodes[function_index],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    }
    nodes = syntax_index.depth_first_nodes
    parent_indices = syntax_index.parent_indices
    references: dict[
        tuple[str, tuple[str, ...], ClassReferenceRole],
        tuple[int, int, bool, tuple[str, ...]],
    ] = {}
    for node_type in (ast.Name, ast.Attribute):
        for node_index in syntax_index.node_indices_by_type.get(node_type, ()):
            node = nodes[node_index]
            if not isinstance(getattr(node, "ctx", None), ast.Load):
                continue
            parent_index = parent_indices[node_index]
            parent = nodes[parent_index] if parent_index >= 0 else None
            if isinstance(parent, ast.Attribute) and parent.value is node:
                continue
            scope = syntax_index.scopes[syntax_index.scope_ids[node_index]]
            receiver_class = _classmethod_receiver_class(
                node, scope, nodes, parent_indices, module_bound_names
            )
            if receiver_class is not None:
                parts = receiver_class
            else:
                parts = ExpressionReference.decoded(node).attribute_chain
                if parts is None or parts[0] not in module_bound_names:
                    continue
                if any(
                    parts[0] in locals_by_function_index.get(function_index, ())
                    for function_index in scope.function_node_indices
                ):
                    continue
            grandparent_index = parent_indices[parent_index] if parent is not None else -1
            grandparent = nodes[grandparent_index] if grandparent_index >= 0 else None
            role = ClassReferenceRole.of_reference(node, parent, grandparent, import_aliases)
            key = (".".join(scope.names), parts, role)
            positional, unpacked = (
                (
                    len(parent.args),
                    any(isinstance(argument, ast.Starred) for argument in parent.args)
                    or bool(CallKeywords.of(parent.keywords).splats),
                )
                if role is ClassReferenceRole.CONSTRUCTION and isinstance(parent, ast.Call)
                else (0, False)
            )
            copied = (
                _copied_keyword_fields(parent)
                if role is ClassReferenceRole.CONSTRUCTION and isinstance(parent, ast.Call)
                else ()
            )
            line, seen_positional, seen_unpacked, seen_copied = references.get(
                key, (node.lineno, 0, False, ())
            )
            references[key] = (
                min(line, node.lineno),
                max(seen_positional, positional),
                seen_unpacked or unpacked,
                max(seen_copied, copied, key=lambda names: (len(names), names)),
            )
    return tuple(
        CompactClassReference(consumer, parts, role, *facts)
        for (consumer, parts, role), facts in sorted(
            references.items(), key=lambda item: (item[1][0], item[0][0], item[0][1])
        )
    )


def _is_abstract_class(
    node: ast.ClassDef, module_binding_snapshot: ModuleNominalBindingSnapshot
) -> bool:
    if {"ABC", "ABCMeta"} & {
        terminal_name
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    }:
        return True
    return any(
        module_binding_snapshot.decorated_by(statement, (ABSTRACT_METHOD_DECORATOR,))
        for statement in FunctionDefinition.within(node.body)
    )


def _class_scope_qualified_import_name(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    reference: ast.AST,
    preceding_class_bound_names: frozenset[str],
) -> str | None:
    parts = ExpressionReference.decoded(
        ClassSymbolResolutionAuthority.reference_node(reference)
    ).attribute_chain
    if parts is None:
        return None
    if (class_alias := class_aliases.get(parts[0])) is not None:
        return ".".join((class_alias, *parts[1:]))
    if parts[0] in preceding_class_bound_names:
        return None
    root_binding = module_binding_snapshot.binding_for(parts[0])
    if root_binding is None:
        return None
    return ".".join((root_binding.qualified_name, *parts[1:]))


def _dataclass_field_role(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    statement: ast.AnnAssign,
    preceding_class_bound_names: frozenset[str],
) -> CompactDataclassFieldRole:
    annotation_root = _dataclass_role_reference(statement.annotation)
    qualified_annotation = (
        None
        if annotation_root is None
        else _class_scope_qualified_import_name(
            module_binding_snapshot,
            class_aliases,
            annotation_root,
            preceding_class_bound_names,
        )
    )
    if (
        semantic_role := CompactDataclassFieldRole.for_qualified_annotation(
            qualified_annotation
        )
    ) is not None:
        return semantic_role
    annotation_is_resolved_plain_field = (
        (
            annotation_root is None
            and _dataclass_plain_annotation_is_resolved(
                module_binding_snapshot,
                class_aliases,
                statement.annotation,
                preceding_class_bound_names,
            )
        )
        or qualified_annotation is not None
        or (
            isinstance(annotation_root, ast.Name)
            and module_binding_snapshot.resolves_unshadowed_builtin(
                annotation_root.id,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
    )
    if not annotation_is_resolved_plain_field:
        return CompactDataclassFieldRole.UNRESOLVED
    return StatementValue.of(statement).answer(
        DataclassFieldDefaultRole(
            module_binding_snapshot, class_aliases, preceding_class_bound_names
        )
    )


@dataclass(frozen=True)
class DataclassFieldDefaultRole(StatementValueQuestion[CompactDataclassFieldRole]):
    """The role a resolved plain field's default gives it; no default stores an init field."""

    module_binding_snapshot: ModuleNominalBindingSnapshot
    class_aliases: dict[str, str]
    preceding_class_bound_names: frozenset[str]

    def bare(self, value: NoStatementValue) -> CompactDataclassFieldRole:
        return CompactDataclassFieldRole.STORED_INIT

    def supplied(self, supplied: SuppliedValue) -> CompactDataclassFieldRole:
        value = supplied.value
        if _is_literal_default(value):
            return CompactDataclassFieldRole.STORED_INIT
        if not isinstance(value, ast.Call):
            return CompactDataclassFieldRole.UNRESOLVED
        qualified_default_factory = _class_scope_qualified_import_name(
            self.module_binding_snapshot,
            self.class_aliases,
            value.func,
            self.preceding_class_bound_names,
        )
        default_factory_declaration = DataclassRuntimeDeclaration.for_qualified_name(
            qualified_default_factory
        )
        if qualified_default_factory is not None and default_factory_declaration is None:
            return CompactDataclassFieldRole.DECLARED_BY_CALL
        if (
            default_factory_declaration is None
            or not default_factory_declaration.is_field_factory
        ):
            return CompactDataclassFieldRole.UNRESOLVED
        if value.args or CallKeywords.of(value.keywords).splats:
            return CompactDataclassFieldRole.UNRESOLVED
        init_values = tuple(
            keyword.value for keyword in value.keywords if keyword.arg == "init"
        )
        if len(init_values) > 1:
            return CompactDataclassFieldRole.UNRESOLVED
        if not init_values:
            return CompactDataclassFieldRole.STORED_INIT
        init_value = init_values[0]
        if not isinstance(init_value, ast.Constant) or not isinstance(
            init_value.value,
            bool,
        ):
            return CompactDataclassFieldRole.UNRESOLVED
        return (
            CompactDataclassFieldRole.STORED_INIT
            if init_value.value
            else CompactDataclassFieldRole.STORED_NON_INIT
        )


def _native_qualified_name(value: object) -> str:
    """The name a module binding resolves one native module attribute to."""

    module = sys.modules[type(value).__module__]
    return next(
        f"{module.__name__}.{name}"
        for name, attribute in vars(module).items()
        if attribute is value
    )


_MISSING_QUALIFIED_NAME = _native_qualified_name(MISSING)
_CAST_QUALIFIED_NAME = f"{typing.cast.__module__}.{typing.cast.__qualname__}"


class CompactFieldDeclarationCall(NamedTuple):
    """A call that may declare a dataclass field: its callee and what it supplies.

    `positional_count` parameters are filled by position and `keywords` by
    name; a `*` or `**` argument sets `splats`, and what the call supplies is
    then unknown. A keyword forwarding a parameter the caller left absent
    supplies nothing and is not listed.
    """

    callee: str
    positional_count: int
    keywords: tuple[str, ...]
    splats: bool

    @classmethod
    def of_call(
        cls, callee: str, call: ast.Call, absent: frozenset[str] = frozenset()
    ) -> Self:
        keywords = CallKeywords.of(call.keywords)
        return cls(
            callee=callee,
            positional_count=len(call.args),
            keywords=tuple(
                sorted(
                    name
                    for name, value in keywords.by_name.items()
                    if not (isinstance(value, ast.Name) and value.id in absent)
                )
            ),
            splats=bool(keywords.splats)
            or any(isinstance(argument, ast.Starred) for argument in call.args),
        )


class FieldDeclarationSignature(NamedTuple):
    """What a field declaration accepts: its positional parameters in order, the
    parameters that give the field a default when supplied, every
    parameter name its own calls supply on the way to `dataclasses.field`, and
    whether those calls already give the field a default."""

    positional: tuple[str, ...]
    missing: tuple[str, ...]
    supplied_on_the_way: tuple[str, ...] = ()
    defaulted: bool = False

    @classmethod
    def of_native(cls, declaration: Callable[..., object]) -> Self:
        parameters = inspect.signature(declaration).parameters.values()
        return cls(
            positional=tuple(
                parameter.name
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ),
            missing=tuple(
                parameter.name
                for parameter in parameters
                if parameter.default is MISSING
                and cls._gives_a_default(declaration(**{parameter.name: list}))
            ),
        )

    @staticmethod
    def _gives_a_default(declared: object) -> bool:
        """Whether a native field declaration, built with one parameter supplied,
        has a default: dataclasses reads a field as defaulted exactly when its
        `default` or `default_factory` is not MISSING (`kw_only` is not one)."""

        return isinstance(declared, Field) and (
            declared.default is not MISSING or declared.default_factory is not MISSING
        )

    def supplied(self, call: CompactFieldDeclarationCall) -> frozenset[str] | None:
        """The parameters the call names, or None when a splat hides them."""

        if call.splats or call.positional_count > len(self.positional):
            return None
        return frozenset((*self.positional[: call.positional_count], *call.keywords))

    def declares_bare_field(self, call: CompactFieldDeclarationCall) -> bool | None:
        """Whether the call declares a field with no default; None when it is open."""

        supplied = self.supplied(call)
        if supplied is None:
            return None
        return not self.defaulted and not supplied.intersection(self.missing)


class _FieldDeclarationFoldOutcome(ABC):
    """What a run of statements does once the absent parameters are folded."""

    @abstractmethod
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        raise NotImplementedError

    def returned(self) -> ast.expr | None:
        return None


@dataclass(frozen=True)
class _FieldDeclarationReturned(_FieldDeclarationFoldOutcome):
    value: ast.expr

    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self

    def returned(self) -> ast.expr | None:
        return self.value


class _FieldDeclarationRaised(_FieldDeclarationFoldOutcome):
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self


class _FieldDeclarationFellThrough(_FieldDeclarationFoldOutcome):
    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return fold.statements(rest)


class _FieldDeclarationUnfolded(_FieldDeclarationFoldOutcome):
    """A statement or test the fold cannot decide: the function stays open."""

    def then(
        self, fold: _FieldDeclarationFold, rest: Sequence[ast.stmt]
    ) -> _FieldDeclarationFoldOutcome:
        return self


class _ReturnedFieldDeclaration(StatementValueQuestion[_FieldDeclarationFoldOutcome]):
    """A `return` hands back its value; a bare `return` declares no field."""

    def supplied(self, value: SuppliedValue) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationReturned(value.value)

    def bare(self, value: NoStatementValue) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationUnfolded()


@dataclass(frozen=True)
class _FieldDeclarationFold:
    """Run a function body with its MISSING-defaulted parameters absent: each
    `parameter is MISSING` test is decided, everything else stays open."""

    snapshot: ModuleNominalBindingSnapshot
    absent: frozenset[str]

    def statements(self, statements: Sequence[ast.stmt]) -> _FieldDeclarationFoldOutcome:
        if not statements:
            return _FieldDeclarationFellThrough()
        return self.statement(statements[0]).then(self, statements[1:])

    @singledispatchmethod
    def statement(self, statement: ast.stmt) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationUnfolded()

    @statement.register
    def _return(self, statement: ast.Return) -> _FieldDeclarationFoldOutcome:
        return StatementValue.of(statement).answer(_ReturnedFieldDeclaration())

    @statement.register
    def _raise(self, statement: ast.Raise) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationRaised()

    @statement.register
    def _pass(self, statement: ast.Pass) -> _FieldDeclarationFoldOutcome:
        return _FieldDeclarationFellThrough()

    @statement.register
    def _if(self, statement: ast.If) -> _FieldDeclarationFoldOutcome:
        taken = self.test(statement.test)
        if taken is None:
            return _FieldDeclarationUnfolded()
        return self.statements(statement.body if taken else statement.orelse)

    @singledispatchmethod
    def test(self, test: ast.expr) -> bool | None:
        return None

    @test.register
    def _compare(self, test: ast.Compare) -> bool | None:
        match test:
            case ast.Compare(
                left=ast.Name(id=name), ops=[op], comparators=[right]
            ) if name in self.absent and self.qualified(right) == _MISSING_QUALIFIED_NAME:
                return self.absence(op)
        return None

    @singledispatchmethod
    def absence(self, op: ast.cmpop) -> bool | None:
        """What `parameter <op> MISSING` says when the parameter is absent."""

        return None

    @absence.register
    def _is(self, op: ast.Is) -> bool | None:
        return True

    @absence.register
    def _is_not(self, op: ast.IsNot) -> bool | None:
        return False

    @test.register
    def _boolean(self, test: ast.BoolOp) -> bool | None:
        values = tuple(self.test(value) for value in test.values)
        if None in values:
            return None
        return self.combined(test.op, values)

    @singledispatchmethod
    def combined(self, op: ast.boolop, values: tuple[bool, ...]) -> bool:
        raise TypeError(f"no boolean operator {type(op).__name__}")

    @combined.register
    def _all(self, op: ast.And, values: tuple[bool, ...]) -> bool:
        return all(values)

    @combined.register
    def _any(self, op: ast.Or, values: tuple[bool, ...]) -> bool:
        return any(values)

    @test.register
    def _unary(self, test: ast.UnaryOp) -> bool | None:
        value = self.test(test.operand)
        return None if value is None else self.unary(test.op, value)

    @singledispatchmethod
    def unary(self, op: ast.unaryop, value: bool) -> bool | None:
        return None

    @unary.register
    def _not(self, op: ast.Not, value: bool) -> bool:
        return not value

    def qualified(self, reference: ast.expr) -> str | None:
        parts = ExpressionReference.decoded(reference).attribute_chain
        if parts is None:
            return None
        binding = self.snapshot.binding_for(parts[0])
        if binding is None:
            return None
        return ".".join((binding.qualified_name, *parts[1:]))

    def returned_call(self, value: ast.expr) -> CompactFieldDeclarationCall | None:
        """The field-declaration call a returned value makes, seeing through `cast`."""

        if not isinstance(value, ast.Call):
            return None
        callee = self.qualified(value.func)
        if callee is None:
            return None
        if callee == _CAST_QUALIFIED_NAME and len(value.args) == 2 and not value.keywords:
            return self.returned_call(value.args[1])
        return CompactFieldDeclarationCall.of_call(callee, value, self.absent)


class CompactFieldDeclarationWrapper(NamedTuple):
    """A module function that, with its MISSING-defaulted parameters absent,
    returns one call: a field declaration when that call is one."""

    qualified_name: str
    signature: FieldDeclarationSignature
    returned: CompactFieldDeclarationCall

    @classmethod
    def of_function(
        cls,
        module_name: str,
        function: ast.FunctionDef,
        snapshot: ModuleNominalBindingSnapshot,
    ) -> Self | None:
        """The wrapper the function is, or None when its body stays open."""

        arguments = function.args
        if variadic_parameters(arguments):
            return None
        positional = (*arguments.posonlyargs, *arguments.args)
        defaulted = (
            *zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults),
            *(
                (argument, default)
                for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
                if default is not None
            ),
        )
        fold = _FieldDeclarationFold(snapshot, frozenset())
        # Every MISSING parameter of a wrapper is read as giving a default when
        # supplied: wrong only toward "a different field", never a false match.
        missing = tuple(
            argument.arg
            for argument, default in defaulted
            if fold.qualified(default) == _MISSING_QUALIFIED_NAME
        )
        fold = _FieldDeclarationFold(snapshot, frozenset(missing))
        value = fold.statements(statements_without_docstring(function.body)).returned()
        returned = None if value is None else fold.returned_call(value)
        if returned is None:
            return None
        if not missing and DataclassRuntimeDeclaration.for_qualified_name(
            returned.callee
        ) is None:
            return None
        return cls(
            qualified_name=f"{module_name}.{function.name}",
            signature=FieldDeclarationSignature(
                positional=tuple(argument.arg for argument in positional),
                missing=missing,
            ),
            returned=returned,
        )


def module_field_declaration_wrappers(
    parsed_module: ParsedModule,
    snapshot: ModuleNominalBindingSnapshot,
) -> tuple[CompactFieldDeclarationWrapper, ...]:
    """Every top-level function of the module that may wrap `dataclasses.field`,
    its names read through the module's final bindings."""

    return tuple(
        wrapper
        for statement in parsed_module.module.body
        if isinstance(statement, ast.FunctionDef)
        if (
            wrapper := CompactFieldDeclarationWrapper.of_function(
                parsed_module.module_name, statement, snapshot
            )
        )
        is not None
    )


@dataclass(frozen=True)
class FieldDeclarationIndex:
    """Which callees declare a dataclass field, across one repository.

    A native field factory declares one directly. A wrapper declares one when
    the call it returns does; the field has a default when that call supplies
    one of the callee's MISSING parameters. A wrapper reaching itself again
    declares nothing.
    """

    wrappers: Mapping[str, CompactFieldDeclarationWrapper]

    @classmethod
    def of_wrappers(cls, wrappers: Iterable[CompactFieldDeclarationWrapper]) -> Self:
        return cls({wrapper.qualified_name: wrapper for wrapper in wrappers})

    @cached_property
    def signatures(self) -> Mapping[str, FieldDeclarationSignature]:
        signatures: dict[str, FieldDeclarationSignature] = {
            member.qualified_name: FieldDeclarationSignature.of_native(member.declaration)
            for member in DataclassRuntimeDeclaration
            if member.is_field_factory
        }
        refused: set[str] = set()

        def settle(name: str, pending: frozenset[str]) -> FieldDeclarationSignature | None:
            if name in signatures:
                return signatures[name]
            wrapper = self.wrappers.get(name)
            if wrapper is None or name in refused or name in pending:
                return None
            callee = settle(wrapper.returned.callee, pending | {name})
            supplied = None if callee is None else callee.supplied(wrapper.returned)
            if callee is None or supplied is None:
                refused.add(name)
                return None
            signature = wrapper.signature._replace(
                supplied_on_the_way=tuple(sorted({*callee.supplied_on_the_way, *supplied})),
                defaulted=callee.defaulted or bool(supplied.intersection(callee.missing)),
            )
            signatures[name] = signature
            return signature

        for name in self.wrappers:
            settle(name, frozenset())
        return signatures

    def declares_bare_field(self, call: CompactFieldDeclarationCall) -> bool | None:
        """Whether the call declares a field without a default: False when its
        callee declares no field, None when a splat leaves it open."""

        signature = self.signatures.get(call.callee)
        if signature is None:
            return False
        return signature.declares_bare_field(call)

    def field_role(self, call: CompactFieldDeclarationCall) -> CompactDataclassFieldRole:
        """A stored init field when the call declares a field and nothing on its
        way to `dataclasses.field` names `init`; otherwise unresolved."""

        signature = self.signatures.get(call.callee)
        if signature is None or signature.supplied(call) is None or self.supplies(call, "init"):
            return CompactDataclassFieldRole.UNRESOLVED
        return CompactDataclassFieldRole.STORED_INIT

    def supplies(self, call: CompactFieldDeclarationCall, parameter: str) -> bool:
        """Whether the call, or a declaration it reaches, names this parameter."""

        signature = self.signatures[call.callee]
        supplied = signature.supplied(call) or frozenset()
        return parameter in supplied or parameter in signature.supplied_on_the_way


def _dataclass_role_reference(annotation: ast.expr) -> ast.expr | None:
    """Return only a top-level reference that can alter dataclass field role."""

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return annotation
    if isinstance(annotation, ast.Subscript):
        return annotation.value
    return annotation if isinstance(annotation, ast.Name | ast.Attribute) else None


_LITERAL_DEFAULT_NODES = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.UnaryOp,
    ast.BinOp,
    ast.unaryop,
    ast.operator,
    ast.expr_context,
)


def _is_literal_default(value: ast.expr) -> bool:
    """A default built only from literals: it names nothing, so it cannot be a
    `dataclasses.Field` and the member is a plain stored field."""

    return all(isinstance(node, _LITERAL_DEFAULT_NODES) for node in ast.walk(value))


def _dataclass_plain_annotation_is_resolved(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    annotation: ast.expr,
    preceding_class_bound_names: frozenset[str],
) -> bool:
    """Prove every evaluated root in a non-role annotation expression."""

    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
    root_names = frozenset(
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    return all(
        root_name in class_aliases
        or module_binding_snapshot.binding_for(root_name) is not None
        or module_binding_snapshot.resolves_unshadowed_builtin(
            root_name,
            preceding_class_bound_names=preceding_class_bound_names,
        )
        for root_name in root_names
    )


def _dataclass_fields(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    node: ast.ClassDef,
    class_binding: DataclassInitBinding,
) -> tuple[CompactDataclassFieldDeclaration, ...]:
    fields: list[CompactDataclassFieldDeclaration] = []
    preceding_bound_names: set[str] = set()
    class_aliases: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            role = _dataclass_field_role(
                module_binding_snapshot,
                class_aliases,
                statement,
                frozenset(preceding_bound_names),
            )
            fields.append(
                CompactDataclassFieldDeclaration(
                    name=statement.target.id,
                    line=statement.lineno,
                    role=role,
                    binding=DataclassInitBinding.of_keywords(
                        (
                            statement.value.keywords
                            if isinstance(statement.value, ast.Call)
                            else ()
                        ),
                        class_binding,
                    ),
                    declaration_call=role.declaration_call(
                        statement,
                        module_binding_snapshot,
                        class_aliases,
                        frozenset(preceding_bound_names),
                    ),
                )
            )
            class_binding = role.binding_of_later_fields(class_binding)
        direct_aliases: dict[str, str] = {}
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            for alias_target, assigned in StatementValues.keyed(
                (target, statement) for target in single_name_targets(statement)
            ).supplied:
                qualified_value = _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    class_aliases,
                    assigned.value,
                    frozenset(preceding_bound_names),
                )
                if qualified_value is not None:
                    direct_aliases[alias_target.id] = qualified_value
        statement_bound_names = _direct_statement_bound_names(statement)
        preceding_bound_names.update(statement_bound_names)
        for bound_name in statement_bound_names:
            class_aliases.pop(bound_name, None)
        class_aliases.update(direct_aliases)
    return tuple(fields)


def _field_member_collision_lines(
    node: ast.ClassDef,
    fields: tuple[CompactDataclassFieldDeclaration, ...],
) -> tuple[int, ...]:
    field_names = frozenset(field.name for field in fields)
    seen_field_bindings: set[str] = set()
    collision_lines: set[int] = set()
    for statement in node.body:
        bound_fields = field_names.intersection(
            _direct_statement_bound_names(statement)
        )
        if seen_field_bindings.intersection(bound_fields):
            collision_lines.add(statement.lineno)
        seen_field_bindings.update(bound_fields)
    return tuple(sorted(collision_lines))


def _dynamic_dataclass_schema_lines(node: ast.ClassDef) -> tuple[int, ...]:
    """Return class-execution sites that can change the direct field schema."""

    lines: set[int] = set()

    class DynamicSchemaVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            return None

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, child: ast.Lambda) -> None:
            return None

        def visit_AnnAssign(self, child: ast.AnnAssign) -> None:
            lines.add(child.lineno)

        def visit_Name(self, child: ast.Name) -> None:
            if child.id == "__annotations__" and isinstance(child.ctx, ast.Store):
                lines.add(child.lineno)

        def visit_Subscript(self, child: ast.Subscript) -> None:
            if (
                isinstance(child.ctx, ast.Store | ast.Del)
                and isinstance(child.value, ast.Name)
                and child.value.id == "__annotations__"
            ):
                lines.add(child.lineno)
            self.generic_visit(child)

        def visit_Call(self, child: ast.Call) -> None:
            called_parts = ExpressionReference.decoded(child.func).attribute_chain
            if called_parts == ("exec",) or called_parts in {
                ("__annotations__", "clear"),
                ("__annotations__", "pop"),
                ("__annotations__", "popitem"),
                ("__annotations__", "setdefault"),
                ("__annotations__", "update"),
            }:
                lines.add(child.lineno)
            elif (
                len(called_parts or ()) == 2
                and called_parts[-1] in {"setdefault", "update"}
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Call)
                and ExpressionReference.decoded(child.func.value.func).attribute_chain
                in {("locals",), ("vars",)}
            ):
                lines.add(child.lineno)
            self.generic_visit(child)

    visitor = DynamicSchemaVisitor()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign):
            continue
        visitor.visit(statement)
    return tuple(sorted(lines))


def _dataclass_declaration(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    qualname: str,
    node: ast.ClassDef,
) -> CompactDataclassFacet:
    decorator_qualified_names = tuple(
        _class_scope_qualified_import_name(
            module_binding_snapshot,
            {},
            decorator.func if isinstance(decorator, ast.Call) else decorator,
            frozenset(),
        )
        for decorator in node.decorator_list
    )
    decorator_runtime_declarations = tuple(
        DataclassRuntimeDeclaration.dataclass_decorator_for_name(qualified_name)
        for qualified_name in decorator_qualified_names
    )
    dataclass_decorator_indexes = tuple(
        index
        for index, runtime_declaration in enumerate(decorator_runtime_declarations)
        if runtime_declaration is not None
    )
    dataclass_like_indexes = tuple(
        index
        for index, decorator in enumerate(node.decorator_list)
        if DataclassRuntimeDeclaration.dataclass_decorator_for_name(
            _terminal_reference_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )
        )
        is not None
    )
    dataclass_candidate_indexes = frozenset(
        (*dataclass_decorator_indexes, *dataclass_like_indexes)
    )
    if not dataclass_candidate_indexes:
        return NoDataclassDeclaration()

    decorator = node.decorator_list[min(dataclass_candidate_indexes)]
    fields = _dataclass_fields(
        module_binding_snapshot,
        node,
        DataclassInitBinding.of_keywords(
            decorator.keywords if isinstance(decorator, ast.Call) else (),
            DataclassInitBinding.POSITIONAL,
        ),
    )
    failures: list[CompactProductDeclarationFailure] = []
    if not dataclass_decorator_indexes:
        failures.extend(
            CompactProductDeclarationFailure(
                node.decorator_list[index].lineno,
                CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR,
            )
            for index in dataclass_candidate_indexes
        )
    if len(dataclass_candidate_indexes) != 1:
        failures.extend(
            CompactProductDeclarationFailure(
                node.decorator_list[index].lineno,
                CompactProductAuthorityViolation.MULTIPLE_DATACLASS_DECORATORS,
            )
            for index in sorted(dataclass_candidate_indexes)
        )
    if dataclass_decorator_indexes:
        failures.extend(
            _dataclass_decorator_failures(
                node.decorator_list[dataclass_decorator_indexes[0]]
            )
        )
    failures.extend(
        CompactProductDeclarationFailure(
            node.decorator_list[index].lineno,
            CompactProductAuthorityViolation.CUSTOM_CLASS_DECORATOR,
        )
        for index, qualified_name in enumerate(decorator_qualified_names)
        if index not in dataclass_decorator_indexes
        if not ClassMethodPromotionSafeDecorator.qualified_name_preserves_product_schema(
            qualified_name
        )
    )
    failures.extend(
        CompactProductDeclarationFailure(
            keyword.value.lineno,
            CompactProductAuthorityViolation.CUSTOM_CLASS_CREATION,
        )
        for keyword in node.keywords
    )
    if "." in qualname:
        failures.append(
            CompactProductDeclarationFailure(
                node.lineno,
                CompactProductAuthorityViolation.NESTED_CLASS_SCOPE,
            )
        )
    lifecycle_violation = CompactProductAuthorityViolation.CUSTOM_PRODUCT_LIFECYCLE
    failures.extend(
        CompactProductDeclarationFailure(statement.lineno, lifecycle_violation)
        for statement in node.body
        if lifecycle_violation.is_violated_by_member_names(
            _direct_statement_bound_names(statement)
        )
    )
    failures.extend(
        CompactProductDeclarationFailure(
            line,
            CompactProductAuthorityViolation.FIELD_MEMBER_COLLISION,
        )
        for line in _field_member_collision_lines(node, fields)
    )
    failures.extend(
        CompactProductDeclarationFailure(
            line,
            CompactProductAuthorityViolation.DYNAMIC_FIELD_SCHEMA,
        )
        for line in _dynamic_dataclass_schema_lines(node)
    )
    return CompactDataclassDeclaration(
        runtime_declaration=next(
            (
                runtime_declaration
                for runtime_declaration in decorator_runtime_declarations
                if runtime_declaration is not None
            ),
            None,
        ),
        fields=fields,
        failures=tuple(dict.fromkeys(failures)),
    )


TYPE_VARIABLE_CONSTRUCTORS = frozenset(
    f"{module}.{name}"
    for module in ("typing", "typing_extensions")
    for name in ("TypeVar", "ParamSpec", "TypeVarTuple")
)


def _module_type_variable_names(parsed_module: ParsedModule) -> tuple[str, ...]:
    """Names the module binds at top level to a type variable constructor."""

    aliases = module_import_aliases(parsed_module)
    names: list[str] = []
    for statement in parsed_module.module.body:
        match statement:
            case ast.Assign(targets=targets, value=ast.Call(func=constructor)):
                parts = ExpressionReference.decoded(constructor).attribute_chain
                if not parts or parts[0] not in aliases:
                    continue
                qualified = ".".join((aliases[parts[0]], *parts[1:]))
                if qualified in TYPE_VARIABLE_CONSTRUCTORS:
                    names.extend(
                        target.id for target in targets if isinstance(target, ast.Name)
                    )
    return tuple(names)


def _compact_callable_member_declarations(
    node: ast.ClassDef, module_binding_snapshot: ModuleNominalBindingSnapshot
) -> tuple[CompactCallableMemberDeclaration, ...]:
    """Each `def` of the class body, except overload stubs and a property's
    setter or deleter, which rebind no new member type."""

    declarations: list[CompactCallableMemberDeclaration] = []
    property_names: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if module_binding_snapshot.decorated_by(
            statement, (OVERLOAD_DECORATOR,)
        ) or _redefines_property(statement, property_names):
            continue
        reads_as_property = module_binding_snapshot.decorated_by(
            statement, PROPERTY_DECORATORS
        )
        if reads_as_property:
            property_names.add(statement.name)
        declared = tuple(
            (CompactAnnotatedProperty if reads_as_property else CompactAnnotatedMethod)(
                statement.name, statement.lineno, ast.unparse(returns)
            )
            for returns in declared_annotations(statement)
        )
        declarations.extend(
            declared or (CompactUnannotatedCallable(statement.name, statement.lineno),)
        )
    return tuple(declarations)


def _redefines_property(
    statement: ast.FunctionDef | ast.AsyncFunctionDef, property_names: Collection[str]
) -> bool:
    """`@name.setter` or `@name.deleter` on a property the class body declared above."""

    return any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id in property_names
        and decorator.attr in PROPERTY_REDEFINITION_ATTRIBUTES
        for decorator in statement.decorator_list
    )


def native_identity(value: Callable[..., object]) -> str:
    """The qualified name a binding resolves a native object to."""

    return f"{value.__module__}.{value.__qualname__}"


PROPERTY_DECORATOR = native_identity(property)
PROPERTY_DECORATORS = frozenset((PROPERTY_DECORATOR, native_identity(cached_property)))
"""The descriptors whose read is the decorated function's return."""
PROPERTY_REDEFINITION_ATTRIBUTES = frozenset(
    (property.setter.__name__, property.deleter.__name__)
)
ABSTRACT_METHOD_DECORATOR = native_identity(abstractmethod)
FINAL_DECORATOR = native_identity(typing.final)
OVERLOAD_DECORATOR = native_identity(typing.overload)
STATIC_METHOD_DECORATOR = native_identity(staticmethod)
CLASS_METHOD_DECORATOR = native_identity(classmethod)


class InstalledClassDeclarations:
    """Classes of modules outside the scan, projected from installed source and never run."""

    MAX_REEXPORTS: ClassVar[int] = 8

    @staticmethod
    @cache
    def module(module_name: str) -> CompactModuleClassProjection | None:
        found = InstalledModule.family_items(
            module_name, CompactModuleClassProjectionFamily
        )
        if found is None:
            return None
        (projection,) = found
        return projection

    @classmethod
    def declaration(
        cls,
        qualified_name: str,
        scan: CompactRepositoryPublicExposureIndex,
    ) -> CompactIndexedClass | None:
        """The class a qualified name reaches, following re-exports.

        A module the scan holds is read from the scan, never from an installed
        copy: the installed bytes of a scanned package can be another checkout
        of it. A module the scan holds more than once is not known. Only a
        module outside the scan is read from its installed source.
        """

        modules = scan.module_projection_multiplicity
        for _hop in range(cls.MAX_REEXPORTS):
            module_name, _dot, name = qualified_name.rpartition(".")
            if module_name in modules.ambiguous_handles:
                return None
            projection = modules.unambiguous_declarations_by_handle.get(module_name) or (
                cls.module(module_name) if module_name else None
            )
            if projection is None:
                return None
            for declared in projection.classes:
                if declared.symbol == qualified_name:
                    return declared
            target = dict(projection.import_aliases).get(name)
            if target is None or target == qualified_name:
                return None
            qualified_name = target
        return None


@dataclass(frozen=True)
class RegisteringMetaclassAuthority(Generic[ClassDeclarationT]):
    """Whether a class's metaclass registers the classes it creates, from what it does.

    The `metaclass=` binding is resolved like a base: to a scanned class, or
    else to an installed module's class read from source, following
    re-exports. The metaclass registers when it, or an ancestor reached the
    same way, stores the class it creates into a mapping
    (`ClassRegistryFacts.stores_created_class`). A native root stores none,
    and an ancestor that cannot be read is not known to register.

    A registry family is a class whose subclasses are stored in a mapping as
    they are created: a registering metaclass creates it
    (`registers`, `inherits_registering`), or it or an ancestor defines an
    `__init_subclass__` that stores the subclass
    (`ClassRegistryFacts.stores_subclass`, `stores_subclasses`).
    """

    scanned: Mapping[str, ClassDeclarationT]
    exposure: CompactRepositoryPublicExposureIndex
    registering: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    """`registers`, remembered per class symbol while ancestors are walked."""
    MAX_ANCESTORS: ClassVar[int] = 64

    @cached_property
    def suffix_index(self) -> _UniqueKnownSymbolSuffixIndex:
        return _unique_known_symbol_by_suffix(frozenset(self.scanned))

    def registers(self, declaration: ClassDeclaration) -> bool:
        pending = [
            (reference, declaration.module_name)
            for reference in declaration.registry_facts.metaclass_references
        ]
        seen: set[str] = set()
        for _step in range(self.MAX_ANCESTORS):
            if not pending:
                break
            reference, module_name = pending.pop()
            found = self.declaration_for(reference, module_name)
            if found is None or found.symbol in seen:
                continue
            seen.add(found.symbol)
            if found.registry_facts.stores_created_class:
                return True
            pending.extend(
                (base, found.module_name)
                for base in found.registry_facts.base_references
            )
        return False

    def is_registry_family(self, declaration: ClassDeclaration) -> bool:
        """Whether the subclasses created under it are stored in a mapping as they are created."""

        return (
            self.registers_remembered(declaration)
            or self.inherits_registering(declaration)
            or self.stores_subclasses(declaration)
        )

    def registry_family_key_type_name(
        self, declaration: CompactIndexedClass
    ) -> str | None:
        """The key type `Family[Key]` names, for the first subscripted base that is a registry family."""

        return next(
            (
                key_type_name
                for reference, key_type_name in zip(
                    declaration.base_references,
                    declaration.base_key_type_names,
                    strict=True,
                )
                if key_type_name is not None
                and (found := self.declaration_for(reference, declaration.module_name))
                is not None
                and self.is_registry_family(found)
            ),
            None,
        )

    def inherits_registering(self, declaration: ClassDeclaration) -> bool:
        """Whether a class it inherits from is created by a registering metaclass.

        Python creates a subclass with its bases' metaclass, so a class whose
        ancestor's metaclass registers is registered too, whatever it declares
        itself.
        """

        return self.some_ancestor(declaration, self.registers_remembered)

    def stores_subclasses(self, declaration: ClassDeclaration) -> bool:
        """Whether its own `__init_subclass__`, or one it inherits, stores each subclass."""

        return declaration.registry_facts.stores_subclass or self.some_ancestor(
            declaration, lambda found: found.registry_facts.stores_subclass
        )

    def registers_remembered(self, declaration: ClassDeclaration) -> bool:
        if declaration.symbol not in self.registering:
            self.registering[declaration.symbol] = self.registers(declaration)
        return self.registering[declaration.symbol]

    def some_ancestor(
        self,
        declaration: ClassDeclaration,
        holds: Callable[[ClassDeclaration], bool],
    ) -> bool:
        """Whether `holds` is true of an ancestor.

        Bases resolve as `declaration_for` resolves them; an ancestor that
        cannot be read is not known to hold. An ancestry left unread past
        `MAX_ANCESTORS` is not known not to hold, so it counts as holding:
        the answer that refuses a move.
        """

        pending = [
            (base, declaration.module_name)
            for base in declaration.registry_facts.base_references
        ]
        seen: set[str] = set()
        for _step in range(self.MAX_ANCESTORS):
            if not pending:
                break
            reference, module_name = pending.pop()
            found = self.declaration_for(reference, module_name)
            if found is None or found.symbol in seen:
                continue
            seen.add(found.symbol)
            if holds(found):
                return True
            pending.extend(
                (base, found.module_name)
                for base in found.registry_facts.base_references
            )
        return bool(pending)

    def declaration_for(
        self, reference: CompactNominalReference, module_name: str
    ) -> ClassDeclaration | None:
        symbol = CompactClassFamilyIndexBuilder._resolved_bound_symbol(
            reference, module_name, self.scanned, self.suffix_index
        )
        if symbol is not None:
            return self.scanned[symbol]
        qualified_name = reference.settled_qualified_name(self, module_name)
        if qualified_name is None:
            return None
        found = InstalledClassDeclarations.declaration(qualified_name, self.exposure)
        return None if found is None else self.scanned.get(found.symbol, found)

    def star_settled_name(
        self, reference: CompactNominalReference, module_name: str
    ) -> str | None:
        """A reference whose root a star import may rebind, settled by the repository's exports."""

        root, *rest = reference.source_parts
        projection = self.exposure.projections_by_module_name.get(module_name)
        settlement = self.exposure.star_settlement(
            module_name,
            root,
            None if projection is None else dict(projection.import_aliases).get(root),
        )
        # settled_qualified_name's str | None protocol.
        return next(
            (
                ".".join((binding, *rest))
                for _, binding in StarSettlements.keyed(((root, settlement),)).settled
            ),
            None,
        )

    def resolved(self) -> dict[str, ClassDeclarationT]:
        return {
            symbol: declaration.with_registering_metaclass(self)
            for symbol, declaration in self.scanned.items()
        }


@dataclass(frozen=True)
class CompactClassFamilyIndexBuilder:
    projections: Sequence[CompactModuleClassHeader]

    def build(self) -> CompactClassFamilyIndex:
        records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                (
                    record
                    for projection in repository_settled_projections(tuple(self.projections))
                    for record in projection.classes
                ),
                lambda record: record.symbol,
            ).values()
        )
        known_symbols = frozenset(record.symbol for record in records)
        unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
        classes_by_symbol = {
            record.symbol: record.with_resolved_base_symbols(
                tuple(
                    resolved
                    for reference in record.base_references
                    if (
                        resolved := self._resolved_bound_symbol(
                            reference,
                            record.module_name,
                            known_symbols,
                            unique_symbols_by_suffix,
                        )
                    )
                    is not None
                )
            )
            for record in records
        }
        return CompactClassFamilyIndex(
            classes_by_symbol=RegisteringMetaclassAuthority(
                classes_by_symbol,
                CompactRepositoryPublicExposureIndex(tuple(self.projections)),
            ).resolved()
        )

    @staticmethod
    def _resolved_bound_symbol(
        reference: CompactNominalReference,
        module_name: str,
        known_symbols: frozenset[str] | dict[str, CompactIndexedClass],
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex,
    ) -> str | None:
        candidate = ".".join(reference.resolved_parts)
        if candidate in known_symbols:
            return candidate
        return reference.known_symbol_beyond(
            candidate, module_name, known_symbols, unique_symbols_by_suffix
        )

    @staticmethod
    def _resolved_symbol(
        parts: tuple[str, ...],
        module_name: str,
        import_aliases_by_module_name: dict[str, dict[str, str]],
        known_symbols: frozenset[str] | dict[str, CompactIndexedClass],
        unique_symbols_by_name: dict[str, str],
        allow_unique_unqualified: bool = True,
        unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex | None = None,
    ) -> str | None:
        import_aliases = import_aliases_by_module_name.get(module_name, {})
        first, *rest = parts
        alias_target = import_aliases.get(first)
        if alias_target is not None:
            candidate = ".".join((alias_target, *rest)) if rest else alias_target
            if candidate in known_symbols:
                return candidate
            unique_by_suffix = unique_symbols_by_suffix
            if unique_by_suffix is None:
                unique_by_suffix = _unique_known_symbol_by_suffix(
                    frozenset(known_symbols)
                )
            match = unique_by_suffix.root_relative_match(candidate)
            if match is not None:
                return match
        qualified = ".".join(parts)
        if qualified in known_symbols:
            return qualified
        module_local = ".".join((module_name, *parts))
        if module_local in known_symbols:
            return module_local
        if allow_unique_unqualified and len(parts) == 1:
            return unique_symbols_by_name.get(parts[0])
        return None


def build_compact_class_family_index(projections: Sequence[CompactModuleClassHeader]) -> CompactClassFamilyIndex:
    """The exact inheritance graph of AST-free per-module facts, built once per
    projection sequence that holds its views."""

    return SequenceViews.of(projections).view(_built_class_family_index)


def _built_class_family_index(
    projections: Sequence[CompactModuleClassHeader],
) -> CompactClassFamilyIndex:
    return CompactClassFamilyIndexBuilder(projections).build()


@dataclass(frozen=True)
class CompactClassReferenceResolver:
    import_aliases_by_module_name: dict[str, dict[str, str]]
    known_symbols: dict[str, CompactIndexedClass]
    unique_symbols_by_name: dict[str, str]
    unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex

    @classmethod
    def from_index(cls, projections: Sequence[CompactModuleClassHeader], class_index: CompactClassFamilyIndex) -> CompactClassReferenceResolver:
        return cls(
            import_aliases_by_module_name={
                projection.module_name: dict(projection.import_aliases)
                for projection in projections
            },
            known_symbols=class_index.classes_by_symbol,
            unique_symbols_by_name={
                name: symbols[0]
                for name, symbols in class_index.symbols_by_simple_name.items()
                if len(symbols) == 1
            },
            unique_symbols_by_suffix=(
                _UniqueKnownSymbolSuffixIndex.from_terminal_buckets(
                    class_index.symbols_by_simple_name
                )
            ),
        )

    def symbol_for(
        self,
        *,
        module_name: str,
        reference_parts: tuple[str, ...],
        allow_unique_unqualified: bool = True,
    ) -> str | None:
        return CompactClassFamilyIndexBuilder._resolved_symbol(
            reference_parts,
            module_name,
            self.import_aliases_by_module_name,
            self.known_symbols,
            self.unique_symbols_by_name,
            allow_unique_unqualified,
            self.unique_symbols_by_suffix,
        )


@dataclass(frozen=True)
class CompactClassMappingKeyAuthority(MappingKeyResolverABC[CompactIndexedClass]):
    """Resolve class-member keys through the referenced declaration authority."""

    projections: Sequence[CompactModuleClassProjection]
    class_index: CompactClassFamilyIndex

    @cached_property
    def class_reference_resolver(self) -> CompactClassReferenceResolver:
        return CompactClassReferenceResolver.from_index(
            self.projections,
            self.class_index,
        )

    def _lexical_mapping_key(
        self,
        reference: LexicalValueReference,
        context: CompactIndexedClass,
    ) -> Hashable:
        if len(reference.attribute_path) != 1:
            raise ValueError("Mapping key reference requires one declaration member")
        symbol = self.class_reference_resolver.symbol_for(
            module_name=context.module_name,
            reference_parts=(reference.root_name,),
            allow_unique_unqualified=False,
        )
        declaration = None if symbol is None else self.class_index.class_for(symbol)
        if declaration is None:
            raise ValueError("Mapping key declaration semantics remain unproved")
        return declaration.enum_declaration.require_mapping_key(
            reference.attribute_path[0]
        )


def _single_assignment_target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
    else:
        target = node.target
    if isinstance(target, ast.Name):
        return target.id
    return None
