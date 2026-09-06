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
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import MISSING, dataclass, field, fields, replace
from enum import StrEnum
from functools import cached_property, lru_cache
from heapq import merge
from typing import (
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

from .annotation_semantics import (
    CLASSVAR_ANNOTATION_AUTHORITY,
    NOMINAL_ANNOTATION_SOURCE_AUTHORITY,
)
from .ast_tools import (
    AstExpressionProjection,
    CompactModuleIdentity,
    CollectedFamily,
    ParsedModule,
    PythonSourcePathPolicy,
    SourceModule,
    _walk_nodes,
    collect_family_items,
    module_syntax_index,
    named_function_nodes,
)
from .class_namespace import ClassNamespaceExecutionEvidence, NATIVE_METHOD_DECORATORS
from .collection_algebra import (
    IdentityHandleMultiplicityProjection,
    UniqueIdentityIndexAuthority,
    sorted_tuple,
)
from .class_mro import ClassMroAuthority
from .declaration_dependencies import ClassScopeDependency
from .descriptor_algebra import AliasProperty
from .enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from .export_tools import PYTHON_PUBLIC_EXPORT_ASSIGNMENT
from .lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from .native_declarations import (
    ClassNamespaceDeclaration,
    NativeDeclaration,
    QualifiedDeclaration,
)
from .native_syntax import NativePythonSyntaxIndex
from .semantic_algebra import DirectedGraph
from .source_geometry import ClassHeaderSourceSpan as ClassHeaderSourceSpan
from .source_identity import resolved_source_path_text


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
    dataclass_declaration: CompactDataclassDeclaration | None = field(
        default=None,
        kw_only=True,
    )
    class_decorators_are_promotion_safe: bool = field(default=True, kw_only=True)

    qualified_name = AliasProperty[str]("symbol")

    def with_resolved_base_symbols(
        self,
        resolved_base_symbols: tuple[str, ...],
    ) -> Self:
        return replace(self, resolved_base_symbols=resolved_base_symbols)


@dataclass(frozen=True)
class IndexedClass(ClassDeclaration, ClassNamespaceDeclaration):
    node: ast.ClassDef

    member_binding_names = AliasProperty[frozenset[str]](
        "namespace_execution.binding_names"
    )

    @cached_property
    def namespace_execution(self) -> ClassNamespaceExecutionEvidence:
        return ClassNamespaceExecutionEvidence.from_class(self.node)

    @property
    def is_final(self) -> bool:
        return any(
            (isinstance(decorator, ast.Name) and decorator.id == "final")
            or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
            for decorator in self.node.decorator_list
        )

    @property
    def declares_autoregister_meta(self) -> bool:
        """Whether this declaration owns an AutoRegister-backed family."""

        return _declares_autoregister_meta(self.node)

    @classmethod
    def from_parsed_class(
        cls,
        parsed_module: ParsedModule,
        qualname: str,
        node: ast.ClassDef,
        module_binding_snapshot: ModuleNominalBindingSnapshot,
    ) -> "IndexedClass":
        return cls(
            symbol=f"{parsed_module.module_name}.{qualname}",
            module_name=parsed_module.module_name,
            qualname=qualname,
            simple_name=qualname.rsplit(".", 1)[-1],
            file_path=parsed_module.file_path,
            line=node.lineno,
            node=node,
            declared_base_names=tuple(
                declared_base_name
                for base in node.bases
                if (
                    declared_base_name
                    := ClassSymbolResolutionAuthority.declared_base_name(base)
                )
                is not None
            ),
            resolved_base_symbols=(),
            dataclass_declaration=_dataclass_declaration(
                module_binding_snapshot,
                qualname,
                node,
            ),
            class_decorators_are_promotion_safe=all(
                ClassMethodPromotionSafeDecorator.for_qualified_name(
                    _class_scope_qualified_import_name(
                        module_binding_snapshot,
                        {},
                        decorator.func
                        if isinstance(decorator, ast.Call)
                        else decorator,
                        frozenset(),
                    )
                )
                is not None
                for decorator in node.decorator_list
            ),
        )


@dataclass(frozen=True)
class CompactClassValueConstruction:
    """One class-owned construction of a nominal value declaration."""

    assigned_name: str
    constructor_name: str
    keyword_names: tuple[str, ...]
    line: int


class CompactNominalReference(NamedTuple):
    """One source reference together with its declaration-time root binding."""

    source_parts: tuple[str, ...]
    root_binding: "CompactNominalBinding | None"

    @property
    def resolved_parts(self) -> tuple[str, ...]:
        if self.root_binding is None:
            return self.source_parts
        return (
            *self.root_binding.qualified_name.split("."),
            *self.source_parts[1:],
        )

    @property
    def qualified_name(self) -> str:
        return ".".join(self.resolved_parts)

    @property
    def simple_name(self) -> str:
        return self.resolved_parts[-1]

    @property
    def permits_root_relative_resolution(self) -> bool:
        return (
            self.root_binding is not None
            and self.root_binding.kind.projects_as_import_alias
        )


class CompactClassMemberDeclaration(NamedTuple):
    """One direct class binding from which all member projections descend."""

    name: str
    line: int
    expression: str | None
    constant_string: str | None
    value_is_none_literal: bool
    constructor_name: str | None
    constructor_keyword_names: tuple[str, ...]
    annotation_expression: str | None

    @property
    def annotation_reference_parts(self) -> tuple[str, ...] | None:
        """Project the one nominal type named by this member annotation."""

        if self.annotation_expression is None:
            return None
        try:
            annotation = ast.parse(self.annotation_expression, mode="eval").body
        except SyntaxError:
            return None
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return None
        if isinstance(annotation, ast.Subscript) and (
            CLASSVAR_ANNOTATION_AUTHORITY.matches(annotation)
        ):
            annotation = annotation.slice
        return NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_parts_or_none(annotation)


class CompactProductAuthorityViolation(StrEnum):
    """One typed reason a dataclass declaration cannot prove a plain product."""

    UNRESOLVED_DATACLASS_DECORATOR = (
        "unresolved_dataclass_decorator",
        "the dataclass decorator does not resolve to the standard-library declaration",
        (),
    )
    MULTIPLE_DATACLASS_DECORATORS = (
        "multiple_dataclass_decorators",
        "more than one dataclass decorator applies to the same declaration",
        (),
    )
    GENERATED_INIT_DISABLED = (
        "generated_init_disabled",
        "the dataclass declaration does not retain its generated initializer",
        (),
    )
    DYNAMIC_DATACLASS_OPTIONS = (
        "dynamic_dataclass_options",
        "the dataclass constructor options cannot be resolved statically",
        (),
    )
    CUSTOM_CLASS_DECORATOR = (
        "custom_class_decorator",
        "another class decorator can replace or mutate the dataclass declaration",
        (),
    )
    CUSTOM_CLASS_CREATION = (
        "custom_class_creation",
        "class keywords or a custom metaclass can change construction semantics",
        (),
    )
    CUSTOM_PRODUCT_LIFECYCLE = (
        "custom_product_lifecycle",
        "a class-owned lifecycle hook can change construction or field projection",
        (
            "__delattr__",
            "__getattr__",
            "__getattribute__",
            "__init__",
            "__init_subclass__",
            "__new__",
            "__post_init__",
            "__setattr__",
            "__slots__",
        ),
    )
    FIELD_MEMBER_COLLISION = (
        "field_member_collision",
        "a dataclass field name is rebound by another class member",
        (),
    )
    UNRESOLVED_FIELD_ROLE = (
        "unresolved_field_role",
        "a field role or default cannot be proven to preserve plain stored projection",
        (),
    )
    INIT_ONLY_FIELD = (
        "init_only_field",
        "an InitVar constructor parameter is not a stored product field",
        (),
    )
    NON_INIT_FIELD = (
        "non_init_field",
        "a stored field is excluded from the generated initializer",
        (),
    )
    DYNAMIC_FIELD_SCHEMA = (
        "dynamic_field_schema",
        "class execution can add or replace annotations outside the direct field declarations",
        (),
    )
    NESTED_CLASS_SCOPE = (
        "nested_class_scope",
        "the enclosing class namespace prevents exact module-binding recovery",
        (),
    )
    INCOMPLETE_BASE_RESOLUTION = (
        "incomplete_base_resolution",
        "not every direct base resolves to one repository declaration",
        (),
    )
    MULTIPLE_PRODUCT_BASES = (
        "multiple_product_bases",
        "the product schema requires an unproved multiple-inheritance linearization",
        (),
    )
    NON_DATACLASS_BASE = (
        "non_dataclass_base",
        "a direct base lacks a closed dataclass product declaration",
        (),
    )
    CYCLIC_PRODUCT_LINEAGE = (
        "cyclic_product_lineage",
        "the product declaration participates in a cyclic inheritance graph",
        (),
    )

    explanation: str
    implicated_member_names: tuple[str, ...]

    def __new__(
        cls,
        value: str,
        explanation: str,
        implicated_member_names: tuple[str, ...],
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.explanation = explanation
        member.implicated_member_names = implicated_member_names
        return member

    def is_violated_by_member_names(self, member_names: frozenset[str]) -> bool:
        return bool(member_names.intersection(self.implicated_member_names))


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

class DataclassRuntimeDeclaration(StrEnum):
    """Standard-library dataclass declarations with nominal qualified identity."""

    DATACLASS = ("dataclass", True, False)
    FIELD = ("field", False, True)

    def __new__(
        cls,
        value: str,
        is_dataclass_decorator: bool,
        is_field_factory: bool,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._is_dataclass_decorator = is_dataclass_decorator
        member._is_field_factory = is_field_factory
        return member

    @property
    def qualified_name(self) -> str:
        return f"dataclasses.{self.value}"

    @property
    def is_dataclass_decorator(self) -> bool:
        return self._is_dataclass_decorator

    @property
    def is_field_factory(self) -> bool:
        return self._is_field_factory

    def matches(self, qualified_name: str | None) -> bool:
        return qualified_name == self.qualified_name

    def matches_reference_name(self, reference_name: str | None) -> bool:
        return reference_name in (self.value, self.qualified_name)

    @classmethod
    def for_qualified_name(cls, qualified_name: str | None) -> Self | None:
        return next(
            (member for member in cls if member.matches(qualified_name)),
            None,
        )

    @classmethod
    def for_reference_name(cls, reference_name: str | None) -> Self | None:
        return next(
            (member for member in cls if member.matches_reference_name(reference_name)),
            None,
        )

    @classmethod
    def dataclass_decorator_for_name(cls, reference_name: str | None) -> Self | None:
        """Resolve a standard dataclass decorator from either source spelling."""

        return next(
            (
                member
                for member in cls
                if member.is_dataclass_decorator
                and member.matches_reference_name(reference_name)
            ),
            None,
        )


class CompactDataclassFieldDeclaration(NamedTuple):
    """One direct annotated member and its exact dataclass role."""

    name: str
    line: int
    role: CompactDataclassFieldRole


class CompactProductDeclarationFailure(NamedTuple):
    """One class-local product obligation retained at its exact source line."""

    line: int
    violation: CompactProductAuthorityViolation


class CompactDataclassDeclaration(NamedTuple):
    """One dataclass-like declaration, including fail-closed unknown semantics."""

    runtime_declaration: DataclassRuntimeDeclaration | None
    fields: tuple[CompactDataclassFieldDeclaration, ...]
    failures: tuple[CompactProductDeclarationFailure, ...] = ()

    @property
    def is_standard_dataclass(self) -> bool:
        return (
            self.runtime_declaration is not None
            and self.runtime_declaration.is_dataclass_decorator
        )


@dataclass(frozen=True)
class CompactProductField:
    """One effective dataclass field retaining its nominal declaration source."""

    name: str
    role: CompactDataclassFieldRole
    declaring_class_symbol: str
    file_path: str
    line: int


@dataclass(frozen=True)
class CompactProductAuthority:
    """One linearly composed dataclass schema with source-visible fields."""

    class_symbol: str
    effective_fields: tuple[CompactProductField, ...]
    file_path: str
    line: int

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


@dataclass(frozen=True)
class CompactProductAuthorityFailure:
    """One failed product obligation attached to its declaration source."""

    class_symbol: str
    file_path: str
    line: int
    violation: CompactProductAuthorityViolation


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
class CompactClassHeader(ClassDeclaration):
    """Class-index surface sufficient for inheritance reconstruction."""

    base_references: tuple[CompactNominalReference, ...]
    direct_base_count: int = 0
    base_references_are_complete: bool = False
    product_base_bindings_are_exact: bool = False
    is_final: bool = False
    mro_bases_are_static: bool = False

    @property
    def base_resolution_is_complete(self) -> bool:
        """Return whether every domain-bearing base resolves in the compact graph."""

        return self.base_references_are_complete and len(
            self.resolved_base_symbols
        ) == declared_nominal_base_count(self)


@dataclass(frozen=True)
class CompactIndexedClass(CompactClassHeader):
    """AST-free class declaration used to reconstruct inheritance globally."""

    direct_member_declarations: tuple[CompactClassMemberDeclaration, ...] = ()
    metaclass_names: tuple[str, ...] = ()
    class_keyword_names: tuple[str, ...] = ()
    class_header_is_reconstructible: bool = True
    keyed_family_key_type_name: str | None = None
    end_line: int | None = None
    method_names: tuple[str, ...] = ()
    abstract_method_names: tuple[str, ...] = ()
    is_abstract: bool = False
    declares_autoregister_meta: bool = False
    is_registration_authority: bool = False
    autoregister_registry_key_attr_name: str | None = None
    autoregister_registry_projection_names: tuple[str, ...] = ()
    keyed_registry_lookup_method_names: tuple[str, ...] = ()
    keyed_registry_reverse_lookup_method_names: tuple[str, ...] = ()
    predicate_selected_methods: tuple[tuple[int, str, str, str], ...] = ()

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
            for name, declaration in self.direct_members_by_name.items()
        }

    @cached_property
    def direct_members_by_name(self) -> dict[str, CompactClassMemberDeclaration]:
        return {
            declaration.name: declaration
            for declaration in self.direct_member_declarations
        }

    @property
    def direct_assignment_expressions(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(self.assignments_by_name.items())

    @property
    def direct_assignment_lines(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (declaration.name, declaration.line)
            for declaration in self.direct_member_declarations
        )

    @property
    def assignment_lines_by_name(self) -> dict[str, int]:
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
            for declaration in self.direct_members_by_name.values()
            if declaration.constructor_name is not None
        )

    @property
    def direct_constant_string_assignments(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (declaration.name, declaration.constant_string)
            for declaration in self.direct_members_by_name.values()
            if declaration.constant_string is not None
        )

    @property
    def direct_non_none_assignment_names(self) -> tuple[str, ...]:
        return sorted_tuple(
            declaration.name
            for declaration in self.direct_members_by_name.values()
            if not declaration.value_is_none_literal
        )


def has_complete_concrete_mro_composite(
    direct_child_symbols: tuple[str, ...],
    concrete_descendants: tuple[IndexedClass, ...] | tuple[CompactIndexedClass, ...],
) -> bool:
    """Return whether one descendant composes every concrete root branch."""

    concrete_descendant_symbols = {
        descendant.symbol for descendant in concrete_descendants
    }
    concrete_branch_symbols = concrete_descendant_symbols.intersection(
        direct_child_symbols
    )
    if len(concrete_branch_symbols) < 2:
        return False
    return any(
        concrete_branch_symbols.issubset(descendant.resolved_base_symbols)
        for descendant in concrete_descendants
    )


class CompactPublicNameExposure(StrEnum):
    """Proof result for one name on a module's declared public surface."""

    PUBLIC = "public", True
    PRIVATE = "private", False
    UNRESOLVED = "unresolved", True

    def __new__(cls, value: str, blocks_closed_boundary: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member._blocks_closed_boundary = blocks_closed_boundary
        return member

    @property
    def blocks_closed_boundary(self) -> bool:
        return self._blocks_closed_boundary

    @property
    def proves_public_exposure(self) -> bool:
        return self is type(self).PUBLIC

    @property
    def introduces_uncertainty(self) -> bool:
        return self is type(self).UNRESOLVED


class CompactModulePublicExportContract(ABC):
    """Representation-independent declaration of one module's export policy."""

    @abstractmethod
    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        raise NotImplementedError

    @abstractmethod
    def allows_binding_relocation(self, name: str) -> bool:
        """Return whether removing this module binding preserves the contract."""

        raise NotImplementedError


@dataclass(frozen=True)
class CompactImplicitPublicExportContract(CompactModulePublicExportContract):
    """Python's implicit convention: leading-underscore bindings are private."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PRIVATE
            if name.startswith("_")
            else CompactPublicNameExposure.PUBLIC
        )

    def allows_binding_relocation(self, name: str) -> bool:
        del name
        return True


@dataclass(frozen=True)
class CompactExplicitPublicExportContract(CompactModulePublicExportContract):
    """One statically complete ``__all__`` membership declaration."""

    exported_names: tuple[str, ...]

    @classmethod
    def from_expression(
        cls,
        value: ast.expr,
        preceding_bound_names: frozenset[str],
    ) -> "CompactExplicitPublicExportContract | None":
        literal_names = _literal_public_export_names(value)
        if literal_names is not None:
            return cls(literal_names)
        if {"tuple", "globals"}.intersection(preceding_bound_names) or not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "tuple"
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.GeneratorExp)
        ):
            return None
        generator = value.args[0]
        if not (
            isinstance(generator.elt, ast.Name)
            and len(generator.generators) == 1
        ):
            return None
        comprehension = generator.generators[0]
        if not (
            isinstance(comprehension.target, ast.Name)
            and generator.elt.id == comprehension.target.id
            and isinstance(comprehension.iter, ast.Call)
            and isinstance(comprehension.iter.func, ast.Name)
            and comprehension.iter.func.id == "globals"
            and not comprehension.iter.args
            and not comprehension.iter.keywords
            and len(comprehension.ifs) == 1
            and not comprehension.is_async
        ):
            return None
        condition = comprehension.ifs[0]
        if not (
            isinstance(condition, ast.UnaryOp)
            and isinstance(condition.op, ast.Not)
            and isinstance(condition.operand, ast.Call)
        ):
            return None
        predicate = condition.operand
        if not (
            isinstance(predicate.func, ast.Attribute)
            and isinstance(predicate.func.value, ast.Name)
            and predicate.func.value.id == comprehension.target.id
            and predicate.func.attr == "startswith"
            and len(predicate.args) == 1
            and isinstance(predicate.args[0], ast.Constant)
            and predicate.args[0].value == "__"
            and not predicate.keywords
        ):
            return None
        return cls(
            sorted_tuple(
                name for name in preceding_bound_names if not name.startswith("__")
            )
        )

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        return (
            CompactPublicNameExposure.PUBLIC
            if name in self.exported_names
            else CompactPublicNameExposure.PRIVATE
        )

    def allows_binding_relocation(self, name: str) -> bool:
        return name not in self.exported_names


@dataclass(frozen=True)
class CompactUnresolvedPublicExportContract(CompactModulePublicExportContract):
    """A dynamic or otherwise incomplete ``__all__`` declaration."""

    def exposure_for(self, name: str) -> CompactPublicNameExposure:
        del name
        return CompactPublicNameExposure.UNRESOLVED

    def allows_binding_relocation(self, name: str) -> bool:
        del name
        return False


@dataclass(frozen=True)
class CompactModuleStarImportOrigin:
    """One module-scope star import and its optional resolved module origin."""

    module_name: str | None

    @property
    def is_resolved(self) -> bool:
        return self.module_name is not None


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
            origin.module_name is not None
            and not self.exposure_for(
                origin.module_name,
                binding_name,
            ).blocks_closed_boundary
            for origin in self.star_import_origins_for(module_name)
        )


@dataclass(frozen=True)
class CompactModuleClassHeader(CompactModuleIdentity):
    """Module namespace and class surface required by the compact family index."""

    import_aliases: tuple[tuple[str, str], ...]
    public_export_contract: CompactModulePublicExportContract
    star_import_origins: tuple[CompactModuleStarImportOrigin, ...]
    classes: tuple[CompactIndexedClass, ...]


@dataclass(frozen=True)
class CompactRepositoryPublicExposureIndex(RepositoryPublicExposureAuthority):
    """Derive transitive public exposure from module-owned namespace contracts."""

    module_projections: tuple[CompactModuleClassHeader, ...]

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
                not origin.is_resolved
                and consumer.public_export_contract.exposure_for(
                    current_name
                ).blocks_closed_boundary
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
class CompactClassSyntaxFacets:
    """Class-family views derived together from the shared module traversal."""

    closed_axis_branch_functions: tuple["CompactClosedAxisBranchFunction", ...] = ()
    exact_type_guards: tuple["CompactExactTypeGuard", ...] = ()
    autoregister_reference_index: "CompactAutoRegisterReferenceIndex | None" = None


@dataclass(frozen=True)
class CompactModuleClassProjection(
    CompactClassSyntaxFacets,
    CompactModuleClassHeader,
):
    """One module's class declarations and import aliases, without its AST."""

    sorted_key_calls: tuple["CompactSortedKeyCall", ...] = ()
    keyed_table_axes: tuple["CompactKeyedTableAxis", ...] = ()
    manual_selector_axes: tuple["CompactManualSelectorAxis", ...] = ()
    top_level_definitions: tuple[tuple[str, int], ...] = ()
    repeated_keyed_family_roots: tuple["CompactRepeatedKeyedFamilyRoot", ...] = ()
    manual_subclass_roster_roots: tuple["CompactManualSubclassRosterRoot", ...] = ()
    latent_rosters: tuple["LatentRosterObservation", ...] = ()
    named_projection_surfaces: tuple["CompactNamedProjectionSurface", ...] = ()
    class_methods: tuple["CompactClassMethod", ...] = ()

    def header_core(self) -> "CompactModuleClassProjection":
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
    ) -> dict[str, object]:
        preserved_names = frozenset(
            dataclass_field.name for dataclass_field in fields(preserved_type)
        )
        values: dict[str, object] = {}
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


class MethodFamilyResidueRole(StrEnum):
    """Declaration role required to absorb one varying method coordinate."""

    CLASS_VARIABLE = "class_variable"
    PROPERTY_HOOK = "property_hook"
    BEHAVIOR_HOOK = "behavior_hook"


MethodFamilyResidueNameBuilder: TypeAlias = Callable[[str, int], str]


class CompactMethodSemanticCoordinateKind(StrEnum):
    """Closed coordinate grammar with declaration-owned residue semantics."""

    CONSTANT = (
        "constant",
        MethodFamilyResidueRole.CLASS_VARIABLE,
        lambda method_name, index: f"{method_name}_constant_{index}".upper(),
    )
    NAME = (
        "name",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_value_{index}",
    )
    SELF_ATTRIBUTE = (
        "self_attr",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_property_{index}",
    )
    ATTRIBUTE = (
        "attribute",
        MethodFamilyResidueRole.PROPERTY_HOOK,
        lambda method_name, index: f"{method_name}_value_{index}",
    )
    CALL = (
        "call",
        MethodFamilyResidueRole.BEHAVIOR_HOOK,
        lambda method_name, index: f"_{method_name}_operation_{index}",
    )

    residue_role: MethodFamilyResidueRole
    _residue_name_builder: MethodFamilyResidueNameBuilder

    def __new__(
        cls,
        value: str,
        residue_role: MethodFamilyResidueRole,
        residue_name_builder: MethodFamilyResidueNameBuilder,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.residue_role = residue_role
        member._residue_name_builder = residue_name_builder
        return member

    def residue_name(self, method_name: str, index: int) -> str:
        return self._residue_name_builder(method_name, index)


@dataclass(frozen=True)
class CompactMethodSemanticCoordinate:
    """One typed varying value in a normalized method body."""

    path: tuple[str, ...]
    kind: CompactMethodSemanticCoordinateKind
    value: str


@dataclass(frozen=True)
class CompactClassMethodSemanticProfile:
    """Semantic body profile derived lazily from one compact class method."""

    skeleton: tuple[str, ...]
    coordinates: tuple[CompactMethodSemanticCoordinate, ...]


@dataclass(frozen=True)
class MethodPromotionInspection:
    """Current method syntax and lexical bindings relevant to promotion safety."""

    method: ast.FunctionDef | ast.AsyncFunctionDef
    module_bound_names: frozenset[str]
    class_bound_names: frozenset[str]
    source_lines: tuple[str, ...]


@dataclass(frozen=True)
class ClassMethodReceiverRequirements:
    """Receiver members required directly or through local aliases."""

    member_names: frozenset[str]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "ClassMethodReceiverRequirements":
        receiver_name = cls.receiver_name(method)
        if receiver_name is None:
            return cls(frozenset())
        aliases = cls.receiver_aliases(method, receiver_name)
        return cls(
            frozenset(
                node.attr
                for node in ast.walk(method)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            )
        )

    @staticmethod
    def receiver_aliases(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        receiver_name: str,
    ) -> frozenset[str]:
        alias_targets_by_source: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                alias_targets_by_source[node.value.id].update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif (
                isinstance(node, ast.AnnAssign | ast.NamedExpr)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                alias_targets_by_source[node.value.id].add(node.target.id)

        aliases = {receiver_name}
        pending = deque((receiver_name,))
        while pending:
            source_name = pending.popleft()
            for target_name in alias_targets_by_source[source_name]:
                if target_name in aliases:
                    continue
                aliases.add(target_name)
                pending.append(target_name)
        return frozenset(aliases)

    @staticmethod
    def receiver_name(
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str | None:
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in method.decorator_list
        ):
            return None
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        return positional_parameters[0].arg if positional_parameters else None


MethodPromotionHazardPredicate: TypeAlias = Callable[[MethodPromotionInspection], bool]


_PROMOTABLE_METHOD_DECORATOR_NAMES = frozenset(
    declaration.__name__ for declaration in NATIVE_METHOD_DECORATORS
)


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


def _has_custom_method_decorator(inspection: MethodPromotionInspection) -> bool:
    shadowed_names = inspection.module_bound_names | inspection.class_bound_names
    return any(
        not isinstance(decorator, ast.Name)
        or decorator.id not in _PROMOTABLE_METHOD_DECORATOR_NAMES
        or decorator.id in shadowed_names
        for decorator in inspection.method.decorator_list
    )


def _is_direct_namespace_sensitive(inspection: MethodPromotionInspection) -> bool:
    name = inspection.method.name
    return name.startswith("__") and name.endswith("__")


def _has_evaluated_default(inspection: MethodPromotionInspection) -> bool:
    return bool(
        inspection.method.args.defaults
        or any(default is not None for default in inspection.method.args.kw_defaults)
    )


def _method_annotation_nodes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    arguments = (
        *method.args.posonlyargs,
        *method.args.args,
        *method.args.kwonlyargs,
        *((method.args.vararg,) if method.args.vararg is not None else ()),
        *((method.args.kwarg,) if method.args.kwarg is not None else ()),
    )
    return (
        *(
            argument.annotation
            for argument in arguments
            if argument.annotation is not None
        ),
        *((method.returns,) if method.returns is not None else ()),
    )


def _has_class_local_annotation_reference(
    inspection: MethodPromotionInspection,
) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in inspection.class_bound_names
        for annotation in _method_annotation_nodes(inspection.method)
        for node in ast.walk(annotation)
    )


def _has_attached_leading_comment(inspection: MethodPromotionInspection) -> bool:
    decorator_lines = tuple(
        decorator.lineno
        for decorator in inspection.method.decorator_list
        if decorator.lineno
    )
    start_line = min((*decorator_lines, inspection.method.lineno))
    if start_line <= 1 or start_line > len(inspection.source_lines):
        return False
    method_line = inspection.source_lines[start_line - 1]
    preceding_line = inspection.source_lines[start_line - 2]
    method_indent = method_line[: len(method_line) - len(method_line.lstrip())]
    preceding_indent = preceding_line[
        : len(preceding_line) - len(preceding_line.lstrip())
    ]
    return preceding_indent == method_indent and preceding_line.lstrip().startswith("#")


class MethodPromotionHazard(StrEnum):
    """One promotion hazard carrying its own syntax recognition behavior."""

    CUSTOM_METHOD_DECORATOR = (
        "custom_method_decorator",
        _has_custom_method_decorator,
    )
    DIRECT_NAMESPACE_SENSITIVE_METHOD = (
        "direct_namespace_sensitive_method",
        _is_direct_namespace_sensitive,
    )
    EVALUATED_DEFAULT = ("evaluated_default", _has_evaluated_default)
    CLASS_LOCAL_ANNOTATION_REFERENCE = (
        "class_local_annotation_reference",
        _has_class_local_annotation_reference,
    )
    ATTACHED_LEADING_COMMENT = (
        "attached_leading_comment",
        _has_attached_leading_comment,
    )

    predicate: MethodPromotionHazardPredicate

    def __new__(
        cls,
        value: str,
        predicate: MethodPromotionHazardPredicate,
    ) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.predicate = predicate
        return member

    def is_present(self, inspection: MethodPromotionInspection) -> bool:
        return self.predicate(inspection)


@dataclass(frozen=True)
class ClassMethodPromotionSafetyProfile:
    """Complete promotion hazards derived from one current method declaration."""

    hazards: tuple[ClassScopeDependency | MethodPromotionHazard, ...]

    @classmethod
    def from_method(
        cls,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        module_bound_names: frozenset[str],
        class_bound_names: frozenset[str] = frozenset(),
        *,
        source_lines: tuple[str, ...],
    ) -> "ClassMethodPromotionSafetyProfile":
        inspection = MethodPromotionInspection(
            method,
            module_bound_names,
            class_bound_names,
            source_lines,
        )
        return cls(
            ClassScopeDependency.from_node(method) + tuple(
                hazard
                for hazard in MethodPromotionHazard
                if hazard.is_present(inspection)
            )
        )


@dataclass(frozen=True)
class CompactClassMethod:
    """AST-free method fact shared by exact promotion and anti-unification."""

    class_symbol: str
    method_name: str
    line: int
    line_count: int
    body_statement_count: int
    statement_sources: tuple[str, ...]
    exact_source_digest: str | None
    promotion_hazards: tuple[ClassScopeDependency | MethodPromotionHazard, ...]
    receiver_member_names: frozenset[str]

    @property
    def statement_count(self) -> int:
        return self.body_statement_count

    @property
    def exact_promotion_source_digest(self) -> str | None:
        """Return the source identity only when exact promotion is safe."""

        return None if self.promotion_hazards else self.exact_source_digest

    @cached_property
    def semantic_profile(self) -> CompactClassMethodSemanticProfile:
        statements = _compact_class_method_statements_from_sources(
            self.statement_sources
        )
        coordinates = tuple(
            coordinate
            for statement_index, statement in enumerate(statements)
            for coordinate in _compact_class_method_coordinates(
                statement,
                ("body", statement_index),
            )
        )
        return CompactClassMethodSemanticProfile(
            skeleton=tuple(
                _compact_class_method_statement_skeleton(statement)
                for statement in statements
            ),
            coordinates=coordinates,
        )


@dataclass(frozen=True)
class CompactManualSubclassRegistrationSite:
    registry_name: str
    guard_summary: str | None
    selector_attr_name: str | None
    requires_concrete_subclass: bool


@dataclass(frozen=True)
class CompactManualSubclassRosterRoot:
    class_symbol: str
    init_subclass_line: int
    registration_sites: tuple[CompactManualSubclassRegistrationSite, ...]
    consumer_locations: tuple[tuple[str, int, str, str], ...]


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
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> tuple["LatentRosterObservation", ...]:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_leading_docstring(list(statement.body))
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and body[0].value is not None
            ):
                return cls.from_value(
                    parsed_module,
                    statement,
                    roster_name=statement.name,
                    value=body[0].value,
                )
            return ()
        target_value: tuple[ast.AST, ast.AST] | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target_value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target_value = statement.target, statement.value
        if target_value is None or not isinstance(target_value[0], ast.Name):
            return ()
        target, value = target_value
        roster_name = (
            f"{roster_prefix}.{target.id}" if roster_prefix is not None else target.id
        )
        return cls.from_value(
            parsed_module,
            statement,
            roster_name=roster_name,
            value=value,
        )

    @classmethod
    def from_mutation_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> tuple["LatentRosterObservation", ...]:
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
    ) -> "LatentRosterObservation":
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

    def for_mutation(self, mutation_name: str) -> "LatentRosterObservation":
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
class CompactNamedProjectionSurface:
    """Top-level tuple/list/dict references used by registry projections."""

    file_path: str
    surface_name: str
    line: int
    sequence_references: tuple[tuple[str, str | None], ...] = ()
    dict_key_references: tuple[tuple[str, str | None], ...] = ()
    dict_value_references: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True)
class CompactSourceLocation:
    """Source position shared by compact structural observations."""

    file_path: str
    line: int


@dataclass(frozen=True)
class CompactAutoRegisterReferenceIndex:
    """Interned receiver-attribute edges for registry consumer resolution."""

    function_qualnames: tuple[str, ...]
    receiver_names: tuple[str, ...]
    attribute_names: tuple[str, ...]
    encoded_edges: str


class RegistryLookupStyle(StrEnum):
    TRY_EXCEPT = "try_except"
    MEMBERSHIP_GUARD = "membership_guard"


class SelectionGuardKind(StrEnum):
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    NOT_EXACTLY_ONE = "not_exactly_one"

    @classmethod
    def proves_exact_selection(
        cls,
        guard_kinds: set["SelectionGuardKind | None"],
    ) -> bool:
        """Return whether the observed guards prove exactly one selected item."""

        return cls.NOT_EXACTLY_ONE in guard_kinds or {
            cls.EMPTY,
            cls.AMBIGUOUS,
        } <= guard_kinds

    @classmethod
    def from_node(
        cls,
        node: ast.AST,
        match_name: str,
    ) -> "SelectionGuardKind | None":
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Name)
            and node.operand.id == match_name
        ):
            return cls.EMPTY
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Name)
            and node.left.args[0].id == match_name
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, int)
        ):
            return None
        operator = node.ops[0]
        comparator = node.comparators[0].value
        if isinstance(operator, ast.NotEq) and comparator == 1:
            return cls.NOT_EXACTLY_ONE
        if isinstance(operator, ast.Gt) and comparator == 1:
            return cls.AMBIGUOUS
        if isinstance(operator, ast.Eq) and comparator == 0:
            return cls.EMPTY
        return None


@dataclass(frozen=True)
class ClsRegistryMembership:
    operator_type: type[ast.cmpop]
    key_expr: str

    @classmethod
    def from_node(cls, node: ast.AST) -> "ClsRegistryMembership | None":
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
    ) -> "RegistryLookupShape | None":
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


@dataclass(frozen=True)
class CompactRepeatedKeyedFamilyRoot(CompactSourceLocation):
    class_name: str
    family_base_name: str
    registry_key_attr_name: str
    lookup_method_name: str
    lookup_style: RegistryLookupStyle
    error_type_name: str | None
    abstract_hook_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactSortedKeyCall(CompactSourceLocation):
    """One sorted call and the semantic attributes used by its key."""

    registry_owner_names: tuple[str, ...]
    key_attribute_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactKeyedTableAxis(CompactSourceLocation):
    """AST-free module-level dictionary keyed by one enum-like axis."""

    table_name: str
    key_type_name: str
    case_names: tuple[str, ...]
    value_shape_name: str | None


@dataclass(frozen=True)
class CompactClosedAxisBranchFact:
    key_type_name: str
    branch_site_count: int
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactQualifiedSourceLocation(CompactSourceLocation):
    """Qualified source position shared by compact structural observations."""

    qualname: str


@dataclass(frozen=True)
class CompactClosedAxisBranchFunction(CompactQualifiedSourceLocation):
    axes: tuple[CompactClosedAxisBranchFact, ...]


@dataclass(frozen=True)
class CompactManualSelectorAxis(CompactSourceLocation):
    family_name: str
    selector_method_name: str
    key_type_name: str
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class CompactExactTypeGuard(CompactQualifiedSourceLocation):
    subject_expression: str
    type_reference_expression: str
    type_reference_parts: tuple[str, ...]
    matches_exact_type_when_true: bool
    expression: str

    @property
    def structural_membership_expression(self) -> str:
        membership = (
            f"isinstance({self.subject_expression}, {self.type_reference_expression})"
        )
        return membership if self.matches_exact_type_when_true else f"not {membership}"


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


@dataclass(frozen=True)
class CompactClassFamilyIndex(ClassDeclarationIndex[CompactIndexedClass]):
    """Repository inheritance graph reconstructed from compact declarations."""

    @cached_property
    def mro_authority(self) -> ClassMroAuthority:
        return ClassMroAuthority(self.classes_by_symbol)

    @classmethod
    def from_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> Self:
        """Collect and join the complete compact class projection once."""

        return CompactClassFamilyIndexBuilder(
            CompactModuleClassProjectionFamily.collect_modules(modules)
        ).build()

    @classmethod
    def from_projection_groups(
        cls,
        projections_by_family: dict[
            type[CollectedFamily],
            tuple[object, ...],
        ],
    ) -> Self:
        """Build the class anchor declared by a compact multi-family join."""

        return CompactClassFamilyIndexBuilder(
            cast(
                tuple[CompactModuleClassProjection, ...],
                projections_by_family[CompactModuleClassProjectionFamily],
            )
        ).build()

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

    @cached_property
    def product_authority_resolutions_by_symbol(
        self,
    ) -> dict[str, CompactProductAuthorityResolution]:
        """Compose every dataclass product through its exact direct-base chain."""

        lineages: dict[str, _CompactProductLineage] = {}

        def compose_lineage(
            class_symbol: str,
            pending_symbols: frozenset[str],
        ) -> _CompactProductLineage:
            if (lineage := lineages.get(class_symbol)) is not None:
                return lineage
            indexed_class = self.classes_by_symbol[class_symbol]
            declaration = indexed_class.dataclass_declaration
            if declaration is None:
                raise ValueError("product lineage requires a dataclass declaration")
            if class_symbol in pending_symbols:
                return _CompactProductLineage(
                    (),
                    (
                        self._product_authority_failure(
                            indexed_class,
                            CompactProductAuthorityViolation.CYCLIC_PRODUCT_LINEAGE,
                        ),
                    ),
                )
            failures = tuple(
                self._product_authority_failure(
                    indexed_class,
                    failure.violation,
                    line=failure.line,
                )
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
                    self._product_authority_failure(
                        indexed_class,
                        CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
                    ),
                )
            elif direct_base_count > 1:
                failures = (
                    *failures,
                    self._product_authority_failure(
                        indexed_class,
                        CompactProductAuthorityViolation.MULTIPLE_PRODUCT_BASES,
                    ),
                )
            elif direct_base_count == 1:
                base_symbol = indexed_class.resolved_base_symbols[0]
                base_class = self.classes_by_symbol[base_symbol]
                if base_class.dataclass_declaration is None:
                    failures = (
                        *failures,
                        self._product_authority_failure(
                            indexed_class,
                            CompactProductAuthorityViolation.NON_DATACLASS_BASE,
                        ),
                    )
                else:
                    base_lineage = compose_lineage(
                        base_symbol,
                        pending_symbols | frozenset((class_symbol,)),
                    )
                    effective_fields = base_lineage.effective_fields
                    failures = (*failures, *base_lineage.failures)

            fields_by_name = {field.name: field for field in effective_fields}
            for field_declaration in declaration.fields:
                fields_by_name[field_declaration.name] = CompactProductField(
                    name=field_declaration.name,
                    role=field_declaration.role,
                    declaring_class_symbol=class_symbol,
                    file_path=indexed_class.file_path,
                    line=field_declaration.line,
                )
            lineage = _CompactProductLineage(
                effective_fields=tuple(fields_by_name.values()),
                failures=tuple(dict.fromkeys(failures)),
            )
            lineages[class_symbol] = lineage
            return lineage

        resolutions: dict[str, CompactProductAuthorityResolution] = {}
        for class_symbol, indexed_class in self.classes_by_symbol.items():
            if indexed_class.dataclass_declaration is None:
                resolutions[class_symbol] = AbsentCompactProductAuthority(class_symbol)
                continue
            lineage = compose_lineage(class_symbol, frozenset())
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
            resolutions[class_symbol] = (
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
        return resolutions

    @staticmethod
    def _product_authority_failure(
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


ClosedLeafMethodAuthorityPredicate: TypeAlias = Callable[
    ["ClosedLeafMethodAuthorityProof"],
    bool,
]
CLASS_METHOD_OWNERSHIP_HOOK_NAMES = frozenset(("__init_subclass__",))


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
        lambda proof: frozenset(proof.common_direct_base_symbols)
        != frozenset((proof.authority_symbol,)),
    )
    AMBIGUOUS_DECLARED_AUTHORITY = (
        "ambiguous_declared_authority",
        "the participants do not have exactly one declared nominal base",
        lambda proof: proof.common_declared_nominal_base_simple_names
        != frozenset((proof.authority_simple_name,)),
    )
    INCOMPLETE_DIRECT_FAMILY = (
        "incomplete_direct_family",
        "the participants are not the complete direct-child family",
        lambda proof: frozenset(proof.authority_direct_child_symbols)
        != proof.participant_symbol_set,
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

    def is_violated_by(self, proof: "ClosedLeafMethodAuthorityProof") -> bool:
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


@dataclass(frozen=True)
class ClassFamilyIndex(ClassDeclarationIndex[IndexedClass]):

    def class_records_excluding_files(
        self,
        file_paths: frozenset[str],
    ) -> tuple[IndexedClass, ...]:
        return tuple(
            indexed_class
            for indexed_class in self.classes_by_symbol.values()
            if resolved_source_path_text(indexed_class.file_path) not in file_paths
        )

    def projected_with_module_overlay(
        self,
        projected_modules: Iterable[ParsedModule],
        changed_modules: Iterable[ParsedModule],
    ) -> "ClassFamilyIndex":
        """Derive an exact projected index using an overlay only when closed."""

        projected_module_tuple = tuple(projected_modules)
        changed_module_tuple = tuple(changed_modules)
        changed_file_paths = frozenset(
            module.file_path for module in changed_module_tuple
        )
        retained_records = tuple(
            record
            for record in self.classes_by_symbol.values()
            if record.file_path not in changed_file_paths
        )
        replaced_symbols = frozenset(self.classes_by_symbol).difference(
            record.symbol for record in retained_records
        )
        projected_symbols = frozenset(
            record.symbol
            for record in ClassFamilyIndexBuilder(
                changed_module_tuple
            ).module_class_records()
        )
        if not replaced_symbols and not projected_symbols:
            return self
        if replaced_symbols != projected_symbols:
            return build_class_family_index(list(projected_module_tuple))
        return ClassFamilyIndexBuilder(
            changed_module_tuple,
            base_records=retained_records,
        ).build()


def iter_class_definitions(
    statements: list[ast.stmt],
    *,
    parent_qualname: str | None = None,
) -> tuple[tuple[str, ast.ClassDef], ...]:
    classes: list[tuple[str, ast.ClassDef]] = []
    for statement in statements:
        if not isinstance(statement, ast.ClassDef):
            continue
        qualname = (
            statement.name
            if parent_qualname is None
            else f"{parent_qualname}.{statement.name}"
        )
        classes.append((qualname, statement))
        classes.extend(
            iter_class_definitions(list(statement.body), parent_qualname=qualname)
        )
    return tuple(classes)


class _UniqueKnownSymbolSuffixIndex:
    """Resolve only requested suffixes from terminal-name candidate buckets."""

    __slots__ = ("_matches_by_suffix", "_symbols_by_terminal_name")

    def __init__(self, known_symbols: frozenset[str]) -> None:
        symbols_by_terminal_name: dict[str, list[str]] = defaultdict(list)
        for symbol in known_symbols:
            symbols_by_terminal_name[symbol.rsplit(".", 1)[-1]].append(symbol)
        self._symbols_by_terminal_name = {
            terminal_name: tuple(symbols)
            for terminal_name, symbols in symbols_by_terminal_name.items()
        }
        self._matches_by_suffix: dict[str, str | None] = {}

    @classmethod
    def from_terminal_buckets(
        cls,
        symbols_by_terminal_name: dict[str, tuple[str, ...]],
    ) -> "_UniqueKnownSymbolSuffixIndex":
        index = cls.__new__(cls)
        index._symbols_by_terminal_name = symbols_by_terminal_name
        index._matches_by_suffix = {}
        return index

    def get(self, suffix: str) -> str | None:
        if suffix in self._matches_by_suffix:
            return self._matches_by_suffix[suffix]
        qualified_suffix = f".{suffix}"
        match: str | None = None
        for symbol in self._symbols_by_terminal_name.get(suffix.rsplit(".", 1)[-1], ()):
            if symbol != suffix and not symbol.endswith(qualified_suffix):
                continue
            if match is not None and match != symbol:
                match = None
                break
            match = symbol
        self._matches_by_suffix[suffix] = match
        return match

    def root_relative_match(self, qualified_name: str) -> str | None:
        """Resolve one imported name across a narrower analysis-root boundary."""

        matches = tuple(
            symbol
            for symbol in self._symbols_by_terminal_name.get(
                qualified_name.rsplit(".", 1)[-1], ()
            )
            if symbol == qualified_name
            or qualified_name.endswith(f".{symbol}")
            or symbol.endswith(f".{qualified_name}")
        )
        return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=8)
def _unique_known_symbol_by_suffix(
    known_symbols: frozenset[str],
) -> _UniqueKnownSymbolSuffixIndex:
    """Share a bounded lazy suffix resolver for one repository symbol set."""

    return _UniqueKnownSymbolSuffixIndex(known_symbols)


class _ModuleScopeNameReferenceCollector(ast.NodeVisitor):
    """Collect one module binding's syntax without entering child namespaces."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.references: list[ast.Name] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name:
            self.references.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _module_scope_name_references(
    module: ast.Module,
    name: str,
) -> tuple[ast.Name, ...]:
    collector = _ModuleScopeNameReferenceCollector(name)
    collector.visit(module)
    return tuple(collector.references)


@dataclass(frozen=True)
class PublicExportNameReference:
    """One exact string literal in a static public-export declaration."""

    literal: ast.Constant

    def renamed_source(self, literal_source: str, new_name: str) -> str:
        if not isinstance(self.literal.value, str):
            raise ValueError("Public export reference must contain a string")
        if literal_source.count(self.literal.value) != 1:
            raise ValueError("Public export reference cannot be reconstructed")
        return literal_source.replace(self.literal.value, new_name, 1)


@dataclass(frozen=True)
class ModulePublicExportSourceAuthority:
    """Exact source declaration from which a module export contract is derived."""

    statement: ast.Assign | ast.AnnAssign
    target: ast.Name
    value: ast.expr

    @classmethod
    def from_statement(
        cls,
        statement: ast.stmt,
    ) -> "ModulePublicExportSourceAuthority | None":
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == PYTHON_PUBLIC_EXPORT_ASSIGNMENT
        ):
            return cls(statement, statement.targets[0], statement.value)
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == PYTHON_PUBLIC_EXPORT_ASSIGNMENT
            and statement.value is not None
        ):
            return cls(statement, statement.target, statement.value)
        return None

    @classmethod
    def from_module(
        cls,
        module: ast.Module,
    ) -> "ModulePublicExportSourceAuthority | None":
        declarations = tuple(
            declaration
            for statement in module.body
            if (declaration := cls.from_statement(statement)) is not None
        )
        if len(declarations) != 1:
            return None
        declaration = declarations[0]
        references = _module_scope_name_references(
            module,
            PYTHON_PUBLIC_EXPORT_ASSIGNMENT,
        )
        return declaration if references == (declaration.target,) else None

    def name_references(self, name: str) -> tuple[PublicExportNameReference, ...]:
        if not isinstance(self.value, ast.List | ast.Tuple | ast.Set):
            return ()
        return tuple(
            PublicExportNameReference(element)
            for element in self.value.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value == name
        )


def _literal_public_export_names(value: ast.expr) -> tuple[str, ...] | None:
    if not isinstance(value, ast.List | ast.Tuple | ast.Set):
        return None
    if any(
        not isinstance(element, ast.Constant) or not isinstance(element.value, str)
        for element in value.elts
    ):
        return None
    return sorted_tuple({element.value for element in value.elts})


def module_public_export_contract(
    parsed_module: ParsedModule,
) -> CompactModulePublicExportContract:
    module = parsed_module.module
    if PYTHON_PUBLIC_EXPORT_ASSIGNMENT not in (
        LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
    ):
        return CompactImplicitPublicExportContract()
    declaration = ModulePublicExportSourceAuthority.from_module(module)
    if declaration is None:
        return CompactUnresolvedPublicExportContract()
    preceding_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        module.body[: module.body.index(declaration.statement)]
    )
    return (
        CompactExplicitPublicExportContract.from_expression(
            declaration.value,
            preceding_bound_names,
        )
        or CompactUnresolvedPublicExportContract()
    )


class CompactNominalBindingKind(StrEnum):
    """Kinds of exact module bindings and their exported-alias behavior."""

    IMPORT = "import", True
    LOCAL_DECLARATION = "local_declaration", False
    BUILTIN = "builtin", False

    projects_as_import_alias: bool

    def __new__(cls, value: str, projects_as_import_alias: bool) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.projects_as_import_alias = projects_as_import_alias
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
        return CompactNominalReference(parts, binding)

    def resolves_unshadowed_builtin(
        self,
        name: str,
        *,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove that one bare name still resolves through Python builtins."""

        return (
            (
                not self.star_import_ambiguity
                or name in self.star_import_excluded_names
            )
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
        parts = AstExpressionProjection.attribute_chain(
            ClassSymbolResolutionAuthority.reference_node(reference)
        )
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
    ) -> str | None:
        """Resolve a single nominal annotation through its declaration-time binding."""

        reference = NOMINAL_ANNOTATION_SOURCE_AUTHORITY.reference_or_none(annotation)
        if reference is None:
            return None
        return self.qualified_name_at(reference, line=line)

    def _direct_nominal_bindings(
        self,
        statement: ast.stmt,
        preceding_bindings: dict[str, CompactNominalBinding],
    ) -> dict[str, CompactNominalBinding]:
        if isinstance(statement, ast.Import | ast.ImportFrom):
            return {
                origin.bound_name: CompactNominalBinding(
                    qualified_name=origin.qualified_name,
                    kind=CompactNominalBindingKind.IMPORT,
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
        parts = AstExpressionProjection.attribute_chain(
            ClassSymbolResolutionAuthority.reference_node(value)
        )
        if parts is None or (root_binding := preceding_bindings.get(parts[0])) is None:
            return {}
        return {
            local_name: CompactNominalBinding(
                qualified_name=".".join((root_binding.qualified_name, *parts[1:])),
                kind=root_binding.kind,
            )
        }


@dataclass(frozen=True)
class FunctionNominalParameterBindingAuthority:
    """Resolve stable parameter types from one nominal function declaration."""

    module_bindings: ModuleNominalBindingAuthority
    function: ast.FunctionDef | ast.AsyncFunctionDef

    @cached_property
    def stable_type_names_by_parameter(self) -> dict[str, str]:
        """Project nominal parameter types whose bindings remain unchanged."""

        rebound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.function.body
        )
        return {
            parameter.arg: type_name
            for parameter in (
                *self.function.args.posonlyargs,
                *self.function.args.args,
                *self.function.args.kwonlyargs,
            )
            if parameter.arg not in rebound_names
            if parameter.annotation is not None
            if (
                type_name := self.module_bindings.nominal_annotation_name_at(
                    parameter.annotation,
                    line=self.function.lineno,
                )
            )
            is not None
        }

    def type_name_for_reference(self, parameter_name: str) -> str | None:
        """Return the declared nominal type unless the parameter is rebound."""

        return self.stable_type_names_by_parameter.get(parameter_name)


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
    ) -> "ModuleNominalBindingWitness | None":
        raise NotImplementedError

    @abstractmethod
    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> "ModuleNominalBindingWitness | None":
        raise NotImplementedError


@dataclass(frozen=True)
class ModuleNominalBindingWitness:
    """One resolved declaration and the lexical root requiring star exclusion."""

    qualified_name: str
    root_name: str


@dataclass(frozen=True)
class RepositoryModuleBindingProof(
    RepositoryPublicExposureAuthority,
    ModuleNominalBindingView,
):
    """Resolve names past star imports only when source export contracts close them."""

    modules: tuple[ParsedModule, ...]

    @cached_property
    def public_export_contract_by_module_name(
        self,
    ) -> dict[str, CompactModulePublicExportContract]:
        return {
            module.module_name: module_public_export_contract(module)
            for module in self.modules
        }

    @cached_property
    def module_name_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for module in self.modules:
            counts[module.module_name] += 1
        return counts

    def contains_module(self, module_name: str) -> bool:
        return self.module_name_counts.get(module_name) == 1

    @cached_property
    def star_import_origins_by_module_name(
        self,
    ) -> dict[str, tuple[CompactModuleStarImportOrigin, ...]]:
        return {
            module.module_name: module_star_import_origins(module)
            for module in self.modules
        }

    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        contract = (
            self.public_export_contract_by_module_name.get(module_name)
            if self.contains_module(module_name)
            else None
        )
        return (
            CompactPublicNameExposure.UNRESOLVED
            if contract is None
            else contract.exposure_for(binding_name)
        )

    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        return self.star_import_origins_by_module_name.get(module_name, ())

    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        if not (
            self.star_imports_exclude(module.module_name, name)
            and ModuleNominalBindingAuthority(module)
            .snapshot_before(
                line,
                policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
            )
            .resolves_unshadowed_builtin(
                name,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        ):
            return None
        return ModuleNominalBindingWitness(f"builtins.{name}", name)

    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        root_name = nominal_reference_root_name(reference)
        if root_name is None or not self.star_imports_exclude(
            module.module_name,
            root_name,
        ):
            return None
        qualified_name = ModuleNominalBindingAuthority(module).qualified_name_at(
            reference,
            line=line,
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        if qualified_name is None:
            return None
        return ModuleNominalBindingWitness(qualified_name, root_name)


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
    if isinstance(statement, ast.Import):
        return frozenset(
            alias.asname or alias.name.split(".", 1)[0] for alias in statement.names
        )
    if isinstance(statement, ast.ImportFrom):
        return frozenset(
            alias.asname or alias.name for alias in statement.names if alias.name != "*"
        )
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
            unresolved_bound_names.intersection_update(
                star_import_excluded_names
            )
            star_import_ambiguity = True
            return
        direct_bindings = authority._direct_nominal_bindings(current, bindings)
        bound_names = _direct_statement_bound_names(current)
        deleted_names = (
            bound_names if isinstance(current, ast.Delete) else frozenset()
        )
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


@lru_cache(maxsize=None)
def _module_import_aliases(parsed_module: ParsedModule) -> dict[str, str]:
    return {
        local_name: binding.qualified_name
        for local_name, binding in _module_nominal_binding_snapshots(
            ModuleNominalBindingAuthority(parsed_module),
            (),
            include_final=True,
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )[None].bindings_by_name.items()
        if binding.kind.projects_as_import_alias
    }


def module_star_import_origins(
    parsed_module: ParsedModule,
) -> tuple[CompactModuleStarImportOrigin, ...]:
    origins: list[CompactModuleStarImportOrigin] = []

    class ModuleScopeStarImportCollector(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if any(alias.name == "*" for alias in node.names):
                origins.append(
                    CompactModuleStarImportOrigin(
                        (
                            parsed_module.module_path_identity.resolve_import_from_module(
                                imported_module=node.module,
                                level=node.level,
                            )
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
        for base in node.bases
        if (
            parts := AstExpressionProjection.attribute_chain(
                ClassSymbolResolutionAuthority.reference_node(base)
            )
        )
        is not None
    )


@dataclass(frozen=True)
class _CompactClassBindingFacets:
    """Class facts whose authority depends on one module binding snapshot."""

    base_references: tuple[CompactNominalReference, ...]
    class_decorators_are_promotion_safe: bool
    dataclass_declaration: CompactDataclassDeclaration | None

    @property
    def product_base_bindings_are_exact(self) -> bool:
        return all(
            reference.root_binding is not None
            for reference in self.base_references
            if ClassSymbolResolutionAuthority.establishes_nominal_family(
                ".".join(reference.source_parts)
            )
        )


def _compact_class_binding_facets(
    node: ast.ClassDef,
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    qualname: str,
    *,
    include_body_facets: bool,
) -> _CompactClassBindingFacets:
    return _CompactClassBindingFacets(
        base_references=_compact_base_references(node, module_binding_snapshot),
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
            else None
        ),
    )


def _compact_indexed_classes(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    *,
    include_body_facets: bool,
) -> tuple[CompactIndexedClass, ...]:
    file_path = parsed_module.file_path
    binding_snapshots = _module_nominal_binding_snapshots(
        ModuleNominalBindingAuthority(parsed_module),
        tuple(node.lineno for _qualname, node in indexed_class_nodes),
        include_final=False,
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
            base_references_are_complete=len(binding_facets.base_references)
            == len(node.bases),
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
            class_keyword_names=tuple(keyword.arg or "**" for keyword in node.keywords),
            class_decorators_are_promotion_safe=(
                binding_facets.class_decorators_are_promotion_safe
            ),
            class_header_is_reconstructible=ClassHeaderSourceSpan(
                node,
                parsed_module.source_segments.lines,
            ).is_reconstructible,
            keyed_family_key_type_name=_keyed_family_key_type_name(node),
            is_final=any(
                (isinstance(decorator, ast.Name) and decorator.id == "final")
                or (isinstance(decorator, ast.Attribute) and decorator.attr == "final")
                for decorator in node.decorator_list
            ),
            end_line=node.end_lineno,
            method_names=tuple(
                statement.name
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            abstract_method_names=sorted_tuple(
                statement.name
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                if any(
                    _terminal_reference_name(decorator) == "abstractmethod"
                    for decorator in statement.decorator_list
                )
            ),
            is_abstract=_is_abstract_class(node),
            dataclass_declaration=binding_facets.dataclass_declaration,
            declares_autoregister_meta=_declares_autoregister_meta(node),
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
        )
        for qualname, node in indexed_class_nodes
        for module_binding_snapshot in (binding_snapshots[node.lineno],)
        for binding_facets in (
            _compact_class_binding_facets(
                node,
                module_binding_snapshot,
                qualname,
                include_body_facets=include_body_facets,
            ),
        )
    )


def _class_nominal_reference_root_names(
    parsed_module: ParsedModule,
) -> frozenset[str]:
    """Return module bindings queried while projecting class declarations."""

    reference_nodes: list[ast.AST] = []
    for _qualname, class_node in iter_class_definitions(
        list(parsed_module.module.body)
    ):
        reference_nodes.extend(class_node.bases)
        reference_nodes.extend(class_node.decorator_list)
        reference_nodes.extend(keyword.value for keyword in class_node.keywords)
        for statement in class_node.body:
            if isinstance(statement, ast.AnnAssign):
                reference_nodes.append(statement.annotation)
                if statement.value is not None:
                    reference_nodes.append(statement.value)
            elif isinstance(statement, ast.Assign):
                reference_nodes.append(statement.value)
    return frozenset(
        node.id
        for reference_node in reference_nodes
        for node in ast.walk(reference_node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _repository_refined_class_projection(
    parsed_module: ParsedModule,
    projection: CompactModuleClassProjection,
    star_import_excluded_names: frozenset[str],
) -> CompactModuleClassProjection:
    """Refine only binding-sensitive class facts from repository export proof."""

    class_nodes_by_qualname = dict(
        iter_class_definitions(list(parsed_module.module.body))
    )
    binding_snapshots = _module_nominal_binding_snapshots(
        ModuleNominalBindingAuthority(parsed_module),
        tuple(node.lineno for node in class_nodes_by_qualname.values()),
        include_final=False,
        star_import_excluded_names=star_import_excluded_names,
    )
    refined_classes = []
    for indexed_class in projection.classes:
        node = class_nodes_by_qualname[indexed_class.qualname]
        binding_snapshot = binding_snapshots[node.lineno]
        binding_facets = _compact_class_binding_facets(
            node,
            binding_snapshot,
            indexed_class.qualname,
            include_body_facets=True,
        )
        refined_classes.append(
            replace(
                indexed_class,
                base_references=binding_facets.base_references,
                base_references_are_complete=len(binding_facets.base_references)
                == len(node.bases),
                product_base_bindings_are_exact=(
                    binding_facets.product_base_bindings_are_exact
                ),
                class_decorators_are_promotion_safe=(
                    binding_facets.class_decorators_are_promotion_safe
                ),
                dataclass_declaration=binding_facets.dataclass_declaration,
            )
        )
    return replace(projection, classes=tuple(refined_classes))


class CompactModuleClassProjectionFamily(CollectedFamily[CompactModuleClassProjection]):
    """Persist class/import facts needed by the global inheritance graph."""

    item_type = CompactModuleClassProjection
    cache_payload_max_bytes = 3_000_000
    source_demand_collector = staticmethod(
        _collect_demanded_class_projection_from_source
    )

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
                    indexed_class.declares_autoregister_meta
                    or (
                        indexed_class.keyed_family_key_type_name is not None
                        and "registry_key_attr" in indexed_class.assignments_by_name
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

        parsed_modules = tuple(parsed_modules)
        repository_binding_proof = RepositoryModuleBindingProof(parsed_modules)
        projections: list[CompactModuleClassProjection] = []
        for parsed_module in parsed_modules:
            star_import_excluded_names = frozenset(
                name
                for name in _class_nominal_reference_root_names(parsed_module)
                if repository_binding_proof.star_imports_exclude(
                    parsed_module.module_name,
                    name,
                )
            )
            module_projections = collect_family_items(parsed_module, cls)
            if star_import_excluded_names and len(module_projections) == 1:
                module_projections = [
                    _repository_refined_class_projection(
                        parsed_module,
                        module_projections[0],
                        star_import_excluded_names,
                    )
                ]
            if len(module_projections) != 1:
                raise ValueError(
                    "Compact class projection family must emit one item per module"
                )
            projections.append(module_projections[0])
        return tuple(projections)

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
    def _collect_header_core(
        cls,
        parsed_module: ParsedModule,
    ) -> list[CompactModuleClassProjection]:
        del cls
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        return [
            CompactModuleClassProjection(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                import_aliases=tuple(
                    sorted(_module_import_aliases(parsed_module).items())
                ),
                public_export_contract=module_public_export_contract(
                    parsed_module
                ),
                star_import_origins=module_star_import_origins(parsed_module),
                classes=_compact_indexed_classes(
                    parsed_module,
                    indexed_class_nodes,
                    include_body_facets=False,
                ),
            ).header_core()
        ]

    @classmethod
    def _collect(
        cls,
        parsed_module: ParsedModule,
        demand: CompactClassProjectionDemand | None,
    ) -> list[CompactModuleClassProjection]:
        del cls
        file_path = parsed_module.file_path
        syntax_facets = _compact_class_syntax_facets(
            parsed_module,
            collect_autoregister=(
                demand is None or demand.include_autoregister_references
            ),
        )
        indexed_class_nodes = iter_class_definitions(list(parsed_module.module.body))
        class_methods = _compact_class_methods(
            parsed_module,
            indexed_class_nodes,
            method_names=(None if demand is None else demand.class_method_names),
        )
        classes = _compact_indexed_classes(
            parsed_module,
            indexed_class_nodes,
            include_body_facets=True,
        )
        return [
            CompactModuleClassProjection(
                module_name=parsed_module.module_name,
                file_path=file_path,
                import_aliases=tuple(
                    sorted(_module_import_aliases(parsed_module).items())
                ),
                public_export_contract=module_public_export_contract(
                    parsed_module
                ),
                star_import_origins=module_star_import_origins(parsed_module),
                classes=classes,
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
                exact_type_guards=syntax_facets.exact_type_guards,
                autoregister_reference_index=(
                    syntax_facets.autoregister_reference_index
                ),
                repeated_keyed_family_roots=_compact_repeated_keyed_family_roots(
                    parsed_module
                ),
                manual_subclass_roster_roots=_compact_manual_subclass_roster_roots(
                    parsed_module
                ),
                latent_rosters=LatentRosterObservation.from_module(parsed_module),
                named_projection_surfaces=_compact_named_projection_surfaces(
                    parsed_module
                ),
                class_methods=class_methods,
            )
        ]


def _compact_predicate_selected_methods(
    node: ast.ClassDef,
) -> tuple[tuple[int, str, str, str], ...]:
    """Project exact registered-types selector shapes without retaining method ASTs."""

    methods: list[tuple[int, str, str, str]] = []
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            _terminal_reference_name(decorator) == "classmethod"
            for decorator in statement.decorator_list
        ):
            continue
        shape = _compact_registered_type_match_shape(statement)
        if shape is None:
            continue
        match_name, predicate_name, context_name = shape
        guard_kinds = {
            SelectionGuardKind.from_node(candidate.test, match_name)
            for candidate in _trim_leading_docstring(list(statement.body))
            if isinstance(candidate, ast.If)
        }
        if not SelectionGuardKind.proves_exact_selection(guard_kinds):
            continue
        if not any(
            isinstance(candidate, ast.Subscript)
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == match_name
            and isinstance(candidate.slice, ast.Constant)
            and candidate.slice.value == 0
            for candidate in ast.walk(statement)
        ):
            continue
        methods.append((statement.lineno, statement.name, predicate_name, context_name))
    return tuple(methods)


def _trim_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _compact_registered_type_match_shape(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    parameter_names = {
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    }
    for statement in _trim_leading_docstring(list(method.body)):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.ListComp)
            and len(statement.value.generators) == 1
        ):
            continue
        match_name = statement.targets[0].id
        comprehension = statement.value
        generator = comprehension.generators[0]
        if not (
            not generator.is_async
            and isinstance(generator.target, ast.Name)
            and isinstance(comprehension.elt, ast.Name)
            and comprehension.elt.id == generator.target.id
            and isinstance(generator.iter, ast.Call)
            and not generator.iter.args
            and not generator.iter.keywords
            and isinstance(generator.iter.func, ast.Attribute)
            and generator.iter.func.attr == "registered_types"
            and isinstance(generator.iter.func.value, ast.Name)
            and generator.iter.func.value.id == "cls"
            and len(generator.ifs) == 1
            and isinstance(generator.ifs[0], ast.Call)
        ):
            continue
        predicate = generator.ifs[0]
        if not (
            not predicate.keywords
            and len(predicate.args) == 1
            and isinstance(predicate.args[0], ast.Name)
            and predicate.args[0].id in parameter_names
            and isinstance(predicate.func, ast.Attribute)
            and predicate.func.attr
            and isinstance(predicate.func.value, ast.Name)
            and predicate.func.value.id == generator.target.id
        ):
            continue
        return match_name, predicate.func.attr, predicate.args[0].id
    return None


def _compact_manual_subclass_roster_roots(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSubclassRosterRoot, ...]:
    roots: list[CompactManualSubclassRosterRoot] = []
    file_path = parsed_module.file_path
    for qualname, node in iter_class_definitions(list(parsed_module.module.body)):
        registry_names = _compact_class_list_registry_names(node)
        if not registry_names:
            continue
        init_subclass = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__init_subclass__"
            ),
            None,
        )
        if init_subclass is None:
            continue
        sites = _compact_manual_subclass_registration_sites(
            init_subclass, registry_names, owner_name=node.name
        )
        if not sites:
            continue
        consumers: list[tuple[str, int, str, str]] = []
        for registry_name in registry_names:
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name == "__init_subclass__":
                    continue
                if _compact_uses_named_registry(
                    method,
                    registry_name=registry_name,
                    owner_names=frozenset({"cls", "type", node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            method.lineno,
                            f"{node.name}.{method.name}",
                            file_path,
                        )
                    )
            for function_qualname, function in _named_functions(parsed_module.module):
                if "." in function_qualname:
                    continue
                if _compact_uses_named_registry(
                    function,
                    registry_name=registry_name,
                    owner_names=frozenset({node.name}),
                ):
                    consumers.append(
                        (
                            registry_name,
                            function.lineno,
                            function_qualname,
                            file_path,
                        )
                    )
        roots.append(
            CompactManualSubclassRosterRoot(
                class_symbol=f"{parsed_module.module_name}.{qualname}",
                init_subclass_line=init_subclass.lineno,
                registration_sites=sites,
                consumer_locations=sorted_tuple(
                    set(consumers), key=lambda item: (item[1], item[2], item[0])
                ),
            )
        )
    return tuple(roots)


def _compact_class_list_registry_names(node: ast.ClassDef) -> tuple[str, ...]:
    return sorted_tuple(
        {
            target_name
            for statement in node.body
            for target_name in (
                (
                    statement.targets[0].id
                    if isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.List)
                    else (
                        statement.target.id
                        if isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and isinstance(statement.value, ast.List)
                        else None
                    )
                ),
            )
            if target_name is not None
        }
    )


def _compact_manual_subclass_registration_sites(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    registry_names: tuple[str, ...],
    *,
    owner_name: str,
) -> tuple[CompactManualSubclassRegistrationSite, ...]:
    sites: dict[str, CompactManualSubclassRegistrationSite] = {}

    def walk_statements(
        statements: list[ast.stmt], guard_stack: tuple[ast.AST, ...]
    ) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_statements(statement.body, (*guard_stack, statement.test))
                walk_statements(statement.orelse, guard_stack)
                continue
            for subnode in ast.walk(statement):
                registry_name = _compact_registration_append_registry_name(
                    subnode, registry_names, owner_name
                )
                if registry_name is None:
                    continue
                sites[registry_name] = CompactManualSubclassRegistrationSite(
                    registry_name=registry_name,
                    guard_summary=(
                        " and ".join(ast.unparse(guard) for guard in guard_stack)
                        if guard_stack
                        else None
                    ),
                    selector_attr_name=next(
                        (
                            attr_name
                            for guard in guard_stack
                            if (attr_name := _compact_guarded_defined_attr_name(guard))
                            is not None
                        ),
                        None,
                    ),
                    requires_concrete_subclass=any(
                        _compact_guard_requires_concrete_subclass(guard)
                        for guard in guard_stack
                    ),
                )

    walk_statements(_trim_leading_docstring(list(method.body)), ())
    return tuple(sites[name] for name in sorted(sites))


def _compact_registration_append_registry_name(
    node: ast.AST, registry_names: tuple[str, ...], owner_name: str
) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in registry_names
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id in {"cls", "type", owner_name}
        and _compact_looks_like_cls_registration_value(node.args[0])
    ):
        return None
    return node.func.value.attr


def _compact_looks_like_cls_registration_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "cls"
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast"
        and node.args
        and _compact_looks_like_cls_registration_value(node.args[-1])
    )


def _compact_class_dict_get_attr_name(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "__dict__"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "cls"
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    return node.args[0].value


def _compact_guarded_defined_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _compact_class_dict_get_attr_name(node)
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.IsNot, ast.NotEq))
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    ):
        return None
    return _compact_class_dict_get_attr_name(node.left)


def _compact_guard_requires_concrete_subclass(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Attribute)
        and isinstance(node.operand.func.value, ast.Name)
        and node.operand.func.value.id == "inspect"
        and node.operand.func.attr == "isabstract"
        and len(node.operand.args) == 1
        and isinstance(node.operand.args[0], ast.Name)
        and node.operand.args[0].id == "cls"
    )


def _compact_uses_named_registry(
    node: ast.AST,
    *,
    registry_name: str,
    owner_names: frozenset[str],
) -> bool:
    return any(
        isinstance(subnode, ast.Attribute)
        and subnode.attr == registry_name
        and isinstance(subnode.value, ast.Name)
        and subnode.value.id in owner_names
        for subnode in ast.walk(node)
    )


def _compact_assignment_target_value(
    statement: ast.stmt,
) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return statement.targets[0], statement.value
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return statement.target, statement.value
    return None


def _compact_named_projection_surfaces(
    parsed_module: ParsedModule,
) -> tuple[CompactNamedProjectionSurface, ...]:
    named_values: dict[str, tuple[int, ast.AST]] = {}
    named_sequences: dict[str, tuple[int, ast.Tuple | ast.List]] = {}
    for statement in _trim_leading_docstring(list(parsed_module.module.body)):
        target_value = _compact_assignment_target_value(statement)
        if target_value is None or not isinstance(target_value[0], ast.Name):
            continue
        target, value = target_value
        named_values[target.id] = (statement.lineno, value)
        if isinstance(value, (ast.Tuple, ast.List)):
            named_sequences[target.id] = (statement.lineno, value)

    surfaces: list[CompactNamedProjectionSurface] = []
    for surface_name, (line, value) in named_sequences.items():
        surfaces.append(
            CompactNamedProjectionSurface(
                file_path=parsed_module.file_path,
                surface_name=surface_name,
                line=line,
                sequence_references=tuple(
                    reference
                    for element in value.elts
                    if (reference := _compact_projection_reference(element)) is not None
                ),
            )
        )
    for surface_name, (line, value) in named_values.items():
        if isinstance(value, ast.Dict):
            surfaces.append(
                CompactNamedProjectionSurface(
                    file_path=parsed_module.file_path,
                    surface_name=surface_name,
                    line=line,
                    dict_key_references=tuple(
                        reference
                        for key in value.keys
                        if key is not None
                        if (reference := _compact_projection_reference(key)) is not None
                    ),
                    dict_value_references=tuple(
                        reference
                        for item in value.values
                        if (reference := _compact_projection_reference(item))
                        is not None
                    ),
                )
            )
    return tuple(surfaces)


def _compact_projection_reference(
    node: ast.AST,
) -> tuple[str, str | None] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, None
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = AstExpressionProjection.attribute_chain(node)
        if parts is None:
            return ast.unparse(node), None
        return ".".join(parts), parts[0]
    return None


def _compact_self_assignment(
    statement: ast.stmt,
) -> tuple[str, ast.AST | None] | None:
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
        value = statement.value
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    return target.attr, value


def _compact_class_field_type_map(
    node: ast.ClassDef,
) -> tuple[tuple[str, str], ...]:
    nominal_fields: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if not CLASSVAR_ANNOTATION_AUTHORITY.matches(statement.annotation):
                nominal_fields.setdefault(
                    statement.target.id,
                    ast.unparse(statement.annotation),
                )
            continue
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name != "__init__":
            continue
        parameter_annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in (
                *statement.args.posonlyargs,
                *statement.args.args,
                *statement.args.kwonlyargs,
            )
            if argument.annotation is not None
        }
        for inner in statement.body:
            assignment = _compact_self_assignment(inner)
            if assignment is None:
                continue
            field_name, value = assignment
            if isinstance(value, ast.Name) and value.id in parameter_annotations:
                nominal_fields.setdefault(
                    field_name,
                    parameter_annotations[value.id],
                )
    return sorted_tuple(nominal_fields.items())


class _CompactClassMethodSemanticSkeletonNormalizer(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        del node
        return ast.arg(arg="ARG", annotation=None, type_comment=None)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr="ATTR",
                ctx=node.ctx,
            ),
            node,
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value="CONST"), node)


def _compact_class_method_statement_skeleton(statement: ast.stmt) -> str:
    normalized = _CompactClassMethodSemanticSkeletonNormalizer().visit(statement)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _compact_class_method_coordinates(
    node: ast.AST,
    path: tuple[object, ...] = (),
) -> tuple[CompactMethodSemanticCoordinate, ...]:
    coordinates: list[CompactMethodSemanticCoordinate] = []
    coordinate_path = tuple(str(item) for item in path)
    if isinstance(node, ast.Constant):
        coordinates.append(
            CompactMethodSemanticCoordinate(
                coordinate_path,
                CompactMethodSemanticCoordinateKind.CONSTANT,
                repr(node.value),
            )
        )
    elif isinstance(node, ast.Name):
        coordinates.append(
            CompactMethodSemanticCoordinate(
                coordinate_path,
                CompactMethodSemanticCoordinateKind.NAME,
                node.id,
            )
        )
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            coordinates.append(
                CompactMethodSemanticCoordinate(
                    coordinate_path,
                    CompactMethodSemanticCoordinateKind.SELF_ATTRIBUTE,
                    node.attr,
                )
            )
        else:
            coordinates.append(
                CompactMethodSemanticCoordinate(
                    coordinate_path,
                    CompactMethodSemanticCoordinateKind.ATTRIBUTE,
                    ast.unparse(node),
                )
            )
    elif isinstance(node, ast.Call):
        coordinates.append(
            CompactMethodSemanticCoordinate(
                tuple(str(item) for item in (*path, "func")),
                CompactMethodSemanticCoordinateKind.CALL,
                ast.unparse(node.func),
            )
        )
    skipped_fields = {"func"} if isinstance(node, ast.Call) else set()
    for field_name, value in ast.iter_fields(node):
        if field_name in skipped_fields:
            continue
        if isinstance(value, ast.AST):
            coordinates.extend(
                _compact_class_method_coordinates(value, (*path, field_name))
            )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ast.AST):
                    coordinates.extend(
                        _compact_class_method_coordinates(
                            item,
                            (*path, field_name, index),
                        )
                    )
    return tuple(coordinates)


def _compact_class_method_statements_from_sources(
    statement_sources: tuple[str, ...],
) -> tuple[ast.stmt, ...]:
    statements = tuple(ast.parse("\n".join(statement_sources)).body)
    if len(statements) != len(statement_sources):
        raise ValueError("Method-family statement source lost its AST boundary")
    return statements


def _compact_class_method(
    class_symbol: str,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: tuple[str, ...],
    module_bound_names: frozenset[str],
    class_bound_names: frozenset[str],
    *,
    include_family_semantics: bool,
) -> CompactClassMethod:
    body = _trim_leading_docstring(list(method.body))
    decorator_lines = tuple(
        decorator.lineno for decorator in method.decorator_list if decorator.lineno
    )
    start_line = min((*decorator_lines, method.lineno))
    is_tiny_role = len(body) <= 2
    safety_profile = (
        ClassMethodPromotionSafetyProfile.from_method(
            method,
            module_bound_names,
            class_bound_names,
            source_lines=source_lines,
        )
        if is_tiny_role
        else None
    )
    return CompactClassMethod(
        class_symbol=class_symbol,
        method_name=method.name,
        line=start_line,
        line_count=max(1, (method.end_lineno or method.lineno) - start_line + 1),
        body_statement_count=len(body),
        statement_sources=(
            tuple(ast.unparse(statement) for statement in body)
            if include_family_semantics and len(body) >= 3
            else ()
        ),
        exact_source_digest=(
            hashlib.blake2s(
                "".join(
                    source_lines[start_line - 1 : (method.end_lineno or method.lineno)]
                ).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
            if is_tiny_role
            else None
        ),
        promotion_hazards=() if safety_profile is None else safety_profile.hazards,
        receiver_member_names=(
            ClassMethodReceiverRequirements.from_method(method).member_names
            if is_tiny_role
            else frozenset()
        ),
    )


def _compact_class_methods(
    parsed_module: ParsedModule,
    indexed_class_nodes: tuple[tuple[str, ast.ClassDef], ...],
    *,
    method_names: frozenset[str] | None = None,
) -> tuple[CompactClassMethod, ...]:
    methods: list[CompactClassMethod] = []
    source_lines = tuple(parsed_module.source.splitlines(keepends=True))
    module_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    for qualname, node in indexed_class_nodes:
        class_symbol = f"{parsed_module.module_name}.{qualname}"
        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        establishes_nominal_family = any(
            ClassSymbolResolutionAuthority.establishes_nominal_family(declared_name)
            for base in node.bases
            if (
                declared_name := ClassSymbolResolutionAuthority.declared_base_name(base)
            )
            is not None
        )
        for statement in node.body:
            if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if method_names is not None and statement.name not in method_names:
                continue
            method = _compact_class_method(
                class_symbol,
                statement,
                source_lines,
                module_bound_names,
                class_bound_names,
                include_family_semantics=establishes_nominal_family,
            )
            if establishes_nominal_family or method.statement_count <= 2:
                methods.append(method)
    return tuple(methods)


def _annotation_type_names(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, ast.Constant) and node.value is None:
        return ()
    if isinstance(node, ast.Name):
        return () if node.id == "None" else (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        return sorted_tuple(
            {name for element in node.elts for name in _annotation_type_names(element)}
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return sorted_tuple(
            {*_annotation_type_names(node.left), *_annotation_type_names(node.right)}
        )
    if isinstance(node, ast.Subscript):
        base_name = _terminal_reference_name(node.value)
        if base_name in {"Optional", "Required", "NotRequired", "Type", "type"}:
            return _annotation_type_names(node.slice)
        if base_name == "Annotated":
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                return _annotation_type_names(node.slice.elts[0])
            return _annotation_type_names(node.slice)
    return ()


def _keyed_family_key_type_name(node: ast.ClassDef) -> str | None:
    for base in node.bases:
        if not isinstance(base, ast.Subscript):
            continue
        if _terminal_reference_name(base.value) != "KeyedNominalFamily":
            continue
        type_names = _annotation_type_names(base.slice)
        if type_names:
            return type_names[0]
    return None


def _compact_keyed_table_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactKeyedTableAxis, ...]:
    axes: list[CompactKeyedTableAxis] = []
    for statement in parsed_module.module.body:
        table_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                table_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            table_name = statement.target.id
            value = statement.value
        if table_name is None or not isinstance(value, ast.Dict):
            continue
        if len(value.keys) < 2 or any(key is None for key in value.keys):
            continue
        case_names = tuple(ast.unparse(key) for key in value.keys if key is not None)
        key_type_names = {
            case_name.split(".", 1)[0] for case_name in case_names if "." in case_name
        }
        if len(key_type_names) != 1:
            continue
        value_shape_name: str | None = None
        value_constructor_names = {
            ast.unparse(item.func)
            for item in value.values
            if isinstance(item, ast.Call)
        }
        if (
            all(isinstance(item, ast.Call) for item in value.values)
            and len(value_constructor_names) == 1
        ):
            value_shape_name = next(iter(value_constructor_names))
        axes.append(
            CompactKeyedTableAxis(
                file_path=parsed_module.file_path,
                line=statement.lineno,
                table_name=table_name,
                key_type_name=next(iter(key_type_names)),
                case_names=sorted_tuple(case_names),
                value_shape_name=value_shape_name,
            )
        )
    return tuple(axes)


def _compact_manual_selector_axes(
    parsed_module: ParsedModule,
) -> tuple[CompactManualSelectorAxis, ...]:
    dict_literals: dict[str, ast.Dict] = {}
    for statement in parsed_module.module.body:
        name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            name = statement.target.id
            value = statement.value
        if name is not None and isinstance(value, ast.Dict):
            dict_literals[name] = value
    case_names_by_mapping = {
        name: tuple(ast.unparse(key) for key in mapping.keys if key is not None)
        for name, mapping in dict_literals.items()
    }
    known_mapping_names = frozenset(
        name
        for name, case_names in case_names_by_mapping.items()
        if len(case_names) >= 2
    )
    axes: list[CompactManualSelectorAxis] = []
    for node in _walk_nodes(parsed_module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not method.name.startswith("for_") or not any(
                _terminal_reference_name(decorator) == "classmethod"
                for decorator in method.decorator_list
            ):
                continue
            parameter_names = {
                item.arg
                for item in (
                    *method.args.posonlyargs,
                    *method.args.args,
                    *method.args.kwonlyargs,
                )
                if item.arg not in {"self", "cls"}
            }
            if not parameter_names:
                continue
            mapping_name: str | None = None
            for subnode in ast.walk(method):
                if not (
                    isinstance(subnode, ast.Subscript)
                    and isinstance(subnode.value, ast.Name)
                    and subnode.value.id in known_mapping_names
                    and ast.unparse(subnode.slice) in parameter_names
                ):
                    continue
                mapping_name = subnode.value.id
                break
            if mapping_name is None:
                continue
            case_names = case_names_by_mapping[mapping_name]
            key_type_names = {
                case_name.split(".", 1)[0]
                for case_name in case_names
                if "." in case_name
            }
            if len(key_type_names) != 1:
                continue
            axes.append(
                CompactManualSelectorAxis(
                    file_path=parsed_module.file_path,
                    line=method.lineno,
                    family_name=node.name,
                    selector_method_name=method.name,
                    key_type_name=next(iter(key_type_names)),
                    case_names=case_names,
                )
            )
    return tuple(axes)


def _exact_type_predicate(
    node: ast.AST,
) -> tuple[ast.AST, ast.AST, bool, str] | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negated = True
        comparison = node.operand
    else:
        negated = False
        comparison = node
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and len(comparison.comparators) == 1
    ):
        return None
    operator = comparison.ops[0]
    if not isinstance(operator, (ast.Is, ast.Eq, ast.IsNot, ast.NotEq)):
        return None
    left = comparison.left
    right = comparison.comparators[0]
    subject = _type_call_subject(left)
    type_reference = right
    if subject is None:
        subject = _type_call_subject(right)
        type_reference = left
    if subject is None:
        return None
    matches_exact_type = isinstance(operator, (ast.Is, ast.Eq))
    return (
        subject,
        type_reference,
        not matches_exact_type if negated else matches_exact_type,
        ast.unparse(node),
    )


def _type_call_subject(node: ast.AST) -> ast.AST | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and not node.keywords
    ):
        return None
    return node.args[0]


def _fail_loud_block(statements: list[ast.stmt]) -> bool:
    non_terminating_prefix_types = (
        ast.AnnAssign,
        ast.Assign,
        ast.AugAssign,
        ast.Delete,
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Pass,
    )
    return (
        bool(statements)
        and isinstance(statements[-1], ast.Raise)
        and all(
            isinstance(statement, non_terminating_prefix_types)
            for statement in statements[:-1]
        )
    )


def _named_functions(
    module: ast.Module,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    return named_function_nodes(module)


def _enum_member_refs_by_key_type(node: ast.AST) -> dict[str, tuple[str, ...]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for subnode in ast.walk(node):
        parts = AstExpressionProjection.attribute_chain(subnode)
        if parts is None or len(parts) < 2:
            continue
        key_type_name = parts[-2]
        refs[key_type_name].add(f"{key_type_name}.{parts[-1]}")
    return {
        key_type_name: sorted_tuple(case_names)
        for key_type_name, case_names in refs.items()
    }


def _direct_class_assignments(node: ast.ClassDef) -> dict[str, ast.AST | None]:
    assignments: dict[str, ast.AST | None] = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _string_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _module_string_constant_assignments(
    parsed_module: ParsedModule,
) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in parsed_module.module.body:
        target_name: str | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        if target_name is not None and (string_value := _string_constant(value)):
            constants[target_name] = string_value
    return constants


def _class_direct_string_member_assignments(
    node: ast.ClassDef,
) -> dict[str, str]:
    return {
        name: string_value
        for name, value in _direct_class_assignments(node).items()
        if (string_value := _string_constant(value)) is not None
    }


def _module_string_enum_member_assignments(
    parsed_module: ParsedModule,
) -> dict[tuple[str, str], str]:
    members: dict[tuple[str, str], str] = {}
    for statement in parsed_module.module.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        if not PYTHON_ENUM_BASE_AUTHORITY.matches_any(
            _terminal_reference_name(base) for base in statement.bases
        ):
            continue
        for member_name, string_value in _class_direct_string_member_assignments(
            statement
        ).items():
            members[(statement.name, member_name)] = string_value
    return members


def _enum_member_value_reference(node: ast.AST | None) -> tuple[str, str] | None:
    if not (
        isinstance(node, ast.Attribute)
        and node.attr == "value"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
    ):
        return None
    return node.value.value.id, node.value.attr


def _autoregister_constant_name(
    node: ast.AST | None,
    parsed_module: ParsedModule,
) -> str | None:
    if (string_value := _string_constant(node)) is not None:
        return string_value
    if (enum_ref := _enum_member_value_reference(node)) is not None:
        return _module_string_enum_member_assignments(parsed_module).get(enum_ref)
    if isinstance(node, ast.Name):
        return (
            _module_string_constant_assignments(parsed_module).get(node.id) or node.id
        )
    if isinstance(node, ast.Attribute) and node.attr == "__registry_key__":
        return node.attr
    return None


def _registry_family_key_attr_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryFamily" or not node.args:
        return None
    key_arg = node.args[0]
    if (key_literal := _string_constant(key_arg)) is not None:
        return key_literal
    if isinstance(key_arg, ast.Attribute):
        return key_arg.attr.lower()
    return None


def _registry_config_keyword(
    node: ast.AST | None,
    keyword_name: str,
) -> ast.AST | None:
    if not isinstance(node, ast.Call):
        return None
    if _terminal_reference_name(node.func) != "RegistryConfig":
        return None
    values = tuple(
        keyword.value for keyword in node.keywords if keyword.arg == keyword_name
    )
    return values[0] if len(values) == 1 else None


def _autoregister_registry_key_attr_name(
    parsed_module: ParsedModule,
    node: ast.ClassDef,
) -> str | None:
    assignments = _direct_class_assignments(node)
    explicit_key = _autoregister_constant_name(
        assignments.get("__registry_key__"), parsed_module
    )
    if explicit_key is not None:
        return explicit_key
    configured_key = _autoregister_constant_name(
        _registry_config_keyword(
            assignments.get("__registry_config__"),
            "key_attribute",
        ),
        parsed_module,
    )
    if configured_key is not None:
        return configured_key
    stable_key_axis = assignments.get("stable_key_axis")
    if (
        isinstance(stable_key_axis, ast.Name)
        and stable_key_axis.id == "__registry_key__"
    ):
        return stable_key_axis.id
    stable_key_name = _autoregister_constant_name(stable_key_axis, parsed_module)
    if stable_key_name is not None:
        return stable_key_name
    return _registry_family_key_attr_name(assignments.get("__registry_family__"))


def _autoregister_registry_projection_names(
    node: ast.ClassDef,
) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        if any(
            isinstance(subnode, ast.Attribute)
            and subnode.attr in {"__registry__", "_registry", "registry"}
            for subnode in ast.walk(method)
        )
    )


def _is_classmethod(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _terminal_reference_name(decorator) == "classmethod"
        for decorator in method.decorator_list
    )


def _method_references_cls_registry(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return RegistryLookupShape.references_registry(method)


def _keyed_registry_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        if _is_classmethod(method) and _method_references_cls_registry(method)
    )


def _keyed_registry_reverse_lookup_method_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(
        method.name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
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
) -> tuple[CompactRepeatedKeyedFamilyRoot, ...]:
    roots: list[CompactRepeatedKeyedFamilyRoot] = []
    for node in parsed_module.module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if "AutoRegisterByClassVar" not in {
            terminal_name
            for base in node.bases
            if (terminal_name := _terminal_reference_name(base)) is not None
        }:
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
            for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
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
                family_base_name="AutoRegisterByClassVar",
                registry_key_attr_name=registry_key_attr_name,
                lookup_method_name=lookup_method.name,
                lookup_style=lookup_shape.style,
                error_type_name=lookup_shape.error_type_name,
                abstract_hook_names=tuple(
                    method.name
                    for method in node.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if any(
                        _terminal_reference_name(decorator) == "abstractmethod"
                        for decorator in method.decorator_list
                    )
                ),
            )
        )
    return tuple(roots)


@dataclass
class _CompactAutoRegisterFunctionReferenceBuilder:
    qualname: str
    receiver_attribute_refs: set[tuple[str, str]]


def _compact_autoregister_reference_projection(
    builders: tuple[_CompactAutoRegisterFunctionReferenceBuilder, ...],
) -> CompactAutoRegisterReferenceIndex | None:
    consumer_builders = tuple(
        builder for builder in builders if builder.receiver_attribute_refs
    )
    if not consumer_builders:
        return None
    receiver_names = sorted_tuple(
        {
            receiver_name
            for builder in consumer_builders
            for receiver_name, _attr_name in builder.receiver_attribute_refs
        }
    )
    attribute_names = sorted_tuple(
        {
            attr_name
            for builder in consumer_builders
            for _receiver_name, attr_name in builder.receiver_attribute_refs
        }
    )
    receiver_indexes = {name: index for index, name in enumerate(receiver_names)}
    attribute_indexes = {name: index for index, name in enumerate(attribute_names)}
    return CompactAutoRegisterReferenceIndex(
        function_qualnames=tuple(builder.qualname for builder in consumer_builders),
        receiver_names=receiver_names,
        attribute_names=attribute_names,
        encoded_edges=";".join(
            f"{function_index},{receiver_index},{attribute_index}"
            for function_index, receiver_index, attribute_index in sorted(
                {
                    (
                        function_index,
                        receiver_indexes[receiver_name],
                        attribute_indexes[attr_name],
                    )
                    for function_index, builder in enumerate(consumer_builders)
                    for receiver_name, attr_name in builder.receiver_attribute_refs
                }
            )
        ),
    )


@lru_cache(maxsize=None)
def _compact_class_syntax_facets(
    parsed_module: ParsedModule,
    *,
    collect_autoregister: bool = True,
) -> CompactClassSyntaxFacets:
    syntax_index = module_syntax_index(parsed_module.module)
    file_path = parsed_module.file_path
    collect_autoregister = (
        collect_autoregister
        and not PythonSourcePathPolicy.is_test_path(parsed_module.path)
    )
    builders_by_function_id = (
        {
            id(function): _CompactAutoRegisterFunctionReferenceBuilder(
                qualname=qualname,
                receiver_attribute_refs=set(),
            )
            for qualname, function in syntax_index.named_functions
        }
        if collect_autoregister
        else {}
    )
    function_bindings_by_id = {
        id(function): LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(function.body)
        | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(function)
        for _qualname, function in syntax_index.named_functions
    }
    module_binds_type = "type" in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        parsed_module.module.body
    )
    active_function_ids_by_scope = tuple(
        tuple(
            id(syntax_index.depth_first_nodes[function_index])
            for function_index in scope.function_node_indices
        )
        for scope in syntax_index.scopes
    )
    active_builders_by_scope = tuple(
        (
            tuple(
                builders_by_function_id[function_id]
                for function_id in active_function_ids
            )
            if builders_by_function_id
            else ()
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_binds_type = tuple(
        module_binds_type
        or any(
            "type" in function_bindings_by_id[function_id]
            for function_id in active_function_ids
        )
        for active_function_ids in active_function_ids_by_scope
    )
    scope_qualnames = tuple(".".join(scope.names) for scope in syntax_index.scopes)
    branch_site_counts_by_function_id: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    case_names_by_function_and_key: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    exact_type_guards: list[CompactExactTypeGuard] = []
    indices_by_type = syntax_index.node_indices_by_type

    if builders_by_function_id:
        for node_index in indices_by_type.get(ast.Attribute, ()):
            node = syntax_index.depth_first_nodes[node_index]
            if not isinstance(node.value, ast.Name):
                continue
            receiver_reference = (node.value.id, node.attr)
            for builder in active_builders_by_scope[syntax_index.scope_ids[node_index]]:
                builder.receiver_attribute_refs.add(receiver_reference)

    for node_index in indices_by_type.get(ast.If, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        for key_type_name, case_names in _enum_member_refs_by_key_type(
            node.test
        ).items():
            branch_site_counts_by_function_id[active_function_id][key_type_name] += 1
            case_names_by_function_and_key[active_function_id][key_type_name].update(
                case_names
            )

    for node_index in indices_by_type.get(ast.Match, ()):
        active_function_ids = active_function_ids_by_scope[
            syntax_index.scope_ids[node_index]
        ]
        if not active_function_ids:
            continue
        active_function_id = active_function_ids[-1]
        node = syntax_index.depth_first_nodes[node_index]
        refs_by_key: dict[str, set[str]] = defaultdict(set)
        for case in node.cases:
            for key_type_name, case_names in _enum_member_refs_by_key_type(
                case.pattern
            ).items():
                refs_by_key[key_type_name].update(case_names)
            if case.guard is not None:
                for key_type_name, case_names in _enum_member_refs_by_key_type(
                    case.guard
                ).items():
                    refs_by_key[key_type_name].update(case_names)
        for key_type_name, case_names in refs_by_key.items():
            branch_site_counts_by_function_id[active_function_id][key_type_name] += 1
            case_names_by_function_and_key[active_function_id][key_type_name].update(
                case_names
            )

    for node_index in merge(
        indices_by_type.get(ast.If, ()),
        indices_by_type.get(ast.Assert, ()),
    ):
        scope_id = syntax_index.scope_ids[node_index]
        if not active_function_ids_by_scope[scope_id]:
            continue
        node = syntax_index.depth_first_nodes[node_index]
        predicate = _exact_type_predicate(node.test)
        if predicate is None:
            continue
        if isinstance(node, ast.If):
            matches_exact_type_when_true = predicate[2]
            rejects_descendants = (
                not matches_exact_type_when_true and _fail_loud_block(node.body)
            ) or (matches_exact_type_when_true and _fail_loud_block(node.orelse))
            if not rejects_descendants:
                continue
        elif not predicate[2]:
            continue
        if scope_binds_type[scope_id]:
            continue
        subject, type_reference, matches_exact_type_when_true, expression = predicate
        reference_node = ClassSymbolResolutionAuthority.reference_node(type_reference)
        parts = AstExpressionProjection.attribute_chain(reference_node)
        if parts is None:
            continue
        exact_type_guards.append(
            CompactExactTypeGuard(
                file_path=file_path,
                line=node.lineno,
                qualname=scope_qualnames[scope_id],
                subject_expression=ast.unparse(subject),
                type_reference_expression=ast.unparse(type_reference),
                type_reference_parts=parts,
                matches_exact_type_when_true=matches_exact_type_when_true,
                expression=expression,
            )
        )

    builders = (
        tuple(
            builders_by_function_id[id(function)]
            for _qualname, function in syntax_index.named_functions
        )
        if collect_autoregister
        else ()
    )
    autoregister_index = _compact_autoregister_reference_projection(builders)
    closed_axis_functions: list[CompactClosedAxisBranchFunction] = []
    for qualname, function in syntax_index.named_functions:
        function_id = id(function)
        axes = tuple(
            CompactClosedAxisBranchFact(
                key_type_name=key_type_name,
                branch_site_count=branch_site_count,
                case_names=sorted_tuple(
                    case_names_by_function_and_key[function_id][key_type_name]
                ),
            )
            for key_type_name, branch_site_count in sorted(
                branch_site_counts_by_function_id[function_id].items()
            )
        )
        if axes:
            closed_axis_functions.append(
                CompactClosedAxisBranchFunction(
                    file_path=file_path,
                    line=function.lineno,
                    qualname=qualname,
                    axes=axes,
                )
            )
    return CompactClassSyntaxFacets(
        autoregister_reference_index=autoregister_index,
        closed_axis_branch_functions=tuple(closed_axis_functions),
        exact_type_guards=tuple(exact_type_guards),
    )


def _registration_authority_base_name(base_name: str) -> bool:
    tokens = frozenset(
        token.lower()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
            base_name,
        )
        if token
    )
    return bool(
        tokens & {"autoregister", "registered", "registry"}
        or (
            "registration" in tokens
            and bool(tokens & {"authority", "base", "family", "meta", "root"})
        )
        or ("stable" in tokens and bool(tokens & {"axis", "key"}))
        or ("key" in tokens and "family" in tokens)
        or (
            "nominal" in tokens
            and "base" in tokens
            and bool(tokens & {"axis", "family", "formula", "policy"})
        )
    )


def _declares_autoregister_meta(node: ast.ClassDef) -> bool:
    return any(
        terminal_name == "AutoRegisterMeta"
        or terminal_name.endswith("AutoRegisterMeta")
        for keyword in node.keywords
        if keyword.arg == "metaclass"
        if (terminal_name := _terminal_reference_name(keyword.value)) is not None
    )


def _is_registration_authority(node: ast.ClassDef) -> bool:
    assignments = _direct_class_assignments(node)
    inherits_named_authority = any(
        _registration_authority_base_name(terminal_name)
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    )
    declares_named_authority = (
        "AutoRegister" in node.name
        or "Registered" in node.name
        or node.name.endswith("KeyFamily")
        or _registration_authority_base_name(node.name)
    )
    return bool(
        _declares_autoregister_meta(node)
        or inherits_named_authority
        or declares_named_authority
        or ("__registry__" in assignments and "__registry_key__" in assignments)
        or "stable_key_axis" in assignments
    )


def _is_abstract_class(node: ast.ClassDef) -> bool:
    if {"ABC", "ABCMeta"} & {
        terminal_name
        for base in node.bases
        if (terminal_name := _terminal_reference_name(base)) is not None
    }:
        return True
    return any(
        _terminal_reference_name(decorator) == "abstractmethod"
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in statement.decorator_list
    )


def _class_scope_qualified_import_name(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    reference: ast.AST,
    preceding_class_bound_names: frozenset[str],
) -> str | None:
    parts = AstExpressionProjection.attribute_chain(
        ClassSymbolResolutionAuthority.reference_node(reference)
    )
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
    if statement.value is None or isinstance(statement.value, ast.Constant):
        return CompactDataclassFieldRole.STORED_INIT
    if not isinstance(statement.value, ast.Call):
        return CompactDataclassFieldRole.UNRESOLVED
    qualified_default_factory = _class_scope_qualified_import_name(
        module_binding_snapshot,
        class_aliases,
        statement.value.func,
        preceding_class_bound_names,
    )
    default_factory_declaration = DataclassRuntimeDeclaration.for_qualified_name(
        qualified_default_factory
    )
    if (
        default_factory_declaration is None
        or not default_factory_declaration.is_field_factory
    ):
        return CompactDataclassFieldRole.UNRESOLVED
    if statement.value.args or any(
        keyword.arg is None for keyword in statement.value.keywords
    ):
        return CompactDataclassFieldRole.UNRESOLVED
    init_values = tuple(
        keyword.value for keyword in statement.value.keywords if keyword.arg == "init"
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
) -> tuple[CompactDataclassFieldDeclaration, ...]:
    fields: list[CompactDataclassFieldDeclaration] = []
    preceding_bound_names: set[str] = set()
    class_aliases: dict[str, str] = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            fields.append(
                CompactDataclassFieldDeclaration(
                    name=statement.target.id,
                    line=statement.lineno,
                    role=_dataclass_field_role(
                        module_binding_snapshot,
                        class_aliases,
                        statement,
                        frozenset(preceding_bound_names),
                    ),
                )
            )
        direct_aliases: dict[str, str] = {}
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            alias_name = _single_assignment_target_name(statement)
            if alias_name is not None and statement.value is not None:
                qualified_value = _class_scope_qualified_import_name(
                    module_binding_snapshot,
                    class_aliases,
                    statement.value,
                    frozenset(preceding_bound_names),
                )
                if qualified_value is not None:
                    direct_aliases[alias_name] = qualified_value
        statement_bound_names = _direct_statement_bound_names(statement)
        preceding_bound_names.update(statement_bound_names)
        for bound_name in statement_bound_names:
            class_aliases.pop(bound_name, None)
        class_aliases.update(direct_aliases)
    return tuple(fields)


def _dataclass_decorator_failures(
    decorator: ast.expr,
) -> tuple[CompactProductDeclarationFailure, ...]:
    if not isinstance(decorator, ast.Call):
        return ()
    violations: list[CompactProductAuthorityViolation] = []
    keyword_names = tuple(keyword.arg for keyword in decorator.keywords)
    if decorator.args or any(name is None for name in keyword_names):
        violations.append(CompactProductAuthorityViolation.DYNAMIC_DATACLASS_OPTIONS)
    init_values = tuple(
        keyword.value for keyword in decorator.keywords if keyword.arg == "init"
    )
    if len(init_values) > 1 or (
        init_values
        and (
            not isinstance(init_values[0], ast.Constant)
            or not isinstance(init_values[0].value, bool)
        )
    ):
        violations.append(CompactProductAuthorityViolation.DYNAMIC_DATACLASS_OPTIONS)
    elif init_values and not init_values[0].value:
        violations.append(CompactProductAuthorityViolation.GENERATED_INIT_DISABLED)
    return tuple(
        CompactProductDeclarationFailure(decorator.lineno, violation)
        for violation in dict.fromkeys(violations)
    )


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
            called_parts = AstExpressionProjection.attribute_chain(child.func)
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
                and AstExpressionProjection.attribute_chain(child.func.value.func)
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
) -> CompactDataclassDeclaration | None:
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
        return None

    fields = _dataclass_fields(module_binding_snapshot, node)
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


def _compact_class_member_declarations(
    node: ast.ClassDef,
) -> tuple[CompactClassMemberDeclaration, ...]:
    declarations: list[CompactClassMemberDeclaration] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign):
            targets = (statement.target,)
            value = statement.value
            annotation = statement.annotation
        elif isinstance(statement, ast.Assign):
            targets = tuple(statement.targets)
            value = statement.value
            annotation = None
        else:
            continue
        constructor_name = (
            _terminal_reference_name(value.func)
            if isinstance(value, ast.Call)
            else None
        )
        constructor_keyword_names = (
            sorted_tuple(
                keyword.arg for keyword in value.keywords if keyword.arg is not None
            )
            if isinstance(value, ast.Call) and constructor_name is not None
            else ()
        )
        declarations.extend(
            CompactClassMemberDeclaration(
                name=target.id,
                line=statement.lineno,
                expression=ast.unparse(value) if value is not None else None,
                constant_string=(
                    value.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                    else None
                ),
                value_is_none_literal=(
                    isinstance(value, ast.Constant) and value.value is None
                ),
                constructor_name=constructor_name,
                constructor_keyword_names=constructor_keyword_names,
                annotation_expression=(
                    ast.unparse(annotation) if annotation is not None else None
                ),
            )
            for target in targets
            if isinstance(target, ast.Name)
        )
    return tuple(declarations)


def _compact_sorted_key_calls(
    parsed_module: ParsedModule,
) -> tuple[CompactSortedKeyCall, ...]:
    calls: list[CompactSortedKeyCall] = []
    for node in _walk_nodes(parsed_module.module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
        ):
            continue
        registry_owner_names = sorted_tuple(
            {
                child.value.id
                for argument in node.args
                for child in ast.walk(argument)
                if isinstance(child, ast.Attribute)
                and child.attr == "__registry__"
                and isinstance(child.value, ast.Name)
            }
        )
        key_attribute_names: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg != "key" or keyword.value is None:
                continue
            key_attribute_names.update(
                child.attr
                for child in ast.walk(keyword.value)
                if isinstance(child, ast.Attribute)
            )
            for child in ast.walk(keyword.value):
                if not (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "attrgetter"
                ):
                    continue
                key_attribute_names.update(
                    argument.value
                    for argument in child.args
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                )
        if key_attribute_names:
            calls.append(
                CompactSortedKeyCall(
                    file_path=parsed_module.file_path,
                    line=node.lineno,
                    registry_owner_names=registry_owner_names,
                    key_attribute_names=sorted_tuple(key_attribute_names),
                )
            )
    return tuple(calls)


def _terminal_reference_name(node: ast.AST) -> str | None:
    parts = AstExpressionProjection.attribute_chain(
        ClassSymbolResolutionAuthority.reference_node(node)
    )
    return None if parts is None else parts[-1]


@dataclass(frozen=True)
class CompactClassFamilyIndexBuilder:
    projections: tuple[CompactModuleClassProjection, ...]

    def build(self) -> CompactClassFamilyIndex:
        records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                (
                    record
                    for projection in self.projections
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
        return CompactClassFamilyIndex(classes_by_symbol=classes_by_symbol)

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
        if reference.root_binding is not None:
            if reference.permits_root_relative_resolution:
                return unique_symbols_by_suffix.root_relative_match(candidate)
            return None
        module_local = ".".join((module_name, *reference.source_parts))
        if module_local in known_symbols:
            return module_local
        return None

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


def build_compact_class_family_index(
    projections: tuple[CompactModuleClassProjection, ...],
) -> CompactClassFamilyIndex:
    """Build an exact inheritance graph from AST-free per-module facts."""

    return CompactClassFamilyIndexBuilder(projections).build()


@dataclass(frozen=True)
class CompactClassReferenceResolver:
    import_aliases_by_module_name: dict[str, dict[str, str]]
    known_symbols: dict[str, CompactIndexedClass]
    unique_symbols_by_name: dict[str, str]
    unique_symbols_by_suffix: _UniqueKnownSymbolSuffixIndex

    @classmethod
    def from_index(
        cls,
        projections: tuple[CompactModuleClassProjection, ...],
        class_index: CompactClassFamilyIndex,
    ) -> "CompactClassReferenceResolver":
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
class ClassSymbolResolutionAuthority:
    """Resolve AST name chains to indexed class symbols under an explicit policy."""

    NOMINAL_FAMILY_NEUTRAL_BASE_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"ABC", "Generic", "Protocol", "object"}
    )

    parsed_module: ParsedModule
    import_aliases: dict[str, str]
    known_symbols: frozenset[str]
    unique_symbols_by_name: dict[str, str]
    allow_unique_unqualified: bool

    def symbol_for_node(self, node: ast.AST) -> str | None:
        parts = AstExpressionProjection.attribute_chain(self.reference_node(node))
        if parts is None:
            return None
        alias_symbol = self._import_alias_symbol(parts)
        if alias_symbol is not None:
            return alias_symbol
        module_local_symbol = self._module_local_symbol(parts)
        if module_local_symbol is not None:
            return module_local_symbol
        if self.allow_unique_unqualified:
            return self._unique_unqualified_symbol(parts)
        return None

    @staticmethod
    def reference_node(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Subscript):
            return node.value
        return node

    def _import_alias_symbol(self, parts: tuple[str, ...]) -> str | None:
        first, *rest = parts
        alias_target = self.import_aliases.get(first)
        if alias_target is None:
            return None
        candidate = ".".join((alias_target, *rest)) if rest else alias_target
        if candidate in self.known_symbols:
            return candidate
        # A scan may start at a package subdirectory, making indexed symbols
        # source-root-relative while imports retain their full package prefix.
        # Resolve only a unique suffix match so unrelated same-named classes do
        # not create a speculative inheritance edge.
        unique_symbol_by_suffix = _unique_known_symbol_by_suffix(self.known_symbols)
        return unique_symbol_by_suffix.root_relative_match(candidate)

    def _module_local_symbol(self, parts: tuple[str, ...]) -> str | None:
        candidate = ".".join((self.parsed_module.module_name, *parts))
        if candidate in self.known_symbols:
            return candidate
        return None

    def _unique_unqualified_symbol(self, parts: tuple[str, ...]) -> str | None:
        if len(parts) != 1:
            return None
        return self.unique_symbols_by_name.get(parts[0])

    @classmethod
    def declared_base_name(cls, node: ast.AST) -> str | None:
        reference_node = cls.reference_node(node)
        if AstExpressionProjection.attribute_chain(reference_node) is None:
            return None
        return ast.unparse(reference_node)

    @classmethod
    def establishes_nominal_family(cls, declared_base_name: str) -> bool:
        """Return whether a base declaration can own a domain class family."""

        return (
            declared_base_name.rsplit(".", 1)[-1]
            not in cls.NOMINAL_FAMILY_NEUTRAL_BASE_NAMES
        )


@dataclass(frozen=True)
class ModuleClassReferenceResolver:
    """Resolve class references in expression syntax against a class index."""

    parsed_module: ParsedModule
    class_index: ClassFamilyIndex

    @cached_property
    def known_symbols(self) -> frozenset[str]:
        return self.class_index.known_symbols

    @cached_property
    def unique_symbols_by_name(self) -> dict[str, str]:
        return self.class_index.unique_symbols_by_name

    @cached_property
    def import_aliases(self) -> dict[str, str]:
        return _module_import_aliases(self.parsed_module)

    @cached_property
    def constructor_assignment_symbols(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for statement in self.parsed_module.module.body:
            if not isinstance(statement, ast.Assign | ast.AnnAssign):
                continue
            target_name = _single_assignment_target_name(statement)
            if target_name is None:
                continue
            value = statement.value
            if value is None:
                continue
            symbol = self._direct_constructor_symbol(value)
            if symbol is not None:
                assignments[target_name] = symbol
        return assignments

    @cached_property
    def reference_resolution(self) -> ClassSymbolResolutionAuthority:
        return ClassSymbolResolutionAuthority(
            parsed_module=self.parsed_module,
            import_aliases=self.import_aliases,
            known_symbols=self.known_symbols,
            unique_symbols_by_name=self.unique_symbols_by_name,
            allow_unique_unqualified=False,
        )

    def symbols_for_node(self, node: ast.AST) -> tuple[str, ...]:
        collector = ClassReferenceSymbolCollector(self)
        collector.visit(node)
        return sorted_tuple(collector.symbols)

    def symbol_for_reference(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            return self._direct_constructor_symbol(node)
        if isinstance(node, ast.Name):
            constructor_symbol = self.constructor_assignment_symbols.get(node.id)
            if constructor_symbol is not None:
                return constructor_symbol
        return self.reference_resolution.symbol_for_node(node)

    def _direct_constructor_symbol(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        return self.reference_resolution.symbol_for_node(node.func)


class ClassReferenceSymbolCollector(ast.NodeVisitor):
    """Collect expression nodes that reference classes without counting members."""

    def __init__(self, resolver: ModuleClassReferenceResolver) -> None:
        self.resolver = resolver
        self.symbols: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        self._add_symbol(self.resolver._direct_constructor_symbol(node))
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_symbol(self.resolver.reference_resolution.symbol_for_node(node))

    def visit_Name(self, node: ast.Name) -> None:
        self._add_symbol(self.resolver.symbol_for_reference(node))

    def _add_symbol(self, symbol: str | None) -> None:
        if symbol is not None:
            self.symbols.add(symbol)


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


def build_class_family_index(modules: list[ParsedModule]) -> ClassFamilyIndex:
    return _build_class_family_index_cached(tuple(modules))


def overlay_class_family_index(
    base_index: ClassFamilyIndex,
    changed_modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    """Rebuild a class index by replacing changed-module class records."""

    changed_path_texts = _resolved_module_path_texts(changed_modules)
    unchanged_records = base_index.class_records_excluding_files(changed_path_texts)
    return ClassFamilyIndexBuilder(
        changed_modules,
        base_records=unchanged_records,
    ).build()


@lru_cache(maxsize=None)
def _build_class_family_index_cached(
    modules: tuple[ParsedModule, ...],
) -> ClassFamilyIndex:
    return ClassFamilyIndexBuilder(modules).build()


@dataclass(frozen=True)
class ClassFamilyIndexBuilder:
    modules: tuple[ParsedModule, ...]
    base_records: tuple[IndexedClass, ...] = ()

    def build(self) -> ClassFamilyIndex:
        class_records = tuple(
            UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                (*self.base_records, *self.module_class_records()),
                lambda record: record.symbol,
            ).values()
        )
        known_symbols = frozenset(record.symbol for record in class_records)
        classes_by_symbol = {
            record.symbol: self.resolved_record(record, known_symbols)
            for record in class_records
        }
        return ClassFamilyIndex(classes_by_symbol=classes_by_symbol)

    def module_class_records(self) -> tuple[IndexedClass, ...]:
        records: list[IndexedClass] = []
        for parsed_module in self.modules:
            indexed_class_nodes = iter_class_definitions(
                list(parsed_module.module.body)
            )
            binding_snapshots = _module_nominal_binding_snapshots(
                ModuleNominalBindingAuthority(parsed_module),
                tuple(node.lineno for _qualname, node in indexed_class_nodes),
                include_final=False,
            )
            for qualname, node in indexed_class_nodes:
                records.append(
                    IndexedClass.from_parsed_class(
                        parsed_module,
                        qualname,
                        node,
                        binding_snapshots[node.lineno],
                    )
                )
        return tuple(records)

    def resolved_record(
        self,
        record: IndexedClass,
        known_symbols: frozenset[str],
    ) -> IndexedClass:
        parsed_module = self.parsed_module_by_name.get(record.module_name)
        if parsed_module is None:
            return self.base_record_with_current_bases(record, known_symbols)
        base_references = _compact_base_references(
            record.node,
            ModuleNominalBindingAuthority(parsed_module).snapshot_before(record.line),
        )
        unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
        return record.with_resolved_base_symbols(
            tuple(
                resolved
                for reference in base_references
                if (
                    resolved := CompactClassFamilyIndexBuilder._resolved_bound_symbol(
                        reference,
                        record.module_name,
                        known_symbols,
                        unique_symbols_by_suffix,
                    )
                )
                is not None
            )
        )

    @cached_property
    def parsed_module_by_name(self) -> dict[str, ParsedModule]:
        return {module.module_name: module for module in self.modules}

    @staticmethod
    def base_record_with_current_bases(
        record: IndexedClass,
        known_symbols: frozenset[str],
    ) -> IndexedClass:
        return record.with_resolved_base_symbols(
            tuple(
                base_symbol
                for base_symbol in record.resolved_base_symbols
                if base_symbol in known_symbols
            )
        )


def _resolved_module_path_texts(modules: tuple[ParsedModule, ...]) -> frozenset[str]:
    return frozenset(module.resolved_file_path for module in modules)
