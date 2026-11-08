from __future__ import annotations

import ast
import copy
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from functools import cached_property
from typing import (
    ClassVar,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from .authority_extraction import SourceDerivedAuthorityProjectionOperation
from .class_base import ClassAuthorityReferenceProof
from .semantic_mirror import (
    FindingRecipeParts,
    MappingSemanticMirrorRecipeStrategy,
    PartsBackedMappingRecipeBuilder,
    RecipePartsT,
    SemanticMirrorImportBoundary,
    SemanticMirrorRecipeSeedLocations,
)
from ..assignment_projection import SingleAssignmentAndValueNameProjection
from ..ast_field_decoders import variadic_parameters
from ..ast_projection import (
    CallKeywords,
    ExpressionReference,
    StatementValue,
)
from ..ast_tools import (
    ROOT_NAME_PROJECTION,
    statements_without_docstring,
    walk_function_body_nodes,
)
from ..class_families.parsed_index import ModuleClassReferenceResolver
from ..codemod import (
    NominalConstructorCall,
    PatchTargetOperation,
)
from ..codemod_authority_claims import AstTargetAuthorityClaim
from ..codemod_declaration_source import PythonExpressionSourceFormatter
from ..codemod_module_declarations import ModuleSymbolTable
from ..codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
)
from ..codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from ..codemod_selector_models import SourceRewriteTarget
from ..codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextReplacement,
    SourceTextSpan,
)
from ..codemod_target_selectors import SourceIndexTargetSelector
from ..lexical_bindings import ImportBoundNameProjection
from ..lexical_scopes import FunctionBindingProjection
from ..models import (
    MappingMetrics,
    RefactorFinding,
)
from ..refactor_concepts import (
    ConstructorKwargCarrierProjectionConcept,
    DataclassPayloadProjectionConcept,
)
from ..semantic_descent import SemanticAuthorityKind
from ..semantic_match import Maybe
from ..source_index import (
    AstTargetDigest,
    AstTargetNodeKind,
)


@dataclass(frozen=True)
class ProductFieldValue:
    """One named product field and the expression assigned to it."""

    field_name: str
    value_node: ast.expr


@dataclass(frozen=True)
class ReturnFieldValue(ProductFieldValue):
    """One named return-product field and the expression assigned to it."""


@dataclass(frozen=True)
class ReturnDictFieldValue(ReturnFieldValue):
    """One string-key return-dict field and the expression assigned to it."""


@dataclass(frozen=True)
class FunctionProjectionTarget:
    """Common identity for a projection located inside one function or method."""

    source_path: str
    function_qualname: str

    @property
    def owner_qualname(self) -> str | None:
        """Return the enclosing nominal declaration, when this is a method."""

        owner_qualname, separator, _member_name = self.function_qualname.rpartition(".")
        return owner_qualname if separator else None


@dataclass(frozen=True)
class ResolvedFunctionProjectionTarget(FunctionProjectionTarget):
    """Uniquely resolved source-index function that contains a projection."""

    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @staticmethod
    def from_rewrite_target(
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> ResolvedFunctionProjectionTarget:
        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_function_like or not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            raise ValueError("Projection must target one exact function")
        return ResolvedFunctionProjectionTarget(
            source_path=target.file_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )

    @staticmethod
    def from_function_identity(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=function_qualname,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=context.source_index.target_by_id[target_ids[0]],
        )

    @staticmethod
    def from_source_line(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        line: int,
    ) -> ResolvedFunctionProjectionTarget | None:
        target = context.source_index.targets_by_file.smallest_enclosing_target(
            source_path,
            line,
            line,
        )
        if target is None:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=target,
        )

    @staticmethod
    def from_target(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        target: AstTargetDigest,
    ) -> ResolvedFunctionProjectionTarget | None:
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return ResolvedFunctionProjectionTarget(
            source_path=source_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )


@dataclass(frozen=True)
class FunctionReturnProjectionTarget(ResolvedFunctionProjectionTarget):
    """Uniquely resolved return statement inside a source-index function."""

    return_node: ast.Return

    @staticmethod
    def from_return_location(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
    ) -> FunctionReturnProjectionTarget | None:
        function = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
        )
        if function is None:
            return None
        matches = tuple(
            child
            for child in walk_function_body_nodes(function.node.body)
            if isinstance(child, ast.Return) and child.lineno == line
        )
        if len(matches) != 1:
            return None
        return FunctionReturnProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=matches[0],
        )


ProjectionTargetT = TypeVar("ProjectionTargetT", bound=FunctionProjectionTarget)


@dataclass(frozen=True)
class ReturnDictProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a return dict with named string-key fields."""

    dict_node: ast.Dict
    field_values: tuple[ReturnDictFieldValue, ...]


@dataclass(frozen=True)
class ReturnKeyValueSequenceFieldValue(ReturnFieldValue):
    """One ``("field", value)`` return-sequence item and its source element."""

    element_node: ast.Tuple | ast.List


@dataclass(frozen=True)
class ReturnKeyValueSequenceProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a returned sequence of string-key value pairs."""

    sequence_node: ast.Tuple | ast.List
    field_values: tuple[ReturnKeyValueSequenceFieldValue, ...]


ReturnCollectionProjectionTarget: TypeAlias = (
    ReturnDictProjectionTarget | ReturnKeyValueSequenceProjectionTarget
)


class ReturnDictFieldValueExtractor:
    """Shared extraction of selected string-key fields from return dictionaries."""

    finding: RefactorFinding

    def field_values(self, dict_node: ast.Dict) -> tuple[ReturnDictFieldValue, ...]:
        return ReturnDictProjectionTargetAuthority.field_values(
            dict_node,
            self.finding.metrics.plan_field_names,
        )

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        return ReturnDictProjectionTargetAuthority.string_key_value(node)


class ReturnDictProjectionTargetAuthority:
    """Resolve return-dict projection targets from source-index function facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Dict,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        if not isinstance(return_node.value, ast.Dict):
            return None
        dict_node = return_node.value
        field_values = cls.field_values(dict_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnDictProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            dict_node=dict_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        dict_node: ast.Dict,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnDictFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnDictFieldValue] = []
        for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
            field_name = cls.string_key_value(key_node)
            if field_name in selected_field_names:
                values.append(
                    ReturnDictFieldValue(
                        field_name=field_name,
                        value_node=value_node,
                    )
                )
        return tuple(values)

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


class ReturnKeyValueSequenceProjectionTargetAuthority:
    """Resolve returned ``("field", value)`` sequence projections from source facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        if not isinstance(return_node.value, ast.Tuple | ast.List):
            return None
        sequence_node = return_node.value
        field_values = cls.field_values(sequence_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnKeyValueSequenceProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            sequence_node=sequence_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        sequence_node: ast.Tuple | ast.List,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnKeyValueSequenceFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnKeyValueSequenceFieldValue] = []
        for element in sequence_node.elts:
            field_value = cls.field_value(element)
            if (
                field_value is not None
                and field_value.field_name in selected_field_names
            ):
                values.append(field_value)
        return tuple(values)

    @classmethod
    def field_value(
        cls,
        element: ast.expr,
    ) -> ReturnKeyValueSequenceFieldValue | None:
        if not isinstance(element, ast.Tuple | ast.List) or len(element.elts) != 2:
            return None
        key_node, value_node = element.elts
        field_name = ReturnDictProjectionTargetAuthority.string_key_value(key_node)
        if field_name is None:
            return None
        return ReturnKeyValueSequenceFieldValue(
            field_name=field_name,
            value_node=value_node,
            element_node=element,
        )


@dataclass(frozen=True)
class DataclassPayloadAuthorityTarget(ResolvedClassTarget):
    """Dataclass authority that owns projected payload field names."""

    @classmethod
    def from_rewrite_target(
        cls,
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> Self:
        authority = super().from_rewrite_target(context, target_reference)
        if "." in authority.qualname:
            raise ValueError("Dataclass projection authority must be top level")
        if not authority.is_dataclass:
            raise ValueError("Dataclass projection authority must be a dataclass")
        if not authority.field_names:
            raise ValueError("Dataclass projection authority has no direct fields")
        return authority

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.field_names_for_node(self.node)

    @property
    def field_annotations(self) -> tuple[tuple[str, str], ...]:
        """Project direct payload-field annotations in declaration order."""

        selected_names = frozenset(self.field_names)
        return tuple(
            (statement.target.id, ast.unparse(statement.annotation))
            for statement in self.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in selected_names
        )

    @property
    def is_dataclass(self) -> bool:
        return self.node_is_dataclass(self.node)

    @classmethod
    def node_is_dataclass(cls, node: ast.ClassDef) -> bool:
        return any(
            cls.decorator_name(decorator) == "dataclass"
            for decorator in node.decorator_list
        )

    @classmethod
    def decorator_name(cls, node: ast.expr) -> str | None:
        if isinstance(node, ast.Call):
            return cls.decorator_name(node.func)
        return ExpressionReference.decoded(node).terminal_name

    @staticmethod
    def field_names_for_node(node: ast.ClassDef) -> tuple[str, ...]:
        excluded_annotation_names = {"ClassVar", "InitVar", "KW_ONLY"}
        return tuple(
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and excluded_annotation_names.isdisjoint(
                child.id
                for child in ast.walk(statement.annotation)
                if isinstance(child, ast.Name)
            )
        )

    def family_defines_method(
        self,
        context: CodemodSelectorContext,
        method_name: str,
    ) -> bool:
        """Return whether this authority or an ancestor owns a method name."""

        authority_symbol = self.symbol(context)
        if authority_symbol is None:
            return True
        class_index = context.required_class_family_index
        return any(
            indexed_class is not None
            and any(
                isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
                and statement.name == method_name
                for statement in indexed_class.node.body
            )
            for symbol in (
                authority_symbol,
                *class_index.ancestor_symbols(authority_symbol),
            )
            for indexed_class in (class_index.class_for(symbol),)
        )

    def require_complete_owned_schema(
        self,
        context: CodemodSelectorContext,
    ) -> None:
        authority_symbol = self.symbol(context)
        if authority_symbol is None:
            raise ValueError("Dataclass projection authority has no nominal identity")
        class_index = context.required_class_family_index
        if any(
            (ancestor := class_index.class_for(ancestor_symbol)) is not None
            and self.node_is_dataclass(ancestor.node)
            for ancestor_symbol in class_index.ancestor_symbols(authority_symbol)
        ):
            raise ValueError(
                "Dataclass projection authority must own its complete field schema"
            )

    def require_transparent_direct_construction(self) -> None:
        """Require construction whose only behavior assigns declared fields."""

        if (
            self.node.bases
            or self.node.keywords
            or len(self.node.decorator_list) != 1
            or not self.has_generated_initializer()
        ):
            raise ValueError(
                "Dataclass constructor projection requires a generated direct "
                "initializer"
            )
        behavior_changing_methods = {
            "__getattr__",
            "__getattribute__",
            "__init__",
            "__post_init__",
            "__setattr__",
        }
        if any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and statement.name in behavior_changing_methods
            for statement in self.node.body
        ):
            raise ValueError(
                "Dataclass constructor projection requires behavior-free field "
                "construction"
            )
        if any(
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in self.field_names
            and isinstance(statement.value, ast.Call)
            and ExpressionReference.decoded(statement.value.func).terminal_name
            == "field"
            and self.call_keyword_bool(statement.value, "init", default=True)
            is not True
            for statement in self.node.body
        ):
            raise ValueError(
                "Dataclass constructor projection requires every authority field "
                "in the generated initializer"
            )

    def has_generated_initializer(self) -> bool:
        dataclass_decorators = tuple(
            decorator
            for decorator in self.node.decorator_list
            if self.decorator_name(decorator) == "dataclass"
        )
        if len(dataclass_decorators) != 1:
            return False
        decorator = dataclass_decorators[0]
        return not isinstance(decorator, ast.Call) or (
            self.call_keyword_bool(decorator, "init", default=True) is True
        )

    @staticmethod
    def call_keyword_bool(
        call: ast.Call,
        keyword_name: str,
        *,
        default: bool,
    ) -> bool | None:
        matches = tuple(
            keyword for keyword in call.keywords if keyword.arg == keyword_name
        )
        if not matches:
            return default
        if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant):
            return None
        value = matches[0].value.value
        return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class DataclassAuthorityReferenceProof:
    """Resolved dataclass authority identity at one source boundary."""

    reference: ClassAuthorityReferenceProof
    generated_import_source: str | None
    top_level_target_binding_is_nominal: bool

    @property
    def target_name(self) -> str:
        return self.reference.authority.name

    @property
    def target_symbol(self) -> str:
        return self.reference.authority_symbol

    @property
    def resolver(self) -> ModuleClassReferenceResolver:
        return self.reference.resolver

    @property
    def symbol_table(self) -> ModuleSymbolTable:
        return self.reference.symbol_table

    @classmethod
    def from_target(
        cls,
        context: CodemodSelectorContext,
        source_path: str,
        target: DataclassPayloadAuthorityTarget,
        generated_import_source: str | None,
    ) -> DataclassAuthorityReferenceProof | None:
        try:
            reference = ClassAuthorityReferenceProof.from_context(
                context,
                target,
                source_path,
            )
        except ValueError:
            return None
        return cls(
            reference=reference,
            generated_import_source=generated_import_source,
            top_level_target_binding_is_nominal=(
                cls.top_level_target_binding_is_nominal(
                    reference.symbol_table,
                    reference.projection_module.file_path,
                    target,
                )
            ),
        )

    def resolves(self, reference: ast.expr) -> bool:
        if (
            isinstance(reference, ast.Name)
            and reference.id in self.symbol_table.top_level_names
            and not self.top_level_target_binding_is_nominal
        ):
            return False
        if self.resolver.symbol_for_reference(reference) == self.target_symbol:
            return True
        return bool(
            self.generated_import_source is not None
            and isinstance(reference, ast.Name)
            and reference.id == self.target_name
            and reference.id not in self.symbol_table.available_names
        )

    @staticmethod
    def top_level_target_binding_is_nominal(
        symbol_table: ModuleSymbolTable,
        source_path: str,
        target: DataclassPayloadAuthorityTarget,
    ) -> bool:
        bindings = symbol_table.binding_statements(target.name)
        return bool(
            source_path == target.file_path
            and len(bindings) == 1
            and isinstance(bindings[0], ast.ClassDef)
            and bindings[0].name == target.name
        )

    def annotation_resolves(self, annotation: ast.expr) -> bool:
        reference = self.annotation_reference(annotation)
        return reference is not None and self.resolves(reference)

    @staticmethod
    def annotation_reference(annotation: ast.expr) -> ast.expr | None:
        if not (
            isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
        ):
            return annotation
        try:
            return ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None

    @classmethod
    def annotation_is_self(cls, annotation: ast.expr) -> bool:
        reference = cls.annotation_reference(annotation)
        return bool(
            (isinstance(reference, ast.Name) and reference.id == "Self")
            or (isinstance(reference, ast.Attribute) and reference.attr == "Self")
        )


@dataclass(frozen=True)
class DataclassInstanceFieldProjection:
    """Exhaustive declaration-ordered field reads from one stable instance."""

    owner_node: ast.expr

    @classmethod
    def from_field_values(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        field_values: tuple[ReturnFieldValue, ...],
    ) -> DataclassInstanceFieldProjection | None:
        if (
            not field_values
            or tuple(field.field_name for field in field_values)
            != authority.field_names
        ):
            return None
        owner_nodes: list[ast.expr] = []
        for field_value in field_values:
            value_node = field_value.value_node
            if (
                not isinstance(value_node, ast.Attribute)
                or value_node.attr != field_value.field_name
                or not cls.is_stable_owner_path(value_node.value)
            ):
                return None
            owner_nodes.append(value_node.value)
        owner_identity = ast.dump(owner_nodes[0], include_attributes=False)
        if any(
            ast.dump(owner_node, include_attributes=False) != owner_identity
            for owner_node in owner_nodes[1:]
        ):
            return None
        return cls(owner_node=owner_nodes[0])

    @classmethod
    def is_stable_owner_path(cls, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return True
        return isinstance(node, ast.Attribute) and cls.is_stable_owner_path(node.value)

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        proof = DataclassAuthorityReferenceProof.from_target(
            context,
            projection.source_path,
            authority,
            authority_import_source,
        )
        if proof is None:
            return False
        if isinstance(self.owner_node, ast.Name):
            return self.name_has_nominal_authority_type(
                self.owner_node.id,
                context,
                authority,
                projection,
                proof,
            )
        if not (
            isinstance(self.owner_node, ast.Attribute)
            and isinstance(self.owner_node.value, ast.Name)
            and self.owner_node.value.id == "self"
        ):
            return False
        enclosing_class = self.enclosing_class_node(context, projection)
        return enclosing_class is not None and any(
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == self.owner_node.attr
            and proof.annotation_resolves(statement.annotation)
            for statement in enclosing_class.body
        )

    def name_has_nominal_authority_type(
        self,
        name: str,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        if name == "self":
            return self.enclosing_class_is_authority(
                context,
                authority,
                projection,
            )
        arguments = projection.node.args
        if any(
            argument.arg == name
            and argument.annotation is not None
            and proof.annotation_resolves(argument.annotation)
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ):
            return True
        assignments = tuple(
            statement
            for statement in ast.walk(projection.node)
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
            )
            or (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
            )
        )
        if len(assignments) != 1:
            return False
        assignment = assignments[0]
        if isinstance(assignment, ast.AnnAssign):
            return proof.annotation_resolves(assignment.annotation)
        return self.call_constructs_authority(
            assignment.value,
            context,
            authority,
            projection,
            proof,
        )

    @classmethod
    def call_constructs_authority(
        cls,
        value: ast.expr,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        if not isinstance(value, ast.Call):
            return False
        if cls.unshadowed_reference_resolves(value.func, projection, proof):
            return True
        if not isinstance(value.func, ast.Attribute):
            return False
        return cls.unshadowed_reference_resolves(
            value.func.value,
            projection,
            proof,
        ) and value.func.attr in cls.authority_factory_method_names(context, authority)

    @staticmethod
    def unshadowed_reference_resolves(
        reference: ast.expr,
        projection: ReturnCollectionProjectionTarget,
        proof: DataclassAuthorityReferenceProof,
    ) -> bool:
        roots = ROOT_NAME_PROJECTION.root_names(reference)
        bindings = FunctionBindingProjection.from_function(projection.node)
        return roots.isdisjoint(bindings.local_names) and proof.resolves(reference)

    @classmethod
    def authority_factory_method_names(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
    ) -> frozenset[str]:
        proof = DataclassAuthorityReferenceProof.from_target(
            context,
            authority.file_path,
            authority,
            None,
        )
        if proof is None:
            return frozenset()
        return frozenset(
            statement.name
            for statement in authority.node.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            and any(
                ExpressionReference.decoded(decorator).terminal_name == "classmethod"
                for decorator in statement.decorator_list
            )
            and statement.returns is not None
            and (
                proof.annotation_is_self(statement.returns)
                or proof.annotation_resolves(statement.returns)
            )
        )

    @classmethod
    def enclosing_class_is_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnCollectionProjectionTarget,
    ) -> bool:
        enclosing_class = cls.enclosing_class_node(context, projection)
        if enclosing_class is None or projection.owner_qualname is None:
            return False
        class_symbol = context.required_class_family_index.symbol_for(
            file_path=projection.source_path,
            qualname=projection.owner_qualname,
        )
        return class_symbol is not None and class_symbol == authority.symbol(context)

    @staticmethod
    def enclosing_class_node(
        context: CodemodSelectorContext,
        projection: ReturnCollectionProjectionTarget,
    ) -> ast.ClassDef | None:
        if projection.owner_qualname is None:
            return None
        target_ids = SourceIndexTargetSelector(
            node_kinds=(AstTargetNodeKind.CLASS,),
            file_paths=(projection.source_path,),
            qualnames=(projection.owner_qualname,),
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        node = context.ast_target_nodes_by_id.get(target_ids[0])
        return node if isinstance(node, ast.ClassDef) else None


@dataclass(frozen=True)
class DataclassInstanceFieldRunProjection:
    """One contiguous exhaustive dict run read from a nominal instance."""

    instance: DataclassInstanceFieldProjection
    first_key_node: ast.expr
    last_value_node: ast.expr
    dict_node: ast.Dict
    comprehension_prefix: ClassVar[str] = "**"

    def replacement_span(self, geometry: SourceTextGeometry) -> SourceTextSpan | None:
        first = geometry.node_offsets(self.first_key_node)
        last = geometry.node_offsets(self.last_value_node)
        if first is None or last is None:
            return None
        return SourceTextSpan(start_offset=first[0], end_offset=last[1])

    @property
    def owner_node(self) -> ast.expr:
        return self.instance.owner_node

    @classmethod
    def from_targets(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> DataclassInstanceFieldRunProjection | None:
        instance = DataclassInstanceFieldProjection.from_field_values(
            authority,
            projection.field_values,
        )
        if instance is None:
            return None
        selected_value_ids = frozenset(
            id(field.value_node) for field in projection.field_values
        )
        matched_indices = tuple(
            index
            for index, value_node in enumerate(projection.dict_node.values)
            if id(value_node) in selected_value_ids
        )
        if not matched_indices or matched_indices != tuple(
            range(matched_indices[0], matched_indices[-1] + 1)
        ):
            return None
        first_key_node = projection.dict_node.keys[matched_indices[0]]
        if not isinstance(first_key_node, ast.expr):
            return None
        projection_type = (
            WholeDataclassPayloadProjection
            if len(projection.field_values) == len(projection.dict_node.values)
            else cls
        )
        return projection_type(
            instance=instance,
            first_key_node=first_key_node,
            last_value_node=projection.dict_node.values[matched_indices[-1]],
            dict_node=projection.dict_node,
        )

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        return self.instance.owner_has_nominal_authority_type(
            context,
            authority,
            projection,
            authority_import_source,
        )


@dataclass(frozen=True)
class WholeDataclassPayloadProjection(DataclassInstanceFieldRunProjection):
    """An exhaustive payload owns the entire dict expression, without a merge."""

    comprehension_prefix: ClassVar[str] = ""

    def replacement_span(self, geometry: SourceTextGeometry) -> SourceTextSpan | None:
        offsets = geometry.node_offsets(self.dict_node)
        if offsets is None:
            return None
        return SourceTextSpan(start_offset=offsets[0], end_offset=offsets[1])


@dataclass(frozen=True)
class DataclassKeyValueElementRunProjection:
    """One contiguous exhaustive pair run read from a nominal instance."""

    instance: DataclassInstanceFieldProjection
    first_element_node: ast.Tuple | ast.List
    last_element_node: ast.Tuple | ast.List

    @property
    def owner_node(self) -> ast.expr:
        return self.instance.owner_node

    @classmethod
    def from_targets(
        cls,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
    ) -> DataclassKeyValueElementRunProjection | None:
        instance = DataclassInstanceFieldProjection.from_field_values(
            authority,
            projection.field_values,
        )
        if instance is None:
            return None
        selected_element_ids = frozenset(
            id(field.element_node) for field in projection.field_values
        )
        matched_indices = tuple(
            index
            for index, element in enumerate(projection.sequence_node.elts)
            if id(element) in selected_element_ids
        )
        if not matched_indices or matched_indices != tuple(
            range(matched_indices[0], matched_indices[-1] + 1)
        ):
            return None
        first_element = projection.sequence_node.elts[matched_indices[0]]
        last_element = projection.sequence_node.elts[matched_indices[-1]]
        if not isinstance(first_element, ast.Tuple | ast.List) or not isinstance(
            last_element,
            ast.Tuple | ast.List,
        ):
            return None
        return cls(
            instance=instance,
            first_element_node=first_element,
            last_element_node=last_element,
        )

    def owner_has_nominal_authority_type(
        self,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
        authority_import_source: str | None,
    ) -> bool:
        return self.instance.owner_has_nominal_authority_type(
            context,
            authority,
            projection,
            authority_import_source,
        )


@dataclass(frozen=True)
class DataclassFieldNameCollectionProjectionTarget(ResolvedFunctionProjectionTarget):
    """One local collection that exhaustively names dataclass fields."""

    collection_node: ast.Tuple | ast.List

    @classmethod
    def candidates_from_function(
        cls,
        function: ResolvedFunctionProjectionTarget,
        authority: DataclassPayloadAuthorityTarget,
    ) -> tuple[DataclassFieldNameCollectionProjectionTarget, ...]:
        return tuple(
            cls(
                source_path=function.source_path,
                function_qualname=function.function_qualname,
                target=function.target,
                node=function.node,
                collection_node=collection,
            )
            for statement in walk_function_body_nodes(function.node.body)
            if (pair := SingleAssignmentAndValueNameProjection(statement).pair)
            is not None
            if isinstance((collection := pair[1]), ast.Tuple | ast.List)
            if cls.string_elements(collection) == authority.field_names
        )

    @classmethod
    def from_binding_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        binding_name: str,
        line: int,
        field_names: frozenset[str],
    ) -> DataclassFieldNameCollectionProjectionTarget | None:
        function = ResolvedFunctionProjectionTarget.from_source_line(
            context,
            source_path=source_path,
            line=line,
        )
        if function is None:
            return None
        collections = tuple(
            collection
            for statement in ast.walk(function.node)
            for collection in cls.bound_collection(statement, binding_name, line)
            if len(collection.elts) == len(field_names)
            and frozenset(cls.string_elements(collection)) == field_names
        )
        if len(collections) != 1:
            return None
        return cls(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            collection_node=collections[0],
        )

    @staticmethod
    def bound_collection(
        statement: ast.AST,
        binding_name: str,
        line: int,
    ) -> tuple[ast.Tuple | ast.List, ...]:
        if not isinstance(statement, ast.stmt) or statement.lineno != line:
            return ()
        pair = SingleAssignmentAndValueNameProjection(statement).pair
        if (
            pair is None
            or pair[0] != binding_name
            or not isinstance(pair[1], ast.Tuple | ast.List)
        ):
            return ()
        return (pair[1],)

    @staticmethod
    def string_elements(collection: ast.Tuple | ast.List) -> tuple[str, ...]:
        if not all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in collection.elts
        ):
            return ()
        return tuple(cast(ast.Constant, element).value for element in collection.elts)

    def derived_source(
        self,
        dataclasses_reference: DataclassesModuleReference,
        authority: DataclassPayloadAuthorityTarget,
    ) -> str:
        field_projection = (
            f"field.name for field in {dataclasses_reference.expression}.fields("
            f"{authority.name})"
        )
        if isinstance(self.collection_node, ast.Tuple):
            return f"tuple({field_projection})"
        return f"[{field_projection}]"

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.string_elements(self.collection_node)


@dataclass(frozen=True)
class DataclassesModuleReference:
    """Collision-checked module reference for public dataclass reflection."""

    expression: str
    import_source: str | None

    @classmethod
    def from_projection(
        cls,
        context: CodemodSelectorContext,
        projection: (
            ReturnCollectionProjectionTarget
            | DataclassFieldNameCollectionProjectionTarget
        ),
    ) -> DataclassesModuleReference | None:
        module = context.module_nodes_by_file_path.get(projection.source_path)
        source = context.sources_by_file_path.get(projection.source_path)
        if module is None or source is None:
            return None
        imported_aliases = tuple(
            ImportBoundNameProjection(statement).alias_bound_name(alias)
            for statement in module.body
            if isinstance(statement, ast.Import)
            for alias in statement.names
            if alias.name == "dataclasses"
        )
        if len(imported_aliases) > 1:
            return None
        expression = imported_aliases[0] if imported_aliases else "dataclasses"
        bindings = FunctionBindingProjection.from_function(projection.node)
        if expression in bindings.local_names:
            return None
        symbol_table = ModuleSymbolTable(
            file_path=projection.source_path,
            source=source,
            module=module,
        )
        if imported_aliases:
            return cls(expression=expression, import_source=None)
        if expression in symbol_table.available_names:
            return None
        return cls(expression=expression, import_source="import dataclasses")


class DataclassAuthorityMappingRecipeBuilder(
    PartsBackedMappingRecipeBuilder[RecipePartsT],
    Generic[ProjectionTargetT, RecipePartsT],
    ABC,
):
    """Shared seed-to-authority workflow for dataclass projection recipes."""

    def is_applicable(self) -> bool:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return False
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        if seed is None:
            return False
        resolved_target = self.resolved_authority_target(seed)
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        return (
            resolved_target is not None
            and self.is_dataclass_authority(resolved_target.node)
            and import_boundary is not None
            and self.projection_shape_is_applicable(
                seed,
                import_boundary.projection_path,
            )
        )

    @cached_property
    def parts(self) -> RecipePartsT | None:
        return (
            Maybe.of(SemanticMirrorRecipeSeedLocations.from_finding(self.finding))
            .project(self.parts_from_seed)
            .unwrap_or_none()
        )

    def parts_from_seed(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> RecipePartsT | None:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return None
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        if import_boundary is None:
            return None
        if import_boundary.import_would_create_cycle(self):
            return None
        authority = self.authority_target(seed)
        projection = self.projection_target(seed, import_boundary.projection_path)
        return (
            Maybe.of((authority, projection))
            .filter(lambda row: row[0] is not None and row[1] is not None)
            .project(lambda row: self.recipe_parts(row[0], row[1]))
            .unwrap_or_none()
        )

    def authority_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> DataclassPayloadAuthorityTarget | None:
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return (
            Maybe.of(self.resolved_authority_target(seed))
            .filter(
                lambda resolved_target: (self.resolved_target_is_exhaustive_dataclass)(
                    resolved_target,
                    field_names,
                )
            )
            .map(
                lambda resolved_target: DataclassPayloadAuthorityTarget(
                    target=resolved_target.target,
                    node=resolved_target.node,
                )
            )
            .unwrap_or_none()
        )

    def resolved_authority_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
    ) -> ResolvedClassTarget | None:
        authority_name = self.finding.metrics.plan_source_name
        if authority_name is None:
            return None
        return MappingSemanticMirrorRecipeStrategy.authority_class_target(
            self,
            seed.authority,
            authority_name,
        )

    def resolved_target_is_exhaustive_dataclass(
        self,
        resolved_target: ResolvedClassTarget,
        field_names: frozenset[str],
    ) -> bool:
        authority = DataclassPayloadAuthorityTarget(
            target=resolved_target.target,
            node=resolved_target.node,
        )
        if (
            not authority.is_dataclass
            or field_names != frozenset(authority.field_names)
            or not authority.field_names
        ):
            return False
        try:
            authority.require_complete_owned_schema(self)
        except ValueError:
            return False
        return True

    @abstractmethod
    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        """Return whether this leaf owns the projection syntax in the finding."""

        raise NotImplementedError

    @abstractmethod
    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ProjectionTargetT | None:
        raise NotImplementedError

    @abstractmethod
    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ProjectionTargetT,
    ) -> RecipePartsT | None:
        raise NotImplementedError

    @staticmethod
    def is_dataclass_authority(node: ast.ClassDef) -> bool:
        return DataclassPayloadAuthorityTarget.node_is_dataclass(node)


@dataclass(frozen=True)
class DataclassProjectionBoundary:
    """Exact dataclass authority and projection-function source boundary."""

    authority: DataclassPayloadAuthorityTarget
    function: ResolvedFunctionProjectionTarget
    authority_import_source: str | None

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassProjectionBoundary:
        authority = DataclassPayloadAuthorityTarget.from_rewrite_target(
            context,
            authority_reference,
        )
        authority.require_complete_owned_schema(context)
        function = ResolvedFunctionProjectionTarget.from_rewrite_target(
            context,
            projection_reference,
        )
        reference = ClassAuthorityReferenceProof.from_context(
            context,
            authority,
            function.source_path,
        )
        return cls(
            authority=authority,
            function=function,
            authority_import_source=reference.required_import_source(context),
        )


@dataclass(frozen=True)
class SourceDerivedDataclassProjection(Generic[ProjectionTargetT]):
    """Current-source proof and edits derived for one dataclass projection."""

    authority: DataclassPayloadAuthorityTarget
    projection: ProjectionTargetT
    source_replacement: SourceTextReplacement
    import_sources: tuple[str, ...]


@dataclass(frozen=True)
class DataclassPayloadProjectionCandidate:
    """One exhaustive return-dict projection proved against a dataclass."""

    projection: ReturnDictProjectionTarget
    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassPayloadProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnDictProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-dict projection."""

    field_run: DataclassInstanceFieldRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassPayloadProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary.authority,
                    boundary.function,
                    node,
                    boundary.authority_import_source,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-dict projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=tuple(
                import_source
                for import_source in (
                    boundary.authority_import_source,
                    candidate.dataclasses_reference.import_source,
                )
                if import_source is not None
            ),
            field_run=candidate.field_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        authority_import_source: str | None,
    ) -> DataclassPayloadProjectionCandidate | None:
        projection = ReturnDictProjectionTargetAuthority.from_return_node(
            function,
            return_node,
            authority.field_names,
        )
        if projection is None:
            return None
        field_run = DataclassInstanceFieldRunProjection.from_targets(
            authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            field_run is None
            or dataclasses_reference is None
            or not field_run.owner_has_nominal_authority_type(
                context,
                authority,
                projection,
                authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            authority,
            projection,
            field_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassPayloadProjectionCandidate(
            projection=projection,
            field_run=field_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
        field_run: DataclassInstanceFieldRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        replacement_span = field_run.replacement_span(geometry)
        owner_source = geometry.segment_for_node(field_run.owner_node)
        if replacement_span is None or owner_source is None:
            return None
        if replacement_span.contains_comment(source):
            return None
        indentation = " " * field_run.first_key_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        replacement_source = (
            f"{field_run.comprehension_prefix}{{\n"
            f"{continuation_indentation}field.name: getattr(\n"
            f"{nested_indentation}{owner_source},\n"
            f"{nested_indentation}field.name,\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation}}}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True)
class DataclassFieldNameCollectionProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass field-name collection."""

    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassFieldNameCollectionProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        if boundary.authority_import_source is not None:
            raise ValueError(
                "Dataclass field-name projection requires an existing runtime "
                "authority reference"
            )
        candidates = (
            DataclassFieldNameCollectionProjectionTarget.candidates_from_function(
                boundary.function,
                boundary.authority,
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive field-name collection; found {len(candidates)}"
            )
        projection = candidates[0]
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if dataclasses_reference is None:
            raise ValueError(
                "Dataclass field-name projection has no collision-free dataclasses "
                "reference"
            )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            dataclasses_reference,
        )
        if source_replacement is None:
            raise ValueError(
                "Dataclass field-name projection cannot preserve its source span"
            )
        return cls(
            authority=boundary.authority,
            projection=projection,
            source_replacement=source_replacement,
            import_sources=(
                (dataclasses_reference.import_source,)
                if dataclasses_reference.import_source is not None
                else ()
            ),
            dataclasses_reference=dataclasses_reference,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.collection_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        return replacement_span.replacement(
            source,
            projection.derived_source(dataclasses_reference, authority),
        )


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionCandidate:
    """One exhaustive return-pair projection proved against a dataclass."""

    projection: ReturnKeyValueSequenceProjectionTarget
    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassKeyValueSequenceProjectionDerivation(
    SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]
):
    """Current-source proof for one exhaustive dataclass return-pair sequence."""

    element_run: DataclassKeyValueElementRunProjection
    dataclasses_reference: DataclassesModuleReference

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassKeyValueSequenceProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            if (
                candidate := cls.candidate_from_return(
                    context,
                    boundary,
                    node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one exhaustive return-pair sequence; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=tuple(
                import_source
                for import_source in (
                    boundary.authority_import_source,
                    candidate.dataclasses_reference.import_source,
                )
                if import_source is not None
            ),
            element_run=candidate.element_run,
            dataclasses_reference=candidate.dataclasses_reference,
        )

    @classmethod
    def candidate_from_return(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        return_node: ast.Return,
    ) -> DataclassKeyValueSequenceProjectionCandidate | None:
        projection = ReturnKeyValueSequenceProjectionTargetAuthority.from_return_node(
            boundary.function,
            return_node,
            boundary.authority.field_names,
        )
        if projection is None:
            return None
        element_run = DataclassKeyValueElementRunProjection.from_targets(
            boundary.authority,
            projection,
        )
        dataclasses_reference = DataclassesModuleReference.from_projection(
            context,
            projection,
        )
        if (
            element_run is None
            or dataclasses_reference is None
            or not element_run.owner_has_nominal_authority_type(
                context,
                boundary.authority,
                projection,
                boundary.authority_import_source,
            )
        ):
            return None
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            element_run,
            dataclasses_reference,
        )
        if source_replacement is None:
            return None
        return DataclassKeyValueSequenceProjectionCandidate(
            projection=projection,
            element_run=element_run,
            dataclasses_reference=dataclasses_reference,
            source_replacement=source_replacement,
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
        element_run: DataclassKeyValueElementRunProjection,
        dataclasses_reference: DataclassesModuleReference,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        first_offsets = geometry.node_offsets(element_run.first_element_node)
        last_offsets = geometry.node_offsets(element_run.last_element_node)
        sequence_offsets = geometry.node_offsets(projection.sequence_node)
        owner_source = geometry.segment_for_node(element_run.owner_node)
        if (
            first_offsets is None
            or last_offsets is None
            or sequence_offsets is None
            or owner_source is None
        ):
            return None
        replacement_span = SourceTextSpan(
            start_offset=first_offsets[0],
            end_offset=last_offsets[1],
        )
        if replacement_span.contains_comment(source):
            return None
        has_trailing_comma = (
            source[last_offsets[1] : sequence_offsets[1]].lstrip().startswith(",")
        )
        indentation = " " * element_run.first_element_node.col_offset
        continuation_indentation = f"{indentation}    "
        nested_indentation = f"{continuation_indentation}    "
        value_indentation = f"{nested_indentation}    "
        replacement_source = (
            "*(\n"
            f"{continuation_indentation}(\n"
            f"{nested_indentation}field.name,\n"
            f"{nested_indentation}getattr(\n"
            f"{value_indentation}{owner_source},\n"
            f"{value_indentation}field.name,\n"
            f"{nested_indentation})\n"
            f"{continuation_indentation})\n"
            f"{continuation_indentation}for field in "
            f"{dataclasses_reference.expression}.fields(\n"
            f"{nested_indentation}{authority.name}\n"
            f"{continuation_indentation})\n"
            f"{indentation})"
            f"{'' if has_trailing_comma else ','}"
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class SourceDerivedDataclassProjectionOperation(
    SourceDerivedAuthorityProjectionOperation,
    Generic[ProjectionTargetT],
    ABC,
):
    """Replay one exact dataclass projection from its current declarations."""

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        derivation = self.required_derivation(snapshot)
        edits = tuple(
            edit
            for import_source in derivation.import_sources
            for edit in self.required_import_mutations(
                derivation.projection.source_path,
                import_source=import_source,
                default_rationale=(
                    "Import a declaration required by the dataclass-derived projection."
                ),
            )
        )
        replacement = derivation.source_replacement
        replacement_edits = PatchTargetOperation(
            target=SourceRewriteTarget(
                target_id=derivation.projection.target.target_id,
            ),
            replacements=(replacement,),
            rationale=("Replace mirrored fields with an authority-owned projection."),
        ).source_edits(snapshot)
        return (*edits, *replacement_edits)

    @abstractmethod
    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ProjectionTargetT]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassPayloadProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnDictProjectionTarget]
):
    """Derive one exhaustive return-dict projection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnDictProjectionTarget]:
        return DataclassPayloadProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassFieldNameCollectionProjectionOperation(
    SourceDerivedDataclassProjectionOperation[
        DataclassFieldNameCollectionProjectionTarget
    ]
):
    """Derive one exhaustive field-name collection from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassFieldNameCollectionProjectionTarget]:
        return DataclassFieldNameCollectionProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassKeyValueSequenceProjectionOperation(
    SourceDerivedDataclassProjectionOperation[ReturnKeyValueSequenceProjectionTarget]
):
    """Derive one exhaustive return-pair sequence from a dataclass authority."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[ReturnKeyValueSequenceProjectionTarget]:
        return DataclassKeyValueSequenceProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True)
class SourceDerivedDataclassProjectionRecipeParts(
    FindingRecipeParts,
    Generic[ProjectionTargetT],
):
    """Exact authority and proof-bearing operation for a dataclass projection."""

    authority: DataclassPayloadAuthorityTarget
    operation: SourceDerivedDataclassProjectionOperation[ProjectionTargetT]

    @classmethod
    def from_proven_operation(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        operation: SourceDerivedDataclassProjectionOperation[ProjectionTargetT],
    ) -> SourceDerivedDataclassProjectionRecipeParts[ProjectionTargetT] | None:
        try:
            operation.required_derivation(context)
        except ValueError:
            return None
        return cls(authority=authority, operation=operation)

    def recipe_for(self, finding: RefactorFinding) -> RefactorRecipe:
        return (
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-derive-dataclass-projection",
                reason="Derive a mirrored projection from its dataclass authority.",
            )
            .with_authority_claim(
                AstTargetAuthorityClaim.from_target(
                    self.authority.target,
                    authority_kind=SemanticAuthorityKind.DATACLASS_SCHEMA,
                )
            )
            .with_operation(self.operation)
        )


@dataclass(frozen=True, kw_only=True)
class DataclassPayloadProjectionMappingRecipeBuilder(
    ReturnDictFieldValueExtractor,
    DataclassAuthorityMappingRecipeBuilder[
        ReturnDictProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive direct-instance mapping from dataclass fields."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass payload projection requires mapping metrics"
        seed = SemanticMirrorRecipeSeedLocations.from_finding(self.finding)
        if seed is None:
            return (
                "dataclass payload projection requires projection and authority "
                "locations"
            )
        import_boundary = SemanticMirrorImportBoundary.from_seed(seed, self)
        if import_boundary is None:
            return "dataclass payload projection requires source-index-resolved files"
        if import_boundary.import_would_create_cycle(self):
            return "dataclass payload projection import would create a module cycle"
        if self.parts is not None:
            return (
                "dataclass payload projection has an executable instance-field recipe"
            )
        return (
            "dataclass payload projection requires one contiguous, exhaustive, "
            "declaration-ordered run of direct field reads from a nominally typed "
            "authority instance"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        return function_return is not None and isinstance(
            function_return.return_node.value,
            ast.Dict,
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnDictProjectionTarget | None:
        return ReturnDictProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnDictProjectionTarget,
    ) -> SourceDerivedDataclassProjectionRecipeParts[ReturnDictProjectionTarget] | None:
        operation = DeriveDataclassPayloadProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassFieldNameCollectionProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        DataclassFieldNameCollectionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive an exhaustive local field-name collection from a dataclass."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass field-name projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass field-name projection has an executable recipe"
        return (
            "dataclass field-name projection requires one local tuple or list that "
            "exhaustively names direct dataclass fields in declaration order, with "
            "the authority already available at runtime"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function = ResolvedFunctionProjectionTarget.from_source_line(
            self,
            source_path=source_path,
            line=seed.projection.line,
        )
        return function is not None and any(
            DataclassFieldNameCollectionProjectionTarget.bound_collection(
                statement,
                seed.projection.subject_symbol,
                seed.projection.line,
            )
            for statement in ast.walk(function.node)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> DataclassFieldNameCollectionProjectionTarget | None:
        return DataclassFieldNameCollectionProjectionTarget.from_binding_location(
            self,
            source_path=source_path,
            binding_name=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=frozenset(self.finding.metrics.plan_field_names),
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassFieldNameCollectionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassFieldNameCollectionProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassFieldNameCollectionProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassKeyValueSequenceProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ReturnKeyValueSequenceProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ],
    ],
    DataclassPayloadProjectionConcept,
):
    """Derive returned ``("field", value)`` items from a dataclass authority."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass key/value sequence projection requires mapping metrics"
        if self.parts is not None:
            return (
                "dataclass key/value sequence projection has an executable "
                "instance-field recipe"
            )
        return (
            "dataclass key/value sequence projection requires one contiguous, "
            "exhaustive, declaration-ordered run of direct pair values read from "
            "a nominally typed authority instance"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_value is not None and field_value.field_name in field_names
            for element in function_return.return_node.value.elts
            for field_value in (
                ReturnKeyValueSequenceProjectionTargetAuthority.field_value(element),
            )
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        return ReturnKeyValueSequenceProjectionTargetAuthority.from_function_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
            field_names=self.finding.metrics.plan_field_names,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ReturnKeyValueSequenceProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            ReturnKeyValueSequenceProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassKeyValueSequenceProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )


@dataclass(frozen=True)
class DataclassConstructorFieldArgument(ProductFieldValue):
    """One authority field and its value at an external constructor call."""


@dataclass(frozen=True)
class DataclassConstructorProjectionTarget(ResolvedFunctionProjectionTarget):
    """External nominal constructor call carrying all dataclass authority fields."""

    constructor: NominalConstructorCall
    field_arguments: tuple[DataclassConstructorFieldArgument, ...]
    remaining_keywords: tuple[ast.keyword, ...]

    @property
    def call_node(self) -> ast.Call:
        return self.constructor.call_node


@dataclass(frozen=True)
class DataclassConstructorProjectionMethod:
    """Direct authority method that forwards fields to one nominal constructor."""

    node: ast.FunctionDef
    constructor: NominalConstructorCall
    receiver_name: str
    parameter_names: tuple[str, ...]

    @property
    def method_name(self) -> str:
        return self.node.name

    @classmethod
    def candidates_from_authority(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> tuple[DataclassConstructorProjectionMethod, ...]:
        return tuple(
            candidate
            for statement in authority.node.body
            if isinstance(statement, ast.FunctionDef)
            if (
                candidate := cls.from_method(
                    context,
                    authority,
                    statement,
                    constructor_symbol,
                    remaining_keyword_names,
                )
            )
            is not None
        )

    @classmethod
    def from_method(
        cls,
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        method_node: ast.FunctionDef,
        constructor_symbol: str,
        remaining_keyword_names: tuple[str, ...],
    ) -> DataclassConstructorProjectionMethod | None:
        body = statements_without_docstring(method_node.body)
        if (
            method_node.decorator_list
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Call)
            or variadic_parameters(method_node.args)
        ):
            return None
        positional_parameters = (
            *method_node.args.posonlyargs,
            *method_node.args.args,
        )
        if not positional_parameters or len(method_node.args.posonlyargs) > 1:
            return None
        receiver = positional_parameters[0]
        if method_node.args.posonlyargs:
            keyword_parameters = (
                *method_node.args.args,
                *method_node.args.kwonlyargs,
            )
        else:
            keyword_parameters = (
                *method_node.args.args[1:],
                *method_node.args.kwonlyargs,
            )
        parameter_names = tuple(parameter.arg for parameter in keyword_parameters)
        if len(frozenset(parameter_names)) != len(parameter_names) or frozenset(
            parameter_names
        ) != frozenset(remaining_keyword_names):
            return None
        constructor = NominalConstructorCall.from_context(
            context,
            authority.file_path,
            method_node,
            body[0].value,
        )
        if (
            constructor is None
            or constructor.constructor_symbol != constructor_symbol
            or frozenset(constructor.keyword_names)
            != frozenset((*authority.field_names, *parameter_names))
        ):
            return None
        if any(
            not cls.keyword_forwards_receiver_field(
                constructor,
                field_name,
                receiver.arg,
            )
            for field_name in authority.field_names
        ):
            return None
        if any(
            not cls.keyword_forwards_parameter(constructor, parameter_name)
            for parameter_name in parameter_names
        ):
            return None
        return cls(
            node=method_node,
            constructor=constructor,
            receiver_name=receiver.arg,
            parameter_names=parameter_names,
        )

    @staticmethod
    def keyword_forwards_receiver_field(
        constructor: NominalConstructorCall,
        field_name: str,
        receiver_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(field_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == receiver_name
            and keyword.value.attr == field_name
        )

    @staticmethod
    def keyword_forwards_parameter(
        constructor: NominalConstructorCall,
        parameter_name: str,
    ) -> bool:
        keyword = constructor.keyword_argument(parameter_name)
        return bool(
            keyword is not None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == parameter_name
        )


@dataclass(frozen=True)
class DataclassConstructorProjectionCandidate:
    """One constructor projection and its exact authority-method relation."""

    projection: DataclassConstructorProjectionTarget
    authority_method: DataclassConstructorProjectionMethod
    source_replacement: SourceTextReplacement


@dataclass(frozen=True)
class DataclassConstructorProjectionDerivation(
    SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]
):
    """Current-source proof for one equivalent constructor projection."""

    authority_method: DataclassConstructorProjectionMethod

    @classmethod
    def from_context(
        cls,
        context: CodemodSelectorContext,
        authority_reference: SourceRewriteTarget,
        projection_reference: SourceRewriteTarget,
    ) -> DataclassConstructorProjectionDerivation:
        boundary = DataclassProjectionBoundary.from_context(
            context,
            authority_reference,
            projection_reference,
        )
        boundary.authority.require_transparent_direct_construction()
        candidates = tuple(
            candidate
            for node in walk_function_body_nodes(boundary.function.node.body)
            if isinstance(node, ast.Return)
            for call_node in StatementValue.of(node).walk()
            if isinstance(call_node, ast.Call)
            if (
                candidate := cls.candidate_from_call(
                    context,
                    boundary,
                    call_node,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                "Dataclass authority and projection function must expose exactly "
                f"one equivalent constructor projection; found {len(candidates)}"
            )
        candidate = candidates[0]
        return cls(
            authority=boundary.authority,
            projection=candidate.projection,
            source_replacement=candidate.source_replacement,
            import_sources=(
                (boundary.authority_import_source,)
                if boundary.authority_import_source is not None
                else ()
            ),
            authority_method=candidate.authority_method,
        )

    @classmethod
    def candidate_from_call(
        cls,
        context: CodemodSelectorContext,
        boundary: DataclassProjectionBoundary,
        call_node: ast.Call,
    ) -> DataclassConstructorProjectionCandidate | None:
        constructor = NominalConstructorCall.from_context(
            context,
            boundary.function.source_path,
            boundary.function.node,
            call_node,
        )
        if constructor is None:
            return None
        field_name_set = frozenset(boundary.authority.field_names)
        projected_field_names = tuple(
            name for name in constructor.keyword_names if name in field_name_set
        )
        if projected_field_names != boundary.authority.field_names:
            return None
        field_arguments = tuple(
            DataclassConstructorFieldArgument(
                field_name=field_name,
                value_node=constructor.required_keyword_argument(field_name).value,
            )
            for field_name in boundary.authority.field_names
        )
        remaining_keywords = tuple(
            keyword
            for keyword in constructor.keyword_arguments
            if keyword.arg not in field_name_set
        )
        if not cls.remaining_values_are_post_construction_safe(
            boundary.function,
            remaining_keywords,
        ):
            return None
        authority_methods = (
            DataclassConstructorProjectionMethod.candidates_from_authority(
                context,
                boundary.authority,
                constructor.constructor_symbol,
                tuple(cast(str, keyword.arg) for keyword in remaining_keywords),
            )
        )
        if len(authority_methods) != 1:
            return None
        projection = DataclassConstructorProjectionTarget(
            source_path=boundary.function.source_path,
            function_qualname=boundary.function.function_qualname,
            target=boundary.function.target,
            node=boundary.function.node,
            constructor=constructor,
            field_arguments=field_arguments,
            remaining_keywords=remaining_keywords,
        )
        source_replacement = cls.projection_replacement(
            context,
            boundary.authority,
            projection,
            authority_methods[0],
        )
        if source_replacement is None:
            return None
        return DataclassConstructorProjectionCandidate(
            projection=projection,
            authority_method=authority_methods[0],
            source_replacement=source_replacement,
        )

    @staticmethod
    def remaining_values_are_post_construction_safe(
        function: ResolvedFunctionProjectionTarget,
        keywords: tuple[ast.keyword, ...],
    ) -> bool:
        parameter_names = frozenset(function.target.parameters)
        return all(
            isinstance(keyword.value, ast.Constant)
            or (
                isinstance(keyword.value, ast.Name)
                and keyword.value.id in parameter_names
            )
            for keyword in keywords
        )

    @staticmethod
    def projection_replacement(
        context: CodemodSelectorContext,
        authority: DataclassPayloadAuthorityTarget,
        projection: DataclassConstructorProjectionTarget,
        authority_method: DataclassConstructorProjectionMethod,
    ) -> SourceTextReplacement | None:
        source = context.sources_by_file_path[projection.source_path]
        geometry = SourceTextGeometry(source)
        offsets = geometry.node_offsets(projection.call_node)
        if offsets is None:
            return None
        replacement_span = SourceTextSpan.from_offsets(offsets)
        if replacement_span.contains_comment(source):
            return None
        authority_instance = ast.Call(
            func=ast.Name(id=authority.name, ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(
                    arg=field.field_name,
                    value=copy.deepcopy(field.value_node),
                )
                for field in projection.field_arguments
            ],
        )
        replacement_call = ast.Call(
            func=ast.Attribute(
                value=authority_instance,
                attr=authority_method.method_name,
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                copy.deepcopy(keyword) for keyword in projection.remaining_keywords
            ],
        )
        replacement_source = PythonExpressionSourceFormatter().replacement_source(
            ast.fix_missing_locations(replacement_call),
            line_prefix=geometry.line_indent(replacement_span.start_offset),
        )
        return replacement_span.replacement(source, replacement_source)


@dataclass(frozen=True, kw_only=True)
class DeriveDataclassConstructorProjectionOperation(
    SourceDerivedDataclassProjectionOperation[DataclassConstructorProjectionTarget]
):
    """Derive one constructor call through an equivalent dataclass method."""

    def required_derivation(
        self,
        context: CodemodSelectorContext,
    ) -> SourceDerivedDataclassProjection[DataclassConstructorProjectionTarget]:
        return DataclassConstructorProjectionDerivation.from_context(
            context,
            self.target,
            self.projection_target,
        )


@dataclass(frozen=True, kw_only=True)
class DataclassConstructorProjectionMappingRecipeBuilder(
    DataclassAuthorityMappingRecipeBuilder[
        ResolvedFunctionProjectionTarget,
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ],
    ],
    ConstructorKwargCarrierProjectionConcept,
):
    """Derive constructor keyword mirrors through an existing dataclass method."""

    finding: RefactorFinding

    def rejection_reason(self) -> str:
        if not isinstance(self.finding.metrics, MappingMetrics):
            return "dataclass constructor projection requires mapping metrics"
        if self.parts is not None:
            return "dataclass constructor projection has an executable authority recipe"
        return (
            "dataclass constructor projection requires one nominal constructor call "
            "that is equivalent to a direct authority method"
        )

    def projection_shape_is_applicable(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> bool:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )
        if function_return is None:
            return False
        field_names = frozenset(self.finding.metrics.plan_field_names)
        return any(
            field_names
            <= frozenset(
                CallKeywords.of(call.keywords).names
            )
            for call in ast.walk(function_return.return_node.value)
            if isinstance(call, ast.Call)
        )

    def projection_target(
        self,
        seed: SemanticMirrorRecipeSeedLocations,
        source_path: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        return FunctionReturnProjectionTarget.from_return_location(
            self,
            source_path=source_path,
            function_qualname=seed.projection.subject_symbol,
            line=seed.projection.line,
        )

    def recipe_parts(
        self,
        authority: DataclassPayloadAuthorityTarget,
        projection: ResolvedFunctionProjectionTarget,
    ) -> (
        SourceDerivedDataclassProjectionRecipeParts[
            DataclassConstructorProjectionTarget
        ]
        | None
    ):
        operation = DeriveDataclassConstructorProjectionOperation(
            target=SourceRewriteTarget(target_id=authority.target.target_id),
            projection_target=SourceRewriteTarget(
                target_id=projection.target.target_id
            ),
        )
        return SourceDerivedDataclassProjectionRecipeParts.from_proven_operation(
            self,
            authority=authority,
            operation=operation,
        )
