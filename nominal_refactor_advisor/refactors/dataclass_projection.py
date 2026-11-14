from __future__ import annotations

import ast
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
    cast,
)

from nominal_refactor_advisor.refactors.function_projection_targets import (
    ProjectionTargetT,
    ResolvedFunctionProjectionTarget,
    ReturnCollectionProjectionTarget,
    ReturnDictProjectionTarget,
    ReturnFieldValue,
    ReturnKeyValueSequenceProjectionTarget,
)
from nominal_refactor_advisor.assignment_projection import (
    SingleAssignmentAndValueNameProjection,
)
from nominal_refactor_advisor.ast_projection import (
    ExpressionReference,
    names_mentioned,
)
from nominal_refactor_advisor.ast_tools import (
    ROOT_NAME_PROJECTION,
    walk_function_body_nodes,
)
from nominal_refactor_advisor.class_families.parsed_index import (
    ModuleClassReferenceResolver,
)
from nominal_refactor_advisor.refactors.semantic_mirror import (
    FindingRecipeParts,
    MappingSemanticMirrorRecipeStrategy,
    PartsBackedMappingRecipeBuilder,
    RecipePartsT,
    SemanticMirrorImportBoundary,
    SemanticMirrorRecipeSeedLocations,
)
from nominal_refactor_advisor.refactors.class_base import ClassAuthorityReferenceProof
from nominal_refactor_advisor.refactors.authority_extraction import (
    SourceDerivedAuthorityProjectionOperation,
)
from nominal_refactor_advisor.codemod import PatchTargetOperation
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextReplacement,
    SourceTextSpan,
)
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.lexical_bindings import ImportBoundNameProjection
from nominal_refactor_advisor.lexical_scopes import FunctionBindingProjection
from nominal_refactor_advisor.models import (
    RefactorFinding,
)
from nominal_refactor_advisor.semantic_descent import SemanticAuthorityKind
from nominal_refactor_advisor.semantic_match import Maybe
from nominal_refactor_advisor.source_index import AstTargetNodeKind


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
            and excluded_annotation_names.isdisjoint(names_mentioned(statement.annotation))
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
