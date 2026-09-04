"""Proof objects for query facades displaced from their nominal enum keys."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .ast_tools import (
    AstParentIndex,
    CompactModuleIdentity,
    CollectedFamily,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
    ParsedModule,
    statements_without_docstring,
)
from .class_index import (
    ClassFamilyIndex,
    IndexedClass,
    ModuleClassReferenceResolver,
    ModuleNominalBindingWitness,
    ModuleNominalBindingView,
    NamedImportModuleBindingProjection,
    RepositoryModuleBindingProof,
    build_class_family_index,
)
from .collection_algebra import sorted_tuple
from .enum_semantics import PYTHON_ENUM_BASE_AUTHORITY
from .models import SourceLocation


@dataclass(frozen=True)
class EnumKeyedQueryConsumer:
    """One direct index into a class-owned map keyed by an enum value."""

    line: int
    column: int
    owner_symbol: str


@dataclass(frozen=True)
class EnumKeyedDerivedMapFacadeComponent:
    """Derived map whose key-facing query behavior is externally displaced."""

    file_path: str
    enum_symbol: str
    enum_line: int
    map_owner_symbol: str
    map_owner_line: int
    map_method_name: str
    map_method_line: int
    reverse_method_name: str
    reverse_method_line: int
    property_name: str
    key_variable_name: str
    value_variable_name: str
    value_annotation_source: str
    star_import_exclusion_names: tuple[str, ...]
    consumers: tuple[EnumKeyedQueryConsumer, ...]

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.file_path,
                self.reverse_method_line,
                self.reverse_method_symbol,
            ),
            self.authority_evidence,
            SourceLocation(
                self.file_path,
                self.map_owner_line,
                self.map_owner_symbol,
            ),
            SourceLocation(
                self.file_path,
                self.map_method_line,
                self.map_method_symbol,
            ),
            *(
                SourceLocation(self.file_path, consumer.line, consumer.owner_symbol)
                for consumer in self.consumers
            ),
        )

    @property
    def authority_evidence(self) -> SourceLocation:
        return SourceLocation(self.file_path, self.enum_line, self.enum_symbol)

    @property
    def projection_evidence(self) -> SourceLocation:
        return SourceLocation(
            self.file_path,
            self.map_method_line,
            self.map_method_symbol,
        )

    @property
    def map_method_symbol(self) -> str:
        return f"{self.map_owner_symbol}.{self.map_method_name}"

    @property
    def reverse_method_symbol(self) -> str:
        return f"{self.map_owner_symbol}.{self.reverse_method_name}"


@dataclass(frozen=True)
class _EnumKeyedMapMethod:
    owner: IndexedClass
    method: ast.FunctionDef
    enum_class: IndexedClass
    value_annotation: ast.expr
    binding_witnesses: tuple[ModuleNominalBindingWitness, ...]


@dataclass(frozen=True)
class _EnumKeyedReverseQuery:
    method: ast.FunctionDef
    key_variable_name: str
    value_variable_name: str
    binding_witnesses: tuple[ModuleNominalBindingWitness, ...]


@dataclass(frozen=True)
class EnumKeyedDerivedMapFacadeModuleProjection(CompactModuleIdentity):
    """Module-local facade shapes with explicit repository proof obligations."""

    components: tuple[EnumKeyedDerivedMapFacadeComponent, ...]


@dataclass(frozen=True)
class EnumKeyedDerivedMapFacadeComponentBuilder:
    """Recover enum-facing queries from one module without inventing an owner."""

    module: ParsedModule
    repository_modules: tuple[ParsedModule, ...] = ()

    def proven_components(self) -> tuple[EnumKeyedDerivedMapFacadeComponent, ...]:
        modules = self.repository_modules or (self.module,)
        return self._proven_components(
            build_class_family_index((self.module,)),
            RepositoryModuleBindingProof(modules),
        )

    def projected_components(self) -> tuple[EnumKeyedDerivedMapFacadeComponent, ...]:
        """Project local shapes while preserving star-import proof obligations."""

        return self._proven_components(
            build_class_family_index((self.module,)),
            NamedImportModuleBindingProjection(),
        )

    @classmethod
    def collect_modules(
        cls,
        modules: tuple[ParsedModule, ...],
    ) -> tuple[EnumKeyedDerivedMapFacadeComponent, ...]:
        """Recover all facades while sharing repository proof indexes once."""

        binding_proof = RepositoryModuleBindingProof(modules)
        return sorted_tuple(
            (
                component
                for module in modules
                if cls.can_contain_facade(module)
                for component in cls(module, modules)._proven_components(
                    build_class_family_index((module,)),
                    binding_proof,
                )
            ),
            key=lambda component: (
                component.file_path,
                component.enum_symbol,
                component.map_owner_symbol,
                component.map_method_name,
            ),
        )

    @staticmethod
    def can_contain_facade(module: ParsedModule) -> bool:
        """Return whether one AST can contain the complete facade relation."""

        nodes = tuple(ast.walk(module.module))
        possible_map_method_names = frozenset(
            node.name
            for node in nodes
            if isinstance(node, ast.FunctionDef)
            and isinstance(node.returns, ast.Subscript)
            and isinstance(node.returns.value, ast.Name)
            and node.returns.value.id == "dict"
        )
        if not possible_map_method_names:
            return False
        has_direct_consumer = False
        has_reverse_query = False
        for node in nodes:
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "items"
                and isinstance(node.value, ast.Call)
            ):
                map_call = node.value
                has_reverse_query = has_reverse_query or bool(
                    isinstance(map_call.func, ast.Attribute)
                    and map_call.func.attr in possible_map_method_names
                )
            elif isinstance(node, ast.Subscript):
                map_call = node.value
                has_direct_consumer = has_direct_consumer or bool(
                    isinstance(map_call, ast.Call)
                    and isinstance(map_call.func, ast.Attribute)
                    and map_call.func.attr in possible_map_method_names
                )
            if has_direct_consumer and has_reverse_query:
                return True
        return False

    def _proven_components(
        self,
        class_index: ClassFamilyIndex,
        binding_proof: ModuleNominalBindingView,
    ) -> tuple[EnumKeyedDerivedMapFacadeComponent, ...]:
        resolver = ModuleClassReferenceResolver(self.module, class_index)
        components = []
        for owner in class_index.classes_by_symbol.values():
            if owner.file_path != self.module.file_path:
                continue
            for map_method in self._map_methods(
                owner,
                resolver=resolver,
                binding_proof=binding_proof,
            ):
                for reverse_query in self._reverse_queries(
                    owner,
                    map_method=map_method,
                    binding_proof=binding_proof,
                ):
                    member_binding_witnesses = self._enum_query_member_witnesses(
                        map_method.enum_class,
                        property_name=reverse_query.value_variable_name,
                        reverse_method_name=reverse_query.method.name,
                        binding_proof=binding_proof,
                    )
                    if member_binding_witnesses is None:
                        continue
                    consumers = self._direct_consumers(
                        owner,
                        map_method=map_method,
                        reverse_method=reverse_query.method,
                        resolver=resolver,
                    )
                    if not consumers:
                        continue
                    components.append(
                        EnumKeyedDerivedMapFacadeComponent(
                            file_path=self.module.file_path,
                            enum_symbol=map_method.enum_class.symbol,
                            enum_line=map_method.enum_class.line,
                            map_owner_symbol=owner.symbol,
                            map_owner_line=owner.line,
                            map_method_name=map_method.method.name,
                            map_method_line=map_method.method.lineno,
                            reverse_method_name=reverse_query.method.name,
                            reverse_method_line=reverse_query.method.lineno,
                            property_name=reverse_query.value_variable_name,
                            key_variable_name=reverse_query.key_variable_name,
                            value_variable_name=reverse_query.value_variable_name,
                            value_annotation_source=ast.unparse(
                                map_method.value_annotation
                            ),
                            star_import_exclusion_names=sorted_tuple(
                                {
                                    witness.root_name
                                    for witness in (
                                        *map_method.binding_witnesses,
                                        *reverse_query.binding_witnesses,
                                        *member_binding_witnesses,
                                    )
                                }
                            ),
                            consumers=consumers,
                        )
                    )
        return sorted_tuple(
            components,
            key=lambda component: (
                component.file_path,
                component.enum_symbol,
                component.map_owner_symbol,
                component.map_method_name,
            ),
        )

    def _map_methods(
        self,
        owner: IndexedClass,
        *,
        resolver: ModuleClassReferenceResolver,
        binding_proof: ModuleNominalBindingView,
    ) -> tuple[_EnumKeyedMapMethod, ...]:
        methods = []
        for statement in owner.node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            classmethod_witness = self._builtin_classmethod_witness(
                owner,
                statement,
                binding_proof=binding_proof,
            )
            if classmethod_witness is None:
                continue
            annotation_parts = self._mapping_annotation_parts(
                owner,
                statement,
                binding_proof=binding_proof,
            )
            if annotation_parts is None:
                continue
            key_annotation, value_annotation, dict_witness = annotation_parts
            enum_symbol = resolver.symbol_for_reference(key_annotation)
            enum_class = (
                None
                if enum_symbol is None
                else resolver.class_index.class_for(enum_symbol)
            )
            if enum_class is None:
                continue
            enum_base_witnesses = tuple(
                witness
                for base in enum_class.node.bases
                if (
                    witness := binding_proof.reference_witness_at(
                        self.module,
                        base,
                        line=enum_class.line,
                    )
                )
                is not None
                if PYTHON_ENUM_BASE_AUTHORITY.matches_qualified(witness.qualified_name)
            )
            if len(enum_base_witnesses) != 1:
                continue
            methods.append(
                _EnumKeyedMapMethod(
                    owner=owner,
                    method=statement,
                    enum_class=enum_class,
                    value_annotation=value_annotation,
                    binding_witnesses=(
                        classmethod_witness,
                        dict_witness,
                        enum_base_witnesses[0],
                    ),
                )
            )
        return tuple(methods)

    def _mapping_annotation_parts(
        self,
        owner: IndexedClass,
        method: ast.FunctionDef,
        *,
        binding_proof: ModuleNominalBindingView,
    ) -> tuple[ast.expr, ast.expr, ModuleNominalBindingWitness] | None:
        annotation = method.returns
        dict_witness = binding_proof.unshadowed_builtin_witness(
            self.module,
            "dict",
            line=method.lineno,
            preceding_class_bound_names=self._class_bound_names_before(owner, method),
        )
        if not (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "dict"
            and dict_witness is not None
            and isinstance(annotation.slice, ast.Tuple)
            and len(annotation.slice.elts) == 2
        ):
            return None
        key_annotation, value_annotation = annotation.slice.elts
        return key_annotation, value_annotation, dict_witness

    def _reverse_queries(
        self,
        owner: IndexedClass,
        *,
        map_method: _EnumKeyedMapMethod,
        binding_proof: ModuleNominalBindingView,
    ) -> tuple[_EnumKeyedReverseQuery, ...]:
        queries = []
        for statement in owner.node.body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if statement is map_method.method:
                continue
            classmethod_witness = self._builtin_classmethod_witness(
                owner,
                statement,
                binding_proof=binding_proof,
            )
            if classmethod_witness is None:
                continue
            query = self._reverse_query(
                statement,
                map_method.method.name,
                classmethod_witness,
            )
            if query is not None:
                queries.append(query)
        return tuple(queries)

    @staticmethod
    def _reverse_query(
        method: ast.FunctionDef,
        map_method_name: str,
        classmethod_witness: ModuleNominalBindingWitness,
    ) -> _EnumKeyedReverseQuery | None:
        body = statements_without_docstring(method.body)
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        if not (
            positional_parameters
            and positional_parameters[0].arg == "cls"
            and not method.args.defaults
            and not any(default is not None for default in method.args.kw_defaults)
            and len(body) == 1
            and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Call)
            and len(body[0].value.args) == 1
            and not body[0].value.keywords
            and isinstance(body[0].value.args[0], ast.GeneratorExp)
        ):
            return None
        generator_expression = body[0].value.args[0]
        if len(generator_expression.generators) != 1:
            return None
        generator = generator_expression.generators[0]
        if not (
            isinstance(generator.target, ast.Tuple)
            and len(generator.target.elts) == 2
            and all(isinstance(item, ast.Name) for item in generator.target.elts)
            and isinstance(generator_expression.elt, ast.Name)
            and generator_expression.elt.id == generator.target.elts[0].id
            and isinstance(generator.iter, ast.Call)
            and not generator.iter.args
            and not generator.iter.keywords
            and isinstance(generator.iter.func, ast.Attribute)
            and generator.iter.func.attr == "items"
            and isinstance(generator.iter.func.value, ast.Call)
            and not generator.iter.func.value.args
            and not generator.iter.func.value.keywords
            and isinstance(generator.iter.func.value.func, ast.Attribute)
            and generator.iter.func.value.func.attr == map_method_name
            and isinstance(generator.iter.func.value.func.value, ast.Name)
            and generator.iter.func.value.func.value.id == "cls"
            and len(generator.ifs) == 1
        ):
            return None
        key_target, value_target = generator.target.elts
        map_receiver = generator.iter.func.value.func.value
        if tuple(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Name) and node.id == "cls"
        ) != (map_receiver,) or any(
            isinstance(node, ast.Name)
            and node.id == "__class__"
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "super"
            )
            for node in ast.walk(method)
        ):
            return None
        return _EnumKeyedReverseQuery(
            method=method,
            key_variable_name=key_target.id,
            value_variable_name=value_target.id,
            binding_witnesses=(classmethod_witness,),
        )

    def _direct_consumers(
        self,
        owner: IndexedClass,
        *,
        map_method: _EnumKeyedMapMethod,
        reverse_method: ast.FunctionDef,
        resolver: ModuleClassReferenceResolver,
    ) -> tuple[EnumKeyedQueryConsumer, ...]:
        consumers = []
        parent_index = AstParentIndex(self.module.module)
        for node in ast.walk(self.module.module):
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.expr)
                and not any(
                    isinstance(slice_node, ast.Slice)
                    for slice_node in ast.walk(node.slice)
                )
                and self._is_map_call(
                    node.value,
                    owner=owner,
                    map_method_name=map_method.method.name,
                    resolver=resolver,
                )
            ):
                continue
            if reverse_method in parent_index.ancestors(node):
                continue
            enclosing = next(
                (
                    ancestor
                    for ancestor in parent_index.ancestors(node)
                    if isinstance(ancestor, ast.FunctionDef | ast.AsyncFunctionDef)
                ),
                None,
            )
            consumers.append(
                EnumKeyedQueryConsumer(
                    line=node.lineno,
                    column=node.col_offset,
                    owner_symbol=(
                        owner.simple_name if enclosing is None else enclosing.name
                    ),
                )
            )
        return sorted_tuple(
            consumers,
            key=lambda consumer: (consumer.line, consumer.column),
        )

    @staticmethod
    def _is_map_call(
        node: ast.expr,
        *,
        owner: IndexedClass,
        map_method_name: str,
        resolver: ModuleClassReferenceResolver,
    ) -> bool:
        return bool(
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == map_method_name
            and resolver.symbol_for_reference(node.func.value) == owner.symbol
        )

    def _builtin_classmethod_witness(
        self,
        owner: IndexedClass,
        method: ast.FunctionDef,
        *,
        binding_proof: ModuleNominalBindingView,
    ) -> ModuleNominalBindingWitness | None:
        if not (
            len(method.decorator_list) == 1
            and isinstance(method.decorator_list[0], ast.Name)
            and method.decorator_list[0].id == "classmethod"
        ):
            return None
        return binding_proof.unshadowed_builtin_witness(
            self.module,
            "classmethod",
            line=method.lineno,
            preceding_class_bound_names=(
                self._class_bound_names_before(
                    owner,
                    method,
                )
            ),
        )

    def _enum_query_member_witnesses(
        self,
        enum_class: IndexedClass,
        *,
        property_name: str,
        reverse_method_name: str,
        binding_proof: ModuleNominalBindingView,
    ) -> tuple[ModuleNominalBindingWitness, ...] | None:
        first_method = next(
            (
                statement
                for statement in enum_class.node.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            ),
            None,
        )
        preceding_class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            enum_class.node.body
            if first_method is None
            else (
                statement
                for statement in enum_class.node.body
                if statement.lineno < first_method.lineno
            )
        )
        insertion_line = (
            enum_class.node.end_lineno or enum_class.line
            if first_method is None
            else first_method.lineno
        )
        if not (
            PYTHON_ENUM_BASE_AUTHORITY.permits_new_member(property_name)
            and PYTHON_ENUM_BASE_AUTHORITY.permits_new_member(reverse_method_name)
        ):
            return None
        witnesses = tuple(
            binding_proof.unshadowed_builtin_witness(
                self.module,
                decorator_name,
                line=insertion_line,
                preceding_class_bound_names=preceding_class_bound_names,
            )
            for decorator_name in ("property", "classmethod")
        )
        if any(witness is None for witness in witnesses):
            return None
        return tuple(witness for witness in witnesses if witness is not None)

    @staticmethod
    def _class_bound_names_before(
        owner: IndexedClass,
        method: ast.FunctionDef,
    ) -> frozenset[str]:
        return LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            statement
            for statement in owner.node.body
            if statement.lineno < method.lineno
        )


class EnumKeyedDerivedMapFacadeModuleProjectionFamily(
    CollectedFamily[EnumKeyedDerivedMapFacadeModuleProjection]
):
    """Persist module-local enum-keyed facade shapes for compact joins."""

    item_type = EnumKeyedDerivedMapFacadeModuleProjection

    @classmethod
    def collect(
        cls,
        parsed_module: ParsedModule,
    ) -> list[EnumKeyedDerivedMapFacadeModuleProjection]:
        del cls
        return [
            EnumKeyedDerivedMapFacadeModuleProjection(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                components=(
                    EnumKeyedDerivedMapFacadeComponentBuilder(
                        parsed_module
                    ).projected_components()
                    if EnumKeyedDerivedMapFacadeComponentBuilder.can_contain_facade(
                        parsed_module
                    )
                    else ()
                ),
            )
        ]
