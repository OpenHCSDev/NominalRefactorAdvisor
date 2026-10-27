"""Source-proven codemods that descend projections to nominal authorities."""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from abc import ABC
from collections import defaultdict
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from nominal_refactor_advisor.class_index import ClassMethodPromotionSafetyProfile
from nominal_refactor_advisor.declaration_binding_transfer import (
    DeclarationModuleBindingEnvironment,
    DeclarationModuleBindingTransfer,
)

from .ast_tools import (
    BuiltinCallName,
    ParsedModule,
    statements_without_docstring,
)
from .class_index import (
    ClassFamilyIndex,
    CompactModuleClassProjectionFamily,
    IndexedClass,
    ModuleClassReferenceResolver,
    ModuleNominalBindingAuthority,
    build_compact_class_family_index,
)
from .codemod_declaration_source import (
    ClassBodySourceAuthority,
)
from .codemod_imports import ImportFromSource
from .codemod_native_requirements import (
    NativeDefinitionUseRequirement,
    NativeInvocationUseRequirement,
    DeclaredNativeUseInvariants,
    NativeUseRequirement,
    NativeUseResolution,
)
from .codemod_payload import PayloadRecordValueCodec, codemod_payload_field
from .codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from .descent import Descent, FamilyRoot, Mirror
from .descriptor_algebra import AliasProperty
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext
from .codemod_source_edits import (
    Include,
    NominalSourceEdit,
    SourceNodeSpan,
    SourceTextGeometry,
    SourceTextSpan,
    SourceTextSpanReplacement,
)
from .collection_algebra import sorted_tuple
from .foreign_family import FunctionDefinition
from .lexical_bindings import (
    ImportFromModuleName,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from .native_declarations import NativeDeclaration
from .registry_identity import (
    REGISTRY_ATTRIBUTE_NAME,
    mro_registry_value,
)
from .type_keyed_behavior import (
    TypeKeyedBehaviorProjectionComponent,
    TypeKeyedBehaviorProjectionComponentBuilder,
)


class _TypeKeyedBehaviorSubjectRenamer(ast.NodeTransformer):
    """Rename one projected subject after it becomes the target method receiver."""

    def __init__(self, subject_name: str) -> None:
        self.subject_name = subject_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id != self.subject_name:
            return node
        return ast.copy_location(ast.Name(id="self", ctx=node.ctx), node)


@dataclass(frozen=True)
class _TypeKeyedBehaviorMethodDescent:
    """One source-proven projection method moved onto its mapped target type."""

    projection_method: ast.FunctionDef
    target_class: IndexedClass
    source_module: ParsedModule
    target_module: ParsedModule
    class_family_index: ClassFamilyIndex
    projection_class: IndexedClass

    def transformed_source(self) -> str:
        safety = ClassMethodPromotionSafetyProfile.from_method(
            self.projection_method,
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.source_module.module.body),
            LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                self.projection_class.node.body
            ),
            source_lines=tuple(self.source_module.source.splitlines()),
        )
        if safety.hazards:
            raise ValueError(
                f"projected method {self.projection_method.name!r} "
                f"has ownership dependencies: {', '.join(safety.hazards)}"
            )
        method = copy.deepcopy(self.projection_method)
        if method.decorator_list:
            raise ValueError(
                f"projected method {method.name!r} has decorators that may change ownership"
            )
        positional_parameters = (*method.args.posonlyargs, *method.args.args)
        if len(positional_parameters) < 2:
            raise ValueError(
                f"projected method {method.name!r} lacks receiver and subject parameters"
            )
        receiver_name, subject_name = (
            positional_parameters[0].arg,
            positional_parameters[1].arg,
        )
        if subject_name in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(method.body):
            raise ValueError(
                f"projected method {method.name!r} rebinds its subject parameter"
            )
        if any(
            isinstance(node, ast.Name) and node.id == receiver_name
            for statement in method.body
            for node in ast.walk(statement)
        ):
            raise ValueError(
                f"projected method {method.name!r} depends on its projection receiver"
            )
        if any(
            isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            )
            for statement in method.body
            for node in ast.walk(statement)
        ):
            raise ValueError(
                f"projected method {method.name!r} contains a nested lexical scope"
            )
        self._remove_receiver_parameter(method)
        subject_parameter = (*method.args.posonlyargs, *method.args.args)[0]
        if subject_parameter.arg != subject_name:
            raise ValueError("projected method subject position changed during descent")
        subject_parameter.arg = "self"
        subject_parameter.annotation = None
        method.body, removed_guard = self._body_without_redundant_type_guard(
            method.body,
            subject_name=subject_name,
        )
        method.body = [
            _TypeKeyedBehaviorSubjectRenamer(subject_name).visit(statement)
            for statement in method.body
        ]
        ast.fix_missing_locations(method)
        self._require_target_module_bindings(method)
        return self._rewritten_method_source(
            method,
            subject_name=subject_name,
            removed_guard=removed_guard,
        )

    @staticmethod
    def _remove_receiver_parameter(method: ast.FunctionDef) -> None:
        if method.args.posonlyargs:
            method.args.posonlyargs.pop(0)
            return
        method.args.args.pop(0)

    def _body_without_redundant_type_guard(
        self,
        body: list[ast.stmt],
        *,
        subject_name: str,
    ) -> tuple[list[ast.stmt], ast.If | None]:
        if not body or not isinstance(body[0], ast.If):
            return body, None
        guard = body[0]
        guarded_type = self._negative_isinstance_type(
            guard.test,
            subject_name=subject_name,
        )
        if guarded_type is None:
            return body, None
        snapshot = ModuleNominalBindingAuthority(self.source_module).snapshot_before(
            self.projection_method.lineno
        )
        if not BuiltinCallName.ISINSTANCE.is_unshadowed(
            snapshot.resolves_unshadowed_builtin
        ):
            raise ValueError(
                f"projected method {self.projection_method.name!r} uses a shadowed "
                "isinstance guard"
            )
        resolver = ModuleClassReferenceResolver(
            self.source_module,
            self.class_family_index,
        )
        guarded_symbol = resolver.symbol_for_reference(guarded_type)
        if guarded_symbol != self.target_class.symbol:
            raise ValueError(
                f"projected method {self.projection_method.name!r} guards a type "
                "different from its registry key"
            )
        if (
            guard.orelse
            or len(guard.body) != 1
            or not isinstance(guard.body[0], ast.Return)
        ):
            raise ValueError(
                f"projected method {self.projection_method.name!r} has a non-removable type guard"
            )
        return body[1:], guard

    @staticmethod
    def _negative_isinstance_type(
        test: ast.expr,
        *,
        subject_name: str,
    ) -> ast.expr | None:
        if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
            return None
        call = test.operand
        if not (
            BuiltinCallName.isinstance_arguments(call) is not None
            and BuiltinCallName.is_isinstance_callee(call.func)
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == subject_name
        ):
            return None
        return call.args[1]

    def _require_target_module_bindings(self, method: ast.FunctionDef) -> None:
        DeclarationModuleBindingTransfer(
            source=DeclarationModuleBindingEnvironment(
                self.source_module, self.projection_class.node
            ),
            destination=DeclarationModuleBindingEnvironment(
                self.target_module, self.target_class.node
            ),
        ).require_preserved(method)

    def _rewritten_method_source(
        self,
        transformed_method: ast.FunctionDef,
        *,
        subject_name: str,
        removed_guard: ast.If | None,
    ) -> str:
        source = self.source_module.source
        geometry = SourceTextGeometry(source)
        method_span = SourceNodeSpan(
            self.projection_method,
            Include,
        )
        method_start, method_end = geometry.node_span_offsets(method_span)
        parameter_span = geometry.function_parameter_span(self.projection_method)
        if parameter_span.contains_comment(source):
            raise ValueError(
                f"projected method {self.projection_method.name!r} has parameter comments"
            )
        replacements = [
            SourceTextSpanReplacement.from_offsets(
                start_offset=parameter_span.start_offset,
                end_offset=parameter_span.end_offset,
                replacement_source=self._parameter_source(transformed_method),
            )
        ]
        removed_guard_span = (
            None
            if removed_guard is None
            else SourceTextSpan.from_offsets(
                geometry.node_span_offsets(SourceNodeSpan(removed_guard))
            )
        )
        if removed_guard_span is not None:
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=removed_guard_span.start_offset,
                    end_offset=removed_guard_span.end_offset,
                    replacement_source="",
                )
            )
        replacements.extend(
            SourceTextSpanReplacement.from_offsets(
                start_offset=start_offset,
                end_offset=end_offset,
                replacement_source="self",
            )
            for statement in self.projection_method.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id == subject_name
            for start_offset, end_offset in (geometry.required_node_offsets(node),)
            if removed_guard_span is None
            or not (
                removed_guard_span.start_offset <= start_offset
                and end_offset <= removed_guard_span.end_offset
            )
        )
        rewritten = geometry.source_with_replacements_in_span(
            method_start,
            method_end,
            replacements,
        )
        return textwrap.indent(
            textwrap.dedent(rewritten).rstrip("\r\n"),
            " " * (self.target_class.node.col_offset + 4),
        )

    @staticmethod
    def _parameter_source(method: ast.FunctionDef) -> str:
        declaration = copy.deepcopy(method)
        declaration.decorator_list = []
        declaration.returns = None
        declaration.body = [ast.Pass()]
        ast.fix_missing_locations(declaration)
        source = ast.unparse(declaration)
        node = ast.parse(source).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise ValueError("cannot render descended method parameters")
        span = SourceTextGeometry(source).function_parameter_span(node)
        return span.source_text(source)


@dataclass(frozen=True)
class _ProjectionLookupSequence:
    """One lookup, absence guard, and projected behavior call relation."""

    subject: ast.expr
    behavior_method_name: str
    statements: tuple[ast.stmt, ast.stmt, ast.stmt]

    @classmethod
    def from_statements(
        cls,
        statements: Iterable[ast.stmt],
        *,
        lookup_method_name: str,
        lookup_receiver_matches: Callable[[ast.expr], bool],
        behavior_method_names: frozenset[str],
    ) -> (_ProjectionLookupSequence | None):
        statement_tuple = tuple(statements)
        if len(statement_tuple) != 3:
            return None
        assignment, absent_guard, result = statement_tuple
        assignment_shape = cls._assignment_shape(
            assignment,
            lookup_method_name=lookup_method_name,
            lookup_receiver_matches=lookup_receiver_matches,
        )
        if assignment_shape is None:
            return None
        projection_name, subject = assignment_shape
        if not cls._is_absent_guard(
            absent_guard,
            projection_name=projection_name,
        ):
            return None
        behavior_method_name = cls._behavior_call_name(
            result,
            projection_name=projection_name,
            subject=subject,
        )
        if behavior_method_name not in behavior_method_names:
            return None
        return cls(
            subject=subject,
            behavior_method_name=behavior_method_name,
            statements=cast(tuple[ast.stmt, ast.stmt, ast.stmt], statement_tuple),
        )

    @staticmethod
    def _assignment_shape(
        statement: ast.stmt,
        *,
        lookup_method_name: str,
        lookup_receiver_matches: Callable[[ast.expr], bool],
    ) -> tuple[str, ast.expr] | None:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == lookup_method_name
            and lookup_receiver_matches(statement.value.func.value)
            and len(statement.value.args) == 1
            and not statement.value.keywords
        ):
            return None
        return statement.targets[0].id, statement.value.args[0]

    @staticmethod
    def _is_absent_guard(
        statement: ast.stmt,
        *,
        projection_name: str,
    ) -> bool:
        return bool(
            isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == projection_name
            and len(statement.test.ops) == 1
            and isinstance(statement.test.ops[0], ast.Is)
            and len(statement.test.comparators) == 1
            and isinstance(statement.test.comparators[0], ast.Constant)
            and statement.test.comparators[0].value is None
        )

    @staticmethod
    def _behavior_call_name(
        statement: ast.stmt,
        *,
        projection_name: str,
        subject: ast.expr,
    ) -> str | None:
        if not (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == projection_name
            and len(statement.value.args) == 1
            and not statement.value.keywords
            and ast.dump(statement.value.args[0], include_attributes=False)
            == ast.dump(subject, include_attributes=False)
        ):
            return None
        return statement.value.func.attr

    @property
    def direct_call_source(self) -> str:
        return ast.unparse(
            ast.Call(
                func=ast.Attribute(
                    value=copy.deepcopy(self.subject),
                    attr=self.behavior_method_name,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )


@dataclass(frozen=True)
class _TypeKeyedBehaviorFacade:
    facade_method_name: str
    behavior_method_name: str


@dataclass(frozen=True)
class _TypeKeyedBehaviorSourceDerivation:
    """Full-source proof and rewrite for one external type-keyed behavior family."""

    snapshot: CodemodSourceSnapshot
    component: TypeKeyedBehaviorProjectionComponent
    projection_root: IndexedClass
    lookup_method: ast.FunctionDef
    facades: tuple[_TypeKeyedBehaviorFacade, ...]
    operation: TypedMirror

    lookup_method_name = AliasProperty[str]("lookup_method.name")
    rationale = AliasProperty[str]("operation.rationale")

    def require_native_uses(self) -> None:
        for resolution in self.native_use_resolutions:
            resolution.require_admitted()

    @cached_property
    def native_use_resolutions(self) -> tuple[NativeUseResolution, ...]:
        return self.operation.supported_execution.resolve(self.native_use_requirements)

    @cached_property
    def native_use_requirements(self) -> tuple[NativeUseRequirement, ...]:
        """Bind the selected syntax's distinct native uses to their actual producer."""
        module = self.snapshot.parsed_module_for_source_path(
            self.projection_root.file_path
        )
        environment = self.snapshot.module_binding_proof.native_reference_environment(
            module
        )
        assignment = cast(
            ast.Assign, statements_without_docstring(self.lookup_method.body)[0]
        )
        lookup = cast(ast.Call, assignment.value)
        subject_type = cast(ast.Call, lookup.args[1])
        return tuple(
            requirement_type(
                type(self.operation),
                node,
                (NativeDeclaration(declaration),),
                environment,
                source_state=self.snapshot.product_flow_repository,
            )
            for requirement_type, node, declaration in (
                (
                    NativeDefinitionUseRequirement,
                    self.lookup_method.decorator_list[0],
                    classmethod,
                ),
                (NativeInvocationUseRequirement, lookup.func, mro_registry_value),
                (NativeInvocationUseRequirement, subject_type.func, type),
            )
        )

    @classmethod
    def required(
        cls,
        snapshot: CodemodSourceSnapshot,
        projection_root_symbol: str,
        *,
        operation: TypedMirror,
    ) -> _TypeKeyedBehaviorSourceDerivation:
        projections = CompactModuleClassProjectionFamily.collect_modules(
            snapshot.parsed_modules
        )
        class_index = build_compact_class_family_index(projections)
        component = TypeKeyedBehaviorProjectionComponentBuilder.from_projections(
            projections,
            class_index,
        ).component_for_projection_root(projection_root_symbol)
        if component is None:
            raise ValueError(
                "type-keyed behavior projection is no longer source-proven"
            )
        projection_root = snapshot.required_class_family_index.class_for(
            component.projection_root.symbol
        )
        if projection_root is None:
            raise ValueError("projection root has no current class declaration")
        cls._require_declared_target_contract(
            snapshot,
            projection_root,
            component,
        )
        lookup_method = cls._required_mro_lookup_method(
            projection_root,
            component,
        )
        return cls(
            snapshot=snapshot,
            component=component,
            projection_root=projection_root,
            lookup_method=lookup_method,
            facades=cls._facades(
                projection_root.node,
                lookup_method_name=lookup_method.name,
                behavior_method_names=frozenset(component.behavior_method_names),
            ),
            operation=operation,
        )

    @staticmethod
    def _require_declared_target_contract(
        snapshot: CodemodSourceSnapshot,
        projection_root: IndexedClass,
        component: TypeKeyedBehaviorProjectionComponent,
    ) -> None:
        declarations = tuple(
            statement
            for statement in projection_root.node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == component.key_attribute_name
        )
        if len(declarations) != 1:
            raise ValueError(
                "type-keyed behavior root lacks one annotated registry-key contract"
            )
        declaration = declarations[0]
        annotation = declaration.annotation
        binding_authority = ModuleNominalBindingAuthority(
            snapshot.parsed_module_for_source_path(projection_root.file_path)
        )
        if not (
            isinstance(annotation, ast.Subscript)
            and binding_authority.qualified_name_at(
                annotation.value,
                line=declaration.lineno,
            )
            == "typing.ClassVar"
            and isinstance(annotation.slice, ast.Subscript)
            and isinstance(annotation.slice.value, ast.Name)
            and annotation.slice.value.id == "type"
            and binding_authority.snapshot_before(
                declaration.lineno
            ).resolves_unshadowed_builtin("type")
        ):
            raise ValueError(
                "registry key annotation does not prove ClassVar[type[Target]]"
            )
        resolver = ModuleClassReferenceResolver(
            snapshot.parsed_module_for_source_path(projection_root.file_path),
            snapshot.required_class_family_index,
        )
        if (
            resolver.symbol_for_reference(annotation.slice.slice)
            != component.target_root.symbol
        ):
            raise ValueError(
                "registry key annotation no longer names the target type authority"
            )

    @staticmethod
    def _required_mro_lookup_method(
        projection_root: IndexedClass,
        component: TypeKeyedBehaviorProjectionComponent,
    ) -> ast.FunctionDef:
        candidates = tuple(
            method
            for method in projection_root.node.body
            if isinstance(method, ast.FunctionDef)
            and method.name
            in component.projection_root.autoregister_registry_projection_names
            if _TypeKeyedBehaviorSourceDerivation._has_mro_lookup_shape(method)
        )
        if len(candidates) != 1:
            raise ValueError(
                "type-keyed behavior descent requires one MRO-aware registry lookup"
            )
        return candidates[0]

    @staticmethod
    def _has_mro_lookup_shape(method: ast.FunctionDef) -> bool:
        """Select syntax only; native identity and behavior are separate obligations."""
        parameters = (*method.args.posonlyargs, *method.args.args)
        if len(parameters) < 2 or len(method.decorator_list) != 1:
            return False
        cls_name, subject_name = parameters[0].arg, parameters[1].arg
        body = statements_without_docstring(method.body)
        if not (
            len(body) == 2
            and isinstance(body[0], ast.Assign)
            and len(body[0].targets) == 1
            and isinstance(body[0].targets[0], ast.Name)
            and isinstance(body[0].value, ast.Call)
            and isinstance(body[1], ast.Return)
        ):
            return False
        result_name = body[0].targets[0].id
        lookup_call = body[0].value
        return bool(
            len(lookup_call.args) == 2
            and not lookup_call.keywords
            and isinstance(lookup_call.args[0], ast.Attribute)
            and isinstance(lookup_call.args[0].value, ast.Name)
            and lookup_call.args[0].value.id == cls_name
            and lookup_call.args[0].attr == REGISTRY_ATTRIBUTE_NAME
            and isinstance(lookup_call.args[1], ast.Call)
            and len(lookup_call.args[1].args) == 1
            and not lookup_call.args[1].keywords
            and isinstance(lookup_call.args[1].args[0], ast.Name)
            and lookup_call.args[1].args[0].id == subject_name
            and _TypeKeyedBehaviorSourceDerivation._returns_optional_instance(
                body[1],
                result_name=result_name,
            )
        )

    @staticmethod
    def _returns_optional_instance(
        statement: ast.Return,
        *,
        result_name: str,
    ) -> bool:
        value = statement.value
        return bool(
            isinstance(value, ast.IfExp)
            and isinstance(value.test, ast.Compare)
            and isinstance(value.test.left, ast.Name)
            and value.test.left.id == result_name
            and len(value.test.ops) == 1
            and isinstance(value.test.ops[0], ast.IsNot)
            and len(value.test.comparators) == 1
            and isinstance(value.test.comparators[0], ast.Constant)
            and value.test.comparators[0].value is None
            and isinstance(value.body, ast.Call)
            and isinstance(value.body.func, ast.Name)
            and value.body.func.id == result_name
            and not value.body.args
            and not value.body.keywords
            and isinstance(value.orelse, ast.Constant)
            and value.orelse.value is None
        )

    @staticmethod
    def _facades(
        root: ast.ClassDef,
        *,
        lookup_method_name: str,
        behavior_method_names: frozenset[str],
    ) -> tuple[_TypeKeyedBehaviorFacade, ...]:
        return tuple(
            facade
            for statement in root.body
            if isinstance(statement, ast.FunctionDef)
            if (
                facade := _TypeKeyedBehaviorSourceDerivation._facade(
                    statement,
                    lookup_method_name=lookup_method_name,
                    behavior_method_names=behavior_method_names,
                )
            )
            is not None
        )

    @staticmethod
    def _facade(
        method: ast.FunctionDef,
        *,
        lookup_method_name: str,
        behavior_method_names: frozenset[str],
    ) -> _TypeKeyedBehaviorFacade | None:
        parameters = (*method.args.posonlyargs, *method.args.args)
        body = statements_without_docstring(method.body)
        if len(parameters) != 2 or len(body) != 3:
            return None
        cls_name, subject_name = parameters[0].arg, parameters[1].arg
        sequence = _ProjectionLookupSequence.from_statements(
            body,
            lookup_method_name=lookup_method_name,
            lookup_receiver_matches=lambda receiver: (
                isinstance(receiver, ast.Name) and receiver.id == cls_name
            ),
            behavior_method_names=behavior_method_names,
        )
        if not (
            sequence is not None
            and isinstance(sequence.subject, ast.Name)
            and sequence.subject.id == subject_name
        ):
            return None
        return _TypeKeyedBehaviorFacade(method.name, sequence.behavior_method_name)

    def source_edits(self) -> tuple[NominalSourceEdit, ...]:
        self.require_native_uses()
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]] = defaultdict(
            list
        )
        deleted_spans_by_path = self._deleted_family_spans(replacements_by_path)
        self._method_insertions(replacements_by_path)
        consumer_spans_by_path = self._consumer_replacements(
            replacements_by_path,
            deleted_spans_by_path=deleted_spans_by_path,
        )
        allowed_spans_by_path = {
            file_path: (
                *deleted_spans_by_path.get(file_path, ()),
                *consumer_spans_by_path.get(file_path, ()),
            )
            for file_path in set(deleted_spans_by_path) | set(consumer_spans_by_path)
        }
        self._require_closed_family_references(allowed_spans_by_path)
        self._unused_import_replacements(
            replacements_by_path,
            deleted_spans_by_path=deleted_spans_by_path,
        )
        return tuple(
            edit
            for file_path, replacements in sorted(replacements_by_path.items())
            for edit in SourceTextGeometry(
                self.snapshot.sources_by_file_path[file_path]
            ).physical_edits(
                file_path=file_path,
                replacements=replacements,
                rationale=self.rationale
                or "Descend type-keyed behavior to its nominal type authority.",
            )
        )

    def _family_classes(self) -> tuple[IndexedClass, ...]:
        family_index = self.snapshot.required_class_family_index
        descendant_symbols = family_index.descendant_symbols(
            self.component.projection_root.symbol
        )
        expected_symbols = frozenset(
            binding.projection_class.symbol for binding in self.component.bindings
        )
        if frozenset(descendant_symbols) != expected_symbols:
            raise ValueError(
                "projection family contains declarations outside the proved type bindings"
            )
        family = (
            self.projection_root,
            *(family_index.class_for(symbol) for symbol in descendant_symbols),
        )
        if any(indexed_class is None for indexed_class in family):
            raise ValueError("projection family declaration is incomplete")
        return cast(tuple[IndexedClass, ...], family)

    def _deleted_family_spans(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> dict[str, tuple[SourceTextSpan, ...]]:
        spans_by_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        for indexed_class in self._family_classes():
            geometry = SourceTextGeometry(
                self.snapshot.sources_by_file_path[indexed_class.file_path]
            )
            offsets = geometry.node_span_offsets(
                SourceNodeSpan(
                    indexed_class.node,
                    Include,
                )
            )
            trailing_separator = re.match(
                r"(?:[ \t]*\r?\n)*",
                geometry.source[offsets[1] :],
            )
            span = SourceTextSpan(
                offsets[0],
                offsets[1]
                + (0 if trailing_separator is None else trailing_separator.end()),
            )
            spans_by_path[indexed_class.file_path].append(span)
            replacements_by_path[indexed_class.file_path].append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    replacement_source="",
                )
            )
        return {file_path: tuple(spans) for file_path, spans in spans_by_path.items()}

    def _method_insertions(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
    ) -> None:
        family_index = self.snapshot.required_class_family_index
        parsed_modules = {
            module.file_path: module for module in self.snapshot.parsed_modules
        }
        for binding in self.component.bindings:
            projection_class = family_index.class_for(binding.projection_class.symbol)
            target_class = family_index.class_for(binding.target_class.symbol)
            if projection_class is None or target_class is None:
                raise ValueError("type-keyed behavior binding lost a class declaration")
            methods_by_name = {
                statement.name: statement
                for statement in projection_class.node.body
                if isinstance(statement, ast.FunctionDef)
            }
            target_method_names = frozenset(
                statement.name
                for statement in FunctionDefinition.within(target_class.node.body)
            )
            collisions = sorted_tuple(
                target_method_names.intersection(self.component.behavior_method_names)
            )
            if collisions:
                raise ValueError(
                    f"target {target_class.simple_name!r} already owns methods {collisions!r}"
                )
            member_sources = tuple(
                _TypeKeyedBehaviorMethodDescent(
                    projection_method=methods_by_name[method_name],
                    target_class=target_class,
                    source_module=parsed_modules[projection_class.file_path],
                    target_module=parsed_modules[target_class.file_path],
                    class_family_index=family_index,
                    projection_class=projection_class,
                ).transformed_source()
                for method_name in self.component.behavior_method_names
                if method_name in methods_by_name
            )
            if len(member_sources) != len(self.component.behavior_method_names):
                raise ValueError(
                    f"projection leaf {projection_class.simple_name!r} lost behavior methods"
                )
            target_source = self.snapshot.sources_by_file_path[target_class.file_path]
            insertion_point = ClassBodySourceAuthority(
                node=target_class.node,
                source=target_source,
            )
            replacements_by_path[target_class.file_path].append(
                insertion_point.member_insertion_replacement(member_sources)
            )

    def _consumer_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
        *,
        deleted_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> dict[str, tuple[SourceTextSpan, ...]]:
        spans_by_path: dict[str, list[SourceTextSpan]] = defaultdict(list)
        facade_names = {
            facade.facade_method_name: facade.behavior_method_name
            for facade in self.facades
        }
        for module in self.snapshot.parsed_modules:
            geometry = SourceTextGeometry(module.source)
            deleted_spans = deleted_spans_by_path.get(module.file_path, ())
            resolver = ModuleClassReferenceResolver(
                module,
                self.snapshot.required_class_family_index,
            )
            for node in ast.walk(module.module):
                if not isinstance(node, ast.Call):
                    continue
                offsets = geometry.required_node_offsets(node)
                if self._offsets_within_any(offsets, deleted_spans):
                    continue
                behavior_method_name = self._direct_facade_behavior(
                    node,
                    resolver=resolver,
                    facade_names=facade_names,
                )
                if behavior_method_name is None:
                    continue
                replacement_call = ast.Call(
                    func=ast.Attribute(
                        value=copy.deepcopy(node.args[0]),
                        attr=behavior_method_name,
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                span = SourceTextSpan.from_offsets(offsets)
                spans_by_path[module.file_path].append(span)
                replacements_by_path[module.file_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        replacement_source=ast.unparse(replacement_call),
                    )
                )
            self._local_lookup_sequence_replacements(
                module,
                resolver=resolver,
                geometry=geometry,
                deleted_spans=deleted_spans,
                replacements=replacements_by_path[module.file_path],
                spans=spans_by_path[module.file_path],
            )
        return {file_path: tuple(spans) for file_path, spans in spans_by_path.items()}

    def _direct_facade_behavior(
        self,
        call: ast.Call,
        *,
        resolver: ModuleClassReferenceResolver,
        facade_names: Mapping[str, str],
    ) -> str | None:
        if not (
            isinstance(call.func, ast.Attribute)
            and len(call.args) == 1
            and not call.keywords
            and resolver.symbol_for_reference(call.func.value)
            == self.component.projection_root.symbol
        ):
            return None
        return facade_names.get(call.func.attr)

    def _local_lookup_sequence_replacements(
        self,
        module: ParsedModule,
        *,
        resolver: ModuleClassReferenceResolver,
        geometry: SourceTextGeometry,
        deleted_spans: tuple[SourceTextSpan, ...],
        replacements: list[SourceTextSpanReplacement],
        spans: list[SourceTextSpan],
    ) -> None:
        for function in FunctionDefinition.within(ast.walk(module.module)):
            function_offsets = geometry.required_node_offsets(function)
            if self._offsets_within_any(function_offsets, deleted_spans):
                continue
            body = function.body
            for index in range(len(body) - 2):
                sequence = body[index : index + 3]
                replacement_source = self._local_lookup_sequence_source(
                    sequence,
                    resolver=resolver,
                )
                if replacement_source is None:
                    continue
                start_offset, _ = geometry.node_span_offsets(
                    SourceNodeSpan(sequence[0])
                )
                _, end_offset = geometry.node_span_offsets(SourceNodeSpan(sequence[-1]))
                span = SourceTextSpan(start_offset, end_offset)
                if span.contains_comment(module.source):
                    raise ValueError(
                        "type-keyed behavior consumer sequence contains comments"
                    )
                spans.append(span)
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        replacement_source=replacement_source,
                    )
                )

    def _local_lookup_sequence_source(
        self,
        sequence: list[ast.stmt],
        *,
        resolver: ModuleClassReferenceResolver,
    ) -> str | None:
        relation = _ProjectionLookupSequence.from_statements(
            sequence,
            lookup_method_name=self.lookup_method_name,
            lookup_receiver_matches=lambda receiver: (
                resolver.symbol_for_reference(receiver)
                == self.component.projection_root.symbol
            ),
            behavior_method_names=frozenset(self.component.behavior_method_names),
        )
        if relation is None:
            return None
        return (
            f"{' ' * relation.statements[0].col_offset}"
            f"return {relation.direct_call_source}\n"
        )

    @staticmethod
    def _offsets_within_any(
        offsets: tuple[int, int],
        spans: tuple[SourceTextSpan, ...],
    ) -> bool:
        start_offset, end_offset = offsets
        return any(
            span.start_offset <= start_offset and end_offset <= span.end_offset
            for span in spans
        )

    def _require_closed_family_references(
        self,
        allowed_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> None:
        family_symbols = frozenset(
            indexed_class.symbol for indexed_class in self._family_classes()
        )
        family_names = frozenset(symbol.rsplit(".", 1)[-1] for symbol in family_symbols)
        for module in self.snapshot.parsed_modules:
            resolver = ModuleClassReferenceResolver(
                module,
                self.snapshot.required_class_family_index,
            )
            geometry = SourceTextGeometry(module.source)
            allowed_spans = allowed_spans_by_path.get(module.file_path, ())
            for node in ast.walk(module.module):
                if isinstance(node, ast.Name | ast.Attribute):
                    symbol = resolver.symbol_for_reference(node)
                    if symbol not in family_symbols:
                        continue
                    offsets = geometry.required_node_offsets(node)
                    if not self._offsets_within_any(offsets, allowed_spans):
                        raise ValueError(
                            f"projection family reference remains at "
                            f"{module.file_path}:{node.lineno}"
                        )
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in family_names
                ):
                    offsets = geometry.required_node_offsets(node)
                    if not self._offsets_within_any(offsets, allowed_spans):
                        raise ValueError(
                            f"string reference to projection family remains at "
                            f"{module.file_path}:{node.lineno}"
                        )

    def _unused_import_replacements(
        self,
        replacements_by_path: dict[str, list[SourceTextSpanReplacement]],
        *,
        deleted_spans_by_path: dict[str, tuple[SourceTextSpan, ...]],
    ) -> None:
        family_symbols = frozenset(
            indexed_class.symbol for indexed_class in self._family_classes()
        )
        compact_projections = CompactModuleClassProjectionFamily.collect_modules(
            self.snapshot.parsed_modules
        )
        compact_projection_by_path = {
            projection.file_path: projection for projection in compact_projections
        }
        for module in self.snapshot.parsed_modules:
            geometry = SourceTextGeometry(module.source)
            deleted_names = frozenset(
                node.id
                for node in ast.walk(module.module)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                if self._offsets_within_any(
                    geometry.required_node_offsets(node),
                    deleted_spans_by_path.get(module.file_path, ()),
                )
            )
            imported_family_names = frozenset(
                local_name
                for local_name, target_symbol in compact_projection_by_path[
                    module.file_path
                ].import_aliases
                if target_symbol in family_symbols
            )
            candidate_names = deleted_names | imported_family_names
            if not candidate_names:
                continue
            primary_replacements = tuple(replacements_by_path.get(module.file_path, ()))
            intermediate_source = geometry.source_with_replacements_in_span(
                0,
                geometry.end_offset,
                primary_replacements,
            )
            intermediate_module = ast.parse(
                intermediate_source, filename=module.file_path
            )
            remaining_loaded_names = frozenset(
                node.id
                for node in ast.walk(intermediate_module)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
            removable_names = candidate_names - remaining_loaded_names
            for statement in module.module.body:
                if not isinstance(statement, ast.ImportFrom):
                    continue
                remaining_aliases = tuple(
                    alias
                    for alias in statement.names
                    if (alias.asname or alias.name) not in removable_names
                )
                if len(remaining_aliases) == len(statement.names):
                    continue
                offsets = geometry.node_span_offsets(SourceNodeSpan(statement))
                module_name = ImportFromModuleName.from_node(statement).source
                replacement_source = ImportFromSource(
                    module_name,
                    remaining_aliases,
                ).source
                replacements_by_path[module.file_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=offsets[0],
                        end_offset=offsets[1],
                        replacement_source=replacement_source,
                    )
                )


@dataclass(frozen=True, kw_only=True)
class Typed(Descent, ABC):
    """Input: behavior in a projection keyed by the family's own classes."""

    supported_execution: DeclaredNativeUseInvariants = codemod_payload_field(
        PayloadRecordValueCodec(DeclaredNativeUseInvariants),
        default_factory=DeclaredNativeUseInvariants,
    )

    def preflight_reports(
        self, context: CodemodSelectorContext
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            resolutions = self.required_reproof(
                lambda: self.source_derivation(
                    context.execution_snapshot()
                ).native_use_resolutions
            )
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return tuple(resolution.preflight_report() for resolution in resolutions)

    def native_use_requirements(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NativeUseRequirement, ...]:
        """Expose exact unmet requirements without authoring their acceptance."""
        return self.source_derivation(snapshot).native_use_requirements

    def required_derivation(
        self, snapshot: CodemodSourceSnapshot
    ) -> _TypeKeyedBehaviorSourceDerivation:
        derivation = self.source_derivation(snapshot)
        derivation.require_native_uses()
        return derivation

    def source_derivation(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> _TypeKeyedBehaviorSourceDerivation:
        _target_identifier, target, _node = self.target_node_from_context(snapshot)
        if not target.is_class:
            raise ValueError("type-keyed behavior projection target must be a class")
        return _TypeKeyedBehaviorSourceDerivation.required(
            snapshot,
            snapshot.source_index.symbol_for_target(target),
            operation=self,
        )


@dataclass(frozen=True, kw_only=True)
class TypedMirror(Mirror, Typed, Descent):
    """A type-keyed projection restates its family: its behavior descends onto it."""

    def synthesis_proof(self, snapshot: CodemodSourceSnapshot) -> None:
        """Unmet native uses stay preflight requirements, not a refusal."""

        self.source_derivation(snapshot)

    def source_edits_from_snapshot(
        self,
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[NominalSourceEdit, ...]:
        return self.required_derivation(snapshot).source_edits()

    def family_root(self, snapshot: CodemodSourceSnapshot) -> FamilyRoot:
        root = self.required_derivation(snapshot).component.target_root
        return FamilyRoot(root.file_path, root.qualname)
