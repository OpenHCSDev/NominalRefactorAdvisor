"""Source proof for factoring repeated dataclass fields into one authority."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations

from .ast_projection import ExpressionReference
from .ast_tools import (
    ModuleAnnotationEvaluationMode,
    ParsedModule,
)
from .class_index import (
    ClassBodyModuleScope,
    ClassBodyNormalForm,
    ClassFamilyIndex,
    CompactDataclassFieldRole,
    IndexedClass,
    ModuleNominalBindingSnapshot,
    build_class_family_index,
)
from .collection_algebra import sorted_tuple
from .foreign_family import FunctionDefinition
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from .native_declarations import DataclassRuntimeDeclaration
from .source_geometry import (
    ClassHeaderSourceSpan,
    SourceCommentLineIndex,
    SourceLineSegmentAuthority,
)


@dataclass(frozen=True)
class ExactDataclassDecorator:
    """One resolved dataclass decorator and its reusable source spelling."""

    frozen: bool
    source: str

    @classmethod
    def from_class(
        cls,
        indexed_class: IndexedClass,
        binding_snapshot: ModuleNominalBindingSnapshot,
        source_segments: SourceLineSegmentAuthority,
        comment_lines: SourceCommentLineIndex,
    ) -> ExactDataclassDecorator:
        decorators = indexed_class.node.decorator_list
        if len(decorators) != 1:
            raise ValueError(
                "Field authority factoring requires one dataclass decorator"
            )
        decorator = decorators[0]
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts = ExpressionReference.decoded(target).attribute_chain
        if parts is None:
            raise ValueError("Dataclass decorator has no nominal reference")
        binding = binding_snapshot.binding_for(parts[0])
        qualified_name = (
            None if binding is None else ".".join((binding.qualified_name, *parts[1:]))
        )
        runtime_declaration = DataclassRuntimeDeclaration.dataclass_decorator_for_name(
            qualified_name
        )
        if runtime_declaration is None:
            raise ValueError("Dataclass decorator is not the standard declaration")

        frozen = False
        if isinstance(decorator, ast.Call):
            if decorator.args or any(
                keyword.arg is None for keyword in decorator.keywords
            ):
                raise ValueError(
                    "Dataclass decorator options must be explicit keywords"
                )
            unsupported = tuple(
                keyword.arg for keyword in decorator.keywords if keyword.arg != "frozen"
            )
            if unsupported:
                raise ValueError(
                    "Field authority factoring does not yet prove dataclass options "
                    f"{unsupported!r}"
                )
            frozen_values = tuple(
                keyword.value
                for keyword in decorator.keywords
                if keyword.arg == "frozen"
            )
            if len(frozen_values) > 1 or (
                frozen_values
                and not (
                    isinstance(frozen_values[0], ast.Constant)
                    and isinstance(frozen_values[0].value, bool)
                )
            ):
                raise ValueError("Dataclass frozen option must be one boolean literal")
            frozen = bool(frozen_values and frozen_values[0].value)

        source = source_segments.segment_for_node(decorator)
        if source is None or "\n" in source or comment_lines.intersects(decorator):
            raise ValueError("Dataclass decorator source is not losslessly reusable")
        return cls(frozen, source)


@dataclass(frozen=True)
class ExactDataclassFieldSemantics:
    """One default-free field's class-body normal form, bound only outside the class."""

    name: str
    digest: str
    normal_form: ClassBodyNormalForm = field(compare=False)

    @classmethod
    def from_statement(
        cls,
        statement: ast.AnnAssign,
        scope: ClassBodyModuleScope,
        class_bound_names: frozenset[str],
        comment_lines: SourceCommentLineIndex,
        annotation_mode: type[ModuleAnnotationEvaluationMode],
    ) -> ExactDataclassFieldSemantics:
        if not isinstance(statement.target, ast.Name) or statement.value is not None:
            raise ValueError("Promoted dataclass fields must be default-free names")
        if comment_lines.intersects(statement) or (
            statement.lineno > 1 and statement.lineno - 1 in comment_lines.comment_lines
        ):
            raise ValueError("Promoted dataclass fields must not carry comments")
        if annotation_mode.annotations_execute_at_declaration:
            raise ValueError(
                "Field authority factoring requires postponed annotation evaluation"
            )
        (normal_form,) = ClassBodyNormalForm.of_assignment(
            statement, scope, class_bound_names
        )
        if normal_form.class_scope_names:
            raise ValueError(
                f"Field {statement.target.id!r} depends on class-local binding "
                f"{normal_form.class_scope_names[0]!r}"
            )
        unbound = (
            *normal_form.unresolved_names,
            *(deferred.name for deferred in normal_form.deferred_names),
        )
        if unbound:
            raise ValueError(
                f"Field {statement.target.id!r} has unresolved annotation "
                f"binding {unbound[0]!r}"
            )
        return cls(statement.target.id, normal_form.digest, normal_form)


@dataclass(frozen=True)
class ExactDataclassFieldParticipant:
    """One dataclass whose leading fields can descend from a shared authority."""

    indexed_class: IndexedClass
    decorator: ExactDataclassDecorator
    fields: tuple[ExactDataclassFieldSemantics, ...]


@dataclass(frozen=True)
class ExactDataclassFieldAuthorityComponent:
    """One source-proved repeated field prefix without an existing owner."""

    participants: tuple[ExactDataclassFieldParticipant, ...]
    fields: tuple[ExactDataclassFieldSemantics, ...]

    def __post_init__(self) -> None:
        if len(self.participants) < 2 or len(self.fields) < 2:
            raise ValueError(
                "Exact dataclass field authority requires repeated field structure"
            )
        file_paths = {item.indexed_class.file_path for item in self.participants}
        if len(file_paths) != 1:
            raise ValueError("Exact dataclass field authority must be module-local")
        if any(
            participant.fields[: len(self.fields)] != self.fields
            for participant in self.participants
        ):
            raise ValueError("Exact dataclass field authority has a divergent prefix")

    @property
    def file_path(self) -> str:
        return self.participants[0].indexed_class.file_path

    @property
    def participant_class_names(self) -> tuple[str, ...]:
        return tuple(item.indexed_class.qualname for item in self.participants)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def evidence_field_name(self) -> str:
        return self.fields[-1].name

    @property
    def decorator_source(self) -> str:
        return self.participants[0].decorator.source


@dataclass(frozen=True)
class ExactDataclassFieldAuthorityComponentBuilder:
    """Derive repeated leading-field components from current source."""

    parsed_modules: tuple[ParsedModule, ...]
    class_index: ClassFamilyIndex

    @classmethod
    def from_modules(
        cls,
        parsed_modules: tuple[ParsedModule, ...],
        *,
        class_index: ClassFamilyIndex | None = None,
    ) -> ExactDataclassFieldAuthorityComponentBuilder:
        return cls(
            parsed_modules=parsed_modules,
            class_index=(
                build_class_family_index(list(parsed_modules))
                if class_index is None
                else class_index
            ),
        )

    @cached_property
    def participants(self) -> tuple[ExactDataclassFieldParticipant, ...]:
        participants = []
        classes_by_path: dict[str, list[IndexedClass]] = {}
        for indexed_class in self.class_index.classes_by_symbol.values():
            classes_by_path.setdefault(indexed_class.file_path, []).append(
                indexed_class
            )
        for parsed_module in self.parsed_modules:
            indexed_classes = tuple(classes_by_path.get(parsed_module.file_path, ()))
            scopes = ClassBodyModuleScope.for_lines(
                parsed_module, (item.line for item in indexed_classes)
            )
            source_segments = SourceLineSegmentAuthority(parsed_module.source)
            comment_lines = SourceCommentLineIndex.from_source(parsed_module.source)
            annotation_mode = ModuleAnnotationEvaluationMode.from_module(
                parsed_module.module
            )
            for indexed_class in indexed_classes:
                try:
                    participant = self._participant(
                        indexed_class,
                        scopes[indexed_class.line],
                        source_segments,
                        comment_lines,
                        annotation_mode,
                    )
                except ValueError:
                    continue
                participants.append(participant)
        return sorted_tuple(
            participants,
            key=lambda item: (
                item.indexed_class.file_path,
                item.indexed_class.line,
                item.indexed_class.qualname,
            ),
        )

    @cached_property
    def proven_components(self) -> tuple[ExactDataclassFieldAuthorityComponent, ...]:
        candidate_keys: set[
            tuple[
                str,
                bool,
                tuple[ExactDataclassFieldSemantics, ...],
            ]
        ] = set()
        for left, right in combinations(self.participants, 2):
            if (
                left.indexed_class.file_path != right.indexed_class.file_path
                or left.decorator.frozen != right.decorator.frozen
            ):
                continue
            common_prefix = _common_prefix(
                left.fields,
                right.fields,
            )
            if len(common_prefix) < 2:
                continue
            candidate_keys.add(
                (
                    left.indexed_class.file_path,
                    left.decorator.frozen,
                    common_prefix,
                )
            )

        components = []
        for file_path, frozen, prefix in candidate_keys:
            participants = tuple(
                participant
                for participant in self.participants
                if participant.indexed_class.file_path == file_path
                and participant.decorator.frozen == frozen
                and participant.fields[: len(prefix)] == prefix
            )
            if len(participants) < 2:
                continue
            components.append(
                ExactDataclassFieldAuthorityComponent(
                    participants=participants,
                    fields=participants[0].fields[: len(prefix)],
                )
            )
        return sorted_tuple(
            components,
            key=lambda component: (
                component.file_path,
                component.participants[0].indexed_class.line,
                component.field_names,
                component.participant_class_names,
            ),
        )

    def required_component_for_field(
        self,
        *,
        file_path: str,
        class_qualname: str,
        field_name: str,
    ) -> ExactDataclassFieldAuthorityComponent:
        components = tuple(
            component
            for component in self.proven_components
            if component.file_path == file_path
            and class_qualname in component.participant_class_names
            and component.evidence_field_name == field_name
        )
        if len(components) != 1:
            raise ValueError(
                f"Field {class_qualname}.{field_name} belongs to {len(components)} "
                "current exact dataclass field authority components"
            )
        return components[0]

    def _participant(
        self,
        indexed_class: IndexedClass,
        scope: ClassBodyModuleScope,
        source_segments: SourceLineSegmentAuthority,
        comment_lines: SourceCommentLineIndex,
        annotation_mode: type[ModuleAnnotationEvaluationMode],
    ) -> ExactDataclassFieldParticipant:
        node = indexed_class.node
        declaration = indexed_class.dataclass_declaration
        if (
            "." in indexed_class.qualname
            or node.bases
            or node.keywords
            or declaration is None
            or not declaration.is_standard_dataclass
            or declaration.failures
            or not ClassHeaderSourceSpan(node, source_segments.lines).is_reconstructible
        ):
            raise ValueError("Dataclass is not a base-free promotion participant")
        if any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in {"super", "__class__"}
            for statement in FunctionDefinition.within(node.body)
            for child in ast.walk(statement)
        ):
            raise ValueError("Dataclass methods depend on the current MRO")
        if not declaration.fields or any(
            field.role is not CompactDataclassFieldRole.STORED_INIT
            for field in declaration.fields
        ):
            raise ValueError("Dataclass fields do not form a stored-init product")

        direct_fields: list[ast.AnnAssign] = []
        field_phase = True
        for index, statement in enumerate(node.body):
            if (
                index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                if not field_phase:
                    raise ValueError("Dataclass fields are not one leading run")
                direct_fields.append(statement)
                continue
            field_phase = False
        if tuple(field.name for field in declaration.fields) != tuple(
            statement.target.id for statement in direct_fields
        ):
            raise ValueError("Dataclass field declarations are not source-complete")

        decorator = ExactDataclassDecorator.from_class(
            indexed_class,
            scope.declaration,
            source_segments,
            comment_lines,
        )
        class_bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(node.body)
        fields = tuple(
            ExactDataclassFieldSemantics.from_statement(
                statement,
                scope,
                class_bound_names,
                comment_lines,
                annotation_mode,
            )
            for statement in direct_fields
        )
        return ExactDataclassFieldParticipant(indexed_class, decorator, fields)


def _common_prefix(
    left: tuple[ExactDataclassFieldSemantics, ...],
    right: tuple[ExactDataclassFieldSemantics, ...],
) -> tuple[ExactDataclassFieldSemantics, ...]:
    prefix = []
    for left_field, right_field in zip(left, right, strict=False):
        if left_field != right_field:
            break
        prefix.append(left_field)
    return tuple(prefix)
