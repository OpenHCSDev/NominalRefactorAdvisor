from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from ..ast_projection import CallKeywords
from ..ast_tools import AstParentIndex
from ..binding_proofs import FunctionNominalParameterBindingAuthority
from ..class_index import ModuleNominalBindingAuthority
from ..class_member_authority_codemod import (
    ClassDeclarationPromotionStatement,
    _class_base_source_names,
)
from ..codemod import NominalConstructorCall
from ..codemod_payload import (
    CodemodPayloadRecord,
    PayloadRecordArrayValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from ..codemod_reproof import SourceReprovedOperation
from ..codemod_runtime import CodemodSourceSnapshot
from ..codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from ..codemod_source_edits import (
    PhysicalSourceEdit,
    SourceLineSpan,
    SourceNodeSpan,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from ..collection_algebra import UniqueIdentityIndexAuthority
from ..enum_semantics import PYTHON_ENUM_BASE_AUTHORITY


@dataclass(frozen=True)
class ClassDeclarationPromotionClass:
    """Class-level safety checks for declaration promotion."""

    node: ast.ClassDef

    @property
    def is_enum_class(self) -> bool:
        return PYTHON_ENUM_BASE_AUTHORITY.matches_any(
            _class_base_source_names(self.node)
        )


@dataclass(frozen=True)
class CarrierFieldDeclaration:
    """One annotated field declaration to be owned by a generated carrier."""

    source: str

    @property
    def field_name(self) -> str:
        field_statement = self.parsed_field_statement
        if not isinstance(field_statement, ast.AnnAssign):
            raise ValueError(
                "Carrier collapse requires annotated field declarations; "
                f"got {self.source!r}"
            )
        field_name = ClassDeclarationPromotionStatement(field_statement).name
        if field_name is None:
            raise ValueError(
                f"Carrier field declaration has no field name: {self.source!r}"
            )
        return field_name

    @property
    def parsed_field_statement(self) -> ast.stmt:
        module = ast.parse(self.probe_class_source, filename="<carrier-field>")
        if len(module.body) != 1 or not isinstance(module.body[0], ast.ClassDef):
            raise ValueError(f"Invalid carrier field declaration: {self.source!r}")
        body = module.body[0].body
        if len(body) != 1:
            raise ValueError(
                "Carrier field declaration must parse to one class-body statement: "
                f"{self.source!r}"
            )
        return body[0]

    @property
    def probe_class_source(self) -> str:
        return f"class _CarrierFieldProbe:\n{''.join(self.indented_lines)}"

    @property
    def indented_lines(self) -> tuple[str, ...]:
        source_lines = SourceTargetEditor.source_lines(self.source.strip())
        if not source_lines:
            raise ValueError("Carrier field declaration must not be empty")
        return tuple(
            f"    {line.lstrip()}" if line.strip() else line for line in source_lines
        )


@dataclass(frozen=True)
class CarrierFieldProjection(CodemodPayloadRecord):
    """One explicit primitive-field to carrier-attribute relation."""

    source_field: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    carrier_attribute: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def __post_init__(self) -> None:
        if not self.source_field.isidentifier():
            raise ValueError(
                f"Carrier source field must be an identifier: {self.source_field!r}"
            )
        if not self.carrier_attribute.isidentifier():
            raise ValueError(
                "Carrier projection attribute must be an identifier: "
                f"{self.carrier_attribute!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ReplaceFieldsWithCarrierOperation(SourceReprovedOperation):
    """Replace projected primitive fields with one existing carrier field."""

    field_projections: tuple[CarrierFieldProjection, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CarrierFieldProjection)
    )
    carrier_field_declaration: str = codemod_payload_field(
        RequiredStringPayloadValueCodec()
    )

    @property
    def carrier_field(self) -> CarrierFieldDeclaration:
        return CarrierFieldDeclaration(self.carrier_field_declaration)

    @property
    def carrier_field_name(self) -> str:
        return self.carrier_field.field_name

    @property
    def field_projection_map(self) -> Mapping[str, str]:
        if not self.field_projections:
            raise ValueError("Field carrier replacement requires field projections")
        projections = UniqueIdentityIndexAuthority.declaration_multiplicity_by_handle(
            self.field_projections,
            lambda projection: projection.source_field,
        )
        if projections.ambiguous_handles:
            raise ValueError(
                "Carrier source fields have multiple projections: "
                f"{tuple(sorted(projections.ambiguous_handles))!r}"
            )
        return {
            source_field: projection.carrier_attribute
            for source_field, projection in (
                projections.unambiguous_declarations_by_handle.items()
            )
        }

    def source_edits_from_snapshot(
        self,
        context: CodemodSourceSnapshot,
    ) -> tuple[PhysicalSourceEdit, ...]:
        _target_identifier, target, class_node = self.target_node_from_context(context)
        if not isinstance(class_node, ast.ClassDef):
            raise ValueError("Field carrier replacement requires a class target")
        target_class = ResolvedClassTarget(target, class_node)
        target_symbol = target_class.required_symbol(context)
        source_path = target.file_path
        source = context.sources_by_file_path[source_path]
        geometry = SourceTextGeometry(source)
        root = context.module_nodes_by_file_path[source_path]
        replacements = [
            *self.class_field_replacements(class_node, geometry),
            *self.constructor_projection_replacements(
                context,
                source_path,
                root,
                geometry,
                constructor_symbol=target_symbol,
            ),
        ]
        covered_lines = tuple(
            SourceLineSpan.from_offsets(geometry, item.start_offset, item.end_offset)
            for item in replacements
        )
        replacements.extend(
            self.attribute_projection_replacements(
                context,
                source_path,
                root,
                geometry,
                target_symbol=target_symbol,
                covered_lines=covered_lines,
            )
        )
        if not replacements:
            raise ValueError(
                f"Field carrier replacement found no edits in {source_path!r}"
            )
        return geometry.physical_edits(
            file_path=source_path,
            replacements=replacements,
            rationale=self.rationale
            or (
                f"Replace projected fields on {target.qualname!r} with carrier "
                f"field {self.carrier_field_name!r}."
            ),
        )

    def class_field_replacements(
        self,
        class_node: ast.ClassDef,
        geometry: SourceTextGeometry,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        field_lines = tuple(
            statement
            for statement in class_node.body
            if self.field_name_for_statement(statement) in self.field_projection_map
        )
        existing_carrier_field = any(
            self.field_name_for_statement(statement) == self.carrier_field_name
            for statement in class_node.body
        )
        if not field_lines:
            return ()
        first_field = field_lines[0]
        replacements: list[SourceTextSpanReplacement] = []
        if not existing_carrier_field:
            replacements.append(
                self.line_span_replacement(
                    geometry,
                    first_field,
                    "".join(self.carrier_field.indented_lines),
                )
            )
            removed_tail = field_lines[1:]
        else:
            removed_tail = field_lines
        replacements.extend(
            self.line_span_replacement(geometry, statement, "")
            for statement in removed_tail
        )
        return tuple(replacements)

    def constructor_projection_replacements(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        root: ast.Module,
        geometry: SourceTextGeometry,
        *,
        constructor_symbol: str,
    ) -> tuple[SourceTextSpanReplacement, ...]:
        replacements: list[SourceTextSpanReplacement] = []
        parent_index = AstParentIndex(root)
        for call in (node for node in ast.walk(root) if isinstance(node, ast.Call)):
            nominal_call = NominalConstructorCall.from_context(
                context,
                source_path,
                parent_index.enclosing_function(call),
                call,
            )
            if (
                nominal_call is None
                or nominal_call.constructor_symbol != constructor_symbol
            ):
                continue
            projected_keywords = tuple(
                keyword
                for keyword in call.keywords
                if keyword.arg in self.field_projection_map
            )
            if len(projected_keywords) != len(self.field_projection_map):
                continue
            carrier_source = self.projected_keyword_carrier_source(
                projected_keywords,
                geometry,
            )
            if carrier_source is None:
                continue
            first_keyword = projected_keywords[0]
            replacements.append(
                self.line_span_replacement(
                    geometry,
                    first_keyword.value,
                    (
                        f"{geometry.line_indent(self.node_start_offset(geometry, first_keyword.value))}"
                        f"{self.carrier_field_name}={carrier_source},\n"
                    ),
                )
            )
            replacements.extend(
                self.line_span_replacement(geometry, keyword.value, "")
                for keyword in projected_keywords[1:]
            )
        return tuple(replacements)

    def projected_keyword_carrier_source(
        self,
        projected_keywords: tuple[ast.keyword, ...],
        geometry: SourceTextGeometry,
    ) -> str | None:
        carrier_sources: set[str] = set()
        projection_map = self.field_projection_map
        keywords = CallKeywords.of(projected_keywords)
        if keywords.splats:
            return None
        for keyword in keywords.named:
            expected_attribute = projection_map[keyword.name]
            value = keyword.value
            if not isinstance(value, ast.Attribute):
                return None
            if value.attr != expected_attribute:
                return None
            carrier_source = geometry.segment_for_node(value.value)
            if carrier_source is None:
                return None
            carrier_sources.add(carrier_source)
        if len(carrier_sources) != 1:
            return None
        return next(iter(carrier_sources))

    def attribute_projection_replacements(
        self,
        context: CodemodSelectorContext,
        source_path: str,
        root: ast.Module,
        geometry: SourceTextGeometry,
        *,
        target_symbol: str,
        covered_lines: tuple[SourceLineSpan, ...],
    ) -> tuple[SourceTextSpanReplacement, ...]:
        replacements: list[SourceTextSpanReplacement] = []
        projection_map = self.field_projection_map
        carrier_field_name = self.carrier_field_name
        parent_index = AstParentIndex(root)
        module_bindings = ModuleNominalBindingAuthority(
            context.parsed_module_for_source_path(source_path)
        )
        parameter_bindings_by_function: dict[
            ast.FunctionDef | ast.AsyncFunctionDef,
            FunctionNominalParameterBindingAuthority,
        ] = {}
        for attribute in (
            node for node in ast.walk(root) if isinstance(node, ast.Attribute)
        ):
            carrier_attribute = projection_map.get(attribute.attr)
            if carrier_attribute is None:
                continue
            if SourceNodeSpan(attribute).line_span.overlaps_any(covered_lines):
                continue
            function_scope = parent_index.enclosing_function(attribute.value)
            if not isinstance(attribute.value, ast.Name) or function_scope is None:
                continue
            parameter_bindings = parameter_bindings_by_function.setdefault(
                function_scope,
                FunctionNominalParameterBindingAuthority(
                    module_bindings,
                    function_scope,
                ),
            )
            owner_symbol = parameter_bindings.type_name_for_reference(
                attribute.value.id
            )
            if owner_symbol != target_symbol:
                continue
            value_source = geometry.segment_for_node(attribute.value)
            if value_source is None:
                continue
            start_offset, end_offset = geometry.required_node_offsets(attribute)
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    replacement_source=(
                        f"{value_source}.{carrier_field_name}.{carrier_attribute}"
                    ),
                )
            )
        return tuple(replacements)

    @staticmethod
    def field_name_for_statement(statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.AnnAssign):
            return None
        if not isinstance(statement.target, ast.Name):
            return None
        return statement.target.id

    @staticmethod
    def line_span_replacement(
        geometry: SourceTextGeometry,
        node: ast.stmt | ast.expr,
        replacement_source: str,
    ) -> SourceTextSpanReplacement:
        line_span = SourceNodeSpan(node).line_span
        start_offset, end_offset = geometry.line_span_offsets(
            line_span.start_line,
            line_span.end_line,
        )
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=replacement_source,
        )

    @staticmethod
    def node_start_offset(
        geometry: SourceTextGeometry,
        node: ast.stmt | ast.expr,
    ) -> int:
        return geometry.required_node_offsets(node)[0]
