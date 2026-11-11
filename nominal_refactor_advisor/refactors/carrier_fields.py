from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.refactors.class_member_authority import (
    ClassDeclarationPromotionStatement,
    _class_base_source_names,
)
from nominal_refactor_advisor.codemod_payload import (
    CodemodPayloadRecord,
    PayloadRecordArrayValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.module_node_index import ModuleNodeIndex
from nominal_refactor_advisor.refactors.answered_method import AnsweredField
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import (
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_source_edits import (
    PhysicalSourceEdit,
    SourceNodeSpan,
    SourceTargetEditor,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.collection_algebra import UniqueIdentityIndexAuthority
from nominal_refactor_advisor.enum_semantics import PYTHON_ENUM_BASE_AUTHORITY


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
class ReplaceFieldsWithCarrierOperation(RepositorySourceReprovedOperation):
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
        """Every construction of the class, in any module, passes the projected
        fields off one carrier value and becomes `carrier=value`; every read of a
        projected field on a receiver typed to the class becomes a read through
        the carrier. A construction or a read this proof cannot rewrite refuses."""

        _target_identifier, target, class_node = self.target_node_from_context(context)
        if not isinstance(class_node, ast.ClassDef):
            raise ValueError("Field carrier replacement requires a class target")
        target_class = ResolvedClassTarget(target, class_node)
        target_symbol = target_class.required_symbol(context)
        if context.required_class_family_index.descendant_symbols(target_symbol):
            raise ValueError(
                f"`{target.qualname}` has descendants, which build or read its fields too"
            )
        declarations = {
            self.field_name_for_statement(statement): statement
            for statement in class_node.body
            if self.field_name_for_statement(statement) in self.field_projection_map
        }
        fields = tuple(
            AnsweredField(context, target_class, declarations[name])
            for name in self.field_projection_map
            if name in declarations
        )
        if not fields:
            raise ValueError(f"`{target.qualname}` declares none of the projected fields")
        geometries: dict[str, SourceTextGeometry] = {}

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(context.sources_by_file_path[file_path])
            )

        replacements: dict[str, list[SourceTextSpanReplacement]] = {
            target.file_path: list(self.class_field_replacements(class_node, geometry(target.file_path)))
        }
        for parsed in context.parsed_modules:
            if class_node.name not in parsed.source:
                continue
            parents = parsed.fact(ModuleNodeIndex).parents
            for call in (node for node in ast.walk(parsed.module) if isinstance(node, ast.Call)):
                if fields[0].class_symbol(parsed.file_path, call.func, parents) != target_symbol:
                    continue
                replacements.setdefault(parsed.file_path, []).extend(
                    self.construction_replacements(call, geometry(parsed.file_path), parsed.file_path)
                )
        reached = fields[0].reaching([reference for field in fields for reference in field.references()])
        for field in fields:
            for use in field.answered_uses(
                [(receiver, binding) for receiver, binding in reached if receiver.site.attribute.attr == field.name]
            ):
                if not isinstance(use.attribute.ctx, ast.Load):
                    raise ValueError(f"{use.where} writes the projected field {field.name!r}")
                start, end = geometry(use.file_path).required_node_offsets(use.attribute)
                replacements.setdefault(use.file_path, []).append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=(
                            f"{ast.unparse(use.attribute.value)}.{self.carrier_field_name}."
                            f"{self.field_projection_map[field.name]}"
                        ),
                    )
                )
        return tuple(
            edit
            for file_path, file_replacements in replacements.items()
            for edit in geometry(file_path).physical_edits(
                file_path=file_path,
                replacements=tuple(file_replacements),
                rationale=self.rationale
                or (
                    f"Replace projected fields on {target.qualname!r} with carrier "
                    f"field {self.carrier_field_name!r}."
                ),
            )
        )

    def construction_replacements(
        self, call: ast.Call, geometry: SourceTextGeometry, file_path: str
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """One construction with its projected keywords replaced by the carrier: line
        by line when each keyword owns its line, else the argument list in place."""

        where = f"{file_path}:{call.lineno}"
        if call.args:
            raise ValueError(f"{where} constructs the class by position; keyword it first")
        projected = tuple(keyword for keyword in call.keywords if keyword.arg in self.field_projection_map)
        if len(projected) != len(self.field_projection_map):
            raise ValueError(f"{where} does not pass every projected field")
        carrier = self.projected_keyword_carrier_source(projected, geometry)
        if carrier is None:
            raise ValueError(f"{where} does not read every projected field off one carrier value")
        lines = [keyword.value.lineno for keyword in call.keywords]
        if len(set(lines)) == len(lines) and call.lineno not in lines:
            first, *rest = projected
            indent = geometry.line_indent(self.node_start_offset(geometry, first.value))
            return (
                self.line_span_replacement(
                    geometry, first.value, f"{indent}{self.carrier_field_name}={carrier},\n"
                ),
                *(self.line_span_replacement(geometry, keyword.value, "") for keyword in rest),
            )
        replacement = self.carrier_argument_list(call, projected, carrier, geometry)
        if replacement is None:
            raise ValueError(f"{where} has an argument whose source this edit cannot keep")
        return (replacement,)

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

    def carrier_argument_list(
        self,
        call: ast.Call,
        projected_keywords: tuple[ast.keyword, ...],
        carrier_source: str,
        geometry: SourceTextGeometry,
    ) -> SourceTextSpanReplacement | None:
        """The call's arguments with the projected keywords replaced, in place
        of the first of them, by one carrier keyword."""

        projected = {id(keyword) for keyword in projected_keywords}
        parts: list[str] = []
        for argument in call.args:
            if (segment := geometry.segment_for_node(argument)) is None:
                return None
            parts.append(segment)
        for keyword in call.keywords:
            if keyword is projected_keywords[0]:
                parts.append(f"{self.carrier_field_name}={carrier_source}")
                continue
            if id(keyword) in projected:
                continue
            if (segment := geometry.segment_for_node(keyword.value)) is None:
                return None
            parts.append(
                f"**{segment}" if keyword.arg is None else f"{keyword.arg}={segment}"
            )
        _, func_end = geometry.required_node_offsets(call.func)
        _, call_end = geometry.required_node_offsets(call)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=func_end,
            end_offset=call_end,
            replacement_source=f"({', '.join(parts)})",
        )

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
