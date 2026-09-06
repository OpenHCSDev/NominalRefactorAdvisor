"""Retain evaluation events through binding and value-origin consumers."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    RemoveImportNamesOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

flow = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
repository = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)
conveyor = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/parameter_conveyor.py",
    qualname="ClosedParameterConveyorComponentBuilder",
)
carrier = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/carrier_expansion.py",
    qualname="DeclaredCarrierExpansionBuilder",
)


def member(target: SourceRewriteTarget, name: str) -> SourceRewriteTarget:
    return replace(
        target, qualname=f"{target.qualname}.{name}" if target.qualname else name
    )


def patch(
    target: SourceRewriteTarget, *changes: tuple[str, str]
) -> PatchTargetOperation:
    return PatchTargetOperation(
        target=target,
        replacements=tuple(
            SourceTextReplacement(old_source=old, new_source=new)
            for old, new in changes
        ),
    )


def body(target: SourceRewriteTarget, source: str) -> ReplaceFunctionBodyOperation:
    return ReplaceFunctionBodyOperation(target=target, body_source=dedent(source))


PLAN = CodemodPlanSequence.from_operations(
    (
        InsertClassMemberOperation(
            target=member(flow, "CompactValueOriginViolation"),
            source='OPAQUE_EXPRESSION = "opaque_expression"',
        ),
        InsertBeforeTargetOperation(
            target=member(flow, "CompactCallableReferenceUse"),
            source=dedent('''\
        @dataclass(frozen=True)
        class CompactValueUse:
            """One evaluated argument value, retaining its source event."""

            value: CompactValueExpression
            position: CompactFlowPosition

            lexical_reference = AliasProperty[LexicalValueReference | None]("value.lexical_reference")

            def origin_in(self, flow: CompactFunctionFlow) -> CompactValueOriginResolution:
                reference = self.lexical_reference
                if reference is None:
                    return OpenCompactValueOrigin((), CompactValueOriginViolation.OPAQUE_EXPRESSION)
                return flow.value_origin_for(reference, self.position)

            def reference_equivalents_in(self, flow: CompactFunctionFlow) -> tuple[LexicalValueReference, ...]:
                return tuple(dict.fromkeys(
                    reference for reference in (self.lexical_reference, self.origin_in(flow).exact_origin)
                    if reference is not None
                ))

        '''),
        ),
        ReplaceFunctionSignatureOperation(
            target=member(flow, "CompactFunctionFlow._value_origin_for"),
            signature_suffix="(self, reference: LexicalValueReference, use_position: CompactFlowPosition, visited_mutations: frozenset[CompactLexicalMutation]) -> CompactValueOriginResolution:",
        ),
        body(
            member(flow, "CompactFunctionFlow._value_origin_for"),
            """\
        selection = self.binding_resolution_for(reference.root_name, use_position)
        if selection is None:
            return ExactCompactValueOrigin(reference)
        possible_origins = self._possible_alias_origins(
            reference, self.mutations_by_root_name[reference.root_name]
        )
        mutation = selection.mutation
        if mutation is None:
            return OpenCompactValueOrigin(possible_origins, CompactValueOriginViolation.AMBIGUOUS_BINDING)
        if mutation in visited_mutations:
            return OpenCompactValueOrigin(possible_origins, CompactValueOriginViolation.CYCLIC_ALIAS)
        alias = self.exact_aliases_by_binding_mutation.get(mutation)
        if alias is None:
            return OpenCompactValueOrigin(possible_origins, CompactValueOriginViolation.INTERVENING_REBINDING)
        source_resolution = self._value_origin_for(
            alias.source, alias.source_position, visited_mutations | {mutation}
        )
        return source_resolution.through_alias(reference.attribute_path, mutation)
        """,
        ),
        patch(
            member(flow, "CompactFunctionCall"),
            (
                "CompactCallArguments[CompactValueExpression]",
                "CompactCallArguments[CompactValueUse]",
            ),
            (
                "CompactCallBinding[CompactValueExpression]",
                "CompactCallBinding[CompactValueUse]",
            ),
        ),
        patch(
            member(flow, "CompactProductConstruction"),
            (
                "CompactKeywordArgument[CompactValueExpression]",
                "CompactKeywordArgument[CompactValueUse]",
            ),
        ),
        InsertClassMemberOperation(
            target=member(flow, "_CompactFlowCollector"),
            source=dedent("""\
        def _capture_argument(self, expression: ast.expr) -> CompactValueUse:
            self.visit(expression)
            return CompactValueUse(CompactValueExpression.project(expression), self._position())
        """),
        ),
        body(
            member(flow, "_CompactFlowCollector.visit_Call"),
            """\
        target_reference = LexicalValueReference.from_expression(node.func)
        if target_reference is not None:
            self.loaded_value_root_names.add(target_reference.root_name)
        self._visit_call_target_evaluation(node.func)
        target_use = self._callable_reference_use(node.func)
        arguments = CompactCallArguments[CompactValueUse].from_call(node, self._capture_argument)
        result = self.call_results.get(id(node), CompactCallResult(CompactCallResultUse.EMBEDDED))
        self.calls.append(CompactFunctionCall(
            target_use=target_use, arguments=arguments, result=result,
            position=self._position(), source_span=SourceByteSpan.require_node(node),
        ))
        """,
        ),
        EnsureImportOperation(
            target=repository, import_source="from .product_flow import CompactValueUse"
        ),
        body(
            member(
                repository, "CompactFunctionCallResolution.argument_origin_resolutions"
            ),
            """\
        return tuple(value.origin_in(self.context.flow) for value in self.call.arguments.values)
        """,
        ),
        patch(
            member(repository, "CompactResolvedFunctionCall.binding"),
            (
                "CompactCallBinding[CompactValueExpression]",
                "CompactCallBinding[CompactValueUse]",
            ),
        ),
        InsertClassMemberOperation(
            target=member(repository, "CompactResolvedFunctionCall"),
            source=dedent('''\
        @cached_property
        def bound_value_uses(self) -> dict[str, CompactValueUse]:
            """Single supplied values selected by the existing binding result."""
            return {
                parameter.name: argument.values[0]
                for parameter in self.call_signature.parameters
                if (argument := self.binding.argument_for(parameter.name)) is not None
                and len(argument.values) == 1
            }
        '''),
        ),
        patch(
            member(
                repository,
                "CompactProductFlowRepository.product_runtime_failures_by_authority_symbol",
            ),
            (
                "context.flow.value_origin_for(\n                        reference,\n                        call.position,\n                    ).exact_origin",
                "value.origin_in(context.flow).exact_origin",
            ),
        ),
        RemoveImportNamesOperation(
            target=repository,
            module_name=".value_expression",
            import_names=("CompactValueExpression",),
        ),
        InsertClassMemberOperation(
            target=member(flow, "CompactProductConstruction"),
            source=dedent("""\
        @cached_property
        def field_values(self) -> dict[str, CompactValueUse]:
            return {argument.name: argument.value for argument in self.field_arguments if argument.name is not None}
        """),
        ),
        body(
            member(flow, "CompactProductConstruction.field_names"),
            "return tuple(self.field_values)",
        ),
        EnsureImportOperation(
            target=conveyor, import_source="from .product_flow import CompactValueUse"
        ),
        *(
            DeleteTargetOperation(target=member(conveyor, name))
            for name in (
                "simple_bound_arguments_by_call",
                "bound_argument_origins_by_call",
                "_simple_bound_arguments",
                "_reference_equivalents",
                "_construction_arguments",
            )
        ),
        body(
            member(conveyor, "calls_by_value_projection"),
            """\
        grouped: dict[_ValueProjectionKey, list[CompactResolvedFunctionCall]] = defaultdict(list)
        for edge in self.repository.resolved_function_calls:
            for value in edge.bound_value_uses.values():
                for projection in value.reference_equivalents_in(edge.context.flow):
                    grouped[(edge.context.owner_symbol, projection)].append(edge)
        return {key: tuple(edges) for key, edges in grouped.items()}
        """,
        ),
        body(
            member(conveyor, "_field_bindings"),
            """\
        bindings: list[CarrierCollapseFieldBinding] = []
        argument_equivalents = {
            name: value.reference_equivalents_in(call_edge.context.flow)
            for name, value in call_edge.bound_value_uses.items()
        }
        for field_name in authority.field_names:
            matches = tuple(
                (parameter_name, reference)
                for parameter_name, value in call_edge.bound_value_uses.items()
                if (reference := value.lexical_reference) is not None
                and expected_references_by_field[field_name] & frozenset(argument_equivalents[parameter_name])
            )
            if len(matches) != 1:
                return None
            parameter_name, value_reference = matches[0]
            bindings.append(CarrierCollapseFieldBinding(
                field_name=field_name, parameter_name=parameter_name, value_reference=value_reference,
            ))
        return tuple(bindings)
        """,
        ),
        *(
            patch(
                member(conveyor, name),
                (
                    "self._construction_arguments(construction)",
                    "construction.construction.field_values",
                ),
            )
            for name in ("root_edges", "_root_call_hazards")
        ),
        patch(
            member(conveyor, "root_edges"),
            (
                "if source_reference is None:",
                "if source_reference.lexical_reference is None:",
            ),
            (
                "self._reference_equivalents(\n                    construction.context,\n                    source_reference,\n                    construction.call.position,\n                )",
                "source_reference.reference_equivalents_in(construction.context.flow)",
            ),
        ),
        patch(
            member(conveyor, "_constructed_edge"),
            (
                "construction_arguments: dict[str, LexicalValueReference | None]",
                "construction_arguments: dict[str, CompactValueUse]",
            ),
            (
                "self._reference_equivalents(\n                                construction.context,\n                                construction_arguments[field_name],\n                                construction.call.position,\n                            )\n                            if construction_arguments[field_name] is not None\n                            else ()",
                "construction_arguments[field_name].reference_equivalents_in(construction.context.flow)",
            ),
            (
                "for reference in construction_arguments.values()\n                if reference is not None",
                "for value in construction_arguments.values()\n                if (reference := value.lexical_reference) is not None",
            ),
        ),
        patch(
            member(conveyor, "_root_call_hazards"),
            (
                "field_name: construction.context.flow.value_origin_for(\n                    reference,\n                    construction.call.position,\n                )",
                "field_name: value.origin_in(construction.context.flow)",
            ),
            (
                "if (reference := construction_arguments[field_name]) is not None",
                "if (value := construction_arguments[field_name]).lexical_reference is not None",
            ),
            (
                "construction_arguments[field_name],",
                "construction_arguments[field_name].lexical_reference,",
            ),
        ),
        *(
            patch(
                member(carrier, name),
                (
                    "for parameter in call.call_signature.parameters:\n            argument = call_binding.argument_for(parameter.name)\n            if argument is None or len(argument.values) != 1:\n                continue",
                    "for parameter_name, argument in call.bound_value_uses.items():",
                ),
                ("argument.values[0].lexical_reference", "argument.lexical_reference"),
            )
            for name in ("_call_expansions", "_forwarded_edge")
        ),
        patch(
            member(carrier, "_call_expansions"),
            ("call.call.position,", "argument.position,"),
            ("parameter_name=parameter.name,", "parameter_name=parameter_name,"),
        ),
        patch(
            member(carrier, "_forwarded_edge"),
            (
                "call.context.flow.value_origin_for(\n                reference,\n                call.call.position,\n            ).exact_origin",
                "argument.origin_in(call.context.flow).exact_origin",
            ),
            (
                "parameters_by_origin[reference].add(parameter.name)",
                "parameters_by_origin[reference].add(parameter_name)",
            ),
            (
                "parameters_by_origin[origin].add(parameter.name)",
                "parameters_by_origin[origin].add(parameter_name)",
            ),
        ),
        RemoveImportNamesOperation(
            target=conveyor,
            module_name=".product_flow",
            import_names=("CompactValueOriginResolution",),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
