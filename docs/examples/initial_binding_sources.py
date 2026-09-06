"""Join declaration-owned entry parameters with positioned source writes."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

FLOW = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
REPOSITORY = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)


def member(module: SourceRewriteTarget, name: str) -> SourceRewriteTarget:
    return replace(module, qualname=name)


def body(name: str, source: str) -> ReplaceFunctionBodyOperation:
    return ReplaceFunctionBodyOperation(
        target=member(FLOW, name), body_source=dedent(source)
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        RenameTopLevelDeclarationAuthorityOperation(
            target=member(FLOW, "CompactBindingMutationResolution"),
            new_name="CompactBindingSource",
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "CompactBindingSource"),
            replacement_source=dedent('''\
                class CompactBindingSource(ABC):
                    """Selected source evidence, with distinct value and callable projections."""

                    @property
                    def mutation(self) -> CompactLexicalMutation | None:
                        return None

                    @property
                    @abstractmethod
                    def target_lookup_violation(self) -> CompactFunctionTargetResolutionViolation | None:
                        """Whether this source cannot select a callable through a body write."""
                        raise NotImplementedError

                    @abstractmethod
                    def value_origin(
                        self, flow: CompactFunctionFlow, reference: LexicalValueReference,
                        visited_mutations: frozenset[CompactLexicalMutation],
                    ) -> CompactValueOriginResolution:
                        raise NotImplementedError
            '''),
        ),
        *(
            PatchTargetOperation(
                target=member(FLOW, name),
                replacements=(
                    SourceTextReplacement(
                        old_source="def violation(self)",
                        new_source="def target_lookup_violation(self)",
                    ),
                ),
            )
            for name in ("ExactCompactBindingMutation", "OpenCompactBindingMutation")
        ),
        DeleteTargetOperation(
            target=member(FLOW, "OpenCompactBindingMutation.mutation")
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "ExactCompactBindingMutation"),
            source=dedent("""\
                def value_origin(
                    self, flow: CompactFunctionFlow, reference: LexicalValueReference,
                    visited_mutations: frozenset[CompactLexicalMutation],
                ) -> CompactValueOriginResolution:
                    mutation = self.selected_mutation
                    possible_origins = flow._possible_alias_origins(
                        reference, flow.mutations_by_root_name[reference.root_name]
                    )
                    if mutation in visited_mutations:
                        return OpenCompactValueOrigin(
                            possible_origins, CompactValueOriginViolation.CYCLIC_ALIAS
                        )
                    alias = flow.exact_aliases_by_binding_mutation.get(mutation)
                    if alias is None:
                        return OpenCompactValueOrigin(
                            possible_origins, CompactValueOriginViolation.INTERVENING_REBINDING
                        )
                    source_resolution = flow._value_origin_for(
                        alias.source, alias.source_position, visited_mutations | {mutation}
                    )
                    return source_resolution.through_alias(reference.attribute_path, mutation)
            """),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "OpenCompactBindingMutation"),
            source=dedent("""\
                def value_origin(
                    self, flow: CompactFunctionFlow, reference: LexicalValueReference,
                    visited_mutations: frozenset[CompactLexicalMutation],
                ) -> CompactValueOriginResolution:
                    return OpenCompactValueOrigin(
                        flow._possible_alias_origins(
                            reference, flow.mutations_by_root_name[reference.root_name]
                        ),
                        CompactValueOriginViolation.AMBIGUOUS_BINDING,
                    )
            """),
        ),
        InsertBeforeTargetOperation(
            target=member(FLOW, "CompactValueOriginViolation"),
            source=dedent('''\
                @dataclass(frozen=True)
                class InitialCompactParameterBinding(CompactBindingSource):
                    """The entry value of the exact parameter declared by the flow owner."""

                    parameter: CompactFunctionParameter

                    @property
                    def target_lookup_violation(self) -> CompactFunctionTargetResolutionViolation:
                        return CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING

                    def value_origin(
                        self, flow: CompactFunctionFlow, reference: LexicalValueReference,
                        visited_mutations: frozenset[CompactLexicalMutation],
                    ) -> CompactValueOriginResolution:
                        return ExactCompactValueOrigin(reference)


            '''),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactFlowOwner"),
            source=dedent("""\
                def initial_binding_for(self, root_name: str) -> CompactBindingSource | None:
                    return None
            """),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactFunctionDeclaration"),
            source=dedent("""\
                def initial_binding_for(self, root_name: str) -> CompactBindingSource | None:
                    return next(
                        (InitialCompactParameterBinding(parameter)
                         for parameter in self.signature.parameters
                         if parameter.name == root_name),
                        None,
                    )
            """),
        ),
        ReplaceFunctionSignatureOperation(
            target=member(FLOW, "CompactFlowOwnerKind.deferred_binding_resolution"),
            signature_suffix="(self, bindings: tuple[CompactBindingSource, ...]) -> CompactBindingSource:",
        ),
        body(
            "CompactFlowOwnerKind.deferred_binding_resolution",
            '''\
            """Completed namespaces select their final binding; closures retain alternatives."""
            if len(bindings) == 1 or not self.is_function_scope:
                return bindings[-1]
            return OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
            )
        ''',
        ),
        ReplaceFunctionSignatureOperation(
            target=member(
                FLOW, "CompactFunctionFlow._binding_resolution_for_mutations"
            ),
            signature_suffix="(self, mutations: tuple[CompactLexicalMutation, ...], use_position: CompactFlowPosition | None, initial_binding: CompactBindingSource | None = None) -> CompactBindingSource | None:",
        ),
        body(
            "CompactFunctionFlow._binding_resolution_for_mutations",
            '''\
            """Select entry or write evidence once for lexical and bound-result queries."""
            if not mutations:
                return initial_binding
            if any(mutation.position.branch_path for mutation in mutations):
                return OpenCompactBindingMutation(
                    CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                )
            if use_position is None:
                return self.owner.kind.deferred_binding_resolution(tuple(
                    binding for binding in chain(
                        (initial_binding,),
                        (ExactCompactBindingMutation(mutation) for mutation in mutations),
                    ) if binding is not None
                ))
            dominating = tuple(
                mutation for mutation in mutations
                if mutation.position.dominates(use_position)
            )
            if not dominating:
                return initial_binding if initial_binding is not None else OpenCompactBindingMutation(
                    CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                )
            return ExactCompactBindingMutation(dominating[-1])
        ''',
        ),
        body(
            "CompactFunctionFlow.binding_resolution_for",
            '''\
            """Select a declared entry binding or positioned write; absence permits outer lookup."""
            return self._binding_resolution_for_mutations(
                self.mutations_by_root_name.get(root_name, ()), use_position,
                self.owner.initial_binding_for(root_name),
            )
        ''',
        ),
        body(
            "CompactFunctionFlow._value_origin_for",
            """\
            selection = self.binding_resolution_for(reference.root_name, use_position)
            if selection is None:
                return ExactCompactValueOrigin(reference)
            return selection.value_origin(self, reference, visited_mutations)
        """,
        ),
        PatchTargetOperation(
            target=member(
                REPOSITORY, "CompactProductFlowRepository._scope_binding_resolution"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source=dedent("""\
                        if context.declaration is not None and root_name in {
                            parameter.name for parameter in context.declaration.signature.parameters
                        }:
                            return OpenCompactFunctionTarget(
                                (".".join((context.owner_symbol, *reference.parts)),),
                                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
                            )

                    """).replace("\n", "\n        ").rstrip(),
                    new_source="",
                ),
                SourceTextReplacement(
                    old_source="if selection.violation is not None:",
                    new_source="if selection.target_lookup_violation is not None:",
                ),
                SourceTextReplacement(
                    old_source="                selection.violation,",
                    new_source="                selection.target_lookup_violation,",
                ),
            ),
        ),
        ReplaceFunctionSignatureOperation(
            target=member(
                FLOW, "CompactFunctionFlow._binding_resolution_for_mutations"
            ),
            signature_suffix="(self, mutations: tuple[CompactLexicalMutation, ...], use_position: CompactFlowPosition | None, root_name: str) -> CompactBindingSource | None:",
        ),
        body(
            "CompactFunctionFlow._binding_resolution_for_mutations",
            '''\
            """Select a positioned write before materialising declaration entry evidence."""
            if any(mutation.position.branch_path for mutation in mutations):
                return OpenCompactBindingMutation(
                    CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                )
            if use_position is not None:
                for mutation in reversed(mutations):
                    if mutation.position.dominates(use_position):
                        return ExactCompactBindingMutation(mutation)
            initial_binding = self.owner.initial_binding_for(root_name)
            if not mutations:
                return initial_binding
            if use_position is None:
                return self.owner.kind.deferred_binding_resolution(tuple(
                    binding for binding in chain(
                        (initial_binding,),
                        (ExactCompactBindingMutation(mutation) for mutation in mutations),
                    ) if binding is not None
                ))
            return initial_binding if initial_binding is not None else OpenCompactBindingMutation(
                CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
            )
        ''',
        ),
        body(
            "CompactFunctionFlow.binding_resolution_for",
            '''\
            """Select a declared entry binding or positioned write; absence permits outer lookup."""
            return self._binding_resolution_for_mutations(
                self.mutations_by_root_name.get(root_name, ()), use_position, root_name
            )
        ''',
        ),
        PatchTargetOperation(
            target=member(FLOW, "CompactFunctionFlow.bound_call_result_for"),
            replacements=(
                SourceTextReplacement(
                    old_source="            use_position,\n        )",
                    new_source="            use_position,\n            reference.root_name,\n        )",
                ),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
