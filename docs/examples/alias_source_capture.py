"""Preserve a lexical alias's captured nominal read through resolution."""

from dataclasses import replace
import json
from textwrap import dedent, indent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
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


def method_source(source: str) -> str:
    return indent(dedent(source), "    ")


def patch(
    module: SourceRewriteTarget, name: str, old: str, new: str
) -> PatchTargetOperation:
    return PatchTargetOperation(
        target=member(module, name),
        replacements=(SourceTextReplacement(old_source=old, new_source=new),),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=member(FLOW, "CompactFunctionTargetResolutionViolation"),
            source="CompactBindingVisit: TypeAlias = tuple[str, CompactLexicalMutation]\n\n",
        ),
        PatchTargetOperation(
            target=REPOSITORY,
            replacements=(
                SourceTextReplacement(
                    old_source="CompactBindingVisit: TypeAlias = tuple[str, CompactLexicalMutation]\n",
                    new_source="",
                ),
            ),
        ),
        EnsureImportOperation(
            target=REPOSITORY,
            import_source="from .product_flow import CompactBindingVisit",
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactCallTargetResolverABC"),
            source=dedent('''\
            @abstractmethod
            def _through_attribute_suffix(
                self, resolution: TargetResolutionT, attribute_path: tuple[str, ...],
            ) -> TargetResolutionT:
                """Project attributes of a captured non-lexical target conservatively."""
                raise NotImplementedError
        '''),
        ),
        InsertClassMemberOperation(
            target=member(REPOSITORY, "CompactProductFlowRepository"),
            source=dedent("""\
            def _through_attribute_suffix(
                self, resolution: CompactCallTargetResolution, attribute_path: tuple[str, ...],
            ) -> CompactCallTargetResolution:
                if not attribute_path:
                    return resolution
                return UnboundedCompactFunctionTarget(
                    (), CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER,
                )
        """),
        ),
        patch(
            FLOW,
            "CompactCallTargetResolverABC._lexical_function_target_resolution",
            "        position: CompactFlowPosition,",
            "        position: CompactFlowPosition,\n        pending_bindings: frozenset[CompactBindingVisit] = frozenset(),",
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "CompactCallTargetReference.resolve"),
            replacement_source=method_source('''\
            def resolve(
                self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                context: ResolutionContextT, position: CompactFlowPosition,
                *, pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
                attribute_path: tuple[str, ...] = (),
            ) -> TargetResolutionT:
                """Select the nominal lookup, then project any captured attribute access."""
                return resolver._through_attribute_suffix(
                    self.resolve_target(resolver, context, position), attribute_path,
                )
        '''),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CompactCallTargetReference"),
            source=dedent("""\
            def resolve_target(
                self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                context: ResolutionContextT, position: CompactFlowPosition,
            ) -> TargetResolutionT:
                return resolver._local_function_target_resolution(context, self)
        """),
        ),
        patch(
            FLOW,
            "CurrentClassCallTargetReference.resolve",
            "def resolve(",
            "def resolve_target(",
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "LexicalCallTargetReference.resolve"),
            replacement_source=method_source("""\
            def resolve(
                self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                context: ResolutionContextT, position: CompactFlowPosition,
                *, pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
                attribute_path: tuple[str, ...] = (),
            ) -> TargetResolutionT:
                reference = self.lexical_reference
                return resolver._lexical_function_target_resolution(
                    context,
                    LexicalValueReference(reference.root_name, (*reference.attribute_path, *attribute_path)),
                    position, pending_bindings,
                )
        """),
        ),
        patch(
            FLOW,
            "CurrentClassCallTargetReference",
            "    owner_class_qualname: str",
            "    receiver_name: str\n    owner_class_qualname: str",
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "CurrentClassCallTargetReference.lexical_reference"),
            replacement_source=method_source("""\
            def lexical_reference(self) -> LexicalValueReference | None:
                path = self.lexical_attribute_path
                return None if path is None else LexicalValueReference(self.receiver_name, path)
        """),
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CurrentClassCallTargetReference"),
            source=dedent("""\
            @property
            def lexical_attribute_path(self) -> tuple[str, ...] | None:
                return (self.method_name,)
        """),
        ),
        *(
            patch(
                FLOW,
                name,
                "\n    owner_class_qualname: str",
                "\n    receiver_name: str\n    owner_class_qualname: str",
            )
            for name in (
                "CurrentClassMethodReference",
                "CurrentClassMemberMethodReference",
            )
        ),
        InsertClassMemberOperation(
            target=member(FLOW, "CurrentClassMemberMethodReference"),
            source=dedent("""\
            @property
            def lexical_attribute_path(self) -> tuple[str, ...] | None:
                return None if self.uses_runtime_class_lookup else (self.member_name, self.method_name)
        """),
        ),
        patch(
            FLOW,
            "CurrentClassMemberMethodReference.from_expression",
            "        return cls(\n",
            "        return cls(\n            receiver_name=receiver_name,\n",
        ),
        patch(
            FLOW,
            "_CompactFlowCollector._call_target",
            "return CurrentClassMethodReference(\n",
            "return CurrentClassMethodReference(\n                self.current_class_receiver_name,\n",
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "CompactCallableReferenceUse.resolve"),
            replacement_source=method_source('''\
            def resolve(
                self, resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                context: ResolutionContextT,
                *, pending_bindings: frozenset[CompactBindingVisit] = frozenset(),
                attribute_path: tuple[str, ...] = (),
            ) -> TargetResolutionT:
                """Resolve the captured target, retaining lexical cycle and suffix evidence."""
                return self.target.resolve(
                    resolver, context, self.position,
                    pending_bindings=pending_bindings, attribute_path=attribute_path,
                )
        '''),
        ),
        ReplaceTargetOperation(
            target=member(FLOW, "CompactExactValueAlias"),
            replacement_source=dedent('''\
            class CompactExactValueAlias:
                """An exact binding retaining the already-collected source read."""

                source_use: CompactCallableReferenceUse
                binding_mutation: CompactLexicalMutation

                source = AliasProperty[LexicalValueReference]("source_use.target.lexical_reference")
                source_position = AliasProperty[CompactFlowPosition]("source_use.position")

                def __post_init__(self) -> None:
                    if self.source is None:
                        raise ValueError("Exact value aliases require a lexical source read")

                @property
                def target(self) -> LexicalValueReference:
                    return self.binding_mutation.reference

                def source_for(self, reference: LexicalValueReference) -> LexicalValueReference:
                    """Project lexical origin syntax without replacing nominal lookup evidence."""
                    return LexicalValueReference(
                        self.source.root_name, (*self.source.attribute_path, *reference.attribute_path),
                    )
        '''),
        ),
        patch(
            FLOW,
            "_CompactFlowCollector._record_exact_value_aliases",
            "        assert source is not None\n",
            "        source_use = self.callable_reference_uses[-1]\n        assert source_use.target.lexical_reference == source\n",
        ),
        patch(
            FLOW,
            "_CompactFlowCollector._record_exact_value_aliases",
            "                source=source,\n                source_position=mutations[0].position,",
            "                source_use=source_use,",
        ),
        patch(
            REPOSITORY,
            "CompactProductFlowRepository._scope_binding_resolution",
            "            resolution = self._lexical_function_target_resolution(\n                context,\n                alias.source_for(reference),\n                alias.source_position,\n                pending_bindings,\n            )",
            "            resolution = alias.source_use.resolve(\n                self, context, pending_bindings=pending_bindings,\n                attribute_path=reference.attribute_path,\n            )",
        ),
        patch(
            REPOSITORY,
            "CompactProductFlowRepository._possible_binding_symbols",
            "for symbol in self._lexical_function_target_resolution(\n                            context,\n                            alias.source_for(reference),\n                            alias.source_position,\n                            pending_bindings | {(context.owner_symbol, mutation)},\n                        ).possible_symbols",
            "for symbol in alias.source_use.resolve(\n                            self, context,\n                            attribute_path=reference.attribute_path,\n                            pending_bindings=pending_bindings | {(context.owner_symbol, mutation)},\n                        ).possible_symbols",
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
