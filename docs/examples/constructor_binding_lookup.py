"""Resolve class and function definitions through the same lexical lookup."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    InsertAfterTargetOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    RenameTopLevelDeclarationAuthorityOperation,
    ReplaceDeclaredCallArgumentsOperation,
    ReplaceDeclaredCallOperation,
    ReplaceFunctionBodyOperation,
    ReplaceTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

syntax = SourceRewriteTarget(file_path="nominal_refactor_advisor/product_flow.py")
repository = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/product_flow_authority.py"
)
target = replace(repository, qualname="CompactCallTargetResolution")

PLAN = CodemodPlanSequence.from_operations(
    (
        RenameTopLevelDeclarationAuthorityOperation(
            target=replace(repository, qualname="CompactFunctionTargetResolution"),
            new_name="CompactCallTargetResolution",
        ),
        ReplaceTargetOperation(
            target=target,
            replacement_source=dedent('''\
                class CompactCallTargetResolution(ABC):
                    """A lexical callable target, with function and construction projections."""

                    @property
                    def declaration(self) -> CompactFunctionDeclaration | None:
                        """The function declaration, when this target denotes a function."""
                        return None

                    @property
                    @abstractmethod
                    def possible_symbols(self) -> tuple[str, ...]:
                        raise NotImplementedError

                    def through_alias(
                        self, alias: CompactExactValueAlias, context: CompactProductFlowContext
                    ) -> CompactCallTargetResolution:
                        return self

                    def through_descriptor(
                        self, access: CompactDescriptorAccess
                    ) -> CompactCallTargetResolution:
                        return self

                    def resolve_call(
                        self, context: CompactProductFlowContext, call: CompactFunctionCall
                    ) -> CompactFunctionCallResolution:
                        return CompactOpenFunctionCall(context, call, self)

                    def resolve_construction(
                        self,
                        repository: CompactProductFlowRepository,
                        context: CompactProductFlowContext,
                        call: CompactFunctionCall,
                    ) -> CompactResolvedProductConstruction | None:
                        return None
                '''),
        ),
        *(
            DeleteTargetOperation(
                target=replace(
                    repository, qualname=f"OpenCompactFunctionTarget.{name}"
                ),
            )
            for name in ("declaration", "through_descriptor", "resolve_call")
        ),
        InsertAfterTargetOperation(
            target=target,
            source=dedent('''\
                @dataclass(frozen=True)
                class ResolvedCompactClassTarget(CompactCallTargetResolution):
                    """A class definition selected by the ordinary lexical binding resolver."""

                    resolved_declaration: CompactIndexedClass

                    @property
                    def possible_symbols(self) -> tuple[str, ...]:
                        return (self.resolved_declaration.symbol,)

                    def resolve_construction(
                        self,
                        repository: CompactProductFlowRepository,
                        context: CompactProductFlowContext,
                        call: CompactFunctionCall,
                    ) -> CompactResolvedProductConstruction | None:
                        construction = call.product_construction()
                        authority = repository.product_authorities_by_symbol.get(
                            self.resolved_declaration.symbol
                        )
                        if construction is None or authority is None:
                            return None
                        return CompactResolvedProductConstruction(
                            context, call, construction, authority
                        )
                '''),
        ),
        InsertClassMemberOperation(
            target=replace(repository, qualname="CompactProductFlowRepository"),
            source=dedent("""\
                def _selected_class_resolution(
                    self, symbol: str, binding: CompactLexicalMutation
                ) -> CompactCallTargetResolution:
                    declaration = self.class_index.class_for(symbol)
                    if declaration is None:
                        return OpenCompactFunctionTarget(
                            (symbol,), CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
                        )
                    if declaration.line != binding.line:
                        return OpenCompactFunctionTarget(
                            (symbol,), CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
                        )
                    return ResolvedCompactClassTarget(declaration)
                """),
        ),
        *(
            InsertClassMemberOperation(
                target=replace(syntax, qualname="CompactCallTargetResolverABC"),
                source=dedent(f'''\
                    @abstractmethod
                    def _selected_{kind}_resolution(
                        self, symbol: str, binding: CompactLexicalMutation
                    ) -> TargetResolutionT:
                        """Resolve a selected {kind} definition at its exact source site."""
                        raise NotImplementedError
                    '''),
            )
            for kind in ("function", "class")
        ),
        ReplaceTargetOperation(
            target=replace(syntax, qualname="CompactMutationKind"),
            replacement_source=dedent('''\
                class CompactMutationKind(StrEnum):
                    """Source operations and their nominal declaration lookup behaviour."""

                    ASSIGNMENT = "assignment"
                    AUGMENTED_ASSIGNMENT = "augmented_assignment"
                    DELETION = "deletion"
                    FUNCTION_DEFINITION = "function_definition", lambda resolver, symbol, binding: resolver._selected_function_resolution(symbol, binding)
                    CLASS_DEFINITION = "class_definition", lambda resolver, symbol, binding: resolver._selected_class_resolution(symbol, binding)
                    IMPORT = "import"
                    ITERATION_BINDING = "iteration_binding"
                    CONTEXT_BINDING = "context_binding"
                    EXCEPTION_BINDING = "exception_binding"
                    PATTERN_BINDING = "pattern_binding"

                    def __new__(
                        cls,
                        value: str,
                        declaration_resolution: Callable[
                            [CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT], str, CompactLexicalMutation],
                            TargetResolutionT,
                        ] | None = None,
                    ) -> Self:
                        member = str.__new__(cls, value)
                        member._value_ = value
                        member._declaration_resolution = declaration_resolution
                        return member

                    @property
                    def is_import_binding(self) -> bool:
                        return self is type(self).IMPORT

                    @property
                    def is_definition_binding(self) -> bool:
                        return self._declaration_resolution is not None

                    @property
                    def preserves_nominal_identity(self) -> bool:
                        return self.is_import_binding or self.is_definition_binding

                    def resolve_definition(
                        self,
                        resolver: CompactCallTargetResolverABC[ResolutionContextT, TargetResolutionT],
                        symbol: str,
                        binding: CompactLexicalMutation,
                    ) -> TargetResolutionT:
                        if self._declaration_resolution is None:
                            raise ValueError("Only definition mutations resolve a declaration")
                        return self._declaration_resolution(resolver, symbol, binding)

                    def validate_import_origin(self, origin: str | None) -> None:
                        if origin is not None and not self.is_import_binding:
                            raise ValueError("Only import mutations carry an imported origin")
                '''),
        ),
        *(
            ReplaceDeclaredCallArgumentsOperation(
                target=replace(syntax, qualname=f"_CompactFlowCollector.visit_{node}"),
                callee=replace(
                    syntax, qualname="_CompactFlowCollector._record_mutation"
                ),
                arguments_source=f"LexicalValueReference(node.name), node, CompactMutationKind.{kind}_DEFINITION",
            )
            for node, kind in (("FunctionDef", "FUNCTION"), ("ClassDef", "CLASS"))
        ),
        ReplaceDeclaredCallOperation(
            target=replace(
                repository,
                qualname="CompactProductFlowRepository._scope_binding_resolution",
            ),
            callee=replace(
                repository,
                qualname="CompactProductFlowRepository._selected_function_resolution",
            ),
            expression_source="binding.kind.resolve_definition(self, binding_symbol, binding)",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                repository,
                qualname="CompactProductFlowRepository.resolve_product_construction",
            ),
            body_source=dedent("""\
                return self.resolve_function_target(
                    context, call.target, call.position
                ).resolve_construction(self, context, call)
                """),
        ),
        DeleteTargetOperation(
            target=replace(
                repository,
                qualname="CompactProductFlowRepository._has_dynamic_local_binding",
            ),
        ),
        PatchTargetOperation(
            target=replace(repository, qualname="CompactOpenFunctionCall"),
            replacements=(
                SourceTextReplacement(
                    old_source="One projected call whose target is not nominally closed.",
                    new_source="One projected call without a resolved function declaration.",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
