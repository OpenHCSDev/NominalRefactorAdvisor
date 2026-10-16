"""Factor current callable metadata into its existing native source owner.

This is an authored syntax plan. Source correspondence and current default
associations remain evidence, not assumptions of callable body behavior.
"""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_declaration_source import FunctionSourceAuthority

PATH = "nominal_refactor_advisor/native_call.py"


def callable_metadata_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    source = snapshot.parsed_module_for_source_path(PATH).source
    replacements = []
    for name, annotation, projection in (
        ("signature", "CompactFunctionSignature", "signature"),
        ("python_defaults", "tuple[NativeParameterDefault, ...]", "defaults"),
        ("python_definition", "ast.FunctionDef | ast.AsyncFunctionDef", "definition"),
    ):
        selector = SourceRewriteTarget(
            file_path=PATH, qualname=f"NativeCallAuthority.{name}"
        )
        identifier = selector.required_target_id(snapshot.source_index)
        authority = FunctionSourceAuthority(
            node=snapshot.ast_target_nodes_by_id[identifier], source=source
        )
        replacements.append(
            SourceTextReplacement(
                authority.declaration_line_span.source_from(source),
                f'    {name} = AliasProperty[{annotation}]("python_source.{projection}")\n',
            )
        )
    return CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativePythonFunctionSource"
                ),
                source='''@property
def defaults(self) -> tuple[NativeParameterDefault, ...]:
    """Current associations, not values reconstructed from source defaults."""
    _ = self.definition
    return NativeParameterDefault.from_function(self.function)
''',
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativePythonFunctionSource"
                ),
                source='''@property
def signature(self) -> CompactFunctionSignature:
    """Source parameter structure with the function's current default presence."""
    return CompactFunctionSignature.from_arguments(self.definition.args).with_default_names(
        frozenset(default.parameter_name for default in self.defaults)
    )
''',
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativeCallAuthority"
                ),
                source="""@property
def python_source(self) -> NativePythonFunctionSource:
    return NativePythonFunctionSource.from_function(self.declaration.declaration)
""",
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativeCallAuthority"
                ),
                replacements=tuple(replacements),
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/source_execution.py",
                    qualname="NativeSourceClassEntryABC",
                ),
                source='''@property
def constructor_source(self) -> NativePythonFunctionSource:
    """Current selected constructor operands, not an admitted body activation."""
    return NativePythonFunctionSource.from_function(
        NativeClassMroDeclaration(self.metaclass_declaration.declaration).python_constructor()
    )
''',
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/source_execution.py",
                    qualname="NativeSourceClassEntryABC.construction_admission",
                ),
                body_source="""self.require_construction_inputs()
_ = self.constructor_source.signature
raise ValueError("Native class construction over prepared inputs remains unproved")
""",
            ),
        )
    )
