"""Keep current-code validation outside retained syntax projections.

This authored syntax plan removes stored AST and cached parameter observations;
it does not admit a captured-function activation or metaclass construction.
"""

import ast
from textwrap import dedent, indent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    InsertClassMemberOperation,
    PatchTargetOperation,
    PrependFunctionBodyOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionDecoratorsOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
)
from nominal_refactor_advisor.codemod_declaration_source import (
    FunctionBodySourceAuthority,
    FunctionSourceAuthority,
)

PATH = "nominal_refactor_advisor/native_call.py"


def current_body_owner_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    caller = SourceRewriteTarget(
        file_path=PATH, qualname="NativeDataclassFactoryCall.require_closed"
    )

    identifier = caller.required_target_id(snapshot.source_index)
    (constructor,) = (
        node
        for node in ast.walk(snapshot.ast_target_nodes_by_id[identifier])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NativeReturnedClosureFactorySource"
    )
    source = snapshot.parsed_module_for_source_path(PATH).source
    span = SourceTextSpan.from_offsets(
        SourceTextGeometry(source).required_node_offsets(constructor)
    )
    binding_target = SourceRewriteTarget(
        file_path=PATH,
        qualname="NativeDataclassDefinitionApplicationABC.require_factory_processor_call",
    )
    binding_id = binding_target.required_target_id(snapshot.source_index)
    binding_source = FunctionSourceAuthority(
        snapshot.ast_target_nodes_by_id[binding_id], source
    ).declaration_line_span.source_from(source)
    binding_observation_source = binding_source
    for name in ("factory_parameters", "processor_parameters"):
        binding_observation_source = binding_observation_source.replace(
            f"self.{name}", name
        )
    return CodemodPlanSequence.from_operations(
        (
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativePythonFunctionSource.flow"
                ),
                replacements=(SourceTextReplacement("def flow(", "def _flow("),),
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativePythonFunctionSource"
                ),
                source='''@property
def flow(self) -> CompactFunctionFlow:
    """Revalidate current code before using the retained body projection."""
    current_span = SourceByteSpan.require_node(self.definition)
    flow = self._flow
    declaration = flow.owner.declaration
    if declaration is None or declaration.source_span != current_span:
        raise ValueError("Native function body changed after source-flow capture")
    return flow
''',
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativeReturnedClosureFactorySource"
                ),
                replacements=(
                    SourceTextReplacement(
                        "    function: FunctionType\n    definition: ast.FunctionDef\n",
                        '    source: NativePythonFunctionSource\n\n    function = AliasProperty[FunctionType]("source.function")\n',
                    ),
                ),
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=PATH,
                    qualname="NativeReturnedClosureFactorySource.from_function",
                ),
                body_source="""result = cls(NativePythonFunctionSource.from_function(function))
_ = result.definition
return result
""",
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativeReturnedClosureFactorySource"
                ),
                source='''@property
def definition(self) -> ast.FunctionDef:
    """The source owner rejoins actual code and exposes fresh syntax."""
    definition = self.source.definition
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError("Native closure factory requires a synchronous function")
    return definition
''',
            ),
            *(
                ReplaceFunctionDecoratorsOperation(
                    target=SourceRewriteTarget(
                        file_path=PATH,
                        qualname=f"NativeDataclassDefinitionApplicationABC.{name}",
                    ),
                    decorators_source="@property",
                )
                for name in (
                    "factory_parameters",
                    "processor_parameters",
                    "selector_parameter",
                )
            ),
            PatchTargetOperation(
                target=binding_target,
                replacements=(
                    SourceTextReplacement(binding_source, binding_observation_source),
                ),
            ),
            PrependFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=PATH,
                    qualname="NativeDataclassDefinitionApplicationABC.require_factory_processor_call",
                ),
                body_source="factory_parameters = self.factory_parameters\nprocessor_parameters = self.processor_parameters\n",
            ),
            PatchTargetOperation(
                target=caller,
                replacements=(
                    span.replacement(
                        source, "NativeReturnedClosureFactorySource(self.python_source)"
                    ),
                ),
            ),
        )
    )


def bounded_body_query_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    """Use the existing invocation cache without retaining validation results."""
    source = snapshot.parsed_module_for_source_path(PATH).source
    operations = []
    for name in (
        "NativeReturnedClosureFactorySource.require_returned_closure",
        "NativeDataclassDefinitionApplicationABC.result",
        "NativeDataclassFactoryCall.require_closed",
    ):
        target = SourceRewriteTarget(file_path=PATH, qualname=name)
        identifier = target.required_target_id(snapshot.source_index)
        authority = FunctionBodySourceAuthority(
            snapshot.ast_target_nodes_by_id[identifier], source
        )
        body = dedent(authority.layout.span.source_text(source))
        first, *_ = ast.parse(body).body
        docstring = ""
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and type(first.value.value) is str
        ):
            span = SourceTextSpan.from_offsets(
                SourceTextGeometry(body).required_node_offsets(first)
            )
            docstring = span.source_text(body) + "\n"
            body = body[span.end_offset :].lstrip("\n")
        operations.append(
            ReplaceFunctionBodyOperation(
                target=target,
                body_source=docstring
                + "with ScanCache.scope():\n"
                + indent(body, "    "),
            )
        )
    return CodemodPlanSequence.from_operations(tuple(operations))


def current_code_span_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    """Let geometry consumers rejoin current code without requesting fresh AST."""
    compilation_path = "nominal_refactor_advisor/native_compilation.py"
    target = SourceRewriteTarget(
        file_path=compilation_path,
        qualname="NativePythonCompilation.function_definition",
    )
    identifier = target.required_target_id(snapshot.source_index)
    node = snapshot.ast_target_nodes_by_id[identifier]
    source = snapshot.parsed_module_for_source_path(compilation_path).source
    geometry = SourceTextGeometry(source)
    validation, code, span = node.body[1:4]
    replacement = SourceTextSpan.from_offsets(
        (
            geometry.required_node_offsets(validation)[0],
            geometry.required_node_offsets(span)[1],
        )
    ).replacement(source, "span = self.function_source_span(function)")
    return CodemodPlanSequence.from_operations(
        (
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=compilation_path,
                    qualname="NativePythonCompilation",
                ),
                source='''def function_source_span(self, function: FunctionType) -> SourceByteSpan:
    """Rejoin actual current code to immutable source geometry, without exposing AST."""
    if type(function) is not FunctionType:
        raise ValueError("Python implementation requires an exact function")
    code = function.__code__
    return self._function_source_span(
        NativeCreationBackend.current().code_contents(code),
        code.co_filename,
        code.co_firstlineno,
    )
''',
            ),
            PatchTargetOperation(target=target, replacements=(replacement,)),
            *(
                PatchTargetOperation(
                    target=SourceRewriteTarget(file_path=PATH, qualname=name),
                    replacements=(
                        SourceTextReplacement(
                            "SourceByteSpan.require_node(self.definition)",
                            "self.compilation.function_source_span(self.function)",
                        ),
                    ),
                )
                for name in (
                    "NativePythonFunctionSource.flow",
                    "NativePythonFunctionSource._flow",
                )
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=PATH,
                    qualname="NativePythonFunctionSource._from_current_function",
                ),
                replacements=(
                    SourceTextReplacement(
                        "compilation.function_definition(function)",
                        "compilation.function_source_span(function)",
                    ),
                ),
            ),
        )
    )


def fresh_definition_window_plan(
    snapshot: CodemodSourceSnapshot,
) -> CodemodPlanSequence:
    """Parse fresh syntax only inside the compiler-authenticated body window."""
    return CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path="nominal_refactor_advisor/native_compilation.py",
                    qualname="NativePythonCompilation.function_definition",
                ),
                body_source='''"""Match current code, then expose fresh positioned syntax for its body only.

Defaults, closure values, globals, activation and effects need independent
evidence. Neither executable code nor mutable syntax is retained by this query.
"""
span = self.function_source_span(function)
start_line_index = function.__code__.co_firstlineno - 1
window = SourceByteSpan(
    start_line_index=start_line_index,
    end_line_index=span.end_line_index,
    start_byte=0,
    end_byte=span.end_byte,
)
source = window.segment(tuple(self.source.splitlines(keepends=True)))
if span.start_byte:
    source = "if True:\\n" + source
parsed = ast.parse(source)
ast.increment_lineno(parsed, start_line_index - bool(span.start_byte))
definitions = tuple(
    node
    for node in ast.walk(parsed)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and SourceByteSpan.require_node(node) == span
)
if len(definitions) != 1:
    raise ValueError("Native body has no unique function source declaration")
return definitions[0]
''',
            ),
        )
    )
