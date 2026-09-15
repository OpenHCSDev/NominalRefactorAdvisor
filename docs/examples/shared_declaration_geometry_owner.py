"""Share exact decorator markers and retain only immutable body-window geometry.

This authored syntax plan changes ownership; it does not prove body activation.
"""

from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    InsertClassMemberOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.codemod_declaration_source import FunctionSourceAuthority

GEOMETRY = "nominal_refactor_advisor/source_geometry.py"
EDITS = "nominal_refactor_advisor/codemod_source_edits.py"
COMPILATION = "nominal_refactor_advisor/native_compilation.py"


def shared_geometry_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    source = snapshot.parsed_module_for_source_path(EDITS).source
    movements = []
    for name in ("iter_tokens", "tokens"):
        target = SourceRewriteTarget(
            file_path=EDITS, qualname=f"SourceTextGeometry.{name}"
        )
        identifier = target.required_target_id(snapshot.source_index)
        declaration = FunctionSourceAuthority(
            snapshot.ast_target_nodes_by_id[identifier], source
        ).declaration_line_span.source_from(source)
        movements.extend(
            (
                InsertClassMemberOperation(
                    target=SourceRewriteTarget(
                        file_path=GEOMETRY, qualname="SourceLineSegmentAuthority"
                    ),
                    source=dedent(declaration),
                ),
                PatchTargetOperation(
                    target=SourceRewriteTarget(
                        file_path=EDITS, qualname="SourceTextGeometry"
                    ),
                    replacements=(SourceTextReplacement(declaration, ""),),
                ),
            )
        )
    return CodemodPlanSequence.from_operations(
        (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=GEOMETRY),
                import_source="from bisect import bisect_left",
            ),
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=GEOMETRY),
                import_source="from operator import attrgetter",
            ),
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=COMPILATION),
                import_source="from .source_geometry import SourceLineSegmentAuthority",
            ),
            *movements,
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=GEOMETRY, qualname="SourceLineSegmentAuthority"
                ),
                source='''def decorated_node_start_line(
    self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
) -> int:
    """Recover exact decorator markers omitted by AST expression positions."""
    if not node.decorator_list:
        return node.lineno
    first = node.decorator_list[0]
    position = (
        first.lineno,
        SourceByteSpan.character_column(self.lines[first.lineno - 1], first.col_offset),
    )
    token_index = bisect_left(self.tokens, position, key=attrgetter("start"))
    for index in range(token_index - 1, -1, -1):
        token = self.tokens[index]
        if token.exact_type == tokenize.AT:
            return token.start[0]
    raise ValueError("Decorated declaration has no source decorator marker")
''',
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=EDITS, qualname="SourceTextGeometry.node_start_line"
                ),
                body_source='''"""Project the exact generic marker through this span's decorator policy."""
node = span.node
if span.decorator_policy.includes_decorators and isinstance(
    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
):
    return self.decorated_node_start_line(node)
return node.lineno
''',
            ),
            AddClassBaseOperation(
                target=SourceRewriteTarget(
                    file_path=COMPILATION, qualname="NativePythonCompilation"
                ),
                base_name="SourceLineSegmentAuthority",
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=COMPILATION, qualname="NativePythonCompilation"
                ),
                replacements=(SourceTextReplacement("    source: str\n", ""),),
            ),
            InsertClassMemberOperation(
                target=SourceRewriteTarget(
                    file_path=COMPILATION, qualname="NativePythonCompilation"
                ),
                source='''@ScanCache.cached
def _definition_start_line(self, span: SourceByteSpan) -> int:
    """Pure original-source geometry; no mutable AST or current-code verdict retained."""
    definitions = tuple(
        node
        for node in ast.walk(ast.parse(self.source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and SourceByteSpan.require_node(node) == span
    )
    if len(definitions) != 1:
        raise ValueError("Native body has no unique function source declaration")
    return self.decorated_node_start_line(definitions[0])
''',
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=COMPILATION,
                    qualname="NativePythonCompilation.function_definition",
                ),
                replacements=(
                    SourceTextReplacement(
                        "function.__code__.co_firstlineno - 1",
                        "self._definition_start_line(span) - 1",
                    ),
                    SourceTextReplacement(
                        "window.segment(tuple(self.source.splitlines(keepends=True)))",
                        "window.segment(self.lines)",
                    ),
                ),
            ),
        )
    )
