"""Derive replacement indentation from declaration ownership and preserve literals."""

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
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

GEOMETRY = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_source_edits.py"
)
STATEMENT = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_statement_source.py"
)
DECLARATION = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod_declaration_source.py"
)
OPERATIONS = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")


def member(module: SourceRewriteTarget, name: str) -> SourceRewriteTarget:
    return replace(module, qualname=name)


def patch(
    module: SourceRewriteTarget, name: str, old: str, new: str
) -> PatchTargetOperation:
    return PatchTargetOperation(
        target=member(module, name),
        replacements=(SourceTextReplacement(old_source=old, new_source=new),),
    )


PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=GEOMETRY, import_source="from collections.abc import Iterator"
        ),
        InsertClassMemberOperation(
            target=member(GEOMETRY, "SourceTextGeometry"),
            source=dedent('''\
            def iter_tokens(self) -> Iterator[tokenize.TokenInfo]:
                """Read source tokens lazily when only a prefix is required."""
                return tokenize.generate_tokens(io.StringIO(self.source).readline)
        '''),
        ),
        ReplaceFunctionBodyOperation(
            target=member(GEOMETRY, "SourceTextGeometry.tokens"),
            body_source="return tuple(self.iter_tokens())",
        ),
        EnsureImportOperation(target=STATEMENT, import_source="import tokenize"),
        EnsureImportOperation(
            target=STATEMENT, import_source="from typing import cast"
        ),
        InsertBeforeTargetOperation(
            target=member(STATEMENT, "StatementScopeSource"),
            source=dedent('''\
            @dataclass(frozen=True)
            class PythonBlockSource(SourceTextGeometry):
                """An authored Python suite whose structural indentation can be relocated."""

                @cached_property
                def leading_indentation(self) -> str:
                    for token in self.iter_tokens():
                        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
                            continue
                        return token.string if token.type == tokenize.INDENT else ""
                    return ""

                @property
                def parse_prefix(self) -> str:
                    return "if True:\\n" if self.leading_indentation else ""

                @cached_property
                def parsed_module(self) -> ast.Module:
                    return ast.parse(self.parse_prefix + self.source)

                @cached_property
                def statements(self) -> tuple[ast.stmt, ...]:
                    statements = self.parsed_module.body
                    if self.leading_indentation:
                        if len(statements) != 1:
                            raise ValueError("Python block escapes its initial indentation")
                        statements = cast(ast.If, statements[0]).body
                    return tuple(statements)

                def indented_source(self, indentation: str) -> str:
                    """Relocate code and comments while preserving complete literal spans."""
                    if not self.statements:
                        raise ValueError("Replacement source block must contain a statement")
                    continuation_lines = self.literal_continuation_lines(self.parsed_module)
                    return "".join(
                        indentation + line.removeprefix(self.leading_indentation)
                        if number not in continuation_lines and line.strip()
                        else line
                        for number, line in enumerate(
                            self.lines, start=1 + self.parse_prefix.count("\\n"),
                        )
                    )


        '''),
        ),
        DeleteTargetOperation(
            target=member(GEOMETRY, "SourceTextGeometry.indented_source")
        ),
        EnsureImportOperation(
            target=DECLARATION,
            import_source="from .codemod_statement_source import PythonBlockSource",
        ),
        EnsureImportOperation(
            target=DECLARATION,
            import_source="from .descriptor_algebra import AliasProperty",
        ),
        EnsureImportOperation(
            target=OPERATIONS,
            import_source="from .codemod_statement_source import PythonBlockSource",
        ),
        patch(
            STATEMENT,
            "AssignmentSource.replacement_source",
            "SourceTextGeometry(self.source[start:end])",
            "PythonBlockSource(self.source[start:end])",
        ),
        patch(
            DECLARATION,
            "ClassMemberSource.from_source",
            "            module = ast.parse(source)",
            "            block = PythonBlockSource(source)\n            statements = block.statements",
        ),
        patch(
            DECLARATION,
            "ClassMemberSource.from_source",
            "len(module.body) != 1",
            "len(statements) != 1",
        ),
        patch(
            DECLARATION,
            "ClassMemberSource.from_source",
            "            module.body[0],",
            "            statements[0],",
        ),
        patch(
            DECLARATION,
            "ClassMemberSource.from_source",
            "bound_names(module.body)",
            "bound_names(statements)",
        ),
        patch(
            DECLARATION,
            "ClassMemberSource.from_source",
            "SourceTextGeometry(source).indented_source(indentation)",
            "block.indented_source(indentation)",
        ),
        patch(
            DECLARATION,
            "DeclarationDecoratorsSourceAuthority.replacement",
            'scaffold = SourceTextGeometry(prefix + "def _decorated(): pass\\n")',
            'scaffold = PythonBlockSource(prefix + "def _decorated(): pass\\n")',
        ),
        patch(
            DECLARATION,
            "DeclarationDecoratorsSourceAuthority.replacement",
            "module = ast.parse(scaffold.source)",
            "module = scaffold.parsed_module",
        ),
        patch(
            DECLARATION,
            "FunctionSuiteLayout.render",
            "SourceTextGeometry(source).indented_source(self.indentation)",
            "PythonBlockSource(source).indented_source(self.indentation)",
        ),
        # Authored transfer: automatic promotion cannot yet prove the unchanged
        # cached_property declarations in this ancestor's class namespace.
        InsertClassMemberOperation(
            target=member(DECLARATION, "NamedDeclarationSourceAuthority"),
            source=dedent('''\
            @property
            def declaration_indentation(self) -> str:
                """Project the declaration header's actual enclosing indentation."""
                lines = self.geometry.lines
                if not 1 <= self.node.lineno <= len(lines):
                    raise ValueError("Declaration header is outside its source geometry")
                line = lines[self.node.lineno - 1]
                return line[: len(line) - len(line.lstrip())]
        '''),
        ),
        DeleteTargetOperation(
            target=member(DECLARATION, "ClassHeaderSpanSourceAuthority.indentation")
        ),
        InsertClassMemberOperation(
            target=member(DECLARATION, "ClassHeaderSpanSourceAuthority"),
            source='indentation = AliasProperty[str]("declaration_indentation")',
        ),
        InsertClassMemberOperation(
            target=member(OPERATIONS, "ReplaceTargetOperation"),
            source=dedent("""\
            @cached_property
            def replacement_block(self) -> PythonBlockSource:
                return PythonBlockSource(self.replacement_source)
        """),
        ),
        ReplaceFunctionBodyOperation(
            target=member(OPERATIONS, "ReplaceTargetOperation.replacement_declaration"),
            body_source=dedent('''\
            """Validate the declaration against the same block used for rendering."""
            try:
                statements = self.replacement_block.statements
            except SyntaxError as error:
                raise ValueError(f"Replacement source is not valid Python: {error}") from error
            if len(statements) != 1 or not isinstance(statements[0], AstTargetNode):
                raise ValueError(
                    "Replacement source must contain exactly one class or function declaration"
                )
            self.decorator_policy.validate_replacement(statements[0])
            return statements[0]
        '''),
        ),
        patch(
            OPERATIONS,
            "ReplaceTargetOperation.source_edits_from_snapshot",
            "        span = SourceTextGeometry(\n            snapshot.sources_by_file_path[target.file_path]\n        ).node_line_span(SourceNodeSpan(target_node, self.decorator_policy))",
            "        declaration = NamedDeclarationSourceAuthority(\n            target_node, snapshot.sources_by_file_path[target.file_path],\n        )\n        span = declaration.geometry.node_line_span(\n            SourceNodeSpan(target_node, self.decorator_policy),\n        )",
        ),
        patch(
            OPERATIONS,
            "ReplaceTargetOperation.source_edits_from_snapshot",
            "                    self.replacement_source\n",
            "                    self.replacement_block.indented_source(\n                        declaration.declaration_indentation,\n                    )\n",
        ),
        PatchTargetOperation(
            target=OPERATIONS,
            replacements=(
                SourceTextReplacement(old_source="import textwrap\n", new_source=""),
            ),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
