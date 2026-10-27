"""Write a spelled declaration as the reference it spells.

Three operations, each re-derived from the current source of one file and
proved to change nothing but the strings it replaces (`require_only_replaced`):
`UnquoteResolvedTypeNamesOperation` writes every quoted type string whose names
all resolve unquoted as the expression it spells;
`ReferenceImportedModuleOperation` writes a module's dotted name, where a string
selects a module the file already imports eagerly or is the file's own
package, as the module reference; `ReferenceClassNameOperation` writes a class
name spelled as a value as `Name.__name__`. Where the names resolve is owned by
`quoted_reference` and `class_identity_observations`, which the detector asks
too.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from pathlib import Path

from .codemod import FindingRecipeSynthesizer
from .codemod_reproof import SourceReprovedOperation
from .codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from .codemod_selection_context import CodemodSelectorContext
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from .finding_recipe_actions import FindingRecipeActionKey
from .class_identity_observations import (
    ClassNameSpelling,
    IdentityPositions,
    IdentitySpelling,
    ModuleSpelling,
    TypeSpelling,
)
from .models import DeclarationReferenceMetrics, RefactorFinding
from .quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
    SpelledModules,
)
from .refactor_concepts import DeclarationReferenceConcept


@dataclass(frozen=True)
class SpanRewrite:
    """One expression of the current source and the reference that replaces it."""

    node: ast.AST
    replacement_source: str

    def replacement(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(self.node)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=self.replacement_source,
        )

    @staticmethod
    def span(node: ast.AST) -> tuple[str, int, int, int | None, int | None]:
        return (
            type(node).__name__,
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset,
        )

    @property
    def reference(self) -> ast.expr:
        return ast.parse(self.replacement_source, mode="eval").body

    def expected(self, node: ast.AST) -> ast.AST:
        """What the replaced node is after the rewrite, in the old module's tree."""

        return self.reference


@dataclass(frozen=True)
class ImportedNamesRewrite(SpanRewrite):
    """Names added to a `from` statement: its last alias is written again followed by them.

    `node` is that last alias; `statement` the `from` statement it ends.
    """

    statement: ast.ImportFrom
    added: tuple[str, ...]

    @classmethod
    def of(
        cls, source: str, statement: ast.ImportFrom, added: tuple[str, ...]
    ) -> ImportedNamesRewrite:
        last = statement.names[-1]
        geometry = SourceTextGeometry(source)
        start, end = geometry.required_node_offsets(last)
        return cls(last, ", ".join((source[start:end], *added)), statement, added)

    def expected(self, node: ast.AST) -> ast.AST:
        return node

    def extended(self, statement: ast.ImportFrom) -> ast.ImportFrom:
        statement.names = [*statement.names, *(ast.alias(name) for name in self.added)]
        return statement


@dataclass(frozen=True, kw_only=True)
class DeclarationReferenceOperation(SourceReprovedOperation, ABC):
    """Rewrite one file's spelled declarations as references, from its current source."""

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        path = self.required_source_path(snapshot, type(self).__name__)
        rewrites = self.proved_rewrites(snapshot, path)
        if not rewrites:
            raise ValueError(
                f"{path} spells no declaration this operation can reference"
            )
        geometry = SourceTextGeometry(snapshot.sources_by_file_path[path])
        return (
            geometry.nominal_edit(
                file_path=path,
                replacements=tuple(
                    rewrite.replacement(geometry) for rewrite in rewrites
                ),
                rationale=self.rationale_text(self.default_rationale),
            ),
        )

    def rewritten_lines(self, snapshot: CodemodSelectorContext) -> tuple[int, ...]:
        """The lines whose spellings this operation rewrites in the current source."""

        path = self.required_source_path(snapshot, type(self).__name__)
        return tuple(
            sorted(
                {
                    rewrite.node.lineno
                    for rewrite in self.proved_rewrites(snapshot, path)
                }
            )
        )

    default_rationale: ClassVar[str]
    position_tags: ClassVar[frozenset[str]]
    """The detector positions whose sites this operation rewrites."""

    def proved_rewrites(
        self, context: CodemodSelectorContext, path: str
    ) -> tuple[SpanRewrite, ...]:
        """This file's rewrites, once the rewritten source is proved to differ only in them."""

        parsed = context.parsed_module_for_source_path(path)
        rewrites = self.rewrites(context, parsed.module, path)
        self.require_only_replaced(context.sources_by_file_path[path], rewrites)
        return rewrites

    @abstractmethod
    def rewrites(
        self, context: CodemodSelectorContext, module: ast.Module, path: str
    ) -> tuple[SpanRewrite, ...]:
        raise NotImplementedError

    @staticmethod
    def require_only_replaced(source: str, rewrites: tuple[SpanRewrite, ...]) -> None:
        """The rewritten source parses to the old source's module with each replaced expression swapped for its reference.

        Both sides are parsed afresh from text, so the proof reads nothing a
        later step could have mutated; a replaced expression is found by its
        kind and exact span.
        """

        if not rewrites:
            return
        geometry = SourceTextGeometry(source)
        rewritten = source
        for start, end, replacement in sorted(
            (
                (
                    *geometry.required_node_offsets(rewrite.node),
                    rewrite.replacement_source,
                )
                for rewrite in rewrites
            ),
            reverse=True,
        ):
            rewritten = rewritten[:start] + replacement + rewritten[end:]
        references = {
            SpanRewrite.span(rewrite.node): rewrite
            for rewrite in rewrites
            if not isinstance(rewrite, ImportedNamesRewrite)
        }
        imports = {
            SpanRewrite.span(rewrite.statement): rewrite
            for rewrite in rewrites
            if isinstance(rewrite, ImportedNamesRewrite)
        }

        class Substituted(ast.NodeTransformer):
            def visit(self, node: ast.AST) -> ast.AST:
                if isinstance(node, ast.expr):
                    rewrite = references.get(SpanRewrite.span(node))
                    if rewrite is not None:
                        return rewrite.expected(node)
                if isinstance(node, ast.ImportFrom):
                    added = imports.get(SpanRewrite.span(node))
                    if added is not None:
                        return added.extended(node)
                return super().visit(node)

        expected = ast.dump(Substituted().visit(ast.parse(source)))
        if ast.dump(ast.parse(rewritten)) != expected:
            raise ValueError(
                "the rewrite would change more than the spelled declarations"
            )


@dataclass(frozen=True, kw_only=True)
class UnquoteResolvedTypeNamesOperation(DeclarationReferenceOperation):
    """Write each quoted type string whose every name resolves unquoted as the expression it spells.

    That the names resolve where they stand is `ModuleReferenceScope.resolves`.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((TypeSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Write quoted type names as the references they spell."
    )

    def rewrites(self, context, module, path):
        return tuple(
            SpanRewrite(quoted.node, self.reference_source(quoted.expression))
            for quoted in ModuleReferenceScope(module).unquotable(
                DeclaredInterpreter.of(Path(path))
            )
        )

    @staticmethod
    def reference_source(expression: ast.expr) -> str:
        """The expression's source, parenthesized unless it is atomic where it lands."""

        source = ast.unparse(expression)
        if isinstance(expression, (ast.Name, ast.Attribute, ast.Subscript)):
            return source
        return f"({source})"


@dataclass(frozen=True, kw_only=True)
class ReferenceImportedModuleOperation(DeclarationReferenceOperation):
    """Write a module's dotted name, where it selects a module the file already has, as the reference.

    The file's top-level `import` has already loaded the module, and the name it
    binds is bound nowhere else, so `import_module("a.b")` and
    `sys.modules["a.b"]` are `a.b`, `files("a.b")` is `files(a.b)`, and
    `-m "a.b"` is `a.b.__name__`; the file's own package, as a string, is
    `__package__` (`ModuleReferenceScope.module_reference`). A module the file
    does not import stays spelled: importing it eagerly would change when it
    runs, which no source proof here can settle.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((ModuleSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Reference the imported module instead of spelling its name."
    )

    def rewrites(self, context, module, path):
        scope = ModuleReferenceScope(
            module,
            package=IdentityPositions.package_of(
                context.parsed_module_for_source_path(path)
            ),
        )
        return tuple(
            SpanRewrite(spelled.replaced, reference)
            for spelled in SpelledModules(module)
            if spelled.replaced is not None
            and (reference := scope.module_reference(spelled)) is not None
        )


@dataclass(frozen=True, kw_only=True)
class ReferenceClassNameOperation(DeclarationReferenceOperation):
    """Write a class name spelled as a value (passed, returned, compared, matched) as `Name.__name__`.

    `Name.__name__` evaluates to the literal when `Name` is bound, wherever the
    literal runs, to a scanned class that keeps its statement's name
    (`ClassNameOwners.keeps_name`). Either the file binds it already, by a
    top-level class statement or `from` import that runs first and is shadowed
    nowhere, or the file binds the name nowhere and a `from` statement before
    the literal can also import the class without changing the import order
    (`ClassNameOwners.importable`): the name is added to that statement. Any
    other class name stays spelled: a new import statement changes when its
    module runs, which no source proof here can settle.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((ClassNameSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Name the class object's __name__ instead of spelling the class name."
    )
    NAME_OF_CLASS: ClassVar[str] = "__name__"

    def rewrites(self, context, module, path):
        identities = context.declared_identities
        positions = IdentitySpelling.positions()
        found: dict[int, SpanRewrite] = {}
        imported: dict[int, dict[str, None]] = {}
        for node, detail in IdentityPositions(
            context.parsed_module_for_source_path(path)
        ).spellings():
            position = positions[detail[0]]
            if not (
                position.rewrite(detail, identities, path).rewritable(
                    DeclaredInterpreter.of(Path(path))
                )
                and position.rewrite_kind is ClassNameSpelling
            ):
                continue
            binding, holding, literal = detail[-3:]
            found[id(node)] = SpanRewrite(node, f"{literal}.{self.NAME_OF_CLASS}")
            if not binding:
                statement = identities.class_names.importable(
                    path, int(holding), literal
                )
                if statement is not None:
                    imported.setdefault(statement.index, {})[literal] = None
        source = context.sources_by_file_path[path]
        return (
            *found.values(),
            *(
                ImportedNamesRewrite.of(source, module.body[index], tuple(names))
                for index, names in sorted(imported.items())
            ),
        )


class DeclarationReferenceFindingRecipeSynthesizer(
    FindingRecipeSynthesizer, DeclarationReferenceConcept, ABC
):
    """One recipe per scan-wide rewrite, offered on the finding the detector names as its carrier.

    The detector names one finding per rewrite kind and the files that rewrite
    touches (`DeclarationReferenceMetrics`). Its recipe holds one proved
    operation per file, so the whole rewrite is one candidate; every other
    finding it removes is rejected here as carried by that one.
    """

    reference_operations: tuple[type[DeclarationReferenceOperation], ...] = (
        UnquoteResolvedTypeNamesOperation,
        ReferenceImportedModuleOperation,
        ReferenceClassNameOperation,
    )

    def carried(
        self, finding: RefactorFinding
    ) -> tuple[tuple[type[DeclarationReferenceOperation], tuple[str, ...]], ...]:
        metrics = finding.metrics
        if not isinstance(metrics, DeclarationReferenceMetrics):
            return ()
        return tuple(
            (operation, files)
            for kind, files in metrics.carried_rewrites
            for operation in self.reference_operations
            if kind in operation.position_tags
        )

    def action_keys_for_finding(
        self, finding: RefactorFinding
    ) -> tuple[FindingRecipeActionKey, ...]:
        evidence = finding.primary_evidence
        if evidence is None:
            return ()
        return FindingRecipeActionKey.from_finding_file_subjects(
            finding,
            tuple(
                (evidence.file_path, f"<{operation.operation_key()}>")
                for operation, _files in self.carried(finding)
            ),
        )

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "a declaration reference needs the current source and a primary witness"
            )
        carried = self.carried(finding)
        if not carried:
            return self.rejected_evaluation(
                "no recipe rewrites this site, or its rewrite is carried by the "
                "finding at the scan's first rewritable site"
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-declaration-references",
            reason="Write the scan's spelled declarations as the references they spell.",
        )
        for operation_type, files in carried:
            for file_path in files:
                operation = operation_type(
                    target=SourceRewriteTarget(file_path=file_path)
                )
                try:
                    rewritten = operation.rewritten_lines(context)
                except (SyntaxError, ValueError) as error:
                    return self.rejected_evaluation(
                        f"{operation_type.__name__} on {file_path}: {error}"
                    )
                if not rewritten:
                    return self.rejected_evaluation(
                        f"{operation_type.__name__} no longer rewrites {file_path}"
                    )
                recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)
