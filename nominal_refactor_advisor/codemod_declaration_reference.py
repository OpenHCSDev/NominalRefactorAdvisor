"""Write a spelled declaration as the reference it spells.

Two operations, each re-derived from the current source of one file:
`UnquoteResolvedTypeNamesOperation` writes every quoted type string whose names
all resolve unquoted as the expression it spells, and proves the rewrite changes
nothing but those strings; `ReferenceImportedModuleOperation` writes a module's
dotted name, where a string selects a module that the file already imports
eagerly, as the module reference. Where the names are resolvable is owned by
`quoted_reference`, which the detector asks too.
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
from .class_identity_observations import ModuleSpelling, TypeSpelling
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

    node: ast.expr
    replacement_source: str

    def replacement(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        start_offset, end_offset = geometry.required_node_offsets(self.node)
        return SourceTextSpanReplacement.from_offsets(
            start_offset=start_offset,
            end_offset=end_offset,
            replacement_source=self.replacement_source,
        )


@dataclass(frozen=True, kw_only=True)
class DeclarationReferenceOperation(SourceReprovedOperation, ABC):
    """Rewrite one file's spelled declarations as references, from its current source."""

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        path = self.required_source_path(snapshot, type(self).__name__)
        source = snapshot.sources_by_file_path[path]
        rewrites = self.rewrites(source, path)
        if not rewrites:
            raise ValueError(
                f"{path} spells no declaration this operation can reference"
            )
        geometry = SourceTextGeometry(source)
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
                    for rewrite in self.rewrites(
                        snapshot.sources_by_file_path[path], path
                    )
                }
            )
        )

    default_rationale: ClassVar[str]
    position_tags: ClassVar[frozenset[str]]
    """The detector positions whose sites this operation rewrites."""

    @abstractmethod
    def rewrites(self, source: str, path: str) -> tuple[SpanRewrite, ...]:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class UnquoteResolvedTypeNamesOperation(DeclarationReferenceOperation):
    """Write each quoted type string whose every name resolves unquoted as the expression it spells.

    The proof is structural: the rewritten module parses to exactly the old
    module with those strings replaced by their parsed expressions, so nothing
    else changes; that the names resolve where they stand is
    `ModuleReferenceScope.resolves`.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((TypeSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Write quoted type names as the references they spell."
    )

    def rewrites(self, source: str, path: str) -> tuple[SpanRewrite, ...]:
        module = ast.parse(source)
        unquoted = tuple(
            ModuleReferenceScope(module).unquotable(DeclaredInterpreter.of(Path(path)))
        )
        rewrites = tuple(
            SpanRewrite(quoted.node, self.reference_source(quoted.expression))
            for quoted in unquoted
        )
        self.require_only_strings_change(
            source, module, {id(quoted.node): quoted.expression for quoted in unquoted}
        )
        return rewrites

    @staticmethod
    def reference_source(expression: ast.expr) -> str:
        """The expression's source, parenthesized unless it is atomic where it lands."""

        source = ast.unparse(expression)
        if isinstance(expression, (ast.Name, ast.Attribute, ast.Subscript)):
            return source
        return f"({source})"

    @staticmethod
    def require_only_strings_change(
        source: str, module: ast.Module, expressions: dict[int, ast.expr]
    ) -> None:
        if not expressions:
            return
        geometry = SourceTextGeometry(source)
        rewritten = source
        spans = sorted(
            (
                (
                    *geometry.required_node_offsets(node),
                    UnquoteResolvedTypeNamesOperation.reference_source(
                        expressions[id(node)]
                    ),
                )
                for node in ast.walk(module)
                if id(node) in expressions
            ),
            reverse=True,
        )
        for start, end, replacement in spans:
            rewritten = rewritten[:start] + replacement + rewritten[end:]

        class Substituted(ast.NodeTransformer):
            def visit_Constant(self, node: ast.Constant) -> ast.expr:
                return expressions.get(id(node), node)

        expected = ast.dump(Substituted().visit(module))
        if ast.dump(ast.parse(rewritten)) != expected:
            raise ValueError("unquoting would change more than the quoted type names")


@dataclass(frozen=True, kw_only=True)
class ReferenceImportedModuleOperation(DeclarationReferenceOperation):
    """Write a module's dotted name, where it selects a module the file imports eagerly, as the reference.

    The file's top-level `import` has already loaded the module, and the name it
    binds is bound nowhere else, so `import_module("a.b")` and
    `sys.modules["a.b"]` are `a.b`, `files("a.b")` is `files(a.b)`, and
    `-m "a.b"` is `a.b.__name__`. A module the file does not import stays
    spelled: importing it eagerly would change when it runs, which no source
    proof here can settle.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((ModuleSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Reference the imported module instead of spelling its name."
    )

    def rewrites(self, source: str, path: str) -> tuple[SpanRewrite, ...]:
        module = ast.parse(source)
        scope = ModuleReferenceScope(module)
        interpreter = DeclaredInterpreter.of(Path(path))
        return tuple(
            SpanRewrite(
                spelled.replaced,
                f"{scope.imported_modules[spelled.name]}{spelled.suffix}",
            )
            for spelled in SpelledModules(module)
            if spelled.replaced is not None
            and scope.module_rewrite(spelled).rewritable(interpreter)
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
