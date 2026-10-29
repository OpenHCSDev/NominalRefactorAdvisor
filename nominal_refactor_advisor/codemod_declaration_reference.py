"""Write a spelled declaration as the reference it spells.

Three operations, each re-derived from the current source of one file and
proved to change nothing but the strings it replaces (`require_only_replaced`):
`UnquoteResolvedTypeNamesOperation` writes every quoted type string whose names
all resolve unquoted as the expression it spells;
`ReferenceImportedModuleOperation` writes a module's dotted name, where a string
selects a module the file already imports eagerly or is the file's own
package, as the module reference; `ReferenceClassNameOperation` writes a class
name spelled as a value as `Name.__name__`, and `ReferenceEnumMemberOperation`
an enum member's value spelled as a string as `Enum.MEMBER.value`. Where the
names resolve is owned by `quoted_reference` and `class_identity_observations`,
which the detector asks too.
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
    DeclaredIdentities,
    EnumValueSpelling,
    IdentityPositions,
    IdentitySpelling,
    ModuleSpelling,
    NameImport,
    TypeSpelling,
)
from .models import DeclarationReferenceMetrics, RefactorFinding
from .quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
    QuotedTypeName,
    SpelledModules,
)
from .refactor_concepts import DeclarationReferenceConcept


@dataclass(frozen=True)
class SpanRewrite:
    """One expression of the current source and the reference that replaces it."""

    node: ast.AST
    replacement_source: str

    def offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        return geometry.required_node_offsets(self.node)

    def replacement(self, geometry: SourceTextGeometry) -> SourceTextSpanReplacement:
        start_offset, end_offset = self.offsets(geometry)
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


@dataclass(frozen=True)
class InsertedImportsRewrite(SpanRewrite):
    """New `from module import a, b` statements written right after `node`, the file's last top-level import."""

    inserted: tuple[ast.ImportFrom, ...]

    @classmethod
    def of(
        cls, statement: ast.stmt, imports: dict[str, tuple[str, ...]]
    ) -> InsertedImportsRewrite:
        ordered = sorted(imports.items())
        return cls(
            statement,
            "".join(
                f"\nfrom {module} import {', '.join(names)}"
                for module, names in ordered
            ),
            tuple(
                ast.ImportFrom(
                    module=module, names=[ast.alias(name) for name in names], level=0
                )
                for module, names in ordered
            ),
        )

    def offsets(self, geometry: SourceTextGeometry) -> tuple[int, int]:
        end = geometry.required_node_offsets(self.node)[1]
        return end, end

    def expected(self, node: ast.AST) -> ast.AST:
        return node


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
                (*rewrite.offsets(geometry), rewrite.replacement_source)
                for rewrite in rewrites
            ),
            reverse=True,
        ):
            rewritten = rewritten[:start] + replacement + rewritten[end:]
        references = {
            SpanRewrite.span(rewrite.node): rewrite
            for rewrite in rewrites
            if not isinstance(rewrite, (ImportedNamesRewrite, InsertedImportsRewrite))
        }
        imports = {
            SpanRewrite.span(rewrite.statement): rewrite
            for rewrite in rewrites
            if isinstance(rewrite, ImportedNamesRewrite)
        }
        insertions = {
            SpanRewrite.span(rewrite.node): rewrite
            for rewrite in rewrites
            if isinstance(rewrite, InsertedImportsRewrite)
        }

        class Substituted(ast.NodeTransformer):
            def visit(self, node: ast.AST) -> ast.AST | list[ast.AST]:
                if isinstance(node, ast.expr):
                    rewrite = references.get(SpanRewrite.span(node))
                    if rewrite is not None:
                        return rewrite.expected(node)
                if not isinstance(node, ast.stmt):
                    return super().visit(node)
                added = imports.get(SpanRewrite.span(node))
                visited = (
                    added.extended(node)
                    if isinstance(node, ast.ImportFrom) and added is not None
                    else super().visit(node)
                )
                inserted = insertions.get(SpanRewrite.span(node))
                return visited if inserted is None else [visited, *inserted.inserted]

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

    OPERAND_PARENTS: ClassVar[tuple[type[ast.AST], ...]] = (
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Attribute,
        ast.Subscript,
        ast.Call,
        ast.Await,
        ast.Starred,
    )
    ATOMS: ClassVar[tuple[type[ast.expr], ...]] = (
        ast.Name,
        ast.Attribute,
        ast.Subscript,
        ast.Call,
        ast.Constant,
    )

    def rewrites(self, context, module, path):
        parents = {
            id(child): parent
            for parent in ast.walk(module)
            for child in ast.iter_child_nodes(parent)
        }
        return tuple(
            SpanRewrite(
                quoted.node,
                self.reference_source(quoted, parents.get(id(quoted.node))),
            )
            for quoted in ModuleReferenceScope(module).unquotable(
                DeclaredInterpreter.of(Path(path))
            )
        )

    @classmethod
    def reference_source(cls, quoted: QuotedTypeName, parent: ast.AST | None) -> str:
        """The string's own text, parenthesized only where it lands as an operand.

        An annotation, base, cast or TypeVar argument, or a subscript slice or
        collection element, holds any expression bare, so `"A | None"` becomes
        `A | None`. As the operand of an operator, attribute or call
        (`None | "A | B"`), a compound expression needs its parentheses; the
        replay proof rejects any rewrite whose parse differs.
        """

        text = quoted.node.value.strip()
        if "\n" in text:
            text = ast.unparse(quoted.expression)
        if isinstance(quoted.expression, cls.ATOMS):
            return text
        operand = isinstance(parent, cls.OPERAND_PARENTS) and not (
            isinstance(parent, ast.Subscript) and parent.slice is quoted.node
        )
        return f"({text})" if operand else text


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
class ReferenceNamedClassOperation(DeclarationReferenceOperation, ABC):
    """Write each rewritable site of the operation's position as an expression on a class, importing the class where needed.

    Each site names its class and replacement (`named`) and how the file comes
    to bind the class there (`NameImport`): already bound, added to a `from`
    statement before the literal, or a new `from owner import Name` after the
    file's last top-level import.
    """

    def rewrites(self, context, module, path):
        identities = context.declared_identities
        positions = IdentitySpelling.positions()
        found: dict[int, SpanRewrite] = {}
        imported: dict[int, dict[str, None]] = {}
        created: dict[str, dict[str, None]] = {}
        for node, detail in IdentityPositions(
            context.parsed_module_for_source_path(path)
        ).spellings():
            position = positions[detail[0]]
            if not (
                position.rewrite(detail, identities, path).rewritable(
                    DeclaredInterpreter.of(Path(path))
                )
                and position.rewrite_kind.tag in self.position_tags
            ):
                continue
            named = self.named(detail, identities, path)
            if named is None:
                continue
            replacement, how = named
            found[id(node)] = SpanRewrite(node, replacement)
            if how.statement is not None:
                imported.setdefault(how.statement, {})[how.name] = None
            elif how.created:
                created.setdefault(how.created, {})[how.name] = None
        source = context.sources_by_file_path[path]
        last_import = identities.class_names.by_path[path].last_import
        return (
            *found.values(),
            *(
                ImportedNamesRewrite.of(source, module.body[index], tuple(names))
                for index, names in sorted(imported.items())
            ),
            *(
                (
                    InsertedImportsRewrite.of(
                        module.body[last_import],
                        {owner: tuple(names) for owner, names in created.items()},
                    ),
                )
                if created
                else ()
            ),
        )

    @abstractmethod
    def named(
        self, detail: tuple[str, ...], identities: DeclaredIdentities, path: str
    ) -> tuple[str, NameImport] | None:
        """The rewritable site's replacement source, and how the file binds the class it names."""

        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ReferenceClassNameOperation(ReferenceNamedClassOperation):
    """Write a class name spelled as a value (passed, returned, compared, matched) as `Name.__name__`.

    `Name.__name__` evaluates to the literal when `Name` is bound, wherever the
    literal runs, to a scanned class that keeps its statement's name
    (`ClassNameOwners.keeps_name`). Either the file binds it already, by a
    top-level class statement or `from` import that runs first and is shadowed
    nowhere, or the file binds the name nowhere and a `from` statement before
    the literal can also import the class without changing the import order
    (`ClassNameOwners.importable`): the name is added to that statement.
    Otherwise a new `from owner import Name` goes after the file's last
    top-level import when the owner module cannot import this one back
    (`ClassNameOwners.new_import`).
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((ClassNameSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Name the class object's __name__ instead of spelling the class name."
    )
    NAME_OF_CLASS: ClassVar[str] = "__name__"

    def named(self, detail, identities, path):
        binding, holding, literal = detail[-3:]
        reference = f"{literal}.{self.NAME_OF_CLASS}"
        if binding:
            return reference, NameImport(literal)
        owners = identities.class_names
        statement = owners.importable(path, int(holding), literal)
        if statement is not None:
            return reference, NameImport(literal, statement=statement.index)
        owner = owners.new_import(path, int(holding), literal)
        if owner is not None:
            return reference, NameImport(literal, created=owner)
        return None


@dataclass(frozen=True, kw_only=True)
class ReferenceEnumMemberOperation(ReferenceNamedClassOperation):
    """Write a string equal to one enum member's value as `Enum.MEMBER.value`.

    `.value` is the member's string itself, so the rewritten expression equals
    the literal wherever it stands, as a key, a comparison operand, an argument
    or a match value; how the file binds the enum is
    `ClassNameOwners.import_for`.
    """

    position_tags: ClassVar[frozenset[str]] = frozenset((EnumValueSpelling.tag,))
    default_rationale: ClassVar[str] = (
        "Reference the enum member instead of spelling its value."
    )

    def named(self, detail, identities, path):
        _tag, holding, literal = detail
        member = identities.enum_member(literal)
        if member is None:
            return None
        how = identities.class_names.import_for(path, int(holding), member.enum)
        return None if how is None else (member.reference, how)


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
        ReferenceEnumMemberOperation,
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
