"""Drop a base another base of the same class already supplies.

`class N(Mixin, ABC)` where `Mixin` is an `ABC` lists `ABC` twice: once by name
and once through `Mixin`. When dropping the name leaves the class's C3 order
exactly as it is (`DeclaredC3.restated_bases`), every lookup is unchanged; what
can still see the drop is code that reads `__bases__` itself. Class creation
code is read by `ClassCreationProof`; every other read in the repository must
either test a named class other than the dropped one, or keep only bases that
are subclasses of a named class the dropped one is not.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    ParsedModuleSet,
    ParsedModuleSetFact,
)
from nominal_refactor_advisor.class_index import CompactIndexedClass
from nominal_refactor_advisor.class_symbols import _referenced_bases
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import RestatedBaseConcept
from nominal_refactor_advisor.refactors.class_creation import ClassCreationProof
from nominal_refactor_advisor.refactors.class_overlap import (
    OverlapClass,
    obsolete_import_edits,
    require_static_c3,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)

BASES_ATTRIBUTES = frozenset(("__bases__", "__orig_bases__"))


@dataclass(frozen=True)
class RestatedBaseDrop:
    """One class's restated bases, proved droppable on the current source."""

    overlap: OverlapClass
    dropped: tuple[tuple[ast.expr, str], ...]
    """Each dropped base expression, with the base that supplies it."""

    @classmethod
    def proved(cls, snapshot: CodemodSourceSnapshot, overlap: OverlapClass) -> RestatedBaseDrop:
        """The class's restated bases, each proved: its order closes statically,
        its creation and every `__bases__` reader see nothing of the drop."""

        node = overlap.node
        referenced = tuple(_referenced_bases(node))
        if len(referenced) != len(node.bases):
            raise NeedsInput(f"`{overlap.qualname}` has a computed base", NeededInput.SOURCE_CHANGE)
        verdicts = snapshot.class_overlap_verdicts
        restated = verdicts.restated_bases(
            overlap.symbol,
            tuple(isinstance(base, (ast.Name, ast.Attribute)) for base, _parts in referenced),
        )
        if not restated:
            raise NeedsInput(f"`{overlap.qualname}` lists no base another base supplies", NeededInput.SOURCE_CHANGE)
        ancestry = require_static_c3(snapshot, overlap.facts)
        listed = verdicts.bases(overlap.symbol)
        dropped_symbols = tuple(listed[index] for index, _supplier in restated)
        header = ClassHeaderSpanSourceAuthority(node, snapshot.sources_by_file_path[overlap.file_path])
        if not header.can_rewrite:
            raise NeedsInput(f"`{overlap.qualname}` has an unrewritable header", NeededInput.OPERATION)
        ClassCreationProof.dropping(snapshot, overlap.facts, ancestry, dropped_symbols).require_unchanged()
        BasesReaders(snapshot).require_unread(overlap.facts, dropped_symbols)
        return cls(overlap, tuple((referenced[index][0], supplier) for index, supplier in restated))

    def header_edits(self, snapshot: CodemodSourceSnapshot, rationale: str) -> tuple[NominalSourceEdit, ...]:
        node = self.overlap.node
        header = ClassHeaderSpanSourceAuthority(node, snapshot.sources_by_file_path[self.overlap.file_path])
        dropped = frozenset(id(base) for base, _supplier in self.dropped)
        return header.source_edits(
            header.with_base_items(
                tuple(header.item_source(base) for base in node.bases if id(base) not in dropped)
            ),
            file_path=self.overlap.file_path,
            rationale=rationale,
        )

    def rationale(self) -> str:
        return " ".join(
            f"Drops `{ast.unparse(base)}` from `{self.overlap.qualname}`'s bases: "
            f"`{supplier.rpartition('.')[2]}` already supplies it."
            for base, supplier in self.dropped
        )


@dataclass(frozen=True)
class BaseListReads(ParsedModuleSetFact):
    """Every read of a class's `__bases__` or `__orig_bases__` in one state's
    modules, found once for every drop proved over that state."""

    reads: tuple[tuple[ParsedModule, ast.expr, ast.AST | None], ...]
    """Each module, the read, and the node it is read in."""

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> BaseListReads:
        found: list[tuple[ParsedModule, ast.expr, ast.AST | None]] = []
        for module in modules:
            reads = tuple(cls.found(module.module))
            if not reads:
                continue
            parents = {
                child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)
            }
            found.extend((module, read, parents.get(read)) for read in reads)
        return cls(tuple(found))

    @staticmethod
    def found(tree: ast.AST) -> Iterator[ast.expr]:
        for node in ast.walk(tree):
            match node:
                case ast.Attribute(attr=attr, ctx=ast.Load()) if attr in BASES_ATTRIBUTES:
                    yield node
                case ast.Call(func=ast.Attribute(attr="get"), args=[ast.Constant(value=str() as key), *_]) if (
                    key in BASES_ATTRIBUTES
                ):
                    yield node
                case ast.Subscript(slice=ast.Constant(value=str() as key)) if key in BASES_ATTRIBUTES:
                    yield node


@dataclass(frozen=True)
class ClassAttributeReaders:
    """Reads of one class attribute (`X.<attribute>`, or `vars(X)` for the
    namespace) anywhere in the repository, each judged by the class it reads."""

    snapshot: CodemodSourceSnapshot

    def reaches(self, module: ParsedModule, read: ast.expr, indexed: CompactIndexedClass) -> bool:
        """Whether the read can see `indexed`'s own attribute: `X.__bases__`,
        `X.__dict__` and `vars(X)` are `X`'s own, and `X.__orig_bases__` is `X`'s or
        an ancestor's, so a read of a named class whose order does not hold
        `indexed` never sees them."""

        match read:
            case ast.Attribute(value=receiver) | ast.Call(func=ast.Name(id="vars"), args=[receiver]):
                pass
            case _:
                return True
        verdicts = self.snapshot.class_overlap_verdicts
        family = self.annotated_family(module, read, receiver)
        if family is not None:
            try:
                return family in verdicts.linearized(indexed.symbol)
            except NeedsInput:
                return True
        named = self.named(module, receiver)
        if named is None or verdicts.class_index.class_for(named) is None:
            return True
        try:
            return indexed.symbol in verdicts.linearized(named)
        except NeedsInput:
            return True

    def annotated_family(self, module: ParsedModule, read: ast.expr, receiver: ast.expr) -> str | None:
        """The scanned class `C` when the receiver is a parameter or local every
        annotation of which is `type[C]`: the read sees only `C` or a subclass."""

        if not isinstance(receiver, ast.Name):
            return None
        scopes = [
            node
            for node in ast.walk(module.module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(inner is read for inner in ast.walk(node))
        ]
        if not scopes:
            return None
        scope = min(scopes, key=lambda node: sum(1 for _ in ast.walk(node)))
        annotations = [
            argument.annotation
            for argument in (*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs)
            if argument.arg == receiver.id
        ] + [
            statement.annotation
            for statement in ast.walk(scope)
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == receiver.id
        ]
        families = set()
        for annotation in annotations:
            match annotation:
                case ast.Subscript(value=ast.Name(id="type" | "Type") | ast.Attribute(attr="type" | "Type"), slice=held):
                    family = self.named(module, held)
                    if family is None or self.snapshot.class_overlap_verdicts.class_index.class_for(family) is None:
                        return None
                    families.add(family)
                case _:
                    return None
        return families.pop() if len(families) == 1 else None

    def named(self, module: ParsedModule, node: ast.expr) -> str | None:
        """The class or function a name or attribute chain spells in `module`: an
        imported or builtin one, or a class the module itself declares."""

        verdicts = self.snapshot.class_overlap_verdicts
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            module, node, line=node.lineno
        )
        if witness is None:
            local = (
                verdicts.class_index.class_for(f"{module.module_name}.{node.id}")
                if isinstance(node, ast.Name)
                else None
            )
            return None if local is None else local.symbol
        indexed = verdicts.ancestor(witness.qualified_name)
        return witness.qualified_name if indexed is None else indexed.symbol


@dataclass(frozen=True)
class BasesReaders(ClassAttributeReaders):
    """Every read of a class's `__bases__` or `__orig_bases__` in the repository."""

    def require_unread(self, indexed: CompactIndexedClass, dropped: tuple[str, ...]) -> None:
        """Each read either tests a named class none of `dropped` is, or keeps only
        bases that subclass a named class none of `dropped` subclasses."""

        verdicts = self.snapshot.class_overlap_verdicts
        orders = {symbol: verdicts.linearized(symbol) for symbol in dropped}
        for module, read, parent in self.snapshot.module_set.fact(BaseListReads).reads:
            if not self.reaches(module, read, indexed):
                continue
            if not self.unaffected(module, read, parent, orders):
                raise NeedsInput(
                    f"{module.file_path}:{read.lineno} `{ast.unparse(parent or read)}` "
                    f"reads bases, and dropping {', '.join(f'`{symbol.rpartition(".")[2]}`' for symbol in dropped)} "
                    f"from `{indexed.qualname}` could change what it sees",
                    NeededInput.PROOF,
                )

    def unaffected(
        self,
        module: ParsedModule,
        read: ast.expr,
        parent: ast.AST | None,
        orders: dict[str, tuple[str, ...]],
    ) -> bool:
        match parent:
            case ast.Compare(left=tested, ops=[ast.In() | ast.NotIn()], comparators=[comparator]) if (
                comparator is read
            ):
                named = self.named(module, tested)
                return named is not None and named not in orders
            case ast.comprehension(iter=iterated, target=ast.Name(id=element), ifs=tests) if iterated is read:
                return any(
                    self.excludes(module, test, element, orders) for test in tests
                )
        return False

    def excludes(
        self, module: ParsedModule, test: ast.expr, element: str, orders: dict[str, tuple[str, ...]]
    ) -> bool:
        """Whether `test` is false for each dropped base: a conjunct that keeps only
        subclasses of a named class no dropped base subclasses, or only generic
        aliases, which a plain base never is."""

        conjuncts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        for conjunct in conjuncts:
            match conjunct:
                case ast.Call(func=ast.Name(id="issubclass"), args=[ast.Name(id=subject), owner], keywords=[]) if (
                    subject == element
                ):
                    named = self.named(module, owner)
                    if named is not None and all(named not in order for order in orders.values()):
                        return True
                case ast.Call(
                    func=ast.Name(id="isinstance"),
                    args=[
                        ast.NamedExpr(
                            value=ast.Call(func=origin, args=[ast.Name(id=subject)], keywords=[])
                        ),
                        ast.Name(id="type"),
                    ],
                    keywords=[],
                ) if subject == element and self.named(module, origin) == "typing.get_origin":
                    return True
        return False


@dataclass(frozen=True, kw_only=True)
class DropRestatedBasesOperation(RepositorySourceReprovedOperation):
    """Drop every base the target class, and each class beside it in its module,
    lists that another of its bases already supplies; then drop each import only
    those bases read, judged once for the whole module."""

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        target = OverlapClass.require(snapshot, self.target)
        drops = [RestatedBaseDrop.proved(snapshot, target)]
        for indexed in snapshot.class_overlap_verdicts.class_index.classes_by_symbol.values():
            if indexed.module_name != target.facts.module_name or indexed.symbol == target.symbol:
                continue
            try:
                drops.append(
                    RestatedBaseDrop.proved(snapshot, OverlapClass.for_symbol(snapshot, indexed.symbol))
                )
            except (NeedsInput, ValueError):
                continue
        rationale = self.rationale_text(" ".join(drop.rationale() for drop in drops))
        return (
            *(edit for drop in drops for edit in drop.header_edits(snapshot, rationale)),
            *obsolete_import_edits(
                snapshot,
                target.file_path,
                tuple(base for drop in drops for base, _supplier in drop.dropped),
                rationale,
            ),
        )


class RestatedBaseFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    RestatedBaseConcept,
):
    """One drop of the restated bases the finding's class lists."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("dropping a restated base needs source context and its class")
        snapshot = context.execution_snapshot()
        indexed = next(
            (
                indexed
                for indexed in snapshot.class_overlap_verdicts.class_index.classes_by_symbol.values()
                if indexed.qualname == evidence.symbol and indexed.file_path == evidence.file_path
            ),
            None,
        )
        if indexed is None:
            return self.rejected_evaluation(f"`{evidence.symbol}` left the source")
        operation = DropRestatedBasesOperation(target=source_rewrite_target(indexed))
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason="Another base already supplies it: the class order and every lookup stay the same.",
            ).with_operation(operation)
        )


__all__ = (
    "BaseListReads",
    "ClassAttributeReaders",
    "BasesReaders",
    "DropRestatedBasesOperation",
    "RestatedBaseDrop",
    "RestatedBaseFindingRecipeSynthesizer",
)
