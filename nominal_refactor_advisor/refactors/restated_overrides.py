"""Delete a member that restates the definition its class would inherit anyway.

`class BackToModel(WatchEnd)` declaring `settled` exactly as `WatchEnd` does
keeps a second copy of one answer: without it, lookup reaches `WatchEnd.settled`,
the same declaration bound to the same names. A copy is the same only when
the class-body normal form says so (kind, syntax with the docstring dropped,
and what each free name is bound to) and nothing in it reads the class it sits
in (`ClassBodyNormalForm.reads_its_class`: the class cell, a mangled private
name, a name the class body binds). What can still see the deletion is code
that reads the class namespace: creation code is read by `ClassCreationProof`,
and every other `vars(X)` or `X.__dict__` read must either be of a named class
whose order does not hold the class, or ask for one constant key it is not.
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
from nominal_refactor_advisor.class_index import PER_CLASS_NAMES, CompactIndexedClass
from nominal_refactor_advisor.class_mro import native_root_type
from nominal_refactor_advisor.class_overlap import ClassOverlapVerdicts
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_source_edits import NominalSourceEdit
from nominal_refactor_advisor.codemod_statement_source import StatementDeletionSource
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import RestatedOverrideConcept
from nominal_refactor_advisor.refactors.class_creation import (
    ClassCreationProof,
    CreationCode,
    CreationReads,
    MroWalk,
    Role,
)
from nominal_refactor_advisor.refactors.class_overlap import (
    OverlapClass,
    RestatedDeclarations,
    obsolete_import_edits,
    require_static_c3,
    source_rewrite_target,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.restated_bases import ClassAttributeReaders


def inherited_restatements(verdicts: ClassOverlapVerdicts, symbol: str) -> tuple[tuple[str, str], ...]:
    """Each member `symbol` declares exactly as the next class in its C3 order
    declaring that name does, with that class: what lookup reaches once the
    member goes. A member is not one when either copy reads its class, when the
    next binder is native or outside the scan, or when its order is unknown; an
    annotated declaration is not one either, since `__annotations__`, like
    `__slots__`, is read from each class's own body."""

    if symbol not in verdicts.bodies.declarations_by_symbol:
        return ()
    try:
        order = verdicts.linearized(symbol)
    except NeedsInput:
        return ()
    found: list[tuple[str, str]] = []
    for name, mine in verdicts.bodies.declarations_of(symbol).items():
        if name in PER_CLASS_NAMES or mine.declaration.reads_its_class or mine.kind.is_field:
            continue
        inherited = next_binder(verdicts, order[1:], name)
        if inherited is None:
            continue
        theirs = verdicts.bodies.declarations_of(inherited)[name]
        if not theirs.declaration.reads_its_class and mine.same_declaration_as(theirs):
            found.append((name, inherited))
    return tuple(found)


def next_binder(verdicts: ClassOverlapVerdicts, order: tuple[str, ...], name: str) -> str | None:
    """The scanned class of `order` whose own body first declares `name`; none
    when a native or unscanned class comes first, since its namespace is not
    the class-body index's."""

    for symbol in order:
        native = native_root_type(symbol)
        if native is not None:
            if name in vars(native):
                return None
            continue
        if verdicts.class_index.class_for(symbol) is None or symbol not in verdicts.bodies.declarations_by_symbol:
            return None
        if name in verdicts.bodies.declarations_of(symbol):
            return symbol
    return None


@dataclass(frozen=True)
class NamespaceReads(ParsedModuleSetFact):
    """Every read of a class namespace, `vars(X)` or `X.__dict__`, in one state's
    modules, with the node it is read in; an instance's own `self.__dict__` holds
    no class member and is no such read."""

    reads: tuple[tuple[ParsedModule, ast.expr, ast.AST | None, ast.AST | None], ...]
    """Each module, the read, the node it is read in and the node around that.
    A read of a parameter annotated as an instance (any annotation but `type`)
    reads that instance's own `__dict__`, which holds no class member."""

    @classmethod
    def derive(cls, modules: ParsedModuleSet) -> NamespaceReads:
        found: list[tuple[ParsedModule, ast.expr, ast.AST | None, ast.AST | None]] = []
        for module in modules:
            reads = tuple(cls.found(module.module))
            if not reads:
                continue
            parents = {
                child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)
            }
            for read in reads:
                if cls.reads_an_instance(read, parents):
                    continue
                parent = parents.get(read)
                found.append((module, read, parent, None if parent is None else parents.get(parent)))
        return cls(tuple(found))

    @staticmethod
    def reads_an_instance(read: ast.expr, parents: dict[ast.AST, ast.AST]) -> bool:
        match read:
            case ast.Attribute(value=ast.Name(id=name)) | ast.Call(args=[ast.Name(id=name)]):
                pass
            case _:
                return False
        scope = parents.get(read)
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = parents.get(scope)
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        arguments = scope.args
        annotations = [
            argument.annotation
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            if argument.arg == name
        ] + [
            statement.annotation
            for statement in ast.walk(scope)
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ]
        return bool(annotations) and all(
            annotation is not None and NamespaceReads.instance_annotation(annotation) for annotation in annotations
        )

    @staticmethod
    def instance_annotation(annotation: ast.expr) -> bool:
        """An annotation naming a class, not `type[...]`: its values are instances."""

        outer = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        match outer:
            case ast.Name(id="type" | "Type") | ast.Attribute(attr="type" | "Type") | ast.Constant():
                return False
        return True

    @staticmethod
    def found(tree: ast.AST) -> Iterator[ast.expr]:
        for node in ast.walk(tree):
            match node:
                case ast.Attribute(value=ast.Name(id="self")) | ast.Call(args=[ast.Name(id="self")]):
                    continue
                case ast.Attribute(attr="__dict__", ctx=ast.Load()):
                    yield node
                case ast.Call(func=ast.Name(id="vars"), args=[_], keywords=[]):
                    yield node


@dataclass(frozen=True)
class NamespaceReaders(ClassAttributeReaders):
    """Every read of a class namespace in the repository, judged against the
    members a deletion takes from one class."""

    def require_unread(self, indexed: CompactIndexedClass, deleted: frozenset[str]) -> None:
        for module, read, parent, around in self.snapshot.module_set.fact(NamespaceReads).reads:
            if not self.reaches(module, read, indexed) or self.asks_another_key(module, read, parent, around, deleted):
                continue
            raise NeedsInput(
                f"{module.file_path}:{read.lineno} `{ast.unparse(parent or read)}` reads a class "
                f"namespace, and deleting {', '.join(f'`{name}`' for name in sorted(deleted))} from "
                f"`{indexed.qualname}` could change what it sees (a parameter or local annotated "
                "with its class shows it reads an instance)",
                NeededInput.PROOF,
            )

    def asks_another_key(
        self,
        module: ParsedModule,
        read: ast.expr,
        parent: ast.AST | None,
        around: ast.AST | None,
        deleted: frozenset[str],
    ) -> bool:
        """Whether the read asks only for keys no deleted name can be: one
        constant key that is not deleted (`ns["k"]`, `ns.get("k")`, `"k" in ns`),
        or `self.<attribute>` whose every write builds a text with a constant
        prefix or suffix no deleted name has."""

        def other(key: ast.AST | None) -> bool:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value not in deleted
            match key:
                case ast.Attribute(value=ast.Name(id="self"), attr=attribute):
                    return self.keys_avoid(module, read, attribute, deleted)
            return False

        match parent:
            case ast.Subscript(value=value, slice=key) if value is read:
                return other(key)
            case ast.Compare(left=key, ops=[ast.In() | ast.NotIn()], comparators=[value]) if value is read:
                return other(key)
            case ast.Attribute(value=value, attr="get") if value is read:
                return isinstance(around, ast.Call) and around.func is parent and bool(around.args) and other(
                    around.args[0]
                )
        return False

    def keys_avoid(self, module: ParsedModule, read: ast.expr, attribute: str, deleted: frozenset[str]) -> bool:
        """Every write of `attribute` in the repository is `self.<attribute> = f"..."`
        inside the class that reads it, and each written text starts or ends with a
        constant that no deleted name starts or ends with."""

        owners = [
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.ClassDef) and any(inner is read for inner in ast.walk(node))
        ]
        if not owners:
            return False
        owner = min(owners, key=lambda node: sum(1 for _ in ast.walk(node)))
        owned = {id(node) for node in ast.walk(owner)}
        writes = 0
        for scanned in self.snapshot.module_set:
            for node in ast.walk(scanned.module):
                match node:
                    case ast.Assign(targets=targets, value=value):
                        written = [target for target in targets if isinstance(target, ast.Attribute) and target.attr == attribute]
                    case ast.AugAssign(target=ast.Attribute(attr=name) as target, value=value) if name == attribute:
                        return False
                    case ast.Call(func=ast.Name(id="setattr"), args=[_, ast.Constant(value=name), *_]) if name == attribute:
                        return False
                    case _:
                        continue
                for target in written:
                    if not (
                        scanned is module
                        and id(target) in owned
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and self.text_avoids(value, deleted)
                    ):
                        return False
                    writes += 1
        return writes > 0

    @staticmethod
    def text_avoids(value: ast.expr, deleted: frozenset[str]) -> bool:
        if not isinstance(value, ast.JoinedStr) or not value.values:
            return False
        first, last = value.values[0], value.values[-1]
        prefix = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else ""
        suffix = last.value if isinstance(last, ast.Constant) and isinstance(last.value, str) else ""
        return bool(prefix or suffix) and not any(
            name.startswith(prefix) and name.endswith(suffix) for name in deleted
        )


@dataclass(frozen=True)
class MemberDeletionCreationProof(ClassCreationProof):
    """Deleting restated members leaves the class's creation unchanged: its order
    stays, and only creation code reading its own namespace can see them go."""

    def reads(self, code: CreationCode, roles: dict[str, Role]) -> CreationReads:
        return MemberDeletionReads(self, code, roles)


@dataclass
class MemberDeletionReads(CreationReads):
    """One creation function's reads, judged against the deleted members."""

    @property
    def unproved(self) -> str:
        return f"NRA needs a proof that deleting {self.proof.owner_name} leaves it unchanged"


@dataclass(frozen=True)
class RestatedOverrideDeletion:
    """One class's members that restate what it inherits, proved deletable."""

    overlap: OverlapClass
    restated: tuple[tuple[RestatedDeclarations, str], ...]
    """Each restatement, with the class whose copy lookup reaches instead."""

    @classmethod
    def proved(cls, snapshot: CodemodSourceSnapshot, overlap: OverlapClass) -> RestatedOverrideDeletion:
        verdicts = snapshot.class_overlap_verdicts
        found = inherited_restatements(verdicts, overlap.symbol)
        if not found:
            raise NeedsInput(
                f"`{overlap.qualname}` restates nothing it would inherit", NeededInput.SOURCE_CHANGE
            )
        names = frozenset(name for name, _inherited in found)
        dunders = sorted(name for name in names if name.startswith("__") and name.endswith("__"))
        if dunders:
            raise NeedsInput(
                f"`{overlap.qualname}` restates {', '.join(f'`{name}`' for name in dunders)}, which "
                "class creation and its decorators treat by name: NRA needs a proof that deleting it "
                "leaves the class unchanged",
                NeededInput.PROOF,
            )
        ancestry = require_static_c3(snapshot, overlap.facts)
        order = verdicts.linearized(overlap.symbol)
        proof = MemberDeletionCreationProof(
            snapshot,
            overlap.facts,
            ancestry,
            " + ".join(f"`{overlap.qualname}.{name}`" for name in sorted(names)),
            names,
            frozenset(),
            MroWalk(order, order, overlap.symbol, names),
        )
        restated = []
        for name, inherited in found:
            binders = proof.binders(order, False, name)
            if binders[:2] != (overlap.symbol, inherited):
                raise NeedsInput(
                    f"`{name}` is bound by {', '.join(f'`{symbol}`' for symbol in binders)} along "
                    f"`{overlap.qualname}`'s order, not first by `{inherited}`",
                    NeededInput.PROOF,
                )
            restatement = RestatedDeclarations.require(
                snapshot, OverlapClass.for_symbol(snapshot, inherited), overlap, (name,)
            )
            for statement in restatement.statements:
                cls.require_once_per_class(proof, overlap, statement)
            restated.append((restatement, inherited))
        cls.require_plain_decorators(proof, overlap)
        proof.require_unchanged()
        NamespaceReaders(snapshot).require_unread(overlap.facts, names)
        return cls(overlap, tuple(restated))

    @staticmethod
    def require_once_per_class(
        proof: ClassCreationProof, overlap: OverlapClass, statement: ast.stmt
    ) -> None:
        """What the class body evaluates for the member does nothing per class:
        it calls nothing, a decorator NRA models aside."""

        evaluated: list[ast.AST] = []
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in statement.decorator_list:
                if proof.function_decorator(overlap.facts, decorator) is None:
                    raise NeedsInput(
                        f"`@{ast.unparse(decorator)}` on `{overlap.qualname}.{statement.name}` runs "
                        "once for each class body: NRA needs a proof that one run serves both",
                        NeededInput.PROOF,
                    )
            evaluated.extend((*statement.args.defaults, *(d for d in statement.args.kw_defaults if d)))
        elif isinstance(statement, ast.Assign):
            evaluated.append(statement.value)
        if any(isinstance(node, ast.Call) for value in evaluated for node in ast.walk(value)):
            raise NeedsInput(
                f"`{overlap.qualname}` builds a value for `{min(_names(statement))}` with a call, once "
                "for each class body: NRA needs a proof that one value serves both",
                NeededInput.PROOF,
            )

    @staticmethod
    def require_plain_decorators(proof: ClassCreationProof, overlap: OverlapClass) -> None:
        """A class decorator may read the namespace it receives; `dataclass` reads
        only annotated names and the methods it generates, neither of which a
        deletion here takes."""

        for decorator in overlap.node.decorator_list:
            called = decorator.func if isinstance(decorator, ast.Call) else decorator
            if proof.qualified(overlap.facts, called, decorator.lineno) != "dataclasses.dataclass":
                raise NeedsInput(
                    f"`{overlap.qualname}` is decorated by `{ast.unparse(decorator)}`, which may read "
                    "its namespace: NRA needs a proof that deleting members leaves it unchanged",
                    NeededInput.PROOF,
                )

    def edits(self, snapshot: CodemodSourceSnapshot, rationale: str) -> tuple[NominalSourceEdit, ...]:
        deletion = StatementDeletionSource(
            source=snapshot.sources_by_file_path[self.overlap.file_path],
            node=self.overlap.node,
            file_path=self.overlap.file_path,
        )
        written = tuple(
            statement for restatement, _inherited in self.restated for statement in restatement.written_statements
        )
        statements = tuple(
            statement for restatement, _inherited in self.restated for statement in restatement.statements
        )
        return (
            *deletion.physical_edits(
                file_path=self.overlap.file_path,
                replacements=deletion.replacements_for_statements(written),
                rationale=rationale,
            ),
            *obsolete_import_edits(snapshot, self.overlap.file_path, statements, rationale),
        )

    def rationale(self) -> str:
        return " ".join(
            f"Deletes `{self.overlap.qualname}.{name}`: `{inherited.rpartition('.')[2]}` declares it the same way."
            for restatement, inherited in self.restated
            for name in restatement.names
        )


def _names(statement: ast.stmt) -> frozenset[str]:
    if isinstance(statement, ast.Assign):
        return frozenset(target.id for target in statement.targets if isinstance(target, ast.Name)) or frozenset(("?",))
    return frozenset((getattr(statement, "name", "?"),))


@dataclass(frozen=True, kw_only=True)
class DeleteRestatedOverridesOperation(RepositorySourceReprovedOperation):
    """Delete every member the target class declares exactly as the class it
    would inherit the member from, then each import only those members read."""

    def source_edits_from_snapshot(self, snapshot: CodemodSourceSnapshot) -> tuple[NominalSourceEdit, ...]:
        deletion = RestatedOverrideDeletion.proved(snapshot, OverlapClass.require(snapshot, self.target))
        return deletion.edits(snapshot, self.rationale_text(deletion.rationale()))


class RestatedOverrideFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    RestatedOverrideConcept,
):
    """One deletion of the members the finding's class restates."""

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation("deleting a restated member needs source context and its class")
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
        operation = DeleteRestatedOverridesOperation(target=source_rewrite_target(indexed))
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-{operation.operation_key()}",
                reason="The class inherits the same declaration: every lookup reaches an identical copy.",
            ).with_operation(operation)
        )


__all__ = (
    "DeleteRestatedOverridesOperation",
    "MemberDeletionCreationProof",
    "NamespaceReaders",
    "NamespaceReads",
    "RestatedOverrideDeletion",
    "RestatedOverrideFindingRecipeSynthesizer",
    "inherited_restatements",
    "next_binder",
)
