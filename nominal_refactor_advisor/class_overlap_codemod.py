"""Execute the two class-overlap moves that keep every class identity (#68).

`ComposeExistingClassOperation` gives a class that restates another class's
whole body that class as its first base and deletes the restated copy.
`FoldSharedDeclarationsIntoBaseOperation` moves a block that every direct child
of a base restates into that base, which becomes a keyword-only dataclass.

Both re-derive the scan's overlap verdict from the current source and refuse
unless it names this move and no guard fired. Each then proves what the move
itself changes: method resolution, attribute ownership, dataclass field order,
frozen-ness and, for the fold, every way a moved field can be bound.
"""

from __future__ import annotations

import ast
import keyword as keyword_module
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Self

from .ast_tools import ParsedModule, is_docstring_statement, module_syntax_index
from .class_index import (
    CLASS_METHOD_OWNERSHIP_HOOK_NAMES,
    ClassReferenceRole,
    CompactIndexedClass,
    module_import_aliases,
)
from .class_member_authority_codemod import (
    ClassMemberMoveProofContext,
    ClassMemberPromotionStatement,
    ModuleDeclarationRelocation,
)
from .class_body import ClassBodyDeclaration
from .class_mro import ClassMroAuthority, NativeMroBase
from .class_overlap import (
    ClassOverlapCandidate,
    ClassOverlapMetrics,
    ClassOverlapMove,
    ComposeExisting,
    PromoteToSharedBase,
)
from .codemod import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    RefactorRecipe,
)
from .codemod_authority_claims import AstTargetAuthorityClaim
from .codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from .codemod_imports import ModuleImportMutation
from .codemod_payload import (
    PayloadRecordValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import CodemodOperationPreflightError
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import FindingRecipeEvaluation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext, ResolvedClassTarget
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import NominalSourceEdit, SourceInsertion
from .codemod_statement_source import StatementDeletionSource, StatementSource
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY, ImportBoundNameProjection
from .native_class_mro import NativeClassMroDeclaration
from .models import RefactorFinding
from .refactor_concepts import ClassBodyCompositionConcept
from .semantic_descent import AuthorityClaim, SemanticAuthorityKind

NATIVE_METACLASS_BASES = frozenset(("builtins.type", "abc.ABCMeta"))


@dataclass(frozen=True)
class OverlapClass:
    """One class of an overlap: its current source node and its compact facts."""

    target: ResolvedClassTarget
    facts: CompactIndexedClass

    @classmethod
    def require(
        cls, snapshot: CodemodSourceSnapshot, reference: SourceRewriteTarget
    ) -> Self:
        target = ResolvedClassTarget.from_rewrite_target(snapshot, reference)
        symbol = target.required_symbol(snapshot)
        facts = snapshot.class_overlap_verdicts.class_index.class_for(symbol)
        if facts is None:
            raise ValueError(f"Class {target.qualname!r} has no compact class facts")
        return cls(target, facts)

    @classmethod
    def for_symbol(cls, snapshot: CodemodSourceSnapshot, symbol: str) -> Self:
        facts = snapshot.class_overlap_verdicts.class_index.class_for(symbol)
        if facts is None:
            raise ValueError(f"Class {symbol!r} has no compact class facts")
        return cls.require(snapshot, source_rewrite_target(facts))

    @property
    def symbol(self) -> str:
        return self.facts.symbol

    @property
    def node(self) -> ast.ClassDef:
        return self.target.node

    @property
    def file_path(self) -> str:
        return self.target.file_path

    @property
    def qualname(self) -> str:
        return self.target.qualname

    def declaration(self, snapshot: CodemodSourceSnapshot, name: str) -> ClassBodyDeclaration:
        """The class's own declaration of `name`, as the class-body index sees it."""

        declaration = snapshot.class_overlap_verdicts.bodies.declarations_of(
            self.symbol
        ).get(name)
        if declaration is None:
            raise ValueError(f"`{self.qualname}` does not declare `{name}` itself")
        if declaration.identity is None:
            raise ValueError(f"`{self.qualname}.{name}` reads names nothing binds")
        return declaration

    def statements_declaring(self, names: Iterable[str]) -> tuple[ast.stmt, ...]:
        """The body statements that declare exactly these names, each wholly."""

        wanted = frozenset(names)
        statements: list[ast.stmt] = []
        found: set[str] = set()
        for statement in self.node.body:
            bound = declared_names(statement)
            if not bound & wanted:
                continue
            if not bound <= wanted:
                raise ValueError(
                    f"`{self.qualname}` binds {sorted(bound - wanted)} in the same "
                    "statement as a moved declaration"
                )
            statements.append(statement)
            found |= bound
        if found != wanted:
            raise ValueError(
                f"`{self.qualname}` has no statement for {sorted(wanted - found)}"
            )
        return tuple(statements)

    @property
    def own_field_names(self) -> tuple[str, ...]:
        """The class's own dataclass fields in declaration order; ClassVars excluded."""

        declaration = self.facts.dataclass_declaration
        if declaration is None:
            return ()
        unplaced = tuple(
            field.name
            for field in declaration.fields
            if not field.role.has_known_field_position
        )
        if unplaced:
            raise ValueError(
                f"`{self.qualname}` has {list(unplaced)} with no known field position"
            )
        return tuple(
            field.name
            for field in declaration.fields
            if field.role.contributes_semantic_field
        )

    @property
    def dataclass_options(self) -> tuple[str, ...] | None:
        """The dataclass decorator's arguments, or None for an undecorated class."""

        decorators = self.node.decorator_list
        if len(decorators) > 1:
            raise ValueError(f"`{self.qualname}` has {len(decorators)} decorators")
        options = self.target.dataclass_argument_sources
        if decorators and options is None:
            raise ValueError(f"`{self.qualname}` has a non-dataclass decorator")
        return options

    def option(self, name: str) -> str:
        for option in self.dataclass_options or ():
            key, separator, value = option.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        return "False"


def source_rewrite_target(indexed_class: CompactIndexedClass) -> SourceRewriteTarget:
    return SourceRewriteTarget(
        file_path=indexed_class.file_path, qualname=indexed_class.qualname
    )


def declared_names(statement: ast.stmt) -> frozenset[str]:
    """The class-scope names one declaration statement binds."""

    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset((statement.name,))
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return frozenset((statement.target.id,))
    if isinstance(statement, ast.Assign):
        return frozenset(
            target.id for target in statement.targets if isinstance(target, ast.Name)
        )
    return frozenset()


@dataclass(frozen=True)
class RestatedDeclarations:
    """Declarations a restating class makes exactly as a source class does.

    Equal means equal class-body normal form: kind, name, syntax and the
    declaration every free name is bound to.
    """

    source: OverlapClass
    restating: OverlapClass
    names: tuple[str, ...]

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        source: OverlapClass,
        restating: OverlapClass,
        names: Iterable[str],
    ) -> Self:
        names = tuple(names)
        if not names:
            raise ValueError("A restatement needs at least one declaration")
        for name in names:
            if name.startswith("__") and not name.endswith("__"):
                raise ValueError(f"`{name}` is name-mangled by its owner")
            mine = source.declaration(snapshot, name)
            theirs = restating.declaration(snapshot, name)
            if mine.identity != theirs.identity:
                raise ValueError(
                    f"`{restating.qualname}.{name}` differs from "
                    f"`{source.qualname}.{name}`"
                )
        return cls(source, restating, names)

    @cached_property
    def statements(self) -> tuple[ast.stmt, ...]:
        return self.restating.statements_declaring(self.names)

    def require_no_owner_bound_values(self, snapshot: CodemodSourceSnapshot) -> None:
        """A value built by a class with `__set_name__` learns the class it sits in.

        Moving it changes which class that hook receives, so a moved value that
        calls anything is admitted only when the callee is a class whose whole
        C3 order is indexed and none of those classes declares the hook.
        """

        class_index = snapshot.class_overlap_verdicts.class_index
        resolver = snapshot.class_reference_resolver_for_source_path(
            self.restating.file_path
        ).reference_resolution
        mro_authority = ClassMroAuthority(class_index.classes_by_symbol)
        for statement in self.statements:
            value = getattr(statement, "value", None)
            for call in () if value is None else ast.walk(value):
                if not isinstance(call, ast.Call):
                    continue
                callee = (
                    call.func.value if isinstance(call.func, ast.Subscript) else call.func
                )
                symbol = resolver.symbol_for_node(callee)
                if symbol is None or class_index.class_for(symbol) is None:
                    raise ValueError(
                        f"`{ast.unparse(call.func)}` builds a moved value; "
                        "its callee is no indexed class"
                    )
                mro_type = mro_authority.resolve(symbol).mro_type
                if mro_type is None:
                    raise ValueError(f"`{symbol}` has an open MRO")
                if any(
                    declaration.name == "__set_name__"
                    for owner in mro_type.declarations
                    for declaration in owner.body_declarations
                ):
                    raise ValueError(
                        f"`{symbol}` learns its owner class through `__set_name__`"
                    )


@dataclass(frozen=True)
class ClassReferenceSite:
    """One read of a class in the repository and the syntax around it."""

    file_path: str
    node: ast.expr
    parent: ast.AST | None
    role: ClassReferenceRole
    is_annotation: bool
    is_base: bool

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.node.lineno}"


@dataclass(frozen=True)
class ClassReferenceSites:
    """Every read of some classes in the current source, by role.

    A read is found by resolving each name or attribute chain whose last
    identifier is a class's name or an import alias of it, so a reference
    through an alias assignment shows up as a read of the class in that
    assignment.
    """

    snapshot: CodemodSourceSnapshot
    symbols: frozenset[str]

    @cached_property
    def sites(self) -> tuple[ClassReferenceSite, ...]:
        terminal_names = frozenset(symbol.rsplit(".", 1)[-1] for symbol in self.symbols)
        return tuple(
            site
            for parsed_module in self.snapshot.parsed_modules
            for site in self._module_sites(parsed_module, terminal_names)
        )

    def _module_sites(
        self, parsed_module: ParsedModule, terminal_names: frozenset[str]
    ) -> Iterable[ClassReferenceSite]:
        aliases = module_import_aliases(parsed_module)
        identifiers = terminal_names | frozenset(
            alias
            for alias, qualified_name in aliases.items()
            if qualified_name.rsplit(".", 1)[-1] in terminal_names
        )
        syntax_index = module_syntax_index(parsed_module.module)
        nodes = syntax_index.depth_first_nodes
        parents = syntax_index.parent_indices
        resolver = self.snapshot.class_reference_resolver_for_source_path(
            parsed_module.file_path
        ).reference_resolution
        for node_type in (ast.Name, ast.Attribute):
            for index in syntax_index.node_indices_by_type.get(node_type, ()):
                node = nodes[index]
                identifier = node.id if isinstance(node, ast.Name) else node.attr
                if identifier not in identifiers or not isinstance(node.ctx, ast.Load):
                    continue
                parent = nodes[parents[index]] if parents[index] >= 0 else None
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                if resolver.symbol_for_node(node) not in self.symbols:
                    continue
                grandparent = (
                    nodes[parents[parents[index]]]
                    if parent is not None and parents[parents[index]] >= 0
                    else None
                )
                yield ClassReferenceSite(
                    file_path=parsed_module.file_path,
                    node=node,
                    parent=parent,
                    role=ClassReferenceRole.of_reference(
                        node, parent, grandparent, aliases
                    ),
                    is_annotation=self._in_annotation(index, nodes, parents),
                    is_base=isinstance(parent, ast.ClassDef) and node in parent.bases,
                )

    @staticmethod
    def _in_annotation(index: int, nodes: tuple[ast.AST, ...], parents) -> bool:
        child = nodes[index]
        position = parents[index]
        while position >= 0:
            parent = nodes[position]
            if (
                (isinstance(parent, (ast.arg, ast.AnnAssign)) and parent.annotation is child)
                or (
                    isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and parent.returns is child
                )
            ):
                return True
            if isinstance(parent, ast.stmt):
                return False
            child, position = parent, parents[position]
        return False

    def require_no_identity_reads(self) -> None:
        """No type test, class pattern or type key reads these classes."""

        for site in self.sites:
            if site.role.distinguishes_identity:
                raise ValueError(
                    f"{site.location} reads `{ast.unparse(site.node)}` as a "
                    f"{site.role.value}: the move changes what it answers"
                )

    def require_keyword_construction(self) -> None:
        """Every read builds the class by keyword, tests it, annotates or subclasses.

        Any other read hands the class on as a value, and whoever calls that
        value may bind fields by position.
        """

        for site in self.sites:
            positional = site.role.positional_bindings(site.parent)
            if positional:
                raise ValueError(
                    f"{site.location} binds {positional} field(s) of "
                    f"`{ast.unparse(site.node)}` by position, as a {site.role.value}"
                )
            if (
                site.role.binds_fields
                or site.role.distinguishes_identity
                or site.is_annotation
                or site.is_base
            ):
                continue
            raise ValueError(
                f"{site.location} passes `{ast.unparse(site.node)}` on as a value; "
                "how its callers bind fields is unproved"
            )


def require_static_c3(
    snapshot: CodemodSourceSnapshot, indexed_class: CompactIndexedClass
) -> tuple[CompactIndexedClass, ...]:
    """The class and its ancestors, when type's own C3 orders them all.

    Every base resolves to an indexed class or a native root, no class passes
    creation keywords other than a metaclass, no metaclass overrides `mro`,
    and no class declares `__init_subclass__`. Returns the closed ancestry.
    """

    class_index = snapshot.class_overlap_verdicts.class_index
    reached: dict[str, CompactIndexedClass] = {}
    pending = [indexed_class]
    while pending:
        current = pending.pop()
        if current.symbol in reached:
            continue
        reached[current.symbol] = current
        if not current.base_references_are_complete:
            raise ValueError(f"`{current.qualname}` has unresolved bases")
        for reference in current.base_references:
            qualified_name = ".".join(reference.resolved_parts)
            base = class_index.class_for(qualified_name)
            if base is not None:
                pending.append(base)
            elif NativeMroBase.for_qualified_name(qualified_name) is None:
                raise ValueError(
                    f"`{current.qualname}` has base `{qualified_name}` outside the index"
                )
        if set(current.class_keyword_names) - {"metaclass"}:
            raise ValueError(f"`{current.qualname}` passes class creation keywords")
        if any(
            declaration.name in CLASS_METHOD_OWNERSHIP_HOOK_NAMES
            for declaration in current.body_declarations
        ):
            raise ValueError(f"`{current.qualname}` runs a subclass creation hook")
        if "metaclass" in current.class_keyword_names:
            MetaclassMroProof(snapshot, current).require_type_mro()
    return tuple(reached.values())


@dataclass(frozen=True)
class MetaclassMroProof:
    """A class's metaclass orders bases with type's own `mro`."""

    snapshot: CodemodSourceSnapshot
    indexed_class: CompactIndexedClass

    def require_type_mro(self) -> None:
        target = OverlapClass.for_symbol(self.snapshot, self.indexed_class.symbol)
        keywords = tuple(
            keyword for keyword in target.node.keywords if keyword.arg == "metaclass"
        )
        if len(keywords) != 1:
            raise ValueError(f"`{target.qualname}` has no single metaclass keyword")
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            self.snapshot.parsed_module_for_source_path(target.file_path),
            keywords[0].value,
            line=target.node.lineno,
        )
        if witness is None:
            raise ValueError(f"`{target.qualname}` has an unbound metaclass")
        class_index = self.snapshot.class_overlap_verdicts.class_index
        if class_index.class_for(witness.qualified_name) is not None:
            self._require_indexed(witness.qualified_name)
        else:
            self._require_native(witness.qualified_name)

    def _require_indexed(self, symbol: str) -> None:
        class_index = self.snapshot.class_overlap_verdicts.class_index
        pending, seen = [symbol], set()
        while pending:
            current = pending.pop()
            if current in seen or current in NATIVE_METACLASS_BASES:
                continue
            seen.add(current)
            metaclass = class_index.class_for(current)
            if metaclass is None or not metaclass.base_references_are_complete:
                raise ValueError(f"Metaclass `{current}` has an unproved ancestry")
            if any(declaration.name == "mro" for declaration in metaclass.body_declarations):
                raise ValueError(f"Metaclass `{current}` overrides `mro`")
            pending.extend(
                ".".join(reference.resolved_parts)
                for reference in metaclass.base_references
            )

    @staticmethod
    def _require_native(qualified_name: str) -> None:
        module_name, _, attribute = qualified_name.rpartition(".")
        try:
            metaclass = getattr(import_module(module_name), attribute)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Metaclass `{qualified_name}` is not importable") from error
        if not isinstance(metaclass, type) or not issubclass(metaclass, type):
            raise ValueError(f"`{qualified_name}` is not a metaclass")
        if NativeClassMroDeclaration(metaclass).member_owner("mro") is not type:
            raise ValueError(f"Metaclass `{qualified_name}` overrides `mro`")




def require_verdict(
    snapshot: CodemodSourceSnapshot,
    classes: Iterable[str],
    move: type[ClassOverlapMove],
    *,
    owner_symbol: str | None = None,
) -> ClassOverlapCandidate:
    """The scan's verdict for this move over these classes, with no guard fired.

    With `owner_symbol`, `classes` restate that owner, possibly in several
    pair verdicts that must all permit the move; without it, `classes` are
    exactly one overlap's two classes.
    """

    wanted = frozenset(classes)
    matches = tuple(
        candidate
        for candidate in snapshot.class_overlap_verdicts.candidates
        if candidate.move is move
        and (
            wanted <= candidate.restating_symbols
            if owner_symbol is not None
            else frozenset((candidate.owner.symbol, *candidate.restating_symbols))
            == wanted
        )
        and owner_symbol in (None, candidate.owner.symbol)
    )
    if not matches or (owner_symbol is None and len(matches) != 1):
        raise ValueError(
            f"The current source has {len(matches)} `{move.value}` verdicts over "
            f"{sorted(wanted)}"
        )
    refusals = tuple(receipt for candidate in matches for receipt in candidate.refusals)
    if refusals:
        raise ValueError(
            "The verdict refuses the move: "
            + "; ".join(dict.fromkeys(receipt.detail for receipt in refusals))
        )
    return matches[0]


def obsolete_import_edits(
    snapshot: CodemodSourceSnapshot,
    file_path: str,
    deleted: tuple[ast.stmt, ...],
    rationale: str,
) -> tuple[ModuleImportMutation, ...]:
    """Drop `from` imports that only the deleted statements read.

    A name stays when anything else in the module reads it, when `__all__`
    names it, or when another module imports it from this one.
    """

    parsed_module = snapshot.parsed_module_for_source_path(file_path)
    deleted_nodes = frozenset(
        id(node) for statement in deleted for node in ast.walk(statement)
    )
    loads = tuple(
        node
        for node in ast.walk(parsed_module.module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    read_by_deleted = frozenset(node.id for node in loads if id(node) in deleted_nodes)
    still_read = frozenset(node.id for node in loads if id(node) not in deleted_nodes)
    exported = frozenset(
        element.value
        for statement in parsed_module.module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        )
        and isinstance(statement.value, (ast.List, ast.Tuple))
        for element in statement.value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )
    reexported = frozenset(
        alias.name
        for other in snapshot.parsed_modules
        if other.file_path != parsed_module.file_path
        for statement in ast.walk(other.module)
        if isinstance(statement, ast.ImportFrom)
        and other.module_path_identity.resolve_import_from_module(
            imported_module=statement.module, level=statement.level
        )
        == parsed_module.module_name
        for alias in statement.names
    )
    imported = frozenset(
        alias.name
        for statement in parsed_module.module.body
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
        if ImportBoundNameProjection(statement).alias_bound_name(alias) == alias.name
    )
    obsolete = sorted(
        (read_by_deleted & imported) - still_read - exported - reexported
    )
    if not obsolete:
        return ()
    return (
        ModuleImportMutation.remove_bound_names(
            file_path=file_path, names=obsolete, rationale=rationale
        ),
    )


def required_name_imports(
    snapshot: CodemodSourceSnapshot,
    statements: tuple[ast.stmt, ...],
    source: OverlapClass,
    destination: OverlapClass,
    rationale: str,
) -> tuple[ModuleImportMutation, ...]:
    """Bind every free name of moved statements at the destination as at the source.

    A name already bound to the same declaration needs nothing; an unbound one
    is imported from the repository module declaring it, refused when that
    import would close a module cycle; one bound to something else refuses.
    """

    bindings = snapshot.module_binding_proof
    source_module = snapshot.parsed_module_for_source_path(source.file_path)
    destination_module = snapshot.parsed_module_for_source_path(destination.file_path)
    imports: list[ModuleImportMutation] = []
    names = sorted(
        {
            node.id
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
    )
    for name in names:
        reference = ast.Name(id=name, ctx=ast.Load())
        expected = bindings.reference_or_builtin_witness_at(
            source_module, reference, line=source.node.lineno
        )
        if expected is None:
            continue  # bound inside the moved statement itself: a parameter or local
        actual = bindings.reference_or_builtin_witness_at(
            destination_module, reference, line=destination.node.lineno
        )
        if actual is not None:
            if actual.qualified_name != expected.qualified_name:
                raise ValueError(
                    f"`{name}` is `{actual.qualified_name}` in "
                    f"{destination.file_path}, not `{expected.qualified_name}`"
                )
            continue
        targets = snapshot.source_index.targets_matching_repository_symbol(
            expected.qualified_name
        )
        if len(targets) != 1:
            raise ValueError(f"`{expected.qualified_name}` has no single declaration")
        imports.append(
            ModuleImportMutation.from_source(
                file_path=destination.file_path,
                import_source=snapshot.module_import_graph.required_import_source(
                    importing_file_path=destination.file_path,
                    imported_file_path=targets[0].file_path,
                    imported_name=name,
                ),
                rationale=rationale,
            )
        )
    return tuple(imports)


@dataclass(frozen=True)
class ClassComposition:
    """A class restates an existing class's whole body; it composes that class instead.

    The owner becomes the first base and the restated declarations go. With
    the owner's own C3 order `[owner, object]` and the owner not already an
    ancestor, C3 places it right after the class and leaves every other
    ancestor in order, so a name the class kept resolves as before and a
    deleted name resolves to the owner's equal declaration. Dataclass
    collection then walks the old ancestors, the owner, the class: the field
    order is compared, not assumed.
    """

    snapshot: CodemodSourceSnapshot
    owner: OverlapClass
    restating: OverlapClass
    restated: RestatedDeclarations

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        owner_reference: SourceRewriteTarget,
        restating_reference: SourceRewriteTarget,
    ) -> Self:
        owner = OverlapClass.require(snapshot, owner_reference)
        restating = OverlapClass.require(snapshot, restating_reference)
        require_verdict(
            snapshot, (restating.symbol,), ComposeExisting, owner_symbol=owner.symbol
        )
        body = owner.node.body[1:] if is_docstring_statement(owner.node.body[0]) else owner.node.body
        if not body or any(not declared_names(statement) for statement in body):
            raise ValueError(f"`{owner.qualname}` runs statements beyond declarations")
        restated = RestatedDeclarations.require(
            snapshot,
            owner,
            restating,
            (declaration.name for declaration in owner.facts.body_declarations),
        )
        composition = cls(snapshot, owner, restating, restated)
        composition.require_owner_is_a_root()
        composition.require_field_order()
        restated.require_no_owner_bound_values(snapshot)
        ClassReferenceSites(
            snapshot, frozenset((owner.symbol,))
        ).require_no_identity_reads()
        if owner.file_path == restating.file_path:
            composition.relocation.require_safe(snapshot)
        return composition

    def require_owner_is_a_root(self) -> None:
        owner, restating = self.owner, self.restating
        if owner.node.keywords:
            raise ValueError(f"`{owner.qualname}` passes class creation keywords")
        if not owner.facts.base_references_are_complete or any(
            ".".join(reference.resolved_parts) != NativeMroBase.OBJECT.value
            for reference in owner.facts.base_references
        ):
            raise ValueError(
                f"`{owner.qualname}` has bases; its C3 order is not [owner, object]"
            )
        ancestry = require_static_c3(self.snapshot, restating.facts)
        if any(ancestor.symbol == owner.symbol for ancestor in ancestry):
            raise ValueError(f"`{owner.qualname}` is already an ancestor")
        if owner.dataclass_options != restating.dataclass_options:
            raise ValueError(
                f"`{owner.qualname}` and `{restating.qualname}` differ as dataclasses"
            )
        header = ClassHeaderSpanSourceAuthority(
            restating.node, self.snapshot.sources_by_file_path[restating.file_path]
        )
        if not header.can_rewrite:
            raise ValueError(f"`{restating.qualname}` has an unrewritable header")

    def require_field_order(self) -> None:
        """Old and new dataclass collection give the same names in the same order.

        Inherited names keep their first position either way, so only names new
        to the class are compared: the class's own fields before; the owner's,
        then the class's remaining ones, after.
        """

        restating = self.restating
        if restating.dataclass_options is None:
            return
        class_index = self.snapshot.class_overlap_verdicts.class_index
        inherited = frozenset(
            field_name
            for symbol in class_index.ancestor_symbols(restating.symbol)
            if (ancestor := class_index.class_for(symbol)) is not None
            and ancestor.dataclass_declaration is not None
            for field_name in OverlapClass.for_symbol(self.snapshot, symbol).own_field_names
        )
        moved = frozenset(self.restated.names)
        before = [name for name in restating.own_field_names if name not in inherited]
        after = [
            name
            for name in dict.fromkeys(
                (
                    *self.owner.own_field_names,
                    *(name for name in restating.own_field_names if name not in moved),
                )
            )
            if name not in inherited
        ]
        if before != after:
            raise ValueError(
                f"Composing `{self.owner.qualname}` reorders the fields of "
                f"`{restating.qualname}`: {before} becomes {after}"
            )

    @cached_property
    def relocation(self) -> ModuleDeclarationRelocation:
        return ModuleDeclarationRelocation.before_user(
            self.snapshot, declaration=self.owner.target, user=self.restating.target
        )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        restating = self.restating
        source = self.snapshot.sources_by_file_path[restating.file_path]
        header = ClassHeaderSpanSourceAuthority(restating.node, source)
        deletion = StatementDeletionSource(
            source=source, node=restating.node, file_path=restating.file_path
        )
        same_module = self.owner.file_path == restating.file_path
        return (
            *(
                self.relocation.source_edits(
                    f"Declare `{self.owner.qualname}` before `{restating.qualname}` "
                    "composes it."
                )
                if same_module
                else self.owner_import_edits(rationale)
            ),
            *header.source_edits(
                header.with_prepended_base(self.owner.target.name),
                file_path=restating.file_path,
                rationale=rationale,
            ),
            *deletion.physical_edits(
                file_path=restating.file_path,
                replacements=deletion.replacements_for_statements(
                    self.restated.statements
                ),
                rationale=rationale,
            ),
            *obsolete_import_edits(
                self.snapshot, restating.file_path, self.restated.statements, rationale
            ),
        )

    def owner_import_edits(self, rationale: str) -> tuple[ModuleImportMutation, ...]:
        name = self.owner.target.name
        resolver = self.snapshot.class_reference_resolver_for_source_path(
            self.restating.file_path
        )
        if resolver.symbol_for_reference(ast.Name(id=name, ctx=ast.Load())) == self.owner.symbol:
            return ()
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.snapshot.parsed_module_for_source_path(self.restating.file_path).module.body
        )
        if name in bound:
            raise ValueError(f"`{name}` is bound to another declaration there")
        return (
            ModuleImportMutation.from_source(
                file_path=self.restating.file_path,
                import_source=self.snapshot.module_import_graph.required_import_source(
                    importing_file_path=self.restating.file_path,
                    imported_file_path=self.owner.file_path,
                    imported_name=name,
                ),
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class BaseFold:
    """Every direct child of a base restates one block; the base declares it once.

    The base becomes a frozen-consistent dataclass whose fields are keyword
    only, so each child's positional parameters keep their order minus the
    moved fields. That is safe only when no caller binds a child's fields by
    position: every read of the base and its descendants must build by
    keyword, test, annotate or subclass, and no `cls(...)`, `type(self)(...)`
    or `self.__class__(...)` in their family passes positions.
    """

    snapshot: CodemodSourceSnapshot
    base: OverlapClass
    children: tuple[OverlapClass, ...]
    restated: tuple[RestatedDeclarations, ...]

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        base_reference: SourceRewriteTarget,
        member_names: tuple[str, ...],
    ) -> Self:
        base = OverlapClass.require(snapshot, base_reference)
        class_index = snapshot.class_overlap_verdicts.class_index
        child_symbols = tuple(sorted(class_index.children_by_symbol.get(base.symbol, ())))
        if len(child_symbols) < 2:
            raise ValueError(f"`{base.qualname}` has fewer than two direct children")
        candidate = require_verdict(snapshot, child_symbols, PromoteToSharedBase)
        if not frozenset(member_names) <= frozenset(candidate.shared_names):
            raise ValueError(
                f"{sorted(frozenset(member_names) - frozenset(candidate.shared_names))} "
                "are not restated by every child"
            )
        children = tuple(OverlapClass.for_symbol(snapshot, symbol) for symbol in child_symbols)
        restated = tuple(
            RestatedDeclarations.require(snapshot, children[0], child, member_names)
            for child in children
        )
        fold = cls(snapshot, base, children, restated)
        fold.require_dataclass_family()
        fold.require_lookup()
        for restatement in restated:
            restatement.require_no_owner_bound_values(snapshot)
            fold.require_movable(restatement)
        family = fold.family_symbols
        ClassReferenceSites(
            snapshot, frozenset((base.symbol, *fold.descendant_symbols))
        ).require_keyword_construction()
        fold.require_no_positional_self_construction(family)
        return fold

    @property
    def names(self) -> tuple[str, ...]:
        return self.restated[0].names

    @cached_property
    def descendant_symbols(self) -> tuple[str, ...]:
        return self.snapshot.class_overlap_verdicts.class_index.descendant_symbols(
            self.base.symbol
        )

    @cached_property
    def family_symbols(self) -> tuple[str, ...]:
        class_index = self.snapshot.class_overlap_verdicts.class_index
        return (
            self.base.symbol,
            *class_index.ancestor_symbols(self.base.symbol),
            *self.descendant_symbols,
        )

    @property
    def frozen(self) -> str:
        return self.children[0].option("frozen")

    @property
    def adds_decorator(self) -> bool:
        return self.base.dataclass_options is None

    def require_dataclass_family(self) -> None:
        base = self.base
        if frozenset(self.names) & base.target.bound_names:
            raise ValueError(f"`{base.qualname}` already binds a moved name")
        if self.adds_decorator:
            if any(isinstance(statement, ast.AnnAssign) for statement in base.node.body):
                raise ValueError(
                    f"`{base.qualname}` annotates names that would become fields"
                )
        elif base.option("kw_only") != "True" or base.option("frozen") != self.frozen:
            raise ValueError(f"`{base.qualname}` is a dataclass of another kind")
        class_index = self.snapshot.class_overlap_verdicts.class_index
        if any(
            (ancestor := class_index.class_for(symbol)) is not None
            and ancestor.dataclass_declaration is not None
            for symbol in class_index.ancestor_symbols(base.symbol)
        ):
            raise ValueError(f"`{base.qualname}` inherits dataclass fields")
        for symbol in self.descendant_symbols:
            descendant = OverlapClass.for_symbol(self.snapshot, symbol)
            if descendant.dataclass_options is None:
                raise ValueError(f"`{descendant.qualname}` is not a dataclass")
            if descendant.option("frozen") != self.frozen:
                raise ValueError(f"`{descendant.qualname}` differs in frozen-ness")
            if descendant.option("order") == "True":
                raise ValueError(f"`{descendant.qualname}` orders by field position")
            descendant.own_field_names
            if {"__init__", "__match_args__"} & descendant.target.bound_names:
                raise ValueError(
                    f"`{descendant.qualname}` declares its own construction surface"
                )

    def require_lookup(self) -> None:
        """Each moved name resolves to the base wherever it resolved to a child.

        In every descendant's C3 order a child comes before the base. Once the
        name leaves the children for the base, lookup still lands on the base
        unless a class between them declares it, so that is what is checked.
        Dunder names are refused: a native root may declare them.
        """

        dunders = sorted(name for name in self.names if name.startswith("__"))
        if dunders:
            raise ValueError(f"Moving {dunders} could expose a native declaration")
        class_index = self.snapshot.class_overlap_verdicts.class_index
        authority = ClassMroAuthority(class_index.classes_by_symbol)
        children = frozenset(child.symbol for child in self.children)
        for symbol in self.descendant_symbols:
            mro_type = authority.resolve(symbol).mro_type
            if mro_type is None:
                raise ValueError(f"`{symbol}` has an open MRO")
            order = tuple(declaration.symbol for declaration in mro_type.declarations)
            if self.base.symbol not in order:
                raise ValueError(f"`{self.base.qualname}` is outside the MRO of `{symbol}`")
            for name in self.names:
                declaring = tuple(
                    declaration.symbol
                    for declaration in mro_type.declarations
                    if name in declaration.direct_declared_member_names
                    or any(item.name == name for item in declaration.body_declarations)
                )
                if not declaring or declaring[0] not in children:
                    continue
                between = tuple(
                    owner
                    for owner in declaring
                    if owner not in children
                    and order.index(owner) < order.index(self.base.symbol)
                )
                if between:
                    raise ValueError(
                        f"`{between[0]}` would answer `{symbol}.{name}` instead of "
                        f"`{self.base.qualname}`"
                    )

    def require_movable(self, restatement: RestatedDeclarations) -> None:
        child = restatement.restating
        context = ClassMemberMoveProofContext(
            source_class=child.target,
            destination_class=self.base.target,
            source=self.snapshot.sources_by_file_path[child.file_path],
            module_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                self.snapshot.parsed_module_for_source_path(child.file_path).module.body
            ),
        )
        for statement in restatement.statements:
            member = ClassMemberPromotionStatement.for_statement(statement)
            if member is None:
                raise ValueError(
                    f"`{child.qualname}` line {statement.lineno} is no movable member"
                )
            member.require_safe_move(context)

    def require_no_positional_self_construction(self, family: tuple[str, ...]) -> None:
        for symbol in family:
            facts = self.snapshot.class_overlap_verdicts.class_index.class_for(symbol)
            if facts is None:
                continue
            owner = OverlapClass.for_symbol(self.snapshot, symbol)
            for node in ast.walk(owner.node):
                if isinstance(node, ast.Call) and node.args and _is_own_class(node.func):
                    raise ValueError(
                        f"{owner.file_path}:{node.lineno} builds its own class with "
                        f"{len(node.args)} positional argument(s)"
                    )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        base = self.base
        source = self.snapshot.sources_by_file_path[base.file_path]
        first = self.restated[0]
        first_source = self.snapshot.sources_by_file_path[first.restating.file_path]
        body = ClassBodySourceAuthority(base.node, source)
        header_line = source.splitlines(keepends=True)[base.node.lineno - 1]
        indentation = header_line[: len(header_line) - len(header_line.lstrip())]
        return (
            *(
                (
                    SourceInsertion(
                        file_path=base.file_path,
                        insertion_line=base.node.lineno,
                        inserted_lines=(
                            f"{indentation}@dataclass(frozen={self.frozen}, kw_only=True)\n",
                        ),
                        rationale=rationale,
                    ),
                    *self.dataclass_import_edits(rationale),
                )
                if self.adds_decorator
                else ()
            ),
            *body.geometry.physical_edits(
                file_path=base.file_path,
                replacements=(
                    body.declaration_block_replacement(
                        tuple(
                            (
                                StatementSource(
                                    source=first_source, node=statement
                                ).member_source(body.indentation),
                                isinstance(
                                    statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                                ),
                            )
                            for statement in first.statements
                        )
                    ),
                ),
                rationale=rationale,
            ),
            *required_name_imports(
                self.snapshot, first.statements, first.restating, base, rationale
            ),
            *(
                edit
                for restatement in self.restated
                for edit in self._deletion_edits(restatement, rationale)
            ),
        )

    def _deletion_edits(
        self, restatement: RestatedDeclarations, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        child = restatement.restating
        deletion = StatementDeletionSource(
            source=self.snapshot.sources_by_file_path[child.file_path],
            node=child.node,
            file_path=child.file_path,
        )
        return (
            *deletion.physical_edits(
                file_path=child.file_path,
                replacements=deletion.replacements_for_statements(restatement.statements),
                rationale=rationale,
            ),
            *obsolete_import_edits(
                self.snapshot, child.file_path, restatement.statements, rationale
            ),
        )

    def dataclass_import_edits(self, rationale: str) -> tuple[ModuleImportMutation, ...]:
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            self.snapshot.parsed_module_for_source_path(self.base.file_path),
            ast.Name(id="dataclass", ctx=ast.Load()),
            line=self.base.node.lineno,
        )
        if witness is not None:
            if witness.qualified_name != "dataclasses.dataclass":
                raise ValueError("`dataclass` is bound to something else in the base module")
            return ()
        return (
            ModuleImportMutation.from_source(
                file_path=self.base.file_path,
                import_source="from dataclasses import dataclass\n",
                rationale=rationale,
            ),
        )


def _is_own_class(function: ast.expr) -> bool:
    """`cls`, `type(self)` or `self.__class__`: the receiver's own class."""

    if isinstance(function, ast.Name):
        return function.id == "cls"
    if isinstance(function, ast.Attribute):
        return (
            function.attr == "__class__"
            and isinstance(function.value, ast.Name)
            and function.value.id == "self"
        )
    return (
        isinstance(function, ast.Call)
        and isinstance(function.func, ast.Name)
        and function.func.id == "type"
        and len(function.args) == 1
        and isinstance(function.args[0], ast.Name)
        and function.args[0].id == "self"
    )


@dataclass(frozen=True, kw_only=True)
class ComposeExistingClassOperation(RepositorySourceReprovedOperation):
    """Make a class that restates an existing class's whole body compose that class."""

    owner: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def composition(self, snapshot: CodemodSourceSnapshot) -> ClassComposition:
        return ClassComposition.require(snapshot, self.owner, self.target)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        composition = self.composition(snapshot)
        return composition.source_edits(
            self.rationale_text(
                f"Compose `{composition.owner.qualname}` instead of restating its body."
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        composition = self.composition(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                composition.owner.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class FoldSharedDeclarationsIntoBaseOperation(RepositorySourceReprovedOperation):
    """Move a block every direct child of a base restates into that base."""

    member_names: tuple[str, ...] = codemod_payload_field(
        StringArrayPayloadValueCodec()
    )

    def __post_init__(self) -> None:
        if not self.member_names or len(set(self.member_names)) != len(self.member_names):
            raise ValueError("A fold needs unique member_names")
        if any(
            not name.isidentifier() or keyword_module.iskeyword(name)
            for name in self.member_names
        ):
            raise ValueError("A fold needs Python identifiers")

    def fold(self, snapshot: CodemodSourceSnapshot) -> BaseFold:
        return BaseFold.require(snapshot, self.target, self.member_names)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        fold = self.fold(snapshot)
        return fold.source_edits(
            self.rationale_text(
                f"Declare {', '.join(self.member_names)} once on `{fold.base.qualname}`."
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        fold = self.fold(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                fold.base.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


class ClassOverlapFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    ClassBodyCompositionConcept,
):
    """The verdict's move as a recipe, when the move keeps every class identity.

    Compose-existing becomes one composition per restating class; promote to
    a shared base becomes one fold. Deleting a class and extracting a new
    mixin change or mint identities and stay findings.
    """

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        metrics = finding.metrics
        if context is None or not isinstance(metrics, ClassOverlapMetrics):
            return self.rejected_evaluation("a class overlap needs its verdict and source")
        if not metrics.move_permitted:
            return self.rejected_evaluation("the verdict refuses the move")
        snapshot = context.execution_snapshot()
        verdicts = snapshot.class_overlap_verdicts
        if metrics.move is ComposeExisting:
            owner = verdicts.class_index.class_for(metrics.owner_symbol)
            restating = tuple(
                verdicts.class_index.class_for(symbol) for symbol in metrics.restating_symbols
            )
            if owner is None or None in restating:
                return self.rejected_evaluation("the overlap's classes left the source")
            operations: tuple[RepositorySourceReprovedOperation, ...] = tuple(
                ComposeExistingClassOperation(
                    target=source_rewrite_target(indexed_class),
                    owner=source_rewrite_target(owner),
                )
                for indexed_class in restating
                if indexed_class is not None
            )
        elif metrics.move is PromoteToSharedBase:
            first, second = (
                verdicts.class_index.class_for(symbol)
                for symbol in (metrics.owner_symbol, *metrics.restating_symbols)
            )
            base = (
                None
                if first is None or second is None
                else verdicts.sole_shared_base(first, second)
            )
            if base is None:
                return self.rejected_evaluation("the two classes share no sole base")
            operations = (
                FoldSharedDeclarationsIntoBaseOperation(
                    target=source_rewrite_target(base),
                    member_names=metrics.shared_declaration_names,
                ),
            )
        else:
            return self.rejected_evaluation(
                f"`{metrics.move.value}` changes a class identity; it stays a finding"
            )
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-class-overlap", reason=finding.summary
        )
        for operation in operations:
            try:
                operation.source_edits(context)
            except CodemodOperationPreflightError as error:
                return self.rejected_evaluation(error.report.message)
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)
