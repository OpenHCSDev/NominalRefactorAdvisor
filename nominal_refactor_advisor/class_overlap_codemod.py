"""Execute the class-overlap moves (#68), and the owner moves a picker chose (#67).

`ComposeExistingClassOperation` gives a class that restates another class's
whole body that class as its first base and deletes the restated copy.
`FoldSharedDeclarationsIntoBaseOperation` moves a block that every direct child
of a base restates into that base, which becomes a keyword-only dataclass.

An extracted block needs an owner a codemod cannot choose. On a picker's
choice, `CreateMixinOperation` declares the block once in a new class, and
`ExtendOwnerOperation` gives an existing partial owner the declarations it
lacks; the restating classes then compose that owner by their own
compose-existing verdicts at the next stage.

Every operation re-derives the scan's verdict from the current source and
refuses unless it names this move and no guard fired. Each then proves what the
move itself changes: method resolution, attribute ownership, dataclass field
order, frozen-ness and, where fields are added, every way a field can be bound.
"""
from __future__ import annotations

import ast
import difflib
from abc import ABC, abstractmethod
import keyword as keyword_module
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from importlib import import_module
from typing import Self

from .ast_projection import CallKeywords
from .ast_tools import ParsedModule, is_docstring_statement
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
from .class_body import ProvedClassBodyDeclaration
from .class_mro import ClassMroAuthority, NativeMroBase
from .class_overlap import (
    ClassOverlapCandidate,
    ClassOverlapMetrics,
    ClassOverlapMove,
    ClassOverlapVerdicts,
    ComposeExisting,
    ExtractMixin,
    PromoteToSharedBase,
)
from .codemod import (
    FindingEvidenceActionKeysMixin,
    SelfDeclaredRecipeSynthesizer,
    RefactorRecipe,
)
from .codemod_authority_claims import AstTargetAuthorityClaim
from .codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from .codemod_imports import ModuleImportMutation
from .codemod_payload import (
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from .codemod_preflight import CodemodOperationPreflightError
from .codemod_reproof import RepositorySourceReprovedOperation
from .codemod_runtime import FindingRecipeEvaluation
from .codemod_runtime import CodemodSourceSnapshot
from .codemod_selection_context import CodemodSelectorContext, ResolvedClassTarget
from .codemod_selector_models import SourceRewriteTarget
from .codemod_source_edits import (
    NominalSourceEdit,
    SourceInsertion,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from .codemod_statement_source import StatementDeletionSource, StatementSource
from .lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY, ImportBoundNameProjection
from .native_class_mro import NativeClassMroDeclaration
from .models import RefactorFinding
from .owner_candidates import (
    ExistingClassOwner,
    ExistingClassesOwner,
    NewMixinOwner,
    OwnerCandidate,
    SharedBaseOwner,
)
from .refactor_concepts import ClassBodyCompositionConcept
from .semantic_descent import AuthorityClaim, SemanticAuthorityKind

NATIVE_METACLASS_BASES = frozenset(("builtins.type", "abc.ABCMeta"))


@dataclass(frozen=True)
class OverlapClass(ABC):
    """One class of an overlap: its current source node and its compact facts."""

    target: ResolvedClassTarget
    facts: CompactIndexedClass

    @staticmethod
    def of(target: ResolvedClassTarget, facts: CompactIndexedClass) -> OverlapClass:
        """The decorator's dataclass arguments, read once, pick the class kind."""

        options = target.dataclass_argument_sources
        if options is None:
            return PlainOverlapClass(target, facts)
        return DataclassOverlapClass(target, facts, options)

    @classmethod
    def require(
        cls, snapshot: CodemodSourceSnapshot, reference: SourceRewriteTarget
    ) -> OverlapClass:
        target = ResolvedClassTarget.from_rewrite_target(snapshot, reference)
        symbol = target.required_symbol(snapshot)
        facts = snapshot.class_overlap_verdicts.class_index.class_for(symbol)
        if facts is None:
            raise ValueError(f"Class {target.qualname!r} has no compact class facts")
        return cls.of(target, facts)

    @classmethod
    def for_symbol(cls, snapshot: CodemodSourceSnapshot, symbol: str) -> OverlapClass:
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

    def declaration(
        self, snapshot: CodemodSourceSnapshot, name: str
    ) -> ProvedClassBodyDeclaration:
        """The class's own declaration of `name`, as the class-body index sees it."""

        declaration = snapshot.class_overlap_verdicts.bodies.declarations_of(
            self.symbol
        ).get(name)
        if declaration is None:
            raise ValueError(f"`{self.qualname}` does not declare `{name}` itself")
        return declaration.require_proved(f"`{self.qualname}.{name}`")

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

        names = self.facts.own_dataclass_field_names
        if names is None:
            unplaced = [
                field.name
                for field in self.facts.dataclass_declaration.fields
                if not field.role.has_known_field_position
            ]
            raise ValueError(
                f"`{self.qualname}` has {unplaced} with no known field position"
            )
        return names

    def _require_single_decorator(self) -> None:
        decorators = self.node.decorator_list
        if len(decorators) > 1:
            raise ValueError(f"`{self.qualname}` has {len(decorators)} decorators")

    @property
    @abstractmethod
    def dataclass_options_key(self) -> tuple[tuple[str, ...], ...]:
        """The dataclass decorator's arguments, if decorated, for kind comparison."""
        raise NotImplementedError

    @abstractmethod
    def keyword_only_composed_by(
        self, composition: ClassComposition
    ) -> KeywordOnlyComposition | None:
        """The keyword-only proof composing this class needs; a plain class has
        no field order, so it needs none."""
        raise NotImplementedError

    @abstractmethod
    def require_order_kept_by(self, composition: CoveringComposition) -> None:
        """Composing the covering owners keeps this composer's field order; a
        plain class has none."""
        raise NotImplementedError

    @abstractmethod
    def require_construction_extended_by(self, extension: OwnerExtension) -> None:
        """Every construction of this owner's family still binds once the
        extension adds its members; a plain class binds no fields."""
        raise NotImplementedError

    @abstractmethod
    def require_accepts_moved_fields(self, frozen: str) -> None:
        """As a fold base, this class can hold the moved fields with this frozen-ness."""
        raise NotImplementedError

    @abstractmethod
    def decorator_edits_for(
        self, fold: BaseFold, indentation: str, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        """The edits that make this fold base a dataclass, if it is not one."""
        raise NotImplementedError

    @abstractmethod
    def option(self, name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def require_dataclass(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class PlainOverlapClass(OverlapClass):
    """An overlap class with no dataclass decorator."""

    @property
    def dataclass_options_key(self) -> tuple[tuple[str, ...], ...]:
        self._require_single_decorator()
        if self.node.decorator_list:
            raise ValueError(f"`{self.qualname}` has a non-dataclass decorator")
        return ()

    def keyword_only_composed_by(
        self, composition: ClassComposition
    ) -> KeywordOnlyComposition | None:
        self.dataclass_options_key
        return None

    def require_order_kept_by(self, composition: CoveringComposition) -> None:
        self.dataclass_options_key

    def require_construction_extended_by(self, extension: OwnerExtension) -> None:
        self.dataclass_options_key

    def decorator_edits_for(
        self, fold: BaseFold, indentation: str, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        self.dataclass_options_key
        return (
            SourceInsertion(
                file_path=self.file_path,
                insertion_line=self.node.lineno,
                inserted_lines=(
                    f"{indentation}@dataclass(frozen={fold.frozen}, kw_only=True)\n",
                ),
                rationale=rationale,
            ),
            *fold.dataclass_import_edits(rationale),
        )

    def require_accepts_moved_fields(self, frozen: str) -> None:
        self.dataclass_options_key
        if any(isinstance(statement, ast.AnnAssign) for statement in self.node.body):
            raise ValueError(
                f"`{self.qualname}` annotates names that would become fields"
            )

    def option(self, name: str) -> str:
        self.dataclass_options_key
        return "False"

    def require_dataclass(self) -> None:
        self.dataclass_options_key
        raise ValueError(f"`{self.qualname}` is not a dataclass")


@dataclass(frozen=True)
class DataclassOverlapClass(OverlapClass):
    """An overlap class decorated `@dataclass(...)`, with its argument sources."""

    options: tuple[str, ...]

    @property
    def dataclass_options_key(self) -> tuple[tuple[str, ...], ...]:
        self._require_single_decorator()
        return (self.options,)

    def keyword_only_composed_by(
        self, composition: ClassComposition
    ) -> KeywordOnlyComposition | None:
        self._require_single_decorator()
        return composition.dataclass_keyword_only()

    def require_order_kept_by(self, composition: CoveringComposition) -> None:
        self._require_single_decorator()
        composition.require_dataclass_order(self)

    def require_construction_extended_by(self, extension: OwnerExtension) -> None:
        self._require_single_decorator()
        extension.require_dataclass_construction()

    def decorator_edits_for(
        self, fold: BaseFold, indentation: str, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        self._require_single_decorator()
        return ()

    def require_accepts_moved_fields(self, frozen: str) -> None:
        if self.option("kw_only") != "True" or self.option("frozen") != frozen:
            raise ValueError(f"`{self.qualname}` is a dataclass of another kind")

    def option(self, name: str) -> str:
        self._require_single_decorator()
        for option in self.options:
            key, separator, value = option.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        return "False"

    def require_dataclass(self) -> None:
        self._require_single_decorator()


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
            if not mine.same_declaration_as(theirs):
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
        syntax_index = parsed_module.syntax_index
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
        if dataclass_options_besides_kw_only(owner) != dataclass_options_besides_kw_only(
            restating
        ):
            raise ValueError(
                f"`{owner.qualname}` and `{restating.qualname}` differ as dataclasses"
            )
        header = ClassHeaderSpanSourceAuthority(
            restating.node, self.snapshot.sources_by_file_path[restating.file_path]
        )
        if not header.can_rewrite:
            raise ValueError(f"`{restating.qualname}` has an unrewritable header")

    def require_field_order(self) -> None:
        """Old and new dataclass collection bind the same values.

        Either the class's own new field names keep their order (the class's
        own fields before; the owner's, then the class's remaining ones,
        after), or the owner is keyword-only and the class becomes so too,
        which `KeywordOnlyComposition` proves safe.
        """

        if self.keyword_only is not None:
            self.keyword_only.require()

    @cached_property
    def keyword_only(self) -> KeywordOnlyComposition | None:
        """The keyword-only proof this composition needs, if its order or its
        keyword-only-ness changes."""

        return self.restating.keyword_only_composed_by(self)

    def dataclass_keyword_only(self) -> KeywordOnlyComposition | None:
        """The keyword-only proof a dataclass composer needs."""

        restating = self.restating
        before, after = composed_field_order(
            self.snapshot, self.owner.own_field_names, restating
        )
        if before == after and self.owner.option("kw_only") == restating.option("kw_only"):
            return None
        if self.owner.option("kw_only") != "True":
            raise ValueError(
                f"Composing `{self.owner.qualname}` reorders the fields of "
                f"`{restating.qualname}`: {before} becomes {after}, and the owner "
                "is not keyword-only"
            )
        return KeywordOnlyComposition(self.snapshot, restating)

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
            *(
                ()
                if self.keyword_only is None
                else self.keyword_only.decorator_edits(rationale)
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
        return _owner_import_edits(self.snapshot, self.owner, self.restating, rationale)


@dataclass(frozen=True)
class CoveringComposition:
    """Classes restating a block that existing classes already own between them
    compose those classes instead (#96).

    The proof is the extract-mixin verdict's permitted `compose_existing`
    owner on the current source. Each composer restates every owner's whole
    body, so deleting its copies loses nothing; each owner is a root, so C3
    places the owners right after the composer, in the given order, ahead of
    its old ancestors. The owners are declared in other modules or before
    every composer, and the composers' dataclass field order must come out
    the same: no keyword-only route is taken for owners other classes use.
    """

    snapshot: CodemodSourceSnapshot
    owners: tuple[OverlapClass, ...]
    composers: tuple[OverlapClass, ...]
    restated: tuple[tuple[RestatedDeclarations, ...], ...]

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        owner_references: Sequence[SourceRewriteTarget],
        composer_references: Sequence[SourceRewriteTarget],
    ) -> Self:
        owners = tuple(OverlapClass.require(snapshot, reference) for reference in owner_references)
        composers = tuple(
            OverlapClass.require(snapshot, reference) for reference in composer_references
        )
        if len(owners) < 2 or not composers:
            raise ValueError("A covering composition needs two owners and a composer")
        block = tuple(
            declaration.name for owner in owners for declaration in owner.facts.body_declarations
        )
        require_owner_candidate(
            snapshot,
            composers[0].symbol,
            "compose_existing:" + "+".join(owner.symbol for owner in owners),
            block,
        )
        restated = tuple(
            tuple(
                RestatedDeclarations.require(
                    snapshot,
                    owner,
                    composer,
                    (declaration.name for declaration in owner.facts.body_declarations),
                )
                for owner in owners
            )
            for composer in composers
        )
        composition = cls(snapshot, owners, composers, restated)
        composition.require_roots()
        composition.require_field_order()
        for restatements in restated:
            for restatement in restatements:
                restatement.require_no_owner_bound_values(snapshot)
        ClassReferenceSites(
            snapshot, frozenset(owner.symbol for owner in owners)
        ).require_no_identity_reads()
        return composition

    def require_roots(self) -> None:
        for owner in self.owners:
            if owner.node.keywords:
                raise ValueError(f"`{owner.qualname}` passes class creation keywords")
            if not owner.facts.base_references_are_complete or any(
                ".".join(reference.resolved_parts) != NativeMroBase.OBJECT.value
                for reference in owner.facts.base_references
            ):
                raise ValueError(
                    f"`{owner.qualname}` has bases; its C3 order is not [owner, object]"
                )
        for composer in self.composers:
            ancestry = {ancestor.symbol for ancestor in require_static_c3(self.snapshot, composer.facts)}
            for owner in self.owners:
                if owner.symbol in ancestry:
                    raise ValueError(f"`{owner.qualname}` is already an ancestor")
                if dataclass_options_besides_kw_only(owner) != dataclass_options_besides_kw_only(
                    composer
                ):
                    raise ValueError(
                        f"`{owner.qualname}` and `{composer.qualname}` differ as dataclasses"
                    )
                if owner.file_path == composer.file_path and (
                    owner.node.lineno > composer.node.lineno
                ):
                    raise ValueError(
                        f"`{owner.qualname}` is declared after `{composer.qualname}`"
                    )
            header = ClassHeaderSpanSourceAuthority(
                composer.node, self.snapshot.sources_by_file_path[composer.file_path]
            )
            if not header.can_rewrite:
                raise ValueError(f"`{composer.qualname}` has an unrewritable header")

    def require_field_order(self) -> None:
        for composer in self.composers:
            composer.require_order_kept_by(self)

    def require_dataclass_order(self, composer: OverlapClass) -> None:
        """Dataclass collection walks the bases last first, so the owners' fields
        come in that order, then the composer's remaining own fields."""

        before, after = composed_field_order(
            self.snapshot,
            tuple(name for owner in reversed(self.owners) for name in owner.own_field_names),
            composer,
        )
        if before != after or any(
            owner.option("kw_only") != composer.option("kw_only") for owner in self.owners
        ):
            raise ValueError(
                f"Composing {', '.join(f'`{owner.qualname}`' for owner in self.owners)} "
                f"reorders the fields of `{composer.qualname}`: {before} becomes {after}"
            )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        edits: list[NominalSourceEdit] = []
        imported: set[tuple[str, str]] = set()
        for composer, restatements in zip(self.composers, self.restated, strict=True):
            source = self.snapshot.sources_by_file_path[composer.file_path]
            header = ClassHeaderSpanSourceAuthority(composer.node, source)
            deletion = StatementDeletionSource(
                source=source, node=composer.node, file_path=composer.file_path
            )
            statements = tuple(
                statement
                for restatement in restatements
                for statement in restatement.statements
            )
            for owner in self.owners:
                if owner.file_path == composer.file_path or (
                    composer.file_path,
                    owner.symbol,
                ) in imported:
                    continue
                imported.add((composer.file_path, owner.symbol))
                edits.extend(_owner_import_edits(self.snapshot, owner, composer, rationale))
            edits.extend(
                (
                    *header.source_edits(
                        header.with_base_items(
                            (
                                *(owner.target.name for owner in self.owners),
                                *header.base_items,
                            )
                        ),
                        file_path=composer.file_path,
                        rationale=rationale,
                    ),
                    *deletion.physical_edits(
                        file_path=composer.file_path,
                        replacements=deletion.replacements_for_statements(statements),
                        rationale=rationale,
                    ),
                    *obsolete_import_edits(
                        self.snapshot, composer.file_path, statements, rationale
                    ),
                )
            )
        return tuple(edits)


def _owner_import_edits(
    snapshot: CodemodSourceSnapshot,
    owner: OverlapClass,
    composer: OverlapClass,
    rationale: str,
) -> tuple[ModuleImportMutation, ...]:
    """The import that binds the owner's name where the composer is declared."""

    name = owner.target.name
    resolver = snapshot.class_reference_resolver_for_source_path(composer.file_path)
    if resolver.symbol_for_reference(ast.Name(id=name, ctx=ast.Load())) == owner.symbol:
        return ()
    bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
        snapshot.parsed_module_for_source_path(composer.file_path).module.body
    )
    if name in bound:
        raise ValueError(f"`{name}` is bound to another declaration there")
    return (
        ModuleImportMutation.from_source(
            file_path=composer.file_path,
            import_source=snapshot.module_import_graph.required_import_source(
                importing_file_path=composer.file_path,
                imported_file_path=owner.file_path,
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
        require_no_positional_self_construction(snapshot, family)
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

    def require_dataclass_family(self) -> None:
        base = self.base
        if frozenset(self.names) & base.target.bound_names:
            raise ValueError(f"`{base.qualname}` already binds a moved name")
        base.require_accepts_moved_fields(self.frozen)
        class_index = self.snapshot.class_overlap_verdicts.class_index
        for symbol in class_index.ancestor_symbols(base.symbol):
            if (ancestor := class_index.class_for(symbol)) is not None:
                ancestor.dataclass_declaration.refuse_inherited_fields(base.qualname)
        for symbol in self.descendant_symbols:
            descendant = OverlapClass.for_symbol(self.snapshot, symbol)
            descendant.require_dataclass()
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

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        base = self.base
        source = self.snapshot.sources_by_file_path[base.file_path]
        first = self.restated[0]
        first_source = self.snapshot.sources_by_file_path[first.restating.file_path]
        body = ClassBodySourceAuthority(base.node, source)
        header_line = source.splitlines(keepends=True)[base.node.lineno - 1]
        indentation = header_line[: len(header_line) - len(header_line.lstrip())]
        return (
            *base.decorator_edits_for(self, indentation, rationale),
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


def require_no_positional_self_construction(
    snapshot: CodemodSourceSnapshot, symbols: Iterable[str]
) -> None:
    """No `cls(...)`, `type(self)(...)` or `self.__class__(...)` in these classes
    passes positional arguments: no read of the class names shows those calls."""

    for symbol in symbols:
        if snapshot.class_overlap_verdicts.class_index.class_for(symbol) is None:
            continue
        owner = OverlapClass.for_symbol(snapshot, symbol)
        for node in ast.walk(owner.node):
            if isinstance(node, ast.Call) and node.args and _is_own_class(node.func):
                raise ValueError(
                    f"{owner.file_path}:{node.lineno} builds its own class with "
                    f"{len(node.args)} positional argument(s)"
                )


@dataclass(frozen=True)
class KeywordOnlyComposition:
    """A composer whose field order changes becomes keyword-only.

    Composing puts the owner's fields ahead of the composer's own, so the
    order dataclass collection gives changes. With `kw_only=True` on the owner
    and on the composer every field of both is keyword-only, and a call that
    binds by keyword binds the same values in either order. That holds when no
    read of the composer or a descendant binds a field by position or hands
    the class on as a value, no `cls(...)` in their bodies passes positions,
    none declares `order=True` (which compares field tuples in order), none
    declares its own `__init__` or `__match_args__`, and no method of theirs or
    of an indexed ancestor digests a whole-instance serialization (`repr`,
    `str`, `astuple`, `asdict`, `fields` or `pickle.dumps` of `self`, in a method
    that calls `hashlib` or `json.dumps` without `sort_keys=True`): that digest
    would change with the field order. Cache identities are refused too until
    they are told apart from persisted ones. The repr text and the order
    `fields()` and `astuple()` report still change.
    """

    snapshot: CodemodSourceSnapshot
    composer: OverlapClass

    @cached_property
    def family(self) -> tuple[str, ...]:
        return (
            self.composer.symbol,
            *self.snapshot.class_overlap_verdicts.class_index.descendant_symbols(
                self.composer.symbol
            ),
        )

    def require(self) -> None:
        for symbol in self.family:
            member = OverlapClass.for_symbol(self.snapshot, symbol)
            if member.option("order") == "True":
                raise ValueError(f"`{member.qualname}` orders by field position")
            if {"__init__", "__match_args__"} & member.target.bound_names:
                raise ValueError(
                    f"`{member.qualname}` declares its own construction surface"
                )
        ClassReferenceSites(
            self.snapshot, frozenset(self.family)
        ).require_keyword_construction()
        require_no_positional_self_construction(self.snapshot, self.family)
        self.require_no_field_order_digest()

    def require_no_field_order_digest(self) -> None:
        class_index = self.snapshot.class_overlap_verdicts.class_index
        symbols = dict.fromkeys(
            symbol
            for member in self.family
            for symbol in (member, *class_index.ancestor_symbols(member))
            if class_index.class_for(symbol) is not None
        )
        for symbol in symbols:
            owner = OverlapClass.for_symbol(self.snapshot, symbol)
            module = self.snapshot.parsed_module_for_source_path(owner.file_path)
            for method in owner.node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                digest = FieldOrderDigest.in_method(self.snapshot, module, method)
                if digest is not None:
                    raise ValueError(
                        f"{owner.file_path}:{digest} digests `{owner.qualname}."
                        f"{method.name}`'s serialization of the instance, whose "
                        "field order the composition changes"
                    )

    def decorator_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        """`kw_only=True` on the composer's dataclass decorator."""

        composer = self.composer
        if composer.option("kw_only") == "True":
            return ()
        source = self.snapshot.sources_by_file_path[composer.file_path]
        geometry = SourceTextGeometry(source)
        (decorator,) = composer.node.decorator_list
        start, end = geometry.required_node_offsets(decorator)
        replacement = keyword_only_decorator_source(source[start:end], decorator)
        return geometry.physical_edits(
            file_path=composer.file_path,
            replacements=(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start, end_offset=end, replacement_source=replacement
                ),
            ),
            rationale=rationale,
        )


class FieldOrderDigest:
    """A method that hashes or dumps a serialization of its whole instance."""

    SERIALIZERS = frozenset(
        (
            "builtins.repr",
            "builtins.str",
            "dataclasses.astuple",
            "dataclasses.asdict",
            "dataclasses.fields",
            "pickle.dumps",
        )
    )

    @classmethod
    def in_method(
        cls,
        snapshot: CodemodSourceSnapshot,
        module: ParsedModule,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int | None:
        """The line of the method's digest of `self`'s serialization, if any."""

        receiver = (
            method.args.args[0].arg
            if method.args.args
            else None
        )
        if receiver is None:
            return None
        proof = snapshot.module_binding_proof
        serializes = False
        sink: int | None = None
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            witness = proof.reference_or_builtin_witness_at(
                module, node.func, line=node.lineno
            )
            if witness is None:
                continue
            name = witness.qualified_name
            if (
                name in cls.SERIALIZERS
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == receiver
            ):
                serializes = True
            if name.startswith("hashlib.") or (
                name == "json.dumps"
                and not any(
                    keyword.arg == "sort_keys"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
            ):
                sink = node.lineno if sink is None else sink
        return sink if serializes else None


def _line_difference_edits(
    file_path: str, before: str, after: str, rationale: str
) -> tuple[NominalSourceEdit, ...]:
    """The line spans that differ between two versions of one source."""

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    offsets = [0]
    for line in before_lines:
        offsets.append(offsets[-1] + len(line))
    geometry = SourceTextGeometry(before)
    return geometry.physical_edits(
        file_path=file_path,
        replacements=tuple(
            SourceTextSpanReplacement.from_offsets(
                start_offset=offsets[first],
                end_offset=offsets[last],
                replacement_source="".join(after_lines[new_first:new_last]),
            )
            for tag, first, last, new_first, new_last in difflib.SequenceMatcher(
                a=before_lines, b=after_lines, autojunk=False
            ).get_opcodes()
            if tag != "equal"
        ),
        rationale=rationale,
    )


def keyword_only_decorator_source(decorator_source: str, decorator: ast.expr) -> str:
    """A dataclass decorator's source with `kw_only=True` added."""

    if not isinstance(decorator, ast.Call):
        return f"{decorator_source}(kw_only=True)"
    if not decorator_source.endswith(")"):
        raise ValueError(f"`{decorator_source}` is an unrewritable decorator")
    opened = decorator_source[:-1].rstrip()
    separator = "" if opened.endswith("(") else ", "
    return f"{opened}{separator}kw_only=True)"


def dataclass_options_besides_kw_only(
    overlap_class: OverlapClass,
) -> tuple[tuple[str, ...], ...]:
    """The dataclass options, keyword-only-ness aside: what composition must keep."""

    return tuple(
        tuple(
            option for option in options if option.partition("=")[0].strip() != "kw_only"
        )
        for options in overlap_class.dataclass_options_key
    )


def composed_field_order(
    snapshot: CodemodSourceSnapshot, owner_fields: Iterable[str], composer: OverlapClass
) -> tuple[list[str], list[str]]:
    """The composer's own new field names before and after composing an owner
    declaring `owner_fields`: inherited names keep their first position."""

    class_index = snapshot.class_overlap_verdicts.class_index
    inherited = frozenset(
        field_name
        for symbol in class_index.ancestor_symbols(composer.symbol)
        if class_index.class_for(symbol) is not None
        for field_name in OverlapClass.for_symbol(snapshot, symbol).own_field_names
    )
    owner_fields = tuple(owner_fields)
    moved = frozenset(owner_fields)
    before = [name for name in composer.own_field_names if name not in inherited]
    after = [
        name
        for name in dict.fromkeys(
            (
                *owner_fields,
                *(name for name in composer.own_field_names if name not in moved),
            )
        )
        if name not in inherited
    ]
    return before, after


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


@dataclass(frozen=True)
class PositionalConstruction:
    """One call that builds a dataclass, or its receiver's class, by position."""

    file_path: str
    call: ast.Call
    parameter_names: tuple[str, ...]

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.call.lineno}"

    def require_keywordable(self) -> None:
        call = self.call
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            raise ValueError(f"{self.location} unpacks `*` arguments")
        if CallKeywords.of(call.keywords).splats:
            raise ValueError(f"{self.location} unpacks `**` arguments")
        if len(call.args) > len(self.parameter_names):
            raise ValueError(
                f"{self.location} passes {len(call.args)} positional argument(s); "
                f"the generated `__init__` takes {len(self.parameter_names)}"
            )
        if any(isinstance(argument, ast.GeneratorExp) for argument in call.args):
            raise ValueError(f"{self.location} passes a bare generator expression")

    def replacements(
        self, geometry: SourceTextGeometry
    ) -> tuple[SourceTextSpanReplacement, ...]:
        """`name=` before each positional argument's first token; a walrus the
        call left bare is parenthesized."""

        call = self.call
        func_end = geometry.required_node_offsets(call.func)[1]
        starts: list[int] = []
        depth = 0
        expecting = True
        for token in geometry.tokens:
            offset = geometry.line_offsets[token.start[0] - 1] + token.start[1]
            if offset < func_end or token.type in _LAYOUT_TOKENS:
                continue
            if depth == 1 and expecting and token.string != ")":
                starts.append(offset)
                expecting = False
            if token.type == tokenize.OP and token.string in "([{":
                depth += 1
            elif token.type == tokenize.OP and token.string in ")]}":
                depth -= 1
                if depth == 0:
                    break
            elif token.type == tokenize.OP and token.string == "," and depth == 1:
                expecting = True
        replacements: list[SourceTextSpanReplacement] = []
        for name, argument, start in zip(self.parameter_names, call.args, starts):
            argument_start, argument_end = geometry.required_node_offsets(argument)
            bare_walrus = isinstance(argument, ast.NamedExpr) and argument_start == start
            replacements.append(
                SourceTextSpanReplacement.from_offsets(
                    start_offset=start,
                    end_offset=start,
                    replacement_source=f"{name}=" + ("(" if bare_walrus else ""),
                )
            )
            if bare_walrus:
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=argument_end,
                        end_offset=argument_end,
                        replacement_source=")",
                    )
                )
        if len(starts) < len(call.args):
            raise ValueError(f"{self.location}: the call's arguments do not tokenize")
        return tuple(replacements)


_LAYOUT_TOKENS = frozenset(
    (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
)


@dataclass(frozen=True)
class ConstructionKeywording:
    """Every positional construction of one dataclass, bound by keyword instead.

    A direct call `C(x, y)` binds the generated `__init__`'s positional
    parameters in order, so `C(a=x, b=y)` binds the same values. A receiver's
    own class (`cls(...)` in a classmethod, `type(self)(...)` or
    `self.__class__(...)` in a method) may be any descendant, so the rewrite
    also needs every descendant's positional parameters to start with the same
    names. The class and every such descendant need a closed dataclass schema
    whose fields' keyword-only-ness is known; any construction that unpacks
    arguments, passes more than the parameters, or passes a bare generator
    refuses the whole rewrite. A read that hands the class on as a value or a
    positional class pattern is left alone: it binds through `__match_args__`
    or an unknown caller, not this call.
    """

    snapshot: CodemodSourceSnapshot
    target: OverlapClass

    @classmethod
    def require(
        cls, snapshot: CodemodSourceSnapshot, reference: SourceRewriteTarget
    ) -> ConstructionKeywording:
        keywording = cls(snapshot, OverlapClass.require(snapshot, reference))
        if not keywording.constructions:
            raise ValueError(f"`{keywording.target.qualname}` is never built by position")
        for construction in keywording.constructions:
            construction.require_keywordable()
        return keywording

    def parameter_names(self, symbol: str) -> tuple[str, ...]:
        resolution = (
            self.snapshot.class_overlap_verdicts.class_index.product_authority_resolutions_by_symbol.get(
                symbol
            )
        )
        authority = None if resolution is None else resolution.authority
        if authority is None:
            raise ValueError(f"`{symbol}` has no closed dataclass schema")
        return authority.positional_parameter_names()

    @cached_property
    def constructions(self) -> tuple[PositionalConstruction, ...]:
        parameters = self.parameter_names(self.target.symbol)
        direct = tuple(
            PositionalConstruction(site.file_path, site.parent, parameters)
            for site in ClassReferenceSites(self.snapshot, frozenset((self.target.symbol,))).sites
            if site.role is ClassReferenceRole.CONSTRUCTION
            and isinstance(site.parent, ast.Call)
            and site.parent.args
        )
        return (*direct, *self.receiver_constructions(parameters))

    def receiver_constructions(
        self, parameters: tuple[str, ...]
    ) -> tuple[PositionalConstruction, ...]:
        module_bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.snapshot.parsed_module_for_source_path(self.target.file_path).module.body
        )
        calls = tuple(
            call
            for function in self.target.node.body
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            for call in _receiver_class_calls(function, module_bound)
        )
        if not calls:
            return ()
        widest = max(len(call.args) for call in calls)
        for symbol in self.snapshot.class_overlap_verdicts.class_index.descendant_symbols(
            self.target.symbol
        ):
            if self.parameter_names(symbol)[:widest] != parameters[:widest]:
                raise ValueError(
                    f"`{symbol}` takes other positional parameters than "
                    f"`{self.target.qualname}`, and its receiver may be one"
                )
        return tuple(
            PositionalConstruction(self.target.file_path, call, parameters) for call in calls
        )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        by_file: dict[str, list[PositionalConstruction]] = {}
        for construction in self.constructions:
            by_file.setdefault(construction.file_path, []).append(construction)
        return tuple(
            edit
            for file_path, constructions in by_file.items()
            for geometry in (SourceTextGeometry(self.snapshot.sources_by_file_path[file_path]),)
            for edit in geometry.physical_edits(
                file_path=file_path,
                replacements=tuple(
                    replacement
                    for construction in constructions
                    for replacement in construction.replacements(geometry)
                ),
                rationale=rationale,
            )
        )


def _receiver_class_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef, module_bound: frozenset[str]
) -> tuple[ast.Call, ...]:
    """Positional calls of the method's receiver's own class: `cls(...)` in a
    classmethod; `type(self)(...)` or `self.__class__(...)` in an undecorated
    method. Raises when the receiver or `type` is rebound inside the method."""

    positional = (*function.args.posonlyargs, *function.args.args)
    if not positional:
        return ()
    receiver = positional[0].arg
    decorators = function.decorator_list
    is_classmethod = (
        len(decorators) == 1
        and isinstance(decorators[0], ast.Name)
        and decorators[0].id == "classmethod"
        and "classmethod" not in module_bound
    )
    if decorators and not is_classmethod:
        return ()

    def builds_receiver_class(node: ast.expr) -> bool:
        if is_classmethod:
            return isinstance(node, ast.Name) and node.id == receiver
        if isinstance(node, ast.Attribute):
            return (
                node.attr == "__class__"
                and isinstance(node.value, ast.Name)
                and node.value.id == receiver
            )
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and "type" not in module_bound
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == receiver
        )

    calls = tuple(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and node.args and builds_receiver_class(node.func)
    )
    if calls:
        rebound = {receiver, "type"} & frozenset(
            name
            for node in ast.walk(function)
            if node is not function
            for name in (
                (node.id,)
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
                else (node.arg,)
                if isinstance(node, ast.arg) and node is not positional[0]
                else (node.name,)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                else tuple(node.names)
                if isinstance(node, (ast.Global, ast.Nonlocal))
                else ()
            )
        )
        if rebound:
            raise ValueError(
                f"line {function.lineno}: `{function.name}` rebinds {sorted(rebound)}, "
                "so its own-class calls are unproved"
            )
    return calls


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
class ComposeExistingClassesOperation(RepositorySourceReprovedOperation):
    """Make every class restating a block compose the existing classes that own it."""

    owners: tuple[SourceRewriteTarget, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteTarget)
    )
    composers: tuple[SourceRewriteTarget, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteTarget), default=()
    )

    def composition(self, snapshot: CodemodSourceSnapshot) -> CoveringComposition:
        return CoveringComposition.require(
            snapshot, self.owners, (self.target, *self.composers)
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        composition = self.composition(snapshot)
        return composition.source_edits(
            self.rationale_text(
                "Compose "
                + " + ".join(f"`{owner.qualname}`" for owner in composition.owners)
                + " instead of restating their declarations."
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        composition = self.composition(context.execution_snapshot())
        return tuple(
            AstTargetAuthorityClaim.from_target(
                owner.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            )
            for owner in composition.owners
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


def require_owner_candidate(
    snapshot: CodemodSourceSnapshot,
    anchor_symbol: str,
    candidate_id: str,
    member_names: Iterable[str],
) -> tuple[ClassOverlapCandidate, OwnerCandidate]:
    """An extract-mixin verdict over the anchor whose block holds these names and
    that offers this owner, with no guard fired on the verdict or the owner."""

    wanted = frozenset(member_names)
    for verdict in snapshot.class_overlap_verdicts.candidates:
        if verdict.move is not ExtractMixin or not wanted <= frozenset(verdict.shared_names):
            continue
        for owner in verdict.owner_candidates:
            if owner.candidate_id == candidate_id and anchor_symbol in owner.composers:
                if not verdict.permitted:
                    raise ValueError(
                        "The verdict refuses the move: "
                        + "; ".join(receipt.detail for receipt in verdict.refusals)
                    )
                if not owner.permitted:
                    raise ValueError(
                        f"`{candidate_id}` is refused: "
                        + "; ".join(receipt.detail for receipt in owner.refusals)
                    )
                return verdict, owner
    raise ValueError(
        f"The current source offers no extract-mixin owner `{candidate_id}` for "
        f"`{anchor_symbol}` over {sorted(wanted)}"
    )


def movable_members(
    snapshot: CodemodSourceSnapshot,
    source: OverlapClass,
    destination: OverlapClass,
    statements: tuple[ast.stmt, ...],
) -> None:
    """Each statement is a member that moves without reading its class's scope."""

    context = ClassMemberMoveProofContext(
        source_class=source.target,
        destination_class=destination.target,
        source=snapshot.sources_by_file_path[source.file_path],
        module_bound_names=LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            snapshot.parsed_module_for_source_path(source.file_path).module.body
        ),
    )
    for statement in statements:
        member = ClassMemberPromotionStatement.for_statement(statement)
        if member is None:
            raise ValueError(
                f"`{source.qualname}` line {statement.lineno} is no movable member"
            )
        member.require_safe_move(context)


@dataclass(frozen=True)
class MixinCreation:
    """A new class declares a block several classes restate; nothing uses it yet.

    The class is a root (`[mixin, object]`) with the anchor's own dataclass
    declaration, inserted right before the anchor in the anchor's module, so
    every free name of the copied statements is bound there exactly as in the
    anchor. Each statement must move without reading the anchor's class scope
    or learning its owner. Creating it changes no behaviour; each restating
    class then composes it through its own compose-existing verdict.
    """

    snapshot: CodemodSourceSnapshot
    anchor: OverlapClass
    restated: RestatedDeclarations
    name: str
    docstring: str
    keyword_only: bool
    """Created keyword-only because composing it reorders some composer's fields;
    each such composer then becomes keyword-only as it composes the mixin."""

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        anchor_reference: SourceRewriteTarget,
        member_names: tuple[str, ...],
        name: str,
        docstring: str,
    ) -> Self:
        anchor = OverlapClass.require(snapshot, anchor_reference)
        if not docstring.strip() or '"""' in docstring or "\\" in docstring:
            raise ValueError("A mixin needs a one-line docstring without quotes")
        module_name = snapshot.parsed_module_for_source_path(anchor.file_path).module_name
        _, owner = require_owner_candidate(
            snapshot, anchor.symbol, f"new_mixin:{module_name}", member_names
        )
        if not name.isidentifier() or keyword_module.iskeyword(name):
            raise ValueError(f"`{name}` is no class name")
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            snapshot.parsed_module_for_source_path(anchor.file_path).module.body
        )
        if name in bound:
            raise ValueError(f"`{name}` is already bound in {anchor.file_path}")
        anchor.dataclass_options_key
        restated = RestatedDeclarations.require(snapshot, anchor, anchor, member_names)
        movable_members(snapshot, anchor, anchor, restated.statements)
        restated.require_no_owner_bound_values(snapshot)
        mixin_fields = tuple(
            name for name in anchor.own_field_names if name in restated.names
        )
        keyword_only = anchor.option("kw_only") != "True" and any(
            before != after
            for symbol in owner.composers
            for before, after in (
                composed_field_order(
                    snapshot, mixin_fields, OverlapClass.for_symbol(snapshot, symbol)
                ),
            )
        )
        return cls(snapshot, anchor, restated, name, docstring, keyword_only)

    def decorator_source(self, source: str, decorator: ast.expr) -> str:
        copied = ast.get_source_segment(source, decorator) or ""
        if not self.keyword_only:
            return copied
        return keyword_only_decorator_source(copied, decorator)

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        anchor = self.anchor
        source = self.snapshot.sources_by_file_path[anchor.file_path]
        lines = source.splitlines(keepends=True)
        first_line = min(
            (decorator.lineno for decorator in anchor.node.decorator_list),
            default=anchor.node.lineno,
        )
        header = lines[first_line - 1]
        indentation = header[: len(header) - len(header.lstrip())]
        member_indentation = indentation + "    "
        decorators = tuple(
            f"{indentation}@{self.decorator_source(source, decorator)}\n"
            for decorator in anchor.node.decorator_list
        )
        members: list[str] = []
        for statement in self.restated.statements:
            member = StatementSource(source=source, node=statement).member_source(
                member_indentation
            )
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and members:
                members.append("\n")
            members.append(member if member.endswith("\n") else member + "\n")
        return (
            SourceInsertion(
                file_path=anchor.file_path,
                insertion_line=first_line,
                inserted_lines=(
                    *decorators,
                    f"{indentation}class {self.name}:\n",
                    f'{member_indentation}"""{self.docstring}"""\n\n',
                    *members,
                    "\n\n",
                ),
                rationale=rationale,
            ),
        )


@dataclass(frozen=True)
class OwnerExtension:
    """An existing class gains the block declarations it lacks.

    The owner already declares the rest of the block identically and is
    composed by other classes. The missing declarations are copied from the
    anchor, which restates the whole block. That changes the owner's
    descendants, so it is proved, not scored: the owner and the anchor are
    created the same way; for a dataclass owner, no added field is required
    while anything constructs the owner or a descendant, every read of them
    builds by keyword, tests, annotates or subclasses, and no `cls(...)` in
    their family passes positions; and for every descendant, the first class
    in its MRO that declares an added name is unchanged.
    """

    snapshot: CodemodSourceSnapshot
    owner: OverlapClass
    anchor: OverlapClass
    restated: RestatedDeclarations

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        owner_reference: SourceRewriteTarget,
        anchor_reference: SourceRewriteTarget,
        member_names: tuple[str, ...],
    ) -> Self:
        owner = OverlapClass.require(snapshot, owner_reference)
        anchor = OverlapClass.require(snapshot, anchor_reference)
        require_owner_candidate(
            snapshot, anchor.symbol, f"existing_class:{owner.symbol}", member_names
        )
        if frozenset(member_names) & owner.target.bound_names:
            raise ValueError(f"`{owner.qualname}` already binds an added name")
        dunders = sorted(name for name in member_names if name.startswith("__"))
        if dunders:
            raise ValueError(f"Adding {dunders} could shadow a native declaration")
        if owner.dataclass_options_key != anchor.dataclass_options_key:
            raise ValueError(
                f"`{owner.qualname}` and `{anchor.qualname}` differ as dataclasses"
            )
        restated = RestatedDeclarations.require(snapshot, anchor, anchor, member_names)
        movable_members(snapshot, anchor, owner, restated.statements)
        restated.require_no_owner_bound_values(snapshot)
        extension = cls(snapshot, owner, anchor, restated)
        extension.require_construction()
        extension.require_lookup()
        return extension

    @cached_property
    def family(self) -> tuple[str, ...]:
        return (
            self.owner.symbol,
            *self.snapshot.class_overlap_verdicts.class_index.descendant_symbols(
                self.owner.symbol
            ),
        )

    def require_construction(self) -> None:
        self.owner.require_construction_extended_by(self)

    def require_dataclass_construction(self) -> None:
        sites = ClassReferenceSites(self.snapshot, frozenset(self.family))
        added_required = tuple(
            ast.unparse(statement.target)
            for statement in self.restated.statements
            if isinstance(statement, ast.AnnAssign) and statement.value is None
        )
        if added_required:
            built = tuple(site for site in sites.sites if site.role.binds_fields)
            if built:
                raise ValueError(
                    f"{built[0].location} builds `{ast.unparse(built[0].node)}`, which "
                    f"would need the new required field(s) {list(added_required)}"
                )
        sites.require_keyword_construction()
        require_no_positional_self_construction(self.snapshot, self.family)

    def require_lookup(self) -> None:
        class_index = self.snapshot.class_overlap_verdicts.class_index
        authority = ClassMroAuthority(class_index.classes_by_symbol)
        for symbol in self.family:
            mro_type = authority.resolve(symbol).mro_type
            if mro_type is None:
                raise ValueError(f"`{symbol}` has an open MRO")
            order = tuple(declaration.symbol for declaration in mro_type.declarations)
            for name in self.restated.names:
                declaring = tuple(
                    declaration.symbol
                    for declaration in mro_type.declarations
                    if name in declaration.direct_declared_member_names
                    or any(item.name == name for item in declaration.body_declarations)
                )
                if declaring and order.index(declaring[0]) > order.index(
                    self.owner.symbol
                ):
                    raise ValueError(
                        f"`{self.owner.qualname}.{name}` would replace `{declaring[0]}`'s "
                        f"answer for `{symbol}`"
                    )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        owner = self.owner
        source = self.snapshot.sources_by_file_path[owner.file_path]
        anchor_source = self.snapshot.sources_by_file_path[self.anchor.file_path]
        body = ClassBodySourceAuthority(owner.node, source)
        return (
            *body.geometry.physical_edits(
                file_path=owner.file_path,
                replacements=(
                    body.declaration_block_replacement(
                        tuple(
                            (
                                StatementSource(
                                    source=anchor_source, node=statement
                                ).member_source(body.indentation),
                                isinstance(
                                    statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                                ),
                            )
                            for statement in self.restated.statements
                        )
                    ),
                ),
                rationale=rationale,
            ),
            *required_name_imports(
                self.snapshot, self.restated.statements, self.anchor, owner, rationale
            ),
        )


@dataclass(frozen=True, kw_only=True)
class CreateMixinOperation(RepositorySourceReprovedOperation):
    """Declare a block several classes restate once, in a new class before the
    anchor, and make each composer compose it, as one proved step.

    The mixin is created on the current source, then each composer composes it
    on the source the previous step left, each through the compose-existing
    proof. Any refusal refuses the whole operation, so a mixin is never created
    without every composer that restates its block.
    """

    member_names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())
    mixin_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    composers: tuple[SourceRewriteTarget, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteTarget), default=()
    )

    def creation(self, snapshot: CodemodSourceSnapshot) -> MixinCreation:
        return MixinCreation.require(
            snapshot, self.target, self.member_names, self.mixin_name, self.docstring
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        rationale = self.rationale_text(
            f"Declare {', '.join(self.member_names)} once on `{self.mixin_name}`."
        )
        creation = self.creation(snapshot)
        if not self.composers:
            return creation.source_edits(rationale)
        mixin = SourceRewriteTarget(
            file_path=self.target.file_path,
            qualname=".".join(
                (*self.target.qualname.split(".")[:-1], self.mixin_name)
            ),
        )
        current = snapshot
        for step in (
            replace(self, composers=()),
            *(
                ComposeExistingClassOperation(target=composer, owner=mixin)
                for composer in self.composers
            ),
        ):
            result = RefactorRecipe(recipe_id=f"create-{self.mixin_name}").with_operation(
                step
            ).simulate(current)
            result.preflight_report.require_clean()
            current = current.with_virtual_sources(result.simulation.rewritten_sources)
        return tuple(
            edit
            for file_path, source in snapshot.sources_by_file_path.items()
            if current.sources_by_file_path.get(file_path, source) != source
            for edit in _line_difference_edits(
                file_path, source, current.sources_by_file_path[file_path], rationale
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        creation = self.creation(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                creation.anchor.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ExtendOwnerOperation(RepositorySourceReprovedOperation):
    """Give an existing owner the block declarations it lacks, copied from the anchor."""

    anchor: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    member_names: tuple[str, ...] = codemod_payload_field(StringArrayPayloadValueCodec())

    def extension(self, snapshot: CodemodSourceSnapshot) -> OwnerExtension:
        return OwnerExtension.require(snapshot, self.target, self.anchor, self.member_names)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        extension = self.extension(snapshot)
        return extension.source_edits(
            self.rationale_text(
                f"Declare {', '.join(self.member_names)} on `{extension.owner.qualname}`, "
                "the owner of the rest of the block."
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        extension = self.extension(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                extension.owner.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class KeywordizeConstructionOperation(RepositorySourceReprovedOperation):
    """Bind every positional construction of a dataclass by keyword (#86)."""

    def keywording(self, snapshot: CodemodSourceSnapshot) -> ConstructionKeywording:
        return ConstructionKeywording.require(snapshot, self.target)

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        keywording = self.keywording(snapshot)
        return keywording.source_edits(
            self.rationale_text(
                f"Build `{keywording.target.qualname}` by keyword, so its field "
                "order no longer decides what each call binds."
            )
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        keywording = self.keywording(context.execution_snapshot())
        return (
            AstTargetAuthorityClaim.from_target(
                keywording.target.target.target,
                authority_kind=SemanticAuthorityKind.CLASS_FAMILY,
            ),
        )


class ClassOverlapFindingRecipeSynthesizer(
    FindingEvidenceActionKeysMixin,
    SelfDeclaredRecipeSynthesizer,
    ClassBodyCompositionConcept,
):
    """The verdict's move as a recipe, when the move keeps every class identity.

    Compose-existing becomes one composition per restating class; promote to
    a shared base becomes one fold. Extracting a block mints or widens an
    owner, so it runs only on a picker's choice (#67): see
    `picked_operations`. Deleting a class changes identities and stays a
    finding.
    """

    @staticmethod
    def picked_operations(
        finding: RefactorFinding,
        metrics: ClassOverlapMetrics,
        verdicts: ClassOverlapVerdicts,
    ) -> tuple[RepositorySourceReprovedOperation, ...] | str:
        """The operation the picker's owner calls for, or why there is none.

        A new mixin is declared before a composing class in its module; an
        existing class first gains the block declarations it lacks; a shared
        base takes the block as a fold. Each restating class then composes the
        owner through its own compose-existing verdict at the next stage. A
        verdict covered by a pick on another finding adds nothing itself.
        """

        pick = metrics.owner_pick
        if pick is None:
            return "an extract-mixin move needs an owner pick (--codemod-owner-picks)"
        if pick.finding != finding.stable_id:
            return f"covered by the owner pick on {pick.finding}"
        owner = next(
            (candidate for candidate in metrics.owner_candidates if candidate.candidate_id == pick.owner),
            None,
        )
        if owner is None or not owner.permitted:
            return f"`{pick.owner}` is not a permitted candidate of this finding"
        composers = tuple(
            indexed_class
            for symbol in owner.composers
            if (indexed_class := verdicts.class_index.class_for(symbol)) is not None
        )
        match owner:
            case NewMixinOwner():
                anchor = next(
                    (
                        indexed_class
                        for indexed_class in composers
                        if indexed_class.module_name == owner.module_name
                    ),
                    None,
                )
                if anchor is None or pick.name is None:
                    return f"no composing class lives in `{owner.module_name}`"
                if pick.judgement is None:
                    return "a new mixin needs the pick's judgement for its docstring"
                return (
                    CreateMixinOperation(
                        target=source_rewrite_target(anchor),
                        member_names=owner.block,
                        mixin_name=pick.name,
                        docstring=pick.judgement[:1].upper() + pick.judgement[1:].rstrip(".") + ".",
                        composers=tuple(
                            source_rewrite_target(indexed_class)
                            for indexed_class in composers
                        ),
                    ),
                )
            case ExistingClassOwner():
                existing = verdicts.class_index.class_for(owner.symbol)
                missing = tuple(name for name in owner.block if name not in owner.held)
                if existing is None or not composers:
                    return "the picked owner left the source"
                if not missing:
                    return f"`{owner.symbol}` already declares the whole block"
                return (
                    ExtendOwnerOperation(
                        target=source_rewrite_target(existing),
                        anchor=source_rewrite_target(composers[0]),
                        member_names=missing,
                    ),
                )
            case ExistingClassesOwner():
                owners = tuple(
                    indexed_class
                    for symbol in owner.symbols
                    if (indexed_class := verdicts.class_index.class_for(symbol)) is not None
                )
                if len(owners) != len(owner.symbols) or not composers:
                    return "the picked owners left the source"
                return (
                    ComposeExistingClassesOperation(
                        target=source_rewrite_target(composers[0]),
                        owners=tuple(source_rewrite_target(item) for item in owners),
                        composers=tuple(
                            source_rewrite_target(item) for item in composers[1:]
                        ),
                    ),
                )
            case SharedBaseOwner():
                base = verdicts.class_index.class_for(owner.symbol)
                if base is None:
                    return "the picked base left the source"
                return (
                    FoldSharedDeclarationsIntoBaseOperation(
                        target=source_rewrite_target(base),
                        member_names=owner.block,
                    ),
                )
        return f"`{owner.kind}` owners have no move"

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
        elif metrics.move is ExtractMixin:
            picked = self.picked_operations(finding, metrics, verdicts)
            if isinstance(picked, str):
                return self.rejected_evaluation(picked)
            operations = picked
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
