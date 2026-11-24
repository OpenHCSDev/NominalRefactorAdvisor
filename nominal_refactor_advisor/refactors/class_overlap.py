from __future__ import annotations

import ast
import builtins
import difflib
import inspect
import keyword as keyword_module
import tokenize
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
    replace,
)
from functools import cached_property
from itertools import chain
from importlib import import_module
from pathlib import Path
from typing import ClassVar, Self, cast

from nominal_refactor_advisor.declared_c3 import BASES_READ_AT_CREATION
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.refactors.class_attribute_reads import BasesReaders
from nominal_refactor_advisor.refactors.class_creation import (
    ClassCreationProof,
    MroWalk,
    declared_names,
)
from nominal_refactor_advisor.refactors.class_member_authority import (
    ClassMemberMoveProofContext,
    ClassMemberPromotionStatement,
    ModuleDeclarationRelocation,
)
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
)
from nominal_refactor_advisor.ast_projection import CallKeywords
from nominal_refactor_advisor.ast_tools import (
    ParsedModule,
    is_docstring_statement,
)
from nominal_refactor_advisor.class_body import ProvedClassBodyDeclaration
from nominal_refactor_advisor.class_index import (
    PER_CLASS_NAMES,
    CompactIndexedClass,
    module_import_aliases,
    native_identity,
)
from nominal_refactor_advisor.class_mro import (
    NativeMroBase,
    namespace_read_native,
    native_declared_names,
)
from nominal_refactor_advisor.field_order_reads import DeclaringClass
from nominal_refactor_advisor.class_overlap import (
    AnswersFor,
    ClassOverlapCandidate,
    ClassOverlapMetrics,
    ClassOverlapMove,
    ClassOverlapVerdicts,
    ComposeExisting,
    ExtractMixin,
    PromoteToSharedBase,
    SubclassAnswers,
)
from nominal_refactor_advisor.class_references import (
    ClassReferenceRole,
    reads_as_type,
    subclass_test_subject,
)
from nominal_refactor_advisor.class_symbols import _referenced_bases
from nominal_refactor_advisor.overlap_receipts import OverlapReceiptKind
from nominal_refactor_advisor.codemod import RefactorRecipe
from nominal_refactor_advisor.codemod_authority_claims import AstTargetAuthorityClaim
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassBodySourceAuthority,
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    RequiredStringPayloadValueCodec,
    StringArrayPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError
from nominal_refactor_advisor.codemod_reproof import RepositorySourceReprovedOperation
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
)
from nominal_refactor_advisor.codemod_selection_context import (
    CodemodSelectorContext,
    ResolvedClassTarget,
)
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceInsertion,
    SourceTextGeometry,
    SourceTextSpanReplacement,
    TextFileChange,
)
from nominal_refactor_advisor.codemod_statement_source import (
    StatementDeletionSource,
    StatementSource,
)
from nominal_refactor_advisor.lexical_bindings import (
    ImportBoundNameProjection,
    LEXICAL_SCOPE_BINDING_AUTHORITY,
)
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.module_node_index import ScopedImportBindings
from nominal_refactor_advisor.native_declarations import DataclassRuntimeDeclaration
from nominal_refactor_advisor.read_sources import EmbeddedSources, ModuleSource, ReadSource
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.owner_candidates import (
    ComposeExistingOwner,
    ExistingClassOwner,
    HoldSourceOwner,
    HoistingMixinOwner,
    ModuleImportGraph,
    NewMixinOwner,
    NewModuleMixinOwner,
    OwnerCandidate,
    PickedOwner,
    SharedBaseOwner,
    common_package,
)
from nominal_refactor_advisor.audit.family import snake_case
from nominal_refactor_advisor.refactor_concepts import ClassBodyCompositionConcept
from nominal_refactor_advisor.refactors.carrier_fields import (
    CarrierFieldProjection,
    ReplaceFieldsWithCarrierOperation,
)
from nominal_refactor_advisor.refactors.module_move import (
    ImportedNameOwner,
    RelocateSymbolsToModuleOperation,
    RelocateSymbolsToNewModuleOperation,
)
from nominal_refactor_advisor.ast_tools import ModuleAnnotationEvaluationMode
from nominal_refactor_advisor.codemod_module_declarations import SourceTopLevelSymbolMoveSelection
from nominal_refactor_advisor.declaration_dependencies import DeclarationDependencyProjection
from nominal_refactor_advisor.semantic_descent import (
    AuthorityClaim,
    SemanticAuthorityKind,
)


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

    def attribute_docstring(self, statement: ast.stmt) -> ast.Expr | None:
        """The string statement right after a field, which documents it."""

        if not isinstance(statement, (ast.AnnAssign, ast.Assign)):
            return None
        body = self.node.body
        index = next(position for position, member in enumerate(body) if member is statement)
        following = body[index + 1] if index + 1 < len(body) else None
        return following if following is not None and is_docstring_statement(following) else None

    def documentation(self, name: str) -> str | None:
        """What the class documents a member with: a definition's own docstring,
        or the string statement after a field."""

        for statement in self.node.body:
            if name not in declared_names(statement):
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return ast.get_docstring(statement, clean=False)
            documented = self.attribute_docstring(statement)
            return None if documented is None else cast(ast.Constant, documented.value).value
        return None

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
                raise NeedsInput(
                    f"`{self.qualname}` binds {sorted(bound - wanted)} in the same "
                    "statement as a moved declaration", NeededInput.SOURCE_CHANGE
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
            raise NeedsInput(
                f"`{self.qualname}` has {unplaced} with no known field position", NeededInput.PROOF
            )
        return names

    def _require_single_decorator(self) -> None:
        decorators = self.node.decorator_list
        if len(decorators) > 1:
            raise NeedsInput(f"`{self.qualname}` has {len(decorators)} decorators", NeededInput.PROOF)

    @property
    def lays_out_instance_dict(self) -> bool:
        """Its instances get a `__dict__` from it: it neither declares
        `__slots__` nor is a slotted dataclass."""

        return self.option("slots") != "True" and not any(
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__slots__" for target in statement.targets)
            for statement in self.node.body
        )

    def require_dataclass_base(self, base: OverlapClass, moved: Iterable[str]) -> None:
        """Composing `base` for the `moved` declarations leaves this class's
        generated methods, its frozenness and its instance layout as they were.

        A dataclass collects fields and generated methods only from dataclass
        bases, so a plain base serves any class, dataclass or not, unless the
        move takes one of the class's own fields into it: that field would stop
        being one.
        """

        layout_changes = base.lays_out_instance_dict and not self.lays_out_instance_dict
        if not base.dataclass_options_key:
            taken = (
                sorted(frozenset(moved) & frozenset(self.own_field_names))
                if self.dataclass_options_key
                else ()
            )
            if taken:
                raise NeedsInput(
                    f"`{base.qualname}` is no dataclass, so moving {taken} into it ends "
                    f"them as fields of `{self.qualname}`",
                    NeededInput.SOURCE_CHANGE,
                )
            if layout_changes:
                raise NeedsInput(
                    f"`{base.qualname}` gives slotted `{self.qualname}` an instance `__dict__`",
                    NeededInput.SOURCE_CHANGE,
                )
            return
        if not self.dataclass_options_key or any(
            not role.base_serves(base.option(name), self.option(name))
            for name, role in DATACLASS_OPTION_ROLES.items()
        ) or layout_changes:
            raise NeedsInput(
                f"`{base.qualname}` and `{self.qualname}` differ as dataclasses",
                NeededInput.SOURCE_CHANGE,
            )

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
            raise NeedsInput(f"`{self.qualname}` has a non-dataclass decorator", NeededInput.PROOF)
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
            raise NeedsInput(
                f"`{self.qualname}` annotates names that would become fields", NeededInput.SOURCE_CHANGE
            )

    def option(self, name: str) -> str:
        self.dataclass_options_key
        return "False"

    def require_dataclass(self) -> None:
        self.dataclass_options_key
        raise NeedsInput(f"`{self.qualname}` is not a dataclass", NeededInput.SOURCE_CHANGE)


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
            raise NeedsInput(f"`{self.qualname}` is a dataclass of another kind", NeededInput.SOURCE_CHANGE)

    def option(self, name: str) -> str:
        self._require_single_decorator()
        for option in self.options:
            key, separator, value = option.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        return DATACLASS_OPTION_DEFAULTS.get(name, "False")

    def require_dataclass(self) -> None:
        self._require_single_decorator()


def source_rewrite_target(indexed_class: CompactIndexedClass) -> SourceRewriteTarget:
    return SourceRewriteTarget(
        file_path=indexed_class.file_path, qualname=indexed_class.qualname
    )


class DataclassOptionRole(ABC):
    """What one `dataclass` option means to a class that inherits a dataclass base.

    The subclass's own decorator decides its methods, except one it does not
    generate: that it inherits. So a base serves a composer only if every method
    the base generates the composer generates too, the two agree on what Python
    requires them to share, and the base adds no instance `__dict__` to a
    composer that has none.
    """

    @abstractmethod
    def base_serves(self, base_value: str, composer_value: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mixin_value(self, name: str, values: frozenset[str]) -> str:
        """The value a new mixin declares, given every composer's."""
        raise NotImplementedError


class GeneratedMethodOption(DataclassOptionRole):
    """`init`, `repr`, `eq`, ...: a method the decorator writes on its own class."""

    def base_serves(self, base_value: str, composer_value: str) -> bool:
        return base_value in ("False", composer_value)

    def mixin_value(self, name: str, values: frozenset[str]) -> str:
        (value,) = values if len(values) == 1 else ("False",)
        return value


class SharedOption(DataclassOptionRole):
    """`frozen`: Python refuses a frozen dataclass over a non-frozen dataclass base
    and the reverse."""

    def base_serves(self, base_value: str, composer_value: str) -> bool:
        return base_value == composer_value

    def mixin_value(self, name: str, values: frozenset[str]) -> str:
        if len(values) != 1:
            raise NeedsInput(
                f"the composers differ in `{name}` ({', '.join(sorted(values))}): "
                "Python lets no dataclass serve both as a base",
                NeededInput.SOURCE_CHANGE,
            )
        (value,) = values
        return value


class OwnLayoutOption(DataclassOptionRole):
    """`kw_only`, `slots`, `weakref_slot`: each class's own fields and layout.
    Field order has its own proof and the layout is compared as a whole; a new
    mixin lays out nothing and gets its keyword-only-ness from that proof."""

    def base_serves(self, base_value: str, composer_value: str) -> bool:
        return True

    def mixin_value(self, name: str, values: frozenset[str]) -> str:
        return DATACLASS_OPTION_DEFAULTS[name]


DATACLASS_OPTION_DEFAULTS: dict[str, str] = {
    name: repr(parameter.default)
    for name, parameter in inspect.signature(dataclass).parameters.items()
    if parameter.kind is parameter.KEYWORD_ONLY
}
"""Each `dataclass` option with the value its own signature gives an unwritten one."""

DATACLASS_OPTION_ROLES: dict[str, DataclassOptionRole] = {
    name: {
        "frozen": SharedOption(),
        "kw_only": OwnLayoutOption(),
        "slots": OwnLayoutOption(),
        "weakref_slot": OwnLayoutOption(),
    }.get(name, GeneratedMethodOption())
    for name in DATACLASS_OPTION_DEFAULTS
}


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
                raise NeedsInput(f"`{name}` is name-mangled by its owner", NeededInput.PROOF)
            mine = source.declaration(snapshot, name)
            theirs = restating.declaration(snapshot, name)
            if not mine.same_declaration_as(theirs):
                raise NeedsInput(
                    f"`{restating.qualname}.{name}` differs from "
                    f"`{source.qualname}.{name}`", NeededInput.SOURCE_CHANGE
                )
        return cls(source, restating, names)

    @cached_property
    def deleted_names(self) -> tuple[str, ...]:
        """The restated names the restating class stops declaring: all but the
        ones class creation reads from each class's own body."""

        return tuple(name for name in self.names if name not in PER_CLASS_NAMES)

    @cached_property
    def statements(self) -> tuple[ast.stmt, ...]:
        return self.restating.statements_declaring(self.deleted_names)

    @cached_property
    def written_statements(self) -> tuple[ast.stmt, ...]:
        """The statements as written: each field followed by the docstring after it,
        which moves and goes with the field."""

        return tuple(
            written
            for statement in self.statements
            for written in (statement, self.restating.attribute_docstring(statement))
            if written is not None
        )

    def member_sources(
        self, snapshot: CodemodSourceSnapshot, indentation: str, documentation: KeptDocumentation
    ) -> tuple[tuple[str, bool], ...]:
        """Each moved statement as the restating class writes it, into `indentation`,
        carrying the docstring `documentation` keeps and no other; each with whether
        it is a function, which a blank line separates from what comes before."""

        source = snapshot.sources_by_file_path[self.restating.file_path]
        members: list[tuple[str, bool]] = []
        for statement in self.statements:
            keeps = documentation.kept(min(declared_names(statement))) is not None
            written = StatementSource(source=source, node=statement).member_source(indentation)
            function = isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                members.append((written if keeps else self.undocumented(source, statement, written), function))
                continue
            members.append((written, function))
            literal = self.restating.attribute_docstring(statement)
            if keeps and literal is not None:
                members.append((StatementSource(source=source, node=literal).member_source(indentation), False))
        return tuple(members)

    def undocumented(
        self, source: str, statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, written: str
    ) -> str:
        """`written`, the rendering of `statement` from its first decorator, without
        its own docstring; `...` stands for a body that was only the docstring."""

        docstring, *rest = statement.body
        if not is_docstring_statement(docstring):
            return written
        lines = source.splitlines(keepends=True)
        before = lines[docstring.lineno - 1].encode()[: docstring.col_offset].decode()
        after = lines[cast(int, docstring.end_lineno) - 1].encode()[cast(int, docstring.end_col_offset) :].decode()
        if before.strip() or after.strip():
            raise NeedsInput(
                f"`{self.restating.qualname}.{statement.name}`'s docstring shares a line with "
                "other source: put it on lines of its own", NeededInput.SOURCE_CHANGE
            )
        first_line = min((decorator.lineno for decorator in statement.decorator_list), default=statement.lineno)
        rendered = written.splitlines(keepends=True)
        start, end = docstring.lineno - first_line, cast(int, docstring.end_lineno) - first_line + 1
        while rest and end < len(rendered) and not rendered[end].strip():
            end += 1
        indentation = rendered[start][: len(rendered[start]) - len(rendered[start].lstrip())]
        return "".join((*rendered[:start], *(() if rest else (f"{indentation}...\n",)), *rendered[end:]))

    def require_no_owner_bound_values(self, snapshot: CodemodSourceSnapshot) -> None:
        """A value built by a class with `__set_name__` learns the class it sits in.

        Moving it changes which class that hook receives, so a moved value that
        calls anything is admitted only when the callee is a class whose whole
        C3 order is indexed and none of those classes declares the hook. The one
        other callee admitted is `dataclasses.field`: the dataclass decorator
        consumes its `Field`, whose own `__set_name__` only forwards to its
        `default`, and a default that calls anything is a call this walk checks.
        """

        class_index = snapshot.class_overlap_verdicts.class_index
        resolver = snapshot.class_reference_resolver_for_source_path(
            self.restating.file_path
        ).reference_resolution
        mro_authority = class_index.mro_authority
        for statement in self.statements:
            value = getattr(statement, "value", None)
            for call in () if value is None else ast.walk(value):
                if not isinstance(call, ast.Call):
                    continue
                callee = (
                    call.func.value if isinstance(call.func, ast.Subscript) else call.func
                )
                if DataclassRuntimeDeclaration.FIELD.matches(
                    resolver.imported_name_for_node(callee)
                ):
                    continue
                symbol = resolver.symbol_for_node(callee)
                if symbol is None or class_index.class_for(symbol) is None:
                    raise NeedsInput(
                        f"`{ast.unparse(call.func)}` builds a moved value; "
                        "its callee is no indexed class", NeededInput.PROOF
                    )
                mro_type = mro_authority.resolve(symbol).mro_type
                if mro_type is None:
                    raise NeedsInput(f"`{symbol}` has an open MRO", NeededInput.PROOF)
                if any(
                    declaration.name == "__set_name__"
                    for owner in mro_type.declarations
                    for declaration in owner.body_declarations
                ):
                    raise NeedsInput(
                        f"`{symbol}` learns its owner class through `__set_name__`", NeededInput.PROOF
                    )


@dataclass(frozen=True)
class KeptDocumentation:
    """The docstring each member keeps when one copy replaces several classes' copies.

    Declarations are judged equal without their docstrings, so the copies a move
    replaces can be documented differently. An existing owner keeps its own copy,
    docstring and all. A copy written anew keeps a docstring only when every copy
    carries it, worded the same: one only some copies carry, or worded otherwise,
    describes one class's copy, and the shared declaration keeps only what every
    copy agrees on. Each docstring the kept copy does not carry is dropped, and the
    step's rationale names it, so whoever picked the owner sees what went.
    """

    copies: tuple[OverlapClass, ...]
    """Every class whose copy is replaced; for a copy written anew, the copy written is one."""
    names: tuple[str, ...]
    owner: OverlapClass | None = None
    """The existing owner whose own copy stays, or none when the copy is written anew."""

    @classmethod
    def composing(cls, restated: RestatedDeclarations) -> Self:
        """The restating class composes the existing class it restates, which keeps its copy."""

        return cls((restated.restating,), restated.deleted_names, restated.source)

    def kept(self, name: str) -> str | None:
        if self.owner is not None:
            return self.owner.documentation(name)
        text, *others = {copy.documentation(name) for copy in self.copies}
        return None if others else text

    @cached_property
    def dropped(self) -> tuple[str, ...]:
        return tuple(
            f"{copy.qualname}.{name}"
            for name in self.names
            for copy in self.copies
            if (text := copy.documentation(name)) is not None and text != self.kept(name)
        )

    def rationale(self, rationale: str) -> str:
        return " ".join(
            (
                rationale,
                *(f"Drops the docstring of `{qualname}`, which the kept copy does not carry." for qualname in self.dropped),
            )
        )


@dataclass(frozen=True)
class ClassReferenceSite:
    """One read of a class in the repository and the syntax around it."""

    source: ReadSource
    node: ast.expr
    parent: ast.AST | None
    role: ClassReferenceRole
    is_annotation: bool
    is_base: bool
    symbol: str
    """The class the read resolves to."""
    tested_subclass: str | None
    """For an `issubclass` test, the scanned class it asks about."""

    @property
    def file_path(self) -> str:
        return self.source.file_path

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.source.line_of(self.node)}"


@dataclass(frozen=True)
class KeywordBinders:
    """Which calls a class handed on as a value reaches only to be built by field
    name, or never built.

    A callee the repository declares a keyword binder (`keyword-binders`) is one:
    an encoder, a decoder, a query binding its filters. So is a scanned top-level
    function whose parameter the class binds is only compared, tested as a type,
    or handed to a keyword binder in turn. The class may travel inside a type
    form (`tuple[X, ...]`), as a decoder's target does.
    """

    snapshot: CodemodSourceSnapshot
    DEPTH: ClassVar[int] = 4
    SUPER: ClassVar[str] = f"{builtins.__name__}.{super.__name__}"
    """`super(C, obj)` reads `C`'s MRO to look names up; it never calls `C`."""

    @staticmethod
    def receiving_call(site: ClassReferenceSite) -> tuple[ast.Call, ast.expr] | None:
        """The call the class is handed to, with the argument that carries it:
        directly, inside a type form (`tuple[X, ...]`), or by keyword."""

        module = site.source.parsed_module
        parents = {
            child: parent for parent in ast.walk(module.module) for child in ast.iter_child_nodes(parent)
        }
        argument: ast.AST = site.node
        parent = parents.get(argument)
        while isinstance(parent, ast.Tuple) or (
            isinstance(parent, ast.Subscript) and parent.slice is argument
        ):
            argument, parent = parent, parents.get(parent)
        if isinstance(parent, ast.keyword):
            argument, parent = parent.value, parents.get(parent)
        if isinstance(parent, ast.Call) and isinstance(argument, ast.expr) and argument is not parent.func:
            return parent, argument
        return None

    def binds_by_keyword(self, site: ClassReferenceSite) -> bool:
        received = self.receiving_call(site)
        return received is not None and self.call_binds_by_keyword(site.source.parsed_module, *received, 0)

    def call_binds_by_keyword(
        self, module: ParsedModule, call: ast.Call, argument: ast.expr, depth: int
    ) -> bool:
        if depth > self.DEPTH or argument is call.func:
            return False
        # A function body looks a global up when it runs: the module's final binding.
        witness = self.snapshot.module_binding_proof.reference_or_builtin_witness_at(
            module, call.func, line=len(module.source.splitlines()) + 1
        )
        if witness is None:
            return False
        if witness.qualified_name == self.SUPER and call.args[:1] == [argument]:
            return True
        if self.declared_binder(module, call.func, witness.qualified_name):
            return True
        callee_module, _, name = witness.qualified_name.rpartition(".")
        function = next(
            (
                statement
                for parsed in self.snapshot.parsed_modules
                if parsed.module_name == callee_module
                for statement in parsed.module.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == name
            ),
            None,
        )
        parameter = self.parameter(function, call, argument) if function is not None else None
        if function is None or parameter is None:
            return False
        reader = next(parsed for parsed in self.snapshot.parsed_modules if parsed.module_name == callee_module)
        parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(function):
            if not (isinstance(node, ast.Name) and node.id == parameter):
                continue
            if not isinstance(node.ctx, ast.Load):
                return False
            parent = parents.get(node)
            match parent:
                case ast.arg() | ast.Compare():
                    continue
                case ast.Call(func=ast.Name(id="isinstance" | "issubclass")):
                    continue
                # Built by name alone: no position for a reordered field to shift.
                case ast.Call(func=func, args=[]) if func is node:
                    continue
                case ast.Call() if self.call_binds_by_keyword(reader, parent, node, depth + 1):
                    continue
            return False
        return True

    def declared(self, module: ParsedModule) -> frozenset[str]:
        """The repository's keyword binders, each naming a function that exists: a
        stale entry would silently prove nothing, so it refuses instead."""

        names = DeclaredRoot.keyword_binders_for(Path(module.file_path))
        for name in sorted(names):
            if not self.names_a_function(name):
                raise NeedsInput(
                    f"keyword-binders names `{name}`, which no scanned or importable "
                    "class or module defines", NeededInput.SOURCE_CHANGE
                )
        return names

    def names_a_function(self, name: str) -> bool:
        owner, _, attribute = name.rpartition(".")
        class_index = self.snapshot.class_overlap_verdicts.class_index
        if class_index.class_for(owner) is not None:
            return bool(class_index.declared_member(owner, attribute).answer(DeclaringClass()))
        if any(
            parsed.module_name == owner
            and any(
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == attribute
                for statement in parsed.module.body
            )
            for parsed in self.snapshot.parsed_modules
        ):
            return True
        try:
            return callable(getattr(import_module(owner), attribute))
        except (ImportError, AttributeError, ValueError):
            return False

    def declared_binder(self, module: ParsedModule, func: ast.expr, qualified_name: str) -> bool:
        """Whether the repository declares this callee a keyword binder: by its
        own name, or, for a method looked up on a scanned class, by the name of
        the class that declares the method."""

        readers = self.declared(module)
        if qualified_name in readers:
            return True
        class_name, _, attribute = qualified_name.rpartition(".")
        class_index = self.snapshot.class_overlap_verdicts.class_index
        if not isinstance(func, ast.Attribute) or class_index.class_for(class_name) is None:
            return False
        declaring = class_index.declared_member(class_name, attribute).answer(DeclaringClass())
        return any(f"{symbol}.{attribute}" in readers for symbol in declaring)

    @staticmethod
    def parameter(
        function: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call, argument: ast.expr
    ) -> str | None:
        """The parameter of a plain function that `argument` binds at `call`."""

        positional = [item.arg for item in (*function.args.posonlyargs, *function.args.args)]
        if argument in call.args:
            index = call.args.index(argument)
            return positional[index] if index < len(positional) else None
        for keyword in call.keywords:
            if keyword.value is argument and keyword.arg in (*positional, *(item.arg for item in function.args.kwonlyargs)):
                return keyword.arg
        return None


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
    def binders(self) -> KeywordBinders:
        return KeywordBinders(self.snapshot)

    @cached_property
    def sites(self) -> tuple[ClassReferenceSite, ...]:
        terminal_names = frozenset(symbol.rsplit(".", 1)[-1] for symbol in self.symbols)
        return tuple(
            site
            for parsed_module in self.snapshot.parsed_modules
            for source in (ModuleSource(parsed_module), *parsed_module.fact(EmbeddedSources).sources)
            for site in self._source_sites(source, terminal_names)
        )

    def _source_sites(
        self, source: ReadSource, terminal_names: frozenset[str]
    ) -> Iterable[ClassReferenceSite]:
        parsed_module = source.parsed_module
        aliases = module_import_aliases(parsed_module)
        identifiers = terminal_names | frozenset(
            alias
            for alias, qualified_name in (
                *aliases.items(),
                *parsed_module.fact(ScopedImportBindings).aliases(),
            )
            if qualified_name.rsplit(".", 1)[-1] in terminal_names
        )
        syntax_index = parsed_module.syntax_index
        nodes = syntax_index.depth_first_nodes
        parents = syntax_index.parent_indices
        resolver = self.snapshot.class_reference_resolver_for_module(source.parsed_module).reference_resolution
        for node_type in (ast.Name, ast.Attribute):
            for index in syntax_index.node_indices_by_type.get(node_type, ()):
                node = nodes[index]
                identifier = node.id if isinstance(node, ast.Name) else node.attr
                if identifier not in identifiers or not isinstance(node.ctx, ast.Load):
                    continue
                parent = nodes[parents[index]] if parents[index] >= 0 else None
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                symbol = resolver.symbol_for_node(node)
                if symbol not in self.symbols:
                    continue
                grandparent = (
                    nodes[parents[parents[index]]]
                    if parent is not None and parents[parents[index]] >= 0
                    else None
                )
                as_type = reads_as_type(index, nodes, parents)
                subject = subclass_test_subject(node, parent, grandparent)
                yield ClassReferenceSite(
                    source=source,
                    node=node,
                    parent=parent,
                    role=ClassReferenceRole.of_reference(
                        node, parent, grandparent, aliases, as_type
                    ),
                    is_annotation=as_type,
                    is_base=(
                        (isinstance(parent, ast.ClassDef) and node in parent.bases)
                        or (
                            isinstance(parent, ast.Subscript)
                            and parent.value is node
                            and isinstance(grandparent, ast.ClassDef)
                            and parent in grandparent.bases
                        )
                    ),
                    symbol=symbol,
                    tested_subclass=None if subject is None else resolver.symbol_for_node(subject),
                )

    def require_no_identity_reads(self, answers: Mapping[str, SubclassAnswers]) -> None:
        """No type test, class pattern or type key reads these classes with an
        answer the move changes; `answers` says, per class, which it changes."""

        for site in self.sites:
            if site.role.distinguishes_identity and answers[site.symbol].changes(site.tested_subclass):
                raise NeedsInput(
                    f"{site.location} reads `{ast.unparse(site.node)}` as a "
                    f"{site.role.value}: the move changes what it answers", NeededInput.SOURCE_CHANGE
                )

    def require_keyword_construction(self) -> None:
        """Every read builds the class by keyword, tests it, annotates or subclasses."""

        for site in self.sites:
            positional = site.role.positional_bindings(site.parent)
            if positional:
                raise NeedsInput(
                    f"{site.location} binds {positional} field(s) of "
                    f"`{ast.unparse(site.node)}` by position, as a {site.role.value}", NeededInput.SOURCE_CHANGE
                )
        self.require_no_value_hand_on()

    def require_no_value_hand_on(self) -> None:
        """No read hands the class on as a value: whoever calls that value may
        bind fields by position, and no keyword rewrite reaches that caller,
        unless the class binds none: a call to one whose fields are all
        keyword-only already names every field, so a move that keeps them
        keyword-only cannot shift it."""

        for site in self.sites:
            if (
                site.role.binds_fields
                or site.role.distinguishes_identity
                or site.is_annotation
                or site.is_base
                or self.binders.binds_by_keyword(site)
                or not proved_construction_parameters(self.snapshot, site.symbol)
            ):
                continue
            received = self.binders.receiving_call(site)
            raise NeedsInput(
                f"{site.location} passes `{ast.unparse(site.node)}` on as a value"
                + ("" if received is None else f" to `{ast.unparse(received[0].func)}`")
                + "; how its callers bind fields is unproved", NeededInput.PROOF
            )


def proved_construction_parameters(snapshot: CodemodSourceSnapshot, symbol: str) -> tuple[str, ...]:
    """The fields a call to `symbol` binds by position, once its creation is
    proved to keep what its declarations bind; refuses when that is unproved."""

    indexed = OverlapClass.for_symbol(snapshot, symbol).facts
    ClassCreationProof.constructing(
        snapshot, indexed, require_static_c3(snapshot, indexed)
    ).require_construction_kept()
    return snapshot.class_overlap_verdicts.class_index.construction_parameters(symbol)


def require_static_c3(
    snapshot: CodemodSourceSnapshot, indexed_class: CompactIndexedClass
) -> tuple[CompactIndexedClass, ...]:
    """The class and its ancestors, when type's own C3 orders them all.

    Every base resolves to a native root (`NativeMroBase`, or one whose
    namespace NRA reads, `NAMESPACE_READ_NATIVES`), a scanned class, or a class an
    installed module declares in `.py` source (`ClassOverlapVerdicts.ancestor`),
    and creating the class keeps that order (`ClassCreationOrderProof`, the proof
    the MRO authority reads). Class keywords and `__init_subclass__` leave the
    order to type's C3; what they read as a class is created is
    `ClassCreationProof`'s. Returns the closed ancestry.
    """

    verdicts = snapshot.class_overlap_verdicts
    reached: dict[str, CompactIndexedClass] = {}
    pending = [indexed_class]
    while pending:
        current = pending.pop()
        if current.symbol in reached:
            continue
        reached[current.symbol] = current
        if not current.base_references_are_complete:
            raise NeedsInput(f"`{current.qualname}` has unresolved bases", NeededInput.PROOF)
        for reference in current.base_references:
            qualified_name = ".".join(reference.resolved_parts)
            if NativeMroBase.for_qualified_name(qualified_name) is not None or namespace_read_native(qualified_name):
                continue
            base = verdicts.ancestor(qualified_name)
            if base is None:
                raise NeedsInput(
                    f"`{current.qualname}` has base `{qualified_name}`, which neither the scan "
                    "nor an installed `.py` source declares", NeededInput.PROOF
                )
            pending.append(base)
    ancestry = tuple(reached.values())
    verdicts.creation.require_order(ancestry)
    return ancestry


def require_verdict(
    snapshot: CodemodSourceSnapshot,
    classes: Iterable[str],
    move: type[ClassOverlapMove],
    *,
    owner_symbol: str | None = None,
) -> ClassOverlapCandidate:
    """The scan's verdict for this move over these classes, with no guard fired.

    With `owner_symbol`, `classes` restate that owner, possibly in several
    pair verdicts that must all permit the move; when the scan reports none,
    each pair is judged on its own, below the reporting floor. Without it,
    `classes` are exactly one overlap's two classes.
    """

    wanted = frozenset(classes)
    matches = tuple(
        candidate
        for candidate in snapshot.class_overlap_verdicts.candidates_touching(
            wanted if owner_symbol is None else frozenset((owner_symbol,))
        )
        if candidate.move is move
        and (
            wanted <= candidate.restating_symbols
            if owner_symbol is not None
            else frozenset((candidate.owner.symbol, *candidate.restating_symbols))
            == wanted
        )
        and owner_symbol in (None, candidate.owner.symbol)
    )
    if not matches and owner_symbol is not None:
        requested = tuple(
            snapshot.class_overlap_verdicts.requested_verdict(owner_symbol, symbol)
            for symbol in sorted(wanted)
        )
        if all(
            candidate is not None
            and candidate.move is move
            and candidate.owner.symbol == owner_symbol
            for candidate in requested
        ):
            matches = tuple(candidate for candidate in requested if candidate is not None)
    if not matches and owner_symbol is None and len(wanted) == 2:
        first, second = sorted(wanted)
        judged = snapshot.class_overlap_verdicts.requested_as(first, second, move)
        matches = () if judged is None or judged.move is not move else (judged,)
    if not matches or (owner_symbol is None and len(matches) != 1):
        raise ValueError(
            f"The current source has {len(matches)} `{move.value}` verdicts over "
            f"{sorted(wanted)}"
        )
    blocking = tuple(receipt for candidate in matches for receipt in candidate.refusals)
    if blocking:
        raise NeedsInput(
            "The move needs: "
            + "; ".join(dict.fromkeys(receipt.detail for receipt in blocking)),
            *(receipt.kind.needs for receipt in blocking),
        )
    return matches[0]


def obsolete_import_edits(
    snapshot: CodemodSourceSnapshot,
    file_path: str,
    deleted: tuple[ast.AST, ...],
    rationale: str,
) -> tuple[ModuleImportMutation, ...]:
    """Drop `from` imports that only the deleted statements and bases read.

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
    reexported = snapshot.names_imported_from_module(
        parsed_module.file_path, parsed_module.module_name
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
                raise NeedsInput(
                    f"`{name}` is `{actual.qualified_name}` in "
                    f"{destination.file_path}, not `{expected.qualified_name}`", NeededInput.SOURCE_CHANGE
                )
            continue
        targets = snapshot.source_index.targets_matching_repository_symbol(
            expected.qualified_name
        )
        if len(targets) != 1:
            raise NeedsInput(f"`{expected.qualified_name}` has no single declaration", NeededInput.PROOF)
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


def _super_looked_up_names(node: ast.ClassDef) -> tuple[frozenset[str], bool]:
    """Direct `super().<attr>` names in `node`, and whether any non-attribute `super()` use exists."""

    attrs: set[str] = set()
    dynamic = False
    attribute_super_ids: set[int] = set()
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Name)
            and item.value.func.id == "super"
        ):
            attrs.add(item.attr)
            attribute_super_ids.add(id(item.value))
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "super"
            and id(item) not in attribute_super_ids
        ):
            dynamic = True
            break
    return frozenset(attrs), dynamic


def _has_super_call(node: ast.ClassDef) -> bool:
    attrs, dynamic = _super_looked_up_names(node)
    return bool(attrs) or dynamic


@dataclass(frozen=True)
class DescendantBaseDrop:
    """Direct owner bases dropped from a descendant of the composer so C3
    still orders the descendant once the composer inherits those owners."""

    descendant: OverlapClass
    dropped: tuple[tuple[ast.expr, str], ...]

    def header_edits(
        self, snapshot: CodemodSourceSnapshot, rationale: str
    ) -> tuple[NominalSourceEdit, ...]:
        node = self.descendant.node
        header = ClassHeaderSpanSourceAuthority(
            node, snapshot.sources_by_file_path[self.descendant.file_path]
        )
        dropped = frozenset(id(base) for base, _supplier in self.dropped)
        return header.source_edits(
            header.with_base_items(
                tuple(
                    header.item_source(base)
                    for base in node.bases
                    if id(base) not in dropped
                )
            ),
            file_path=self.descendant.file_path,
            rationale=rationale,
        )

    def rationale(self) -> str:
        return " ".join(
            f"Drops `{ast.unparse(base)}` from `{self.descendant.qualname}`'s bases: "
            f"`{supplier.rpartition('.')[2]}` now supplies it."
            for base, supplier in self.dropped
        )


def emit_descendant_drop_edits(
    snapshot: CodemodSourceSnapshot,
    drops: Sequence[DescendantBaseDrop],
    rationale: str,
    *,
    skip_obsolete_import_paths: frozenset[str] = frozenset(),
) -> tuple[NominalSourceEdit, ...]:
    """Header edits and per-file obsolete import removals for descendant drops."""

    if not drops:
        return ()
    edits: list[NominalSourceEdit] = []
    dropped_by_file: dict[str, list[ast.expr]] = {}
    for drop in drops:
        edits.extend(drop.header_edits(snapshot, rationale))
        if drop.descendant.file_path not in skip_obsolete_import_paths:
            dropped_by_file.setdefault(drop.descendant.file_path, []).extend(
                base for base, _supplier in drop.dropped
            )
    for file_path, dropped_nodes in dropped_by_file.items():
        edits.extend(
            obsolete_import_edits(snapshot, file_path, tuple(dropped_nodes), rationale)
        )
    return tuple(edits)


@dataclass(frozen=True)
class ComposedAncestry:
    """A composer's C3 order before and after its owners become its first bases.

    Python looks every name up along the class's C3 order, and the specified
    lookup winners are fixed by the requirement, whatever the representation
    (paper1 Theorem 4.17). So composing keeps the class's behaviour exactly
    when every name keeps its winner: a name the class keeps resolves where it
    did, a name it deletes resolves to its owner's equal declaration, and it
    gains no name. The old ancestors keep their order, so dataclass collection
    walks them as before. An ancestor the class gains besides its owners adds
    no fields or metaclass and is read as no identity: a type test of it would
    answer differently for the class. Its creation hook, `__init_subclass__`,
    is a name like any other, so the composer's creation runs the same hook.
    Every scanned descendant of the composer is proved by the same C3 order and
    lookup check; when a descendant already listed an owner directly ahead of
    the composer, dropping that now-redundant direct base is proved and emitted
    with the composition.
    """

    snapshot: CodemodSourceSnapshot
    composer: OverlapClass
    owners: tuple[OverlapClass, ...]
    deleted: frozenset[str]
    peer_overrides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    peer_deleted: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def creation(self) -> ClassCreationProof:
        """What the composer's creation reads. An abstract method moved to an owner
        `ABCMeta` creates stays abstract: `ABCMeta` takes each base's abstract
        names and keeps those the class still resolves to an abstract function,
        which is the owner's equal declaration."""

        proof = ClassCreationProof.composing(
            self.snapshot,
            self.composer.facts,
            self.ancestry,
            tuple((owner.qualname, owner.node) for owner in self.owners),
            MroWalk(self.before, self.after, self.composer.symbol, self.deleted),
        )
        abstract_owners = tuple(
            owner
            for owner in self.owners
            if NativeMroBase.ABSTRACT_ROOT.value in self.c3.linearized(owner.symbol)
        )
        kept = frozenset(
            name for owner in abstract_owners for name in self.names(owner.symbol, composed=False)
        ) - frozenset(
            name
            for owner in self.owners
            if owner not in abstract_owners
            for name in self.names(owner.symbol, composed=False)
        )
        return replace(proof, abstract_moved=proof.abstract_moved - kept)

    @property
    def c3(self) -> ClassOverlapVerdicts:
        """The C3 orders of the ancestry, scanned or installed: `require_static_c3`
        read every one of them through this same lookup."""

        return self.snapshot.class_overlap_verdicts

    def names(self, symbol: str, *, composed: bool) -> frozenset[str]:
        """What a class declares; the composer, once composed, without its deleted names."""

        if NativeMroBase.for_qualified_name(symbol) is not None:
            return frozenset()
        if (native := namespace_read_native(symbol)) is not None:
            return native_declared_names(native)
        declared = frozenset(declaration.name for declaration in self.c3.declared(symbol).body_declarations)
        if composed:
            if symbol == self.composer.symbol:
                return declared - self.deleted
            if symbol in self.peer_deleted:
                return declared - self.peer_deleted[symbol]
        return declared

    @cached_property
    def before(self) -> tuple[str, ...]:
        """The composer's C3 order as declared."""

        return self.c3.linearized(self.composer.symbol)

    @cached_property
    def kept_bases(self) -> tuple[str, ...]:
        return tuple(
            base for base in self.c3.bases(self.composer.symbol) if base != NativeMroBase.OBJECT.value
        )

    @cached_property
    def after(self) -> tuple[str, ...]:
        """The composer's C3 order with its owners as its first bases."""

        owner_symbols = tuple(owner.symbol for owner in self.owners)
        return self.c3.linearized(self.composer.symbol, (*owner_symbols, *self.kept_bases))

    @cached_property
    def restated(self) -> tuple[tuple[ast.expr, str], ...]:
        """Each direct base of the composer another of its bases, its owners first,
        already supplies, with that base (`DeclaredC3.restated_bases`)."""

        referenced = tuple(_referenced_bases(self.composer.node))
        if len(referenced) != len(self.composer.node.bases):
            return ()
        restated = self.c3.restated_bases(
            self.composer.symbol,
            tuple(isinstance(node, (ast.Name, ast.Attribute)) for node, _parts in referenced),
            tuple(owner.symbol for owner in self.owners),
        )
        return tuple((referenced[index][0], supplier) for index, supplier in restated)

    @cached_property
    def redundant_bases(self) -> tuple[ast.expr, ...]:
        return tuple(node for node, _supplier in self.restated)

    def header_bases(self, header: ClassHeaderSpanSourceAuthority) -> tuple[str, ...]:
        """The composer's new base list: its owners first, then every base it
        lists that is not redundant, as the header spells it."""

        dropped = frozenset(id(node) for node in self.redundant_bases)
        return (
            *(owner.target.name for owner in self.owners),
            *(header.item_source(base) for base in self.composer.node.bases if id(base) not in dropped),
        )

    def rationale(self, rationale: str) -> str:
        """The step's rationale, naming each base the composer or a descendant no longer lists."""

        return " ".join(
            (
                rationale,
                *(
                    f"Drops `{ast.unparse(node)}` from `{self.composer.qualname}`'s bases: "
                    f"`{supplier.rpartition('.')[2]}` already supplies it."
                    for node, supplier in self.restated
                ),
                *(drop.rationale() for drop in self.descendant_drops),
            )
        )

    def descendant_edits(
        self,
        rationale: str,
        *,
        skip_obsolete_import_paths: frozenset[str] = frozenset(),
    ) -> tuple[NominalSourceEdit, ...]:
        return emit_descendant_drop_edits(
            self.snapshot,
            self.descendant_drops,
            rationale,
            skip_obsolete_import_paths=skip_obsolete_import_paths,
        )

    @cached_property
    def descendant_drops(self) -> tuple[DescendantBaseDrop, ...]:
        """Prove every scanned descendant of the composer still linearizes and
        keeps its lookups; drop any direct plain owner base whose presence ahead
        of the composer breaks C3 when dropping it is proved behaviour-preserving."""

        class_index = self.snapshot.class_overlap_verdicts.class_index
        descendant_set = frozenset(class_index.descendant_symbols(self.composer.symbol))
        if not descendant_set:
            return ()
        owner_symbols = tuple(owner.symbol for owner in self.owners)
        owner_set = frozenset(owner_symbols)
        ordered = sorted(
            descendant_set,
            key=lambda sym: (
                len(frozenset(class_index.ancestor_symbols(sym)) & descendant_set),
                sym,
            ),
        )
        overrides: dict[str, tuple[str, ...]] = {
            **self.peer_overrides,
            self.composer.symbol: (*owner_symbols, *self.kept_bases),
        }
        dropped_by_symbol: dict[str, frozenset[str]] = {}
        drops: list[DescendantBaseDrop] = []
        for symbol in ordered:
            descendant = OverlapClass.for_symbol(self.snapshot, symbol)
            d_ancestry = require_static_c3(self.snapshot, descendant.facts)
            before_d = self.c3.linearized(symbol)
            dropped_symbols: tuple[str, ...] = ()
            try:
                after_d = self.c3.linearized_with(symbol, overrides)
            except NeedsInput:
                referenced = tuple(_referenced_bases(descendant.node))
                if len(referenced) != len(descendant.node.bases):
                    raise
                listed = self.c3.bases(symbol)
                drop_indices = tuple(
                    index
                    for index, ((base_node, _parts), base_symbol) in enumerate(
                        zip(referenced, listed)
                    )
                    if isinstance(base_node, (ast.Name, ast.Attribute))
                    and base_symbol in owner_set
                    and base_symbol not in BASES_READ_AT_CREATION
                )
                if not drop_indices:
                    raise
                drop_index_set = frozenset(drop_indices)
                new_d_bases = tuple(
                    base for index, base in enumerate(listed) if index not in drop_index_set
                ) or (NativeMroBase.OBJECT.value,)
                after_d = self.c3.linearized_with(symbol, {**overrides, symbol: new_d_bases})
                require_rewritable_header(self.snapshot, descendant)
                dropped_symbols = tuple(listed[index] for index in drop_indices)
                creation = ClassCreationProof.dropping(
                    self.snapshot, descendant.facts, d_ancestry, dropped_symbols
                )
                creation.require_unchanged()
                BasesReaders(self.snapshot).require_unread(
                    descendant.facts, dropped_symbols, creation.installed_packages()
                )
                if descendant.facts.dataclass_declaration.generates_construction and any(
                    owner.own_field_names
                    for owner in self.owners
                    if owner.symbol in dropped_symbols
                ):
                    dropped_names = ", ".join(
                        f"`{item.rpartition('.')[2]}`" for item in dropped_symbols
                    )
                    raise NeedsInput(
                        f"Dropping {dropped_names} "
                        f"from dataclass `{descendant.qualname}` may reorder its fields",
                        NeededInput.PROOF,
                    )
                overrides[symbol] = new_d_bases
                dropped_by_symbol[symbol] = frozenset(dropped_symbols)
                drops.append(
                    DescendantBaseDrop(
                        descendant,
                        tuple(
                            (referenced[index][0], self.composer.symbol)
                            for index in drop_indices
                        ),
                    )
                )
            ancestor_dropped: set[str] = set()
            for ancestor in before_d[1:]:
                ancestor_dropped |= dropped_by_symbol.get(ancestor, frozenset())
            shifted = frozenset(dropped_symbols) | (frozenset(before_d) & ancestor_dropped)
            kept_before = tuple(item for item in before_d if item not in shifted)
            kept_after = tuple(
                item for item in after_d if item in before_d and item not in shifted
            )
            if kept_after != kept_before:
                raise NeedsInput(
                    f"Composing reorders `{descendant.qualname}`'s ancestors: "
                    f"{[item.rpartition('.')[2] for item in before_d]} becomes "
                    f"{[item.rpartition('.')[2] for item in after_d]}",
                    NeededInput.PROOF,
                )
            for shifted_owner in sorted(shifted):
                before_pos = before_d.index(shifted_owner)
                after_pos = after_d.index(shifted_owner)
                passed = tuple(
                    item
                    for item in before_d
                    if before_d.index(item) > before_pos
                    and item in after_d
                    and after_d.index(item) < after_pos
                )
                for checked in (shifted_owner, *passed):
                    if class_index.class_for(checked) is None or _has_super_call(
                        OverlapClass.for_symbol(self.snapshot, checked).node
                    ):
                        raise NeedsInput(
                            f"Dropping `{shifted_owner.rpartition('.')[2]}` shifts "
                            f"`{checked.rpartition('.')[2]}` in `{descendant.qualname}`'s MRO "
                            "across a `super()` call",
                            NeededInput.PROOF,
                        )
                owner_names = self.names(shifted_owner, composed=False) - {"__slots__"}
                for item in passed:
                    overlap = sorted(owner_names & (self.names(item, composed=True) - {"__slots__"}))
                    if overlap:
                        raise NeedsInput(
                            f"Dropping `{shifted_owner.rpartition('.')[2]}` from "
                            f"`{descendant.qualname}` reorders {overlap} with "
                            f"`{item.rpartition('.')[2]}`",
                            NeededInput.PROOF,
                        )
            d_names = frozenset().union(
                *(self.names(item, composed=False) for item in before_d),
                *(self.names(item, composed=False) for item in after_d),
            )
            for name in sorted(d_names):
                was = next(
                    (item for item in before_d if name in self.names(item, composed=False)),
                    None,
                )
                becomes = next(
                    (item for item in after_d if name in self.names(item, composed=True)),
                    None,
                )
                if was == becomes:
                    continue
                if was == self.composer.symbol and name in self.deleted and becomes in owner_set:
                    continue
                if (
                    was is not None
                    and was in self.peer_deleted
                    and name in self.peer_deleted[was]
                    and becomes in owner_set
                ):
                    continue
                if name == "__slots__" and was in shifted and becomes == self.composer.symbol:
                    continue
                raise NeedsInput(
                    f"Composing makes `{descendant.qualname}.{name}` resolve to "
                    f"`{becomes or 'nothing'}` instead of `{was or 'nothing'}`",
                    NeededInput.PROOF,
                )
            self.require_super_lookups(descendant.qualname, before_d, after_d)
            d_bases = overrides.get(symbol, self.c3.bases(symbol))
            non_object_bases = tuple(
                base for base in d_bases if base != NativeMroBase.OBJECT.value
            )
            if len(non_object_bases) > 1:
                self.require_slots_fit(non_object_bases)
        return tuple(drops)

    @cached_property
    def ancestry(self) -> tuple[CompactIndexedClass, ...]:
        """The composed class's closed ancestry."""

        composer = self.composer
        ancestry = {
            indexed.symbol: indexed
            for start in (composer, *self.owners)
            for indexed in require_static_c3(self.snapshot, start.facts)
        }
        owner_symbols = tuple(owner.symbol for owner in self.owners)
        before = self.before
        for owner in self.owners:
            if owner.symbol in before:
                raise ValueError(f"`{owner.qualname}` is already an ancestor")
        kept_bases = self.kept_bases
        after = self.after
        if tuple(symbol for symbol in after if symbol in before) != before:
            raise NeedsInput(
                f"Composing reorders `{composer.qualname}`'s ancestors: "
                f"{[symbol.rpartition('.')[2] for symbol in before]} becomes "
                f"{[symbol.rpartition('.')[2] for symbol in after]}",
                NeededInput.PROOF,
            )
        gained = tuple(
            symbol for symbol in after if symbol not in before and symbol not in owner_symbols
        )
        for symbol in gained:
            self.require_inert_gain(symbol)
        names = frozenset().union(
            *(self.names(symbol, composed=False) for symbol in after)
        )
        for name in sorted(names):
            was = next(
                (symbol for symbol in before if name in self.names(symbol, composed=False)), None
            )
            becomes = next(
                (symbol for symbol in after if name in self.names(symbol, composed=True)), None
            )
            if name in self.deleted and becomes in owner_symbols:
                continue
            if was != becomes:
                raise NeedsInput(
                    f"Composing makes `{composer.qualname}.{name}` resolve to "
                    f"`{becomes or 'nothing'}` instead of `{was or 'nothing'}`",
                    NeededInput.PROOF,
                )
        self.require_super_lookups(composer.qualname, before, after)
        self.require_slots_fit(kept_bases)
        self.descendant_drops
        class_index = self.snapshot.class_overlap_verdicts.class_index
        ClassReferenceSites(self.snapshot, frozenset(gained)).require_no_identity_reads(
            {symbol: AnswersFor.gaining(symbol, (composer.symbol,), class_index) for symbol in gained}
        )
        return tuple(ancestry.values())

    def require_super_lookups(
        self,
        subject_qualname: str,
        before: tuple[str, ...],
        after: tuple[str, ...],
    ) -> None:
        """Every `super().<attr>` lookup along `before -> after` keeps its next-MRO winner."""

        class_index = self.snapshot.class_overlap_verdicts.class_index
        owner_set = frozenset(owner.symbol for owner in self.owners)
        before_set = frozenset(before)
        all_names = frozenset().union(
            *(self.names(item, composed=False) for item in after)
        )

        # 1. Existing classes in `before` whose post-class MRO tail changed.
        for index_before, cls_sym in enumerate(before):
            if cls_sym not in after:
                continue
            b_tail = before[index_before + 1 :]
            a_tail = after[after.index(cls_sym) + 1 :]
            if a_tail == b_tail:
                continue
            if class_index.class_for(cls_sym) is not None:
                attrs, dynamic = _super_looked_up_names(
                    OverlapClass.for_symbol(self.snapshot, cls_sym).node
                )
                check_names = all_names if dynamic else attrs
            elif any(item not in before_set for item in a_tail):
                check_names = all_names
            else:
                continue
            for name in sorted(check_names):
                was = next(
                    (s for s in b_tail if name in self.names(s, composed=False)), None
                )
                becomes = next(
                    (s for s in a_tail if name in self.names(s, composed=True)), None
                )
                if was == becomes:
                    continue
                if (
                    was == self.composer.symbol
                    and name in self.deleted
                    and becomes in owner_set
                ):
                    continue
                if (
                    was is not None
                    and was in self.peer_deleted
                    and name in self.peer_deleted[was]
                    and becomes in owner_set
                ):
                    continue
                raise NeedsInput(
                    f"Composing makes `super().{name}` in `{cls_sym.rpartition('.')[2]}` "
                    f"(via `{subject_qualname}`) resolve to `{becomes or 'nothing'}` "
                    f"instead of `{was or 'nothing'}`",
                    NeededInput.PROOF,
                )

        # 2. Newly inserted owners and gained ancestors calling `super()` into the composed tail.
        for index_after, ins in enumerate(after):
            if ins in before_set or class_index.class_for(ins) is None:
                continue
            attrs, dynamic = _super_looked_up_names(
                OverlapClass.for_symbol(self.snapshot, ins).node
            )
            check_names = all_names if dynamic else attrs
            if not check_names:
                continue
            ins_tail = self.c3.linearized(ins)[1:]
            a_tail = after[index_after + 1 :]
            for name in sorted(check_names):
                was = next(
                    (s for s in ins_tail if name in self.names(s, composed=False)), None
                )
                becomes = next(
                    (s for s in a_tail if name in self.names(s, composed=True)), None
                )
                if was != becomes:
                    raise NeedsInput(
                        f"Composing makes `super().{name}` in `{ins.rpartition('.')[2]}` "
                        f"(via `{subject_qualname}`) resolve to `{becomes or 'nothing'}` "
                        f"instead of `{was or 'nothing'}`",
                        NeededInput.PROOF,
                    )

    def require_inert_gain(self, symbol: str) -> None:
        if NativeMroBase.for_qualified_name(symbol) is not None:
            raise NeedsInput(
                f"Composing makes `{self.composer.qualname}` a `{symbol}`", NeededInput.PROOF
            )
        if namespace_read_native(symbol) is not None:
            return
        indexed = self.c3.declared(symbol)
        if indexed.metaclass_names or "metaclass" in indexed.class_keyword_names:
            raise NeedsInput(
                f"Composing gives `{self.composer.qualname}` the metaclass of `{indexed.qualname}`",
                NeededInput.PROOF,
            )
        if any(declaration.kind.is_field for declaration in indexed.body_declarations):
            raise NeedsInput(
                f"Composing gives `{self.composer.qualname}` the fields of `{indexed.qualname}`",
                NeededInput.PROOF,
            )

    def require_slots_fit(self, kept_bases: tuple[str, ...]) -> None:
        """Two bases with non-empty instance slots cannot share one layout."""

        if not kept_bases:
            return
        for owner in self.owners:
            declared = any(
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__slots__"
                    for target in statement.targets
                )
                and not (isinstance(statement.value, (ast.Tuple, ast.List)) and not statement.value.elts)
                for statement in owner.node.body
            )
            if declared or (owner.option("slots") == "True" and owner.own_field_names):
                raise NeedsInput(
                    f"`{owner.qualname}` lays out instance slots beside "
                    f"`{self.composer.qualname}`'s other bases",
                    NeededInput.PROOF,
                )


def require_no_creation_keywords(owner: OverlapClass) -> None:
    if owner.node.keywords:
        raise NeedsInput(
            f"`{owner.qualname}` passes class creation keywords: NRA needs a proof "
            "that the new base leaves what its class creation sees unchanged",
            NeededInput.PROOF,
        )


def require_rewritable_header(snapshot: CodemodSourceSnapshot, composer: OverlapClass) -> None:
    header = ClassHeaderSpanSourceAuthority(composer.node, snapshot.sources_by_file_path[composer.file_path])
    if not header.can_rewrite:
        raise NeedsInput(f"`{composer.qualname}` has an unrewritable header", NeededInput.OPERATION)


def require_declared_before(owner: OverlapClass, composer: OverlapClass) -> None:
    if owner.file_path == composer.file_path and owner.node.lineno > composer.node.lineno:
        raise NeedsInput(f"`{owner.qualname}` is declared after `{composer.qualname}`", NeededInput.SOURCE_CHANGE)


@dataclass(frozen=True)
class ClassComposition:
    """A class restates an existing class's whole body; it composes that class instead.

    The owner becomes the first base and the restated declarations go. With
    the owner's own C3 order `[owner, object]`, or `[owner, ABC, object]` for a
    class already an `ABC`, and the owner not already an ancestor, C3 places it
    right after the class and leaves every other ancestor in order, so a name the class kept resolves as before and a
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
        # A string statement documents the field before it and runs nothing.
        if not body or any(
            not declared_names(statement) and not is_docstring_statement(statement) for statement in body
        ):
            raise NeedsInput(f"`{owner.qualname}` runs statements beyond declarations", NeededInput.SOURCE_CHANGE)
        restated = RestatedDeclarations.require(
            snapshot,
            owner,
            restating,
            (declaration.name for declaration in owner.facts.body_declarations),
        )
        composition = cls(snapshot, owner, restating, restated)
        class_index = snapshot.class_overlap_verdicts.class_index
        NeedsInput.every(
            composition.require_owner_is_a_root,
            composition.require_field_order,
            lambda: restated.require_no_owner_bound_values(snapshot),
            lambda: ClassReferenceSites(snapshot, frozenset((owner.symbol,))).require_no_identity_reads(
                {owner.symbol: AnswersFor.gaining(owner.symbol, (restating.symbol,), class_index)}
            ),
            lambda: owner.file_path != restating.file_path or composition.relocation.require_safe(snapshot),
        )
        return composition

    def require_owner_is_a_root(self) -> None:
        owner, restating = self.owner, self.restating
        NeedsInput.every(
            lambda: require_no_creation_keywords(owner),
            lambda: self.composed.creation().require_unchanged(),
            lambda: restating.require_dataclass_base(owner, self.restated.names),
            lambda: require_rewritable_header(self.snapshot, restating),
        )

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
    def composed(self) -> ComposedAncestry:
        """The composer's C3 order with the owner first."""

        return ComposedAncestry(
            self.snapshot, self.restating, (self.owner,), frozenset(self.restated.deleted_names)
        )

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
            raise NeedsInput(
                f"Composing `{self.owner.qualname}` reorders the fields of "
                f"`{restating.qualname}`: {before} becomes {after}, and the owner "
                "is not keyword-only", NeededInput.SOURCE_CHANGE
            )
        return KeywordOnlyComposition(self.snapshot, restating)

    @cached_property
    def relocation(self) -> ModuleDeclarationRelocation:
        return ModuleDeclarationRelocation.before_user(
            self.snapshot, declaration=self.owner.target, user=self.restating.target
        )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        restating = self.restating
        rationale = self.composed.rationale(
            KeptDocumentation.composing(self.restated).rationale(rationale)
        )
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
                header.with_base_items(self.composed.header_bases(header)),
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
                    self.restated.written_statements
                ),
                rationale=rationale,
            ),
            *obsolete_import_edits(
                self.snapshot,
                restating.file_path,
                (*self.restated.statements, *self.composed.redundant_bases),
                rationale,
            ),
            *self.composed.descendant_edits(
                rationale,
                skip_obsolete_import_paths=frozenset((restating.file_path,)),
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
    body, so deleting its copies loses nothing; each owner is a root, or an
    `ABC` composed into an `ABC`, so C3 places the owners right after the composer, in the given order, ahead of
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
            ComposeExistingOwner,
            tuple(owner.symbol for owner in owners),
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
        class_index = snapshot.class_overlap_verdicts.class_index
        ClassReferenceSites(
            snapshot, frozenset(owner.symbol for owner in owners)
        ).require_no_identity_reads(
            {
                owner.symbol: AnswersFor.gaining(
                    owner.symbol, (composer.symbol for composer in composers), class_index
                )
                for owner in owners
            }
        )
        return composition

    def require_roots(self) -> None:
        NeedsInput.every(
            *(lambda owner=owner: require_no_creation_keywords(owner) for owner in self.owners),
            *(
                check
                for composer, composed, restatements in zip(
                    self.composers, self.composed, self.restated, strict=True
                )
                for check in (
                    lambda composed=composed: composed.creation().require_unchanged(),
                    *(
                        lambda composer=composer, owner=owner, restatement=restatement: (
                            composer.require_dataclass_base(owner, restatement.names)
                        )
                        for owner, restatement in zip(self.owners, restatements, strict=True)
                    ),
                    *(
                        lambda composer=composer, owner=owner: require_declared_before(owner, composer)
                        for owner in self.owners
                    ),
                    lambda composer=composer: require_rewritable_header(self.snapshot, composer),
                )
            ),
        )

    @cached_property
    def composed(self) -> tuple[ComposedAncestry, ...]:
        """Each composer's C3 order with the owners first."""

        owner_symbols = tuple(owner.symbol for owner in self.owners)
        verdicts = self.snapshot.class_overlap_verdicts
        all_overrides = {
            composer.symbol: (
                *owner_symbols,
                *(
                    base
                    for base in verdicts.bases(composer.symbol)
                    if base != NativeMroBase.OBJECT.value
                ),
            )
            for composer in self.composers
        }
        all_deleted = {
            composer.symbol: frozenset(
                name for restatement in restatements for name in restatement.deleted_names
            )
            for composer, restatements in zip(self.composers, self.restated, strict=True)
        }
        return tuple(
            ComposedAncestry(
                self.snapshot,
                composer,
                self.owners,
                all_deleted[composer.symbol],
                peer_overrides={
                    symbol: bases
                    for symbol, bases in all_overrides.items()
                    if symbol != composer.symbol
                },
                peer_deleted={
                    symbol: names
                    for symbol, names in all_deleted.items()
                    if symbol != composer.symbol
                },
            )
            for composer in self.composers
        )

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
            raise NeedsInput(
                f"Composing {', '.join(f'`{owner.qualname}`' for owner in self.owners)} "
                f"reorders the fields of `{composer.qualname}`: {before} becomes {after}", NeededInput.SOURCE_CHANGE
            )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        for restatements in self.restated:
            for restatement in restatements:
                rationale = KeptDocumentation.composing(restatement).rationale(rationale)
        for composed in self.composed:
            rationale = composed.rationale(rationale)
        edits: list[NominalSourceEdit] = []
        imported: set[tuple[str, str]] = set()
        for composer, restatements, composed in zip(
            self.composers, self.restated, self.composed, strict=True
        ):
            source = self.snapshot.sources_by_file_path[composer.file_path]
            header = ClassHeaderSpanSourceAuthority(composer.node, source)
            deletion = StatementDeletionSource(
                source=source, node=composer.node, file_path=composer.file_path
            )
            statements = tuple(
                statement
                for restatement in restatements
                for statement in restatement.written_statements
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
                        header.with_base_items(composed.header_bases(header)),
                        file_path=composer.file_path,
                        rationale=rationale,
                    ),
                    *deletion.physical_edits(
                        file_path=composer.file_path,
                        replacements=deletion.replacements_for_statements(statements),
                        rationale=rationale,
                    ),
                    *obsolete_import_edits(
                        self.snapshot,
                        composer.file_path,
                        (*statements, *composed.redundant_bases),
                        rationale,
                    ),
                )
            )
        unique_drops: dict[str, DescendantBaseDrop] = {}
        for composed in self.composed:
            for drop in composed.descendant_drops:
                unique_drops.setdefault(drop.descendant.symbol, drop)
        edits.extend(
            emit_descendant_drop_edits(
                self.snapshot,
                tuple(unique_drops.values()),
                rationale,
                skip_obsolete_import_paths=frozenset(
                    composer.file_path for composer in self.composers
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
        raise NeedsInput(f"`{name}` is bound to another declaration there", NeededInput.SOURCE_CHANGE)
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
        fold.require_creation_unchanged()
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

    @cached_property
    def stays_plain(self) -> bool:
        """The block carries no field and the base has no dataclass decorator, so
        the base takes it as a plain class: a dataclass collects fields only from
        dataclass bases, and a decorator here would generate methods every child
        then inherits ahead of its other bases (#421's rule)."""

        return not self.base.facts.dataclass_declaration.generates_construction and (
            self.snapshot.class_overlap_verdicts.takes_no_field(
                None, self.names, *(child.facts for child in self.children)
            )
        )

    def require_dataclass_family(self) -> None:
        base = self.base
        if frozenset(self.names) & base.target.bound_names:
            raise NeedsInput(f"`{base.qualname}` already binds a moved name", NeededInput.SOURCE_CHANGE)
        if self.stays_plain:
            return
        base.require_accepts_moved_fields(self.frozen)
        class_index = self.snapshot.class_overlap_verdicts.class_index
        for symbol in class_index.ancestor_symbols(base.symbol):
            if (ancestor := class_index.class_for(symbol)) is not None:
                ancestor.dataclass_declaration.refuse_inherited_fields(base.qualname)
        for symbol in self.descendant_symbols:
            descendant = OverlapClass.for_symbol(self.snapshot, symbol)
            descendant.require_dataclass()
            if descendant.option("frozen") != self.frozen:
                raise NeedsInput(f"`{descendant.qualname}` differs in frozen-ness", NeededInput.SOURCE_CHANGE)
            if descendant.option("order") == "True":
                raise NeedsInput(f"`{descendant.qualname}` orders by field position", NeededInput.SOURCE_CHANGE)
            descendant.own_field_names
            if {"__init__", "__match_args__"} & descendant.target.bound_names:
                raise NeedsInput(
                    f"`{descendant.qualname}` declares its own construction surface", NeededInput.PROOF
                )

    def require_creation_unchanged(self) -> None:
        """Each child's creation reads none of the names leaving its body: the
        order stays as it is, and only creation code reading them sees the move."""

        authority = self.snapshot.class_overlap_verdicts.class_index.mro_authority
        for child in self.children:
            mro_type = authority.resolve(child.symbol).mro_type
            if mro_type is None:
                raise NeedsInput(f"`{child.symbol}` has an open MRO", NeededInput.PROOF)
            ancestry = mro_type.declarations
            order = tuple(declaration.symbol for declaration in ancestry)
            ClassCreationProof(
                self.snapshot, child.facts, ancestry, self.base.qualname,
                frozenset(self.names), frozenset(),
                MroWalk(order, order, child.symbol, frozenset(self.names)),
            ).require_unchanged()

    def require_lookup(self) -> None:
        """Each moved name resolves to the base wherever it resolved to a child.

        In every descendant's C3 order a child comes before the base. Once the
        name leaves the children for the base, lookup still lands on the base
        unless a class between them declares it, so that is what is checked.
        Dunder names are refused: a native root may declare them.
        """

        dunders = sorted(name for name in self.names if name.startswith("__"))
        if dunders:
            raise NeedsInput(f"Moving {dunders} could expose a native declaration", NeededInput.PROOF)
        class_index = self.snapshot.class_overlap_verdicts.class_index
        authority = self.snapshot.class_overlap_verdicts.class_index.mro_authority
        children = frozenset(child.symbol for child in self.children)
        for symbol in self.descendant_symbols:
            mro_type = authority.resolve(symbol).mro_type
            if mro_type is None:
                raise NeedsInput(f"`{symbol}` has an open MRO", NeededInput.PROOF)
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
                    raise NeedsInput(
                        f"`{between[0]}` would answer `{symbol}.{name}` instead of "
                        f"`{self.base.qualname}`", NeededInput.SOURCE_CHANGE
                    )

    def require_movable(self, restatement: RestatedDeclarations) -> None:
        child = restatement.restating
        movable_members(
            ClassMemberMoveProofContext.between(self.snapshot, child.target, self.base.target),
            child,
            restatement.statements,
        )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        base = self.base
        source = self.snapshot.sources_by_file_path[base.file_path]
        first = self.restated[0]
        body = ClassBodySourceAuthority(base.node, source)
        header_line = source.splitlines(keepends=True)[base.node.lineno - 1]
        indentation = header_line[: len(header_line) - len(header_line.lstrip())]
        documentation = KeptDocumentation(self.children, first.deleted_names)
        rationale = documentation.rationale(rationale)
        return (
            *(() if self.stays_plain else base.decorator_edits_for(self, indentation, rationale)),
            *body.geometry.physical_edits(
                file_path=base.file_path,
                replacements=(
                    body.declaration_block_replacement(
                        first.member_sources(self.snapshot, body.indentation, documentation)
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
                replacements=deletion.replacements_for_statements(restatement.written_statements),
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
                raise NeedsInput("`dataclass` is bound to something else in the base module", NeededInput.SOURCE_CHANGE)
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
                raise NeedsInput(
                    f"{owner.file_path}:{node.lineno} builds its own class with "
                    f"{len(node.args)} positional argument(s)", NeededInput.SOURCE_CHANGE
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
    `str`, `asdict` or `pickle.dumps` of `self`, in a method that calls
    `hashlib` or `json.dumps` without `sort_keys=True`): that digest would
    change with the field order. Nor does their class code read the field
    order itself (`FieldOrderReads`), as a table deriving its stored columns
    does. Cache identities are refused too until they are told apart from
    persisted ones. The repr text, and the order `fields()` and `astuple()`
    report to code outside the classes, still change.
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
                raise NeedsInput(f"`{member.qualname}` orders by field position", NeededInput.SOURCE_CHANGE)
            if {"__init__", "__match_args__"} & member.target.bound_names:
                raise NeedsInput(
                    f"`{member.qualname}` declares its own construction surface", NeededInput.PROOF
                )
        ClassReferenceSites(
            self.snapshot, frozenset(self.family)
        ).require_keyword_construction()
        require_no_positional_self_construction(self.snapshot, self.family)
        self.require_no_field_order_digest()
        self.snapshot.class_overlap_verdicts.field_order_reads.require_unread((self.composer.symbol,))

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
                    raise NeedsInput(
                        f"{owner.file_path}:{digest} digests `{owner.qualname}."
                        f"{method.name}`'s serialization of the instance, whose "
                        "field order the composition changes", NeededInput.SOURCE_CHANGE
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
            "dataclasses.asdict",
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
        raise NeedsInput(f"`{decorator_source}` is an unrewritable decorator", NeededInput.OPERATION)
    opened = decorator_source[:-1].rstrip()
    separator = "" if opened.endswith("(") else ", "
    return f"{opened}{separator}kw_only=True)"


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

    source: ReadSource
    call: ast.Call
    parameter_names: tuple[str, ...]

    @property
    def file_path(self) -> str:
        return self.source.file_path

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.source.line_of(self.call)}"

    def require_keywordable(self) -> None:
        call = self.call
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            raise NeedsInput(f"{self.location} unpacks `*` arguments", NeededInput.PROOF)
        if CallKeywords.of(call.keywords).splats:
            raise NeedsInput(f"{self.location} unpacks `**` arguments", NeededInput.PROOF)
        if len(call.args) > len(self.parameter_names):
            raise NeedsInput(
                f"{self.location} passes {len(call.args)} positional argument(s); "
                f"the generated `__init__` takes {len(self.parameter_names)}", NeededInput.PROOF
            )
        if any(isinstance(argument, ast.GeneratorExp) for argument in call.args):
            raise NeedsInput(f"{self.location} passes a bare generator expression", NeededInput.OPERATION)

    def replacements(self) -> tuple[SourceTextSpanReplacement, ...]:
        """`name=` before each positional argument's first token; a walrus the
        call left bare is parenthesized. Offsets are the holding file's."""

        call = self.call
        geometry = SourceTextGeometry(self.source.parsed_module.source)
        placed = self.source.file_offset
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
                    start_offset=placed(start),
                    end_offset=placed(start),
                    replacement_source=f"{name}=" + ("(" if bare_walrus else ""),
                )
            )
            if bare_walrus:
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=placed(argument_end),
                        end_offset=placed(argument_end),
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
    names. The class and every such descendant need known construction
    parameters (`CompactClassFamilyIndex.construction_parameters`); any construction that unpacks
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

    @cached_property
    def constructions(self) -> tuple[PositionalConstruction, ...]:
        parameters = proved_construction_parameters(self.snapshot, self.target.symbol)
        direct = tuple(
            PositionalConstruction(site.source, site.parent, parameters)
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
            if proved_construction_parameters(self.snapshot, symbol)[:widest] != parameters[:widest]:
                raise NeedsInput(
                    f"`{symbol}` takes other positional parameters than "
                    f"`{self.target.qualname}`, and its receiver may be one", NeededInput.PROOF
                )
        return tuple(
            PositionalConstruction(
                ModuleSource(self.snapshot.parsed_module_for_source_path(self.target.file_path)),
                call,
                parameters,
            )
            for call in calls
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
                    for replacement in construction.replacements()
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
            raise NeedsInput(
                f"line {function.lineno}: `{function.name}` rebinds {sorted(rebound)}, "
                "so its own-class calls are unproved", NeededInput.PROOF
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


class RelocationStep:
    """An operation one of whose steps relocates declarations: the files that
    relocation creates, a new module and any package of it, are created first,
    so every step sees them and the edits fill them as any existing module; the
    files outside the scan it rewrites are rewritten with them."""

    def relocation(
        self, snapshot: CodemodSourceSnapshot
    ) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation | None:
        raise NotImplementedError

    def source_file_creations(self, context: CodemodSelectorContext) -> tuple[SourceFileCreation, ...]:
        relocation = self.relocation(context.execution_snapshot())
        return () if relocation is None else relocation.source_file_creations(context)

    def relocation_state(self, snapshot: CodemodSourceSnapshot) -> CodemodSourceSnapshot:
        """The source the relocation runs on: the operation's own, as it is the
        first step, unless a step of the operation comes before it."""

        return snapshot

    def text_file_changes(self, context: CodemodSelectorContext) -> tuple[TextFileChange, ...]:
        """What the relocation keeps true outside the scanned source, read on
        the source it runs on: a manifest naming a moved name, a dependent
        checkout importing one."""

        snapshot = context.execution_snapshot()
        relocation = self.relocation(snapshot)
        return () if relocation is None else relocation.text_file_changes(self.relocation_state(snapshot))


@dataclass(frozen=True, kw_only=True)
class ComposeMovedOwnerOperation(RelocationStep, RepositorySourceReprovedOperation):
    """Move an existing owner to `home_module`, then make the target and each other
    composer compose it there: composing it where it is closes an import cycle.

    The move is the module-move operation's, which redirects every importer of
    the owner and refuses what it cannot carry; each composition is then proved
    on the source the move left."""

    owner: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    home_module: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    """A module every composer already imports and that imports none of them, or
    a new module of their lowest common package."""
    composers: tuple[SourceRewriteTarget, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(SourceRewriteTarget), default=()
    )

    def relocation(
        self, snapshot: CodemodSourceSnapshot
    ) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation:
        return relocation_to(
            snapshot, self.owner, self.home_module, MovedClosure.of(snapshot, self.owner).qualnames
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        move = self.relocation(snapshot)
        try:
            RefactorRecipe(recipe_id="placement").with_operation(move).simulate(
                snapshot
            ).preflight_report.require_clean()
        except ValueError as error:
            raise NeedsInput(
                f"`{self.owner.qualname}` moves to `{self.home_module}` so composing it "
                f"closes no import cycle, and that move refuses: {error}",
                *NeedsInput.needs_of(error),
            ) from error
        moved = SourceRewriteTarget(file_path=move.destination_path, qualname=self.owner.qualname)
        owner = OverlapClass.require(snapshot, self.owner)
        names = tuple(declaration.name for declaration in owner.facts.body_declarations)
        rationale = self.rationale_text(
            f"Move `{self.owner.qualname}` to `{self.home_module}` and compose it there "
            "instead of restating its body."
        )
        for composer in (self.target, *self.composers):
            restated = RestatedDeclarations(owner, OverlapClass.require(snapshot, composer), names)
            rationale = KeptDocumentation.composing(restated).rationale(rationale)
        return in_order_edits(
            snapshot,
            (
                move,
                *(
                    ComposeExistingClassOperation(target=composer, owner=moved)
                    for composer in (self.target, *self.composers)
                ),
            ),
            f"compose-moved-{self.owner.qualname}",
            rationale,
        )

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        return ComposeExistingClassOperation(
            target=self.target, owner=self.owner
        ).current_source_authority_claims(context)


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
    owner_kind: type[OwnerCandidate],
    subject: tuple[str, ...],
    member_names: Iterable[str],
    *,
    tolerate_inherited: bool = False,
) -> tuple[ClassOverlapCandidate, OwnerCandidate]:
    """An extract-mixin verdict over the anchor whose block holds these names and
    that offers an owner of this kind about this subject, with no guard fired on
    the verdict or the owner. A reported verdict answers first; otherwise the
    verdict requested on the anchor and a class declaring every name itself, as
    a step of a larger move asks below the reporting floor.

    The anchor can restate the block with several classes. Any one proved
    verdict answers: the owner is declared from the anchor's own declarations,
    and each composer's composition is proved by its own step. Only when every
    such verdict refuses does the first refusal stand.

    With ``tolerate_inherited``, INHERITED_DECLARATION receipts do not refuse:
    the caller filters the block to the declarations both classes own."""

    wanted = frozenset(member_names)
    verdicts = snapshot.class_overlap_verdicts
    refused: NeedsInput | None = None
    for verdict in chain(
        verdicts.candidates_touching(frozenset((anchor_symbol,))),
        verdicts.requested_extractions_restating(anchor_symbol, sorted(wanted)),
    ):
        if verdict.move is not ExtractMixin or not wanted <= frozenset(verdict.shared_names):
            continue
        for owner in verdict.owner_candidates:
            if (
                isinstance(owner, owner_kind)
                and owner.subject == subject
                and anchor_symbol in owner.composers
            ):
                blocking = tuple(
                    receipt
                    for receipt in verdict.refusals
                    if not (
                        tolerate_inherited
                        and receipt.kind is OverlapReceiptKind.INHERITED_DECLARATION
                    )
                )
                if blocking:
                    refused = refused or NeedsInput(
                        "The move needs: "
                        + "; ".join(receipt.detail for receipt in blocking),
                        *(receipt.kind.needs for receipt in blocking),
                    )
                elif not owner.permitted:
                    refused = refused or NeedsInput(
                        f"`{owner.candidate_id}` needs: "
                        + "; ".join(receipt.detail for receipt in owner.refusals),
                        *(receipt.kind.needs for receipt in owner.refusals),
                    )
                else:
                    return verdict, owner
    if refused is not None:
        raise refused
    raise ValueError(
        f"The current source offers no extract-mixin {owner_kind.__name__} "
        f"`{'+'.join(subject)}` for "
        f"`{anchor_symbol}` over {sorted(wanted)}"
    )


def movable_members(
    context: ClassMemberMoveProofContext,
    source: OverlapClass,
    statements: tuple[ast.stmt, ...],
) -> None:
    """Each statement is a member that moves without reading its class's scope."""

    for statement in statements:
        member = ClassMemberPromotionStatement.for_statement(statement)
        if member is None:
            raise NeedsInput(
                f"`{source.qualname}` line {statement.lineno} is no movable member", NeededInput.OPERATION
            )
        member.require_safe_move(context)


@dataclass(frozen=True)
class MixinCreation:
    """A new class declares a block several classes restate; nothing uses it yet.

    The class is a root (`[mixin, object]`), or, when the block declares abstract
    methods, an `ABC` (`[mixin, ABC, object]`), inserted right before the anchor in
    the anchor's module, so
    every free name of the copied statements is bound there exactly as in the
    anchor. Each statement must move without reading the anchor's class scope
    or learning its owner. Creating it changes no behaviour; each restating
    class then composes it through its own compose-existing verdict.

    Its dataclass declaration is the one every composer can inherit: each method
    the composers' decorators agree on generating, none they disagree on, their
    common frozenness, and no instance layout of its own. A block in which no
    composer declares a field needs no dataclass: the mixin is a plain class,
    which a dataclass of either frozenness and a plain class all compose, since
    a dataclass collects fields only from dataclass bases. When a composer is
    slotted, the mixin declares `__slots__ = ()`, so each composer slots the
    mixin's fields itself and none gains a `__dict__` or a second layout.

    `ABCMeta` gives a class the abstract names of its own namespace and those of
    each base's `__abstractmethods__` it still resolves to an abstract function.
    A plain root has none, so an abstract method moved there would stop being
    enforced; an `ABC` mixin carries it into each composer, whose
    `__abstractmethods__` then stay as they were. Every composer must derive from
    `ABC` already, so composing the mixin adds no ancestor and keeps its metaclass.
    """

    snapshot: CodemodSourceSnapshot
    anchor: OverlapClass
    restated: RestatedDeclarations
    name: str
    docstring: str
    keyword_only: bool
    """Created keyword-only when the anchor is, when composing it reorders some
    composer's fields, or when some composer is keyword-only already; each other
    composer then becomes keyword-only as it composes the mixin."""
    options: tuple[str, ...]
    """The mixin's `dataclass` arguments besides `kw_only`, as `name=value`."""
    empty_slots: bool
    """Some composer is slotted, so the mixin declares `__slots__ = ()`."""
    holds_fields: bool
    """Some composer declares a field in the block, so the mixin is a dataclass
    carrying the anchor's decorator; otherwise it is a plain class."""
    documentation: KeptDocumentation
    """The docstrings the mixin's one copy keeps, judged across every composer."""
    abstract_names: tuple[str, ...] = ()
    """The block's abstract methods: when there are any, the mixin is an `ABC`."""

    @classmethod
    def require(
        cls,
        snapshot: CodemodSourceSnapshot,
        anchor_reference: SourceRewriteTarget,
        member_names: tuple[str, ...],
        name: str,
        docstring: str,
        home: tuple[str, ...] | None = None,
    ) -> Self:
        """`home` is the subject of the new-mixin candidate that proves the
        creation, when the mixin is declared away from the anchor's module."""

        anchor = OverlapClass.require(snapshot, anchor_reference)
        module_name = snapshot.parsed_module_for_source_path(anchor.file_path).module_name
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            snapshot.parsed_module_for_source_path(anchor.file_path).module.body
        )
        NeedsInput.every(
            lambda: cls.require_pickable(name, docstring, bound, anchor.file_path),
            lambda: anchor.dataclass_options_key,
        )
        _, owner = require_owner_candidate(
            snapshot, anchor.symbol, NewMixinOwner, home or (module_name,), member_names,
            tolerate_inherited=True,
        )
        restated = RestatedDeclarations.require(snapshot, anchor, anchor, member_names)
        NeedsInput.every(
            lambda: movable_members(
                ClassMemberMoveProofContext.between(snapshot, anchor.target, anchor.target).into_new_abc(),
                anchor,
                restated.statements,
            ),
            lambda: restated.require_no_owner_bound_values(snapshot),
        )
        abstract_names = tuple(
            name for name in anchor.facts.abstract_method_names if name in restated.names
        )
        mixin_fields = tuple(
            name for name in anchor.own_field_names if name in restated.names
        )
        # A composer keeps its fields only if they stay in order and it matches the
        # mixin's keyword-only-ness: one already keyword-only, made so by an earlier
        # step, needs a keyword-only mixin just as a reordered one does.
        composers = tuple(OverlapClass.for_symbol(snapshot, symbol) for symbol in owner.composers)
        if abstract_names:
            NeedsInput.every(
                lambda: cls.abc_import(snapshot, anchor),
                *(
                    lambda composer=composer: cls.require_abstract_composer(snapshot, composer, abstract_names)
                    for composer in (anchor, *composers)
                ),
            )
        holds_fields = any(
            frozenset(composer.own_field_names) & frozenset(restated.names)
            for composer in (anchor, *composers)
            if composer.dataclass_options_key
        )
        keyword_only = holds_fields and (
            anchor.option("kw_only") == "True"
            or any(
                before != after or composer.option("kw_only") == "True"
                for composer in composers
                for before, after in (composed_field_order(snapshot, mixin_fields, composer),)
            )
        )
        declared = tuple(
            (name_, role.mixin_value(name_, frozenset(composer.option(name_) for composer in (anchor, *composers))))
            for name_, role in DATACLASS_OPTION_ROLES.items()
        ) if holds_fields else ()
        return cls(
            snapshot, anchor, restated, name, docstring, keyword_only,
            tuple(
                f"{name_}={value}" for name_, value in declared if value != DATACLASS_OPTION_DEFAULTS[name_]
            ),
            any(not composer.lays_out_instance_dict for composer in (anchor, *composers)),
            holds_fields,
            KeptDocumentation((anchor, *composers), restated.deleted_names),
            abstract_names,
        )

    @staticmethod
    def require_abstract_composer(
        snapshot: CodemodSourceSnapshot, composer: OverlapClass, abstract_names: tuple[str, ...]
    ) -> None:
        """The composer derives from `ABC`, so `ABCMeta` or a subclass creates it and
        composing an `ABC` mixin adds no ancestor besides the mixin."""

        if not any(
            ".".join(reference.resolved_parts) == native_identity(ABC)
            for indexed in require_static_c3(snapshot, composer.facts)
            for reference in indexed.base_references
        ):
            raise NeedsInput(
                f"`{composer.qualname}` does not derive from `abc.ABC`: a mixin taking abstract "
                f"{list(abstract_names)} is an `ABC`, which composing would add to "
                f"`{composer.qualname}`'s ancestors. NRA needs `{composer.qualname}` to "
                "derive from `ABC` first",
                NeededInput.SOURCE_CHANGE,
            )

    @staticmethod
    def abc_import(snapshot: CodemodSourceSnapshot, anchor: OverlapClass) -> tuple[str, ...]:
        """The import binding `ABC` where the mixin is declared: none when the
        anchor's module binds it to `abc.ABC` there already; refuses another binding."""

        witness = snapshot.module_binding_proof.reference_or_builtin_witness_at(
            snapshot.parsed_module_for_source_path(anchor.file_path),
            ast.Name(id=ABC.__name__, ctx=ast.Load()),
            line=anchor.node.lineno,
        )
        if witness is None:
            return (f"from {ABC.__module__} import {ABC.__name__}",)
        if witness.qualified_name != native_identity(ABC):
            raise NeedsInput(
                f"`{ABC.__name__}` is `{witness.qualified_name}` in {anchor.file_path}, so the mixin "
                f"cannot name `{native_identity(ABC)}` as its base: NRA needs that "
                "binding renamed",
                NeededInput.SOURCE_CHANGE,
            )
        return ()

    @staticmethod
    def require_pickable(name: str, docstring: str, bound: frozenset[str], file_path: str) -> None:
        """The pick's class name and docstring can be written where the mixin goes."""

        def one_line_docstring() -> None:
            if not docstring.strip() or '"""' in docstring or "\\" in docstring:
                raise NeedsInput("A mixin needs a one-line docstring without quotes", NeededInput.OWNER_PICK)

        def class_name() -> None:
            if not name.isidentifier() or keyword_module.iskeyword(name):
                raise NeedsInput(f"`{name}` is no class name", NeededInput.OWNER_PICK)

        def unbound_name() -> None:
            if name in bound:
                raise NeedsInput(f"`{name}` is already bound in {file_path}", NeededInput.OWNER_PICK)

        NeedsInput.every(one_line_docstring, class_name, unbound_name)

    def decorator_source(self, source: str, decorator: ast.expr) -> str:
        function = decorator.func if isinstance(decorator, ast.Call) else decorator
        options = (*self.options, *(("kw_only=True",) if self.keyword_only else ()))
        named = ast.get_source_segment(source, function) or ""
        return f"{named}({', '.join(options)})" if options else named

    @property
    def bases_source(self) -> str:
        return f"({ABC.__name__})" if self.abstract_names else ""

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
        ) if self.holds_fields else ()
        members: list[str] = []
        for member, function in self.restated.member_sources(self.snapshot, member_indentation, self.documentation):
            if function and members:
                members.append("\n")
            members.append(member if member.endswith("\n") else member + "\n")
        return (
            SourceInsertion(
                file_path=anchor.file_path,
                insertion_line=first_line,
                inserted_lines=(
                    *decorators,
                    f"{indentation}class {self.name}{self.bases_source}:\n",
                    f'{member_indentation}"""{self.docstring}"""\n\n',
                    *((f"{member_indentation}__slots__ = ()\n\n",) if self.empty_slots else ()),
                    *members,
                    "\n\n",
                ),
                rationale=rationale,
            ),
            *(
                ModuleImportMutation.from_source(
                    file_path=anchor.file_path, import_source=import_source, rationale=rationale
                )
                for import_source in (self.abc_import(self.snapshot, anchor) if self.abstract_names else ())
            ),
        )


def home_subject(snapshot: CodemodSourceSnapshot, home_module: str) -> tuple[str, ...]:
    """The subject of the new-mixin candidate placed in `home_module`: the module
    when it exists, else the package a new module of that name is created in."""

    if snapshot.module_import_graph.unique_source_file_for_module_name(home_module):
        return (home_module,)
    return (home_module.rpartition(".")[0],)


@dataclass(frozen=True)
class ClosureRead:
    """One name a moved declaration reads, and the scanned module declaring it."""

    declaration: str
    name: str
    module: str


@dataclass(frozen=True)
class MovedClosure:
    """Module-level declarations with every declaration of their module they
    read, which move with them, and each name those declarations read from
    another scanned module: the modules their new home must import without a
    cycle, and which declaration needs each. The roots are the declarations
    the move is for, which name a new home; the rest go with them."""

    roots: tuple[str, ...]
    qualnames: tuple[str, ...]
    read_names: tuple[ClosureRead, ...]

    @property
    def reads(self) -> frozenset[str]:
        return frozenset(read.module for read in self.read_names)

    @classmethod
    def of(cls, snapshot: CodemodSourceSnapshot, declaration: SourceRewriteTarget) -> Self:
        if declaration.file_path is None or declaration.qualname is None:
            raise NeedsInput("a moved class needs its module and name", NeededInput.OPERATION)
        return cls.of_roots(snapshot, declaration.file_path, (declaration.qualname,))

    @classmethod
    def of_roots(cls, snapshot: CodemodSourceSnapshot, file_path: str, roots: tuple[str, ...]) -> Self:
        module = snapshot.module_nodes_by_file_path[file_path]
        selection = SourceTopLevelSymbolMoveSelection.dependency_closure(file_path, module, roots)
        moved = frozenset(item.name for item in selection.declarations)
        module_name = snapshot.parsed_module_for_source_path(file_path).module_name
        bound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(module.body)
        # A name the module does not bind is a builtin: no module provides it.
        return cls(
            roots,
            tuple(item.name for item in selection.declarations),
            tuple(
                ClosureRead(item.name, name, owner)
                for item in selection.declarations
                for name in sorted(
                    (DeclarationDependencyProjection.from_declarations((item.node,)).names & bound) - moved
                )
                if (owner := ImportedNameOwner.declaring_module(snapshot, module_name, name)) is not None
            ),
        )


def annotation_modes(snapshot: CodemodSourceSnapshot) -> dict[str, type[ModuleAnnotationEvaluationMode]]:
    """How each scanned module evaluates its annotations, by module name."""

    graph = snapshot.module_import_graph
    return {
        module_name: ModuleAnnotationEvaluationMode.from_module(module)
        for file_path, module in snapshot.module_nodes_by_file_path.items()
        if (module_name := graph.module_name_for_file_path(file_path)) is not None
    }


def relocation_to(
    snapshot: CodemodSourceSnapshot,
    declaration: SourceRewriteTarget,
    home_module: str,
    moved: tuple[str, ...],
) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation:
    """The module move that declares the module-level class `declaration` names,
    with the rest of `moved`, the declarations of its module that go with it, in
    `home_module` instead: into that module when it exists, else into a new module
    of that name beside its package's `__init__`."""

    qualname = declaration.qualname
    if declaration.file_path is None or qualname is None or "." in qualname:
        raise NeedsInput(
            f"`{qualname}` is no module-level class to move to `{home_module}`",
            NeededInput.OPERATION,
        )
    graph = snapshot.module_import_graph
    source = SourceRewriteTarget(file_path=declaration.file_path)
    existing = graph.unique_source_file_for_module_name(home_module)
    if existing is not None:
        return RelocateSymbolsToModuleOperation(
            target=source, symbol_qualnames=moved, destination_path=existing.file_path
        )
    package, _, stem = home_module.rpartition(".")
    package_file = graph.unique_source_file_for_module_name(package) if package else None
    if package_file is None or Path(package_file.file_path).name != "__init__.py":
        raise NeedsInput(
            f"`{home_module}` would be a new module, but `{package}` is no package "
            "with an `__init__.py` to create it beside", NeededInput.SOURCE_CHANGE,
        )
    return RelocateSymbolsToNewModuleOperation(
        target=source,
        symbol_qualnames=moved,
        destination_path=(Path(package_file.file_path).parent / f"{stem}.py").as_posix(),
    )


def in_order_edits(
    snapshot: CodemodSourceSnapshot,
    steps: Sequence[RepositorySourceReprovedOperation],
    recipe_id: str,
    rationale: str,
) -> tuple[NominalSourceEdit, ...]:
    """The edits of these steps run in order, each proved on the source the
    previous one left; any refusal refuses them all."""

    return changed_source_edits(snapshot, in_order_states(snapshot, steps, recipe_id)[-1], rationale)


def in_order_states(
    snapshot: CodemodSourceSnapshot,
    steps: Sequence[RepositorySourceReprovedOperation],
    recipe_id: str,
) -> tuple[CodemodSourceSnapshot, ...]:
    """The source before each step and after the last, each step proved on the
    source the previous one left; any refusal refuses them all."""

    states = [snapshot]
    for step in steps:
        result = RefactorRecipe(recipe_id=recipe_id).with_operation(step).simulate(states[-1])
        result.preflight_report.require_clean()
        states.append(states[-1].with_virtual_sources(result.simulation.rewritten_sources))
    return tuple(states)


def changed_source_edits(
    snapshot: CodemodSourceSnapshot, current: CodemodSourceSnapshot, rationale: str
) -> tuple[NominalSourceEdit, ...]:
    """The line edits that take each file of `snapshot` to its source in `current`.
    A file only `current` holds was created by a step without being declared,
    and its source would be lost: that refuses."""

    created = sorted(set(current.sources_by_file_path) - set(snapshot.sources_by_file_path))
    if created:
        raise NeedsInput(
            "a step creates " + ", ".join(f"`{path}`" for path in created)
            + " that the operation does not declare, so its source would be lost",
            NeededInput.OPERATION,
        )
    return tuple(
        edit
        for file_path, source in snapshot.sources_by_file_path.items()
        if current.sources_by_file_path.get(file_path, source) != source
        for edit in _line_difference_edits(
            file_path, source, current.sources_by_file_path[file_path], rationale
        )
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
            snapshot, anchor.symbol, ExistingClassOwner, (owner.symbol,), member_names
        )
        if frozenset(member_names) & owner.target.bound_names:
            raise NeedsInput(f"`{owner.qualname}` already binds an added name", NeededInput.SOURCE_CHANGE)
        dunders = sorted(name for name in member_names if name.startswith("__"))
        if dunders:
            raise NeedsInput(f"Adding {dunders} could shadow a native declaration", NeededInput.PROOF)
        if owner.dataclass_options_key != anchor.dataclass_options_key:
            raise NeedsInput(
                f"`{owner.qualname}` and `{anchor.qualname}` differ as dataclasses",
                NeededInput.SOURCE_CHANGE,
            )
        restated = RestatedDeclarations.require(snapshot, anchor, anchor, member_names)
        movable_members(
            ClassMemberMoveProofContext.between(snapshot, anchor.target, owner.target),
            anchor,
            restated.statements,
        )
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
                raise NeedsInput(
                    f"{built[0].location} builds `{ast.unparse(built[0].node)}`, which "
                    f"would need the new required field(s) {list(added_required)}", NeededInput.SOURCE_CHANGE
                )
        sites.require_keyword_construction()
        require_no_positional_self_construction(self.snapshot, self.family)

    def require_lookup(self) -> None:
        class_index = self.snapshot.class_overlap_verdicts.class_index
        authority = self.snapshot.class_overlap_verdicts.class_index.mro_authority
        for symbol in self.family:
            mro_type = authority.resolve(symbol).mro_type
            if mro_type is None:
                raise NeedsInput(f"`{symbol}` has an open MRO", NeededInput.PROOF)
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
                    raise NeedsInput(
                        f"`{self.owner.qualname}.{name}` would replace `{declaring[0]}`'s "
                        f"answer for `{symbol}`", NeededInput.SOURCE_CHANGE
                    )

    def source_edits(self, rationale: str) -> tuple[NominalSourceEdit, ...]:
        owner = self.owner
        source = self.snapshot.sources_by_file_path[owner.file_path]
        body = ClassBodySourceAuthority(owner.node, source)
        # The anchor's is the only copy, so it keeps every docstring it carries.
        documentation = KeptDocumentation((self.anchor,), self.restated.deleted_names)
        return (
            *body.geometry.physical_edits(
                file_path=owner.file_path,
                replacements=(
                    body.declaration_block_replacement(
                        self.restated.member_sources(self.snapshot, body.indentation, documentation)
                    ),
                ),
                rationale=rationale,
            ),
            *required_name_imports(
                self.snapshot, self.restated.statements, self.anchor, owner, rationale
            ),
        )


@dataclass(frozen=True, kw_only=True)
class CreateMixinOperation(RelocationStep, RepositorySourceReprovedOperation):
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
    home_module: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(), default=None
    )
    """The module the mixin is declared in when declaring it beside the anchor
    closes an import cycle: an existing module, or a new one in its package. The
    mixin is created beside the anchor, moved there, then composed."""

    def creation(self, snapshot: CodemodSourceSnapshot) -> MixinCreation:
        return MixinCreation.require(
            snapshot, self.target, self.member_names, self.mixin_name, self.docstring,
            None if self.home_module is None else home_subject(snapshot, self.home_module),
        )

    def declared(self, creation: MixinCreation) -> SourceRewriteTarget:
        """The mixin where the first step declares it, beside the anchor."""

        qualname = ".".join((*creation.anchor.qualname.split(".")[:-1], self.mixin_name))
        return SourceRewriteTarget(file_path=self.target.file_path, qualname=qualname)

    def relocation(
        self, snapshot: CodemodSourceSnapshot
    ) -> RelocateSymbolsToModuleOperation | RelocateSymbolsToNewModuleOperation | None:
        """A new mixin moves alone: whether the names it reads can be bound there
        was settled when the candidate was placed. The first step alone, which
        has no composers, declares it beside the anchor and moves nothing."""

        if self.home_module is None or not self.composers:
            return None
        mixin = self.declared(self.creation(snapshot))
        return relocation_to(snapshot, mixin, self.home_module, (mixin.qualname,))

    def relocation_state(self, snapshot: CodemodSourceSnapshot) -> CodemodSourceSnapshot:
        """The source the first step leaves, which declares the mixin the
        relocation moves."""

        return in_order_states(snapshot, (self.first_step,), f"create-{self.mixin_name}")[-1]

    @property
    def first_step(self) -> CreateMixinOperation:
        """The mixin declared beside its anchor, composed by nobody yet."""

        return replace(self, composers=())

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        rationale = self.rationale_text(
            f"Declare {', '.join(self.member_names)} once on `{self.mixin_name}`."
        )
        creation = self.creation(snapshot)
        rationale = creation.documentation.rationale(rationale)
        if not self.composers:
            # The first step alone: declared beside the anchor, proved by the
            # candidate for wherever it is going.
            return creation.source_edits(rationale)
        mixin = self.declared(creation)
        move = self.relocation(snapshot)
        moves: tuple[RepositorySourceReprovedOperation, ...] = ()
        if move is not None:
            moves = (move,)
            mixin = SourceRewriteTarget(file_path=move.destination_path, qualname=mixin.qualname)
        return in_order_edits(
            snapshot,
            (
                self.first_step,
                *moves,
                *(
                    ComposeExistingClassOperation(target=composer, owner=mixin)
                    for composer in self.composers
                ),
            ),
            f"create-{self.mixin_name}",
            rationale,
        )

    def keyword_cure(self, snapshot: CodemodSourceSnapshot) -> tuple[str, ...]:
        """The composers and descendants a call builds by position, among those
        composing the keyword-only mixin makes keyword-only: each is bound by
        keyword at an earlier stage. A family that hands its class on as a value
        refuses here, before any rewrite."""

        if not self.creation(snapshot).keyword_only:
            return ()
        verdicts = snapshot.class_overlap_verdicts
        family = tuple(
            dict.fromkeys(
                symbol
                for target in self.composers
                if (composer := OverlapClass.require(snapshot, target)).option("kw_only") != "True"
                for symbol in (composer.symbol, *verdicts.class_index.descendant_symbols(composer.symbol))
            )
        )
        cured = tuple(
            symbol
            for symbol in family
            if (indexed_class := verdicts.class_index.class_for(symbol)) is not None
            and verdicts.uses.constructions_of(indexed_class, positional=True)
        )
        if cured:
            ClassReferenceSites(snapshot, frozenset(family)).require_no_value_hand_on()
        return cured

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
class CreateMixinsInOrderOperation(RepositorySourceReprovedOperation):
    """Create several mixins as one step, each on the source the one before it
    left: a block one class inherits is hoisted out of its base before the rest
    of the block is extracted."""

    steps: tuple[CreateMixinOperation, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(CreateMixinOperation)
    )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        states = in_order_states(
            snapshot, self.steps, "create-" + "-".join(step.mixin_name for step in self.steps)
        )
        rationale = self.rationale_text(
            "Declare "
            + "; ".join(
                f"{', '.join(step.member_names)} once on `{step.mixin_name}`"
                for step in self.steps
            )
            + "."
        )
        for step, state in zip(self.steps, states):
            rationale = step.creation(state).documentation.rationale(rationale)
        return changed_source_edits(snapshot, states[-1], rationale)

    def current_source_authority_claims(
        self, context: CodemodSelectorContext
    ) -> tuple[AuthorityClaim, ...]:
        return self.steps[0].current_source_authority_claims(context)


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
    FindingRecipeSynthesizer,
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
        snapshot: CodemodSourceSnapshot,
    ) -> tuple[RepositorySourceReprovedOperation, ...]:
        """The operations the picker's owner calls for; `NeedsInput` names what a
        pick still needs when there are none.

        A new mixin is declared before a composing class in its module; an
        existing class first gains the block declarations it lacks; a shared
        base takes the block as a fold. Each restating class then composes the
        owner through its own compose-existing verdict at the next stage. A
        verdict covered by a pick on another finding adds nothing itself.
        """

        pick = metrics.owner_pick
        if pick is None:
            if not any(
                owner.permitted or owner.keyword_cure for owner in metrics.owner_candidates
            ):
                blocking = tuple(
                    receipt for owner in metrics.owner_candidates for receipt in owner.refusals
                )
                raise NeedsInput(
                    "every owner candidate needs: "
                    + "; ".join(dict.fromkeys(receipt.detail for receipt in blocking)),
                    *(receipt.kind.needs for receipt in blocking),
                )
            raise NeedsInput(
                "an extract-mixin move needs an owner pick (--codemod-owner-picks)",
                NeededInput.OWNER_PICK,
            )
        owner = pick.candidate
        verdicts = snapshot.class_overlap_verdicts
        # Refused before any stage: keywording constructions for a move that
        # cannot land would only churn the source.
        snapshot.class_overlap_verdicts.field_order_reads.require_unread(owner.reordered)
        if owner.keyword_cure:
            return ClassOverlapFindingRecipeSynthesizer.keywordings(owner.keyword_cure, verdicts)
        composers = tuple(
            indexed_class
            for symbol in owner.composers
            if (indexed_class := verdicts.class_index.class_for(symbol)) is not None
        )
        match owner:
            case HoistingMixinOwner():
                base = verdicts.class_index.class_for(owner.base)
                restater = verdicts.class_index.class_for(owner.restater)
                if base is None or restater is None:
                    raise NeedsInput("the hoisting base left the source", NeededInput.OWNER_PICK)
                if pick.hoisted_name is None or pick.hoisted_judgement is None:
                    raise NeedsInput("a hoist needs the pick's hoisted_name and hoisted_judgement", NeededInput.OWNER_PICK)
                hoist = CreateMixinOperation(
                    target=source_rewrite_target(base),
                    member_names=owner.hoisted,
                    mixin_name=pick.hoisted_name,
                    docstring=pick.hoisted_judgement[:1].upper() + pick.hoisted_judgement[1:].rstrip(".") + ".",
                    composers=tuple(
                        source_rewrite_target(composer)
                        for symbol in owner.hoist_composers
                        if (composer := verdicts.class_index.class_for(symbol)) is not None
                    ),
                )
                rest = ClassOverlapFindingRecipeSynthesizer.new_mixin(owner, pick, composers, verdicts)
                return (
                    CreateMixinsInOrderOperation(
                        target=hoist.target, steps=(hoist, *rest)
                    ),
                )
            case NewMixinOwner():
                creations = ClassOverlapFindingRecipeSynthesizer.new_mixin(owner, pick, composers, verdicts)
                cure = tuple(
                    dict.fromkeys(symbol for creation in creations for symbol in creation.keyword_cure(snapshot))
                )
                if cure:
                    return ClassOverlapFindingRecipeSynthesizer.keywordings(cure, verdicts)
                return creations
            case ExistingClassOwner():
                existing = verdicts.class_index.class_for(owner.symbol)
                missing = tuple(name for name in owner.block if name not in owner.held)
                if existing is None or not composers:
                    raise NeedsInput("the picked owner left the source", NeededInput.OWNER_PICK)
                if not missing:
                    raise NeedsInput(f"`{owner.symbol}` already declares the whole block", NeededInput.OWNER_PICK)
                return (
                    ExtendOwnerOperation(
                        target=source_rewrite_target(existing),
                        anchor=source_rewrite_target(composers[0]),
                        member_names=missing,
                    ),
                )
            case ComposeExistingOwner():
                owners = tuple(
                    indexed_class
                    for symbol in owner.symbols
                    if (indexed_class := verdicts.class_index.class_for(symbol)) is not None
                )
                if len(owners) != len(owner.symbols) or not composers:
                    raise NeedsInput("the picked owners left the source", NeededInput.OWNER_PICK)
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
                    raise NeedsInput("the picked base left the source", NeededInput.OWNER_PICK)
                return (
                    FoldSharedDeclarationsIntoBaseOperation(
                        target=source_rewrite_target(base),
                        member_names=owner.block,
                    ),
                )
            case HoldSourceOwner():
                source = verdicts.class_index.class_for(owner.symbol)
                if source is None or not composers:
                    raise NeedsInput("the picked source left the source", NeededInput.OWNER_PICK)
                if not (pick.name or "").isidentifier():
                    raise NeedsInput("a hold-source pick needs the held field's `name`", NeededInput.OWNER_PICK)
                return tuple(
                    ReplaceFieldsWithCarrierOperation(
                        target=source_rewrite_target(copier),
                        field_projections=tuple(
                            CarrierFieldProjection(source_field=name, carrier_attribute=name)
                            for name in owner.block
                        ),
                        carrier_field_declaration=f"{pick.name}: {source.qualname}",
                    )
                    for copier in composers
                )
        raise NeedsInput(f"`{owner.kind}` owners have no move", NeededInput.OPERATION)

    @staticmethod
    def new_mixin(
        owner: NewMixinOwner,
        pick: PickedOwner,
        composers: tuple[CompactIndexedClass, ...],
        verdicts: ClassOverlapVerdicts,
    ) -> tuple[CreateMixinOperation, ...]:
        """The new mixin of the names both composers declare themselves, declared
        before the anchor in its module, then moved to the owner's module when
        that is another one, a new module included."""

        anchor = next(
            (indexed_class for indexed_class in composers if indexed_class.symbol == owner.anchor),
            None,
        )
        if anchor is None or pick.name is None:
            raise NeedsInput(f"the anchor `{owner.anchor}` left the source", NeededInput.OWNER_PICK)
        home_module = (
            f"{owner.module_name}.{snake_case(pick.name)}"
            if isinstance(owner, NewModuleMixinOwner)
            else None
            if owner.module_name == anchor.module_name
            else owner.module_name
        )
        if isinstance(owner, NewModuleMixinOwner) and home_module in verdicts.module_imports.imports:
            raise NeedsInput(
                f"`{home_module}` already exists and does not avoid the cycle: the new "
                "module needs another mixin name", NeededInput.OWNER_PICK,
            )
        if pick.judgement is None:
            raise NeedsInput("a new mixin needs the pick's judgement for its docstring", NeededInput.OWNER_PICK)
        pair = frozenset(owner.composers[:2])
        verdict_match = next(
            (
                v for v in verdicts.candidates_touching(pair)
                if v.move is ExtractMixin
                and frozenset((v.owner.symbol, *v.restating_symbols)) == pair
                and frozenset(owner.block) <= frozenset(v.shared_names)
            ),
            None,
        )
        inherited = verdict_match.inherited_names if verdict_match else frozenset()
        block = tuple(
            name for name in owner.block if name not in inherited
        )
        if not block:
            raise NeedsInput("every shared declaration is inherited; nothing to extract", NeededInput.OWNER_PICK)
        return (
            CreateMixinOperation(
                target=source_rewrite_target(anchor),
                member_names=block,
                mixin_name=pick.name,
                docstring=pick.judgement[:1].upper() + pick.judgement[1:].rstrip(".") + ".",
                composers=tuple(
                    source_rewrite_target(indexed_class)
                    for indexed_class in composers
                ),
                home_module=home_module,
            ),
        )

    @staticmethod
    def owner_home(
        snapshot: CodemodSourceSnapshot,
        owner: CompactIndexedClass,
        composers: tuple[CompactIndexedClass, ...],
    ) -> str | None:
        """Where an existing owner moves before its composers compose it: None
        when none of them importing it where it is closes a cycle. Otherwise a
        module the owner's and every composer's module already import and that
        imports none of them; failing that, a new module of their lowest common
        package named for the owner. Refuses, naming why, when there is neither."""

        graph = snapshot.module_import_graph
        closing = tuple(
            composer.module_name
            for composer in composers
            if graph.import_would_create_cycle(
                importing_file_path=composer.file_path, imported_file_path=owner.file_path
            )
        )
        if not closing:
            return None
        users = tuple(dict.fromkeys((owner.module_name, *(c.module_name for c in composers))))
        imports = ModuleImportGraph(graph.import_edges_by_module)
        closure = MovedClosure.of(snapshot, source_rewrite_target(owner))
        lower = imports.shared_lower_module(
            users,
            closure.reads,
            homes_among_users=users[1:],
            modes=annotation_modes(snapshot),
            source=owner.module_name,
        )
        if lower is not None:
            return lower.module
        pinned = imports.pinning_reads(users, closure.reads)
        if pinned:
            raise NeedsInput(
                f"`{owner.module_name}` already imports, directly or not, "
                + ", ".join(f"`{module}`" for module in closing)
                + f", so `{owner.qualname}` moves first; but "
                + " and ".join(f"`{name}`" for name in closure.qualnames)
                + " read names from "
                + ", ".join(f"`{module}`" for module in pinned)
                + ", which no module below "
                + ", ".join(f"`{module}`" for module in users)
                + " can import without a cycle",
                NeededInput.SOURCE_CHANGE,
            )
        package = common_package(users)
        if not package:
            raise NeedsInput(
                f"`{owner.module_name}` already imports, directly or not, "
                + ", ".join(f"`{module}`" for module in closing)
                + f", so `{owner.qualname}` moves first; but no module that "
                + ", ".join(f"`{module}`" for module in users)
                + " all import avoids the cycle, and they share no package to hold "
                "a new one",
                NeededInput.SOURCE_CHANGE,
            )
        home_module = f"{package}.{snake_case(owner.qualname)}"
        if graph.unique_source_file_for_module_name(home_module) is not None:
            raise NeedsInput(
                f"`{owner.qualname}` moves first to avoid a cycle, but `{home_module}` "
                "already exists and does not avoid it", NeededInput.SOURCE_CHANGE,
            )
        return home_module

    @staticmethod
    def keywordings(
        symbols: tuple[str, ...], verdicts: ClassOverlapVerdicts
    ) -> tuple[RepositorySourceReprovedOperation, ...]:
        """The stage that clears a positional-construction guard: every construction
        of these classes bound by keyword. The move itself is proved at the next
        stage, on the source this one leaves, where the guard no longer fires."""

        classes = tuple(verdicts.class_index.class_for(symbol) for symbol in symbols)
        if None in classes:
            raise NeedsInput("a class built by position left the source", NeededInput.SOURCE_CHANGE)
        return tuple(
            KeywordizeConstructionOperation(target=source_rewrite_target(indexed_class))
            for indexed_class in classes
            if indexed_class is not None
        )

    def keywording_evaluation(
        self,
        finding: RefactorFinding,
        symbols: tuple[str, ...],
        verdicts: ClassOverlapVerdicts,
        context: CodemodSelectorContext,
    ) -> FindingRecipeEvaluation:
        try:
            operations = self.keywordings(symbols, verdicts)
        except NeedsInput as missing:
            return self.stopped_evaluation(missing)
        recipe = RefactorRecipe(
            recipe_id=f"{finding.stable_id}-keywordize",
            reason="Build the classes this move reorders by keyword, so the move can prove.",
        )
        for operation in operations:
            try:
                operation.source_edits(context)
            except CodemodOperationPreflightError as error:
                return self.stopped_evaluation(error)
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        metrics = finding.metrics
        if context is None or not isinstance(metrics, ClassOverlapMetrics):
            return self.rejected_evaluation("a class overlap needs its verdict and source")
        snapshot = context.execution_snapshot()
        verdicts = snapshot.class_overlap_verdicts
        if not metrics.permitted:
            if not metrics.keyword_cure:
                return self.rejected_evaluation(
                    "the move needs: "
                    + "; ".join(receipt.detail for receipt in metrics.refusals),
                    tuple(receipt.kind.needs for receipt in metrics.refusals),
                )
            return self.keywording_evaluation(finding, metrics.keyword_cure, verdicts, context)
        if metrics.move is ComposeExisting:
            owner = verdicts.class_index.class_for(metrics.owner_symbol)
            restating = tuple(
                verdicts.class_index.class_for(symbol) for symbol in metrics.restating_symbols
            )
            if owner is None or None in restating:
                return self.rejected_evaluation("the overlap's classes left the source")
            composers = tuple(
                indexed_class for indexed_class in restating if indexed_class is not None
            )
            try:
                home_module = self.owner_home(snapshot, owner, composers)
            except NeedsInput as missing:
                return self.stopped_evaluation(missing)
            operations: tuple[RepositorySourceReprovedOperation, ...] = (
                tuple(
                    ComposeExistingClassOperation(
                        target=source_rewrite_target(indexed_class),
                        owner=source_rewrite_target(owner),
                    )
                    for indexed_class in composers
                )
                if home_module is None
                else (
                    ComposeMovedOwnerOperation(
                        target=source_rewrite_target(composers[0]),
                        owner=source_rewrite_target(owner),
                        home_module=home_module,
                        composers=tuple(source_rewrite_target(item) for item in composers[1:]),
                    ),
                )
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
            pick = metrics.owner_pick
            if pick is not None and pick.finding != finding.stable_id:
                return self.covered_evaluation(f"the owner pick on {pick.finding}")
            try:
                operations = self.picked_operations(finding, metrics, snapshot)
            except NeedsInput as missing:
                return self.stopped_evaluation(missing)
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
                return self.stopped_evaluation(error)
            recipe = recipe.with_operation(operation)
        return self.executable_evaluation(recipe)
