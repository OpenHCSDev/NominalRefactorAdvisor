"""Declared identities and the strings that spell them again.

A class, a module and a family member each have one declared identity: the
class statement, the module file, and the class keyword or class constant whose
string is distinct across the member's siblings. A string literal equal to one
of them, in a position where the program uses it to pick, compare, import or
type something, is a second copy of that identity.

`DeclaredIdentityFamily` collects what each module declares; `IdentitySpellingFamily`
collects the literals in identity positions. Which literal spells which
identity is a scan-wide join (`DeclaredIdentities`), because the declaration and
the spelling sit in different modules.
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

from .ast_tools import ModuleSyntaxIndex, ParsedModule, module_syntax_index
from .quoted_reference import (
    DeclaredInterpreter,
    ModuleReferenceScope,
    ReferenceRewrite,
    SpelledModules,
)
from .models import SourceLocation
from .observation_families import ShapeFamily
from .restated_fact_observations import RestatingSite, RestatingSiteFamily


@dataclass(frozen=True)
class DeclaredClass:
    """One class statement: its name, bases, and the strings it declares about itself.

    `keywords` are the class keywords whose value is a string
    (`class A(Base, declared_name="a")`), and `consumed_keywords` the keywords its
    own `__init_subclass__` accepts: declaration options the class machinery
    consumes. `slots` are class-level names annotated `ClassVar[...]` with no
    value: a spelling every subclass declares. `constants` are class-level string
    constants (`kind = "a"`);
    `instance_attributes` are the names it declares as per-object state (an
    annotated field that is not a ClassVar, or `self.name = ...` in a method);
    `lookups` are its classmethods that return a member class of the family for
    an argument.
    """

    name: str
    line: int
    base_names: tuple[str, ...]
    keywords: tuple[tuple[str, str], ...]
    consumed_keywords: tuple[str, ...]
    slots: tuple[str, ...]
    constants: tuple[tuple[str, str], ...]
    instance_attributes: tuple[str, ...]
    lookups: tuple[str, ...]


@dataclass(frozen=True)
class ModuleDeclarations:
    """What one module declares: its import name and its classes."""

    module_name: str
    file_path: str
    classes: tuple[DeclaredClass, ...]


class DeclaredIdentityFamily(ShapeFamily[ModuleDeclarations]):
    item_type = ModuleDeclarations

    SUBCLASS_HOOK = "__init_subclass__"

    @classmethod
    def collect(cls, parsed_module: ParsedModule) -> list[ModuleDeclarations]:
        index = module_syntax_index(parsed_module.module)
        return [
            ModuleDeclarations(
                module_name=parsed_module.module_name,
                file_path=parsed_module.file_path,
                classes=tuple(
                    cls.declared_class(node)
                    for _node_index, node in index.indexed_nodes_of_type(ast.ClassDef)
                ),
            )
        ]

    @classmethod
    def declared_class(cls, node: ast.ClassDef) -> DeclaredClass:
        methods = [
            statement
            for statement in node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return DeclaredClass(
            name=node.name,
            line=node.lineno,
            base_names=tuple(
                name for base in node.bases if (name := cls.base_name(base))
            ),
            keywords=tuple(
                (keyword.arg, keyword.value.value)
                for keyword in node.keywords
                if keyword.arg is not None
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            consumed_keywords=tuple(
                argument.arg
                for method in methods
                if method.name == cls.SUBCLASS_HOOK
                for argument in (
                    *method.args.posonlyargs[1:],
                    *method.args.args[1:],
                    *method.args.kwonlyargs,
                )
            ),
            slots=tuple(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is None
                and ClassVar.__name__ in ast.unparse(statement.annotation)
            ),
            constants=tuple(cls.constants(node)),
            instance_attributes=tuple(
                sorted(set(cls.instance_attributes(node, methods)))
            ),
            lookups=tuple(
                method.name for method in methods if cls.is_lookup(method, node.name)
            ),
        )

    @classmethod
    def base_name(cls, base: ast.expr) -> str:
        """The class a base names: `Base`, `module.Base` and `Base[T]` all name `Base`."""

        match base:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
            case ast.Subscript(value=generic):
                return cls.base_name(generic)
        return ""

    @staticmethod
    def constants(node: ast.ClassDef) -> Iterator[tuple[str, str]]:
        for statement in node.body:
            match statement:
                case ast.Assign(
                    targets=[ast.Name(id=name)],
                    value=ast.Constant(value=str() as value),
                ) | ast.AnnAssign(
                    target=ast.Name(id=name), value=ast.Constant(value=str() as value)
                ):
                    yield name, value

    @staticmethod
    def instance_attributes(
        node: ast.ClassDef, methods: Sequence[ast.FunctionDef | ast.AsyncFunctionDef]
    ) -> Iterator[str]:
        for statement in node.body:
            match statement:
                case ast.AnnAssign(
                    target=ast.Name(id=name), annotation=annotation, value=None
                ):
                    if ClassVar.__name__ not in ast.unparse(annotation):
                        yield name
        for method in methods:
            parameters = [*method.args.posonlyargs, *method.args.args]
            if not parameters:
                continue
            own = parameters[0].arg
            for statement in method.body:
                for assigned in ast.walk(statement):
                    match assigned:
                        case ast.Attribute(
                            value=ast.Name(id=receiver), attr=name, ctx=ast.Store()
                        ) if (receiver == own):
                            yield name

    @staticmethod
    def is_lookup(function: ast.FunctionDef | ast.AsyncFunctionDef, owner: str) -> bool:
        """A classmethod taking an argument and returning `type[Self]` or `type[Owner]`."""

        decorated = any(
            isinstance(decorator, ast.Name) and decorator.id == classmethod.__name__
            for decorator in function.decorator_list
        )
        parameters = [*function.args.posonlyargs, *function.args.args]
        match function.returns:
            case ast.Subscript(
                value=ast.Name(id="type"),
                slice=ast.Name(id=returned) | ast.Constant(value=str() as returned),
            ):
                return (
                    decorated and len(parameters) >= 2 and returned in ("Self", owner)
                )
        return False


class IdentitySpelling(ABC):
    """How a literal in one kind of position spells an identity.

    Each position class declares its `tag`, which the site carries as its first
    detail so the join can find the position again; the class alone knows
    which declared identity its literal can spell.
    """

    tag: ClassVar[str]

    @classmethod
    def detail(cls, *values: str) -> tuple[str, ...]:
        return (cls.tag, *(str(value) for value in values))

    @classmethod
    def rewrite(cls, detail: Sequence[str]) -> ReferenceRewrite:
        """Where this site can be rewritten as its reference; most positions cannot."""

        return ReferenceRewrite.NONE

    @classmethod
    def spelling(cls, detail: Sequence[str]) -> str:
        """What the site writes: the literal, unless the position says otherwise."""

        return repr(detail[-1])

    @classmethod
    def positions(cls) -> dict[str, type[IdentitySpelling]]:
        return {position.tag: position for position in cls.__subclasses__()}

    @classmethod
    @abstractmethod
    def resolve(
        cls, site: RestatingSite, identities: DeclaredIdentities
    ) -> ResolvedIdentity | None:
        """The identity this site's literal spells, if the scan declares it."""

        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedIdentity:
    """A declared identity: what it is, and where it is declared."""

    identity: str
    declared_at: tuple[SourceLocation, ...]

    def restated_by(self, site: RestatingSite) -> RestatingSite:
        return replace(site, group=self.identity)


class TypeSpelling(IdentitySpelling):
    """A class name inside a quoted annotation, base, TypeVar bound or cast type."""

    tag = "type"

    @classmethod
    def rewrite(cls, detail):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[-1])


class LazyAnnotationSpelling(IdentitySpelling):
    """A class name quoted in an annotation that only an eager interpreter needs quoted.

    The name is not bound when the annotation would run eagerly (a later class,
    an import under `if TYPE_CHECKING:`), so the quote is required unless the
    project's interpreter evaluates annotations lazily (Python 3.14 and later).
    """

    tag = "annotation"

    @classmethod
    def rewrite(cls, detail):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        if not ReferenceRewrite.LAZY_INTERPRETER.rewritable(
            DeclaredInterpreter.of(Path(site.file_path))
        ):
            return None
        return identities.class_named(site.detail[-1])


class WordSpelling(IdentitySpelling):
    """A multi-word string passed or returned: a class's name there.

    A one-word class name (Read, Open) is also an ordinary word, so only a
    multi-word spelling is taken as naming a class outside a type position.
    A member's spelling passed along is not taken as naming it: field, column
    and wire names share those spellings by coincidence.
    """

    tag = "word"

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[2])


class ComparedWordSpelling(IdentitySpelling):
    """A multi-word string compared or matched: a class's name or a family member's spelling.

    Comparing a value with the spelling asks which declaration it is; a
    one-word spelling (main, completed) is also an ordinary word, so only a
    multi-word one is taken as naming a declaration here.
    """

    tag = "compared word"

    @classmethod
    def resolve(cls, site, identities):
        literal = site.detail[1]
        return identities.class_named(literal) or identities.member_named(literal)


class ClassNameSpelling(IdentitySpelling):
    """A string compared with a `__name__` or `__qualname__`: a class name, one word or more."""

    tag = "class name"

    @classmethod
    def resolve(cls, site, identities):
        return identities.class_named(site.detail[-1])


@dataclass(frozen=True)
class Receiver:
    """What a site knows about the object an attribute is read from.

    `name` is the receiver's last segment (a class when the scan declares one of
    that name); `class_object` and `instance` are the class an enclosing
    parameter annotation declares it to be (`type[C]`, or `C`), or empty.
    """

    text: str
    name: str
    class_object: str
    instance: str

    def detail(self) -> tuple[str, ...]:
        return (self.text, self.name, self.class_object, self.instance)

    @classmethod
    def from_detail(cls, detail: Sequence[str]) -> Receiver:
        text, name, class_object, instance = detail
        return cls(text, name, class_object, instance)


class MemberSpelling(IdentitySpelling):
    """A string compared or matched with `x.attribute`, where the attribute is a family's member spelling."""

    tag = "member"

    @classmethod
    def resolve(cls, site, identities):
        _tag, attribute, *receiver, literal = site.detail
        found = identities.member_attribute(attribute, Receiver.from_detail(receiver))
        return None if found is None else found.spelled(literal)


class FlattenedSpelling(IdentitySpelling):
    """A family member's keyword spelling read off a class or object and passed or returned as text."""

    tag = "flattened"

    @classmethod
    def spelling(cls, detail):
        _tag, attribute, text, *_receiver = detail
        return f"{text}.{attribute}"

    @classmethod
    def resolve(cls, site, identities):
        _tag, attribute, *receiver = site.detail
        read = Receiver.from_detail(receiver)
        found = identities.member_attribute(attribute, read)
        if found is None:
            return None
        return found.read_off(read.text, identities.reads_any_member(read, found))


class ModuleSpelling(IdentitySpelling):
    """A dotted module name given to an importer, `sys.modules`, or an interpreter's `-m`."""

    tag = "module"

    @classmethod
    def rewrite(cls, detail):
        return ReferenceRewrite(detail[1])

    @classmethod
    def resolve(cls, site, identities):
        return identities.module_named(site.detail[-1])


class LookupSpelling(IdentitySpelling):
    """A string passed to a family's classmethod that returns the member it names."""

    tag = "lookup"

    @classmethod
    def resolve(cls, site, identities):
        _tag, receiver, method, literal = site.detail
        return identities.member_looked_up(receiver, method, literal)


# A finding names this many declaring sites of one identity.
LISTED_DECLARATIONS = 4


@dataclass(frozen=True)
class MemberSpellings:
    """A class keyword or constant whose explicit values are distinct among the siblings declaring it.

    `members` are the explicit values with their classes; `family_bases` are the
    classes below which it names members: the bases of those siblings, the
    classes whose `__init_subclass__` consumes it, and the classes that declare
    it as a `ClassVar` slot. A keyword may also be derived
    by the class machinery, so any spelling compared with it names a member; a
    class constant names one only by an explicit value.
    """

    attribute: str
    members: tuple[tuple[str, SourceLocation], ...]
    family_bases: frozenset[str]
    declared_by_keyword: bool
    slotted: bool
    declared_at: tuple[SourceLocation, ...]

    @property
    def identity(self) -> str:
        return f"the `{self.attribute}` spelling of a family member"

    CODEC_BOUNDARY = (
        " (codec boundary: needs judgment — the one wire decode may spell the tag)"
    )

    def read_off(self, receiver: str, any_member: bool) -> ResolvedIdentity | None:
        """A keyword spelling read as text; a class constant read is only a value.

        Reading it off one named member restates that member. Reading it off
        whichever member arrives (a receiver typed by the family's base, or not
        typed) is what a codec does at its boundary, so the finding says it
        needs judgment.
        """

        if not self.declared_by_keyword:
            return None
        boundary = self.CODEC_BOUNDARY if any_member else ""
        return ResolvedIdentity(
            f"{self.identity}, read off `{receiver}` as text{boundary}",
            self.declared_at[:LISTED_DECLARATIONS],
        )

    def spelled(self, literal: str) -> ResolvedIdentity | None:
        named = tuple(location for value, location in self.members if value == literal)
        if not named and not self.declared_by_keyword:
            return None
        who = (
            " and ".join(f"`{location.symbol}`" for location in named)
            if named
            else "a member whose spelling the family derives"
        )
        return ResolvedIdentity(
            f"{self.identity} {literal!r} ({who})",
            (named or self.declared_at)[:LISTED_DECLARATIONS],
        )


@dataclass(frozen=True)
class DeclaredIdentities:
    """Every identity the scan declares: classes, modules, member spellings and lookups."""

    classes: dict[str, tuple[SourceLocation, ...]]
    modules: dict[str, SourceLocation]
    members: dict[str, MemberSpellings]
    lookups: dict[str, tuple[str, ...]]
    bases: dict[str, tuple[str, ...]]
    instance_attributes: dict[str, frozenset[str]]
    carriers: dict[tuple[str, str, str], tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_modules(cls, modules: Iterable[ModuleDeclarations]) -> DeclaredIdentities:
        classes: dict[str, list[SourceLocation]] = defaultdict(list)
        module_locations: dict[str, SourceLocation] = {}
        siblings: dict[tuple[str, str, bool], list[tuple[str, SourceLocation]]] = (
            defaultdict(list)
        )
        consumers: dict[str, list[SourceLocation]] = defaultdict(list)
        slots: dict[str, list[SourceLocation]] = defaultdict(list)
        lookups: dict[str, set[str]] = defaultdict(set)
        bases: dict[str, set[str]] = defaultdict(set)
        instance_attributes: dict[str, set[str]] = defaultdict(set)
        explicit: list[tuple[str, str, SourceLocation]] = []
        for module in modules:
            module_locations[module.module_name] = SourceLocation(
                module.file_path, 1, module.module_name
            )
            for declared in module.classes:
                location = SourceLocation(
                    module.file_path, declared.line, declared.name
                )
                classes[declared.name].append(location)
                lookups[declared.name].update(declared.lookups)
                bases[declared.name].update(declared.base_names)
                instance_attributes[declared.name].update(declared.instance_attributes)
                for keyword in declared.consumed_keywords:
                    consumers[keyword].append(location)
                for slot in declared.slots:
                    slots[slot].append(location)
                for by_keyword, spellings in (
                    (True, declared.keywords),
                    (False, declared.constants),
                ):
                    for attribute, value in spellings:
                        explicit.append((attribute, value, location))
                        for base in declared.base_names:
                            siblings[(attribute, base, by_keyword)].append(
                                (value, location)
                            )
        members: dict[str, list[tuple[str, SourceLocation]]] = defaultdict(list)
        family_bases: dict[str, set[str]] = defaultdict(set)
        keyword_attributes: set[str] = set(consumers)
        for (attribute, base, by_keyword), group in siblings.items():
            values = [value for value, _location in group]
            if len(values) >= 2 and len(set(values)) == len(values):
                members[attribute].extend(group)
                family_bases[attribute].add(base)
                if by_keyword:
                    keyword_attributes.add(attribute)
        for declaring_by_attribute in (consumers, slots):
            for attribute, declaring in declaring_by_attribute.items():
                family_bases[attribute].update(
                    location.symbol for location in declaring
                )
        return cls(
            classes={name: tuple(found) for name, found in classes.items()},
            modules=module_locations,
            members={
                attribute: MemberSpellings(
                    attribute=attribute,
                    members=tuple(members[attribute]),
                    family_bases=frozenset(found),
                    declared_by_keyword=attribute in keyword_attributes,
                    slotted=attribute in slots,
                    declared_at=(
                        *consumers.get(attribute, ()),
                        *slots.get(attribute, ()),
                        *(location for _value, location in members[attribute]),
                    ),
                )
                for attribute, found in family_bases.items()
            },
            lookups={name: tuple(sorted(found)) for name, found in lookups.items()},
            bases={name: tuple(sorted(found)) for name, found in bases.items()},
            instance_attributes={
                name: frozenset(found) for name, found in instance_attributes.items()
            },
        ).with_family_values(explicit)

    def with_family_values(
        self, explicit: Iterable[tuple[str, str, SourceLocation]]
    ) -> DeclaredIdentities:
        """Every explicit value of a keyword or slot spelling on a class of its family names that member.

        A family that consumes a keyword, or declares a slot every subclass fills,
        declares the spelling per member, so a member that declares one alone
        among its siblings still names itself.
        """

        added: dict[str, list[tuple[str, SourceLocation]]] = defaultdict(list)
        for attribute, value, location in explicit:
            spellings = self.members.get(attribute)
            if (
                spellings is not None
                and (spellings.declared_by_keyword or spellings.slotted)
                and (value, location) not in spellings.members
                and self.in_family(location.symbol, spellings)
            ):
                added[attribute].append((value, location))
        return replace(
            self,
            members={
                attribute: replace(
                    spellings, members=(*spellings.members, *added.get(attribute, ()))
                )
                for attribute, spellings in self.members.items()
            },
        )

    def member_named(self, literal: str) -> ResolvedIdentity | None:
        """The family member whose explicit spelling the literal is."""

        for spellings in self.members.values():
            if any(value == literal for value, _location in spellings.members):
                return spellings.spelled(literal)
        return None

    def class_named(self, name: str) -> ResolvedIdentity | None:
        declared = self.classes.get(name)
        if declared is None:
            return None
        return ResolvedIdentity(f"class `{name}`", declared[:LISTED_DECLARATIONS])

    def module_named(self, name: str) -> ResolvedIdentity | None:
        declared = self.modules.get(name)
        if declared is None:
            return None
        return ResolvedIdentity(f"module `{name}`", (declared,))

    def ancestors(self, name: str) -> tuple[str, ...]:
        """The class and every scanned class it inherits from, by name."""

        found = {name: None}
        frontier = [name]
        for current in frontier:
            for base in self.bases.get(current, ()):
                if base not in found:
                    found[base] = None
                    frontier.append(base)
        return tuple(found)

    def in_family(self, name: str, spellings: MemberSpellings) -> bool:
        return any(
            ancestor in spellings.family_bases for ancestor in self.ancestors(name)
        )

    def per_object(self, name: str, attribute: str) -> bool:
        """Whether the class or an ancestor declares the attribute as per-object state."""

        return any(
            attribute in self.instance_attributes.get(ancestor, ())
            for ancestor in self.ancestors(name)
        )

    def member_attribute(
        self, attribute: str, receiver: Receiver
    ) -> MemberSpellings | None:
        """The family spelling `receiver.attribute` reads, when the receiver's declared type says so.

        A class named directly, or a parameter annotated `type[C]`, reads the
        class's own spelling; a parameter annotated `C` reads it unless C declares
        the attribute as per-object state; a receiver of no declared type reads
        a family spelling only if no class anywhere declares that name as
        per-object state.
        """

        spellings = self.members.get(attribute)
        if spellings is None:
            return None
        for class_object in (receiver.name, receiver.class_object):
            if class_object in self.classes:
                return spellings if self.in_family(class_object, spellings) else None
        if receiver.instance in self.classes:
            return (
                spellings
                if self.in_family(receiver.instance, spellings)
                and not self.per_object(receiver.instance, attribute)
                else None
            )
        if any(attribute in declared for declared in self.instance_attributes.values()):
            return None
        return spellings if spellings.members else None

    def resolved_class(self, receiver: Receiver) -> str:
        """The scanned class the receiver is, or is an instance of, by its declared type."""

        return next(
            (
                name
                for name in (receiver.name, receiver.class_object, receiver.instance)
                if name in self.classes
            ),
            "",
        )

    def reads_any_member(self, receiver: Receiver, spellings: MemberSpellings) -> bool:
        """Whether the receiver is whichever member arrives rather than one named member."""

        resolved = self.resolved_class(receiver)
        return not resolved or resolved in spellings.family_bases

    def member_looked_up(
        self, receiver: str, method: str, literal: str
    ) -> ResolvedIdentity | None:
        declaring = next(
            (
                ancestor
                for ancestor in self.ancestors(receiver)
                if method in self.lookups.get(ancestor, ())
            ),
            None,
        )
        if declaring is None:
            return None
        for spellings in self.members.values():
            if self.in_family(receiver, spellings) and any(
                value == literal for value, _location in spellings.members
            ):
                return spellings.spelled(literal)
        return ResolvedIdentity(
            f"the `{receiver}` member spelled {literal!r}",
            self.classes[declaring][:LISTED_DECLARATIONS],
        )

    def resolve(self, site: RestatingSite) -> RestatingSite | None:
        position = IdentitySpelling.positions()[site.detail[0]]
        resolved = position.resolve(site, self)
        return None if resolved is None else resolved.restated_by(site)

    def with_carriers(self, sites: Sequence[RestatingSite]) -> DeclaredIdentities:
        """Name, per file and rewritable position, the one finding that carries the file's rewrite.

        A rewrite is file-wide, so one finding per file and position carries it as
        the candidate: the one whose site is the file's first rewritable site of
        that position (by line, then identity).
        """

        positions = IdentitySpelling.positions()
        first: dict[tuple[str, str], RestatingSite] = {}
        for site in sites:
            position = site.detail[0]
            rewrite = positions[position].rewrite(site.detail)
            if not rewrite.rewritable(DeclaredInterpreter.of(Path(site.file_path))):
                continue
            key = (site.file_path, position)
            current = first.get(key)
            if current is None or (site.lineno, site.group) < (
                current.lineno,
                current.group,
            ):
                first[key] = site
        carriers: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for (_file_path, position), site in sorted(first.items()):
            carriers[(site.file_path, site.symbol, site.group)].append(position)
        return replace(
            self, carriers={key: tuple(found) for key, found in carriers.items()}
        )

    def carried_positions(self, sites: Sequence[RestatingSite]) -> tuple[str, ...]:
        """The positions whose file-wide rewrite this finding's sites carry."""

        return self.carriers.get(
            (sites[0].file_path, sites[0].symbol, sites[0].group), ()
        )

    def declarations_of(
        self, sites: Sequence[RestatingSite]
    ) -> tuple[SourceLocation, ...]:
        """The declaring sites of the identity one group of resolved sites restates."""

        resolved = IdentitySpelling.positions()[sites[0].detail[0]].resolve(
            sites[0], self
        )
        return () if resolved is None else resolved.declared_at


class IdentityPositions:
    """Which expressions of one module are identity positions, and what each says."""

    MULTIWORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+")
    IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
    CLASS_NAME_ATTRIBUTES = frozenset(("__name__", "__qualname__"))
    COLLECTIONS = frozenset(
        (
            "tuple",
            "list",
            "set",
            "frozenset",
            "Sequence",
            "MutableSequence",
            "Iterable",
            "Iterator",
            "Collection",
            "AbstractSet",
            "MutableSet",
        )
    )

    def __init__(self, parsed_module: ParsedModule) -> None:
        self.parsed_module = parsed_module
        self.index: ModuleSyntaxIndex = module_syntax_index(parsed_module.module)
        self.node_indices = {
            id(node): node_index
            for node_index, node in enumerate(self.index.depth_first_nodes)
        }
        self.references = ModuleReferenceScope(parsed_module.module)

    def site(self, node: ast.expr, detail: tuple[str, ...]) -> RestatingSite:
        return RestatingSiteFamily.site(
            self.parsed_module,
            node,
            self.index.lexical_symbol(self.node_indices[id(node)]),
            detail[-1],
            detail,
        )

    @classmethod
    def multiword(cls, literal: str) -> bool:
        return (
            cls.IDENTIFIER.fullmatch(literal) is not None
            and len(cls.MULTIWORD.findall(literal)) >= 2
        )

    @staticmethod
    def string(node: ast.AST) -> str | None:
        match node:
            case ast.Constant(value=str() as value):
                return value
        return None

    called = staticmethod(ModuleReferenceScope.called)

    def sites(self) -> Iterator[RestatingSite]:
        yield from self.type_sites()
        for node_index, node in self.index.indexed_nodes_of_type(ast.Compare):
            yield from self.compare_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Match):
            yield from self.match_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Call):
            yield from self.call_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Return):
            if node.value is not None:
                yield from self.passed_sites(node_index, node.value, "return")
        yield from self.module_sites()

    # Type positions -----------------------------------------------------------

    def type_sites(self) -> Iterator[RestatingSite]:
        """Class names in quoted type strings: a reference wherever the name would resolve.

        Where it resolves on any interpreter the site is a `type` site; an
        annotation that resolves only when evaluated lazily is an `annotation`
        site, which the join keeps only for projects whose Python is lazy.
        """

        for quoted in self.references.quoted_type_names():
            rewrite = self.references.rewrite_of(quoted)
            for name, root in quoted.names:
                if self.references.resolves_eagerly(
                    root, quoted.node.lineno, quoted.annotation
                ):
                    yield self.site(quoted.node, TypeSpelling.detail(rewrite, name))
                elif quoted.annotation:
                    yield self.site(
                        quoted.node, LazyAnnotationSpelling.detail(rewrite, name)
                    )

    # Comparison and match positions ---------------------------------------------

    def compared(
        self, node_index: int, subject: ast.expr, literal_node: ast.expr
    ) -> Iterator[RestatingSite]:
        literal = self.string(literal_node)
        if literal is None or isinstance(subject, ast.Constant):
            return
        match subject:
            case ast.Attribute(attr=attribute) if (
                attribute in self.CLASS_NAME_ATTRIBUTES
            ):
                yield self.site(literal_node, ClassNameSpelling.detail(literal))
                return
            case ast.Attribute(value=receiver, attr=attribute):
                yield self.site(
                    literal_node,
                    MemberSpelling.detail(
                        attribute,
                        *self.receiver(node_index, receiver).detail(),
                        literal,
                    ),
                )
        if self.multiword(literal):
            yield self.site(literal_node, ComparedWordSpelling.detail(literal))

    def compare_sites(
        self, node_index: int, node: ast.Compare
    ) -> Iterator[RestatingSite]:
        left = node.left
        for operator, right in zip(node.ops, node.comparators):
            match operator, right:
                case ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot(), _:
                    yield from self.compared(node_index, left, right)
                    yield from self.compared(node_index, right, left)
                case ast.In() | ast.NotIn(), ast.Set(elts=items) | ast.Tuple(
                    elts=items
                ) | ast.List(elts=items):
                    for item in items:
                        yield from self.compared(node_index, left, item)
            left = right

    def match_sites(self, node_index: int, node: ast.Match) -> Iterator[RestatingSite]:
        for case in node.cases:
            for literal_node in self.pattern_literals(case.pattern):
                yield from self.compared(node_index, node.subject, literal_node)

    @classmethod
    def pattern_literals(cls, pattern: ast.pattern) -> Iterator[ast.expr]:
        match pattern:
            case ast.MatchValue(value=ast.Constant(value=str()) as literal):
                yield literal
            case ast.MatchOr(patterns=alternatives):
                for alternative in alternatives:
                    yield from cls.pattern_literals(alternative)

    # Argument and return positions -----------------------------------------------

    def call_sites(self, node_index: int, node: ast.Call) -> Iterator[RestatingSite]:
        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            yield from self.passed_sites(node_index, argument, "argument")
        match node:
            case ast.Call(
                func=ast.Attribute(value=receiver, attr=method),
                args=[ast.Constant(value=str() as literal) as literal_node, *_],
            ) if isinstance(receiver, (ast.Name, ast.Attribute)):
                yield self.site(
                    literal_node,
                    LookupSpelling.detail(self.called(receiver), method, literal),
                )

    def passed_sites(
        self, node_index: int, value: ast.expr, position: str
    ) -> Iterator[RestatingSite]:
        for passed in self.passed_values(value):
            literal = self.string(passed)
            if literal is not None and self.multiword(literal):
                yield self.site(passed, WordSpelling.detail(position, literal))
            match passed:
                case ast.Attribute(value=receiver, attr=attribute) if not self.own(
                    node_index, receiver
                ):
                    yield self.site(
                        passed,
                        FlattenedSpelling.detail(
                            attribute, *self.receiver(node_index, receiver).detail()
                        ),
                    )

    def receiver(self, node_index: int, receiver: ast.expr) -> Receiver:
        """The receiver's text, last segment, and the type its binding declares for it."""

        class_object = instance = ""
        if isinstance(receiver, ast.Name):
            declared = self.declared_type(node_index, receiver.id)
            match declared:
                case ast.Subscript(value=ast.Name(id="type"), slice=element):
                    class_object = self.called(element)
                case ast.Name() | ast.Attribute():
                    instance = self.called(declared)
        return Receiver(
            ast.unparse(receiver), self.called(receiver), class_object, instance
        )

    def declared_type(self, node_index: int, name: str) -> ast.expr | None:
        """The type the nearest binding of `name` around this node declares.

        A parameter declares its annotation; a `for` or comprehension target
        declares the element type of what it iterates, when that is a parameter
        annotated as a collection (or a `zip` of such), unpacked by the target.
        Nothing wider: no assignment or flow is followed.
        """

        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            for target, iterated in self.loops(ancestor):
                if name in self.target_names(target):
                    return self.unpacked(
                        target,
                        self.elements(self.node_indices[id(ancestor)], iterated),
                        name,
                    )
            match ancestor:
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    ):
                        if argument.arg == name:
                            return argument.annotation
        return None

    @staticmethod
    def loops(node: ast.AST) -> Iterator[tuple[ast.expr, ast.expr]]:
        match node:
            case ast.For(target=target, iter=iterated) | ast.AsyncFor(
                target=target, iter=iterated
            ):
                yield target, iterated
            case (
                ast.ListComp(generators=generators)
                | ast.SetComp(generators=generators)
                | ast.GeneratorExp(generators=generators)
                | ast.DictComp(generators=generators)
            ):
                for generator in generators:
                    yield generator.target, generator.iter

    @staticmethod
    def target_names(target: ast.expr) -> frozenset[str]:
        return frozenset(
            node.id for node in ast.walk(target) if isinstance(node, ast.Name)
        )

    def elements(self, node_index: int, iterated: ast.expr) -> ast.expr | None:
        """The element type of an iterated parameter, or a tuple of them for `zip`."""

        match iterated:
            case ast.Name(id=name):
                return self.collection_element(self.declared_type(node_index, name))
            case ast.Call(func=ast.Name(id="zip"), args=arguments):
                return ast.Tuple(
                    elts=[
                        self.elements(node_index, argument) or ast.Constant(None)
                        for argument in arguments
                    ]
                )
        return None

    @classmethod
    def collection_element(cls, annotation: ast.expr | None) -> ast.expr | None:
        match annotation:
            case ast.Subscript(
                value=collection,
                slice=ast.Tuple(elts=[element, ast.Constant(value=builtins_ellipsis)]),
            ) if (
                builtins_ellipsis is Ellipsis
                and cls.called(collection) in cls.COLLECTIONS
            ):
                return element
            case ast.Subscript(value=collection, slice=element) if cls.called(
                collection
            ) in cls.COLLECTIONS and not isinstance(element, ast.Tuple):
                return element
        return None

    @classmethod
    def unpacked(
        cls, target: ast.expr, element: ast.expr | None, name: str
    ) -> ast.expr | None:
        match target, element:
            case ast.Name(id=bound), _ if bound == name:
                return element
            case ast.Tuple(elts=targets), ast.Tuple(elts=elements) | ast.Subscript(
                value=ast.Name(id="tuple"), slice=ast.Tuple(elts=elements)
            ) if len(targets) == len(elements):
                for inner, inner_element in zip(targets, elements):
                    if name in cls.target_names(inner):
                        return cls.unpacked(inner, inner_element, name)
        return None

    @classmethod
    def passed_values(cls, value: ast.expr) -> Iterator[ast.expr]:
        """The value itself, or the items of a literal tuple, list, set or boolean chain."""

        match value:
            case (
                ast.Tuple(elts=items)
                | ast.List(elts=items)
                | ast.Set(elts=items)
                | ast.BoolOp(values=items)
            ):
                for item in items:
                    yield from cls.passed_values(item)
            case _:
                yield value

    def own(self, node_index: int, receiver: ast.expr) -> bool:
        """Whether the receiver is the enclosing method's own object or class."""

        if not isinstance(receiver, ast.Name):
            return False
        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            match ancestor:
                case ast.ClassDef(name=name):
                    return receiver.id == name
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    parameters = [*arguments.posonlyargs, *arguments.args]
                    if parameters and parameters[0].arg == receiver.id:
                        return True
        return False

    # Module positions ---------------------------------------------------------

    def module_sites(self) -> Iterator[RestatingSite]:
        for spelled in SpelledModules(self.parsed_module.module):
            yield self.site(
                spelled.literal,
                ModuleSpelling.detail(
                    self.references.module_rewrite(spelled), spelled.name
                ),
            )


class IdentitySpellingFamily(RestatingSiteFamily):
    """Literals in positions where a string can stand for a declared identity."""

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        yield from IdentityPositions(parsed_module).sites()
