"""A class reference read from source: the name it spells and the role it plays."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import NamedTuple

from nominal_refactor_advisor.ast_projection import ExpressionReference


class ClassReferenceRole(StrEnum):
    """How one expression uses a module-level name that may be a class."""

    TYPE_TEST = "type_test"
    CLASS_PATTERN = "class_pattern"
    TYPE_KEY = "type_key"
    CONSTRUCTION = "construction"
    OTHER = "other"

    @property
    def distinguishes_identity(self) -> bool:
        """The use answers differently for two classes even when their bodies agree."""

        return self in (
            ClassReferenceRole.TYPE_TEST,
            ClassReferenceRole.CLASS_PATTERN,
            ClassReferenceRole.TYPE_KEY,
        )

    @property
    def binds_fields(self) -> bool:
        """The read builds the class or matches against its fields."""

        return self in (ClassReferenceRole.CONSTRUCTION, ClassReferenceRole.CLASS_PATTERN)

    def positional_bindings(self, parent: ast.AST | None) -> int:
        """How many arguments or subpatterns this read binds to fields by position."""

        if self is ClassReferenceRole.CONSTRUCTION and isinstance(parent, ast.Call):
            return len(parent.args)
        if self is ClassReferenceRole.CLASS_PATTERN and isinstance(parent, ast.MatchClass):
            return len(parent.patterns)
        return 0

    @classmethod
    def of_reference(
        cls,
        node: ast.expr,
        parent: ast.AST | None,
        grandparent: ast.AST | None,
        import_aliases: Mapping[str, str],
        as_type: bool,
    ) -> ClassReferenceRole:
        """The role one read of a name plays, from the syntax around it.

        `as_type` says the read sits in a type form (`reads_as_type`). There a
        subscript `G[C]` names `C` as a type parameter, as a bare `x: C` does: it
        looks nothing up by `C`, so it is no type key.
        """

        if isinstance(parent, ast.Call) and parent.func is node:
            return cls.CONSTRUCTION
        type_test_operand = (
            parent.args[1]
            if isinstance(parent, ast.Call)
            and (_is_builtin_call(parent, "isinstance") or _is_builtin_call(parent, "issubclass"))
            and len(parent.args) == 2
            else None
        )
        if type_test_operand is node or (
            isinstance(parent, ast.Tuple)
            and isinstance(grandparent, ast.Call)
            and (
                _is_builtin_call(grandparent, "isinstance")
                or _is_builtin_call(grandparent, "issubclass")
            )
            and len(grandparent.args) == 2
            and grandparent.args[1] is parent
        ):
            return cls.TYPE_TEST
        if (
            isinstance(parent, ast.Compare)
            and any(isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) for op in parent.ops)
            and any(
                _is_builtin_call(operand, "type")
                for operand in (parent.left, *parent.comparators)
            )
        ):
            return cls.TYPE_TEST
        if isinstance(parent, ast.MatchClass) and parent.cls is node:
            return cls.CLASS_PATTERN
        if isinstance(parent, ast.Dict) and any(key is node for key in parent.keys):
            return cls.TYPE_KEY
        if isinstance(parent, ast.Subscript) and parent.slice is node:
            if as_type:
                return cls.OTHER
            subscripted = ExpressionReference.decoded(parent.value).attribute_chain
            root = (
                None
                if subscripted is None
                else import_aliases.get(subscripted[0], subscripted[0])
            )
            if subscripted is not None and (
                (subscripted[0] not in import_aliases and hasattr(builtins, subscripted[0]))
                or (root is not None and root.split(".", 1)[0] in _GENERIC_ALIAS_ROOTS)
            ):
                return cls.OTHER
            return cls.TYPE_KEY
        return cls.OTHER


def reads_as_type(index: int, nodes: Sequence[ast.AST], parents: Sequence[int]) -> bool:
    """Whether the node at `index` sits in a type: an annotation, or the class
    of a `type[...]` form, which calling builds a `type`, never the class; the
    class comes back out of either only through a decoder."""

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
            or (isinstance(parent, ast.TypeVar) and parent.bound is child)
            or (isinstance(parent, ast.TypeAlias) and parent.value is child)
            or (
                isinstance(parent, ast.Subscript)
                and parent.slice is child
                and isinstance(parent.value, ast.Name)
                and parent.value.id == "type"
            )
        ):
            return True
        if isinstance(parent, ast.stmt):
            return False
        child, position = parent, parents[position]
    return False


def subclass_test_subject(
    node: ast.expr, parent: ast.AST | None, grandparent: ast.AST | None
) -> ast.expr | None:
    """The class an `issubclass` call asks about, when `node` is what it is
    tested against, alone or in a tuple; a class's ancestry decides the answer."""

    call = grandparent if isinstance(parent, ast.Tuple) else parent
    if not (_is_builtin_call(call, "issubclass") and len(call.args) == 2):
        return None
    tested = call.args[1]
    if tested is node or (isinstance(tested, ast.Tuple) and tested is parent):
        return call.args[0]
    return None


class CompactClassReference(NamedTuple):
    """One read of a module-level name by one consumer, and the role it plays.

    `consumer` is the enclosing function or class qualname, or the module
    itself (empty); `reference_parts` the full name or attribute chain read.
    For a call, `positional_arguments` is the most positional arguments any
    call in the consumer passes and `unpacked_arguments` whether one unpacks
    `*args` or `**kwargs`, which hides how it binds fields.
    """

    consumer: str
    reference_parts: tuple[str, ...]
    role: ClassReferenceRole
    line: int
    positional_arguments: int = 0
    unpacked_arguments: bool = False
    copied_fields: tuple[str, ...] = ()
    """For a construction, the most keywords one call passes as `x.<same name>`
    from a single value `x`: fields copied off another object."""
    tested_subclass: tuple[str, ...] = ()
    """For an `issubclass` test, the module-level name chain of the class it
    asks about, when it spells one."""


class CompactScriptScope(NamedTuple):
    """The class reads of one script a module's string literal holds, which a
    child interpreter runs as a module of its own: that module's name, its own
    imports, and its reads, each line the holding file's."""

    module_name: str
    import_aliases: tuple[tuple[str, str], ...]
    class_references: tuple[CompactClassReference, ...]


_GENERIC_ALIAS_ROOTS = frozenset(
    ("builtins", "typing", "typing_extensions", "collections")
)


def _is_builtin_call(node: ast.AST | None, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )
