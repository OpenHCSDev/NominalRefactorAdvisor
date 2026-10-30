"""Declared receiver typing: each form resolves from declarations alone, or names why not."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import (
    AssignedClassMember,
    CallableClassMember,
    CompactAnnotatedMethod,
    CompactAnnotatedProperty,
    CompactClassFamilyIndexBuilder,
    CompactModuleClassProjectionFamily,
    CompactUnannotatedCallable,
    MissingClassMember,
    OpenClassMember,
    module_import_aliases,
)
from nominal_refactor_advisor import declared_receivers
from nominal_refactor_advisor.declared_receivers import (
    DeclaredReceiverTyping,
    ReceiverPathBuilder,
    ScannedClass,
    UndeclaredType,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.typeshed_stubs import (
    AbsentStubRoot,
    RunningInterpreterBranches,
)

MODELS = """
from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Optional, Self, TypeVar

T = TypeVar("T")


class Shape:
    marker: int

    def grown(self) -> Self:
        return self

    @classmethod
    def made(cls) -> Self:
        return cls()

    def rename(self):
        return self


class Square(Shape):
    pass


class Circle(Shape):
    pass


class Unrelated:
    marker: int


@dataclass(frozen=True)
class Board:
    shapes: tuple[Shape, ...]
    pair: tuple[Square, Circle]
    listed: list[Shape]
    by_name: dict[str, Shape]
    by_shape: Mapping[Shape, str]
    sequence: Sequence[Shape]
    mixed: Square | Circle
    foreign: Square | Unrelated
    maybe: Shape | None
    kind: type[Square] | None
    optional_kind: Optional[type[Square]]
    none_first_kind: None | type[Square]
    kinds: type[Square] | type[Circle]
    alien_kinds: type[Square] | type[Unrelated]

    @property
    def first(self) -> Shape:
        return self.shapes[0]

    @cached_property
    def lazy(self) -> Square:
        return Square()

    def iterate(self) -> Iterator[Shape]:
        return iter(self.shapes)

    def lookup(self, name: str) -> Shape:
        return self.by_name[name]


class Index(Generic[T]):
    by_symbol: dict[str, T]


class ShapeIndex(Index[Shape]):
    pass


def make_board() -> Board:
    raise NotImplementedError


def unannotated_board():
    raise NotImplementedError
"""


def _typed(tmp_path: Path, body: str) -> dict[int, object]:
    """The declared type of every `X.marker` receiver in `body`, by line."""

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "models.py").write_text(MODELS)
    (package / "api.py").write_text("from pkg.models import Square\n")
    (package / "use.py").write_text(
        "from __future__ import annotations\n"
        "from pkg.models import (\n"
        "    Board, Shape, ShapeIndex, Square, Unrelated, make_board, unannotated_board,\n"
        ")\n"
        "from pkg.api import Square as Exported\n\n"
        + body
    )
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False).modules
        projections = [
            CompactModuleClassProjectionFamily._required_module_projection(module)
            for module in modules
        ]
        index = CompactClassFamilyIndexBuilder(projections).build()
        typing = DeclaredReceiverTyping.from_projections(projections, index)
        (use,) = (module for module in modules if module.module_name.endswith("use"))
        builder = ReceiverPathBuilder(
            use.module_name, use.module, module_import_aliases(use)
        )
        return {
            node.lineno: builder.path_of(node.value).declared(typing)
            for node in ast.walk(use.module)
            if isinstance(node, ast.Attribute) and node.attr == "marker"
        }


def _symbols(tmp_path: Path, body: str) -> list[tuple[str, ...]]:
    return [
        tuple(symbol.removeprefix("pkg.models.") for symbol in declared.class_symbols())
        for _line, declared in sorted(_typed(tmp_path, body).items())
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        pytest.param(
            "def f(board: Board):\n    for shape in board.shapes:\n        shape.marker\n",
            ("Shape",),
            id="loop over tuple[F, ...]",
        ),
        pytest.param(
            "def f(board: Board):\n    return [shape.marker for shape in board.listed]\n",
            ("Shape",),
            id="comprehension over list[F]",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.sequence:\n        shape.marker\n",
            ("Shape",),
            id="loop over Sequence[F]",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.iterate():\n        shape.marker\n",
            ("Shape",),
            id="loop over a method returning Iterator[F]",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.by_name.values():\n        shape.marker\n",
            ("Shape",),
            id="mapping values",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.by_shape:\n        shape.marker\n",
            ("Shape",),
            id="bare mapping iteration gives keys",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.by_shape.keys():\n        shape.marker\n",
            ("Shape",),
            id="mapping keys",
        ),
        pytest.param(
            "def f(board: Board):\n    for _name, shape in board.by_name.items():\n        shape.marker\n",
            ("Shape",),
            id="mapping items destructured",
        ),
        pytest.param(
            "def f(board: Board):\n    for _index, shape in enumerate(board.shapes):\n        shape.marker\n",
            ("Shape",),
            id="enumerate",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in sorted(board.listed):\n        shape.marker\n",
            ("Shape",),
            id="sorted keeps the element",
        ),
        pytest.param(
            "def f(board: Board):\n    board.pair[1].marker\n",
            ("Circle",),
            id="tuple position by constant index",
        ),
        pytest.param(
            "def f(board: Board):\n    board.pair[-2].marker\n",
            ("Square",),
            id="tuple position by negative index",
        ),
        pytest.param(
            "def f(board: Board, i: int):\n    board.listed[i].marker\n",
            ("Shape",),
            id="list element by any index",
        ),
        pytest.param(
            "def f(board: Board, k: str):\n    board.by_name[k].marker\n",
            ("Shape",),
            id="mapping value by key",
        ),
        pytest.param(
            "def f(board: Board, k: str):\n    shape = board.by_name.get(k)\n    shape.marker\n",
            ("Shape",),
            id="mapping get, the None arm excluded",
        ),
        pytest.param(
            "def f():\n    board = make_board()\n    board.first.marker\n",
            ("Shape",),
            id="local from a function's declared return, then a property",
        ),
        pytest.param(
            "def f(board: Board):\n    shape = board.lookup('x')\n    shape.marker\n",
            ("Shape",),
            id="local from a method's declared return",
        ),
        pytest.param(
            "def f(square: Square):\n    grown = square.grown()\n    grown.marker\n",
            ("Square",),
            id="Self return resolves to the receiver, through the MRO",
        ),
        pytest.param(
            "def f():\n    Square.made().marker\n",
            ("Square",),
            id="classmethod Self return on the class object",
        ),
        pytest.param(
            "def f(board: Board):\n    board.lazy.marker\n",
            ("Square",),
            id="cached property",
        ),
        pytest.param(
            "def f(board: Board):\n    board.mixed.marker\n",
            ("Circle", "Square"),
            id="union of one family",
        ),
        pytest.param(
            "def f(board: Board):\n    board.maybe.marker\n",
            ("Shape",),
            id="optional field read as its present arm",
        ),
        pytest.param(
            "def f(board: Board, flag: bool):\n"
            "    shape = board.pair[0]\n"
            "    if flag:\n"
            "        shape = board.pair[0]\n"
            "    shape.marker\n",
            ("Square",),
            id="rebound to the same type",
        ),
        pytest.param(
            "def f(board: Board, flag: bool):\n"
            "    shape = board.pair[0]\n"
            "    if flag:\n"
            "        shape = board.pair[1]\n"
            "    shape.marker\n",
            ("Circle", "Square"),
            id="rebound within one family joins as a union does",
        ),
        pytest.param(
            "def f(board: Board):\n    board.kind.marker\n",
            ("Square",),
            id="type[A] | None",
        ),
        pytest.param(
            "def f(board: Board):\n    board.optional_kind.marker\n",
            ("Square",),
            id="Optional[type[A]]",
        ),
        pytest.param(
            "def f(board: Board):\n    board.none_first_kind.marker\n",
            ("Square",),
            id="None | type[A]",
        ),
        pytest.param(
            "def f(board: Board):\n    board.kinds.marker\n",
            ("Circle", "Square"),
            id="type[A] | type[B] of one family",
        ),
        pytest.param(
            "def f(board: Board):\n    board.listed.marker\n",
            ("builtins.list",),
            id="a declared list[A] is a builtins.list, not collections.abc.Iterable",
        ),
        pytest.param(
            "def f(board: Board):\n    board.by_name.marker\n",
            ("builtins.dict",),
            id="a declared dict is a builtins.dict",
        ),
        pytest.param(
            "def f(board: Board):\n    board.sequence.marker\n",
            ("collections.abc.Sequence",),
            id="a declared Sequence is the class it spells",
        ),
        pytest.param(
            "def f(board: Board):\n    board.pair.marker\n",
            ("builtins.tuple",),
            id="a declared tuple is a builtins.tuple",
        ),
        pytest.param(
            "def f(board: Board):\n    board.by_name.values().marker\n",
            (),
            id="a mapping view names no class",
        ),
        pytest.param(
            "def f(index: ShapeIndex, k: str):\n    index.by_symbol[k].marker\n",
            ("Shape",),
            id="a generic base's type variable bound by the subclass",
        ),
        pytest.param(
            "def f(square: Exported):\n    square.marker\n",
            ("Square",),
            id="a class re-exported by another module",
        ),
        pytest.param(
            "class Holder:\n"
            "    def f(self) -> None:\n"
            "        self.marker\n",
            ("pkg.use.Holder",),
            id="a method's own receiver",
        ),
    ),
)
def test_each_declared_form_resolves(tmp_path, body, expected):
    assert _symbols(tmp_path, body) == [expected]


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        pytest.param(
            "def f(board: Board, flag: bool):\n"
            "    shape = board.pair[0]\n"
            "    if flag:\n"
            "        shape = Unrelated()\n"
            "    shape.marker\n",
            "no one family",
            id="rebound to another family",
        ),
        pytest.param(
            "def f(board: Board):\n    board.alien_kinds.marker\n",
            "no one family",
            id="type[A] | type[B] of two families",
        ),
        pytest.param(
            "def f(board: Board):\n    board.foreign.marker\n",
            "no one family",
            id="heterogeneous union",
        ),
        pytest.param(
            "def f():\n    board = unannotated_board()\n    board.marker\n",
            "calls no scanned class or annotated function",
            id="unannotated callee",
        ),
        pytest.param(
            "def f(square: Square):\n    renamed = square.rename()\n    renamed.marker\n",
            "declares no return annotation",
            id="unannotated method",
        ),
        pytest.param(
            "def f(board):\n    board.marker\n",
            "declares no annotation",
            id="unannotated parameter",
        ),
        pytest.param(
            "def f(board: Board):\n    for shape in board.shapes:\n        shape += 1\n        shape.marker\n",
            "augmented assignment",
            id="augmented rebind",
        ),
        pytest.param(
            "def f(board: Board):\n    with open('x') as shape:\n        shape.marker\n",
            "`with` statement",
            id="with binding",
        ),
    ),
)
def test_an_undeclared_receiver_stays_unresolved_with_its_reason(tmp_path, body, reason):
    (declared,) = _typed(tmp_path, body).values()
    assert isinstance(declared, UndeclaredType)
    assert reason in declared.reason


def _index(tmp_path: Path, source: str):
    (tmp_path / "models.py").write_text(source)
    with ScanCache.scope():
        modules = parse_python_modules(tmp_path, use_parse_cache=False).modules
        projections = [
            CompactModuleClassProjectionFamily._required_module_projection(module)
            for module in modules
        ]
        return projections, CompactClassFamilyIndexBuilder(projections).build()


def test_member_lookup_reads_the_first_binding_along_the_mro(tmp_path):
    _projections, index = _index(
        tmp_path,
        "import abc\n"
        "class Base:\n    name: str\n    def run(self) -> int: ...\n\n"
        "class Child(Base):\n    @property\n    def name(self) -> str: ...\n\n"
        "class Open(abc.Unknown):\n    pass\n",
    )
    assert isinstance(index.member_in_mro("models.Base", "name"), AssignedClassMember)
    child_name = index.member_in_mro("models.Child", "name")
    assert isinstance(child_name, CallableClassMember)
    assert isinstance(child_name.declaration, CompactAnnotatedProperty)
    run = index.member_in_mro("models.Child", "run")
    assert isinstance(run, CallableClassMember) and run.owner.symbol == "models.Base"
    assert isinstance(run.declaration, CompactAnnotatedMethod)
    assert isinstance(index.member_in_mro("models.Child", "absent"), MissingClassMember)
    assert isinstance(index.member_in_mro("models.Open", "absent"), OpenClassMember)


def test_callable_member_declarations_skip_overloads_and_setters(tmp_path):
    projections, _index_ = _index(
        tmp_path,
        "from typing import overload\n"
        "class Holder:\n"
        "    @property\n    def value(self) -> int: ...\n"
        "    @value.setter\n    def value(self, new: int) -> None: ...\n"
        "    @overload\n    def pick(self, key: int) -> int: ...\n"
        "    def pick(self, key): ...\n",
    )
    (projection,) = projections
    (holder,) = projection.classes
    assert holder.callable_member_declarations == (
        CompactAnnotatedProperty("value", 4, "int"),
        CompactUnannotatedCallable("pick", 9),
    )


def test_decorators_are_the_objects_their_names_bind(tmp_path):
    projections, _index_ = _index(
        tmp_path,
        "import abc\n"
        "import typing as t\n"
        "from functools import cached_property as cp\n\n\n"
        "def property(function):\n    return function\n\n\n"
        "@t.final\n"
        "class Holder:\n"
        "    @property\n    def local(self) -> int: ...\n"
        "    @cp\n    def cached(self) -> int: ...\n"
        "    @abc.abstractmethod\n    def asked(self) -> int: ...\n",
    )
    (projection,) = projections
    (holder,) = projection.classes
    # The module's own `property` is a plain function, so `local` is a method.
    assert holder.callable_member_declarations == (
        CompactAnnotatedMethod("local", 13, "int"),
        CompactAnnotatedProperty("cached", 15, "int"),
        CompactAnnotatedMethod("asked", 17, "int"),
    )
    assert holder.abstract_method_names == ("asked",)
    assert holder.is_final


def test_a_cached_class_projection_carries_the_declared_returns(tmp_path):
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "models.py").write_text(
        "class Holder:\n    def make(self) -> 'Holder': ...\n\n"
        "def build() -> Holder: ...\n"
    )
    cache_dir = tmp_path / "cache"

    def projection():
        with ScanCache.scope():
            (module,) = parse_python_modules(
                source_root, use_parse_cache=True, cache_dir=cache_dir
            ).modules
            return CompactModuleClassProjectionFamily._required_module_projection(module)

    cold = projection()
    warm = projection()
    for loaded in (cold, warm):
        (holder,) = loaded.classes
        assert holder.callable_member_declarations == (
            CompactAnnotatedMethod("make", 2, "'Holder'"),
        )
        assert [declared.name for declared in loaded.annotated_function_returns] == [
            "build"
        ]


STUB_FIELD_READS = (
    "import ast\n\n"
    "def f(\n"
    "    assigned: ast.AnnAssign,\n"
    "    handler: ast.ExceptHandler,\n"
    "    case: ast.match_case,\n"
    "    statement: ast.stmt,\n"
    "):\n"
    "    assigned.value.marker\n"
    "    handler.type.marker\n"
    "    case.guard.marker\n"
    "    statement.end_lineno.marker\n"
)


def test_a_stdlib_class_is_read_from_its_typeshed_stub(tmp_path):
    """`ast.AnnAssign.value: expr | None` is typeshed's declaration, read where
    it is declared; reading the receiver excludes the None arm."""

    assert _symbols(tmp_path, STUB_FIELD_READS) == [
        ("ast.expr",),
        ("ast.expr",),
        ("ast.expr",),
        ("builtins.int",),
    ]


def test_the_stub_member_is_its_declared_optional(tmp_path):
    with ScanCache.scope():
        typing = DeclaredReceiverTyping.from_projections(*_index(tmp_path, "x = 1\n"))
        assert typing.chased("ast.AnnAssign") == ScannedClass("ast.AnnAssign")
        stubs = typing.declaring("ast.AnnAssign")
        assert stubs is not typing
        for symbol, name, annotation in (
            ("ast.AnnAssign", "value", "expr | None"),
            ("ast.ExceptHandler", "type", "expr | None"),
            ("ast.match_case", "guard", "expr | None"),
            ("ast.stmt", "end_lineno", "int | None"),
        ):
            stubs.chased(symbol)
            found = stubs.class_index.declared_member(symbol, name)
            assert isinstance(found, AssignedClassMember)
            assert found.declaration.annotation_expression == annotation
            assert Path(found.owner.file_path).name == "ast.pyi"


def test_without_typeshed_the_stdlib_class_names_the_absent_stubs(tmp_path, monkeypatch):
    monkeypatch.setattr(declared_receivers, "installed_stub_root", AbsentStubRoot)
    declared = _typed(tmp_path, STUB_FIELD_READS)
    assert {type(found) for found in declared.values()} == {UndeclaredType}
    assert {found.reason for found in declared.values()} == {
        "no typeshed stubs installed for `ast`"
    }


@pytest.mark.parametrize(
    ("condition", "taken"),
    (
        pytest.param("sys.version_info >= (3, 0)", "kept", id="version at least"),
        pytest.param("sys.version_info < (3, 0)", "other", id="version below"),
        pytest.param(f"sys.platform == {sys.platform!r}", "kept", id="this platform"),
        pytest.param(f"sys.platform != {sys.platform!r}", "other", id="another platform"),
        pytest.param(
            f"sys.version_info >= (3, 0) and sys.platform != {sys.platform!r}",
            "other",
            id="conjunction",
        ),
    ),
)
def test_stub_branches_resolve_against_the_running_interpreter(condition, taken):
    resolved = RunningInterpreterBranches().visit(
        ast.parse(
            f"class Holder:\n    if {condition}:\n        kept: int\n    else:\n        other: int\n"
        )
    )
    (holder,) = resolved.body
    (member,) = holder.body
    assert isinstance(member, ast.AnnAssign) and ast.unparse(member.target) == taken


def test_an_empty_resolved_branch_leaves_a_body():
    resolved = RunningInterpreterBranches().visit(
        ast.parse("class Holder:\n    if sys.version_info < (3, 0):\n        old: int\n")
    )
    (holder,) = resolved.body
    assert [type(statement) for statement in holder.body] == [ast.Pass]


def test_an_undecidable_branch_keeps_both_arms():
    resolved = RunningInterpreterBranches().visit(
        ast.parse("if FLAG:\n    a: int\nelse:\n    b: int\n")
    )
    (branch,) = resolved.body
    assert isinstance(branch, ast.If)
