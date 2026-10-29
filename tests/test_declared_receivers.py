"""Declared receiver typing: each form resolves from declarations alone, or names why not."""

from __future__ import annotations

import ast
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
from nominal_refactor_advisor.declared_receivers import (
    DeclaredReceiverTyping,
    ReceiverPathBuilder,
    UndeclaredType,
)
from nominal_refactor_advisor.scan_cache import ScanCache

MODELS = """
from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Self, TypeVar

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
        "from pkg.models import Board, Shape, ShapeIndex, Square, make_board, unannotated_board\n"
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
            "        shape = board.pair[1]\n"
            "    shape.marker\n",
            "rebound to a different type",
            id="rebound to a different type",
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
