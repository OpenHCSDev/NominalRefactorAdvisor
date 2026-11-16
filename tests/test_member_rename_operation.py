from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.refactors.member_rename import RenameClassMemberOperation

SHAPES = (
    "class Shape:\n"
    "    def area(self) -> float:\n"
    "        return 0.0\n\n"
    "    def describe(self) -> str:\n"
    "        return f'{type(self).__name__}:{self.area()}'\n\n\n"
    "class Square(Shape):\n"
    "    def __init__(self, side: float) -> None:\n"
    "        self.side = side\n\n"
    "    def area(self) -> float:\n"
    "        return self.side * self.side\n\n\n"
    "class Doubled(Square):\n"
    "    def area(self) -> float:\n"
    "        return 2 * super().area()\n\n\n"
    "class Room:\n"
    "    def area(self) -> float:\n"
    "        return 9.0\n"
)
USE = (
    "from pkg.shapes import Room, Shape, Square\n\n\n"
    "def total(shapes: list[Shape]) -> float:\n"
    "    return sum(shape.area() for shape in shapes)\n\n\n"
    "def report(square: Square, room: Room) -> str:\n"
    "    return f'{square.area()} {room.area()} {square.describe()}'\n"
)
SHAPES_PROBE = (
    "from pkg.shapes import Doubled, Room, Square; from pkg.use import report, total; "
    "print(total([Square(2), Doubled(1)]), report(Square(3), Room()))"
)
COLORS = (
    "from enum import Enum\n\n\n"
    "class Color(Enum):\n"
    "    RED = 'red'\n"
    "    GREEN = 'green'\n\n\n"
    "def warm(color: Color) -> bool:\n"
    "    return color is Color.RED\n"
)
COLORS_PROBE = "from pkg.colors import Color, warm; print(warm(Color('red')), warm(Color.GREEN), Color('red').value)"
POINTS = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass, replace\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Point:\n"
    "    x: int\n"
    "    y: int\n\n"
    "    def norm(self) -> int:\n"
    "        return abs(self.x) + abs(self.y)\n\n"
    "    def shifted(self) -> Point:\n"
    "        return replace(self, x=self.x + 1)\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Labeled(Point):\n"
    "    label: str = ''\n\n\n"
    "def build() -> Point:\n"
    "    return Labeled(x=1, y=2, label='a').shifted()\n"
)
POINTS_PROBE = "from pkg.points import Point, build; print(build().norm(), Point(3, 4).norm())"


def _write(root: Path, sources: dict[str, str]) -> None:
    (root / "pkg").mkdir()
    (root / "pkg/__init__.py").write_text("")
    for name, source in sources.items():
        (root / f"pkg/{name}.py").write_text(source)


def _probe(root: Path, program: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", program], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _rename(root: Path, module: str, qualname: str, member: str, new_name: str):
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    operation = RenameClassMemberOperation(
        target=SourceRewriteTarget(file_path=(root / f"pkg/{module}.py").as_posix(), qualname=qualname),
        member_name=member,
        new_name=new_name,
    )
    replayed = RefactorRecipeOperation.from_json_value(json_report_object(operation))
    return RefactorRecipe(recipe_id="rename").with_operation(replayed).simulate(snapshot)


def _apply(root: Path, result) -> None:
    assert result.is_clean is True
    for path, source in result.simulation.rewritten_sources.items():
        Path(path).write_text(source)


def _source(root: Path, module: str) -> str:
    return (root / f"pkg/{module}.py").read_text()


@pytest.mark.native_proof
def test_an_overridden_method_is_renamed_across_its_family_and_its_typed_reads(tmp_path: Path) -> None:
    _write(tmp_path, {"shapes": SHAPES, "use": USE})
    before = _probe(tmp_path, SHAPES_PROBE)

    _apply(tmp_path, _rename(tmp_path, "shapes", "Shape", "area", "surface"))

    shapes = _source(tmp_path, "shapes")
    assert shapes.count("def surface(self) -> float:") == 3
    assert "self.surface()" in shapes and "super().surface()" in shapes
    assert "class Room:\n    def area(self) -> float:" in shapes
    use = _source(tmp_path, "use")
    assert "shape.surface() for shape in shapes" in use
    assert "{square.surface()} {room.area()}" in use
    assert _probe(tmp_path, SHAPES_PROBE) == before


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "sources,new_name,needed",
    (
        ({"shapes": SHAPES, "use": USE + "\n\ndef loose(thing):\n    return thing.area()\n"},
         "surface", r"needs .*pkg/use\.py:13 reads 'area' on a receiver typed by no declaration"),
        ({"shapes": SHAPES, "use": USE}, "describe", r"needs another name: 'Shape' .* already binds 'describe'"),
        ({"shapes": SHAPES, "use": USE + "\n\ndef dynamic(shape: Shape) -> float:\n    return getattr(shape, 'area')()\n"},
         "surface", r"needs .*pkg/use\.py:13 reaches 'area' by a string"),
    ),
    ids=("untyped-receiver", "name-collision", "getattr-read"),
)
def test_a_rename_says_what_it_needs_instead_of_editing(
    tmp_path: Path, sources: dict[str, str], new_name: str, needed: str
) -> None:
    _write(tmp_path, sources)
    with pytest.raises(CodemodOperationPreflightError, match=needed) as raised:
        _rename(tmp_path, "shapes", "Shape", "area", new_name)
    assert "refus" not in str(raised.value)


@pytest.mark.native_proof
def test_a_class_outside_the_family_keeps_its_own_member_until_it_joins(tmp_path: Path) -> None:
    _write(tmp_path, {"shapes": SHAPES.replace("class Room:", "class Room(Shape):"), "use": USE})
    before = _probe(tmp_path, SHAPES_PROBE)

    _apply(tmp_path, _rename(tmp_path, "shapes", "Shape", "area", "surface"))

    assert _source(tmp_path, "shapes").count("def surface(self) -> float:") == 4
    assert "{square.surface()} {room.surface()}" in _source(tmp_path, "use")
    assert _probe(tmp_path, SHAPES_PROBE) == before


@pytest.mark.native_proof
def test_a_rename_from_below_the_declaring_class_needs_the_declaring_class(tmp_path: Path) -> None:
    _write(tmp_path, {"shapes": SHAPES, "use": USE})
    with pytest.raises(CodemodOperationPreflightError, match=r"needs the rename to start where dispatch starts: 'Shape'"):
        _rename(tmp_path, "shapes", "Square", "area", "surface")


@pytest.mark.native_proof
def test_an_enum_member_rename_updates_its_reads(tmp_path: Path) -> None:
    _write(tmp_path, {"colors": COLORS})
    before = _probe(tmp_path, COLORS_PROBE)

    _apply(tmp_path, _rename(tmp_path, "colors", "Color", "RED", "CRIMSON"))

    colors = _source(tmp_path, "colors")
    assert "    CRIMSON = 'red'\n" in colors and "color is Color.CRIMSON" in colors
    assert "RED" not in colors
    assert _probe(tmp_path, COLORS_PROBE) == before


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "colors,needed",
    (
        (COLORS.replace("from enum import Enum", "from enum import Enum, auto").replace("'red'", "auto()"),
         r"needs the member's value given: `auto\(\)` derives it from the name"),
        (COLORS + "\n\ndef named() -> Color:\n    return Color['RED']\n",
         r"needs .*pkg/colors\.py:14 shown not to read the member by its name 'RED'"),
    ),
    ids=("auto-value", "lookup-by-name"),
)
def test_an_enum_member_read_by_its_name_needs_input(tmp_path: Path, colors: str, needed: str) -> None:
    _write(tmp_path, {"colors": colors})
    with pytest.raises(CodemodOperationPreflightError, match=needed):
        _rename(tmp_path, "colors", "Color", "RED", "CRIMSON")


@pytest.mark.native_proof
def test_a_dataclass_field_rename_rewrites_every_proved_construction(tmp_path: Path) -> None:
    _write(tmp_path, {"points": POINTS})
    before = _probe(tmp_path, POINTS_PROBE)

    _apply(tmp_path, _rename(tmp_path, "points", "Point", "x", "east"))

    points = _source(tmp_path, "points")
    assert "    east: int\n" in points
    assert "abs(self.east) + abs(self.y)" in points
    assert "replace(self, east=self.east + 1)" in points
    assert "Labeled(east=1, y=2, label='a')" in points
    assert _probe(tmp_path, POINTS_PROBE) == before


@pytest.mark.native_proof
@pytest.mark.parametrize(
    "extra,needed",
    (
        ("\n\ndef spread(values: dict[str, int]) -> Point:\n    return Point(**values)\n",
         r"needs the keywords .*pkg/points\.py:\d+ unpacks with `\*\*`"),
        ("\n\nFACTORY = Point\n", r"needs the constructions behind .*pkg/points\.py:\d+"),
        ("\n\ndef keys(point: Point) -> list[str]:\n    from dataclasses import asdict\n    return list(asdict(point))\n",
         r"needs .*pkg/points\.py:\d+ shown: it reads the field names as strings"),
    ),
    ids=("unpacked-keywords", "class-as-value", "asdict"),
)
def test_a_dataclass_field_with_unseen_constructions_needs_them(tmp_path: Path, extra: str, needed: str) -> None:
    _write(tmp_path, {"points": POINTS + extra})
    with pytest.raises(CodemodOperationPreflightError, match=needed):
        _rename(tmp_path, "points", "Point", "x", "east")


GAUGES = (
    "class Gauge:\n"
    "    def __init__(self) -> None:\n"
    "        self._level = 0\n\n"
    "    @property\n"
    "    def level(self) -> int:\n"
    "        return self._level\n\n"
    "    @level.setter\n"
    "    def level(self, value: int) -> None:\n"
    "        self._level = value\n\n\n"
    "def bump(gauge: Gauge) -> int:\n"
    "    gauge.level = gauge.level + 1\n"
    "    return gauge.level\n"
)


@pytest.mark.native_proof
def test_a_property_with_a_setter_is_renamed_with_its_decorator_and_writes(tmp_path: Path) -> None:
    _write(tmp_path, {"gauges": GAUGES})
    program = "from pkg.gauges import Gauge, bump; g = Gauge(); bump(g); print(bump(g))"
    before = _probe(tmp_path, program)

    _apply(tmp_path, _rename(tmp_path, "gauges", "Gauge", "level", "reading"))

    gauges = _source(tmp_path, "gauges")
    assert gauges.count("def reading(self") == 2 and "@reading.setter" in gauges
    assert "gauge.reading = gauge.reading + 1" in gauges
    assert "self._level" in gauges
    assert _probe(tmp_path, program) == before
