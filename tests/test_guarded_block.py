"""A block guarded by a type test becomes the tested class's override."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    GuardedBlockToOverrideOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)

SHAPES = (
    "class Shape:\n"
    "    def __init__(self, name: str) -> None:\n"
    "        self.name = name\n\n\n"
    "class Square(Shape):\n"
    "    def corners(self) -> int:\n"
    "        return 4\n\n\n"
    "class Circle(Shape):\n"
    "    pass\n\n\n"
    "def describe(shape: Shape, out: list[str]) -> None:\n"
    "    out.append(shape.name)\n"
    "    if isinstance(shape, Square):\n"
    "        count = shape.corners()\n"
    "        out.append(f'{shape.name} has {count}')\n"
    "    out.append('done')\n"
)

PROBE = (
    "from pkg.shapes import Circle, Square, describe\n"
    "out = []\n"
    "describe(Square('a'), out)\n"
    "describe(Circle('b'), out)\n"
    "print(out)\n"
)


def _run(root: Path) -> str:
    return subprocess.run(
        [sys.executable, "-c", PROBE], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def _package(root: Path, source: str) -> str:
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/shapes.py").write_text(source)
    return (root / "pkg/shapes.py").as_posix()


def _operation(path: str, source: str) -> GuardedBlockToOverrideOperation:
    line = next(number for number, text in enumerate(source.splitlines(), 1) if "if isinstance" in text)
    return GuardedBlockToOverrideOperation(
        target=SourceRewriteTarget(file_path=path, qualname="describe"),
        base=SourceRewriteTarget(file_path=path, qualname="Shape"),
        method_name="describe_corners",
        line=line,
    )


def _apply(root: Path, operation: GuardedBlockToOverrideOperation) -> None:
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root))
    simulation = RefactorRecipe("guarded").with_operation(operation).simulate(snapshot)
    assert simulation.is_clean, simulation.preflight_report
    for path, source in simulation.simulation.rewritten_sources.items():
        Path(path).write_text(source)


def test_the_block_becomes_the_tested_class_s_override_and_the_base_default(tmp_path: Path) -> None:
    path = _package(tmp_path, SHAPES)
    before = _run(tmp_path)
    _apply(tmp_path, _operation(path, SHAPES))
    source = Path(path).read_text()
    assert "isinstance" not in source
    assert "    shape.describe_corners(out=out)\n    out.append('done')\n" in source
    assert (
        "    def describe_corners(self, out) -> None:\n"
        "        count = self.corners()\n"
        "        out.append(f'{self.name} has {count}')\n"
    ) in source
    assert _run(tmp_path) == before == "['a', 'a has 4', 'done', 'b', 'done']\n"


def test_the_else_block_becomes_the_base_default(tmp_path: Path) -> None:
    source = SHAPES.replace(
        "        out.append(f'{shape.name} has {count}')\n",
        "        out.append(f'{shape.name} has {count}')\n    else:\n        out.append('round')\n",
    )
    path = _package(tmp_path, source)
    before = _run(tmp_path)
    _apply(tmp_path, _operation(path, source))
    assert _run(tmp_path) == before == "['a', 'a has 4', 'done', 'b', 'round', 'done']\n"
    assert "        out.append('round')\n" in Path(path).read_text().split("class Square")[0]


@pytest.mark.parametrize(
    ("change", "refusal"),
    (
        (("        count = shape.corners()\n", "        count = shape.corners() + offset\n"),
         "reads the function's locals"),
        (("        count = shape.corners()\n", "        return\n"), "returns or yields"),
        (("    out.append('done')\n", "    out.append(str(count))\n"), "which the function also uses outside it"),
        (("        count = shape.corners()\n", "        shape = Circle('c')\n        count = 4\n"),
         "rebinds"),
        (("        count = shape.corners()\n", "        count = shape.corners()\n        out = []\n"),
         "which the function also uses outside it"),
        (("        count = shape.corners()\n", "        count = [shape for shape in out]\n"), "rebinds"),
        (("        count = shape.corners()\n", "        count = super().corners()\n"), "super"),
    ),
)
def test_a_block_that_cannot_run_as_a_method_names_what_it_needs(
    tmp_path: Path, change: tuple[str, str], refusal: str
) -> None:
    source = SHAPES.replace(*change)
    if "offset" in source:
        source = source.replace("    out.append(shape.name)\n", "    offset = 1\n    out.append(shape.name)\n")
    path = _package(tmp_path, source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match=refusal):
        _operation(path, source).source_edits(snapshot)


def test_another_answer_on_the_way_is_refused(tmp_path: Path) -> None:
    """A class between the base and the tested class already answering the name would
    take the call: the old block never ran for it."""
    source = SHAPES.replace(
        "class Circle(Shape):\n    pass\n",
        "class Circle(Shape):\n    def describe_corners(self, out) -> None:\n        out.append('x')\n",
    )
    path = _package(tmp_path, source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    with pytest.raises(CodemodOperationPreflightError, match="declares 'describe_corners'"):
        _operation(path, source).source_edits(snapshot)


def test_closures_and_comprehensions_in_the_block_read_the_new_receiver(tmp_path: Path) -> None:
    source = SHAPES.replace(
        "        out.append(f'{shape.name} has {count}')\n",
        "        out.extend(f'{shape.name}{n}' for n in range(count))\n"
        "        out.append((lambda: shape.name.upper())())\n",
    )
    path = _package(tmp_path, source)
    before = _run(tmp_path)
    _apply(tmp_path, _operation(path, source))
    assert "lambda: self.name.upper()" in Path(path).read_text()
    assert _run(tmp_path) == before == "['a', 'a0', 'a1', 'a2', 'a3', 'A', 'done', 'b', 'done']\n"


def test_the_override_s_module_imports_what_the_block_reads(tmp_path: Path) -> None:
    """The block moves to the tested class's module: the names it reads there are
    imported the way the function's module binds them (move_member's import carry)."""
    shapes = SHAPES.split("def describe")[0].rstrip() + "\n"
    describe = (
        "import json\n\nfrom pkg.shapes import Shape, Square\n\n\n"
        + "def describe" + SHAPES.split("def describe")[1].replace(
            "        out.append(f'{shape.name} has {count}')\n",
            "        out.append(json.dumps([shape.name, count]))\n",
        )
    )
    path = _package(tmp_path, shapes)
    (tmp_path / "pkg/describe.py").write_text(describe)
    probe = PROBE.replace("from pkg.shapes import Circle, Square, describe", "from pkg.shapes import Circle, Square\nfrom pkg.describe import describe")
    run = lambda: subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    before = run()
    operation = _operation((tmp_path / "pkg/describe.py").as_posix(), describe)
    _apply(tmp_path, GuardedBlockToOverrideOperation(
        target=operation.target, base=SourceRewriteTarget(file_path=path, qualname="Shape"),
        method_name=operation.method_name, line=operation.line,
    ))
    assert "import json" in Path(path).read_text()
    assert run() == before == '[\'a\', \'["a", 4]\', \'done\', \'b\', \'done\']\n'


@pytest.mark.parametrize(
    ("imports", "annotation"),
    (
        ("from pkg import messages\n", "messages.Message"),
        ("from pkg.messages import Message\n", "Message"),
    ),
)
def test_a_re_exported_annotation_names_its_declaring_class(
    tmp_path: Path, imports: str, annotation: str
) -> None:
    """`messages.Message` and `from pkg.messages import Message`, where `pkg.messages`
    imports `Message` from `pkg.message`, both name `pkg.message.Message` (Textual's
    `Driver.process_message(self, message: messages.Message)`)."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/message.py").write_text(
        "class Message:\n    pass\n\n\nclass Close(Message):\n    pass\n"
    )
    (tmp_path / "pkg/messages.py").write_text("from pkg.message import Close, Message\n")
    driver = (
        f"{imports}from pkg.message import Close\n\n\n"
        f"def process(message: {annotation}, out: list[str]) -> None:\n"
        "    if isinstance(message, Close):\n"
        "        out.append('closed')\n"
        "        out.append(type(message).__name__)\n"
    )
    (tmp_path / "pkg/driver.py").write_text(driver)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    path = (tmp_path / "pkg/driver.py").as_posix()
    operation = GuardedBlockToOverrideOperation.at_site(
        snapshot, SourceRewriteTarget(file_path=path, qualname="process"), 6, "on_process"
    )
    assert operation.base.qualname == "Message"
    assert operation.base.file_path == (tmp_path / "pkg/message.py").as_posix()
    _apply(tmp_path, operation)
    assert "    message.on_process(out=out)\n" in Path(path).read_text()


@pytest.mark.parametrize(
    ("loop", "outcome"),
    (
        ("def describe_all(shapes: list[Shape], out: list[str]) -> None:\n    for shape in shapes:\n", None),
        (
            "def describe_all(shapes: list[Shape | None], out: list[str]) -> None:\n    for shape in shapes:\n",
            "whose values are not all instances",
        ),
        (
            "def describe_all(shapes: list[Shape | int], out: list[str]) -> None:\n    for shape in shapes:\n",
            "which is not one class the scan declares",
        ),
    ),
)
def test_a_local_subject_takes_the_type_checker_s_class(
    tmp_path: Path, loop: str, outcome: str | None
) -> None:
    """A loop variable has no annotation; the type checker's type at the test is its
    declared class, and a value the old test skipped that has no method refuses."""
    source = SHAPES.split("def describe")[0] + loop + (
        "        if isinstance(shape, Square):\n"
        "            out.append(f'{shape.name} has {shape.corners()}')\n"
    )
    path = _package(tmp_path, source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    line = next(number for number, text in enumerate(source.splitlines(), 1) if "if isinstance" in text)
    target = SourceRewriteTarget(file_path=path, qualname="describe_all")
    if outcome is not None:
        with pytest.raises(ValueError, match=outcome):
            GuardedBlockToOverrideOperation.at_site(snapshot, target, line, "describe_corners")
        return
    operation = GuardedBlockToOverrideOperation.at_site(snapshot, target, line, "describe_corners")
    assert operation.base.qualname == "Shape"
    _apply(tmp_path, operation)
    probe = "from pkg.shapes import Circle, Square, describe_all\nout = []\ndescribe_all([Square('a'), Circle('b')], out)\nprint(out)\n"
    assert subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout == "['a has 4']\n"
    assert "        shape.describe_corners(out=out)\n" in Path(path).read_text()
