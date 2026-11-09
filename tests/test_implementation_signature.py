"""An implementation signature changes exactly when code a declaration reaches changes."""

import importlib
import sys
from pathlib import Path

from nominal_refactor_advisor.implementation_identity import implementation_signature

HELPERS = '''
ALLOWED = frozenset({"a", "b", "c"})


def used(value):
    return value in ALLOWED


def unused(value):
    return value * 2
'''

DETECTOR = '''
from sigpkg.helpers import used


class Detector:
    def detect(self, value):
        return used(value)
'''


def _signature(tmp_path: Path, helpers: str, detector: str = DETECTOR) -> str:
    package = tmp_path / "sigpkg"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text(helpers)
    (package / "detector.py").write_text(detector)
    for name in ("sigpkg.detector", "sigpkg.helpers", "sigpkg"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("sigpkg.detector")
        return implementation_signature((module.Detector,))
    finally:
        sys.path.remove(str(tmp_path))


def test_an_edit_to_unreached_code_keeps_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    after = _signature(tmp_path, HELPERS.replace("value * 2", "value * 3"))
    assert after == before


def test_moving_reached_code_down_the_file_keeps_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    after = _signature(tmp_path, "\n\n\n# a comment above\n" + HELPERS)
    assert after == before


def test_an_edit_to_a_reached_function_changes_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    after = _signature(tmp_path, HELPERS.replace("return value in ALLOWED", "return value not in ALLOWED"))
    assert after != before


def test_an_edit_to_a_reached_constant_changes_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    # A different length: an edit of the same size inside one mtime second would
    # reload the stale bytecode, and the test would compare the old code to itself.
    after = _signature(tmp_path, HELPERS.replace('"c"', '"cd"'))
    assert after != before


def test_an_edit_to_the_declaration_itself_changes_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    after = _signature(tmp_path, HELPERS, DETECTOR.replace("return used(value)", "return not used(value)"))
    assert after != before


def test_a_global_read_only_inside_a_nested_function_is_reached(tmp_path: Path) -> None:
    nested = DETECTOR.replace(
        "        return used(value)",
        "        check = lambda item: used(item)\n        return check(value)",
    )
    before = _signature(tmp_path, HELPERS, nested)
    after = _signature(tmp_path, HELPERS.replace("return value in ALLOWED", "return value not in ALLOWED"), nested)
    assert after != before


def test_moving_the_declaration_down_its_file_keeps_the_signature(tmp_path: Path) -> None:
    before = _signature(tmp_path, HELPERS)
    moved = DETECTOR.replace("class Detector:", "# a comment above the class\n\n\nclass Detector:")
    after = _signature(tmp_path, HELPERS, moved)
    assert after == before
