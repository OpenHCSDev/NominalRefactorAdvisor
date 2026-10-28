"""Facts derived from a parse are held by it and released with it (#89)."""

import gc
import weakref
from pathlib import Path

from nominal_refactor_advisor.ast_tools import SourceModule


def test_a_parse_s_syntax_facts_are_released_with_the_parse():
    parsed = SourceModule(Path("m.py"), "m", "class A:\n    def f(self):\n        return 1\n").parse()
    tree = weakref.ref(parsed.module)
    assert parsed.syntax_index is parsed.syntax_index
    assert [name for name, _ in parsed.syntax_index.named_functions] == ["A.f"]
    del parsed
    gc.collect()
    assert tree() is None
