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


def test_a_declared_fact_is_derived_once_and_released_with_its_parse():
    from dataclasses import dataclass

    from nominal_refactor_advisor.ast_tools import ParsedModuleFact
    from nominal_refactor_advisor.class_index import ModuleImportBindings

    derivations = []

    @dataclass(frozen=True)
    class LineCount(ParsedModuleFact):
        lines: int

        @classmethod
        def derive(cls, parsed_module):
            derivations.append(parsed_module.path)
            return cls(parsed_module.source.count("\n"))

    parsed = SourceModule(Path("m.py"), "m", "import os as o\nx = 1\n").parse()
    assert parsed.fact(LineCount) is parsed.fact(LineCount)
    assert parsed.fact(LineCount).lines == 2
    assert derivations == [Path("m.py")]
    assert "o" in parsed.fact(ModuleImportBindings).bindings
    tree = weakref.ref(parsed.module)
    del parsed
    gc.collect()
    assert tree() is None
