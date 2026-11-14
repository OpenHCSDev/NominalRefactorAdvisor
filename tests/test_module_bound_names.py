"""A parse says once which names its module binds."""
from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule

SOURCE = (
    "import os\n"
    "from a import b as c\n"
    "X = 1\n"
    "def f(p):\n    inner = 1\n\n"
    "class K:\n    field = 1\n\n"
    "if X:\n    in_if = 1\nelse:\n    in_else = 2\n"
    "try:\n    in_try = 1\nexcept ValueError as caught:\n    in_handler = 2\n"
    "for looped in ():\n    pass\n"
    "with open('x') as handle:\n    pass\n"
    "squares = [n * n for n in range(3)]\n"
    "(walrus := 1)\n"
)


def test_a_module_binds_what_its_own_scope_binds() -> None:
    module = ParsedModule(
        path=Path("pkg", "names.py"),
        module_name="pkg.names",
        module=ast.parse(SOURCE),
        source=SOURCE,
    )

    assert module.bound_names == {
        "os", "c", "X", "f", "K",
        # every branch of a compound statement runs in the module's scope
        "in_if", "in_else", "in_try", "caught", "in_handler", "looped", "handle",
        "squares", "walrus",
    }
    # a parameter, a function's local, a class's member and a comprehension's variable
    # belong to their own scopes
    assert module.bound_names.isdisjoint({"p", "inner", "field", "n"})
    assert module.bound_names is module.bound_names
