"""Plain class creation uses the captured native root, not its source spelling."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _require_last_class(source):
    module = ParsedModule(
        Path("native_root.py"), "native_root", False, ast.parse(source), source
    )
    execution = SourceModuleExecution.from_module(module)
    node = module.module.body[-1]
    execution.require_class_creation(node)
    return execution.class_entry(node)


@pytest.mark.parametrize(
    "source",
    (
        "class Owner(object): pass\n",
        "import builtins\nclass Owner(builtins.object): pass\n",
        "from builtins import object as Root\nclass Owner(Root): pass\n",
        "Root = object\nobject = type\nclass Owner(Root): pass\n",
        "class Base(object): pass\nclass Owner(Base): pass\n",
        "class Base: pass\nclass Owner(Base, object): pass\n",
        "class Base(object): pass\nclass Left(Base): pass\n"
        "class Right(Base): pass\nclass Owner(Left, Right): pass\n",
    ),
)
def test_native_object_root_and_retained_aliases_are_admitted(source):
    namespace = {}
    exec(source, namespace)
    entry = _require_last_class(source)
    assert tuple(owner.node.name for owner in entry.mro_type.declarations) == tuple(
        owner.__name__ for owner in namespace["Owner"].__mro__[:-1]
    )


@pytest.mark.parametrize(
    "source, reason",
    (
        ("object = type\nclass Owner(object): pass\n", "required native declaration"),
        ("Root = type\nclass Owner(Root): pass\n", "required native declaration"),
        (
            "print('unproved earlier effect')\nclass Owner(object): pass\n",
            "unproved_execution_effects",
        ),
        (
            "class Base(object):\n    def __init_subclass__(cls): unknown()\n"
            "class Owner(Base): pass\n",
            "unproved native protocol hooks: __init_subclass__",
        ),
    ),
)
def test_native_root_admission_does_not_grant_other_protocols(source, reason):
    with pytest.raises(ValueError, match=reason):
        _require_last_class(source)


def test_inconsistent_native_root_order_remains_rejected():
    source = "class Base(object): pass\nclass Owner(object, Base): pass\n"
    with pytest.raises(TypeError):
        exec(source, {})
    with pytest.raises(ValueError, match="hierarchy remains unproved"):
        _require_last_class(source)
