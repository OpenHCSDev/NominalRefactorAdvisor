"""Subclass hooks belong to type-construction inputs, not global name writes."""

import ast

import pytest

from test_source_function_result import execution, native


@pytest.mark.parametrize(
    "body",
    (
        "global __init_subclass__\n__init_subclass__ = None",
        "global __init_subclass__\ndef __init_subclass__(cls):\n    raise RuntimeError('global function is not a class hook')",
    ),
)
@pytest.mark.parametrize(
    "descendants",
    (
        "class Child(Base): pass\n",
        "class Middle(Base): pass\nclass Child(Middle): pass\n",
        "class Left(Base): pass\nclass Right(Base): pass\nclass Child(Left, Right): pass\n",
    ),
)
def test_global_binding_does_not_invent_a_local_subclass_hook(body, descendants):
    source = (
        "class Base:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
        + descendants
        + "result = Child()\n"
    )
    native(
        source
        + "assert type(result) is Child\nassert '__init_subclass__' not in vars(Base)\n"
    )
    environment = execution(source)
    environment.require_call(environment.module.module.body[-1].value)


@pytest.mark.parametrize("decorator", ("", "    @classmethod\n"))
def test_actual_member_hook_remains_unproved_and_reports_its_owner(decorator):
    source = (
        "events = []\nclass Base:\n"
        + decorator
        + "    def __init_subclass__(cls):\n        events.append('hook')\n"
        + "class Child(Base): pass\n"
    )
    native(source + "assert events == ['hook']\n")
    environment = execution(source)
    with pytest.raises(ValueError, match="Base.*unproved.*__init_subclass__"):
        environment.require_class_creation(environment.module.module.body[-1])


def test_old_construction_input_does_not_hide_a_later_hook_write():
    source = (
        "class Base: pass\n"
        "Base.__init_subclass__ = classmethod(lambda cls: None)\n"
        "class Child(Base): pass\n"
    )
    native(source)
    environment = execution(source)
    base = environment.module.module.body[0]
    assert isinstance(base, ast.ClassDef)
    environment.require_class_creation(base)
    with pytest.raises(ValueError):
        environment.require_class_creation(environment.module.module.body[-1])
