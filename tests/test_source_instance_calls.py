"""Source-class identity and default instance creation are different proofs."""

import ast
from dataclasses import fields, replace

import pytest

from nominal_refactor_advisor.native_call import CallAuthority
from test_source_function_result import execution, native


@pytest.mark.parametrize(
    "declaration",
    (
        "class Payload: pass\n",
        "class Base: pass\nclass Payload(Base): pass\n",
        "class Left: pass\nclass Right: pass\nclass Payload(Left, Right): pass\n",
        "class Payload:\n    def __getattribute__(self, name):\n        raise RuntimeError('not during creation')\n",
        "class Payload:\n    global __new__\n    __new__ = None\n",
        "class Payload:\n    __abstractmethods__ = ('run',)\n",
        "class Payload:\n    __dict__ = None\n",
        "class Payload:\n    __weakref__ = None\n",
        "class Payload:\n    __class__ = None\n",
        "class Payload:\n    __mro__ = None\n",
    ),
)
def test_default_source_instance_retains_original_call_without_fabricated_identity(
    declaration,
):
    source = declaration + "result = Payload()\n"
    native(source + "assert type(result) is Payload\n")
    environment = execution(source)
    node = environment.module.module.body[-1].value
    environment.require_call(node)
    result = environment.capture_value(node)
    assert isinstance(result, CallAuthority)
    assert tuple(field.name for field in fields(result)) == ("environment", "operation")
    context, call = environment.source_call(node)
    assert result.operation is environment.source_operation(context, call)
    result.require_closed()
    result.require_class_installation()
    with pytest.raises(ValueError):
        result.source_definition()
    with pytest.raises(ValueError):
        result.require_release()


@pytest.mark.parametrize(
    "declaration",
    (
        "class Payload:\n    def __init__(self): pass\n",
        "class Base:\n    def __init__(self): pass\nclass Payload(Base): pass\n",
        "class Payload:\n    def __new__(cls): return object()\n",
        "class Meta(type):\n    def __call__(cls): return object()\nclass Payload(metaclass=Meta): pass\n",
    ),
)
def test_source_constructor_callbacks_remain_unproved(declaration):
    source = declaration + "result = Payload()\n"
    native(source)
    environment = execution(source)
    with pytest.raises(ValueError):
        environment.require_call(environment.module.module.body[-1].value)


@pytest.mark.parametrize("arguments", ("None", "flag=None", "*()", "**{}"))
def test_default_constructor_does_not_invent_argument_acceptance(arguments):
    environment = execution(f"class Payload: pass\nresult = Payload({arguments})\n")
    with pytest.raises(ValueError):
        environment.require_call(environment.module.module.body[-1].value)


@pytest.mark.parametrize("hook", (False, True))
def test_instance_installation_keeps_its_own_native_hook_obligation(hook):
    declaration = (
        "class Payload:\n    def __set_name__(self, owner, name): pass\n"
        if hook
        else "class Payload: pass\n"
    )
    source = declaration + "class Owner:\n    item = Payload()\n"
    native(source + "assert type(Owner.item) is Payload\n")
    environment = execution(source)
    owner = environment.module.module.body[-1]
    call_node = next(node for node in ast.walk(owner) if isinstance(node, ast.Call))
    environment.require_call(call_node)
    if hook:
        with pytest.raises(ValueError):
            environment.require_class_creation(owner)
    else:
        environment.require_class_creation(owner)


def test_equal_call_copy_cannot_supply_source_instance_evidence():
    environment = execution("class Payload: pass\nresult = Payload()\n")
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    with pytest.raises(ValueError):
        environment.call_authority(context, replace(call))


@pytest.mark.parametrize(
    "change",
    (
        "Payload.__init__ = poison",
        "setattr(Payload, '__init__', poison)",
        "Base.__init__ = poison",
        "saved = Payload\nsaved.__init__ = poison",
    ),
)
def test_creation_receipt_does_not_certify_class_state_after_mutation(change):
    source = (
        "class Base: pass\nclass Payload(Base): pass\n"
        "def poison(self): pass\n"
        f"{change}\nresult = Payload()\n"
    )
    native(source + "assert type(result) is Payload\n")
    environment = execution(source)
    with pytest.raises(ValueError):
        environment.require_call(environment.module.module.body[-1].value)
