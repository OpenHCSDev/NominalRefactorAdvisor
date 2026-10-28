"""Native descriptor binding uses Python's constructor, not guessed arities."""

import builtins
from itertools import product

import pytest

from nominal_refactor_advisor.native_call import NativeDescriptorCall
from test_native_namespace_calls import execution

SHAPES = (
    *(", ".join("None" for _ in range(count)) for count in range(6)),
    "doc=None",
    "fget=None",
    "fset=None",
    "fdel=None",
    "function=None",
    "unknown=None",
    "None, doc=None",
    "None, fget=None",
    "None, unknown=None",
    "fget=None, fset=None, fdel=None, doc=None",
)


def _accepted(constructor, arguments):
    """Whether the real builtin constructor accepts these arguments."""
    try:
        eval(
            f"{constructor}({arguments})", {constructor: getattr(builtins, constructor)}
        )
    except TypeError:
        return False
    return True


@pytest.mark.parametrize(
    "constructor,arguments",
    [
        # An accepted construction must be proved natively; a rejected one is
        # refused before any native proof.
        pytest.param(
            constructor,
            arguments,
            marks=pytest.mark.native_proof if _accepted(constructor, arguments) else (),
        )
        for constructor, arguments in product(
            ("property", "classmethod", "staticmethod"), SHAPES
        )
    ],
)
def test_binding_acceptance_matches_actual_native_constructor(constructor, arguments):
    expression = f"{constructor}({arguments})"
    valid = _accepted(constructor, arguments)
    environment = execution(f"result = {expression}\n")
    node = environment.module.module.body[-1].value
    if valid:
        result = environment.capture_value(node)
        assert isinstance(result, NativeDescriptorCall)
        result.require_closed()
        result.require_class_installation()
        # No analyzer-created descriptor is exported as the target's identity.
        assert not isinstance(result, (property, classmethod, staticmethod))
    else:
        with pytest.raises(ValueError):
            environment.capture_value(node).require_closed()


@pytest.mark.parametrize("arguments", ("*(None,)", "**{'fget': None}"))
def test_expansion_does_not_turn_into_a_guessed_argument_count(arguments):
    environment = execution(f"result = property({arguments})\n")
    with pytest.raises(ValueError):
        environment.capture_value(
            environment.module.module.body[-1].value
        ).require_closed()


@pytest.mark.parametrize(
    "arguments,valid",
    (("", False), pytest.param("None", True, marks=pytest.mark.native_proof)),
)
def test_qualified_import_alias_retains_actual_constructor_binding(arguments, valid):
    environment = execution(
        "from builtins import staticmethod as declared\n"
        f"result = declared({arguments})\n"
    )
    node = environment.module.module.body[-1].value
    if valid:
        environment.capture_value(node).require_closed()
    else:
        with pytest.raises(ValueError):
            environment.capture_value(node).require_closed()
