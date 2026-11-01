"""Original header/slot joins do not imply current storage or relocation."""

from copy import copy
from typing import cast

import pytest

from nominal_refactor_advisor.lexical_bindings import (
    DefaultedParameterSource,
    FunctionParameterSource,
)
from nominal_refactor_advisor.native_compilation import (
    NativeConstantTupleElement,
    NativeConstantValue,
)
from nominal_refactor_advisor.source_execution import SourceFunctionDefault
from test_source_call_binding import source_call
from test_source_function_installation import module_function

pytestmark = pytest.mark.native_proof


def joined_default(header: str, name: str) -> SourceFunctionDefault:
    environment, function = module_function(f"def chosen({header}): pass\n")
    source = next(
        source
        for source in FunctionParameterSource.from_arguments(function.node.args)
        if source.argument.arg == name
    )
    assert isinstance(source, DefaultedParameterSource)
    return SourceFunctionDefault(
        function, source, environment.required_prefix(environment.entry.context, None)
    )


@pytest.mark.parametrize(
    "header,name,expected",
    (
        ("left=0, right=0", "left", 0),
        ("left=0, right=0", "right", 0),
        ("*, option=None", "option", None),
        ("first, /, second=0, *, option=None", "second", 0),
        ("first, /, second=0, *, option=None", "option", None),
        ("value=(0, True)", "value", (0, True)),
    ),
)
def test_original_default_joins_its_source_read_and_native_slot(
    header: str, name: str, expected: object
) -> None:
    join = joined_default(header, name)
    result = join.result()
    result.require_constant_contents(expected)
    assert join.operand.parameter_name == name
    join.store.require_value(join.operand.storage)
    # A matching endpoint is not the admitted original cut.
    with pytest.raises(ValueError):
        SourceFunctionDefault(join.function, join.parameter, copy(join.prefix)).result()


def test_equal_contents_do_not_join_a_different_positional_slot() -> None:
    join = joined_default("left=0, right=0", "left")
    projection = NativeConstantTupleElement(cast(NativeConstantValue, join.operand.storage), 1)
    with pytest.raises(ValueError, match="original storage slot"):
        projection.resolve_operand(join)


def test_matching_foreign_syntax_does_not_supply_the_original_default() -> None:
    original = joined_default("value=0", "value")
    foreign = joined_default("value=0", "value")
    with pytest.raises(ValueError, match="original function header"):
        SourceFunctionDefault(
            original.function, foreign.parameter, original.prefix
        ).result()


def test_new_join_does_not_replace_a_broader_existing_parameter_entry() -> None:
    _, _, authority = source_call("value=0.5", "chosen()")
    assert authority.initial_entries["value"].native_type is float
    # Existing lifetime refusal is independent of source parameter entry.
    with pytest.raises(ValueError, match="Native instance lifetime remains unproved"):
        authority.require_closed()
    source = FunctionParameterSource.from_arguments(authority.callee.node.args)[0]
    assert isinstance(source, DefaultedParameterSource)
    with pytest.raises(ValueError, match="Native original tuple elements remain unproved"):
        SourceFunctionDefault(
            authority.callee, source, authority.caller_prefix
        ).result()
