"""Native completion dispatch never invents a Python value for a void transfer."""

import ast

import pytest

from nominal_refactor_advisor.native_compilation import (
    NativeCompletionResolverABC,
    NativeConstantValue,
    NativeDiscardValue,
    NativeItemStoreValue,
    NativePythonCompilation,
    NativeStackEffectABC,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan


class CompletionRecorder(NativeCompletionResolverABC):
    def __init__(self):
        self.values = []
        self.effects = []

    def _native_value_completion(self, value):
        self.values.append(value)

    def _native_stack_effect_completion(self, effect):
        self.effects.append(effect)


def test_completion_contract_requires_both_obligation_families():
    with pytest.raises(TypeError, match="abstract"):
        NativeCompletionResolverABC()


def test_values_and_effects_select_their_own_completion_contract():
    value = NativeConstantValue(2, (), None)
    discard = NativeDiscardValue(4, (value,))
    item = NativeItemStoreValue(6, (value, value, value))
    recorder = CompletionRecorder()
    for production in (value, discard, item):
        assert production.require_completion(recorder) is None
    assert recorder.values == [value]
    assert recorder.effects == [discard, item]


@pytest.mark.parametrize("expression", ("receiver[key] = value", "unproved()"))
def test_real_compiler_transfers_do_not_enter_python_result_completion(expression):
    source = expression + "\n"
    node = ast.parse(source).body[0]
    effect_type = (
        NativeItemStoreValue if isinstance(node, ast.Assign) else NativeDiscardValue
    )
    compilation = NativePythonCompilation(source, "completion_dispatch.py")
    span = SourceByteSpan.require_node(
        node.targets[0] if isinstance(node, ast.Assign) else node
    )
    receipt = compilation.return_after_effect(span, effect_type)
    effect = receipt.effect_for(span, effect_type)
    recorder = CompletionRecorder()
    for production in receipt.values:
        production.require_completion(recorder)
    assert any(original is effect for original in recorder.effects)
    assert all(
        not isinstance(original, NativeStackEffectABC) for original in recorder.values
    )
    assert all(
        isinstance(original, NativeStackEffectABC) for original in recorder.effects
    )
