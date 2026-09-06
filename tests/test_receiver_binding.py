"""Receiver provenance must constrain both calls and escaping method values."""

import ast
from dataclasses import fields
from pathlib import Path
import pickle
from textwrap import indent

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceDeclaredCallArgumentsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository


def module_for(source: str) -> ParsedModule:
    return ParsedModule(
        path=Path("receiver.py"),
        module_name="receiver",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def source_for(body: str) -> str:
    return (
        "class _Other:\n"
        "    def method(self, value): return 'other'\n"
        "class _Owner:\n"
        "    def method(self, value): return 'owner'\n"
        "    def run(self):\n" + indent(body, "    ")
    )


@pytest.mark.parametrize(
    "use",
    (
        "    return self.method(7)\n",
        "    callback = self.method\n    return callback(7)\n",
    ),
)
def test_rebound_receiver_does_not_select_the_original_method(use: str) -> None:
    source = source_for("    self = _Other()\n" + use)
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run() == "other"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    assert resolution.resolved_call is None
    participants = frozenset(("receiver._Owner.method", "receiver._Other.method"))
    assert (
        resolution.target_resolution.candidate_symbols_within(participants)
        == participants
    )
    proof = repository.callable_component_authority_proof(
        {symbol: frozenset(("value",)) for symbol in participants},
        frozenset(),
    )
    assert set(proof.unresolved_consumer_symbols) == participants
    assert not proof.is_closed
    if "callback" in use:
        assert repository.callable_escapes_for("receiver._Other.method")


@pytest.mark.parametrize(
    "prefix",
    (
        "",
        "    self = self\n",
        "    original = self\n    self = original\n",
    ),
)
def test_entry_receiver_aliases_preserve_nominal_lookup(prefix: str) -> None:
    source = source_for(prefix + "    return self.method(7)\n")
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run() == "owner"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    assert (
        resolution.target_resolution.declaration.identity.symbol
        == "receiver._Owner.method"
    )
    assert resolution.target_resolution.candidate_symbols_within(
        frozenset(("receiver._Owner.method", "receiver._Other.method"))
    ) == frozenset(("receiver._Owner.method",))


def test_target_capture_precedes_argument_receiver_rebinding() -> None:
    source = source_for(
        "    before = self.method((self := _Other()))\n"
        "    return before, self.method(7)\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run() == ("owner", "other")
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    first, second = (
        repository.resolve_function_call(context, call)
        for call in context.flow.calls
        if call.target.terminal_name == "method"
    )
    assert (
        first.target_resolution.declaration.identity.symbol == "receiver._Owner.method"
    )
    assert second.resolved_call is None


def test_class_receiver_rebinding_is_not_an_original_class_method() -> None:
    source = (
        "class _Other:\n"
        "    @classmethod\n"
        "    def method(cls): return 'other'\n"
        "class _Owner:\n"
        "    @classmethod\n"
        "    def method(cls): return 'owner'\n"
        "    @classmethod\n"
        "    def run(cls):\n"
        "        cls = _Other\n"
        "        return cls.method()\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"].run() == "other"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    assert (
        repository.resolve_function_call(context, context.flow.calls[-1]).resolved_call
        is None
    )


@pytest.mark.parametrize("receiver", ("self.renderer", "type(self).renderer"))
def test_member_lookup_requires_the_entry_receiver(receiver: str) -> None:
    source = (
        "class _Renderer:\n    def render(self): return 'owner'\n"
        "class _Alternative:\n    def render(self): return 'other'\n"
        "class _Other:\n    renderer: _Alternative = _Alternative()\n"
        "class _Owner:\n"
        "    renderer: _Renderer = _Renderer()\n"
        "    def run(self):\n"
        "        self = _Other()\n"
        f"        return {receiver}.render()\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run() == "other"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    assert resolution.resolved_call is None
    symbols = frozenset(("receiver._Renderer.render", "receiver._Alternative.render"))
    assert resolution.target_resolution.candidate_symbols_within(symbols) == symbols


def test_declared_call_edit_does_not_ignore_an_unbounded_receiver_call() -> None:
    source = source_for(
        "    first = _Other.method(_Other(), 1)\n"
        "    self = _Other()\n"
        "    return first, self.method(2)\n"
    )
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(
                    file_path="receiver.py", qualname="_Owner.run"
                ),
                callee=SourceRewriteTarget(
                    file_path="receiver.py", qualname="_Other.method"
                ),
                arguments_source="_Other(), 3",
            ),
        )
    )
    with pytest.raises(ValueError, match="authority is unresolved"):
        plan.simulate(CodemodSourceSnapshot.from_modules((module_for(source),)))


def test_shadowed_runtime_type_lookup_has_no_proved_receiver_bound() -> None:
    source = (
        "class _Renderer:\n    def render(self): return 'owner'\n"
        "class _Alternative:\n    def render(self): return 'other'\n"
        "class _Other:\n    renderer: _Alternative = _Alternative()\n"
        "class _Owner:\n"
        "    renderer: _Renderer = _Renderer()\n"
        "    def run(self, type):\n"
        "        return type(self).renderer.render()\n"
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run(lambda value: namespace["_Other"]) == "other"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    assert resolution.resolved_call is None
    alternative = frozenset(("receiver._Alternative.render",))
    assert (
        resolution.target_resolution.candidate_symbols_within(alternative)
        == alternative
    )


@pytest.mark.parametrize(
    "binding",
    (
        "callback = self.method",
        "callback: object = self.method",
        "first = callback = self.method",
    ),
)
def test_exact_alias_reuses_the_nominal_source_read(binding: str) -> None:
    repository = CompactProductFlowRepository.from_modules(
        (module_for(source_for(f"    {binding}\n    return callback(7)\n")),)
    )
    flow = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"].flow
    for candidate in (flow, pickle.loads(pickle.dumps(flow))):
        for alias in candidate.exact_value_aliases:
            assert any(
                alias.source_use is use for use in candidate.callable_reference_uses
            )
            assert alias.source_position.dominates(alias.binding_mutation.position)
            assert alias.source is not None
            assert alias.source.root_name == "self"
            assert alias.source.attribute_path == ("method",)
            assert {field.name for field in fields(alias)} == {
                "source_use",
                "binding_mutation",
            }


@pytest.mark.parametrize(
    "body",
    (
        "    self = _Other()\n    if True:\n        callback = self.method\n    return callback(7)\n",
        "    self = _Other()\n    callback = self.method\n    return callback.__call__(7)\n",
        "    self = _Other()\n    callback = self.method\n    forwarded = callback\n    return forwarded(7)\n",
    ),
)
def test_unbounded_receiver_evidence_survives_alias_projection(body: str) -> None:
    source = source_for(body)
    namespace = {}
    exec(source, namespace)
    assert namespace["_Owner"]().run() == "other"
    repository = CompactProductFlowRepository.from_modules((module_for(source),))
    context = repository.flow_contexts_by_owner_symbol["receiver._Owner.run"]
    resolution = repository.resolve_function_call(context, context.flow.calls[-1])
    alternative = frozenset(("receiver._Other.method",))
    assert resolution.resolved_call is None
    assert (
        resolution.target_resolution.candidate_symbols_within(alternative)
        == alternative
    )
