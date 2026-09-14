"""Receiver bounds need actual capture, not a selected source declaration."""

import ast
from dataclasses import fields, replace
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.product_flow import CompactFlowValue, CompactItemTarget
from nominal_refactor_advisor.product_flow_authority import (
    CompactProductFlowRepository,
    ProductFlowRepository,
    SourceProductFlowRepository,
)


def module(source):
    return ParsedModule(
        Path("/repo/receiver.py"), "receiver", False, ast.parse(source), source
    )


@pytest.mark.parametrize(
    "source, name",
    (
        ("class Owner: pass\nOwner.changed = 1\n", "Owner"),
        ("def callback(): pass\ncallback.changed = 1\n", "callback"),
        ("class Owner: pass\nalias = Owner\nalias.changed = 1\n", "Owner"),
        ("def callback(): pass\nalias = callback\nalias.changed = 1\n", "callback"),
        (
            "class Outer:\n    global Owner\n    class Owner: pass\nOwner.changed = 1\n",
            "Outer.Owner",
        ),
        (
            "class Outer:\n    global callback\n    def callback(): pass\ncallback.changed = 1\n",
            "Outer.callback",
        ),
    ),
)
def test_source_capture_proves_plain_receiver_while_compact_queries_stay_open(
    source, name
):
    receiver = ast.unparse(ast.parse(source).body[-1].targets[0].value)
    native = subprocess.run(
        [sys.executable, "-c", source + f"print({receiver}.__name__)\n"],
        check=True,
        capture_output=True,
        text=True,
    )
    # A global-directed declaration retains its source producer even though
    # Python installs it (and gives it a runtime qualname) at module scope.
    assert native.stdout.strip() == name.rsplit(".", 1)[-1]
    repository = SourceProductFlowRepository.from_modules((module(source),))
    context = repository.module_flow_contexts["receiver"]
    mutation = context.flow.mutations[-1]
    participants = frozenset(
        (f"receiver.{name}", "receiver.Unrelated", "receiver.Other")
    )
    assert mutation.resolve(repository, context).candidate_symbols_within(
        participants
    ) == {f"receiver.{name}"}
    compact = CompactProductFlowRepository(
        repository.product_projections, repository.class_projections
    )
    assert (
        mutation.resolve(compact, context).candidate_symbols_within(participants)
        == participants
    )
    assert (
        SourceProductFlowRepository._receiver_mutation_resolution
        is CompactProductFlowRepository._receiver_mutation_resolution
    )


@pytest.mark.parametrize(
    "declaration",
    (
        "def replace(original): return Product\n@replace\nclass Owner: pass\n",
        "class Meta(type):\n    def __new__(meta, name, bases, namespace): return Product\nclass Owner(metaclass=Meta): pass\n",
        "def replace(original): return Product\n@replace\ndef Owner(): pass\n",
    ),
)
def test_replacing_construction_never_inherits_a_lexical_receiver_bound(declaration):
    source = (
        "class Product: pass\n"
        + declaration
        + "Owner.changed = 1\nassert Product.changed == 1\n"
    )
    subprocess.run([sys.executable, "-c", source], check=True)
    repository = SourceProductFlowRepository.from_modules((module(source),))
    context = repository.module_flow_contexts["receiver"]
    mutation = context.flow.mutations[-1]
    participants = frozenset(("receiver.Product", "receiver.Owner"))
    assert (
        mutation.resolve(repository, context).candidate_symbols_within(participants)
        == participants
    )


def test_repository_payloads_have_no_independently_supplied_source_projections():
    assert [field.name for field in fields(SourceProductFlowRepository)] == ["modules"]
    assert [field.name for field in fields(CompactProductFlowRepository)] == [
        "product_projections",
        "class_projections",
    ]
    with pytest.raises(TypeError):
        ProductFlowRepository()
    state = CodemodSourceSnapshot.from_source_mapping(
        {"/repo/receiver.py": "class Owner: pass\n"}
    )
    assert state.module_binding_proof is state.product_flow_repository


def test_source_capture_requires_actual_context_and_original_use():
    parsed = module("class Owner: pass\nOwner.changed = 1\n")
    repository = SourceProductFlowRepository.from_modules((parsed,))
    context = repository.module_flow_contexts["receiver"]
    use = context.flow.mutations[-1].target.receiver_use
    repository.captured_value(CompactFlowValue(context, use)).require_closed()
    with pytest.raises(ValueError, match="unique original operation"):
        repository.captured_value(CompactFlowValue(context, replace(use)))
    other = SourceProductFlowRepository.from_modules((parsed,))
    with pytest.raises(ValueError, match="unique original operation"):
        other.captured_value(CompactFlowValue(context, use))


def test_duplicate_module_names_do_not_select_a_runtime_activation():
    first = module("class Owner: pass\nOwner.changed = 1\n")
    second = replace(first, path=Path("/elsewhere/receiver.py"))
    repository = SourceProductFlowRepository.from_modules((first, second))
    context = repository.sources[0].compact.flow_contexts[0]
    value = CompactFlowValue(context, context.flow.mutations[-1].target.receiver_use)
    with pytest.raises(ValueError, match="unique original source owner"):
        repository.captured_value(value)


@pytest.mark.parametrize(
    "source",
    (
        "data = {}\ndata['item'] = 1\n",
        "import builtins\ndata = dict(vars(builtins))\ndata['item'] = 1\n",
        "from __future__ import annotations\nclass Owner:\n    item: object = None\n",
    ),
)
def test_proved_dictionary_receiver_excludes_definition_targets_without_admitting_compact_facts(
    source,
):
    subprocess.run([sys.executable, "-c", source], check=True)
    repository = SourceProductFlowRepository.from_modules((module(source),))
    compact = CompactProductFlowRepository(
        repository.product_projections, repository.class_projections
    )
    ((context, mutation),) = (
        (context, mutation)
        for context in repository.flow_contexts
        for mutation in context.flow.mutations
        if isinstance(mutation.target, CompactItemTarget)
    )
    participants = frozenset(("receiver.Owner", "receiver.Other"))
    captured = repository.captured_value(
        CompactFlowValue(context, mutation.target.receiver_use)
    )
    captured.require_closed()
    assert captured.native_type is dict
    with pytest.raises(ValueError, match="Source definition identity"):
        captured.source_definition()
    assert not mutation.resolve(repository, context).candidate_symbols_within(
        participants
    )
    assert (
        mutation.resolve(compact, context).candidate_symbols_within(participants)
        == participants
    )


@pytest.mark.parametrize("annotation", (False, True))
def test_item_receiver_kind_cannot_be_inferred_from_annotation_syntax_or_a_local_name(
    annotation,
):
    prefix = (
        "from __future__ import annotations\n"
        "events = []\n"
        "class Meta(type):\n"
        "    def __setitem__(cls, key, value): events.append(key)\n"
        "class Target(metaclass=Meta): pass\n"
    )
    source = prefix + (
        "class Owner:\n    __annotations__ = Target\n    item: object = None\n"
        if annotation
        else "data = Target\ndata['item'] = object\n"
    )
    subprocess.run(
        [sys.executable, "-c", source + "assert events == ['item']\n"], check=True
    )
    repository = SourceProductFlowRepository.from_modules((module(source),))
    context, mutation = next(
        (context, mutation)
        for context in reversed(repository.flow_contexts)
        for mutation in reversed(context.flow.mutations)
        if isinstance(mutation.target, CompactItemTarget)
    )
    participants = frozenset(("receiver.Target", "receiver.Owner", "receiver.Other"))
    assert (
        mutation.resolve(repository, context).candidate_symbols_within(participants)
        == participants
    )
