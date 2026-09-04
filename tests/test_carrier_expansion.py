from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.carrier_expansion import (
    DeclaredCarrierExpansionAuthorityViolation,
    DeclaredCarrierExpansionBuilder,
)
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    CollapseDeclaredCarrierExpansionOperation,
    FindingRecipeSynthesisStatus,
    RefactorRecipe,
    RefactorRecipeOperation,
    SourceRewriteTarget,
    codemod_plan_from_findings,
)
from nominal_refactor_advisor.detectors import DeclaredCarrierExpansionDetector
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.models import ParameterThreadMetrics, SourceLocation
from nominal_refactor_advisor.patterns import PatternId


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _closed_expansion_source() -> str:
    return (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _Context:\n"
        "    first: object\n"
        "    second: object\n"
        "\n"
        "    @classmethod\n"
        "    def merge(cls, base) -> '_Context':\n"
        "        return base\n"
        "\n"
        "def _leaf(first, second):\n"
        "    return first, second\n"
        "\n"
        "def _middle(left, right):\n"
        "    return _leaf(left, right)\n"
        "\n"
        "def caller(base):\n"
        "    context = _Context.merge(base)\n"
        "    return _middle(context.first, context.second)\n"
    )


def test_builder_derives_field_mapping_from_bound_nominal_carrier() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.models",
                "class Context:\n"
                "    title: str\n"
                "    metrics: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n",
            ),
            _module(
                "pkg.worker",
                "from pkg.models import Context\n"
                "\n"
                "def consume(value, *, title=None, metrics=None):\n"
                "    return value, title, metrics\n"
                "\n"
                "def build(value, base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(\n"
                "        value,\n"
                "        title=context.title,\n"
                "        metrics=context.metrics,\n"
                "    )\n",
            ),
        )
    )

    assert len(builder.expansions) == 1
    expansion = builder.expansions[0]
    assert expansion.carrier_class_symbol == "pkg.models.Context"
    assert expansion.callee_symbol == "pkg.worker.consume"
    assert expansion.field_mapping == (
        ("title", "title"),
        ("metrics", "metrics"),
    )


def test_builder_rejects_untyped_or_rebound_carrier_values() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.open_values",
                "def make(base):\n"
                "    return base\n"
                "\n"
                "def consume(*, left=None, right=None):\n"
                "    return left, right\n"
                "\n"
                "def untyped(base):\n"
                "    context = make(base)\n"
                "    return consume(left=context.left, right=context.right)\n"
                "\n"
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def rebound(base, replacement):\n"
                "    context = Context.merge(base)\n"
                "    context = replacement\n"
                "    return consume(left=context.left, right=context.right)\n",
            ),
        )
    )

    assert builder.expansions == ()


def test_builder_follows_complete_flat_field_forwarding_graph() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.chain",
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def leaf(*, first=None, second=None):\n"
                "    return first, second\n"
                "\n"
                "def middle(*, left=None, right=None):\n"
                "    return leaf(first=left, second=right)\n"
                "\n"
                "def root(base):\n"
                "    context = Context.merge(base)\n"
                "    return middle(left=context.alpha, right=context.beta)\n",
            ),
        )
    )

    assert len(builder.components) == 1
    component = builder.components[0]
    assert component.root_edges[0].field_mapping == (
        ("alpha", "left"),
        ("beta", "right"),
    )
    assert tuple(
        (edge.caller_symbol, edge.callee_symbol, edge.field_mapping)
        for edge in component.forwarding_edges
    ) == (
        (
            "pkg.chain.middle",
            "pkg.chain.leaf",
            (("alpha", "first"), ("beta", "second")),
        ),
    )
    assert component.field_mapping_by_participant == {
        "pkg.chain.middle": (("alpha", "left"), ("beta", "right")),
        "pkg.chain.leaf": (("alpha", "first"), ("beta", "second")),
    }


def test_builder_unifies_connected_root_expansions_into_one_component() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.connected",
                "class Context:\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> 'Context':\n"
                "        return base\n"
                "\n"
                "def consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def first(base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(context.left, context.right)\n"
                "\n"
                "def second(base):\n"
                "    context = Context.merge(base)\n"
                "    return consume(context.left, context.right)\n",
            ),
        )
    )

    assert len(builder.components) == 1
    component = builder.components[0]
    assert tuple(edge.caller_symbol for edge in component.root_edges) == (
        "pkg.connected.first",
        "pkg.connected.second",
    )
    assert component.participant_symbols == ("pkg.connected.consume",)
    assert (
        builder.assessed_components()[0]
        .proof.callable_component.incomplete_call_family_symbols
        == ()
    )


def test_builder_proves_one_closed_private_carrier_expansion() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.closed",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n",
            ),
        )
    )

    assessments = builder.assessed_components()

    assert len(assessments) == 1
    assert assessments[0].proof.is_proven
    assert assessments[0].proof.batch_compression_delta == 2
    assert builder.proven_components() == assessments


def test_builder_rejects_calls_outside_the_atomic_component() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.open_call",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n"
                "\n"
                "def other(left, right):\n"
                "    return _consume(left, right)\n",
            ),
        )
    )

    proof = builder.assessed_components()[0].proof

    assert (
        DeclaredCarrierExpansionAuthorityViolation.INCOMPLETE_CALL_FAMILY
        in proof.violations
    )
    assert builder.proven_components() == ()


def test_builder_rejects_mutated_projected_parameters() -> None:
    builder = DeclaredCarrierExpansionBuilder.from_modules(
        (
            _module(
                "pkg.mutated",
                "from dataclasses import dataclass\n"
                "\n"
                "@dataclass(frozen=True)\n"
                "class _Context:\n"
                "    left: object\n"
                "    right: object\n"
                "\n"
                "    @classmethod\n"
                "    def merge(cls, base) -> '_Context':\n"
                "        return base\n"
                "\n"
                "def _consume(left, right):\n"
                "    left = right\n"
                "    return left, right\n"
                "\n"
                "def root(base):\n"
                "    context = _Context.merge(base)\n"
                "    return _consume(context.left, context.right)\n",
            ),
        )
    )

    proof = builder.assessed_components()[0].proof

    assert (
        DeclaredCarrierExpansionAuthorityViolation.REBINDING_OR_MUTATION
        in proof.violations
    )


def test_declared_carrier_operation_rewrites_the_complete_forwarding_graph() -> None:
    module = _module("pkg.rewrite", _closed_expansion_source())
    snapshot = CodemodSourceSnapshot.from_modules((module,), ())
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if snapshot.source_index.symbol_for_target(target)
        == "pkg.rewrite._Context"
    )
    operation = CollapseDeclaredCarrierExpansionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id)
    )
    operation_payload = json_report_object(operation)
    assert RefactorRecipeOperation.from_json_value(operation_payload) == operation
    assert "field_mapping" not in operation_payload

    recipe = RefactorRecipe(recipe_id="collapse-declared-carrier").with_operation(
        operation
    )
    simulation = snapshot.simulate_rewrites(recipe.source_rewrite_batch(snapshot))

    assert simulation.parse_validation.parse_valid
    rewritten_source = simulation.rewritten_sources[module.file_path]
    assert (
        "def _leaf(*, context: '_Context'):\n"
        "    return context.first, context.second\n"
    ) in rewritten_source
    assert (
        "def _middle(*, context: '_Context'):\n"
        "    return _leaf(context=context)\n"
    ) in rewritten_source
    assert "return _middle(context=context)" in rewritten_source
    original_namespace = {"__name__": "pkg.rewrite_original"}
    rewritten_namespace = {"__name__": "pkg.rewrite_rewritten"}
    exec(
        compile(module.source, module.file_path, "exec", dont_inherit=True),
        original_namespace,
    )
    exec(
        compile(rewritten_source, module.file_path, "exec", dont_inherit=True),
        rewritten_namespace,
    )
    original_context = original_namespace["_Context"]("left", "right")
    rewritten_context = rewritten_namespace["_Context"]("left", "right")
    assert original_namespace["caller"](original_context) == rewritten_namespace[
        "caller"
    ](rewritten_context)


def test_declared_carrier_detector_synthesizes_the_atomic_operation() -> None:
    module = _module("pkg.detected", _closed_expansion_source())

    findings = DeclaredCarrierExpansionDetector().detect(
        [module],
        DetectorConfig(),
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.detector_id == "declared_carrier_expansion"
    assert finding.pattern_id is PatternId.AUTHORITATIVE_CONTEXT
    assert finding.certification == "certified"
    assert finding.authority_evidence == SourceLocation(
        "pkg/detected.py",
        4,
        "pkg.detected._Context",
    )
    assert finding.metrics == ParameterThreadMetrics(
        function_count=2,
        shared_parameter_count=2,
        shared_parameter_names=("first", "second"),
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), findings)
    plan = codemod_plan_from_findings(findings, selector_context=snapshot)

    assert len(plan.records) == 1
    assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    operation = plan.document.recipes[0].operations[0]
    assert isinstance(operation, CollapseDeclaredCarrierExpansionOperation)
    simulation = plan.document.simulate(snapshot)
    assert simulation.is_clean
    rewritten_source = simulation.simulation.rewritten_sources[module.file_path]
    assert "def _middle(*, context: '_Context'):" in rewritten_source
    assert "return _middle(context=context)" in rewritten_source


def test_declared_carrier_operation_imports_cross_module_authority() -> None:
    models_module = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class Context:\n"
        "    first: object\n"
        "    second: object\n"
        "\n"
        "    @classmethod\n"
        "    def merge(cls, base) -> 'Context':\n"
        "        return base\n",
    )
    worker_module = _module(
        "pkg.worker",
        "def _consume(first, second):\n"
        "    return first, second\n",
    )
    entry_module = _module(
        "pkg.entry",
        "from pkg.models import Context\n"
        "from pkg.worker import _consume\n"
        "\n"
        "def caller(base):\n"
        "    context = Context.merge(base)\n"
        "    return _consume(context.first, context.second)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules(
        (models_module, worker_module, entry_module),
        (),
    )
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if snapshot.source_index.symbol_for_target(target) == "pkg.models.Context"
    )
    operation = CollapseDeclaredCarrierExpansionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id)
    )

    simulation = snapshot.simulate_rewrites(
        RefactorRecipe(recipe_id="cross-module-carrier")
        .with_operation(operation)
        .source_rewrite_batch(snapshot)
    )

    assert simulation.parse_valid
    rewritten_worker = simulation.rewritten_sources[worker_module.file_path]
    assert "from pkg.models import Context\n" in rewritten_worker
    assert "def _consume(*, context: 'Context'):" in rewritten_worker
    rewritten_entry = simulation.rewritten_sources[entry_module.file_path]
    assert "return _consume(context=context)" in rewritten_entry
    compile(rewritten_worker, worker_module.file_path, "exec")
    compile(rewritten_entry, entry_module.file_path, "exec")


def test_declared_carrier_operation_rejects_an_open_current_call_family() -> None:
    module = _module(
        "pkg.open_rewrite",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class _Context:\n"
        "    first: object\n"
        "    second: object\n"
        "\n"
        "    @classmethod\n"
        "    def merge(cls, base) -> '_Context':\n"
        "        return base\n"
        "\n"
        "def _consume(first, second):\n"
        "    return first, second\n"
        "\n"
        "def root(base):\n"
        "    context = _Context.merge(base)\n"
        "    return _consume(context.first, context.second)\n"
        "\n"
        "def outside(first, second):\n"
        "    return _consume(first, second)\n",
    )
    snapshot = CodemodSourceSnapshot.from_modules((module,), ())
    authority_target = next(
        target
        for target in snapshot.source_index.ast_targets
        if snapshot.source_index.symbol_for_target(target)
        == "pkg.open_rewrite._Context"
    )
    operation = CollapseDeclaredCarrierExpansionOperation(
        target=SourceRewriteTarget(target_id=authority_target.target_id)
    )

    with pytest.raises(
        ValueError,
        match="carrier expansion rewrite requires a proven component",
    ):
        operation.source_edits_from_snapshot(snapshot)
