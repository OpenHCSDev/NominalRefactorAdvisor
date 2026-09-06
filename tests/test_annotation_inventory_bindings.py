"""Annotation inventory preserves source bindings without inventing root order."""

import ast

import pytest

from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyUse,
    FunctionParameterBinding,
    ModuleBindingResolutionPhase,
    ModuleLexicalDependencyProjection,
)
from nominal_refactor_advisor.lexical_scopes import LexicalNameResolution


@pytest.mark.parametrize(
    "signature",
    (
        "value: Removed = first(), other: Remaining = second()",
        "value: Removed = first(), /, other: Remaining = second()",
        "*, value: Removed = first(), other: Remaining = second()",
        "*, value: Removed, other: Remaining = second()",
        "*items, value: Removed = first(), other: Remaining = second(), **options",
    ),
)
def test_parameter_removal_inventory_does_not_require_default_realignment(signature):
    function = ast.parse(f"def sample({signature}) -> Returned: return value\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    binding = FunctionParameterBinding(function, "value")
    projected = binding.without_binding()

    # These are deliberately partial binding projections, not rewritten callable
    # signatures. Original default expressions remain owned by their source.
    for original, retained in (
        (function.args.defaults, projected.args.defaults),
        (function.args.kw_defaults, projected.args.kw_defaults),
    ):
        assert len(retained) == len(original)
        assert all(
            actual is expected
            for actual, expected in zip(retained, original, strict=True)
        )
    inventory = ModuleLexicalDependencyProjection.from_module(
        ast.Module(body=[projected], type_ignores=[])
    )
    annotations = {
        surface.reference.id for surface in inventory.direct_annotation_name_surfaces
    }
    assert annotations == {"Remaining", "Returned"}
    defaults = (
        *function.args.defaults,
        *(value for value in function.args.kw_defaults if value is not None),
    )
    for default in defaults:
        assert isinstance(default, ast.Call)
        assert any(
            surface.reference is default.func
            for surface in inventory.direct_name_surfaces
        )
    (reference,) = binding.required_references()
    assert reference is function.body[0].value


@pytest.mark.parametrize("prefix", ("def", "async def"))
@pytest.mark.parametrize(
    "signature",
    (
        "pos: Target, /, ordinary: (Target := replacement)",
        "pos: (Target := replacement), /, ordinary: Target",
        "*rest: Target, keyword: (Target := replacement)",
        "*rest: (Target := replacement), keyword: Target",
    ),
)
def test_cross_root_walrus_keeps_class_annotation_ownership_unproved(prefix, signature):
    # This inventory accepts source ASTs independently of the executing
    # interpreter's annotation mode. It must not assume eager evaluation.
    module = ast.parse(
        "class Owner:\n"
        "    Stable = stable_source\n"
        f"    {prefix} method({signature}): pass\n"
        "    after = Target\n"
        "    unchanged = Stable\n"
    )
    owner = module.body[0]
    function = owner.body[1]
    reads = tuple(
        node
        for node in ast.walk(function.args)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "Target"
    )
    assert len(reads) == 1
    inventory = ModuleLexicalDependencyProjection.from_module(module)
    (surface,) = tuple(
        surface
        for surface in inventory.direct_name_surfaces
        if surface.reference is reads[0]
    )
    assert surface.use is DeclarationDependencyUse.EVALUATED_ANNOTATION
    assert surface.resolution is LexicalNameResolution.UNPROVED
    (after,) = tuple(
        surface
        for surface in inventory.direct_name_surfaces
        if surface.reference is owner.body[2].value
    )
    assert after.resolution is LexicalNameResolution.UNPROVED
    assert not any(
        surface.reference is owner.body[3].value
        for surface in inventory.direct_name_surfaces
    )
    (replacement,) = tuple(
        surface
        for surface in inventory.direct_name_surfaces
        if surface.reference.id == "replacement"
    )
    assert replacement.resolution is LexicalNameResolution.EXTERNAL
    with pytest.raises(ValueError, match="Unproved class namespace binding"):
        inventory.external_name_references


@pytest.mark.parametrize("future", ("", "from __future__ import annotations\n"))
def test_annotation_inventory_leaves_unaffected_class_bindings_owned(future):
    module = ast.parse(
        future + "class Owner:\n"
        "    Stable = stable_source\n"
        "    def method(value: Stable) -> External: return Deferred\n"
        "    after = Stable\n"
    )
    inventory = ModuleLexicalDependencyProjection.from_module(module)
    assert not any(
        surface.reference.id == "Stable" for surface in inventory.name_surfaces
    )
    (annotation,) = tuple(
        surface
        for surface in inventory.direct_annotation_name_surfaces
        if surface.reference.id == "External"
    )
    assert annotation.resolution is LexicalNameResolution.EXTERNAL
    (body,) = tuple(
        surface
        for surface in inventory.direct_name_surfaces
        if surface.reference.id == "Deferred"
    )
    assert body.binding_phase is ModuleBindingResolutionPhase.FINAL_MODULE
    assert body.use is DeclarationDependencyUse.EXECUTION
    assert body.resolution is LexicalNameResolution.EXTERNAL


@pytest.mark.parametrize("future", ("", "from __future__ import annotations\n"))
def test_string_annotation_inventory_remains_deferred_and_complete(future):
    module = ast.parse(
        future + "class Owner:\n"
        "    Stable = stable_source\n"
        "    def method(value: 'External[Inner]') -> 'Returned': pass\n"
        "    after = Stable\n"
    )
    inventory = ModuleLexicalDependencyProjection.from_module(module)
    deferred = tuple(
        surface
        for surface in inventory.name_surfaces
        if surface.use is DeclarationDependencyUse.DEFERRED_ANNOTATION
    )
    assert {surface.reference.id for surface in deferred} == {
        "External",
        "Inner",
        "Returned",
    }
    assert all(
        surface.resolution is LexicalNameResolution.EXTERNAL for surface in deferred
    )
    assert not any(
        surface.reference.id == "Stable" for surface in inventory.name_surfaces
    )
