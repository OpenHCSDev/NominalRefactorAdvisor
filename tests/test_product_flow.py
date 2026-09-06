from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import ImportBoundNameProjection, ParsedModule
from nominal_refactor_advisor.product_flow import (
    BareCallTargetReference,
    CompactCallArgument,
    CompactCallBindingViolation,
    CompactValueDestinationKind,
    CompactControlBranchKind,
    CompactFunctionBindingKind,
    CompactFunctionSignature,
    CompactFlowOwner,
    CompactFlowOwnerKind,
    CompactFunctionTargetResolutionViolation,
    CompactMutationKind,
    CompactNamespaceFlowOwner,
    CompactKeywordArgument,
    CompactParameterKind,
    CompactProductFlowModuleProjectionFamily,
    CurrentClassMemberMethodReference,
    CurrentClassMethodReference,
    DynamicCallTargetReference,
    ExactCompactValueOrigin,
    LexicalValueReference,
    OpenCompactValueOrigin,
    OpenCompactBindingMutation,
    compact_product_flow_projection,
)

def _parsed_module(source: str) -> ParsedModule:
    return ParsedModule(
        path=Path("pkg/sample.py"),
        module_name="pkg.sample",
        is_package_init=False,
        module=ast.parse(source, filename="pkg/sample.py"),
        source=source,
    )


def _signature(source: str) -> CompactFunctionSignature:
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return CompactFunctionSignature.from_arguments(function.args)


def _value(name: str) -> LexicalValueReference:
    return LexicalValueReference(name)


@pytest.mark.parametrize(
    "use_source,expected",
    (
        ("saved = (result,)", (LexicalValueReference("result"),)),
        ("return result", (LexicalValueReference("result"),)),
        ("publish(result)", (LexicalValueReference("result"),)),
        (
            "saved = result.callback",
            (
                LexicalValueReference("result"),
                LexicalValueReference("result", ("callback",)),
            ),
        ),
        (
            "saved = result.child.callback",
            (
                LexicalValueReference("result"),
                LexicalValueReference("result", ("child",)),
                LexicalValueReference("result", ("child", "callback")),
            ),
        ),
        (
            "saved = [result, unknown]",
            (LexicalValueReference("result"), LexicalValueReference("unknown")),
        ),
        ("result.execute()", (LexicalValueReference("result"),)),
        (
            "result.callback = replacement",
            (LexicalValueReference("replacement"), LexicalValueReference("result")),
        ),
    ),
)
def test_flow_retains_reference_uses_without_a_callable_name_inventory(
    use_source: str, expected: tuple[LexicalValueReference, ...]
) -> None:
    projection = compact_product_flow_projection(
        _parsed_module(f"def run():\n    result = Factory()\n    {use_source}\n")
    )
    flow = next(flow for flow in projection.flows if flow.owner.qualname == "run")

    assert (
        tuple(use.target.lexical_reference for use in flow.callable_reference_uses)
        == expected
    )
    assert all(
        flow.calls[0].position.dominates(use.position)
        for use in flow.callable_reference_uses
    )


@pytest.mark.parametrize(
    "source,expected",
    (
        ("import pkg.library", (("pkg", "pkg"),)),
        ("import pkg.library as lib", (("lib", "pkg.library"),)),
        ("from library import consume as run", (("run", "library.consume"),)),
        ("from .library import consume", (("consume", "pkg.library.consume"),)),
        ("from . import library", (("library", "pkg.library"),)),
        ("from ...library import consume", (("consume", None),)),
        ("from library import *", ()),
    ),
)
def test_import_origins_share_the_name_projection(source: str, expected: tuple) -> None:
    module = _parsed_module(source)
    projection = ImportBoundNameProjection(module.module.body[0])
    origins = projection.origins(module.module_path_identity)
    assert (
        tuple((origin.bound_name, origin.qualified_name) for origin in origins)
        == expected
    )
    assert projection.names() == tuple(origin.bound_name for origin in origins)
    flow = compact_product_flow_projection(module).flows[0]
    assert (
        tuple(
            (mutation.reference.root_name, mutation.imported_origin.qualified_name)
            for mutation in flow.mutations
        )
        == expected
    )
    assert all(
        mutation.target.origin is mutation.imported_origin
        and mutation.target.bound_name == mutation.imported_origin.bound_name
        for mutation in flow.mutations
    )


def test_annotation_only_statements_bind_locals_but_not_namespaces() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "value: int\nclass Owner:\n value: int\ndef run():\n value: int\n"
        )
    )
    for flow in projection.flows:
        assert (
            "value" in flow.mutations_by_root_name
        ) is flow.owner.kind.is_function_scope


def test_binding_selection_distinguishes_source_position_and_final_namespace() -> None:
    projection = compact_product_flow_projection(
        _parsed_module("def consume(): pass\nconsume()\nconsume = None\n")
    )
    flow = projection.flows[0]
    immediate = flow.binding_resolution_for("consume", flow.calls[0].position)
    deferred = flow.binding_resolution_for("consume")
    assert immediate is not None and immediate.mutation is not None
    assert deferred is not None and deferred.mutation is not None
    assert immediate.mutation.kind is CompactMutationKind.FUNCTION_DEFINITION
    assert deferred.mutation.kind is CompactMutationKind.ASSIGNMENT
    assert flow.binding_resolution_for("absent") is None
    assert pickle.loads(pickle.dumps(immediate)) == immediate


def test_deferred_closure_binding_keeps_multiple_writes_unresolved() -> None:
    projection = compact_product_flow_projection(
        _parsed_module("def outer():\n value = 1\n value = 2\n")
    )
    flow = next(flow for flow in projection.flows if flow.owner.qualname == "outer")
    resolution = flow.binding_resolution_for("value")
    assert isinstance(resolution, OpenCompactBindingMutation)
    assert (
        resolution.target_lookup_violation
        is CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )


def _contains_ast(value: object) -> bool:
    if isinstance(value, ast.AST):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_ast(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_ast(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_ast(key) or _contains_ast(item) for key, item in value.items()
        )
    return False


def test_lexical_value_reference_accepts_only_exact_name_attribute_chains() -> None:
    assert LexicalValueReference.from_expression(
        ast.parse("request.key.value").body[0].value
    ) == LexicalValueReference("request", ("key", "value"))
    assert (
        LexicalValueReference.from_expression(ast.parse("request[0]").body[0].value)
        is None
    )
    assert (
        LexicalValueReference.from_expression(
            ast.parse("request().value").body[0].value
        )
        is None
    )


def test_function_signature_owns_full_python_argument_binding() -> None:
    signature = _signature(
        "def target(a, /, b=1, *rest, c, d=2, **extras):\n    pass\n"
    )

    assert [parameter.kind for parameter in signature.parameters] == [
        CompactParameterKind.POSITIONAL_ONLY,
        CompactParameterKind.POSITIONAL_OR_KEYWORD,
        CompactParameterKind.VAR_POSITIONAL,
        CompactParameterKind.KEYWORD_ONLY,
        CompactParameterKind.KEYWORD_ONLY,
        CompactParameterKind.VAR_KEYWORD,
    ]
    binding = signature.bind(
        (
            CompactCallArgument(_value("first")),
            CompactCallArgument(_value("second")),
            CompactCallArgument(_value("third")),
        ),
        (
            CompactKeywordArgument("c", _value("required")),
            CompactKeywordArgument("extra", _value("extension")),
        ),
    )

    assert binding.is_exact
    assert binding.argument_for("a").values == (_value("first"),)
    assert binding.argument_for("rest").values == (_value("third"),)
    assert binding.argument_for("c").values == (_value("required"),)
    assert binding.argument_for("extras").keyword_names == ("extra",)
    assert binding.argument_for("d") is None


def test_function_projection_retains_nominal_annotation_declarations() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "def build(item: pkg.models.Item, values: list[int]) -> 'pkg.Result':\n"
            "    return pkg.Result(item, values)\n"
        )
    )
    declaration = projection.function_declarations[0]
    item, values = declaration.signature.parameters

    assert item.annotation_expression == "pkg.models.Item"
    assert item.annotation_reference_parts == ("pkg", "models", "Item")
    assert item.has_annotation
    assert not item.is_plain_required
    assert values.annotation_expression == "list[int]"
    assert values.annotation_reference_parts is None
    assert declaration.return_annotation_expression == "'pkg.Result'"
    assert declaration.return_annotation_reference_parts == ("pkg", "Result")


def test_function_flow_proves_unchanged_bound_call_result() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "def caller(value, replacement):\n"
            "    result = make(value)\n"
            "    consume(result)\n"
            "    result = replacement\n"
            "    consume(result)\n"
        )
    )
    flow = next(item for item in projection.flows if item.owner.qualname == "caller")
    make_call = next(call for call in flow.calls if call.target.terminal_name == "make")
    first_consume, second_consume = tuple(
        call for call in flow.calls if call.target.terminal_name == "consume"
    )
    result = LexicalValueReference("result")

    assert flow.bound_call_result_for(result, first_consume.position) == make_call
    assert flow.bound_call_result_for(result, second_consume.position) is None


@pytest.mark.parametrize(
    "intervening",
    (
        "if flag:\n        result = replacement",
        "for result in replacements:\n        pass",
        "try:\n        result = replacement\n    except Exception:\n        pass",
        "with manager as result:\n        pass",
        "match replacement:\n        case {'value': result}:\n            pass",
        "if flag:\n        del result",
    ),
)
def test_bound_call_result_respects_selected_binding(intervening: str) -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "def caller():\n"
            "    result = make()\n"
            f"    {intervening}\n"
            "    consume(result)\n"
        )
    )
    flow = next(item for item in projection.flows if item.owner.qualname == "caller")
    consume = next(
        call for call in flow.calls if call.target.terminal_name == "consume"
    )
    selection = flow.binding_resolution_for("result", consume.position)
    assert selection is not None
    assert selection.mutation is None or selection.mutation.kind is not (
        CompactMutationKind.ASSIGNMENT
    )
    assert flow.bound_call_result_for(_value("result"), consume.position) is None


@pytest.mark.parametrize(
    "mutation,retains_result",
    (
        ("owner = replacement", False),
        ("owner.child = replacement", False),
        ("owner.child.result = replacement", False),
        ("if flag:\n        owner.child = replacement", False),
        ("owner.sibling = replacement", False),
        ("owner.child.result.field = replacement", False),
        ("unrelated = replacement", True),
    ),
)
def test_bound_attribute_result_requires_unchanged_binding_and_closed_slot_writes(
    mutation: str, retains_result: bool
) -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "def caller():\n"
            "    owner.child.result = make()\n"
            f"    {mutation}\n"
            "    consume(owner.child.result)\n"
        )
    )
    flow = next(item for item in projection.flows if item.owner.qualname == "caller")
    make, consume = flow.calls
    reference = LexicalValueReference("owner", ("child", "result"))
    assert flow.bound_call_result_for(reference, consume.position) == (
        make if retains_result else None
    )


def test_function_signature_rejects_every_non_exact_binding_boundary() -> None:
    signature = _signature("def target(a, *, b):\n    pass\n")

    assert signature.bind((), ()).violation is (
        CompactCallBindingViolation.MISSING_REQUIRED_ARGUMENT
    )
    assert (
        signature.bind(
            (
                CompactCallArgument(_value("a")),
                CompactCallArgument(_value("extra")),
            ),
            (CompactKeywordArgument("b", _value("b")),),
        ).violation
        is CompactCallBindingViolation.TOO_MANY_POSITIONAL_ARGUMENTS
    )
    assert (
        signature.bind(
            (CompactCallArgument(_value("a")),),
            (CompactKeywordArgument("unknown", _value("b")),),
        ).violation
        is CompactCallBindingViolation.UNEXPECTED_KEYWORD_ARGUMENT
    )
    assert (
        signature.bind(
            (CompactCallArgument(_value("a")),),
            (
                CompactKeywordArgument("a", _value("duplicate")),
                CompactKeywordArgument("b", _value("b")),
            ),
        ).violation
        is CompactCallBindingViolation.DUPLICATE_ARGUMENT
    )
    assert (
        signature.bind(
            (CompactCallArgument(_value("items"), is_unpacked=True),),
            (),
        ).violation
        is CompactCallBindingViolation.VARIADIC_UNPACKING
    )


def test_product_flow_projection_preserves_complete_construction_forwarding_context() -> (
    None
):
    projection = compact_product_flow_projection(
        _parsed_module(
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def consume(*, left, right):\n"
            "    return left, right\n"
            "\n"
            "def wrapper(left, right, cached):\n"
            "    cache_key = CacheKey(left=left, right=right)\n"
            "    if cached:\n"
            "        return cached\n"
            "    return consume(left=cache_key.left, right=cache_key.right)\n"
            "\n"
            "escaped = consume\n"
        )
    )

    declarations = {
        declaration.identity.qualname: declaration
        for declaration in projection.function_declarations
    }
    wrapper = next(
        flow
        for flow in projection.flows
        if flow.owner.kind is CompactFlowOwnerKind.FUNCTION
        and flow.owner.qualname == "wrapper"
    )
    construction_call, consumer_call = wrapper.calls
    construction = construction_call.product_construction()

    assert construction is not None
    assert construction.field_names == ("left", "right")
    assert construction.result_binding == LexicalValueReference("cache_key")
    assert construction.position.dominates(consumer_call.position)
    assert consumer_call.result_use is CompactValueDestinationKind.RETURNED
    assert isinstance(consumer_call.target, BareCallTargetReference)
    assert wrapper.local_candidate_symbols(consumer_call.target, "pkg.sample") == (
        "pkg.sample.wrapper.consume",
        "pkg.sample.consume",
    )
    binding = consumer_call.bind_to(declarations["consume"])
    assert binding.is_exact
    for name, supplied in zip(
        ("left", "right"), consumer_call.arguments.values, strict=True
    ):
        assert binding.argument_for(name).values[0] is supplied
        assert supplied.lexical_reference == LexicalValueReference("cache_key", (name,))
    module_flow = projection.flows[0]
    assert [
        use.target.terminal_name for use in module_flow.callable_reference_uses
    ] == ["consume"]


def test_control_positions_reject_branch_local_binding_as_post_branch_authority() -> (
    None
):
    projection = compact_product_flow_projection(
        _parsed_module(
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "def wrapper(flag, value):\n"
            "    if flag:\n"
            "        carrier = Carrier(value=value)\n"
            "    return consume(carrier.value)\n"
        )
    )
    wrapper = next(
        flow for flow in projection.flows if flow.owner.qualname == "wrapper"
    )
    construction_call, consumer_call = wrapper.calls

    assert construction_call.position.branch_path[0].kind is (
        CompactControlBranchKind.IF_BODY
    )
    assert not construction_call.position.dominates(consumer_call.position)


def test_product_flow_projects_exact_value_alias_events_in_lexical_scopes() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "module_value = source\n"
            "\n"
            "def wrapper(source, owner):\n"
            "    direct = source\n"
            "    first = second = source\n"
            "    annotated: object = direct\n"
            "    transformed = normalize(source)\n"
            "    owner.value = source\n"
            "    global module_value\n"
            "    module_value = source\n"
            "    return direct, first, second, annotated\n"
        )
    )
    module_flow = projection.flows[0]
    wrapper = next(
        flow for flow in projection.flows if flow.owner.qualname == "wrapper"
    )

    assert tuple(
        (alias.target.root_name, alias.source.root_name)
        for alias in module_flow.exact_value_aliases
    ) == (("module_value", "source"),)
    assert tuple(
        (alias.target.root_name, alias.source.root_name)
        for alias in wrapper.exact_value_aliases
    ) == (
        ("direct", "source"),
        ("first", "source"),
        ("second", "source"),
        ("annotated", "direct"),
    )
    assert all(
        alias.binding_mutation in wrapper.mutations
        for alias in wrapper.exact_value_aliases
    )


def test_product_flow_resolves_straight_line_alias_origins_and_attribute_suffixes() -> (
    None
):
    projection = compact_product_flow_projection(
        _parsed_module(
            "def target(value):\n"
            "    return value\n"
            "\n"
            "def wrapper(source):\n"
            "    first = source\n"
            "    second = first\n"
            "    return target(second.value)\n"
        )
    )
    wrapper = next(
        flow for flow in projection.flows if flow.owner.qualname == "wrapper"
    )
    call = wrapper.calls[0]

    resolution = wrapper.value_origin_for(
        LexicalValueReference("second", ("value",)),
        call.position,
    )

    assert isinstance(resolution, ExactCompactValueOrigin)
    assert resolution.exact_origin == LexicalValueReference("source", ("value",))
    assert tuple(
        mutation.reference.root_name for mutation in resolution.alias_chain
    ) == (
        "first",
        "second",
    )


def test_product_flow_keeps_branch_and_rebinding_alias_origins_open() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "def target(left, right):\n"
            "    return left, right\n"
            "\n"
            "def wrapper(flag, left, right):\n"
            "    if flag:\n"
            "        first = left\n"
            "    second = right\n"
            "    second = normalize(second)\n"
            "    return target(first, second)\n"
        )
    )
    wrapper = next(
        flow for flow in projection.flows if flow.owner.qualname == "wrapper"
    )
    call = wrapper.calls[-1]

    branch_resolution = wrapper.value_origin_for(
        LexicalValueReference("first"),
        call.position,
    )
    rebound_resolution = wrapper.value_origin_for(
        LexicalValueReference("second"),
        call.position,
    )

    assert isinstance(branch_resolution, OpenCompactValueOrigin)
    assert LexicalValueReference("left") in branch_resolution.possible_origins
    assert isinstance(rebound_resolution, OpenCompactValueOrigin)
    assert LexicalValueReference("right") in rebound_resolution.possible_origins


def test_function_binding_and_dynamic_call_shapes_remain_nominal() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "class Owner:\n"
            "    @classmethod\n"
            "    def from_value(cls, value):\n"
            "        return cls(value)\n"
            "\n"
            "    @staticmethod\n"
            "    def normalize(value):\n"
            "        return value\n"
            "\n"
            "    def execute(self, callback, value):\n"
            "        escaped = self.normalize\n"
            "        return callback()(value)\n"
        )
    )
    declarations = {
        declaration.identity.qualname: declaration
        for declaration in projection.function_declarations
    }
    execute = next(
        flow for flow in projection.flows if flow.owner.qualname == "Owner.execute"
    )

    assert declarations["Owner.from_value"].binding_kind is (
        CompactFunctionBindingKind.CLASS_METHOD
    )
    assert declarations["Owner.normalize"].binding_kind is (
        CompactFunctionBindingKind.STATIC_METHOD
    )
    assert declarations["Owner.execute"].binding_kind is (
        CompactFunctionBindingKind.INSTANCE_METHOD
    )
    assert [use.target.terminal_name for use in execute.callable_reference_uses] == [
        "self",
        "normalize",
        "value",
    ]
    assert any(
        isinstance(call.target, DynamicCallTargetReference) for call in execute.calls
    )


def test_receiver_identity_and_decorator_safety_are_declaration_derived() -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            "class Owner:\n"
            "    @instrument\n"
            "    def normalize(receiver, value):\n"
            "        return value\n"
            "\n"
            "    def execute(receiver, value):\n"
            "        return receiver.normalize(value)\n"
        )
    )
    declarations = {
        declaration.identity.qualname: declaration
        for declaration in projection.function_declarations
    }
    execute = next(
        flow for flow in projection.flows if flow.owner.qualname == "Owner.execute"
    )
    call = execute.calls[0]

    assert isinstance(call.target, CurrentClassMethodReference)
    assert declarations["Owner.normalize"].signature_decorator_hazard
    assert call.bind_to(declarations["Owner.normalize"]).violation is (
        CompactCallBindingViolation.SIGNATURE_DECORATOR_HAZARD
    )


def test_annotated_member_method_targets_retain_current_class_lookup_semantics() -> (
    None
):
    projection = compact_product_flow_projection(
        _parsed_module(
            "class Owner:\n"
            "    renderer: Renderer\n"
            "\n"
            "    def direct(self, value):\n"
            "        return self.renderer.render(value)\n"
            "\n"
            "    def runtime(self, value):\n"
            "        return type(self).renderer.render(value)\n"
        )
    )

    targets = tuple(
        call.target
        for flow in projection.flows
        for call in flow.calls
        if isinstance(call.target, CurrentClassMemberMethodReference)
    )

    assert targets == (
        CurrentClassMemberMethodReference("self", "Owner", "renderer", "render", False),
        CurrentClassMemberMethodReference("self", "Owner", "renderer", "render", True),
    )


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("callback(value)", ("callback", "value")),
        ("self.renderer.render(value)", ("self", "value")),
        ("type(self).renderer.render(value)", ("self", "type", "value")),
        ("factory().callback", ("factory",)),
    ),
)
def test_loaded_names_are_derived_from_reference_facts(
    expression: str, expected: tuple[str, ...]
) -> None:
    projection = compact_product_flow_projection(
        _parsed_module(
            f"class Owner:\n    def run(self):\n        return {expression}\n"
        )
    )
    flow = next(flow for flow in projection.flows if flow.owner.qualname == "Owner.run")
    assert flow.loaded_value_root_names == expected
    assert "loaded_value_root_names" not in {field.name for field in fields(flow)}
    empty = replace(flow, calls=(), callable_reference_uses=())
    assert empty.loaded_value_root_names == ()
    assert pickle.loads(pickle.dumps(flow)).loaded_value_root_names == expected


def test_function_declaration_is_the_flow_owner_and_module_view_is_derived() -> None:
    projection = compact_product_flow_projection(
        _parsed_module("def run(value):\n    return value\n")
    )
    declaration = projection.function_declarations[0]
    flow = next(flow for flow in projection.flows if flow.owner.qualname == "run")
    assert flow.owner is declaration
    assert flow.owner.declaration is declaration
    assert flow.owner.kind is CompactFlowOwnerKind.FUNCTION
    assert flow.owner.qualname == declaration.identity.qualname
    assert "function_declarations" not in {field.name for field in fields(projection)}
    namespace_flows = tuple(
        flow for flow in projection.flows if flow.owner.declaration is None
    )
    assert replace(projection, flows=namespace_flows).function_declarations == ()
    restored = pickle.loads(pickle.dumps(projection))
    restored_flow = next(
        flow for flow in restored.flows if flow.owner.qualname == "run"
    )
    assert restored_flow.owner is restored.function_declarations[0]


def test_function_flow_ownership_requires_a_declaration() -> None:
    with pytest.raises(TypeError):
        CompactFlowOwner()
    with pytest.raises(ValueError, match="owned by their declaration"):
        CompactNamespaceFlowOwner(CompactFlowOwnerKind.FUNCTION, "run")
    for kind, qualname in (
        (CompactFlowOwnerKind.MODULE, ""),
        (CompactFlowOwnerKind.CLASS_BODY, "Owner"),
    ):
        owner = CompactNamespaceFlowOwner(kind, qualname)
        assert owner.declaration is None
        assert owner.kind is kind
        assert owner.qualname == qualname


def test_opaque_attribute_reads_remain_explicit_flow_evidence() -> None:
    projection = compact_product_flow_projection(
        _parsed_module("def run():\n    return factory().callback\n")
    )
    flow = next(flow for flow in projection.flows if flow.owner.qualname == "run")
    assert len(flow.callable_reference_uses) == 1
    use = flow.callable_reference_uses[0]
    assert isinstance(use.target, DynamicCallTargetReference)
    assert flow.calls[0].position.dominates(use.position)


def test_product_flow_family_payload_is_ast_free_and_pickle_stable() -> None:
    module = _parsed_module(
        "def target(value):\n"
        "    return value\n"
        "\n"
        "def caller(value):\n"
        "    return target(value)\n"
    )
    projection = CompactProductFlowModuleProjectionFamily.collect(module)[0]

    assert not _contains_ast(projection)
    assert pickle.loads(pickle.dumps(projection)) == projection
    assert CompactProductFlowModuleProjectionFamily.item_type is type(projection)
