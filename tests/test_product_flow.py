from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
import pickle

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.product_flow import (
    BareCallTargetReference,
    CompactCallArgument,
    CompactCallBindingViolation,
    CompactCallResultUse,
    CompactControlBranchKind,
    CompactFunctionBindingKind,
    CompactFunctionSignature,
    CompactFlowOwnerKind,
    CompactKeywordArgument,
    CompactParameterKind,
    CompactProductFlowModuleProjectionFamily,
    CurrentClassMethodReference,
    DynamicCallTargetReference,
    ExactCompactValueOrigin,
    LexicalValueReference,
    OpenCompactValueOrigin,
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
    assert consumer_call.result_use is CompactCallResultUse.RETURNED
    assert isinstance(consumer_call.target, BareCallTargetReference)
    assert wrapper.local_candidate_symbols(consumer_call.target, "pkg.sample") == (
        "pkg.sample.wrapper.consume",
        "pkg.sample.consume",
    )
    binding = consumer_call.bind_to(declarations["consume"])
    assert binding.is_exact
    assert binding.argument_for("left").values == (
        LexicalValueReference("cache_key", ("left",)),
    )
    assert binding.argument_for("right").values == (
        LexicalValueReference("cache_key", ("right",)),
    )
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


def test_product_flow_projects_only_exact_local_value_alias_events() -> None:
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

    assert module_flow.exact_local_value_aliases == ()
    assert tuple(
        (alias.target.root_name, alias.source.root_name)
        for alias in wrapper.exact_local_value_aliases
    ) == (
        ("direct", "source"),
        ("first", "source"),
        ("second", "source"),
        ("annotated", "direct"),
    )
    assert all(
        alias.binding_mutation in wrapper.mutations
        for alias in wrapper.exact_local_value_aliases
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
    assert tuple(mutation.reference.root_name for mutation in resolution.alias_chain) == (
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
        "normalize"
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
