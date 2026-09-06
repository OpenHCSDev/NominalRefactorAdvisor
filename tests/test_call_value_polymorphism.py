"""The signature binder transports values without interpreting their shape."""

import ast
from dataclasses import dataclass
import inspect

from nominal_refactor_advisor.call_binding import CompactFunctionSignature
from nominal_refactor_advisor.product_flow import CompactCallArguments


@dataclass(frozen=True)
class SourceToken:
    spelling: str
    line: int


def test_projection_and_binding_preserve_arbitrary_value_identity() -> None:
    def consume(first, /, *rest, flag, **extras):
        pass

    node = (
        ast.parse("consume(first, second, third, flag=enabled, detail=other)")
        .body[0]
        .value
    )
    projected = []

    def project(expression: ast.expr) -> SourceToken:
        value = SourceToken(ast.unparse(expression), expression.lineno)
        projected.append(value)
        return value

    arguments = CompactCallArguments[SourceToken].from_call(node, project)
    assert len(projected) == 5
    assert all(
        actual is expected
        for actual, expected in zip(arguments.values, projected, strict=True)
    )
    declaration = ast.parse("def consume(first, /, *rest, flag, **extras): pass").body[
        0
    ]
    signature = CompactFunctionSignature.from_arguments(declaration.args)
    result = signature.bind(arguments.positional, arguments.keywords)
    native = inspect.signature(consume).bind(
        *projected[:3], flag=projected[3], detail=projected[4]
    )
    assert result.is_exact
    for name, expected in native.arguments.items():
        argument = result.argument_for(name)
        assert argument is not None
        expected_values = (
            tuple(expected.values())
            if isinstance(expected, dict)
            else expected if isinstance(expected, tuple) else (expected,)
        )
        assert all(
            actual is value
            for actual, value in zip(argument.values, expected_values, strict=True)
        )


def test_projection_retains_unpacking_as_an_explicit_binding_limit() -> None:
    node = ast.parse("consume(*items, **options)", mode="eval").body
    arguments = CompactCallArguments[ast.expr].from_call(
        node, lambda expression: expression
    )
    assert arguments.positional[0].value is node.args[0].value
    assert arguments.keywords[0].value is node.keywords[0].value
    declaration = ast.parse("def consume(*items, **options): pass").body[0]
    signature = CompactFunctionSignature.from_arguments(declaration.args)
    assert not signature.bind(arguments.positional, arguments.keywords).is_exact


def test_projection_preserves_the_nominal_argument_list_refinement() -> None:
    class TokenArguments(CompactCallArguments[SourceToken]):
        pass

    node = ast.parse("consume(item)", mode="eval").body
    arguments = TokenArguments.from_call(
        node, lambda expression: SourceToken(ast.unparse(expression), expression.lineno)
    )
    assert type(arguments) is TokenArguments
    assert arguments.values == (SourceToken("item", 1),)
