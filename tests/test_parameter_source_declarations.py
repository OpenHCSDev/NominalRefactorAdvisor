"""Source parameter ownership derives signature views without retaining ASTs."""

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect
import pickle

import pytest

from nominal_refactor_advisor.call_binding import (
    CompactFunctionSignature,
    CompactParameterKind as PublicParameterKind,
)
from nominal_refactor_advisor.lexical_bindings import (
    CompactParameterKind,
    FunctionParameterSource,
)


def _arguments(source: str) -> ast.arguments:
    (statement,) = ast.parse(source).body
    if isinstance(statement, ast.Expr):
        assert isinstance(statement.value, ast.Lambda)
        return statement.value.args
    assert isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    return statement.args


def test_parameter_kind_reexport_preserves_nominal_identity() -> None:
    assert PublicParameterKind is CompactParameterKind
    for kind in CompactParameterKind:
        assert PublicParameterKind(kind.value) is kind


@pytest.mark.parametrize("prefix", ["def", "async def"])
@pytest.mark.parametrize(
    "parameters, default_start",
    [
        ("a, b, /, c, d", 4),
        ("a, b, /, c, d=40", 3),
        ("a, b, /, c=30, d=40", 2),
        ("a, b=20, /, c=30, d=40", 1),
        ("a=10, b=20, /, c=30, d=40", 0),
    ],
)
def test_positional_default_cutoff_retains_exact_nodes(
    prefix: str, parameters: str, default_start: int
) -> None:
    arguments = _arguments(f"{prefix} sample({parameters}): pass")
    sources = FunctionParameterSource.from_arguments(arguments)
    assert isinstance(sources, tuple)
    native_arguments = (*arguments.posonlyargs, *arguments.args)
    assert len(sources) == len(native_arguments)
    for index, (source, argument) in enumerate(
        zip(sources, native_arguments, strict=True)
    ):
        assert source.argument is argument
        expected_kind = (
            CompactParameterKind.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else CompactParameterKind.POSITIONAL_OR_KEYWORD
        )
        assert source.kind is expected_kind
        expected_default = (
            None if index < default_start else arguments.defaults[index - default_start]
        )
        assert source.default is expected_default


def test_variadics_and_keyword_only_defaults_keep_source_identity() -> None:
    arguments = _arguments(
        "def sample(a: First, /, b: Second=None, *items: Third, "
        "required: Fourth, optional: Fifth=None, valued: Sixth=7, "
        "**options: Seventh) -> Result: pass"
    )
    sources = FunctionParameterSource.from_arguments(arguments)
    expected_arguments = (
        *arguments.posonlyargs,
        *arguments.args,
        arguments.vararg,
        *arguments.kwonlyargs,
        arguments.kwarg,
    )
    expected_kinds = (
        CompactParameterKind.POSITIONAL_ONLY,
        CompactParameterKind.POSITIONAL_OR_KEYWORD,
        CompactParameterKind.VAR_POSITIONAL,
        CompactParameterKind.KEYWORD_ONLY,
        CompactParameterKind.KEYWORD_ONLY,
        CompactParameterKind.KEYWORD_ONLY,
        CompactParameterKind.VAR_KEYWORD,
    )
    expected_defaults = (
        None,
        arguments.defaults[0],
        None,
        *arguments.kw_defaults,
        None,
    )
    for source, argument, kind, default in zip(
        sources, expected_arguments, expected_kinds, expected_defaults, strict=True
    ):
        assert source.argument is argument
        assert source.argument.annotation is argument.annotation
        assert source.kind is kind
        assert source.default is default
    required, optional = sources[3:5]
    assert required.default is None
    assert isinstance(optional.default, ast.Constant)
    assert optional.default.value is None


def test_lambda_parameters_use_same_source_projection() -> None:
    arguments = _arguments(
        "lambda a, /, b=None, *items, required, optional=None, **options: None"
    )
    sources = FunctionParameterSource.from_arguments(arguments)
    compact = CompactFunctionSignature.from_arguments(arguments)
    assert tuple(source.argument.arg for source in sources) == (
        "a",
        "b",
        "items",
        "required",
        "optional",
        "options",
    )
    for source, parameter in zip(sources, compact.parameters, strict=True):
        assert parameter.name == source.argument.arg
        assert parameter.kind is source.kind
        assert parameter.has_default is (source.default is not None)


def test_empty_parameter_list_has_empty_source_and_compact_views() -> None:
    arguments = _arguments("def sample(): pass")
    assert FunctionParameterSource.from_arguments(arguments) == ()
    assert CompactFunctionSignature.from_arguments(arguments).parameters == ()


def test_compact_signature_consumes_the_shared_source_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments("def sample(first: int, second: str=None): pass")
    selected = FunctionParameterSource(
        arguments.args[1], CompactParameterKind.KEYWORD_ONLY, arguments.defaults[0]
    )
    requests = []

    def project(cls, requested):
        assert cls is FunctionParameterSource
        requests.append(requested)
        return (selected,)

    monkeypatch.setattr(FunctionParameterSource, "from_arguments", classmethod(project))
    compact = CompactFunctionSignature.from_arguments(arguments)
    assert len(requests) == 1
    assert requests[0] is arguments
    (parameter,) = compact.parameters
    assert parameter.name == selected.argument.arg
    assert parameter.kind is selected.kind
    assert parameter.has_default
    assert parameter.annotation_expression == ast.unparse(selected.argument.annotation)


def test_source_receipt_is_frozen_but_owns_original_syntax() -> None:
    arguments = _arguments("def sample(value=None): pass")
    (source,) = FunctionParameterSource.from_arguments(arguments)
    assert source.argument is arguments.args[0]
    assert source.default is arguments.defaults[0]
    with pytest.raises(FrozenInstanceError):
        source.default = None


@pytest.mark.parametrize(
    "mutation", ["excess_positional", "missing_keyword", "excess_keyword"]
)
def test_malformed_default_association_is_rejected(mutation: str) -> None:
    arguments = _arguments("def sample(value, *, required, optional=None): pass")
    if mutation == "excess_positional":
        arguments.defaults.extend([ast.Constant(value=1), ast.Constant(value=2)])
    elif mutation == "missing_keyword":
        arguments.kw_defaults.pop()
    else:
        arguments.kw_defaults.append(ast.Constant(value=3))
    with pytest.raises(ValueError):
        FunctionParameterSource.from_arguments(arguments)
    with pytest.raises(ValueError):
        CompactFunctionSignature.from_arguments(arguments)


@pytest.mark.parametrize(
    "source",
    [
        "def sample(a, b=2, /, c=3, *items, required, optional=None, **options): pass",
        "async def sample(a, /, b=None, *, required, optional=4): pass",
        "def sample(*items, **options): pass",
        "def sample(*, required, optional=None): pass",
        "def sample(): pass",
        "lambda a, /, b=None, *items, required, optional=3, **options: None",
    ],
)
def test_compact_kind_and_default_presence_match_trusted_native_signature(
    source: str,
) -> None:
    # These are fixed inert test fixtures, never repository/analyzed source.
    namespace = {}
    if source.startswith("lambda"):
        native = eval(compile(source, "<trusted-signature-fixture>", "eval"), namespace)
    else:
        exec(compile(source, "<trusted-signature-fixture>", "exec"), namespace)
        native = namespace["sample"]
    native_parameters = tuple(inspect.signature(native).parameters.values())
    compact = CompactFunctionSignature.from_arguments(_arguments(source))
    for parameter, expected in zip(compact.parameters, native_parameters, strict=True):
        assert parameter.name == expected.name
        assert parameter.kind.name == expected.kind.name
        assert parameter.has_default is (
            expected.default is not inspect.Parameter.empty
        )


def test_compact_signature_pickle_contains_no_source_ast() -> None:
    arguments = _arguments(
        "def sample(a: list[int], /, b: tuple[str, ...]=None, "
        "*items: bytes, required: float, optional: dict[str, int]=None, "
        "**options: object): pass"
    )
    compact = CompactFunctionSignature.from_arguments(arguments)
    restored = pickle.loads(pickle.dumps(compact))
    assert restored == compact
    pending = [compact, restored]
    while pending:
        value = pending.pop()
        assert not isinstance(value, ast.AST)
        if is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
