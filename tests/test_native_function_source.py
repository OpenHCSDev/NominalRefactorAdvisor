"""Native Python source correspondence precedes, and does not imply, effects."""

import ast
import dataclasses
import marshal
from pathlib import Path
import runpy
import sys
from types import FunctionType, ModuleType

import pytest

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.native_call import NativePythonFunctionSource
from nominal_refactor_advisor.scan_cache import ScanCache
from test_native_dataclass_factory import factory_environment


def compiled_function(tmp_path, monkeypatch, source):
    path = tmp_path / "native_function_source_fixture.py"
    path.write_text(source)
    module = ModuleType("native_function_source_fixture")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    compilation = NativePythonCompilation(source, str(path))
    exec(compilation.compile(), vars(module))
    return compilation, module.sample


def test_unrelated_body_with_new_native_constants_does_not_block_correspondence(
    tmp_path, monkeypatch
):
    compilation, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sibling(value):\n    return value[1:]\n"
        "def sample(value):\n    return value\n",
    )
    assert compilation.function_definition(function).name == "sample"


def test_cached_native_source_owner_does_not_share_mutable_definition_syntax(
    tmp_path, monkeypatch
):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    with ScanCache.scope():
        source = NativePythonFunctionSource.from_function(function)
        definition = source.definition
        definition.name = "forged"
        definition.body = [ast.Return(value=ast.Constant(value="forged"))]
        current = NativePythonFunctionSource.from_function(function)
        assert current is source
        assert current.definition is not definition
        assert current.definition.name == "sample"
        assert isinstance(current.definition.body[0].value, ast.Name)


@pytest.mark.skipif(sys.version_info < (3, 14), reason="native slice constants")
def test_selected_body_with_unsupported_native_constants_remains_unproved(
    tmp_path, monkeypatch
):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value[1:]\n"
    )
    with pytest.raises(ValueError, match="unmarshallable object"):
        compilation.function_definition(function)


@pytest.mark.parametrize(
    "source",
    (
        "def sample(value):\n    return value\n",
        "async def sample(value):\n    return value\n",
        "def sample(value):\n    yield value\n",
        "def sample(value):\n    return value in {1, 2, 3}\n",
        "def outer(seed):\n    def sample(value=7, *, option=True):\n"
        "        return seed + value\n    return sample\nsample = outer(3)\n",
        "def identity(value):\n    return value\n@identity\n"
        "def sample(value: int = 3) -> int:\n    return value\n",
        "from __future__ import annotations\ndef sample(value: int) -> int:\n"
        "    return value\n",
    ),
)
def test_original_module_context_matches_without_calling_target(
    tmp_path, monkeypatch, source
):
    compilation, function = compiled_function(tmp_path, monkeypatch, source)
    observed = []
    code = function.__code__

    def observe(frame, event, argument):
        if frame.f_code is code:
            observed.append(event)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        definition = compilation.function_definition(function)
        reread = NativePythonCompilation.from_function(function)
        assert ast.dump(reread.function_definition(function)) == ast.dump(definition)
    finally:
        sys.setprofile(previous)
    assert not observed
    assert definition.name == "sample"


@pytest.mark.parametrize(
    "change",
    (
        {"co_qualname": "different"},
        {"co_filename": "different.py"},
        {"co_stacksize": 100},
        {"co_consts": (None, 999)},
        {"co_linetable": b""},
        {"co_exceptiontable": b"\x01"},
    ),
)
def test_warmed_source_proof_rejects_changed_code_on_same_function(
    tmp_path, monkeypatch, change
):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample():\n    return 17\n"
    )
    original = function.__code__
    with ScanCache.scope():
        compilation.function_definition(function)
        function.__code__ = original.replace(**change)
        with pytest.raises(
            ValueError, match="differs from compiled source|different source path"
        ):
            compilation.function_definition(function)
        function.__code__ = original
        assert compilation.function_definition(function).name == "sample"


def test_nested_code_metadata_is_part_of_source_correspondence(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sample():\n    def inner():\n        return 17\n    return inner\n",
    )
    compilation.function_definition(function)
    original = function.__code__
    function.__code__ = original.replace(
        co_consts=tuple(
            (
                value.replace(co_qualname="different")
                if type(value) is type(original)
                else value
            )
            for value in original.co_consts
        )
    )
    # Code equality alone ignores this observable nested metadata.
    assert function.__code__ == original
    with pytest.raises(ValueError, match="differs from compiled source"):
        compilation.function_definition(function)


def test_equal_constant_contents_cannot_erase_different_alias_relations(
    tmp_path, monkeypatch
):
    compilation, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sample():\n    return ((1000,), (1000,))\n",
    )
    original = function.__code__
    pair = original.co_consts[1]
    assert pair[0] is pair[1]
    distinct = tuple(list(pair[0]))
    assert distinct == pair[0] and distinct is not pair[0]
    with ScanCache.scope():
        compilation.function_definition(function)
        function.__code__ = original.replace(
            co_consts=tuple(
                (pair[0], distinct) if value is pair else value
                for value in original.co_consts
            )
        )
        assert marshal.dumps(function.__code__, 2) == marshal.dumps(original, 2)
        with pytest.raises(ValueError, match="differs from compiled source"):
            compilation.function_definition(function)


def test_same_code_does_not_mean_same_closure_or_defaults(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def outer(seed):\n    def sample(value=7):\n        return seed + value\n"
        "    return sample\nsample = outer(3)\n",
    )
    original = ast.dump(compilation.function_definition(function))
    function.__defaults__ = (22,)
    function.__closure__[0].cell_contents = 100
    assert function() == 122
    assert ast.dump(compilation.function_definition(function)) == original
    # This API returns syntax only, never argument values or an execution result.


def test_completed_correspondence_cache_is_bounded_and_syntax_is_not_shared(
    tmp_path, monkeypatch
):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    compile_calls = []
    native_compile = NativePythonCompilation.compile

    def observed_compile(owner, *, filename=None):
        compile_calls.append(owner)
        return native_compile(owner, filename=filename)

    monkeypatch.setattr(NativePythonCompilation, "compile", observed_compile)
    with ScanCache.scope():
        definition = compilation.function_definition(function)
        definition.name = "damaged"
        assert compilation.function_definition(function).name == "sample"
        assert len(compile_calls) == 1
    compilation.function_definition(function)
    assert len(compile_calls) == 2


def test_replaced_constants_never_execute_equality_callbacks(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample():\n    return 17\n"
    )

    class Explosive:
        def __eq__(self, other):
            raise AssertionError("Analyzed constant equality was invoked")

    function.__code__ = function.__code__.replace(co_consts=(None, Explosive()))
    with pytest.raises(ValueError, match="unmarshallable object"):
        compilation.function_definition(function)


def test_different_compilation_cannot_reuse_completed_source_proof(
    tmp_path, monkeypatch
):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample():\n    return 17\n"
    )
    with ScanCache.scope():
        compilation.function_definition(function)
        changed = NativePythonCompilation(
            compilation.source.replace("17", "18"), compilation.file_path
        )
        with pytest.raises(ValueError, match="differs from compiled source"):
            changed.function_definition(function)


def test_dataclass_factory_rechecks_current_implementation_after_warming(monkeypatch):
    environment, operation = factory_environment("frozen=True")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    with ScanCache.scope():
        authority.require_closed()
        monkeypatch.setattr(
            dataclasses.dataclass,
            "__code__",
            dataclasses.dataclass.__code__.replace(co_qualname="changed"),
        )
        with pytest.raises(ValueError, match="differs from compiled source"):
            authority.require_closed()


def test_non_function_has_no_python_implementation():
    with pytest.raises(ValueError, match="exact function"):
        NativePythonCompilation.from_function(len)


def test_stdlib_dataclass_code_is_matched_to_current_source():
    function = dataclasses.dataclass
    assert type(function) is FunctionType
    with ScanCache.scope():
        definition = NativePythonCompilation.from_function(
            function
        ).function_definition(function)
        assert definition.name == function.__name__


def test_executable_filename_preserves_backslashes_separately_from_source_identity():
    source = "def sample():\n    return 17\n"
    filename = r"package\sample.py"
    compilation = NativePythonCompilation(source, filename)
    namespace = {}
    exec(compile(source, filename, "exec", dont_inherit=True, optimize=0), namespace)
    function = namespace["sample"]
    assert compilation.file_path == "package/sample.py"
    assert function.__code__.co_filename == filename
    assert compilation.function_definition(function).name == "sample"
    assert compilation.identity.file_path == "package/sample.py"
    assert compilation.compile(filename=filename).co_filename == filename
    with pytest.raises(ValueError, match="different source path"):
        compilation.compile(filename="unrelated.py")


def test_current_source_metadata_owns_live_default_associations(tmp_path, monkeypatch):
    _, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sample(value=7, *, option=True):\n"
        "    raise AssertionError('target body must not run')\n",
    )
    with ScanCache.scope():
        source = NativePythonFunctionSource.from_function(function)
        before = source.signature
        observations = source.defaults
        assert before.bind((), ()).is_exact
        assert tuple(default.value for default in observations) == (7, True)
        with monkeypatch.context() as mutation:
            mutation.setattr(function, "__defaults__", None)
            mutation.setattr(function, "__kwdefaults__", None)
            assert not source.signature.bind((), ()).is_exact
            assert source.defaults == ()
            assert before.bind(
                (), ()
            ).is_exact  # Previous observation, not current proof.
            assert source.definition.args.defaults  # Source was not rewritten.
        assert source.signature.bind((), ()).is_exact


@pytest.mark.parametrize("query", ("signature", "defaults"))
def test_warmed_current_metadata_rejoins_code_before_query(
    tmp_path, monkeypatch, query
):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value=7):\n    return value\n"
    )
    with ScanCache.scope():
        source = NativePythonFunctionSource.from_function(function)
        _ = getattr(source, query)
        original = function.__code__
        with monkeypatch.context() as mutation:
            mutation.setattr(function, "__code__", original.replace(co_stacksize=100))
            with pytest.raises(ValueError, match="differs from compiled source"):
                _ = getattr(source, query)
        assert getattr(source, query)


def test_current_source_signature_ignores_wrapper_and_annotation_hooks(
    tmp_path, monkeypatch
):
    _, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sample(value: int = 7):\n"
        "    raise AssertionError('target body must not run')\n",
    )
    with monkeypatch.context() as mutation:
        mutation.setattr(function, "__signature__", object(), raising=False)
        mutation.setattr(function, "__wrapped__", len, raising=False)
        mutation.setattr(function, "__annotations__", {"value": object()})
        source = NativePythonFunctionSource.from_function(function)
        assert source.signature.bind((), ()).is_exact
        assert source.signature.parameters[0].annotation_expression == "int"


def test_native_call_projects_metadata_from_current_source_owner():
    environment, operation = factory_environment("")
    authority = environment.call_authority(
        environment.context_for_owner(operation.owner), operation.event
    )
    with ScanCache.scope():
        source = authority.python_source
        assert source is NativePythonFunctionSource.from_function(dataclasses.dataclass)
        assert authority.signature == source.signature
        projected = authority.python_defaults
        observed = source.defaults
        assert tuple(default.parameter_name for default in projected) == tuple(
            default.parameter_name for default in observed
        )
        assert all(
            left.value is right.value
            for left, right in zip(projected, observed, strict=True)
        )
        assert ast.dump(authority.python_definition) == ast.dump(source.definition)
        authority.require_closed()
        assert not environment.entry.operation_conditions


def test_callable_metadata_dsl_batches_source_ownership_without_effect_admission():
    from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

    plan = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "docs/examples/native_callable_metadata_owner.py"
        )
    )["callable_metadata_plan"]
    native_path = "nominal_refactor_advisor/native_call.py"
    execution_path = "nominal_refactor_advisor/source_execution.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            native_path: (
                "class NativeCallAuthority:\n"
                "    @property\n    def signature(self): pass\n"
                "    @property\n    def python_defaults(self): pass\n"
                "    @property\n    def python_definition(self): pass\n"
                "class NativePythonFunctionSource:\n    function = None\n"
            ),
            execution_path: (
                "class NativeSourceClassEntryABC:\n"
                "    @property\n    def construction_admission(self):\n"
                "        raise ValueError('construction remains unproved')\n"
            ),
        }
    )
    result = plan(snapshot).simulate(snapshot)
    assert result.is_clean
    assert result.stage_count == 6
    assert set(result.simulation.changed_file_paths) == {native_path, execution_path}
    parsed = result.final_snapshot.parsed_module_for_source_path(native_path)
    caller, source = parsed.module.body
    assert {node.name for node in caller.body if isinstance(node, ast.FunctionDef)} == {
        "python_source"
    }
    assert {node.name for node in source.body if isinstance(node, ast.FunctionDef)} == {
        "signature",
        "defaults",
    }
    execution_module = result.final_snapshot.parsed_module_for_source_path(
        execution_path
    )
    assert any(
        isinstance(node, ast.Raise) for node in ast.walk(execution_module.module)
    )
    original = snapshot.parsed_module_for_source_path(native_path)
    assert original is not parsed
    assert original.source != parsed.source
