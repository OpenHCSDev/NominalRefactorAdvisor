"""Native Python source correspondence precedes, and does not imply, effects."""

import abc
import ast
import dataclasses
import marshal
from pathlib import Path
import runpy
import sys
from types import FunctionType, ModuleType

import pytest

from nominal_refactor_advisor.native_compilation import NativePythonCompilation
from nominal_refactor_advisor.native_call import (
    NativePythonFunctionSource,
    NativeReturnedClosureFactorySource,
)
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


def test_actual_frozen_abc_constructor_joins_current_source_without_invocation():
    function = vars(abc.ABCMeta)["__new__"].__func__
    assert function.__globals__ is vars(abc)
    observed = []

    def observe(frame, event, argument):
        if frame.f_code is function.__code__:
            observed.append(event)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        with ScanCache.scope():
            owner = NativePythonFunctionSource.from_function(function)
            assert owner.definition.name == "__new__"
            assert tuple(
                parameter.name for parameter in owner.signature.parameters
            ) == ("mcls", "name", "bases", "namespace", "kwargs")
    finally:
        sys.setprofile(previous)
    assert not observed


def test_source_acquisition_does_not_execute_a_custom_loader(tmp_path, monkeypatch):
    source = "def sample(value):\n    raise AssertionError('target must not run')\n"
    _, function = compiled_function(tmp_path, monkeypatch, source)
    namespace = function.__globals__
    exec(
        compile(
            source,
            str(tmp_path / "not_retained_executable.py"),
            "exec",
            dont_inherit=True,
            optimize=0,
        ),
        namespace,
    )
    function = namespace["sample"]
    observed = []

    class ActiveLoader:
        def get_source(self, name):
            observed.append(name)
            return source

    function.__globals__["__loader__"] = ActiveLoader()
    owner = NativePythonFunctionSource.from_function(function)
    assert owner.definition.name == "sample"
    assert not observed


def test_physical_source_does_not_require_registered_module_metadata(tmp_path):
    path = tmp_path / "unregistered.py"
    source = "def sample(value):\n    return value\n"
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec", dont_inherit=True, optimize=0), namespace)
    function = namespace["sample"]
    assert "__file__" not in function.__globals__
    assert (
        NativePythonFunctionSource.from_function(function).definition.name == "sample"
    )


def test_ambiguous_physical_source_context_is_not_selected(tmp_path, monkeypatch):
    source = "def sample(value):\n    return value\nafter = 1\n"
    _, function = compiled_function(tmp_path, monkeypatch, source)
    other = tmp_path / "other.py"
    other.write_text(source.replace("after = 1", "after = 2"))
    function.__globals__["__file__"] = str(other)
    with pytest.raises(ValueError, match="ambiguous physical source"):
        NativePythonFunctionSource.from_function(function)


def test_warmed_source_acquisition_rereads_edited_physical_body(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    with ScanCache.scope():
        owner = NativePythonFunctionSource.from_function(function)
        assert owner.definition.name == "sample"
        Path(compilation.file_path).write_text("def sample(value):\n    return 99\n")
        with pytest.raises(ValueError, match="differs from compiled source"):
            NativePythonFunctionSource.from_function(function)
        # The prior immutable source still corresponds to the unchanged code;
        # it is neither an observation of the edited file nor an activation proof.
        assert owner.definition.name == "sample"


def test_source_path_subclass_is_rejected_without_path_hooks(tmp_path, monkeypatch):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    observed = []

    class ActivePath(str):
        def __fspath__(self):
            observed.append("path")
            return str(self)

    function.__globals__["__file__"] = ActivePath(function.__globals__["__file__"])
    with pytest.raises(ValueError, match="exact native string"):
        NativePythonFunctionSource.from_function(function)
    assert not observed


def test_source_namespace_rejects_active_keys_before_lookup(tmp_path, monkeypatch):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    observed = []

    class ActiveKey:
        def __hash__(self):
            observed.append("hash")
            return 3

    function.__globals__[ActiveKey()] = "foreign"
    observed.clear()
    with pytest.raises(ValueError):
        NativePythonFunctionSource.from_function(function)
    assert not observed


def test_loader_only_source_remains_unproved_without_executing_loader(
    tmp_path, monkeypatch
):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    Path(function.__globals__["__file__"]).unlink()
    observed = []

    class ActiveLoader:
        def get_source(self, name):
            observed.append(name)
            return "def sample(value):\n    return value\n"

    function.__globals__["__loader__"] = ActiveLoader()
    with pytest.raises(ValueError, match="no inspectable physical source"):
        NativePythonFunctionSource.from_function(function)
    assert not observed


def test_physical_source_reader_respects_python_encoding_cookie(tmp_path):
    path = tmp_path / "encoded.py"
    source = b"# coding: latin-1\ndef sample():\n    return 'caf\xe9'\n"
    path.write_bytes(source)
    namespace = {}
    exec(compile(source, str(path), "exec", dont_inherit=True, optimize=0), namespace)
    owner = NativePythonFunctionSource.from_function(namespace["sample"])
    assert owner.definition.body[0].value.value == "caf\u00e9"


def test_physical_source_acquisition_dsl_keeps_input_and_original_signature():
    from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

    path = "nominal_refactor_advisor/native_compilation.py"
    source = "class NativePythonCompilation:\n    @classmethod\n"
    source += "    def from_function(cls, function): return cls('', '')\n"
    original = CodemodSourceSnapshot.from_source_mapping({path: source})
    plan = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "docs/examples/native_physical_source_acquisition.py"
        )
    )["physical_source_plan"](original)
    result = plan.simulate(original)
    assert result.is_clean and result.stage_count == 2
    assert original.sources_by_file_path[path] == source
    final = result.final_snapshot.sources_by_file_path[path]
    assert "import tokenize" in final and "inspect.findsource" not in final
    module = result.final_snapshot.parsed_module_for_source_path(path)
    (definition,) = (
        node
        for node in ast.walk(module.module)
        if isinstance(node, ast.FunctionDef) and node.name == "from_function"
    )
    assert tuple(argument.arg for argument in definition.args.args) == (
        "cls",
        "function",
    )
    assert definition.decorator_list[0].id == "classmethod"


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


def test_warmed_compact_flow_rejoins_current_function_code(tmp_path, monkeypatch):
    _, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    with ScanCache.scope():
        source = NativePythonFunctionSource.from_function(function)
        original_flow = source.flow
        with monkeypatch.context() as mutation:
            mutation.setattr(
                function, "__code__", function.__code__.replace(co_stacksize=100)
            )
            with pytest.raises(ValueError, match="differs from compiled source"):
                _ = source.flow
        assert source.flow is original_flow


def test_warmed_compact_flow_rejects_another_valid_source_body(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path,
        monkeypatch,
        "def sample(value):\n    return value\n" "def other(value):\n    return 999\n",
    )
    namespace = {}
    exec(compilation.compile(), namespace)
    replacement = namespace["other"]
    with ScanCache.scope():
        source = NativePythonFunctionSource.from_function(function)
        original_flow = source.flow
        with monkeypatch.context() as mutation:
            mutation.setattr(function, "__code__", replacement.__code__)
            assert source.definition.name == "other"  # Correspondence alone is valid.
            with pytest.raises(ValueError, match="changed after source-flow capture"):
                _ = source.flow
            current = NativePythonFunctionSource.from_function(function)
            assert current is not source
            assert current.flow.owner.declaration.qualname == "other"
        assert source.flow is original_flow


def test_warmed_closure_factory_rejoins_current_function_code(monkeypatch):
    with ScanCache.scope():
        proof = NativeReturnedClosureFactorySource.from_function(dataclasses.dataclass)
        parameter = proof.selected_parameter_name
        proof.require_returned_closure(parameter)
        with monkeypatch.context() as mutation:
            mutation.setattr(
                dataclasses.dataclass,
                "__code__",
                dataclasses.dataclass.__code__.replace(co_stacksize=100),
            )
            with pytest.raises(ValueError, match="differs from compiled source"):
                proof.require_returned_closure(parameter)
        proof.require_returned_closure(parameter)


def test_closure_factory_exposes_fresh_definition_syntax():
    with ScanCache.scope():
        proof = NativeReturnedClosureFactorySource.from_function(dataclasses.dataclass)
        definition = proof.definition
        definition.body.clear()
        assert proof.definition is not definition
        assert len(proof.body) == 3
        proof.require_returned_closure(proof.selected_parameter_name)


@pytest.mark.parametrize("dependency", ("dataclass", "_process_class"))
def test_reused_dataclass_application_revalidates_body_dependencies(
    monkeypatch, dependency
):
    from test_definition_application_activation import controlled_execution

    environment = controlled_execution(
        "from dataclasses import dataclass\n@dataclass\nclass Target: pass\n"
    )
    entry = environment.class_entry(environment.module.module.body[-1])
    _, application = entry.creation_results
    authority = application.application_authority
    assert authority.result() is application.argument
    function = getattr(dataclasses, dependency)
    with monkeypatch.context() as mutation:
        mutation.setattr(
            function, "__code__", function.__code__.replace(co_stacksize=100)
        )
        with pytest.raises(ValueError, match="differs from compiled source"):
            authority.result()
    assert authority.result() is application.argument


def test_current_body_owner_dsl_preserves_docstring_and_explicit_source_ownership():
    from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

    plans = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/current_native_body_owner.py")
    )
    path = "nominal_refactor_advisor/native_call.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            path: "class NativePythonFunctionSource:\n"
            "    @cached_property\n    def flow(self): return None\n"
            "class NativeReturnedClosureFactorySource:\n"
            "    function: FunctionType\n    definition: ast.FunctionDef\n"
            "    @classmethod\n    def from_function(cls, function): return cls(function, None)\n"
            "    def require_returned_closure(self, parameter_name):\n"
            '        """Preserved documentation."""\n        return None\n'
            "class NativeDataclassDefinitionApplicationABC:\n"
            "    @cached_property\n    def factory_parameters(self): return ()\n"
            "    @cached_property\n    def processor_parameters(self): return ()\n"
            "    @cached_property\n    def selector_parameter(self): return None\n"
            "    def require_factory_processor_call(self):\n"
            "        return self.factory_parameters, self.processor_parameters\n"
            "    def result(self): return None\n"
            "class NativeDataclassFactoryCall:\n"
            "    def require_closed(self):\n"
            "        NativeReturnedClosureFactorySource(self.declaration, self.definition)\n"
        }
    )
    ownership = plans["current_body_owner_plan"](snapshot).simulate(snapshot)
    assert ownership.is_clean and ownership.stage_count == 11
    bounded = plans["bounded_body_query_plan"](ownership.final_snapshot).simulate(
        ownership.final_snapshot
    )
    assert bounded.is_clean and bounded.stage_count == 3
    parsed = bounded.final_snapshot.parsed_module_for_source_path(path)
    _, closure, dataclass, factory = parsed.module.body
    method = next(
        node
        for node in closure.body
        if isinstance(node, ast.FunctionDef) and node.name == "require_returned_closure"
    )
    assert ast.get_docstring(method) == "Preserved documentation."
    assert isinstance(method.body[1], ast.With)
    fields = {
        node.target.id for node in closure.body if isinstance(node, ast.AnnAssign)
    }
    assert fields == {"source"}
    for owner in (dataclass, factory):
        query = next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"result", "require_closed"}
        )
        assert isinstance(query.body[0], ast.With)
    assert snapshot.parsed_module_for_source_path(path).source != parsed.source


def test_current_flow_geometry_does_not_reparse_syntax(tmp_path, monkeypatch):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "def sample(value):\n    return value\n"
    )
    with ScanCache.scope():
        owner = NativePythonFunctionSource.from_function(function)
        flow = owner.flow

        def reject_parse(*args, **kwargs):
            raise AssertionError("Current source geometry must not require fresh AST")

        monkeypatch.setattr(ast, "parse", reject_parse)
        assert owner.flow is flow
        assert (
            compilation.function_source_span(function)
            == flow.owner.declaration.source_span
        )


@pytest.mark.parametrize(
    "body",
    (
        "def sample(value):\n    return value\n",
        "async def sample(value):\n    return value\n",
        "class Container:\n    @staticmethod\n    def sample(value):\n        return value\n"
        "sample = Container.sample\n",
        "class Container:\n    @(\n        staticmethod\n    )\n"
        "    def sample(value):\n        return value\n"
        "sample = Container.sample\n",
    ),
)
def test_fresh_definition_parses_only_authenticated_body_with_original_geometry(
    tmp_path, monkeypatch, body
):
    compilation, function = compiled_function(
        tmp_path, monkeypatch, "padding = 'unrelated context'\n" * 100 + body
    )
    expected = next(
        node
        for node in ast.walk(ast.parse(compilation.source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sample"
    )
    parse = ast.parse
    inputs = []

    def observe(source, *args, **kwargs):
        inputs.append(source)
        return parse(source, *args, **kwargs)

    with ScanCache.scope():
        # Warm immutable code/source geometry, not a current-code verdict or AST.
        _ = compilation.function_definition(function)
        with monkeypatch.context() as observation:
            observation.setattr(ast, "parse", observe)
            first = compilation.function_definition(function)
            second = compilation.function_definition(function)
        assert first is not second
        assert ast.dump(first, include_attributes=True) == ast.dump(
            expected, include_attributes=True
        )
        assert ast.dump(second, include_attributes=True) == ast.dump(
            expected, include_attributes=True
        )
        assert len(inputs) == 2
        assert all("unrelated context" not in source for source in inputs)


def test_code_span_and_generic_geometry_dsl_chain_has_one_source_owner():
    from nominal_refactor_advisor.codemod import CodemodSourceSnapshot

    root = Path(__file__).parents[1]
    native_plans = runpy.run_path(
        str(root / "docs/examples/current_native_body_owner.py")
    )
    geometry_plan = runpy.run_path(
        str(root / "docs/examples/shared_declaration_geometry_owner.py")
    )["shared_geometry_plan"]
    native = "nominal_refactor_advisor/native_call.py"
    compilation = "nominal_refactor_advisor/native_compilation.py"
    geometry = "nominal_refactor_advisor/source_geometry.py"
    edits = "nominal_refactor_advisor/codemod_source_edits.py"
    original = CodemodSourceSnapshot.from_source_mapping(
        {
            native: "class NativePythonFunctionSource:\n"
            "    def flow(self): return SourceByteSpan.require_node(self.definition)\n"
            "    def _flow(self): return SourceByteSpan.require_node(self.definition)\n"
            "    def _from_current_function(function, compilation, code):\n"
            "        return compilation.function_definition(function)\n",
            compilation: "class NativePythonCompilation:\n"
            "    source: str\n    file_path: str\n"
            "    def function_definition(self, function):\n"
            '        """Original declaration."""\n'
            "        if function is None: raise ValueError('missing')\n"
            "        code = function.__code__\n"
            "        span = self._function_source_span(code, '', 1)\n"
            "        return span\n",
            geometry: "class SourceLineSegmentAuthority:\n    source: str\n",
            edits: "class SourceTextGeometry(SourceLineSegmentAuthority):\n"
            "    def iter_tokens(self): return iter(())\n"
            "    @cached_property\n    def tokens(self): return tuple(self.iter_tokens())\n"
            "    def node_start_line(self, span): return span.node.lineno\n",
        }
    )
    spans = native_plans["current_code_span_plan"](original).simulate(original)
    assert spans.is_clean and spans.stage_count == 5
    windows = native_plans["fresh_definition_window_plan"](
        spans.final_snapshot
    ).simulate(spans.final_snapshot)
    assert windows.is_clean and windows.stage_count == 1
    shared = geometry_plan(windows.final_snapshot).simulate(windows.final_snapshot)
    assert shared.is_clean and shared.stage_count == 13
    final = shared.final_snapshot
    owner = final.parsed_module_for_source_path(geometry).module.body[-1]
    consumer = final.parsed_module_for_source_path(compilation).module.body[-1]
    editor = final.parsed_module_for_source_path(edits).module.body[-1]
    fields = lambda node: {
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
    }
    assert fields(owner) == {"source"}
    assert fields(consumer) == {"file_path"}
    assert consumer.bases[0].id == "SourceLineSegmentAuthority"
    assert not any(
        isinstance(statement, ast.FunctionDef)
        and statement.name in {"tokens", "iter_tokens"}
        for statement in editor.body
    )
    assert set(shared.simulation.changed_file_paths) == {compilation, geometry, edits}
    assert fields(
        original.parsed_module_for_source_path(compilation).module.body[0]
    ) == {
        "source",
        "file_path",
    }
