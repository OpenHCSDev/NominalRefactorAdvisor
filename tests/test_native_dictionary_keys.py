"""Actual static-type keys share original namespace and mutation evidence."""

import pytest

import nominal_refactor_advisor.registry_identity as registry_identity
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    NativeNamespace,
)
from nominal_refactor_advisor.codemod_native_requirements import NativeUseProvenance
from nominal_refactor_advisor.native_compilation import (
    NativeCreationBackend,
    SpanOnlyCreationBackend,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from test_native_behavior_proof import (
    invocation_requirement,
    snapshot_with_environment,
)


@pytest.mark.parametrize("key", (object, type, str, int, bool, type(None)))
def test_static_type_key_has_native_identity_protocol_and_safe_release(key):
    backend = NativeCreationBackend.current()
    assert backend.require_dictionary_key(key) is key
    backend.require_dictionary_store(key)
    namespace = NativeNamespace({key: property})
    namespace.member(key).require_native_identity(NativeDeclaration(property))
    assert CapturedNativeObject(key).require_dictionary_key() is key
    # Dictionary-key support does not turn class identity into scalar contents.
    with pytest.raises(ValueError, match="exact primitive"):
        CapturedNativeObject(key).require_native_scalar()


@pytest.mark.parametrize(
    "assignments,declaration,expected",
    (
        ("registry[object]=property\n", "type(None)", property),
        ("registry[object]=property\nregistry[str]=int\n", "str", int),
        ("registry[str]=int\nregistry[object]=property\n", "str", int),
        ("registry[object]=property\nregistry[int]=str\n", "bool", str),
        ("registry[object]=property\nregistry[str]=None\n", "str", None),
        ("registry[str]=property\n", "int", None),
    ),
)
def test_real_mro_hit_uses_original_stores_and_actual_nominal_priority(
    tmp_path, assignments, declaration, expected
):
    source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        "registry={}\n"
        + assignments
        + f"result=mro_registry_value(registry, {declaration})\n"
    )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    environment = requirement.environment
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    result = environment.call_authority(context, call).result()
    if expected is None:
        result.require_constant_contents(None)
    else:
        result.require_native_identity(NativeDeclaration(expected))
    namespace = {}
    exec(environment.module.native_compilation.compile(), namespace)
    assert namespace["result"] is expected
    assert not environment.entry.operation_conditions


def test_heap_type_key_does_not_acquire_static_identity_contract():
    class Heap:
        pass

    with pytest.raises(ValueError, match="immutable static type"):
        NativeCreationBackend.current().require_dictionary_key(Heap)
    with pytest.raises(TypeError, match="immutable static type"):
        NativeNamespace({Heap: property})


def test_key_representation_admission_does_not_inherit_storage_or_construction_laws():
    backend = SpanOnlyCreationBackend()
    assert backend.require_dictionary_key("initial") == "initial"
    with pytest.raises(ValueError, match="item storage remains unproved"):
        backend.require_dictionary_store("initial")
    with pytest.raises(ValueError, match="immutable static type"):
        backend.require_dictionary_key(object)


@pytest.mark.parametrize("operand", (None, 1, "class-name", (), object()))
def test_class_lookup_law_requires_an_actual_class_operand(operand):
    with pytest.raises(ValueError, match="static type lifetime"):
        NativeCreationBackend.current().require_dictionary_class_lookup(operand)


def test_integer_hash_collision_does_not_alias_a_static_type_key(tmp_path):
    source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        f"registry={{}}\nregistry[{hash(object)}]=int\nregistry[object]=property\n"
        "result=mro_registry_value(registry, object)\n"
    )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert requirement.inspect().provenance is NativeUseProvenance.PROVED
    namespace = {}
    exec(requirement.module.native_compilation.compile(), namespace)
    assert namespace["result"] is property


def test_metaclass_hash_equality_and_metadata_hooks_are_never_used_by_admission():
    events = []

    class Meta(type):
        def __hash__(cls):
            events.append("hash")
            return 0

        def __eq__(cls, other):
            events.append("eq")
            return True

        def __getattribute__(cls, name):
            events.append("getattribute")
            return super().__getattribute__(name)

    class Heap(metaclass=Meta):
        pass

    storage = {Heap: property}
    events.clear()
    with pytest.raises(TypeError, match="immutable static type"):
        NativeNamespace(storage)
    assert events == []


def test_query_rejects_foreign_key_before_hashing_or_equality():
    events = []

    class Key:
        def __hash__(self):
            events.append("hash")
            return 0

        def __eq__(self, other):
            events.append("eq")
            return True

    namespace = NativeNamespace({object: property})
    with pytest.raises(TypeError, match="exact scalar key"):
        namespace.member(Key())
    assert events == []


def test_sequential_dsl_stores_select_the_projected_hit_not_the_prior_result(tmp_path):
    source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        "registry={}\nregistry[object]=property\nregistry[str]=int\n"
        "result=mro_registry_value(registry, str)\n"
    )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    plan = CodemodPlanSequence.from_operations(
        (
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=path),
                replacements=(
                    SourceTextReplacement("registry[str]=int", "registry[str]=bool"),
                ),
            ),
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=path),
                replacements=(
                    SourceTextReplacement("registry[str]=bool", "registry[str]=str"),
                ),
            ),
        )
    )
    preview = plan.simulate(snapshot)
    assert preview.is_clean
    assert len(preview.stage_reports) == 2
    for state, expected in ((snapshot, int), (preview.final_snapshot, str)):
        requirement = invocation_requirement(
            state, path, registry_identity.mro_registry_value
        )
        assert requirement.inspect().provenance is NativeUseProvenance.PROVED
        environment = requirement.environment
        context, call = environment.source_call(
            environment.module.module.body[-1].value
        )
        environment.call_authority(context, call).result().require_native_identity(
            NativeDeclaration(expected)
        )
        namespace = {}
        exec(environment.module.native_compilation.compile(), namespace)
        assert namespace["result"] is expected


@pytest.mark.parametrize(
    "prefix",
    (
        "class Local: pass\nregistry={}\nregistry[Local]=property\n",
        "def callback(): return None\nregistry={}\nregistry[object]=callback\n",
    ),
)
def test_unproved_class_keys_and_result_release_remain_unresolved(tmp_path, prefix):
    source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        + prefix
        + "result=mro_registry_value(registry, object)\n"
    )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(
        snapshot, path, registry_identity.mro_registry_value
    )
    assert not requirement.inspect().provenance.is_admitted


@pytest.mark.parametrize(
    "helper_source",
    (
        "def mro_registry_value(next, declaration_type):\n"
        "    return next((next[owner] for owner in declaration_type.__mro__ if owner in next), None)\n",
        "def mro_registry_value(registry, declaration_type):\n"
        "    return next((registry[registry] for registry in declaration_type.__mro__ if registry in registry), None)\n",
    ),
)
def test_matching_source_shape_does_not_authenticate_shadowed_lexical_roles(
    tmp_path, monkeypatch, helper_source
):
    provider_path = tmp_path / "shadowed_provider.py"
    provider_path.write_bytes(helper_source.encode("utf-8"))
    provider = {}
    exec(compile(helper_source, str(provider_path), "exec"), provider)
    original = registry_identity.mro_registry_value
    monkeypatch.setattr(original, "__code__", provider["mro_registry_value"].__code__)
    source = (
        "from nominal_refactor_advisor.registry_identity import mro_registry_value\n"
        "registry={}\nregistry[object]=property\n"
        "result=mro_registry_value(registry, object)\n"
    )
    path, snapshot = snapshot_with_environment(tmp_path, source)
    requirement = invocation_requirement(snapshot, path, original)
    assert not requirement.inspect().provenance.is_admitted
    with pytest.raises(TypeError):
        original({object: property}, object)
