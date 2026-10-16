"""Source snapshots retain native proofs only for their own parsed source state."""

import pytest

from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.native_declarations import NativeDeclaration


def test_snapshot_reuses_actual_producer_and_preserves_captured_identity():
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {"/repo/subject.py": "chosen = property\n"}
    )
    module = snapshot.parsed_module_for_source_path("/repo/subject.py")
    proof = snapshot.module_binding_proof
    environment = proof.native_reference_environment(module)
    node = module.module.body[0].value
    read = environment.source.reference_reads_by_node[node]
    capture = environment.capture(node)
    assert environment.source.module is module
    assert (
        capture.require_native((NativeDeclaration(property),)).declaration is property
    )
    for _ in range(3):
        assert snapshot.module_binding_proof is proof
        assert proof.native_reference_environment(module) is environment
        assert environment.source.reference_reads_by_node[node] is read
        assert (
            environment.require_native(node, (NativeDeclaration(property),)).declaration
            is property
        )


def test_virtual_source_change_cannot_reuse_prior_admission_or_source_nodes():
    original = CodemodSourceSnapshot.from_source_mapping(
        {"/repo/subject.py": "chosen = property\n"}
    )
    changed = original.with_virtual_sources({"/repo/subject.py": "chosen = object\n"})
    original_module = original.parsed_module_for_source_path("/repo/subject.py")
    changed_module = changed.parsed_module_for_source_path("/repo/subject.py")
    assert original.module_binding_proof is not changed.module_binding_proof
    before = original.module_binding_proof.native_reference_environment(original_module)
    after = changed.module_binding_proof.native_reference_environment(changed_module)
    assert before is not after
    assert before.source.module is original_module
    assert after.source.module is changed_module
    original_node = original_module.module.body[0].value
    changed_node = changed_module.module.body[0].value
    assert original_node not in after.source.reference_reads_by_node
    assert changed_node not in before.source.reference_reads_by_node
    assert (
        before.require_native(original_node, (NativeDeclaration(property),)).declaration
        is property
    )
    assert (
        after.require_native(changed_node, (NativeDeclaration(object),)).declaration
        is object
    )


def test_distinct_source_files_have_distinct_native_execution_owners():
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            "/repo/first.py": "chosen = property\n",
            "/repo/second.py": "chosen = property\n",
        }
    )
    first, second = snapshot.parsed_modules
    proof = snapshot.module_binding_proof
    first_environment = proof.native_reference_environment(first)
    second_environment = proof.native_reference_environment(second)
    assert first_environment is not second_environment
    assert first_environment.source.module is first
    assert second_environment.source.module is second
    assert (
        first.module.body[0].value
        not in second_environment.source.reference_reads_by_node
    )


def test_virtual_edit_retains_only_unaffected_module_proofs_and_rebuilds_global_queries():
    provider_path = "/repo/provider.py"
    consumer_path = "/repo/consumer.py"
    provider_source = "def render(value):\n    return value\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            provider_path: provider_source,
            consumer_path: (
                "from provider import render\n" "def run():\n" "    return render(1)\n"
            ),
        }
    )
    repository = snapshot.product_flow_repository
    consumer = snapshot.parsed_module_for_source_path(consumer_path)
    provider = snapshot.parsed_module_for_source_path(provider_path)
    consumer_source = repository.source_projection(consumer)
    consumer_execution = repository.native_reference_environment(consumer)
    original_call = repository.function_call_resolutions[0]
    assert original_call.target_resolution.declaration is not None

    changed = snapshot.with_virtual_sources(
        {
            provider_path: (
                provider_source + "def render(value):\n" + "    return None\n"
            )
        }
    )
    changed_repository = changed.product_flow_repository
    changed_consumer = changed.parsed_module_for_source_path(consumer_path)
    changed_provider = changed.parsed_module_for_source_path(provider_path)

    assert changed_repository is not repository
    assert changed_consumer is consumer
    assert changed_provider is not provider
    assert changed_repository.source_projection(consumer) is consumer_source
    assert (
        changed_repository.native_reference_environment(consumer) is consumer_execution
    )
    assert id(changed_provider) not in changed_repository._source_projections
    assert id(changed_provider) not in changed_repository._native_executions
    assert "function_call_resolutions" not in vars(changed_repository)

    changed_call = changed_repository.function_call_resolutions[0]
    assert changed_call.target_resolution.declaration is None
    assert repository.function_call_resolutions[0] is original_call


def test_virtual_source_creation_is_visible_without_recollecting_existing_source():
    provider_path = "/repo/provider.py"
    provider_source = "def render(value):\n    return value\n"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {provider_path: provider_source}
    )
    repository = snapshot.product_flow_repository
    provider = snapshot.parsed_module_for_source_path(provider_path)
    provider_projection = repository.source_projection(provider)

    changed = snapshot.with_virtual_sources(
        {
            "/repo/consumer.py": (
                "from provider import render\n" "def run():\n" "    return render(1)\n"
            )
        }
    )
    changed_repository = changed.product_flow_repository

    assert changed_repository.source_projection(provider) is provider_projection
    assert len(changed_repository.function_call_resolutions) == 1
    assert (
        changed_repository.function_call_resolutions[0].target_resolution.declaration
        is not None
    )


def test_multistage_virtual_edits_retain_each_still_current_proof_owner():
    provider_path = "/repo/provider.py"
    first_consumer_path = "/repo/first_consumer.py"
    second_consumer_path = "/repo/second_consumer.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            provider_path: "def render(value):\n    return value\n",
            first_consumer_path: (
                "from provider import render\n" "def run():\n" "    return render(1)\n"
            ),
        }
    )
    repository = snapshot.product_flow_repository
    provider = snapshot.parsed_module_for_source_path(provider_path)
    first_consumer = snapshot.parsed_module_for_source_path(first_consumer_path)
    provider_execution = repository.native_reference_environment(provider)
    first_consumer_execution = repository.native_reference_environment(first_consumer)

    added = snapshot.with_virtual_sources(
        {
            second_consumer_path: (
                "from provider import render\n" "def run():\n" "    return render(2)\n"
            )
        }
    )
    added_repository = added.product_flow_repository
    second_consumer = added.parsed_module_for_source_path(second_consumer_path)
    second_consumer_execution = added_repository.native_reference_environment(
        second_consumer
    )
    assert added_repository.native_reference_environment(provider) is provider_execution
    assert (
        added_repository.native_reference_environment(first_consumer)
        is first_consumer_execution
    )
    assert len(added_repository.resolved_function_calls) == 2

    changed = added.with_virtual_sources(
        {first_consumer_path: added.sources_by_file_path[first_consumer_path] + "\n"}
    )
    changed_repository = changed.product_flow_repository
    assert (
        changed_repository.native_reference_environment(provider) is provider_execution
    )
    assert (
        changed_repository.native_reference_environment(second_consumer)
        is second_consumer_execution
    )
    assert (
        changed.parsed_module_for_source_path(first_consumer_path) is not first_consumer
    )
    assert (
        id(changed.parsed_module_for_source_path(first_consumer_path))
        not in changed_repository._native_executions
    )
    assert len(changed_repository.resolved_function_calls) == 2


def test_repository_projection_rejects_a_foreign_source_owner():
    path = "/repo/subject.py"
    first = CodemodSourceSnapshot.from_source_mapping({path: "chosen = property\n"})
    foreign = CodemodSourceSnapshot.from_source_mapping({path: "chosen = property\n"})

    with pytest.raises(ValueError, match="different parsed module owners"):
        first.product_flow_repository.projected_with_source_projection(
            foreign.source_projection({path: "chosen = object\n"})
        )
