"""Source snapshots retain native proofs only for their own parsed source state."""

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
