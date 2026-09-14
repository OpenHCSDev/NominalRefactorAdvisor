"""Source receipts derive revision and original nodes from one retained owner."""

import ast
import pickle
from dataclasses import fields, replace
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevision
from nominal_refactor_advisor.product_flow import source_product_flow_projection


def module(source="chosen = property\n"):
    return ParsedModule(
        Path("source_owner.py"), "source_owner", False, ast.parse(source), source
    )


def test_source_projection_retains_actual_parsed_owner_without_compact_ast_debt():
    original = module()
    source = source_product_flow_projection(original)
    assert source.module is original
    assert source.module.module.body[0].value in source.reference_reads_by_node
    assert "module" not in {field.name for field in fields(source.compact)}


def test_another_revision_reusing_ast_cannot_relabel_existing_source_observations():
    original = module()
    source = source_product_flow_projection(original)
    forged = replace(original, source="chosen = object  \n")
    assert forged.module is original.module
    assert forged.module.body[0].value in source.reference_reads_by_node
    # AST membership alone accepts both owners. Revision must derive from the
    # actual retained producer, never this independently supplied metadata.
    assert source.module is not forged
    actual_revision = CodemodSourceRevision.hash_source(source.module.source)
    assert actual_revision == CodemodSourceRevision.hash_source(original.source)
    assert actual_revision != CodemodSourceRevision.hash_source(forged.source)


def test_reparsed_source_owns_distinct_canonical_read_nodes():
    original = module()
    reparsed = module()
    source = source_product_flow_projection(original)
    other = source_product_flow_projection(reparsed)
    assert source.module is original
    assert other.module is reparsed
    assert reparsed.module.body[0].value not in source.reference_reads_by_node
    assert original.module.body[0].value not in other.reference_reads_by_node


def test_pickle_preserves_shared_module_node_identity_inside_source_projection():
    source = pickle.loads(pickle.dumps(source_product_flow_projection(module())))
    node = source.module.module.body[0].value
    assert node in source.reference_reads_by_node
    assert any(operation.node is node for operation in source.operations)
