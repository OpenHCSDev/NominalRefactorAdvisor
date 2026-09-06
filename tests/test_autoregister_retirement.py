"""Registry observations remain useful without an unsupported design prescription."""

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor import class_index as class_index_module
from nominal_refactor_advisor.ast_tools import ParsedModule, SharedRegistryRootBase
from nominal_refactor_advisor.class_index import CompactClassFamilyIndex
from nominal_refactor_advisor.codemod import AutoRegisterClassRegistryConcept
from nominal_refactor_advisor.codemod_architecture_guards import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_module_declarations import (
    AssignedSourceTopLevelDeclaration,
    NamedSourceTopLevelDeclaration,
    SourceTopLevelDeclaration,
)
from nominal_refactor_advisor.codemod_workflow import (
    CodemodRefactorGoalRunner,
    CodemodWorkflowStopReason,
)
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector


def test_registry_observations_do_not_register_a_prescriptive_rent_detector() -> None:
    declarations = IssueDetector.registered_detector_types()
    assert all(
        detector.effective_detector_id() != "autoregister_meta_under_rented"
        for detector in declarations
    )
    assert any(
        detector.effective_detector_id() == "inherited_autoregister_config_boilerplate"
        for detector in declarations
    )


@pytest.mark.parametrize(
    "source, expected_type, expected_name",
    (
        ("class Example: pass", NamedSourceTopLevelDeclaration, "Example"),
        ("def example(): pass", NamedSourceTopLevelDeclaration, "example"),
        ("EXAMPLE = 1", AssignedSourceTopLevelDeclaration, "EXAMPLE"),
    ),
)
def test_real_delegated_registry_keeps_native_selection(
    source,
    expected_type,
    expected_name,
) -> None:
    # Inherited configuration and delegated registry traversal need no new key
    # declaration, copied registry read, or consumer-count justification.
    assert (
        SourceTopLevelDeclaration.__registry_key__
        == SharedRegistryRootBase.__registry_key__
    )
    assert "__registry_key__" not in SourceTopLevelDeclaration.__dict__
    declaration = SourceTopLevelDeclaration.from_statement(
        "example.py", ast.parse(source).body[0]
    )
    assert type(declaration) is expected_type
    assert declaration.name == expected_name


def test_inherited_registry_configuration_remains_neutral_source_evidence() -> None:
    package = Path(__file__).resolve().parents[1] / "nominal_refactor_advisor"
    modules = tuple(
        ParsedModule(
            path=package / f"{name}.py",
            module_name=f"nominal_refactor_advisor.{name}",
            is_package_init=False,
            module=ast.parse(source),
            source=source,
        )
        for name in ("ast_tools", "codemod_module_declarations")
        for source in ((package / f"{name}.py").read_text(),)
    )
    partial = CompactClassFamilyIndex.from_modules(modules[1:])
    root_symbol = (
        "nominal_refactor_advisor.codemod_module_declarations.SourceTopLevelDeclaration"
    )
    partial_root = partial.class_for(root_symbol)
    assert partial_root is not None
    assert partial_root.autoregister_registry_key_attr_name is None
    complete = CompactClassFamilyIndex.from_modules(modules)
    inherited = complete.class_for(
        "nominal_refactor_advisor.ast_tools.SharedRegistryRootBase"
    )
    assert inherited is not None
    assert inherited.symbol in complete.ancestor_symbols(root_symbol)
    assert inherited.autoregister_registry_key_attr_name == "__registry_token__"


def test_registry_receiver_collection_does_not_rewalk_factory_functions(
    monkeypatch,
) -> None:
    source = (
        "def materialize():\n"
        "    Helpers.select()\n"
        "    return AutoRegisterMeta('Generated', (), {'run': transform})\n"
    )
    parsed = ParsedModule(
        path=Path("/repo/registry.py"),
        module_name="registry",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    # The module traversal is shared. Collecting its receiver index must not
    # start an additional walk merely because a function calls AutoRegisterMeta.
    class_index_module.module_syntax_index(parsed.module)

    def unexpected_walk(node):
        raise AssertionError("Registry collection repeated the shared AST walk")

    with monkeypatch.context() as patch:
        patch.setattr(ast, "walk", unexpected_walk)
        facets = class_index_module._compact_class_syntax_facets(parsed)
    index = facets.autoregister_reference_index
    assert index is not None
    assert index.function_qualnames == ("materialize",)
    assert index.receiver_names == ("Helpers",)
    assert index.attribute_names == ("select",)
    assert index.encoded_edges == "0,0,0"


def test_registry_factoring_no_longer_acquires_a_fictitious_rent_obligation(
    tmp_path,
) -> None:
    source_path = tmp_path / "registry.py"
    source = (
        "REGISTRY = {}\n"
        "class AlphaHandler:\n"
        "    pass\n"
        "class BetaHandler:\n"
        "    pass\n"
        "REGISTRY['alpha'] = AlphaHandler\n"
        "REGISTRY['beta'] = BetaHandler\n"
    )
    source_path.write_text(source)
    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(),
        parse_workers=1,
        dry_run=True,
        migration_type=AutoRegisterClassRegistryConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert report.stop_reason is CodemodWorkflowStopReason.ACHIEVED
    assert report.stages
    assert report.replay_sequence.documents
    assert report.trajectory_proof.unjustified_debt_terminals == ()
    assert source_path.read_text() == source
