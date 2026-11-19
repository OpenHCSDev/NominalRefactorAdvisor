from __future__ import annotations

from dataclasses import replace

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteTargetOperation,
    ExplicitOpenRequirement,
    MaintenanceContract,
    MaintenanceContractSuite,
    MaintenanceOpen,
    MaintenanceProved,
    MaintenanceViolated,
    NoCompetingDeclaration,
    RefactorRecipe,
    ReplaceDeclaredCallTargetOperation,
    RequiredDeclaredCall,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

PATH = "/repo/subject.py"
SOURCE = '''\
class AlphaAuthority:
    @staticmethod
    def run(value):
        return value


class Legacy:
    @staticmethod
    def run(value):
        return value


class Consumer:
    def render(self, value):
        return Legacy.run(value)


class DynamicConsumer:
    def render(self, callback, value):
        return callback(value)
'''


def source_target(qualname: str) -> SourceRewriteTarget:
    return SourceRewriteTarget(file_path=PATH, qualname=qualname)


@pytest.fixture
def subject():
    snapshot = CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})
    contract = MaintenanceContract(
        "render-owner",
        (
            RequiredDeclaredCall(
                "render-result",
                consumer=source_target("Consumer.render"),
                provider=source_target("AlphaAuthority.run"),
            ),
            NoCompetingDeclaration(
                "render-result",
                competitor=source_target("Legacy"),
            ),
        ),
    )
    return snapshot, contract


def test_contract_payload_derives_nested_source_targets_and_registered_requirement():
    contract = MaintenanceContract.from_json_value(
        {
            "contract_id": "render-owner",
            "requirements": [
                {
                    "requirement": "required_declared_call",
                    "fact_id": "render-result",
                    "consumer": {
                        "file_path": PATH,
                        "target_qualname": "Consumer.render",
                    },
                    "provider": {
                        "file_path": PATH,
                        "target_qualname": "AlphaAuthority.run",
                    },
                }
            ],
        }
    )

    assert isinstance(contract.requirements[0], RequiredDeclaredCall)
    assert tuple(
        (target.file_path, target.qualname)
        for target in contract.referenced_source_targets()
    ) == (
        (PATH, "Consumer.render"),
        (PATH, "AlphaAuthority.run"),
    )


def test_terminal_contract_reports_missing_route_and_competing_provider(subject):
    snapshot, contract = subject

    result = CodemodPlanDocument(
        recipes=(RefactorRecipe("check-before"),),
        maintenance_contracts=MaintenanceContractSuite((contract,)),
    ).simulate(snapshot)

    assert result.is_clean is False
    assert tuple(
        (requirement.status, requirement.fact_id)
        for requirement in result.maintenance_contract_report.results
    ) == (
        (MaintenanceViolated, "render-result"),
        (MaintenanceViolated, "render-result"),
    )


def test_sequence_proves_required_route_and_deleted_competitor_at_final_state(subject):
    snapshot, contract = subject
    sequence = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallTargetOperation(
                target=source_target("Consumer.render"),
                callee=source_target("Legacy.run"),
                expression_source="AlphaAuthority.run",
                selection_count=SelectionCountExpectation(exact=1),
            ),
            DeleteTargetOperation(target=source_target("Legacy")),
        )
    )
    sequence = replace(
        sequence,
        maintenance_contracts=MaintenanceContractSuite((contract,)),
    )

    result = sequence.simulate(snapshot)

    assert result.is_clean is True
    assert tuple(
        (requirement.status, requirement.fact_id)
        for requirement in result.maintenance_contract_report.results
    ) == (
        (MaintenanceProved, "render-result"),
        (MaintenanceProved, "render-result"),
    )
    assert all(stage.document_simulation.is_clean for stage in result.stage_reports)
    assert result.final_snapshot.source_index.targets_by_qualname.tuple_for_key("Legacy") == ()


def test_open_requirement_is_retained_in_json_and_blocks_apply(subject):
    snapshot, _contract = subject
    contract = MaintenanceContract(
        "dynamic-owner",
        (
            ExplicitOpenRequirement(
                "dynamic-render",
                consumer=source_target("DynamicConsumer.render"),
                reason="runtime callback provider is not source-proved",
            ),
        ),
    )
    result = CodemodPlanDocument(
        recipes=(RefactorRecipe("open-edge"),),
        maintenance_contracts=MaintenanceContractSuite((contract,)),
    ).simulate(snapshot)

    assert result.is_clean is False
    assert result.maintenance_contract_report.results[0].status is MaintenanceOpen
    assert json_report_object(result)["maintenance_contract_report"]["open_count"] == 1
    with pytest.raises(ValueError, match="1 open maintenance requirement"):
        result.apply()


def test_unknown_requirement_kind_fails_at_the_plan_boundary():
    with pytest.raises(ValueError, match="Unsupported maintenance requirement"):
        CodemodPlanDocument.from_json_value(
            {
                "maintenance_contracts": [
                    {
                        "contract_id": "unknown",
                        "requirements": [
                            {"requirement": "mystery", "fact_id": "answer"}
                        ],
                    }
                ],
            }
        )


def test_contract_ids_are_unique_within_a_suite(subject):
    _snapshot, contract = subject

    with pytest.raises(ValueError, match="ids must be unique"):
        MaintenanceContractSuite((contract, contract))
