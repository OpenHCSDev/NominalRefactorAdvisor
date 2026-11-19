"""Positive maintenance relations carried and checked by codemod plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Self, cast

from metaclass_registry import AutoRegisterMeta

from nominal_refactor_advisor.case_root import Case
from nominal_refactor_advisor.codemod_call_source import (
    DeclaredCallAuthority,
    UnresolvedDeclaredCall,
)
from nominal_refactor_advisor.codemod_payload import (
    DiscriminatedPayloadRecord,
    IntegerPayloadValueCodec,
    PayloadRecordArrayValueCodec,
    PayloadRecordValueCodec,
    PayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_selector_models import (
    SelectionCountExpectation,
    SourceRewriteReferences,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import (
    JSON_REPORT_VALUE_PROJECTION,
    DataclassJsonReport,
    JsonValue,
    json_report_property,
)
from nominal_refactor_advisor.needed_input import NeededInput
from nominal_refactor_advisor.semantic_descent import AuthorityClaim

if TYPE_CHECKING:
    from nominal_refactor_advisor.codemod_runtime import (
        CodemodPlanDocumentSimulation,
        CodemodSourceSnapshot,
    )


class MaintenanceContractStatus(Case):
    """Whether source proves one admitted maintenance requirement."""

    is_clean: ClassVar[bool] = False


class MaintenanceProved(MaintenanceContractStatus):
    value = "proved"
    is_clean = True


class MaintenanceOpen(MaintenanceContractStatus):
    value = "open"


class MaintenanceViolated(MaintenanceContractStatus):
    value = "violated"


class MaintenanceEvaluationPhase(StrEnum):
    """The source state against which a requirement is meaningful."""

    TRANSITION = "transition"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class MaintenanceRequirementResult(DataclassJsonReport):
    """One admitted requirement's source-backed verdict."""

    contract_id: str
    fact_id: str
    requirement_kind: str
    phase: MaintenanceEvaluationPhase
    status: type[MaintenanceContractStatus]
    detail: str
    needs: tuple[NeededInput, ...] = ()

    @json_report_property()
    def is_clean(self) -> bool:
        return self.status.is_clean


@dataclass(frozen=True)
class MaintenanceTransitionContext:
    """One document's exact before/after source transition."""

    simulation: CodemodPlanDocumentSimulation

    @property
    def before(self) -> CodemodSourceSnapshot:
        return self.simulation.after_snapshot_projection.base_snapshot

    @property
    def after(self) -> CodemodSourceSnapshot:
        return self.simulation.required_after_snapshot


@dataclass(frozen=True)
class MaintenanceTerminalContext:
    """The final source state of a document or composed sequence."""

    snapshot: CodemodSourceSnapshot


@dataclass(frozen=True)
class MaintenanceRequirement(
    DiscriminatedPayloadRecord,
    SourceRewriteReferences,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """One positive, forbidden, competing, or explicitly open relation."""

    __registry__: ClassVar[dict[str, type[MaintenanceRequirement]]] = {}
    __registry_key__ = "requirement_key_value"
    __skip_if_no_key__ = True
    discriminator_field_name: ClassVar[str] = "requirement"
    requirement_key_value: ClassVar[str]

    fact_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    @classmethod
    def record_type_for_discriminator(cls, discriminator: str) -> type[Self]:
        requirement = cls.__registry__.get(discriminator)
        if requirement is None or not issubclass(requirement, cls):
            raise ValueError(f"Unsupported maintenance requirement: {discriminator}")
        return cast(type[Self], requirement)

    @classmethod
    def discriminator_key(cls) -> str:
        return cls.requirement_key_value

    def transition_results(
        self,
        contract_id: str,
        context: MaintenanceTransitionContext,
    ) -> tuple[MaintenanceRequirementResult, ...]:
        """No transition-local relation is declared by default."""

        return ()

    def terminal_results(
        self,
        contract_id: str,
        context: MaintenanceTerminalContext,
    ) -> tuple[MaintenanceRequirementResult, ...]:
        """No terminal relation is declared by default."""

        return ()

    def result(
        self,
        contract_id: str,
        phase: MaintenanceEvaluationPhase,
        status: type[MaintenanceContractStatus],
        detail: str,
        *needs: NeededInput,
    ) -> MaintenanceRequirementResult:
        return MaintenanceRequirementResult(
            contract_id=contract_id,
            fact_id=self.fact_id,
            requirement_kind=self.discriminator_key(),
            phase=phase,
            status=status,
            detail=detail,
            needs=NeededInput.distinct(needs) if needs else (),
        )


class TerminalMaintenanceRequirement(MaintenanceRequirement, ABC):
    """A requirement checked on the completed projected source state."""

    @abstractmethod
    def terminal_result(
        self,
        contract_id: str,
        context: MaintenanceTerminalContext,
    ) -> MaintenanceRequirementResult: ...

    def terminal_results(self, contract_id, context):
        return (self.terminal_result(contract_id, context),)


@dataclass(frozen=True)
class RequiredDeclaredCall(TerminalMaintenanceRequirement):
    """One consumer must call one provider through resolved declarations."""

    requirement_key_value = "required_declared_call"

    consumer: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    provider: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    count: int = codemod_payload_field(IntegerPayloadValueCodec(), default=1)

    def terminal_result(self, contract_id, context):
        snapshot = context.snapshot
        try:
            consumer_id = self.consumer.required_target_id(snapshot.source_index)
            provider_id = self.provider.required_target_id(snapshot.source_index)
            calls = DeclaredCallAuthority(
                snapshot=snapshot,
                caller=snapshot.source_index.target_by_id[consumer_id],
                callee=snapshot.source_index.target_by_id[provider_id],
            ).calls(SelectionCountExpectation(exact=self.count))
        except UnresolvedDeclaredCall as error:
            return self.result(
                contract_id,
                MaintenanceEvaluationPhase.TERMINAL,
                MaintenanceOpen,
                str(error),
                NeededInput.PROOF,
            )
        except ValueError as error:
            return self.result(
                contract_id,
                MaintenanceEvaluationPhase.TERMINAL,
                MaintenanceViolated,
                str(error),
                NeededInput.SOURCE_CHANGE,
            )
        return self.result(
            contract_id,
            MaintenanceEvaluationPhase.TERMINAL,
            MaintenanceProved,
            f"{len(calls)} declaration-resolved call(s) derive from the provider",
        )


@dataclass(frozen=True)
class NoCompetingDeclaration(TerminalMaintenanceRequirement):
    """One independently writable declaration must be absent at the terminal state."""

    requirement_key_value = "no_competing_declaration"

    competitor: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )

    def terminal_result(self, contract_id, context):
        source_index = context.snapshot.source_index
        target = self.competitor.optional_target_id(source_index)
        if target is not None:
            declaration = source_index.target_by_id[target]
            return self.result(
                contract_id,
                MaintenanceEvaluationPhase.TERMINAL,
                MaintenanceViolated,
                f"competing declaration remains: {declaration.qualname}",
                NeededInput.SOURCE_CHANGE,
            )
        file_path = self.competitor.optional_file_path(source_index)
        candidates = self.competitor.candidate_targets(source_index, file_path)
        if candidates:
            return self.result(
                contract_id,
                MaintenanceEvaluationPhase.TERMINAL,
                MaintenanceOpen,
                "competing declaration target is ambiguous",
                NeededInput.PROOF,
            )
        return self.result(
            contract_id,
            MaintenanceEvaluationPhase.TERMINAL,
            MaintenanceProved,
            "competing declaration is absent",
        )


@dataclass(frozen=True)
class ExplicitOpenRequirement(TerminalMaintenanceRequirement):
    """A named behavior/dynamic edge that source proof cannot yet establish."""

    requirement_key_value = "explicit_open"

    consumer: SourceRewriteTarget = codemod_payload_field(
        PayloadRecordValueCodec(SourceRewriteTarget)
    )
    reason: str = codemod_payload_field(RequiredStringPayloadValueCodec())

    def terminal_result(self, contract_id, context):
        target_id = self.consumer.required_target_id(context.snapshot.source_index)
        target = context.snapshot.source_index.target_by_id[target_id]
        return self.result(
            contract_id,
            MaintenanceEvaluationPhase.TERMINAL,
            MaintenanceOpen,
            f"{target.qualname}: {self.reason}",
            NeededInput.PROOF,
        )


@dataclass(frozen=True)
class MaintenanceContract(SourceRewriteReferences):
    """One admitted fact family and every source requirement it imposes."""

    contract_id: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    requirements: tuple[MaintenanceRequirement, ...] = codemod_payload_field(
        PayloadRecordArrayValueCodec(MaintenanceRequirement)
    )

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("Maintenance contract requires at least one requirement")

    @property
    def authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return self.records_of_type(AuthorityClaim)

    def transition_report(
        self,
        context: MaintenanceTransitionContext,
    ) -> MaintenanceContractReport:
        return MaintenanceContractReport(
            tuple(
                result
                for requirement in self.requirements
                for result in requirement.transition_results(self.contract_id, context)
            )
        )

    def terminal_report(
        self,
        context: MaintenanceTerminalContext,
    ) -> MaintenanceContractReport:
        return MaintenanceContractReport(
            tuple(
                result
                for requirement in self.requirements
                for result in requirement.terminal_results(self.contract_id, context)
            )
        )


@dataclass(frozen=True)
class MaintenanceContractReport(DataclassJsonReport):
    """Every proved, open, and violated admitted source relation."""

    results: tuple[MaintenanceRequirementResult, ...] = ()

    @classmethod
    def compose(
        cls,
        reports: tuple[MaintenanceContractReport, ...],
    ) -> MaintenanceContractReport:
        return cls(tuple(result for report in reports for result in report.results))

    @json_report_property()
    def is_clean(self) -> bool:
        return all(result.is_clean for result in self.results)

    @json_report_property()
    def open_count(self) -> int:
        return sum(result.status is MaintenanceOpen for result in self.results)

    @json_report_property()
    def violation_count(self) -> int:
        return sum(result.status is MaintenanceViolated for result in self.results)


@dataclass(frozen=True)
class MaintenanceContractSuite:
    """Nominal carrier for a plan's admitted maintenance contracts."""

    contracts: tuple[MaintenanceContract, ...] = ()

    def __post_init__(self) -> None:
        identities = tuple(contract.contract_id for contract in self.contracts)
        if len(identities) != len(set(identities)):
            raise ValueError("Maintenance contract ids must be unique")

    @property
    def is_empty(self) -> bool:
        return not self.contracts

    @property
    def authority_claims(self) -> tuple[AuthorityClaim, ...]:
        return tuple(
            dict.fromkeys(
                claim for contract in self.contracts for claim in contract.authority_claims
            )
        )

    def with_contract(self, contract: MaintenanceContract) -> MaintenanceContractSuite:
        return replace(self, contracts=(*self.contracts, contract))

    def merge(self, *suites: MaintenanceContractSuite) -> MaintenanceContractSuite:
        return type(self)(
            tuple(
                dict.fromkeys(
                    (
                        *self.contracts,
                        *(contract for suite in suites for contract in suite.contracts),
                    )
                )
            )
        )

    def transition_report(
        self,
        context: MaintenanceTransitionContext,
    ) -> MaintenanceContractReport:
        return MaintenanceContractReport.compose(
            tuple(contract.transition_report(context) for contract in self.contracts)
        )

    def terminal_report(
        self,
        context: MaintenanceTerminalContext,
    ) -> MaintenanceContractReport:
        return MaintenanceContractReport.compose(
            tuple(contract.terminal_report(context) for contract in self.contracts)
        )


@dataclass(frozen=True)
class MaintenanceContractSuitePayloadValueCodec(
    PayloadValueCodec[MaintenanceContractSuite]
):
    """Optional maintenance-contract array owned by its nominal suite."""

    def read(
        self,
        payload: Mapping[str, JsonValue],
        field_name: str,
    ) -> MaintenanceContractSuite:
        rows = payload.get(field_name, ())
        if not isinstance(rows, (list, tuple)):
            raise TypeError(f"{field_name} must be an array")
        return MaintenanceContractSuite(
            tuple(MaintenanceContract.from_json_value(row) for row in rows)
        )

    def serialize(self, value: object) -> JsonValue:
        if not isinstance(value, MaintenanceContractSuite):
            raise TypeError("maintenance-contract payload codec requires a suite")
        return JSON_REPORT_VALUE_PROJECTION.project(value.contracts)

    def nested_records(self, value: object) -> tuple[MaintenanceContract, ...]:
        if not isinstance(value, MaintenanceContractSuite):
            raise TypeError("maintenance-contract payload codec requires a suite")
        return value.contracts
