"""Authoring workflow dependency planning for codemod bundles."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .codemod import JsonObject, JsonValue

PayloadItem = TypeVar("PayloadItem")


@dataclass(frozen=True)
class CodemodAuthoringPayloadReader:
    """Typed field reader for authoring workflow JSON payloads."""

    payload: Mapping[str, JsonValue]

    def string(self, key: str) -> str:
        value = self.payload[key]
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string")
        return value

    def string_tuple(self, key: str) -> tuple[str, ...]:
        return self.typed_tuple(key, str, "strings")

    def object_tuple(self, key: str) -> tuple[JsonObject, ...]:
        return self.typed_tuple(key, dict, "objects")

    def typed_tuple(
        self,
        key: str,
        item_type: type[PayloadItem],
        item_label: str,
    ) -> tuple[PayloadItem, ...]:
        values: list[PayloadItem] = []
        for item in self.sequence(key):
            if not isinstance(item, item_type):
                raise TypeError(f"{key} must contain only {item_label}")
            values.append(item)
        return tuple(values)

    def optional_string(self, key: str) -> str | None:
        value = self.payload.get(key)
        if value is None or isinstance(value, str):
            return value
        raise TypeError(f"{key} must be a string")

    def optional_int(self, key: str) -> int | None:
        value = self.payload.get(key)
        if value is None or isinstance(value, int):
            return value
        raise TypeError(f"{key} must be an integer")

    def integer(self, key: str) -> int:
        value = self.payload[key]
        if isinstance(value, int):
            return value
        raise TypeError(f"{key} must be an integer")

    def sequence(self, key: str) -> tuple[JsonValue, ...]:
        value = self.payload[key]
        if not isinstance(value, list | tuple):
            raise TypeError(f"{key} must be a sequence")
        return tuple(value)


@dataclass(frozen=True)
class CodemodAuthoringCommandModel:
    """Executable command node in an authoring workflow artifact graph."""

    action_id: str
    required_artifacts: tuple[str, ...]
    generated_artifacts: tuple[str, ...]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "CodemodAuthoringCommandModel":
        reader = CodemodAuthoringPayloadReader(payload)
        return cls(
            action_id=reader.string("action_id"),
            required_artifacts=reader.string_tuple("required_artifacts"),
            generated_artifacts=reader.string_tuple("generated_artifacts"),
        )


@dataclass(frozen=True)
class CodemodAuthoringWorkflowModel:
    """Ordered command set for one authoring workflow."""

    workflow_id: str
    command_action_ids: tuple[str, ...]
    default_next_action_id: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "CodemodAuthoringWorkflowModel":
        reader = CodemodAuthoringPayloadReader(payload)
        return cls(
            workflow_id=reader.string("workflow_id"),
            command_action_ids=reader.string_tuple("command_action_ids"),
            default_next_action_id=reader.string("default_next_action_id"),
        )


@dataclass(frozen=True)
class CodemodAuthoringCommandReadiness:
    """Current artifact readiness for a command node."""

    command: CodemodAuthoringCommandModel
    missing_artifacts: tuple[str, ...]

    @property
    def runnable(self) -> bool:
        return not self.missing_artifacts

    def to_dict(self) -> JsonObject:
        return {
            "action_id": self.command.action_id,
            "required_artifacts": self.command.required_artifacts,
            "generated_artifacts": self.command.generated_artifacts,
            "missing_artifacts": self.missing_artifacts,
            "runnable": self.runnable,
        }


@dataclass(frozen=True)
class CodemodAuthoringRecordReference:
    """Reference to one record in an authoring bundle."""

    record_index: int


@dataclass(frozen=True)
class CodemodAuthoringTargetAction:
    """Target action id within an authoring workflow."""

    target_action_id: str


@dataclass(frozen=True)
class CodemodAuthoringActionPlan(CodemodAuthoringTargetAction):
    """Command sequence required to reach one target workflow action."""

    action_ids: tuple[str, ...]
    missing_artifacts: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return bool(self.missing_artifacts)

    @property
    def first_action_id(self) -> str | None:
        for action_id in self.action_ids:
            return action_id
        return None

    def to_dict(self) -> JsonObject:
        return {
            "target_action_id": self.target_action_id,
            "action_ids": self.action_ids,
            "missing_artifacts": self.missing_artifacts,
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class CodemodAuthoringWorkflowReadiness:
    """Runnable commands and dependency plans for one workflow."""

    workflow: CodemodAuthoringWorkflowModel
    next_action_id: str | None
    command_readiness: tuple[CodemodAuthoringCommandReadiness, ...]
    action_plans: tuple[CodemodAuthoringActionPlan, ...]

    @property
    def runnable_action_ids(self) -> tuple[str, ...]:
        return tuple(
            readiness.command.action_id
            for readiness in self.command_readiness
            if readiness.runnable
        )

    @property
    def blocked_action_ids(self) -> tuple[str, ...]:
        return tuple(
            readiness.command.action_id
            for readiness in self.command_readiness
            if not readiness.runnable
        )

    def to_dict(self) -> JsonObject:
        return {
            "workflow_id": self.workflow.workflow_id,
            "default_next_action_id": self.workflow.default_next_action_id,
            "next_action_id": self.next_action_id,
            "runnable_action_ids": self.runnable_action_ids,
            "blocked_action_ids": self.blocked_action_ids,
            "command_readiness": tuple(
                readiness.to_dict() for readiness in self.command_readiness
            ),
            "action_plans": tuple(plan.to_dict() for plan in self.action_plans),
        }


@dataclass(frozen=True)
class CodemodAuthoringBundleReadiness:
    """Readiness report for every workflow in one authoring bundle record."""

    available_artifacts: tuple[str, ...]
    workflows: tuple[CodemodAuthoringWorkflowReadiness, ...]

    def to_dict(self) -> JsonObject:
        return {
            "available_artifacts": self.available_artifacts,
            "workflows": tuple(workflow.to_dict() for workflow in self.workflows),
        }


@dataclass(frozen=True)
class CodemodAuthoringArtifactInventory:
    """Filesystem-backed artifact inventory for one bundle record."""

    bundle_root: Path
    commands: tuple[CodemodAuthoringCommandModel, ...]

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        artifacts: list[str] = []
        for command in self.commands:
            artifacts.extend(command.required_artifacts)
            artifacts.extend(command.generated_artifacts)
        return tuple(dict.fromkeys(artifacts))

    @property
    def available_artifacts(self) -> tuple[str, ...]:
        return tuple(
            artifact_path
            for artifact_path in self.artifact_paths
            if (self.bundle_root / artifact_path).exists()
        )


@dataclass(frozen=True)
class CodemodAuthoringBundleRecordStatus(CodemodAuthoringRecordReference):
    """Current workflow status for one authoring bundle record."""

    finding_id: str | None
    detector_id: str | None
    readiness: CodemodAuthoringBundleReadiness

    def to_dict(self) -> JsonObject:
        return {
            "record_index": self.record_index,
            "finding_id": self.finding_id,
            "detector_id": self.detector_id,
            "workflow_readiness": self.readiness.to_dict(),
        }


@dataclass(frozen=True)
class CodemodAuthoringBundleStatus:
    """Current workflow status for an authoring bundle index."""

    bundle_index_path: Path
    records: tuple[CodemodAuthoringBundleRecordStatus, ...]

    @property
    def bundle_root(self) -> Path:
        return self.bundle_index_path.parent

    def to_dict(self) -> JsonObject:
        return {
            "bundle_index_path": self.bundle_index_path.as_posix(),
            "bundle_root": self.bundle_root.as_posix(),
            "records": tuple(record.to_dict() for record in self.records),
        }


@dataclass(frozen=True)
class CodemodAuthoringBundleStatusReporter:
    """Recompute authoring bundle readiness from the current filesystem."""

    bundle_index_path: Path
    bundle_index: Mapping[str, JsonValue]

    @classmethod
    def from_index_path(
        cls,
        bundle_index_path: Path,
    ) -> "CodemodAuthoringBundleStatusReporter":
        payload = json.loads(bundle_index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("authoring bundle index JSON must be an object")
        return cls(bundle_index_path, payload)

    @property
    def bundle_root(self) -> Path:
        return self.bundle_index_path.parent

    def status(self) -> CodemodAuthoringBundleStatus:
        reader = CodemodAuthoringPayloadReader(self.bundle_index)
        return CodemodAuthoringBundleStatus(
            bundle_index_path=self.bundle_index_path,
            records=tuple(
                self.record_status(record_payload)
                for record_payload in reader.object_tuple("records")
            ),
        )

    def record_status(
        self,
        record_payload: Mapping[str, JsonValue],
    ) -> CodemodAuthoringBundleRecordStatus:
        reader = CodemodAuthoringPayloadReader(record_payload)
        commands = self.commands(record_payload)
        workflows = self.workflows(record_payload)
        available_artifacts = CodemodAuthoringArtifactInventory(
            self.bundle_root,
            commands,
        ).available_artifacts
        return CodemodAuthoringBundleRecordStatus(
            record_index=reader.integer("record_index"),
            finding_id=reader.optional_string("finding_id"),
            detector_id=reader.optional_string("detector_id"),
            readiness=CodemodAuthoringWorkflowPlanner(
                commands=commands,
                workflows=workflows,
            ).bundle_readiness(available_artifacts),
        )

    def record_payload_at(
        self,
        record_index: int,
    ) -> JsonObject:
        reader = CodemodAuthoringPayloadReader(self.bundle_index)
        for position, record_payload in enumerate(reader.object_tuple("records")):
            record_reader = CodemodAuthoringPayloadReader(record_payload)
            payload_record_index = record_reader.optional_int("record_index")
            if payload_record_index == record_index:
                return record_payload
            if payload_record_index is None and position == record_index:
                return record_payload
        raise IndexError(f"authoring bundle record {record_index} was not found")

    def commands(
        self,
        record_payload: Mapping[str, JsonValue],
    ) -> tuple[CodemodAuthoringCommandModel, ...]:
        reader = CodemodAuthoringPayloadReader(record_payload)
        return tuple(
            CodemodAuthoringCommandModel.from_payload(command_payload)
            for command_payload in reader.object_tuple("commands")
        )

    def workflows(
        self,
        record_payload: Mapping[str, JsonValue],
    ) -> tuple[CodemodAuthoringWorkflowModel, ...]:
        reader = CodemodAuthoringPayloadReader(record_payload)
        return tuple(
            CodemodAuthoringWorkflowModel.from_payload(workflow_payload)
            for workflow_payload in reader.object_tuple("workflows")
        )


@dataclass(frozen=True)
class CodemodAuthoringCommandInvocation:
    """Executable argv/cwd pair for one authoring command."""

    action_id: str
    argv: tuple[str, ...]
    cwd: Path

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, JsonValue],
    ) -> "CodemodAuthoringCommandInvocation":
        reader = CodemodAuthoringPayloadReader(payload)
        return cls(
            action_id=reader.string("action_id"),
            argv=reader.string_tuple("argv"),
            cwd=Path(reader.string("cwd")),
        )

    def run(self) -> "CodemodAuthoringCommandRun":
        result = subprocess.run(
            self.argv,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return CodemodAuthoringCommandRun(
            invocation=self,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


@dataclass(frozen=True)
class CodemodAuthoringCommandRun:
    """Captured result for one executed authoring command."""

    invocation: CodemodAuthoringCommandInvocation
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def stdout_json(self) -> JsonValue:
        try:
            return json.loads(self.stdout)
        except json.JSONDecodeError:
            return None

    def to_dict(self) -> JsonObject:
        return {
            "action_id": self.invocation.action_id,
            "argv": self.invocation.argv,
            "cwd": self.invocation.cwd.as_posix(),
            "returncode": self.returncode,
            "succeeded": self.succeeded,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_json": self.stdout_json,
        }


@dataclass(frozen=True)
class CodemodAuthoringActionRunRequest(
    CodemodAuthoringRecordReference,
    CodemodAuthoringTargetAction,
):
    """Target action request for one authoring bundle record."""

    workflow_id: str | None


@dataclass(frozen=True)
class CodemodAuthoringActionRunReport(CodemodAuthoringActionRunRequest):
    """Execution report for one planned authoring action chain."""

    workflow: CodemodAuthoringWorkflowModel
    action_plan: CodemodAuthoringActionPlan
    command_runs: tuple[CodemodAuthoringCommandRun, ...]

    @property
    def completed(self) -> bool:
        return (
            not self.action_plan.blocked
            and len(self.command_runs) == len(self.action_plan.action_ids)
            and all(command_run.succeeded for command_run in self.command_runs)
        )

    @property
    def exit_code(self) -> int:
        if self.completed:
            return 0
        return 1

    def to_dict(self) -> JsonObject:
        return {
            "record_index": self.record_index,
            "workflow_id": self.workflow.workflow_id,
            "target_action_id": self.target_action_id,
            "completed": self.completed,
            "action_plan": self.action_plan.to_dict(),
            "command_runs": tuple(
                command_run.to_dict() for command_run in self.command_runs
            ),
        }


@dataclass(frozen=True)
class CodemodAuthoringBundleActionRunner(CodemodAuthoringActionRunRequest):
    """Execute one target action through the bundle workflow planner."""

    bundle_index_path: Path

    def run(self) -> CodemodAuthoringActionRunReport:
        reporter = CodemodAuthoringBundleStatusReporter.from_index_path(
            self.bundle_index_path
        )
        record_payload = reporter.record_payload_at(self.record_index)
        commands = reporter.commands(record_payload)
        workflows = reporter.workflows(record_payload)
        workflow = self.selected_workflow(workflows)
        action_plan = CodemodAuthoringWorkflowPlanner(
            commands=commands,
            workflows=workflows,
        ).plan_to_action(
            self.target_action_id,
            CodemodAuthoringArtifactInventory(
                reporter.bundle_root,
                commands,
            ).available_artifacts,
        )
        return CodemodAuthoringActionRunReport(
            record_index=self.record_index,
            workflow_id=self.workflow_id,
            target_action_id=self.target_action_id,
            workflow=workflow,
            action_plan=action_plan,
            command_runs=self.command_runs(record_payload, action_plan),
        )

    def selected_workflow(
        self,
        workflows: tuple[CodemodAuthoringWorkflowModel, ...],
    ) -> CodemodAuthoringWorkflowModel:
        if self.workflow_id is not None:
            for workflow in workflows:
                if workflow.workflow_id == self.workflow_id:
                    self.require_workflow_contains_target(workflow)
                    return workflow
            raise ValueError(f"workflow {self.workflow_id!r} was not found")
        candidate_workflows = tuple(
            workflow
            for workflow in workflows
            if self.target_action_id in workflow.command_action_ids
        )
        if len(candidate_workflows) != 1:
            raise ValueError(
                f"target action {self.target_action_id!r} matched "
                f"{len(candidate_workflows)} workflows"
            )
        return candidate_workflows[0]

    def require_workflow_contains_target(
        self,
        workflow: CodemodAuthoringWorkflowModel,
    ) -> None:
        if self.target_action_id not in workflow.command_action_ids:
            raise ValueError(
                f"workflow {workflow.workflow_id!r} does not contain target action "
                f"{self.target_action_id!r}"
            )

    def command_runs(
        self,
        record_payload: Mapping[str, JsonValue],
        action_plan: CodemodAuthoringActionPlan,
    ) -> tuple[CodemodAuthoringCommandRun, ...]:
        if action_plan.blocked:
            return ()
        invocations_by_action_id = self.invocations_by_action_id(record_payload)
        command_runs: list[CodemodAuthoringCommandRun] = []
        for action_id in action_plan.action_ids:
            command_run = invocations_by_action_id[action_id].run()
            command_runs.append(command_run)
            if not command_run.succeeded:
                break
        return tuple(command_runs)

    @staticmethod
    def invocations_by_action_id(
        record_payload: Mapping[str, JsonValue],
    ) -> dict[str, CodemodAuthoringCommandInvocation]:
        reader = CodemodAuthoringPayloadReader(record_payload)
        return {
            invocation.action_id: invocation
            for invocation in (
                CodemodAuthoringCommandInvocation.from_payload(command_payload)
                for command_payload in reader.object_tuple("commands")
            )
        }


@dataclass(frozen=True)
class CodemodAuthoringWorkflowPlanner:
    """Plan authoring workflow command chains from artifact dependencies."""

    commands: tuple[CodemodAuthoringCommandModel, ...]
    workflows: tuple[CodemodAuthoringWorkflowModel, ...]

    @classmethod
    def from_payloads(
        cls,
        command_payloads: tuple[JsonObject, ...],
        workflow_payloads: tuple[JsonObject, ...],
    ) -> "CodemodAuthoringWorkflowPlanner":
        return cls(
            commands=tuple(
                CodemodAuthoringCommandModel.from_payload(payload)
                for payload in command_payloads
            ),
            workflows=tuple(
                CodemodAuthoringWorkflowModel.from_payload(payload)
                for payload in workflow_payloads
            ),
        )

    def bundle_readiness(
        self,
        available_artifacts: tuple[str, ...],
    ) -> CodemodAuthoringBundleReadiness:
        return CodemodAuthoringBundleReadiness(
            available_artifacts=available_artifacts,
            workflows=tuple(
                self.workflow_readiness(workflow, available_artifacts)
                for workflow in self.workflows
            ),
        )

    def workflow_readiness(
        self,
        workflow: CodemodAuthoringWorkflowModel,
        available_artifacts: tuple[str, ...],
    ) -> CodemodAuthoringWorkflowReadiness:
        command_readiness = tuple(
            self.command_readiness(action_id, available_artifacts)
            for action_id in workflow.command_action_ids
        )
        action_plans = tuple(
            self.plan_to_action(action_id, available_artifacts)
            for action_id in workflow.command_action_ids
        )
        default_plan = self.plan_to_action(
            workflow.default_next_action_id,
            available_artifacts,
        )
        return CodemodAuthoringWorkflowReadiness(
            workflow=workflow,
            next_action_id=default_plan.first_action_id,
            command_readiness=command_readiness,
            action_plans=action_plans,
        )

    def command_readiness(
        self,
        action_id: str,
        available_artifacts: tuple[str, ...],
    ) -> CodemodAuthoringCommandReadiness:
        command = self.command_by_action_id()[action_id]
        available = frozenset(available_artifacts)
        return CodemodAuthoringCommandReadiness(
            command=command,
            missing_artifacts=tuple(
                artifact
                for artifact in command.required_artifacts
                if artifact not in available
            ),
        )

    def plan_to_action(
        self,
        target_action_id: str,
        available_artifacts: tuple[str, ...],
    ) -> CodemodAuthoringActionPlan:
        command_by_action_id = self.command_by_action_id()
        generators_by_artifact = self.generators_by_artifact()
        available = set(available_artifacts)
        action_ids: list[str] = []
        missing_artifacts: list[str] = []

        def restore_plan_state(
            available_snapshot: set[str],
            action_id_snapshot: list[str],
        ) -> None:
            available.clear()
            available.update(available_snapshot)
            action_ids[:] = action_id_snapshot

        def ensure_artifact(artifact: str, visiting_artifacts: frozenset[str]) -> bool:
            if artifact in available:
                return True
            if artifact in visiting_artifacts:
                missing_artifacts.append(artifact)
                return False
            generators = generators_by_artifact.get(artifact)
            if generators is None:
                missing_artifacts.append(artifact)
                return False
            next_visiting_artifacts = visiting_artifacts | frozenset((artifact,))
            generator_missing_artifacts: list[str] = []
            for generator in generators:
                available_snapshot = set(available)
                action_id_snapshot = list(action_ids)
                missing_start = len(missing_artifacts)
                if ensure_command(generator, next_visiting_artifacts):
                    return True
                generator_missing_artifacts.extend(missing_artifacts[missing_start:])
                del missing_artifacts[missing_start:]
                restore_plan_state(available_snapshot, action_id_snapshot)
            if generator_missing_artifacts:
                missing_artifacts.extend(generator_missing_artifacts)
            else:
                missing_artifacts.append(artifact)
            return False

        def ensure_command(
            command: CodemodAuthoringCommandModel,
            visiting_artifacts: frozenset[str],
        ) -> bool:
            available_snapshot = set(available)
            action_id_snapshot = list(action_ids)
            unresolved_requirements = tuple(
                artifact
                for artifact in command.required_artifacts
                if not ensure_artifact(artifact, visiting_artifacts)
            )
            if unresolved_requirements:
                restore_plan_state(available_snapshot, action_id_snapshot)
                return False
            if command.action_id not in action_ids:
                action_ids.append(command.action_id)
            available.update(command.generated_artifacts)
            return True

        ensure_command(command_by_action_id[target_action_id], frozenset())
        return CodemodAuthoringActionPlan(
            target_action_id=target_action_id,
            action_ids=tuple(action_ids),
            missing_artifacts=tuple(dict.fromkeys(missing_artifacts)),
        )

    def command_by_action_id(self) -> dict[str, CodemodAuthoringCommandModel]:
        return {command.action_id: command for command in self.commands}

    def generators_by_artifact(
        self,
    ) -> dict[str, tuple[CodemodAuthoringCommandModel, ...]]:
        generators: dict[str, list[CodemodAuthoringCommandModel]] = {}
        for command in self.commands:
            for artifact in command.generated_artifacts:
                if artifact not in generators:
                    generators[artifact] = []
                generators[artifact].append(command)
        return {
            artifact: tuple(artifact_generators)
            for artifact, artifact_generators in generators.items()
        }
