"""Authoring workflow dependency planning for codemod bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .codemod import JsonObject, JsonValue


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
        if key not in self.payload:
            return ()
        value = self.payload[key]
        if not isinstance(value, list | tuple):
            raise TypeError(f"{key} must be a sequence of strings")
        values: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"{key} must contain only strings")
            values.append(item)
        return tuple(values)


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
class CodemodAuthoringActionPlan:
    """Command sequence required to reach one target workflow action."""

    target_action_id: str
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
