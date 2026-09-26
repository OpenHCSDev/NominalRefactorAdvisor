"""Private authored async action migration using existing NRA DSL.

This is a bounded fixture, NOT a parser, detector or automatic rewrite proof.
It does not edit or invoke any running agent service. Async method/statement
movement is not supported by DispatchToPolymorphismOperation: the new request
classes and method body here are explicitly authored, then behavior-tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    ArchitectureGuardRule,
    ArchitectureGuardSuite,
    ArchitectureGuardTargetScope,
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceRevision,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    ForbiddenDispatchArchitectureGuardConstraint,
    InsertBeforeTargetOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)

BEFORE = dedent('''\
    import json
    from dataclasses import asdict

    class Rpc:
        def __init__(self, agent):
            self.agent = agent

        async def handle(self, session_id, request, writer):
            action = request.get("action")
            if action == "set_goal":
                text = request.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("A goal requires text.")
                goal = await self.agent.set_goal(session_id, text)
                writer.write((json.dumps({"result": {"goal": asdict(goal)}}) + "\\n").encode())
                await writer.drain()
            elif action == "clear_goal":
                cleared = await self.agent.clear_goal(session_id)
                writer.write((json.dumps({"result": {"cleared": cleared}}) + "\\n").encode())
                await writer.drain()
            else:
                raise ValueError(f"Unknown action: {action!r}")
''')

# An authored target architecture, not an inferred operation result. JSON
# vocabulary selection stays at the adapter boundary; per-case validation and
# invocation live with the case; transport framing is one common algorithm.
REQUESTS = dedent('''\
    class Request(ABC, metaclass=AutoRegisterMeta):
        __registry__ = {}
        __registry_key__ = "action"
        __skip_if_no_key__ = True
        action = None

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            # This bounded fixture admits concrete extensions with explicit keys;
            # abstract intermediates and live registry mutation are not covered.
            key = cls.__dict__.get("action")
            if not isinstance(key, str) or not key:
                raise ValueError("Request subclasses require an explicit action.")
            if key in Request.__registry__:
                raise ValueError(f"Duplicate action: {key!r}")

        @classmethod
        def decode(cls, payload):
            action = payload.get("action")
            # Decoded JSON may contain unhashable non-string action values.
            case = cls.__registry__.get(action) if isinstance(action, str) else None
            if case is None:
                raise ValueError(f"Unknown action: {action!r}")
            return case.from_payload(payload)

        @classmethod
        @abstractmethod
        def from_payload(cls, payload):
            raise NotImplementedError

        @abstractmethod
        async def execute(self, agent, session_id):
            raise NotImplementedError

        async def respond(self, agent, session_id, writer):
            result = await self.execute(agent, session_id)
            writer.write((json.dumps({"result": result}) + "\\n").encode())
            await writer.drain()

    class SetGoal(Request):
        action = "set_goal"

        def __init__(self, text):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("A goal requires text.")
            self.text = text

        @classmethod
        def from_payload(cls, payload):
            return cls(payload.get("text"))

        async def execute(self, agent, session_id):
            goal = await agent.set_goal(session_id, self.text)
            return {"goal": asdict(goal)}

    class ClearGoal(Request):
        action = "clear_goal"

        @classmethod
        def from_payload(cls, payload):
            return cls()

        async def execute(self, agent, session_id):
            cleared = await agent.clear_goal(session_id)
            return {"cleared": cleared}
''')


@dataclass(frozen=True)
class ActionRehearsal:
    """Private source-pinned consumer of NRA, NOT an exportable plan root.

    Exporting the underlying authored DSL without this receipt check is not a
    source-pinned replay. The existing NRA simulator and application transaction
    remain the sole execution authority; this boundary only admits the fixture.
    """

    path: str

    @property
    def guard_suite(self):
        return _authored_sequence(self.path).guard_suite

    def simulate(self, snapshot: CodemodSourceSnapshot):
        if set(snapshot.sources_by_file_path) != {self.path}:
            raise ValueError("Fixture rehearsal requires its exact one-file scope")
        actual_source = snapshot.sources_by_file_path[self.path]
        expected = CodemodSourceRevision.from_sources(self.path, {self.path: BEFORE})
        if not expected.matches_source(actual_source):
            raise ValueError("Changed fixture baseline requires a new ownership review")
        return _authored_sequence(self.path).simulate(snapshot)


def build_plan(path: str, *, source: str) -> ActionRehearsal:
    """Prepare a private rehearsal whose source receipt is checked on every use."""
    expected = CodemodSourceRevision.from_sources(path, {path: BEFORE})
    if not expected.matches_source(source):
        raise ValueError("Changed fixture baseline requires a new ownership review")
    return ActionRehearsal(path)


def _authored_sequence(path: str) -> CodemodPlanSequence:
    """Unbound authored DSL; only ActionRehearsal admits its actual snapshot."""
    module = SourceRewriteTarget(file_path=path)
    owner = SourceRewriteTarget(file_path=path, qualname="Rpc")
    handle = SourceRewriteTarget(file_path=path, qualname="Rpc.handle")
    sequence = CodemodPlanSequence.from_operations((
        EnsureImportOperation(target=module, import_source="from abc import ABC, abstractmethod"),
        EnsureImportOperation(target=module, import_source="from metaclass_registry import AutoRegisterMeta"),
        InsertBeforeTargetOperation(target=owner, source=REQUESTS),
        PatchTargetOperation(
            target=handle,
            replacements=(SourceTextReplacement(
                old_source=BEFORE[BEFORE.index("    async def handle"):].rstrip("\n"),
                new_source=("    async def handle(self, session_id, request, writer):\n"
                            "        command = Request.decode(request)\n"
                            "        await command.respond(self.agent, session_id, writer)"),
            ),),
        ),
    ))
    # Terminal guard checks source architecture, not dynamic dispatch behavior.
    guard = ArchitectureGuardRule(
        rule_id="rpc-handle-no-action-case-recovery",
        constraints=(ForbiddenDispatchArchitectureGuardConstraint(("action", 'request.get("action")', 'request["action"]')),),
        scopes=(ArchitectureGuardTargetScope(path, "Rpc.handle"),),
        reason="Action-specific validation and invocation are request-owned.",
    )
    return CodemodPlanSequence.compose((
        sequence,
        CodemodPlanDocument(guard_suite=ArchitectureGuardSuite((guard,))),
    ))
