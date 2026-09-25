from __future__ import annotations

import ast
import asyncio
import json
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

from action_batch import BEFORE, build_plan

from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    DispatchToPolymorphismOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)


@dataclass
class Goal:
    text: str
    revision: int = 1


def load(source: str):
    module = types.ModuleType("rpc_fixture")
    exec(compile(source, "rpc_fixture.py", "exec"), module.__dict__)  # noqa: S102 - trusted test fixture and its simulated rewrite only
    return module


def observe(source: str, payload, failure=None):
    module = load(source)
    events = []

    class Agent:
        async def set_goal(self, session_id, text):
            events.append(("set_goal", session_id, text))
            await asyncio.sleep(0)
            if failure == "agent":
                raise RuntimeError("agent failed")
            if failure == "cancel":
                raise asyncio.CancelledError("cancelled")
            return Goal({"not_json"} if failure == "serialization" else text)

        async def clear_goal(self, session_id):
            events.append(("clear_goal", session_id))
            await asyncio.sleep(0)
            return False

    class Writer:
        def write(self, data):
            events.append(("write", data))
            if failure == "write":
                raise OSError("write failed")

        async def drain(self):
            events.append(("drain",))
            await asyncio.sleep(0)
            if failure == "drain":
                raise OSError("drain failed")

    try:
        result = asyncio.run(module.Rpc(Agent()).handle("session", payload, Writer()))
    except (ValueError, RuntimeError, TypeError, OSError, asyncio.CancelledError) as error:
        return (tuple(events), (type(error).__name__, str(error)), None)
    return (tuple(events), None, result)


class ActionBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="nra-action-fixture-")
        cls.path = Path(cls.temp.name) / "rpc_fixture.py"
        cls.path.write_text(BEFORE)
        cls.snapshot = CodemodSourceSnapshot.from_source_mapping({str(cls.path): BEFORE})
        cls.plan = build_plan(str(cls.path), source=BEFORE)
        cls.result = cls.plan.simulate(cls.snapshot)
        if not cls.result.is_clean:
            raise AssertionError(cls.result)
        cls.after = cls.result.final_snapshot.sources_by_file_path[str(cls.path)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_one_batched_simulation_no_source_writes(self):
        self.assertTrue(self.result.is_clean)
        self.assertEqual(self.path.read_text(), BEFORE)
        self.assertEqual(self.snapshot.sources_by_file_path[str(self.path)], BEFORE)
        self.assertEqual(self.result.stage_count, 5)
        tree = ast.parse(self.after)
        handle = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle")
        self.assertFalse(any(isinstance(n, ast.If) for n in ast.walk(handle)))
        self.assertEqual(self.after.count("writer.write("), 1)
        self.assertEqual(self.after.count("await writer.drain()"), 1)
        module = load(self.after)
        self.assertEqual(set(module.Request.__registry__), {"set_goal", "clear_goal"})
        self.assertIs(module.Request.__registry__["set_goal"], module.SetGoal)

    def test_decoded_json_input_contract_and_unknown_actions(self):
        values = [None, False, 12, [], {}, "", "unknown", " set_goal", "set_goal", "clear_goal"]
        texts = [None, False, 12, [], {}, "", " \t", " hello ", "λ\nline"]
        for action in values:
            for text in texts:
                with self.subTest(action=action, text=text):
                    payload = json.loads(json.dumps({"action": action, "text": text}))
                    self.assertEqual(observe(BEFORE, payload), observe(self.after, payload))
        for payload in ({}, {"action": "set_goal"}, {"action": "clear_goal", "ignored": 1}):
            self.assertEqual(observe(BEFORE, payload), observe(self.after, payload))

    def test_exact_bytes_and_raw_text_not_trimmed(self):
        events, error, result = observe(self.after, {"action": "set_goal", "text": " hi "})
        self.assertIsNone(error)
        self.assertIsNone(result)
        self.assertEqual(events, (
            ("set_goal", "session", " hi "),
            ("write", b'{"result": {"goal": {"text": " hi ", "revision": 1}}}\n'),
            ("drain",),
        ))

    def test_error_and_cancellation_ordering(self):
        for failure in ("agent", "cancel", "serialization", "write", "drain"):
            with self.subTest(failure=failure):
                payload = {"action": "set_goal", "text": "hello"}
                self.assertEqual(observe(BEFORE, payload, failure), observe(self.after, payload, failure))

    def test_existing_automatic_dispatch_operation_refuses_async_method(self):
        recipe = RefactorRecipe(recipe_id="not-an-automatic-async-rewrite").with_operation(
            DispatchToPolymorphismOperation(target=SourceRewriteTarget(file_path=str(self.path), qualname="Rpc.handle"))
        )
        with self.assertRaisesRegex(ValueError, "function target"):
            recipe.operations[0].source_edits(self.snapshot)
        self.assertEqual(self.path.read_text(), BEFORE)

    def test_architecture_guard_rejects_original_case_recovery(self):
        self.assertFalse(self.plan.guard_suite.evaluate(
            self.snapshot.source_index, self.snapshot.sources_by_file_path
        ).is_clean)
        self.assertTrue(self.result.architecture_guard_report.is_clean)

    def test_changed_baseline_rejects_authoring_and_replay(self):
        changed = BEFORE.replace('        action = request.get("action")', '        audit(request)\n        action = request.get("action")')
        with self.assertRaisesRegex(ValueError, "Changed fixture baseline"):
            build_plan(str(self.path), source=changed)
        changed_snapshot = CodemodSourceSnapshot.from_source_mapping({str(self.path): changed})
        with self.assertRaises(ValueError):
            self.plan.simulate(changed_snapshot)
        self.assertEqual(self.path.read_text(), BEFORE)

    def test_replay_revalidates_whole_module_and_context_not_only_method(self):
        for suffix in ('\nRequest = None\n', '\n# harmless but unreviewed\n', '\njson = None\n'):
            with self.subTest(suffix=suffix):
                changed = BEFORE + suffix
                snapshot = CodemodSourceSnapshot.from_source_mapping({str(self.path): changed})
                with self.assertRaisesRegex(ValueError, "Changed fixture baseline"):
                    self.plan.simulate(snapshot)
                self.assertEqual(self.path.read_text(), BEFORE)
        extra_scope = CodemodSourceSnapshot.from_source_mapping({str(self.path): BEFORE, str(self.path.with_name('extra.py')): 'Request = None\n'})
        with self.assertRaisesRegex(ValueError, "exact one-file scope"):
            self.plan.simulate(extra_scope)

    def test_guard_covers_local_action_and_direct_payload_spellings_only(self):
        for expression in ('action', 'request.get("action")', 'request["action"]'):
            changed = self.after.replace('        command = Request.decode(request)', f'        if {expression} == "set_goal":\n            return None\n        command = Request.decode(request)')
            snapshot = CodemodSourceSnapshot.from_source_mapping({str(self.path): changed})
            self.assertFalse(self.plan.guard_suite.evaluate(snapshot.source_index, snapshot.sources_by_file_path).is_clean)

    def test_duplicate_and_inherited_action_keys_rejected_without_overwriting(self):
        module = load(self.after)
        original = dict(module.Request.__registry__)
        with self.assertRaisesRegex(ValueError, "Duplicate action"):
            type("Duplicate", (module.SetGoal,), {"action": "set_goal"})
        with self.assertRaisesRegex(ValueError, "explicit action"):
            type("Inherited", (module.SetGoal,), {})
        self.assertEqual(module.Request.__registry__, original)

    def test_application_rejects_changed_physical_source(self):
        stale = BEFORE + "\n# unrelated concurrent edit\n"
        self.path.write_text(stale)
        try:
            with self.assertRaises(ValueError):
                self.result.apply()
            self.assertEqual(self.path.read_text(), stale)
        finally:
            self.path.write_text(BEFORE)

    def test_add_case_without_editing_transport_or_case_roster(self):
        module = load(self.after)
        old_handle = module.Rpc.handle
        old_respond = module.Request.respond
        class Ping(module.Request):
            action = "ping"
            @classmethod
            def from_payload(cls, payload):
                return cls()
            async def execute(self, agent, session_id):
                return {"pong": True}
        self.assertIs(module.Request.__registry__["ping"], Ping)
        self.assertIs(module.Rpc.handle, old_handle)
        self.assertIs(module.Request.respond, old_respond)
        self.assertIsInstance(module.Request.decode({"action": "ping"}), Ping)
        emitted = []
        class Writer:
            def write(self, data):
                emitted.append(data)
            async def drain(self):
                emitted.append("drained")
        asyncio.run(module.Rpc(object()).handle("session", {"action": "ping"}, Writer()))
        self.assertEqual(emitted, [b'{"result": {"pong": true}}\n', "drained"])


if __name__ == "__main__":
    unittest.main()
