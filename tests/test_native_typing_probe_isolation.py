"""Native binding probes must not execute unrelated analyzer cache entries."""

from pathlib import Path
import subprocess
import sys
from textwrap import dedent

CLASSVAR_COLLISION_SETUP = dedent("""
    import ast
    from pathlib import Path
    import typing

    from nominal_refactor_advisor.ast_tools import ParsedModule
    from nominal_refactor_advisor.native_compilation import NativeCreationBackend
    from nominal_refactor_advisor.native_subscription import InertNativeArgumentWitness
    from nominal_refactor_advisor.source_execution import SourceModuleExecution

    events = []

    class Collision:
        armed = False

        def __init__(self, target):
            self.target = target

        def __hash__(self):
            return hash(self.target)

        def __eq__(self, other):
            if self.armed:
                events.append("prior-cache-key-equality")
                raise RuntimeError("preexisting typing cache key executed")
            return self is other

    number = 98765432123456789123456789
    key = Collision(number)
    typing.ClassVar[key]
    key.armed = True

    def environment_for(source):
        module = ParsedModule(
            Path("cache_binding.py"), "cache_binding", False, ast.parse(source), source
        )
        return SourceModuleExecution.from_module(module)

    def assert_public_cache_effect():
        # The analyzer has not cleared/replaced the target's cache, nor proved
        # its effects safe merely by validating an inert binding witness.
        assert events == [], events
        try:
            typing.ClassVar[number]
        except RuntimeError as error:
            assert str(error) == "preexisting typing cache key executed"
        else:
            raise AssertionError("Expected actual cached subscription to run prior key")
        assert events == ["prior-cache-key-equality"], events
    """)


CLASSVAR_COLLISION_PROBE = CLASSVAR_COLLISION_SETUP + dedent("""
    environment = environment_for(f"held = {number}\\n")
    node = environment.module.module.body[0].value
    read = environment.source.value_reads_by_node[node]
    witness = InertNativeArgumentWitness(environment, read)

    # Binding is not execution of the target's cached public subscription.
    NativeCreationBackend.current().require_classvar_binding(witness)
    assert_public_cache_effect()
    print("binding-probe-isolated; public-subscription-cache-effect-preserved")
    """)


CLASSVAR_ABSOLUTE_COMPLETION_PROBE = CLASSVAR_COLLISION_SETUP + dedent("""
    source = f"from typing import ClassVar\\nheld = ClassVar[{number}]\\n"
    environment = environment_for(source)
    node = environment.module.module.body[1].value
    argument = environment.source.value_reads_by_node[node.slice]
    NativeCreationBackend.current().require_classvar_binding(
        InertNativeArgumentWitness(environment, argument)
    )
    assert events == [], events

    read = environment.source.value_reads_by_node[node]
    obligations = (
        lambda: environment.require_subscription(node),
        lambda: environment.required_prefix(read.context, read.use.position),
    )
    outcomes = []
    for obligation in obligations:
        try:
            obligation()
        except ValueError:
            outcomes.append("unproved")
        else:
            outcomes.append("closed")
    assert_public_cache_effect()
    assert outcomes == ["unproved", "unproved"], outcomes
    print("binding-useful; absolute-invocation-and-result-cut-unproved")
    """)


def run_probe(source, expected_marker):
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == expected_marker


def test_classvar_native_binding_does_not_execute_preexisting_cache_key_hooks():
    run_probe(
        CLASSVAR_COLLISION_PROBE,
        "binding-probe-isolated; public-subscription-cache-effect-preserved",
    )


def test_standard_entry_does_not_prove_cached_operation_or_post_invocation_completion():
    run_probe(
        CLASSVAR_ABSOLUTE_COMPLETION_PROBE,
        "binding-useful; absolute-invocation-and-result-cut-unproved",
    )
