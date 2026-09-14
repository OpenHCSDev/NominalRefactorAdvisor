"""Native call execution, captured result and copy timing remain separate."""

import ast
import builtins
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_call import (
    CopiedNativeNamespace,
    NativeDictCopyCall,
    NativeVarsCall,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def execution(source):
    return SourceModuleExecution.from_module(
        ParsedModule(
            Path("native_copy.py"), "native_copy", False, ast.parse(source), source
        )
    )


def captured_final(source):
    environment = execution(source)
    node = environment.module.module.body[-1].value
    return environment, environment.capture_value(node)


def test_vars_borrows_the_admitted_actual_storage():
    environment, result = captured_final(
        "import builtins\nborrowed = vars(builtins)\nresult = borrowed\n"
    )
    assert isinstance(result, CapturedNativeObject)
    assert result.value is vars(builtins)
    assert result.dictionary_namespace(
        environment.initial
    ) is environment.initial.namespace_for_storage(vars(builtins))


def test_copy_is_one_symbolic_creation_not_an_analyzer_dictionary():
    environment, result = captured_final(
        "import builtins\ncopied = dict(vars(builtins))\nalias = copied\nresult = alias\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    assert not isinstance(result, CapturedNativeObject)
    assert environment.call_result(result.context, result.call) is result
    assert environment.capture_value(environment.module.module.body[-1].value) is result
    assert result.as_builtin_namespace(environment.initial) is result
    assert (
        result.member("property")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_keyword_value_keeps_earlier_capture_after_its_name_changes():
    _, result = captured_final(
        "import builtins\nselected = builtins.property\n"
        "copied = dict(vars(builtins), property=selected)\n"
        "selected = builtins.object\nresult = copied\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    assert (
        result.member("property")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_copy_reads_parent_at_outer_call_and_ignores_later_parent_writes():
    _, result = captured_final(
        "import builtins\ncopied = dict(vars(builtins))\n"
        "builtins.property = builtins.object\nresult = copied\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    assert (
        result.member("property")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_callee_alias_keeps_native_identity_after_its_module_slot_changes():
    _, result = captured_final(
        "import builtins\nconstructor = builtins.dict\n"
        "builtins.dict = builtins.object\nresult = constructor(vars(builtins))\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    assert (
        result.member("property")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


def test_explicit_keyword_overrides_parent_write_with_earlier_value():
    _, result = captured_final(
        "import builtins\nselected = builtins.property\n"
        "builtins.property = builtins.object\n"
        "result = dict(vars(builtins), property=selected)\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    assert (
        result.member("property")
        .require_native_identity(NativeDeclaration(property))
        .declaration
        is property
    )


@pytest.mark.parametrize(
    "expression",
    (
        "vars()",
        "vars(builtins, builtins)",
        "vars(vars(builtins))",
        "dict(builtins)",
        "dict(*[vars(builtins)])",
        "dict(vars(builtins), **vars(builtins))",
    ),
)
def test_unsupported_native_argument_protocol_is_open(expression):
    _, result = captured_final(f"import builtins\nresult = {expression}\n")
    assert isinstance(result, OpenCapturedReference)


def test_shadowed_same_named_function_is_not_a_native_protocol():
    _, result = captured_final(
        "import builtins\ndef dict(value): return value\nresult = dict(vars(builtins))\n"
    )
    assert isinstance(result, OpenCapturedReference)


def test_actual_call_rejects_foreign_context_and_copied_operand():
    environment = execution("import builtins\nresult = dict(vars(builtins))\n")
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    with pytest.raises(ValueError, match="original"):
        environment.source_call(ast.parse(ast.unparse(node), mode="eval").body)
    with pytest.raises(ValueError, match="canonical context"):
        environment.call_result(replace(context), call)
    with pytest.raises(ValueError, match="unique original"):
        environment.call_result(context, replace(call))


def test_bound_protocol_rejects_a_copied_source_operation():
    environment = execution("import builtins\nresult = vars(builtins)\n")
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    operation = environment.source_operation(context, call)
    protocol = NativeVarsCall(environment, operation)
    assert protocol.node is node
    assert protocol.call is call
    assert protocol.context is context
    with pytest.raises(ValueError, match="original canonical"):
        NativeVarsCall(environment, replace(operation))


def test_direct_wrong_leaf_cannot_claim_another_calls_native_protocol():
    environment = execution("import builtins\nresult = dict(builtins)\n")
    context, call = environment.source_call(environment.module.module.body[-1].value)
    with pytest.raises(ValueError, match="not the required native declaration"):
        NativeVarsCall(environment, environment.source_operation(context, call))


def test_direct_constructor_does_not_skip_keyword_evaluation_effects():
    environment = execution(
        "import builtins\nresult = dict(vars(builtins), property=unknown())\n"
    )
    context, call = environment.source_call(environment.module.module.body[-1].value)
    with pytest.raises(ValueError, match="capture remains open"):
        NativeDictCopyCall(environment, environment.source_operation(context, call))


@pytest.mark.parametrize(
    "shape", ("refinement", "multiple_inheritance", "incomparable")
)
def test_native_call_refinement_uses_mro_not_family_enumeration_order(shape):
    # Trusted test protocol declarations are process-local so their nominal
    # descendants cannot alter another test's production family discovery.
    definitions = {
        "refinement": "class Selected(NativeVarsCall): pass\n",
        "multiple_inheritance": "class Left(NativeVarsCall): pass\nclass Right(NativeVarsCall): pass\nclass Selected(Left, Right): pass\n",
        "incomparable": "class Left(NativeVarsCall): pass\nclass Right(NativeVarsCall): pass\n",
    }
    program = (
        """
import ast
from pathlib import Path
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.native_call import NativeCallAuthority, NativeVarsCall
from nominal_refactor_advisor.source_execution import SourceModuleExecution
"""
        + definitions[shape]
        + """
source = "import builtins\\nresult = vars(builtins)\\n"
environment = SourceModuleExecution.from_module(ParsedModule(Path("mro.py"), "mro", False, ast.parse(source), source))
context, call = environment.source_call(environment.module.module.body[-1].value)
"""
    )
    if shape == "incomparable":
        program += """
try:
    NativeCallAuthority.for_call(environment, context, call)
except ValueError as error:
    assert "most-specific" in str(error)
else:
    raise AssertionError("Incomparable protocols must stay open")
"""
    else:
        program += "assert type(NativeCallAuthority.for_call(environment, context, call)) is Selected\n"
    native = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert native.returncode == 0, native.stderr


def test_duplicate_original_event_cannot_reuse_result_cache():
    environment = execution("import builtins\nresult = dict(vars(builtins))\n")
    node = environment.module.module.body[-1].value
    context, call = environment.source_call(node)
    environment.call_result(context, call)
    operation = environment.source_operation(context, call)
    duplicate_source = replace(
        environment.source, operations=(*environment.source.operations, operation)
    )
    environment.entry.__dict__["source"] = duplicate_source
    with pytest.raises(ValueError, match="unique original"):
        environment.call_result(context, call)


def test_forged_copy_is_not_the_canonical_creation():
    environment, result = captured_final(
        "import builtins\ncopied = dict(vars(builtins))\nresult = copied\n"
    )
    assert isinstance(result, CopiedNativeNamespace)
    with pytest.raises(ValueError, match="canonical admitted"):
        replace(result).require_admitted(environment.initial)
    foreign = execution("import builtins\n")
    with pytest.raises(ValueError, match="canonical admitted"):
        result.require_admitted(foreign.initial)


def test_chained_assignment_result_remains_explicitly_unsupported():
    _, result = captured_final(
        "import builtins\nfirst = second = dict(vars(builtins))\nresult = second\n"
    )
    assert isinstance(result, OpenCapturedReference)


def test_descriptor_execution_does_not_fabricate_an_actual_descriptor():
    environment = execution("result = property()\n")
    node = environment.module.module.body[0].value
    environment.require_call(node)
    result = environment.capture_value(node)
    result.require_class_installation()
    with pytest.raises(ValueError):
        result.require_native_identity(NativeDeclaration(property))


def test_native_copy_timing_controls_execute_only_in_isolated_process():
    source = """
import builtins
borrowed = vars(builtins)
assert borrowed is builtins.__dict__
parent = {"selected": object()}
before = parent["selected"]
def keyword():
    parent["selected"] = object()
    return before
copied = dict(parent, earlier=keyword())
assert copied["selected"] is parent["selected"]
assert copied["earlier"] is before
retained = copied["selected"]
parent["selected"] = object()
assert copied["selected"] is retained
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("qualified", (False, True))
def test_copied_builtin_frame_preserves_qualified_native_decorator_only(qualified):
    decorator = "builtins.property" if qualified else "property"
    source = f"""import builtins
events = []
class Replacement:
    def __init__(self, *args): pass
    def __set_name__(self, owner, name): events.append(name)
__builtins__ = dict(vars(builtins), property=Replacement)
class Authority:
    @{decorator}
    def cached(self): pass
"""
    environment = execution(source)
    owner = environment.module.module.body[-1]
    operand = owner.body[0].decorator_list[0]
    result = environment.capture(operand)
    if qualified:
        assert (
            result.require_native_identity(NativeDeclaration(property)).declaration
            is property
        )
        environment.require_class_creation(owner)
    else:
        with pytest.raises(ValueError, match="identity remains open"):
            result.require_native_identity(NativeDeclaration(property))
        with pytest.raises(ValueError):
            environment.require_class_creation(owner)
    expected = "[]" if qualified else "['cached']"
    native = subprocess.run(
        [sys.executable, "-c", source + f"\nassert events == {expected}\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native.returncode == 0, native.stderr


def test_plain_class_keeps_module_builder_when_copied_builtins_override_it():
    source = """import builtins
class Replacement: pass
__builtins__ = dict(vars(builtins), __build_class__=Replacement)
class Authority:
    @builtins.property
    def cached(self): pass
"""
    environment = execution(source)
    # The containing plain module already captured its builder; its ordinary
    # class can still be created. A generated wrapper that captures the copied
    # frame must separately prove its builder before admission (covered by the
    # compiler/frame integration suite, not inferred from this plain control).
    owner = environment.module.module.body[-1]
    environment.require_class_creation(owner)
