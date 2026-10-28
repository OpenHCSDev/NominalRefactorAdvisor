"""Call edits distinguish declaration identity from receiver binding."""

import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule, parse_python_modules
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ReplaceDeclaredCallArgumentsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.product_flow import (
    CompactCallArguments,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.product_flow_authority import CompactProductFlowRepository
from nominal_refactor_advisor.value_expression import CompactValueExpression
from nominal_refactor_advisor.call_binding import CompactCallBindingViolation


def _run_call(source: str, scope: str = "Owner.run"):
    module = ParsedModule(
        path=Path("probe.py"),
        module_name="probe",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )
    repository = CompactProductFlowRepository(
        product_projections=(compact_product_flow_projection(module),),
        class_projections=CompactModuleClassProjectionFamily.collect_modules((module,)),
    )
    return next(
        result.resolved_call
        for result in repository.function_call_resolutions
        if result.context.owner_symbol == f"probe.{scope}"
        and result.call.target.terminal_name == "consume"
    )


@pytest.mark.parametrize(
    "decorator,parameters",
    (
        ("", "self, value"),
        ("@classmethod\n    ", "cls, value"),
        ("@staticmethod\n    ", "value"),
    ),
)
@pytest.mark.parametrize(
    "caller_decorator,receiver", (("", "self"), ("@classmethod\n    ", "cls"))
)
def test_resolved_call_signature_agrees_with_python_descriptor_lookup(
    decorator: str, parameters: str, caller_decorator: str, receiver: str
) -> None:
    source = (
        "class Owner:\n"
        + f"    {decorator}def consume({parameters}): return value + 1\n"
    )
    namespace = {}
    exec(source, namespace)
    owner = namespace["Owner"]
    runtime_receiver = owner() if receiver == "self" else owner
    runtime_target = runtime_receiver.consume
    native_signature = inspect.signature(runtime_target)
    arguments = (
        "instance, value=3" if "self" in native_signature.parameters else "value=3"
    )
    source += f"    {caller_decorator}def run({receiver}, instance): return {receiver}.consume({arguments})\n"
    resolved = _run_call(source)
    assert resolved is not None
    assert resolved.binding.is_exact
    assert tuple(
        parameter.name for parameter in resolved.call_signature.parameters
    ) == tuple(native_signature.parameters)
    assert resolved.target_resolution is resolved.resolved_target
    replacement = CompactCallArguments.from_call(
        ast.parse("consume(value=3)", mode="eval").body, CompactValueExpression.project
    )
    try:
        native_signature.bind(value=3)
    except TypeError:
        assert not resolved.target_resolution.bind_arguments(replacement).is_exact
    else:
        assert resolved.target_resolution.bind_arguments(replacement).is_exact


def test_raw_classmethod_descriptor_retains_identity_but_is_not_callable() -> None:
    source = (
        "class Owner:\n"
        "    @classmethod\n"
        "    def consume(cls, value): return value\n"
        "    result = consume(3)\n"
    )
    resolved = _run_call(source, "Owner")
    assert resolved is not None
    assert resolved.callee.identity.symbol == "probe.Owner.consume"
    assert (
        resolved.binding.violation
        is CompactCallBindingViolation.INVALID_DESCRIPTOR_ACCESS
    )


def test_raw_function_in_class_body_has_no_implicit_receiver() -> None:
    source = (
        "class Owner:\n    def consume(value): return value\n    result = consume(3)\n"
    )
    resolved = _run_call(source, "Owner")
    assert resolved is not None
    assert resolved.binding.is_exact
    namespace = {}
    exec(source, namespace)
    assert namespace["Owner"].result == 3


@pytest.mark.parametrize("arguments", ("value=3", "other, value=3"))
def test_mixed_access_cannot_share_an_incompatible_argument_edit(
    tmp_path: Path, arguments: str
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        "    def consume(self, value): return value\n"
        "    def run(self, other): return self.consume(3), Owner.consume(other, 3)\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner.run"
                ),
                callee=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner.consume"
                ),
                arguments_source=arguments,
            ),
        )
    )
    with pytest.raises(ValueError, match="do not bind"):
        plan.simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize(
    "arguments,accepted", (("owner, value=3", True), ("value=3", False))
)
def test_cli_validates_unbound_class_access_with_explicit_receiver(
    tmp_path: Path, arguments: str, accepted: bool
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "class Owner:\n"
        "    def consume(self, value): return value + 1\n"
        "def run(owner): return Owner.consume(owner, 3)\n"
        "print(run(Owner()))\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    plan = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallArgumentsOperation(
                target=SourceRewriteTarget(file_path=path.as_posix(), qualname="run"),
                callee=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname="Owner.consume"
                ),
                arguments_source=arguments,
            ),
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(plan)),
        text=True,
        capture_output=True,
    )
    if accepted:
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["applied"]
        assert subprocess.check_output([sys.executable, str(path)], text=True) == before
    else:
        assert result.returncode != 0
        assert path.read_bytes() == source.encode()
