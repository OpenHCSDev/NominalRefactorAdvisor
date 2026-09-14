"""Native creation probes distinguish stored arguments from inspected arguments."""

import ast
import builtins
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import typing

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import RepositoryModuleBindingProof
from nominal_refactor_advisor.class_namespace import ClassNamespaceExecutionEvidence
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    InitialNativeIsland,
)
from nominal_refactor_advisor.native_compilation import NativeCreationBackend
from nominal_refactor_advisor.native_subscription import NativeSubscriptionAuthority
from nominal_refactor_advisor.product_flow import CompactSubscription
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_operation_completion_premise import supplied_entry


def _verify_namespace(
    tmp_path: Path, source: str, *, safe: bool, injected: bool
) -> None:
    path = tmp_path / "probe.py"
    path.write_text(
        source + "\nprint('injected' in vars(Owner))\n", encoding="utf-8", newline=""
    )
    assert subprocess.check_output(
        [sys.executable, str(path)], text=True
    ).strip() == str(injected)
    modules = parse_python_modules(tmp_path)
    module = modules[0]
    owner = next(
        node
        for node in module.module.body
        if isinstance(node, ast.ClassDef) and node.name == "Owner"
    )
    evidence = ClassNamespaceExecutionEvidence.from_class(owner)
    bindings = RepositoryModuleBindingProof(modules)
    if safe:
        evidence.require_closed(bindings, module, owner)
    else:
        with pytest.raises(ValueError):
            evidence.require_closed(bindings, module, owner)


@pytest.mark.parametrize(
    "annotation,safe",
    (
        ("CV[str]", True),
        ("CV[Text]", True),
        ("CV[dict[str, tuple[int, ...]]]", True),
        ("CV['Poison']", True),
        ("CV[Poison]", False),
        ("CV[list[Poison]]", False),
    ),
)
def test_subscription_argument_hashing_uses_native_provenance(
    tmp_path: Path, annotation: str, safe: bool
) -> None:
    setup = (
        "hash_calls = []\n"
        "class Meta(type):\n"
        "    def __hash__(cls):\n"
        "        hash_calls.append(cls)\n"
        "        import sys\n"
        "        frame = sys._getframe(1)\n"
        "        while frame is not None:\n"
        "            if frame.f_code.co_name == 'Owner': frame.f_locals['injected'] = True\n"
        "            frame = frame.f_back\n"
        "        return 17\n"
        "class Poison(metaclass=Meta): pass\n"
    )
    source = (
        "from typing import ClassVar as CV\n"
        "from builtins import str as Text\n"
        f"class Owner:\n    field: {annotation} = 1\n"
    )
    eager = sys.version_info < (3, 14)
    # Run the complete authored control separately. Analysis starts after the
    # custom classes already exist, so their construction cannot mask hashing.
    output = subprocess.check_output(
        [sys.executable, "-c", setup + source + "\nprint('injected' in vars(Owner))"],
        text=True,
    )
    assert output.strip() == str(not safe and eager)
    prepared = ModuleType("subscription_setup")
    exec(setup, vars(prepared))  # Authored setup only, never the analyzed source.
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    (module,) = parse_python_modules(tmp_path)
    bindings = RepositoryModuleBindingProof((module,))
    observed = bindings.source_projection(module)
    initial = InitialNativeIsland((builtins, typing, prepared))
    entry = SourceModuleEntryPremise(
        observed,
        initial,
        {name: CapturedNativeObject(value) for name, value in vars(prepared).items()},
        initial.namespace_for_storage(vars(builtins)),
    )
    environment = SourceModuleExecution(entry)
    owner = module.module.body[-1]
    operation = next(
        (
            operation
            for operation in observed.operations
            if operation.node is owner.body[0].annotation
            and isinstance(operation.event, CompactSubscription)
        ),
        None,
    )
    if not eager:
        assert operation is None
        environment.require_class_creation(owner)
        assert not prepared.hash_calls
        return
    assert operation is not None
    environment = SourceModuleExecution(supplied_entry(environment, (operation,)))
    authority = NativeSubscriptionAuthority.for_subscription(
        environment,
        environment.context_for_owner(operation.owner),
        operation.event,
    )
    if safe:
        NativeCreationBackend.current().require_classvar_binding(
            authority.inspected_argument
        )
        environment.require_class_creation(owner)
    else:
        with pytest.raises(ValueError):
            NativeCreationBackend.current().require_classvar_binding(
                authority.inspected_argument
            )
        with pytest.raises(ValueError):
            environment.require_class_creation(owner)
    assert not prepared.hash_calls


@pytest.mark.parametrize("constructor", ("staticmethod", "classmethod", "property"))
@pytest.mark.parametrize("argument,safe", (("payload", False), ("lambda: None", True)))
def test_native_descriptor_arguments_can_execute_metadata_hooks(
    tmp_path: Path, constructor: str, argument: str, safe: bool
) -> None:
    source = (
        "class Metadata:\n"
        "    def __getattribute__(self, name):\n"
        "        import sys\n"
        "        frame = sys._getframe(1)\n"
        "        while frame is not None:\n"
        "            if frame.f_code.co_name == 'Owner': frame.f_locals['injected'] = True\n"
        "            frame = frame.f_back\n"
        "        return object.__getattribute__(self, name)\n"
        "payload = Metadata()\n"
        f"class Owner:\n    field = {constructor}({argument})\n"
    )
    _verify_namespace(tmp_path, source, safe=safe, injected=not safe)


def test_computed_subscription_reference_remains_explicitly_unproved(
    tmp_path: Path,
) -> None:
    # Native execution succeeds, but the computed origin has no captured nominal
    # reference. Report an unproved effect rather than leaking a lookup KeyError.
    _verify_namespace(
        tmp_path,
        "class Owner:\n    field = property((list if True else dict)[str])\n",
        safe=False,
        injected=False,
    )
