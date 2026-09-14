"""Registration conventions differ from source names and import permission."""

import ast
import builtins
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceViolation,
    InitialNativeIsland,
    NativeNamespace,
    NativeTypePremise,
    OpenCapturedReference,
)
from nominal_refactor_advisor.product_flow import source_product_flow_projection
from nominal_refactor_advisor.source_entry import (
    DirectScriptEntryPremise,
    ImportedSourceModuleEntryPremise,
    SourceModuleEntryPremise,
)
from nominal_refactor_advisor.source_execution import SourceModuleExecution


@pytest.fixture(scope="module")
def direct_file_evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("direct_entry")
    path = root / "nominal_refactor_advisor" / "codemod.py"
    path.parent.mkdir()
    path.write_text(
        "initial_keys = tuple(globals())\n"
        "import sys\n"
        "import nominal_refactor_advisor.codemod as installed\n"
        "print(repr((initial_keys, __name__,\n"
        "    sys.modules[__name__].__dict__ is globals(),\n"
        "    installed is sys.modules[__name__],\n"
        "    installed.__file__ == __file__,\n"
        "    (sys.implementation.name, sys.version))))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return ast.literal_eval(result.stdout)


def source(text="class Plain: pass\n", name="ast"):
    parsed = ParsedModule(Path("entry_probe.py"), name, False, ast.parse(text), text)
    return source_product_flow_projection(parsed)


def facts(initial, evidence):
    # This is an explicit complete-facts premise for the observed convention,
    # not a universal key roster or a fabricated runtime module dictionary.
    values = {
        key: OpenCapturedReference(CapturedReferenceViolation.UNPROVED_BINDING)
        for key in evidence[0]
    }
    values["__name__"] = NativeTypePremise(type(evidence[1]))
    namespace = initial.namespace_for_storage(vars(builtins))
    values[SourceModuleEntryPremise.bootstrap_binding_name()] = CapturedNativeObject(
        namespace.storage
    )
    return values


def test_native_file_execution_is_main_not_its_qualified_import(direct_file_evidence):
    _, runtime_name, registered_globals, same_module, same_file, interpreter = (
        direct_file_evidence
    )
    assert runtime_name == DirectScriptEntryPremise.registration_name
    assert registered_globals
    assert not same_module
    assert not same_file
    assert interpreter == source().module.native_compilation.identity.interpreter


def test_script_can_preserve_same_named_cached_import(direct_file_evidence):
    projected = source("import ast\nclass Plain: pass\n")
    initial = InitialNativeIsland((builtins, ast))
    namespace = initial.namespace_for_storage(vars(builtins))
    entry = DirectScriptEntryPremise(
        projected, initial, facts(initial, direct_file_evidence), namespace
    )
    assert entry.registration_name == "__main__"
    assert entry.source.module.module_name == "ast"
    assert initial.modules_by_name["ast"] is ast
    execution = SourceModuleExecution(entry)
    execution.require_import(projected.module.module.body[0])
    execution.require_class_creation(projected.module.module.body[-1])
    with pytest.raises(ValueError, match="replace an admitted native module"):
        ImportedSourceModuleEntryPremise.from_standard_source_loader(
            projected, initial, namespace
        )


def test_script_registration_rejects_admitted_main_module(direct_file_evidence):
    initial = InitialNativeIsland((builtins, sys.modules["__main__"]))
    with pytest.raises(ValueError, match="replace an admitted native module"):
        DirectScriptEntryPremise(
            source(),
            initial,
            facts(initial, direct_file_evidence),
            initial.namespace_for_storage(vars(builtins)),
        )


def test_import_registration_does_not_confuse_unrelated_main(direct_file_evidence):
    projected = source(name="fresh_registered_module")
    initial = InitialNativeIsland((builtins, sys.modules["__main__"]))
    entry = ImportedSourceModuleEntryPremise.from_standard_source_loader(
        projected, initial, initial.namespace_for_storage(vars(builtins))
    )
    assert entry.registration_name == projected.module.module_name
    assert entry.registration_name != DirectScriptEntryPremise.registration_name


def test_import_direct_constructor_also_enforces_shared_collision_invariant(
    direct_file_evidence,
):
    initial = InitialNativeIsland((builtins, ast))
    with pytest.raises(ValueError, match="replace an admitted native module"):
        ImportedSourceModuleEntryPremise(
            source(),
            initial,
            facts(initial, direct_file_evidence),
            initial.namespace_for_storage(vars(builtins)),
        )


def test_frame_validation_precedes_registration_claim(direct_file_evidence):
    initial = InitialNativeIsland((builtins, sys.modules["__main__"]))
    foreign_namespace = NativeNamespace(vars(builtins))
    with pytest.raises(ValueError, match="belongs to a different admission"):
        DirectScriptEntryPremise(
            source(), initial, facts(initial, direct_file_evidence), foreign_namespace
        )


def test_script_convention_does_not_admit_cold_imports(direct_file_evidence):
    projected = source("import ast\n", name="fresh_script")
    initial = InitialNativeIsland((builtins,))
    entry = DirectScriptEntryPremise(
        projected,
        initial,
        facts(initial, direct_file_evidence),
        initial.namespace_for_storage(vars(builtins)),
    )
    with pytest.raises(ValueError, match="unadmitted_native_import"):
        SourceModuleExecution(entry).require_import(projected.module.module.body[0])


def test_script_constructor_never_executes_source_or_copies_registration_name(
    direct_file_evidence,
):
    projected = source("raise AssertionError('target must never run')\n")
    initial = InitialNativeIsland((builtins,))
    entry = DirectScriptEntryPremise(
        projected,
        initial,
        facts(initial, direct_file_evidence),
        initial.namespace_for_storage(vars(builtins)),
    )
    assert entry.source is projected
    assert "registration_name" not in {field.name for field in fields(entry)}
    assert "registration_name" not in entry.__dict__


def test_default_factory_stays_an_imported_entry():
    projected = source(name="fresh_imported_module")
    execution = SourceModuleExecution.from_module(projected.module)
    assert isinstance(execution.entry, ImportedSourceModuleEntryPremise)
    assert not isinstance(execution.entry, DirectScriptEntryPremise)
    assert not hasattr(SourceModuleEntryPremise, "from_standard_source_loader")
