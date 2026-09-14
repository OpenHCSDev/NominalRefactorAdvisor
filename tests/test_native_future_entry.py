"""Native language syntax dependencies still execute through import admission."""

import __future__

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import (
    ModuleAnnotationEvaluationMode,
    ParsedModule,
)
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    OpenCapturedReference,
)
from nominal_refactor_advisor.native_declarations import NativeLanguageFeature
from nominal_refactor_advisor.source_execution import SourceModuleExecution


def _execution(source: str, module_name: str = "future_entry") -> SourceModuleExecution:
    return SourceModuleExecution.from_module(
        ParsedModule(
            path=Path("future_entry.py"),
            module_name=module_name,
            is_package_init=False,
            module=ast.parse(source),
            source=source,
        )
    )


def _require_class(source: str) -> SourceModuleExecution:
    execution = _execution(source)
    node = execution.module.module.body[-1]
    assert isinstance(node, ast.ClassDef)
    execution.require_class_creation(node)
    return execution


@pytest.mark.parametrize("name", tuple(__future__.all_feature_names))
def test_feature_name_and_source_derive_from_native_export_identity(name: str) -> None:
    declaration = NativeLanguageFeature(vars(__future__)[name])
    assert declaration.export_name == name
    assert declaration.qualified_name == f"__future__.{name}"
    assert declaration.import_source == f"from __future__ import {name}\n"
    assert declaration.imported_by(
        ast.parse(f"from __future__ import {name} as selected")
    )
    assert not declaration.imported_by(ast.parse(f"from .__future__ import {name}"))


@pytest.mark.parametrize("feature_name", ("annotations", "division", "absolute_import"))
@pytest.mark.parametrize("alias", (False, True))
def test_standard_source_entry_proves_runtime_future_imports(
    feature_name: str, alias: bool
) -> None:
    binding = "feature" if alias else feature_name
    source = f"from __future__ import {feature_name}" + (
        " as feature\n" if alias else "\n"
    )
    source += "import builtins\nclass Owner:\n    @builtins.property\n    def value(self): return 7\n"
    execution = _require_class(source)
    program = (
        source
        + f"print({binding} is __import__('__future__').{feature_name}, Owner().value)\n"
    )
    assert (
        subprocess.check_output([sys.executable, "-c", program], text=True).strip()
        == "True 7"
    )
    observed = execution.initial.module(NativeLanguageFeature.module.__name__)
    assert isinstance(observed, CapturedNativeObject)
    assert observed.value is __future__


def test_local_importer_spelling_does_not_shadow_frame_builtin_importer() -> None:
    _require_class(
        "from __future__ import annotations\ndef __import__(*args): raise AssertionError\nimport __future__ as features\nclass Owner:\n    pass\n"
    )


def test_rebound_builtin_importer_does_not_admit_later_feature_import() -> None:
    with pytest.raises(ValueError):
        _require_class(
            "from __future__ import annotations\nimport builtins\nclass Replacement:\n    pass\nbuiltins.__import__ = Replacement\nimport __future__ as features\nclass Owner:\n    pass\n"
        )


def test_rebound_native_feature_member_is_not_a_native_import_result() -> None:
    with pytest.raises(ValueError):
        _require_class(
            "import __future__ as features\nclass Replacement:\n    pass\n"
            "features.annotations = Replacement\n"
            "import __future__ as changed_features\n"
            "selected = changed_features.annotations\nclass Owner:\n    pass\n"
        )


@pytest.mark.parametrize("module_name", ("builtins", "typing", "__future__"))
def test_source_entry_cannot_replace_admitted_native_module(module_name: str) -> None:
    with pytest.raises(ValueError, match="replace an admitted native module"):
        _execution("class Owner:\n    pass\n", module_name=module_name)


def test_source_entry_cannot_replace_observed_native_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "nra_native_future_alias"
    monkeypatch.setitem(sys.modules, alias, __future__)
    with pytest.raises(ValueError, match="replace an admitted native module"):
        _execution("class Owner:\n    pass\n", module_name=alias)


def test_native_source_loader_replaces_the_module_import_association(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "__future__.py"
    source_path.write_text("import __future__ as observed\n", encoding="utf-8")
    program = f"""import __future__
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
native = __future__
loader = SourceFileLoader(native.__name__, {str(source_path)!r})
spec = spec_from_file_location(native.__name__, loader.path, loader=loader)
module = module_from_spec(spec)
sys.modules[spec.name] = module
loader.exec_module(module)
print(module.observed is module, module.observed is native)
"""
    assert (
        subprocess.check_output([sys.executable, "-c", program], text=True).strip()
        == "True False"
    )


def test_native_feature_module_alias_is_observed_not_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "nra_native_future_alias"
    monkeypatch.setitem(sys.modules, alias, __future__)
    execution = _require_class(f"import {alias} as features\nclass Owner:\n    pass\n")
    observed = execution.initial.module(alias)
    assert isinstance(observed, CapturedNativeObject)
    assert observed.value is __future__


def test_counterfeit_module_name_does_not_supply_native_future_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType(__future__.__name__)
    fake.annotations = __future__.annotations
    monkeypatch.setitem(sys.modules, __future__.__name__, fake)
    execution = _execution(
        "from __future__ import annotations\nclass Owner:\n    pass\n"
    )
    assert isinstance(
        execution.initial.module(__future__.__name__), OpenCapturedReference
    )
    with pytest.raises(ValueError):
        execution.require_class_creation(execution.module.module.body[-1])


def test_stringized_mode_uses_native_declaration_for_syntax() -> None:
    mode = ModuleAnnotationEvaluationMode.STRINGIZED
    (feature,) = mode.native_features
    assert feature.declaration is __future__.annotations
    assert mode.new_module_prelude == feature.import_source
    assert (
        ModuleAnnotationEvaluationMode.from_module(
            ast.parse("from __future__ import annotations as selected")
        )
        is mode
    )
    assert (
        ModuleAnnotationEvaluationMode.from_module(
            ast.parse("from .__future__ import annotations")
        )
        is ModuleAnnotationEvaluationMode.runtime_default()
    )


def test_feature_directive_placement_still_uses_native_compiler_validation() -> None:
    execution = _execution(
        "value = 1\nfrom __future__ import annotations\nclass Owner:\n    pass\n"
    )
    with pytest.raises(SyntaxError):
        execution.module.native_compilation.compile()
