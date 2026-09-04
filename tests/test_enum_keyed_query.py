from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.analysis import analyze_compact_roots_with_cache
from nominal_refactor_advisor.analysis_cache import AnalysisCacheStatus
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodBackend,
    CodemodSourceSnapshot,
    DescendEnumKeyedDerivedMapFacadeOperation,
    FindingRecipeSynthesisStatus,
    RefactorRecipeOperation,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._runtime import (
    EnumKeyedDerivedMapFacadeDetector,
)
from nominal_refactor_advisor.enum_keyed_query import (
    EnumKeyedDerivedMapFacadeComponentBuilder,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.models import SourceLocation


def _enum_keyed_facade_source() -> str:
    return """\
from __future__ import annotations

from enum import StrEnum


class Handler:
    pass


class AlternateHandler(Handler):
    pass


class Mode(StrEnum):
    FAST = "fast"
    SAFE = "safe"


class HandlerCatalog:
    @classmethod
    def handler_types_by_mode(cls) -> dict[Mode, type[Handler]]:
        return {
            Mode.FAST: Handler,
            Mode.SAFE: AlternateHandler,
        }

    @classmethod
    def modes_for_handler_name(cls, name: str) -> tuple[Mode, ...]:
        return tuple(
            mode
            for mode, handler_type in cls.handler_types_by_mode().items()
            if handler_type.__name__ == name
        )


def handler_name(mode: Mode) -> str:
    return HandlerCatalog.handler_types_by_mode()[mode].__name__


def modes_for_name(name: str) -> tuple[Mode, ...]:
    return HandlerCatalog.modes_for_handler_name(name)
"""


def _write_package_module(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module_path = package / "mod.py"
    module_path.write_text(source)
    return module_path


def _with_second_facade(source: str, *, value_name: str = "fallback_type") -> str:
    methods = f"""\

    @classmethod
    def fallback_types_by_mode(cls) -> dict[Mode, type[Handler]]:
        return {{
            Mode.FAST: AlternateHandler,
            Mode.SAFE: Handler,
        }}

    @classmethod
    def modes_for_fallback_name(cls, name: str) -> tuple[Mode, ...]:
        return tuple(
            mode
            for mode, {value_name} in cls.fallback_types_by_mode().items()
            if {value_name}.__name__ == name
        )
"""
    return source.replace(
        "\n\ndef handler_name",
        f"{methods}\n\ndef handler_name",
    ) + (
        "\n\ndef fallback_name(mode: Mode) -> str:\n"
        "    return HandlerCatalog.fallback_types_by_mode()[mode].__name__\n"
    )


def _findings(tmp_path: Path):
    modules = parse_python_modules(tmp_path)
    findings = tuple(
        EnumKeyedDerivedMapFacadeDetector().detect(
            list(modules),
            DetectorConfig(),
        )
    )
    return modules, findings


def test_enum_keyed_facade_builder_recovers_nominal_owner(tmp_path: Path) -> None:
    module_path = _write_package_module(tmp_path, _enum_keyed_facade_source())
    modules = parse_python_modules(tmp_path)

    components = EnumKeyedDerivedMapFacadeComponentBuilder.collect_modules(modules)

    assert len(components) == 1
    component = components[0]
    assert component.enum_symbol == "pkg.mod.Mode"
    assert component.map_owner_symbol == "pkg.mod.HandlerCatalog"
    assert component.map_method_name == "handler_types_by_mode"
    assert component.reverse_method_name == "modes_for_handler_name"
    assert component.property_name == "handler_type"
    assert component.authority_evidence == SourceLocation(
        module_path.as_posix(),
        14,
        "pkg.mod.Mode",
    )
    assert len(component.consumers) == 1


def test_enum_keyed_facade_compact_cache_preserves_exact_finding(
    tmp_path: Path,
) -> None:
    module_path = _write_package_module(tmp_path, _enum_keyed_facade_source())
    cache_dir = tmp_path / "cache"
    analysis_cache_dir = tmp_path / "analysis"

    cold = analyze_compact_roots_with_cache(
        (module_path.parent,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=(EnumKeyedDerivedMapFacadeDetector,),
    )
    warm = analyze_compact_roots_with_cache(
        (module_path.parent,),
        cache_dir=cache_dir,
        analysis_cache_dir=analysis_cache_dir,
        detector_types=(EnumKeyedDerivedMapFacadeDetector,),
    )

    assert len(cold.findings) == 1
    assert warm.cache_status is AnalysisCacheStatus.HIT
    assert warm.findings == cold.findings


def test_enum_keyed_facade_recipe_moves_queries_and_preserves_behavior(
    tmp_path: Path,
) -> None:
    source = _enum_keyed_facade_source()
    module_path = _write_package_module(tmp_path, source)
    modules, findings = _findings(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("enum_keyed_derived_map_facade",),
    )

    assert len(findings) == 1
    assert plan.records[0].status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
    operation = plan.document.recipes[0].operations[0]
    assert isinstance(operation, DescendEnumKeyedDerivedMapFacadeOperation)
    assert set(json_report_object(operation)) == {"operation", "target_id", "rationale"}
    assert isinstance(
        RefactorRecipeOperation.from_dict(json_report_object(operation)),
        DescendEnumKeyedDerivedMapFacadeOperation,
    )
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    assert "def handler_type(self) -> type[Handler]:" in rewritten
    assert "return HandlerCatalog.handler_types_by_mode()[self]" in rewritten
    assert "Mode.modes_for_handler_name(name)" in rewritten
    assert "mode.handler_type.__name__" in rewritten
    catalog_source = rewritten.split("class HandlerCatalog:", maxsplit=1)[1]
    assert "def modes_for_handler_name" not in catalog_source

    original_namespace: dict[str, object] = {}
    rewritten_namespace: dict[str, object] = {}
    exec(compile(source, module_path.as_posix(), "exec"), original_namespace)
    exec(compile(rewritten, module_path.as_posix(), "exec"), rewritten_namespace)
    for mode_name in ("FAST", "SAFE"):
        original_mode = original_namespace["Mode"][mode_name]
        rewritten_mode = rewritten_namespace["Mode"][mode_name]
        assert original_namespace["handler_name"](original_mode) == rewritten_namespace[
            "handler_name"
        ](rewritten_mode)
    for handler_name in ("Handler", "AlternateHandler", "Missing"):
        assert tuple(
            mode.value for mode in original_namespace["modes_for_name"](handler_name)
        ) == tuple(
            mode.value for mode in rewritten_namespace["modes_for_name"](handler_name)
        )

    rewritten_modules = tuple(
        module.with_source(rewritten) if module.path == module_path else module
        for module in modules
    )
    assert not EnumKeyedDerivedMapFacadeComponentBuilder.collect_modules(
        rewritten_modules
    )


@pytest.mark.parametrize(
    "source_mutation",
    (
        "dict = object\n\n",
        "classmethod = lambda function: function\n\n",
        "property = lambda function: function\n\n",
        "StrEnum = type('StrEnum', (), {})\n\n",
    ),
)
def test_enum_keyed_facade_requires_unshadowed_python_declarations(
    tmp_path: Path,
    source_mutation: str,
) -> None:
    source = _enum_keyed_facade_source().replace(
        "class Handler:\n",
        f"{source_mutation}class Handler:\n",
    )
    _write_package_module(tmp_path, source)

    _modules, findings = _findings(tmp_path)

    assert not findings


def test_enum_keyed_facade_does_not_replace_inherited_enum_members(
    tmp_path: Path,
) -> None:
    source = (
        _enum_keyed_facade_source()
        .replace(
            "mode, handler_type",
            "mode, value",
        )
        .replace(
            "handler_type.__name__",
            "value.__name__",
        )
    )
    _write_package_module(tmp_path, source)

    _modules, findings = _findings(tmp_path)

    assert not findings


def test_enum_keyed_facade_requires_receiver_independent_reverse_query(
    tmp_path: Path,
) -> None:
    source = _enum_keyed_facade_source().replace(
        "if handler_type.__name__ == name",
        "if handler_type.__name__ == name and cls is not None",
    )
    _write_package_module(tmp_path, source)

    _modules, findings = _findings(tmp_path)

    assert not findings


def test_enum_keyed_facade_star_import_uses_declared_export_contract(
    tmp_path: Path,
) -> None:
    module_path = _write_package_module(
        tmp_path,
        _enum_keyed_facade_source().replace(
            "from enum import StrEnum\n",
            "from enum import StrEnum\nfrom .support import *\n",
        ),
    )
    support_path = module_path.with_name("support.py")
    support_path.write_text('__all__ = ("HELPER",)\n\nHELPER = object()\n')

    _modules, findings = _findings(tmp_path)

    assert len(findings) == 1

    support_path.write_text('__all__ = ("dict",)\n')
    _modules, findings = _findings(tmp_path)
    assert not findings


def test_enum_keyed_facade_derives_aliased_binding_exclusion(
    tmp_path: Path,
) -> None:
    module_path = _write_package_module(
        tmp_path,
        _enum_keyed_facade_source()
        .replace(
            "from enum import StrEnum\n",
            "from enum import StrEnum as EnumBase\nfrom .support import *\n",
        )
        .replace("class Mode(StrEnum):", "class Mode(EnumBase):"),
    )
    support_path = module_path.with_name("support.py")
    support_path.write_text('__all__ = ("StrEnum",)\n')

    _modules, findings = _findings(tmp_path)

    assert len(findings) == 1

    support_path.write_text('__all__ = ("EnumBase",)\n')
    _modules, findings = _findings(tmp_path)
    assert not findings


def test_enum_keyed_facade_recipe_rejects_noncall_method_reference(
    tmp_path: Path,
) -> None:
    source = (
        _enum_keyed_facade_source()
        + "\nMODE_QUERY = HandlerCatalog.modes_for_handler_name\n"
    )
    _write_package_module(tmp_path, source)
    modules, findings = _findings(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("enum_keyed_derived_map_facade",),
    )

    assert len(findings) == 1
    assert (
        plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "cannot be rewritten nominally" in plan.records[0].reason


def test_enum_keyed_facade_recipe_rejects_shadowed_enum_replacement(
    tmp_path: Path,
) -> None:
    source = (
        _enum_keyed_facade_source() + "\n\ndef shadowed_query(Mode, name: str):\n"
        "    return HandlerCatalog.modes_for_handler_name(name)\n"
    )
    _write_package_module(tmp_path, source)
    modules, findings = _findings(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("enum_keyed_derived_map_facade",),
    )

    assert len(findings) == 1
    assert (
        plan.records[0].status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
    )
    assert "cannot be rewritten nominally" in plan.records[0].reason


def test_enum_keyed_facade_rejects_non_value_map_slice(tmp_path: Path) -> None:
    source = _enum_keyed_facade_source().replace(
        "HandlerCatalog.handler_types_by_mode()[mode].__name__",
        "HandlerCatalog.handler_types_by_mode()[mode:].__name__",
    )
    _write_package_module(tmp_path, source)

    _modules, findings = _findings(tmp_path)

    assert not findings


def test_enum_keyed_facade_recipes_batch_independent_owner_relations(
    tmp_path: Path,
) -> None:
    module_path = _write_package_module(
        tmp_path,
        _with_second_facade(_enum_keyed_facade_source()),
    )
    modules, findings = _findings(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("enum_keyed_derived_map_facade",),
    )

    assert len(findings) == 2
    assert all(
        record.status is FindingRecipeSynthesisStatus.EXECUTABLE_CANDIDATE
        for record in plan.records
    )
    simulation = plan.simulate(snapshot, backend=CodemodBackend.AST_SPAN)
    assert simulation.is_clean is True
    rewritten = simulation.simulation.rewritten_sources[module_path.as_posix()]
    assert rewritten.count("def handler_type(") == 1
    assert rewritten.count("def fallback_type(") == 1
    assert "mode.handler_type.__name__" in rewritten
    assert "mode.fallback_type.__name__" in rewritten


def test_enum_keyed_facade_batch_rejects_competing_member_derivations(
    tmp_path: Path,
) -> None:
    _write_package_module(
        tmp_path,
        _with_second_facade(
            _enum_keyed_facade_source(),
            value_name="handler_type",
        ),
    )
    modules, findings = _findings(tmp_path)
    snapshot = CodemodSourceSnapshot.from_modules(modules, findings)

    plan = snapshot.plan_from_findings(
        findings,
        detector_ids=("enum_keyed_derived_map_facade",),
    )

    assert len(findings) == 2
    assert all(
        record.status is FindingRecipeSynthesisStatus.CONFLICTING_TRAJECTORY_BRANCHES
        for record in plan.records
    )
