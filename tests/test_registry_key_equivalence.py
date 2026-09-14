"""Registry uniqueness follows runtime mapping keys, not member spellings."""

import ast
from pathlib import Path
from types import ModuleType

import pytest

from registry_test_sources import keyed_registry_source
from nominal_refactor_advisor.analysis import analyze_detector_types
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.detectors import DetectorConfig, IssueDetector
from nominal_refactor_advisor.manual_registry import DirectManualRegistryComponent
from nominal_refactor_advisor.source_index import build_source_index


_DISTINCT_ENUM = """class Mode(Enum):
    ALPHA = auto()
    BETA = auto()"""

_KEY_CASES = (
    pytest.param(_DISTINCT_ENUM, False, 2, id="distinct-auto-positive"),
    pytest.param(
        """class Mode(Enum):
    ALPHA = auto()
    BETA = ALPHA""",
        True,
        1,
        id="direct-alias",
    ),
    pytest.param(
        """class Mode(Enum):
    ALPHA = 1
    BETA = True""",
        True,
        1,
        id="equal-native-values",
    ),
    pytest.param(
        """class Mode(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return 1

    ALPHA = auto()
    BETA = auto()""",
        True,
        1,
        id="custom-auto-collision",
    ),
    pytest.param(
        """class Mode(Enum):
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return 0

    ALPHA = auto()
    BETA = auto()""",
        False,
        1,
        id="distinct-objects-equal-mapping-keys",
    ),
)

_MANUAL_SOURCE = """from enum import Enum, auto

class Mode(Enum):
    ALPHA = auto()
    BETA = auto()

REGISTRY = {}

class AlphaHandler:
    pass

class BetaHandler:
    pass

REGISTRY[Mode.ALPHA] = AlphaHandler
REGISTRY[Mode.BETA] = BetaHandler
"""


def _source_with_enum(template: str, enum_source: str) -> str:
    assert template.count(_DISTINCT_ENUM) == 1
    return template.replace(_DISTINCT_ENUM, enum_source)


def _native_keys(source: str, aliases: bool, key_count: int) -> ModuleType:
    # Execute only this test's authored fixture, never analyzed repository code.
    runtime = ModuleType("registry_key_equivalence")
    exec(compile(source, "<registry-key-equivalence>", "exec"), runtime.__dict__)
    assert (runtime.Mode.ALPHA is runtime.Mode.BETA) is aliases
    assert len({runtime.Mode.ALPHA, runtime.Mode.BETA}) == key_count
    return runtime


def _parsed(source: str) -> ParsedModule:
    return ParsedModule(
        Path("/repo/registry_key_equivalence.py"),
        "registry_key_equivalence",
        False,
        ast.parse(source),
        source,
    )


@pytest.mark.parametrize("enum_source,aliases,key_count", _KEY_CASES)
def test_injective_finding_requires_distinct_mapping_keys(
    enum_source: str, aliases: bool, key_count: int
) -> None:
    source = _source_with_enum(keyed_registry_source(), enum_source)
    runtime = _native_keys(source, aliases, key_count)
    assert len(runtime.ModeRunner._registry) == key_count
    assert runtime.run_beta() == "beta"
    assert runtime.run_alpha() == ("alpha" if key_count == 2 else "beta")
    detectors = tuple(
        detector
        for detector in IssueDetector.registered_detector_types()
        if detector.effective_detector_id() == "injective_type_registry"
    )
    assert len(detectors) == 1
    findings = analyze_detector_types(
        [_parsed(source)], DetectorConfig(), detector_types=detectors
    )
    assert len(findings) == (1 if key_count == 2 else 0), (
        "Member spelling and distinct object identity do not prove unequal mapping keys"
    )


@pytest.mark.parametrize("enum_source,aliases,key_count", _KEY_CASES)
def test_manual_registry_component_requires_distinct_mapping_keys(
    enum_source: str, aliases: bool, key_count: int
) -> None:
    source = _source_with_enum(_MANUAL_SOURCE, enum_source)
    runtime = _native_keys(source, aliases, key_count)
    assert len(runtime.REGISTRY) == key_count
    assert runtime.REGISTRY[runtime.Mode.BETA] is runtime.BetaHandler
    module = _parsed(source)
    if key_count == 1:
        with pytest.raises(ValueError):
            DirectManualRegistryComponent.from_module_anchor(
                module.module, "AlphaHandler"
            )
        return

    component = DirectManualRegistryComponent.from_module_anchor(
        module.module, "AlphaHandler"
    )
    component.require_complete()
    assert len(component.entries) == 2
    assert component.class_names == ("AlphaHandler", "BetaHandler")


@pytest.mark.parametrize("enum_source,aliases,key_count", _KEY_CASES)
def test_manual_conversion_does_not_assume_member_spellings_are_unique(
    enum_source: str, aliases: bool, key_count: int
) -> None:
    source = _source_with_enum(_MANUAL_SOURCE, enum_source)
    runtime = _native_keys(source, aliases, key_count)
    assert len(runtime.REGISTRY) == key_count
    module = _parsed(source)
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([module], ()), {module.file_path: source}
    )
    operation = ConvertManualRegistryToAutoregisterOperation(
        target=SourceRewriteTarget(
            file_path=module.file_path, qualname="AlphaHandler"
        )
    )
    if key_count == 1:
        # The operation currently promises unique keys. Preserving dict overwrite
        # behavior by accident does not establish that precondition.
        with pytest.raises(ValueError):
            operation.source_edits_from_snapshot(snapshot)
        return

    result = RefactorRecipe("distinct-registry-keys").with_operation(
        operation
    ).simulate(snapshot)
    assert result.is_clean
    converted = _native_keys(
        result.simulation.rewritten_sources[module.file_path], aliases, key_count
    )
    assert len(converted.REGISTRY) == 2
    assert converted.REGISTRY[converted.Mode.ALPHA] is converted.AlphaHandler
    assert converted.REGISTRY[converted.Mode.BETA] is converted.BetaHandler
