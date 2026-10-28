"""A source context retains shared facts through its final executable consumer."""

import ast
import weakref

import pytest

from nominal_refactor_advisor import analysis
from nominal_refactor_advisor.analysis_cache import AnalysisCacheIdentity, AnalysisFindingCache
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.detectors import ClosedParameterConveyorDetector
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._heuristic_leads import LongConditionFactoringLeadDetector


@pytest.mark.parametrize("fail_global", (False, True))
def test_original_source_facts_reach_late_consumer_and_release_after_result(
    tmp_path, monkeypatch, fail_global,
):
    modules = [
        ParsedModule(
            path=tmp_path / f"module{index}.py", module_name=f"module{index}",
            is_package_init=False, source="def plain(): pass\n",
            module=ast.parse("def plain(): pass\n"),
        )
        for index in range(2)
    ]
    original_local = LongConditionFactoringLeadDetector._findings_for_module
    original_global = ClosedParameterConveyorDetector._candidate_items
    original_facts = {}
    consumers = []

    def observe_local(self, module, config):
        original_facts[module.file_path] = weakref.ref(module.syntax_index)
        return original_local(self, module, config)

    def observe_global(self, modules, config):
        for module in modules:
            assert original_facts[module.file_path]() is module.syntax_index
        consumers.append("global")
        if fail_global:
            raise RuntimeError("late source consumer failed")
        return original_global(self, modules, config)

    monkeypatch.setattr(LongConditionFactoringLeadDetector, "_findings_for_module", observe_local)
    monkeypatch.setattr(ClosedParameterConveyorDetector, "_candidate_items", observe_global)
    monkeypatch.setattr(
        analysis, "default_detector_types_for_analysis",
        lambda: (LongConditionFactoringLeadDetector, ClosedParameterConveyorDetector),
    )
    config = DetectorConfig()
    resolver = analysis.IncrementalAnalysisCacheResolver(
        cache_identity=AnalysisCacheIdentity.from_modules((tmp_path,), tuple(modules), config),
        modules=modules, config=config, analysis_cache=AnalysisFindingCache(None),
        analysis_workers=1,
        semantic_descent_source=analysis.SemanticDescentGraphAnalysisSource(),
    )
    if fail_global:
        with pytest.raises(RuntimeError, match="late source consumer failed"):
            resolver.result()
    else:
        assert resolver.result().findings == []
    assert consumers == ["global"]
