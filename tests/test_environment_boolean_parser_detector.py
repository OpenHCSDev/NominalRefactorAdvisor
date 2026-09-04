"""Focused calibration for environment-boolean authority drift detection."""

from __future__ import annotations

import json
from pathlib import Path

from nominal_refactor_advisor.analysis import (
    accumulate_compact_global_projections_for_roots,
)
from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.calibration import run_calibration_manifest
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    FindingRecipeSynthesisStatus,
)
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    EnvironmentBooleanAuthorityDriftDetector,
)
from nominal_refactor_advisor.json_reports import json_report_object
from nominal_refactor_advisor.models import (
    EnvironmentReadKind,
    FixedKeyEnvironmentAuthorityWrapperMetrics,
    LocalEnvironmentBooleanParserMetrics,
)

_CENTRAL_AUTHORITY_SOURCE = """
import os

DECISIONS = {"1": True, "0": False}


def declared_environment_flag_decision(
    name: str,
    *,
    absent_decision: bool,
) -> bool:
    value = os.environ.get(name)
    if value is None:
        return absent_decision
    normalized = value.strip().lower()
    if normalized not in DECISIONS:
        raise ValueError(normalized)
    return DECISIONS[normalized]


class DeclaredEnvironmentFlagAuthority:
    @staticmethod
    def enabled(name: str) -> bool:
        return declared_environment_flag_decision(
            name,
            absent_decision=False,
        )
"""


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _focused_findings(root: Path):
    modules = parse_python_modules(root, use_parse_cache=False)
    return EnvironmentBooleanAuthorityDriftDetector().detect(
        modules,
        DetectorConfig(),
    )


def test_detects_environment_token_parsers_and_declared_authority_wrapper(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "pkg/central.py", _CENTRAL_AUTHORITY_SOURCE)
    _write_module(
        tmp_path,
        "pkg/local.py",
        """
import os

DISABLED_VALUES = ("0", "false", "no", "off")


class RuntimeProfileEnvironmentAuthority:
    @staticmethod
    def enabled() -> bool:
        value = os.environ.get("RUNTIME_PROFILE")
        if value is None:
            value = ""
        return value.lower() not in DISABLED_VALUES
""",
    )
    _write_module(
        tmp_path,
        "pkg/direct.py",
        """
from os import environ, getenv


def trace_enabled() -> bool:
    return getenv("TRACE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def audit_enabled() -> bool:
    return environ["AUDIT"].casefold() not in frozenset(
        ("0", "false", "no", "off")
    )


def verbose_enabled() -> bool:
    value = getenv(key="VERBOSE")
    return (value or "").strip().lower() not in ("0", "false", "no", "off")


def optional_enabled() -> bool:
    value = getenv("OPTIONAL")
    return value not in ("0", "false", "no", "off")
""",
    )
    _write_module(
        tmp_path,
        "pkg/wrapper.py",
        """
from .central import DeclaredEnvironmentFlagAuthority


class FeatureEnvironmentAuthority:
    FEATURE_ENV = "FEATURE_FLAG"

    @staticmethod
    def enabled() -> bool:
        return DeclaredEnvironmentFlagAuthority.enabled(
            name=FeatureEnvironmentAuthority.FEATURE_ENV
        )
""",
    )

    findings = _focused_findings(tmp_path / "pkg")
    detector_type = EnvironmentBooleanAuthorityDriftDetector
    projected_findings = accumulate_compact_global_projections_for_roots(
        (tmp_path / "pkg",),
        (detector_type,),
        use_parse_cache=False,
    ).findings_by_detector(DetectorConfig())[detector_type]
    assert sorted(
        (json_report_object(finding) for finding in projected_findings),
        key=lambda item: item["summary"],
    ) == sorted(
        (json_report_object(finding) for finding in findings),
        key=lambda item: item["summary"],
    )
    summaries_by_symbol = {
        finding.evidence[0].symbol.split(":", 1)[0]: finding for finding in findings
    }

    assert set(summaries_by_symbol) == {
        "FeatureEnvironmentAuthority.enabled",
        "RuntimeProfileEnvironmentAuthority.enabled",
        "audit_enabled",
        "optional_enabled",
        "trace_enabled",
        "verbose_enabled",
    }
    profile_finding = summaries_by_symbol["RuntimeProfileEnvironmentAuthority.enabled"]
    assert "fallback '' makes absence enabled" in profile_finding.summary
    assert profile_finding.evidence[1].symbol == (
        "DeclaredEnvironmentFlagAuthority.enabled"
    )
    assert "environment-read default '0' makes absence disabled" in (
        summaries_by_symbol["trace_enabled"].summary
    )
    assert "environ[...]" in summaries_by_symbol["audit_enabled"].summary
    assert "implicit missing `None` value makes absence enabled" in (
        summaries_by_symbol["optional_enabled"].summary
    )
    assert "environment-read default '' makes absence enabled" in (
        summaries_by_symbol["verbose_enabled"].summary
    )
    wrapper_finding = summaries_by_symbol["FeatureEnvironmentAuthority.enabled"]
    assert "one-return fixed-key wrapper" in wrapper_finding.summary
    assert wrapper_finding.evidence[1].symbol == (
        "DeclaredEnvironmentFlagAuthority.enabled"
    )
    assert isinstance(
        wrapper_finding.metrics,
        FixedKeyEnvironmentAuthorityWrapperMetrics,
    )
    assert isinstance(profile_finding.metrics, LocalEnvironmentBooleanParserMetrics)
    assert profile_finding.metrics.read_kind is EnvironmentReadKind.ENVIRON_GET
    assert profile_finding.metrics.token_values == ("0", "false", "no", "off")
    assert profile_finding.metrics.matched_decision is False
    assert profile_finding.metrics.absent_decision is True

    snapshot = CodemodSourceSnapshot.from_modules(
        parse_python_modules(tmp_path / "pkg", use_parse_cache=False),
        findings,
    )
    plan = snapshot.plan_from_findings(findings)
    records_by_symbol = {
        record.finding.evidence[0].subject_symbol: record for record in plan.records
    }
    assert all(record.action_keys for record in plan.records)
    assert all(
        record.status is FindingRecipeSynthesisStatus.REJECTED_BY_SAFETY_CHECK
        for record in plan.records
    )
    assert "token and absent-state semantics are not proven equivalent" in (
        records_by_symbol["RuntimeProfileEnvironmentAuthority.enabled"].reason
    )
    assert "complete call and import reference closure" in (
        records_by_symbol["FeatureEnvironmentAuthority.enabled"].reason
    )
    assert records_by_symbol["trace_enabled"].reason == (
        "local environment parser has no source-proven declared authority"
    )


def test_ignores_non_boolean_environment_reads_and_declared_decision_maps(
    tmp_path: Path,
) -> None:
    _write_module(
        tmp_path,
        "pkg/clean.py",
        """
import os as operating_system
from os import getenv as environment_value

DECLARED_DECISIONS = {"yes": True, "no": False}


def cache_root() -> str | None:
    return operating_system.environ.get("CACHE_ROOT")


def backend_selected() -> bool:
    return environment_value("BACKEND") in ("cpu", "gpu")


def declared_environment_flag_decision(
    name: str,
    *,
    absent_decision: bool,
) -> bool:
    value = operating_system.environ.get(name)
    if value is None:
        return absent_decision
    normalized = value.strip().lower()
    if normalized not in DECLARED_DECISIONS:
        raise ValueError(normalized)
    return DECLARED_DECISIONS[normalized]


class RuntimeConsumer:
    def enabled(self, configuration) -> bool:
        return configuration.feature_enabled
""",
    )

    assert _focused_findings(tmp_path / "pkg") == []


def test_calibration_manifest_tracks_environment_boolean_detector(
    tmp_path: Path,
) -> None:
    _write_module(tmp_path, "calibration_pkg/central.py", _CENTRAL_AUTHORITY_SOURCE)
    _write_module(
        tmp_path,
        "calibration_pkg/local.py",
        """
import os

FALSE_VALUES = ("0", "false", "no", "off")


class LocalEnvironmentAuthority:
    @staticmethod
    def enabled() -> bool:
        value = os.getenv("LOCAL_FEATURE", "")
        return value.lower() not in FALSE_VALUES


class WrappedEnvironmentAuthority:
    @staticmethod
    def enabled() -> bool:
        return DeclaredEnvironmentFlagAuthority.enabled("WRAPPED_FEATURE")
""",
    )
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "environment-boolean-authority",
                        "path": "calibration_pkg",
                        "expected_detectors": [
                            {
                                "detector_id": ("environment_boolean_authority_drift"),
                                "min_count": 2,
                                "max_count": 2,
                            }
                        ],
                        "require_payoff_guard": False,
                        "max_scan_seconds": 20.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_calibration_manifest(
        manifest_path,
        use_parse_cache=False,
    )

    assert report.passes, report.regression_reasons
    assert (
        report.target_results[0].detector_count("environment_boolean_authority_drift")
        == 2
    )
