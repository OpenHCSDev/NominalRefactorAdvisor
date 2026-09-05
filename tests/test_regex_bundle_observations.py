"""Regex observations require imported declaration and lexical ownership evidence."""

import ast
from inspect import Parameter
from pathlib import Path
import weakref

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.analysis import release_module_analysis_memory
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.detectors._runtime import RepeatedLocalRegexBundleDetector
from nominal_refactor_advisor.detectors._regex_bundle import (
    RegexBundleModuleProjection,
    RegexPatternOperation,
)

PATTERNS = (
    r"\bname\s+([A-Za-z_][A-Za-z0-9_]*)",
    r"^\s*namespace\s+([A-Za-z0-9_.]+)\s*$",
    r"^\s*end(?:\s+[A-Za-z0-9_.]+)?\s*$",
)


def _module(header, call, *, parameters="", local_prefix="", local_suffix=""):
    source = header + "\n".join(
        f"def {name}({parameters}):\n"
        + local_prefix
        + "".join(
            f"    p{index} = {call.format(pattern=repr(pattern))}\n"
            for index, pattern in enumerate(PATTERNS)
        )
        + local_suffix
        + "    return p0, p1, p2\n"
        for name in ("first", "second")
    )
    return ParsedModule(
        path=Path("regex_sample.py"),
        module_name="regex_sample",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


@pytest.mark.parametrize(
    ("header", "call"),
    (
        ("import re\n", "re.compile({pattern})"),
        ("import re as regex\n", "regex.compile({pattern})"),
        ("from re import compile as pattern\n", "pattern({pattern})"),
        ("import re\ncompiler = re.compile\n", "compiler({pattern})"),
        ("import re\ncompiler = re.compile\nre = object()\n", "compiler({pattern})"),
        ("import re\n", "re.compile(pattern={pattern})"),
    ),
)
def test_import_aliases_and_keyword_arguments_identify_the_same_patterns(header, call):
    module = _module(header, call)
    namespace = {}
    exec(compile(module.source, str(module.path), "exec"), namespace)
    assert tuple(pattern.pattern for pattern in namespace["first"]()) == PATTERNS
    assert tuple(pattern.pattern for pattern in namespace["second"]()) == PATTERNS
    findings = RepeatedLocalRegexBundleDetector().detect([module], DetectorConfig())
    assert len(findings) == 1
    assert findings[0].authority_evidence is None


@pytest.mark.parametrize(
    ("header", "parameters", "prefix", "suffix"),
    (
        ("import re\n", "re", "", ""),
        ("import re\n", "", "    re = object()\n", ""),
        ("import re\n", "", "", "    re = object()\n"),
        ("import re\nre = object()\n", "", "", ""),
        ("from another_library import re\n", "", "", ""),
        ("", "re", "", ""),
    ),
)
def test_spelling_cannot_claim_a_regex_origin(header, parameters, prefix, suffix):
    module = _module(
        header,
        "re.compile({pattern})",
        parameters=parameters,
        local_prefix=prefix,
        local_suffix=suffix,
    )
    assert RepeatedLocalRegexBundleDetector().detect([module], DetectorConfig()) == []


@pytest.mark.parametrize(
    "call",
    (
        "re.compile({pattern}, pattern={pattern})",
        "re.compile({pattern}, unknown_option=True)",
        "re.compile(*({pattern},))",
    ),
)
def test_unproved_argument_bindings_do_not_count_as_pattern_uses(call):
    module = _module("import re\n", call)
    assert RepeatedLocalRegexBundleDetector().detect([module], DetectorConfig()) == []


@pytest.mark.parametrize("operation", tuple(RegexPatternOperation))
def test_operation_arguments_derive_from_the_native_declaration(operation):
    parameters = tuple(operation.call_signature.parameters.values())
    suffix = "".join(
        ", 'text'"
        for parameter in parameters[1:]
        if parameter.default is Parameter.empty
    )
    call = f"re.{operation.function.__name__}({{pattern}}{suffix})"
    module = _module("import re\n", call)
    namespace = {}
    exec(compile(module.source, str(module.path), "exec"), namespace)
    assert len(namespace["first"]()) == 3
    assert (
        len(RepeatedLocalRegexBundleDetector().detect([module], DetectorConfig())) == 1
    )


def test_unrelated_imports_do_not_force_lexical_or_declaration_projection():
    module = _module("import other\n", "other.compile({pattern})")
    projection = RegexBundleModuleProjection(module)
    call = next(node for node in ast.walk(module.module) if isinstance(node, ast.Call))
    assert projection.pattern_literal(call) is None
    assert "external_reference_ids" not in vars(projection)
    assert "stable_binding_names" not in vars(projection)


def test_cached_source_projection_does_not_cache_configuration_or_retain_released_modules():
    module = _module("import re\n", "re.compile({pattern})")
    projection = RegexBundleModuleProjection.from_module(module)
    assert RegexBundleModuleProjection.from_module(module) is projection
    assert len(RegexBundleModuleProjection.collect(module, DetectorConfig())) == 1
    assert (
        RegexBundleModuleProjection.collect(
            module, DetectorConfig(min_repeated_local_regex_literals=4)
        )
        == ()
    )
    retained = weakref.ref(module)
    del module, projection
    assert retained() is not None
    release_module_analysis_memory()
    assert RegexBundleModuleProjection.from_module.cache_info().currsize == 0
    assert retained() is None
