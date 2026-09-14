"""Requested native policy compatibility does not prove target execution."""

import ast
from pathlib import Path
from types import ModuleType

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    ConvertManualRegistryToAutoregisterOperation,
    DeriveAutoregisterInstanceViewOperation,
    RefactorRecipe,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.source_index import build_source_index

_PLAIN = """REGISTRY = {}
class Alpha:
    pass
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""

_DECORATED = """def replace_class(cls):
    return object()

@replace_class
class Alpha:
    pass
class Beta:
    pass
REGISTRY = {'alpha': Alpha, 'beta': Beta}
"""

_SUBCLASS_HOOK = """class Handler:
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.registry_key = 'rewritten'

class Alpha(Handler):
    pass
class Beta(Handler):
    pass
REGISTRY = {'alpha': Alpha, 'beta': Beta}
"""

_ALIASED_OBSERVATION = """REGISTRY = {}
alias = REGISTRY

def observe():
    return len(alias)

class Alpha:
    pass
observed = observe()
class Beta:
    pass
REGISTRY['alpha'] = Alpha
REGISTRY['beta'] = Beta
"""

_SPOOFED_METACLASS = """class AutoRegisterMeta(type):
    pass

class Handler(metaclass=AutoRegisterMeta):
    __registry__ = {}
    __registry_key__ = 'registry_key'
    __skip_if_no_key__ = True

class Alpha(Handler):
    pass
class Beta(Handler):
    pass
REGISTRY = {'alpha': Alpha(), 'beta': Beta()}
"""


def _execute(source: str) -> ModuleType:
    # Only execute the small fixtures authored in this file, never repository code.
    module = ModuleType("registry_policy_fixture")
    exec(compile(source, "<registry-policy-fixture>", "exec"), module.__dict__)
    return module


def _simulate(source: str, operation_type: type, anchor: str):
    parsed = ParsedModule(
        Path("/repo/registry_policy_fixture.py"),
        "registry_policy_fixture",
        False,
        ast.parse(source),
        source,
    )
    snapshot = CodemodSourceSnapshot.from_indexed_sources(
        build_source_index([parsed], ()), {parsed.file_path: source}
    )
    result = (
        RefactorRecipe("native-policy-integrity")
        .with_operation(
            operation_type(
                target=SourceRewriteTarget(file_path=parsed.file_path, qualname=anchor)
            )
        )
        .simulate(snapshot)
    )
    # Post-render refusals obey the same admission contract as source preflight.
    result.preflight_report.require_clean()
    return result, parsed.file_path


def _assert_direct_registry(module: ModuleType) -> None:
    assert type(module.REGISTRY) is dict
    assert tuple(module.REGISTRY) == ("alpha", "beta")
    assert module.REGISTRY["alpha"] is module.Alpha
    assert module.REGISTRY["beta"] is module.Beta


def test_plain_native_registration_still_converts():
    before = _execute(_PLAIN)
    _assert_direct_registry(before)
    result, path = _simulate(
        _PLAIN, ConvertManualRegistryToAutoregisterOperation, "Alpha"
    )
    assert result.is_clean
    _assert_direct_registry(_execute(result.simulation.rewritten_sources[path]))


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(_DECORATED, id="decorator-changes-final-binding"),
        pytest.param(_SUBCLASS_HOOK, id="subclass-hook-rewrites-key"),
    ),
)
def test_requested_policy_does_not_replace_actual_class_activation(source: str):
    before = _execute(source)
    _assert_direct_registry(before)
    try:
        result, path = _simulate(
            source, ConvertManualRegistryToAutoregisterOperation, "Alpha"
        )
    except ValueError:
        return  # Explicit refusal is valid until target activation is proved.
    assert result.is_clean
    _assert_direct_registry(_execute(result.simulation.rewritten_sources[path]))


def test_registration_timing_preserves_alias_mediated_observations():
    before = _execute(_ALIASED_OBSERVATION)
    _assert_direct_registry(before)
    assert before.observed == 0
    assert before.alias is before.REGISTRY
    try:
        result, path = _simulate(
            _ALIASED_OBSERVATION, ConvertManualRegistryToAutoregisterOperation, "Alpha"
        )
    except ValueError:
        return
    assert result.is_clean
    after = _execute(result.simulation.rewritten_sources[path])
    _assert_direct_registry(after)
    assert after.alias is after.REGISTRY
    assert after.observed == before.observed


def test_installed_metaclass_probe_does_not_authenticate_source_spelling():
    before = _execute(_SPOOFED_METACLASS)
    assert type(before.REGISTRY) is dict
    assert tuple(before.REGISTRY) == ("alpha", "beta")
    assert type(before.REGISTRY["alpha"]) is before.Alpha
    assert type(before.REGISTRY["beta"]) is before.Beta
    assert before.Handler.__registry__ == {}
    try:
        result, path = _simulate(
            _SPOOFED_METACLASS, DeriveAutoregisterInstanceViewOperation, "Handler"
        )
    except ValueError:
        return
    assert result.is_clean
    after = _execute(result.simulation.rewritten_sources[path])
    assert type(after.REGISTRY) is dict
    assert tuple(after.REGISTRY) == ("alpha", "beta")
    assert type(after.REGISTRY["alpha"]) is after.Alpha
    assert type(after.REGISTRY["beta"]) is after.Beta


@pytest.mark.parametrize(
    "prefix, observe",
    (
        pytest.param(
            "REGISTRY={}\nalias=REGISTRY\nclass Alpha:pass\n"
            "observed=dict(alias)\nclass Beta:pass\n",
            lambda module: tuple(module.observed),
            id="registry-copy-between-classes",
        ),
        pytest.param(
            "REGISTRY={}\nalias=REGISTRY\nclass Alpha:pass\nclass Beta:pass\n"
            "observed=dict(alias)\n",
            lambda module: tuple(module.observed),
            id="registry-copy-before-original-stores",
        ),
        pytest.param(
            "REGISTRY={}\nalias=REGISTRY\nclass Alpha:pass\n"
            "class Beta:\n    observed=dict(alias)\n",
            lambda module: tuple(module.Beta.observed),
            id="registry-copy-inside-child-activation",
        ),
        pytest.param(
            "REGISTRY={}\nclass Alpha:pass\nobserved=dict(globals())\n"
            "class Beta:pass\n",
            lambda module: tuple(module.observed),
            id="generated-module-bindings-observed",
        ),
    ),
)
def test_registration_motion_preserves_intermediate_storage_observations(
    prefix, observe
):
    source = prefix + "REGISTRY['alpha']=Alpha\nREGISTRY['beta']=Beta\n"
    before = _execute(source)
    _assert_direct_registry(before)
    try:
        result, path = _simulate(
            source, ConvertManualRegistryToAutoregisterOperation, "Alpha"
        )
    except ValueError:
        return
    assert result.is_clean
    after = _execute(result.simulation.rewritten_sources[path])
    _assert_direct_registry(after)
    assert observe(after) == observe(before)


@pytest.mark.parametrize(
    "prefix",
    (
        pytest.param(
            "REGISTRY={}\nalias=REGISTRY\nobserved=dict(alias)\n"
            "class Alpha:pass\nclass Beta:pass\n",
            id="snapshot-before-registration-moves",
        ),
        pytest.param(
            "REGISTRY={}\nother={}\nclass Alpha:pass\n"
            "observed=dict(other)\nclass Beta:pass\n",
            id="independent-storage",
        ),
        pytest.param(
            "REGISTRY={}\nalias=REGISTRY\nfrozen=dict(alias)\nclass Alpha:pass\n"
            "observed=dict(frozen)\nclass Beta:pass\n",
            id="independent-earlier-copy",
        ),
    ),
)
def test_registration_motion_keeps_independent_storage_observations_valid(prefix):
    source = prefix + "REGISTRY['alpha']=Alpha\nREGISTRY['beta']=Beta\n"
    before = _execute(source)
    result, path = _simulate(
        source, ConvertManualRegistryToAutoregisterOperation, "Alpha"
    )
    assert result.is_clean
    after = _execute(result.simulation.rewritten_sources[path])
    _assert_direct_registry(after)
    assert tuple(after.observed) == tuple(before.observed)
