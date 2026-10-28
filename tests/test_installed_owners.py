"""Registration owners from a project's installed dependencies (#71)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nominal_refactor_advisor import detectors
from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.detectors import DetectorConfig
from nominal_refactor_advisor.installed_owners import (
    InstalledDistributionOwners,
    OwnerPackagesDeclaration,
    RegistrationOwners,
    RegistrationRole,
)

OWNER = "metaclass_registry.core.AutoRegisterMeta"

HAND_ROLLED = (
    "class RegistryMeta(type):\n"
    "    _registry = {}\n\n"
    "    def __new__(mcs, name, bases, namespace):\n"
    "        created = super().__new__(mcs, name, bases, namespace)\n"
    "        if bases:\n"
    "            mcs._registry[namespace.get('key', name)] = created\n"
    "        return created\n\n\n"
    "class Plugin(metaclass=RegistryMeta):\n    pass\n\n\n"
    "class Alpha(Plugin):\n    key = 'alpha'\n\n\n"
    "class Beta(Plugin):\n    key = 'beta'\n"
)

MANUAL = (
    "class Alpha:\n    pass\n\n\n"
    "class Beta:\n    pass\n\n\n"
    "REGISTRY = {}\n"
    "REGISTRY['alpha'] = Alpha\n"
    "REGISTRY['beta'] = Beta\n"
)


def _project(root: Path, pyproject: str | None) -> Path:
    package = root / "plugins"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "family.py").write_text(HAND_ROLLED)
    (package / "manual.py").write_text(MANUAL)
    if pyproject is not None:
        (root / "pyproject.toml").write_text(pyproject)
    return root


def _summaries(root: Path, detector: type) -> list[str]:
    modules = parse_python_modules(
        root,
        use_parse_cache=False,
        source_policy=PythonSourcePathPolicy(include_tests=False),
    )
    return [finding.summary for finding in detector().detect(modules, DetectorConfig())]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NRA_CACHE_HOME", str(tmp_path / "cache"))
    InstalledDistributionOwners.of.cache_clear()


def test_an_installed_owner_is_found_by_what_its_source_does(tmp_path: Path) -> None:
    owners = InstalledDistributionOwners.of("metaclass-registry")

    assert [(owner.symbol, owner.role) for owner in owners.owners] == [
        (OWNER, RegistrationRole.METACLASS)
    ]
    (cached,) = (tmp_path / "cache" / "installed-owners").iterdir()
    assert cached.name.startswith(f"metaclass-registry-{owners.version}-")
    assert InstalledDistributionOwners.decoded(json.loads(cached.read_text())) == owners


def test_owner_packages_absent_declares_nothing(tmp_path: Path) -> None:
    root = _project(tmp_path / "repo", "[project]\nname = 'plugins'\n")

    assert OwnerPackagesDeclaration.declared_for(
        root / "plugins" / "family.py"
    ) == OwnerPackagesDeclaration()
    assert _summaries(root, detectors.HandRolledRegistrationDetector) == []
    (manual,) = _summaries(root, detectors.ManualClassRegistrationDetector)
    assert "No installed dependency declares a class-creation registration owner (gap)" in manual
    assert "owner-packages: not declared" in manual


def test_a_declared_owner_package_names_the_installed_owner(tmp_path: Path) -> None:
    pyproject = '[tool.nominal-refactor-advisor]\nowner-packages = ["metaclass-registry"]\n'
    root = _project(tmp_path / "repo", pyproject)
    digest = hashlib.sha256(pyproject.encode()).hexdigest()

    (hand_rolled,) = _summaries(root, detectors.HandRolledRegistrationDetector)
    (manual,) = _summaries(root, detectors.ManualClassRegistrationDetector)

    assert hand_rolled.startswith(
        "`RegistryMeta` stores each class it creates in a mapping. Installed owners "
        f"(confidence uncalibrated): 1.00 inherit or compose installed `{OWNER}` "
        "(metaclass-registry "
    )
    # Declared but not yet imported, so the move adds the dependency; only the
    # hand-rolled metaclass plays the owner's role.
    assert "(-0.25 import_added, +1.00 plays_same_role)" in hand_rolled
    assert "(-0.25 import_added)" in manual
    assert f"Move: make the family inherit or compose `{OWNER}`" in hand_rolled
    assert f"Move: register through `{OWNER}` at class creation" in manual
    for summary in (hand_rolled, manual):
        assert f"(sha256 {digest[:12]})" in summary


def test_a_declared_package_that_is_not_installed_is_a_gap(tmp_path: Path) -> None:
    pyproject = (
        "[tool.nominal-refactor-advisor]\n"
        'owner-packages = ["metaclass-registry", "no-such-distribution-nra"]\n'
    )
    root = _project(tmp_path / "repo", pyproject)

    scope = RegistrationOwners.for_scan((), root / "plugins" / "family.py")

    assert [owner.symbol for owner in scope.owners] == [OWNER]
    (gap,) = scope.gaps
    assert gap.startswith("`no-such-distribution-nra` is declared in owner-packages")
