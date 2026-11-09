"""A family that walks a package for its members registers a member only where the
walk reaches; a move refuses to take one out of it."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.codemod import ArchitectureGuardSuite
from nominal_refactor_advisor.codemod_workflow import CodemodRefactorGoalRunner
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.refactor_concepts import PackageMoveConcept
from tests.test_package_move_batch import _moved, _runs, _write


def _family(recursive: str) -> dict[str, str]:
    return {
        "pkg/__init__.py": "",
        "pkg/messaging/__init__.py": "",
        "pkg/family.py": (
            "from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict, RegistryConfig\n\n\n"
            "class Plugin(metaclass=AutoRegisterMeta):\n"
            "    __registry_config__ = RegistryConfig(\n"
            "        registry_dict=LazyDiscoveryDict(),\n"
            '        key_attribute="key",\n'
            "        discovery_package=__package__,\n"
            f"{recursive}"
            "    )\n"
            "    key = None\n"
        ),
        "pkg/alpha.py": (
            "from pkg.family import Plugin\n\n\nclass Alpha(Plugin):\n    key = 'alpha'\n"
        ),
        "pkg/app.py": "from pkg.family import Plugin\n\nVALUE = sorted(Plugin.__registry__)\n",
    }


def test_a_move_out_of_a_flat_walk_refuses_naming_the_family(tmp_path: Path) -> None:
    """`Plugin` walks `pkg`'s direct modules. `pkg/messaging/alpha.py` is one level
    down, where the walk never looks, so `Alpha` would silently stop registering.
    The move refuses with the family and its root named, and the app still sees it."""

    _write(tmp_path, _family(""))

    report = CodemodRefactorGoalRunner(
        roots=(tmp_path,),
        config=DetectorConfig(package_destination_tables=("| `messaging` | `alpha.py` | |\n",)),
        dry_run=True,
        migration_type=PackageMoveConcept,
        guard_suite=ArchitectureGuardSuite(),
    ).run()
    assert (
        "Alpha is a member of the Plugin family, and moving it to pkg.messaging.alpha "
        "would stop it registering: it walks pkg's direct modules (the walk is flat)"
    ) in report.to_markdown()
    assert not report.replay_sequence.documents
    assert _runs(tmp_path, "VALUE") == "['alpha']"


def test_a_move_inside_a_recursive_walk_still_moves(tmp_path: Path) -> None:
    """The same family walking `pkg`'s whole tree still reaches `pkg.messaging.alpha`."""

    _write(tmp_path, _family("        discovery_recursive=True,\n"))

    _stages, root = _moved(tmp_path, "| `messaging` | `alpha.py` | |\n")
    assert (root / "pkg/messaging/alpha.py").exists()
    assert _runs(root, "VALUE") == "['alpha']"
