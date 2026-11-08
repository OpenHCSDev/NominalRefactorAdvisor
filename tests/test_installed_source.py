"""An installed module's collected families come from the scan's family cache once stored."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.cache_paths import ActiveScanCache
from nominal_refactor_advisor.class_index import CompactModuleClassProjectionFamily
from nominal_refactor_advisor.installed_source import InstalledModule


def test_an_installed_modules_classes_are_read_from_the_family_cache_once_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm agent-comms scan parsed 31 installed modules again (0.8 s) to read
    their classes, though their source had not changed (11-08)."""

    site = tmp_path / "site"
    site.mkdir()
    (site / "nra_test_cached_install.py").write_text("class Installed:\n    pass\n")
    monkeypatch.syspath_prepend(str(site))
    InstalledModule.located_source.cache_clear()
    cache_dir = tmp_path / "collected-family"
    with ActiveScanCache.opened(cache_dir):
        (stored,) = InstalledModule.family_items(
            "nra_test_cached_install", CompactModuleClassProjectionFamily
        )

    # Patching anything NRA's family implementation reaches (InstalledModule,
    # the family's collect) would change the family's cache key by design, so
    # the spy sits on `ast.parse`, which that dependency walk does not expand.
    InstalledModule.parsed.cache_clear()
    real_parse = ast.parse

    def parse(source, filename="<unknown>", *args, **kwargs):
        if "nra_test_cached_install" in str(filename):
            pytest.fail("installed source parsed again")
        return real_parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)
    with ActiveScanCache.opened(cache_dir):
        (read,) = InstalledModule.family_items(
            "nra_test_cached_install", CompactModuleClassProjectionFamily
        )

    assert read == stored
    assert [declared.symbol for declared in read.classes] == [
        "nra_test_cached_install.Installed"
    ]
