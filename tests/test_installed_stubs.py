"""An installed module with no `.py` is read from its stub, and a package's stub states the
names its `.py` binds only at run time: both give a scanned class over them a proved order."""

import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.class_index import (
    CompactRepositoryPublicExposureIndex,
    InstalledClassDeclarations,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._sibling_overrides import (
    SiblingOverrideDetector,
)
from nominal_refactor_advisor.installed_source import InstalledModule
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


@pytest.fixture
def toolkit(tmp_path: Path, monkeypatch):
    """An installed toolkit: `toolkit.native` is compiled (a stub only), and
    `toolkit.widgets` binds its widgets in `__getattr__` and states them in a stub."""

    installed = tmp_path / "site"
    _write(
        installed,
        {
            "toolkit/__init__.py": "",
            "toolkit/native.pyi": (
                "class Object:\n    def name(self) -> str: ...\n\n\n"
                "class Widget(Object):\n    def show(self) -> None: ...\n"
            ),
            "toolkit/widgets/__init__.py": (
                "import importlib\n\n\ndef __getattr__(name):\n"
                "    return getattr(importlib.import_module('toolkit.widgets._' + name.lower()), name)\n"
            ),
            "toolkit/widgets/__init__.pyi": "from ._static import Static as Static\n",
            "toolkit/widgets/_static.py": "class Static:\n    def render(self) -> str:\n        return ''\n",
            "toolkit/both.py": "class FromSource:\n    pass\n",
            "toolkit/both.pyi": "class FromStub: ...\n",
        },
    )
    monkeypatch.syspath_prepend(str(installed))
    for cached in (InstalledModule.located_source, InstalledModule.parsed,
                   InstalledClassDeclarations.module, InstalledClassDeclarations.stub,
                   InstalledClassDeclarations.lazy_getattr_aliases):
        cached.cache_clear()
    yield installed
    for cached in (InstalledModule.located_source, InstalledModule.parsed,
                   InstalledClassDeclarations.module, InstalledClassDeclarations.stub,
                   InstalledClassDeclarations.lazy_getattr_aliases):
        cached.cache_clear()
    for name in [name for name in sys.modules if name == "toolkit" or name.startswith("toolkit.")]:
        del sys.modules[name]


def test_a_module_with_no_source_is_read_from_its_stub_and_a_source_wins_over_its_stub(toolkit: Path):
    assert InstalledModule.source_path("toolkit.native") == toolkit / "toolkit/native.pyi"
    assert InstalledModule.stub_path("toolkit.native") is None
    assert InstalledModule.source_path("toolkit.both") == toolkit / "toolkit/both.py"
    assert InstalledModule.stub_path("toolkit.both") == toolkit / "toolkit/both.pyi"
    assert InstalledModule.stub_path("toolkit.widgets") == toolkit / "toolkit/widgets/__init__.pyi"
    assert [declared.qualname for declared in InstalledClassDeclarations.module("toolkit.native").classes] == [
        "Object", "Widget"
    ]
    assert [declared.qualname for declared in InstalledClassDeclarations.module("toolkit.both").classes] == [
        "FromSource"
    ]


def test_classes_over_a_stubbed_base_and_a_run_time_reexport_have_a_proved_order(toolkit: Path, tmp_path: Path):
    scanned = tmp_path / "scan"
    _write(
        scanned,
        {
            "app/__init__.py": "",
            "app/shared.py": (
                "from abc import abstractmethod\n\n\n"
                "class Scope:\n    @abstractmethod\n    def scope(self) -> str: ...\n"
            ),
            "app/native_view.py": (
                "from toolkit.native import Widget\n\nfrom app.shared import Scope\n\n\n"
                "class NativeView(Scope, Widget):\n    def scope(self) -> str:\n        return 'thread'\n"
            ),
            "app/text_view.py": (
                "from toolkit.widgets import Static\n\nfrom app.shared import Scope\n\n\n"
                "class TextView(Scope, Static):\n    def scope(self) -> str:\n        return 'thread'\n"
            ),
        },
    )
    with ScanCache.scope():
        (finding,) = SiblingOverrideDetector().detect(
            parse_python_modules(
                scanned, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=False)
            ),
            DetectorConfig(),
        )
    assert [evidence.symbol for evidence in finding.evidence] == ["NativeView.scope", "TextView.scope"]
    # Both orders are proved: the compiled base from its stub, the lazy widget from the
    # stub beside the package that binds it in `__getattr__`.
    assert "is not proved" not in finding.summary
    assert "`Scope` already exists to hold it" in finding.summary


def test_installed_star_reexport_and_pep562_type_checking_reexport_are_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    """Installed modules re-exporting via `from mod import *` (like `collections.abc`)
    or PEP 562 `def __getattr__` + `if TYPE_CHECKING:` (like `pydantic`) resolve
    statically without executing third-party code; non-`__getattr__` modules never
    expose `if TYPE_CHECKING:` imports as runtime re-exports."""

    installed = tmp_path / "site"
    _write(
        installed,
        {
            "fakelib/__init__.py": (
                "import typing\n\n"
                "if typing.TYPE_CHECKING:\n"
                "    from .main import BaseModel\n\n"
                "def __getattr__(name: str):\n"
                "    raise AttributeError(name)\n"
            ),
            "fakelib/main.py": "class BaseModel:\n    pass\n",
            "fakelib/star_impl.py": "class StarBase:\n    pass\n",
            "fakelib/star_pub.py": "from fakelib.star_impl import *\n",
            "fakelib/no_getattr.py": (
                "from typing import TYPE_CHECKING\n\n"
                "if TYPE_CHECKING:\n"
                "    from .main import BaseModel\n"
            ),
            "fakelib/cycle_a.py": "from fakelib.cycle_b import *\n",
            "fakelib/cycle_b.py": "from fakelib.cycle_a import *\n",
        },
    )
    monkeypatch.syspath_prepend(str(installed))
    caches = (
        InstalledModule.located_source,
        InstalledModule.parsed,
        InstalledClassDeclarations.module,
        InstalledClassDeclarations.stub,
        InstalledClassDeclarations.lazy_getattr_aliases,
    )
    for cached in caches:
        cached.cache_clear()
    try:
        scanned = tmp_path / "scan"
        _write(
            scanned,
            {
                "app/__init__.py": "",
                "app/shared.py": (
                    "from abc import abstractmethod\n\n\n"
                    "class Scope:\n    @abstractmethod\n    def scope(self) -> str: ...\n"
                ),
                "app/views.py": (
                    "from collections.abc import MutableMapping\n"
                    "from fakelib import BaseModel\n"
                    "from fakelib.star_pub import StarBase\n"
                    "from app.shared import Scope\n\n\n"
                    "class ModelView(Scope, BaseModel):\n    def scope(self) -> str:\n        return 'm'\n\n\n"
                    "class StarView(Scope, StarBase):\n    def scope(self) -> str:\n        return 'm'\n\n\n"
                    "class MapView(Scope, MutableMapping):\n    def scope(self) -> str:\n        return 'm'\n"
                ),
            },
        )
        modules = parse_python_modules(
            scanned,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
        exposure = CompactRepositoryPublicExposureIndex(())
        assert (
            InstalledClassDeclarations.declaration("fakelib.BaseModel", exposure).symbol
            == "fakelib.main.BaseModel"
        )
        assert (
            InstalledClassDeclarations.declaration("fakelib.star_pub.StarBase", exposure).symbol
            == "fakelib.star_impl.StarBase"
        )
        assert (
            InstalledClassDeclarations.declaration("collections.abc.MutableMapping", exposure).symbol
            in {"collections.abc.MutableMapping", "_collections_abc.MutableMapping"}
        )
        assert InstalledClassDeclarations.declaration("fakelib.no_getattr.BaseModel", exposure) is None
        assert InstalledClassDeclarations.declaration("fakelib.cycle_a.Missing", exposure) is None

        with ScanCache.scope():
            (finding,) = SiblingOverrideDetector().detect(modules, DetectorConfig())
        assert "is not proved" not in finding.summary
        assert "`Scope` already exists to hold it" in finding.summary
    finally:
        for cached in caches:
            cached.cache_clear()
