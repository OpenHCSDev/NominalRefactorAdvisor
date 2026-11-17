"""A dataclass member's place in the field order, settled by the repository.

CPython's `dataclasses` makes a default a field specification only when it is
a `dataclasses.Field`, and reads a member as `ClassVar`, `InitVar` or `KW_ONLY`
only from its annotation; everything else is a plain field in order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import InstalledClassDeclarations
from nominal_refactor_advisor.codemod import CodemodSourceSnapshot
from nominal_refactor_advisor.installed_source import InstalledModule

SORTS = (
    "class ThreadSort:\n    pass\n\n\n"
    "class CreatedThreadSort(ThreadSort):\n    pass\n"
)
TEXT = (
    "from nra_installed_text import Choice\n\n\n"
    "class ThreadText(Choice):\n    pass\n\n\n"
    "class NoThreadText(ThreadText):\n    pass\n"
)
CHANNELS = (
    "from __future__ import annotations\n\n"
    "from dataclasses import dataclass\n\n"
    "from pkg.sorts import CreatedThreadSort, ThreadSort\n"
    "from pkg.text import NoThreadText, ThreadText\n\n\n"
    "@dataclass(frozen=True)\n"
    "class Channel:\n"
    "    name: str\n"
    "    title: ThreadText = NoThreadText()\n"
    "    tags: frozenset[str] = frozenset()\n"
    "    order: type[ThreadSort] = CreatedThreadSort\n\n\n"
    "@dataclass(frozen=True)\n"
    "class CursorRow:\n"
    "    cursor: ConversationCursor\n"
    "    scope: 'DMDisplayScope'\n\n\n"
    "class ConversationCursor:\n    pass\n\n\n"
    "class DMDisplayScope:\n    pass\n"
)


def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatch: str) -> None:
    site = tmp_path / "site"
    (site / "nra_installed_text").mkdir(parents=True)
    (site / "nra_installed_text/__init__.py").write_text("from .choice import Choice\n")
    (site / "nra_installed_text/choice.py").write_text(dispatch)
    monkeypatch.syspath_prepend(str(site))
    InstalledModule.located_source.cache_clear()
    InstalledModule.parsed.cache_clear()
    InstalledClassDeclarations.module.cache_clear()


def _field_names(tmp_path: Path, files: dict[str, str], symbol: str) -> tuple[str, ...] | None:
    root = tmp_path / "project"
    for relative, source in {"pkg/__init__.py": "", **files}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))
    indexed = snapshot.class_overlap_verdicts.class_index.class_for(symbol)
    assert indexed is not None
    return indexed.own_dataclass_field_names


PLAIN_CHOICE = (
    "class Choice:\n"
    "    def __init_subclass__(cls, **kwargs):\n"
    "        super().__init_subclass__(**kwargs)\n"
)


def test_a_class_call_a_builtin_type_call_and_a_class_reference_are_plain_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`NoThreadText()` builds an instance: its installed base defines no
    `__new__` and no metaclass replaces `type.__call__`. `frozenset()` is a
    builtin type's instance, and `CreatedThreadSort` a class object: none is
    a `Field`."""

    _installed(tmp_path, monkeypatch, PLAIN_CHOICE)
    names = _field_names(
        tmp_path,
        {"pkg/sorts.py": SORTS, "pkg/text.py": TEXT, "pkg/channels.py": CHANNELS},
        "pkg.channels.Channel",
    )
    assert names == ("name", "title", "tags", "order")


def test_an_annotation_naming_a_class_declared_later_is_a_plain_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `from __future__ import annotations`, `dataclasses._is_type` finds
    no `ConversationCursor` in the module while `CursorRow` is created, and
    neither for the explicit string `'DMDisplayScope'`: both are plain fields."""

    _installed(tmp_path, monkeypatch, PLAIN_CHOICE)
    names = _field_names(
        tmp_path,
        {"pkg/sorts.py": SORTS, "pkg/text.py": TEXT, "pkg/channels.py": CHANNELS},
        "pkg.channels.CursorRow",
    )
    assert names == ("cursor", "scope")


@pytest.mark.parametrize(
    ("default", "dispatch"),
    (
        ("made()", PLAIN_CHOICE),
        ("NoThreadText()", "class Meta(type):\n    def __call__(cls):\n        return 0\n\n\nclass Choice(metaclass=Meta):\n    pass\n"),
        ("NoThreadText()", "class Choice:\n    def __new__(cls):\n        return 0\n"),
        ("MADE", PLAIN_CHOICE),
    ),
    ids=("unknown-function", "metaclass-call", "new", "module-variable"),
)
def test_a_default_that_may_be_a_field_leaves_the_order_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default: str, dispatch: str
) -> None:
    """A call of a function NRA cannot see, a class whose metaclass `__call__`
    or `__new__` may return anything, and a module variable may each be a
    `dataclasses.Field`."""

    _installed(tmp_path, monkeypatch, dispatch)
    records = (
        "from dataclasses import dataclass\n\n"
        "from pkg.text import NoThreadText\n"
        "from pkg.helpers import MADE, made\n\n\n"
        "@dataclass(frozen=True)\n"
        f"class Record:\n    name: str\n    value: object = {default}\n"
    )
    helpers = (
        "from dataclasses import field\n\nMADE = field(default=0)\n_PENDING = []\n\n\n"
        "def made():\n    return _PENDING.pop()\n"
    )
    names = _field_names(
        tmp_path,
        {"pkg/text.py": TEXT, "pkg/helpers.py": helpers, "pkg/records.py": records},
        "pkg.records.Record",
    )
    assert names is None
