"""Name-only subclasses and undistinguished enum members, replayed across modules.

Each case writes a package whose declarations and consumers span modules,
runs it, applies every recipe the detector's findings offer, checks the
exact rewritten source of every module, runs it again and re-scans.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_false_distinctions import (
    FoldEnumMembersOperation,
    RedirectEmptySubclassOperation,
)
from nominal_refactor_advisor.false_distinction_observations import NameUse, NameUseKind
from nominal_refactor_advisor.detectors import (
    DetectorConfig,
    NameOnlySubclassDetector,
    UndistinguishedEnumMembersDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache

HANDLERS = "class Handler:\n    def handle(self):\n        return 'handled'\n"
USERS = (
    "from pkg.handlers import Handler\n\n\nclass LocalUser(Handler):\n    pass\n\n\n"
    "class RemoteUser(Handler):\n    \"\"\"Reached over the network.\"\"\"\n\n\n"
    "def default():\n    return LocalUser()\n"
)
APP = (
    "from pkg.handlers import Handler\nfrom pkg.users import LocalUser, RemoteUser\n\n\n"
    "def run():\n    return [LocalUser().handle(), RemoteUser().handle(), Handler().handle()]\n"
)
OTHER = "from pkg import users\n\n\ndef make():\n    return users.RemoteUser()\n"
USERS_MAIN = (
    "from pkg.app import run\nfrom pkg.other import make\nfrom pkg.users import default\n\n"
    "print(run(), make().handle(), default().handle())\n"
)
ERRORS = (
    "class AppError(Exception):\n    pass\n\n\nclass LoadError(AppError):\n    pass\n\n\n"
    "class SaveError(AppError):\n    pass\n"
)
STORE = (
    "from .errors import AppError, LoadError, SaveError\n\n\ndef load(ok):\n"
    "    if not ok:\n        raise LoadError('load')\n    return 'loaded'\n\n\n"
    "def save():\n    raise SaveError('save')\n\n\ndef attempt(action):\n    try:\n"
    "        return action()\n    except AppError as error:\n        return str(error)\n"
)
STORE_MAIN = (
    "from pkg.store import attempt, load, save\n\n"
    "print(attempt(lambda: load(True)), attempt(lambda: load(False)), attempt(save))\n"
)

REDIRECTS = {
    "name-and-module-imports": (
        {
            "pkg/__init__.py": "",
            "pkg/handlers.py": HANDLERS,
            "pkg/users.py": USERS,
            "pkg/app.py": APP,
            "pkg/other.py": OTHER,
            "main.py": USERS_MAIN,
        },
        {
            "pkg/users.py": "from pkg.handlers import Handler\n\n\ndef default():\n    return Handler()\n",
            "pkg/app.py": (
                "from pkg.handlers import Handler\n\n\n"
                "def run():\n    return [Handler().handle(), Handler().handle(), Handler().handle()]\n"
            ),
            "pkg/other.py": "from pkg import users\n\n\ndef make():\n    return users.Handler()\n",
        },
        2,
    ),
    "relative-imports-and-raise": (
        {
            "pkg/__init__.py": "",
            "pkg/errors.py": ERRORS,
            "pkg/store.py": STORE,
            "main.py": STORE_MAIN,
        },
        {
            "pkg/errors.py": "class AppError(Exception):\n    pass\n",
            "pkg/store.py": STORE.replace(
                "from .errors import AppError, LoadError, SaveError\n",
                "from .errors import AppError\n",
            )
            .replace("raise LoadError('load')", "raise AppError('load')")
            .replace("raise SaveError('save')", "raise AppError('save')"),
        },
        2,
    ),
}

MODES = "from enum import Enum\n\n\nclass Mode(Enum):\n    FAST = 1\n    QUICK = 2\n    SLOW = 3\n"
BUDGET = (
    "from pkg.modes import Mode\n\n\ndef budget(mode: Mode) -> int:\n"
    "    if mode in (Mode.FAST, Mode.QUICK):\n        return 10\n    return 20\n"
)
LABEL = (
    "from pkg import modes\n\n\ndef label(mode):\n    match mode:\n"
    "        case modes.Mode.FAST | modes.Mode.QUICK:\n            return 'fast'\n"
    "        case _:\n            return 'slow'\n\n\n"
    "def is_fast(mode):\n    return mode is modes.Mode.FAST or mode is modes.Mode.QUICK\n"
)
MODES_MAIN = (
    "from pkg.budget import budget\nfrom pkg.label import is_fast, label\n"
    "from pkg.modes import Mode\n\n"
    "print(budget(Mode.SLOW), label(Mode.SLOW), is_fast(Mode.SLOW))\n"
)
MODE_PACKAGE = {
    "pkg/__init__.py": "",
    "pkg/modes.py": MODES,
    "pkg/budget.py": BUDGET,
    "pkg/label.py": LABEL,
    "main.py": MODES_MAIN,
}
MODE_FOLDED = {
    "pkg/modes.py": "from enum import Enum\n\n\nclass Mode(Enum):\n    FAST = 1\n    SLOW = 3\n",
    "pkg/budget.py": BUDGET.replace("(Mode.FAST, Mode.QUICK)", "(Mode.FAST,)"),
    "pkg/label.py": LABEL.replace(
        "case modes.Mode.FAST | modes.Mode.QUICK:", "case modes.Mode.FAST:"
    ).replace(
        "mode is modes.Mode.FAST or mode is modes.Mode.QUICK", "mode is modes.Mode.FAST"
    ),
}


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def _run(root: Path) -> str:
    return subprocess.check_output([sys.executable, "main.py"], cwd=root, text=True)


def _findings(detector: type, root: Path, *, include_tests: bool = False):
    """The detector's findings on a scan that, like NRA's, leaves tests out."""

    with ScanCache.scope():
        return detector().detect(
            parse_python_modules(
                root,
                use_parse_cache=False,
                source_policy=PythonSourcePathPolicy(include_tests=include_tests),
            ),
            DetectorConfig(),
        )


def _snapshot(root: Path) -> CodemodSourceSnapshot:
    return CodemodSourceSnapshot.from_modules(parse_python_modules(root, use_parse_cache=False))


def _apply_every_recipe(detector: type, root: Path, expected_findings: int) -> None:
    assert len(_findings(detector, root)) == expected_findings
    for _ in range(expected_findings):
        finding = _findings(detector, root)[0]
        snapshot = _snapshot(root)
        evaluation = detector().evaluate_recipe_for_finding(finding, snapshot)
        assert evaluation.rejection_reason == ""
        simulation = CodemodPlanSequence.from_operations(
            evaluation.required_recipe.operations
        ).simulate(snapshot)
        assert simulation.is_clean
        simulation.apply()


def _target(root: Path, name: str) -> SourceRewriteTarget:
    path = next(path for path in root.rglob("*.py") if f"class {name}(" in path.read_text())
    return SourceRewriteTarget(file_path=path.as_posix(), qualname=name)


@pytest.mark.parametrize(
    "detector",
    (NameOnlySubclassDetector, UndistinguishedEnumMembersDetector),
)
def test_each_detector_states_its_fact_and_declares_its_operation(detector):
    spec = detector.finding_spec
    assert spec.certification.value == "certified"
    assert detector.restated.fact in spec.why
    assert detector.restated.operations


def test_patterns_continue_the_stable_sequence():
    assert [
        PatternId.IDENTICAL_ARMS,
        PatternId.UNDISTINGUISHED_TYPE_TEST,
        PatternId.NAME_ONLY_SUBCLASS,
        PatternId.UNDISTINGUISHED_ENUM_MEMBERS,
    ] == [45, 46, 47, 48]


@pytest.mark.parametrize("name", sorted(REDIRECTS))
def test_name_only_subclasses_redirect_across_modules(tmp_path: Path, name: str):
    files, rewritten, count = REDIRECTS[name]
    _write(tmp_path, files)
    expected = _run(tmp_path)
    _apply_every_recipe(NameOnlySubclassDetector, tmp_path, count)
    for module, text in files.items():
        assert (tmp_path / module).read_text() == rewritten.get(module, text), module
    assert _run(tmp_path) == expected
    assert _findings(NameOnlySubclassDetector, tmp_path) == []


def test_name_only_subclass_finding_names_its_siblings(tmp_path: Path):
    _write(tmp_path, REDIRECTS["name-and-module-imports"][0])
    summaries = sorted(finding.summary for finding in _findings(NameOnlySubclassDetector, tmp_path))
    assert summaries[0].startswith(
        "`LocalUser` differs from its base `Handler` only by name: every "
        "spelling of it constructs or raises it; its sibling `RemoteUser` differs the same way."
    )


@pytest.mark.parametrize(
    ("consumer", "reason"),
    (
        (
            "from pkg.users import LocalUser\n\n\ndef local(user):\n    return isinstance(user, LocalUser)\n",
            "pkg/probe.py:5 uses `LocalUser` as more than a constructor",
        ),
        (
            "from pkg.users import LocalUser\n\n\ndef local(action):\n    try:\n        action()\n"
            "    except LocalUser:\n        return 1\n",
            "pkg/probe.py:7 uses `LocalUser` as more than a constructor",
        ),
        (
            "from pkg.users import LocalUser\n\n\ndef local(user: LocalUser):\n    return user\n",
            "pkg/probe.py:4 uses `LocalUser` as more than a constructor",
        ),
        (
            "KIND = 'LocalUser'\n",
            "has a string name surface",
        ),
    ),
    ids=("isinstance", "except", "annotation", "string"),
)
def test_redirect_refuses_a_reference_that_tells_the_subclass_apart(
    tmp_path: Path, consumer: str, reason: str
):
    files = dict(REDIRECTS["name-and-module-imports"][0])
    files["pkg/probe.py"] = consumer
    _write(tmp_path, files)
    operation = RedirectEmptySubclassOperation(target=_target(tmp_path, "LocalUser"))
    with pytest.raises(ValueError) as refused:
        operation.source_edits(_snapshot(tmp_path))
    assert reason in str(refused.value).replace(tmp_path.as_posix() + "/", "")
    assert all(
        "LocalUser" not in finding.summary.split(" differs")[0]
        for finding in _findings(NameOnlySubclassDetector, tmp_path)
    )


@pytest.mark.parametrize(
    "source",
    (
        # The base stores every subclass it creates.
        "class Base:\n    registry = {}\n\n    def __init_subclass__(cls, **kw):\n"
        "        super().__init_subclass__(**kw)\n        Base.registry[cls.__name__] = cls\n\n\n"
        "class Child(Base):\n    pass\n\n\nChild()\n",
        # A registering metaclass.
        "class Meta(type):\n    seen = []\n\n    def __new__(mcs, name, bases, ns):\n"
        "        cls = super().__new__(mcs, name, bases, ns)\n        Meta.seen.append(cls)\n"
        "        return cls\n\n\nclass Base(metaclass=Meta):\n    pass\n\n\n"
        "class Child(Base):\n    pass\n\n\nChild()\n",
        # An abstract base cannot be constructed instead.
        "from abc import ABC, abstractmethod\n\n\nclass Base(ABC):\n    @abstractmethod\n"
        "    def go(self): ...\n\n\nclass Middle(Base):\n    def go(self):\n        return 1\n\n\n"
        "class Child(Middle):\n    pass\n\n\nclass Shape(Base):\n    pass\n",
        # A base from outside the scan.
        "import somelib\n\n\nclass Child(somelib.Base):\n    pass\n\n\nChild()\n",
        # Declares something.
        "class Base:\n    pass\n\n\nclass Child(Base):\n    size = 1\n\n\nChild()\n",
        # Decorated.
        "def mark(cls):\n    return cls\n\n\nclass Base:\n    pass\n\n\n@mark\nclass Child(Base):\n    pass\n",
    ),
    ids=("init-subclass", "metaclass", "abstract", "foreign", "declares", "decorated"),
)
def test_distinguished_subclasses_are_not_found(tmp_path: Path, source: str):
    _write(tmp_path, {"m.py": source})
    assert not any(
        finding.summary.startswith(("`Child`", "`Shape`"))
        for finding in _findings(NameOnlySubclassDetector, tmp_path)
    )


def test_enum_members_named_together_fold_across_modules(tmp_path: Path):
    _write(tmp_path, MODE_PACKAGE)
    expected = _run(tmp_path)
    (finding,) = _findings(UndistinguishedEnumMembersDetector, tmp_path)
    assert finding.summary.startswith(
        "`Mode` has members that every spelling names together: `Mode.FAST`, `Mode.QUICK`."
    )
    _apply_every_recipe(UndistinguishedEnumMembersDetector, tmp_path, 1)
    for module, text in MODE_PACKAGE.items():
        assert (tmp_path / module).read_text() == MODE_FOLDED.get(module, text), module
    assert _run(tmp_path) == expected
    assert _findings(UndistinguishedEnumMembersDetector, tmp_path) == []


@pytest.mark.parametrize(
    ("consumer", "reason"),
    (
        (
            "from pkg.modes import Mode\n\nQUICK_VALUE = Mode.QUICK.value\n",
            "pkg/probe.py:3 names `Mode.QUICK` outside a grouped test",
        ),
        (
            "from pkg.modes import Mode\n\nDECODED = Mode(2)\n",
            "pkg/probe.py:3 uses `Mode` other than by a member or as a type",
        ),
        (
            "from pkg.modes import Mode\n\nNAMES = [mode.name for mode in Mode]\n",
            "pkg/probe.py:3 uses `Mode` other than by a member or as a type",
        ),
        (
            "from pkg.modes import Mode\n\n\ndef quick(mode):\n    return mode is Mode.QUICK\n",
            "pkg/probe.py:5 names `Mode.QUICK` outside a grouped test",
        ),
        (
            "from pkg.modes import Mode\n\n\ndef quick(mode):\n"
            "    return mode in (Mode.QUICK, Mode.SLOW)\n",
            "pkg/probe.py:5 names `Mode.QUICK` without `FAST`",
        ),
    ),
    ids=("value-read", "decode-by-value", "iterate", "alone", "split-group"),
)
def test_fold_refuses_a_spelling_that_tells_members_apart(
    tmp_path: Path, consumer: str, reason: str
):
    files = dict(MODE_PACKAGE)
    files["pkg/probe.py"] = consumer
    _write(tmp_path, files)
    operation = FoldEnumMembersOperation(
        target=_target(tmp_path, "Mode"), members=("FAST", "QUICK")
    )
    with pytest.raises(ValueError) as refused:
        operation.source_edits(_snapshot(tmp_path))
    assert reason in str(refused.value).replace(tmp_path.as_posix() + "/", "")
    assert _findings(UndistinguishedEnumMembersDetector, tmp_path) == []


@pytest.mark.parametrize(
    "modes",
    (
        "from enum import StrEnum\n\n\nclass Mode(StrEnum):\n    FAST = 'f'\n    QUICK = 'q'\n",
        "from enum import Enum, auto\n\n\nclass Mode(Enum):\n    FAST = auto()\n    QUICK = auto()\n",
        "from enum import Enum\n\n\nclass Mode(Enum):\n    FAST = 1\n    QUICK = 2\n\n"
        "    def label(self):\n        return 'x'\n",
    ),
    ids=("str-enum", "auto-values", "methods"),
)
def test_enums_whose_members_can_differ_are_not_found(tmp_path: Path, modes: str):
    files = dict(MODE_PACKAGE)
    files["pkg/modes.py"] = modes
    files["main.py"] = "print('ok')\n"
    _write(tmp_path, files)
    assert _findings(UndistinguishedEnumMembersDetector, tmp_path) == []


def test_an_explicit_re_export_is_a_public_spelling(tmp_path: Path):
    files = dict(REDIRECTS["name-and-module-imports"][0])
    files["pkg/api.py"] = "from pkg.users import LocalUser as LocalUser\n"
    _write(tmp_path, files)
    assert [
        finding.summary.split(" differs")[0]
        for finding in _findings(NameOnlySubclassDetector, tmp_path)
    ] == ["`RemoteUser`"]


ERROR_PACKAGE = {
    "pkg/__init__.py": "",
    "pkg/errors.py": (
        "class LoadError(ValueError):\n    pass\n\n\ndef load():\n    raise LoadError('load')\n"
    ),
}
CATCHING_TESTS = {
    "pytest-raises": (
        "import pytest\n\nfrom pkg.errors import LoadError, load\n\n\n"
        "def test_load():\n    with pytest.raises(LoadError):\n        load()\n"
    ),
    "pytest-raises-tuple": (
        "import pytest\n\nfrom pkg.errors import LoadError, load\n\n\n"
        "def test_load():\n    with pytest.raises((LoadError, KeyError)):\n        load()\n"
    ),
    "assert-raises": (
        "import unittest\n\nfrom pkg.errors import LoadError, load\n\n\n"
        "class LoadTest(unittest.TestCase):\n    def test_load(self):\n"
        "        self.assertRaises(LoadError, load)\n"
    ),
    "except": (
        "from pkg.errors import LoadError, load\n\n\ndef test_load():\n    try:\n"
        "        load()\n    except LoadError:\n        pass\n"
    ),
}


@pytest.mark.parametrize("name", sorted(CATCHING_TESTS))
def test_an_exception_a_test_catches_is_distinguished(tmp_path: Path, name: str):
    _write(tmp_path, {**ERROR_PACKAGE, "tests/test_errors.py": CATCHING_TESTS[name]})
    assert _findings(NameOnlySubclassDetector, tmp_path) == []


def test_an_exception_nothing_catches_still_differs_only_by_name(tmp_path: Path):
    _write(tmp_path, ERROR_PACKAGE)
    (finding,) = _findings(NameOnlySubclassDetector, tmp_path)
    assert finding.summary.startswith("`LoadError` differs from its base `ValueError`")
    assert "tests" not in finding.evidence[0].file_path


def test_a_test_that_only_constructs_the_subclass_blocks_the_redirect(tmp_path: Path):
    files = dict(REDIRECTS["name-and-module-imports"][0])
    files["tests/test_users.py"] = (
        "from pkg.users import LocalUser\n\n\ndef test_local():\n    assert LocalUser().handle()\n"
    )
    _write(tmp_path, files)
    finding = next(
        finding
        for finding in _findings(NameOnlySubclassDetector, tmp_path)
        if finding.summary.startswith("`LocalUser`")
    )
    evaluation = NameOnlySubclassDetector().evaluate_recipe_for_finding(
        finding, _snapshot(tmp_path)
    )
    assert "tests/test_users.py:5 spells `LocalUser` in a test (construct)" in (
        evaluation.rejection_reason.replace(tmp_path.as_posix() + "/", "")
    )


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("try:\n    pass\nexcept Target:\n    pass\n", NameUseKind.CATCH),
        ("try:\n    pass\nexcept (Target, KeyError):\n    pass\n", NameUseKind.CATCH),
        ("pytest.raises(Target)\n", NameUseKind.CATCH),
        ("pytest.raises((KeyError, Target))\n", NameUseKind.CATCH),
        ("self.assertRaises(Target, f)\n", NameUseKind.CATCH),
        ("self.assertRaisesRegex(Target, 'x', f)\n", NameUseKind.CATCH),
        ("raise Target\n", NameUseKind.RAISE),
        ("Target()\n", NameUseKind.CONSTRUCT),
        ("pytest.raises(KeyError, Target)\n", NameUseKind.OTHER),
    ),
)
def test_name_use_kinds_wherever_they_appear(tmp_path: Path, source: str, kind: NameUseKind):
    (tmp_path / "m.py").write_text(source)
    (module,) = parse_python_modules(tmp_path, use_parse_cache=False)
    assert {use.kind for use in NameUse.in_module(module) if use.name == "Target"} == {kind}


def test_a_test_naming_a_member_alone_keeps_the_members_apart(tmp_path: Path):
    files = dict(MODE_PACKAGE)
    files["tests/test_budget.py"] = (
        "from pkg.budget import budget\nfrom pkg.modes import Mode\n\n\n"
        "def test_quick():\n    assert budget(Mode.QUICK) == 10\n"
    )
    _write(tmp_path, files)
    assert _findings(UndistinguishedEnumMembersDetector, tmp_path) == []
    operation = FoldEnumMembersOperation(
        target=_target(tmp_path, "Mode"), members=("FAST", "QUICK")
    )
    with pytest.raises(ValueError) as refused:
        operation.source_edits(_snapshot(tmp_path))
    assert "tests/test_budget.py:6 spells `Mode.QUICK` in a test" in str(refused.value).replace(
        tmp_path.as_posix() + "/", ""
    )


def test_a_class_declared_in_a_test_is_never_reported(tmp_path: Path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/base.py": "class Base:\n    pass\n",
            "tests/test_base.py": (
                "from pkg.base import Base\n\n\nclass FakeBase(Base):\n    pass\n\n\n"
                "def test_fake():\n    assert FakeBase()\n"
            ),
        },
    )
    assert _findings(NameOnlySubclassDetector, tmp_path, include_tests=True) == []
