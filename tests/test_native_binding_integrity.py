"""Native proof must retain object capture separately from later slot lookup."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PromoteClassMembersToAncestorOperation,
    SourceRewriteTarget,
)


def _module(
    header: str, decorator: str, *, authority_header: str = "class Authority:"
) -> ParsedModule:
    source = (
        "import builtins\n"
        "events = []\n"
        "class Replacement:\n"
        "    def __init__(self, function): pass\n"
        "    def __set_name__(self, owner, name):\n"
        "        events.append('shared' in vars(owner))\n"
        f"{header}\n"
        f"{authority_header}\n"
        f"    @{decorator}\n"
        "    def cached(self): return 1\n"
        "class Leaf(Authority):\n"
        "    def shared(self): return 2\n"
        "print(events)\n"
    )
    return ParsedModule(
        path=Path("descriptor_probe.py"),
        module_name="descriptor_probe",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


def _promotion(module: ParsedModule) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            PromoteClassMembersToAncestorOperation(
                target=SourceRewriteTarget(file_path=module.file_path, qualname="Leaf"),
                destination=SourceRewriteTarget(
                    file_path=module.file_path, qualname="Authority"
                ),
                member_names=("shared",),
            ),
        )
    )


def _output(source: str) -> str:
    # Each program mutates its own builtins, never those of the analyzer or pytest.
    return subprocess.check_output([sys.executable, "-c", source], text=True).strip()


@pytest.mark.parametrize(
    "header, decorator",
    (
        ("builtins.property = Replacement", "builtins.property"),
        ("builtins.property = Replacement", "property"),
        (
            "builtins.property = Replacement\n"
            "from builtins import property as captured",
            "captured",
        ),
        (
            "alias = builtins\nalias.property = Replacement",
            "builtins.property",
        ),
        ("alias = builtins\nalias.property = Replacement", "alias.property"),
        ("if True:\n    builtins.property = Replacement", "builtins.property"),
        ("class Setup:\n    builtins.property = Replacement", "builtins.property"),
        (
            "def install():\n    builtins.property = Replacement\ninstall()",
            "builtins.property",
        ),
        (
            "def namespace(): return builtins\nnamespace().property = Replacement",
            "builtins.property",
        ),
    ),
    ids=(
        "qualified-read",
        "builtin-read",
        "import-after-write",
        "write-through-alias",
        "read-through-alias",
        "conditional-write",
        "class-body-write",
        "called-function-write",
        "computed-receiver-write",
    ),
)
def test_changed_native_lookup_does_not_authorize_promotion(
    header: str, decorator: str
) -> None:
    module = _module(header, decorator)
    assert _output(module.source) == "[False]"
    with pytest.raises(ValueError, match="(?i)unproved|binding|namespace|native"):
        _promotion(module).simulate(CodemodSourceSnapshot.from_modules((module,)))


@pytest.mark.parametrize(
    "header, decorator",
    (
        ("", "builtins.property"),
        (
            "saved = builtins.property\nbuiltins.property = Replacement",
            "saved",
        ),
        (
            "from builtins import property as saved\n"
            "builtins.property = Replacement",
            "saved",
        ),
        ("builtins.unrelated = Replacement", "builtins.property"),
    ),
    ids=("unchanged", "capture-before-write", "import-before-write", "other-slot"),
)
def test_unchanged_captured_native_object_remains_usable(
    header: str, decorator: str
) -> None:
    module = _module(header, decorator)
    before = _output(module.source)
    assert before == "[]"
    result = _promotion(module).simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert result.is_clean
    assert (
        _output(result.final_snapshot.sources_by_file_path[module.file_path]) == before
    )


@pytest.mark.parametrize(
    "authority_header",
    (
        pytest.param("class Authority:", id="plain-body-frame"),
        pytest.param(
            "class Authority[T]:",
            id="generic-body-frame",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12),
                reason="Type-parameter syntax requires Python 3.12 or newer",
            ),
        ),
    ),
)
def test_rebound_builtin_dictionary_does_not_authenticate_class_body_property(
    authority_header: str,
) -> None:
    module = _module(
        "__builtins__ = dict(vars(builtins), property=Replacement)",
        "property",
        authority_header=authority_header,
    )
    # Both ordinary and generic class-body functions capture the newly named
    # dictionary. This differs from an ordinary class's builder lookup in the
    # parent frame, which still uses that frame's earlier captured builtins.
    assert _output(module.source) == "[False]"
    with pytest.raises(ValueError, match="(?i)unproved|binding|namespace|native"):
        _promotion(module).simulate(CodemodSourceSnapshot.from_modules((module,)))


@pytest.mark.parametrize(
    "authority_header",
    (
        pytest.param("class Authority:", id="plain-body-frame"),
        pytest.param(
            "class Authority[T]:",
            id="generic-body-frame",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12),
                reason="Type-parameter syntax requires Python 3.12 or newer",
            ),
        ),
    ),
)
def test_rebound_builtin_dictionary_preserves_qualified_native_module_lookup(
    authority_header: str,
) -> None:
    module = _module(
        "__builtins__ = dict(vars(builtins), property=Replacement)",
        "builtins.property",
        authority_header=authority_header,
    )
    before = _output(module.source)
    assert before == "[]"
    result = _promotion(module).simulate(CodemodSourceSnapshot.from_modules((module,)))
    assert result.is_clean
    assert (
        _output(result.final_snapshot.sources_by_file_path[module.file_path]) == before
    )


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Type-parameter syntax requires Python 3.12+"
)
def test_generic_implicit_base_hook_mutation_does_not_authorize_promotion():
    module = _module(
        "import typing\n"
        "original_entries = typing._GenericAlias.__mro_entries__\n"
        "def changed_entries(self, bases):\n"
        "    builtins.property = Replacement\n"
        "    return original_entries(self, bases)\n"
        "typing._GenericAlias.__mro_entries__ = changed_entries",
        "builtins.property",
        authority_header="class Authority[T]:",
    )
    # The native generic alias calls this protocol even though the source class
    # has no explicit base expressions. Only the isolated program is mutated.
    assert _output(module.source) == "[False]"
    with pytest.raises(ValueError, match="(?i)unproved|binding|namespace|native"):
        _promotion(module).simulate(CodemodSourceSnapshot.from_modules((module,)))


def test_native_class_body_builtins_are_captured_before_base_expression():
    module = _module(
        "def base():\n"
        "    global __builtins__\n"
        "    __builtins__ = dict(vars(builtins), property=Replacement)\n"
        "    return object",
        "property",
        authority_header="class Authority(base()):",
    )
    source = module.source.replace(
        "print(events)\n",
        "class Later:\n"
        "    @property\n"
        "    def cached(self): return 3\n"
        "print(type(vars(Authority)['cached']) is builtins.property, "
        "isinstance(vars(Later)['cached'], Replacement), events)\n",
    )
    # Native capture timing does not assert that promotion already admits
    # computed bases or their effects.
    assert _output(source) == "True True [False]"
