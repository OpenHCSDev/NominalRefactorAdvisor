"""Decorator edits preserve the selected declaration's header, suite and literals."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    ExtractSymbolsToNewModuleOperation,
    ReplaceFunctionDecoratorsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object


def _plan(
    path: Path, source: str, qualname: str = "Owner.value"
) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionDecoratorsOperation(
                target=SourceRewriteTarget(
                    file_path=path.as_posix(), qualname=qualname
                ),
                decorators_source=source,
            ),
        )
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_cli_changes_property_to_cached_property_without_rewriting_its_body(
    tmp_path: Path, newline: str
) -> None:
    path = tmp_path / "probe.py"
    suffix = (
        "    def value(self) -> str: # retain header\n"
        "        self.calls += 1\n"
        "        return 'café' # retain body\n"
        "instance = Owner()\n"
        "print(ascii(instance.value), ascii(instance.value))\n"
        "print(instance.calls)\n"
    ).replace("\n", newline)
    source = (
        "from functools import cached_property\n"
        "class Owner:\n"
        "    calls = 0\n"
        "    @property\n"
    ).replace("\n", newline) + suffix
    path.write_text(source, encoding="utf-8", newline="")
    before = subprocess.check_output([sys.executable, str(path)], text=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(path),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(_plan(path, "@cached_property"))),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["applied"]
    after = subprocess.check_output([sys.executable, str(path)], text=True)
    assert before.splitlines()[0] == after.splitlines()[0]
    assert before.splitlines()[1] == "2"
    assert after.splitlines()[1] == "1"
    assert (
        path.read_bytes().decode().endswith("    @cached_property" + newline + suffix)
    )


@pytest.mark.parametrize(
    "prefix",
    (
        "",
        "    @first\n    @second\n",
        "    @(\n        first\n    )\n",
    ),
)
@pytest.mark.parametrize("replacement", ("", "@third", "@outer\n@inner"))
def test_add_replace_and_remove_decorators_preserve_async_eof(
    tmp_path: Path, prefix: str, replacement: str
) -> None:
    path = tmp_path / "probe.py"
    suffix = "    async def value(self): return 1"
    path.write_text("class Owner:\n" + prefix + suffix, encoding="utf-8", newline="")
    simulation = _plan(path, replacement).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    assert simulation.is_clean
    simulation.apply()
    expected = "".join("    " + line + "\n" for line in replacement.splitlines())
    assert path.read_bytes().decode() == "class Owner:\n" + expected + suffix


def test_multiline_decorator_preserves_literal_indentation_and_runtime_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.py"
    source = (
        "events = []\n"
        "def mark(value):\n"
        "    def decorate(fn):\n"
        "        events.append(value)\n"
        "        return fn\n"
        "    return decorate\n"
        "class Owner:\n"
        "    @mark('old')\n"
        "    def value(self): return 3\n"
        "print(events, Owner().value())\n"
    )
    path.write_text(source, encoding="utf-8", newline="")
    replacement = "@mark('''a\n  b\n''')\n@mark(\n    'inner'\n)"
    simulation = _plan(path, replacement).simulate(
        CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    )
    simulation.apply()
    assert (
        subprocess.check_output([sys.executable, str(path)], text=True).strip()
        == "['inner', 'a\\n  b\\n'] 3"
    )
    node = ast.parse(path.read_text(encoding="utf-8")).body[2].body[0]
    assert len(node.decorator_list) == 2


@pytest.mark.parametrize(
    "prefix",
    (
        "    @property # keep\n",
        "    @wrapper(\n        # keep\n        option\n    )\n",
        "    @property\n    # keep\n",
    ),
)
def test_existing_decorator_comments_are_not_discarded(
    tmp_path: Path, prefix: str
) -> None:
    path = tmp_path / "probe.py"
    source = "class Owner:\n" + prefix + "    def value(self): return 1\n"
    path.write_text(source, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="discard comments"):
        _plan(path, "@cached_property").simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize(
    "replacement", ("value = 1", "def extra(): pass", "@", "@wrapper(1")
)
def test_decorator_payload_rejects_statements_and_invalid_python(
    tmp_path: Path, replacement: str
) -> None:
    path = tmp_path / "probe.py"
    source = "class Owner:\n    def value(self): return 1\n"
    path.write_text(source, encoding="utf-8", newline="")
    with pytest.raises((ValueError, SyntaxError)):
        _plan(path, replacement).simulate(
            CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        )
    assert path.read_bytes() == source.encode()


def test_module_extraction_retains_parenthesized_decorators(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    path = package / "source.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "@(\n    dataclass\n)\n"
        "class Item:\n    value: int\n",
        encoding="utf-8",
        newline="",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from pkg.source import Item\nprint(Item(3).value)\n", encoding="utf-8"
    )
    before = subprocess.check_output([sys.executable, str(probe)], text=True)
    destination = package / "item.py"
    plan = CodemodPlanSequence.from_operations(
        (
            ExtractSymbolsToNewModuleOperation(
                target=SourceRewriteTarget(file_path=path.as_posix()),
                symbol_qualnames=("Item",),
                destination_path=destination.as_posix(),
            ),
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nominal_refactor_advisor",
            str(package),
            "--codemod-plan",
            "-",
            "--codemod-apply",
            "--json",
        ],
        input=json.dumps(json_report_object(plan)),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "@(\n    dataclass\n)\nclass Item" in destination.read_text(encoding="utf-8")
    assert subprocess.check_output([sys.executable, str(probe)], text=True) == before
