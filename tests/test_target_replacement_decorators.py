from __future__ import annotations

from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodSourceSnapshot,
    RefactorRecipe,
    RefactorRecipeOperation,
    ReplaceDeclarationDecoratorsOperation,
    ReplaceTargetOperation,
    SourceNodeDecoratorPolicy,
    SourceRewriteTarget,
)


@pytest.mark.parametrize(
    "header", ("class Value:", "def Value():", "async def Value():")
)
@pytest.mark.parametrize("existing_decorator", ("", "@decorate\n"))
def test_body_only_replacement_rejects_decorator_payload_before_writing(
    tmp_path: Path, header: str, existing_decorator: str
) -> None:
    path = tmp_path / "sample.py"
    source = f"{existing_decorator}{header}\n    pass\n"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceTargetOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Value"),
        replacement_source=f"@decorate\n{header}\n    pass\n",
    )

    with pytest.raises(
        CodemodOperationPreflightError, match="decorator-inclusive source region"
    ):
        RefactorRecipe("replace-value").with_operation(operation).simulate(snapshot)
    assert path.read_text() == source


def test_body_only_replacement_preserves_the_existing_decorator_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.py"
    source = "from dataclasses import dataclass\n@(\n    dataclass(frozen=True)\n)\nclass Value:\n    old: int\n"
    path.write_text(source)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    operation = ReplaceTargetOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Value"),
        replacement_source="class Value:\n    new: int\n",
    )

    result = (
        RefactorRecipe("replace-value").with_operation(operation).simulate(snapshot)
    )
    assert result.is_clean
    result.apply()
    rewritten = path.read_text()
    assert rewritten.startswith(source[: source.index("class Value:")])
    namespace = {}
    exec(rewritten, namespace)
    assert namespace["Value"](new=7).new == 7


def test_nominal_policy_controls_both_payload_and_complete_decorator_geometry(
    tmp_path: Path,
) -> None:
    class DecoratorInclusivePolicy:
        decorator_policy = SourceNodeDecoratorPolicy.INCLUDE

    class DecoratorInclusiveProbeOperation(
        DecoratorInclusivePolicy, ReplaceTargetOperation
    ):
        pass

    try:
        path = tmp_path / "sample.py"
        path.write_text(
            "calls = []\n"
            "def old(value):\n    calls.append('old')\n    return value\n"
            "def new(value):\n    calls.append('new')\n    return value\n"
            "@(\n    old\n)\nclass Value:\n    number = 1\n"
        )
        snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
        operation = DecoratorInclusiveProbeOperation(
            target=SourceRewriteTarget(file_path=str(path), qualname="Value"),
            replacement_source="@new\nclass Value:\n    number = 2\n",
        )
        result = (
            RefactorRecipe("replace-value").with_operation(operation).simulate(snapshot)
        )
        assert result.is_clean
        result.apply()
        namespace = {}
        exec(path.read_text(), namespace)
        assert namespace["calls"] == ["new"]
        assert namespace["Value"].number == 2
    finally:
        del RefactorRecipeOperation.__registry__[
            DecoratorInclusiveProbeOperation.operation_key()
        ]


def test_body_and_decorator_edits_compose_on_the_same_source_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.py"
    path.write_text(
        "from dataclasses import dataclass\n"
        "@(\n    dataclass(frozen=True)\n)\nclass Value:\n    old: int\n"
    )
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(tmp_path))
    target = SourceRewriteTarget(file_path=str(path), qualname="Value")
    recipe = (
        RefactorRecipe("replace-value")
        .with_operation(
            ReplaceTargetOperation(
                target=target, replacement_source="class Value:\n    new: int\n"
            )
        )
        .with_operation(
            ReplaceDeclarationDecoratorsOperation(
                target=target, decorators_source="@dataclass(frozen=False)\n"
            )
        )
    )
    result = recipe.simulate(snapshot)
    assert result.is_clean
    result.apply()
    namespace = {}
    exec(path.read_text(), namespace)
    value = namespace["Value"](new=7)
    value.new = 9
    assert value.new == 9
