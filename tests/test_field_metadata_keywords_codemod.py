"""Selected field keyword edits preserve remaining declarations and refuse open cases."""

from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodOperationPreflightError,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PruneFieldMetadataKeywordsOperation,
    RefactorRecipeOperation,
    RemoveFieldMetadataKeywordsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_source_edits import CodemodSourceRevisionError
from nominal_refactor_advisor.json_reports import json_report_object

PREFIX = """from typing import Annotated
class Column:
    def __init__(self, check='', primary_key=False):
        self.check = check
        self.primary_key = primary_key
class FrozenColumn(Column):
    immutable = True
"""


def operation(
    path,
    *,
    field="identity",
    callee="Column",
    operation_type=PruneFieldMetadataKeywordsOperation,
):
    return operation_type(
        target=SourceRewriteTarget(file_path=str(path), qualname="Row"),
        callee=SourceRewriteTarget(file_path=str(path), qualname=callee),
        field_name=field,
        keyword_names=("check",),
    )


def simulate(path, source, *operations):
    snapshot = CodemodSourceSnapshot.from_source_mapping({str(path): source})
    return CodemodPlanSequence.from_operations(
        operations or (operation(path),)
    ).simulate(snapshot)


def test_real_annotation_consumers_keep_frozen_and_other_metadata(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = PREFIX + """class Row:
    identity: Annotated[str, Column(check="length(identity)=32")]
    frozen: Annotated[str, FrozenColumn(check="length(frozen)=32")]
    keyed: Annotated[str, "marker", Column(primary_key=True, check="length(keyed)=32")]
    extra: Annotated[str, "preserved", Column(check="length(extra)=32")]
    untouched: Annotated[str, Column(check="independent")]
"""
    operations = (
        operation(path),
        operation(
            path,
            field="frozen",
            callee="FrozenColumn",
            operation_type=RemoveFieldMetadataKeywordsOperation,
        ),
        operation(path, field="keyed"),
        operation(path, field="extra"),
    )
    result = simulate(path, source, *operations)
    assert result.is_clean
    rewritten = result.simulation.rewritten_sources[str(path)]
    assert "identity: str" in rewritten
    assert "frozen: Annotated[str, FrozenColumn()]" in rewritten
    assert 'keyed: Annotated[str, "marker", Column(primary_key=True)]' in rewritten
    assert 'extra: Annotated[str, "preserved"]' in rewritten
    assert 'untouched: Annotated[str, Column(check="independent")]' in rewritten
    namespace = {}
    exec(compile(rewritten, str(path), "exec"), namespace)
    annotations = get_type_hints(
        namespace["Row"], globalns=namespace, include_extras=True
    )
    assert annotations["identity"] is str
    assert get_args(annotations["frozen"])[1].immutable
    assert get_args(annotations["keyed"])[2].primary_key
    assert get_args(annotations["extra"]) == (str, "preserved")


@pytest.mark.parametrize(
    "call",
    (
        'Column(check="x", primary_key=True)',
        'Column(primary_key=True, check="x")',
        'Column(primary_key=True, check="x",)',
        'Column(\n        check="x",\n        primary_key=True,\n    )',
    ),
)
def test_surviving_keyword_source_is_retained(tmp_path: Path, call):
    path = tmp_path / "fields.py"
    source = PREFIX + f"class Row:\n    identity: Annotated[str, {call}]\n"
    rewritten = simulate(path, source).simulation.rewritten_sources[str(path)]
    assert "check=" not in rewritten.removeprefix(PREFIX)
    assert "primary_key=True" in rewritten
    compile(rewritten, str(path), "exec")


def test_unicode_and_unselected_default_call_are_preserved(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = (
        PREFIX
        + 'class Row:\n    identité: Annotated[str, Column(check="é")] = Column(check="default")\n'
    )
    rewritten = simulate(
        path, source, operation(path, field="identité")
    ).simulation.rewritten_sources[str(path)]
    assert 'identité: str = Column(check="default")' in rewritten


def test_generated_field_initializer_uses_the_same_keyword_edit(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = PREFIX + """from typing import ClassVar
class Generated:
    def __init__(self, annotation, column):
        self.column = column
class Row:
    identity: ClassVar[Generated] = Generated(str, Column(check="x", primary_key=True))
"""
    selected = operation(path, operation_type=RemoveFieldMetadataKeywordsOperation)
    rewritten = simulate(path, source, selected).simulation.rewritten_sources[str(path)]
    assert "Generated(str, Column(primary_key=True))" in rewritten
    namespace = {}
    exec(compile(rewritten, str(path), "exec"), namespace)
    assert namespace["Row"].identity.column.primary_key


def test_nested_executable_field_call_remains_open(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = (
        PREFIX + 'class Row:\n    identity: object = lambda Column: Column(check="x")\n'
    )
    selected = operation(path, operation_type=RemoveFieldMetadataKeywordsOperation)
    with pytest.raises(
        CodemodOperationPreflightError, match="Expected one metadata call"
    ):
        simulate(path, source, selected)


@pytest.mark.parametrize(
    "suffix,match",
    (
        (
            "class Row:\n    identity: Annotated[str, Column(check=make_check())]\n",
            "malformed node|literal",
        ),
        (
            'class Row:\n    identity: Annotated[str, Column(check="x", **options)]\n',
            "Expanded keyword",
        ),
        (
            'class Row:\n    identity: Annotated[str, Column(check="x"), Column(check="y")]\n',
            "Expected one metadata call",
        ),
        (
            "class Row:\n    identity: Annotated[str, Column(primary_key=True)]\n",
            "exactly once",
        ),
        (
            'class Row:\n    identity: Annotated[str, Column(check="x")]\n    identity: str\n',
            "ambiguous",
        ),
        (
            'class Row:\n    Column = object\n    identity: Annotated[str, Column(check="x")]\n',
            "Expected one metadata call",
        ),
        (
            'class Row:\n    identity: Annotated[str, Column(check="x")]\nColumn = object\n',
            "Expected one metadata call",
        ),
        (
            'class Row:\n    identity: Annotated[str, Column(\n        check="x", # retained explanation\n    )]\n',
            "comment",
        ),
        (
            'class Annotated: pass\nclass Row:\n    identity: Annotated[str, Column(check="x")]\n',
            "original typing.Annotated",
        ),
    ),
)
def test_open_or_unpreservable_sites_refuse(tmp_path: Path, suffix, match):
    path = tmp_path / "fields.py"
    with pytest.raises(CodemodOperationPreflightError, match=match):
        simulate(path, PREFIX + suffix)


def test_json_round_trip_and_source_change_rejection(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = PREFIX + 'class Row:\n    identity: Annotated[str, Column(check="x")]\n'
    path.write_text(source)
    selected = operation(path)
    assert (
        RefactorRecipeOperation.from_json_value(json_report_object(selected))
        == selected
    )
    result = simulate(path, source, selected)
    path.write_text(source + "# newer source\n")
    with pytest.raises(CodemodSourceRevisionError):
        result.apply()
    assert path.read_text().endswith("# newer source\n")


def test_nested_class_cannot_mistake_an_enclosing_parameter_for_the_callee(
    tmp_path: Path,
):
    path = tmp_path / "fields.py"
    source = PREFIX + """def build(Column):
    class Row:
        identity: Annotated[str, Column(check="x")]
    return Row
"""
    selected = PruneFieldMetadataKeywordsOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="build.Row"),
        callee=SourceRewriteTarget(file_path=str(path), qualname="Column"),
        field_name="identity",
        keyword_names=("check",),
    )
    with pytest.raises(CodemodOperationPreflightError, match="module-level class"):
        simulate(path, source, selected)


def test_multiple_keyword_runs_preserve_unselected_metadata(tmp_path: Path):
    path = tmp_path / "fields.py"
    source = (
        PREFIX
        + 'class Row:\n    identity: Annotated[str, Column(check="x", primary_key=True, nullable=False)]\n'
    )
    selected = PruneFieldMetadataKeywordsOperation(
        target=SourceRewriteTarget(file_path=str(path), qualname="Row"),
        callee=SourceRewriteTarget(file_path=str(path), qualname="Column"),
        field_name="identity",
        keyword_names=("check", "nullable"),
    )
    rewritten = simulate(path, source, selected).simulation.rewritten_sources[str(path)]
    assert "identity: Annotated[str, Column(primary_key=True)]" in rewritten


def test_cross_module_alias_and_changed_callee_source(tmp_path: Path):
    declaration = tmp_path / "columns.py"
    caller = tmp_path / "rows.py"
    sources = {
        str(declaration): "class Column: pass\n",
        str(
            caller
        ): 'from typing import Annotated as A\nfrom columns import Column as Metadata\nclass Row:\n    identity: A[str, Metadata(check="x")]\n',
    }
    for name, source in sources.items():
        Path(name).write_text(source)
    selected = PruneFieldMetadataKeywordsOperation(
        target=SourceRewriteTarget(file_path=str(caller), qualname="Row"),
        callee=SourceRewriteTarget(file_path=str(declaration), qualname="Column"),
        field_name="identity",
        keyword_names=("check",),
    )
    result = CodemodPlanSequence.from_operations((selected,)).simulate(
        CodemodSourceSnapshot.from_source_mapping(sources)
    )
    assert "identity: str" in result.simulation.rewritten_sources[str(caller)]
    declaration.write_text("class Column:\n    changed = True\n")
    with pytest.raises(CodemodSourceRevisionError):
        result.apply()
    assert caller.read_text() == sources[str(caller)]
