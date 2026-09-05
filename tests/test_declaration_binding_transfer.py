"""Declaration environment changes retain module authority and evaluation phase."""

import ast
from pathlib import Path
import sys
from typing import get_type_hints

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.declaration_binding_transfer import (
    DeclarationModuleBindingEnvironment,
    DeclarationModuleBindingTransfer,
)
from nominal_refactor_advisor.declaration_dependencies import (
    DeclarationDependencyUse,
    ModuleLexicalDependencyProjection,
)


def _environment(name: str, source: str) -> DeclarationModuleBindingEnvironment:
    module = ast.parse(source)
    return DeclarationModuleBindingEnvironment(
        ParsedModule(
            path=Path(f"pkg/{name}.py"),
            module_name=f"pkg.{name}",
            is_package_init=False,
            module=module,
            source=source,
        ),
        next(node for node in module.body if isinstance(node, ast.ClassDef)),
    )


@pytest.mark.parametrize(
    "source_header, destination_header, expression, preserved",
    (
        ("import math", "import math", "math.sqrt(4)", True),
        (
            "from math import sqrt as scale",
            "from math import sqrt as scale",
            "scale(4)",
            True,
        ),
        (
            "from math import sqrt as scale",
            "from math import ceil as scale",
            "scale(4)",
            False,
        ),
        ("label = 3", "label = 3", "label", False),
        ("label = 3", "label = 4", "label", False),
        ("from pkg.destination import label", "label = 3", "label", True),
        ("", "str = lambda value: 'changed'", "str(4)", False),
        ("str = lambda value: 'source'", "", "str(4)", False),
        (
            "from math import sqrt as scale",
            "from math import sqrt as scale\nscale = lambda x: 7",
            "scale(4)",
            False,
        ),
        (
            "from math import sqrt as scale\ndel scale",
            "from math import sqrt as scale",
            "scale(4)",
            False,
        ),
        ("from unknown import *", "", "str(4)", False),
        ("", "", "sum(item for item in range(3))", True),
        ("", "", "{item: item + 1 for item in range(3)}", True),
    ),
)
def test_transfer_requires_binding_identity_not_equal_names_or_values(
    source_header: str, destination_header: str, expression: str, preserved: bool
) -> None:
    source = _environment(
        "source",
        source_header + f"\nclass Source:\n def moved(self):\n  return {expression}\n",
    )
    destination = _environment(
        "destination", destination_header + "\nclass Destination: pass\n"
    )
    transfer = DeclarationModuleBindingTransfer(source, destination)
    method = source.scope.body[0]
    if preserved:
        transfer.require_preserved(method)
        transfer.require_preserved(method)
        assert source.snapshots is source.snapshots
        assert source.binding_authority is source.binding_authority
    else:
        with pytest.raises(ValueError, match="binding authority"):
            transfer.require_preserved(method)


def test_same_module_runtime_global_keeps_its_owned_cell() -> None:
    source = _environment(
        "source",
        "class Source:\n def moved(self): return label\nlabel = object()\nclass Destination: pass\n",
    )
    destination = DeclarationModuleBindingEnvironment(
        source.module, source.module.module.body[-1]
    )
    DeclarationModuleBindingTransfer(source, destination).require_preserved(
        source.scope.body[0]
    )


def test_reference_views_derive_from_one_collection_without_fabricated_source_spans() -> (
    None
):
    module = ast.parse(
        "def moved(value: 'Input') -> list['Output']:\n return convert(value)\n"
    )
    projection = ModuleLexicalDependencyProjection.from_module(module)
    original_nodes = frozenset(ast.walk(module))
    assert {surface.reference.id for surface in projection.name_surfaces} == {
        "Input",
        "Output",
        "list",
        "convert",
    }
    assert all(
        surface.reference in original_nodes
        for surface in projection.direct_name_surfaces
    )
    assert projection.direct_name_surfaces is projection.direct_name_surfaces
    assert projection.names_for_use(DeclarationDependencyUse.EXECUTION) == {"convert"}
    assert projection.names_for_use(DeclarationDependencyUse.EVALUATED_ANNOTATION) == {
        "list"
    }
    assert projection.names_for_use(DeclarationDependencyUse.DEFERRED_ANNOTATION) == {
        "Input",
        "Output",
    }


@pytest.mark.parametrize("annotation", ("Result", "'Result'"))
@pytest.mark.parametrize("postponed", (False, True))
def test_annotations_resolve_at_their_actual_evaluation_phase(
    annotation: str, postponed: bool
) -> None:
    prelude = "from __future__ import annotations\n" if postponed else ""
    source = _environment(
        "source",
        prelude
        + "from builtins import str as Result\nclass Source:\n def moved(self) -> "
        + annotation
        + ": return 'result'\n",
    )
    destination = _environment(
        "destination",
        prelude + "class Destination: pass\nfrom builtins import str as Result\n",
    )
    transfer = DeclarationModuleBindingTransfer(source, destination)
    eager = not postponed and annotation == "Result" and sys.version_info < (3, 14)
    if eager:
        with pytest.raises(ValueError, match="'Result'.*binding authority"):
            transfer.require_preserved(source.scope.body[0])
    else:
        transfer.require_preserved(source.scope.body[0])


def test_annotation_representation_change_is_not_silently_accepted() -> None:
    source = _environment(
        "source",
        "from __future__ import annotations\nclass Source:\n def moved(self) -> None: pass\n",
    )
    destination = _environment("destination", "class Destination: pass\n")
    with pytest.raises(ValueError, match="annotation evaluation mode changes"):
        DeclarationModuleBindingTransfer(source, destination).require_preserved(
            source.scope.body[0]
        )


def test_same_qualified_name_does_not_prove_redefined_declaration_identity() -> None:
    source = _environment(
        "source",
        "def Result(): return 'first'\nclass Source:\n def moved(self) -> Result: pass\ndef Result(): return 'second'\n",
    )
    destination = _environment(
        "destination", "from pkg.source import Result\nclass Destination: pass\n"
    )
    with pytest.raises(ValueError, match="rebound declaration"):
        DeclarationModuleBindingTransfer(source, destination).require_preserved(
            source.scope.body[0]
        )


@pytest.mark.parametrize("annotation", ("'Result'", "list['Result']"))
def test_quoted_annotation_dependencies_keep_their_native_type_authority(
    annotation: str,
) -> None:
    source = _environment(
        "source",
        f"from builtins import str as Result\nclass Source:\n def moved(self) -> {annotation}: pass\n",
    )
    destination = _environment(
        "destination",
        f"from builtins import int as Result\nclass Destination:\n def moved(self) -> {annotation}: pass\n",
    )
    source_namespace = {}
    destination_namespace = {}
    exec(
        compile(source.module.source, source.module.file_path, "exec"), source_namespace
    )
    exec(
        compile(destination.module.source, destination.module.file_path, "exec"),
        destination_namespace,
    )
    assert get_type_hints(source_namespace["Source"].moved) != get_type_hints(
        destination_namespace["Destination"].moved
    )
    with pytest.raises(ValueError, match="'Result'.*binding authority"):
        DeclarationModuleBindingTransfer(source, destination).require_preserved(
            source.scope.body[0]
        )


def test_batched_snapshots_retain_native_declaration_and_final_bindings() -> None:
    environment = _environment(
        "source",
        "from math import floor as operation\nclass Source:\n captured = operation\nfrom math import ceil as operation\n",
    )
    snapshots = ModuleNominalBindingAuthority(environment.module).snapshots_before(
        (2, None, 2)
    )
    namespace = {}
    exec(
        compile(environment.module.source, environment.module.file_path, "exec"),
        namespace,
    )
    assert set(snapshots) == {2, None}
    assert (
        snapshots[2].binding_for("operation").qualified_name
        == f"math.{namespace['Source'].captured.__name__}"
    )
    assert (
        snapshots[None].binding_for("operation").qualified_name
        == f"math.{namespace['operation'].__name__}"
    )
