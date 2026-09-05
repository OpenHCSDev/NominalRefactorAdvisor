"""Native lazy metadata must not invalidate implementation dependency traversal."""

import sys
from types import ModuleType

import pytest

from nominal_refactor_advisor.implementation_identity import implementation_module_names


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Native deferred annotations require Python 3.14"
)
@pytest.mark.parametrize(
    "annotation",
    (
        'Root.__annotations__["value"]',
        'tuple[Root.__annotations__["value"], ...]',
    ),
)
def test_deferred_annotations_can_materialise_class_metadata(annotation: str) -> None:
    module = ModuleType("deferred_annotation_fixture")
    source = (
        "class Payload: pass\n"
        "Payload.__module__ = 'external_payload'\n"
        "class Root:\n"
        "    value: Payload\n"
        f"    def action(self, value: {annotation}): return value\n"
    )
    exec(
        compile(source, "<native deferred annotations>", "exec", dont_inherit=True),
        vars(module),
    )
    root = module.Root
    before = tuple(vars(root))
    dependencies = implementation_module_names((root,))
    assert "external_payload" in dependencies
    assert "deferred_annotation_fixture" in dependencies
    assert before != tuple(vars(root))
    assert implementation_module_names((root,)) == dependencies
