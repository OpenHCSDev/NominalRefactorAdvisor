"""Declaration consumers retain rich proof facts without unrelated analyses."""

from pathlib import Path

import pytest

from nominal_refactor_advisor import class_index
from nominal_refactor_advisor.ast_tools import SourceModule
from nominal_refactor_advisor.class_index import (
    CompactClassFamilyIndex,
    CompactModuleClassHeader,
    CompactModuleClassProjectionFamily,
    build_compact_class_family_index,
)
from nominal_refactor_advisor.product_flow_authority import SourceProductFlowRepository
from nominal_refactor_advisor.scan_cache import ScanCache


def parsed(name, source):
    return SourceModule(Path(name + ".py"), name, source).parse()


@pytest.mark.parametrize("star_exports", ("__all__ = ('Foreign',)\n", ""))
def test_declaration_and_rich_consumers_keep_the_same_inheritance_and_proof_answers(
    star_exports,
):
    modules = (
        parsed("foreign", star_exports + "class Foreign: pass\n"),
        parsed(
            "records",
            "from foreign import *\n"
            "from dataclasses import dataclass as record\n"
            "@record(frozen=True)\n"
            "class Pair:\n    left: int\n    right: int\n"
            "    def project(self): return self.left, self.right\n"
            "class Child(Pair): pass\n"
            "class Open(factory()): pass\n",
        ),
    )
    with ScanCache.scope():
        declarations = CompactModuleClassHeader.from_modules(modules)
        rich = CompactModuleClassProjectionFamily.collect_modules(modules)
        declaration_index = build_compact_class_family_index(declarations)
        rich_index = build_compact_class_family_index(rich)

        assert declaration_index.classes_by_symbol == rich_index.classes_by_symbol
        assert (
            declaration_index.product_authority_resolutions_by_symbol
            == rich_index.product_authority_resolutions_by_symbol
        )
        assert declaration_index.mro_authority.resolve("records.Open").mro_type is None
        assert declaration_index.descendant_symbols("records.Pair") == (
            "records.Child",
        )
        assert CompactClassFamilyIndex.from_modules(modules).classes_by_symbol == (
            rich_index.classes_by_symbol
        )


def test_original_source_consumer_does_not_construct_method_comparison_surfaces():
    module = parsed(
        "sample",
        "from dataclasses import dataclass\n@dataclass(frozen=True)\n"
        "class Pair:\n    left: int\n    right: int\n"
        "    def project(self): return self.left, self.right\n",
    )

    with ScanCache.scope():
        repository = SourceProductFlowRepository.from_modules((module,))
        authority = repository.declared_product_authorities_by_symbol["sample.Pair"]
        assert authority.field_names == ("left", "right")
        assert repository.class_index.class_for("sample.Pair").method_names == (
            "project",
        )
    held_facts = vars(module).get("_facts", {})
    assert class_index.ModuleClassSyntaxFacets not in held_facts
    assert class_index.ModuleClassSyntaxFacetsWithoutAutoregister not in held_facts


def test_changed_original_source_reproves_declarations_within_the_same_invocation():
    module = parsed("sample", "class Pair:\n    field: int\n")
    changed = module.with_source("class Pair:\n    field: str\n")
    with ScanCache.scope():
        first = CompactClassFamilyIndex.from_modules((module,))
        second = CompactClassFamilyIndex.from_modules((changed,))
        first_member, = first.class_for("sample.Pair").direct_member_declarations
        second_member, = second.class_for("sample.Pair").direct_member_declarations
        assert first_member.annotation_reference_parts == ("int",)
        assert second_member.annotation_reference_parts == ("str",)
