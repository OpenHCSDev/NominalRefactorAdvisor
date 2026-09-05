"""Derive class-index views after declaring ClassDeclarationIndex and DirectedGraph."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_index.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        *(
            operation
            for owner, record_type in (
                ("CompactClassFamilyIndex", "CompactIndexedClass"),
                ("ClassFamilyIndex", "IndexedClass"),
            )
            for operation in (
                AddClassBaseOperation(
                    target=replace(module, qualname=owner),
                    base_name=f"ClassDeclarationIndex[{record_type}]",
                ),
                DeleteClassAssignmentsOperation(
                    target=replace(module, qualname=owner),
                    assignment_names=(
                        "classes_by_symbol",
                        "symbols_by_simple_name",
                        "symbols_by_file_and_qualname",
                        "children_by_symbol",
                        "ancestors_by_symbol",
                        "descendants_by_symbol",
                    ),
                ),
                *(
                    DeleteTargetOperation(
                        target=replace(module, qualname=f"{owner}.{name}")
                    )
                    for name in (
                        "class_for",
                        "symbol_for",
                        "ancestor_symbols",
                        "descendant_symbols",
                    )
                ),
            )
        ),
        *(
            DeleteTargetOperation(
                target=replace(module, qualname=f"ClassFamilyIndex.{name}")
            )
            for name in ("known_symbols", "unique_symbols_by_name")
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="CompactClassFamilyIndexBuilder.build"),
            body_source=dedent("""\
                records = tuple(
                    UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                        (record for projection in self.projections for record in projection.classes),
                        lambda record: record.symbol,
                    ).values()
                )
                known_symbols = frozenset(record.symbol for record in records)
                unique_symbols_by_suffix = _unique_known_symbol_by_suffix(known_symbols)
                classes_by_symbol = {
                    record.symbol: record.with_resolved_base_symbols(
                        tuple(
                            resolved
                            for reference in record.base_references
                            if (
                                resolved := self._resolved_bound_symbol(
                                    reference, record.module_name, known_symbols,
                                    unique_symbols_by_suffix,
                                )
                            ) is not None
                        )
                    )
                    for record in records
                }
                return CompactClassFamilyIndex(classes_by_symbol=classes_by_symbol)
                """),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(module, qualname="ClassFamilyIndexBuilder.build"),
            body_source=dedent("""\
                class_records = tuple(
                    UniqueIdentityIndexAuthority.unambiguous_declarations_by_handle(
                        (*self.base_records, *self.module_class_records()),
                        lambda record: record.symbol,
                    ).values()
                )
                known_symbols = frozenset(record.symbol for record in class_records)
                classes_by_symbol = {
                    record.symbol: self.resolved_record(record, known_symbols)
                    for record in class_records
                }
                return ClassFamilyIndex(classes_by_symbol=classes_by_symbol)
                """),
        ),
        *(
            DeleteTargetOperation(target=replace(module, qualname=f"{owner}.{name}"))
            for owner, names in (
                (
                    "CompactClassFamilyIndexBuilder",
                    (
                        "_children_by_symbol",
                        "_ancestors_by_symbol",
                        "_descendants_by_symbol",
                    ),
                ),
                (
                    "ClassFamilyIndexBuilder",
                    (
                        "children_by_symbol",
                        "ancestors_by_symbol",
                        "descendants_by_symbol",
                        "symbols_by_simple_name_multimap",
                    ),
                ),
            )
            for name in names
        ),
        ReplaceFunctionBodyOperation(
            target=SourceRewriteTarget(
                file_path="nominal_refactor_advisor/semantic_descent.py",
                qualname="_rebase_class_family_index",
            ),
            body_source=dedent("""\
                if class_index is None:
                    return None
                return replace(
                    class_index,
                    classes_by_symbol={
                        symbol: replace(
                            indexed_class,
                            file_path=rebase_checkout_path(
                                indexed_class.file_path, source_roots, target_roots,
                            ),
                        )
                        for symbol, indexed_class in class_index.classes_by_symbol.items()
                    },
                )
                """),
        ),
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
