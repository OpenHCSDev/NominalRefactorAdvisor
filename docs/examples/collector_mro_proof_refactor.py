"""Replace direct-base exclusions with a native inherited-method proof."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source="from .source_native_mro import NativeClassBaseSubstitution, SourceNativeClassMro",
        ),
        EnsureImportOperation(
            target=module,
            import_source="from .native_class_mro import NativeClassMroDeclaration",
        ),
        PatchTargetOperation(
            target=replace(
                module, qualname="DeriveCandidateCollectorOperation.required_migration"
            ),
            replacements=(
                SourceTextReplacement(
                    old_source="""        collector_symbols = {
            NativeDeclaration(declaration).qualified_name
            for declaration in DerivedCandidateCollectorMixin.registered_collector_base_types()
        }
        if any(
            bindings.qualified_name_at(base, line=owner.node.lineno)
            in collector_symbols
            for base in owner.node.bases
        ):
            raise ValueError(
                "Collector migration has a competing native collector base"
            )
""",
                    new_source="",
                ),
                SourceTextReplacement(
                    old_source="        return CandidateCollectorMigration(",
                    new_source="""        SourceNativeClassMro(
            snapshot,
            DerivedCandidateCollectorMixin.registered_collector_base_types(),
        ).require_inherited_method(
            NativeClassBaseSubstitution(
                snapshot.required_class_family_index.classes_by_symbol[
                    owner.required_symbol(snapshot)
                ],
                replaced_bases[0],
                NativeClassMroDeclaration(candidate.recommended_base_type),
            ),
            candidate.method_name,
        )
        return CandidateCollectorMigration(""",
                ),
            ),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
