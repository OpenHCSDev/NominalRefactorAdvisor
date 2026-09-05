"""Resolve collector bases through native declaration authority and exact source."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteTargetOperation,
    InsertBeforeTargetOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(file_path="nominal_refactor_advisor/codemod.py")
candidate = SourceRewriteTarget(file_path="nominal_refactor_advisor/detectors/_base.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        InsertBeforeTargetOperation(
            target=replace(
                candidate,
                qualname="CandidateCollectorBoilerplateCandidate.recommended_base_name",
            ),
            source="""    @property
    def recommended_base_type(self) -> type[DerivedCandidateCollectorMixin]:
        return DerivedCandidateCollectorMixin.collector_base_types_by_shape()[
            CandidateCollectorBaseShape(self.collector_scope, self.uses_config)
        ]

""",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                candidate,
                qualname="CandidateCollectorBoilerplateCandidate.recommended_base_name",
            ),
            body_source="return self.recommended_base_type.__name__\n",
        ),
        DeleteTargetOperation(
            target=replace(
                candidate,
                qualname="CandidateCollectorBoilerplateCandidate.replaced_base_name",
            )
        ),
        InsertBeforeTargetOperation(
            target=replace(
                module, qualname="CandidateCollectorMigration.contextual_base_source"
            ),
            source="    replaced_base: ast.expr\n\n",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                module, qualname="CandidateCollectorMigration.class_header_replacements"
            ),
            body_source=dedent("""\
            header = ClassHeaderSpanSourceAuthority(node=self.node, source=self.source)
            return header.source_edits(
                header.with_replaced_base(ast.unparse(self.replaced_base), self.contextual_base_source),
                file_path=self.target.file_path,
                rationale=self.rationale or f"Derive {self.node.name!r} candidate traversal from its collector.",
            )
            """),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                module, qualname="DeriveCandidateCollectorOperation.required_migration"
            ),
            body_source=dedent("""\
            candidate = witness.candidate
            owner = ResolvedClassTarget.from_rewrite_target(
                snapshot,
                SourceRewriteTarget(file_path=candidate.file_path, qualname=candidate.class_name),
            )
            original = ClassAuthorityReferenceProof.from_native_declaration(
                snapshot, candidate.collector_scope.forwarding_detector_type, owner.file_path,
            )
            replacement = ClassAuthorityReferenceProof.from_native_declaration(
                snapshot, candidate.recommended_base_type, owner.file_path,
            )
            bindings = ModuleNominalBindingAuthority(snapshot.parsed_module_for_source_path(owner.file_path))
            collector_symbols = {
                NativeDeclaration(declaration).qualified_name
                for declaration in DerivedCandidateCollectorMixin.registered_collector_base_types()
            }
            if any(
                bindings.qualified_name_at(base, line=owner.node.lineno) in collector_symbols
                for base in owner.node.bases
            ):
                raise ValueError("Collector migration has a competing native collector base")
            replaced_bases = tuple(
                base for base in owner.node.bases
                if bindings.qualified_name_at(base, line=owner.node.lineno) == original.authority_symbol
            )
            if len(replaced_bases) != 1:
                raise ValueError("Collector migration requires one exact native forwarding base")
            return CandidateCollectorMigration(
                candidate=candidate, target=owner.target, node=owner.node,
                source=snapshot.sources_by_file_path[owner.file_path],
                import_source=replacement.required_import_source(snapshot),
                rationale=self.rationale, replaced_base=replaced_bases[0],
            )
            """),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
