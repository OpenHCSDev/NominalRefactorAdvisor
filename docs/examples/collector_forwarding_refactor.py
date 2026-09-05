"""Derive collector facts and capture proof from authored shared declarations."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    DeleteClassAssignmentsOperation,
    DeleteTargetOperation,
    EnsureImportOperation,
    InsertAfterTargetOperation,
    InsertBeforeTargetOperation,
    PrependFunctionBodyOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

candidate = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/detectors/_base.py",
    qualname="CandidateCollectorBoilerplateCandidate",
)
migration = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/codemod.py",
    qualname="CandidateCollectorMigration",
)
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=replace(candidate, qualname=""),
            import_source="from ..positional_forwarding import PositionalForwardingCall",
        ),
        EnsureImportOperation(
            target=replace(migration, qualname=""),
            import_source="from .declaration_binding_transfer import ClassBodyReferenceCapture",
        ),
        *(
            ReplaceScopeAssignmentOperation(
                target=replace(candidate, qualname="CandidateCollectorScope"),
                assignment_name=name,
                source=f'{name} = "{value}"',
            )
            for name, value in (
                ("MODULE", "module"),
                ("FLATTENED_MODULE", "flattened_module"),
                ("CROSS_MODULE", "cross_module"),
            )
        ),
        *(
            DeleteTargetOperation(
                target=replace(candidate, qualname=f"CandidateCollectorScope.{name}")
            )
            for name in ("__new__", "collector_argument_name")
        ),
        InsertBeforeTargetOperation(
            target=replace(
                candidate, qualname="CandidateCollectorScope.forwarding_detector_type"
            ),
            source="""    @cached_property
    def direct_forwarding_calls(self) -> tuple[PositionalForwardingCall, ...]:
        return tuple(
            forwarding
            for declaration in DerivedCandidateCollectorMixin.registered_collector_base_types()
            if declaration.collector_scope is self
            for forwarding in (PositionalForwardingCall.from_callable(declaration._candidate_items),)
            if forwarding is not None
        )
""",
        ),
        ReplaceScopeAssignmentOperation(
            target=candidate,
            assignment_name="collector_name",
            source="forwarding: PositionalForwardingCall",
        ),
        DeleteClassAssignmentsOperation(
            target=candidate, assignment_names=("uses_config",)
        ),
        ReplaceFunctionBodyOperation(
            target=replace(candidate, qualname=f"{candidate.qualname}.from_class"),
            body_source=dedent("""\
            detector_shape = cls.detector_shape(node)
            if detector_shape is None:
                return ()
            forwarding_base_name, candidate_type_source = detector_shape
            method = next(
                (statement for statement in node.body
                 if isinstance(statement, ast.FunctionDef)
                 and statement.name == CandidateFindingDetector._candidate_items.__name__),
                None,
            )
            if method is None:
                return ()
            collector_calls = tuple(
                (scope, call)
                for scope in CandidateCollectorScope.for_forwarding_base_name(forwarding_base_name)
                for call in (cls.collector_call(method, scope),)
                if call is not None
            )
            if len(collector_calls) != 1:
                return ()
            collector_scope, forwarding = collector_calls[0]
            candidate = cls(
                file_path=module.file_path, line=method.lineno,
                class_name=node.name, method_name=method.name,
                forwarding=forwarding, collector_scope=collector_scope,
                candidate_type_source=candidate_type_source,
            )
            return () if candidate.recommended_base_name == node.name else (candidate,)
            """),
        ),
        ReplaceFunctionSignatureOperation(
            target=replace(candidate, qualname=f"{candidate.qualname}.collector_call"),
            signature_suffix="(method: ast.FunctionDef, collector_scope: CandidateCollectorScope) -> PositionalForwardingCall | None:",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(candidate, qualname=f"{candidate.qualname}.collector_call"),
            body_source=dedent("""\
            forwarding = PositionalForwardingCall.from_function(method)
            if forwarding is None:
                return None
            return forwarding if any(
                forwarding.parameter_names[1:] == declared.parameter_names[1:]
                and forwarding.argument_names == declared.argument_names
                for declared in collector_scope.direct_forwarding_calls
            ) else None
            """),
        ),
        InsertAfterTargetOperation(
            target=replace(candidate, qualname=f"{candidate.qualname}.collector_call"),
            source="\n    @property\n    def collector_name(self) -> str:\n        return ast.unparse(self.forwarding.callee)\n\n    @property\n    def uses_config(self) -> bool:\n        return len(self.forwarding.argument_names) == 2\n",
        ),
        InsertAfterTargetOperation(
            target=replace(
                migration,
                qualname=f"{migration.qualname}.candidate_declaration_insertion",
            ),
            source="\n    @cached_property\n    def candidate_method(self) -> ast.FunctionDef:\n        method = next((statement for statement in self.node.body if isinstance(statement, ast.FunctionDef) and statement.name == self.candidate.method_name), None)\n        if method is None:\n            raise ValueError(f'{self.candidate.symbol!r} is no longer declared by the target class')\n        return method\n",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                migration, qualname=f"{migration.qualname}.candidate_method_deletion"
            ),
            body_source=dedent("""\
            return NamedDeclarationSourceAuthority(
                self.candidate_method, self.source,
            ).declaration_line_span.line_deletion(
                file_path=self.target.file_path,
                rationale=self.rationale or "Delete candidate traversal now owned by the collector base.",
            )
            """),
        ),
        PrependFunctionBodyOperation(
            target=replace(migration, qualname=f"{migration.qualname}.source_edits"),
            body_source=dedent("""\
            ClassBodyReferenceCapture(
                context.parsed_module_for_source_path(self.target.file_path),
                self.node, self.candidate_method, self.candidate.forwarding.callee,
            ).require_preserved()
            """),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
