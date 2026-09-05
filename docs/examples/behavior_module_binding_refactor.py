"""Replace name-presence checks with a declaration-owned binding transfer proof."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/projection_descent_codemod.py"
)
binding_authority = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/class_index.py",
    qualname="ModuleNominalBindingAuthority",
)
BATCH_SNAPSHOT_OPERATIONS = (
    ReplaceFunctionSignatureOperation(
        target=replace(
            binding_authority, qualname=f"{binding_authority.qualname}.snapshots_before"
        ),
        signature_suffix="""(self, lines: Iterable[int | None], *, policy: ModuleNominalBindingSnapshotPolicy = ModuleNominalBindingSnapshotPolicy.EXACT) -> dict[int | None, ModuleNominalBindingSnapshot]:""",
    ),
    ReplaceFunctionBodyOperation(
        target=replace(
            binding_authority, qualname=f"{binding_authority.qualname}.snapshots_before"
        ),
        body_source=dedent('''\
            """Resolve requested declaration positions and final bindings in one pass."""
            requested_lines = tuple(dict.fromkeys(lines))
            return _module_nominal_binding_snapshots(
                self,
                tuple(line for line in requested_lines if line is not None),
                include_final=None in requested_lines,
                policy=policy,
            )
            '''),
    ),
    ReplaceFunctionBodyOperation(
        target=replace(
            binding_authority, qualname=f"{binding_authority.qualname}.snapshot_before"
        ),
        body_source="return self.snapshots_before((line,), policy=policy)[line]\n",
    ),
)
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module,
            import_source=(
                "from nominal_refactor_advisor.declaration_binding_transfer import "
                "DeclarationModuleBindingEnvironment, DeclarationModuleBindingTransfer\n"
            ),
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                module,
                qualname="_TypeKeyedBehaviorMethodDescent._require_target_module_bindings",
            ),
            body_source=dedent("""\
                DeclarationModuleBindingTransfer(
                    source=DeclarationModuleBindingEnvironment(
                        self.source_module, self.projection_class.node
                    ),
                    destination=DeclarationModuleBindingEnvironment(
                        self.target_module, self.target_class.node
                    ),
                ).require_preserved(method)
                """),
        ),
        PatchTargetOperation(
            target=module,
            replacements=(
                SourceTextReplacement(old_source="import builtins\n", new_source=""),
            ),
        ),
        *BATCH_SNAPSHOT_OPERATIONS,
    )
)


if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
