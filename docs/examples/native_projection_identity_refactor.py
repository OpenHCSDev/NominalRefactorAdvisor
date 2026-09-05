"""Reuse inspected source by native identity; recheck each proposed declaration."""

from dataclasses import replace
import json

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    RemoveImportNamesOperation,
    ReplaceDeclarationDecoratorsOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.json_reports import json_report_object

module = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/native_declarations.py"
)
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=module, import_source="from functools import lru_cache"
        ),
        EnsureImportOperation(target=module, import_source="from typing import cast"),
        ReplaceDeclarationDecoratorsOperation(
            target=replace(module, qualname="NativeDeclaration"),
            decorators_source="@dataclass(frozen=True, eq=False)",
        ),
        InsertBeforeTargetOperation(
            target=replace(module, qualname="NativeDeclaration.qualified_name"),
            source="""    def __hash__(self) -> int:
        return id(self.declaration)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.declaration is cast(NativeDeclaration, other).declaration

""",
        ),
        ReplaceDeclarationDecoratorsOperation(
            target=replace(module, qualname="NativeDeclaration.node"),
            decorators_source="@property\n@lru_cache(maxsize=None)",
        ),
        RemoveImportNamesOperation(
            target=module, module_name="functools", import_names=("cached_property",)
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
