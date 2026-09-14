"""Reuse the scan's engine signature instead of reconstructing it per source."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    EnsureImportOperation,
    ReplaceDeclarationDecoratorsOperation,
    SourceRewriteTarget,
)

SOURCE = "nominal_refactor_advisor/analysis_cache.py"
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=SourceRewriteTarget(file_path=SOURCE),
            import_source="from .scan_cache import ScanCache",
        ),
        ReplaceDeclarationDecoratorsOperation(
            target=SourceRewriteTarget(
                file_path=SOURCE, qualname="AnalysisEngineSignature.current"
            ),
            decorators_source="@classmethod\n@ScanCache.cached",
        ),
        *(
            ReplaceDeclarationDecoratorsOperation(
                target=SourceRewriteTarget(file_path=SOURCE, qualname=name),
                decorators_source="@ScanCache.cached",
            )
            for name in (
                "_module_source_signature_from_path",
                "_detector_module_file_hash",
            )
        ),
    )
)
