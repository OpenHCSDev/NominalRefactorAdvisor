"""Replay the roster-selection factoring on the cache-isolation PR baseline."""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)

PATH = "nominal_refactor_advisor/analysis_cache.py"
OWNER = SourceRewriteTarget(
    file_path=PATH, qualname="DetectorRegistrySignature.current"
)

PLAN = CodemodPlanSequence.from_operations(
    (
        ReplaceFunctionSignatureOperation(
            target=OWNER,
            signature_suffix='(cls, *, detector_types: tuple[type[IssueDetector], ...] | None = None) -> "DetectorRegistrySignature":',
        ),
        ReplaceFunctionBodyOperation(
            target=OWNER,
            body_source='''"""Resolve omitted/full and explicit rosters before caching their signatures."""
if detector_types is None:
    detector_types = IssueDetector.registered_detector_types()
return cls.from_detector_types(detector_types)''',
        ),
        *(
            PatchTargetOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname=f"AnalysisCacheIdentity.{method}"
                ),
                replacements=(
                    SourceTextReplacement(
                        old_source="""detector_registry=(
                DetectorRegistrySignature.current()
                if detector_types is None
                else DetectorRegistrySignature.from_detector_types(detector_types)
            )""",
                        new_source="detector_registry=DetectorRegistrySignature.current(detector_types=detector_types)",
                    ),
                ),
            )
            for method in ("from_source_paths", "from_modules")
        ),
    )
)
