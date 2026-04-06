"""Public package surface for the nominal refactor advisor.

The package exposes the high-level analysis and planning entrypoints together with
the canonical result records and taxonomy values that downstream tooling is most
likely to consume.
"""

from .cli import analyze_path, plan_path
from .models import (
    AnalysisReport,
    ImpactDelta,
    OutcomeEstimate,
    RefactorFinding,
    RefactorPlan,
    SourceLocation,
)
from .patterns import PATTERN_SPECS, PatternSpec
from .planner import build_refactor_plans
from .taxonomy import (
    CapabilityTag,
    CertificationLevel,
    ConfidenceLevel,
    ObservationTag,
)

__all__ = sorted(name for name in globals() if not name.startswith("_"))
