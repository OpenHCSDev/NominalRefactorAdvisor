"""AST-driven refactoring advisor for DQ-Dock."""

from .cli import analyze_path
from .models import RefactorFinding, SourceLocation
from .patterns import PATTERN_SPECS, PatternSpec

__all__ = [
    "PATTERN_SPECS",
    "PatternSpec",
    "RefactorFinding",
    "SourceLocation",
    "analyze_path",
]
