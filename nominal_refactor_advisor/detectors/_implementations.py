"""Concrete detector implementation aggregator."""

from __future__ import annotations

from . import (  # noqa: F401 - importing a detector module registers it
    _carrier_collapse,
    _class_overlap,
    _environment,
    _heuristic_leads,
    _semantic_descent,
    _record_checks,
    _restated_facts,
    _restated_identity,
    _systemic,
    _structural,
    _runtime,
    _surface,
)
