"""Registry normal-form planning policy records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegistryNormalFormPolicy:
    stage_order: int
    normal_form: str
    stage_label: str
    step_template: str
    blocks_metaclass: bool = False
