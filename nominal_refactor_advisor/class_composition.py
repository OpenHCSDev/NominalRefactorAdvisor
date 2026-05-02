"""Nominal class-composition derivation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompositeClassSpec:
    class_name: str
    bases: tuple[type[Any], ...]
    namespace: Mapping[str, Any] = ()

    def build(self, module_name: str) -> type[Any]:
        return type(self.class_name, self.bases, {'__module__': module_name, **dict(self.namespace)})
