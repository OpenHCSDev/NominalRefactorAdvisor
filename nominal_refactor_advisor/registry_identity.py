"""Shared registry-key derivation for semantic inheritance families."""

from __future__ import annotations

import re

DEFAULT_REGISTRY_KEY_ATTRIBUTE = "registry_key"


def class_name_registry_key(name: str, cls: type[object]) -> str:
    """Derive a stable snake-case registry key from a concrete class name."""

    del cls
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+", name)
    return "_".join(token.lower() for token in tokens)
