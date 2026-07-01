"""Shared filesystem locations for advisor caches."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ParseCachePolicy:
    """Boolean policy controlling parse-cache use."""

    use_parse_cache: bool = True


@dataclass(frozen=True, kw_only=True)
class ParseCacheDirectory(ParseCachePolicy):
    """Filesystem authority for parse-cache backed analysis."""

    parse_cache_dir: Path | None = None

    @property
    def collected_family_cache_dir(self) -> Path | None:
        if not self.use_parse_cache or self.parse_cache_dir is None:
            return None
        return self.parse_cache_dir / "collected-family"


@dataclass(frozen=True)
class AdvisorCacheLayout:
    """Nominal filesystem layout for persistent advisor cache state."""

    application_cache_dir_name: str = "nominal-refactor-advisor"
    environment_cache_home_name: str = "NRA_CACHE_HOME"
    xdg_cache_home_name: str = "XDG_CACHE_HOME"
    ast_parse_entry_name: str = "ast"
    analysis_entry_name: str = "analysis"
    semantic_descent_entry_name: str = "semantic_descent"

    def base_for(self, root: Path) -> Path:
        if root.is_file():
            return root.parent
        return root

    def persistent_cache_home(self) -> Path:
        explicit_cache_home = os.environ.get(self.environment_cache_home_name)
        if explicit_cache_home:
            return Path(explicit_cache_home)
        xdg_cache_home = os.environ.get(self.xdg_cache_home_name)
        if xdg_cache_home:
            return Path(xdg_cache_home) / self.application_cache_dir_name
        return Path.home() / ".cache" / self.application_cache_dir_name

    def root_identity_path(self, root: Path) -> Path:
        base_path = self.base_for(root).resolve()
        digest = hashlib.blake2s(
            str(base_path).encode("utf-8"), digest_size=8
        ).hexdigest()
        return Path(f"{base_path.name}-{digest}")

    def persistent_cache_base(self, root: Path) -> Path:
        return self.persistent_cache_home() / self.root_identity_path(root)

    def parse_cache_dir(self, root: Path) -> Path:
        return self.persistent_cache_base(root) / self.ast_parse_entry_name

    def analysis_cache_dir(self, root: Path) -> Path:
        return self.persistent_cache_base(root) / self.analysis_entry_name

    def semantic_descent_cache_dir(self, root: Path) -> Path:
        return self.persistent_cache_base(root) / self.semantic_descent_entry_name

    def analysis_sibling(self, parse_cache_dir: Path) -> Path:
        if parse_cache_dir.name == self.ast_parse_entry_name:
            return parse_cache_dir.parent / self.analysis_entry_name
        return parse_cache_dir.with_name(
            f"{parse_cache_dir.name}-{self.analysis_entry_name}"
        )

    def semantic_descent_sibling(self, parse_cache_dir: Path) -> Path:
        if parse_cache_dir.name == self.ast_parse_entry_name:
            return parse_cache_dir.parent / self.semantic_descent_entry_name
        return parse_cache_dir.with_name(
            f"{parse_cache_dir.name}-{self.semantic_descent_entry_name}"
        )


advisor_cache_layout = AdvisorCacheLayout()


def default_cache_base(root: Path) -> Path:
    """Return the filesystem root that should own default advisor cache state."""

    return advisor_cache_layout.persistent_cache_base(root)


def default_parse_cache_dir(root: Path) -> Path:
    """Return the default persistent AST cache directory for one scan root."""

    return advisor_cache_layout.parse_cache_dir(root)


def default_analysis_cache_dir(root: Path) -> Path:
    """Return the default persistent finding cache directory for one scan root."""

    return advisor_cache_layout.analysis_cache_dir(root)


def default_semantic_descent_cache_dir(root: Path) -> Path:
    """Return the default semantic-descent graph cache directory for one scan root."""

    return advisor_cache_layout.semantic_descent_cache_dir(root)


def analysis_cache_sibling(parse_cache_dir: Path) -> Path:
    """Return the finding cache directory paired with one AST cache directory."""

    return advisor_cache_layout.analysis_sibling(parse_cache_dir)


def semantic_descent_cache_sibling(parse_cache_dir: Path) -> Path:
    """Return the semantic-descent cache directory paired with one AST cache."""

    return advisor_cache_layout.semantic_descent_sibling(parse_cache_dir)
