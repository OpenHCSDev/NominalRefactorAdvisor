"""Shared filesystem locations for advisor caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdvisorCacheLayout:
    """Nominal filesystem layout for persistent advisor cache state."""

    directory_name: str = ".nra-cache"
    ast_parse_entry_name: str = "ast"
    analysis_entry_name: str = "analysis"

    def base_for(self, root: Path) -> Path:
        if root.is_file():
            return root.parent
        return root

    @property
    def ast_parse_relative_path(self) -> Path:
        return Path(self.directory_name) / self.ast_parse_entry_name

    @property
    def analysis_relative_path(self) -> Path:
        return Path(self.directory_name) / self.analysis_entry_name

    def parse_cache_dir(self, root: Path) -> Path:
        return self.base_for(root) / self.ast_parse_relative_path

    def analysis_cache_dir(self, root: Path) -> Path:
        return self.base_for(root) / self.analysis_relative_path

    def analysis_sibling(self, parse_cache_dir: Path) -> Path:
        return parse_cache_dir.parent / self.analysis_entry_name


advisor_cache_layout = AdvisorCacheLayout()


def default_cache_base(root: Path) -> Path:
    """Return the filesystem root that should own default advisor cache state."""

    return advisor_cache_layout.base_for(root)


def default_parse_cache_dir(root: Path) -> Path:
    """Return the default persistent AST cache directory for one scan root."""

    return advisor_cache_layout.parse_cache_dir(root)


def default_analysis_cache_dir(root: Path) -> Path:
    """Return the default persistent finding cache directory for one scan root."""

    return advisor_cache_layout.analysis_cache_dir(root)


def analysis_cache_sibling(parse_cache_dir: Path) -> Path:
    """Return the finding cache directory paired with one AST cache directory."""

    return advisor_cache_layout.analysis_sibling(parse_cache_dir)
