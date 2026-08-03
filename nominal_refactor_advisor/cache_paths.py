"""Shared filesystem locations for advisor caches."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import time


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


@dataclass(frozen=True, kw_only=True)
class AdvisorCacheRetentionPolicy:
    """Bound persistent derived state without affecting analysis correctness."""

    max_root_count: int = 128
    max_total_bytes: int = 4 * 1024**3
    max_root_bytes: int = 2 * 1024**3
    maintenance_interval_seconds: float = 3600.0
    active_lock_seconds: float = 600.0


@dataclass(frozen=True)
class AdvisorCacheMaintenanceReport:
    """Observable effects from one cache-retention pass."""

    removed_root_count: int = 0
    removed_file_count: int = 0
    removed_bytes: int = 0
    skipped: bool = False


@dataclass(frozen=True)
class AdvisorCacheRetention:
    """Throttled LRU maintenance for the default persistent cache home."""

    cache_home: Path
    policy: AdvisorCacheRetentionPolicy = AdvisorCacheRetentionPolicy()

    _maintenance_stamp_name = ".retention-maintained"
    _maintenance_lock_name = ".retention.lock"

    def maintain(self, active_cache_base: Path) -> AdvisorCacheMaintenanceReport:
        try:
            self.cache_home.mkdir(parents=True, exist_ok=True)
            active_cache_base.mkdir(parents=True, exist_ok=True)
        except OSError:
            return AdvisorCacheMaintenanceReport(skipped=True)
        try:
            os.utime(active_cache_base)
        except OSError:
            pass
        if not self._maintenance_due():
            return AdvisorCacheMaintenanceReport(skipped=True)
        lock_descriptor = self._acquire_lock()
        if lock_descriptor is None:
            return AdvisorCacheMaintenanceReport(skipped=True)
        try:
            if not self._maintenance_due():
                return AdvisorCacheMaintenanceReport(skipped=True)
            try:
                report = self._prune(active_cache_base)
            except OSError:
                report = AdvisorCacheMaintenanceReport(skipped=True)
            try:
                self._maintenance_stamp_path.touch(exist_ok=True)
            except OSError:
                pass
            return report
        finally:
            try:
                os.close(lock_descriptor)
            except OSError:
                pass
            try:
                self._maintenance_lock_path.unlink()
            except FileNotFoundError:
                pass

    @property
    def _maintenance_stamp_path(self) -> Path:
        return self.cache_home / self._maintenance_stamp_name

    @property
    def _maintenance_lock_path(self) -> Path:
        return self.cache_home / self._maintenance_lock_name

    def _maintenance_due(self) -> bool:
        try:
            age_seconds = time() - self._maintenance_stamp_path.stat().st_mtime
        except OSError:
            return True
        return age_seconds >= self.policy.maintenance_interval_seconds

    def _acquire_lock(self) -> int | None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            return os.open(self._maintenance_lock_path, flags, 0o600)
        except FileExistsError:
            try:
                lock_age = time() - self._maintenance_lock_path.stat().st_mtime
                if lock_age < self.policy.active_lock_seconds:
                    return None
                self._maintenance_lock_path.unlink()
                return os.open(self._maintenance_lock_path, flags, 0o600)
            except (FileExistsError, FileNotFoundError, OSError):
                return None
        except OSError:
            return None

    def _prune(self, active_cache_base: Path) -> AdvisorCacheMaintenanceReport:
        active_absolute = Path(os.path.abspath(active_cache_base))
        root_rows = sorted(
            (
                (self._modified_time(path), path)
                for path in self.cache_home.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            reverse=True,
        )
        protected_rows = [
            row for row in root_rows if Path(os.path.abspath(row[1])) == active_absolute
        ]
        other_rows = [row for row in root_rows if row not in protected_rows]
        retained_rows = [
            *protected_rows,
            *other_rows[: max(0, self.policy.max_root_count - len(protected_rows))],
        ]
        evicted_rows = other_rows[
            max(0, self.policy.max_root_count - len(protected_rows)) :
        ]
        removed_root_count = 0
        removed_file_count = 0
        removed_bytes = 0
        for _, root_path in evicted_rows:
            root_size, root_file_count = self._tree_usage(root_path)
            if self._remove_root(root_path):
                removed_root_count += 1
                removed_file_count += root_file_count
                removed_bytes += root_size

        retained_usage: list[tuple[float, Path, int]] = []
        for modified_time, root_path in retained_rows:
            root_size, _ = self._tree_usage(root_path)
            if root_size > self.policy.max_root_bytes:
                file_count, byte_count = self._prune_root_files(
                    root_path,
                    target_bytes=self.policy.max_root_bytes,
                    current_bytes=root_size,
                )
                removed_file_count += file_count
                removed_bytes += byte_count
                root_size -= byte_count
            retained_usage.append((modified_time, root_path, root_size))

        total_bytes = sum(row[2] for row in retained_usage)
        for _, root_path, root_size in sorted(retained_usage):
            if total_bytes <= self.policy.max_total_bytes:
                break
            if Path(os.path.abspath(root_path)) == active_absolute:
                continue
            _, root_file_count = self._tree_usage(root_path)
            if self._remove_root(root_path):
                removed_root_count += 1
                removed_file_count += root_file_count
                removed_bytes += root_size
                total_bytes -= root_size
        return AdvisorCacheMaintenanceReport(
            removed_root_count=removed_root_count,
            removed_file_count=removed_file_count,
            removed_bytes=removed_bytes,
        )

    @staticmethod
    def _modified_time(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _tree_usage(root: Path) -> tuple[int, int]:
        total_bytes = 0
        file_count = 0
        for directory_path, _, file_names in os.walk(root, followlinks=False):
            for file_name in file_names:
                file_path = Path(directory_path) / file_name
                try:
                    total_bytes += file_path.stat(follow_symlinks=False).st_size
                    file_count += 1
                except OSError:
                    continue
        return total_bytes, file_count

    def _prune_root_files(
        self,
        root: Path,
        *,
        target_bytes: int,
        current_bytes: int,
    ) -> tuple[int, int]:
        candidates: list[tuple[float, int, Path]] = []
        for directory_path, _, file_names in os.walk(root, followlinks=False):
            for file_name in file_names:
                if file_name.endswith((".lock", ".tmp")):
                    continue
                file_path = Path(directory_path) / file_name
                try:
                    stat = file_path.stat(follow_symlinks=False)
                except OSError:
                    continue
                candidates.append((stat.st_mtime, stat.st_size, file_path))
        removed_file_count = 0
        removed_bytes = 0
        for _, file_size, file_path in sorted(candidates):
            if current_bytes - removed_bytes <= target_bytes:
                break
            try:
                file_path.unlink()
            except OSError:
                continue
            removed_file_count += 1
            removed_bytes += file_size
        return removed_file_count, removed_bytes

    @staticmethod
    def _remove_root(root: Path) -> bool:
        try:
            shutil.rmtree(root)
        except OSError:
            return False
        return True


def default_cache_base(root: Path) -> Path:
    """Return the filesystem root that should own default advisor cache state."""

    return advisor_cache_layout.persistent_cache_base(root)


def maintain_default_cache(root: Path) -> AdvisorCacheMaintenanceReport:
    """Touch the active default root and periodically enforce retention bounds."""

    cache_home = advisor_cache_layout.persistent_cache_home()
    return AdvisorCacheRetention(cache_home).maintain(
        advisor_cache_layout.persistent_cache_base(root)
    )


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
