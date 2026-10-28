"""Importable Python module identity derived from source paths."""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePath

from .source_identity import (
    PACKAGE_INIT_FILE_NAME,
    SourceFileIdentity,
    source_path_is_package_init,
)


@dataclass(frozen=True)
class PythonModulePathIdentity(SourceFileIdentity):
    """Module import identity derived from one source path and analysis root."""

    path: Path
    import_name: str

    @classmethod
    def from_path(
        cls,
        path: Path,
        analysis_root: Path,
    ) -> PythonModulePathIdentity:
        import_root = cls.import_root(path, analysis_root)
        return cls.from_import_root(path, import_root)

    @classmethod
    def from_import_root(
        cls,
        path: Path,
        import_root: Path,
    ) -> PythonModulePathIdentity:
        relative = path.resolve().relative_to(import_root.resolve())
        return cls(path=path, import_name=cls.import_name_of(relative))

    @staticmethod
    def import_name_of(relative: PurePath) -> str:
        """The dotted module name of a source path relative to its import root;
        a package's `__init__.py` names the package."""

        module_parts = relative.with_suffix("").parts
        if source_path_is_package_init(relative):
            module_parts = module_parts[:-1]
        return ".".join(module_parts) if module_parts else "__init__"

    @classmethod
    def from_source_path(cls, path: Path) -> PythonModulePathIdentity:
        """Derive context-free identity when no parsed project declarations exist."""

        import_root = Path(path.anchor) if path.anchor else Path.cwd()
        return cls.from_import_root(path, import_root)

    def resolve_import_from_module(
        self,
        *,
        imported_module: str | None,
        level: int,
    ) -> str | None:
        """Use Python's package-boundary rules without importing analysed modules."""
        if level == 0:
            return imported_module
        try:
            return resolve_name(
                "." * level + (imported_module or ""), self.package_name
            )
        except ImportError:
            return None

    @property
    def package_parts(self) -> tuple[str, ...]:
        """The package this module's relative imports resolve in: itself for a
        package's `__init__.py`, its parent package otherwise; empty at top level."""

        parts = tuple(self.import_name.split("."))
        return parts if self.is_package_init else parts[:-1]

    @property
    def package_name(self) -> str:
        return ".".join(self.package_parts)

    @staticmethod
    def analysis_root_for_scan_root(root: Path) -> Path:
        return root.parent if root.is_file() else root

    @staticmethod
    def import_root(path: Path, analysis_root: Path) -> Path:
        """Use the outer edge of the source package as import-name authority."""

        package_directory = path.parent
        import_root: Path | None = None
        while (package_directory / PACKAGE_INIT_FILE_NAME).is_file():
            import_root = package_directory.parent
            package_directory = package_directory.parent
        return analysis_root if import_root is None else import_root

    @property
    def declared_source_relative_path(self) -> Path:
        module_parts = tuple(self.import_name.split("."))
        if self.is_package_init:
            if module_parts == ("__init__",):
                return Path(PACKAGE_INIT_FILE_NAME)
            return Path(*module_parts, PACKAGE_INIT_FILE_NAME)
        return Path(*module_parts[:-1], f"{module_parts[-1]}.py")

    @property
    def declared_import_root(self) -> Path:
        import_root = self.path.resolve()
        for _part in self.declared_source_relative_path.parts:
            import_root = import_root.parent
        if (
            import_root / self.declared_source_relative_path
        ).resolve() != self.path.resolve():
            raise ValueError(
                f"Module name {self.import_name!r} does not describe {self.path}"
            )
        return import_root

    @property
    def is_importable(self) -> bool:
        return python_module_name_is_importable(self.import_name)


def python_module_name_is_importable(module_name: str) -> bool:
    """Return whether a dotted source identity is valid Python import syntax."""

    parts = module_name.split(".")
    return bool(
        parts
        and all(part.isidentifier() and not keyword.iskeyword(part) for part in parts)
    )
