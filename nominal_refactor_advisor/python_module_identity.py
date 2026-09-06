"""Importable Python module identity derived from source paths."""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path

from .source_identity import SourceFileIdentity


@dataclass(frozen=True)
class PythonModulePathIdentity(SourceFileIdentity):
    """Module import identity derived from one source path and analysis root."""

    path: Path
    import_name: str
    is_package_init: bool

    @classmethod
    def from_path(
        cls,
        path: Path,
        analysis_root: Path,
    ) -> "PythonModulePathIdentity":
        import_root = cls.import_root(path, analysis_root)
        return cls.from_import_root(path, import_root)

    @classmethod
    def from_import_root(
        cls,
        path: Path,
        import_root: Path,
    ) -> "PythonModulePathIdentity":
        relative = path.resolve().relative_to(import_root.resolve())
        module_parts = list(relative.with_suffix("").parts)
        is_package_init = bool(module_parts and module_parts[-1] == "__init__")
        if is_package_init:
            module_parts = module_parts[:-1]
        import_name = ".".join(module_parts) if module_parts else "__init__"
        return cls(
            path=path,
            import_name=import_name,
            is_package_init=is_package_init,
        )

    @classmethod
    def from_source_path(cls, path: Path) -> "PythonModulePathIdentity":
        """Derive context-free identity when no parsed project declarations exist."""

        import_root = Path(path.anchor) if path.is_absolute() else Path.cwd()
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
        package = (
            self.import_name
            if self.is_package_init
            else self.import_name.rpartition(".")[0]
        )
        try:
            return resolve_name("." * level + (imported_module or ""), package)
        except ImportError:
            return None

    @staticmethod
    def analysis_root_for_scan_root(root: Path) -> Path:
        return root.parent if root.is_file() else root

    @staticmethod
    def import_root(path: Path, analysis_root: Path) -> Path:
        """Use the outer edge of the source package as import-name authority."""

        package_directory = path.parent
        import_root: Path | None = None
        while (package_directory / "__init__.py").is_file():
            import_root = package_directory.parent
            package_directory = package_directory.parent
        return analysis_root if import_root is None else import_root

    @property
    def declared_source_relative_path(self) -> Path:
        module_parts = tuple(self.import_name.split("."))
        if self.is_package_init:
            if module_parts == ("__init__",):
                return Path("__init__.py")
            return Path(*module_parts, "__init__.py")
        return Path(*module_parts[:-1], f"{module_parts[-1]}.py")

    @property
    def declared_import_root(self) -> Path:
        if self.is_package_init != (self.path.name == "__init__.py"):
            raise ValueError(f"Package-init identity does not describe {self.path}")
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
