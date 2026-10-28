"""Classes in a project's installed dependencies that already own registration.

A project that registers its classes by hand may depend on a package whose
class already does it. NRA finds that class from what the package's source
does, never from its name: a metaclass whose creation path stores each class
it creates into a mapping, or a base whose `__init_subclass__` stores each
subclass (`CreatedClassStore`). No package or class is named in NRA.

**Which packages.** The installed distributions whose top-level modules the
scanned project imports, plus any the project declares in its
``pyproject.toml``::

    [tool.nominal-refactor-advisor]
    owner-packages = ["metaclass-registry"]

A declared package counts even when nothing imports it yet. The declaration
is recorded with the file's sha256, so a changed declaration is a different
premise; without the key, only imported packages count. A declared package
that is not installed is reported as a gap, not skipped. "Installed" means
visible to the interpreter running NRA, which reads each module's source and
never imports it.

**Cache.** One distribution's owners depend only on its files, so they are
cached under the distribution's name, version and RECORD sha256, together
with the digest of the source that judged them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cache
from importlib import metadata
from pathlib import Path

from .cache_paths import AdvisorCacheLayout
from .class_creation_store import CreatedClassStore
from .class_index import CompactModuleClassHeader
from .declared_root import DeclaredRoot


class RegistrationRole(StrEnum):
    """How an owner registers the classes of its family, with how it is said."""

    receipt_text: str
    finding_text: str

    def __new__(cls, value: str, receipt_text: str, finding_text: str) -> RegistrationRole:
        member = str.__new__(cls, value)
        member._value_ = value
        member.receipt_text = receipt_text
        member.finding_text = finding_text
        return member

    METACLASS = (
        "metaclass",
        "is a metaclass that stores each class it creates in a mapping",
        "stores each class it creates in a mapping",
    )
    """Its `__new__` or `__init__` stores each class it creates."""

    SUBCLASS_HOOK = (
        "subclass_hook",
        "stores each subclass in a mapping from `__init_subclass__`",
        "stores each subclass in a mapping from `__init_subclass__`",
    )
    """Its `__init_subclass__` stores each subclass."""

    @classmethod
    def played(
        cls, stores_created_class: bool, stores_subclass: bool
    ) -> tuple[RegistrationRole, ...]:
        """The roles a class plays, from what its creation path stores."""

        return tuple(
            role
            for role, stores in (
                (cls.METACLASS, stores_created_class),
                (cls.SUBCLASS_HOOK, stores_subclass),
            )
            if stores
        )


@dataclass(frozen=True)
class OwnerPackagesDeclaration:
    """The ``owner-packages`` a project declares, with the declaring file's sha256."""

    pyproject: str
    digest: str
    packages: tuple[str, ...]

    @classmethod
    def declared_for(cls, source: Path) -> OwnerPackagesDeclaration | None:
        pyproject = DeclaredRoot.nearest_pyproject(source)
        if pyproject is None:
            return None
        status = pyproject.stat()
        return cls.read(pyproject, status.st_mtime_ns, status.st_size)

    @classmethod
    @cache
    def read(
        cls, pyproject: Path, modified_ns: int, size: int
    ) -> OwnerPackagesDeclaration | None:
        """The declaration in one version of the file; a rewrite is read again."""

        del modified_ns, size
        content = pyproject.read_bytes()
        match tomllib.loads(content.decode()):
            case {"tool": {"nominal-refactor-advisor": {"owner-packages": list(names)}}}:
                return cls(
                    str(pyproject),
                    hashlib.sha256(content).hexdigest(),
                    tuple(sorted({str(name) for name in names})),
                )
        return None


@dataclass(frozen=True)
class InstalledRegistrationOwner:
    """One installed class that registers its family's classes, with its receipt."""

    distribution: str
    version: str
    record_digest: str
    symbol: str
    file_path: str
    line: int
    role: RegistrationRole

    @property
    def receipt(self) -> str:
        return (
            f"`{self.symbol}` ({self.distribution} {self.version}, RECORD sha256 "
            f"{self.record_digest[:12]}) {self.role.receipt_text}, read from "
            f"{self.file_path}:{self.line}"
        )


@dataclass(frozen=True)
class InstalledDistributionOwners:
    """The registration owners one installed distribution declares."""

    distribution: str
    version: str
    record_digest: str
    owners: tuple[InstalledRegistrationOwner, ...]

    JUDGE_SOURCES = ("class_creation_store.py", "installed_owners.py")

    @staticmethod
    @cache
    def judge_digest() -> str:
        """The digest of the source that decides what an owner is."""

        here = Path(__file__).parent
        digest = hashlib.sha256(f"{sys.version_info[:2]}".encode())
        for name in InstalledDistributionOwners.JUDGE_SOURCES:
            digest.update((here / name).read_bytes())
        return digest.hexdigest()

    @staticmethod
    def cache_dir() -> Path:
        return AdvisorCacheLayout().persistent_cache_home() / "installed-owners"

    @classmethod
    @cache
    def of(cls, distribution_name: str) -> InstalledDistributionOwners | None:
        """The distribution's owners, from the cache when its files are unchanged."""

        try:
            distribution = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            return None
        record = distribution.read_text("RECORD") or ""
        name = distribution.metadata["Name"]
        version = distribution.version
        record_digest = hashlib.sha256(record.encode()).hexdigest()
        cache_path = cls.cache_dir() / (
            f"{name}-{version}-{record_digest[:16]}-{cls.judge_digest()[:16]}.json"
        )
        if cache_path.is_file():
            return cls.decoded(json.loads(cache_path.read_text()))
        found = cls(
            name,
            version,
            record_digest,
            tuple(cls._owners(distribution, name, version, record_digest)),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(found), sort_keys=True))
        return found

    @classmethod
    def decoded(cls, payload: dict[str, object]) -> InstalledDistributionOwners:
        owners = payload["owners"]
        assert isinstance(owners, list)
        return cls(
            str(payload["distribution"]),
            str(payload["version"]),
            str(payload["record_digest"]),
            tuple(
                InstalledRegistrationOwner(
                    **{**owner, "role": RegistrationRole(owner["role"])}
                )
                for owner in owners
            ),
        )

    @staticmethod
    def _modules(distribution: metadata.Distribution) -> Iterator[tuple[str, Path]]:
        for file in distribution.files or ():
            parts = file.parts
            if (
                file.suffix != ".py"
                or not parts
                or not all(part.isidentifier() for part in (*parts[:-1], file.stem))
            ):
                continue
            names = [*parts[:-1], file.stem]
            if names[-1] == "__init__":
                names.pop()
            if names:
                yield ".".join(names), Path(str(distribution.locate_file(file)))

    @classmethod
    def _owners(
        cls,
        distribution: metadata.Distribution,
        name: str,
        version: str,
        record_digest: str,
    ) -> Iterator[InstalledRegistrationOwner]:
        for module_name, path in sorted(cls._modules(distribution)):
            try:
                module = ast.parse(path.read_text(), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
                continue
            for node in module.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                store = CreatedClassStore.of(node, module)
                for role in RegistrationRole.played(
                    store.stores_created_class(), store.stores_subclass()
                )[:1]:
                    yield InstalledRegistrationOwner(
                        name,
                        version,
                        record_digest,
                        f"{module_name}.{node.name}",
                        str(path),
                        node.lineno,
                        role,
                    )


@dataclass(frozen=True)
class RegistrationOwners:
    """The installed registration owners one scan may compose, and what is missing."""

    owners: tuple[InstalledRegistrationOwner, ...]
    distributions: tuple[str, ...]
    gaps: tuple[str, ...]
    declaration: OwnerPackagesDeclaration | None

    @classmethod
    def for_scan(
        cls,
        imported_top_levels: Iterable[str],
        source: Path | None,
    ) -> RegistrationOwners:
        declaration = (
            None if source is None else OwnerPackagesDeclaration.declared_for(source)
        )
        by_top_level = _distributions_by_top_level()
        wanted = {
            distribution
            for top_level in set(imported_top_levels)
            for distribution in by_top_level.get(top_level, ())
        }
        declared = () if declaration is None else declaration.packages
        found: list[InstalledDistributionOwners] = []
        gaps: list[str] = []
        for distribution in sorted(wanted | set(declared)):
            owners = InstalledDistributionOwners.of(distribution)
            if owners is None:
                if distribution in declared:
                    gaps.append(
                        f"`{distribution}` is declared in owner-packages "
                        f"({declaration.pyproject if declaration else ''}) "
                        "but is not installed for the interpreter running NRA"
                    )
                continue
            found.append(owners)
        return cls(
            tuple(owner for distribution in found for owner in distribution.owners),
            tuple(f"{item.distribution} {item.version}" for item in found),
            tuple(gaps),
            declaration,
        )

    @classmethod
    def for_package(
        cls,
        projections: Iterable[CompactModuleClassHeader],
        module_name: str,
        source: Path,
    ) -> RegistrationOwners:
        """The owners for one scanned module: the top-level modules its package
        imports (the scan's own excluded), and the declaration above ``source``."""

        projections = tuple(projections)
        package = module_name.split(".")[0]
        scanned = {projection.module_name.split(".")[0] for projection in projections}
        return cls.for_scan(
            {
                target.split(".")[0]
                for projection in projections
                if projection.module_name.split(".")[0] == package
                for _local, target in projection.import_aliases
            }
            - scanned,
            source,
        )

    def owners_besides(self, symbol: str) -> tuple[InstalledRegistrationOwner, ...]:
        return tuple(owner for owner in self.owners if owner.symbol != symbol)

    @property
    def premise(self) -> str:
        if self.declaration is None:
            return "owner-packages: not declared"
        return (
            f"owner-packages {list(self.declaration.packages)} declared in "
            f"{self.declaration.pyproject} (sha256 {self.declaration.digest[:12]})"
        )


@cache
def _distributions_by_top_level() -> dict[str, tuple[str, ...]]:
    return {
        top_level: tuple(sorted(set(names)))
        for top_level, names in metadata.packages_distributions().items()
    }


__all__ = (
    "InstalledDistributionOwners",
    "InstalledRegistrationOwner",
    "OwnerPackagesDeclaration",
    "RegistrationOwners",
    "RegistrationRole",
)
