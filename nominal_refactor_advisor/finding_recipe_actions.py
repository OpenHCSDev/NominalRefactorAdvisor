from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from .models import RefactorFinding
from .json_reports import DataclassJsonReport


@dataclass(frozen=True)
class FindingRecipeActionIdentity(DataclassJsonReport):
    """Detector-independent identity of one source semantic action."""

    subject_separator: ClassVar[str] = "::"

    file_path: str
    subject_name: str

    @classmethod
    def child_subject(cls, parent_subject: str, child_subject: str) -> str:
        return f"{parent_subject}{cls.subject_separator}{child_subject}"

    def conflicts_with(self, other: "FindingRecipeActionIdentity") -> bool:
        return self.file_path == other.file_path and self.subject_conflicts_with(
            other.subject_name
        )

    def subject_conflicts_with(self, other_subject: str) -> bool:
        if self.subject_name == other_subject:
            return True
        return self.subject_name.startswith(
            f"{other_subject}{self.subject_separator}",
        ) or other_subject.startswith(
            f"{self.subject_name}{self.subject_separator}",
        )


@dataclass(frozen=True)
class FindingRecipeActionKey(FindingRecipeActionIdentity):
    """A detector claim projected onto one stable source action identity."""

    detector_id: str

    @classmethod
    def from_finding_file_subjects(
        cls,
        finding: RefactorFinding,
        file_subjects: Iterable[tuple[str, str]],
    ) -> tuple["FindingRecipeActionKey", ...]:
        return tuple(
            cls(
                detector_id=finding.detector_id,
                file_path=file_path,
                subject_name=subject_name,
            )
            for file_path, subject_name in file_subjects
        )
