"""Nominal authorities for Python annotation syntax semantics."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassVarAnnotationAuthority:
    """Own recognition of typing.ClassVar annotation roots."""

    root_name: str = "ClassVar"

    def matches(self, annotation: ast.AST) -> bool:
        return self.annotation_root_name(annotation) == self.root_name

    def matches_source(self, annotation_source: str) -> bool:
        return self.annotation_source_root_name(annotation_source) == self.root_name

    def annotation_root_name(self, annotation: ast.AST) -> str | None:
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Attribute):
            return annotation.attr
        if isinstance(annotation, ast.Subscript):
            return self.annotation_root_name(annotation.value)
        return None

    @staticmethod
    def annotation_source_root_name(annotation_source: str) -> str:
        annotation_root = annotation_source.partition("[")[0].strip()
        return annotation_root.rsplit(".", maxsplit=1)[-1]


CLASSVAR_ANNOTATION_AUTHORITY = ClassVarAnnotationAuthority()
