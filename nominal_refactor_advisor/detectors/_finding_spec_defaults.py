"""Declaration-derived analysis of redundant ``FindingSpec`` call fields."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from ..ast_tools import AstExpressionProjection, ParsedModule
from ._base import (
    CANDIDATE_COLLECTION_AUTHORITY,
    FindingSpecDefaultFieldCandidate,
    FindingSpecFactory,
    FindingSpecSemanticDefaults,
    FindingSpecSemanticField,
    FindingSpecSemanticValue,
    finding_spec_factory_for_constructor_name,
    finding_spec_factory_for_defaults,
)


@dataclass(frozen=True)
class FindingSpecSemanticKeyword:
    """One recognized semantic field and value in a ``FindingSpec`` call."""

    field: FindingSpecSemanticField
    value_name: str | None
    semantic_value: FindingSpecSemanticValue | None

    @classmethod
    def from_keyword(
        cls,
        keyword: ast.keyword,
    ) -> "FindingSpecSemanticKeyword | None":
        field = FindingSpecSemanticField.from_keyword_name(keyword.arg)
        if field is None:
            return None
        value_name = AstExpressionProjection.terminal_name(keyword.value)
        return cls(
            field=field,
            value_name=value_name,
            semantic_value=field.semantic_value_from_import_name(value_name),
        )

    def is_redundant_against(self, defaults: FindingSpecSemanticDefaults) -> bool:
        return (
            self.semantic_value is not None
            and self.value_name is not None
            and self.semantic_value == defaults.value_for_field(self.field)
        )


@dataclass(frozen=True)
class FindingSpecSemanticKeywords:
    """Typed semantic keyword projection for one ``FindingSpec`` call."""

    items: tuple[FindingSpecSemanticKeyword, ...]

    @classmethod
    def from_keywords(
        cls,
        keywords: tuple[ast.keyword, ...] | list[ast.keyword],
    ) -> "FindingSpecSemanticKeywords":
        return cls(
            tuple(
                semantic_keyword
                for keyword in keywords
                if (
                    semantic_keyword := FindingSpecSemanticKeyword.from_keyword(keyword)
                )
                is not None
            )
        )

    def defaults_over(
        self,
        defaults: FindingSpecSemanticDefaults,
    ) -> FindingSpecSemanticDefaults:
        overrides = {
            semantic_keyword.field.value: semantic_keyword.semantic_value
            for semantic_keyword in self.items
            if semantic_keyword.semantic_value is not None
        }
        return defaults if not overrides else replace(defaults, **overrides)

    def redundant_against(
        self,
        defaults: FindingSpecSemanticDefaults,
    ) -> tuple[FindingSpecSemanticKeyword, ...]:
        return tuple(
            keyword for keyword in self.items if keyword.is_redundant_against(defaults)
        )


@dataclass(frozen=True)
class FindingSpecCallSemantics:
    """Nominal interpretation of one recognized ``FindingSpec`` constructor call."""

    constructor_name: str
    factory: FindingSpecFactory
    keywords: FindingSpecSemanticKeywords

    @classmethod
    def from_call(cls, node: ast.Call) -> "FindingSpecCallSemantics | None":
        constructor_name = AstExpressionProjection.terminal_name(node.func)
        factory = (
            None
            if constructor_name is None
            else finding_spec_factory_for_constructor_name(constructor_name)
        )
        if factory is None:
            return None
        keywords = FindingSpecSemanticKeywords.from_keywords(node.keywords)
        return cls(constructor_name, factory, keywords) if keywords.items else None

    @property
    def recommended_factory(self) -> FindingSpecFactory | None:
        return finding_spec_factory_for_defaults(
            self.keywords.defaults_over(self.factory.semantic_defaults)
        )

    def candidate(
        self,
        module: ParsedModule,
        node: ast.Call,
    ) -> FindingSpecDefaultFieldCandidate | None:
        recommended_factory = self.recommended_factory
        if recommended_factory is None:
            return None
        redundant_keywords = self.keywords.redundant_against(
            recommended_factory.semantic_defaults
        )
        if not redundant_keywords:
            return None
        return FindingSpecDefaultFieldCandidate(
            file_path=module.file_path,
            line=node.lineno,
            constructor_name=self.constructor_name,
            recommended_constructor_name=recommended_factory.constructor_name,
            redundant_keyword_names=tuple(
                keyword.field.value for keyword in redundant_keywords
            ),
            redundant_keyword_values=tuple(
                keyword.value_name
                for keyword in redundant_keywords
                if keyword.value_name is not None
            ),
        )


class FindingSpecDefaultFieldCandidateCollector:
    """Collect redundant semantic call fields through their typed interpretation."""

    @staticmethod
    def candidate(
        module: ParsedModule,
        node: ast.Call,
    ) -> tuple[FindingSpecDefaultFieldCandidate, ...]:
        semantics = FindingSpecCallSemantics.from_call(node)
        if semantics is None:
            return ()
        candidate = semantics.candidate(module, node)
        return () if candidate is None else (candidate,)

    @classmethod
    def collect(
        cls,
        module: ParsedModule,
    ) -> tuple[FindingSpecDefaultFieldCandidate, ...]:
        return CANDIDATE_COLLECTION_AUTHORITY.ast_node_candidates(
            module,
            module.module,
            ast.Call,
            cls.candidate,
        )
