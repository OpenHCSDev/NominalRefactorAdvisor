"""Nominal role-token authorities shared by semantic graph analyses."""

from __future__ import annotations

from enum import StrEnum


class SemanticRoleIdentityToken(StrEnum):
    """Role tokens that identify semantic variants, keys, or family members."""

    FAMILY_TOKEN = "family"
    FORMAT_TOKEN = "format"
    ID_TOKEN = "id"
    KEY_TOKEN = "key"
    KIND_TOKEN = "kind"
    MODE_TOKEN = "mode"
    NAME_TOKEN = "name"
    ROLE_TOKEN = "role"
    STEP_ID_TOKEN = "step_id"
    TOKEN_TOKEN = "token"
    TYPE_TOKEN = "type"
    VALUE_TOKEN = "value"

    @classmethod
    def authority_affinity_weak_values(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.KIND_TOKEN,
                cls.NAME_TOKEN,
                cls.TYPE_TOKEN,
                cls.VALUE_TOKEN,
            )
        )

    @classmethod
    def inheritance_identity_attr_suffixes(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.ID_TOKEN,
                cls.KEY_TOKEN,
                cls.KIND_TOKEN,
                cls.NAME_TOKEN,
                cls.TOKEN_TOKEN,
                cls.TYPE_TOKEN,
            )
        )

    @classmethod
    def inheritance_identity_attr_names(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.FAMILY_TOKEN,
                cls.FORMAT_TOKEN,
                cls.KIND_TOKEN,
                cls.MODE_TOKEN,
                cls.NAME_TOKEN,
                cls.ROLE_TOKEN,
                cls.STEP_ID_TOKEN,
                cls.TYPE_TOKEN,
            )
        )

    @classmethod
    def identity_axis_values(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.FAMILY_TOKEN,
                cls.KIND_TOKEN,
                cls.KEY_TOKEN,
                cls.MODE_TOKEN,
                cls.NAME_TOKEN,
                cls.ROLE_TOKEN,
                cls.TYPE_TOKEN,
            )
        )

    @classmethod
    def bridge_axis_source_values(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.FORMAT_TOKEN,
                cls.KIND_TOKEN,
                cls.MODE_TOKEN,
                cls.TYPE_TOKEN,
            )
        )

    @classmethod
    def string_identifier_values(cls) -> frozenset[str]:
        return frozenset(
            token.value for token in (cls.ID_TOKEN, cls.KEY_TOKEN, cls.NAME_TOKEN)
        )

    @classmethod
    def pluralized_string_identifier_values(cls) -> frozenset[str]:
        return cls._pluralized_values(cls.string_identifier_values())

    @classmethod
    def runtime_semantic_branch_axis_values(cls) -> frozenset[str]:
        return frozenset(
            token.value for token in (cls.FAMILY_TOKEN, cls.KIND_TOKEN, cls.MODE_TOKEN)
        )

    @classmethod
    def literal_discriminator_axis_values(cls) -> frozenset[str]:
        return frozenset(
            token.value
            for token in (
                cls.FAMILY_TOKEN,
                cls.FORMAT_TOKEN,
                cls.KEY_TOKEN,
                cls.KIND_TOKEN,
                cls.MODE_TOKEN,
                cls.TYPE_TOKEN,
            )
        )

    @classmethod
    def relation_comparison_axis_values(cls) -> frozenset[str]:
        return frozenset(
            token.value for token in (cls.FAMILY_TOKEN, cls.KEY_TOKEN, cls.TYPE_TOKEN)
        )

    @classmethod
    def semantic_suffix_only_values(cls) -> frozenset[str]:
        return frozenset(
            token.value for token in (cls.FAMILY_TOKEN, cls.ID_TOKEN, cls.ROLE_TOKEN)
        )

    @classmethod
    def semantic_extra_keyword_values(cls) -> frozenset[str]:
        return frozenset(
            token.value for token in (cls.MODE_TOKEN, cls.NAME_TOKEN, cls.TYPE_TOKEN)
        )

    @staticmethod
    def _pluralized_values(values: frozenset[str]) -> frozenset[str]:
        return frozenset((*values, *(f"{value}s" for value in values)))
