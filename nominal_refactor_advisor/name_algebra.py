"""Shared nominal-name algebra used by detectors and codemod synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassNameAlgebra:
    ignored_tokens: frozenset[str] = frozenset(
        {"abc", "abstract", "base", "mixin", "spec"}
    )

    def token_set(self, name: str) -> frozenset[str]:
        return frozenset(self.ordered_tokens(name))

    def ordered_tokens(self, name: str) -> tuple[str, ...]:
        return tuple(
            (
                token.lower()
                for token in re.findall(
                    "[A-Z]+(?=[A-Z][a-z0-9]|$)|[A-Z]?[a-z0-9]+",
                    name.lstrip("_"),
                )
                if token.lower() not in self.ignored_tokens
            )
        )

    def longest_common_prefix(self, values: tuple[str, ...]) -> str:
        token_prefix = self.longest_common_token_prefix(values)
        if token_prefix:
            return self.public_name_from_tokens(token_prefix)
        return self.longest_common_character_prefix(values)

    def longest_common_suffix(self, values: tuple[str, ...]) -> str:
        token_suffix = self.longest_common_token_suffix(values)
        if token_suffix:
            return self.public_name_from_tokens(token_suffix)
        return self.longest_common_character_suffix(values)

    def longest_common_token_prefix(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            return ()
        sequences = tuple(self.ordered_tokens(value) for value in values)
        if not all(sequences):
            return ()
        prefix: list[str] = []
        for index, token in enumerate(sequences[0]):
            if all(index < len(sequence) and sequence[index] == token for sequence in sequences):
                prefix.append(token)
                continue
            break
        return tuple(prefix)

    def longest_common_token_suffix(self, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            return ()
        reversed_sequences = tuple(
            tuple(reversed(self.ordered_tokens(value))) for value in values
        )
        if not all(reversed_sequences):
            return ()
        suffix: list[str] = []
        for index, token in enumerate(reversed_sequences[0]):
            if all(
                index < len(sequence) and sequence[index] == token
                for sequence in reversed_sequences
            ):
                suffix.append(token)
                continue
            break
        return tuple(reversed(suffix))

    @staticmethod
    def public_name_from_tokens(tokens: tuple[str, ...]) -> str:
        return "".join(token.capitalize() for token in tokens)

    def longest_common_character_prefix(self, values: tuple[str, ...]) -> str:
        if not values:
            return ""
        prefix = values[0]
        for value in values[1:]:
            while prefix and (not value.startswith(prefix)):
                prefix = prefix[:-1]
        return prefix

    def longest_common_character_suffix(self, values: tuple[str, ...]) -> str:
        if not values:
            return ""
        reversed_values = tuple(value[::-1] for value in values)
        return self.longest_common_character_prefix(reversed_values)[::-1]


CLASS_NAME_ALGEBRA = ClassNameAlgebra()
