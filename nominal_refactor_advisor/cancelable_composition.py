"""Detect cancelable product-carrier compositions.

The signal is algebraic rather than tied to carrier names. It identifies
functions that map product fields through pack, forward, and unpack steps
without changing those fields or owning an invariant. These identity-like
morphisms can be cancelled before a codemod materialises a rewrite.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from typing import ClassVar

from .ast_tools import AstExpressionProjection, FunctionDefinitionNode
from .collection_algebra import sorted_tuple
from .semantic_match import Maybe
from .source_index import (
    AstTargetDigest,
    AstTargetNodeIndex,
    SourceIndex,
    SourceTargetSpan,
)


@dataclass(frozen=True, kw_only=True)
class ProductForwardIdentity:
    """Product carrier/source/field identity shared by forward projections."""

    carrier_name: str
    source_name: str
    field_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class _ProductForward(ProductForwardIdentity):
    """AST-local product-forward projection fact."""

    @classmethod
    def from_fields(
        cls,
        *,
        carrier_name: str,
        source_name: str,
        field_names: tuple[str, ...],
    ) -> "_ProductForward | None":
        """Construct one meaningful product projection from unique fields."""

        return (
            Maybe.of(sorted_tuple(set(field_names)))
            .filter(lambda unique_fields: len(unique_fields) >= 2)
            .map(
                lambda unique_fields: cls(
                    carrier_name=carrier_name,
                    source_name=source_name,
                    field_names=unique_fields,
                )
            )
            .unwrap_or_none()
        )

    @classmethod
    def from_intersection(
        cls,
        packed: "_ProductForward",
        unpacked: "ProductForwardFieldProjection",
    ) -> "_ProductForward | None":
        """Retain only fields preserved by both sides of a composition."""

        return cls.from_fields(
            carrier_name=packed.carrier_name,
            source_name=packed.source_name,
            field_names=tuple(
                set(packed.field_names) & set(unpacked.product_fields)
            ),
        )


class _CancelableCompositionProjectionABC(ABC):
    """Leaf execution owned by one cancelable-composition kind."""

    load_bearing_bonus: ClassVar[int]

    @abstractmethod
    def product_forward(
        self,
        node: FunctionDefinitionNode,
    ) -> _ProductForward | None:
        """Project one function body into this exact composition kind."""

        raise NotImplementedError


class _ProductPackForwardProjection(_CancelableCompositionProjectionABC):
    """Project a direct product construction returned from a function."""

    load_bearing_bonus = 25

    def product_forward(
        self,
        node: FunctionDefinitionNode,
    ) -> _ProductForward | None:
        match node.body:
            case [ast.Return(value=ast.Call() as call)]:
                return ProductForwardCallAuthority(call).product_forward()
            case _:
                return None


class _PackUnpackForwardProjection(_CancelableCompositionProjectionABC):
    """Project a product construction immediately unpacked into another."""

    load_bearing_bonus = 75

    def product_forward(
        self,
        node: FunctionDefinitionNode,
    ) -> _ProductForward | None:
        match node.body:
            case [
                ast.Assign(
                    targets=[ast.Name(id=carrier_variable_name)],
                    value=ast.Call() as pack_call,
                ),
                ast.Return(value=ast.expr() as returned_value),
            ]:
                return (
                    Maybe.of(
                        ProductForwardCallAuthority(pack_call).product_forward()
                    )
                    .combine(
                        lambda _packed: ProductForwardFieldProjection.from_return_value(
                            returned_value,
                            carrier_variable_name,
                        ),
                        _ProductForward.from_intersection,
                    )
                    .unwrap_or_none()
                )
            case _:
                return None


class CancelableCompositionKind(StrEnum):
    """Kinds of product-carrier compositions with member-owned execution."""

    PRODUCT_PACK_FORWARD = (
        "product_pack_forward",
        _ProductPackForwardProjection(),
    )
    PACK_UNPACK_FORWARD = (
        "pack_unpack_forward",
        _PackUnpackForwardProjection(),
    )

    def __new__(
        cls,
        value: str,
        projection: _CancelableCompositionProjectionABC,
    ) -> "CancelableCompositionKind":
        member = str.__new__(cls, value)
        member._value_ = value
        member._projection = projection
        return member

    @property
    def load_bearing_bonus(self) -> int:
        """Return the prioritisation rent derived from the leaf projector."""

        return self._projection.load_bearing_bonus

    def signal_for(
        self,
        authority: "CancelableCompositionSignalTargetAuthority",
    ) -> "CancelableCompositionSignal | None":
        """Execute this member's projection for one indexed function target."""

        return (
            Maybe.of(self._projection.product_forward(authority.node))
            .map(lambda product: authority.cancelable_signal(self, product))
            .unwrap_or_none()
        )


@dataclass(frozen=True, kw_only=True)
class CancelableCompositionSignal(SourceTargetSpan, ProductForwardIdentity):
    """Generic factorable morphism over product carrier fields."""

    composition_kind: CancelableCompositionKind
    covered_finding_ids: tuple[str, ...] = ()

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def covered_finding_count(self) -> int:
        return len(self.covered_finding_ids)

    @property
    def load_bearing_score(self) -> int:
        return (
            self.field_count * 50
            + self.covered_finding_count * 100
            + self.composition_kind.load_bearing_bonus
        )

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.target_id,)


def detect_cancelable_composition_signals(
    source_index: SourceIndex,
    source_by_path: Mapping[str, str],
) -> tuple[CancelableCompositionSignal, ...]:
    """Detect generic pack/unpack/forward compositions worth factoring away."""

    nodes_by_target_id = AstTargetNodeIndex.from_source_mapping(
        source_index,
        source_by_path,
    ).function_nodes_by_target_id
    signals = []
    for target in source_index.ast_targets:
        if not target.is_function_like:
            continue
        node = nodes_by_target_id.get(target.target_id)
        if node is None:
            continue
        signal = CancelableCompositionSignalTargetAuthority(
            source_index, target, node
        ).signal()
        if signal is not None:
            signals.append(signal)
    return sorted_tuple(
        signals,
        key=lambda item: (
            -item.load_bearing_score,
            item.file_path,
            item.line,
            item.qualname,
        ),
    )


@dataclass(frozen=True)
class CancelableCompositionSignalTargetAuthority:
    """Build cancelable-composition signals for one function target."""

    source_index: SourceIndex
    target: AstTargetDigest
    node: FunctionDefinitionNode

    def signal(self) -> CancelableCompositionSignal | None:
        return next(
            filter(
                lambda signal: signal is not None,
                (kind.signal_for(self) for kind in CancelableCompositionKind),
            ),
            None,
        )

    def cancelable_signal(
        self,
        composition_kind: CancelableCompositionKind,
        product_forward: _ProductForward,
    ) -> CancelableCompositionSignal:
        return CancelableCompositionSignal(
            target_id=self.target.target_id,
            file_path=self.target.file_path,
            qualname=self.target.qualname,
            line=self.target.line,
            end_line=self.target.end_line,
            composition_kind=composition_kind,
            carrier_name=product_forward.carrier_name,
            source_name=product_forward.source_name,
            field_names=product_forward.field_names,
            covered_finding_ids=self.source_index.finding_ids_for_target_id(
                self.target.target_id
            ),
        )


@dataclass(frozen=True)
class ProductForwardFieldProjection:
    """Fields projected from one product carrier construction call."""

    source_name: str | None = None
    field_names: tuple[str, ...] = ()

    @classmethod
    def from_arguments(
        cls,
        arguments: Sequence[ast.expr],
        keywords: Sequence[ast.keyword] = (),
        *,
        source_name: str | None = None,
    ) -> "ProductForwardFieldProjection | None":
        """Project one argument list through the shared optional-effect algebra."""

        positional_projection = reduce(
            lambda projection, argument: projection.project(
                lambda fields: fields.with_positional_argument(argument)
            ),
            arguments,
            Maybe.of(cls(source_name=source_name)),
        )
        return reduce(
            lambda projection, keyword: projection.project(
                lambda fields: fields.with_keyword(keyword)
            ),
            keywords,
            positional_projection,
        ).unwrap_or_none()

    @classmethod
    def from_return_value(
        cls,
        value: ast.AST,
        carrier_variable_name: str,
    ) -> "ProductForwardFieldProjection | None":
        """Project supported unpack-return syntax from one product carrier."""

        match value:
            case ast.Call(args=arguments, keywords=keywords):
                return cls.from_arguments(
                    arguments,
                    keywords,
                    source_name=carrier_variable_name,
                )
            case ast.Tuple(elts=arguments) | ast.List(elts=arguments):
                return cls.from_arguments(
                    arguments,
                    source_name=carrier_variable_name,
                )
            case _:
                return None

    @property
    def product_fields(self) -> tuple[str, ...]:
        return sorted_tuple(set(self.field_names))

    def with_positional_argument(
        self,
        argument: ast.expr,
    ) -> "ProductForwardFieldProjection | None":
        return (
            Maybe.of(AstExpressionProjection(argument).attribute_projection())
            .project(lambda projected: self.with_projected_field(*projected))
            .unwrap_or_none()
        )

    def with_keyword(
        self,
        keyword: ast.keyword,
    ) -> "ProductForwardFieldProjection | None":
        return (
            Maybe.of(keyword.arg)
            .combine(
                lambda _keyword_name: AstExpressionProjection(
                    keyword.value
                ).attribute_projection(),
                lambda keyword_name, projected: (keyword_name, *projected),
            )
            .filter(
                lambda projection: projection[0] == projection[2]
            )
            .project(
                lambda projection: self.with_projected_field(
                    projection[1],
                    projection[2],
                )
            )
            .unwrap_or_none()
        )

    def with_projected_field(
        self,
        candidate_source_name: str,
        field_name: str,
    ) -> "ProductForwardFieldProjection | None":
        return (
            Maybe.of(candidate_source_name)
            .filter(
                lambda source_name: self.source_name in (None, source_name)
            )
            .map(
                lambda source_name: ProductForwardFieldProjection(
                    source_name=source_name,
                    field_names=(*self.field_names, field_name),
                )
            )
            .unwrap_or_none()
        )

    def product_forward(self, carrier_name: str) -> _ProductForward | None:
        return (
            Maybe.of(self.source_name)
            .project(
                lambda source_name: _ProductForward.from_fields(
                    carrier_name=carrier_name,
                    source_name=source_name,
                    field_names=self.product_fields,
                )
            )
            .unwrap_or_none()
        )


@dataclass(frozen=True)
class ProductForwardCallAuthority:
    """Project product-carrier construction calls into cancelable forward facts."""

    call: ast.Call

    def product_forward(self) -> _ProductForward | None:
        return (
            Maybe.of(AstExpressionProjection(self.call.func).qualified_name())
            .combine(
                lambda carrier_name: self.field_projection(),
                lambda carrier_name, projection: projection.product_forward(
                    carrier_name
                ),
            )
            .unwrap_or_none()
        )

    def field_projection(self) -> ProductForwardFieldProjection | None:
        return ProductForwardFieldProjection.from_arguments(
            self.call.args,
            self.call.keywords,
        )
