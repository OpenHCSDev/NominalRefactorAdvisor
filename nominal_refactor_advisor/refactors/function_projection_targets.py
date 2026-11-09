from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import (
    TypeAlias,
    TypeVar,
)

from nominal_refactor_advisor.ast_tools import walk_function_body_nodes
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.source_index import AstTargetDigest


@dataclass(frozen=True)
class ProductFieldValue:
    """One named product field and the expression assigned to it."""

    field_name: str
    value_node: ast.expr


@dataclass(frozen=True)
class ReturnFieldValue(ProductFieldValue):
    """One named return-product field and the expression assigned to it."""


@dataclass(frozen=True)
class ReturnDictFieldValue(ReturnFieldValue):
    """One string-key return-dict field and the expression assigned to it."""


@dataclass(frozen=True)
class FunctionProjectionTarget:
    """Common identity for a projection located inside one function or method."""

    source_path: str
    function_qualname: str

    @property
    def owner_qualname(self) -> str | None:
        """Return the enclosing nominal declaration, when this is a method."""

        owner_qualname, separator, _member_name = self.function_qualname.rpartition(".")
        return owner_qualname if separator else None


@dataclass(frozen=True)
class ResolvedFunctionProjectionTarget(FunctionProjectionTarget):
    """Uniquely resolved source-index function that contains a projection."""

    target: AstTargetDigest
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @staticmethod
    def from_rewrite_target(
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> ResolvedFunctionProjectionTarget:
        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_function_like or not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            raise ValueError("Projection must target one exact function")
        return ResolvedFunctionProjectionTarget(
            source_path=target.file_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )

    @staticmethod
    def from_function_identity(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
    ) -> ResolvedFunctionProjectionTarget | None:
        target_ids = SourceIndexTargetSelector.for_function_or_method(
            file_path=source_path,
            qualname=function_qualname,
        ).target_ids(context)
        if len(target_ids) != 1:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=context.source_index.target_by_id[target_ids[0]],
        )

    @staticmethod
    def from_source_line(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        line: int,
    ) -> ResolvedFunctionProjectionTarget | None:
        target = context.source_index.targets_by_file.smallest_enclosing_target(
            source_path,
            line,
            line,
        )
        if target is None:
            return None
        return ResolvedFunctionProjectionTarget.from_target(
            context,
            source_path=source_path,
            target=target,
        )

    @staticmethod
    def from_target(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        target: AstTargetDigest,
    ) -> ResolvedFunctionProjectionTarget | None:
        node = context.ast_target_nodes_by_id.get(target.target_id)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        return ResolvedFunctionProjectionTarget(
            source_path=source_path,
            function_qualname=target.qualname,
            target=target,
            node=node,
        )


@dataclass(frozen=True)
class FunctionReturnProjectionTarget(ResolvedFunctionProjectionTarget):
    """Uniquely resolved return statement inside a source-index function."""

    return_node: ast.Return

    @staticmethod
    def from_return_location(
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
    ) -> FunctionReturnProjectionTarget | None:
        function = ResolvedFunctionProjectionTarget.from_function_identity(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
        )
        if function is None:
            return None
        matches = tuple(
            child
            for child in walk_function_body_nodes(function.node.body)
            if isinstance(child, ast.Return) and child.lineno == line
        )
        if len(matches) != 1:
            return None
        return FunctionReturnProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=matches[0],
        )


ProjectionTargetT = TypeVar("ProjectionTargetT", bound=FunctionProjectionTarget)


@dataclass(frozen=True)
class ReturnDictProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a return dict with named string-key fields."""

    dict_node: ast.Dict
    field_values: tuple[ReturnDictFieldValue, ...]


@dataclass(frozen=True)
class ReturnKeyValueSequenceFieldValue(ReturnFieldValue):
    """One ``("field", value)`` return-sequence item and its source element."""

    element_node: ast.Tuple | ast.List


@dataclass(frozen=True)
class ReturnKeyValueSequenceProjectionTarget(FunctionReturnProjectionTarget):
    """Source-index target for a returned sequence of string-key value pairs."""

    sequence_node: ast.Tuple | ast.List
    field_values: tuple[ReturnKeyValueSequenceFieldValue, ...]


ReturnCollectionProjectionTarget: TypeAlias = (
    ReturnDictProjectionTarget | ReturnKeyValueSequenceProjectionTarget
)


class ReturnDictFieldValueExtractor:
    """Shared extraction of selected string-key fields from return dictionaries."""

    finding: RefactorFinding

    def field_values(self, dict_node: ast.Dict) -> tuple[ReturnDictFieldValue, ...]:
        return ReturnDictProjectionTargetAuthority.field_values(
            dict_node,
            self.finding.metrics.plan_field_names,
        )

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        return ReturnDictProjectionTargetAuthority.string_key_value(node)


class ReturnDictProjectionTargetAuthority:
    """Resolve return-dict projection targets from source-index function facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Dict,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnDictProjectionTarget | None:
        if not isinstance(return_node.value, ast.Dict):
            return None
        dict_node = return_node.value
        field_values = cls.field_values(dict_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnDictProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            dict_node=dict_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        dict_node: ast.Dict,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnDictFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnDictFieldValue] = []
        for key_node, value_node in zip(dict_node.keys, dict_node.values, strict=True):
            field_name = cls.string_key_value(key_node)
            if field_name in selected_field_names:
                values.append(
                    ReturnDictFieldValue(
                        field_name=field_name,
                        value_node=value_node,
                    )
                )
        return tuple(values)

    @staticmethod
    def string_key_value(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


class ReturnKeyValueSequenceProjectionTargetAuthority:
    """Resolve returned ``("field", value)`` sequence projections from source facts."""

    @classmethod
    def from_function_location(
        cls,
        context: CodemodSelectorContext,
        *,
        source_path: str,
        function_qualname: str,
        line: int,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        function_return = FunctionReturnProjectionTarget.from_return_location(
            context,
            source_path=source_path,
            function_qualname=function_qualname,
            line=line,
        )
        if function_return is None or not isinstance(
            function_return.return_node.value,
            ast.Tuple | ast.List,
        ):
            return None
        return cls.from_return_node(
            function_return,
            function_return.return_node,
            field_names,
        )

    @classmethod
    def from_return_node(
        cls,
        function: ResolvedFunctionProjectionTarget,
        return_node: ast.Return,
        field_names: tuple[str, ...],
    ) -> ReturnKeyValueSequenceProjectionTarget | None:
        if not isinstance(return_node.value, ast.Tuple | ast.List):
            return None
        sequence_node = return_node.value
        field_values = cls.field_values(sequence_node, field_names)
        if frozenset(field.field_name for field in field_values) != frozenset(
            field_names
        ):
            return None
        return ReturnKeyValueSequenceProjectionTarget(
            source_path=function.source_path,
            function_qualname=function.function_qualname,
            target=function.target,
            node=function.node,
            return_node=return_node,
            sequence_node=sequence_node,
            field_values=field_values,
        )

    @classmethod
    def field_values(
        cls,
        sequence_node: ast.Tuple | ast.List,
        field_names: tuple[str, ...],
    ) -> tuple[ReturnKeyValueSequenceFieldValue, ...]:
        selected_field_names = frozenset(field_names)
        values: list[ReturnKeyValueSequenceFieldValue] = []
        for element in sequence_node.elts:
            field_value = cls.field_value(element)
            if (
                field_value is not None
                and field_value.field_name in selected_field_names
            ):
                values.append(field_value)
        return tuple(values)

    @classmethod
    def field_value(
        cls,
        element: ast.expr,
    ) -> ReturnKeyValueSequenceFieldValue | None:
        if not isinstance(element, ast.Tuple | ast.List) or len(element.elts) != 2:
            return None
        key_node, value_node = element.elts
        field_name = ReturnDictProjectionTargetAuthority.string_key_value(key_node)
        if field_name is None:
            return None
        return ReturnKeyValueSequenceFieldValue(
            field_name=field_name,
            value_node=value_node,
            element_node=element,
        )
