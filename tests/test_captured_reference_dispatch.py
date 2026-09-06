"""Import and write leaves dispatch actual evidence to independent ABC consumers."""

import ast
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import pickle

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.lexical_bindings import (
    ImportBoundNameProjection,
    ImportedNameOrigin,
    ImportOriginResolverABC,
)
from nominal_refactor_advisor.product_flow import (
    CompactAttributeTarget,
    CompactItemTarget,
    CompactMutation,
    CompactMutationResolverABC,
    CompactPositionedReference,
    CompactValueUse,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.value_graph import DataclassGraphValue


class Route(Enum):
    MODULE = auto()
    MEMBER = auto()
    BINDING = auto()
    RECEIVER = auto()
    ATTRIBUTE = auto()
    ITEM = auto()


@dataclass(frozen=True)
class ImportReceipt:
    route: Route
    origin: ImportedNameOrigin
    context: object


class ImportConsumer(ImportOriginResolverABC[object, ImportReceipt]):
    def _module_import_resolution(self, origin, context):
        return ImportReceipt(Route.MODULE, origin, context)

    def _member_import_resolution(self, origin, context):
        return ImportReceipt(Route.MEMBER, origin, context)


def _module(source):
    return ParsedModule(
        path=Path("dispatch.py"),
        module_name="dispatch",
        is_package_init=False,
        module=ast.parse(source),
        source=source,
    )


@pytest.mark.parametrize(
    "source, route, bound_name, qualified_name, requested_module",
    (
        ("import package.child", Route.MODULE, "package", "package", "package.child"),
        (
            "import package.child as package",
            Route.MODULE,
            "package",
            "package.child",
            "package.child",
        ),
        (
            "from package import child",
            Route.MEMBER,
            "child",
            "package.child",
            "package",
        ),
        (
            "from package.child import child as saved",
            Route.MEMBER,
            "saved",
            "package.child.child",
            "package.child",
        ),
        ("from .. import value", Route.MEMBER, "value", None, None),
    ),
)
def test_import_declaration_dispatch_preserves_exact_origin_and_context(
    source,
    route,
    bound_name,
    qualified_name,
    requested_module,
):
    module = _module(source)
    declaration = ImportBoundNameProjection(module.module.body[0]).declaration
    (origin,) = declaration.origins(module.module_path_identity)
    context = object()
    receipt = origin.resolve(ImportConsumer(), context)
    assert receipt.route is route
    assert receipt.origin is origin
    assert receipt.origin.declaration is declaration
    assert receipt.context is context
    assert origin.bound_name == bound_name
    assert origin.qualified_name == qualified_name
    assert origin.requested_module_name == requested_module


def test_grouped_import_dispatch_preserves_shared_declaration_after_pickle():
    module = _module("import package.child as first, package.child as second")
    origins = ImportBoundNameProjection(module.module.body[0]).origins(
        module.module_path_identity
    )
    restored = pickle.loads(pickle.dumps(origins))
    assert restored[0].declaration is restored[1].declaration
    consumer = ImportConsumer()
    for origin in restored:
        receipt = origin.resolve(consumer, consumer)
        assert receipt.origin is origin
        assert receipt.route is Route.MODULE
        assert receipt.context is consumer


@dataclass(frozen=True)
class MutationReceipt:
    route: Route
    mutation: CompactMutation
    context: object
    receiver: CompactValueUse | None = None


class ReceiverConsumer(CompactMutationResolverABC[object, MutationReceipt]):
    def _binding_mutation_resolution(self, context, mutation, name):
        assert name == mutation.target.bound_name
        return MutationReceipt(Route.BINDING, mutation, context)

    def _receiver_mutation_resolution(self, context, mutation, receiver_use):
        return MutationReceipt(Route.RECEIVER, mutation, context, receiver_use)


class AccessConsumer(ReceiverConsumer):
    def _attribute_mutation_resolution(
        self,
        context: object,
        mutation: CompactMutation[CompactAttributeTarget],
    ) -> MutationReceipt:
        return MutationReceipt(
            Route.ATTRIBUTE, mutation, context, mutation.target.receiver_use
        )

    def _item_mutation_resolution(
        self,
        context: object,
        mutation: CompactMutation[CompactItemTarget],
    ) -> MutationReceipt:
        return MutationReceipt(
            Route.ITEM, mutation, context, mutation.target.receiver_use
        )


@pytest.mark.parametrize("consumer", (ReceiverConsumer(), AccessConsumer()))
def test_access_leaf_dispatch_preserves_receiver_capture_before_augmented_rhs(consumer):
    source = "saved = original\nsaved.attribute += (saved := replacement)\nsaved[index] = value\n"
    (flow,) = compact_product_flow_projection(_module(source)).flows
    context = object()
    receipts = [mutation.resolve(consumer, context) for mutation in flow.mutations]
    assert all(receipt.context is context for receipt in receipts)
    assert all(
        receipt.mutation is mutation
        for receipt, mutation in zip(receipts, flow.mutations, strict=True)
    )
    receiver_receipts = [
        receipt for receipt in receipts if receipt.receiver is not None
    ]
    assert len(receiver_receipts) == 2
    attribute, item = receiver_receipts
    assert attribute.receiver is attribute.mutation.target.receiver_use
    assert item.receiver is item.mutation.target.receiver_use
    first, rebound = (
        mutation for mutation in flow.mutations if mutation.target.bound_name == "saved"
    )
    assert (
        flow.binding_resolution_for("saved", attribute.receiver.position).mutation
        is first
    )
    assert (
        flow.binding_resolution_for("saved", item.receiver.position).mutation is rebound
    )
    assert attribute.receiver.position.dominates(rebound.position)
    assert rebound.position.dominates(attribute.mutation.position)


def test_specialized_access_consumer_overrides_only_the_relevant_leaf_hooks():
    (flow,) = compact_product_flow_projection(
        _module("name = value\nreceiver.attribute = value\nreceiver[index] = value\n")
    ).flows
    context = object()
    assert [
        mutation.resolve(ReceiverConsumer(), context).route
        for mutation in flow.mutations
    ] == [Route.BINDING, Route.RECEIVER, Route.RECEIVER]
    assert [
        mutation.resolve(AccessConsumer(), context).route for mutation in flow.mutations
    ] == [Route.BINDING, Route.ATTRIBUTE, Route.ITEM]


def test_positioned_reference_shares_origin_logic_without_changing_record_equality():
    (flow,) = compact_product_flow_projection(
        _module("alias = original\nresult = alias\n")
    ).flows
    read = flow.exact_value_aliases[-1].source_use
    value = CompactValueUse(read.lexical_reference, read.position)
    assert isinstance(read, CompactPositionedReference)
    assert isinstance(value, CompactPositionedReference)
    assert read.origin_in(flow).exact_origin == value.origin_in(flow).exact_origin
    assert read.reference_equivalents_in(flow) == value.reference_equivalents_in(flow)
    assert pickle.loads(pickle.dumps(read)) == read
    equivalent = CompactValueUse(value.value, value.position)
    assert equivalent == value
    assert hash(equivalent) == hash(value)
    assert CompactValueUse.__eq__ is DataclassGraphValue.__eq__
    assert CompactValueUse.__hash__ is DataclassGraphValue.__hash__
    assert hash(read) == hash(pickle.loads(pickle.dumps(read)))
