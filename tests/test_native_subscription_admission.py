"""Subscription protocols consume source capture, without lexical-name admission."""

import ast
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.captured_reference import (
    CapturedNativeObject,
    CapturedReferenceEffectsABC,
    CapturedReferenceKernel,
    CapturedReferenceRejection,
    CapturedReferenceViolation,
    InitialNativeFrame,
    InitialNativeIsland,
    OpenCapturedReference,
    SingleFlowPrefix,
)
from nominal_refactor_advisor.native_declarations import NativeDeclaration
from nominal_refactor_advisor.native_reference import NativeReferenceEnvironment
from nominal_refactor_advisor.native_subscription import (
    BuiltinGenericAliasSubscription,
    ClassVariableSubscription,
    NativeArgumentInspection,
    NativeSubscriptionAuthority,
)
from nominal_refactor_advisor.product_flow import (
    CompactFlowContext,
    CompactSubscription,
    source_product_flow_projection,
)
from nominal_refactor_advisor.source_entry import SourceModuleEntryPremise
from nominal_refactor_advisor.source_execution import SourceModuleExecution
from test_source_operation_completion_premise import supplied_entry


@dataclass(frozen=True)
class ReferenceFixtureEffects(CapturedReferenceEffectsABC):
    """Explicit test premise: inert reads and tuple/list construction only.

    Held initial objects retain their identity; no analyzed program is executed.
    Calls, imports, attributes, subscriptions and class construction are excluded.
    Operation-specific metadata/hash protocols are tested separately below.
    """

    module: ParsedModule
    context: CompactFlowContext
    frame: InitialNativeFrame

    def admit(self, context, position):
        allowed = {
            ast.Module,
            ast.Assign,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Tuple,
            ast.List,
            ast.Constant,
        }
        if context is not self.context or any(
            type(node) not in allowed for node in ast.walk(self.module.module)
        ):
            return OpenCapturedReference(CapturedReferenceViolation.UNPROVED_EFFECTS)
        return SingleFlowPrefix(context, self.frame, position)


class ReferenceFixtureEnvironment(NativeReferenceEnvironment):
    def require_discard(self, node):
        raise ValueError("Fixture does not admit evaluated result release")

    def require_call(self, node):
        raise ValueError("Fixture does not admit call execution")

    def require_binding_write(self, node):
        raise ValueError("Fixture does not admit lexical storage effects")

    def __init__(self, source, kernel):
        self._source = source
        self._kernel = kernel

    @property
    def source(self):
        return self._source

    @property
    def kernel(self):
        return self._kernel

    def require_namespace_write(self, node):
        raise ValueError("Fixture does not admit namespace writes")

    def require_class_creation(self, node):
        raise ValueError("Fixture does not admit class construction")

    def require_import(self, node):
        raise ValueError("Fixture does not admit import execution")

    def require_import_operation(self, operation):
        raise ValueError("Fixture does not admit import execution")


def _reference(value=list, expression="chosen"):
    source = f"result = {expression}\n"
    tree = ast.parse(source)
    module = ParsedModule(Path("subscription.py"), "subscription", False, tree, source)
    observed = source_product_flow_projection(module)
    globals_storage = {"chosen": value}
    initial = InitialNativeIsland((builtins,), (globals_storage,))
    namespace = initial.namespace_for_storage(globals_storage)
    frame = InitialNativeFrame(
        namespace, namespace, initial.namespace_for_storage(vars(builtins))
    )
    effects = ReferenceFixtureEffects(module, observed.compact.flow_contexts[0], frame)
    environment = ReferenceFixtureEnvironment(
        observed, CapturedReferenceKernel(initial, effects)
    )
    return tree.body[0].value, environment


def _subscription(
    receiver=CapturedNativeObject(list), argument=CapturedNativeObject(int)
):
    """A real source invocation under explicit complete initial binding evidence."""
    text = "result = chosen[argument]\n"
    module = ParsedModule(
        Path("subscription.py"), "subscription", False, ast.parse(text), text
    )
    source = source_product_flow_projection(module)
    initial = InitialNativeIsland((builtins,))
    environment = SourceModuleExecution(
        SourceModuleEntryPremise(
            source,
            initial,
            {"chosen": receiver, "argument": argument},
            initial.namespace_for_storage(vars(builtins)),
        )
    )
    operation = source.node_operation(module.module.body[0].value, CompactSubscription)
    return operation, environment


def _authority(operation, environment):
    return NativeSubscriptionAuthority.for_subscription(
        environment,
        environment.context_for_owner(operation.owner),
        operation.event,
    )


def test_subscription_obeys_shared_gate_rejection():
    rejection = ValueError("Actual read is not an admitted native object")
    evidence = OpenCapturedReference(
        CapturedReferenceViolation.UNADMITTED_IMPORT, cause=rejection
    )
    operation, environment = _subscription(evidence)
    with pytest.raises(CapturedReferenceRejection) as caught:
        _authority(operation, environment)
    assert caught.value.violation is CapturedReferenceViolation.UNADMITTED_IMPORT
    assert caught.value.__cause__ is rejection


@pytest.mark.parametrize(
    "native,expected",
    ((list, BuiltinGenericAliasSubscription), (ClassVar, ClassVariableSubscription)),
)
def test_subscription_dispatches_on_admitted_object_not_lexical_spelling(
    native, expected
):
    operation, environment = _subscription(CapturedNativeObject(native))
    environment = SourceModuleExecution(
        supplied_entry(environment, frozenset((operation,)))
    )
    authority = _authority(operation, environment)
    assert type(authority) is expected
    assert authority.operation is operation
    assert authority.invocation is operation.event
    assert authority.receiver.proves_same_object(CapturedNativeObject(native))
    authority.require_closed()


def test_same_qualified_name_does_not_substitute_for_the_admitted_object():
    counterfeit = type("list", (), {"__module__": "builtins"})
    native = NativeDeclaration(counterfeit)
    assert native.qualified_name == NativeDeclaration(list).qualified_name
    assert native != NativeDeclaration(list)
    operation, environment = _subscription(CapturedNativeObject(counterfeit))
    with pytest.raises(ValueError, match="required native declaration"):
        _authority(operation, environment)


def test_actual_counterfeit_initial_object_is_not_native_by_name():
    events = []

    class Counterfeit(list):
        @classmethod
        def __class_getitem__(cls, item):
            events.append(item)
            return object

    operation, environment = _subscription(CapturedNativeObject(Counterfeit))
    with pytest.raises(ValueError, match="required native declaration"):
        _authority(operation, environment)
    assert events == []
    assert Counterfeit[int] is object
    assert events == [int]


def test_existing_shared_gate_still_selects_native_builtin_subscription():
    operation, environment = _subscription()
    authority = _authority(operation, environment)
    assert type(authority) is BuiltinGenericAliasSubscription
    authority.require_closed()
    authority.result().require_closed()


def test_environment_capture_uses_original_operand():
    reference, environment = _reference()
    assert environment.capture(reference).value is list
    copy = ast.copy_location(ast.Name(id=reference.id, ctx=ast.Load()), reference)
    assert ast.dump(copy, include_attributes=True) == ast.dump(
        reference, include_attributes=True
    )
    with pytest.raises(ValueError, match="identity remains open"):
        environment.require_native(copy, (NativeDeclaration(list),))


def test_fixture_effect_authority_rejects_calls():
    node, environment = _reference(expression="chosen()")
    assert (
        environment.capture(node.func).violation
        is CapturedReferenceViolation.UNPROVED_EFFECTS
    )


def test_equivalent_native_aliases_do_not_create_multiple_identities():
    reference, environment = _reference(OSError)
    candidates = (NativeDeclaration(OSError), NativeDeclaration(IOError))
    assert environment.require_native(reference, candidates).declaration is OSError


def test_single_identity_requirement_delegates_to_shared_selection():
    reference, environment = _reference()
    native = NativeDeclaration(list)
    assert environment.capture(reference).require_native_identity(native) is native


def test_argument_inspection_uses_actual_operand_reads():
    node, environment = _reference(str, "(chosen, int)")
    NativeArgumentInspection(environment).visit(node)
    operation, environment = _subscription(
        CapturedNativeObject(ClassVar), CapturedNativeObject(str)
    )
    environment = SourceModuleExecution(
        supplied_entry(environment, frozenset((operation,)))
    )
    authority = _authority(operation, environment)
    assert type(authority) is ClassVariableSubscription
    authority.require_closed()


def test_non_native_argument_metadata_is_not_executed():
    events = []

    class Payload:
        def __getattribute__(self, name):
            events.append(name)
            return super().__getattribute__(name)

    node, environment = _reference(Payload())
    with pytest.raises(ValueError, match="required native declaration"):
        NativeArgumentInspection(environment).visit(node)
    assert events == []


def test_generic_argument_storage_does_not_claim_classvar_hash_safety():
    events = []

    class Meta(type):
        def __hash__(cls):
            events.append("hash")
            return super().__hash__()

    class Payload(metaclass=Meta):
        pass

    operation, environment = _subscription(
        CapturedNativeObject(list), CapturedNativeObject(Payload)
    )
    _authority(operation, environment).require_closed()
    operation, environment = _subscription(
        CapturedNativeObject(ClassVar), CapturedNativeObject(Payload)
    )
    with pytest.raises(ValueError, match="required native declaration"):
        _authority(operation, environment).require_closed()
    assert events == []
    assert list[Payload].__args__ == (Payload,)
    assert events == []
    assert ClassVar[Payload].__args__ == (Payload,)
    assert events and set(events) == {"hash"}
