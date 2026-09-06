from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleClassProjectionFamily,
    CompactProductAuthorityViolation,
    OpenCompactProductAuthority,
)
from nominal_refactor_advisor.product_flow import (
    CurrentClassMemberMethodReference,
    LexicalCallTargetReference,
    LexicalValueReference,
    compact_product_flow_projection,
)
from nominal_refactor_advisor.product_flow_authority import (
    CompactFunctionTargetResolutionViolation,
    CompactOpenFunctionCall,
    CompactProductFlowRepository,
)


def test_target_resolution_obeys_declared_mro_instead_of_concrete_class_dispatch() -> (
    None
):
    class LexicallyBoundMember(
        LexicalCallTargetReference, CurrentClassMemberMethodReference
    ):
        @property
        def lexical_reference(self) -> LexicalValueReference:
            return LexicalValueReference(self.method_name)

    repository = _repository(
        _module(
            "pkg.local",
            "def consume(value): return value\n"
            "def caller(value): return consume(value)\n",
        )
    )
    context = repository.flow_contexts_by_owner_symbol["pkg.local.caller"]
    target = LexicallyBoundMember("Owner", "member", "consume", False)
    resolution = repository.resolve_function_target(
        context, target, context.flow.calls[0].position
    )
    assert resolution.declaration is not None
    assert resolution.declaration.identity.symbol == "pkg.local.consume"


def _module(module_name: str, source: str) -> ParsedModule:
    path = Path(*module_name.split(".")).with_suffix(".py")
    return ParsedModule(
        path=path,
        module_name=module_name,
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _repository(*modules: ParsedModule) -> CompactProductFlowRepository:
    return CompactProductFlowRepository(
        product_projections=tuple(
            compact_product_flow_projection(module) for module in modules
        ),
        class_projections=CompactModuleClassProjectionFamily.collect_modules(modules),
    )


def _open_product_violations(
    repository: CompactProductFlowRepository,
    class_symbol: str,
) -> frozenset[CompactProductAuthorityViolation]:
    resolution = repository.class_index.product_authority_resolutions_by_symbol[
        class_symbol
    ]
    assert isinstance(resolution, OpenCompactProductAuthority)
    return frozenset(failure.violation for failure in resolution.failures)


@pytest.mark.parametrize(
    "import_source,call",
    (
        ("from library import consume\n", "consume(1)"),
        ("import library as lib\n", "lib.consume(1)"),
    ),
)
def test_imported_function_requires_live_export_binding(
    import_source: str, call: str
) -> None:
    repository = _repository(
        _module("library", "def consume(value): return value\nconsume = replacement\n"),
        _module("probe", import_source + f"def run(): return {call}\n"),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is None
    assert (
        resolution.target_resolution.violation
        is CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
    )
    assert "library.consume" in resolution.target_resolution.possible_symbols


def test_reexport_chain_resolves_declaring_function() -> None:
    repository = _repository(
        _module("library", "def consume(value): return value\n"),
        _module("facade", "from library import consume as exposed\n"),
        _module(
            "probe",
            "from facade import exposed as consume\ndef run(): return consume(1)\n",
        ),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is not None
    assert resolution.resolved_call.callee.identity.symbol == "library.consume"


@pytest.mark.parametrize("class_scope", (False, True))
@pytest.mark.parametrize(
    "capture",
    (
        "alias = first\nfirst = replacement\n",
        "alias = first\nfirst = alias\n",
        "alias = first\nalias = alias\n",
        "first = alias = first\n",
        "alias: object = first\n",
    ),
)
def test_callable_alias_captures_source_before_assignment_targets(
    class_scope: bool, capture: str
) -> None:
    prefix = "class Owner:\n" if class_scope else ""
    indent = "    " if class_scope else ""
    definition = (
        "    @staticmethod\n    def first(value): return value\n"
        if class_scope
        else "def first(value): return value\n"
    )
    source = (
        prefix
        + definition
        + "".join(indent + line + "\n" for line in capture.splitlines())
    )
    source += (
        "def run(): return " + ("Owner.alias" if class_scope else "alias") + "(3)\n"
    )
    repository = _repository(_module("probe", source))
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is not None
    assert resolution.resolved_call.callee.identity.symbol == (
        "probe.Owner.first" if class_scope else "probe.first"
    )


def test_local_alias_resolves_module_function_and_local_function() -> None:
    repository = _repository(
        _module(
            "probe",
            "def first(value): return value\n"
            "def run():\n"
            "    def second(value): return value\n"
            "    alias = first\n"
            "    alias(1)\n"
            "    alias = second\n"
            "    return alias(2)\n",
        )
    )
    calls = tuple(
        r
        for r in repository.resolved_function_calls
        if r.context.owner_symbol == "probe.run"
    )
    assert tuple(r.callee.identity.symbol for r in calls) == (
        "probe.first",
        "probe.run.second",
    )


@pytest.mark.parametrize(
    "decorator,parameters",
    (
        ("", "self, value"),
        ("@classmethod\n    ", "cls, value"),
        ("@staticmethod\n    ", "value"),
    ),
)
def test_inherited_method_alias_preserves_descriptor_binding(
    decorator: str, parameters: str
) -> None:
    repository = _repository(
        _module(
            "probe",
            "class Parent:\n"
            f"    {decorator}def first({parameters}): return value\n"
            "    alias = first\n"
            "class Child(Parent):\n"
            "    def run(self): return self.alias(3)\n",
        )
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.Child.run"
    )
    assert resolution.resolved_call is not None
    assert resolution.resolved_call.callee.identity.symbol == "probe.Parent.first"
    call = resolution.resolved_call.call
    assert resolution.resolved_call.callee.bind_call(
        call.arguments.positional, call.arguments.keywords
    ).is_exact


def test_alias_reexport_cycle_is_open_without_recursion() -> None:
    repository = _repository(
        _module(
            "first",
            "from second import alias as implementation\nalias = implementation\n",
        ),
        _module(
            "second",
            "from first import alias as implementation\nalias = implementation\n",
        ),
        _module("probe", "from first import alias\ndef run(): return alias(1)\n"),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is None
    assert (
        resolution.target_resolution.violation
        is CompactFunctionTargetResolutionViolation.CYCLIC_BINDING
    )


@pytest.mark.parametrize(
    "source",
    (
        "def function(self, value): return value\nclass Owner:\n    alias = function\n    def run(self): return self.alias(3)\n",
        "class Base:\n    def first(self, value): return value\nclass Owner:\n    alias = Base.first\n    def run(self): return self.alias(3)\n",
        "class Base:\n    def first(self, value): return value\nalias = Base.first\ndef run(): return alias(3)\n",
    ),
)
def test_descriptor_rebinding_is_not_inferred_from_callable_identity(
    source: str,
) -> None:
    repository = _repository(_module("probe", source))
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol.endswith(".run")
    )
    assert resolution.resolved_call is None
    assert (
        resolution.target_resolution.violation
        is CompactFunctionTargetResolutionViolation.UNSUPPORTED_RECEIVER
    )


def test_import_alias_is_selected_at_its_source_position() -> None:
    repository = _repository(
        _module("first", "def consume(value): return value\n"),
        _module("second", "def consume(value): return value + 1\n"),
        _module(
            "probe",
            "from first import consume\nconsume(1)\nfrom second import consume\nconsume(2)\n",
        ),
    )
    resolutions = tuple(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe"
    )
    assert tuple(r.resolved_call.callee.identity.symbol for r in resolutions) == (
        "first.consume",
        "second.consume",
    )


def test_cyclic_reexports_remain_explicitly_unresolved() -> None:
    repository = _repository(
        _module("first", "from second import consume\n"),
        _module("second", "from first import consume\n"),
        _module("probe", "from first import consume\ndef run(): return consume(1)\n"),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is None
    assert (
        resolution.target_resolution.violation
        is CompactFunctionTargetResolutionViolation.CYCLIC_BINDING
    )


def test_reexported_class_resolves_its_method() -> None:
    repository = _repository(
        _module(
            "pkg.library",
            "class Owner:\n @staticmethod\n def consume(value): return value\n",
        ),
        _module("pkg.facade", "from .library import Owner\n"),
        _module(
            "pkg.probe",
            "from .facade import Owner\ndef run(): return Owner.consume(1)\n",
        ),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "pkg.probe.run"
    )
    assert resolution.resolved_call is not None
    assert (
        resolution.resolved_call.callee.identity.symbol == "pkg.library.Owner.consume"
    )


def test_exported_class_rebinding_does_not_authorise_old_method() -> None:
    repository = _repository(
        _module(
            "library",
            "class Owner:\n @staticmethod\n def consume(value): return value\nOwner = replacement\n",
        ),
        _module(
            "probe", "from library import Owner\ndef run(): return Owner.consume(1)\n"
        ),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is None


def test_imported_nested_class_resolves_each_namespace_binding() -> None:
    repository = _repository(
        _module(
            "library",
            "class Outer:\n class Inner:\n  @staticmethod\n  def consume(value): return value\n",
        ),
        _module(
            "probe",
            "from library import Outer\ndef run(): return Outer.Inner.consume(1)\n",
        ),
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is not None
    assert (
        resolution.resolved_call.callee.identity.symbol == "library.Outer.Inner.consume"
    )


def test_repository_joins_local_calls_constructions_and_callable_escapes() -> None:
    repository = _repository(
        _module(
            "pkg.sample",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class CacheKey:\n"
            "    left: object\n"
            "    right: object\n"
            "\n"
            "def _consume(*, left, right):\n"
            "    return left, right\n"
            "\n"
            "def wrapper(left, right):\n"
            "    key = CacheKey(left=left, right=right)\n"
            "    return _consume(left=key.left, right=key.right)\n"
            "\n"
            "escaped = _consume\n",
        )
    )

    construction = repository.resolved_product_constructions[0]
    assert construction.authority.class_symbol == "pkg.sample.CacheKey"
    assert construction.authority.field_names == ("left", "right")
    assert construction.construction.field_names == ("left", "right")
    incoming = repository.incoming_calls_for("pkg.sample._consume")
    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.sample.wrapper"
    escapes = repository.callable_escapes_for("pkg.sample._consume")
    assert len(escapes) == 1
    assert escapes[0].context.owner_symbol == "pkg.sample"


def test_repository_resolves_imported_and_qualified_function_authorities() -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class CacheKey:\n"
        "    left: object\n"
        "    right: object\n",
    )
    sink = _module(
        "pkg.sink",
        "def consume(*, left, right):\n    return left, right\n",
    )
    worker = _module(
        "pkg.worker",
        "from pkg.models import CacheKey\n"
        "from pkg.sink import consume as send\n"
        "import pkg.sink as sink\n"
        "\n"
        "def imported(left, right):\n"
        "    key = CacheKey(left=left, right=right)\n"
        "    return send(left=key.left, right=key.right)\n"
        "\n"
        "def qualified(left, right):\n"
        "    key = CacheKey(left=left, right=right)\n"
        "    return sink.consume(left=key.left, right=key.right)\n",
    )

    repository = _repository(models, sink, worker)

    assert len(repository.resolved_product_constructions) == 2
    assert {
        edge.context.owner_symbol
        for edge in repository.incoming_calls_for("pkg.sink.consume")
    } == {"pkg.worker.imported", "pkg.worker.qualified"}


def test_repository_resolves_callee_declared_return_class() -> None:
    repository = _repository(
        _module(
            "pkg.models",
            "class Carrier:\n"
            "    @classmethod\n"
            "    def merge(cls, value) -> 'Carrier':\n"
            "        return cls(value)\n",
        ),
        _module(
            "pkg.worker",
            "from pkg.models import Carrier\n"
            "\n"
            "def consume(carrier):\n"
            "    return carrier\n"
            "\n"
            "def prepare(value):\n"
            "    carrier = Carrier.merge(value)\n"
            "    return consume(carrier)\n",
        ),
    )
    call = repository.incoming_calls_for("pkg.models.Carrier.merge")[0]
    prepare = repository.flow_contexts_by_owner_symbol["pkg.worker.prepare"]
    consume_call = next(
        projected_call
        for projected_call in prepare.flow.calls
        if projected_call.target.terminal_name == "consume"
    )

    assert repository.declared_return_class_symbol_for(call) == "pkg.models.Carrier"
    assert repository.declared_bound_value_class_symbol(
        prepare,
        LexicalValueReference("carrier"),
        consume_call.position,
    ) == "pkg.models.Carrier"


def test_repository_does_not_resolve_a_rebound_import_as_a_product_constructor() -> (
    None
):
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class CacheKey:\n"
        "    left: object\n"
        "    right: object\n",
    )
    worker = _module(
        "pkg.worker",
        "from pkg.models import CacheKey\n"
        "CacheKey = replacement\n"
        "\n"
        "def make(left, right):\n"
        "    return CacheKey(left=left, right=right)\n",
    )

    repository = _repository(models, worker)

    assert repository.resolved_product_constructions == ()


@pytest.mark.parametrize(
    "runtime_opening_use",
    (
        "Product.extra = replacement\n",
        "register(Product)\n",
        "Alias = Product\n",
        "def mutate():\n    Product.extra = replacement\n",
        "def replace():\n    global Product\n    Product = replacement\n",
    ),
)
def test_repository_rejects_locally_opened_product_class_semantics(
    runtime_opening_use: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.runtime_open",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            f"{runtime_opening_use}"
            "def make(left, right):\n"
            "    return Product(left, right)\n",
        )
    )

    assert "pkg.runtime_open.Product" not in (repository.product_authorities_by_symbol)
    assert "pkg.runtime_open.Product" in (
        repository.product_runtime_failures_by_authority_symbol
    )


def test_function_local_class_names_do_not_open_the_module_product() -> None:
    repository = _repository(
        _module(
            "pkg.local_class_name",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            "def local():\n"
            "    Product = replacement\n"
            "    return Product\n",
        )
    )

    assert "pkg.local_class_name.Product" in repository.product_authorities_by_symbol


def test_repository_rejects_parameter_assignment_and_module_shadowing() -> None:
    repository = _repository(
        _module(
            "pkg.shadowed",
            "def _consume(value):\n"
            "    return value\n"
            "\n"
            "def parameter(_consume, value):\n"
            "    return _consume(value)\n"
            "\n"
            "def assignment(callback, value):\n"
            "    _consume = callback\n"
            "    return _consume(value)\n"
            "\n"
            "_consume = replacement\n"
            "\n"
            "def global_lookup(value):\n"
            "    return _consume(value)\n",
        )
    )

    assert repository.incoming_calls_for("pkg.shadowed._consume") == ()


def test_repository_resolves_dominating_nested_function_not_outer_namesake() -> None:
    repository = _repository(
        _module(
            "pkg.nested",
            "def _consume(value):\n"
            "    return value\n"
            "\n"
            "def caller(value):\n"
            "    def _consume(value):\n"
            "        return value + 1\n"
            "    return _consume(value)\n",
        )
    )

    assert len(repository.incoming_calls_for("pkg.nested.caller._consume")) == 1
    assert repository.incoming_calls_for("pkg.nested._consume") == ()


def test_repository_resolves_inherited_method_by_nominal_mro() -> None:
    repository = _repository(
        _module(
            "pkg.methods",
            "class Base:\n"
            "    def _consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Base):\n"
            "    def caller(self, value):\n"
            "        return self._consume(value)\n",
        )
    )

    incoming = repository.incoming_calls_for("pkg.methods.Base._consume")
    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.methods.Leaf.caller"


def test_repository_resolves_annotated_member_method_from_current_class() -> None:
    repository = _repository(
        _module(
            "pkg.models",
            "class Renderer:\n"
            "    def render(self, value):\n"
            "        return value\n",
        ),
        _module(
            "pkg.member",
            "from typing import ClassVar\n"
            "from pkg.models import Renderer\n"
            "\n"
            "class Owner:\n"
            "    renderer: ClassVar[Renderer]\n"
            "\n"
            "    def execute(self, value):\n"
            "        return type(self).renderer.render(value)\n",
        )
    )

    incoming = repository.incoming_calls_for("pkg.models.Renderer.render")

    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.member.Owner.execute"


@pytest.mark.parametrize("bases", ("Base", "Left, Right"))
def test_repository_resolves_annotated_member_from_native_mro(bases: str) -> None:
    repository = _repository(
        _module(
            "pkg.inherited_member",
            "class Renderer:\n"
            "    def render(self, value):\n"
            "        return value\n"
            "\n"
            "class Base:\n"
            "    renderer: Renderer\n"
            "\n"
            "class Left(Base): pass\n"
            "class Right(Base): pass\n"
            f"class Owner({bases}):\n"
            "    def execute(self, value):\n"
            "        return self.renderer.render(value)\n",
        )
    )

    incoming = repository.incoming_calls_for(
        "pkg.inherited_member.Renderer.render"
    )

    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.inherited_member.Owner.execute"


def test_repository_keeps_runtime_class_member_call_open_when_type_is_shadowed() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.shadowed_type",
            "class Renderer:\n"
            "    def render(self, value):\n"
            "        return value\n"
            "\n"
            "class Owner:\n"
            "    renderer: Renderer\n"
            "\n"
            "    def execute(self, type, value):\n"
            "        return type(self).renderer.render(value)\n",
        )
    )

    resolution = next(
        resolution
        for resolution in repository.function_call_resolutions
        if resolution.context.owner_symbol == "pkg.shadowed_type.Owner.execute"
        and resolution.call.target.terminal_name == "render"
    )

    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
    )
    assert repository.incoming_calls_for("pkg.shadowed_type.Renderer.render") == ()


def test_repository_uses_export_contract_to_exclude_builtin_shadowing() -> None:
    repository = _repository(
        _module(
            "pkg.exports",
            "__all__ = ('public_helper',)\n"
            "\n"
            "def public_helper(value):\n"
            "    return value\n",
        ),
        _module(
            "pkg.star_member",
            "from pkg.exports import *\n"
            "\n"
            "class Renderer:\n"
            "    def render(self, value):\n"
            "        return value\n"
            "\n"
            "class Owner:\n"
            "    renderer: Renderer\n"
            "\n"
            "    def execute(self, value):\n"
            "        return type(self).renderer.render(value)\n",
        ),
    )

    incoming = repository.incoming_calls_for("pkg.star_member.Renderer.render")

    assert len(incoming) == 1
    assert incoming[0].context.owner_symbol == "pkg.star_member.Owner.execute"


def test_repository_keeps_inherited_lookup_open_across_unprojectable_base() -> None:
    repository = _repository(
        _module(
            "pkg.unprojectable_base",
            "class Base:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Base, base_factory()):\n"
            "    def caller(self, value):\n"
            "        return self.consume(value)\n",
        )
    )

    resolution = next(
        resolution
        for resolution in repository.function_call_resolutions
        if resolution.context.owner_symbol == "pkg.unprojectable_base.Leaf.caller"
    )
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.INCOMPLETE_RECEIVER_FAMILY
    )
    assert repository.resolved_function_calls == ()


def test_repository_retains_missing_import_resolution_without_outer_fallback() -> None:
    repository = _repository(
        _module(
            "pkg.import_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "from missing_sink import consume\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    assert repository.resolved_function_calls == ()
    assert len(repository.function_call_resolutions) == 1
    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.MISSING_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == ("missing_sink.consume",)


def test_repository_stops_at_dynamic_enclosing_function_binding() -> None:
    repository = _repository(
        _module(
            "pkg.enclosing_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "def outer(callback):\n"
            "    consume = callback\n"
            "\n"
            "    def inner(value):\n"
            "        return consume(value)\n"
            "\n"
            "    return inner\n",
        )
    )

    resolution = next(
        resolution
        for resolution in repository.function_call_resolutions
        if resolution.context.owner_symbol == "pkg.enclosing_shadow.outer.inner"
    )
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING
    )
    assert repository.incoming_calls_for("pkg.enclosing_shadow.consume") == ()


def test_repository_resolves_multiple_inheritance_by_native_mro() -> None:
    repository = _repository(
        _module(
            "pkg.ambiguous_methods",
            "class Left:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Right:\n"
            "    def consume(self, value):\n"
            "        return value\n"
            "\n"
            "class Leaf(Left, Right):\n"
            "    def caller(self, value):\n"
            "        return self.consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0].resolved_call
    assert resolution is not None
    assert resolution.callee.identity.symbol == "pkg.ambiguous_methods.Left.consume"


@pytest.mark.parametrize(
    "write,violation",
    (
        (
            "    consume = replacement\n",
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        ),
        (
            "    consume: object = replacement\n",
            CompactFunctionTargetResolutionViolation.MISSING_DECLARATION,
        ),
        (
            "    if flag:\n        consume = replacement\n",
            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
        ),
        ("    del consume\n", CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING),
        (
            "    class consume: pass\n",
            CompactFunctionTargetResolutionViolation.DYNAMIC_BINDING,
        ),
    ),
)
@pytest.mark.parametrize("base", ("Owner", "Child"))
def test_method_selection_respects_class_body_writes(
    write: str, violation: CompactFunctionTargetResolutionViolation, base: str
) -> None:
    repository = _repository(
        _module(
            "probe",
            "class Owner:\n"
            "    @staticmethod\n"
            "    def consume(value): return value\n"
            + write
            + "class Child(Owner): pass\n"
            f"def run(): return {base}.consume(1)\n",
        )
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is None
    assert resolution.target_resolution.violation is violation
    assert "probe.Owner.consume" in resolution.target_resolution.possible_symbols


@pytest.mark.parametrize(
    "before,after",
    (
        ("    consume = replacement\n", ""),
        ("", "    consume: object\n"),
    ),
)
def test_method_selection_uses_final_class_binding(before: str, after: str) -> None:
    repository = _repository(
        _module(
            "probe",
            "class Owner:\n" + before + "    @staticmethod\n"
            "    def consume(value): return value\n"
            + after
            + "def run(): return Owner.consume(1)\n",
        )
    )
    resolution = next(
        r
        for r in repository.function_call_resolutions
        if r.context.owner_symbol == "probe.run"
    )
    assert resolution.resolved_call is not None
    assert resolution.resolved_call.callee.identity.symbol == "probe.Owner.consume"


def test_repository_rejects_duplicate_nominal_function_declaration() -> None:
    repository = _repository(
        _module(
            "pkg.duplicate",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "def consume(value):\n"
            "    return value + 1\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == ("pkg.duplicate.consume",)
    assert repository.resolved_function_calls == ()


def test_repository_keeps_star_import_function_binding_open() -> None:
    repository = _repository(
        _module(
            "pkg.star_shadow",
            "def consume(value):\n"
            "    return value\n"
            "\n"
            "from external_sink import *\n"
            "\n"
            "def caller(value):\n"
            "    return consume(value)\n",
        )
    )

    resolution = repository.function_call_resolutions[0]
    assert isinstance(resolution, CompactOpenFunctionCall)
    assert resolution.target_resolution.violation is (
        CompactFunctionTargetResolutionViolation.AMBIGUOUS_DECLARATION
    )
    assert resolution.target_resolution.possible_symbols == (
        "external_sink.consume",
        "pkg.star_shadow.consume",
    )
    assert repository.resolved_function_calls == ()


def test_repository_derives_single_inheritance_dataclass_product_fields() -> None:
    repository = _repository(
        _module(
            "pkg.models",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class BaseKey:\n"
            "    left: object\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class CacheKey(BaseKey):\n"
            "    right: object\n"
            "\n"
            "def make(left, right):\n"
            "    return CacheKey(left=left, right=right)\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.models.CacheKey"
    ].field_names == (
        "left",
        "right",
    )
    assert tuple(
        field.declaring_class_symbol
        for field in repository.product_authorities_by_symbol[
            "pkg.models.CacheKey"
        ].fields
    ) == ("pkg.models.BaseKey", "pkg.models.CacheKey")


def test_repository_derives_exact_qualified_dataclass_roles_without_classvar_mirror() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.qualified",
            "import dataclasses as dc\n"
            "from typing import ClassVar\n"
            "\n"
            "@dc.dataclass(frozen=True)\n"
            "class CacheKey:\n"
            "    cache: ClassVar[dict[str, object]] = {}\n"
            "    _: dc.KW_ONLY\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    authority = repository.product_authorities_by_symbol["pkg.qualified.CacheKey"]

    assert authority.field_names == ("left", "right")
    assert tuple(field.line for field in authority.fields) == (8, 9)


def test_repository_treats_resolved_union_annotations_as_stored_fields() -> None:
    repository = _repository(
        _module(
            "pkg.union_fields",
            "from dataclasses import dataclass\n"
            "\n"
            "class Left:\n"
            "    pass\n"
            "\n"
            "class Right:\n"
            "    pass\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: Left | None\n"
            "    right: tuple[Right, ...] | None\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.union_fields.Product"
    ].field_names == ("left", "right")


def test_repository_keeps_unresolved_union_annotation_open() -> None:
    repository = _repository(
        _module(
            "pkg.open_union_field",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: Missing | None\n"
            "    right: object\n",
        )
    )

    assert CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE in (
        _open_product_violations(repository, "pkg.open_union_field.Product")
    )


def test_repository_accepts_a_direct_nominal_dataclass_alias() -> None:
    repository = _repository(
        _module(
            "pkg.direct_alias",
            "from dataclasses import dataclass as dc\n"
            "\n"
            "@dc\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.direct_alias.Product"
    ].field_names == ("left", "right")


def test_repository_treats_explicit_object_as_a_neutral_product_base() -> None:
    repository = _repository(
        _module(
            "pkg.object_base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product(object):\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.object_base.Product"
    ].field_names == ("left", "right")


def test_repository_invalidates_dataclass_binding_after_star_import() -> None:
    repository = _repository(
        _module(
            "pkg.star_dataclass",
            "from dataclasses import dataclass\n"
            "from decorators import *\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR in (
        _open_product_violations(repository, "pkg.star_dataclass.Product")
    )


def test_repository_proves_dataclass_binding_excluded_by_star_export_contract() -> (
    None
):
    repository = _repository(
        _module("pkg.decorators", "__all__ = ('public_decorator',)\n"),
        _module(
            "pkg.proved_star_dataclass",
            "from dataclasses import dataclass\n"
            "from pkg.decorators import *\n"
            "\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n",
        ),
    )

    assert repository.product_authorities_by_symbol[
        "pkg.proved_star_dataclass.Product"
    ].field_names == ("left", "right")


def test_repository_keeps_product_construction_open_across_a_star_import() -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n",
    )
    consumer = _module(
        "pkg.star_construction",
        "from pkg.models import Product\n"
        "from external import *\n"
        "def make(left, right):\n"
        "    return Product(left=left, right=right)\n",
    )

    repository = _repository(models, consumer)

    assert repository.resolved_product_constructions == ()


@pytest.mark.parametrize(
    ("consumer_source", "expected_construction_count"),
    (
        (
            "result = Product(left=left, right=right)\n"
            "from pkg.models import Product\n",
            0,
        ),
        (
            "from pkg.models import Product\n"
            "result = Product(left=left, right=right)\n",
            1,
        ),
    ),
)
def test_repository_resolves_module_construction_from_its_execution_position(
    consumer_source: str,
    expected_construction_count: int,
) -> None:
    models = _module(
        "pkg.models",
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Product:\n"
        "    left: object\n"
        "    right: object\n",
    )
    consumer = _module("pkg.module_construction", consumer_source)

    repository = _repository(models, consumer)

    assert len(repository.resolved_product_constructions) == (
        expected_construction_count
    )


@pytest.mark.parametrize(
    "prefix",
    (
        "@dataclass\n"
        "class Product(Base):\n"
        "    left: object\n"
        "    right: object\n"
        "from pkg.base import Base\n",
        "from pkg.base import Base\n"
        "Base = replacement\n"
        "@dataclass\n"
        "class Product(Base):\n"
        "    left: object\n"
        "    right: object\n",
    ),
)
def test_repository_resolves_bases_from_the_class_declaration_position(
    prefix: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Base:\n"
            "    base: object\n",
        ),
        _module(
            "pkg.source_order",
            "from dataclasses import dataclass\n" + prefix,
        ),
    )

    assert (
        repository.class_index.classes_by_symbol[
            "pkg.source_order.Product"
        ].resolved_base_symbols
        == ()
    )
    assert CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION in (
        _open_product_violations(repository, "pkg.source_order.Product")
    )


def test_repository_composes_a_base_imported_before_the_product_declaration() -> None:
    repository = _repository(
        _module(
            "pkg.base",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Base:\n"
            "    inherited: object\n",
        ),
        _module(
            "pkg.imported_base",
            "from dataclasses import dataclass\n"
            "from pkg.base import Base\n"
            "@dataclass\n"
            "class Product(Base):\n"
            "    left: object\n"
            "    right: object\n",
        ),
    )

    assert repository.class_index.classes_by_symbol[
        "pkg.imported_base.Product"
    ].resolved_base_symbols == ("pkg.base.Base",)
    assert repository.product_authorities_by_symbol[
        "pkg.imported_base.Product"
    ].field_names == ("inherited", "left", "right")


@pytest.mark.parametrize(
    ("schema_change", "failure_line"),
    (
        ("if enabled:\n        conditional: object\n", 7),
        ("__annotations__.update(extra_annotations)\n", 6),
        ("exec(source)\n", 6),
    ),
)
def test_repository_keeps_dynamic_dataclass_field_schemas_open(
    schema_change: str,
    failure_line: int,
) -> None:
    repository = _repository(
        _module(
            "pkg.dynamic_schema",
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            f"    {schema_change}",
        )
    )

    resolution = repository.class_index.product_authority_resolutions_by_symbol[
        "pkg.dynamic_schema.Product"
    ]
    assert isinstance(resolution, OpenCompactProductAuthority)
    failures = tuple(
        failure
        for failure in resolution.failures
        if failure.violation is CompactProductAuthorityViolation.DYNAMIC_FIELD_SCHEMA
    )
    assert len(failures) == 1
    assert failures[0].line == failure_line


def test_repository_resolves_module_and_class_aliases_of_dataclass_field_roles() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.role_aliases",
            "from dataclasses import KW_ONLY, dataclass\n"
            "from typing import ClassVar\n"
            "CV = ClassVar\n"
            "KO = KW_ONLY\n"
            "@dataclass\n"
            "class Product:\n"
            "    LocalCV = CV\n"
            "    LocalKO = KO\n"
            "    cache: LocalCV[object] = None\n"
            "    _: LocalKO\n"
            "    left: object\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.role_aliases.Product"
    ].field_names == ("left", "right")


@pytest.mark.parametrize(
    "source",
    (
        (
            "from dataclasses import dataclass\n"
            "CV = external_role\n"
            "@dataclass\n"
            "class Product:\n"
            "    cache: CV[object]\n"
            "    left: object\n"
            "    right: object\n"
        ),
        (
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    CV = external_role\n"
            "    cache: CV[object]\n"
            "    left: object\n"
            "    right: object\n"
        ),
        (
            "from dataclasses import dataclass\n"
            "object = external_role\n"
            "@dataclass\n"
            "class Product:\n"
            "    cache: object[int]\n"
            "    left: int\n"
            "    right: str\n"
        ),
        (
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Product:\n"
            "    cache: Mystery[object]\n"
            "    left: object\n"
            "    right: object\n"
        ),
    ),
)
def test_repository_keeps_unresolved_dataclass_annotation_roles_open(
    source: str,
) -> None:
    repository = _repository(_module("pkg.unresolved_role", source))

    assert CompactProductAuthorityViolation.UNRESOLVED_FIELD_ROLE in (
        _open_product_violations(repository, "pkg.unresolved_role.Product")
    )


@pytest.mark.parametrize(
    "base_field",
    (
        "right: object = field(init=False)",
        "right: InitVar[object]",
    ),
)
def test_child_field_override_repairs_an_inherited_product_role(
    base_field: str,
) -> None:
    repository = _repository(
        _module(
            "pkg.override_repair",
            "from dataclasses import InitVar, dataclass, field\n"
            "@dataclass\n"
            "class Base:\n"
            "    left: object\n"
            f"    {base_field}\n"
            "@dataclass\n"
            "class Product(Base):\n"
            "    right: object\n",
        )
    )

    assert repository.product_authorities_by_symbol[
        "pkg.override_repair.Product"
    ].field_names == ("left", "right")


@pytest.mark.parametrize(
    ("class_source", "expected_violation"),
    (
        (
            "def dataclass(cls):\n    return cls\n\n"
            "@dataclass\nclass Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.UNRESOLVED_DATACLASS_DECORATOR,
        ),
        (
            "@dataclass(init=False)\n"
            "class Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.GENERATED_INIT_DISABLED,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object = field(init=False)\n",
            CompactProductAuthorityViolation.NON_INIT_FIELD,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: InitVar[object]\n",
            CompactProductAuthorityViolation.INIT_ONLY_FIELD,
        ),
        (
            "@dataclass\n"
            "class Product:\n"
            "    left: object\n"
            "    right: object\n"
            "    def __post_init__(self):\n"
            "        self.left = normalize(self.left)\n",
            CompactProductAuthorityViolation.CUSTOM_PRODUCT_LIFECYCLE,
        ),
        (
            "@wrap\n@dataclass\nclass Product:\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.CUSTOM_CLASS_DECORATOR,
        ),
        (
            "@dataclass\n"
            "class Product(metaclass=Meta):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.CUSTOM_CLASS_CREATION,
        ),
        (
            "@dataclass\n"
            "class Product(External):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
        ),
        (
            "@dataclass\n"
            "class Product(factory()):\n    left: object\n    right: object\n",
            CompactProductAuthorityViolation.INCOMPLETE_BASE_RESOLUTION,
        ),
    ),
)
def test_repository_keeps_unproved_dataclass_semantics_explicitly_open(
    class_source: str,
    expected_violation: CompactProductAuthorityViolation,
) -> None:
    repository = _repository(
        _module(
            "pkg.open_product",
            "from dataclasses import InitVar, dataclass, field\n\n" + class_source,
        )
    )

    assert expected_violation in _open_product_violations(
        repository,
        "pkg.open_product.Product",
    )
    assert "pkg.open_product.Product" not in repository.product_authorities_by_symbol


def test_repository_keeps_partial_product_construction_visible_for_proof_rejection() -> (
    None
):
    repository = _repository(
        _module(
            "pkg.partial",
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class CacheKey:\n"
            "    left: object\n"
            "    right: object = None\n"
            "\n"
            "def make(left):\n"
            "    key = CacheKey(left=left)\n"
            "    return key\n",
        )
    )

    construction = repository.resolved_product_constructions[0]
    assert construction.authority.field_names == ("left", "right")
    assert construction.construction.field_names == ("left",)
