from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    FunctionNominalParameterBindingAuthority,
    ModuleNominalBindingAuthority,
)
from nominal_refactor_advisor.exact_field_authority import (
    ExactDataclassFieldAuthorityComponentBuilder,
)


def _module(source: str) -> ParsedModule:
    path = Path("pkg/models.py")
    return ParsedModule(
        path=path,
        module_name="pkg.models",
        is_package_init=False,
        module=ast.parse(source, filename=str(path)),
        source=source,
    )


def _source(
    *,
    beta_identity_fields: str = "    module_name: str\n    file_path: str\n",
    decorator: str = "@dataclass(frozen=True)",
    alpha_method: str = "",
) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
        f"{decorator}\n"
        "class AlphaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    alpha_value: int\n"
        f"{alpha_method}"
        "\n\n"
        f"{decorator}\n"
        "class BetaProjection:\n"
        f"{beta_identity_fields}"
        "    beta_value: float\n"
        "\n\n"
        f"{decorator}\n"
        "class GammaProjection:\n"
        "    module_name: str\n"
        "    file_path: str\n"
        "    gamma_value: bytes\n"
    )


def test_builder_derives_repeated_leading_fields_as_one_component() -> None:
    module = _module(_source())
    builder = ExactDataclassFieldAuthorityComponentBuilder.from_modules((module,))

    assert len(builder.proven_components) == 1
    component = builder.proven_components[0]
    assert component.field_names == ("module_name", "file_path")
    assert component.evidence_field_name == "file_path"
    assert component.participant_class_names == (
        "AlphaProjection",
        "BetaProjection",
        "GammaProjection",
    )
    assert (
        builder.required_component_for_field(
            file_path=module.file_path,
            class_qualname="BetaProjection",
            field_name="file_path",
        )
        is component
    )


def test_builder_rejects_field_order_changes() -> None:
    module = _module(
        _source(
            beta_identity_fields="    file_path: str\n    module_name: str\n",
        )
    )

    component = ExactDataclassFieldAuthorityComponentBuilder.from_modules(
        (module,)
    ).proven_components[0]
    assert component.participant_class_names == (
        "AlphaProjection",
        "GammaProjection",
    )


def test_builder_rejects_defaulted_fields() -> None:
    module = _module(
        _source(
            beta_identity_fields=(
                "    module_name: str = 'pkg'\n    file_path: str = 'models.py'\n"
            ),
        )
    )

    component = ExactDataclassFieldAuthorityComponentBuilder.from_modules(
        (module,)
    ).proven_components[0]
    assert component.participant_class_names == (
        "AlphaProjection",
        "GammaProjection",
    )


def test_builder_rejects_dataclass_options_without_proved_inheritance_semantics() -> (
    None
):
    module = _module(_source(decorator="@dataclass(frozen=True, slots=True)"))

    assert (
        ExactDataclassFieldAuthorityComponentBuilder.from_modules(
            (module,)
        ).proven_components
        == ()
    )


def test_builder_rejects_methods_that_observe_the_old_mro() -> None:
    module = _module(
        _source(
            alpha_method=(
                "\n    def describe(self) -> str:\n"
                "        return super().__repr__()\n"
            ),
        )
    )

    component = ExactDataclassFieldAuthorityComponentBuilder.from_modules(
        (module,)
    ).proven_components[0]
    assert component.participant_class_names == (
        "BetaProjection",
        "GammaProjection",
    )


def test_module_binding_authority_resolves_many_positions_in_one_projection() -> None:
    module = _module(
        "from first import Value\n\n"
        "class First:\n"
        "    pass\n\n"
        "from second import Value\n\n"
        "class Second:\n"
        "    pass\n"
    )
    snapshots = ModuleNominalBindingAuthority(module).snapshots_before((3, 8))

    assert snapshots[3].binding_for("Value").qualified_name == "first.Value"
    assert snapshots[8].binding_for("Value").qualified_name == "second.Value"


def test_function_parameter_binding_authority_requires_stable_nominal_types() -> None:
    module = _module(
        "class Target:\n"
        "    pass\n\n\n"
        "TargetAlias = Target\n\n\n"
        "def consume(stable: TargetAlias, rebound: Target, untyped):\n"
        "    rebound = untyped\n"
        "    return stable, rebound\n"
    )
    function = next(
        node for node in module.module.body if isinstance(node, ast.FunctionDef)
    )
    authority = FunctionNominalParameterBindingAuthority(
        ModuleNominalBindingAuthority(module),
        function,
    )

    assert authority.stable_type_names_by_parameter == {
        "stable": "pkg.models.Target"
    }
    assert authority.type_name_for_reference("rebound") is None
    assert authority.type_name_for_reference("untyped") is None
