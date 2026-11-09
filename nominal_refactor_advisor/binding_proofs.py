from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

from nominal_refactor_advisor.ast_field_decoders import ParameterAnnotations
from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.class_index import (
    CompactModuleStarImportOrigin,
    ModuleNominalBindingAuthority,
    ModuleNominalBindingSnapshotPolicy,
    ModuleNominalBindingView,
    ModuleNominalBindingWitness,
    RepositoryPublicExposureAuthority,
    module_star_import_origins,
    nominal_reference_root_name,
)
from nominal_refactor_advisor.class_namespace import ClassNamespaceExecutionEvidence
from nominal_refactor_advisor.lexical_bindings import LEXICAL_SCOPE_BINDING_AUTHORITY
from nominal_refactor_advisor.lexical_scopes import FunctionBindingProjection
from nominal_refactor_advisor.public_exports import (
    CompactModulePublicExportContract,
    CompactPublicNameExposure,
    module_public_export_contract,
)


@dataclass(frozen=True)
class FunctionNominalParameterBindingAuthority:
    """Resolve stable parameter types from one nominal function declaration."""

    module_bindings: ModuleNominalBindingAuthority
    function: ast.FunctionDef | ast.AsyncFunctionDef

    @cached_property
    def stable_parameter_annotations(self) -> dict[str, ast.AST]:
        rebound = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.function.body)
        return {
            parameter.arg: parameter.annotation
            for parameter in ParameterAnnotations.of(
                (
                    *self.function.args.posonlyargs,
                    *self.function.args.args,
                    *self.function.args.kwonlyargs,
                )
            ).annotated
            if parameter.arg not in rebound
        }

    @cached_property
    def class_bound_names(self) -> frozenset[str]:
        parents = self.module_bindings.parsed_module.syntax_index.parent_by_node
        names: set[str] = set()
        current: ast.AST = self.function
        while (parent := parents.get(current)) is not None:
            if isinstance(parent, ast.ClassDef):
                names.update(
                    ClassNamespaceExecutionEvidence.from_class(parent).binding_names
                )
            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.update(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(parent.body))
                names.update(LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(parent))
            current = parent
        return frozenset(names)

    @cached_property
    def stable_type_names_by_parameter(self) -> dict[str, str]:
        """Project nominal instance types whose bindings remain unchanged."""
        return {
            name: type_name
            for name, annotation in self.stable_parameter_annotations.items()
            if (
                type_name := self.module_bindings.nominal_annotation_name_at(
                    annotation,
                    line=self.function.lineno,
                    class_bound_names=self.class_bound_names,
                )
            )
            is not None
        }

    def class_type_name_for_reference(self, parameter_name: str) -> str | None:
        """The represented class of a stable type[C] parameter, never an instance."""
        annotation = self.stable_parameter_annotations.get(parameter_name)
        return (
            None
            if annotation is None
            else self.module_bindings.class_annotation_name_at(
                annotation,
                line=self.function.lineno,
                class_bound_names=self.class_bound_names,
            )
        )

    def type_name_for_reference(self, parameter_name: str) -> str | None:
        """Return the declared nominal type unless the parameter is rebound."""

        return self.stable_type_names_by_parameter.get(parameter_name)

    def comprehension_owner_for_reference(
        self, reference: ast.Name
    ) -> ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp | None:
        parents = self.module_bindings.parsed_module.syntax_index.parent_by_node
        return next(
            (
                node
                for node in LEXICAL_SCOPE_BINDING_AUTHORITY.enclosing_comprehensions(
                    reference, self.function, parents
                )
                if reference.id
                in FunctionBindingProjection.from_comprehension(node).local_names
            ),
            None,
        )

    def comprehension_iterable_for_reference(
        self, reference: ast.Name
    ) -> ast.expr | None:
        """The exact target's iterable; rebinding/unpacking/async remain open."""
        owner = self.comprehension_owner_for_reference(reference)
        if owner is None or reference.id in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            (owner,)
        ):
            return None
        generators = [
            generator
            for generator in owner.generators
            if reference.id
            in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names((generator.target,))
        ]
        if len(generators) != 1:
            return None
        generator = generators[0]
        if generator.is_async or not isinstance(generator.target, ast.Name):
            return None
        return generator.iter

    def runtime_bound_names_for_reference(self, reference: ast.Name) -> frozenset[str]:
        """Names which prevent a runtime reference proving a module/builtin binding."""
        names = set(self.class_bound_names)
        names.update(LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(self.function))
        names.update(LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(self.function.body))
        parents = self.module_bindings.parsed_module.syntax_index.parent_by_node
        for scope in LEXICAL_SCOPE_BINDING_AUTHORITY.enclosing_comprehensions(
            reference, self.function, parents
        ):
            names.update(
                FunctionBindingProjection.from_comprehension(scope).local_names
            )
        return frozenset(names)

    @cached_property
    def receiver_field_annotations(self) -> dict[str, ast.AST | None]:
        """Constructor assignments, retaining unresolved writes as explicit entries.

        Nominal source evidence only; not a proof of runtime object integrity.
        """
        positional = (*self.function.args.posonlyargs, *self.function.args.args)
        if self.function.name != "__init__" or not positional:
            return {}
        receiver = positional[0].arg
        rebound = receiver in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
            self.function.body
        )
        syntax = self.module_bindings.parsed_module.syntax_index
        writes: dict[str, list[ast.AST | None]] = defaultdict(list)
        for index, target in syntax.indexed_nodes_of_type(ast.Attribute):
            if self.function not in syntax.ancestor_nodes(index):
                continue
            if not isinstance(target.ctx, (ast.Store, ast.Del)):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != receiver:
                continue
            statement = syntax.parent_node(index)
            value = None
            if statement in self.function.body and isinstance(statement, ast.Assign):
                if statement.targets == [target]:
                    value = statement.value
            writes[target.attr].append(value)
        result: dict[str, ast.AST | None] = {}
        for name, values in writes.items():
            result[name] = None
            if rebound or self.function.decorator_list:
                continue
            if len(values) == 1 and isinstance(values[0], ast.Name):
                result[name] = self.stable_parameter_annotations.get(values[0].id)
        return result

    @property
    def receiver_field_type_names(self) -> dict[str, str | None]:
        return {
            name: None
            if annotation is None
            else self.module_bindings.nominal_annotation_name_at(
                annotation,
                line=self.function.lineno,
                class_bound_names=self.class_bound_names,
            )
            for name, annotation in self.receiver_field_annotations.items()
        }

    def type_name_for_receiver_field(self, field_name: str) -> str | None:
        """Return the declared type of one stable parameter-to-field assignment."""
        return self.receiver_field_type_names.get(field_name)


@dataclass(frozen=True)
class RepositoryModuleBindingProof(
    RepositoryPublicExposureAuthority,
    ModuleNominalBindingView,
):
    """Resolve names past star imports only when source export contracts close them."""

    modules: tuple[ParsedModule, ...]

    @cached_property
    def public_export_contract_by_module_name(
        self,
    ) -> dict[str, CompactModulePublicExportContract]:
        return {
            module.module_name: module_public_export_contract(module)
            for module in self.modules
        }

    @cached_property
    def module_name_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for module in self.modules:
            counts[module.module_name] += 1
        return counts

    def contains_module(self, module_name: str) -> bool:
        return self.module_name_counts.get(module_name) == 1

    @cached_property
    def star_import_origins_by_module_name(
        self,
    ) -> dict[str, tuple[CompactModuleStarImportOrigin, ...]]:
        return {
            module.module_name: module_star_import_origins(module)
            for module in self.modules
        }

    def exposure_for(
        self,
        module_name: str,
        binding_name: str,
    ) -> CompactPublicNameExposure:
        contract = (
            self.public_export_contract_by_module_name.get(module_name)
            if self.contains_module(module_name)
            else None
        )
        return (
            CompactPublicNameExposure.UNRESOLVED
            if contract is None
            else contract.exposure_for(binding_name)
        )

    def star_import_origins_for(
        self,
        module_name: str,
    ) -> tuple[CompactModuleStarImportOrigin, ...]:
        return (
            self.star_import_origins_by_module_name[module_name]
            if self.contains_module(module_name)
            else ()
        )

    def unshadowed_builtin_witness(
        self,
        module: ParsedModule,
        name: str,
        *,
        line: int,
        preceding_class_bound_names: frozenset[str],
    ) -> ModuleNominalBindingWitness | None:
        if not (
            self.star_imports_exclude(module.module_name, name)
            and ModuleNominalBindingAuthority(module)
            .snapshot_before(
                line,
                policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
            )
            .resolves_unshadowed_builtin(
                name,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        ):
            return None
        return ModuleNominalBindingWitness(f"builtins.{name}", name)

    def reference_witness_at(
        self,
        module: ParsedModule,
        reference: ast.expr,
        *,
        line: int,
    ) -> ModuleNominalBindingWitness | None:
        root_name = nominal_reference_root_name(reference)
        if root_name is None or not self.star_imports_exclude(
            module.module_name,
            root_name,
        ):
            return None
        qualified_name = ModuleNominalBindingAuthority(module).qualified_name_at(
            reference,
            line=line,
            policy=ModuleNominalBindingSnapshotPolicy.NAMED_IMPORT_PROJECTION,
        )
        if qualified_name is None:
            return None
        return ModuleNominalBindingWitness(qualified_name, root_name)
