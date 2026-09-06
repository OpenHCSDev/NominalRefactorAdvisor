"""Share the native C3 carrier across source and loaded declaration representations."""

from dataclasses import replace
import json
from textwrap import dedent

from nominal_refactor_advisor.codemod import (
    AddClassBaseOperation,
    CodemodPlanSequence,
    EnsureImportOperation,
    InsertBeforeTargetOperation,
    InsertAfterTargetOperation,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    ReplaceModuleAssignmentOperation,
    ReplaceScopeAssignmentOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.json_reports import json_report_object

native = SourceRewriteTarget(
    file_path="nominal_refactor_advisor/native_declarations.py"
)
source = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_index.py")
mro = SourceRewriteTarget(file_path="nominal_refactor_advisor/class_mro.py")
PLAN = CodemodPlanSequence.from_operations(
    (
        EnsureImportOperation(
            target=native, import_source="from abc import ABC, abstractmethod"
        ),
        InsertBeforeTargetOperation(
            target=replace(native, qualname="NativeDeclaration"),
            source=dedent('''\
                class QualifiedDeclaration(ABC):
                    """A declaration with a qualified source name, independent of representation."""

                    @property
                    @abstractmethod
                    def qualified_name(self) -> str:
                        raise NotImplementedError


                class ClassNamespaceDeclaration(QualifiedDeclaration):
                    """Names whose class-level binding must be accounted for in member lookup."""

                    @property
                    @abstractmethod
                    def member_binding_names(self) -> frozenset[str]:
                        raise NotImplementedError


                '''),
        ),
        AddClassBaseOperation(
            target=replace(native, qualname="NativeDeclaration"),
            base_name="QualifiedDeclaration",
        ),
        *(
            EnsureImportOperation(
                target=target,
                import_source="from .native_declarations import QualifiedDeclaration",
            )
            for target in (source, mro)
        ),
        EnsureImportOperation(
            target=source, import_source="from .descriptor_algebra import AliasProperty"
        ),
        AddClassBaseOperation(
            target=replace(source, qualname="ClassDeclaration"),
            base_name="QualifiedDeclaration",
        ),
        InsertBeforeTargetOperation(
            target=replace(
                source, qualname="ClassDeclaration.with_resolved_base_symbols"
            ),
            source='    qualified_name = AliasProperty[str]("symbol")\n\n',
        ),
        EnsureImportOperation(
            target=mro, import_source="from typing import Generic, TypeVar"
        ),
        InsertAfterTargetOperation(
            target=replace(mro, qualname="NativeMroBase.for_qualified_name"),
            source="""
    @classmethod
    def for_python_type(cls, python_type: type) -> NativeMroBase | None:
        return next(
            (member for member in cls if member.python_type is python_type), None
        )
""",
        ),
        InsertBeforeTargetOperation(
            target=replace(mro, qualname="DeclarationMroType"),
            source='MroDeclarationT = TypeVar("MroDeclarationT", bound=QualifiedDeclaration)\n\n\n',
        ),
        AddClassBaseOperation(
            target=replace(mro, qualname="DeclarationMroType"),
            base_name="Generic[MroDeclarationT]",
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(mro, qualname="DeclarationMroType"),
            assignment_name="declaration",
            source="declaration: MroDeclarationT",
        ),
        ReplaceFunctionSignatureOperation(
            target=replace(mro, qualname="DeclarationMroType.from_declaration"),
            signature_suffix="(cls, declaration: MroDeclarationT, bases: tuple[type, ...]) -> DeclarationMroType[MroDeclarationT]:",
        ),
        PatchTargetOperation(
            target=replace(mro, qualname="DeclarationMroType.from_declaration"),
            replacements=(
                SourceTextReplacement(
                    old_source="declaration.symbol",
                    new_source="declaration.qualified_name",
                ),
            ),
        ),
        ReplaceFunctionSignatureOperation(
            target=replace(mro, qualname="DeclarationMroType.declarations"),
            signature_suffix="(self) -> tuple[MroDeclarationT, ...]:",
        ),
        ReplaceScopeAssignmentOperation(
            target=replace(mro, qualname="ResolvedClassMro"),
            assignment_name="declaration_type",
            source="declaration_type: DeclarationMroType[CompactIndexedClass]",
        ),
        *(
            ReplaceFunctionSignatureOperation(
                target=replace(mro, qualname=f"{owner}.mro_type"),
                signature_suffix=f"(self) -> DeclarationMroType[CompactIndexedClass]{suffix}:",
            )
            for owner, suffix in (
                ("ClassMroResolution", " | None"),
                ("ResolvedClassMro", ""),
            )
        ),
        EnsureImportOperation(
            target=source,
            import_source="from .native_declarations import ClassNamespaceDeclaration",
        ),
        EnsureImportOperation(
            target=source,
            import_source="from .class_namespace import ClassNamespaceExecutionEvidence, NATIVE_METHOD_DECORATORS",
        ),
        AddClassBaseOperation(
            target=replace(source, qualname="IndexedClass"),
            base_name="ClassNamespaceDeclaration",
        ),
        InsertBeforeTargetOperation(
            target=replace(source, qualname="IndexedClass.is_final"),
            source="""    member_binding_names = AliasProperty[frozenset[str]]("namespace_execution.binding_names")

    @cached_property
    def namespace_execution(self) -> ClassNamespaceExecutionEvidence:
        return ClassNamespaceExecutionEvidence.from_class(self.node)

""",
        ),
        ReplaceModuleAssignmentOperation(
            target=source,
            source="_PROMOTABLE_METHOD_DECORATOR_NAMES = frozenset(declaration.__name__ for declaration in NATIVE_METHOD_DECORATORS)",
        ),
        InsertBeforeTargetOperation(
            target=replace(
                source, qualname="ModuleNominalBindingView.require_native_type_in_class"
            ),
            source="""    def reference_or_builtin_witness_at(
        self, module: ParsedModule, reference: ast.expr, *, line: int,
        preceding_class_bound_names: frozenset[str] = frozenset(),
    ) -> ModuleNominalBindingWitness | None:
        root = nominal_reference_root_name(reference)
        if root in preceding_class_bound_names:
            return None
        witness = self.reference_witness_at(module, reference, line=line)
        if witness is None and isinstance(reference, ast.Name):
            return self.unshadowed_builtin_witness(
                module, reference.id, line=line,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        return witness

""",
        ),
        ReplaceFunctionBodyOperation(
            target=replace(
                source, qualname="ModuleNominalBindingView.require_native_type_in_class"
            ),
            body_source=dedent('''\
                """Prove the native type emitted into a class resolves to its declaration."""
                name = declaration.__name__
                class_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(owner.body)
                if name in class_names:
                    raise ValueError(f"Class namespace shadows native type {name!r}")
                witness = self.reference_or_builtin_witness_at(
                    module, ast.Name(id=name, ctx=ast.Load()), line=owner.lineno,
                    preceding_class_bound_names=class_names,
                )
                qualified_name = NativeDeclaration(declaration).qualified_name
                if witness is None or witness.qualified_name != qualified_name:
                    raise ValueError(
                        f"Class creation does not prove native type binding {qualified_name!r}"
                    )
                '''),
        ),
    )
)

if __name__ == "__main__":
    print(json.dumps(json_report_object(PLAN), indent=2))
