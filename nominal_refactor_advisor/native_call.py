"""Native call declarations own execution, returned values and installation separately."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import CodeType, FunctionType
from typing import (
    ClassVar,
    TYPE_CHECKING,
    cast,
)

from .ast_tools import SourceModule
from .captured_reference import (
    AdmittedExecutionPrefixABC,
    CapturedReferenceKernel,
    CapturedReferenceResolution,
    CapturedReferenceViolation,
    CreatedNamespaceDictionary,
    InitialNativeIsland,
    NamespaceContentsABC,
    NamespaceEvidenceABC,
    NamespaceMemberInventory,
    NativeTypeCapture,
    OpaqueCapturedObjectOperations,
    OpenCapturedReference,
    CompletedSourceOperation,
    SourceDefinitionApplicationAuthorityABC,
)
from .call_binding import (
    CompactCallArgument,
    CompactCallBinding,
    CompactFunctionParameter,
    CompactFunctionSignature,
)
from .descriptor_algebra import AliasProperty
from .lexical_bindings import FunctionParameterSource
from .native_class_mro import NativeClassMroDeclaration
from .native_compilation import (
    NativeCreationBackend,
    NativePythonCompilation,
)
from .native_declarations import (
    DataclassRuntimeDeclaration,
    NativeConstantContentsABC,
    NativeDeclaration,
    NativeDeclarationFamily,
    NativeParameterDefault,
    NativeScalar,
)
from .native_subscription import NativeArgumentInspection
from .product_flow import (
    CompactBranchPredicateResolverABC,
    CompactCallableReferenceUse,
    CompactFlowContext,
    CompactFunctionFlow,
    CompactFunctionCall,
    compact_function_flow_projection,
    CompactValueUse,
    SourceFlowOperation,
)
from .scan_cache import ScanCache
from .source_geometry import SourceByteSpan

if TYPE_CHECKING:
    from .native_reference import NativeReferenceEnvironment


@dataclass(frozen=True, eq=False)
class CallAuthority(CompletedSourceOperation):
    """One invocation, retaining its original source operation and activation."""

    node = AliasProperty[ast.Call]("operation.node")
    call = AliasProperty[CompactFunctionCall]("operation.event")

    def require_inert_native_binding(self, declaration: NativeDeclaration) -> None:
        """Bind an explicitly proved value-independent native protocol with inert values.

        This never evaluates source arguments or provides returned-object identity.
        Callers must separately prove real argument effects and the chosen protocol.
        """
        if any(
            argument.is_unpacked
            for argument in (
                *self.call.arguments.positional,
                *self.call.arguments.keywords,
            )
        ):
            raise ValueError("Native argument expansion remains unproved")
        try:
            declaration.declaration(
                *(None for argument in self.call.arguments.positional),
                **{keyword.name: None for keyword in self.call.arguments.keywords},
            )
        except TypeError as error:
            raise ValueError("Native argument binding rejected") from error

    @classmethod
    def for_call(
        cls,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> CallAuthority:
        return cls(environment, environment.source_operation(context, call))

    @cached_property
    def callee(self) -> CapturedReferenceResolution:
        return self.environment.kernel._read_use(
            self.call.target_use, self.context, frozenset()
        )

    def require_operation_kind(self) -> None:
        if not isinstance(self.operation.node, ast.Call) or not isinstance(
            self.operation.event, CompactFunctionCall
        ):
            raise TypeError("Call requires an actual invocation operation")
        if not any(call is self.call for call in self.context.flow.calls):
            raise ValueError("Call is absent from its actual flow")


class SignatureCallAuthorityABC(CallAuthority):
    """An original invocation bound by its actual Python signature declaration.

    Exact parameter binding and evaluated arguments are shared obligations.
    They supply neither body execution nor a returned-object identity.
    """

    @property
    @abstractmethod
    def signature(self) -> CompactFunctionSignature:
        raise NotImplementedError

    @property
    def bound_arguments(self) -> CompactCallBinding[CompactValueUse]:
        self.require_original_operation()
        binding = self.signature.bind(
            self.call.arguments.positional, self.call.arguments.keywords
        )
        if not binding.is_exact:
            raise ValueError(f"Argument binding rejected: {binding.violation}")
        for value in self.call.arguments.values:
            self.environment.kernel._read_use(
                value, self.context, frozenset()
            ).require_closed()
        return binding


@dataclass(frozen=True, eq=False)
class NativeCallAuthority(SignatureCallAuthorityABC, NativeDeclarationFamily):
    """One native invocation, retaining its canonical original operation only."""

    @property
    def signature(self) -> CompactFunctionSignature:
        return CompactFunctionSignature.from_arguments(
            self.python_definition.args
        ).with_default_names(
            frozenset(default.parameter_name for default in self.python_defaults)
        )

    @property
    def python_defaults(self) -> tuple[NativeParameterDefault, ...]:
        return NativeParameterDefault.from_function(self.declaration.declaration)

    @property
    def python_definition(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Derive Python syntax from the current implementation, never its name."""
        function = self.declaration.declaration
        if type(function) is not FunctionType:
            raise ValueError("Python implementation requires an exact function")
        return NativePythonCompilation.from_function(function).function_definition(
            function
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        _ = self.declaration

    @cached_property
    def declaration(self) -> NativeDeclaration:
        return self.callee.require_native(self.native_declarations)

    @classmethod
    def for_call(
        cls,
        environment: NativeReferenceEnvironment,
        context: CompactFlowContext,
        call: CompactFunctionCall,
    ) -> NativeCallAuthority:
        operation = environment.source_operation(context, call)
        callee = environment.kernel._read_use(call.target_use, context, frozenset())
        return cls.select_from_capture(callee)(environment, operation)


class DefaultObjectConstruction(CallAuthority):
    """Default object construction shared by source and native class evidence."""

    native_constructor = NativeDeclaration(object)
    construction_hooks = tuple(
        NativeDeclaration(hook)
        for hook in (
            native_constructor.declaration.__new__,
            native_constructor.declaration.__init__,
        )
    )
    installation_hooks = (NativeDeclaration(property.__set_name__),)

    @abstractmethod
    def require_instance_hooks(self, hooks: tuple[NativeDeclaration, ...]) -> None:
        """Require default instance behavior on this invocation's actual class."""
        raise NotImplementedError

    def require_closed(self) -> None:
        self.require_instance_hooks(self.construction_hooks)
        self.require_inert_native_binding(self.native_constructor)

    def result(self) -> CapturedReferenceResolution:
        self.require_closed()
        return self

    def require_class_installation(self) -> None:
        self.require_closed()
        self.require_instance_hooks(self.installation_hooks)


class NativeObjectConstruction(
    NativeTypeCapture, DefaultObjectConstruction, NativeCallAuthority
):
    """Exact builtin object allocation, not a source subclass or analyzer identity."""

    native_declarations = (DefaultObjectConstruction.native_constructor,)
    native_type = DefaultObjectConstruction.native_constructor.declaration

    def require_instance_hooks(self, hooks: tuple[NativeDeclaration, ...]) -> None:
        self.callee.require_native(self.native_declarations)
        NativeCreationBackend.current().require_static_type_mro(self.native_type)
        declaration = NativeClassMroDeclaration(self.native_type)
        for hook in hooks:
            name = hook.declaration.__name__
            owner = declaration.member_owner(name)
            if (
                owner is not None
                and declaration.stored_namespace(owner)[name] is not hook.declaration
            ):
                raise ValueError("Native object hook differs from the default protocol")


@dataclass(frozen=True, eq=False)
class NativePythonFunctionSource:
    """Current function code joined to its exact compact source-flow declaration."""

    function: FunctionType
    compilation: NativePythonCompilation
    definition: ast.FunctionDef | ast.AsyncFunctionDef

    @classmethod
    def from_function(cls, function: FunctionType) -> NativePythonFunctionSource:
        if type(function) is not FunctionType:
            raise ValueError("Native source flow requires an exact Python function")
        compilation = NativePythonCompilation.from_function(function)
        code_contents = NativeCreationBackend.current().code_contents(function.__code__)
        return cls._from_current_function(function, compilation, code_contents)

    @staticmethod
    @ScanCache.cached
    def _from_current_function(
        function: FunctionType,
        compilation: NativePythonCompilation,
        code_contents: tuple[bytes, tuple[int, ...]],
    ) -> NativePythonFunctionSource:
        if (
            NativeCreationBackend.current().code_contents(function.__code__)
            != code_contents
        ):
            raise ValueError("Native function code changed during source capture")
        return NativePythonFunctionSource(
            function,
            compilation,
            compilation.function_definition(function),
        )

    @cached_property
    def flow(self) -> CompactFunctionFlow:
        parsed = SourceModule.from_source_path(
            Path(self.compilation.file_path), self.compilation.source
        ).parse()
        source_span = SourceByteSpan.require_node(self.definition)
        return compact_function_flow_projection(parsed, source_span)


@dataclass(frozen=True)
class NativeReturnedClosureFactorySource:
    """Current-source proof of one inert branch returning a fresh closure.

    This proof admits only an exact synchronous Python function whose selected
    ``parameter is None`` path creates and returns one undecorated closure. All
    other parameters must be captured by that returned closure, so frame cleanup
    cannot destroy an argument. The closure body remains entirely unexecuted.
    """

    function: FunctionType
    definition: ast.FunctionDef

    @classmethod
    def from_function(
        cls, function: FunctionType
    ) -> NativeReturnedClosureFactorySource:
        source = NativePythonFunctionSource.from_function(function)
        if not isinstance(source.definition, ast.FunctionDef):
            raise ValueError("Native closure factory requires a synchronous function")
        return cls(function, source.definition)

    @property
    def body(self) -> tuple[ast.stmt, ...]:
        statements = tuple(self.definition.body)
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and type(statements[0].value.value) is str
        ):
            statements = statements[1:]
        return statements

    @property
    def closure_definition(self) -> ast.FunctionDef:
        if len(self.body) != 3 or not isinstance(self.body[0], ast.FunctionDef):
            raise ValueError(
                "Native factory has no unique returned closure declaration"
            )
        closure = self.body[0]
        if (
            closure.decorator_list
            or closure.args.defaults
            or any(default is not None for default in closure.args.kw_defaults)
            or closure.returns is not None
            or any(
                argument.annotation is not None
                for argument in (
                    *closure.args.posonlyargs,
                    *closure.args.args,
                    *closure.args.kwonlyargs,
                )
            )
            or closure.args.vararg is not None
            or closure.args.kwarg is not None
            or (sys.version_info >= (3, 12) and closure.type_params)
        ):
            raise ValueError("Returned closure creation has executable header inputs")
        return closure

    @property
    def selected_parameter_name(self) -> str:
        branch = self.body[1]
        closure = self.closure_definition
        if (
            not isinstance(branch, ast.If)
            or branch.orelse
            or len(branch.body) != 1
            or not isinstance(branch.body[0], ast.Return)
            or not isinstance(branch.body[0].value, ast.Name)
            or branch.body[0].value.id != closure.name
            or not isinstance(branch.test, ast.Compare)
            or not isinstance(branch.test.left, ast.Name)
            or len(branch.test.ops) != 1
            or not isinstance(branch.test.ops[0], ast.Is)
            or len(branch.test.comparators) != 1
            or not isinstance(branch.test.comparators[0], ast.Constant)
            or branch.test.comparators[0].value is not None
        ):
            raise ValueError("Native factory has no inert None-selected return branch")
        return branch.test.left.id

    @property
    def closure_code(self) -> CodeType:
        closure = self.closure_definition
        candidates = tuple(
            constant
            for constant in self.function.__code__.co_consts
            if isinstance(constant, CodeType)
            and constant.co_name == closure.name
            and constant.co_firstlineno == closure.lineno
        )
        if len(candidates) != 1:
            raise ValueError("Returned closure has no unique native code declaration")
        return candidates[0]

    def require_returned_closure(self, parameter_name: str) -> None:
        """Prove closure creation/retention for the selected declaration parameter."""
        if parameter_name != self.selected_parameter_name:
            raise ValueError("Native factory selector differs from its declaration")
        closure = self.closure_definition
        fallback = self.body[2]
        if (
            not isinstance(fallback, ast.Return)
            or not isinstance(fallback.value, ast.Call)
            or fallback.value.keywords
            or len(fallback.value.args) != 1
            or not isinstance(fallback.value.func, ast.Name)
            or fallback.value.func.id != closure.name
            or not isinstance(fallback.value.args[0], ast.Name)
            or fallback.value.args[0].id != parameter_name
        ):
            raise ValueError(
                "Native factory fallback differs from its closure protocol"
            )
        parameters = FunctionParameterSource.from_arguments(self.definition.args)
        parameter_names = frozenset(parameter.argument.arg for parameter in parameters)
        retained_parameter_names = parameter_names - {parameter_name}
        code = self.function.__code__
        if (
            frozenset(code.co_cellvars) != retained_parameter_names
            or frozenset(self.closure_code.co_freevars) != retained_parameter_names
            or frozenset(code.co_varnames) != parameter_names | {closure.name}
        ):
            raise ValueError("Returned closure does not retain the factory parameters")


class NativeDefinitionApplicationAuthorityABC(NativeDeclarationFamily, ABC):
    """Nominal native semantics for one original definition application."""

    @classmethod
    def for_application(
        cls,
        application: SourceDefinitionApplicationAuthorityABC,
        callee: CapturedReferenceResolution,
    ) -> NativeDefinitionApplicationAuthorityABC:
        authority = cls.select_from_capture(callee)
        return authority(application)

    @abstractmethod
    def result(self) -> CapturedReferenceResolution:
        raise NotImplementedError

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        """Require transformation-specific preservation of the class-base protocol."""
        raise ValueError("Definition application may change the class-base protocol")


@dataclass(frozen=True, eq=False)
class NativeDataclassDefinitionApplicationABC(
    CompactBranchPredicateResolverABC,
    ABC,
):
    """Current dataclass source proving one in-place class transformation."""

    application: SourceDefinitionApplicationAuthorityABC

    factory_declaration: ClassVar[NativeDeclaration] = (
        DataclassRuntimeDeclaration.DATACLASS.native_declaration
    )
    processor_declaration: ClassVar[NativeDeclaration] = NativeDeclaration(
        dataclasses._process_class
    )

    @cached_property
    def factory_source(self) -> NativeReturnedClosureFactorySource:
        return NativeReturnedClosureFactorySource.from_function(
            cast(FunctionType, self.factory_declaration.declaration)
        )

    @cached_property
    def processor_source(self) -> NativePythonFunctionSource:
        return NativePythonFunctionSource.from_function(
            cast(FunctionType, self.processor_declaration.declaration)
        )

    @cached_property
    def factory_parameters(self) -> tuple[CompactFunctionParameter, ...]:
        return CompactFunctionSignature.from_arguments(
            self.factory_source.definition.args
        ).parameters

    @cached_property
    def processor_parameters(self) -> tuple[CompactFunctionParameter, ...]:
        declaration = self.processor_source.flow.owner.declaration
        if declaration is None:
            raise ValueError("Native dataclass processor has no function declaration")
        return declaration.signature.parameters

    @cached_property
    def selector_parameter(self) -> CompactFunctionParameter:
        name = self.factory_source.selected_parameter_name
        matches = tuple(
            parameter for parameter in self.factory_parameters if parameter.name == name
        )
        if len(matches) != 1:
            raise ValueError("Native dataclass selector has no unique parameter")
        return matches[0]

    @abstractmethod
    def explicit_parameter_value(
        self, parameter: CompactFunctionParameter
    ) -> NativeConstantContentsABC | None:
        """Return an explicitly supplied factory value or no supplied value."""
        raise NotImplementedError

    def parameter_value(
        self, parameter: CompactFunctionParameter
    ) -> NativeConstantContentsABC:
        if parameter.name == self.selector_parameter.name:
            return self.application.argument
        explicit = self.explicit_parameter_value(parameter)
        if explicit is not None:
            return explicit
        defaults = tuple(
            default
            for default in NativeParameterDefault.from_function(
                cast(FunctionType, self.factory_declaration.declaration)
            )
            if default.parameter_name == parameter.name
        )
        if len(defaults) != 1:
            raise ValueError("Native dataclass parameter has no unique default")
        return defaults[0]

    @staticmethod
    def _require_boolean(value: NativeConstantContentsABC) -> bool:
        for expected in (False, True):
            try:
                value.require_constant_contents(expected)
            except ValueError:
                continue
            return expected
        raise ValueError("Native dataclass option is not an exact Boolean")

    def proves_boolean(
        self,
        predicate_use: CompactCallableReferenceUse,
        expected: bool,
    ) -> bool:
        binding = self.processor_source.flow.initial_parameter_binding_for_predicate(
            predicate_use
        )
        if binding is None:
            return False
        parameter = next(
            (
                candidate
                for candidate in self.factory_parameters
                if candidate.name == binding.parameter.name
            ),
            None,
        )
        if parameter is None:
            return False
        try:
            self.parameter_value(parameter).require_constant_contents(expected)
        except ValueError:
            return False
        return True

    def require_factory_processor_call(self) -> None:
        source = self.factory_source
        selector = self.selector_parameter
        source.require_returned_closure(selector.name)
        closure = source.closure_definition
        closure_parameters = FunctionParameterSource.from_arguments(closure.args)
        if (
            len(closure_parameters) != 1
            or closure_parameters[0].argument.arg != selector.name
            or len(closure.body) != 1
            or not isinstance(closure.body[0], ast.Return)
            or not isinstance(closure.body[0].value, ast.Call)
        ):
            raise ValueError("Native dataclass closure has an unproved call body")
        call = closure.body[0].value
        if (
            call.keywords
            or not isinstance(call.func, ast.Name)
            or call.func.id != self.processor_declaration.declaration.__name__
            or any(not isinstance(argument, ast.Name) for argument in call.args)
        ):
            raise ValueError("Native dataclass closure calls another processor")
        argument_names = tuple(cast(ast.Name, argument).id for argument in call.args)
        factory_names = tuple(parameter.name for parameter in self.factory_parameters)
        processor_names = tuple(
            parameter.name for parameter in self.processor_parameters
        )
        if argument_names != factory_names or processor_names != factory_names:
            raise ValueError(
                "Native dataclass processor binding differs from its factory"
            )
        binding = CompactFunctionSignature(self.processor_parameters).bind(
            tuple(
                CompactCallArgument(parameter) for parameter in self.factory_parameters
            ),
            (),
        )
        if not binding.is_exact or any(
            bound is None
            or len(bound.values) != 1
            or bound.values[0] is not factory_parameter
            for processor_parameter, factory_parameter in zip(
                self.processor_parameters,
                self.factory_parameters,
                strict=True,
            )
            for bound in (binding.argument_for(processor_parameter.name),)
        ):
            raise ValueError("Native dataclass processor call binding is not exact")

    def require_processor_at_application(self) -> None:
        environment = self.application.environment
        namespace = environment.initial.namespace_for_storage(vars(dataclasses))
        value = NamespaceMemberInventory(
            environment.kernel,
            namespace,
            self.application.application_prefix,
        ).require_member(self.processor_declaration.declaration.__name__)
        value.require_native_identity(self.processor_declaration)

    def require_factory_activation(self) -> None:
        """Require any earlier factory call before using its captured options."""

    def result(self) -> CapturedReferenceResolution:
        self.application.require_original_application()
        self.application.argument.require_definition_application_argument()
        self.require_factory_activation()
        self.require_factory_processor_call()
        self.require_processor_at_application()
        for parameter in self.factory_parameters:
            if parameter.name != self.selector_parameter.name:
                self._require_boolean(self.parameter_value(parameter))
        binding = self.processor_source.flow.require_returned_parameter_binding(
            self.selector_parameter.name,
            self,
        )
        if binding.parameter.name != self.selector_parameter.name:
            raise ValueError("Native dataclass processor returned another parameter")
        return self.application.argument

    def require_plain_class_base(
        self,
        resolver: CapturedReferenceKernel,
        context: CompactFlowContext,
        position: CompactFlowPosition,
    ) -> type:
        """Preserve the input class's inheritance protocol for in-place dataclasses."""
        _ = self.result()
        return self.application.argument.require_plain_class_base(
            resolver,
            context,
            position,
        )


@dataclass(frozen=True, eq=False)
class NativeDataclassDefinitionApplication(
    NativeDataclassDefinitionApplicationABC,
    NativeDefinitionApplicationAuthorityABC,
):
    """Direct ``@dataclass`` application selected by captured function identity."""

    native_declarations = (DataclassRuntimeDeclaration.DATACLASS.native_declaration,)

    def explicit_parameter_value(self, parameter: CompactFunctionParameter) -> None:
        del parameter
        return None


class NativeDataclassFactoryCall(NativeTypeCapture, NativeCallAuthority):
    """Create a dataclass decorator, without claiming its later application.

    Binding and the returned-closure proof use the current exact declaration,
    never a placeholder call into Python code or an independently maintained
    keyword schema. The later decorator application remains separate.
    """

    native_declarations = (DataclassRuntimeDeclaration.DATACLASS.native_declaration,)
    native_type = FunctionType

    def definition_application_authority(
        self, application: SourceDefinitionApplicationAuthorityABC
    ) -> NativeDefinitionApplicationAuthorityABC:
        return NativeDataclassFactoryDefinitionApplication(application, self)

    def require_closed(self) -> None:
        binding = self.bound_arguments
        declaration = self.python_definition
        if not isinstance(declaration, ast.FunctionDef):
            raise ValueError(
                "Native factory requires a synchronous function declaration"
            )
        # The native protocol distinguishes its first positional input by None.
        # Its spelling, default, and remaining options belong to the declaration.
        parameter = FunctionParameterSource.from_arguments(declaration.args)[0]
        argument = binding.argument_for(parameter.argument.arg)
        if argument is None:
            defaults = tuple(
                default
                for default in self.python_defaults
                if default.parameter_name == parameter.argument.arg
            )
            if len(defaults) != 1:
                raise ValueError("Native dataclass factory default remains unproved")
            defaults[0].require_constant_contents(None)
        else:
            self.environment.kernel._read_use(
                argument.values[0], self.context, frozenset()
            ).require_constant_contents(None)
        NativeReturnedClosureFactorySource(
            self.declaration.declaration, declaration
        ).require_returned_closure(parameter.argument.arg)


@dataclass(frozen=True, eq=False)
class NativeDataclassFactoryDefinitionApplication(
    NativeDataclassDefinitionApplicationABC
):
    """Application of the exact decorator returned by a dataclass factory call."""

    factory: NativeDataclassFactoryCall

    def require_factory_activation(self) -> None:
        self.factory.require_closed()

    def explicit_parameter_value(
        self, parameter: CompactFunctionParameter
    ) -> NativeConstantContentsABC | None:
        argument = self.factory.bound_arguments.argument_for(parameter.name)
        if argument is None:
            return None
        if len(argument.values) != 1:
            raise ValueError("Native dataclass option has no unique source argument")
        return self.factory.environment.kernel._read_use(
            argument.values[0], self.factory.context, frozenset()
        )


class NativeDescriptorArgumentABC(ABC):
    """Creation evidence for metadata inspected by native descriptor wrapping."""

    @abstractmethod
    def require_descriptor_argument(self) -> None:
        raise NotImplementedError


class NativeDescriptorResult(
    NativeDescriptorArgumentABC, OpaqueCapturedObjectOperations
):
    """Native descriptor creation and installation, without runtime identity."""

    native_declarations = tuple(
        NativeDeclaration(native) for native in (classmethod, property, staticmethod)
    )
    violation = CapturedReferenceViolation.UNPROVED_ACCESS

    def result(self) -> CapturedReferenceResolution:
        self.require_closed()
        return self

    def require_descriptor_argument(self) -> None:
        self.require_closed()

    def require_class_installation(self) -> None:
        self.require_closed()


class NativeDescriptorCall(NativeDescriptorResult, NativeCallAuthority):
    def require_closed(self) -> None:
        if any(keyword.is_unpacked for keyword in self.call.arguments.keywords):
            raise ValueError("Native keyword expansion remains unproved")
        inspection = NativeArgumentInspection(self.environment)
        for argument in (
            *self.node.args,
            *(keyword.value for keyword in self.node.keywords),
        ):
            inspection.visit(argument)
        self.require_inert_native_binding(self.declaration)


class NativeDictionaryResultCall(NativeCallAuthority, ABC):

    def require_closed(self) -> None:
        """Admit the input namespace before borrowing or copying its dictionary."""
        self.namespace()

    def namespace(self) -> NamespaceEvidenceABC:
        namespace = self._namespace()
        if isinstance(namespace, OpenCapturedReference):
            namespace.require_closed()
        namespace = cast(NamespaceEvidenceABC, namespace)
        namespace.require_admitted(self.environment.kernel.initial)
        return namespace

    @abstractmethod
    def _namespace(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        """Select the namespace consumed by the exact native call protocol."""
        raise NotImplementedError

    def _single_argument(self) -> CapturedReferenceResolution:
        arguments = self.call.arguments.positional
        if len(arguments) != 1 or arguments[0].is_unpacked:
            raise ValueError("Native namespace call requires one explicit object")
        return self.environment.kernel._read_use(
            arguments[0].value, self.context, frozenset()
        )


class NativeBorrowedNamespaceCall(NativeDictionaryResultCall, ABC):
    """A native call returning an already admitted namespace's dictionary."""

    def result(self) -> CapturedReferenceResolution:
        return self.namespace().captured_dictionary()


class NativeVarsCall(NativeBorrowedNamespaceCall):
    native_declarations = (NativeDeclaration(builtins.vars),)

    def _namespace(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        if self.call.arguments.keywords:
            raise ValueError("Native vars does not accept keyword arguments")
        return self._single_argument().object_namespace(self.environment.kernel.initial)


class NativeGlobalsCall(NativeBorrowedNamespaceCall):
    native_declarations = (NativeDeclaration(builtins.globals),)

    def _namespace(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        if self.call.arguments.positional or self.call.arguments.keywords:
            raise ValueError("Native globals requires no arguments")
        return self.prefix.endpoint.frame.globals


class CopiedNativeNamespace(
    CreatedNamespaceDictionary, NativeDictionaryResultCall, NamespaceContentsABC
):
    """The actual copy invocation owns its fresh dictionary, not a result-table wrapper."""

    kernel = AliasProperty[CapturedReferenceKernel]("environment.kernel")
    initial = AliasProperty[InitialNativeIsland]("environment.kernel.initial")

    @property
    def parent(self) -> NamespaceEvidenceABC:
        return self.namespace()

    def result(self) -> CapturedReferenceResolution:
        self.require_closed()
        return self

    names = AliasProperty[frozenset[NativeScalar]]("initial_names")

    @classmethod
    def from_call(
        cls,
        environment: NativeReferenceEnvironment,
        operation: SourceFlowOperation,
    ) -> CopiedNativeNamespace:
        """Require the canonical result of an independently admitted native copy."""
        context = environment.context_for_owner(operation.owner)
        authority = environment.call_authority(context, operation.event)
        if (
            not isinstance(authority, NativeDictCopyCall)
            or authority.operation is not operation
        ):
            raise ValueError("Selected call is not an admitted native dictionary copy")
        authority.require_closed()
        result = environment.capture_value(operation.node)
        if not isinstance(result, cls):
            raise ValueError("Selected call has no canonical copied namespace")
        result.require_closed()
        return result

    @cached_property
    def initial_names(self) -> frozenset[NativeScalar]:
        self.require_admitted(self.initial)
        return self.kernel.namespace_names(
            self.parent, self.context, self.call.position
        ) | frozenset(argument.name for argument in self.call.arguments.keywords)

    def require_available(
        self,
        kernel: CapturedReferenceKernel,
        prefix: AdmittedExecutionPrefixABC,
    ) -> None:
        self.require_admitted(kernel.initial)
        if kernel is not self.kernel:
            raise ValueError("Copied namespace belongs to another execution kernel")
        prefix.require_event(self.context, self.call, self.prefix.endpoint.frame)

    def require_admitted(self, initial: InitialNativeIsland) -> None:
        if (
            initial is not self.initial
            or self.environment.call_authority(self.context, self.call) is not self
        ):
            raise ValueError(
                "Copied namespace has no canonical admitted call activation"
            )
        self.namespace()

    def _member(self, key: NativeScalar) -> CapturedReferenceResolution | None:
        for argument in self.call.arguments.keywords:
            if argument.name == key:
                return self.kernel._read_use(argument.value, self.context, frozenset())
        return self.kernel._slot(
            self.parent, key, self.context, self.call.position, frozenset()
        )


class NativeDictCopyCall(CopiedNativeNamespace):
    native_declarations = (NativeDeclaration(builtins.dict),)

    def _namespace(self) -> NamespaceEvidenceABC | OpenCapturedReference:
        keywords = self.call.arguments.keywords
        if any(argument.is_unpacked for argument in keywords) or len(
            {argument.name for argument in keywords}
        ) != len(keywords):
            raise ValueError("Native dict copy requires explicit unique keyword values")
        return self._single_argument().dictionary_namespace(
            self.environment.kernel.initial
        )
