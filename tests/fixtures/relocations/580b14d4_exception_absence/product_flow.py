# Extracted from 580b14d4:nominal_refactor_advisor/product_flow.py


class _CompactFlowCollector(
    EagerFunctionAnnotationVisitor,
    DictionaryEvaluationVisitor,
    VariableAnnotationVisitorABC,
):
    """Collect one source scope without descending into nested scope bodies."""

    def visit_Tuple(self, node: ast.Tuple) -> CompactTupleValue | None:
        if not isinstance(node.ctx, ast.Load) or any(
            isinstance(element, ast.Starred) for element in node.elts
        ):
            self.generic_visit(node)
            return None
        return CompactTupleValue(
            tuple(self._capture_value(element) for element in node.elts)
        )

    def visit_Constant(self, node: ast.Constant) -> CompilerStoredValue | None:
        if (
            self.documentation_statement is not None
            and node is self.documentation_statement.value
        ):
            return CompilerStoredValue()
        return None

    def _visit_assignment_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        result: CompactEvaluatedResult | None,
    ) -> tuple[CompactMutation, ...]:
        previous = self.assignment_result
        self.assignment_result = (
            (targets[0], result)
            if result is not None
            and len(targets) == 1
            and isinstance(targets[0], (ast.Name, ast.Attribute, ast.Subscript))
            else None
        )
        try:
            return self._visit_mutation_targets(targets, CompactMutationKind.ASSIGNMENT)
        finally:
            self.assignment_result = previous

    def _cursor_position(self) -> CompactFlowPosition:
        """Snapshot the shared event cursor without inventing a source event."""
        return CompactFlowPosition(
            self.branch_path,
            self.statement_index,
            self.event_index,
            self.evaluation_path,
        )

    def _record_operation(
        self, node: SourcePositionedNode, event: SourceFlowEvent
    ) -> None:
        """Task collectors can retain this exact generated operation without replay."""

    def visit_unordered_annotations(self, roots: tuple[ast.expr, ...]) -> None:
        """Preserve each root's order without ordering sibling root evaluations."""
        parent_path = self.evaluation_path
        parent_slot = self.event_index
        try:
            for member, expression in enumerate(roots):
                self.evaluation_path = (
                    *parent_path,
                    CompactEvaluationBranch(parent_slot, member),
                )
                self.event_index = 0
                self.visit_annotation(expression)
        finally:
            self.evaluation_path = parent_path
            self.event_index = parent_slot + 1

    def _capture_result(
        self,
        expression: ast.expr | None,
        destination: CompactValueDestination,
        statement: SourcePositionedNode,
    ) -> CompactEvaluatedResult:
        if isinstance(expression, ast.Call):
            self.call_results[id(expression)] = destination
        value_use = None if expression is None else self._capture_value(expression)
        return self._record_evaluated_result(value_use, destination, statement)

    def _record_evaluated_result(
        self,
        value_use: CompactValueUse | None,
        destination: CompactValueDestination,
        statement: SourcePositionedNode,
    ) -> CompactEvaluatedResult:
        result = CompactEvaluatedResult(
            destination,
            value_use,
            self._position(),
            SourceByteSpan.require_node(statement),
        )
        self.evaluated_results.append(result)
        self._record_operation(statement, result)
        return result

    def visit_Subscript(self, node: ast.Subscript) -> SubscriptionResultValue | None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
            return None
        receiver = self._capture_value(node.value)
        argument = self._capture_value(node.slice)
        invocation = CompactSubscription(
            receiver_use=receiver,
            argument_use=argument,
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self.subscriptions.append(invocation)
        self._record_operation(node, invocation)
        return SubscriptionResultValue(invocation)

    def _record_target_mutation(
        self,
        node: ast.expr,
        kind: CompactMutationKind | None = None,
    ) -> None:
        target = self.mutation_targets.visit(node)
        result = (
            self.assignment_result[1]
            if self.assignment_result is not None and self.assignment_result[0] is node
            else None
        )
        self._record_mutation(target, node, kind, result=result)

    def _capture_definition_input(self, expression: ast.expr) -> None:
        self.definition_input_uses.append(self._capture_value(expression))

    visit_annotation = _capture_definition_input

    def _capture_value(self, expression: ValueExpressionNode) -> CompactValueUse:
        value = self.visit(expression)
        if value is None:
            value = CompactValueExpression.project(expression)
        use = CompactValueUse(value, self._position())
        self._record_operation(expression, use)
        return use

    def __init__(
        self,
        *,
        owner: CompactFlowOwner,
        definition_owners: dict[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            CompactDefinitionFlowOwner,
        ],
        module_identity: PythonModulePathIdentity,
        annotation_mode: type[ModuleAnnotationEvaluationMode],
        lexical_scope_qualnames: tuple[str, ...],
        current_class_qualname: str | None,
        current_class_receiver_name: str | None,
    ) -> None:
        self.owner = owner
        self.definition_owners = definition_owners
        self.module_identity = module_identity
        self.annotation_mode = annotation_mode
        self.lexical_scope_qualnames = lexical_scope_qualnames
        self.current_class_qualname = current_class_qualname
        self.current_class_receiver_name = current_class_receiver_name
        self.calls: list[CompactFunctionCall] = []
        self.subscriptions: list[CompactSubscription] = []
        self.native_captures: list[CompactNativeCapture] = []
        self.evaluated_results: list[CompactEvaluatedResult] = []
        self.definition_input_uses: list[CompactValueUse] = []
        self.callable_reference_uses: list[CompactCallableReferenceUse] = []
        self.mutations: list[CompactMutation] = []
        self.exact_value_aliases: list[CompactExactValueAlias] = []
        self.global_binding_names: set[str] = set()
        self.nonlocal_binding_names: set[str] = set()
        self.branch_path: tuple[CompactControlBranch, ...] = ()
        self.statement_index = 0
        self.event_index = 0
        self.evaluation_path: tuple[CompactEvaluationBranch, ...] = ()
        self.call_results: dict[int, CompactValueDestination] = {}
        self.mutation_kind = CompactMutationKind.ASSIGNMENT
        self.mutation_targets = _CompactMutationTargetCollector(self)
        self.assignment_result: tuple[ast.expr, CompactEvaluatedResult] | None = None

    def collect(self, statements: list[ast.stmt]) -> CompactFunctionFlow:
        self.documentation_statement = self.owner.kind.documentation_statement(
            statements
        )
        self._collect_statements(statements)
        return CompactFunctionFlow(
            owner=self.owner,
            lexical_scope_qualnames=self.lexical_scope_qualnames,
            calls=tuple(self.calls),
            subscriptions=tuple(self.subscriptions),
            native_captures=tuple(self.native_captures),
            evaluated_results=tuple(self.evaluated_results),
            callable_reference_uses=tuple(self.callable_reference_uses),
            mutations=tuple(self.mutations),
            exact_value_aliases=tuple(self.exact_value_aliases),
            global_binding_names=tuple(sorted(self.global_binding_names)),
            nonlocal_binding_names=tuple(sorted(self.nonlocal_binding_names)),
        )

    def _collect_statements(self, statements: list[ast.stmt]) -> None:
        saved_statement_index = self.statement_index
        saved_event_index = self.event_index
        for statement_index, statement in enumerate(statements):
            self.statement_index = statement_index
            self.event_index = 0
            self.visit(statement)
        self.statement_index = saved_statement_index
        self.event_index = saved_event_index

    def _collect_branch(
        self,
        statements: list[ast.stmt],
        kind: CompactControlBranchKind,
        alternative_index: int = 0,
        predicate_use: CompactCallableReferenceUse | None = None,
    ) -> None:
        saved_path = self.branch_path
        self.branch_path = (
            *saved_path,
            CompactControlBranch(
                self.statement_index,
                kind,
                alternative_index,
                predicate_use,
            ),
        )
        self._collect_statements(statements)
        self.branch_path = saved_path

    def _position(self) -> CompactFlowPosition:
        position = self._cursor_position()
        self.event_index += 1
        return position

    def _record_mutation(
        self,
        target: CompactAssignmentTargetABC,
        node: SourcePositionedNode,
        kind: CompactMutationKind | None = None,
        *,
        result: CompactEvaluatedResult | None = None,
    ) -> CompactMutation:
        factory = (
            CompactMutation
            if result is None
            else partial(CompactEvaluatedAssignment, result=result)
        )
        mutation = factory(
            target=target,
            kind=self.mutation_kind if kind is None else kind,
            position=self._position(),
            line=node.lineno,
        )
        self.mutations.append(mutation)
        self._record_operation(node, mutation)
        return mutation

    def _call_target(self, expression: ast.expr) -> CompactCallTargetReference:
        if isinstance(expression, ast.Name):
            return BareCallTargetReference(expression.id)
        member_method = CurrentClassMemberMethodReference.from_expression(
            expression,
            owner_class_qualname=self.current_class_qualname,
            receiver_name=self.current_class_receiver_name,
        )
        if member_method is not None:
            return member_method
        reference = LexicalValueReference.from_expression(expression)
        if reference is None and isinstance(expression, ast.Attribute):
            receiver = expression.value
            return ReceiverAttributeCallTargetReference(
                expression.attr,
                (
                    self._call_target(receiver.func)
                    if isinstance(receiver, ast.Call)
                    else None
                ),
                receiver.lineno,
            )
        if reference is None:
            return DynamicCallTargetReference()
        if (
            self.current_class_qualname is not None
            and self.current_class_receiver_name is not None
            and reference.root_name == self.current_class_receiver_name
            and len(reference.attribute_path) == 1
        ):
            return CurrentClassMethodReference(
                self.current_class_receiver_name,
                self.current_class_qualname,
                reference.terminal_name,
            )
        return QualifiedCallTargetReference(reference)

    def _callable_reference_use(self, node: ast.expr) -> CompactCallableReferenceUse:
        use = CompactCallableReferenceUse(
            target=self._call_target(node),
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self._record_operation(node, use)
        return use

    def visit_Call(self, node: ast.Call) -> CallResultValue:
        self._visit_reference_evaluation(node.func)
        target_use = self._callable_reference_use(node.func)
        arguments = CompactCallArguments[CompactValueUse].from_call(
            node, self._capture_value
        )
        result = self.call_results.get(
            id(node), CompactValueDestination(CompactValueDestinationKind.EMBEDDED)
        )
        invocation = CompactFunctionCall(
            target_use=target_use,
            arguments=arguments,
            result=result,
            position=self._position(),
            source_span=SourceByteSpan.require_node(node),
        )
        self.calls.append(invocation)
        self._record_operation(node, invocation)
        return CallResultValue(invocation)

    def _visit_reference_evaluation(self, expression: ast.expr) -> None:
        """Evaluate a reference's receiver and indices before its terminal access."""
        if isinstance(expression, ast.Attribute):
            self.visit(expression.value)
        elif not isinstance(expression, ast.Name):
            self.visit(expression)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
            return
        self._visit_reference_evaluation(node)
        self.callable_reference_uses.append(self._callable_reference_use(node))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._record_target_mutation(node)
        elif isinstance(node.ctx, ast.Load):
            self.callable_reference_uses.append(self._callable_reference_use(node))

    def visit_Assign(self, node: ast.Assign) -> None:
        result = self._capture_result(
            node.value, CompactValueDestination.for_assignment(node.targets), node
        )
        mutations = self._visit_assignment_targets(node.targets, result)
        self._record_exact_value_aliases(
            node.targets, result.lexical_reference, mutations
        )

    def _visit_annotated_assignment(self, node: ast.AnnAssign) -> None:
        result = None
        try:
            value = StatementValue.of(node).require_value("a bare annotation")
        except AbsentStatementValue:
            self.mutation_targets.visit(node.target)
            if not (
                isinstance(node.target, ast.Name) and self.owner.kind.is_function_scope
            ):
                return
        else:
            result = self._capture_result(
                value, CompactValueDestination.for_assignment((node.target,)), node
            )
        mutations = self._visit_assignment_targets((node.target,), result)
        self._record_exact_value_aliases(
            (node.target,),
            None if result is None else result.lexical_reference,
            mutations,
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotated_assignment(node)
        self.owner.kind.visit_assignment_annotation(self, node, self.annotation_mode)

    def _capture_compiler_operand(
        self, node: ast.AnnAssign, role: NativeItemStoreOperand
    ) -> CompactCompilerOperandUse:
        use = CompactCompilerOperandUse(CompilerOperandValue(role), self._position())
        self._record_operation(node, use)
        return use

    def _store_variable_annotation(
        self, node: ast.AnnAssign, result: CompactEvaluatedResult
    ) -> None:
        if not isinstance(node.target, ast.Name):
            raise ValueError("Simple variable annotation requires a name target")
        receiver = self._capture_compiler_operand(node, NativeItemStoreOperand.RECEIVER)
        key = self._capture_compiler_operand(node, NativeItemStoreOperand.KEY)
        self._record_mutation(
            CompactVariableAnnotationTarget(receiver, key, node.target.id),
            node,
            result=result,
        )

    def visit_eager_variable_annotation(self, node: ast.AnnAssign) -> None:
        if node.simple:
            result = self._capture_result(
                node.annotation,
                CompactValueDestination(CompactValueDestinationKind.EMBEDDED),
                node.annotation,
            )
            self._store_variable_annotation(node, result)
        else:
            self.visit(node.annotation)

    def visit_stringized_variable_annotation(self, node: ast.AnnAssign) -> None:
        if node.simple:
            use = self._capture_compiler_operand(node, NativeItemStoreOperand.VALUE)
            result = self._record_evaluated_result(
                use,
                CompactValueDestination(CompactValueDestinationKind.EMBEDDED),
                node.annotation,
            )
            self._store_variable_annotation(node, result)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        target = self.mutation_targets.visit(node.target)
        self.callable_reference_uses.append(self._callable_reference_use(node.target))
        self.visit(node.value)
        self._record_mutation(
            target,
            node.target,
            CompactMutationKind.AUGMENTED_ASSIGNMENT,
        )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ForwardedResultValue:
        result = self._capture_result(
            node.value, CompactValueDestination.for_assignment((node.target,)), node
        )
        self._visit_assignment_targets((node.target,), result)
        return ForwardedResultValue(result)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._visit_mutation_targets(node.targets, CompactMutationKind.DELETION)

    def visit_Return(self, node: ast.Return) -> None:
        self._capture_result(
            node.value,
            CompactValueDestination(CompactValueDestinationKind.RETURNED),
            node,
        )

    def visit_Expr(self, node: ast.Expr) -> None:
        destination = (
            CompactValueDestination(
                CompactValueDestinationKind.BOUND,
                LexicalValueReference(
                    CPythonClassConstructionField.DOCUMENTATION.value
                ),
            )
            if node is self.documentation_statement
            else CompactValueDestination(CompactValueDestinationKind.DISCARDED)
        )
        result = self._capture_result(node.value, destination, node)
        if destination.direct_binding_name is not None:
            self._record_mutation(
                CompactBindingTarget(destination.direct_binding_name),
                node,
                result=result,
            )

    def _visit_mutation_targets(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        kind: CompactMutationKind,
    ) -> tuple[CompactMutation, ...]:
        mutation_start = len(self.mutations)
        saved_kind = self.mutation_kind
        self.mutation_kind = kind
        try:
            for target in targets:
                self.visit(target)
        finally:
            self.mutation_kind = saved_kind
        return tuple(self.mutations[mutation_start:])

    def _is_exact_value_alias_assignment(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        source: LexicalValueReference | None,
    ) -> bool:
        return bool(
            source is not None
            and targets
            and all(
                isinstance(target, ast.Name)
                and (
                    not self.owner.kind.is_function_scope
                    or target.id not in self.global_binding_names
                    and target.id not in self.nonlocal_binding_names
                )
                for target in targets
            )
        )

    def _record_exact_value_aliases(
        self,
        targets: tuple[ast.expr, ...] | list[ast.expr],
        source: LexicalValueReference | None,
        mutations: tuple[CompactMutation, ...],
    ) -> None:
        if not self._is_exact_value_alias_assignment(targets, source):
            return
        source_use = self.callable_reference_uses[-1]
        assert source_use.target.lexical_reference == source
        self.exact_value_aliases.extend(
            CompactExactValueAlias(
                source_use=source_use,
                binding_mutation=mutation,
            )
            for mutation in mutations
        )

    def visit_Global(self, node: ast.Global) -> None:
        self.global_binding_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_binding_names.update(node.names)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        for origin in ImportBoundNameProjection(node).origins(self.module_identity):
            self._record_mutation(
                CompactImportTarget(origin), node, CompactMutationKind.IMPORT
            )

    visit_ImportFrom = visit_Import

    def _capture_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> tuple[CompactValueUse, ...]:
        return tuple(
            self._capture_value(decorator) for decorator in node.decorator_list
        )

    def _bind_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        decorator_uses: tuple[CompactValueUse, ...],
        input_start: int,
    ) -> None:
        self._record_mutation(
            CompactDefinitionTarget(
                self.definition_owners[node],
                decorator_uses,
                tuple(self.definition_input_uses[input_start:]),
                self._position(),
            ),
            node,
            CompactMutationKind.DEFINITION,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        input_start = len(self.definition_input_uses)
        self._bind_definition(
            node, self._visit_definition_expressions(node), input_start
        )

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[CompactValueUse, ...]:
        decorator_uses = self._capture_decorators(node)
        for default in self.default_roots(node.args):
            self._capture_definition_input(default)
        if self.annotation_mode.annotations_execute_at_declaration:
            self.visit_function_annotations(node)
        return decorator_uses

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        input_start = len(self.definition_input_uses)
        decorator_uses = self._capture_decorators(node)
        declaration = cast(CompactClassDeclaration, self.definition_owners[node])
        declaration.capture.resolve(_ClassHeaderCaptureProjection(self, node))
        for base in node.bases:
            self._capture_definition_input(base)
        for keyword in node.keywords:
            self._capture_definition_input(keyword.value)
        self._bind_definition(node, decorator_uses, input_start)

    def _direct_predicate_use(
        self, expression: ast.expr
    ) -> CompactCallableReferenceUse | None:
        """Retain the exact direct-name read used by one control predicate."""
        read_start = len(self.callable_reference_uses)
        self.visit(expression)
        reads = tuple(self.callable_reference_uses[read_start:])
        if (
            not isinstance(expression, ast.Name)
            or len(reads) != 1
            or reads[0].source_span != SourceByteSpan.require_node(expression)
        ):
            return None
        return reads[0]

    def visit_If(self, node: ast.If) -> None:
        predicate_use = self._direct_predicate_use(node.test)
        self._collect_branch(
            node.body,
            CompactControlBranchKind.IF_BODY,
            predicate_use=predicate_use,
        )
        self._collect_branch(
            node.orelse,
            CompactControlBranchKind.IF_ELSE,
            predicate_use=predicate_use,
        )

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_mutation_targets(
            (node.target,), CompactMutationKind.ITERATION_BINDING
        )
        self._collect_branch(node.body, CompactControlBranchKind.LOOP_BODY)
        self._collect_branch(node.orelse, CompactControlBranchKind.LOOP_ELSE)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._collect_branch(node.body, CompactControlBranchKind.LOOP_BODY)
        self._collect_branch(node.orelse, CompactControlBranchKind.LOOP_ELSE)

    def visit_Try(self, node: ast.Try) -> None:
        self._collect_branch(node.body, CompactControlBranchKind.TRY_BODY)
        for index, handler in enumerate(node.handlers):
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._record_mutation(
                    CompactBindingTarget(handler.name),
                    handler,
                    CompactMutationKind.EXCEPTION_BINDING,
                )
            self._collect_branch(
                handler.body,
                CompactControlBranchKind.TRY_HANDLER,
                index,
            )
        self._collect_branch(node.orelse, CompactControlBranchKind.TRY_ELSE)
        self._collect_branch(node.finalbody, CompactControlBranchKind.TRY_FINALLY)

    visit_TryStar = visit_Try

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_mutation_targets(
                    (item.optional_vars,), CompactMutationKind.CONTEXT_BINDING
                )
        self._collect_branch(node.body, CompactControlBranchKind.WITH_BODY)

    visit_AsyncWith = visit_With

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for index, case in enumerate(node.cases):
            for name in _match_bound_names(case.pattern):
                self._record_mutation(
                    CompactBindingTarget(name),
                    case.pattern,
                    CompactMutationKind.PATTERN_BINDING,
                )
            if case.guard is not None:
                self.visit(case.guard)
            self._collect_branch(
                case.body,
                CompactControlBranchKind.MATCH_CASE,
                index,
            )
