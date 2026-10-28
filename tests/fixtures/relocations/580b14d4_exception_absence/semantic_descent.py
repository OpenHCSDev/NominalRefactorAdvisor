# Extracted from 580b14d4:nominal_refactor_advisor/semantic_descent.py


class _ProjectionVisitor(ClassFunctionStackNodeVisitor):
    def __init__(
        self,
        parsed_module: ParsedModule,
        class_index: ClassFamilyIndex | None,
        *,
        include_presentations: bool = True,
    ) -> None:
        super().__init__()
        self.parsed_module = parsed_module
        self.include_presentations = include_presentations
        self.class_reference_resolver = (
            None
            if class_index is None or not include_presentations
            else ModuleClassReferenceResolver(parsed_module, class_index)
        )
        self.deferred_class_reference_collector = (
            DeferredModuleClassReferenceCollector(parsed_module)
            if class_index is None and include_presentations
            else None
        )
        self.projections: list[PresentationProjection] = []
        self.owner_construction_stack: list[ProjectionOwnerConstructionFrame] = []
        self.type_scopes: list[ProjectionTypeScope] = []
        self.class_supplement_stack: list[CompactSemanticClassSupplementFrame] = []
        self.active_class_method_frames: list[CompactSemanticClassSupplementFrame] = []
        self._class_supplements: list[SemanticClassSupplement | None] = []

    @property
    def class_supplements(self) -> tuple[SemanticClassSupplement, ...]:
        return tuple(
            supplement
            for supplement in self._class_supplements
            if supplement is not None
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        frame: CompactSemanticClassSupplementFrame | None = None
        if not self.function_stack:
            frame = CompactSemanticClassSupplementFrame(
                node=node,
                qualname=".".join((*self.class_stack, node.name)),
                direct_method_ids=frozenset(
                    id(statement)
                    for statement in FunctionDefinition.within(node.body)
                ),
                supplement_index=len(self._class_supplements),
            )
            self._class_supplements.append(None)
            self.class_supplement_stack.append(frame)
        try:
            super().visit_ClassDef(node)
        finally:
            if frame is not None:
                self.class_supplement_stack.pop()
                self._class_supplements[frame.supplement_index] = (
                    SemanticClassSupplement.from_class_declaration(
                        f"{self.parsed_module.module_name}.{frame.qualname}",
                        node,
                        constructed_type_names=frame.constructed_type_names,
                    )
                )

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        direct_class_frames = tuple(
            frame
            for frame in self.class_supplement_stack
            if id(node) in frame.direct_method_ids
        )
        self.active_class_method_frames.extend(direct_class_frames)
        frame = (
            ProjectionOwnerConstructionFrame(
                constructions=set(),
                projection_indices=[],
            )
            if self.include_presentations
            else None
        )
        if frame is not None:
            self.owner_construction_stack.append(frame)
        self.type_scopes.append(
            ProjectionTypeScope.from_function(
                node,
                self.type_scopes[-1] if self.type_scopes else None,
            )
        )
        try:
            super().visit_FunctionDef(node)
            if self.include_presentations:
                collector = MappingReadProjectionCollector()
                bound_names = LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                    node.body
                ) | LEXICAL_SCOPE_BINDING_AUTHORITY.argument_names(node)
                for statement in node.body:
                    collector.visit(statement)
                for subject, reads in collector.reads.items():
                    keys = sorted_tuple({key for _, key, _ in reads})
                    if len(keys) < 2:
                        continue
                    common_references = set.intersection(
                        *(
                            {argument.reference_parts for argument in targets}
                            for _, _, targets in reads
                        )
                    )
                    classmethod_receiver = (
                        node.args.args[0].arg
                        if self.class_stack
                        and node.args.args
                        and any(
                            isinstance(d, ast.Name) and d.id == "classmethod"
                            for d in node.decorator_list
                        )
                        else None
                    )
                    constructors = tuple(
                        MappingReadConstructor(
                            (
                                tuple(self.class_stack)
                                if parts == (classmethod_receiver,)
                                else parts
                            ),
                            tuple(
                                sorted(
                                    {
                                        argument
                                        for _, _, targets in reads
                                        for argument in targets
                                        if argument.reference_parts == parts
                                    },
                                    key=repr,
                                )
                            ),
                        )
                        for parts in sorted(common_references)
                        if parts[0] not in bound_names
                        or (
                            parts == (classmethod_receiver,)
                            and classmethod_receiver
                            not in LEXICAL_SCOPE_BINDING_AUTHORITY.bound_names(
                                node.body
                            )
                        )
                    )
                    self.function_stack.append(node.name)
                    self._append_projection(
                        node,
                        PresentationProjectionKind.MAPPING_READ,
                        f"{node.name}:{subject}",
                        tuple(
                            PresentationToken(
                                key,
                                PresentationTokenKind.STRING_LITERAL,
                                PresentationTokenRole.DICT_KEY,
                            )
                            for key in keys
                        ),
                        mapping_read_constructors=constructors,
                    )
                    if self.class_reference_resolver is not None:
                        projection = self.projections[-1]
                        self.projections[-1] = replace(
                            projection,
                            class_symbols=projection.resolved_read_constructor_symbols(
                                lambda parts: self.class_reference_resolver.symbol_for_reference(
                                    ast.parse(".".join(parts), mode="eval").body
                                ),
                                self.class_reference_resolver.class_index,
                            ),
                        )
                    self.function_stack.pop()
        finally:
            self.type_scopes.pop()
            if frame is not None:
                self.owner_construction_stack.pop()
                owner_constructions = frozenset(frame.constructions)
                for projection_index in frame.projection_indices:
                    projection = self.projections[projection_index]
                    self.projections[projection_index] = replace(
                        projection,
                        owner_constructions=sorted_tuple(
                            frozenset(
                                (*projection.owner_constructions, *owner_constructions)
                            ),
                            key=lambda item: (item.type_name, item.field_tokens),
                        ),
                    )
            if direct_class_frames:
                del self.active_class_method_frames[-len(direct_class_frames) :]

    visit_AsyncFunctionDef = visit_FunctionDef

    def _axis_type_names_for_node(self, node: ast.AST) -> tuple[str, ...]:
        return (
            self.type_scopes[-1].projection_type_names(node) if self.type_scopes else ()
        )

    def visit_Call(self, node: ast.Call) -> None:
        if self.active_class_method_frames:
            self._record_class_construction_type_names(
                PresentationAuthorityConstructionCollector.construction_type_names(node)
            )
        if self.include_presentations:
            self._record_owner_constructions(
                PresentationAuthorityConstructionCollector.constructions_for_call(node)
            )
        self.generic_visit(node)

    def _record_class_construction_type_names(
        self,
        type_names: Iterable[str],
    ) -> None:
        type_name_tuple = tuple(type_names)
        if not type_name_tuple:
            return
        for frame in self.active_class_method_frames:
            frame.constructed_type_names.update(type_name_tuple)

    def _record_class_construction_type_names_for_node(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._record_class_construction_type_names(
                    PresentationAuthorityConstructionCollector.construction_type_names(
                        child
                    )
                )

    def _record_owner_constructions(
        self,
        constructions: tuple[PresentationAuthorityConstruction, ...],
    ) -> None:
        if not constructions:
            return
        for frame in self.owner_construction_stack:
            frame.constructions.update(constructions)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        if self._collect_assignment_projection(node, node.value):
            return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        try:
            value = StatementValue.of(node).require_value("a bare annotation")
        except AbsentStatementValue:
            self.generic_visit(node)
            return
        if self._collect_assignment_projection(
            node,
            value,
            axis_type_names=ProjectionTypeScope.annotation_type_names(node.annotation),
        ):
            return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        try:
            value = StatementValue.of(node).require_value("a bare return")
        except AbsentStatementValue:
            self.generic_visit(node)
            return
        if self._collect_return_projection(node, value):
            return
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        tokens = tuple(
            PresentationTokenProjection.tokens_for_node(
                node.test,
                PresentationTokenRole.CONDITION,
            )
        )
        if len({token.value for token in tokens}) >= 2:
            self._append_projection(
                node,
                PresentationProjectionKind.BRANCH_LITERAL,
                PresentationProjectionSiteKind.BRANCH.label_for(self.qualname),
                tokens,
                axis_type_names=self._axis_type_names_for_node(node.test),
            )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        if not self.include_presentations:
            self.generic_visit(node)
            return
        tokens: list[PresentationToken] = []
        for case in node.cases:
            tokens.extend(
                PresentationTokenProjection.tokens_for_node(
                    case.pattern,
                    PresentationTokenRole.MATCH_CASE,
                )
            )
        if len({token.value for token in tokens}) >= 2:
            self._append_projection(
                node,
                PresentationProjectionKind.MATCH_LITERAL,
                PresentationProjectionSiteKind.MATCH.label_for(self.qualname),
                tuple(tokens),
                axis_type_names=self._axis_type_names_for_node(node.subject),
            )
        self.generic_visit(node)

    def _collect_assignment_projection(
        self,
        node: ast.stmt,
        value: ast.AST,
        *,
        axis_type_names: tuple[str, ...] = (),
    ) -> bool:
        label = _assignment_label(node, owner_symbol=self.qualname)
        if ProjectionSuppressionPolicy(label).suppresses_semantic_projection():
            return False
        if SingleAssignmentAndValueNameProjection(node).pair is None:
            return False
        return self._collect_value_projection(
            node,
            value,
            label=label,
            allow_call_projection=self.current_function_name is None,
            axis_type_names=(axis_type_names or self._axis_type_names_for_node(value)),
        )

    def _collect_return_projection(self, node: ast.Return, value: ast.AST) -> bool:
        return self._collect_value_projection(
            node,
            value,
            label=PresentationProjectionSiteKind.RETURN.label_for(self.qualname),
            allow_call_projection=False,
            axis_type_names=(
                self.type_scopes[-1].projection_type_names(
                    value,
                    include_return=True,
                )
                if self.type_scopes
                else ()
            ),
        )

    def _collect_value_projection(
        self,
        node: ast.stmt,
        value: ast.AST,
        *,
        label: str,
        allow_call_projection: bool,
        axis_type_names: tuple[str, ...],
    ) -> bool:
        projection_kind = self._projection_kind(value, allow_call_projection)
        if projection_kind is None:
            return False
        key_value_pairs = (
            self._projection_key_value_pairs(value)
            if isinstance(value, ast.Dict)
            else ()
        )
        projection_constructions = self._projection_constructions(value)
        self._record_owner_constructions(projection_constructions)
        class_symbols = (
            self.class_reference_resolver.symbols_for_node(value)
            if self.class_reference_resolver is not None
            else ()
        )
        class_reference_parts = (
            self.deferred_class_reference_collector.reference_parts_for_node(value)
            if self.deferred_class_reference_collector is not None
            else ()
        )
        tokens = tuple(
            PresentationTokenProjection.tokens_for_node(
                value,
                PresentationTokenRole.COLLECTION_ITEM,
            )
        )
        if (
            self.current_function_name is not None
            and not projection_kind.admits_function_local_tokens(
                has_string_literal=any(token.kind.is_string_literal for token in tokens)
            )
        ):
            return False
        if len({token.value for token in tokens}) < 2:
            return False
        self._record_class_construction_type_names_for_node(value)
        self._append_projection(
            node,
            projection_kind,
            label,
            tokens,
            projection_constructions,
            key_value_pairs,
            class_symbols,
            class_reference_parts,
            axis_type_names,
        )
        return True

    @staticmethod
    def _projection_kind(
        value: ast.AST,
        allow_call_projection: bool,
    ) -> PresentationProjectionKind | None:
        if isinstance(value, ast.Dict):
            return PresentationProjectionKind.MAPPING_LITERAL
        if isinstance(value, ast.List | ast.Tuple | ast.Set):
            return PresentationProjectionKind.COLLECTION_LITERAL
        if isinstance(value, ast.Call) and allow_call_projection:
            return PresentationProjectionKind.CALL_LITERAL
        return None

    @staticmethod
    def _projection_constructions(
        value: ast.AST,
    ) -> tuple[PresentationAuthorityConstruction, ...]:
        return PresentationAuthorityConstructionCollector.constructions_for_projection(
            value
        )

    def _projection_key_value_pairs(
        self,
        value: ast.Dict,
    ) -> tuple[PresentationKeyValuePair, ...]:
        pairs: list[PresentationKeyValuePair] = []
        for key, item_value in zip(value.keys, value.values, strict=True):
            if key is None:
                continue
            if self.class_reference_resolver is not None:
                pairs.append(
                    PresentationKeyValuePair.from_nodes(
                        key=key,
                        value=item_value,
                        class_reference_resolver=self.class_reference_resolver,
                    )
                )
                continue
            if self.deferred_class_reference_collector is None:
                raise RuntimeError("deferred class-reference collector disappeared")
            pairs.append(
                PresentationKeyValuePair(
                    key_source=ast.unparse(key),
                    value_source=ast.unparse(item_value),
                    value_tokens=sorted_tuple(
                        {
                            token.value
                            for token in PresentationTokenProjection.tokens_for_node(
                                item_value,
                                PresentationTokenRole.DICT_VALUE,
                            )
                        }
                    ),
                    value_class_reference_parts=(
                        self.deferred_class_reference_collector.reference_parts_for_node(
                            item_value
                        )
                    ),
                )
            )
        return sorted_tuple(
            pairs,
            key=lambda item: (item.key_source, item.value_source),
        )

    def _append_projection(
        self,
        node: ast.stmt,
        kind: PresentationProjectionKind,
        label: str,
        tokens: tuple[PresentationToken, ...],
        projection_constructions: tuple[PresentationAuthorityConstruction, ...] = (),
        key_value_pairs: tuple[PresentationKeyValuePair, ...] = (),
        class_symbols: tuple[str, ...] = (),
        class_reference_parts: tuple[tuple[str, ...], ...] = (),
        axis_type_names: tuple[str, ...] = (),
        mapping_read_constructors: tuple[MappingReadConstructor, ...] = (),
    ) -> None:
        line = node.lineno
        projection_id = f"{self.parsed_module.file_path}:{line}:{self.qualname}:{kind.value}:{label}"
        self.projections.append(
            PresentationProjection(
                projection_id=projection_id,
                kind=kind,
                label=label,
                owner_symbol=self.qualname,
                location=SourceLocation(self.parsed_module.file_path, line, label),
                tokens=sorted_tuple(
                    frozenset(tokens),
                    key=lambda item: (
                        item.value,
                        item.kind,
                        item.role,
                        item.qualifier or "",
                    ),
                ),
                source_text="",
                owner_constructions=sorted_tuple(
                    frozenset(projection_constructions),
                    key=lambda item: (item.type_name, item.field_tokens),
                ),
                projection_constructions=sorted_tuple(
                    frozenset(projection_constructions),
                    key=lambda item: (item.type_name, item.field_tokens),
                ),
                key_value_pairs=key_value_pairs,
                class_symbols=class_symbols,
                class_reference_parts=class_reference_parts,
                axis_type_names=axis_type_names,
                mapping_read_constructors=mapping_read_constructors,
            )
        )
        if self.owner_construction_stack:
            self.owner_construction_stack[-1].projection_indices.append(
                len(self.projections) - 1
            )
