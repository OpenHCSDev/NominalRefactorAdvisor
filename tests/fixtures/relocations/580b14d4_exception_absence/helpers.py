# Extracted from 580b14d4:nominal_refactor_advisor/detectors/_helpers.py


def _guarded_delegator_candidates_for_class(
    module: ParsedModule, node: ast.ClassDef
) -> Iterable[GuardedDelegatorCandidate]:
    if not _is_observation_spec_class(node) or CLASS_NODE_AUTHORITY.is_abstract(node):
        return
    for statement in node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if statement.name not in {
            "build_from_function",
            "build_from_assign",
            "build_from_observation",
            "build_from_context",
        }:
            continue
        body = statements_without_docstring(statement.body)
        while body and isinstance(body[0], ast.Assign):
            body = body[1:]
        if len(body) != 2:
            continue
        guard, return_stmt = body
        if not isinstance(
            guard, ast.If
        ) or not HELPER_SUPPORT_PROJECTION_AUTHORITY.if_returns_none_only(guard):
            continue
        if not isinstance(return_stmt, ast.Return):
            continue
        try:
            returned = StatementValue.of(return_stmt).require_value("a bare return")
        except AbsentStatementValue:
            continue
        delegate_name = _delegate_name_from_return(returned)
        if delegate_name is None:
            continue
        yield GuardedDelegatorCandidate(
            file_path=module.file_path,
            line=statement.lineno,
            subject_name=node.name,
            name_family=(
                guard.test.__class__.__name__,
                delegate_name,
                _scope_role_name(guard.test),
            ),
            method_name=statement.name,
            guard_role=_guard_role_name(guard.test),
            delegate_name=delegate_name,
            scope_role=_scope_role_name(guard.test),
        )


def _linear_query_return_expr(loop: ast.For, result_name: str) -> ast.AST | None:
    return_exprs: list[ast.expr] = []
    for current in _walk_nodes(loop):
        if not isinstance(current, ast.Return):
            continue
        try:
            return_exprs.append(
                StatementValue.of(current).require_value("a bare return")
            )
        except AbsentStatementValue:
            continue
    if len(return_exprs) != 1 or not _name_mentions(return_exprs[0], result_name):
        return None
    return return_exprs[0]


def _schema_accessor_fetch_assignment(
    statement: ast.stmt,
) -> tuple[str, str, str, str] | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            return None
        target_name = statement.targets[0].id
        fetch = _schema_accessor_fetch_call(statement.value)
    elif isinstance(statement, ast.AnnAssign) and isinstance(
        statement.target, ast.Name
    ):
        try:
            value = StatementValue.of(statement).require_value("a bare annotation")
        except AbsentStatementValue:
            return None
        target_name = statement.target.id
        fetch = _schema_accessor_fetch_call(value)
    else:
        return None
    if fetch is None:
        return None
    fetch_mode, enum_name, field_name = fetch
    return target_name, fetch_mode, enum_name, field_name


def _schema_accessor_returns_local(
    function: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> bool:
    for node in _walk_nodes(function):
        if not isinstance(node, ast.Return):
            continue
        try:
            returned = StatementValue.of(node).require_value("a bare return")
        except AbsentStatementValue:
            continue
        if _node_mentions_name(returned, local_name):
            return True
    return False
