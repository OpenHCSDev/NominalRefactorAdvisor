# Extracted from 674a5ed9:nominal_refactor_advisor/ast_field_decoders.py


def declared_annotations(
    node: ast.arg | ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    """The annotation a parameter or function declares: none when unannotated."""

    match node:
        case (
            ast.arg(annotation=ast.expr() as annotation)
            | ast.FunctionDef(returns=ast.expr() as annotation)
            | ast.AsyncFunctionDef(returns=ast.expr() as annotation)
        ):
            return (annotation,)
    return ()
