# Extracted from 580b14d4:nominal_refactor_advisor/ast_projection.py


class AbsentStatementValue(ValueError):
    """A bare annotation or bare `return` supplies no value."""
