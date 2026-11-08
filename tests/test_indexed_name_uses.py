"""Name-use answers consume the parsed source's existing syntax views."""

import ast
from pathlib import Path
from unittest.mock import patch

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.false_distinction_observations import NameUse, NameUseKind


def test_name_uses_keep_order_and_parent_classification_without_rewalking(tmp_path: Path) -> None:
    (tmp_path / "uses.py").write_text(
        "from m import Reexport as Reexport\n"
        "Target()\n"
        "raise Target\n"
        "try: pass\n"
        "except Target: pass\n"
        "try: pass\n"
        "except (Target, pkg.Other): pass\n"
        "with pytest.raises(Target): pass\n"
        "Target\n"
        "'Target StringWord'\n"
        "# Target VisibleWord\n"
        "pkg.Target()\n",
        encoding="utf-8",
    )
    (module,) = parse_python_modules(tmp_path, use_parse_cache=False)
    index = module.syntax_index
    # Both views already belong to the parse; collection must not rebuild either.
    index.parent_by_node
    with patch("ast.walk", wraps=ast.walk) as whole_module_walks:
        spellings = tuple(NameUse.spellings(module))

    assert spellings[0] == ("Reexport", NameUseKind.OTHER, 1)
    assert [(kind, line) for name, kind, line in spellings if name == "Target"] == [
        (NameUseKind.RAISE, 3),
        (NameUseKind.OTHER, 9),
        (NameUseKind.OTHER, 10),
        (NameUseKind.CONSTRUCT, 2),
        (NameUseKind.CATCH, 5),
        (NameUseKind.CONSTRUCT, 12),
        (NameUseKind.CATCH, 7),
        (NameUseKind.CATCH, 8),
        (NameUseKind.OTHER, 11),
    ]
    assert ("Other", NameUseKind.CATCH, 7) in spellings
    assert ("StringWord", NameUseKind.OTHER, 10) in spellings
    assert spellings[-2:] == (
        ("Target", NameUseKind.OTHER, 11),
        ("VisibleWord", NameUseKind.OTHER, 11),
    )
    assert whole_module_walks.call_count == 0
