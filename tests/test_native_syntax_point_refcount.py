from __future__ import annotations

import gc
import sys

from nominal_refactor_advisor.native_syntax import NativePythonSyntaxIndex


def test_point_row_and_column_return_owned_references_above_small_int_cache() -> None:
    """Guard against py-tree-sitter 0.26.0's missing Py_INCREF in point_get_row/column.

    On CPython 3.14, values > 256 are heap-allocated PyLongs; reading Point.row
    or Point.column without an owned reference corrupts the PyLong freelist and
    segfaults during subsequent allocations or GC.
    """
    source = "# pad\n" * 300 + " " * 260 + "x = 1\n"
    index = NativePythonSyntaxIndex.from_source(source)
    assignment = index.common_captures()["assignment"][0]

    start = assignment.start_point
    row_tuple = start[0]
    row_before = sys.getrefcount(row_tuple)
    row_attr = start.row
    assert row_attr == 300
    assert sys.getrefcount(row_tuple) == row_before + 1

    col_tuple = start[1]
    col_before = sys.getrefcount(col_tuple)
    col_attr = start.column
    assert col_attr == 260
    assert sys.getrefcount(col_tuple) == col_before + 1

    del row_attr, col_attr, row_tuple, col_tuple, start, assignment, index
    gc.collect()
    assert [-n for n in range(1000, 1200)][0] == -1000
