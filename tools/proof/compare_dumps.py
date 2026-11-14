"""Compare two finding dumps written by stable_ids.py or fixture_scan.py.

Usage: python tools/proof/compare_dumps.py BEFORE.json AFTER.json

Prints the rows only one side has, counted by detector.  A summary can print the files
digest of an installed package; when that package is this checkout, any change to its
source changes the digest and nothing else.  Rows equal up to that digest are counted
apart, so they cannot hide a real difference and cannot be mistaken for one.  Exits 1 when
any other row differs.
"""
import json
import re
import sys
from collections import Counter

DIGEST = re.compile(r"files sha256 [0-9a-f]{12}")


def rows_of(path: str) -> list[tuple[str, str]]:
    """Each finding as (owner, row text): the owner is the fixture for a fixture scan."""

    data = json.loads(open(path).read())
    if isinstance(data, dict):
        return [
            (fixture, row if isinstance(row, str) else json.dumps(row, sort_keys=True))
            for fixture, rows in sorted(data.items())
            for row in (rows if isinstance(rows, list) else [rows])
        ]
    return [("", json.dumps(row, sort_keys=True)) for row in data]


def detector_of(row: str) -> str:
    try:
        value = json.loads(row)
    except ValueError:
        return "<not a finding>"
    if isinstance(value, dict):
        return str(value.get("detector_id", "<not a finding>"))
    return value[1] if isinstance(value, list) and len(value) > 1 else "<not a finding>"


before, after = (Counter(rows_of(path)) for path in sys.argv[1:3])
removed = list((before - after).elements())
added = list((after - before).elements())
masked_removed = Counter((owner, DIGEST.sub("files sha256 *", row)) for owner, row in removed)
masked_added = Counter((owner, DIGEST.sub("files sha256 *", row)) for owner, row in added)
digest_only = sum((masked_removed & masked_added).values())
really_removed = list((masked_removed - masked_added).elements())
really_added = list((masked_added - masked_removed).elements())
print(
    f"rows: {sum(before.values())} -> {sum(after.values())}"
    f" | identical: {sum((before & after).values())}"
    f" | differ only in an installed package's files digest: {digest_only}"
    f" | removed: {len(really_removed)} | added: {len(really_added)}"
)
for label, rows in (("removed", really_removed), ("added", really_added)):
    by_detector = Counter(detector_of(row) for _owner, row in rows)
    if by_detector:
        print(f"{label} by detector:", dict(by_detector.most_common()))
    for owner, row in rows[:8]:
        print(f"  {label[0]} {owner} {row[:420]}")
sys.exit(1 if really_removed or really_added else 0)
