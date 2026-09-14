# Native scalar dictionary slots

Reference checkpoint, 13 September 2026. The native-proof publication batch is
still unfinished.

## Declaration ownership

Dictionary item writes, lookup, membership and native copies use the existing
`NativeScalar` contents contract. Exact strings, integers, booleans and `None`
are admitted. Tuple, floating-point and arbitrary object keys remain outside
this storage proof. Subclasses cannot borrow their base type's contract.

`NamespaceEvidenceABC` validates initial keys before copying or querying storage.
`CapturedSlotQuery` retains the original prefix, mutation identity and installed
value. `NamespaceMemberInventory` includes every admitted key, and
`CopiedNativeNamespace` observes the complete parent mapping at the original
copy cut. Neither inventory filters out non-text keys.

`RegistryEntries.require_class_mapping` consumes this complete-content
contract without repeating a text-only key restriction. Original class identity
and complete membership still have to match. Numeric keys and their string
spellings remain different mappings.

`AdmittedExecutionPrefixABC.binding_sources` contributes lexical installations
only for exact Unicode names. Non-text dictionary slots still replay item
mutations through the same slot query. A dictionary key is not converted into
a variable name.

Scalar equality follows dictionary semantics: `False` and `0` select the same
slot, as do `True` and `1`. This is distinct from the type-sensitive constant
comparison used to authenticate source/native expression correspondence.

## Native effects

`CPythonContainerConstruction.require_dictionary_scalar_store` validates the
exact scalar key domain and delegates temporary-key release to the existing
native lifetime authority. Prior-value and temporary-receiver release retain
their separate original-cut requirements. Equal boolean/integer keys do not
make an overwrite independent of its previous value.

`CPythonValueLifetime` includes exact integers. These objects own native digits,
without an instance dictionary, weak-reference storage or user-owned referents.
Python 3.11 inherits object deallocation and supplies `PyObject_Free`; Python 3.14
uses `long_dealloc` and its native free list. Integer subclasses remain outside
the exact-type contract. The corresponding native declarations are in
[CPython 3.11.11 longobject.c](https://github.com/python/cpython/blob/v3.11.11/Objects/longobject.c)
and [CPython 3.14.6 longobject.c](https://github.com/python/cpython/blob/v3.14.6/Objects/longobject.c).

## Validation

The focused suite reports 241 passed and two existing skips on Python 3.11 in
3.82 seconds, and 243 passed on Python 3.14 in 4.60 seconds. Coverage includes
original class identity, complete membership, copies at historical cuts,
boolean/integer aliases, overwritten-value release, hostile subclasses and
unsupported key families. Existing integer-only rejection cases now test
unsupported floating-point keys; positive integer coverage is explicit.

Expanded content, copy and lifetime-boundary checks report 83 passed on
Python 3.14 in 2.63 seconds. They include registry comparisons through both
positioned contents and an actual native copy. Integer deletion and replacement
have positive cases; replacement of a prior source-created class retains its
separate release rejection.

The final disjoint Python 3.11 runs report 5,415 passed, 148 failed and 68 skipped
in the support partition (115.20 seconds), and 760 passed and 35 failed in the
core partition (67.78 seconds). The combined total is **6,175 passed, 183 failed
and 68 skipped**. Both exact failed-node sets match the preceding call-target
checkpoint. The new storage support advances source proof, but has not closed
the remaining end-to-end conversion failures.

The original `False` registry fixture previously failed source capture with
`Native text requires an exact Unicode value`. Its source-edit planning now
completes. This does not establish candidate metaclass construction or complete
conversion equivalence.

The next direct candidate trace fails with `unadmitted_native_import`, wrapped
as `unproved_execution_effects`. The default imported-source entry admits
`builtins`, `typing` and the native future-feature module, but not the generated
metaclass import. This is earlier than metaclass construction; no new import
whitelist or native-behaviour premise was added at this checkpoint.

Reports are under `/home/ts/nra-global-scan-CXVD7T`:
`registry-boundary-trace.txt`, `scalar-dict-registry-trace.txt`,
`scalar-dict-focused-final-311.txt` and `scalar-dict-focused-314.txt`.
The expanded Python 3.14 report is `scalar-dict-registry-final-314.txt`.
Final broad reports are `scalar-dict-registry-support-311.txt` and
`scalar-dict-core-311.txt`. Earlier `scalar-dict-support-311.txt` includes eight
outdated scalar-rejection expectations and is not the final regression result.
`scalar-dict-candidate-trace.txt` records the subsequent candidate import gate.

## Full production scan

The existing disposable OpenHCS plus eight-library tree completes all 79
detectors with no omissions. All 180 emitted findings match the preceding
same-path `target-read-optimised-edit2.json` report exactly.

| Cache/input state | Scan seconds | Command wall seconds | Peak RSS (KiB) |
| --- | ---: | ---: | ---: |
| Empty cache | 50.948 | 54.52 | 1,331,624 |
| Unchanged cached source | 1.402 | 2.52 | 135,648 |
| Fresh equivalent source edit | 17.519 | 21.17 | 1,327,296 |

Cold preparation/analysis take 36.015/14.933 seconds; edit preparation/analysis
take 3.987/13.532 seconds. These are regression measurements, not a paired
attribution of speedup to scalar-key support. Command-wall edit time still
exceeds 20 seconds.

The edit changes only `TimeTravelScopeChange.needs_navigation` in the disposable
ObjectState copy from `0 != (len(self.meta_changed_keys) + 0)` to
`0 != (len(self.meta_changed_keys) + 0 + 0)`. The cache was seeded before that
edit. The live OpenHCS checkout was not modified.

Reports are `scalar-dict-cold.json`, `scalar-dict-warm.json` and
`scalar-dict-edit.json`, with corresponding `.stderr` timing records. The isolated
cache is `cache-scalar-dict-cold-StpYMR` under the same report directory.
