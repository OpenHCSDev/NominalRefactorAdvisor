# Native implicit call operands

Reference checkpoint, 13 September 2026. The native-proof publication batch
remains unfinished.

## Operand ownership

`NativeCallSlotABC` describes the prefix operand that distinguishes a protocol
NULL from a Python object passed as the first argument. `NativeCallMarker`
contributes no Python arguments. `NativeProducedValue` contributes its original
value, including when that value contains `None`.

`NativeCallOperandOrder` retains the compiler's NULL layout. Its selected operand
owns prefix interpretation through nominal dispatch. The Python-object form has
the same order on both supported interpreters:

| Interpreter | NULL form | Implicit-object form |
| --- | --- | --- |
| Python 3.11 | NULL, callable | callable, object |
| Python 3.14 | callable, NULL | callable, object |

Explicit arguments follow the prefix. The instruction argument count excludes
the implicit object. These layouts are documented by the native
[Python 3.11 CALL contract](https://docs.python.org/3.11/library/dis.html#opcode-CALL)
and [Python 3.14 CALL contract](https://docs.python.org/3.14/library/dis.html#opcode-CALL).

`NativeOperandStack.invoke` consumes either form through the shared slot
contract. `NativeCallValue.argument_slot` retains the original prefix object;
its ordinary input graph includes the callable and all Python arguments in
invocation order. A protocol marker cannot become a Python value or stored
operand. A marker in the other interpreter's NULL position is rejected.

## Graph comparison

`NativeCallValue` inherits `DataclassGraphValue` comparison and hashing through
MRO, with `eq=False` on its declaration. The existing implementation visits
shared DAG nodes once per operation. It retains the dataclass field hash
contract without persisting derived hashes or introducing another cache.

A direct 10,000-call chain previously raised `RecursionError` while hashing at
the ordinary recursion limit of 1,000. The iterative implementation handles the
same graph in 0.0754 seconds. Tests compare and hash independent 1,500-call
graphs. Separate runtime-fixture checks retain the prefix object's identity
with the first argument after serialisation.

## Proof boundaries

This establishes conditional native operand transfer. It does not close
function-creation input linkage, decorator callee correspondence, final
decorated-result installation or unknown callable effects. Those remain
separate source/native obligations.

The earlier registry candidate trace also remains unresolved. The default
source entry's limited import admission is intentional: an explicitly supplied
cached-module premise can establish native identity, but neither that identity
nor a declared use invariant establishes class-construction effects. Tests in
`test_registry_candidate_requirements.py` and `test_registry_candidate_gate.py`
exercise that distinction. No default package whitelist was added.

## Validation records

`test_native_implicit_call_slots.py` tests both layouts, invalid prefixes, NULL
versus `None`, shared argument identity and graph operations. Its runtime
fixtures replace a tiny compiled call's two-slot prefix with the native implicit
form and execute that authored bytecode. The observed arguments match the
operand graph; the fixtures do not claim correspondence to target source.

Reports under `/home/ts/nra-global-scan-CXVD7T` include
`implicit-call-first-311.txt` and `implicit-call-first-314.txt` (40 passed each),
`implicit-call-graph-311.txt` (70 passed), and
`implicit-call-hash-before.txt` / `implicit-call-hash-after.txt`.
The expanded final Python 3.14 suite reports 227 passed in 3.86 seconds in
`implicit-call-final-314.txt`.

The final disjoint Python 3.11 runs report 5,426 passed, 148 failed and 68 skipped
in 111.02 seconds for the support suite, and 760 passed with 35 failed in 67.16
seconds for the core suite. Combined coverage is **6,186 passed, 183 failed and
68 skipped**. Both exact failed-node sets match the scalar-dictionary checkpoint;
these runs establish no new broad regression, not a passing publication gate.
Reports are `implicit-call-final-support-311.txt` and
`implicit-call-final-core-311.txt`.

## Whole-package scan check

All runs below cover OpenHCS and its eight external production libraries on the
same disposable checkout. All 79 detectors complete without omissions. The 180
emitted findings match exactly across cold, unchanged-cache and new-edit runs,
and match the preceding scalar-dictionary cold report at the same source paths.

| Run | Preparation seconds | Analysis seconds | Scan seconds | Command-wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Empty cache | 36.457 | 15.178 | 51.635 | 55.23 |
| Unchanged cache | 0.000 | 1.431 | 1.431 | 2.49 |
| New function-body edit | 3.986 | 13.500 | 17.486 | 21.04 |

Peak RSS is 1,331,432, 135,812 and 1,327,412 KiB respectively. The edit adds one
further `+ 0` to the integer length expression in `needs_navigation` in the
disposable ObjectState module. The live OpenHCS checkout was not changed.
The cache is `cache-implicit-call-cold-2l0aJ1`; reports are
`implicit-call-{cold,warm,edit}.json` with corresponding `.stderr` records under
the report directory above. These are regression measurements, not an isolated
attribution of whole-scan speedup to call-slot handling. Single-edit scan time
is below 20 seconds; command-wall time remains above it.
