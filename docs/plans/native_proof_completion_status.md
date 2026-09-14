# Native proof completion status

Reference record, 13 September 2026. NRA is based on commit `5f96c763` with an
unfinished working-tree batch. This record separates implemented capabilities
from the proof obligations that still prevent publication.

## Original keyword-call capture

Python 3.11 `KW_NAMES` and Python 3.14 `CALL_KW` now feed the shared native
invocation model. `NativeKeywordCallValue` retains the original constant metadata
as a graph dependency and derives positional and named arguments from the call's
ordered inputs. The VM's implicit receiver stays positional. The observer owns
its original code context and clears keyword state after each invocation.

Call and class-header correspondence share argument-shape validation. Source
expression admission authenticates each direct native dependency, including
keyword metadata. Copied metadata, malformed names, unpacking and unproved
callee effects remain rejected. Class-header capture does not establish native
metaclass construction, registration effects or returned-class identity.

The expanded Python 3.14 selection passes 129 tests, including real authored
bytecode with an implicit receiver and keyword arguments. Broad Python 3.11
support/core checks report 5,765/760 passed and 118/35 failed, with 69 support
tests skipped: **6,525 passed, 153 failed and 69 skipped** combined. Both exact
failure sets match the abstract-member checkpoint. Runs took 118.74/70.06 seconds.
Ruff, Black and whitespace checks pass for the changed implementation/tests.

Full-global cold/unchanged/novel-edit scans take 52.437/1.394/17.197 seconds;
command-wall times are 55.76/2.47/20.36 seconds. All 79 detectors complete with
zero omissions and exactly the preceding 180 emitted/215 underlying findings.
These are regression measurements, not a new speed optimisation. Reports use
`keyword-*` under `/home/ts/nra-global-scan-CXVD7T` and cache
`cache-keyword-JIi09H`. Cross-edit completed-proof reuse, the 20-second edited
command-wall target and the larger native-proof batch remain unfinished.

## Native abstract-member inspection

Captured values now expose `require_nonabstract_member(prefix)` independently
of class-member installation. The CPython backend owns the marker lookup law:
the actual immutable native MRO must select a supported ordinary getter and
contain neither an abstract marker nor a fallback getter. Type-only admission
also requires absent instance attribute storage. Captured functions with custom
attributes, descriptor marker getters and source-defined lookup hooks remain
unproved; the analyser does not invoke their attributes to obtain an answer.

`SourceCreatedFunctionCapture` combines its existing empty-function-storage
contract at the original source cut with the native type lookup law. This requires
the explicit external-noninterference premise. A function type alone, another
activation's prefix or a captured runtime function supplies no empty-storage
proof. Ordinary values without instance attribute storage use their existing
exact-type evidence.

`AutoRegisterClassEntry` checks the actual prepared members through this contract
before construction. This checks the source of ABCMeta's member marker queries;
it does not establish inherited abstractness, registry effects, class construction
or the returned class identity. These remain separate obligations. Earlier
prepared-body tests now identify the missing interference premise instead of
expecting the later generic construction error; they continue to require rejection.

`NativeCreationBackend.require_static_type_mro` shares immutable hierarchy
admission between member installation and lookup. A three-operation DSL batch
added that owner and migrated both consumers. Tests verify that each actual MRO
owner is checked once. The implementation does not keep a parallel hierarchy or
execute analysed getters. Authored controls demonstrate active marker callbacks
and verify that NRA rejects them without calling them.

Final focused checks report 60 passed in 2.12 seconds on Python 3.11; the expanded
selection reports 197 passed in 5.15 seconds on Python 3.14. Broad support/core
validation reports 5,742/760 passed and 118/35 failed, with 69 support tests skipped:
**6,502 passed, 153 failed and 69 skipped** combined. Support/core runs took
117.59/68.87 seconds. Both failed-node sets equal the preceding checkpoint;
no rejection has been turned into a construction proof to obtain these results.

At this checkpoint, the next shared native-call gap was confirmed on both interpreters: a positional
call produces an original value-store receipt, while an otherwise equivalent
keyword call does not. The observer has no `KW_NAMES` (3.11) or `CALL_KW` (3.14)
support. A class declaration with `metaclass=Creator` reaches the same missing
receipt. The keyword-call section above records the subsequent shared capture
and correspondence implementation; native construction remains unfinished.

All-detector regression scans retain the exact 180 emitted/215 underlying
findings with zero omissions. Cold/unchanged/novel-edit scans take
51.138/1.391/17.573 seconds; command-wall times are 54.33/2.49/20.78 seconds.
Cold preparation/analysis take 35.780/15.358 seconds, and edited input takes
3.598/13.975 seconds. These validate the proof extension, not a new performance
optimisation. Reports are `abc-member-{cold,warm,edit}.{json,stderr}` under
`/home/ts/nra-global-scan-CXVD7T`, using cache `cache-abc-member-R1gwor`.
Both engine signatures derive the three changed implementation modules. Ruff,
Black on the implementation/new tests, and whitespace checks pass. Nothing has
been committed or pushed.

## Immutable family representation reuse

`CollectedFamilySchemaIdentity` now derives its representation once per immutable
instance through `CachedDataclassRepresentation`. Dataclass fields and their
`repr` flags supply the text. The opt-in contract requires acyclic stored records
whose field representations remain stable; it does not apply to mutable source
activation or proof objects. `StoredDataclassState` excludes the derived text from
copy/pickle transport, and replacement constructs a fresh representation.

The existing scan-scoped schema ownership supplies the cache lifetime. There is
no additional global registry, field-name list or cache-key format. Tests compare
the text with native dataclass rendering, exercise inherited and hidden fields,
and check replacement, implementation changes, transport and lifetime release.

The refreshed full-global edit profile reduced representation work from 3.764 to
0.599 seconds and bundle-marker construction from 1.472 to 0.229 seconds. Family
schema text was rendered 23 times. Total profiled calls fell from 79.15 million
to 70.68 million; garbage collection remained near 0.28 seconds. These are
instrumented costs, separate from the command-wall measurements below.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 50.958 | 54.10 | 1,366,016 |
| Unchanged input | 1.416 | 2.48 | 136,004 |
| Novel equivalent edit | 17.275 | 20.37 | 1,361,224 |

All 79 detectors completed with zero omissions. Full findings match the preceding
checkpoint exactly across these runs and the profiled edit: 180 emitted findings,
215 underlying findings. The preceding command-wall measurements were
56.10/2.50/21.60 seconds; the profile identifies the reduced repeated work, while
whole-command timings also include scheduling and filesystem variation. The
edited command-wall target of 20 seconds remains unmet.

Reports under `/home/ts/nra-global-scan-CXVD7T` use the `schema-repr-*` prefix and
cache `cache-schema-repr-mfYHLN`. The before/after profiles are
`current-edit-costs.prof` and `schema-repr-edit-costs.prof`. Cross-edit completed
proof reuse and the native-proof completion batch remain unfinished.

Final support/core validation reports 5,722/760 passed and 118/35 failed, with
69 support tests skipped: **6,482 passed, 153 failed and 69 skipped** combined.
Both exact failed-node sets match the preceding checkpoint. Support/core runs
took 115.82/68.64 seconds. The expanded Python 3.11 cache selection reports
217 passed and three already-known native-proof failures in 15.39 seconds;
the Python 3.14 representation/lifetime/graph selection passes all 43 tests in
3.34 seconds. Both engine signatures derive `value_graph` through declaration
dependencies. The changed helper and new tests pass Ruff; `ast_tools.py` retains
the same 42 import/re-export diagnostics present at HEAD. Black and whitespace
checks pass. No production/test files changed during measurements or test runs.

## Item storage completion and overwrite queries

`NativeStackEffectABC` owns operand consumption for transfers that produce no
Python result. `NativeDiscardValue` and `NativeItemStoreValue` declare their
operand counts. Their opcode declarations select the shared capture method;
the item store retains its original value, receiver and key in stack order.
`NativeReturn.effect_for` authenticates the selected effect and its complete
original operand graph. `return_after_effect` selects the corresponding frame
return. Discard completion and import cleanup use the same general contract.

`SourceStackEffectReturnABC` owns frame authentication and return continuation.
`SourceItemStoreReturn` joins the three original item operands under the existing
source item-write contract. Assignment RHS evidence is shared with lexical
stores through `SourceAssignmentValueABC`; the event owner supplies their common
completion prefix. The DSL moved that prefix declaration to its common owner.

`SourceCompletionResolver` uses the existing mutation-target visitor for lexical
and item stores. Actual evaluated assignments own their RHS results, including
embedded destinations; independent expression results retain their disposition
contract. Unsupported receiver mutations remain explicit failures.

Slot release now resolves the actual previous value through the existing kernel
value-query contract, then checks that value's release obligation. A retained
class binding can justify an overwrite; unrestricted class destruction is not
inferred. An authored destructor fixture demonstrates the callback that NRA
rejects. Unproved container release is also rejected and not cached as completed
storage.

`InstalledValueSlotQuery` preserves its query type across later writes and
examines original mutation occurrences newest first. Later interference and
pending-cycle checks remain in the shared traversal. A profiled 32-write fixture
now resolves one matching stored value instead of recursively resolving all 32;
the original conservative effect-only query is unchanged. The traversal change
and prefix hoist were applied together using the DSL.

Focused Python 3.11 validation reports 325 passed in 3.60 seconds; the expanded
Python 3.14 selection reports 269 passed in 4.29 seconds. Final release,
memoisation and item-completion controls report 107 passed on each interpreter
in 2.39/3.08 seconds. Authored execution, native snapshot transport, copied-event
and operand rejection, prefix identity, overwrite retention and callback effects
are covered.

Final support validation reports 5,715 passed, 118 failed and 69 skipped in
119.55 seconds; core validation reports 760 passed and 35 failed in 68.96 seconds.
Combined coverage is **6,475 passed, 153 failed and 69 skipped**. The exact failed
nodes are unchanged from the preceding source-completion checkpoint. Reports
under `/home/ts/nra-global-scan-CXVD7T` use the `item-completion-*` prefix.

Both analysis cache signatures derive all three changed implementation modules
from declaration dependencies. No hand-maintained invalidation entry was added.

Full-global scans complete every detector with zero omissions. The 180 emitted
findings (215 underlying) agree exactly across cold, unchanged and novel-edit
runs and with the preceding source-completion checkpoint.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 52.479 | 56.10 | 1,365,832 |
| Unchanged input | 1.401 | 2.50 | 135,916 |
| Novel equivalent edit | 17.956 | 21.60 | 1,360,648 |

Cold preparation/analysis take 37.096/15.383 seconds; edited-input preparation/
analysis take 3.999/13.957 seconds. Reports are
`item-completion-{cold,warm,edit}.{json,stderr}`, using cache
`cache-item-completion-mqMgnY`. These are whole-scan regression checks, not an
attributed speedup; the edited command-wall target remains unmet.

Registry construction, source annotation-storage completion and cross-edit
completed-proof reuse remain unfinished; this batch has not been published.

## Source completion boundaries

`SourceCompletedReturnABC` owns the original operation, source completion
evaluation and native return boundary. `SourceInstalledReturnABC` specialises
that contract for installed bindings. `SourceEventReturnABC` supplies the shared
event-to-frame join for binding and discard completions.

`SourceDiscardReturn` joins the original discarded operand, independently proved
source release and native return. It neither manufactures a namespace binding
nor treats a conditional native receipt as proof of source effects. Unknown
calls, copied source events, foreign prefixes and altered native operands remain
rejected.

`SourceCompletionResolver.completed_body` considers original bindings and
independently applied expression results. Bound results remain with their
existing bindings; independent dispositions derive from the existing destination
declarations. The final candidate is selected by source dominance in one
unambiguous frame. `PreparedNamespaceTail` validates the selected completion and
the original final source evaluation, including any later statements.

Native discard lookup belongs to `NativeReturn`. Both source expression
completion and import cleanup use its unique original discard/operand check;
the compilation index selects the corresponding original return. This removes
the separate discard predicate formerly in source import cleanup.

Unreleased interface names changed with the broader contract:

| Earlier interface | Current interface |
| --- | --- |
| `SourceInstalledReturnResolver` | `SourceCompletionResolver` |
| `PreparedNamespaceTail.store` | `PreparedNamespaceTail.completion` |
| `installation_evaluation` | `completion_evaluation` |

Focused Python 3.11 checks report 135 passed in 2.71 seconds; the expanded Python
3.14 selection reports 165 passed in 4.39 seconds. The new controls execute
authored fixtures and check original/copy identity, source ordering, prefixes,
native operand substitution and unknown effects.

The predecessor-order control now observes completion property returns before
prefix construction. Its previous check for a cached `completed` field no longer
matched the revalidating property contract. The order/activation selection
reports 77 passed in 2.53 seconds. Other negative cache checks already pair their
storage assertions with actual admission failures or verify that completion is
not cached; those controls are retained.

Final broad support checks report 5,672 passed, 118 failed and 69 skipped in
121.40 seconds; the core suite reports 760 passed and 35 failed in 69.96 seconds.
Combined coverage is **6,432 passed, 153 failed and 69 skipped**. Compared with
the ordered-sequence checkpoint, the exact failed-node sets remove the two
discarded-expression failures and the stale predecessor-order assertion, and
add none. Final Python 3.14 checks covering discard completion, namespace tails,
predecessor ordering, activation and construction report 87 passed and two
skipped in 3.21 seconds.

A combined 16-worker experiment reached the 165-second command limit without
a pytest summary. Its displayed 100% progress is not evidence of completed
validation. The completed eight-worker split runs above remain authoritative;
the default worker configuration is unchanged. Reports are
`discard-completion-final-support-311.txt`, `discard-completion-core-311.txt`,
`discard-completion-final-selection-314.txt` and
`discard-completion-full-16.{txt,stderr}`.

Full-global scans complete all 79 detectors with zero omissions. All 180 emitted
findings (215 underlying) agree exactly across cold, unchanged and novel-edit
runs and with the preceding ordered-sequence checkpoint.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 51.667 | 55.25 | 1,365,504 |
| Unchanged input | 1.442 | 2.51 | 135,872 |
| Novel equivalent edit | 17.679 | 21.31 | 1,361,276 |

Cold preparation/analysis take 36.325/15.342 seconds; edited-input preparation/
analysis take 4.021/13.658 seconds. Reports are
`discard-completion-{cold,warm,edit}.{json,stderr}`, using cache
`cache-discard-completion-NtnKEM`. These are regression measurements, not an
attributed optimisation. Edited command-wall time remains above 20 seconds.

Annotation-only bodies still require their original dictionary-write completion
path. Compiler-elided constant expressions need a distinct native boundary;
they do not acquire a fictional discard. Registry construction and cross-edit
completed-proof reuse also remain unfinished. Reports under
`/home/ts/nra-global-scan-CXVD7T` use the `discard-completion-*` prefix.

## Ordered sequence construction

`NativeSequenceValue.capture` owns the shared ordered-input transfer for tuple
and list builders. Their declarations retain distinct native types and resolver
hooks; opcode members select those declarations directly. The former tuple-only
stack implementation is removed.
The shared declaration is abstract; a concrete builder must supply its native
type before it can be instantiated.

`NativeListExtensionValue` retains its original fresh-list receiver and iterable.
Its source-literal contract accepts an exact constant tuple of appended items,
using the existing callback-free constant comparison. Invalid receiver depths,
aliased stack receivers and unknown iterables remain unproved. Nested sequence
contents are checked through their production declarations, without identifying
an analyser-created collection with the runtime object.

The source list join authenticates every original operand and consumes the
existing `SourceLiteralCapture` at its original read. Dynamic list elements,
general unpacking, mutation and reference-release effects remain separate
obligations. Literal contents do not grant runtime identity or general release
of content-bearing values. This completes the previously failing list-valued
class metadata case.

Focused Python 3.11 checks report 121 passed in 2.53 seconds. The expanded Python
3.14 selection reports 148 passed and three skipped in 3.29 seconds. Controls
cover real authored fixture execution, nested literals, compiler expansion,
snapshot transport, changed/copy-substituted operands, exact scalar types,
unknown effects and callback-bearing foreign containers. Reports under
`/home/ts/nra-global-scan-CXVD7T` use the `native-sequence-*` prefix.
After adding the abstract-declaration control, the final Python 3.14 selection
reports 149 passed and three skipped in 3.43 seconds
(`native-sequence-final-314.txt`).

Initial broad Python 3.11 support checks report 5,646 passed, 121 failed and 69 skipped
in 121.95 seconds; the core suite reports 760 passed and 35 failed in 68.73
seconds. Combined coverage is **6,406 passed, 156 failed and 69 skipped**.
Compared with the import/runtime checkpoint, the exact failed-node sets remove
only the list-valued class metadata failure and add none. Reports are
`native-sequence-support-311.txt` and `native-sequence-core-311.txt`.

Final broad checks, including the abstract-declaration control, report 5,647
passed, 121 failed and 69 skipped in 121.91 seconds for support, and 760 passed
with 35 failed in 69.57 seconds for core. Combined coverage is **6,407 passed,
156 failed and 69 skipped**. The same exact failed-node comparison holds.
Reports are `native-sequence-final-{support,core}-311.txt`.

Both edited implementation owners, `native_compilation` and `source_execution`,
are included in the derived analysis and detector-semantic cache signatures.
This was checked against the current declaration dependency traversal; no
manual cache-version or dependency-list entry was introduced.

Final full-package scans complete all 79 detectors with zero omissions. The
180 emitted findings (215 underlying) agree exactly across cold, unchanged and
novel-edit runs and with the preceding import/runtime checkpoint.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 53.443 | 57.12 | 1,365,560 |
| Unchanged input | 1.384 | 2.46 | 135,908 |
| Novel equivalent edit | 18.043 | 21.72 | 1,361,576 |

Cold preparation/analysis take 37.834/15.609 seconds; edited-input preparation/
analysis take 4.037/14.006 seconds. Reports are
`native-sequence-final-{cold,warm,edit}.json` and `.stderr`, with cache
`cache-native-sequence-final-7mqsjh` under the report directory. These are
regression measurements for the added proof coverage, not an attributed
whole-scan speedup. The edited command-wall time remains above 20 seconds.

## Source import installation and cleanup

`SourceInstalledReturnResolver` uses the existing import declaration's visitor
to select `SourceModuleImportStore` or `SourceMemberImportStore`. Their common
base owns the original binding, source cut, frame and native stored-value
receipt. Source admission continues through
`SourceModuleExecution.require_import_operation`, including its builtins-only
importer lookup, original import origin and destination storage checks.

The native request is checked against its declaration: module spelling,
relative level, from-list and selected member. `ImportFromModuleName` owns its
derived module/level components. Constant input comparison uses the existing
callback-free constant-content authority, including exact types and ordered
tuple contents. Neither name lookup nor native transfer observation supplies
import execution by itself.

`NativeDiscardValue` records `POP_TOP` with its original operand. Member-import
completion selects the unique cleanup after the final original alias, checks
that it releases the retained request result, and consumes the admitted module
association through `InitialNativeIsland.require_registered_module_retention`.
The prefix contract already requires captured `sys.modules` associations to
remain valid. That registered reference supplies retention; analyzer-held
references and arbitrary entries in a frame namespace do not. The general
frame-retention helper is unchanged.

`SourceInstalledReturnABC.native_completion_offset` distinguishes the final
binding from a source-proved cleanup boundary. Prepared namespace tails use
that shared contract before checking later compiler work. An earlier alias can
be installed without being the final cleanup boundary. A later failed import
does not invalidate an earlier successful binding, and unknown subsequent
source effects still prevent completion.

Python 3.11 focused checks report 125 passed in 2.72 seconds. The expanded
Python 3.14 selection reports 172 passed in 4.61 seconds. Controls cover actual
fixture execution, module/class frames, global destinations, importer shadows,
partial imports, copied or foreign cuts, altered requests, substituted cleanup
operands and missing module associations. Reports under
`/home/ts/nra-global-scan-CXVD7T` use the `import-installation-*` prefix.

Native call/discard sequences now have conditional return receipts even when
their source effects are unknown. Native interruption tests use a genuinely
unsupported transfer, while a dedicated source test verifies that an unknown
call still cannot acquire source completion. Unknown imports, ambiguous repeated
bindings, unproved dotted source traversal and arbitrary callbacks remain open.

### Runtime relative-import targets

Member imports now resolve their target through the actual global `__package__`
at the original import read. `ImportFromModuleName` owns relative-level
resolution and the package lookup key; the captured-reference kernel supplies
the runtime read. Class-local shadows and later global rebindings do not change
that earlier import. Absolute imports do not read package metadata. The same
resolved target supplies the registered-module retention check during cleanup.

This corrects an observed false acceptance. With the catalogue name
`typing.synthetic`, the source `__package__ = "builtins"; from . import Any`
previously acquired `typing.Any` in the proof, although actual Python execution
raises `ImportError`. The catalogue path remains available for static import
discovery, but no longer supplies runtime identity. Unknown package contents,
invalid relative levels and unproved `__spec__`/`__name__` fallback behaviour
remain open.

The final Python 3.14 boundary selection reports 114 passed and three skipped
in 2.80 seconds (`runtime-relative-imports-final-314.txt`). Python 3.11 support
checks report 5,604 passed, 122 failed and 69 skipped in 121.30 seconds; the core
suite reports 760 passed and 35 failed in 67.77 seconds. Combined coverage is
**6,364 passed, 157 failed and 69 skipped**. Reports use the
`import-runtime-final-{support,core}-311.txt` names. Relative to the shared-segment
checkpoint, the exact failed-node sets remove only the two imported-member
class-completion failures and add none. These remaining failures still prevent
publication of the full native-proof batch.

The full-package cold, unchanged and novel-edit scans complete all 79 detectors
with no omissions. Their 180 emitted findings (215 underlying) agree exactly
with each other and with the preceding shared-segment checkpoint.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 53.342 | 57.06 | 1,365,484 |
| Unchanged input | 1.403 | 2.50 | 135,844 |
| Novel equivalent edit | 18.238 | 21.97 | 1,360,768 |

Cold preparation/analysis take 37.712/15.630 seconds; edited-input preparation/
analysis take 4.024/14.214 seconds. Reports use
`import-runtime-final-{cold,warm,edit}.json` and `.stderr`, with cache
`cache-import-installation-cold-5Cr0qM` under the report directory. These check
the expanded proof coverage rather than demonstrate a new optimisation. The
edited command-wall time remains above 20 seconds.

## Shared operand segments

`NativeContinuationWindow` owns one straight-line operand segment and its
original stores. `NativeStoreStream` observes each instruction once in that
segment. `NativeValueStoreStream` derives stored-value receipts from the same
segments; it no longer creates a separate operand window for every store.
The instruction and function-creation transfers share
`NativeOperandStack.capture_instruction` with the entry-window observer.

Stored operands and the final return inventory now retain the same original
objects. Instruction-offset membership rejects copied operands and conflicting
store observations. Jumps, reversed or repeated instructions, unsupported
operations and unproved returns end the continuing segment. Earlier stored
values remain available as conditional transfer receipts without acquiring a
later return. Historical native membership does not permit replay at a later
source cut; the source availability checks remain independent.

`NativeImportValue` records the original module request and its level/from-list
inputs. `NativeImportMemberValue` retains the original module below each member
result. Multiple member stores and restored snapshots share that module operand.
These declarations describe native transfers, not the importer's identity,
import execution effects or the returned object's identity. Repeated aliases
retain the existing ambiguity boundary.

At this earlier checkpoint, `POP_TOP` remained unsupported in the continuing operand walk. Member-import
stores therefore do not acquire a return while the retained module's release
is unproved. The source import-origin and importer admission contracts remain
unchanged; imported-name source installation was not yet implemented. The source
import checkpoint above records the subsequent implementation.

The first focused Python 3.11 selection reports 196 passed in 2.97 seconds.
After replacing an assertion tied to the old duplicated operand inventories,
the boundary selection reports 73 passed in 2.26 seconds. It explicitly checks
that a historical operand remains unavailable for replay at a later source cut.
Import/segment controls report 82 passed in 2.25 seconds. The expanded Python
3.14 selection reports 219 passed in 4.59 seconds. Reports under
`/home/ts/nra-global-scan-CXVD7T` use the `shared-segment-*` and
`native-import-operands-*` prefixes.

Final broad Python 3.11 support checks report 5,564 passed, 124 failed and 69
skipped in 124.25 seconds. The core suite reports 760 passed and 35 failed in
67.92 seconds. Combined coverage is **6,324 passed, 159 failed and 69 skipped**.
The exact failed-node sets are unchanged from the class-result checkpoint.
Reports are `shared-segment-final-support-311.txt` and
`shared-segment-final-core-311.txt`. The remaining failures still prevent
publication of the full native-proof batch.

Full-package scans retain all 79 detectors, zero omissions, and exact agreement
on 180 emitted findings (215 underlying) across cold, warm and novel-edit runs
and with the preceding class-result checkpoint:

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 52.409 | 55.98 | 1,365,724 |
| Unchanged input | 1.422 | 2.52 | 135,904 |
| Novel equivalent edit | 17.851 | 21.53 | 1,360,760 |

Cold preparation/analysis take 36.942/15.467 seconds; edited-input preparation/
analysis take 3.982/13.869 seconds. Reports are
`shared-segment-final-{cold,warm,edit}.json` and `.stderr`, with cache
`cache-shared-segment-cold-SQLQoq` under the report directory. These are
regression measurements for the consolidated walk and import coverage. The
small timing differences do not establish an attributable whole-scan speedup;
the edited command-wall time still exceeds 20 seconds.

## Class-result installation

`NativeClassBuilderValue` retains the original `LOAD_BUILD_CLASS` production
without inferring the builder's runtime identity. The opcode declaration also
supplies the backend's builder lookup key. `ExactNativeClassCapture.construction_in`
joins the final call to its original builder, body function, operand inventory
and creator frame. The source class result checks the compiler-supplied name and
joins each base at its original source read.

The builder's actual value comes from `SourceClassBodyEntryABC.builder_value`,
factored from its existing builtins-only lookup. A module-global variable named
`__build_class__` does not replace that lookup. Builder identity, body builtins,
source construction and final installation retain their separate checks.

`ExactNativeClassCapture.creation` now derives from `body.require_creation()`.
It is no longer a second stored capture site. Missing creation ranges leave the
class capture open rather than producing a synthetic site independently of the
body's receipt. Snapshot restoration preserves the shared original creation.

`SourceDefinitionResultABC` supplies the common installed-return contract for
definition results. The class-result factory returns that nominal role, so the
existing return resolver can select a completed class result without a cast or
another dispatch table. `SourceCreatedClassCapture` joins its original store and
return through that contract. This supports nested class tails and classes with
multiple already-admitted source bases, including bodies ending in decorated
methods. Replacing decorators and unknown construction hooks remain unproved.

Focused Python 3.11 checks report 144 passed and nine skipped in 3.34 seconds.
Expanded Python 3.14 checks report 247 passed and one existing generated-frame
failure in 4.59 seconds. The same failure and message occur in the earlier
`prepared-tail-broad-314.txt` report. Final class-capture/result controls on
Python 3.14 report 43 passed in 3.43 seconds, including missing-range rejection.
Reports under `/home/ts/nra-global-scan-CXVD7T` use the `class-result-*` prefix.

Final Python 3.11 support checks report 5,546 passed, 124 failed and 69 skipped
in 122.14 seconds. The core suite reports 760 passed and 35 failed in 67.38
seconds. Combined coverage is **6,306 passed, 159 failed and 69 skipped**.
Seven earlier failures are closed, covering nested activation, global lookup
and writes, receiver capture and original-prefix derivation. No new failed
nodes appear relative to the decorated-result checkpoint; the core set is
unchanged. Reports are `class-result-final-support-311.txt` and
`class-result-core-311.txt`.

The former test requiring blanket rejection of nested-class continuation now
checks the installed class call, its original binding and completed outer body.
It explicitly distinguishes that result from the class-body function operand.
The final focused selection reports 49 passed and eight skipped in 3.14 seconds.
The same selection on Python 3.14 reports 57 passed in 3.47 seconds
(`class-result-final-selection-314.txt`).

At the class-result checkpoint, imported-name continuation remained separate work. For a multi-member
`from ... import ...`, the native instruction sequence retains the module across
the member stores, then releases it with `POP_TOP`. The current operand-window
contract requires an empty stack at a captured store. Supporting this path needs
original retained-stack and release evidence, alongside the existing
declaration-owned import-origin and builtins-importer checks. An ordinary name
read or an ignored `POP_TOP` would not supply that evidence.

The class-result global scans retain all 79 detectors, zero omissions, and
the same 180 emitted findings (215 underlying) as the decorated-result
checkpoint and across cold, warm and novel-edit inputs:

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 52.572 | 56.18 | 1,365,532 |
| Unchanged input | 1.410 | 2.49 | 135,960 |
| Novel equivalent edit | 18.116 | 21.89 | 1,361,084 |

Cold preparation/analysis take 36.896/15.676 seconds; edited-input preparation/
analysis take 4.005/14.111 seconds. Reports are
`class-result-final-{cold,warm,edit}.json` and `.stderr`, using
`cache-class-result-cold-nuyAZi` under the report directory. These measurements
check the added proof coverage; they do not establish a new speedup. The
edited command-wall time remains above 20 seconds.

## Decorated-result installation

Original native application sites now join their callee and implicit argument
to the final store's operand inventory. `NativeCaptureSite` validates function
creation operands; `NativeDefinitionApplication` validates application operands.
The original predecessor declaration selects that behaviour, rather than a
source-side dispatch table. The existing inventory owns the unambiguous lookup
by instruction offset and rejects substituted operand identities.

`SourceNativeDecoratorApplication` joins its callee through
`SourceNativeOperandJoin` at the callee's original source read. The shared
production/completion contracts also serve ordinary expression joins. A wrapped
result does not need to impersonate an ordinary expression to use that proof.

`SourceDefinitionResultABC` owns definition-result availability at an original
source cut. `SourceDefinitionCapture` adds raw definition identity. Wrapped
results share availability, but do not inherit raw identity. Only the final
wrapper supplies a native installation and return continuation; intermediate
wrappers and the decorated raw function cannot borrow that store.
`SourceInstalledReturnResolver` selects the final result from the original
definition's result chain. Prepared class completion consequently consumes the
wrapped installation instead of requesting an unproved raw-function store.

Keyword-only defaults retain their compiler-generated dictionary operands.
`BUILD_MAP` and `BUILD_CONST_KEY_MAP` project input transfers through their
declared operand operations. Nonempty results carry a known dictionary type and
original inputs, without certifying contents, hashing safety or source effects.
The empty-map operation retains its existing fresh-empty-map proof. Constant-key
construction requires the original constant tuple with matching length.

Controls exercise module/class results, positional and keyword-only defaults,
annotations, nested wrappers, actual authored runtime results, original callee
cuts, copied source cuts and native operands, changed callees, foreign frames,
intermediate results and substituted stores. Expanded Python 3.14 checks report
260 passed and three skipped in 4.96 seconds. Those skips concern the
`BUILD_CONST_KEY_MAP` opcode absent from that interpreter; its actual dictionary
and keyword-default paths are tested. Selected final Python 3.11 checks report
56 passed in 2.34 seconds.

Final support checks report 5,520 passed, 131 failed and 68 skipped in 121.46
seconds. The core suite reports 760 passed and 35 failed in 67.49 seconds.
Combined broad coverage is **6,280 passed, 166 failed and 68 skipped**. Sixteen
earlier support failures are closed, including captured builtin lookup,
future-import cases, snapshot identity and composed member-promotion/module-move
operations. The core failed-node set is unchanged. Reports are
`decorated-final-support-311.txt` and `decorated-final-core-311.txt` under
`/home/ts/nra-global-scan-CXVD7T`.

Five tests tied to the former blanket rejection of nonempty map observation were
updated. They now distinguish conditional native transfers from empty-map
identity and source installation, and still reject invalid counts, stack
underflow and unpacked transfers. Their focused run reports 49 passed before
the final broad rerun. These changes do not admit source dictionary contents or
effects from type-only transfer evidence.

The final dictionary-boundary checks also pass on Python 3.14: 46 passed and
three opcode-specific skips in 2.72 seconds.

Whole-package measurements retain all 79 detectors, zero omissions, and the
same 180 emitted findings (215 underlying findings) across all three runs and
the preceding function-operand checkpoint:

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 52.853 | 56.51 | 1,366,512 |
| Unchanged input | 1.387 | 2.44 | 135,668 |
| Novel equivalent edit | 18.044 | 21.74 | 1,362,076 |

Cold preparation/analysis take 37.287/15.566 seconds; edited-input preparation/
analysis take 4.042/14.002 seconds. Reports are `decorated-final-{cold,warm,edit}.json`
and their `.stderr` files, using `cache-decorated-installation-cold-W4XidN` in
the report directory. The novel edit affects only the disposable benchmark's
ObjectState source. These are regression measurements for added proof coverage,
not an attributed optimisation; edited command-wall time still exceeds 20 seconds.

The larger native-proof batch and dependency-aware completed-proof reuse across
edits remain unfinished. This checkpoint does not certify native metaclass
construction, arbitrary decorators, generated function bodies or callback effects.

## Function-creation operands

The existing native instruction walk now retains raw function-creation inputs,
attribute attachments, decorator-call operands and their final store in one
operand graph. `NativeCodeObservation` owns the current creation emission for
its frame. Body operand capture authenticates the original load, creation and
attachment instructions there; initial class-helper capture retains its separate
strict prologue boundary.

`NativeFunctionAttributeValue` retains its original function creation. An
attachment does not represent another function object. Creation and operand
observations share the same final `NativeBindingTransfer`; the compact graph
does not retain executable code objects or transient compiler emissions.

Binding snapshot transport serialises the existing graph's inputs before their
users. Its state derives from the dataclass fields through `StoredDataclassState`,
and pickle preserves shared operand identities without recursively descending a
long decorator chain. This does not establish arbitrary source execution or
source/native callee correspondence.

The new tests cover module and class creators, defaults, decorator order,
attachment identity, snapshot restoration without recompilation, copied or
foreign instruction evidence, unsupported backends and the initial/body boundary.
Expanded Python 3.14 validation reports 184 passed in 2.86 seconds. The Python
3.11 support suite reports 5,469 passed, 147 failed and 68 skipped in 117.71
seconds, with the exact failed-node set unchanged from store convergence.
The core suite reports 760 passed and 35 failed in 65.99 seconds, also with
unchanged failures. Combined broad coverage is **6,229 passed, 182 failed and
68 skipped**.
Reports are `function-operands-first-314.txt` and
`function-operands-support-311.txt`, with `function-operands-core-311.txt`, under
`/home/ts/nra-global-scan-CXVD7T`.

All three whole-package runs complete all 79 detectors without omissions and
retain the same 180 emitted findings (215 underlying findings) as store
convergence. Reports are `function-operands-{cold,warm,edit}.json` and their
`.stderr` files in the directory above; the cache is
`cache-function-operands-cold-ZFEUbA`.

| Input | Scan seconds | Command-wall seconds | Peak RSS, KiB |
| --- | ---: | ---: | ---: |
| Empty cache | 53.057 | 56.59 | 1,363,536 |
| Unchanged input | 1.382 | 2.42 | 136,436 |
| Novel equivalent edit | 17.824 | 21.47 | 1,359,028 |

The cold run spends 37.423 seconds in preparation and 15.634 in analysis;
the edited run spends 4.019 and 13.805 seconds respectively. These single-run
regression measurements are slower than the preceding cold/edit checkpoint,
not evidence of a speedup. Edited-input command-wall time remains above 20
seconds. The edit is confined to the disposable benchmark ObjectState source.

This earlier checkpoint did not yet supply source-native decorated-result
installation. That capability is described above. Dependency-aware reuse of
completed proofs across edits is still unfinished.

## Converging native store observations

`NativeStoreStream.record_store` returns the original store for an instruction
offset within its current continuation. The continuation's sole store collection
is now an insertion-ordered mapping; the published `NativeReturn.stores` tuple
is derived from its values. Function-creation emissions retain the returned
original record rather than independently retaining a second store description.

`NativeBindingTransfer.require_observation` checks compatibility. An operand-free
description can refer to an already observed store with matching instruction
metadata. A second operand must be the same original object, not an equal copy.
An operand-free first record cannot acquire a later, independently produced
value through this join. Matching uses the dataclass declarations with only the
operand excluded; there is no separately maintained list of metadata fields.
The ordinary same-record path returns before allocating comparison projections.

This checkpoint supplied the prerequisite for observing function-creation
operands in the shared native walk. It did not itself enable those operands or
complete decorated-result installation. Duplicate records supplied directly to a return receipt still
fail its existing uniqueness check; that proof gate was not relaxed.

The authored batch is `.codex-temp/native_store_convergence.py`, followed by a
DSL patch adding the same-record fast path. Eight new tests cover record identity,
conflicting metadata, copied operands, observation ordering and distinct stores
with equal names. Focused Python 3.11 checks report 99 passed and 17 existing
skips; expanded Python 3.14 checks report 175 passed. The support suite reports
5,450 passed, 147 failed and 68 skipped in 116.02 seconds, with its exact failed
set unchanged from the attribute checkpoint. Reports under
`/home/ts/nra-global-scan-CXVD7T` are `store-convergence-focused-311.txt`,
`store-convergence-final-314.txt` and `store-convergence-support-311.txt`.
The core suite reports 760 passed and 35 failed in 67.42 seconds, also with an
unchanged failed-node set (`store-convergence-core-311.txt`). Combined broad
coverage is **6,210 passed, 182 failed and 68 skipped**.

The empty-cache whole-package scan completes all 79 detectors without omissions
in 50.284 scan seconds / 53.90 command-wall seconds. Preparation takes 35.299
seconds and analysis takes 14.985; peak RSS is 1,332,028 KiB. Its 180 emitted
findings exactly match the same-path attribute checkpoint. The report is
`store-convergence-cold.json` with its `.stderr` timing record, using
`cache-store-convergence-cold-7qxWn5` in the report directory above. This is a
regression measurement, not a separately attributed speedup.

The unchanged-cache run completes in 1.355 scan seconds / 2.42 command-wall
seconds, with 135,516 KiB peak RSS. A novel edit in the disposable ObjectState
method completes in 17.380 scan seconds / 20.97 command-wall seconds (4.007
preparation, 13.373 analysis; 1,327,164 KiB RSS). Reports are
`store-convergence-warm.json` and `store-convergence-edit.json` with their
`.stderr` files. All three runs retain identical findings and complete detector
coverage. Edited-input command-wall time remains above 20 seconds.

## Ordinary attribute operands

`NativeAttributeValue` extends the existing native read-value declaration.
`LOAD_ATTR` consumes its original receiver and retains the attribute name and
source span. `SourceNativeExpressionABC` joins that receiver at its canonical
source read, then uses the existing captured-reference proof for the attribute
result. The analyser does not perform an additional Python attribute lookup.

This closes the native store/return gap for a class ending with an assignment
such as `descriptor = builtins.property`. Conditional native observation also
retains unknown attribute chains, but does not admit their source execution.
Copied productions, changed attribute names and substituted receiver sites are
rejected. Historical receiver reads are joined before later assignments; they
do not establish release safety for those later assignments.

The supported operation replaces one receiver with one result, as documented
for [Python 3.11](https://docs.python.org/3.11/library/dis.html#opcode-LOAD_ATTR)
and the ordinary [Python 3.14 form](https://docs.python.org/3.14/library/dis.html#opcode-LOAD_ATTR).
Method-call loads remain open: their two-result protocol depends on actual
method binding and cannot be represented by inventing a NULL or self operand.
Unsupported backends still contribute no native primitive operations.

The authored DSL batch is `.codex-temp/native_attribute_operands.py`. The 15 new
cases in `tests/test_native_attribute_operands.py` pass on Python 3.11. The
expanded Python 3.14 suite reports 143 passed in 3.54 seconds, covering native
reads/calls, source correspondence and prepared namespace tails. Reports are
`attribute-cuts-311.txt` and `attribute-final-focused-314.txt` under
`/home/ts/nra-global-scan-CXVD7T`.

Final disjoint Python 3.11 runs report 5,442 passed, 147 failed and 68 skipped
in 117.26 seconds for the support suite, plus 760 passed and 35 failed in 68.97
seconds for the core suite. Combined coverage is **6,202 passed, 182 failed and
68 skipped**. Comparing exact failed-node sets removes the `builtins.property`
class-completion failure and introduces no new failures. Reports are
`attribute-support-311.txt` and `attribute-core-311.txt`.

The full 1,014-file OpenHCS plus eight-library production scan completes all 79
detectors without omissions. The 180 emitted findings match exactly across all
three runs and the preceding same-path implicit-call cold report.

| Run | Preparation seconds | Analysis seconds | Scan seconds | Command-wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Empty cache | 35.860 | 15.114 | 50.974 | 54.57 |
| Unchanged cache | 0.000 | 1.404 | 1.404 | 2.46 |
| New function-body edit | 3.914 | 13.528 | 17.442 | 21.06 |

Peak RSS is 1,331,728, 135,708 and 1,327,504 KiB respectively. Reports are
`attribute-{cold,warm,edit}.json` and their `.stderr` timing records, using
`cache-attribute-cold-szSUBK` under the report directory above. The new edit adds
one further `+ 0` to the integer length expression in the disposable ObjectState
module's `needs_navigation` method; live OpenHCS source is unchanged. These
measurements check the native-proof change for regression, not a separately
attributed performance improvement. Edited-input command-wall time still
exceeds 20 seconds, and the remaining native-proof batch is not publishable.

## Implicit native call operands

The [implicit call-slot reference](native_implicit_call_slots.md) records shared
native handling of NULL-prefixed and implicit-object calls. Operand declarations
own the distinction; version-specific NULL layouts remain on the existing enum.
Call values also inherit the existing iterative DAG equality/hash contract.
This closes an operand-transfer gap and a recursive-hashing failure, not the
remaining decorator installation or native class-construction proofs.
Final broad checks report **6,186 passed, 183 failed and 68 skipped**, with both
failed-node sets unchanged. Whole-package cold, warm and new-edit scans complete
all 79 detectors with identical findings; their scan times are 51.635, 1.431 and
17.486 seconds. The linked reference records command-wall times and boundaries.

## Scalar dictionary keys

The [scalar dictionary slot checkpoint](scalar_dictionary_slots.md) removes the
Unicode-only restriction from ordinary dictionary storage while keeping lexical
binding lookup text-only. Membership and native copies retain all admitted keys;
boolean/integer aliases retain dictionary equality and overwrite obligations.
Focused suites pass on Python 3.11 and 3.14. Final broad runs report **6,175
passed, 183 failed and 68 skipped**, with both failed-node sets unchanged from
the call-target checkpoint. Candidate metaclass construction and the remaining
native-proof batch are still unfinished.

## Call-target projection regression checkpoint

The [shared call-target projection](call_target_reuse.md) changes an aggregate
view to consume existing resolved targets rather than repeat their resolution.
No native admission rule changes at this checkpoint. Full disjoint suites report
**6,135 passed, 183 failed and 68 skipped**, with failed-node sets identical to
the decorator-input checkpoint below. Publication still requires completing the
remaining native proofs as well as the performance work.

## Original decorator argument chains

`NativeDefinitionApplication.argument` retains the original preceding native
result site. The first application refers to the same raw creation receipt;
each subsequent application refers to the preceding application. The existing
`NativeEmissionBinding` owns construction and lifetime of this chain. Application
frames derive from their arguments, and nonforward argument/preparation/call
offsets are rejected. The sites do not substitute for runtime function objects.

`NativeDefinitionApplication` inherits the existing `DataclassGraphValue`
comparison and hash implementation through MRO. This keeps linked-input
comparison iterative. A compiled 1,200-decorator fixture roundtrips its complete
compilation through serialisation, preserves predecessor identity, compares and
hashes the final application, and distinguishes a changed argument.

`SourceNativeDecoratorApplication.native_application` joins each application to
its original source read, compiler source span, preceding native input and frame.
Closure revalidates the current decorator order and chain length, including after
warming. Copied input/frame evidence and reordered, duplicated or omitted source
decorators are rejected. Native callee operands and final decorated-result
installation remain separate unfinished obligations.

Focused checks report **132 passed** on Python 3.11 in 3.08 seconds and
**129 passed, three existing skips** on Python 3.14 in 3.34 seconds. After the
cold/warm fixture was corrected to avoid closing the source result during setup,
all 21 new cases passed again on both versions in 1.99 and 2.26 seconds.
The support suite reports **5,366 passed, 148 failed and 68 skipped** in
117.17 seconds; core reports **760 passed and 35 failed** in 67.31 seconds.
Combined coverage is **6,126 passed, 183 failed and 68 skipped**. Exact failed-node
sets match the global-call checkpoint. No checks ran concurrently.

Reports under `/home/ts/nra-global-scan-CXVD7T` are
`application-input-final-311/314.txt`,
`application-input-cold-controls-311/314.txt`,
`application-input-support-311.txt` and `application-input-core-311.txt`.
Applied DSL batches are `.codex-temp/native_application_arguments.py`,
`.codex-temp/source_application_arguments.py` and
`.codex-temp/application_argument_graph.py`.

The full empty-cache OpenHCS plus eight-library scan completes all 79 detectors
without omissions and retains the same 215 findings as the positional-call
checkpoint. It takes **52.300 scan seconds** (35.275 preparation, 17.025 analysis),
55.90 command-wall seconds and 1,331,420 KiB peak RSS. The unchanged-cache repeat
takes **1.423 scan seconds**, 2.47 command-wall seconds and 135,624 KiB peak RSS,
with identical findings and complete cached coverage. Reports are
`application-input-cold.json` and `application-input-warm.json`, with `.stderr`
timing records; the isolated cache is `cache-application-input-cold-bpl4r7` in
the same report directory. A novel-edit measurement has not been repeated after
this change. The native-proof publication batch remains unfinished.

## Call-bearing global reads

`NativeReadValue` owns read capture and delegates input consumption and stack
placement to its declaration. `NativeGlobalValue` consumes no operand and emits
one original Python value. For call-bearing loads, it also places the existing
`NativeCallMarker` according to the backend's `NativeCallOperandOrder`. The marker
shares the original instruction site but is not a second Python production.
Explicit `PUSH_NULL` uses the same marker constructor. This closes the global-load
protocol gap recorded in the preceding checkpoint.

The applied DSL batch is `.codex-temp/native_global_call_load.py`. New tests cover
class-body global callees, canonical results, original source frames, plain reads,
both declared stack layouts and rejection of copied production evidence. A direct
native function-body window checks conditional operand observation separately
from source activation. The public inventory continues to index module and class
entry streams; it does not eagerly index every function body.

Focused checks report **138 passed** on Python 3.11 in 2.72 seconds and
**135 passed, three existing skips** on Python 3.14 in 3.49 seconds. The support
suite reports **5,345 passed, 148 failed and 68 skipped** in 118.18 seconds;
core reports **760 passed and 35 failed** in 66.74 seconds. Both failed-node
sets exactly match the positional-call checkpoint. Combined coverage is
**6,105 passed, 183 failed and 68 skipped**. Reports are
`native-global-call-final-focused-311/314.txt`,
`native-global-call-support-311.txt` and `native-global-call-core-311.txt`
under `/home/ts/nra-global-scan-CXVD7T`.

Current full OpenHCS plus eight-library measurements use the disposable source
copy and the previously populated `cache-identity-edit-optimised-G22fXo` cache:

| Run | Preparation seconds | Analysis seconds | Scan seconds | Command-wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Implementation refresh | 34.972 | 17.469 | 52.441 | 56.74 |
| New ObjectState function-body edit | 4.612 | 15.176 | 19.788 | 23.98 |
| Unchanged edited input | 0.000 | 1.377 | 1.377 | 2.45 |

The refresh is not an empty-cache measurement. After it completed, the disposable
copy's `needs_navigation` predicate changed from
`bool(self.changed_paths) or 0 != len(self.meta_changed_keys)` to
`bool(self.changed_paths) or bool(len(self.meta_changed_keys))`. The new expression
had not previously populated this cache. Live OpenHCS remained unchanged.
All runs complete all 79 detectors without omissions and retain identical
findings, also matching the earlier same-path family-identity result. Peak RSS
is 1,337,916, 1,333,772 and 135,708 KiB respectively. Reports are
`native-global-edit-seed.json`, `native-global-novel-edit.json` and
`native-global-edited-warm.json`, with `.stderr` timing records.

The new-edit scan is below 20 seconds; end-to-end command time is not.
Dependency-aware completed-proof reuse remains unfinished. Keyword and expanded
calls, attribute callees, implicit decorator operands, source-function metadata
inspection, import/nested-class continuations and native metaclass construction
still have outstanding proof obligations. This batch remains uncommitted.

## Canonical positional-call results

`NativeCallValue` records the original callee and ordered explicit operands of
a positional `CALL`. `NativeCallOperandOrder`, selected by the existing compiler
backend, owns the two supported stack layouts. The protocol NULL is represented
by `NativeCallMarker`, not by a fabricated Python object or a second value
inventory. Ordinary value consumers reject that marker. A prepared invocation
requires its declared next operation, matching argument counts and source ranges.
Decorator application and ordinary invocation share the backend preparation
validator; decorated result installation remains a separate proof obligation.

`CompactFlowValue` and `CompactFlowRead` select their own original source
operations. This preserves callable-reference reads without fabricating ordinary
value events for callees. `SourceNativeExpressionABC` joins the native callee
and positional operands at their original read cuts, then consumes the existing
canonical result from `SourceModuleExecution.call_result`. It introduces no
parallel result cache and does not equate separate constructor invocations.

Focused checks report **131 passed** on Python 3.11 in 2.74 seconds and
**128 passed, three existing skips** on Python 3.14 in 3.31 seconds. They include
authored compiled calls, constructors, class-body `globals()`, builtin descriptor
arguments, original read membership, malformed preparation, marker rejection,
and warm serialisation without recompilation. Reports are
`native-call-owner-focused-311/314.txt` under `/home/ts/nra-global-scan-CXVD7T`.

The support suite reports **5,338 passed, 148 failed and 68 skipped** in
117.86 seconds. Exact comparison with the expression-join checkpoint removes
three failed nodes and adds none: constructor-result installation, class-body
globals identity, and the corresponding namespace frame-membership case.
Core reports 760 passed and 35 failed in 67.77 seconds, with its failed-node
set unchanged. Combined, the suites report **6,098 passed, 183 failed and
68 skipped**. Reports are `native-call-support-311.txt` and
`native-call-core-311.txt`. No tests or scans ran concurrently.

The full empty-cache OpenHCS plus eight-library scan completes all 79 detectors
with no omissions and findings identical to the preceding expression-join result.
It takes 53.233 scan seconds (36.453 preparation, 16.780 analysis), 56.79
command-wall seconds and 1,331,296 KiB peak RSS. The unchanged-cache run takes
1.408 scan seconds and 2.47 command-wall seconds with identical findings and
complete cached coverage. Reports are `native-call-cold.json` and
`native-call-warm.json`, with their `.stderr` timing records. Novel-edit timing
has not been repeated after this native-proof change.

Two former native-observation refusal cases now have source-execution rejection
coverage in `test_native_call_store.py`: observing an unknown call's operands
does not prove that it executes successfully. Source-defined descriptor metadata
inspection remains unproved; the new operand join does not bypass it. Keyword
and expanded calls, implicit decorator operands, and call-bearing multi-output
global loads still require their native protocol evidence. These remain part
of the unfinished native-proof work, alongside import/nested-class continuations
and native metaclass construction.

The applied DSL batches are `.codex-temp/native_call_operands.py`,
`.codex-temp/source_native_call_join.py`,
`.codex-temp/native_call_canonical_result.py` and
`.codex-temp/native_call_result_owner.py`. Initial simulations rejected a
multi-declaration member insertion and an ambiguous two-match replacement without
writing production files; the corrected batches select individual declarations.

## Original expression operand joins

`SourceNativeExpressionABC` now owns source/native value joining at an original
expression read. Assignment storage and nested operand joins inherit the same
lookup, constant comparison and identity checks. `NativeProducedValue` retains
its compiler source span, assigned by the shared operand emitter; the existing
native inventory still requires the original production object.

`NativeTupleValue` represents `BUILD_TUPLE` with its ordered native inputs.
Each input joins the corresponding original `SourceTupleCapture` input at that
input's own read cut. Source spans, operand count and original source/native
membership are checked. Folded tuple constants retain their existing content
comparison. Equal contents or matching types do not establish identity between
different evaluations.

Lookup and completed-creation cuts are separate. Fresh dictionary operands
must occur in the enclosing completed source prefix, while their lookup uses
the expression's earlier read prefix. This permits nested empty dictionaries
without treating pre-expression admission as completed creation.

The authored batches are `.codex-temp/native_expression_join.py` and
`.codex-temp/native_expression_completion_cut.py`. New tests exercise actual
compiled module/class output, nested operands, historical bindings, distinct
repeated-read cuts, copied and mismatched evidence, and incomplete creation.
Unknown calls, conditional expressions and unobserved compound effects remain
unproved. The previously conservative tuple-store test now verifies the joined
original source value and rejects cross-execution identity.

The warm-source check exposed stale tuple-shape admission: reversing, duplicating
or omitting AST elements after warming still passed. `SourceTupleCapture.production`
is now a validated derived property rather than a cached structural proof.
Original element captures remain cached; their source association is checked
again before reuse. `.codex-temp/source_tuple_shape_validation.py` applies this
change. All six cold/warm structural-mutation cases pass after the fix.

Final focused checks report 187 passed and one known failure on each of Python
3.11 and 3.14, in 2.92 and 3.68 seconds. The outstanding focused case
is `test_instance_installation_keeps_its_own_native_hook_obligation[False]`:
ordinary call-result storage still lacks native operand observation. Reports
use `native-expression-shape-focused-311/314.txt` in the report directory below.

Final disjoint Python 3.11 suites report **6,073 passed, 186 failed and 68 skipped**.
Support reports 5,313 passed, 151 failed and 68 skipped in 116.93 seconds;
core reports 760 passed and 35 failed in 69.24 seconds. The exact failed-node
sets match the family-identity checkpoint in both suites. Reports are
`native-expression-final-support-311.txt` and
`native-expression-final-core-311.txt` under `/home/ts/nra-global-scan-CXVD7T`.
The native-proof batch remains unfinished.

The empty-cache full OpenHCS plus eight-library scan completes all 79 detectors
with no omissions and findings identical to the family-identity result. It takes
52.281 scan seconds (35.604 preparation, 16.677 analysis), 55.97 command-wall
seconds and 1,331,104 KiB peak RSS. The unchanged-cache run takes 1.422 scan
seconds and 2.49 command-wall seconds, with the same findings and complete
cached coverage. Reports are `native-expression-cold.json` and
`native-expression-warm.json`, with `.stderr` timing files. A novel-edit timing
has not been repeated after this native-proof change; the prior A/B edit
measurements remain specific to the family-identity checkpoint.

## Family identity regression checkpoint

After scan-scoped family-identity reuse, the disjoint Python 3.11 suites report
**6,048 passed, 186 failed and 68 skipped**. The exact failed-node sets are
unchanged from the constant-content checkpoint below. The
[family identity reference](family_identity_reuse.md) records the scope and
transport checks, Python 3.14 focused validation, and isolated cold/new-edit
comparisons. Cache-key reuse does not supply completed native proofs.

## Compiler-folded constant contents

`NativeConstantContentsABC` owns exact scalar and tuple-content comparison.
`NativeScalarValueABC` inherits that contract while retaining its narrower
scalar query. The primitive domain remains the declared `NativeScalar` alias;
tuple comparison checks arity, element order and exact nested types. Unknown
objects, subclasses, mutable containers and unsupported primitive types are
rejected before equality or iteration can invoke their protocols.
Comparison visits each original value pair once, including shared tuple
subgraphs; its worklist retains the input roots for the comparison lifetime.

`NativeOperandStack` now retains admitted tuple constants in the original
`NativeConstantValue` instead of reducing them to type-only evidence.
`SourceTupleCapture` derives constant contents from its original ordered
element captures. `SourceAssignmentStore` uses the shared comparison to join
that capture to its original native production. Equal contents do not establish
identity between different source evaluations, fresh allocation, or release
behaviour. Nonconstant tuple operands and call-result installation remain
separate proof obligations.

The authored DSL batches `.codex-temp/native_constant_contents.py` and
`.codex-temp/native_constant_predicate.py` applied the change and moved the
primitive predicate onto the shared contents owner. The initial insertion
targeted a type-alias assignment unsupported by that selector; simulation
rejected the plan without writing files. The successful plan inserted the ABC
before the existing scalar declaration.

Final focused checks report **201 passed, one failed and one skipped** on
Python 3.11 in 2.61 seconds, and **202 passed and one failed** on Python 3.14
in 3.02 seconds. The previously failing final assignment
`__abstractmethods__ = ('run',)` now passes; the remaining case exercises
the separate `Payload()` result-store boundary. Reports are
`constant-join-final-focused-311.txt` and `constant-join-final-focused-314.txt`. Tests cover
authored compiled module/class output, exact nested contents, unsupported
values and subclass protocols, original receipt membership, distinct source
evaluations, deep iterative comparison, shared subgraphs and warm serialisation.

The unsupported-constant test now uses a tuple containing a float; supported
integer tuples have direct content and installation coverage. Non-scalar store
queries still reject tuples and now accept the contents-owning constant's
scalar-domain diagnostic as well as the type-only production diagnostic.

The final disjoint Python 3.11 suites report **6,040 passed, 186 failed and
68 skipped**. Support reports 5,280 passed, 151 failed and 68 skipped in
115.03 seconds; core reports 760 passed and 35 failed in 65.78 seconds.
Compared with the shared-guard checkpoint, the tuple-assignment case is the
only changed failed node; no new failures remain. Reports are
`constant-join-final-support-311.txt` and `constant-join-core-311.txt` under
`/home/ts/nra-global-scan-CXVD7T`. The native-proof batch remains unfinished.

The full OpenHCS plus eight-library production scan completes all 79 detectors
with zero omissions and identical findings. Implementation refresh reports
55.308 scan seconds (36.435 preparation, 18.873 analysis), 69.86 command-wall
seconds and 1,404,768 KiB peak RSS. The following exact-cache run reports
1.419 scan seconds and 2.51 command-wall seconds with identical findings.
Reports are `constant-join-refresh-live.json` and `constant-join-warm-live.json`,
with their `.stderr` timing records. These are not cold or novel-edit benchmarks.
The refresh wall time is higher than the preceding 62.92-second checkpoint;
these observations alone do not attribute the difference to a particular phase.

## Shared created-function transition guard

`CreatedNativeFunctionOperation.advance` owns the prerequisite that a native
function creation has been observed. Attachment, application preparation,
application and installation supply `_advance_created` hooks. Attachment also
inherits the existing function-production contract through MRO. Global
installation continues to inherit the local installation algorithm with its
declared storage operation.

The authored DSL batch `.codex-temp/created_operation_guard.py` factors four
repeated guards and removes the unused emission argument from their leaf hooks.
It does not broaden native observation or source-proof admission. Focused checks
report 162 passed and three skipped on each of Python 3.11 and 3.14, in 3.05 and
3.77 seconds respectively (`created-operation-guard-focused-311.txt` and
`created-operation-guard-focused-314.txt`). These include the existing authored
execution controls and new checks of shared dispatch, abstract registration,
attachment MRO and rejection of uncreated input without modifying the emission.

The subsequent disjoint Python 3.11 suites report **6,008 passed, 187 failed
and 68 skipped**. Support reports 5,248 passed, 152 failed and 68 skipped in
115.10 seconds; core reports 760 passed and 35 failed in 63.85 seconds. Exact
failed-node sets match the operand checkpoint. Reports are
`created-operation-guard-support-311.txt` and
`created-operation-guard-core-311.txt`. The native-proof publication batch
remains unfinished.

The full production scan after this refactor completes all 79 detectors with
identical findings: 54.176 scan seconds (35.118 preparation, 19.058 analysis),
62.92 command-wall seconds and 1,390,552 KiB peak RSS. The unchanged repeat is
an exact cache hit in 1.486 scan seconds and 2.61 command-wall seconds, again
with identical findings. Reports are `created-operation-guard-refresh-live.json`
and `created-operation-guard-warm-live.json` with their `.stderr` files. These
are implementation-refresh and unchanged-input checks, not cold-cache or
novel-edit measurements.

## Complete native operand graphs

The operand checkpoint's disjoint Python 3.11 suites report **6,002 passed, 187 failed and 68
skipped**. Support reports 5,242 passed, 152 failed and 68 skipped in 116.63
seconds; core reports 760 passed and 35 failed in 64.14 seconds. Exact failed-node
sets match the definition-application checkpoint. Reports are
`native-operand-final-support-311.txt` and `native-operand-final-core-311.txt`
under `/home/ts/nra-global-scan-CXVD7T`.

`NativeValueStoreWindow` now observes a complete operand graph ending in one
consuming store. Every production must feed that stored root, the operand stack
must be empty after storage, and native instruction offsets must progress.
Unknown operations and jump entries close the observation. The root producer's
span supplies the production query; original binding geometry supplies the
target query. Entry documentation retains its single-production, text and
same-span requirements.

`NativeProducedValue` participates in `DataclassGraphNode` traversal, with graph
children derived from its declared `inputs`. The shared graph contract retains
dataclass-field traversal as its default. Operand metadata fields do not become
execution edges. The authored DSL batch
`.codex-temp/native_operand_graph_edges.py` introduced that projection hook and
the declaration-owned operand alias.

`NativeValueStore` derives its value inventory from the stored root instead of
retaining another collection. `NativeValueInventoryABC` checks original,
unambiguous production membership for source assignments as well as existing
native consumers. Each input must precede its consumer; iterative traversal
handles deep graphs and rejects cyclic or forward operand edges.

Focused checks report **148 passed, two skipped on Python 3.11** and **150
passed on Python 3.14** (`native-operand-final-focused-311.txt` and
`native-operand-final-focused-314.txt`). They include native nested-tuple
execution controls, original/copy/foreign membership, ambiguous addresses,
a 4,096-node graph, cycle rejection, metadata-edge exclusion, serialisation,
and existing source-assignment and documentation guards.

The old test that rejected any tuple operand observation now has separate
native-observation and source-identity checks. Tuple graphs are observable;
their source object-identity proof remains unavailable. Two copied-value tests
now check the shared inventory's diagnostic while retaining rejection.
Observing more native operands does not discharge source-value, lookup-timing
or effect obligations. In particular, future compound-read and decorator-input
joins must resolve each read at its original source evaluation cut.

The full OpenHCS plus eight-library scan completes all 79 detectors with no
omissions and identical findings. Implementation refresh reports 54.096 scan
seconds (35.268 preparation, 18.828 analysis), 61.89 command-wall seconds and
1,379,856 KiB peak RSS. The following unchanged scan reports an exact cache hit
in 1.402 scan seconds and 2.52 command-wall seconds, with the same findings.
Reports are `native-operand-refresh-live.json` and
`native-operand-warm-live.json`, with their `.stderr` records. These precede the
shared transition-guard refactor above and are not cold or novel-edit timings.

## Definition-application observations

The definition-application checkpoint's disjoint Python 3.11 suites report **5,990 passed, 187 failed and 68
skipped**. Support reports 5,230 passed, 152 failed and 68 skipped in 115.85
seconds; core reports 760 passed and 35 failed in 64.19 seconds. Their exact
failed-node sets match the completed-deletion checkpoint. Reports are
`native-application-final-support-311.txt` and `native-application-core-311.txt`
under `/home/ts/nra-global-scan-CXVD7T`.

`NativeCreationOperation.advance` now receives the selected backend authority.
The CPython backends declare their application preparation sequence using the
existing operation classes: `PRECALL` before `CALL` in 3.11, and direct `CALL`
in 3.14. The creation-chain observer validates arity, instruction roles and
source ranges, retaining the original application sites and result store.

`AppliedNativeFunctionExecution` keeps raw function creation separate from the
installed application result. Raw-installation queries remain unavailable for
that result. `NativeDefinitionApplication` inherits `NativeCaptureSite`, sharing
its frame and offset contract directly. The result binding feeds the existing
return observer and original-binding lookup.

The authored DSL batches `.codex-temp/creation_backend_context.py` and
`.codex-temp/application_site_inheritance.py` performed the backend-context
factoring and site inheritance/relocation with reference updates. These are
explicit syntax transformations; source-value and invocation proof obligations
are unchanged.

Matched focused checks report **94 passed, one skipped on Python 3.11**, and
**92 passed, three skipped on Python 3.14**. Reports are
`native-application-final-focused-311.txt` and
`native-application-final-focused-314.txt`. Authored native-execution fixtures
check application order, original function code, distinct returned objects,
module/class/global installation, and retained frame identity. Further checks
cover malformed preparation, arity and ranges, missing return continuations,
raw-result separation, and serialisation without recompilation.

This completes observation of the native application chain, not source-side
decorated-function completion. Callee values, implicit argument/result joins,
activation, invocation effects and metadata access remain separate obligations.
The shared operand-production boundary also serves assignments and compound
expressions. The native-proof batch remains unfinished and uncommitted.

The full OpenHCS plus eight-library regression scan completes all 79 detectors
with identical findings. Implementation refresh reports 52.744 scan seconds
(34.090 preparation, 18.654 analysis), 61.02 command-wall seconds and
1,371,740 KiB peak RSS. The following exact-cache run reports 1.412 scan seconds
and 2.48 command-wall seconds. Reports are `native-application-refresh-live.json`
and `native-application-warm-live.json` with their `.stderr` timing records.
These are existing-cache implementation-refresh and unchanged-input checks,
not new cold-cache or novel-edit benchmarks.

## Completed deletion integration

The deletion checkpoint's disjoint Python 3.11 runs report **5,973 passed, 187 failed and 68
skipped**. Support reports 5,213 passed, 152 failed and 68 skipped in 114.04
seconds; core reports 760 passed and 35 failed in 63.57 seconds. Reports are
`native-deletion-support-311.txt` and `native-deletion-core-311.txt` under
`/home/ts/nra-global-scan-CXVD7T`. Compared with the source-point-query checkpoint,
six previously failing cases pass and no new failed nodes appear. The core
failed-node set is unchanged.

`SourceBindingReturnABC` owns the original source binding, operation and frame
used by assignment and deletion continuations. `SourceDeletionReturn` requires
the canonical completed source cut, then joins the original emitted deletion
and return in that executing frame. Source execution still proves destination
presence and displaced-value release. The continuation does not replay a
deletion against the completed namespace.

`NativePrimitiveOperation` declares deletion capture and its conditional
completion boundary. `NativeStoreStream` feeds that original binding into the
existing uninterrupted return observer. `NativeBindingTransfer` owns target
geometry; `NativeValueStore.source_span` derives from it. The authored DSL batch
`.codex-temp/derive_store_target_span.py` removed the duplicated field and its
constructor argument. This is an explicitly authored syntax transformation,
not automatic semantic promotion.

The fixed regression cases cover deletion of `__doc__`, class-local and global
bindings, and deleted `__init__`, `__new__` and `__init_subclass__` hooks. Added
checks compare actual subprocess class-body return namespaces, exercise
multiple deletion targets, and reject copied events, foreign frames, ambiguous
bindings and instruction addresses, incomplete ranges, jump-entry cuts,
unavailable names, and unproved work before or after the deletion.

Exact Boolean release now joins the existing backend-owned singleton lifetime
contract. This uses CPython's static singleton and non-subclassable exact type
in [3.11.11](https://raw.githubusercontent.com/python/cpython/v3.11.11/Objects/boolobject.c)
and its immortal singleton implementation in
[3.14.6](https://raw.githubusercontent.com/python/cpython/v3.14.6/Objects/boolobject.c),
under the backend's valid reference-ownership premise. It does not admit
integer, float or dictionary instance lifetimes.

Matched focused validation reports **132 passed on Python 3.11** and **130
passed, two skipped on Python 3.14**. Reports are
`native-deletion-final-focused-311.txt` and `native-deletion-focused-314.txt`.
The environments retain metaclass-registry 0.1.4 and 0.2.1 respectively.

The publication batch remains unfinished. Decorated results, nested class and
import completions, non-scalar production boundaries and explicit native
metaclass construction remain separate obligations.

The full OpenHCS plus eight-library scan completed all 79 detectors with no
omissions and identical findings. Implementation refresh reports 52.247 seconds
for the scan (33.479 preparation, 18.768 analysis), 58.77 seconds command wall
time and 1,354,840 KiB peak RSS. The following unchanged run reports an exact
cache hit in 1.413 scan seconds and 2.46 command-wall seconds. Reports are
`native-deletion-refresh-live.json` and `native-deletion-warm-live.json` with
their `.stderr` timing records. These use the existing cache after an
implementation change; they are not new cold-cache or novel-input-edit timings.

## Current construction integration

At the construction-integration checkpoint, disjoint Python 3.11 runs reported
**5,927 passed, 193 failed and 68 skipped**. Support reported 5,167 passed,
158 failed and 68 skipped in 110.63
seconds; core reports 760 passed and 35 failed in 62.85 seconds. Reports are
`native-construction-global-store-support-311.txt` and
`native-construction-global-store-core-311.txt` under
`/home/ts/nra-global-scan-CXVD7T`.

Compared with the initial construction integration, preparation-first validation
resolved seven failures and global-function installation resolved four, with no
new failed nodes in either comparison. The support suite still has 53 failures
additional to its 105-failure preintegration baseline. The core failed-node set
is unchanged. These are historical construction-integration counts; the
completed-deletion checkpoint above supersedes them.

Ordinary `SourceClassEntry` construction now consumes `native_tail`, including
compiler-generated namespace changes after the last source store. The shared
`SourceClassBodyEntryABC.completed` property validates preparation and the
original completed source body before consulting cached construction admission.
Explicit native-metaclass construction remains a separate open obligation.

Fresh empty-map production joins the original source dictionary creation through
the existing storage interpreter. Its empty operand tuple belongs to the
declaration; callers cannot supply a different tuple. Nonempty and unpacked
maps do not acquire this proof.

Some compilers produce the return operand for a one-line `pass` class during
the prologue. The join authenticates the original class capture, return receipt,
and actual returned operand before selecting the unique original prologue
production. It resolves that production in its prologue context. Copies,
foreign frames, ambiguous addresses, and other historical operands are rejected.

Raw global-function installation shares `InstallNativeFunction.advance` with
local installation. The leaf declares `STORE_GLOBAL` and derives its registry
name from that operation. Real-execution tests check that the function appears
in module storage, not in the enclosing class dictionary, and that its code is
the originally observed function code. Decorated-function and fast-local
installation retain their separate proof requirements.

Matched focused global-function, construction, empty-map and continuation checks
report **81 passed on Python 3.11** and **79 passed, two skipped on Python 3.14**.
The environments use metaclass-registry 0.1.4 and 0.2.1 respectively. Reports are
`global-function-installation-matched-311.txt` and
`global-function-installation-focused-314.txt` in the same directory. A separate
Python 3.11 selection including subclass-hook tests reports 75 passed and one
failure for a decorated hook (`global-function-installation-focused-311.txt`).

These changes are unfinished integration work, not a passing publication batch.
Earlier sections retain measurements from their named implementation stages.

## Implemented boundaries

- `CollectedFamilyBatch` retains the full-family collector's publication
  signature. `CompactProjectionCacheSource` supplies the collector's actual
  module identity and cache directory, including explicitly selected caches.
  Shard construction consumes that receipt instead of publishing the same
  family twice or rereading fresh metadata.
- `CliArguments` owns one parsed invocation and one decoded plan. Commands and
  execution modes derive their deadline policy from that invocation. Exact
  recipe batches have no analysis deadline; architecture guards and requested
  findings retain the scan deadline. This does not remove process-level limits
  imposed by a caller.
- `SourceClassBodyEntryABC.final_evaluation` selects the original final
  statement and canonical completed source cut. Changed statement order,
  removed trailing statements and copied nodes cannot reuse this evidence.
- `ModuleSyntaxIndex.children_by_node` derives ordered children from the
  existing unambiguous parent projection. Completion queries reuse this view
  instead of walking every syntax node for every class. Ambiguous original
  nodes remain excluded; a later AST edit does not change the retained view.
- `NativeValueStore` retains an adjacent production and binding; its target
  span derives from the binding. `NativeOperandStack` and `NativeBindingTransfer` own the
  interpretation. Scalar queries require scalar contents from the original
  produced value. `NativeConstantStore` retains the narrower entry-only,
  text-only, same-span documentation contract.
- `SourceAssignmentStore` joins an original evaluated assignment, its
  canonical completed activation, original RHS value and compiler receipt.
  The historical store does not establish the namespace after subsequent
  writes or after compiler-generated class-body instructions.
- `SourceNativeFrameResolver.require_event_available_in` owns canonical-cut,
  original-event membership and source-frame checks for both scalar and
  definition installation. Definition captures retain their original-operation
  and creation checks; scalar installation retains its value/store join.

The indexed-completion change is expressed by the two-stage DSL plan in
`docs/examples/indexed_class_completion.py`. The earlier completion query is
expressed by `docs/examples/class_body_completion.py`. These plans do not cover
all hand-written changes in the broader working-tree batch.
The five-stage plan in `docs/examples/source_installation_availability.py`
factors the shared frame checks into the existing base and updates both
consumers in one simulation.

## Verification

Focused Python 3.11 completion, scalar-installation, original-suite and syntax
index checks: 85 passed. Python 3.14 scalar-installation, scalar-store,
completion, syntax-index and initial CLI checks: 88 passed. The expanded CLI
suite passes all 21 tests on Python 3.11, including real subprocess simulation
and application of dependent stages, stdin consumption and guard deadlines.
After the shared frame factoring, the combined Python 3.14 installation,
completion, native scalar, original-event and CLI checks report 188 passed.

The indexed-completion full Python 3.11 run reported 140 failed, 5753 passed and
66 skipped in 162.11 seconds. Its exact failed-test set matches the preceding
140-failure run. The four regressions introduced during the recent work were
repaired relative to the earlier 144-failure run. This full result predates the
shared frame factoring; its focused Python 3.11 checks report 144 passed.
Two subsequent whole-suite attempts reached the 100% progress indicator but
exited at the external 165-second deadline before a final summary, including
one with traceback output disabled. Neither is recorded as a completed run.

The final checkout was subsequently tested in two disjoint groups, each with
eight workers and the same 165-second external limit:

| Group | Passed | Failed | Skipped | Seconds |
| --- | ---: | ---: | ---: | ---: |
| `tests/test_refactor_advisor.py` | 755 | 35 | 0 | 77.66 |
| All other tests | 4999 | 105 | 66 | 90.22 |
| Combined coverage | 5754 | 140 | 66 | 167.88 |

Both processes completed normally with pytest's failure exit code. Their
combined failed-test set exactly matches `proof-completion-indexed-311.txt`.
The partition changes execution grouping, not test selection or assertions.
The shutdown delay in the unpartitioned runs has not been diagnosed.

The post-store cold OpenHCS package scan took 109.537 seconds and reported all
79 detectors, zero omitted detectors and the exact same grouped findings
payload as `optimised-one-edit.json`. A short focused test overlapped the start
of that scan; it is a completeness/budget check, not an isolated speedup ratio.
Broader performance measurements are in `full_package_scan_performance.md`.

NRA's own production-package cold scan completed in 30.741 seconds, with all
79 detectors and zero omitted detectors. It returned zero findings. This
records the supported detector results, not a proof that the package has no
architectural debt or unsupported refactoring cases.
The unchanged repeat took 0.499 seconds with validated exact cache coverage
and the same findings payload (`nra-production-warm.json`).

Raw test and scan records are in `/home/ts/nra-edit-latency-Y2OuOK/`:
`proof-completion-baseline.txt`, `proof-completion-after.txt`,
`proof-completion-cold.json`, `proof-completion-indexed-311.txt` and
`proof-completion-shared-frame-311.txt`. The self-scan is `nra-production.json`.
Completed final regression reports are `proof-shared-frame-core-311.txt` and
`proof-shared-frame-support-311.txt`.

## Remaining proof obligations

The earlier 140 outstanding failures included registry conversion, collector inheritance,
class-member moves, product mutation, native subscriptions and dependent cache
or CLI cases. They are not all obsolete assertions.

Explicit native-metaclass entry still refuses completed construction. Ordinary
class construction now uses the native tail; its integration exposes missing
completed-source/native boundaries for other final operations. Native tail
interpretation retains original compiler cells and overwritten-value lifetime
checks. Final scalar, name-read, empty-map and plain-function stores, plus
event-free original bodies, have joins. Final deletions, imports, nested class
results, decorated results and multi-instruction productions still need work.

Registration additionally requires a source-bound native result/effect law and
the complete selected mapping relation. Native creator identity alone does not
prove these results. Standard-library imports, dataclass decorator results and
subscription callback noninterference are separate outstanding obligations.
Broadening caller-side import exceptions or treating successful compilation as
an execution proof would not discharge them.

The eager caller lookup identified in the support trace has been replaced with
repository-owned point queries. Source queries inspect every possible module
owner, retain original declaration multiplicity, and reuse canonical source
projections. Full enumeration and transitive exposure checks retain their global
scope. The [performance record](performance_and_native_proof_completion.md#source-symbol-point-queries)
describes the implementation and its measured effect on DSL call edits.

No NRA commit or push has been made for this unfinished batch. OpenHCS's
separate documentation publication is complete at `20b2abb00` on
`openhcsdev/main`; it is not an NRA proof or release result.

## Native store continuations

`NativePrimitiveOperation` now supplies the actual return behaviour through the
existing operand interpreter: `RETURN_VALUE` consumes one result;
`RETURN_CONST` produces and returns its constant. Missing or extra operands
are rejected. Return instructions no longer discard their value.

`NativeStoreStream` observes one continuation for each uninterrupted primitive
suffix. Scalar assignment and function-installation observers supply their
original store bindings to this shared owner. Its stores share a `NativeReturn`
receipt with the original
frame, return operand, subsequent transfers and original store membership.
Unknown instructions, jump entries and instructions following a return close
that continuation. A later scalar store can start a new one; it cannot grant
an earlier store the missing evidence. No suffix is reinterpreted per store.

The membership projection uses native instruction offsets and retained binding
objects. It survives serialisation without persisting transient process IDs.
`SourceAssignmentStore.return_continuation` joins the receipt to the
original installed assignment and the complete canonical source cut. It does
not admit native lookup effects or turn a prepared namespace into a constructed
class. `SourceCreatedFunctionCapture.return_continuation` now joins final method
stores through the same completed-cut contract, owned by
`SourceNativeFrameResolver.require_complete_source_cut`.

Final focused checks report **135 passed, 13 skipped on Python 3.11**, and
**148 passed on Python 3.14**. They compare actual compiler instructions,
including the later `__static_attributes__` overwrite; reject foreign frames,
copied bindings, jump entries and unproved source execution; and verify shared
receipt identity after a warmed pickle round trip. A disposable subprocess
executes an authored class and checks its returned value and final native field
types. A 200-store case checks that interpretation remains linear and membership
validation neither hashes operand graphs nor materialises later-transfer views.
Reports are `native-return-final-311.txt` and `native-return-final-314.txt` under
`/home/ts/nra-global-scan-CXVD7T`.

The broad support run reports **5,055 passed, 105 failed and 66 skipped in
106.60 seconds**. It preceded the final separation of membership validation from
transfer-list materialisation and the added subprocess control, both covered
by the final focused checks. The core run after that separation reports
**760 passed and 35 failed in 60.86 seconds**. Each failed-test set exactly
matches its preceding baseline. Reports are `native-return-support-311.txt` and
`native-return-core-311.txt`. The 140 outstanding failures remain; this native
continuation evidence alone does not complete their missing proofs.

### Shared function and scalar continuation validation

Function stores use the existing `NativeCodeEmission.installation` and
`InstalledNativeFunctionExecution` evidence. No function-creation interpreter
was added. Native inventory frame binding publishes each shared continuation
once, and the compiled index retains its original frame and store membership.
`_NativeCompilationOutcome.require_function` owns canonical function-receipt
validation for both return continuations and fresh function namespace queries.

Expanded focused tests initially reported **187 passed on Python 3.11** and
**187 passed on Python 3.14**, covering method and module definitions, defaults and annotations,
async definitions, scalar/function receipt sharing, decorated-function rejection,
unknown suffixes, foreign or copied receipts, duplicate return evidence, warmed
serialisation and canonical completed source cuts. Existing function-birth and
storage tests are included. Reports are `shared-return-final-311.txt` and
`shared-return-final-314.txt` in `/home/ts/nra-global-scan-CXVD7T`.

The prepared-root test now proves the final method's continuation while still
rejecting the unproved metaclass result. Construction requires the completed
namespace after those native transfers and the selected native result/effect
law. Those obligations remain open. Broad regression runs for the shared-store
extension are in progress.

The first broad run found a value-comparison regression in serialised compilation
snapshots: new return receipts used object equality. `NativeReturn` and
`NativeValueStore` now inherit the existing `DataclassGraphValue` contract.
Original-store authentication still uses canonical object identity, independently
of snapshot equality. The warmed round-trip test checks both that snapshots
compare equal and that the old instance cannot authenticate a restored store.
The extended checks, including native function-creation tests, report **200
passed and 1 skipped on Python 3.11**, and **201 passed on Python 3.14**
(`shared-return-value-311.txt`, `shared-return-value-314.txt`). The support suite
rerun after this correction reports **5,074 passed, 105 failed and 66 skipped in
107.36 seconds**, restoring the exact preceding failed-test set
(`shared-return-support-final-311.txt`). The earlier 106-failure run is not a
clean regression result. The core run reported **760 passed and the same 35
failures in 60.87 seconds** (`shared-return-core-311.txt`).

A subsequent owner-boundary audit found that validating a receipt against an
index alone did not authenticate the index against its source compilation.
`NativePythonCompilation.execution_outcome` now owns that cache-to-source join
for all its native queries; the compiled index separately owns original receipt
membership. Injecting a foreign cached outcome is rejected before either birth
or continuation evidence is available. After this addition, all native/source
test files on Python 3.11 report **1,803 passed, 4 failed and 49 skipped in 12.93
seconds** (`shared-return-owner-broad-311.txt`). The four failures are exactly
the existing native subscription provenance failures from the broad baseline.

The corresponding Python 3.14 sweep reports **1,849 passed and 7 failed in
16.24 seconds** (`shared-return-owner-broad-314.txt`). All seven failed nodes
also occur in the retained pre-extension report
`/tmp/nra_native_preparation_diagnostics_full_314.log`: six subscription cases
and the generic-body-frame native binding case. These remain unfinished proof
work, not new failures introduced by the continuation extension.

### Prepared namespace tail and original operand inventory

`PreparedNamespaceTail` joins the completed prepared namespace to native work
after its final source store. A historical continuation can contain later
source assignments; replaying those transfers over completed source storage
would apply them twice. `SourceInstalledReturnABC` supplies the original
installation evaluation for both scalar and function stores.
`SourceModuleExecution.require_terminal_evaluation` checks the remaining
operations and effects from that evaluation's actual exit, using the same
completion check as `SourceClassBodyEntryABC.final_evaluation`.

The join rejects earlier stores, foreign frames, copied evaluations and changed
original body statements. An event-free trailing `pass` is supported. The
result retains compiler-generated writes, including the Python 3.14
`__static_attributes__` overwrite. It does not yet resolve their values or
release effects against the completed namespace, or admit class construction.
The historical DSL batch is `.codex-temp/prepared_namespace_tail.py`.

`NativeValueInventoryABC` owns original operand membership for both
`ExactNativeClassPrologue` and `NativeReturn`. Return receipts retain the
existing operand walk's productions, including the returned value. Membership
is derived by unambiguous instruction offset and original object identity;
equal copied operands cannot authenticate. The derived index survives warmed
serialisation and does not hash operand graphs. The historical DSL batch is
`.codex-temp/native_value_inventory.py`.

The prepared-tail checks report **118 passed on Python 3.11** and **118 passed
on Python 3.14** (`prepared-tail-311.txt`, `prepared-tail-314.txt`). After the
shared production inventory, focused checks report **110 passed, 13 skipped
on Python 3.11** and **123 passed on Python 3.14** (`native-values-311.txt`,
`native-values-314.txt`). The expanded Python 3.11 native/source sweep reports
**1,827 passed, 4 failed and 49 skipped in 12.13 seconds**
(`prepared-tail-broad-311.txt`). The four failed nodes are the same subscription
provenance cases as the preceding sweep. Reports are under
`/home/ts/nra-global-scan-CXVD7T`.

The corresponding Python 3.14 sweep reports **1,873 passed and 7 failed in
17.11 seconds** (`prepared-tail-broad-314.txt`). Sorted failed-node comparisons
are identical to both preceding native/source sweeps. These comparisons do
not establish that the unfinished native-proof batch is publishable.

The disjoint Python 3.11 support suite then completed with **5,099 passed,
105 failed and 66 skipped in 108.39 seconds**
(`prepared-tail-support-311.txt`). Its exact failed-node set matches
`shared-return-support-final-311.txt`.

The core suite completed with **760 passed and 35 failed in 63.30 seconds**
(`prepared-tail-core-311.txt`), again with the exact preceding failed-node set.
The two disjoint runs cover **5,859 passed, 140 failed and 66 skipped**. Both
completed within their individual 165-second limits. The outstanding failures
still prevent publication of the requested complete native-proof batch.

### Shared native namespace interpretation

`SourceNativeNamespaceABC` owns native value interpretation, local/global
lookup and local transfers. `SourceClassBodyEntryABC` supplies the entry cut
and initially empty prepared storage. `PreparedNamespaceTail` supplies the
completed source cut and resolves earlier native transfers before consulting
source storage. Its `member` query checks the supported tail operations before
exposing final slot values. Neither context owns a second runtime namespace.

`SourceNativeTypeCapture`, renamed from `SourcePrologueTypeCapture`, retains the
context that authenticates its original production. Captures cannot borrow
entry or tail productions from the other cut. `SourceFreshCellCapture` uses
that context's canonical class entry and actual bindings. Python 3.14's
`__classdictcell__` installation is joined to the prologue's original
`MAKE_CELL` evidence; foreign namespaces remain rejected.

Local overwrites use the existing captured-value lifetime check. A final method
stored under a compiler-overwritten field does not acquire an inferred release
guarantee. Tail global writes and cell writes still reject unsupported effects.
Class construction and registration are not admitted by these namespace queries.

The DSL batch `.codex-temp/native_namespace_context.py` expresses the factoring
and the declaration rename; `.codex-temp/native_cell_context.py` shares the cell
join. The semantic member-promotion operation refused the module's currently
unproved execution effects. The reviewed move was instead expressed as explicit
DSL member insertions/deletions, deriving existing member text through
`ClassMemberSourceSelection`. This is an authored syntax transformation, not a
successful semantic-promotion proof. The declaration rename and import updates
used `RenameTopLevelDeclarationAuthorityOperation` successfully.

Focused namespace, cell and existing ownership checks report **138 passed,
13 skipped on Python 3.11** and **151 passed on Python 3.14**
(`native-namespace-final-311.txt`, `native-namespace-final-314.txt`). Added cell
field and old-value lifetime checks report **62 passed, 15 skipped on Python
3.11** and **77 passed on Python 3.14** (`native-namespace-extra-311.txt`,
`native-namespace-extra-314.txt`).

The complete namespace comparison runs the authored registry example in a
subprocess and traces its class-body return, before metaclass construction.
Every prepared member's type and the returned operand's type match the
interpreted namespace. Its suite reports **10 passed, 2 skipped on Python 3.11**
and **12 passed on Python 3.14** (`native-namespace-runtime-311.txt`,
`native-namespace-runtime-314.txt`). Skips reflect compiler-generated operations
absent on 3.11. These are runtime checks of the authored fixture, not execution
of analysed user source by NRA.

The native/source sweeps report **1,836 passed, 4 failed, 51 skipped in 11.93
seconds on Python 3.11**, and **1,884 passed, 7 failed in 15.99 seconds on Python
3.14** (`native-namespace-broad-311.txt`, `native-namespace-broad-314.txt`). They
precede the final subprocess comparison, which is covered separately above.
Both retain the preceding failed-node sets. Reports remain under
`/home/ts/nra-global-scan-CXVD7T`.

The final disjoint Python 3.11 runs report **5,109 passed, 105 failed and 68
skipped in 110.34 seconds** for support, and **760 passed, 35 failed in 61.65
seconds** for core (`native-namespace-support-311.txt`,
`native-namespace-core-311.txt`). Their combined coverage is **5,869 passed,
140 failed and 68 skipped**, with both failed-node sets identical to the
preceding prepared-tail baseline. The native-proof batch remains unfinished.

### Declaration-selected stores and event-free bodies

`CompactBindingValueResolverABC` owns interpretation of an already-selected
binding; alias traversal remains in `CompactBindingResolverABC`.
`SourceInstalledReturnResolver` uses the binding operation and definition owner
to select scalar or function installation evidence. `SourceClassBodyEntryABC`
derives its final store from its existing flow mutations. Names and a separate
function-versus-assignment dispatch table do not select that evidence.

`PreparedNamespaceContinuationABC` owns namespace lookup, native transfer
interpretation, original-value authentication and completion. Its two boundary
implementations admit instructions strictly after an installed source store,
or at and after the native prologue boundary of an event-free source body.
`EntryOnlyNamespaceContinuation` requires the entire original body interval to
contain no source operations or effects. An absent assignment alone does not
satisfy this condition.

`NativePythonCompilation.return_from` authenticates the original body receipt
and selects its unique return through the frame origin. Entry-only completion
also requires the shared uninterrupted observer to cover the original prologue
boundary. A later native suffix cannot replace that coverage. `NOP` uses the
existing operand interpreter's no-operation behaviour; jump entries still
invalidate a continuation. No synthetic source event or target-code execution
is introduced.

The authored DSL batches are `.codex-temp/selected_store_continuation.py` and
`.codex-temp/entry_only_continuation.py`. They use explicit member operations
and derive moved method text from its original declaration. They are historical
applications, not replayable plans against the already-edited checkout and not
automatically proved semantic promotions.

Focused checks report **99 passed, 2 skipped on Python 3.11** and **101 passed
on Python 3.14** (`entry-tail-final-311.txt`, `entry-tail-final-314.txt`). They
include actual subprocess class-frame returns, all prepared member types,
event-free and eventful bodies, original-source mutation, foreign and copied
receipts, warm serialisation, ambiguous returns, missing boundary coverage and
real loop jump targets. The reports are under
`/home/ts/nra-global-scan-CXVD7T`. These checks do not admit metaclass construction
or registration, or final call-result, import, deletion or class-result stores.
The subsequent assignment extension below adds final alias stores. Broader
regression results follow.

Expanded prologue and continuation coverage reports **136 passed, 15 skipped
on Python 3.11** and **151 passed on Python 3.14**
(`entry-tail-prologue-311.txt`, `entry-tail-prologue-314.txt`). An older
unsupported-instruction fixture used `NOP`; it now uses the unsupported
`RAISE_VARARGS` instruction, retaining the same rejection assertion.

The final disjoint Python 3.11 runs report **5,141 passed, 105 failed and 68
skipped in 111.64 seconds** for support and **760 passed, 35 failed in 64.35
seconds** for core (`entry-tail-support-final-311.txt`, `entry-tail-core-311.txt`).
Combined coverage is **5,901 passed, 140 failed and 68 skipped**. Both exact
failed-node sets match the preceding namespace baseline. The initial support
run with the stale `NOP` fixture is superseded by this completed rerun. The
larger native-proof batch remains unfinished and unpublished.

### General adjacent stores and original source reads

`NativeValueStore` replaces `NativeScalarStore`: storage provenance belongs to
the receipt, while scalar contents belong to `NativeConstantValue`.
`NativeValueStoreStream` and the compilation's `value_stores` inventory contain
all adjacent producer/store pairs. `value_store_for` and `scalar_store_for`
query that same original inventory; the scalar query additionally requires
scalar production evidence. General receipts do not admit unknown source reads,
calls or effects. Documentation retains its separate entry-only text contract.

`SourceNativeStorageABC` now owns the shared native lookup/transfer interpreter.
`SourceNativeNamespaceABC` adds class-specific compiler-cell evidence.
`SourceAssignmentStore`, renamed from `SourceScalarAssignmentStore`, supplies
the original RHS capture prefix and its actual local, global and builtin
namespaces. Its native name reads use that shared interpreter. They join the
same source capture or require the existing same-object proof; matching types
alone is insufficient. Typed scalar productions retain their exact type/content
comparison. This does not infer scalar contents from a general produced type.

Final aliases to builtin objects, scalar values, source-created dictionaries and
deferred functions now supply native continuations. A registry root may end
with `__registry__ = REGISTRY`; its metaclass construction remains separately
unproved. Historical stores retain their original RHS value after later source
overwrites, and warm queries still reject changes to the original class suite.

The authored DSL batches are `.codex-temp/native_value_store.py` and
`.codex-temp/source_assignment_join.py`. Declaration renames update their
consumers and imports; the remaining changes use explicit member operations.
These are historical applications, not replayable plans against the edited
checkout or claims of automatically proved semantic promotion.

Focused validation reports **195 passed, 2 skipped on Python 3.11** and **197
passed on Python 3.14** (`assignment-join-final-311.txt`,
`assignment-join-final-314.txt`). The checks include real subprocess namespace
comparisons before class construction, original read points, local/global
lookup, wrong-object rejection, foreign/copy/ambiguity guards, mixed-store
receipt sharing and warmed serialisation. Reports remain under
`/home/ts/nra-global-scan-CXVD7T`.

The final disjoint Python 3.11 runs report **5,189 passed, 105 failed and 68
skipped in 108.61 seconds** for support and **760 passed, 35 failed in 62.16
seconds** for core (`assignment-join-support-311.txt`,
`assignment-join-core-311.txt`). Combined coverage is **5,949 passed, 140 failed
and 68 skipped**. Both failed-node sets exactly match the preceding entry-tail
baseline. Native class construction and registration remain unfinished; no
commit or push has been made for this batch.
