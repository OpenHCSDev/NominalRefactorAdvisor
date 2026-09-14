# Native-proof failure classification, 14 September 2026

This inventory classifies the 209 failures retained by
`checkpoint/native-proof-performance-20260914` at commit `54124b7`.  It is a
root-cause map for completing the proof model, not a waiver for failed tests.
The exact node IDs remain in
[checkpoint_failures_20260914.txt](checkpoint_failures_20260914.txt).

## Reproduction

The 23 affected files were rerun from the checkpoint with eight workers and a
165-second bound.  The result reproduced the checkpoint exactly: **209 failed,
1328 passed, 1 skipped in 91.62 seconds**.

## File-level clusters

| Shared boundary | Failing tests | Affected test files | Current evidence |
| --- | ---: | --- | --- |
| Source/runtime admission chain | 182 | `parameter_conveyor` (46), `registry_destination_admission` (37), `refactor_advisor` (35), `registry_destination_declaration` (14), `collector_inherited_dispatch` (13), `collector_migration_semantics` (6), `carrier_expansion` (4), `collector_base_authority` (4), `adjacent_declaration_insertion` (4), `registry_policy_integrity` (4), `analysis_cache` (3), `semantic_descent` (3), `registry_original_values` (2), `conveyor_mutation_targets` (2), `class_member_move_operation` (2), `argument_capture` (1), `autoregister_retirement` (1), `call_target_capture` (1) | These consumers converge on open original-source captures.  Representative positive cases terminate at `unproved_execution_effects`; the complete parameter-conveyor fixture reaches an unconditioned `dataclasses.dataclass` invocation, while source-created functions stop at the deliberate `Source function body execution remains unproved` gate.  The cluster therefore needs execution authority, not consumer-specific exceptions. |
| Product authority and mutation attribution | 21 | `product_flow_authority` (16), `product_mutation_targets` (5) | Declaration candidates exist, but runtime mutation attribution is removed or fanned out when source captures stay open.  In the canonical `_CacheKey` fixture both annotation writes are misclassified as unresolved receiver mutations because the compiler-created annotation namespace cannot be reached past the open decorator call.  Function-local imports and enclosing bindings additionally require real function frames. |
| Registry-key identity/equivalence | 4 | `registry_key_equivalence` (3), `registry_identity_integrity` (1) | The mapping-key proof does not yet retain enough original identity/equivalence evidence for the distinct positive case.  This remains separate from function activation; spelling equality must not substitute for key identity. |
| DSL/runtime source rewrite | 2 | `signature_codemod_runtime` (2) | The historical extraction batch exits nonzero.  This must be re-evaluated after source/runtime admission closes so a downstream preflight rejection is not patched locally. |
| **Total** | **209** | **23 files** | |

## First coherent implementation boundary

The source/runtime cluster already has separate authorities for source flow,
compiler entry-to-return receipts, function declarations, exact call binding,
effect occurrences, and namespace access.  The missing join is one canonical
activation per original invocation:

1. `SourceFunctionCall` owns the invocation and selects its exact callee and
   bound arguments.
2. A function-entry declaration must derive fresh activation-local storage from
   that invocation, while retaining the callee creator's globals and builtins.
3. Initial parameter values must dispatch from `InitialCompactParameterBinding`
   to that entry.  They must not pass through the generic dynamic-binding
   rejection or a caller-maintained parameter-kind table.
4. The function execution must retain its own kernel and frame.  Distinct calls
   may not share locals merely because they share a declaration.
5. Only source and native return paths that join at the same original body may
   publish a completed result. Final locals retain their ordinary release
   obligations unless the joined return value supplies the escaping reference.
   Defaults, variadic activation containers, closures, suspended execution,
   branches, and external writes remain closed until their own evidence is
   implemented.

This boundary is upstream of the registry, collector, product, conveyor, and DSL
consumers.  Those consumers should be rerun after each proof increment; none
should acquire a local fallback or a parallel registry to compensate for an open
activation.

## Separate follow-on boundaries

- Derive supported native Python callable behavior from its current verified
  implementation source.  An absent hand-written operation condition is not by
  itself evidence that a standard-library operation failed, but neither may an
  operation condition be manufactured automatically.
- Transport completed child effects back into later caller cuts, including
  global writes, before admitting caller state after a source invocation.
- Add default, variadic tuple/dictionary, closure-cell, return-value, and release
  evidence through their nominal declarations.
- Re-evaluate the four key-identity failures independently after the execution
  cluster contracts.
- Re-evaluate the two DSL runtime failures only after their production proof
  prerequisites pass.
