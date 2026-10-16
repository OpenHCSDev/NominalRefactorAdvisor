# Dependency-aware source-proof reuse

Checkpoint record, 14 September 2026. This is the source-proof reuse boundary
for the cross-edit requirement on
`checkpoint/native-proof-performance-20260914`.

## Ownership and invalidation

`ParsedModuleSourceProjection` remains the sole authority for a virtual source
transition. `SourceProductFlowRepository.projected_with_source_projection`
first authenticates the projection's exact original `ParsedModule` owners. It
then creates a fresh repository for the projected source state and retains only
the source projection and module-local native execution belonging to each
unchanged owner object.

Changed and created modules receive new parsed owners and therefore cannot
inherit source observations or native execution evidence. The projected
repository receives none of the prior repository's global cached properties.
Call resolution, product construction, public export, star-import ambiguity,
cycle, absence, and declaration-multiplicity queries are consequently derived
again from the complete projected module set. An unchanged consumer can reuse
its local source proof while a changed provider or newly created consumer still
changes the repository-wide result.

`CodemodSourceSnapshot` carries the projected repository as the original proof
object, not as copied metadata. Access re-authenticates the repository's parsed
owners against the snapshot before exposing it through the existing
`product_flow_repository` and `module_binding_proof` contracts. Independent
snapshots continue to create independent source activations; only a proved
virtual transition can retain an unchanged activation.

The production edit was applied through the NRA DSL batch
[`dependency_proof_reuse.py`](../examples/dependency_proof_reuse.py) using
declaration-targeted import, member-insertion, and exact target-patch operations.

## Validation

After integration with native admission, the focused source-transition,
codemod-runtime, product-flow, and ownership surface reports **266 passed and 8
skipped** with eight workers in 17.87 seconds on Python 3.11. It covers unchanged
projection and execution identity, changed-owner exclusion, a changed provider
invalidating an unchanged consumer's call resolution, a newly created consumer
becoming visible, multistage retention, foreign-owner rejection, and
non-transfer of global call-resolution caches.

The complete Python 3.11 suite passes in the prescribed eight-worker split:
**7,136 passed and 71 skipped**. Python 3.14 passes **7,171 tests with 36
skipped**. No known test failure is retained by the combined branch.

## Persistence boundary

This reuse intentionally belongs to a multi-stage in-memory DSL trajectory.
Completed native proofs retain their original loaded declaration, source owner,
activation, evaluation cuts, and observed events. Serialising a summary and
restoring it as a completed proof in another process would replace that evidence
with a structurally similar claim and is therefore outside the proof contract.

Persistent scan caches remain a separate derived-result system. Their global
detector identities already derive from complete demanded-family content
signatures; changed family contents invalidate the affected detector result.
Evidence-local partial findings are explicitly nonterminal and cannot certify a
complete scan. Neither cache surface is treated as native execution evidence.

The final integrated full-package scan completes in 46.757 seconds cold, 1.197
seconds unchanged-warm, and 4.527 seconds after one novel source edit. All three
runs cover 79 of 79 detectors with the same 180 active findings and identical
semantic report content. The complete semantic projection hash is
`4191f7d7c49191222549f74d03ad17dd924d0c3f4d08cc9acc17e92a487d8743`.
