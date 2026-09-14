# Dependency-aware source-proof reuse

Checkpoint record, 14 September 2026. This is a conservative first slice of
the cross-edit proof-reuse requirement on
`checkpoint/native-proof-performance-20260914`; it does not complete persistent
reuse across separate scan invocations.

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
`.codex-temp/dependency_proof_reuse.py` using declaration-targeted import,
member-insertion, and exact target-patch operations.

## Validation

The focused source-transition, source-call, source-read, flow-identity,
registry-correspondence, and dependency suites report **94 passed** with eight
workers in 2.63 seconds on Python 3.11. They cover unchanged projection and
execution identity, changed-owner exclusion, a changed provider invalidating an
unchanged consumer's call resolution, a newly created consumer becoming visible,
foreign-owner rejection, and non-transfer of global call-resolution caches.

A broader codemod, source, product-flow, and registry run reports **2,148 passed,
69 failed, and 11 skipped** in 55.68 seconds. The exact 69 failed nodes match an
immutable `ee1ae0c` run, which reports the same **2,148 passed, 69 failed, and 11
skipped** in 55.78 seconds. These are the checkpoint's existing native-admission
failures, not regressions from source-proof reuse.

## Remaining work

This slice improves multi-stage in-memory DSL refactoring, where successive
virtual edits share exact unaffected source proof owners. Separate process scans
still require a persistable dependency receipt capable of proving positive and
negative global dependencies before completed global proofs can be reused.
Source-path invalidation alone remains insufficient: new consumers, changed
providers, declaration ambiguity, public-export changes, star imports, escaping
references, and cycles must all participate in that receipt.
