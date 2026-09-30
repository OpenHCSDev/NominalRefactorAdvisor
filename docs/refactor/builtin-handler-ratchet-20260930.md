# Built-in handler dispatch ratchet

Owner: Kepler. Scope: user item3, 2026-09-30.

Extend the existing refactor-audit Measure owner in
`skills/refactor-audit/scripts/audit/measures.py`. Count MroDispatch `@handles`
arms over dict/list/str/int/float/bool/tuple outside admitted codec modules as
type dispatch. These handlers classify raw representation shapes; changing
syntax from isinstance to decorators does not give the domain nominal owners.
Use declaration-owned measurement and existing consumers, not another checker
or a string registry. Relevant patterns: IMPL-3, IMPL-12, BOUND-2.

Prove growth on the actual Core PR421 acp_failure.py historical specimen and
verify aliases, codec admission and per-function dispatch ownership. Inspect
actual Core/OpenHCS ratchet admission of StringDispatch and TypeSwitch; add
missing per-function admission through their existing owners. Keep the primitive
handler measure visible separately so a one-arm decoration cannot escape a
three-arm function threshold.

Rebuild the authoritative `.skill` archive from the same updated skill source
after the existing tool regression suite passes. Record exact source/archive
identity, changed files, deleted lines and consumer integration gaps. Parent
coordinates downstream OpenHCS pin publication.

No production ACP rewrite: Arendt owns that independent surface. No frozen
release/package mutations, provider calls or public inputs. The NRA main
checkout's dirty uv.lock and other work are preserved. This worktree is isolated
under `/home/ts/wt/nra-builtin-handler-ratchet-20260930`.

Acceptance: real historical debt growth is measured; the existing family and
CLI discover the new declaration; authorized codecs remain explicit; actual
consumer ratchets include the per-function measures; archive matches source.
Draft scope only: no implementation or readiness claimed yet.
