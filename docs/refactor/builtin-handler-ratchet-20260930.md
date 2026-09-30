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
## Scoped acceptance

PR14 is preserved and closed in favor of stacked PR15. Verified remote base
`checkpoint/native-proof-performance-20260914` is
`9c4546964e899d06c74c896143b083f6b343da24`; no PR for that prerequisite branch
exists. The item3 diff contains audit, collector, package and proof files, not
the four prerequisite commits or newer main's unrelated native changes.

Code/package checkpoint: `d392c5e4cd189ce1203127a746337ad498a0330d`.
There is one classifier, `audit.handler_declarations.BuiltinHandlerDeclarations`.
The audit Measure and Core466's original per-file Measure delegate to it. The
lightweight distribution exposes the same source directory, with no NRA parser
or tree-sitter dependency and no copied detector in Core.

The existing skill suite passes four cases. The real PR421 specimen grows from
zero to six primitive arms while the older TypeSwitch count falls from one to
zero. Fixture provenance retains both exact Git revisions and source hashes.
Archive SHA256 is
`78bbad2b86182f64a65d643a96760bc0dfbb4de15cfeeeb5ca63a47e84034d80`;
each of its 32 entries was verified byte-equal to source.

Receiving Core466 checkpoint `e863bbbc174910aec4faeb6e67497a88b95b3b4e`
builds and installs a real wheel in its own worktree. The installed
`agent-comms-ratchet` rejects the original PR421 Git comparison with exit1 and
`BuiltinHandlerTypeSwitch:src/agent_comms/acp_failure.py` delta +6. Installed
command and collector bytes match their reviewed sources. The exact collector
Git dependency is the NRA code checkpoint above. Twenty-four original
per-function command cases passed; both new primitive growth/codec command
cases pass after fixing the initially missing abstract occurrence hook. The
first nonadmitted report is retained in Core's receipt.

Deleted lines: 2 in NRA and 1 in Core at these code checkpoints. This is new
ratchet coverage; existing StringDispatch/TypeSwitch definitions remain owners.
No provider calls, public owner actions or default installation changes occurred.
This proves the affected installed command, not unrelated UI/ACP readiness.

OpenHCS consumer gap: no local ratchet was found in its current tools, scripts,
tests or .github tree. Parent owns notifying its active agent of the exact
published NRA/Core pins and closing that consumer admission gap.
