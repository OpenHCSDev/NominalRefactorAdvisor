# Original family boundary collector

## Source and ownership

The original NRA owner handed `feat/original-family-flattened-collector` and its
clean checkout at `e19f0c4c24db6f85c400c3605cca49a9a95a7ee9` to the receiving
integration owner. The predecessor is the existing
`checkpoint/native-proof-performance-20260914` publication branch; this change
does not fold the unrelated NRA main backlog into the collector.

`FamilyFlattened` and BOUND-8 were already present in the original skill archive,
SHA `ef0367d878cc57565f2b257a8e6647a234d4ba4298f61391709d9c3c82bec9f7`.
The class and the BOUND-8 section are promoted verbatim. The newer primitive
handler and dispatch-arm collectors stay in place.

## Owner and consumers

- `Measure` registers the new class through the existing `Family` declaration.
- `_dispatch` selects its declared `Call` and `Return` nodes.
- `measure_source` parses and counts through that dispatch.
- `Repository.measure` and `Change.delta` read the original Git source.
- `Tally.as_record`, `Census`, the census CLI and merge review discover the
  measure through `Measure.members`; no second roster or scanner is added.
- The existing setuptools package mapping publishes the owner as
  `refactor_audit.measures.FamilyFlattened` for the agent-comms ratchet.

The count is a syntax lead. It counts member names passed to calls or returned,
including tuple and boolean alternatives. It omits formatted display text and
passing the member itself. It does not prove domain ownership or resolve dynamic
aliases. Codec and schema site exemptions belong to F4's reviewed ratchet policy;
the original collector is unchanged.

## Before and remaining confirmation

The existing `Package.load` collector parsed all 18 skill Python modules at the
predecessor with zero omissions. The declaration registered 23 measures before
this change. All collector, repository, record and CLI consumers above were read.

Source promotion is implemented. The final check will bind a real historical
compaction event change to the count, run the affected audit regressions, and
import the original class from the normal file wheel. No full NRA runtime,
installed application, native process or provider acceptance is claimed.
