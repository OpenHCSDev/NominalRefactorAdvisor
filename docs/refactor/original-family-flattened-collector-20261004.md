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

## Confirmation

The existing `Package.load` collector parsed all 18 skill Python modules at the
predecessor with zero omissions. The declaration registered 23 measures before
this change. All collector, repository, record and CLI consumers above were read.

After the change, all 20 skill Python modules parsed with zero omissions and the
original family registered 24 measures. The class source and BOUND-8 section
match the original archive exactly.

The real agent-comms commit `c4ceec4243ebd925a51ce80222ef8bb2e32a59c8`
introduced `reason=self.reason.declared_name` in `CompactionProgress.apply`.
The original full `pi_events.py` count rises from 4 to 5; the exact class fragment
rises from 0 to 1. The compact fixtures retain source revisions, line ranges and
full-file and fragment hashes. Both original Git files were read and their hashes
and full counts were checked. The affected audit batch passed all five tests in
0.478 seconds, including removal, member transport and formatted display text.

One normal no-isolation setuptools build produced a 23,532-byte file wheel.
All ten Python package members match their source. A fresh isolated interpreter
imported `refactor_audit.measures.FamilyFlattened` directly from that wheel,
registered it as `family_flattened`, and exposed its count through `Census.record`.
Setuptools 84.0.0 and wheel 0.48.0 were already installed; no environment or
application dependency was installed. Wheel SHA:
`8d489731728ebf9f7e0f90ef2b14e7e3b5d2f90ed82bd8b1482d735f82869886`.

The checkout's skill archive was rebuilt from its 35 tracked source members,
then each member was read back and compared. Archive SHA:
`9c9b4b24aef8d0fd07645d0dd9d3012c26bb7d18ca5447e9ce48e37cccefff96`.
The original archive outside this checkout retains its original SHA. The owned
build and temporary fixture directories were removed after completion; the
small file wheel remains in `.artifacts/family-flattened-collector-20261004/wheels`.

These checks qualify the collector and package seam. They do not claim full NRA
runtime, installed application, native process or provider acceptance. Core F4
still owns the complete site triage, explicit mechanism exemptions and required
ratchet integration; publishing this library does not complete those changes.
