"""Bench: fine-grained vs fresh on the saved move, with the update's messages and any exception shown."""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fg_cursor import Cursor
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._restated_facts import ForeignAbsenceProbeDetector, ProbeOwnerQuery
from nominal_refactor_advisor.probe_owner import ProbeOwnerContext
from nominal_refactor_advisor.type_checker_session import SourceTexts
root = Path(sys.argv[1]).resolve(); bench = json.loads(Path(sys.argv[2]).read_text())
config = DetectorConfig()
scan = CodemodWorkflowScan.parsed((root,), config, cache_dir=None, use_parse_cache=True)
cursor = Cursor(tuple(scan.modules)); print(f"fine-grained base build {cursor.build_seconds:.1f}s", flush=True)
after = scan.source_snapshot.with_virtual_sources(bench["sources"]).module_set
detector = ForeignAbsenceProbeDetector()
try:
    groups = type(detector).compact_module_projection_groups(after)
    query = ProbeOwnerQuery.of(groups)
    t = time.perf_counter(); fresh = query.answered(SourceTexts(tuple(m for m in after if m.file_path in query.file_paths))); fresh_s = time.perf_counter() - t
    t = time.perf_counter(); changed, removed, update_s = cursor.move_to(tuple(after)); answers = cursor.answers(query.spans); fine_s = time.perf_counter() - t
    print(f"changed={changed} removed={removed} update={update_s:.2f}s fine total={fine_s:.2f}s fresh={fresh_s:.2f}s", flush=True)
    messages = cursor.fgm.previous_messages
    print(f"update messages: {len(messages)}; blocking={cursor.fgm.blocking_error}", flush=True)
    for m in messages[:5]: print("   ", m, flush=True)
    fine = ProbeOwnerContext(query.classes, {s: a.owner(query.classes) for s, a in answers.items()}, query.typing)
    diffs = [s for s in query.spans if fresh.typed.get(s) != fine.typed.get(s)]
    print(f"spans={len(query.spans)} differing={len(diffs)}", flush=True)
    for s in diffs[:5]: print("   ", s, "fresh:", fresh.typed.get(s), "fine:", fine.typed.get(s), flush=True)
    ff = sorted(f.stable_id for f in detector._findings_from_compact_projection_groups_context(groups, fresh, config))
    gf = sorted(f.stable_id for f in detector._findings_from_compact_projection_groups_context(groups, fine, config))
    print(f"findings fresh={len(ff)} fine={len(gf)} identical={ff == gf}", flush=True)
except BaseException:
    traceback.print_exc()
