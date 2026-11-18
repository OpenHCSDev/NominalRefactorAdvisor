"""Which concepts' reproofs fall back to a full projected_scan, and the detector ids that send them there."""
import sys
from pathlib import Path
from nominal_refactor_advisor.codemod_workflow import CodemodWorkflowScan
from nominal_refactor_advisor.detectors._base import DetectorConfig, IssueDetector
from nominal_refactor_advisor.refactors.finding_recipes import FindingRecipeSynthesizer
from nominal_refactor_advisor.refactor_concepts import RefactorConcept
root = Path(sys.argv[1]).resolve(); config = DetectorConfig()
scan = CodemodWorkflowScan.parsed((root,), config, cache_dir=None, use_parse_cache=True)
registered = {d.effective_detector_id() for d in IssueDetector.registered_detector_types()}
mirrors = IssueDetector.semantic_mirror_detector_ids()
for concept in RefactorConcept.declaration_types():
    targets = FindingRecipeSynthesizer.findings_for_concept(scan.findings, concept, scan.source_snapshot)
    if not targets or len(targets) > 100:
        continue
    ids = concept.stage_proof.reproof_detector_ids(frozenset((*(f.detector_id for f in targets), *FindingRecipeSynthesizer.detector_ids_for_concept(concept))), mirrors)
    missing = sorted(ids - registered)
    print(f"{concept.concept_key()}: targets={len(targets)} reproof ids={len(ids)} {'FALLBACK missing ' + str(missing) if missing else 'reproof'} stage_proof={concept.stage_proof.__name__}", flush=True)
