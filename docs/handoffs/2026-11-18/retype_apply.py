"""Apply every type-only retype the cut planner offers on one tree, to a fixpoint (measurement)."""
import collections, sys, time
from pathlib import Path
from nominal_refactor_advisor.ast_tools import PythonSourcePathPolicy, parse_python_modules
from nominal_refactor_advisor.codemod import CodemodPlanSequence, CodemodSourceSnapshot
from nominal_refactor_advisor.refactors.hidden_import_cuts import HiddenImportCuts, HubRetype
from nominal_refactor_advisor.refactors.type_only_readers import NoLowerBase, RetypeTypeOnlyReadsOperation

root = Path(sys.argv[1])


def snapshot():
    return CodemodSourceSnapshot.from_modules(
        parse_python_modules(root, use_parse_cache=False, source_policy=PythonSourcePathPolicy(include_tests=True))
    )


def key(read):
    return (read.reader, read.name)


def attempt(state, reads):
    """Apply the retype of `reads` if its proof holds; else the refusal's text."""
    operation = RetypeTypeOnlyReadsOperation.of(tuple(reads))
    try:
        simulation = CodemodPlanSequence.from_operations((operation,)).simulate(state)
    except ValueError as error:
        return str(error)
    if not simulation.is_clean:
        return "simulation not clean"
    simulation.apply()
    return None


# One round per process: apply the first retype whose proof holds, then exit 0;
# exit 3 when none does, after printing every refusal. A shell loop runs rounds.
state = snapshot()
plans = HiddenImportCuts.of(state).hubs()
if len(sys.argv) > 2 and sys.argv[2] == "--census":
    no_base = [need for plan in plans for _r, need in plan.needs if isinstance(need, NoLowerBase)]
    offered = [read for plan in plans if isinstance(plan, HubRetype) for read in plan.reads]
    print(f"{root.name}: {len(offered)} type-only reads offered over "
          f"{sum(isinstance(p, HubRetype) for p in plans)} hubs; {len(no_base)} need a declared contract", flush=True)
    for need in no_base:
        print("   needs contract:", str(need)[:220], flush=True)
    sys.exit(0)
# Keys a previous round refused on an earlier state are skipped while SKIP names
# a file; the shell runs one final round without it, so nothing is left untried.
import os
skip_file = os.environ.get("SKIP")
skipped = set(Path(skip_file).read_text().splitlines()) if skip_file and Path(skip_file).exists() else set()
refused = {}
for plan in plans:
    if not isinstance(plan, HubRetype):
        continue
    batches = [list(plan.reads)] + ([[read] for read in plan.reads] if len(plan.reads) > 1 else [])
    for reads in batches:
        if all(f"{r.reader}:{r.name}" in skipped for r in reads):
            continue
        why = attempt(state, reads)
        if why is None:
            for read in reads:
                print(f"  OK  {read.reader}: {read.name} -> {read.base_module}.{read.base_name}", flush=True)
            sys.exit(0)
        if len(reads) == 1:
            refused[key(reads[0])] = why
            if skip_file:
                with open(skip_file, "a") as out:
                    out.write(f"{reads[0].reader}:{reads[0].name}\n")
for (reader, name), why in sorted(refused.items()):
    print(f"  NO  {reader}:{name}: {why[:300]}", flush=True)
sys.exit(3)
