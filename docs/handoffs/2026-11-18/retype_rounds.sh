#!/bin/bash
# Type-only retype rounds on one agent-comms tree, one process per round (fork M's driver).
# A round applies the first retype whose mypy proof holds (exit 0) or prints every refusal (exit 3).
# Rounds skip keys refused on an earlier state; a final round retries everything; stop when it applies nothing.
#
# usage: retype_rounds.sh NRA_CHECKOUT AGENT_COMMS_TREE OUT_DIR [PYTHON]
#   AGENT_COMMS_TREE: a fresh `git worktree add --detach X origin/main`, made right after `git fetch`.
#   Heavy: holds OUT_DIR/nra_scan.lock per round so it never runs beside another scan.
set -u
NRA=$1; TREE=$2; OUT=$3; PY=${4:-$NRA/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; log=$OUT/retype_rounds.out; skip=$OUT/retype_skip.txt; round_out=$OUT/round.out
: > "$log"; : > "$skip"
round=0; mode=skip
while [ $round -lt 80 ]; do
  round=$((round+1))
  env_skip=""; [ $mode = skip ] && env_skip="SKIP=$skip"
  flock "$OUT/nra_scan.lock" bash -c "ulimit -v 9000000; cd '$NRA' && env $env_skip PYTHONPATH='$NRA' timeout -s INT 3000 '$PY' '$HERE/retype_apply.py' '$TREE' > '$round_out' 2>&1; echo rc=\$? >> '$round_out'"
  echo "== round $round ($mode) $(date +%T)" >> "$log"; cat "$round_out" >> "$log"
  if grep -q '^rc=0' "$round_out"; then mode=skip
  elif grep -q '^rc=3' "$round_out"; then [ $mode = full ] && break; mode=full
  else break; fi
done
echo "rounds done $(date +%T)" >> "$log"
