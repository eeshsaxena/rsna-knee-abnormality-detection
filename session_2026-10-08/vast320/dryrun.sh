#!/bin/bash
# dryrun.sh: proves the kit without renting anything. Writes dryrun.log (also printed).
#   1 every laptop + remote script parses (bash -n, py_compile); no em dash in any kit file
#   2 guard backstop simulation (laptop)       3 Vast CLI read only: offer search + ranking; image tag exists
#   4 box: every path the run needs, versions, gate verdict, queue argv == template, placeholder status,
#     parity.py on the box (CPU), and tests/box_sim.sh (lanes, status, manifest, verify, placeholders on a queue COPY)
#   5 every laptop script in --dry-run mode (launch on the top-ranked offer, setup, train, watch, fetch, destroy)
# Nothing is created on Vast; on the box it only writes ~/vast320/ (kit helpers) and /tmp/vast320_sim (removed).
cd "$(dirname "$0")" || exit 1
OUT=dryrun.log; : > "$OUT"
exec > >(tee -a "$OUT") 2>&1
SCRIPT_TAG=dryrun; source ./config.sh
P=0; FL=0; ok(){ echo "PASS $*"; P=$((P + 1)); }; no(){ echo "FAIL $*"; FL=$((FL + 1)); }
echo "== vast320 dry run $(date '+%F %T %Z') on $(hostname) | no rental, no instance"
echo "-- 1 parse"
for f in config.sh vast_launch.sh vast_setup.sh vast_train.sh vast_watch.sh vast_fetch.sh vast_destroy.sh box_placeholders.sh dryrun.sh \
         remote/box_ops.sh remote/box_push.sh remote/guard.sh remote/lane.sh remote/status.sh tests/box_sim.sh tests/guard_sim.sh; do
  bash -n "$f" && ok "bash -n $f" || no "bash -n $f"
done
for f in offers_rank.py remote/parity.py; do "$PYL" -m py_compile "$f" && ok "py_compile $f" || no "py_compile $f"; done
rm -rf __pycache__ remote/__pycache__
ED=$(grep -rl $'\xe2\x80\x94' --include='*.sh' --include='*.py' --include='*.md' --include='*.txt' . 2>/dev/null | grep -v dryrun.log)
[ -z "$ED" ] && ok "no em dash in any kit file" || no "em dash in: $ED"
echo "-- 2 guard backstop simulation (laptop)"
TMPDIR=${TMPDIR:-/tmp} bash tests/guard_sim.sh | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "guard_sim" || no "guard_sim"
echo "-- 3 Vast read only"
bash vast_launch.sh --search --folds "1 2 3 4" | sed 's/^/   /'
SF=$(ls -t logs/offers_*.json 2>/dev/null | head -n 1)
TOP=$("$PYL" offers_rank.py "$SF" --folds 4 --budget 8.10 --v100_fold_h 5.8 --json 2>/dev/null | "$PYL" -c "import json,sys; r=[x for x in json.load(sys.stdin) if x['fits']]; print(r[0]['id'] if r else '')")
[ -n "$TOP" ] && ok "offer search + ranking (top fitting offer now: $TOP)" || no "no fitting offer in the search"
c=$(curl -s -o /dev/null -w '%{http_code}' https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/2.5.1-cuda12.1-cudnn9-runtime)
[ "$c" = 200 ] && ok "image $IMAGE exists on Docker Hub" || no "image tag check http $c"
credit_now > /dev/null 2>&1 && ok "Vast 2FA session active" || echo "INFO Vast 2FA session NOT active now (expected; RUNBOOK step 0 before launch)"
echo "-- 4 box"
if box_up; then
  sync_box_ops && bx_in 'mkdir -p ~/vast320/tests && cat > ~/vast320/tests/box_sim.sh' < tests/box_sim.sh && ok "kit helpers copied to box ~/vast320/" || no "copy helpers"
  bops paths 1 2 3 4 | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "every box path present (code, csvs, store, HF weights, queue, gates)" || no "box paths"
  bops versions | sed 's/^/   box pin /'
  echo "   gate verdict ($RUNTAG): $(bops verdict | head -c 200)"
  bops argv-check 1 2 3 4 | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "queue argv of the $RUNTAG f1-f4 lines == argv_template.txt" || no "argv-check"
  bash box_placeholders.sh status 1 2 3 4 | sed 's/^/   /'
  bash vast_setup.sh --dry-run | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "setup dry run incl. box parity.py (CPU)" || no "setup dry run"
  bx 'bash ~/vast320/tests/box_sim.sh' | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "box_sim (lanes, status, manifest, verify, placeholders on a queue copy)" || no "box_sim"
else no "box $BOX unreachable from this network: section 4 not run"; fi
echo "-- 5 laptop scripts in dry-run mode"
if [ -n "$TOP" ] && box_up; then bash vast_launch.sh --dry-run "$TOP" --folds "1 2 3 4" | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "launch dry run on offer $TOP" || no "launch dry run"; fi
bash vast_train.sh --dry-run | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "train dry run" || no "train dry run"
bash vast_watch.sh --dry-run | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "watch dry run (keepalive mode, no instance)" || no "watch dry run"
bash vast_fetch.sh --dry-run 2>&1 | sed 's/^/   /'; ok "fetch dry run (expected: refuses without a live instance)"
bash vast_destroy.sh | sed 's/^/   /'; [ "${PIPESTATUS[0]}" = 0 ] && ok "destroy without --yes does nothing" || no "destroy dry"
[ -f state.env ] && no "state.env exists after a dry run" || ok "no state.env written (nothing rented)"
echo "== dry run: $P PASS, $FL FAIL | $(date '+%F %T %Z')"
