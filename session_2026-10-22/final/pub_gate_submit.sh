#!/bin/bash
# pub_gate_submit.sh (10-22 ~16:15 IST, Claude): PREREG Amendment 23 public entries. For each kernel: wait for its commit (wait_fetch,
# up to 80 min), retry the fetch on 502 pages, run its commit gate, and submit version 1 ONLY on the exact PASS line. Logs to pub_gate_submit.log.
R=/c/Users/eeshs/Downloads/rsna-knee; F=$R/session_2026-10-22/final; K=$R/session_2026-10-21/levers/kernels; LOG=$F/pub_gate_submit.log
C4T='{"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker'"'"'s": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}'
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gate_moacf() { py -3.12 -I $K/moacf_commit_gate.py "$1" --a224 0.25 --table "$C4T" --aoa 0.25 --mode pw --aloamm 0.05 --amclmm 0.1 --poa 1 --amoacf 0.05 2>&1 | tail -1; }
gate_cons()  { py -3.12 -I $K/sink_commit_gate.py "$1" --a224 0.25 --table "$C4T" --mode pw --aoa 0.25 --aloamm 0.05 --amclmm 0.1 --poa 2 2>&1 | tail -1; }
one() {
  key=$1; slug=$2; pass=$3; desc=$4; gfn=$5; out=$F/out_$key
  for attempt in 1 2 3; do
    rm -rf "$out"; py -3.12 $R/session_2026-10-15/v224f/wait_fetch.py "$slug" "$out" 80 >> "$LOG" 2>&1
    bad=$(grep -l "Server Error" "$out"/* 2>/dev/null | wc -l)
    if [ -f "$out/submission.csv" ] && [ "$bad" = "0" ]; then break; fi
    say "$key fetch attempt $attempt incomplete (bad $bad) -> retry"; sleep 60
  done
  gate=$($gfn "$out"); say "$key gate: $gate"
  if [ "$gate" = "PROBE_CHECK $pass" ]; then
    r=$(py -3.12 -m kaggle competitions submit rsna-knee-abnormality-detection -k "$slug" -v 1 -f submission.csv -m "$desc" 2>&1 | tail -1)
    say "$key SUBMITTED: $r"
    echo "[$(date '+%m-%d %H:%M:%S') IST] A23 $key SUBMITTED $slug v1: $r" >> $R/session_2026-10-14/submits.log
  else
    say "GATE_FAIL_$key -> NOT submitted"
  fi
}
one moacf eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10moacf05j "PASS 29/29" "PUB22-MOACF: swide 57464326 + Medial OA <- Contusion, Fracture a .05 (public push; never a final pick)" gate_moacf &
one cons eeshsaxena/rsna-sub-v224r6c4pw25lomm05mcl10cvx2j "PASS 27/27" "PUB22-CONS: swide 57464326 with convex E-OA p 2 (public push; never a final pick)" gate_cons &
wait
say "PUB_DONE"
