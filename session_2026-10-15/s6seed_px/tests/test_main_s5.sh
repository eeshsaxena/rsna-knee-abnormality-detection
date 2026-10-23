#!/bin/bash
# Stub tests of main_s5_unzip_px.sh (10-15 review S6R-2): a FAKE HOME with stub unzip_parts.py / px_build_s6.py / ct_s6v100.py,
# 4,407 fake studies (58 gold). Never touches ~/rsna_comp, ~/s6seed/px320, the GPU or cores 12-27 (a taskset shim in PATH maps
# the script's 'taskset -c 12-27' to 28-39). Usage: test_main_s5.sh [MAIN_S5]  ->  MAIN_S5_TESTS PASS n/n
set -u
S5="${1:-$HOME/s6seed/main_s5_unzip_px.sh}"
REALPY=$HOME/miniconda3/envs/kmimic/bin/python
T="$(mktemp -d /tmp/s6s5.XXXXXX)"
PASS=0; N=0
ok() { N=$((N+1)); if [ "$1" = "1" ]; then PASS=$((PASS+1)); echo "PASS $2"; else echo "FAIL $2"; fi; }
mkdir -p "$T/bin"
cat > "$T/bin/taskset" <<'EOF'
#!/bin/bash
[ "$1" = "-c" ] && shift 2
exec /usr/bin/taskset -c 28-39 "$@"
EOF
chmod +x "$T/bin/taskset"
mkhome() {   # $1 = case dir -> fake HOME
  local H="$1"; mkdir -p "$H/s6seed/logs" "$H/miniconda3/envs/kmimic/bin" "$H/miniconda3/envs/rsna/bin" "$H/rsna_comp"
  ln -sf "$REALPY" "$H/miniconda3/envs/kmimic/bin/python"; ln -sf "$REALPY" "$H/miniconda3/envs/rsna/bin/python"
  "$REALPY" - "$H/s6seed/targets_s6seed.csv" <<'EOF'
import sys
uids = ["1.2.3.%04d" % i for i in range(4407)]
open(sys.argv[1], "w").write("StudyInstanceUID,is_gold\n" + "".join("%s,%d\n" % (u, int(i < 58)) for i, u in enumerate(uids)))
EOF
  cat > "$H/s6seed/unzip_parts.py" <<'EOF'
import os, sys
a = sys.argv; out = a[a.index("--out") + 1]
if "--list-only" in a:
    print("list-only: train.csv train_series.csv train_series/"); sys.exit(0)
uids = ["1.2.3.%04d" % i for i in range(4407)]
os.makedirs(os.path.join(out, "train_series"), exist_ok=True)
open(os.path.join(out, "train.csv"), "w").write("StudyInstanceUID\n" + "".join(u + "\n" for u in uids))
open(os.path.join(out, "train_series.csv"), "w").write("x\n1\n")
for u in uids:
    os.makedirs(os.path.join(out, "train_series", u), exist_ok=True)
print("UNZIP OK 4407 studies")
EOF
  cat > "$H/s6seed/px_build_s6.py" <<'EOF'
import json, os, sys, glob
a = sys.argv; out = a[a.index("--out") + 1]; os.makedirs(out, exist_ok=True)
mode = os.environ.get("STUB_MODE", "ok")
if "--finalize" in a:
    done, err = {}, {}
    for p in sorted(glob.glob(out + "/part*.json")):
        s = json.load(open(p)); done.update(s["done"]); err.update({u: e for u, e in s["errors"].items()})
    err = {u: e for u, e in err.items() if u not in done}
    json.dump({"done": done, "errors": err, "order": sorted(done)}, open(out + "/index.json", "w"))
    open(out + "/FINALIZED_MARK", "w").write("x"); print("FINALIZED done %d errors %d" % (len(done), len(err))); sys.exit(0)
data = a[a.index("--data") + 1]
uids = [l.strip() for l in open(os.path.join(data, "train.csv")).read().splitlines()[1:] if l.strip()]
jp = out + "/part000.json"; st = json.load(open(jp)) if os.path.exists(jp) else {"done": {}, "errors": {}}
npass = st.get("passes", 0) + 1; st["passes"] = npass
for i, u in enumerate(uids):
    if u in st["done"]:
        continue
    if mode == "rc3" and i >= 2200: break
    if mode == "rc4" and i >= 2900: break
    if mode == "missing" and i >= 3407: break
    if mode == "errors_then_ok" and npass == 1 and 100 <= i < 112: st["errors"][u] = "TimeoutError"; continue
    if mode == "errors_persist" and 100 <= i < 140: st["errors"][u] = "ValueError no rows"; continue
    if mode == "gold_missing" and i == 3: st["errors"][u] = "ValueError no rows"; continue
    st["done"][u] = [i, 600, 3]; st["errors"].pop(u, None)
json.dump(st, open(jp, "w"))
print("stub build pass %d mode %s done %d errors %d" % (npass, mode, len(st["done"]), len([u for u in st["errors"] if u not in st["done"]])))
sys.exit({"rc3": 3, "rc4": 4}.get(mode, 0))
EOF
  cat > "$H/s6seed/ct_s6v100.py" <<'EOF'
import csv, json, sys
a = sys.argv; px = a[a.index("--px") + 1]; tg = a[a.index("--targets") + 1]
idx = json.load(open(px + "/index.json")); done = {u for u, v in idx["done"].items() if v[1] > 0}
rows = list(csv.DictReader(open(tg)))
pool = [r for r in rows if r["is_gold"] == "0" and r["StudyInstanceUID"] in done]
gold = [r for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]
print("pool %d gold %d folds [0, 0, 0, 0, 0] | stub" % (len(pool), len(gold))); print("check-targets OK")
EOF
}
run() {   # $1 case, $2 mode
  local H="$T/$1"; mkhome "$H"
  HOME="$H" PATH="$T/bin:$PATH" STUB_MODE="$2" PARTS="$H/rsna_comp/part.0" DATA="$H/rsna_comp/data" bash "$S5" > "$H/s5.log" 2>&1
  RC=$?; FIN=0; [ -e "$H/s6seed/px320/FINALIZED_MARK" ] && FIN=1
  VER="$(cat "$H/s6seed/gates/PX_BUILD.verdict" 2>/dev/null)"
}
run m1 rc3;            ok "$([ $RC = 3 ] && [ $FIN = 0 ] && echo "$VER" | grep -q '^FAIL.*exit 3' && echo 1 || echo 0)" "M1 MEM_CAP stop (exit 3): no finalize, exit 3, verdict FAIL (rc $RC, finalized $FIN)"
run m2 rc4;            ok "$([ $RC = 4 ] && [ $FIN = 0 ] && echo "$VER" | grep -q '^FAIL.*exit 4' && echo 1 || echo 0)" "M2 dead worker (exit 4): no finalize, exit 4, verdict FAIL (rc $RC, finalized $FIN)"
run m3 missing;        ok "$([ $RC != 0 ] && echo "$VER" | grep -q '^FAIL.*1000 studies neither done nor errored' && echo 1 || echo 0)" "M3 rc 0 but 1,000 studies missing: verdict FAIL (rc $RC)"
run m4 errors_then_ok; ok "$([ $RC = 0 ] && echo "$VER" | grep -q '^PASS.* pool 4349 gold 58 errors 0' && grep -q 'retry of 12' "$T/m4/s5.log" && echo 1 || echo 0)" "M4 12 errors -> one retry pass -> PASS pool 4349 (rc $RC)"
run m5 errors_persist; ok "$([ $RC != 0 ] && echo "$VER" | grep -q '^FAIL.*pool 4309 < floor 4328' && echo 1 || echo 0)" "M5 40 persistent errors: pool 4309 < 4328 -> FAIL (rc $RC)"
run m6 gold_missing;   ok "$([ $RC != 0 ] && echo "$VER" | grep -q '^FAIL.*gold 57 != 58' && echo 1 || echo 0)" "M6 a gold study errors: gold 57 -> FAIL (rc $RC)"
run m7 ok;             ok "$([ $RC = 0 ] && [ $FIN = 1 ] && echo "$VER" | grep -q '^PASS.* pool 4349 gold 58 errors 0 missing 0' && echo 1 || echo 0)" "M7 clean build -> PASS pool 4349 (rc $RC)"
echo "MAIN_S5_TESTS $([ $PASS = $N ] && echo PASS || echo FAIL) $PASS/$N  (script $(sha256sum "$S5" | cut -c1-12), logs in $T)"
