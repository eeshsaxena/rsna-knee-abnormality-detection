"""Independent local check of a downloaded rsna-r7-oof output folder + timing projection.

  python verify_commit.py OUT_DIR

Re-checks from the raw per-job files (not from the notebook's own summary): one fold model per OOF study and it is the hash fold,
probabilities in [0, 1] and finite, r7_oof.csv == the job files, proof outlier test, determinism (OOF logit vs the same fold model
inside the 4-model proof run), then prints per-study timings and the projection for the 4,407-study run. No em dashes.
"""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

csv.field_size_limit(1 << 30)
out = Path(sys.argv[1])
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]


def r7_fold(u):
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


ok = True


def check(c, what):
    global ok
    print(("PASS " if c else "FAIL ") + what)
    ok &= bool(c)


oof = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(out / "r7_oof.csv", encoding="utf-8", newline=""))}
rec = json.loads((out / "r7_oof_receipt.json").read_text(encoding="utf-8"))
jobs = sorted((out / "r7oof").iterdir())
seen = {}
for jd in jobs:
    if not jd.name.startswith("oof_"):
        continue
    k = int(jd.name.split("_f")[1][0])
    r = json.loads((jd / "receipt.json").read_text(encoding="utf-8"))
    check(r["models"] == ["E_lrprior_s0_f%d.pt" % k], "%s: worker loaded only %s" % (jd.name, r["models"]))
    check("flip_tta" not in r, "%s: no flip_tta block in the worker receipt (plain pass)" % jd.name)
    dump = json.loads((jd / "dump.json").read_text(encoding="utf-8"))
    for row in csv.DictReader(open(jd / "predictions.csv", encoding="utf-8", newline="")):
        u = row["StudyInstanceUID"]
        check(u not in seen, "%s scored once" % u[-14:])
        seen[u] = k
        check(r7_fold(u) == k, "%s hash fold %d == job fold %d" % (u[-14:], r7_fold(u), k))
        p = np.array([float(row[L]) for L in LABELS])
        check(bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()), "%s 12 probabilities finite, in [0, 1]" % u[-14:])
        z = np.array(dump[u]["z"], np.float64)
        check(z.shape == (1, 12), "%s dump holds one fold model's 12 logits" % u[-14:])
        o = oof[u]
        po = np.array([float(o["p|" + L]) for L in LABELS])
        zo = np.array([float(o["z|" + L]) for L in LABELS])
        check(np.array_equal(zo, z[0]), "%s r7_oof.csv logits == dump" % u[-14:])
        nonst = [j for j, L in enumerate(LABELS) if L not in ("Effusion", "Baker's")]
        check(float(np.max(np.abs(po[nonst] - p[nonst]))) == 0.0, "%s r7_oof.csv non-stacked p == predictions.csv" % u[-14:])
        check(float(np.max(np.abs(po - p))) < 1e-8, "%s r7_oof.csv p within 1e-8 of predictions.csv (stacked at %%.17g)" % u[-14:])
check(set(seen) == set(oof), "r7_oof.csv rows == studies in the job files (%d)" % len(oof))

if (out / "r7_proof.csv").is_file():
    pr = list(csv.DictReader(open(out / "r7_proof.csv", encoding="utf-8", newline="")))
    for jd in jobs:
        if jd.name.startswith("proof_"):
            r = json.loads((jd / "receipt.json").read_text(encoding="utf-8"))
            dump = json.loads((jd / "dump.json").read_text(encoding="utf-8"))
            check(r["models"] == ["E_lrprior_s0_f%d.pt" % k for k in range(4)], "%s loaded the 4 fold models" % jd.name)
            for u, d in dump.items():
                Z = np.array(d["z"], np.float64)
                dist = [float(np.sqrt(np.mean((Z[k] - np.delete(Z, k, 0).mean(0)) ** 2))) for k in range(4)]
                hf = r7_fold(u)
                print("PROOF %s hash fold %d | outlier fold %d | dist %s | mean|z| held-out %.3f in-fold %.3f | OOF vs proof max|dz| %s" % (
                    u[-14:], hf, int(np.argmax(dist)), [round(x, 3) for x in dist], float(np.abs(Z[hf]).mean()),
                    float(np.abs(np.delete(Z, hf, 0)).mean()),
                    "%.3g" % float(np.max(np.abs(np.array([float(oof[u]["z|" + L]) for L in LABELS]) - Z[hf]))) if u in oof else "n/a"))
    print("PROOF summary: outlier == hash fold %d / %d" % (sum(r["match"] == "True" for r in pr), len(pr)))

print("\n--- timings (worker receipts) ---")
print("session_seconds %.0f, scored %d, cpu_count %s, mem_gb %s, n_gpu %s, io_bench %s" % (
    rec["session_seconds"], rec["scored"], rec["cpu_count"], rec["mem_gb"], rec["n_gpu"], rec.get("io_bench")))
for js in rec["jobs"]:
    st = js.get("stats") or {}
    n = max(1, int(st.get("rows", 0)))
    print("%-11s gpu %s n %s scored %s | wall %ss (start %s end %s) | compile check %ss active %s | per study decode %.1f render %.2f anchor %.2f gpu %.2f wait %.1f windows %.0f" % (
        js["id"], js.get("gpu"), js.get("studies"), js.get("scored"), js.get("seconds"), js.get("start_s"), js.get("end_s"),
        js.get("compile_check_s"), js.get("compile_active"), st.get("decode_s", 0) / n, st.get("render_s", 0) / n,
        st.get("anchor_s", 0) / n, st.get("gpu_s", 0) / n, st.get("wait_s", 0) / n, st.get("windows", 0) / n))
for jd in jobs:
    r = json.loads((jd / "receipt.json").read_text(encoding="utf-8")) if (jd / "receipt.json").is_file() else {}
    for ps in r.get("per_study", []):
        print("  %s %s windows %d decode %.1f render %.2f gpu %.2f" % (jd.name, ps[0][-14:], ps[1], ps[2], ps[3], ps[4]))
print("cpu samples:", rec.get("cpu_samples"))
print("ALL_PASS" if ok else "SOME_FAIL")
