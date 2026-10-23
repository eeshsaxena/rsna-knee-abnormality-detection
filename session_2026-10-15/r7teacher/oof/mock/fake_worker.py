"""Fake compact_infer.py for a logic test of the R7-OOF controller + merge cells (no model, no GPU, no DICOM).
Writes the same files as the real worker: predictions.csv (%.9g), fluid_stacker.csv / state_stacker.csv (%.17g), dump.json, receipt.json.
Per (study, fold model) logits are deterministic; the hash-fold model of a study gets a larger offset (mimics an out-of-fold model).
FAKE_SLOW_S (env) = seconds per study, to test the deadline."""
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
for k in ("--data", "--split", "--code", "--arm", "--assignment", "--output", "--amp", "--workers", "--deadline", "--models",
          "--dump-logits", "--backbone", "--size", "--crop", "--eval-stride", "--views", "--state-spec", "--side-model",
          "--side-threshold", "--side-position-mm", "--side-max-x", "--slices", "--phys-mm", "--trainer-md5", "--fluid-stacker",
          "--state-stacker", "--anchor-weights", "--tmp"):
    ap.add_argument(k, default=None)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--dump-state", action="store_true")
ap.add_argument("--flip-tta", action="store_true")
a = ap.parse_args()
out = Path(a.output)
uids = json.loads(Path(a.assignment).read_text())
files = sorted(Path(a.models).glob(a.arm + "_s*_f*.pt"))
folds = [int(p.name.split("_f")[1].split(".")[0]) for p in files]
print("MODEL_READY", json.dumps({"models": len(files)}), flush=True)


def rng(*s):
    return np.random.default_rng(int(hashlib.md5("|".join(map(str, s)).encode()).hexdigest()[:8], 16))


fh = open(out / "predictions.csv", "w", newline="")
wr = csv.writer(fh)
wr.writerow(["StudyInstanceUID"] + LABELS)
ffh = open(out / "fluid_stacker.csv", "w", newline="")
fwr = csv.writer(ffh)
fwr.writerow(["StudyInstanceUID", "status", "n_sag_fluid_series", "series", "median", "n", "spacing", "eff_sag_p99_open", "bak_sag_p99_open",
              "fluid_s", "Effusion|z_feature", "Effusion|z_out", "Effusion|p_model", "Effusion|p_out"])
sfh = open(out / "state_stacker.csv", "w", newline="")
swr = csv.writer(sfh)
swr.writerow(["StudyInstanceUID", "Baker's|z_feature", "Baker's|z_out", "Baker's|p_model", "Baker's|p_out"])
dump, fallback, per_study = {}, {}, []
t0 = time.time()
pos = 0
for u in uids:
    if u.endswith("0"):                       # a deterministic "decode failure" for ~10 % of studies
        fallback[u] = "loader: fake failure"
        pos += 1
        continue
    time.sleep(float(os.environ.get("FAKE_SLOW_S", "0")))
    base = rng(u).normal(0, 2, 12)
    hf = int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4
    zs = np.stack([base + (rng(u, f).normal(0, 1.5, 12) if f == hf else rng(u, f).normal(0, 0.2, 12)) for f in folds]).astype(np.float32)
    zmean = zs.mean(0)
    p = (1 / (1 + np.exp(-zmean.astype(np.float32)))).astype(np.float32).astype(np.float64)
    ze = float(-0.5 + 0.3 * zmean[7])
    pe = 1 / (1 + math.exp(-ze))
    zb = float(0.1 + 0.9 * zmean[9])
    pb = 1 / (1 + math.exp(-zb))
    fwr.writerow([u, "ok", 1, "s", 1, 1, 1, 12.5, 3.25, 0.1, "%.17g" % zmean[7], "%.17g" % ze, "%.17g" % p[7], "%.17g" % pe])
    swr.writerow([u, "%.17g" % zmean[9], "%.17g" % zb, "%.17g" % p[9], "%.17g" % pb])
    p[7], p[9] = pe, pb
    wr.writerow([u] + ["%.9g" % x for x in p])
    fh.flush()
    rec = {"z": zs.tolist()}
    if a.dump_state:
        rec["state_plain"] = [rng(u, f, "s").random(84).astype(np.float32).tolist() for f in folds]
    dump[u] = rec
    per_study.append([u, 500, 30.0, 1.5, 2.0])
    pos += 1
    if a.deadline and time.time() > float(a.deadline):
        break
for u in uids[pos:]:
    fallback[u] = "not reached before the deadline"
fh.close(); ffh.close(); sfh.close()
if a.dump_logits:
    Path(a.dump_logits).write_text(json.dumps(dump))
n = len(dump)
rec = {"status": "PASS", "models": [p.name for p in files], "studies": len(uids), "scored": n, "fallback": fallback,
       "timed_out": pos < len(uids), "seconds": round(time.time() - t0, 1), "amp": "fp16", "torch": "fake", "timm": "fake",
       "compile": {"active": True, "check_seconds": 1.0, "fallback": None},
       "stats": {"decode_s": 30.0 * n, "render_s": 1.5 * n, "gpu_s": 2.0 * n, "windows": 500 * n, "wait_s": 0.0, "rows": n, "anchor_s": 1.0 * n},
       "per_study": per_study[:50]}
(out / "receipt.json").write_text(json.dumps(rec, indent=1))
print("WORKER_DONE", json.dumps({"studies": len(uids), "scored": n}), flush=True)
