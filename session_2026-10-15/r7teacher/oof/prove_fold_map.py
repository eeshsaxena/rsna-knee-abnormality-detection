"""Step 1 of the R7-OOF teacher job: prove how heliosli's R7 trainer (compact_train.py v9.8) assigns RSNA train studies
to its 4 folds, and write the held-out fold of every one of the 4,407 train studies.

Reads only local copies: src/compact_train.py (extracted from the R7 notebook SOURCES dict), src/r7_manifest.json (the R7
bundle manifest), src/fluid_stacker.json + src/state_stacker.json (bundle files), _data/train.csv + train_series.csv.
Writes r7_fold_map.csv and prints the evidence.  No em dashes.
"""
import csv, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = HERE / "src"
csv.field_size_limit(1 << 30)

ok_all = True


def check(cond, what):
    global ok_all
    print(("PASS " if cond else "FAIL ") + what)
    ok_all &= bool(cond)


man = json.loads((SRC / "r7_manifest.json").read_text(encoding="utf-8"))
rc = man["run_config"]
tr_bytes = (SRC / "compact_train.py").read_bytes()
tr_text = tr_bytes.decode("utf-8")

# (a) the trainer we read is the trainer the bundle pins
check(hashlib.sha256(tr_bytes).hexdigest() == man["src_sha256"]["compact_train.py"], "compact_train.py sha256 == manifest src_sha256 (%s)" % man["src_sha256"]["compact_train.py"][:12])
check(hashlib.md5(tr_bytes).hexdigest() == man["trainer_md5"], "compact_train.py md5 == manifest trainer_md5 (%s)" % man["trainer_md5"])

# (b) the fold rule and the train / holdout split, verbatim lines of the pinned trainer
LINES = {
    "pool": 'pool = sorted(r["StudyInstanceUID"] for r in rows if r["is_gold"] == "0" and r["StudyInstanceUID"] in done',
    "gold": 'gold = [r["StudyInstanceUID"] for r in rows if r["is_gold"] == "1" and r["StudyInstanceUID"] in done]',
    "disjoint": "assert not (set(pool) & set(gold))",
    "K": "K = a.folds",
    "fold_rule": "pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])",
    "train_set": "tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)",
    "holdout_set": "ho = np.flatnonzero(pool_fold == k)",
    "model_name": '"%s_s%d_f%d.pt" % (arm, seed, k)',
}
for key, line in LINES.items():
    n = tr_text.count(line)
    check(n >= 1, "trainer contains %-11s x%d: %s" % (key, n, line[:110]))
check(tr_text.count("pool_fold = ") == 1, "pool_fold is assigned exactly once in the trainer (no second fold rule)")

# (c) the run that produced the bundle
check(rc["folds"] == 4, "run_config folds == 4")
check(rc["holdout_fold"] is None, "run_config holdout_fold is None (fold k model trains on pool_fold != k only)")
check(rc["seeds"] == 1 and rc["seed_base"] == 0, "run_config seeds 1, seed_base 0 (files E_lrprior_s0_f*.pt)")
check(man["models"] == ["E_lrprior_s0_f%d.pt" % k for k in range(4)], "manifest models == E_lrprior_s0_f0..f3.pt (fold index k in the file name)")
check(rc["include_review_holdout"] is True, "include_review_holdout true: review-holdout rows are in the pool, folded by the same hash")
check(rc["warm_start"] is None and rc["reeval_run"] is None and rc["px_extra"] is None, "no warm start, no re-eval run, no extra pixel cache")
check(man["gold_in_training"] == 0, "manifest gold_in_training == 0")
check(rc["pool_studies"] == 4344 and rc["gold_studies"] == 58 and rc["feature_studies"] == 4403,
      "run_config pool 4344, gold 58, feature (px) studies 4403")
fl = json.loads((SRC / "fluid_stacker.json").read_text(encoding="utf-8"))
st = json.loads((SRC / "state_stacker.json").read_text(encoding="utf-8"))
check(fl["feature"]["lines"][0]["pool"] == 4344 and st["rows"] == 4344,
      "stackers were fit on the R7 OOF file (%s, md5 %s) with 4344 pool rows" % (fl["feature"]["lines"][0]["oof"], fl["feature"]["lines"][0]["oof_md5"][:8]))
check(fl["folds"] == 4 and st["folds"] == 4, "stacker cross-fitting folds == 4")

# (d) every train study's held-out fold
train = list(csv.DictReader(open(ROOT / "_data" / "train.csv", encoding="utf-8", newline="")))
labels = list(train[0].keys())[2:]
ser = list(csv.DictReader(open(ROOT / "_data" / "train_series.csv", encoding="utf-8", newline="")))
nser = {}
for r in ser:
    nser[r["StudyInstanceUID"]] = nser.get(r["StudyInstanceUID"], 0) + 1
ids = [r["StudyInstanceUID"] for r in train]
check(len(ids) == 4407 and len(set(ids)) == 4407, "train.csv: 4407 unique studies")


def r7_fold(u):          # the trainer's rule, K = 4
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


def shard(u, n):         # our session shards: md5, independent of the sha1 fold rule
    return int(hashlib.md5(u.encode()).hexdigest()[:8], 16) % n


gold = {r["StudyInstanceUID"] for r in train if any(r[L] != "" for L in labels)}
check(len(gold) == 58, "58 gold studies (all 12 labels present) in train.csv")
counts = [0] * 4
gcounts = [0] * 4
rows = []
for u in ids:
    k = r7_fold(u)
    counts[k] += 1
    gcounts[k] += int(u in gold)
    rows.append([u, k, int(u in gold), nser.get(u, 0), shard(u, 2), shard(u, 3), shard(u, 4)])
print("held-out fold counts, all 4407 train studies:", counts)
print("of which gold (never trained by any fold):", gcounts, "| non-gold:", [c - g for c, g in zip(counts, gcounts)])
print("R7 pool (non-gold, with px, not blanked) = 4344, so %d non-gold train studies were in NO fold's training set "
      "(4 without px rows + 1 blanked report, per run_config feature_studies 4403); any fold model is out-of-fold for them" % (4349 - 4344))
with open(HERE / "r7_fold_map.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["StudyInstanceUID", "r7_fold", "is_gold", "n_series", "shard_of_2", "shard_of_3", "shard_of_4"])
    w.writerows(rows)
print("wrote", HERE / "r7_fold_map.csv")
print("ALL_PASS" if ok_all else "SOME_CHECK_FAILED")
sys.exit(0 if ok_all else 1)
