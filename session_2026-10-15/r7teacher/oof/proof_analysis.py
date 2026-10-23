"""Empirical check of the R7 fold mapping from proof runs (all four fold models on the same train studies).

  python proof_analysis.py OUT_DIR [OUT_DIR ...]

Test A (outlier): the model whose RMS logit distance to the mean of the other three is largest == the sha1 hash fold?
  Null (no train-fit signal or a wrong mapping): 1/4 per study.
Test B (margin rank, the powered one): for every cell where our LLM label is confident (>= .9 or <= .1), margin_k = (2y - 1) z_k.
  A model that trained on the study (with R7's report-derived targets, which agree with confident LLM labels) should push its logit
  further toward the label than the model that did not. Rank of the hash-fold model's margin among the 4 (0 = lowest).
  Null: uniform on {0,1,2,3}, mean 1.5. Inference by study-level permutation (each study's ranks kept together), and per fold.
Control: gold studies (never trained by any fold): the hash-fold model's rank must look uniform. No em dashes.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
csv.field_size_limit(1 << 30)


def r7_fold(u):
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


llm = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(ROOT / "_labels" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv", encoding="utf-8"))}
train = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(ROOT / "_data" / "train.csv", encoding="utf-8", newline=""))}
gold = {u for u, r in train.items() if r["ACL"] != ""}
Z = {}
for d in sys.argv[1:]:
    for jd in sorted((Path(d) / "r7oof").glob("proof_*")):
        rec = json.loads((jd / "receipt.json").read_text(encoding="utf-8"))
        assert rec["models"] == ["E_lrprior_s0_f%d.pt" % k for k in range(4)], rec["models"]
        for u, v in json.loads((jd / "dump.json").read_text(encoding="utf-8")).items():
            Z[u] = np.array(v["z"] if isinstance(v, dict) else v, np.float64)
print("proof studies with 4-model logits: %d (gold controls %d)" % (len(Z), sum(u in gold for u in Z)))


def ranks_for(u, k, ylab):
    out = []
    for j, L in enumerate(LABELS):
        y = ylab[j]
        if y is None or 0.1 < y < 0.9:
            continue
        m = (1.0 if y >= 0.9 else -1.0) * Z[u][:, j]
        out.append(int((m < m[k]).sum()) + 0.5 * int((m == m[k]).sum() - 1))
    return out


rng = np.random.default_rng(0)


def report(name, uids, label_src):
    A = []
    B = {}
    for u in uids:
        k = r7_fold(u)
        dist = [float(np.sqrt(np.mean((Z[u][i] - np.delete(Z[u], i, 0).mean(0)) ** 2))) for i in range(4)]
        A.append(int(np.argmax(dist)) == k)
        ylab = [float(label_src[u][L]) if label_src[u][L] != "" else None for L in LABELS]
        B[u] = ranks_for(u, k, ylab)
    allr = np.array([r for v in B.values() for r in v], float)
    per_study = np.array([np.mean(v) for v in B.values() if v])
    # study-level permutation null: each cell's rank replaced by a uniform draw; statistic = mean over cells
    sizes = [len(v) for v in B.values() if v]
    null = np.array([np.mean(np.concatenate([rng.integers(0, 4, s) for s in sizes])) for _ in range(20000)])
    p = float((null <= allr.mean()).mean())
    print("%s: studies %d | A outlier == hash fold %d / %d (null 1/4 each) | B cells %d, held-out margin rank counts (0 lowest) %s, "
          "mean %.3f (null 1.5), one-sided p %.2g; studies with mean rank < 1.5: %d / %d" % (
              name, len(uids), sum(A), len(A), len(allr), np.bincount(np.floor(allr).astype(int), minlength=4).tolist(), allr.mean(), p,
              int((per_study < 1.5).sum()), len(per_study)))
    return B


def fold_permutation_test(name, uids, label_src, n_perm=20000):
    """Test C: statistic = mean over confident cells of the margin rank of the model ASSIGNED to the study; observed assignment = the
    sha1 hash fold; null = the hash folds shuffled across the same studies (keeps every model's own calibration in both). Also per
    model k: its mean margin rank on studies where it is the hash fold vs where it is not."""
    if len(uids) < 2:
        return
    ylabs = {u: [float(label_src[u][L]) if label_src[u][L] != "" else None for L in LABELS] for u in uids}
    R = {}
    for u in uids:
        R[u] = np.array([ranks_for(u, k, ylabs[u]) for k in range(4)], float)    # [4, cells]
    assign = np.array([r7_fold(u) for u in uids])
    def stat(a):
        return np.mean(np.concatenate([R[u][a[i]] for i, u in enumerate(uids)]))
    obs = stat(assign)
    null = np.array([stat(rng.permutation(assign)) for _ in range(n_perm)])
    print("%s: C fold-shuffle test: mean margin rank of the hash-fold model %.3f, shuffled-fold null mean %.3f (sd %.3f), one-sided p %.3g" % (
        name, obs, null.mean(), null.std(), float((null <= obs).mean())))
    for k in range(4):
        own = np.concatenate([R[u][k] for i, u in enumerate(uids) if assign[i] == k] or [np.array([])])
        oth = np.concatenate([R[u][k] for i, u in enumerate(uids) if assign[i] != k] or [np.array([])])
        print("  model f%d: mean margin rank when held-out %.3f (cells %d) vs when trained on the study %.3f (cells %d)" % (
            k, own.mean() if len(own) else float("nan"), len(own), oth.mean() if len(oth) else float("nan"), len(oth)))


non_gold = [u for u in Z if u not in gold]
B = report("NON-GOLD vs LLM labels", non_gold, llm)
fold_permutation_test("NON-GOLD vs LLM labels", non_gold, llm)
for k in range(4):
    uk = [u for u in non_gold if r7_fold(u) == k]
    if uk:
        rk = np.array([r for u in uk for r in B[u]], float)
        print("  fold %d: studies %d, cells %d, mean held-out margin rank %.3f" % (k, len(uk), len(rk), rk.mean() if len(rk) else float("nan")))
gu = [u for u in Z if u in gold]
if gu:
    report("GOLD CONTROLS (never trained) vs TRUE labels, 'hash fold' model", gu, train)
    report("GOLD CONTROLS vs LLM labels", gu, llm)
    fold_permutation_test("GOLD CONTROLS vs TRUE labels", gu, train)
