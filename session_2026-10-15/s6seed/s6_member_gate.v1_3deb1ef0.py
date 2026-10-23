"""Amendment 7 member gate (BOX_PLAN_1015 sec 4.2), the gold58 part, CPU only:
  (g1) our gold58 macro AUC >= nartaa gold58 macro AUC - .015
  (g2) rank-mean(nartaa, ours) gold58 macro AUC >= nartaa - .003
Inputs: the two models' gold58 predictions through the SERVE path (compact_infer 34f433fe predictions.csv: StudyInstanceUID + the 12
labels as probabilities, any column order; read BY NAME) and the competition train.csv (the 58 rows with official labels).
Macro AUC = mean over the labels that have both classes among the 58 (rank AUC, average ties); the label set is printed.
rank-mean = per label, mean of the two models' average-tie percentile ranks over the 58 studies.
Writes nothing unless --out is given (a one-line verdict).  Also run as `--selftest` (synthetic data, no files)."""
import argparse, csv, json, sys
import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def rankdata(x):
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(y, s):
    y = np.asarray(y, np.float64)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro(Y, P):
    per = {}
    for j, L in enumerate(LABELS):
        a = auc(Y[:, j], P[:, j])
        if a is not None:
            per[L] = a
    return float(np.mean(list(per.values()))), per


def read_preds(path, uids):
    rows = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
    miss = [u for u in uids if u not in rows]
    if miss:
        raise SystemExit("%s lacks %d of the %d gold studies" % (path, len(miss), len(uids)))
    P = np.array([[float(rows[u][L]) for L in LABELS] for u in uids], np.float64)
    if not np.isfinite(P).all():
        raise SystemExit("%s has non-finite predictions" % path)
    return P


def gate(Y, Pn, Po):
    mn, per_n = macro(Y, Pn)
    mo, per_o = macro(Y, Po)
    R = np.stack([(rankdata(Pn[:, j]) + rankdata(Po[:, j])) / 2.0 for j in range(len(LABELS))], 1)
    mr, per_r = macro(Y, R)
    g1 = mo >= mn - 0.015
    g2 = mr >= mn - 0.003
    return {"nartaa_gold58": round(mn, 5), "ours_gold58": round(mo, 5), "rankmean_gold58": round(mr, 5),
            "g1_ours_ge_nartaa_minus_.015": bool(g1), "g2_rankmean_ge_nartaa_minus_.003": bool(g2), "labels_scored": sorted(per_n),
            "verdict": "PASS" if (g1 and g2) else "FAIL"}


def selftest():
    rng = np.random.default_rng(0)
    Y = (rng.random((58, 12)) < 0.3).astype(float)
    Y[:, 11] = 0                                   # a label with one class only: must be skipped
    base = Y + rng.normal(0, 0.8, Y.shape)
    good = base + rng.normal(0, 0.05, Y.shape)      # a near-copy of nartaa: must PASS
    bad = rng.normal(0, 1, Y.shape)
    ok = True
    r1 = gate(Y, base, good)
    r2 = gate(Y, base, bad)
    ok &= r1["verdict"] == "PASS" and "Fracture" not in r1["labels_scored"]
    ok &= r2["verdict"] == "FAIL" and not r2["g1_ours_ge_nartaa_minus_.015"]
    ok &= abs(auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-12 and abs(auc([0, 1, 0, 1], [1, 1, 1, 1]) - 0.5) < 1e-12
    r3 = gate(Y, base, base)                       # identical models: rank-mean == each
    ok &= abs(r3["rankmean_gold58"] - r3["nartaa_gold58"]) < 1e-12
    print("S6_MEMBER_GATE_SELFTEST %s %s" % ("PASS" if ok else "FAIL", json.dumps({"good": r1, "bad": r2["verdict"]})))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv")
    ap.add_argument("--nartaa")
    ap.add_argument("--ours")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    tr = list(csv.DictReader(open(a.train_csv, encoding="utf-8-sig")))
    gold = [r for r in tr if all((r.get(L) or "").strip() != "" for L in LABELS)]
    uids = [r["StudyInstanceUID"] for r in gold]
    if len(uids) != 58:
        raise SystemExit("expected 58 gold studies, found %d" % len(uids))
    Y = np.array([[float(r[L]) for L in LABELS] for r in gold], np.float64)
    res = gate(Y, read_preds(a.nartaa, uids), read_preds(a.ours, uids))
    line = "%s A7_GOLD58 %s" % (res["verdict"], json.dumps(res))
    print(line)
    if a.out:
        open(a.out, "w").write(line + "\n")
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
