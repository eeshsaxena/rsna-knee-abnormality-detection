"""Targets for compact_train.py (R7 recipe) from OUR report labels (no official label enters a training cell).
Writes:
  targets.csv       StudyInstanceUID, is_gold, review_holdout, fold, then per class L|prior, L|lr_prior, L|strict
                    gold (studies with official labels in train.csv) -> is_gold 1, every cell blank (the trainer asserts this)
  cell_weights.csv  StudyInstanceUID, L|w  (arm 'E_lrprior=lr:lr_prior:w'); w from the FINAL2 agreement tier when given, else 1
prior   = clip(soft label, lo, hi)
lr_prior = odds(prior) / odds(base_c), base_c = mean prior over the non-gold pool (so a constant model sits at the pool rate)
strict  = 1[prior >= 0.5]
fold    = 0 for every non-gold study (the trainer makes its own sha1 hash folds; only fold -3 has a meaning: review holdout)."""
import argparse, csv
import numpy as np
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TIER_W = {"very_high": 1.0, "high": 0.85, "medium": 0.65, "low": 0.45}
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="soft labels csv (StudyInstanceUID + 12 class columns), e.g. llm_labels_v4_blend.csv")
ap.add_argument("--train", required=True, help="competition train.csv (only used to find the gold studies: rows with official labels)")
ap.add_argument("--tiers", default=None, help="optional FINAL2_labels_with_confidence.csv: <class>_tier -> cell weight")
ap.add_argument("--out", required=True)
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.98)
a = ap.parse_args()
import pandas as pd
tr = pd.read_csv(a.train, dtype={"StudyInstanceUID": str})
gold = set(tr.loc[tr[LABELS].notna().all(1), "StudyInstanceUID"])
lab = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
tiers = pd.read_csv(a.tiers, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID") if a.tiers else None
uids = sorted(set(tr["StudyInstanceUID"]))
pool = [u for u in uids if u not in gold and u in lab.index]
q = lab.loc[pool, LABELS].astype(float).clip(a.lo, a.hi)
base = q.mean(0)
print("studies %d gold %d pool %d (no label: %d)" % (len(uids), len(gold), len(pool), len([u for u in uids if u not in gold and u not in lab.index])))
print("pool base rate per class:", {l: round(float(base[l]), 4) for l in LABELS})
with open(a.out + "/targets.csv", "w", newline="", encoding="utf-8") as fh, open(a.out + "/cell_weights.csv", "w", newline="", encoding="utf-8") as fw:
    w = csv.DictWriter(fh, ["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [l + "|" + s for l in LABELS for s in ("prior", "lr_prior", "strict")])
    ww = csv.DictWriter(fw, ["StudyInstanceUID"] + [l + "|w" for l in LABELS])
    w.writeheader(); ww.writeheader()
    for u in uids:
        g = u in gold
        if not g and u not in lab.index:
            continue
        r = {"StudyInstanceUID": u, "is_gold": int(g), "review_holdout": 0, "fold": 0}
        rw = {"StudyInstanceUID": u}
        for l in LABELS:
            if g:
                r[l + "|prior"] = r[l + "|lr_prior"] = r[l + "|strict"] = ""
                rw[l + "|w"] = "0"
                continue
            p = float(q.at[u, l]); b = float(base[l])
            r[l + "|prior"] = "%.6f" % p
            r[l + "|lr_prior"] = "%.6g" % ((p / (1 - p)) / (b / (1 - b)))
            r[l + "|strict"] = "%d" % (p >= 0.5)
            t = None if tiers is None or u not in tiers.index else tiers.at[u, l + "_tier"]
            rw[l + "|w"] = "%.3f" % TIER_W.get(str(t), 1.0)
        w.writerow(r); ww.writerow(rw)
print("wrote", a.out + "/targets.csv", a.out + "/cell_weights.csv")
