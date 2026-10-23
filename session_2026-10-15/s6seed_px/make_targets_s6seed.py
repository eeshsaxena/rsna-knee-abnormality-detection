"""targets_s6seed.csv (10-15, BOX_PLAN_1015 S3) for the S6 member s43: compact_train 14767c7a --targets format (Heliosli's targets
layout, the one nartaa's E_prior arm reads): StudyInstanceUID, is_gold, review_holdout, fold, then per class L|prior, L|lr_prior,
L|strict.  Source = ~/raptor2/labels_raptor_rsna_blendjev_sol.csv (our blendjev structure, sol read; RSNA only).
  gold58 (rows with official labels in the competition train.csv == gold74_eval_set source original_gold58) -> is_gold 1, every
         cell blank (the trainer asserts it and never trains on them)
  other  -> is_gold 0, review_holdout 0, fold 0 (the trainer makes its own sha1 hash folds; all-data uses the whole pool)
  prior    = the blendjev_sol soft label, unclipped (arm E_prior = BCE on L|prior; values already in (0, 1))
  lr_prior = odds(prior) / odds(base_c), base_c = mean prior over the non-gold pool (format completeness; E_prior does not read it)
  strict   = 1[prior >= 0.5]"""
import argparse, csv, hashlib
import numpy as np
import pandas as pd
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--train", required=True, help="competition train.csv (official labels mark the gold studies)")
ap.add_argument("--gold74", required=True, help="gold74_eval_set.csv (cross-check: source original_gold58)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
tr = pd.read_csv(a.train, dtype={"StudyInstanceUID": str})
uids = sorted(set(tr["StudyInstanceUID"]))
gold = set(tr.loc[tr[LABELS].notna().all(1), "StudyInstanceUID"])
g74 = pd.read_csv(a.gold74, dtype={"StudyInstanceUID": str})
g58 = set(g74.loc[g74["source"] == "original_gold58", "StudyInstanceUID"])
assert len(uids) == 4407, len(uids)
assert gold == g58 and len(gold) == 58, (len(gold), len(g58), len(gold ^ g58))
lab = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS].astype(float)
assert lab.index.is_unique and set(lab.index) == set(uids), "label table does not cover the train studies 1:1"
pool = [u for u in uids if u not in gold]
q = lab.loc[pool]
assert np.isfinite(q.values).all() and (q.values > 0).all() and (q.values < 1).all()
base = q.mean(0)
cols = ["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [l + "|" + s for l in LABELS for s in ("prior", "lr_prior", "strict")]
with open(a.out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, cols)
    w.writeheader()
    for u in uids:
        g = u in gold
        r = {"StudyInstanceUID": u, "is_gold": int(g), "review_holdout": 0, "fold": 0}
        for l in LABELS:
            if g:
                r[l + "|prior"] = r[l + "|lr_prior"] = r[l + "|strict"] = ""
                continue
            p, b = float(q.at[u, l]), float(base[l])
            r[l + "|prior"] = "%.8g" % p
            r[l + "|lr_prior"] = "%.6g" % ((p / (1 - p)) / (b / (1 - b)))
            r[l + "|strict"] = "%d" % (p >= 0.5)
        w.writerow(r)
print("labels md5 %s | train.csv md5 %s" % (md5(a.labels), md5(a.train)))
print("studies %d gold %d pool %d | pool mean prior per class: %s" % (len(uids), len(gold), len(pool),
      " ".join("%.3f" % base[l] for l in LABELS)))
print("wrote %s md5 %s sha256 %s" % (a.out, md5(a.out), hashlib.sha256(open(a.out, "rb").read()).hexdigest()))
