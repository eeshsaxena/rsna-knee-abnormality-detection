"""targets_s6seed.csv for ct_s6v100.py / compact_train 14767c7a (BOX_PLAN_1015 S3): the E_prior arm reads <label>|prior.

Source: the blendjev_sol soft labels (one row per train study, 12 probabilities).  Gold studies (the train.csv rows with official
labels, gold58) get is_gold=1 and BLANK prior / strict cells (compact_train asserts it and never trains on them).  Every other
study: is_gold=0, review_holdout=0, fold = folds5 fold, <label>|prior = the blendjev_sol probability, <label>|strict blank.
Never reads anything else; writes only --out (+ a sha256 receipt)."""
import argparse, csv, hashlib, json, math, sys

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels_raptor_rsna_blendjev_sol.csv")
    ap.add_argument("--train-csv", required=True, help="competition train.csv (gold = rows with all 12 official labels)")
    ap.add_argument("--folds", required=True, help="folds5.csv (StudyInstanceUID, fold)")
    ap.add_argument("--uids", default=None, help="optional subset: one uid per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8-sig"))}
    tr = list(csv.DictReader(open(a.train_csv, encoding="utf-8-sig")))
    gold = {r["StudyInstanceUID"] for r in tr if all((r.get(L) or "").strip() != "" for L in LABELS)}
    folds = {r["StudyInstanceUID"]: int(r["fold"]) for r in csv.DictReader(open(a.folds, encoding="utf-8-sig"))}
    uids = sorted({r["StudyInstanceUID"] for r in tr})
    if a.uids:
        want = [l.strip() for l in open(a.uids) if l.strip()]
        miss = [u for u in want if u not in set(uids)]
        if miss:
            raise SystemExit("%d subset uids are not in train.csv" % len(miss))
        uids = sorted(set(want))
    bad, rows = [], []
    for u in uids:
        g = u in gold
        r = {"StudyInstanceUID": u, "is_gold": "1" if g else "0", "review_holdout": "0", "fold": str(folds.get(u, 0))}
        if u not in folds:
            bad.append("no fold: " + u)
        src = lab.get(u)
        for L in LABELS:
            r[L + "|strict"] = ""
            if g:
                r[L + "|prior"] = ""
                continue
            if src is None or (src.get(L) or "") == "":
                bad.append("no label %s for %s" % (L, u))
                r[L + "|prior"] = ""
                continue
            v = float(src[L])
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                bad.append("label %s=%r out of [0,1] for %s" % (L, src[L], u))
            r[L + "|prior"] = src[L]
        rows.append(r)
    if bad:
        print("\n".join(bad[:20]))
        raise SystemExit("make_targets_s6: %d problems" % len(bad))
    cols = ["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [L + "|prior" for L in LABELS] + [L + "|strict" for L in LABELS]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    rec = {"out": a.out, "out_sha256": h(a.out), "labels": a.labels, "labels_sha256": h(a.labels), "train_csv_sha256": h(a.train_csv),
           "folds_sha256": h(a.folds), "rows": len(rows), "gold_rows": sum(r["is_gold"] == "1" for r in rows),
           "pool_candidates": sum(r["is_gold"] == "0" for r in rows)}
    json.dump(rec, open(a.out + ".receipt.json", "w"), indent=1)
    print("TARGETS " + json.dumps(rec))


if __name__ == "__main__":
    main()
