"""Merge the r7_oof.csv files of several rsna-r7-oof sessions into one teacher table and report coverage.

  python merge_sessions.py OUT.csv SESSION_DIR_OR_CSV [...]

Checks: every row's r7_fold == sha1(uid)[:8] % 4; a study scored in two sessions must agree (max |dp| <= 1e-6, else both kept
out and listed); all 4,407 train ids accounted for (scored / missing list written next to OUT). No em dashes.
"""
import csv, hashlib, sys
from pathlib import Path

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]


def r7_fold(u):
    return int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4


out = Path(sys.argv[1])
srcs = []
for s in sys.argv[2:]:
    p = Path(s)
    srcs.append(p / "r7_oof.csv" if p.is_dir() else p)
train = [r["StudyInstanceUID"] for r in csv.DictReader(open(ROOT / "_data" / "train.csv", encoding="utf-8", newline=""))]
rows, header, conflicts = {}, None, []
for f in srcs:
    with open(f, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        header = header or rd.fieldnames
        assert rd.fieldnames == header, "column layout differs: %s" % f
        for r in rd:
            u = r["StudyInstanceUID"]
            assert int(r["r7_fold"]) == r7_fold(u), (u, r["r7_fold"])
            if u in rows:
                d = max(abs(float(rows[u]["p|" + L]) - float(r["p|" + L])) for L in LABELS)
                if d > 1e-6:
                    conflicts.append((u, d))
                if rows[u]["precision"] != "full" and r["precision"] == "full":
                    rows[u] = r
                continue
            rows[u] = r
bad = {u for u, _ in conflicts}
with open(out, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=header)
    w.writeheader()
    for u in sorted(rows):
        if u not in bad:
            w.writerow(rows[u])
missing = [u for u in train if u not in rows or u in bad]
with open(out.with_name(out.stem + "_missing.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(missing) + ("\n" if missing else ""))
per_fold = [sum(1 for u in rows if r7_fold(u) == k and u not in bad) for k in range(4)]
print("MERGED %d sessions -> %s: %d studies (full precision %d), per fold %s, conflicts %d, missing %d of %d" % (
    len(srcs), out, len(rows) - len(bad), sum(r["precision"] == "full" for u, r in rows.items() if u not in bad), per_fold,
    len(conflicts), len(missing), len(train)))
