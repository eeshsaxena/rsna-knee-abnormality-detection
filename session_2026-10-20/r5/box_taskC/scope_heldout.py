# scope_heldout.py (r5 task C, 10-20): AGGREGATES ONLY. Counts OAI follow-up knee-visits that the OAI-trained
# T123v00 fleet (DT) never saw (knee not in the V00 training ids), with a strict subject-level variant.
import os, re, pandas as pd, numpy as np
H = os.path.expanduser
tr = set(l.strip() for l in open(H("~/oai_build/teacher/ids_v00.txt")) if l.strip())
trs = set(s.split("_")[1] for s in tr)
L = pd.read_csv(H("~/oai_build/labels_v/oai_labels_T123_allv.csv"))
idc = L.columns[0]
ids = L[idc].astype(str)
knee = ids.str.replace(r"_v\d+$", "", regex=True)
subj = ids.str.split("_").str[1]
isv00 = ~ids.str.contains(r"_v\d+$", regex=True)
print("allv rows", len(L), "| v00-form rows", int(isv00.sum()), "| follow-up rows", int((~isv00).sum()))
hk = (~knee.isin(tr)) & (~isv00)
hs = (~subj.isin(trs)) & (~isv00)
store = H("~/oai_store")
have = ids.map(lambda s: os.path.exists(f"{store}/{s}.npz"))
for tag, m in (("knee-heldout", hk), ("subject-heldout", hs)):
    mm = m & have
    print(f"{tag}: rows {int(m.sum())} with npz {int(mm.sum())} | knees {knee[mm].nunique()} | subjects {subj[mm].nunique()}")
    sub = L[mm]
    for c in L.columns[1:]:
        y = sub[c]; ok = y.notna(); pos = int((y[ok] >= 0.5).sum())
        print(f"   {c:18s} labelled {int(ok.sum()):5d} pos(>=.5) {pos:5d}")
