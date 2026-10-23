# make_ids.py (r5 task C): writes the held-out follow-up OAI ids to a box-only file. Prints counts only.
import os, pandas as pd
H = os.path.expanduser
tr = set(l.strip() for l in open(H("~/oai_build/teacher/ids_v00.txt")) if l.strip())
L = pd.read_csv(H("~/oai_build/labels_v/oai_labels_T123_allv.csv")); ids = L[L.columns[0]].astype(str)
knee = ids.str.replace(r"_v\d+$", "", regex=True); fu = ids.str.contains(r"_v\d+$", regex=True)
keep = [s for s, k, f in zip(ids, knee, fu) if f and k not in tr and os.path.exists(H(f"~/oai_store/{s}.npz"))]
open(H("~/r5_oai/taskC/ids_heldout.txt"), "w").write("\n".join(keep) + "\n"); print("written", len(keep))
