#!/usr/bin/env python
# ho_infer.py (r5 task C, 10-20, Claude): copy of ~/oai_build/oai_infer_gpu.py (same tta_bench pass A, offset 0, no mirror)
# WITHOUT the OAI-trained refusal, run ONLY on follow-up OAI knee-visits whose knee was never in the T123v00 training ids.
# Purpose: measure the in-domain held-out OAI gain of the OAI-trained twin (DT) over the OAI-free twin (D). Inference only.
# Box only; predictions stay in ~/r5_oai/taskC/preds and never leave the box.
import os, sys, time
sys.dont_write_bytecode = True
H = os.path.expanduser
sys.path.insert(0, H("~/novel/code"))
import numpy as np, pandas as pd, torch
import tta_bench as tb
ids = [l.strip() for l in open(H("~/r5_oai/taskC/ids_heldout.txt")) if l.strip()]
assert all(s.startswith("oai_") for s in ids)
out = H("~/r5_oai/taskC/preds"); os.makedirs(out, exist_ok=True)
for p in sys.argv[1:]:
    run = os.path.basename(os.path.dirname(p)); dst = f"{out}/{run}__raw__A__gpu.csv"
    if os.path.exists(dst): print("skip", run, flush=True); continue
    m, info = tb.load_member(p, "cuda"); P = np.full((len(ids), 12), np.nan, np.float32); t1 = time.time()
    with torch.no_grad():
        for k, s in enumerate(ids):
            st = tb.study_stacks(H("~/oai_store"), s); x, pr, ct, nv = tb.make_input(st, 0.0, False, "cuda")
            if nv: P[k] = tb.forward(m, x, pr, ct, "cuda")
            if k and k % 400 == 0: print(f"  {run} {k}/{len(ids)} {(time.time() - t1) / k:.2f}s/study", flush=True)
    tmp = dst + ".tmp"; pd.DataFrame(P, columns=tb.TARGETS, index=pd.Index(ids, name="StudyInstanceUID")).to_csv(tmp); os.replace(tmp, dst)
    print(f"{run} ep{info['epoch_1based']} cv {info['cv']:.4f} | {len(ids)} studies, nan rows {int(np.isnan(P).any(1).sum())}, {time.time() - t1:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
print("ALL_DONE", flush=True)
