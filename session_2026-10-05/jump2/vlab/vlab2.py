# vlab2.py: KX token budget per report with the real Qwen2.5 tokenizer + chat template (CPU, tokenizer only, no model).
import os, re, unicodedata, numpy as np, pandas as pd
os.environ["HF_HUB_OFFLINE"] = "1"
src = open(os.path.expanduser("~/jump2/kx/kx_pilot.py"), encoding="utf-8").read()
seg = src[src.index("SYSTEM = ("): src.index("# ---------------- model")]
ns = {"re": re, "unicodedata": unicodedata}; exec(seg, ns)
SYSTEM, DEFS, Q1, Q2, norm_report = ns["SYSTEM"], ns["DEFS"], ns["Q1"], ns["Q2"], ns["norm_report"]
L = list(DEFS.keys())
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
def chat(m): return tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)
ID = "StudyInstanceUID"; RT = os.path.expanduser("~/rsna_train")
tr = pd.read_csv(f"{RT}/train.csv", dtype={ID: str}).drop_duplicates(ID).set_index(ID)
lang = pd.read_csv(f"{RT}/_reports_lang.csv", dtype={ID: str}).drop_duplicates(ID).set_index(ID).lang
rows = []
for u, t in tr.Report.fillna("").astype(str).items():
    rep = norm_report(t); user0 = 'REPORT:\n"""\n' + rep + '\n"""\n\n'
    s1 = {l: chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0 + Q1.format(lab=l, d=DEFS[l])}]) for l in L}
    seqs = list(s1.values()); n = min(map(len, seqs)); k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs): k += 1
    k -= 1
    q1 = sum(len(s) - k for s in seqs)
    q2 = 0
    for l in L:
        s2 = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0 + Q1.format(lab=l, d=DEFS[l])},
                   {"role": "assistant", "content": "A"}, {"role": "user", "content": Q2}])
        q2 += len(s2) - len(s1[l])
    rows.append(dict(uid=u, lang=lang.get(u, "?"), prefix=k, q1=q1, q2=q2, total=k + q1 + q2, chars=len(t), capped=len(t) > 3500))
D = pd.DataFrame(rows)
sys_only = len(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]))
print("system+template-only tokens", sys_only)
print(D[["prefix", "q1", "q2", "total"]].describe(percentiles=[.5, .9, .99]).round(0).to_string())
print(D.groupby("lang")[["prefix", "total"]].median().round(0).to_string())
print("reports > 3500 chars (truncated by KX):", int(D.capped.sum()))
tot = D.total.sum(); print("TOTAL tokens over 4407 reports:", int(tot))
# FLOP model: ~2 * 7.07e9 (non-embedding params + lm_head on every position, as HF computes all logits) per token
fl = 2 * 7.07e9 * tot
for tf in (25, 40, 60, 112):
    print(f"  at {tf} TFLOPS sustained: {fl / (tf * 1e12) / 3600:.2f} h  ({fl / (tf * 1e12) / len(D):.2f} s/report)  [forward-only, excludes per-call overhead]")
print("forward calls per report (unbatched):", 1 + 2 * len(L))
