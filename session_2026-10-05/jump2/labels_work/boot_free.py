# Bootstrap the free (CPU) label variants vs mixSA_logit and mix50_logit; writes the best free variant csv to ~/jump2.
import numpy as np, pandas as pd
from scipy.stats import rankdata
exec(open("text_ens.py").read().split("SA=lambda")[0])
SA = lambda vd: np.where(vd == "UNK", .75, .5)
def mixwith(text, te):
    out = pd.DataFrame(index=ids)
    for l in L:
        vd = pk[l + "__verdict"].fillna("UNK").values; w = SA(vd)
        out[l] = (1 - w) * text[l].values + w * lm(np.clip(te[l].values, 1e-4, 1 - 1e-4), text[l].values)
    return out
VAR = {"mix50_logit": pd.read_csv(f"{RT}/labels_mix50_logit.csv").set_index(ID).reindex(ids)[L],
       "mixSA_logit": pd.read_csv(f"{RT}/labels_mixSA_logit.csv").set_index(ID).reindex(ids)[L],
       "SA_wd_text(v4+jev)": mixwith(TX["v4+jev (rank .5/.5)"], wd),
       "SA_wd_text(v4,jev,pilk,lixin)": mixwith(TX["v4 .6 + jev .2 + pilk .1 + lixin .1"], wd),
       "SA_raptorproxy_text(v4)": mixwith(v4[L], rp),
       "SA_raptorproxy_text(v4+jev)": mixwith(TX["v4+jev (rank .5/.5)"], rp)}
G = gold.values; rng = np.random.default_rng(1); B = rng.integers(0, 58, (3000, 58))
def fa(y, s):
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def bm(d):
    D = d.loc[gold.index].values; out = []
    for ix in B:
        out.append(np.mean([fa(G[ix, j], D[ix, j]) for j in range(12) if 0 < G[ix, j].sum() < 58]))
    return np.array(out)
BM = {k: bm(v) for k, v in VAR.items()}
ref1, ref2 = BM["mixSA_logit"], BM["mix50_logit"]
for k, v in VAR.items():
    m = score(v)[0]; d = BM[k] - ref1
    print(f"{k:32s} {m:.4f}  P>mixSA {np.mean(d > 0):.2f}  P>mix50 {np.mean(BM[k] > ref2):.2f}  d90 vs mixSA [{np.quantile(d, .05):+.4f},{np.quantile(d, .95):+.4f}]")
VAR["SA_wd_text(v4+jev)"].rename_axis(ID).reset_index().to_csv("labels_SAwd_textv4jev.csv", index=False)
