# Pre-registered: does ensembling existing independent text reads into the TEXT half help? (all variants reported)
import numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
exec(open("gate_grid.py").read().split("W={")[0])  # reuse loaders (v4, wd, rp, pk, gold, lm, score)
LI="labels_in"
jev=pd.read_csv(f"{LI}/jev_final2_scores.csv").set_index(ID).reindex(ids)
pkv=pk[L].astype(float).fillna(pk[L].astype(float).median())
lx=pd.read_csv(f"{LI}/lixin_gpt56_conf.csv").set_index(ID).reindex(ids)
lxs=pd.DataFrame({l:np.where(lx[l]>=.5,.5+.5*lx[l+"__conf"],.5-.5*lx[l+"__conf"]) for l in L},index=ids)
sf=pd.read_csv(f"{LI}/steven_full.csv").set_index(ID).reindex(ids)
def rk(d): return d[L].rank(pct=True)
def back(r):  # map a rank-mean back onto the v4 marginal (ECDF) so the mixing code sees v4-like scale
    out=pd.DataFrame(index=ids)
    for l in L: out[l]=np.quantile(v4[l].values, r[l].rank(pct=True).values-0.5/len(r))
    return out
TX={
 "v4":v4[L],
 "v4+jev (rank .5/.5)": back((rk(v4)+rk(jev))/2),
 "v4 .6 + jev .2 + pilk .1 + lixin .1": back(.6*rk(v4)+.2*rk(jev)+.1*rk(pkv)+.1*rk(lxs)),
 "v4+jev+pilk+lixin+steven_full equal": back((rk(v4)+rk(jev)+rk(pkv)+rk(lxs)+rk(sf))/5),
}
SA=lambda vd,v: np.where(vd=="UNK",.75,.5)
for n,t in TX.items():
    m0,a0=score(t)
    out=pd.DataFrame(index=ids)
    for l in L:
        vd=pk[l+"__verdict"].fillna("UNK").values; w=SA(vd,None)
        out[l]=(1-w)*t[l].values + w*lm(np.clip(wd[l].values,1e-4,1-1e-4), t[l].values)
    m1,a1=score(out)
    print(f"{n:40s} text-only {m0:.4f} | mixSA(wd) {m1:.4f} | "+" ".join(f"{x:.3f}" for x in a0))
