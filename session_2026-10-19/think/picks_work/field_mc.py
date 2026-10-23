# field_mc.py (10-19, think/picks): private-rank Monte Carlo for final-pick pairs.
# Our side: private margins over the heliosli copy block from picks_post.py (post_<SC>_<REG>.npz; MODEL).
# Field at the deadline (10-22 23:59 UTC), everything relative to the copy block's PRIVATE score (= 0):
#   counts and public positions from the 10-19 11:00 UTC snapshot (MEASURED) projected with the MEASURED growth rates (MODEL);
#   private = shrunk public + noise (MODEL / GUESS parameters, scenario flag).
# Team score = max over its two picks; rank = 1 + #field teams with higher private score (+ the copy block if we are below it).
import sys, json, numpy as np
from scipy.stats import norm
SC = sys.argv[1] if len(sys.argv) > 1 else "base"; FS = sys.argv[2] if len(sys.argv) > 2 else "base"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
Z = np.load(f"post_{SC}.npz"); ENT = list(Z["ent"]); PV = Z["priv"].astype(float)
rng = np.random.default_rng(22)
P = dict(N=6336, n_copy=750,
         # >= .959 at the deadline: today's MEASURED band counts + ~75 projected arrivals (20-25/day, mostly .959 / .960)
         t1_bands={0.959: 92, 0.960: 65, 0.961: 25, 0.962: 27, 0.963: 18, 0.964: 10, 0.965: 6, 0.966: 6, 0.967: 5, 0.968: 3, 0.969: 1},
         f959=9.8e-4,               # .959 floor above the copy block (c4 ~ copy + 8.7e-4, ~1.1e-4 under .959; MODEL)
         t1_b=1.5e-4, t1_s=1.2e-3, lin_sd=2e-4,      # T1: selection bias, private-vs-public noise, common other-lineage shock
         n_V=40, V_mu0=2e-4, V_tau=4e-4, V_sp=6e-4, V_b=0.5e-4,          # in-band above c4
         n_W=710, W_spread=0.15, Wn_tau=1.5e-4, Wn_sp=2e-4, Wn_b=0.3e-4, # W zone between copy and c4 (near-copy tweaks + spread)
         c4pub=8.7e-4,
         below={0.957: 200, 0.956: 330, 0.955: 35, 0.954: 145}, below_s=1.0e-3, below_b=0.5e-4,
         # fraction of each >= .959 band that are heliosli-family derivatives like us (GUESS): they shrink like us instead of keeping m
         q={0.959: 0.6, 0.960: 0.4, 0.961: 0.25}, q_hi=0.1, D_mu0=3e-4, D_tau=4e-4, D_sp=7e-4,
         bonus=0.7e-4)            # max-of-2-picks bonus for non-copy field teams with distinct picks (GUESS)
if FS == "T1keep": P["q"] = {}; P["q_hi"] = 0.0
if FS == "T1deriv": P["q"] = {0.959: 0.8, 0.960: 0.6, 0.961: 0.4}; P["q_hi"] = 0.2
if FS == "noBonus": P["bonus"] = 0.0
if FS == "T1tight": P["t1_s"] = 0.8e-3
if FS == "T1loose": P["t1_s"] = 1.6e-3
if FS == "Wbig": P["n_W"] = 850; P["W_spread"] = 0.25
if FS == "Wnoisy": P["Wn_sp"] = 3e-4; P["V_sp"] = 8e-4
if FS == "belowLoose": P["below_s"] = 1.3e-3
if FS == "fieldGood": P["V_mu0"] = 4e-4; P["t1_b"] = 0.5e-4      # field additions are as real as ours
if FS == "fieldBad": P["V_mu0"] = 0.0; P["t1_b"] = 3e-4           # field public scores are probe-inflated
if FS == "N6400": P["N"] = 6400
silver = P["N"] // 20; bronze = P["N"] // 10; gold = 10 + P["N"] // 500
def shrink(m, mu0, tau, sp):
    sv = 0.65 * sp; lam = tau ** 2 / (tau ** 2 + sp ** 2)
    return mu0 + lam * (m - mu0), np.sqrt(lam * sp ** 2 + sv ** 2)
# fixed public positions of the field (relative to copy)
bands = []; m1 = []
for b, n in P["t1_bands"].items():
    mm = P["f959"] + (b - 0.959) * 1e3 * 1e-3 + rng.uniform(0, 1e-3, n); m1.append(mm); bands.append(np.full(n, b))
m1 = np.concatenate(m1); bands = np.concatenate(bands)
qv = np.array([P["q"].get(round(b, 3), P["q_hi"]) for b in bands]); isD = rng.uniform(0, 1, m1.size) < qv
mV = rng.uniform(P["c4pub"], P["f959"], P["n_V"])
nWs = int(P["n_W"] * P["W_spread"]); mWs = rng.uniform(0, P["c4pub"], nWs); mWn = rng.uniform(0, 3e-4, P["n_W"] - nWs)
mB = np.concatenate([-(0.75e-4 + (0.958 - b) * 1e3 * 1e-3 - 1e-3) - rng.uniform(0, 1e-3, n) for b, n in P["below"].items()])
muD, sD = shrink(m1 - P["V_b"], P["D_mu0"], P["D_tau"], P["D_sp"])
mu1 = np.where(isD, muD + P["bonus"], m1 - P["t1_b"]); s1 = np.where(isD, sD, P["t1_s"])
muV, sV = shrink(mV - P["V_b"], P["V_mu0"], P["V_tau"], P["V_sp"]); muWs, sWs = shrink(mWs - P["V_b"], P["V_mu0"], P["V_tau"], P["V_sp"])
muWn, sWn = shrink(mWn - P["Wn_b"], 0.0, P["Wn_tau"], P["Wn_sp"])
MU = np.concatenate([mu1, muV + P["bonus"], muWs + P["bonus"], muWn, mB - P["below_b"]])
SD = np.concatenate([s1, np.full(mV.size, sV), np.full(mWs.size, sWs), np.full(mWn.size, sWn), np.full(mB.size, P["below_s"])])
OTHER = np.concatenate([(~isD).astype(float), np.zeros(mV.size + mWs.size + mWn.size), np.ones(mB.size)])   # gets the other-lineage shock
nF = MU.size
def ranks(x, chunk=2000):
    # x: (NS,) our private margin over copy -> rank per sim (fresh field draw each sim)
    out = np.empty(x.size, int)
    for i in range(0, x.size, chunk):
        xs = x[i:i + chunk]; n = xs.size
        y = MU + SD * rng.standard_normal((n, nF)) + OTHER * (P["lin_sd"] * rng.standard_normal((n, 1)))
        out[i:i + chunk] = 1 + (y > xs[:, None]).sum(1) + np.where(xs < 0, P["n_copy"], 0)
    return out
idx = rng.choice(PV.shape[0], NS, replace=False); e = {k: i for i, k in enumerate(ENT)}
X = PV[idx]
# common random numbers: same field draws for every pick option -> draw field once, evaluate all options
y_all = []
res = {"SC": SC, "FS": FS, "params": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in P.items()},
       "silver_r": silver, "bronze_r": bronze, "field_size": int(nF)}
OPTS = [("c4", None), ("c4", "R6u"), ("c4", "HW"), ("c4", "a25"), ("c4", "fr100"), ("c4", "pf10"), ("c4", "S6W"), ("c4", "s615"), ("c4", "hw3"), ("c4", "a15"),
        ("c4", "c4a15"), ("c4", "fr65"), ("c4", "pf50"), ("c4", "wr45"), ("HW", "R6u"), ("a25", "R6u"), ("R6u", None), ("HW", None), ("c4", "c4noS6"), ("c4noS6", None), ("c4", "S6F")]
chunk = 1000; R = {o: [] for o in OPTS}
for i in range(0, NS, chunk):
    xs = X[i:i + chunk]; n = xs.shape[0]
    y = MU + SD * rng.standard_normal((n, nF)) + OTHER * (P["lin_sd"] * rng.standard_normal((n, 1)))
    ys = np.sort(y, axis=1)
    for a, b in OPTS:
        xa = xs[:, e[a]] if b is None else np.maximum(xs[:, e[a]], xs[:, e[b]])
        above = nF - np.array([np.searchsorted(ys[k], xa[k], side="right") for k in range(n)])
        R[(a, b)].append(1 + above + np.where(xa < 0, P["n_copy"], 0))
res["opts"] = {}
for o in OPTS:
    r = np.concatenate(R[o]); key = o[0] + ("+" + o[1] if o[1] else "")
    res["opts"][key] = dict(silver=float((r <= silver).mean()), bronze_plus=float((r <= bronze).mean()), gold=float((r <= gold).mean()),
                            med=int(np.median(r)), p10=int(np.percentile(r, 10)), p90=int(np.percentile(r, 90)), E_rank_capped=float(np.minimum(r, 2000).mean()))
json.dump(res, open(f"field_{SC}_{FS}.json", "w"), indent=1)
print(f"== our {SC} | field {FS} | silver r{silver} bronze r{bronze} | field n {nF}")
for k, v in res["opts"].items():
    print(f"  {k:12s} silver {v['silver']:.3f} bronze+ {v['bronze_plus']:.3f} med r{v['med']} p10-p90 r{v['p10']}-{v['p90']}")
