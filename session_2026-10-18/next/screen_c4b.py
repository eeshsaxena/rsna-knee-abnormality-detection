# screen_a25.py: gold58 screen of one-line changes on the FULL a25 recipe (kernel_v224r6hwa25j cells 14 + 18):
#   Raptor slot rr = (1 - A224) pct(r384) + A224 pct(r224);  stage 3 b = w_c rr + (1 - w_c) pct(ours), w_c = table.get(c, W_RAPTOR);
#   final = (1 - W_S6F) pct(b) + W_S6F pct(S6fall).  pct = average-tie percentile rank over the scored set (= pandas rank pct).
# Components on gold58 (n = 58): r384 = argv[2] (kernel serve k94 + mirror TTA on CPU fp32, or the crop320_k24 proxy);
# r224 = kernel serve (cell 11 verbatim, box V100, fp16); ours = R7 OOF (stand-in for R6, no R6 gold preds exist);
# S6fall = nartaa GPU serve preds (n42gpu). Raptor r384 / r224 were selected on gold58 (optimistic for Raptor-up moves).
# Bootstrap 1 ("fixed"): predictions fixed on all 58, studies resampled (multiplicity-weighted Mann-Whitney AUC, exact).
# Bootstrap 2 ("rerank", A17 style): ranks recomputed inside each resample. Resamples with a label lacking a class are skipped.
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D, R384F, B, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B2 = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(R384F) if R384F.endswith(".npy") else pd.read_csv(R384F, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert R384.shape == R224.shape == R7.shape == S6.shape == (n, 12)
pct = lambda v: rankdata(v) / len(v)


def final(idx=None, A=0.25, wr=0.5, table=HW, ws6=0.2):
    idx = np.arange(n) if idx is None else idx
    out = np.zeros((len(idx), 12))
    for j, c in enumerate(L):
        rr = pct(R384[idx, j])
        rr = (1 - A) * rr + A * pct(R224[idx, j])          # kernel: r224 present -> always mixed (A = 0 leaves rr unchanged)
        w = table.get(c, wr)
        b = w * rr + (1 - w) * pct(R7[idx, j])
        out[:, j] = (1 - ws6) * pct(b) + ws6 * pct(S6[idx, j])
    return out


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


mac = lambda P, idx=None: np.array([auc(P[:, j], Y[np.arange(n) if idx is None else idx, j]) for j in range(12)])
T = lambda **kw: dict(table={**HW, **kw})
C4 = {'PF OA': 0.5, "Baker's": 0.5, 'Lateral Meniscus': 0.4, 'Fracture': 0.65}
CFG = {
    "a35 (HW)": dict(A=0.35), "a15": dict(A=0.15), "a05": dict(A=0.05), "a00": dict(A=0.0),
    "fr65": T(Fracture=0.65), "fr80": T(Fracture=0.8),
    "c4": dict(table={**HW, **C4}), "c4a15": dict(table={**HW, **C4}, A=0.15), "c4a05": dict(table={**HW, **C4}, A=0.05),
    "pf50": T(**{'PF OA': 0.5}), "bk50": T(**{"Baker's": 0.5}), "lm40": T(**{'Lateral Meniscus': 0.4}),
    "acl40": T(ACL=0.4), "acl60": T(ACL=0.6), "mcl40": T(MCL=0.4), "mcl60": T(MCL=0.6),
    "mm40": T(**{'Medial Meniscus': 0.4}), "mm60": T(**{'Medial Meniscus': 0.6}), "lm60": T(**{'Lateral Meniscus': 0.6}),
    "eff40": T(Effusion=0.4), "eff60": T(Effusion=0.6), "con40": T(Contusion=0.4), "con60": T(Contusion=0.6),
    "moa40": T(**{'Medial OA': 0.4}), "moa60": T(**{'Medial OA': 0.6}), "lat10": T(**{'Lateral OA': 0.1}), "lat30": T(**{'Lateral OA': 0.3}),
    "pf70": T(**{'PF OA': 0.7}), "syn70": T(Synovitis=0.7), "syn90": T(Synovitis=0.9), "bk25": T(**{"Baker's": 0.25}),
    "fr35": T(Fracture=0.35), "wr45": dict(wr=0.45), "wr55": dict(wr=0.55),
    "s6f25": dict(ws6=0.25), "s6f15": dict(ws6=0.15), "s6f30": dict(ws6=0.3),
    "fr65a15": dict(table={**HW, 'Fracture': 0.65}, A=0.15), "fr80a15": dict(table={**HW, 'Fracture': 0.8}, A=0.15),
    "c4f80": dict(table={**HW, **C4, 'Fracture': 0.8}), "c4f80a15": dict(table={**HW, **C4, 'Fracture': 0.8}, A=0.15),
}
base_P = final(); base = mac(base_P)
res = {"r384_source": R384F, "n": n, "pos": dict(zip(L, Y.sum(0).tolist())), "base_macro": float(base.mean()),
       "base_per_label": dict(zip(L, base.round(5).tolist()))}
comp = {"r384": R384, "r224": R224, "ours(R7 OOF)": R7, "S6fall": S6}
res["components"] = {k: dict(macro=float(mac(v).mean()), per_label=dict(zip(L, mac(v).round(4).tolist()))) for k, v in comp.items()}
res["A224_curve"] = {str(a): float(mac(final(A=a)).mean()) for a in (0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1.0)}
res["W_S6F_curve"] = {str(w): float(mac(final(ws6=w)).mean()) for w in (0, .1, .15, .2, .25, .3, .4)}
# per-label sweeps on the a25 base (a label's AUC depends on its own column only): Raptor weight w_c and A224 (in-sample, optimistic)
WG = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0); AG = (0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0)
res["per_label_wc_sweep"] = {c: {str(w): float(mac(final(table={**HW, c: w}))[j]) for w in WG} for j, c in enumerate(L)}
res["per_label_A224_sweep"] = {c: {str(a): float(mac(final(A=a))[j]) for a in AG} for j, c in enumerate(L)}
for j, c in enumerate(L):
    sw, sa = res["per_label_wc_sweep"][c], res["per_label_A224_sweep"][c]
    print("SWEEP %-16s w_c now %.2f | AUC by w_c %s | by A224 %s" % (c, HW.get(c, 0.5), " ".join("%s:%.3f" % (w, v) for w, v in sw.items()),
                                                                    " ".join("%s:%.3f" % (a, v) for a, v in sa.items())))
# ---- bootstrap 1 (fixed predictions): pairwise comparison matrices, multiplicity weights -----------------------------------------
rng = np.random.default_rng(20261018)
Wm = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)   # (B, n)
ok = np.ones(B, bool)
for j in range(12):
    pw = Wm[:, Y[:, j] == 1].sum(1); nw = Wm[:, Y[:, j] == 0].sum(1); ok &= (pw > 0) & (nw > 0)
Wm = Wm[ok]


def boot_auc(P):
    A = np.zeros((len(Wm), 12))
    for j in range(12):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        sp, sn = P[pos, j][:, None], P[neg, j][None, :]
        Cm = (sp > sn) + 0.5 * (sp == sn)
        wp, wn = Wm[:, pos], Wm[:, neg]
        A[:, j] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return A


bb = boot_auc(base_P)
# ---- bootstrap 2 (rerank inside each resample) ---------------------------------------------------------------------------------
rng2 = np.random.default_rng(1); idxs = []
while len(idxs) < B2:
    idx = rng2.integers(0, n, n)
    if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == n).any(): continue
    idxs.append(idx)
bb2 = np.array([mac(final(idx), idx) for idx in idxs])
res["n_boot_fixed"], res["n_boot_rerank"] = int(len(Wm)), len(idxs)
res["cfg"] = {}
for k, kw in CFG.items():
    P = final(**kw); pt = mac(P) - base
    d = (boot_auc(P) - bb).mean(1)
    d2 = np.array([(mac(final(idx, **kw), idx) - b2).mean() for idx, b2 in zip(idxs, bb2)])
    top = np.argsort(-np.abs(pt))[:3]
    res["cfg"][k] = dict(kw={kk: (vv if kk != 'table' else {c: vv[c] for c in vv if HW.get(c, 0.5) != vv[c]}) for kk, vv in kw.items()},
                         point=float(pt.mean()), sd=float(d.std()), p_gt0=float((d > 0).mean()), p_eq0=float((d == 0).mean()),
                         p_gt0_rerank=float((d2 > 0).mean()), sd_rerank=float(d2.std()),
                         per_label=dict(zip(L, pt.round(5).tolist())), drivers=[(L[j], round(float(pt[j]), 4)) for j in top if pt[j] != 0])
    print("%-9s point %+.5f | sd %.5f | P>0 %.3f (rerank %.3f) | P=0 %.2f | %s" % (
        k, pt.mean(), d.std(), (d > 0).mean(), (d2 > 0).mean(), (d == 0).mean(),
        ", ".join("%s %+.4f" % (L[j], pt[j]) for j in top if pt[j] != 0)), flush=True)
# ---- calibration: the same gold58 proxy applied to steps whose LB direction is KNOWN (R1 reads) ------------------------------------
CAL = [("V224: r224 at .35 into the uniform no-S6 recipe", dict(A=0.0, table={}, ws6=0.0), dict(A=0.35, table={}, ws6=0.0), "positive (V224 lift)"),
       ("S6fall .2 added (uniform, A .35)", dict(A=0.35, table={}, ws6=0.0), dict(A=0.35, table={}, ws6=0.2), "positive (S6F entry)"),
       ("HW table (uniform -> HW, A .35)", dict(A=0.35, table={}), dict(A=0.35), "positive (HW best-id move)"),
       ("hw3: S6fall .2 -> .3 (HW, A .35)", dict(A=0.35), dict(A=0.35, ws6=0.3), "NEGATIVE"),
       ("s615: S6fall .2 -> .15 (HW, A .35)", dict(A=0.35), dict(A=0.35, ws6=0.15), "NEGATIVE"),
       ("wr45: W_RAPTOR .5 -> .45 (HW, A .35)", dict(A=0.35), dict(A=0.35, wr=0.45), "NEGATIVE"),
       ("a25: A224 .35 -> .25 (HW)", dict(A=0.35), dict(A=0.25), "positive (a25 best-id move)")]
res["calibration"] = []
for nm, k0, k1, lb in CAL:
    P0, P1 = final(**k0), final(**k1); pt = mac(P1) - mac(P0); d = (boot_auc(P1) - boot_auc(P0)).mean(1)
    res["calibration"].append(dict(step=nm, lb=lb, point=float(pt.mean()), p_gt0=float((d > 0).mean())))
    print("CAL %-50s gold %+.5f P>0 %.3f | LB %s" % (nm, pt.mean(), (d > 0).mean(), lb), flush=True)
# ---- conditional pairs (next step given a positive read tonight): config B vs config A, both on the full recipe ---------------------
C4B = {**HW, **C4}
CFG.update({
 "c4pf10": dict(table={**C4B, 'PF OA': 0.1}), "c4pf30": dict(table={**C4B, 'PF OA': 0.3}),
 "c4ct20": dict(table={**C4B, 'Contusion': 0.2}), "c4ct30": dict(table={**C4B, 'Contusion': 0.3}),
 "c4ef65": dict(table={**C4B, 'Effusion': 0.65}), "c4ef60": dict(table={**C4B, 'Effusion': 0.6}),
 "c4bk65": dict(table={**C4B, "Baker's": 0.65}), "c4lat00": dict(table={**C4B, 'Lateral OA': 0.0}),
 "c4mm60": dict(table={**C4B, 'Medial Meniscus': 0.6}), "c4syn60": dict(table={**C4B, 'Synovitis': 0.6}),
 "c5": dict(table={**C4B, 'PF OA': 0.1, 'Contusion': 0.2, 'Effusion': 0.65, 'Fracture': 0.8}),
 "c5down": dict(table={**C4B, 'PF OA': 0.1, 'Contusion': 0.2, 'Lateral OA': 0.0}),
 "c4pfct": dict(table={**C4B, 'PF OA': 0.1, 'Contusion': 0.2}),
 "c4lat10": dict(table={**C4B, 'Lateral OA': 0.1}), "c3d": dict(table={**C4B, 'PF OA': 0.1, 'Lateral OA': 0.1}),
 "c6": dict(table={**C4B, 'PF OA': 0.1, 'Lateral OA': 0.1, 'Fracture': 0.8}), "c4fr100": dict(table={**C4B, 'Fracture': 1.0}), "c4fr90": dict(table={**C4B, 'Fracture': 0.9}),
 "c6f90": dict(table={**C4B, 'PF OA': 0.1, 'Lateral OA': 0.1, 'Fracture': 0.9}),
})
PAIRS = [("c4", x) for x in ["c4f80","c4fr90","c4fr100","c4pf10","c4lat10","c3d","c6","c6f90"]]
res["pairs"] = []
for a_, b_ in PAIRS:
    P0, P1 = final(**CFG[a_]), final(**CFG[b_]); pt = mac(P1) - mac(P0); d = (boot_auc(P1) - boot_auc(P0)).mean(1)
    top = np.argsort(-np.abs(pt))[:3]
    res["pairs"].append(dict(a=a_, b=b_, point=float(pt.mean()), sd=float(d.std()), p_gt0=float((d > 0).mean()),
                             drivers=[(L[j], round(float(pt[j]), 4)) for j in top if pt[j] != 0]))
    print("PAIR %-6s -> %-9s %+.5f sd %.5f P>0 %.3f | %s" % (a_, b_, pt.mean(), d.std(), (d > 0).mean(),
          ", ".join("%s %+.4f" % (L[j], pt[j]) for j in top if pt[j] != 0)), flush=True)
print("base a25 macro %.5f | components %s" % (base.mean(), {k: round(v['macro'], 4) for k, v in res['components'].items()}))
print("A224 curve", {k: round(v, 5) for k, v in res["A224_curve"].items()})
print("W_S6F curve", {k: round(v, 5) for k, v in res["W_S6F_curve"].items()})
json.dump(res, open(OUT, "w"), indent=1)
