# a224_perlabel.py (10-18, Claude): per-label A224 on the c4 recipe (team best 57343435), gold58, light CPU only.
# Kernel blend per label c:  rr = (1 - A_c) pct(r384) + A_c pct(r224);  b = w_c rr + (1 - w_c) pct(ours);  final = .8 pct(b) + .2 pct(S6fall).
# c4 w_c table; A_c = .25 everywhere is the baseline. Grid A_c in {0, .1, .15, .25, .35, .5}.
# Parts: (1) full-data per-label sweep (+ fixed and rerank bootstraps vs .25); (2) nested split-half selection, held-out gain vs .25;
# (3) direction agreement (half A vs full data, half A vs half B); (4) LB calibration (HW .35->.25 positive, c4 .25->.15 negative).
# usage: python a224_perlabel.py <g58 dir> <ours csv> <NREP> <out.json>
import sys, json, time, numpy as np, pandas as pd
from scipy.stats import rankdata
D, OURS, NREP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B, B2 = 4000, 1000
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
HW = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
WC = np.array([C4.get(l, 0.5) for l in L]); WHW = np.array([HW.get(l, 0.5) for l in L])
AG = [0.0, 0.1, 0.15, 0.25, 0.35, 0.5]; A0 = 0.25; I0 = AG.index(A0)
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids


def load_csv(f):
    d = pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    cols = ["p|" + l for l in L] if "p|ACL" in d.columns else L
    return d.loc[ids, cols].values.astype(float)


OUR = load_csv(OURS)
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = load_csv(D + "/n42gpu_gold58.csv")
assert R384.shape == R224.shape == OUR.shape == S6.shape == (n, 12) and np.isfinite(OUR).all()
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    if p == 0 or q == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def legs(idx, j):
    return pct(R384[idx, j]), pct(R224[idx, j]), pct(OUR[idx, j]), pct(S6[idx, j])


def score_from(lg, a, w, ws6=0.2):
    p384, p224, po, ps = lg
    b = w * ((1 - a) * p384 + a * p224) + (1 - w) * po
    return (1 - ws6) * pct(b) + ws6 * ps


ALL = np.arange(n)
LG = [legs(ALL, j) for j in range(12)]
t0 = time.time()
res = dict(ours=OURS, n=n, grid=AG, pos=dict(zip(L, Y.sum(0).tolist())))
# ---------------- (1) full-data sweep ---------------------------------------------------------------------------------------------
FULL = np.array([[auc(score_from(LG[j], a, WC[j]), Y[:, j]) for a in AG] for j in range(12)])   # (12, 6) label AUC
res["component_macro"] = {k: float(np.nanmean([auc(v[:, j], Y[:, j]) for j in range(12)])) for k, v in
                          dict(r384=R384, r224=R224, ours=OUR, S6fall=S6).items()}
res["component_per_label"] = {k: {l: round(float(auc(v[:, j], Y[:, j])), 4) for j, l in enumerate(L)} for k, v in
                              dict(r384=R384, r224=R224, ours=OUR, S6fall=S6).items()}
res["c4_macro"] = float(FULL[:, I0].mean())
res["full_label_auc"] = {l: {str(a): round(float(FULL[j, i]), 5) for i, a in enumerate(AG)} for j, l in enumerate(L)}
res["full_macro_contrib_vs_25"] = {l: {str(a): float((FULL[j, i] - FULL[j, I0]) / 12) for i, a in enumerate(AG)} for j, l in enumerate(L)}
best_full = {l: AG[max(range(len(AG)), key=lambda i: (FULL[j, i], -abs(AG[i] - A0)))] for j, l in enumerate(L)}
res["full_best"] = best_full
res["full_best_table_point"] = float(np.mean([FULL[j, AG.index(best_full[l])] - FULL[j, I0] for j, l in enumerate(L)]))
GA = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
res["global_A_curve_c4"] = {str(a): float(np.mean([auc(score_from(LG[j], a, WC[j]), Y[:, j]) for j in range(12)])) for a in GA}
res["global_A_curve_hw"] = {str(a): float(np.mean([auc(score_from(LG[j], a, WHW[j]), Y[:, j]) for j in range(12)])) for a in GA}
FINE = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
res["fine_label_curve"] = {l: {str(a): round(float(auc(score_from(LG[j], a, WC[j]), Y[:, j])), 4) for a in FINE} for j, l in enumerate(L)}
# fixed-prediction bootstrap (multiplicity weights, exact pairwise AUC)
rng = np.random.default_rng(20261018)
Wm = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)
ok = np.ones(B, bool)
for j in range(12):
    ok &= (Wm[:, Y[:, j] == 1].sum(1) > 0) & (Wm[:, Y[:, j] == 0].sum(1) > 0)
Wm = Wm[ok]


def boot_label(s, j):
    pos, neg = Y[:, j] == 1, Y[:, j] == 0
    sp, sn = s[pos][:, None], s[neg][None, :]
    Cm = (sp > sn) + 0.5 * (sp == sn); wp, wn = Wm[:, pos], Wm[:, neg]
    return np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))


BOOT = np.stack([np.stack([boot_label(score_from(LG[j], a, WC[j]), j) for a in AG]) for j in range(12)])   # (12, 6, B)
# rerank bootstrap (ranks recomputed inside each resample)
rng2 = np.random.default_rng(1); idxs = []
while len(idxs) < B2:
    idx = rng2.integers(0, n, n)
    if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == n).any(): continue
    idxs.append(idx)
RR = np.zeros((12, len(AG), B2))
for b, idx in enumerate(idxs):
    for j in range(12):
        lg = legs(idx, j)
        for i, a in enumerate(AG): RR[j, i, b] = auc(score_from(lg, a, WC[j]), Y[idx, j])
res["boot"] = {}
for j, l in enumerate(L):
    res["boot"][l] = {}
    for i, a in enumerate(AG):
        if i == I0: continue
        d = BOOT[j, i] - BOOT[j, I0]; d2 = RR[j, i] - RR[j, I0]
        res["boot"][l][str(a)] = dict(point=float(FULL[j, i] - FULL[j, I0]), p_gt0=float((d > 0).mean()), p_lt0=float((d < 0).mean()),
                                      sd=float(d.std()), p_gt0_rerank=float((d2 > 0).mean()), p_lt0_rerank=float((d2 < 0).mean()))
print("full-data done %.1fs | c4 macro %.5f | n_boot %d" % (time.time() - t0, FULL[:, I0].mean(), len(Wm)), flush=True)
for j, l in enumerate(L):
    print("FULL %-16s pos %2d | %s | best %.2f | P>0 fixed/rerank of best: %s" % (
        l, Y[:, j].sum(), " ".join("%.2f:%.4f" % (a, FULL[j, i]) for i, a in enumerate(AG)), best_full[l],
        "-" if best_full[l] == A0 else "%.3f/%.3f" % (res["boot"][l][str(best_full[l])]["p_gt0"], res["boot"][l][str(best_full[l])]["p_gt0_rerank"])))
print("global A curve c4:", {k: round(v, 5) for k, v in res["global_A_curve_c4"].items()})
# ---------------- (4) LB calibration on this proxy --------------------------------------------------------------------------------
def boot_macro_step(a0, w0, a1, w1):
    pts, ds = [], []
    for j in range(12):
        s0, s1 = score_from(LG[j], a0, w0[j]), score_from(LG[j], a1, w1[j])
        pts.append(auc(s1, Y[:, j]) - auc(s0, Y[:, j])); ds.append(boot_label(s1, j) - boot_label(s0, j))
    d = np.mean(ds, 0); return float(np.mean(pts)), float((d > 0).mean()), dict(zip(L, np.round(pts, 4).tolist()))


CAL = [("HW table: A224 .35 -> .25 (a25 best-id move)", 0.35, WHW, 0.25, WHW, "POSITIVE"),
       ("c4 table: A224 .25 -> .15 (c4a15 vs c4)", 0.25, WC, 0.15, WC, "NEGATIVE (c4a15 read below c4)"),
       ("c4 table: A224 .25 -> .35 (not run on LB)", 0.25, WC, 0.35, WC, "unknown")]
res["calibration"] = []
for nm, a0, w0, a1, w1, lb in CAL:
    pt, p, pl = boot_macro_step(a0, w0, a1, w1)
    res["calibration"].append(dict(step=nm, lb=lb, point=pt, p_gt0=p, per_label=pl))
    print("CAL %-46s gold58 %+.5f P>0 %.3f | LB %s | per-label %s" % (nm, pt, p, lb, {k: v for k, v in pl.items() if v != 0}), flush=True)
# ---------------- named per-label candidates on c4 (point + fixed / rerank bootstrap vs c4) -------------------------------------------
NAMED = {"syn00": {"Synovitis": 0.0}, "syn10": {"Synovitis": 0.1}, "syn15": {"Synovitis": 0.15},
         "eff35": {"Effusion": 0.35}, "eff50": {"Effusion": 0.5}, "syn10eff35": {"Synovitis": 0.1, "Effusion": 0.35},
         "syn00eff35": {"Synovitis": 0.0, "Effusion": 0.35}, "pf15": {"PF OA": 0.15}, "pf00": {"PF OA": 0.0},
         "lm00": {"Lateral Meniscus": 0.0}, "fr50": {"Fracture": 0.5}, "mcl50": {"MCL": 0.5},
         "fullbest": {l: a for l, a in best_full.items() if a != A0}}
res["named"] = {}
for nm, tab in NAMED.items():
    pts, ds, d2s = [], [], []
    for j, l in enumerate(L):
        a = tab.get(l, A0)
        if a == A0: continue
        i = AG.index(a) if a in AG else None
        s1 = score_from(LG[j], a, WC[j])
        pts.append(auc(s1, Y[:, j]) - FULL[j, I0]); ds.append(boot_label(s1, j) - BOOT[j, I0])
        d2s.append(RR[j, i] - RR[j, I0] if i is not None else np.zeros(B2))
    pt, d, d2 = np.sum(pts) / 12, np.sum(ds, 0) / 12, np.sum(d2s, 0) / 12
    res["named"][nm] = dict(table=tab, point=float(pt), p_gt0=float((d > 0).mean()), p_lt0=float((d < 0).mean()), sd=float(d.std()),
                            p_gt0_rerank=float((d2 > 0).mean()), p_lt0_rerank=float((d2 < 0).mean()))
    print("NAMED %-11s point %+.5f | fixed P>0 %.3f P<0 %.3f | rerank P>0 %.3f P<0 %.3f | %s" % (
        nm, pt, (d > 0).mean(), (d < 0).mean(), (d2 > 0).mean(), (d2 < 0).mean(), tab), flush=True)
# ---------------- slate-conditional: A224 on Fracture / PF OA when their w_c moves (A19 slate: Fracture .9 / 1.0, PF OA .1) ------------
res["slate_conditional"] = {}
for l, ws in (("Fracture", (0.65, 0.9, 1.0)), ("PF OA", (0.5, 0.1))):
    j = L.index(l); res["slate_conditional"][l] = {}
    for w in ws:
        s0 = score_from(LG[j], A0, w); a0v = auc(s0, Y[:, j]); b0 = boot_label(s0, j); row = {}
        for a in AG:
            s1 = score_from(LG[j], a, w); d = boot_label(s1, j) - b0
            row[str(a)] = dict(auc=float(auc(s1, Y[:, j])), macro_vs_25=float((auc(s1, Y[:, j]) - a0v) / 12), p_gt0=float((d > 0).mean()),
                               p_lt0=float((d < 0).mean()))
        res["slate_conditional"][l][str(w)] = row
        print("SLATE %-9s w_c %.2f | %s" % (l, w, " ".join("%s:%.4f(P>0 %.2f)" % (a, v["auc"], v["p_gt0"]) for a, v in row.items())), flush=True)
# ---------------- (2)+(3) nested split-half --------------------------------------------------------------------------------------
rng3 = np.random.default_rng(0)
VAR = ["free", "shrink50", "min005", "down_only", "up_only", "global"]
held = {k: [] for k in VAR}; held_lab = {k: np.zeros(12) for k in VAR if k != "global"}
pickA = np.zeros((NREP, 12)); pickB = np.zeros((NREP, 12)); gpick = []
fixed_held = np.zeros((NREP, 12, len(AG)))     # held-out (half B) gain of each fixed value vs .25, per label AUC units
reps = tries = 0
while reps < NREP:
    tries += 1
    perm = rng3.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if any(Y[A_, j].sum() in (0, len(A_)) or Y[B_, j].sum() in (0, len(B_)) for j in range(12)): continue
    trA = np.zeros((12, len(AG))); teB = np.zeros((12, len(AG)))
    for j in range(12):
        lgA, lgB = legs(A_, j), legs(B_, j)
        for i, a in enumerate(AG):
            trA[j, i] = auc(score_from(lgA, a, WC[j]), Y[A_, j]); teB[j, i] = auc(score_from(lgB, a, WC[j]), Y[B_, j])
    gains = {k: np.zeros(12) for k in VAR if k != "global"}
    for j in range(12):
        key = lambda i, M=trA[j]: (M[i], -abs(AG[i] - A0))
        iA = max(range(len(AG)), key=key); iB = max(range(len(AG)), key=lambda i, M=teB[j]: (M[i], -abs(AG[i] - A0)))
        pickA[reps, j], pickB[reps, j] = AG[iA], AG[iB]
        iD = max([i for i in range(len(AG)) if AG[i] <= A0], key=key); iU = max([i for i in range(len(AG)) if AG[i] >= A0], key=key)
        te0 = teB[j, I0]
        gains["free"][j] = teB[j, iA] - te0
        ash = (AG[iA] + A0) / 2
        gains["shrink50"][j] = auc(score_from(legs(B_, j), ash, WC[j]), Y[B_, j]) - te0
        gains["min005"][j] = (teB[j, iA] - te0) if trA[j, iA] - trA[j, I0] >= 0.005 else 0.0
        gains["down_only"][j] = teB[j, iD] - te0; gains["up_only"][j] = teB[j, iU] - te0
        fixed_held[reps, j] = teB[j] - te0
    for k in gains:
        held[k].append(gains[k].mean()); held_lab[k] += gains[k]
    gi = max(range(len(AG)), key=lambda i: (trA[:, i].mean(), -abs(AG[i] - A0)))
    gpick.append(AG[gi]); held["global"].append(teB[:, gi].mean() - teB[:, I0].mean())
    reps += 1
res["nested"] = dict(reps=reps, accept_rate=reps / tries, variants={})
for k in VAR:
    v = np.array(held[k])
    res["nested"]["variants"][k] = dict(mean=float(v.mean()), sd=float(v.std()), se=float(v.std() / np.sqrt(len(v))),
                                        p_gt0=float((v > 0).mean()), p_lt0=float((v < 0).mean()),
                                        per_label=None if k == "global" else {l: round(float(held_lab[k][j] / reps / 12), 6) for j, l in enumerate(L)})
    print("NESTED %-9s held-out macro gain vs A .25: mean %+.5f (se %.5f) sd %.5f P>0 %.3f P<0 %.3f" % (
        k, v.mean(), v.std() / np.sqrt(len(v)), v.std(), (v > 0).mean(), (v < 0).mean()), flush=True)
res["nested"]["global_pick_freq"] = {str(a): float(np.mean(np.array(gpick) == a)) for a in AG}
print("global pick freq", res["nested"]["global_pick_freq"])
# per-label: direction agreement + fixed-value held-out
res["direction"] = {}
for j, l in enumerate(L):
    fdir = np.sign(best_full[l] - A0)
    dA, dB = np.sign(pickA[:, j] - A0), np.sign(pickB[:, j] - A0)
    both = (dA != 0) & (dB != 0)
    fh = fixed_held[:, j] / 12
    entry = dict(full_best=best_full[l], full_dir=int(fdir),
                 halfA_pick_freq={str(a): float(np.mean(pickA[:, j] == a)) for a in AG},
                 halfA_down=float((dA < 0).mean()), halfA_stay=float((dA == 0).mean()), halfA_up=float((dA > 0).mean()),
                 halfA_agrees_full=float((dA == fdir).mean()),
                 halves_agree_given_both_move=float((dA[both] == dB[both]).mean()) if both.any() else None,
                 halves_opposite=float(((dA * dB) < 0).mean()),
                 fixed_heldout_macro={str(a): float(fh[:, i].mean()) for i, a in enumerate(AG)},
                 fixed_heldout_p_gt0={str(a): float((fh[:, i] > 0).mean()) for i, a in enumerate(AG)},
                 free_pick_heldout_macro=res["nested"]["variants"]["free"]["per_label"][l])
    res["direction"][l] = entry
    print("DIR %-16s full best %.2f | half-A down/stay/up %.2f/%.2f/%.2f | A agrees full %.2f | halves opposite %.2f | agree|both move %s | free-pick held %+.5f" % (
        l, best_full[l], entry["halfA_down"], entry["halfA_stay"], entry["halfA_up"], entry["halfA_agrees_full"], entry["halves_opposite"],
        "-" if entry["halves_agree_given_both_move"] is None else "%.2f" % entry["halves_agree_given_both_move"], entry["free_pick_heldout_macro"]))
for nm, tab in NAMED.items():
    h = np.zeros(reps)
    for l, a in tab.items():
        if a != A0: h += fixed_held[:, L.index(l), AG.index(a)] / 12
    res["named"][nm].update(split_half_held_mean=float(h.mean()), split_half_held_p_gt0=float((h > 0).mean()),
                            split_half_held_p_lt0=float((h < 0).mean()))
    print("NAMED-HELD %-11s split-half held-out (fixed value, NOT selection-honest) %+.5f P>0 %.3f P<0 %.3f" % (
        nm, h.mean(), (h > 0).mean(), (h < 0).mean()))
res["elapsed_s"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
print("done %.1fs" % (time.time() - t0))
