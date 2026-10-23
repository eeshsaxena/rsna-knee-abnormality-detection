# nested_r6.py (10-19, Claude): nested_sel + direction_stability rerun with ours = R6 serve (real ours leg, gold_in_training 0).
# Blend = c4 kernel 57343435: rr = .75 pct(r384) + .25 pct(r224); b = w_c rr + (1 - w_c) pct(ours); final = .8 pct(b) + .2 pct(S6fall).
# One stream of repeated random splits of gold58 into A (train) / B (held-out). Per split, the per-label AUC curve over a .025 weight grid
# is computed on A and on B (ranks recomputed inside each half, pct before weighting = same FP order as screen_a25 / cv_shrink).
# Part 1 (nested_sel): pick w per label on A (.05 grid, ties -> nearest c4, then lower), score on B vs c4.
#   variants: free, shrink50, shrink25, down_only, up_only, free_min005, step rules (A chooses {c4-s, c4, c4+s}), dirpick_s (sign of A pick, fixed step s).
# Part 2 (direction_stability): every label, both directions, steps .1 / .25 / .4 from c4 (clipped to [0, 1]) + slate extras:
#   A-side direction agreement (free pick side, and local "A prefers this exact move"), held-out B gain of the fixed candidate, conditional held-out.
# Part 3: fixed slate tables (LEAKY: chosen on all 58) on the same B halves, incl. c3max.
# Usage: python nested_r6.py <g58 dir> <ours csv> <reps> <train_frac> <seed> <out.json>
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D, OURSF, NREP, TRF, SEED, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
W0 = np.array([C4.get(l, 0.5) for l in L]); A224, WS6 = 0.25, 0.2
SLATE = {"c3max": {"PF OA": 0.1, "Fracture": 1.0, "Lateral OA": 0.1}, "c2max": {"PF OA": 0.1, "Fracture": 1.0}, "fr100": {"Fracture": 1.0},
         "c2": {"PF OA": 0.1, "Fracture": 0.9}, "pf10": {"PF OA": 0.1}, "fr90": {"Fracture": 0.9}, "lat10": {"Lateral OA": 0.1},
         "frlat": {"Fracture": 1.0, "Lateral OA": 0.1}, "lat00": {"Lateral OA": 0.0}, "c3lat0": {"PF OA": 0.1, "Fracture": 1.0, "Lateral OA": 0.0}}
C3 = {"PF OA": 0.1, "Fracture": 1.0, "Lateral OA": 0.1}
SLATE.update({"syn55": {"Synovitis": 0.55}, "syn40": {"Synovitis": 0.4}, "acl25": {"ACL": 0.25}, "mm75": {"Medial Meniscus": 0.75},
              "c3syn55": {**C3, "Synovitis": 0.55}, "c3syn40": {**C3, "Synovitis": 0.4}, "c3acl25": {**C3, "ACL": 0.25},
              "c3s55a25": {**C3, "Synovitis": 0.55, "ACL": 0.25}, "c3mm75": {**C3, "Medial Meniscus": 0.75}, "c3pf00": {**C3, "PF OA": 0.0},
              "c3pf25": {**C3, "PF OA": 0.25}, "c3fr90": {**C3, "Fracture": 0.9},
              "free58": {"ACL": 0.25, "MCL": 0.45, "Medial Meniscus": 0.9, "Lateral Meniscus": 0.6, "Medial OA": 0.55, "Lateral OA": 0.05, "PF OA": 0.0,
                         "Effusion": 0.8, "Synovitis": 0.15, "Fracture": 1.0}})
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
od = pd.read_csv(OURSF, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
cols = L if all(l in od.columns for l in L) else ["p|" + l for l in L]
OU = od.loc[ids, cols].values.astype(float)
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert R384.shape == R224.shape == OU.shape == S6.shape == (n, 12) and np.isfinite(OU).all()
FG = np.round(np.arange(0, 1.0001, 0.025), 3)            # fine grid (evaluation)
PG = np.round(np.arange(0, 1.0001, 0.05), 2)             # pick grid (as nested_sel / cv_shrink)
PGI = np.array([int(np.argmin(np.abs(FG - w))) for w in PG])
gi = lambda w: int(np.argmin(np.abs(FG - w)))
I0 = np.array([gi(w) for w in W0])


def grid_auc(idx, Wm):
    m = len(idx)
    rr = (1 - A224) * (rankdata(R384[idx], axis=0) / m) + A224 * (rankdata(R224[idx], axis=0) / m)
    po = rankdata(OU[idx], axis=0) / m; ps = rankdata(S6[idx], axis=0) / m
    w = Wm[:, :, None]
    b = w * rr.T[:, None, :] + (1 - w) * po.T[:, None, :]
    f = (1 - WS6) * (rankdata(b, axis=2) / m) + WS6 * ps.T[:, None, :]
    r = rankdata(f, axis=2)
    y = Y[idx].T.astype(float); p = y.sum(1); q = m - p
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * y[:, None, :]).sum(2) - (p * (p + 1) / 2)[:, None]) / (p * q)[:, None]
    a[(p == 0) | (q == 0)] = np.nan
    return a


FGW = np.tile(FG, (12, 1))

# ---- sanity: vectorised scorer == slow per-label scorer (quick_r6_slate.final) on all 58 ----
pct = lambda v: rankdata(v) / len(v)
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def slow_macro(table):
    out = []
    for j, c in enumerate(L):
        rr = (1 - A224) * pct(R384[:, j]) + A224 * pct(R224[:, j]); w = table.get(c, 0.5)
        b = w * rr + (1 - w) * pct(OU[:, j]); out.append(auc((1 - WS6) * pct(b) + WS6 * pct(S6[:, j]), Y[:, j]))
    return np.array(out)
full = grid_auc(np.arange(n), FGW)
c4_full = full[np.arange(12), I0]
assert np.allclose(c4_full, slow_macro(C4), atol=1e-12), "scorer mismatch"
for k, t in SLATE.items():
    tw = np.array([{**C4, **t}.get(l, 0.5) for l in L])
    assert np.allclose(full[np.arange(12), [gi(w) for w in tw]], slow_macro({**C4, **t}), atol=1e-12)
print("full-58 c4 macro (ours R6) %.5f" % np.nanmean(c4_full))


def pick(aA, j, allowed):
    """allowed: bool mask over PG. argmax A-AUC over allowed pick-grid points; ties -> nearest c4, then lower."""
    cand = np.flatnonzero(allowed); v = aA[j, PGI[cand]]
    best = cand[v >= v.max() - 1e-12]
    k = best[np.argmin(np.abs(PG[best] - W0[j]) + 1e-9 * PG[best])]
    return PG[k]


STEPS = [0.1, 0.25, 0.4]
VARS = ["free", "shrink50", "shrink25", "down_only", "up_only", "free_min005"] + ["step%.2f" % s for s in STEPS] + ["dirpick%.2f" % s for s in STEPS]
# direction-stability candidates
CAND = []
for j, l in enumerate(L):
    for sgn in (-1, 1):
        for s in STEPS:
            w = float(np.clip(round(W0[j] + sgn * s, 3), 0, 1))
            if abs(w - W0[j]) > 1e-9 and (l, w) not in CAND: CAND.append((l, w))
for x in [("Fracture", 0.8), ("Effusion", 0.65), ("Contusion", 0.2), ("Baker's", 0.65), ("Lateral OA", 0.05), ("PF OA", 0.2), ("Fracture", 0.95)]:
    if x not in CAND: CAND.append(x)

rng = np.random.default_rng(SEED); ntr = int(round(n * TRF))
G = {v: [] for v in VARS}; PL = {v: np.zeros(12) for v in VARS}; PLP = {v: np.zeros(12) for v in VARS}; PLN = {v: np.zeros(12) for v in VARS}
PLPICK = {v: [] for v in VARS}
DS = {c: dict(dir_agree=0, pick_eq=0, loc_pos=0, loc_neg=0, held=[], held_if_dir=[], held_if_loc=[], conc=0, conc_n=0) for c in CAND}
GS = {k: [] for k in SLATE}; PLS = {k: np.zeros(12) for k in SLATE}
reps = tries = 0
while reps < NREP:
    tries += 1
    perm = rng.permutation(n); A_, B_ = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    ya, yb = Y[A_].sum(0), Y[B_].sum(0)
    if ((ya == 0) | (ya == len(A_)) | (yb == 0) | (yb == len(B_))).any(): continue
    reps += 1
    aA = grid_auc(A_, FGW); aB = grid_auc(B_, FGW)
    gainB = aB - aB[np.arange(12), I0][:, None]            # (12, FG) held-out per-label gain vs c4
    gainA = aA - aA[np.arange(12), I0][:, None]
    W = {v: W0.copy() for v in VARS}
    for j in range(12):
        allp = np.ones(len(PG), bool); w0 = W0[j]
        wf = pick(aA, j, allp); W["free"][j] = wf
        W["shrink50"][j] = round((wf + w0) / 2, 3); W["shrink25"][j] = round(w0 + 0.25 * (wf - w0), 4)
        W["down_only"][j] = pick(aA, j, PG <= w0 + 1e-9); W["up_only"][j] = pick(aA, j, PG >= w0 - 1e-9)
        W["free_min005"][j] = wf if gainA[j, gi(wf)] >= 0.005 else w0
        for s in STEPS:
            lo, hi = max(0.0, round(w0 - s, 3)), min(1.0, round(w0 + s, 3))
            gl, gh = (gainA[j, gi(lo)] if lo < w0 - 1e-9 else -1), (gainA[j, gi(hi)] if hi > w0 + 1e-9 else -1)
            ws = w0
            if max(gl, gh) > 1e-12 and abs(gl - gh) > 1e-12: ws = lo if gl > gh else hi
            W["step%.2f" % s][j] = ws
            W["dirpick%.2f" % s][j] = w0 if abs(wf - w0) < 1e-9 else (hi if wf > w0 else lo)
    for v in VARS:
        wv = W[v]; d = np.zeros(12)
        for j in range(12):
            k = gi(wv[j]); assert abs(FG[k] - wv[j]) < 1e-6 or v == "shrink25", (v, wv[j])
            if v == "shrink25" and abs(FG[k] - wv[j]) > 1e-6:      # .0125 points: interpolate-free exact evaluation
                d[j] = grid_auc(B_, np.full((12, 1), wv[j]))[j, 0] - aB[j, I0[j]]
            else:
                d[j] = gainB[j, k]
        G[v].append(d.mean()); PL[v] += d; PLP[v] += d > 1e-12; PLN[v] += d < -1e-12; PLPICK[v].append(wv.copy())
    for (l, w) in CAND:
        j = L.index(l); w0 = W0[j]; k = gi(w); st = DS[(l, w)]
        sd = np.sign(w - w0); wf = W["free"][j]
        same = (wf - w0) * sd > 1e-9; st["dir_agree"] += same; st["pick_eq"] += abs(wf - w0) < 1e-9
        la = gainA[j, k]; hb = gainB[j, k] / 12.0
        st["loc_pos"] += la > 1e-12; st["loc_neg"] += la < -1e-12; st["held"].append(hb)
        if same: st["held_if_dir"].append(hb)
        if la > 1e-12: st["held_if_loc"].append(hb)
        if abs(la) > 1e-12 and abs(hb) > 1e-12: st["conc_n"] += 1; st["conc"] += np.sign(la) == np.sign(hb)
    for k, t in SLATE.items():
        d = np.zeros(12)
        for l, w in t.items(): j = L.index(l); d[j] = gainB[j, gi(w)]
        GS[k].append(d.mean()); PLS[k] += d
    if reps % 500 == 0: print("rep", reps, flush=True)


def summ(x):
    x = np.array(x); return dict(mean=float(x.mean()), sd=float(x.std()), p_gt0=float((x > 1e-12).mean()), p_lt0=float((x < -1e-12).mean()))


res = dict(ours=OURSF, n=n, ntrain=ntr, reps=reps, tries=tries, seed=SEED, c4_full_per_label=dict(zip(L, np.round(c4_full, 5).tolist())),
           c4_full_macro=float(np.nanmean(c4_full)), nested={}, direction={}, slate={}, full_curve={})
print("\n=== Part 1 nested selection (train %d / test %d, %d reps): held-out macro gain vs c4 ===" % (ntr, n - ntr, reps))
for v in VARS:
    s = summ(G[v]); s["per_label"] = dict(zip(L, np.round(PL[v] / reps / 12, 6).tolist()))
    s["per_label_p_gt0"] = dict(zip(L, np.round(PLP[v] / reps, 3).tolist())); s["per_label_p_lt0"] = dict(zip(L, np.round(PLN[v] / reps, 3).tolist()))
    P = np.array(PLPICK[v]); s["moved_frac"] = dict(zip(L, np.round((np.abs(P - W0) > 1e-9).mean(0), 3).tolist()))
    s["mean_pick"] = dict(zip(L, np.round(P.mean(0), 3).tolist()))
    res["nested"][v] = s
    print("%-12s mean %+.5f sd %.5f P>0 %.3f P<0 %.3f" % (v, s["mean"], s["sd"], s["p_gt0"], s["p_lt0"]))
print("per-label macro contribution (free):", {l: round(x, 5) for l, x in res["nested"]["free"]["per_label"].items()})
print("\n=== Part 2 direction stability ===")
for (l, w), st in DS.items():
    h = np.array(st["held"])
    r = dict(label=l, w=w, c4=float(W0[L.index(l)]), dir=("up" if w > W0[L.index(l)] else "down"), step=round(abs(w - W0[L.index(l)]), 3),
             dir_agree=round(st["dir_agree"] / reps, 3), pick_eq_c4=round(st["pick_eq"] / reps, 3),
             loc_pos=round(st["loc_pos"] / reps, 3), loc_neg=round(st["loc_neg"] / reps, 3),
             held_mean=float(h.mean()), held_p_gt0=float((h > 1e-12).mean()), held_p_lt0=float((h < -1e-12).mean()),
             held_if_dir=float(np.mean(st["held_if_dir"])) if st["held_if_dir"] else None,
             held_if_loc=float(np.mean(st["held_if_loc"])) if st["held_if_loc"] else None,
             concord=round(st["conc"] / st["conc_n"], 3) if st["conc_n"] else None,
             full_gain=float((full[L.index(l), gi(w)] - c4_full[L.index(l)]) / 12))
    res["direction"]["%s %.3f" % (l, w)] = r
    print("%-16s %4s %.3f (c4 %.2f) dirA %.2f locA+ %.2f locA- %.2f | held %+.5f P>0 %.2f P<0 %.2f | if dirA %s | conc %s | full %+.5f" % (
        l, r["dir"], w, r["c4"], r["dir_agree"], r["loc_pos"], r["loc_neg"], r["held_mean"], r["held_p_gt0"], r["held_p_lt0"],
        "%+.5f" % r["held_if_dir"] if r["held_if_dir"] is not None else "  -  ", r["concord"], r["full_gain"]))
print("\n=== Part 3 fixed slate tables on B (LEAKY) ===")
for k in SLATE:
    s = summ(GS[k]); s["per_label"] = {l: round(float(PLS[k][L.index(l)] / reps / 12), 6) for l in SLATE[k]}
    tw = np.array([{**C4, **SLATE[k]}.get(l, 0.5) for l in L]); s["full_gain"] = float(np.nanmean(full[np.arange(12), [gi(w) for w in tw]] - c4_full))
    res["slate"][k] = s
    print("%-7s held %+.5f sd %.5f P>0 %.3f P<0 %.3f | full-58 %+.5f" % (k, s["mean"], s["sd"], s["p_gt0"], s["p_lt0"], s["full_gain"]))
res["paired"] = {}
for v in ("free", "down_only", "dirpick0.25", "step0.25", "shrink50"):
    for k in ("c3max", "c2max", "fr100", "pf10", "c3syn55"):
        d = np.array(G[v]) - np.array(GS[k]); res["paired"]["%s - %s" % (v, k)] = dict(mean=float(d.mean()), p_gt0=float((d > 0).mean()))
for a_, b_ in (("c3max", "c2max"), ("c3max", "c3lat0"), ("c3syn55", "c3max"), ("c3s55a25", "c3max"), ("c3max", "fr100"), ("c2max", "fr100"), ("c2max", "c2"), ("c3max", "pf10")):
    d = np.array(GS[a_]) - np.array(GS[b_]); res["paired"]["%s - %s" % (a_, b_)] = dict(mean=float(d.mean()), p_gt0=float((d > 1e-12).mean()), p_lt0=float((d < -1e-12).mean()))
print("paired:", {k: (round(v["mean"], 5), round(v["p_gt0"], 3)) for k, v in res["paired"].items()})
for j, l in enumerate(L):
    res["full_curve"][l] = {"%.3f" % FG[k]: round(float(full[j, k] - c4_full[j]), 5) for k in PGI}
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
