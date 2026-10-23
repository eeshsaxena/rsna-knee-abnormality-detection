# cv_shrink_r6.py (10-19, Claude; copy of 10-18 cv_shrink.py, ours = R6 serve; adds c3max / lat10 / frlat slate rows and LOA masks): cross-validated SHRUNK per-label Raptor-weight table around c4.
# Blend = c4 kernel 57343435: rr = .75 pct(r384) + .25 pct(r224); b = w_c rr + (1 - w_c) pct(ours); final = .8 pct(b) + .2 pct(S6fall).
# Repeated random splits of gold58 into train half A / held-out half B:
#   on A pick per label w_pick = argmax AUC over GRID (ties -> nearest c4); w = c4 + lambda (w_pick - c4) on labels in subset S;
#   score the table on B vs c4 (ranks recomputed inside B). Macro gain = mean over 12 labels (unchanged labels contribute 0).
# Subset rules S: ALL; NEST_DIRt = labels whose direction agreement INSIDE A (K half-subsamples of A, same side of c4 as A's pick)
#   >= t (honest analogue of "dir agree >= .75"); TOPk = k labels with the largest in-sample gain on A (honest analogue of how the
#   A19 slate was built); GAIN005 = in-sample gain >= .005; FULL_DIR75 / FIX_FR / FIX_FRPF = subsets chosen on all 58 (LEAKY).
# Fixed A19 slate tables are also scored on the same B halves (LEAKY: chosen on all 58 incl. B).
# Full-data table: the same procedure run on all 58 (direction agreement from half-splits of 58).
# Light CPU only (numpy/scipy on 58 studies).
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
D, OURSF, NREP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
TRF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
KIN = int(sys.argv[6]) if len(sys.argv) > 6 else 40
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 20261018
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
HW = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
KB = int(sys.argv[9]) if len(sys.argv) > 9 else 0      # >0: also bagged picks from KB bootstrap resamples of the training half (separate RNG)
BASE = sys.argv[8] if len(sys.argv) > 8 else "c4"     # "c4" (main) or "hw" (calibration: was the LB-positive HW -> c4 move CV-visible?)
BT = {"c4": C4, "hw": HW}[BASE]
W0 = np.array([BT.get(l, 0.5) for l in L])
A224, WS6 = 0.25, 0.2
SLATE = {  # A19 tables (from a19/build_receipt_*.json), as overrides on c4
    "c2max": {"PF OA": 0.1, "Fracture": 1.0}, "fr100": {"Fracture": 1.0}, "fr90": {"Fracture": 0.9},
    "c2": {"PF OA": 0.1, "Fracture": 0.9}, "pf10": {"PF OA": 0.1}, "fr80": {"Fracture": 0.8},
    "c5": {"PF OA": 0.1, "Fracture": 0.8, "Contusion": 0.2, "Effusion": 0.65}, "c4pfct": {"PF OA": 0.1, "Contusion": 0.2},
    "ef65": {"Effusion": 0.65},
    "c3max": {"PF OA": 0.1, "Fracture": 1.0, "Lateral OA": 0.1}, "lat10": {"Lateral OA": 0.1}, "frlat": {"Fracture": 1.0, "Lateral OA": 0.1}}
if BASE == "hw": SLATE = {"c4": {}, **SLATE}
SLW = {k: np.array([{**C4, **v}.get(l, 0.5) for l in L]) for k, v in SLATE.items()}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
od = pd.read_csv(OURSF, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
cols = L if all(l in od.columns for l in L) else ["p|" + l for l in L]
assert all(c in od.columns for c in cols), ("ours columns", list(od.columns)[:20])
missing = [i for i in ids if i not in od.index]; assert not missing, ("ours missing ids", len(missing))
OU = od.loc[ids, cols].values.astype(float)
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(D + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
assert R384.shape == R224.shape == OU.shape == S6.shape == (n, 12) and np.isfinite(OU).all()
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
LAMS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
MAIN_LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def grid_auc(idx, Wm):
    """AUC per label for each candidate weight. idx: study indices; Wm: (12, W) weights. Returns (12, W)."""
    m = len(idx)
    rr = (1 - A224) * (rankdata(R384[idx], axis=0) / m) + A224 * (rankdata(R224[idx], axis=0) / m)   # pct first (FP order = screen_a25)
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


GW = np.tile(GRID, (12, 1)); I0 = np.array([int(np.argmin(np.abs(GRID - w0))) for w0 in W0])


def picks(idx):
    """Per-label grid argmax (ties -> nearest c4) and in-sample gain over c4. NaN-safe (label lacking a class -> c4, gain 0)."""
    a = grid_auc(idx, GW); pk = W0.copy(); gn = np.zeros(12)
    for j in range(12):
        if np.isnan(a[j]).any(): continue
        best = np.flatnonzero(a[j] >= a[j].max() - 1e-12)
        k = best[np.argmin(np.abs(GRID[best] - W0[j]) + 1e-9 * GRID[best])]
        pk[j] = GRID[k]; gn[j] = a[j, k] - a[j, I0[j]]
    return pk, gn


def bag_picks(idx, K, rb):
    """Bagged picks on idx: mean of K bootstrap-resample picks (BAGMEAN) and argmax of the mean bootstrap AUC curve (BAGCURVE)."""
    m = len(idx); cur = np.zeros((12, len(GRID))); cn = np.zeros(12); sp = np.zeros(12)
    for _ in range(K):
        bi = np.sort(rb.choice(idx, m, replace=True)); a = grid_auc(bi, GW); ok = ~np.isnan(a).any(1)
        cur[ok] += a[ok]; cn += ok
        for j in np.flatnonzero(ok):
            best = np.flatnonzero(a[j] >= a[j].max() - 1e-12); sp[j] += GRID[best[np.argmin(np.abs(GRID[best] - W0[j]) + 1e-9 * GRID[best])]]
    bm = np.where(cn > 0, sp / np.maximum(cn, 1), W0); bc = W0.copy()
    for j in range(12):
        if cn[j] == 0: continue
        c = cur[j] / cn[j]; best = np.flatnonzero(c >= c.max() - 1e-12); bc[j] = GRID[best[np.argmin(np.abs(GRID[best] - W0[j]) + 1e-9 * GRID[best])]]
    return bm, bc


def dir_agree(idx, pk, K, rng):
    """Fraction of K half-subsamples of idx whose pick lies on the same side of c4 as pk (NaN where pk == c4)."""
    m = len(idx); h = m // 2; same = np.zeros(12); valid = np.zeros(12)
    for _ in range(K):
        sub = np.sort(rng.choice(idx, h, replace=False))
        ys = Y[sub].sum(0); okl = (ys > 0) & (ys < h)
        ps, _ = picks(sub)
        same += okl * ((ps - W0) * (pk - W0) > 0); valid += okl
    with np.errstate(invalid="ignore", divide="ignore"):
        ag = same / valid
    ag[pk == W0] = np.nan
    return ag


# ---------------- full-data quantities (the table we would actually ship) ----------------
rng_full = np.random.default_rng(7)
pk_full, gn_full = picks(np.arange(n))
ag_full = dir_agree(np.arange(n), pk_full, 400, rng_full)
auc_c4_full = grid_auc(np.arange(n), W0[:, None])[:, 0]
HWv = np.array([HW.get(l, 0.5) for l in L])
print("full-data base(" + BASE + ") macro %.5f | HW macro %.5f (screen_c4b HW base 0.93573 when ours = R7)" % (
    np.nanmean(auc_c4_full), np.nanmean(grid_auc(np.arange(n), HWv[:, None])[:, 0])))
for j, l in enumerate(L):
    print("FULL %-16s base %.2f pick %.2f in-sample gain %+.4f dir-agree(29 of 58) %s" % (
        l, W0[j], pk_full[j], gn_full[j], "  -" if np.isnan(ag_full[j]) else "%.2f" % ag_full[j]))
FULL_DIR75 = np.nan_to_num(ag_full) >= 0.75
FIX_FR = np.array([l == "Fracture" for l in L]); FIX_FRPF = np.array([l in ("Fracture", "PF OA") for l in L])
FIX_PF = np.array([l == "PF OA" for l in L])
FIX_LOA = np.array([l == "Lateral OA" for l in L]); FIX_FRLOA = np.array([l in ("Fracture", "Lateral OA") for l in L])
FIX_FRPFLOA = np.array([l in ("Fracture", "PF OA", "Lateral OA") for l in L])


def masks(pk, gn, ag):
    order = np.argsort(-gn); M = {"ALL": np.ones(12, bool)}
    for t in (0.6, 0.75, 0.9): M["NEST_DIR%02d" % int(round(t * 100))] = np.nan_to_num(ag) >= t
    for k in (1, 2, 3):
        mk = np.zeros(12, bool); mk[order[:k]] = True; M["TOP%d" % k] = mk & (gn > 0)
    M["GAIN005"] = gn >= 0.005
    M["FULL_DIR75(leaky)"] = FULL_DIR75; M["FIX_FR(leaky)"] = FIX_FR; M["FIX_FRPF(leaky)"] = FIX_FRPF; M["FIX_PF(leaky)"] = FIX_PF
    M["FIX_LOA(leaky)"] = FIX_LOA; M["FIX_FRLOA(leaky)"] = FIX_FRLOA; M["FIX_FRPFLOA(leaky)"] = FIX_FRPFLOA
    return M


VAR = list(masks(pk_full, gn_full, ag_full).keys()) + (["BAGMEAN_ALL", "BAGCURVE_ALL"] if KB else [])
rb = np.random.default_rng(SEED + 99)
# ---------------- repeated split CV ----------------
rng = np.random.default_rng(SEED); ntr = int(round(n * TRF))
G = {v: {lam: [] for lam in LAMS} for v in VAR}; PL = {v: {lam: np.zeros(12) for lam in LAMS} for v in VAR}
PLP = {lam: np.zeros(12) for lam in LAMS}; PLN = {lam: np.zeros(12) for lam in LAMS}   # per-label P>0 / P<0 of the single-label move (ALL rule)
GS = {k: [] for k in SLATE}; PLS = {k: np.zeros(12) for k in SLATE}
pick_log = []; agree_log = []; msize = {v: [] for v in VAR}; top1_lab = []; top1_g = []
reps = 0; tries = 0
while reps < NREP:
    tries += 1
    perm = rng.permutation(n); A_, B_ = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    ya, yb = Y[A_].sum(0), Y[B_].sum(0)
    if ((ya == 0) | (ya == len(A_)) | (yb == 0) | (yb == len(B_))).any(): continue
    reps += 1
    pk, gn = picks(A_); ag = dir_agree(A_, pk, KIN, rng)
    pick_log.append(pk); agree_log.append(ag)
    # held-out AUCs on B: c4, each lambda-shrunk pick, and each slate table, in one vectorised call
    if KB: bm, bc = bag_picks(A_, KB, rb)
    Wc = np.stack([W0] + [np.round(W0 + lam * (pk - W0), 6) for lam in LAMS] + [SLW[k] for k in SLATE]
                  + ([np.round(W0 + lam * (bm - W0), 6) for lam in LAMS] + [np.round(W0 + lam * (bc - W0), 6) for lam in LAMS] if KB else []), 1)
    aB = grid_auc(B_, Wc); base = aB[:, 0]
    gl = {lam: aB[:, 1 + i] - base for i, lam in enumerate(LAMS)}
    for lam in LAMS: PLP[lam] += gl[lam] > 0; PLN[lam] += gl[lam] < 0
    for i, k in enumerate(SLATE):
        d = aB[:, 1 + len(LAMS) + i] - base; GS[k].append(d.mean()); PLS[k] += d
    MK = masks(pk, gn, ag)
    if KB:
        o0 = 1 + len(LAMS) + len(SLATE)
        for vb, off in (("BAGMEAN_ALL", o0), ("BAGCURVE_ALL", o0 + len(LAMS))):
            msize[vb].append(12)
            for i, lam in enumerate(LAMS):
                d = aB[:, off + i] - base; G[vb][lam].append(d.mean()); PL[vb][lam] += d
    if MK["TOP1"].any():
        j1 = int(np.flatnonzero(MK["TOP1"])[0]); top1_lab.append(j1); top1_g.append([gl[lam][j1] / 12 for lam in LAMS])
    for v, mk in MK.items():
        msize[v].append(int(mk.sum()))
        for lam in LAMS:
            d = np.where(mk, gl[lam], 0.0); G[v][lam].append(d.mean()); PL[v][lam] += d
    if reps % 200 == 0: print("rep", reps, flush=True)


def summ(x):
    x = np.array(x); return dict(mean=float(x.mean()), sd=float(x.std()), se=float(x.std() / np.sqrt(len(x))),
                                 p_gt0=float((x > 0).mean()), p_lt0=float((x < 0).mean()), p_eq0=float((x == 0).mean()))


res = dict(ours=OURSF, base=BASE, n=n, ntrain=ntr, reps=reps, tries=tries, k_inner=KIN, seed=SEED,
           full=dict(c4_per_label=dict(zip(L, np.round(auc_c4_full, 5).tolist())), c4_macro=float(np.nanmean(auc_c4_full)),
                     pick=dict(zip(L, pk_full.tolist())), insample_gain=dict(zip(L, np.round(gn_full, 5).tolist())),
                     dir_agree=dict(zip(L, [None if np.isnan(a) else round(float(a), 3) for a in ag_full]))),
           cv={}, slate={}, picks_freq={}, nested_agree_mean={})
print("\n=== held-out macro gain vs c4 (train %d / test %d, %d reps) ===" % (ntr, n - ntr, reps))
for v in VAR:
    res["cv"][v] = {}
    for lam in LAMS:
        s = summ(G[v][lam]); s["per_label"] = dict(zip(L, np.round(PL[v][lam] / reps, 5).tolist())); s["mean_size"] = float(np.mean(msize[v]))
        if v == "ALL": s["per_label_p_gt0"] = dict(zip(L, np.round(PLP[lam] / reps, 3).tolist())); s["per_label_p_lt0"] = dict(zip(L, np.round(PLN[lam] / reps, 3).tolist()))
        res["cv"][v][str(lam)] = s
    print("%-18s |S|=%.2f | " % (v, np.mean(msize[v])) + " ".join(
        "l%.3g %+.5f(P>0 %.2f)" % (lam, res["cv"][v][str(lam)]["mean"], res["cv"][v][str(lam)]["p_gt0"]) for lam in MAIN_LAMS))
print("\n=== fixed A19 slate tables on the same B halves (LEAKY: chosen on all 58) ===")
for k in SLATE:
    s = summ(GS[k]); s["per_label"] = dict(zip(L, np.round(PLS[k] / reps, 5).tolist())); res["slate"][k] = s
    print("%-7s %+.5f sd %.5f P>0 %.3f" % (k, s["mean"], s["sd"], s["p_gt0"]))
P = np.array(pick_log); AG = np.array(agree_log)
for j, l in enumerate(L):
    vals, cn = np.unique(P[:, j], return_counts=True)
    res["picks_freq"][l] = {str(v): round(float(c) / reps, 3) for v, c in zip(vals, cn)}
    res["nested_agree_mean"][l] = None if np.isnan(AG[:, j]).all() else round(float(np.nanmean(AG[:, j])), 3)
# TOP1 decomposition: which label half A promotes, and its held-out macro contribution
tl = np.array(top1_lab); tg = np.array(top1_g); res["top1_by_label"] = {}
for j in np.unique(tl):
    sel = tl == j
    res["top1_by_label"][L[j]] = dict(freq=round(float(sel.sum()) / reps, 3), **{"heldout_macro_l%.3g" % lam: float(tg[sel, i].mean()) for i, lam in enumerate(LAMS) if lam in MAIN_LAMS})
print("TOP1 label freq / held-out macro at l=1:", {k: (v["freq"], round(v["heldout_macro_l1"], 5)) for k, v in res["top1_by_label"].items()})
# paired: best honest procedures vs slate on the same splits
res["paired"] = {}
for v, lam in [("ALL", 0.25), ("ALL", 0.5), ("NEST_DIR75", 0.5), ("NEST_DIR75", 0.75), ("NEST_DIR75", 1.0), ("TOP1", 1.0), ("TOP2", 1.0), ("TOP2", 0.75), ("NEST_DIR90", 1.0)]:
    for k in [k for k in ("c3max", "c2max", "fr100", "c2", "fr90", "pf10", "lat10", "c4") if k in SLATE]:
        d = np.array(G[v][lam]) - np.array(GS[k]); res["paired"]["%s l%.3g - %s" % (v, lam, k)] = dict(mean=float(d.mean()), p_gt0=float((d > 0).mean()))
# full-data tables per rule and lambda
res["full_tables"] = {}
for v, mk in masks(pk_full, gn_full, ag_full).items():
    res["full_tables"][v] = {}
    for lam in LAMS:
        w = np.where(mk, W0 + lam * (pk_full - W0), W0)
        a = grid_auc(np.arange(n), np.round(w, 6)[:, None])[:, 0]
        res["full_tables"][v][str(lam)] = dict(changes={L[j]: round(float(w[j]), 4) for j in range(12) if abs(w[j] - W0[j]) > 1e-9},
                                               insample_gain=float(np.nanmean(a - auc_c4_full)))
if KB:
    bm, bc = bag_picks(np.arange(n), 400, np.random.default_rng(11))
    for vb, pv in (("BAGMEAN_ALL", bm), ("BAGCURVE_ALL", bc)):
        res["full_tables"][vb] = {}
        for lam in LAMS:
            w = W0 + lam * (pv - W0); a = grid_auc(np.arange(n), np.round(w, 6)[:, None])[:, 0]
            res["full_tables"][vb][str(lam)] = dict(changes={L[j]: round(float(w[j]), 4) for j in range(12) if abs(w[j] - W0[j]) > 1e-9},
                                                    insample_gain=float(np.nanmean(a - auc_c4_full)))
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
