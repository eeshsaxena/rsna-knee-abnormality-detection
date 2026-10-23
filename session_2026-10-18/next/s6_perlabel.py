# s6_perlabel.py (10-18, Claude): per-label S6fall weight ws[label] on the c4 recipe, gold58, laptop CPU (numpy/scipy only).
# Kernel c4 57343435 blend, per label j:
#   rr = .75 pct(r384) + .25 pct(r224);  b = w_c rr + (1 - w_c) pct(ours);  final = (1 - ws_j) pct(b) + ws_j pct(S6fall)
# A label's AUC depends only on its own column, so the per-label ws sweep is separable (macro = mean of the 12 label AUCs).
# Parts: (1) full-data per-label sweep + bootstraps (fixed-prediction multiplicity bootstrap and rerank bootstrap);
#        (2) nested split-half: pick ws_j on half A from the grid, score held-out on half B vs .2 (several pick rules);
#        (3) direction agreement: sign(pick_A - .2) vs sign(pick_B - .2), and vs the full-data direction;
#        (4) global .2 -> .15 / .3 decomposed per label, on c4 and on the LB-time recipe (HW table, A224 .35).
# usage: python s6_perlabel.py <g58 dir> <ours csv> <nrep> <out json> [tables md]
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

D, OURSF, NREP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
TAB = sys.argv[5] if len(sys.argv) > 5 else None
B1, B2 = 4000, 1500
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
HW = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
G = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]; W0 = 0.2
GF = np.round(np.arange(0, 0.5001, 0.025), 3)               # fine grid (holds every shrink-50 midpoint)
iG = np.array([int(np.argmin(np.abs(GF - w))) for w in G]); i0 = int(np.argmin(np.abs(GF - W0)))
fi = lambda w: int(np.argmin(np.abs(GF - w)))

g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids


def load_csv(f):
    df = pd.read_csv(f, dtype={"StudyInstanceUID": str})
    if df.StudyInstanceUID.duplicated().any():
        print("WARN duplicated ids in", f, "-> keeping first"); df = df.drop_duplicates("StudyInstanceUID")
    df = df.set_index("StudyInstanceUID")
    cols = ["p|" + l for l in L] if all("p|" + l in df.columns for l in L) else L
    miss = [i for i in ids if i not in df.index]; assert not miss, "%d gold58 ids missing in %s" % (len(miss), f)
    v = df.loc[ids, cols].values.astype(float); assert np.isfinite(v).all(); return v, cols


OURS, ocols = load_csv(OURSF)
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6, _ = load_csv(D + "/n42gpu_gold58.csv")
assert R384.shape == R224.shape == OURS.shape == S6.shape == (n, 12)
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def aucm(S, y):                       # S (k, m): one AUC per row
    p = y.sum(); q = len(y) - p; r = rankdata(S, axis=1); return (r[:, y == 1].sum(1) - p * (p + 1) / 2) / (p * q)


def parts(idx, j, table=C4, A=0.25):
    rr = (1 - A) * pct(R384[idx, j]) + A * pct(R224[idx, j])
    w = table.get(L[j], 0.5)
    return pct(w * rr + (1 - w) * pct(OURS[idx, j])), pct(S6[idx, j])


def lab_grid(idx, j, grid=GF, **kw):  # scores (len(grid), m) for label j on subset idx
    b, s = parts(idx, j, **kw); grid = np.asarray(grid)
    return (1 - grid)[:, None] * b[None, :] + grid[:, None] * s[None, :]


def lab_auc_grid(idx, j, grid=GF, **kw): return aucm(lab_grid(idx, j, grid, **kw), Y[idx, j])


ALL = np.arange(n)
res = dict(ours_source=OURSF, ours_cols=ocols[:2], n=n, pos=dict(zip(L, Y.sum(0).tolist())), grid=G, nrep=NREP)
# ---------------- components -------------------------------------------------------------------------------------------------
comp = {}
for j, l in enumerate(L):
    b, s = parts(ALL, j)
    comp[l] = dict(ours=auc(OURS[:, j], Y[:, j]), r384=auc(R384[:, j], Y[:, j]), r224=auc(R224[:, j], Y[:, j]), b=auc(b, Y[:, j]),
                   S6=auc(S6[:, j], Y[:, j]), rho_b_S6=float(spearmanr(b, s)[0]), w_c=C4.get(l, 0.5))
res["components"] = comp
# ---------------- (1) full-data sweep -----------------------------------------------------------------------------------------
full = np.array([lab_auc_grid(ALL, j) for j in range(12)])          # (12, len(GF))
fullG = full[:, iG]; d_full = fullG - full[:, [i0]]
res["full_sweep"] = {l: {str(w): float(fullG[j, k]) for k, w in enumerate(G)} for j, l in enumerate(L)}
res["full_best"] = {l: G[int(np.lexsort((np.abs(np.array(G) - W0), -fullG[j]))[0])] for j, l in enumerate(L)}
res["global_macro_c4"] = {str(w): float(fullG[:, k].mean()) for k, w in enumerate(G)}
# fixed-prediction bootstrap with multiplicity weights (exact weighted Mann-Whitney)
rng = np.random.default_rng(20261018)
Wm = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B1)]).astype(np.float64)
ok = np.ones(B1, bool)
for j in range(12):
    ok &= (Wm[:, Y[:, j] == 1].sum(1) > 0) & (Wm[:, Y[:, j] == 0].sum(1) > 0)
Wm = Wm[ok]


def boot_lab(j, grid=G, **kw):         # (B, len(grid)) bootstrap AUCs, predictions fixed on all 58
    S = lab_grid(ALL, j, grid, **kw); pos, neg = Y[:, j] == 1, Y[:, j] == 0; wp, wn = Wm[:, pos], Wm[:, neg]
    out = np.zeros((len(Wm), len(grid)))
    for k in range(len(grid)):
        sp, sn = S[k, pos][:, None], S[k, neg][None, :]; Cm = (sp > sn) + 0.5 * (sp == sn)
        out[:, k] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return out


bootG = np.stack([boot_lab(j) for j in range(12)], 1)               # (B, 12, len(G))
k0 = G.index(W0)
bd = bootG - bootG[:, :, [k0]]
res["boot_fixed_p_gt0"] = {l: {str(w): float((bd[:, j, k] > 0).mean()) for k, w in enumerate(G) if w != W0} for j, l in enumerate(L)}
res["boot_fixed_p_lt0"] = {l: {str(w): float((bd[:, j, k] < 0).mean()) for k, w in enumerate(G) if w != W0} for j, l in enumerate(L)}
# rerank bootstrap (ranks recomputed inside each resample)
rng2 = np.random.default_rng(1); idxs = []
while len(idxs) < B2:
    idx = rng2.integers(0, n, n)
    if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == n).any(): continue
    idxs.append(idx)
rr_boot = np.array([[lab_auc_grid(idx, j, G) for j in range(12)] for idx in idxs])   # (B2, 12, len(G))
rd = rr_boot - rr_boot[:, :, [k0]]
res["boot_rerank_p_gt0"] = {l: {str(w): float((rd[:, j, k] > 0).mean()) for k, w in enumerate(G) if w != W0} for j, l in enumerate(L)}
res["boot_rerank_p_lt0"] = {l: {str(w): float((rd[:, j, k] < 0).mean()) for k, w in enumerate(G) if w != W0} for j, l in enumerate(L)}
res["boot_rerank_sd"] = {l: {str(w): float(rd[:, j, k].std()) for k, w in enumerate(G) if w != W0} for j, l in enumerate(L)}
res["boot_rerank_macro_sd"] = {str(w): float(rd[:, :, k].mean(1).std()) for k, w in enumerate(G) if w != W0}
# ---------------- (4) global moves, per-label decomposition, c4 and LB-time recipe (HW, A .35) -------------------------------------
glob = {}
for rname, kw in [("c4_A25", dict(table=C4, A=0.25)), ("HW_A35 (LB-time)", dict(table=HW, A=0.35))]:
    fa = np.array([lab_auc_grid(ALL, j, G, **kw) for j in range(12)])
    bb = np.stack([boot_lab(j, G, **kw) for j in range(12)], 1)
    gg = {}
    for w in (0.15, 0.3, 0.1, 0.25):
        k = G.index(w); dl = fa[:, k] - fa[:, k0]; dm = (bb[:, :, k] - bb[:, :, k0]).mean(1)
        gg[str(w)] = dict(macro=float(dl.mean()), p_gt0_fixed=float((dm > 0).mean()), per_label_macro_contrib={l: float(dl[j] / 12) for j, l in enumerate(L)})
    glob[rname] = dict(base_macro=float(fa[:, k0].mean()), moves=gg)
res["global_moves"] = glob
# ---------------- (2)+(3) nested split-half ----------------------------------------------------------------------------------------
rng3 = np.random.default_rng(0)
VAR = ["free", "shrink50", "min005", "min010", "dir_step05", "dir_step10"]
held = {v: [] for v in VAR}; held_glob = []
picksA = np.zeros((NREP, 12)); picksB = np.zeros((NREP, 12)); gainsA = np.zeros((NREP, 12))
fixed_held = np.zeros((NREP, 12, len(G)))
reps = 0; tried = 0
while reps < NREP:
    tried += 1
    perm = rng3.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if any(Y[A_, j].sum() in (0, len(A_)) or Y[B_, j].sum() in (0, len(B_)) for j in range(12)): continue
    aA = np.array([lab_auc_grid(A_, j) for j in range(12)]); aB = np.array([lab_auc_grid(B_, j) for j in range(12)])
    gv = {v: np.zeros(12) for v in VAR}
    for j in range(12):
        def pick(a):
            kk = np.lexsort((np.abs(np.array(G) - W0), -a[iG]))[0]; return G[int(kk)]
        pa, pb = pick(aA[j]), pick(aB[j]); picksA[reps, j], picksB[reps, j] = pa, pb
        trg = aA[j, fi(pa)] - aA[j, i0]; gainsA[reps, j] = trg
        dA = np.sign(pa - W0)
        cand = dict(free=pa, shrink50=(pa + W0) / 2, min005=pa if trg >= 0.005 else W0, min010=pa if trg >= 0.010 else W0,
                    dir_step05=W0 + 0.05 * dA, dir_step10=W0 + 0.1 * dA)
        for v, w in cand.items():
            gv[v][j] = aB[j, fi(w)] - aB[j, i0]
        fixed_held[reps, j] = aB[j, iG] - aB[j, i0]
    for v in VAR: held[v].append(gv[v])
    mA = aA[:, iG].mean(0); kg = int(np.lexsort((np.abs(np.array(G) - W0), -mA))[0])
    held_glob.append((G[kg], float((aB[:, iG[kg]] - aB[:, i0]).mean())))
    reps += 1
nest = {}
for v in VAR:
    H = np.array(held[v])                                          # (NREP, 12) label-AUC units
    m = H.mean(1)
    nest[v] = dict(macro_mean=float(m.mean()), macro_sd=float(m.std()), macro_p_gt0=float((m > 0).mean()), macro_p_lt0=float((m < 0).mean()),
                   per_label={l: dict(mean=float(H[:, j].mean()), p_gt0=float((H[:, j] > 0).mean()), p_lt0=float((H[:, j] < 0).mean()),
                                      p_eq0=float((H[:, j] == 0).mean())) for j, l in enumerate(L)})
hg = np.array([x[1] for x in held_glob]); gp = [x[0] for x in held_glob]
nest["global_pick"] = dict(macro_mean=float(hg.mean()), macro_sd=float(hg.std()), macro_p_gt0=float((hg > 0).mean()),
                           macro_p_lt0=float((hg < 0).mean()), pick_freq={str(w): gp.count(w) / reps for w in G})
res["nested"] = nest; res["nested_reps"] = reps; res["nested_tried"] = tried
dirs = {}
for j, l in enumerate(L):
    sa, sb = np.sign(picksA[:, j] - W0), np.sign(picksB[:, j] - W0); both = (sa != 0) & (sb != 0)
    fd = np.sign(res["full_best"][l] - W0)
    pu, pdn = float((sa > 0).mean()), float((sa < 0).mean())
    chance = (pu ** 2 + pdn ** 2) / max((pu + pdn) ** 2, 1e-9)
    dirs[l] = dict(pA_up=pu, pA_down=pdn, pA_stay=float((sa == 0).mean()),
                   AB_agree_given_both_move=float((sa[both] == sb[both]).mean()) if both.any() else None, AB_both_move=float(both.mean()),
                   chance_agree=float(chance), full_dir=int(fd), A_matches_full_dir=float((sa == fd).mean()),
                   pick_freq={str(w): float((picksA[:, j] == w).mean()) for w in G},
                   fixed_heldout={str(w): dict(mean=float(fixed_held[:, j, k].mean()), p_gt0=float((fixed_held[:, j, k] > 0).mean()))
                                  for k, w in enumerate(G) if w != W0},
                   heldout_step10_if_A_up=float(np.mean([fixed_held[r, j, G.index(0.3)] for r in range(reps) if sa[r] > 0])) if (sa > 0).any() else None,
                   heldout_step10_if_A_down=float(np.mean([fixed_held[r, j, G.index(0.1)] for r in range(reps) if sa[r] < 0])) if (sa < 0).any() else None)
res["directions"] = dirs
# ---------------- (5) slate-conditional: S6 sweep for Fracture / PF OA under the 10-19 slate w_c values ---------------------------
SLATE_WC = [("Fracture", 0.65), ("Fracture", 0.9), ("Fracture", 1.0), ("PF OA", 0.5), ("PF OA", 0.1)]
slate = {}
rng4 = np.random.default_rng(7); splits = []
while len(splits) < min(NREP, 1000):
    perm = rng4.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if any(Y[A_, j].sum() in (0, len(A_)) or Y[B_, j].sum() in (0, len(B_)) for j in (L.index("Fracture"), L.index("PF OA"))): continue
    splits.append((A_, B_))
for l, wc in SLATE_WC:
    j = L.index(l); tb = {**C4, l: wc}
    fa = lab_auc_grid(ALL, j, G, table=tb); bb = boot_lab(j, G, table=tb)
    hs = []; ups = 0
    for A_, B_ in splits:
        aA = lab_auc_grid(A_, j, G, table=tb); aB = lab_auc_grid(B_, j, G, table=tb)
        kk = int(np.lexsort((np.abs(np.array(G) - W0), -aA))[0]); hs.append(aB[kk] - aB[k0]); ups += int(G[kk] > W0) - int(G[kk] < W0)
    hs = np.array(hs)
    slate["%s w_c %.2f" % (l, wc)] = dict(auc_at_02=float(fa[k0]), delta={str(w): float(fa[k] - fa[k0]) for k, w in enumerate(G) if w != W0},
                                          p_gt0_fixed={str(w): float(((bb[:, k] - bb[:, k0]) > 0).mean()) for k, w in enumerate(G) if w != W0},
                                          nested_free_mean=float(hs.mean()), nested_free_p_gt0=float((hs > 0).mean()), nested_free_p_lt0=float((hs < 0).mean()),
                                          net_up_minus_down=ups / len(splits))
res["slate_conditional"] = slate
json.dump(res, open(OUT, "w"), indent=1)
# ---------------- printout + markdown tables ---------------------------------------------------------------------------------------
md = []
md.append("Components (gold58 AUC, full 58):\n\n| label | pos | w_c | ours | r384 | r224 | b (pre-S6) | S6fall | rho(b,S6) |\n|---|---|---|---|---|---|---|---|---|")
for j, l in enumerate(L):
    c = comp[l]
    md.append("| %s | %d | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f |" % (l, Y[:, j].sum(), c["w_c"], c["ours"], c["r384"], c["r224"], c["b"], c["S6"], c["rho_b_S6"]))
md.append("\nFull-data per-label AUC delta vs ws .2 (label-AUC units, x1000; c4 recipe). Brackets: fixed-bootstrap P(delta>0) / rerank P(delta>0).\n")
md.append("| label | AUC@.2 | " + " | ".join("%.2f" % w for w in G if w != W0) + " | full-data best |\n|---|---|" + "---|" * (len(G) - 1) + "---|")
for j, l in enumerate(L):
    cells = []
    for k, w in enumerate(G):
        if w == W0: continue
        cells.append("%+.1f (%.2f/%.2f)" % (1000 * d_full[j, k], res["boot_fixed_p_gt0"][l][str(w)], res["boot_rerank_p_gt0"][l][str(w)]))
    md.append("| %s | %.4f | %s | %.2f |" % (l, fullG[j, k0], " | ".join(cells), res["full_best"][l]))
md.append("| **macro** | %.5f | %s | |" % (fullG[:, k0].mean(), " | ".join("%+.2f" % (1000 * d_full[:, k].mean()) for k, w in enumerate(G) if w != W0)))
md.append("\nRerank-bootstrap SD of the label delta (x1000), .2 -> .1 / .3: " + ", ".join(
    "%s %.1f/%.1f" % (l, 1000 * res["boot_rerank_sd"][l]["0.1"], 1000 * res["boot_rerank_sd"][l]["0.3"]) for l in L) +
    "; macro %.2f/%.2f" % (1000 * res["boot_rerank_macro_sd"]["0.1"], 1000 * res["boot_rerank_macro_sd"]["0.3"]))
md.append("\nNested split-half (%d reps): pick ws_j on half A, held-out gain on half B vs .2 (label-AUC units x1000), per pick rule:\n" % reps)
md.append("| label | free mean | free P>0 / P<0 | shrink50 mean | min005 mean | dir_step05 mean | dir_step10 mean | half-A pick up / down / stay | A-B dir agree (chance) |\n|---|---|---|---|---|---|---|---|---|")
for j, l in enumerate(L):
    f = nest["free"]["per_label"][l]; dd = dirs[l]
    md.append("| %s | %+.1f | %.2f / %.2f | %+.1f | %+.1f | %+.1f | %+.1f | %.2f / %.2f / %.2f | %s (%.2f) |" % (
        l, 1000 * f["mean"], f["p_gt0"], f["p_lt0"], 1000 * nest["shrink50"]["per_label"][l]["mean"], 1000 * nest["min005"]["per_label"][l]["mean"],
        1000 * nest["dir_step05"]["per_label"][l]["mean"], 1000 * nest["dir_step10"]["per_label"][l]["mean"],
        dd["pA_up"], dd["pA_down"], dd["pA_stay"], "%.2f" % dd["AB_agree_given_both_move"] if dd["AB_agree_given_both_move"] is not None else "-",
        dd["chance_agree"]))
md.append("| **macro** | %+.2f | %.2f / %.2f | %+.2f | %+.2f | %+.2f | %+.2f | | |" % (
    1000 * nest["free"]["macro_mean"], nest["free"]["macro_p_gt0"], nest["free"]["macro_p_lt0"], 1000 * nest["shrink50"]["macro_mean"],
    1000 * nest["min005"]["macro_mean"], 1000 * nest["dir_step05"]["macro_mean"], 1000 * nest["dir_step10"]["macro_mean"]))
md.append("\nGlobal pick on half A (one ws for all labels): held-out macro %+.2f x1e-3, P>0 %.2f, P<0 %.2f; pick freq %s\n" % (
    1000 * hg.mean(), (hg > 0).mean(), (hg < 0).mean(), {k: round(v, 2) for k, v in nest["global_pick"]["pick_freq"].items()}))
md.append("Fixed per-label values, held-out on half B (no pick; split-half noise only), mean x1000 [P>0]:\n")
md.append("| label | " + " | ".join("%.2f" % w for w in G if w != W0) + " |\n|---|" + "---|" * (len(G) - 1))
for l in L:
    md.append("| %s | %s |" % (l, " | ".join("%+.1f [%.2f]" % (1000 * dirs[l]["fixed_heldout"][str(w)]["mean"], dirs[l]["fixed_heldout"][str(w)]["p_gt0"]) for w in G if w != W0)))
for rname, gr in glob.items():
    md.append("\nGlobal moves on %s (base macro %.5f): per-label macro contribution x1000\n" % (rname, gr["base_macro"]))
    md.append("| move | macro x1000 | P>0 fixed | " + " | ".join(L) + " |\n|---|---|---|" + "---|" * 12)
    for w, mv in gr["moves"].items():
        md.append("| .2 -> %s | %+.2f | %.2f | %s |" % (w, 1000 * mv["macro"], mv["p_gt0_fixed"], " | ".join("%+.2f" % (1000 * mv["per_label_macro_contrib"][l]) for l in L)))
md.append("\nSlate-conditional S6 sweep (label AUC delta vs ws .2, x1000; nested free pick over %d split-halves):\n" % len(splits))
md.append("| label @ w_c | AUC@.2 | " + " | ".join("%.2f" % w for w in G if w != W0) + " | nested free mean [P>0/P<0] | net up-down |\n|---|---|" + "---|" * (len(G) - 1) + "---|---|")
for k_, s in slate.items():
    md.append("| %s | %.4f | %s | %+.1f [%.2f/%.2f] | %+.2f |" % (k_, s["auc_at_02"], " | ".join("%+.1f" % (1000 * s["delta"][str(w)]) for w in G if w != W0),
                                                        1000 * s["nested_free_mean"], s["nested_free_p_gt0"], s["nested_free_p_lt0"], s["net_up_minus_down"]))
txt = "\n".join(md)
print(txt)
if TAB: open(TAB, "w", encoding="utf-8").write(txt + "\n")
print("reps", reps, "tried", tried)
