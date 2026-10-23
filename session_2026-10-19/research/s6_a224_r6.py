# s6_a224_r6.py (10-19, Claude): per-label S6fall weight x per-label A224 on the c4 recipe, R6 as ours, gold58 (R6 held out).
# Kernel blend per label c (c4 57343435):
#   rr = (1 - A_c) pct(r384) + A_c pct(r224);  b = w_c rr + (1 - w_c) pct(ours = R6);  final = (1 - ws_c) pct(b) + ws_c pct(S6fall)
# A label's AUC depends only on its own (w_c, A_c, ws_c), so every per-label sweep is separable (macro = mean of 12 label AUCs).
# Parts:
#  (1) full-data joint sweep A x ws per label (c4 and c3max tables), Bayesian-bootstrap P(delta>0) and rerank-bootstrap P(delta>0);
#  (2) nested split-half (NREP reps): pick on half A, score held-out on half B vs (A .25, ws .2); rules free / shrink50 / min005 /
#      one-step / stability-selected (direction must hold in >= 80 % of Dirichlet resamples of half A) for S6, A224 and joint;
#  (3) direction agreement (half A vs half B free picks; stable-on-A vs free-on-B; half A vs full data);
#  (4) LB facts on R6: per-label decomposition of every reproducible LB comparison (S6 .3 / .15 on HW_A35, c4a15 on c4, ...),
#      and an LB-conditioned posterior: Bayesian-bootstrap draws reweighted by prod Phi(sign * macro_delta / sd_pub).
# Light CPU (numpy / scipy), 58 studies. usage: python s6_a224_r6.py [--nrep 2000] [--reads reads.json]
import argparse, json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm

ap = argparse.ArgumentParser()
ap.add_argument("--g58", default="C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-18/next/g58")
ap.add_argument("--out", default="C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/research")
ap.add_argument("--nrep", type=int, default=2000)
ap.add_argument("--nrep_c3", type=int, default=1000)
ap.add_argument("--bb", type=int, default=6000)
ap.add_argument("--rr", type=int, default=1000)
ap.add_argument("--kstab", type=int, default=100)
ap.add_argument("--sdpub", type=float, default=4e-4)
ap.add_argument("--reads", default=None, help="json list of [base, alt, sign] over the named configs (today's reads)")
ap.add_argument("--tag", default="")
args = ap.parse_args()
D, OUTD = args.g58, args.out
t0 = time.time()

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
HW = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
C2MAX = {**C4, "Fracture": 1.0, "PF OA": 0.1}
C3MAX = {**C2MAX, "Lateral OA": 0.1}
SG = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]; S0 = 0.2
AG = [0.0, 0.1, 0.15, 0.25, 0.35, 0.5]; A0 = 0.25
SF = np.round(np.arange(0, 0.4001, 0.025), 3); AF = np.round(np.arange(0, 0.5001, 0.025), 3)
fS = lambda s: int(np.argmin(np.abs(SF - s))); fA = lambda a: int(np.argmin(np.abs(AF - a)))
iSg = [fS(s) for s in SG]; iAg = [fA(a) for a in AG]; iS0, iA0 = fS(S0), fA(A0)
kS0, kA0 = SG.index(S0), AG.index(A0)

# ---------------- data ---------------------------------------------------------------------------------------------------------
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
assert pd.read_csv(D + "/gold58_r224_ids.csv", dtype=str).StudyInstanceUID.tolist() == ids


def load_csv(f):
    df = pd.read_csv(f, dtype={"StudyInstanceUID": str})
    assert not df.StudyInstanceUID.duplicated().any(), f
    df = df.set_index("StudyInstanceUID")
    cols = ["p|" + l for l in L] if all("p|" + l in df.columns for l in L) else L
    assert all(i in df.index for i in ids), f
    v = df.loc[ids, cols].values.astype(float); assert np.isfinite(v).all(); return v


OURS = load_csv(D + "/R6_gold58_serve.csv")
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
S6 = load_csv(D + "/n42gpu_gold58.csv")
assert R384.shape == R224.shape == OURS.shape == S6.shape == (n, 12)
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


def aucm(S, y):
    p = y.sum(); q = len(y) - p; r = rankdata(S, axis=1); return (r[:, y == 1].sum(1) - p * (p + 1) / 2) / (p * q)


def legs(idx, j): return pct(R384[idx, j]), pct(R224[idx, j]), pct(OURS[idx, j]), pct(S6[idx, j])


def score1(lg, w, a, s):
    p384, p224, po, ps = lg
    return (1 - s) * pct(w * ((1 - a) * p384 + a * p224) + (1 - w) * po) + s * ps


def grid_scores(lg, w, Av, Sv):          # rows ordered A-major: row = iA * len(Sv) + iS
    p384, p224, po, ps = lg; Av = np.asarray(Av, float)[:, None]; Sv = np.asarray(Sv, float); m = len(po)
    b = w * ((1 - Av) * p384 + Av * p224) + (1 - w) * po
    pb = rankdata(b, axis=1) / m
    return ((1 - Sv)[None, :, None] * pb[:, None, :] + Sv[None, :, None] * ps[None, None, :]).reshape(-1, m)


ALL = np.arange(n); LEG = [legs(ALL, j) for j in range(12)]


def cfg(w, A0_=0.25, ws0=0.2, A=None, ws=None): return dict(w=dict(w), A0=A0_, ws0=ws0, A=dict(A or {}), ws=dict(ws or {}))


def P(c, l): return c["w"].get(l, 0.5), c["A"].get(l, c["A0"]), c["ws"].get(l, c["ws0"])


# ---------------- point + Bayesian-bootstrap caches (predictions ranked on all 58) ----------------------------------------------
rngb = np.random.default_rng(20261019)
WB = rngb.dirichlet(np.ones(n), size=args.bb)
POSM = [Y[:, j] == 1 for j in range(12)]
WP = [WB[:, POSM[j]] for j in range(12)]; WN = [WB[:, ~POSM[j]] for j in range(12)]
NORM = [WP[j].sum(1) * WN[j].sum(1) for j in range(12)]
PT, BBC = {}, {}


def key(j, w, a, s): return (j, round(float(w), 4), round(float(a), 4), round(float(s), 4))


def lab_point(j, w, a, s):
    k = key(j, w, a, s)
    if k not in PT: PT[k] = float(auc(score1(LEG[j], w, a, s), Y[:, j]))
    return PT[k]


def wauc(sc, j, wp, wn, nm):
    pos = POSM[j] if len(sc) == n else None
    sp, sn = sc[pos][:, None], sc[~pos][None, :]
    Cm = (sp > sn) + 0.5 * (sp == sn)
    return ((wp @ Cm) * wn).sum(1) / nm


def lab_bb(j, w, a, s):
    k = key(j, w, a, s)
    if k not in BBC: BBC[k] = wauc(score1(LEG[j], w, a, s), j, WP[j], WN[j], NORM[j]).astype(np.float32)
    return BBC[k]


def cfg_point(c): return np.array([lab_point(j, *P(c, l)) for j, l in enumerate(L)])
def cfg_bb(c): return np.stack([lab_bb(j, *P(c, l)) for j, l in enumerate(L)], 1).astype(np.float64)


res = dict(n=n, pos=dict(zip(L, Y.sum(0).tolist())), S_grid=SG, A_grid=AG, args=vars(args))
res["components"] = {l: dict(ours_R6=round(auc(OURS[:, j], Y[:, j]), 4), r384=round(auc(R384[:, j], Y[:, j]), 4),
                             r224=round(auc(R224[:, j], Y[:, j]), 4), S6fall=round(auc(S6[:, j], Y[:, j]), 4)) for j, l in enumerate(L)}

# ---------------- named configs + LB facts -------------------------------------------------------------------------------------
NAMED = {
    "R6uni_noS6": cfg({}, 0.35, 0.0), "R6": cfg({}, 0.35), "HW": cfg(HW, 0.35), "hw3": cfg(HW, 0.35, 0.3), "s615": cfg(HW, 0.35, 0.15),
    "wr45": cfg({**{l: 0.45 for l in L if l not in HW}, **HW}, 0.35), "a25": cfg(HW, 0.25), "c4": cfg(C4), "c4a15": cfg(C4, 0.15),
    "fr65": cfg({**HW, "Fracture": 0.65}), "pf50": cfg({**HW, "PF OA": 0.5}), "a15": cfg(HW, 0.15),
    "c2max": cfg(C2MAX), "fr100": cfg({**C4, "Fracture": 1.0}), "c2": cfg({**C4, "Fracture": 0.9, "PF OA": 0.1}),
    "pf10": cfg({**C4, "PF OA": 0.1}), "c3max": cfg(C3MAX),
}
# (name, base, alt, LB sign of alt - base, set). 'approx' = the ours leg at that time was not R6.
FACTS = [("S6 entry: ws 0 -> .2 (uniform, A .35)", "R6uni_noS6", "R6", +1, "approx"),
         ("HW: heliosli table vs uniform (A .35)", "R6", "HW", +1, "main"),
         ("hw3: HW ws .2 -> .3", "HW", "hw3", -1, "main"),
         ("wr45: HW w .5 -> .45 (7 non-table labels)", "HW", "wr45", -1, "main"),
         ("s615: HW ws .2 -> .15", "HW", "s615", -1, "main"),
         ("a25: HW A .35 -> .25", "HW", "a25", +1, "main"),
         ("c4: a25 table -> c4 table", "a25", "c4", +1, "main"),
         ("c4a15: c4 A .25 -> .15 (read below c4)", "c4", "c4a15", -1, "main"),
         ("fr65 read below c4", "fr65", "c4", +1, "main"),
         ("pf50 read below c4", "pf50", "c4", +1, "main"),
         ("a15 read below c4", "a15", "c4", +1, "main")]
if args.reads:
    for b_, a_, sg in json.load(open(args.reads)):
        FACTS.append(("today: %s vs %s" % (a_, b_), b_, a_, int(sg), "main"))
FP = {}; FB = {}
for nm, b_, a_, sg, st in FACTS:
    FP[nm] = cfg_point(NAMED[a_]) - cfg_point(NAMED[b_])                       # (12,) label deltas
    FB[nm] = cfg_bb(NAMED[a_]) - cfg_bb(NAMED[b_])                             # (BB, 12)


def lb_weights(names, sd):
    lw = np.zeros(args.bb)
    for nm in names:
        sg = [f[3] for f in FACTS if f[0] == nm][0]
        lw += norm.logcdf(sg * FB[nm].mean(1) / sd)
    w = np.exp(lw - lw.max()); return w / w.sum()


MAIN = [f[0] for f in FACTS if f[4] == "main"]
WLB = lb_weights(MAIN, args.sdpub)
WSETS = {"all": MAIN, "noHW": [f for f in MAIN if not f.startswith("HW:")],
         "s6a": [f for f in MAIN if f.split(":")[0] in ("hw3", "s615", "a25", "c4a15", "a15 read below c4")]}
WW = {k: lb_weights(v, args.sdpub) for k, v in WSETS.items()}
ess = lambda w: float(1 / (w ** 2).sum())
res["lb"] = dict(sd_pub=args.sdpub, ess_main=ess(WLB), ess_sets={k: ess(v) for k, v in WW.items()}, sets=WSETS, facts={})
for nm, b_, a_, sg, st in FACTS:
    d = FB[nm]; dm = d.mean(1); w1 = lb_weights([nm], args.sdpub)
    res["lb"]["facts"][nm] = dict(base=b_, alt=a_, lb_sign=sg, set=st, macro_point=float(FP[nm].mean()), bb_p_gt0=float((dm > 0).mean()),
                                  bb_sd=float(dm.std()), agrees=bool(np.sign(FP[nm].mean()) == sg),
                                  per_label_point={l: float(FP[nm][j] / 12) for j, l in enumerate(L)},
                                  per_label_sd={l: float(d[:, j].std() / 12) for j, l in enumerate(L)},
                                  per_label_post_self={l: float((w1 * d[:, j]).sum() / 12) for j, l in enumerate(L)},
                                  per_label_post_all={l: float((WLB * d[:, j]).sum() / 12) for j, l in enumerate(L)},
                                  per_label_post_noHW={l: float((WW["noHW"] * d[:, j]).sum() / 12) for j, l in enumerate(L)},
                                  post_macro={k: float((v * dm).sum()) for k, v in WW.items()},
                                  per_label_p_neg={l: float((d[:, j] < 0).mean()) for j, l in enumerate(L)})
print("LB facts done %.1fs, ESS(main) %.0f / %d" % (time.time() - t0, ess(WLB), args.bb), flush=True)
for sd in (3e-4, 6e-4):
    res["lb"]["ess_sd%.0e" % sd] = ess(lb_weights(MAIN, sd))

# ---------------- (1) full-data joint sweep + bootstraps -----------------------------------------------------------------------
rng2 = np.random.default_rng(1); RIDX = []
while len(RIDX) < args.rr:
    idx = rng2.integers(0, n, n)
    if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == n).any(): continue
    RIDX.append(idx)


def full_sweep(table, tname):
    out = {}
    for j, l in enumerate(L):
        w = table.get(l, 0.5)
        pts = np.array([[lab_point(j, w, a, s) for s in SG] for a in AG])                    # (6, 7)
        bbs = np.stack([np.stack([lab_bb(j, w, a, s) for s in SG]) for a in AG])               # (6, 7, BB)
        d = bbs - bbs[kA0, kS0][None, None, :]
        rr = np.array([aucm(grid_scores(legs(idx, j), w, AG, SG), Y[idx, j]).reshape(len(AG), len(SG)) for idx in RIDX])
        dr = rr - rr[:, [kA0], :][:, :, [kS0]]
        post = (d * WLB[None, None, :]).sum(2)
        sets = {k: dict(mean=(d * v[None, None, :]).sum(2).tolist(), p_gt0=((d > 0) * v[None, None, :]).sum(2).tolist()) for k, v in WW.items()}
        out[l] = dict(lbpost_sets=sets, w_c=w, auc_base=float(pts[kA0, kS0]), delta=(pts - pts[kA0, kS0]).tolist(), bb_p_gt0=(d > 0).mean(2).tolist(),
                      bb_p_lt0=(d < 0).mean(2).tolist(), rr_p_gt0=(dr > 0).mean(0).tolist(), rr_p_lt0=(dr < 0).mean(0).tolist(),
                      bb_sd=d.std(2).tolist(), lbpost_mean=post.tolist(), lbpost_p_gt0=((d > 0) * WLB[None, None, :]).sum(2).tolist())
    return out


FULL = {"c4": full_sweep(C4, "c4"), "c3max": full_sweep(C3MAX, "c3max")}
res["full"] = FULL
print("full sweeps done %.1fs" % (time.time() - t0), flush=True)


# ---------------- (2)+(3) nested split-half ------------------------------------------------------------------------------------
def pick1(vals, grid, base):
    g = np.asarray(grid); return float(g[np.lexsort((np.abs(g - base), -np.asarray(vals)))[0]])


def nested(table, nrep, seed):
    rng = np.random.default_rng(seed); W = [table.get(l, 0.5) for l in L]
    RS = ["free", "shrink50", "min005", "step", "stab80", "stab80_step"]
    RJ = ["joint_free", "joint_shrink50", "stab80_both", "s6_stab_only", "a_stab_only"]
    H = {("S", r): np.zeros((nrep, 12)) for r in RS}; H.update({("A", r): np.zeros((nrep, 12)) for r in RS})
    H.update({("J", r): np.zeros((nrep, 12)) for r in RJ})
    pk = {k: np.zeros((nrep, 12)) for k in ("S_A", "S_B", "A_A", "A_B", "Sst", "Ast", "Sstdir", "Astdir")}
    fixS = np.zeros((nrep, 12, len(SG))); fixA = np.zeros((nrep, 12, len(AG)))
    reps = tried = 0
    while reps < nrep:
        tried += 1
        perm = rng.permutation(n); A_, B_ = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
        if any(Y[A_, j].sum() in (0, len(A_)) or Y[B_, j].sum() in (0, len(B_)) for j in range(12)): continue
        WA = rng.dirichlet(np.ones(len(A_)), size=args.kstab)
        for j in range(12):
            lgA, lgB = legs(A_, j), legs(B_, j)
            GA = aucm(grid_scores(lgA, W[j], AF, SF), Y[A_, j]).reshape(len(AF), len(SF))
            GB = aucm(grid_scores(lgB, W[j], AF, SF), Y[B_, j]).reshape(len(AF), len(SF))
            hb = lambda a, s: GB[fA(a), fS(s)] - GB[iA0, iS0]
            fixS[reps, j] = [hb(A0, s) for s in SG]; fixA[reps, j] = [hb(a, S0) for a in AG]
            # stability on half A: Dirichlet resamples, fixed half-A ranks
            posA = Y[A_, j] == 1; wp, wn = WA[:, posA], WA[:, ~posA]; nm = wp.sum(1) * wn.sum(1)
            def stab(rows, grid, base):
                M = np.zeros((args.kstab, len(grid)))
                for k in range(len(grid)):
                    sc = rows[k]; sp, sn = sc[posA][:, None], sc[~posA][None, :]
                    M[:, k] = ((wp @ ((sp > sn) + 0.5 * (sp == sn))) * wn).sum(1) / nm
                gg = np.asarray(grid); M2 = M - 1e-9 * np.abs(gg - base)[None, :]
                dirs = np.sign(gg[M2.argmax(1)] - base); up, dn = (dirs > 0).mean(), (dirs < 0).mean()
                return (1, up) if up >= dn else (-1, dn)
            for ax, grid, base, cur_A, cur_B, rows, stepv in (
                    ("S", SG, S0, GA[iA0, iSg], GB[iA0, iSg], grid_scores(lgA, W[j], [A0], SG), 0.05),
                    ("A", AG, A0, GA[iAg, iS0], GB[iAg, iS0], grid_scores(lgA, W[j], AG, [S0]), 0.1)):
                pa, pb = pick1(cur_A, grid, base), pick1(cur_B, grid, base)
                pk[ax + "_A"][reps, j], pk[ax + "_B"][reps, j] = pa, pb
                gainA = cur_A[grid.index(pa)] - cur_A[grid.index(base)]
                dA = np.sign(pa - base); sdir, sshare = stab(rows, grid, base)
                stable = sshare >= 0.8 and dA != 0
                pk[ax + "st"][reps, j] = float(stable); pk[ax + "stdir"][reps, j] = sdir if sshare >= 0.8 else 0
                cand = dict(free=pa, shrink50=(pa + base) / 2, min005=pa if gainA >= 0.005 else base,
                            step=base + stepv * dA, stab80=pa if (stable and np.sign(pa - base) == sdir) else base,
                            stab80_step=base + stepv * sdir if sshare >= 0.8 else base)
                for r, v in cand.items():
                    H[(ax, r)][reps, j] = hb(v, S0) if ax == "A" else hb(A0, v)
            # joint
            sub = GA[np.ix_(iAg, iSg)]; best = None
            for ia, a in enumerate(AG):
                for is_, s in enumerate(SG):
                    keyv = (-sub[ia, is_], abs(a - A0) / 0.1 + abs(s - S0) / 0.05)
                    if best is None or keyv < best[0]: best = (keyv, a, s)
            ja, js = best[1], best[2]
            sA = H[("A", "stab80")][reps, j]  # placeholder not used
            aS = pk["A_A"][reps, j] if (pk["Ast"][reps, j] and np.sign(pk["A_A"][reps, j] - A0) == pk["Astdir"][reps, j]) else A0
            sS = pk["S_A"][reps, j] if (pk["Sst"][reps, j] and np.sign(pk["S_A"][reps, j] - S0) == pk["Sstdir"][reps, j]) else S0
            H[("J", "joint_free")][reps, j] = hb(ja, js)
            H[("J", "joint_shrink50")][reps, j] = hb((ja + A0) / 2, (js + S0) / 2)
            H[("J", "stab80_both")][reps, j] = hb(aS, sS)
            H[("J", "s6_stab_only")][reps, j] = hb(A0, sS)
            H[("J", "a_stab_only")][reps, j] = hb(aS, S0)
        reps += 1
    out = dict(reps=reps, tried=tried, rules={}, dirs={})
    for (ax, r), M in H.items():
        m = M.mean(1)
        out["rules"]["%s|%s" % (ax, r)] = dict(macro_mean=float(m.mean()), macro_se=float(m.std() / np.sqrt(reps)), macro_sd=float(m.std()),
                                               p_gt0=float((m > 0).mean()), p_lt0=float((m < 0).mean()),
                                               per_label={l: dict(mean=float(M[:, j].mean()), p_gt0=float((M[:, j] > 0).mean()),
                                                                  p_lt0=float((M[:, j] < 0).mean())) for j, l in enumerate(L)})
    for ax, grid, base, fx in (("S", SG, S0, fixS), ("A", AG, A0, fixA)):
        dd = {}
        for j, l in enumerate(L):
            sa, sb = np.sign(pk[ax + "_A"][:, j] - base), np.sign(pk[ax + "_B"][:, j] - base); both = (sa != 0) & (sb != 0)
            pu, pdn = float((sa > 0).mean()), float((sa < 0).mean())
            st = pk[ax + "st"][:, j] > 0; sdr = pk[ax + "stdir"][:, j]
            fb = FULL["c4" if table is C4 else "c3max"][l]["delta"]
            fv = np.array(fb)[kA0, :] if ax == "S" else np.array(fb)[:, kS0]
            fdir = np.sign(pick1(fv, grid, base) - base)
            dd[l] = dict(up=pu, down=pdn, stay=float((sa == 0).mean()), both_move=float(both.mean()),
                         agree_given_both=float((sa[both] == sb[both]).mean()) if both.any() else None,
                         chance=float((pu ** 2 + pdn ** 2) / max((pu + pdn) ** 2, 1e-9)), full_dir=int(fdir),
                         A_matches_full=float((sa == fdir).mean()),
                         stable_rate=float(st.mean()), stable_up=float(((sdr > 0) & st).mean()), stable_down=float(((sdr < 0) & st).mean()),
                         B_agrees_given_stable=float((np.sign(pk[ax + "_B"][st, j] - base) == sdr[st]).mean()) if st.any() else None,
                         B_opposes_given_stable=float((np.sign(pk[ax + "_B"][st, j] - base) == -sdr[st]).mean()) if st.any() else None,
                         fixed_heldout={str(v): dict(mean=float(fx[:, j, k].mean()), p_gt0=float((fx[:, j, k] > 0).mean()),
                                                     p_lt0=float((fx[:, j, k] < 0).mean())) for k, v in enumerate(grid) if v != base})
        out["dirs"][ax] = dd
    return out


NEST = {"c4": nested(C4, args.nrep, 0)}
print("nested c4 done %.1fs" % (time.time() - t0), flush=True)
NEST["c3max"] = nested(C3MAX, args.nrep_c3, 11)
print("nested c3max done %.1fs" % (time.time() - t0), flush=True)
res["nested"] = NEST

# ---------------- global moves now (on c4 and c3max), per label ---------------------------------------------------------------
GM = {}
for tn, tb in (("c4", C4), ("c3max", C3MAX)):
    base = cfg(tb); bp = cfg_point(base); bbb = cfg_bb(base)
    for nm, c in (("ws .15", cfg(tb, 0.25, 0.15)), ("ws .3", cfg(tb, 0.25, 0.3)), ("ws .1", cfg(tb, 0.25, 0.1)), ("ws .25", cfg(tb, 0.25, 0.25)),
                  ("A .15", cfg(tb, 0.15)), ("A .35", cfg(tb, 0.35)), ("A 0", cfg(tb, 0.0))):
        d = cfg_bb(c) - bbb; dp = cfg_point(c) - bp
        GM["%s | %s" % (tn, nm)] = dict(macro=float(dp.mean()), bb_p_gt0=float((d.mean(1) > 0).mean()),
                                         lbpost_sets={k: [float((v * d.mean(1)).sum()), float((v * (d.mean(1) > 0)).sum())] for k, v in WW.items()},
                                         lbpost=float((WLB * d.mean(1)).sum()), lbpost_p_gt0=float((WLB * (d.mean(1) > 0)).sum()),
                                         per_label={l: float(dp[j] / 12) for j, l in enumerate(L)})
res["global_moves_now"] = GM
json.dump(res, open(OUTD + "/s6_a224_r6%s.json" % args.tag, "w"), indent=1)
print("json written %.1fs" % (time.time() - t0), flush=True)

# ---------------- markdown tables ----------------------------------------------------------------------------------------------
md = []
A_ = lambda x: "%+.1f" % (1000 * x)
md.append("## Components (gold58 label AUC, R6 = ours)\n\n| label | pos | ours R6 | r384 | r224 | S6fall |\n|---|---|---|---|---|---|")
for j, l in enumerate(L):
    c = res["components"][l]; md.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (l, Y[:, j].sum(), c["ours_R6"], c["r384"], c["r224"], c["S6fall"]))
for tn in ("c4", "c3max"):
    F = FULL[tn]
    md.append("\n## Full-data per-label S6fall sweep at A .25, %s table (label-AUC delta vs ws .2, x1000) [Bayes-boot P>0 / rerank P>0]\n" % tn)
    md.append("| label | w_c | AUC@base | " + " | ".join("%.2f" % s for s in SG if s != S0) + " |\n|---|---|---|" + "---|" * (len(SG) - 1))
    for l in L:
        f = F[l]; cells = []
        for k, s in enumerate(SG):
            if s == S0: continue
            cells.append("%s [%.2f/%.2f]" % (A_(f["delta"][kA0][k]), f["bb_p_gt0"][kA0][k], f["rr_p_gt0"][kA0][k]))
        md.append("| %s | %.2f | %.4f | %s |" % (l, f["w_c"], f["auc_base"], " | ".join(cells)))
    md.append("| **macro x1000** | | %.5f | %s |" % (np.mean([F[l]["auc_base"] for l in L]),
                                                  " | ".join("%+.2f" % (1000 * np.mean([F[l]["delta"][kA0][k] for l in L])) for k, s in enumerate(SG) if s != S0)))
    md.append("\n## Full-data per-label A224 sweep at ws .2, %s table (x1000) [Bayes-boot P>0 / rerank P>0]\n" % tn)
    md.append("| label | " + " | ".join("%.2f" % a for a in AG if a != A0) + " |\n|---|" + "---|" * (len(AG) - 1))
    for l in L:
        f = F[l]
        md.append("| %s | %s |" % (l, " | ".join("%s [%.2f/%.2f]" % (A_(f["delta"][k][kS0]), f["bb_p_gt0"][k][kS0], f["rr_p_gt0"][k][kS0])
                                              for k, a in enumerate(AG) if a != A0)))
    md.append("| **macro x1000** | %s |" % " | ".join("%+.2f" % (1000 * np.mean([F[l]["delta"][k][kS0] for l in L])) for k, a in enumerate(AG) if a != A0))
    md.append("\n## Joint full-data best (A, ws) per label, %s table (x1000 vs base; P>0 Bayes / rerank; LB-conditioned posterior mean)\n" % tn)
    md.append("| label | best (A, ws) | delta | P>0 bb / rr | LB-post mean | LB-post P>0 |\n|---|---|---|---|---|---|")
    for l in L:
        f = F[l]; dl = np.array(f["delta"]); best = None
        for ia, a in enumerate(AG):
            for is_, s in enumerate(SG):
                kv = (-dl[ia, is_], abs(a - A0) / 0.1 + abs(s - S0) / 0.05)
                if best is None or kv < best[0]: best = (kv, ia, is_)
        ia, is_ = best[1], best[2]
        md.append("| %s | (%.2f, %.2f) | %s | %.2f / %.2f | %s | %.2f |" % (l, AG[ia], SG[is_], A_(dl[ia, is_]), f["bb_p_gt0"][ia][is_], f["rr_p_gt0"][ia][is_],
                                                                    A_(f["lbpost_mean"][ia][is_]), f["lbpost_p_gt0"][ia][is_]))
for tn in ("c4", "c3max"):
    F = FULL[tn]
    for ax, grid, base in (("S6fall at A .25", SG, S0), ("A224 at ws .2", AG, A0)):
        md.append("\n## LB-conditioned posterior mean of the label delta (x1000), %s, %s table: all facts / no-HW / S6+A facts only [P>0 all]\n" % (ax, tn))
        md.append("| label | " + " | ".join("%.2f" % v for v in grid if v != base) + " |\n|---|" + "---|" * (len(grid) - 1))
        for l in L:
            ps = F[l]["lbpost_sets"]; cells = []
            for k, v in enumerate(grid):
                if v == base: continue
                ia, is_ = (kA0, k) if grid is SG else (k, kS0)
                cells.append("%s / %s / %s [%.2f]" % (A_(ps["all"]["mean"][ia][is_]), A_(ps["noHW"]["mean"][ia][is_]), A_(ps["s6a"]["mean"][ia][is_]), ps["all"]["p_gt0"][ia][is_]))
            md.append("| %s | %s |" % (l, " | ".join(cells)))
for tn in ("c4", "c3max"):
    N = NEST[tn]
    md.append("\n## Nested split-half, %s table (%d reps): held-out macro gain vs (A .25, ws .2), x1000\n" % (tn, N["reps"]))
    md.append("| axis | rule | macro mean (se) | P>0 / P<0 |\n|---|---|---|---|")
    for k, r in N["rules"].items():
        ax, rn = k.split("|"); md.append("| %s | %s | %+.3f (%.3f) | %.2f / %.2f |" % (ax, rn, 1000 * r["macro_mean"], 1000 * r["macro_se"], r["p_gt0"], r["p_lt0"]))
    for ax, nmx in (("S", "S6fall"), ("A", "A224")):
        md.append("\n### %s per label, %s table: held-out label gain x1000 (free / shrink50 / stab80 / stab80_step), direction agreement\n" % (nmx, tn))
        md.append("| label | free | shrink50 | stab80 | stab80_step | half-A up / down / stay | A-B agree given both move (chance) | stable rate (up / down) | B agrees / opposes given stable | full-data dir |\n|---|---|---|---|---|---|---|---|---|---|")
        for l in L:
            r = lambda rn: N["rules"]["%s|%s" % (ax, rn)]["per_label"][l]["mean"]; d = N["dirs"][ax][l]
            md.append("| %s | %s | %s | %s | %s | %.2f / %.2f / %.2f | %s (%.2f) | %.2f (%.2f / %.2f) | %s | %+d |" % (
                l, A_(r("free")), A_(r("shrink50")), A_(r("stab80")), A_(r("stab80_step")), d["up"], d["down"], d["stay"],
                "%.2f" % d["agree_given_both"] if d["agree_given_both"] is not None else "-", d["chance"], d["stable_rate"], d["stable_up"],
                d["stable_down"], "%.2f / %.2f" % (d["B_agrees_given_stable"], d["B_opposes_given_stable"]) if d["B_agrees_given_stable"] is not None else "-",
                d["full_dir"]))
        grid, base = (SG, S0) if ax == "S" else (AG, A0)
        md.append("\nFixed %s values held-out on half B (split-half noise only, NOT selection-honest), mean x1000 [P>0/P<0]:\n" % nmx)
        md.append("| label | " + " | ".join("%.2f" % v for v in grid if v != base) + " |\n|---|" + "---|" * (len(grid) - 1))
        for l in L:
            fh = N["dirs"][ax][l]["fixed_heldout"]
            md.append("| %s | %s |" % (l, " | ".join("%s [%.2f/%.2f]" % (A_(fh[str(v)]["mean"]), fh[str(v)]["p_gt0"], fh[str(v)]["p_lt0"]) for v in grid if v != base)))
md.append("\n## LB facts reproduced on gold58 with R6 (macro x1000; Bayes-boot P>0; agrees = gold58 sign == LB sign); sd_pub %.0e, ESS(main) %.0f / %d\n" % (
    args.sdpub, res["lb"]["ess_main"], args.bb))
md.append("ESS by fact set: %s\n" % {k: round(v) for k, v in res["lb"]["ess_sets"].items()})
md.append("| fact | set | LB sign | gold58 macro | P>0 | sd | agrees | post mean all / noHW / s6a |\n|---|---|---|---|---|---|---|---|")
for nm, f in res["lb"]["facts"].items():
    md.append("| %s | %s | %+d | %+.3f | %.2f | %.2f | %s | %+.3f / %+.3f / %+.3f |" % (
        nm, f["set"], f["lb_sign"], 1000 * f["macro_point"], f["bb_p_gt0"], 1000 * f["bb_sd"], "yes" if f["agrees"] else "NO",
        1000 * f["post_macro"]["all"], 1000 * f["post_macro"]["noHW"], 1000 * f["post_macro"]["s6a"]))
md.append("\n### Per-label decomposition of the LB facts (macro contribution x1000: point / LB-posterior given that fact alone / given all main facts)\n")
md.append("| fact | " + " | ".join(L) + " |\n|---|" + "---|" * 12)
for nm, f in res["lb"]["facts"].items():
    md.append("| %s | %s |" % (nm, " | ".join("%+.2f / %+.2f / %+.2f" % (1000 * f["per_label_point"][l], 1000 * f["per_label_post_self"][l], 1000 * f["per_label_post_all"][l]) for l in L)))
md.append("\n## Global moves on the current tables (macro x1000; Bayes P>0; LB-posterior mean and P>0)\n")
md.append("| move | macro | P>0 | LB-post all (P>0) | LB-post noHW (P>0) | largest per-label contributions |\n|---|---|---|---|---|---|")
for k, v in GM.items():
    top = sorted(v["per_label"].items(), key=lambda x: -abs(x[1]))[:4]
    md.append("| %s | %+.3f | %.2f | %+.3f (%.2f) | %+.3f (%.2f) | %s |" % (
        k, 1000 * v["macro"], v["bb_p_gt0"], 1000 * v["lbpost"], v["lbpost_p_gt0"], 1000 * v["lbpost_sets"]["noHW"][0], v["lbpost_sets"]["noHW"][1],
        ", ".join("%s %+.2f" % (a, 1000 * b) for a, b in top)))
txt = "\n".join(md)
open(OUTD + "/s6_a224_r6_tables%s.md" % args.tag, "w", encoding="utf-8").write(txt + "\n")
print(txt)
print("done %.1fs" % (time.time() - t0))
