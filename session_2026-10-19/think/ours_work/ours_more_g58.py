# ours_more_g58.py (10-19 think, lens "our unused assets"; read-only numpy on existing prediction files, CPU only).
# Extends ours_g58.py to ALL our trained-but-LB-negative members whose gold58 predictions exist on the box (copied read-only to
# ours_work/boxpreds): KC1 (s2001 f0 + s2003 fA), KC-all (6 members), K14 DT set (OAI T123v00, 5 folds), fleet D set (s1044, 5 folds),
# T_b student (s1047, 5 folds), our S6 seeds s43 / s44 and their soups.
# Questions: (1) LB calibration of gold58 for MEMBER additions (KC1 .2 on the R6 entry read -1 display step; R7 prob-mean on HW
# read a non-move); (2) small weights .05 / .10 at the end; (3) only on prespecified structurally plausible labels; (4) replacing
# r224 inside the Raptor slot; (5) mixing into the ours (R6) slot. Paired bootstrap + split-half sign agreement for each.
import sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
D = sys.argv[1]; BP = sys.argv[2]; OUT = sys.argv[3]
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
HW = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
g = pd.read_csv(D + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)


def rdf(f):
    return pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values.astype(np.float64)


R6 = rdf(D + "/R6_gold58_serve.csv"); S6 = rdf(D + "/n42gpu_gold58.csv")
R7 = pd.read_csv(D + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(D + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(D + "/gold58_r224_kernelserve.npy")
mean_of = lambda pat: np.mean([rdf(f) for f in sorted(glob.glob(BP + pat))], axis=0)
M = {
    "KC1": np.mean([rdf(BP + "/fleet_runs/f0_s2001_kc256_r7tb/epochs/gold_f0_s2001_raw_ep12.csv"),
                    rdf(BP + "/fleet_runs/fA_s2003_kc256_dread/epochs/gold_f-1_s2003_raw_ep12.csv")], axis=0),
    "KCall6": mean_of("/fleet_runs/*kc256*/epochs/gold_*_raw_ep12.csv"),
    "K14DT": mean_of("/k14_test/gold/preds/f*_s1043_ttr_frac75oaiT123v00__raw__A.csv"),
    "fleetD": mean_of("/r7t_labels/gold_preds/f*_s1044_ttr_frac75__raw__A.csv"),
    "Tb": mean_of("/r7t_labels/gold_preds/f*_s1047_ttr_r7tb__raw__A.csv"),
    "s43": rdf(BP + "/s6seed/gold58_serve_gpu/ours_s43/out/predictions.csv"),
    "s44": rdf(BP + "/s6seed/gold58_serve_gpu/ours_s44/out/predictions.csv"),
    "soup3": rdf(BP + "/s6seed/gold58_serve_gpu/soup3/out/predictions.csv"),
    "R7oof": R7,
}
M["DT+Tb"] = 0.5 * (M["K14DT"] + M["Tb"])
M["KC1+DT+Tb"] = (M["KC1"] + M["K14DT"] + M["Tb"]) / 3.0
pct = lambda v: rankdata(v) / len(v)


def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)


mac = lambda P: np.array([auc(P[:, j], Y[:, j]) for j in range(12)])


def blend(idx=None, table=C4, A=0.25, ws6=0.2, end=None, wend=0.0, end_labels=None, r224_sub=None, ours_mix=None, wom=0.0,
          om_labels=None, ours_prob=None, vprob=0.0, s6_sub=None):
    idx = np.arange(n) if idx is None else idx
    out = np.zeros((len(idx), 12))
    for j, c in enumerate(L):
        r224 = R224[idx, j] if r224_sub is None else r224_sub[idx, j]
        rr = (1 - A) * pct(R384[idx, j]) + A * pct(r224); w = table.get(c, 0.5)
        if ours_prob is not None: o = pct((1 - vprob) * R6[idx, j] + vprob * ours_prob[idx, j])
        else: o = pct(R6[idx, j])
        if ours_mix is not None and (om_labels is None or c in om_labels): o = (1 - wom) * o + wom * pct(ours_mix[idx, j])
        b = w * rr + (1 - w) * pct(o)
        s6 = S6[idx, j] if s6_sub is None else s6_sub[idx, j]
        f = (1 - ws6) * pct(b) + ws6 * pct(s6)
        if end is not None and (end_labels is None or c in end_labels): f = (1 - wend) * pct(f) + wend * pct(end[idx, j])
        out[:, j] = f
    return out


rng = np.random.default_rng(20261019); B = 2000; Wm = []
while len(Wm) < B:
    w = np.bincount(rng.integers(0, n, n), minlength=n)
    if all((w[Y[:, j] == 1].sum() > 0) and (w[Y[:, j] == 0].sum() > 0) for j in range(12)): Wm.append(w)
Wm = np.array(Wm, float)


def boot_auc(P):
    A = np.zeros((len(Wm), 12))
    for j in range(12):
        pos, neg = Y[:, j] == 1, Y[:, j] == 0
        Cm = (P[pos, j][:, None] > P[neg, j][None, :]) + 0.5 * (P[pos, j][:, None] == P[neg, j][None, :])
        wp, wn = Wm[:, pos], Wm[:, neg]
        A[:, j] = np.einsum('bi,ij,bj->b', wp, Cm, wn) / (wp.sum(1) * wn.sum(1))
    return A


rng2 = np.random.default_rng(5); SPL = []
while len(SPL) < 300:
    perm = rng2.permutation(n); a_, b_ = perm[:29], perm[29:]
    if any(Y[a_, j].sum() in (0, 29) or Y[b_, j].sum() in (0, 29) for j in range(12)): continue
    SPL.append((a_, b_))


def macro_on(P, idx):
    return np.mean([auc(P[:, j], Y[idx, j]) for j in range(12)])


def split_half(kw, basekw):
    # sign agreement of the gain between two disjoint halves (re-ranked inside each half), and held-out gain when half A says +
    agree = []; held = []
    for a_, b_ in SPL:
        dA = macro_on(blend(a_, **kw), a_) - macro_on(blend(a_, **basekw), a_)
        dB = macro_on(blend(b_, **kw), b_) - macro_on(blend(b_, **basekw), b_)
        agree.append(np.sign(dA) == np.sign(dB)); held.append(dB if dA > 0 else np.nan)
    return float(np.mean(agree)), float(np.nanmean(held)) if np.any(~np.isnan(held)) else float('nan')


res = {"members": {}, "cal": {}, "cfg": {}}
base_c4 = blend(); bm = mac(base_c4); bb = boot_auc(base_c4)
res["c4_macro"] = float(bm.mean())
print("c4 gold58 macro %.5f" % bm.mean())
comps = {"r384": R384, "r224": R224, "R6": R6, "S6fall": S6}
comps.update(M)
for k, v in comps.items():
    a = mac(v)
    rho_c4 = float(np.mean([spearmanr(v[:, j], base_c4[:, j])[0] for j in range(12)]))
    rho_r6 = float(np.mean([spearmanr(v[:, j], R6[:, j])[0] for j in range(12)]))
    res["members"][k] = dict(macro=float(a.mean()), rho_c4=rho_c4, rho_R6=rho_r6, per_label=dict(zip(L, np.round(a, 3).tolist())))
    print("  %-10s macro %.4f rho(c4) %.3f rho(R6) %.3f | %s" % (k, a.mean(), rho_c4, rho_r6, " ".join("%s:%.3f" % (l[:5], x) for l, x in zip(L, a))))

# ---- LB calibration for MEMBER additions (gold58 on the base that was actually on the LB at the time) ----
r6entry = dict(table={}, A=0.35, ws6=0.2)                      # R6 57248084: uniform .5, A224 .35, S6fall .2
hwentry = dict(table=HW, A=0.35, ws6=0.2)                      # HW 57262115
CAL = {
    "KC1 .2 end on R6 entry (combo 57250689: LB .957 < .958 = full display step DOWN)": (dict(r6entry, end=M["KC1"], wend=0.2), r6entry),
    "KCall6 .2 end on R6 entry (not on LB)": (dict(r6entry, end=M["KCall6"], wend=0.2), r6entry),
    "R7 prob-mean .5 on HW (R7 late 57291336: LB non-move, practically negative; R7 OOF proxy for public R7 no-flip)": (dict(hwentry, ours_prob=R7, vprob=0.5), hwentry),
    "S6fall .2 added on R6-uniform (LB positive S6F entry, ref)": (r6entry, dict(table={}, A=0.35, ws6=0.0)),
}
for k, (kw, bkw) in CAL.items():
    P, P0 = blend(**kw), blend(**bkw); pt = mac(P) - mac(P0); d = (boot_auc(P) - boot_auc(P0)).mean(1)
    res["cal"][k] = dict(point=float(pt.mean()), p_gt0=float((d > 0).mean()), sd=float(d.std()), per_label=dict(zip(L, np.round(pt, 4).tolist())))
    print("CAL %-110s gold %+.5f P>0 %.3f sd %.5f" % (k, pt.mean(), (d > 0).mean(), d.std()))

# ---- configs on c4 ----
STRUCT = {
    "KC1": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],          # knee-centred sagittal / coronal soft-tissue crop
    "KCall6": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "K14DT": ["Medial OA", "Lateral OA", "PF OA"],                        # OAI T123 pretraining = OA / cartilage grades
    "Tb": None, "fleetD": None, "s44": None, "soup3": None, "s43": None, "DT+Tb": None, "KC1+DT+Tb": None, "R7oof": None,
}
CFG = {}
for m in ("KC1", "KCall6", "K14DT", "fleetD", "Tb", "DT+Tb", "KC1+DT+Tb", "s44", "soup3", "R7oof"):
    for w in (0.05, 0.10):
        CFG["%s end %.2f all" % (m, w)] = dict(end=M[m], wend=w)
    if STRUCT.get(m):
        for w in (0.05, 0.10, 0.20):
            CFG["%s end %.2f struct" % (m, w)] = dict(end=M[m], wend=w, end_labels=STRUCT[m])
for m in ("KC1", "KCall6", "K14DT", "fleetD", "Tb", "DT+Tb", "s44"):
    CFG["%s replaces r224 (A .25)" % m] = dict(r224_sub=M[m])
    CFG["%s ours-slot rank .10" % m] = dict(ours_mix=M[m], wom=0.10)
CFG["r224 dropped (A 0)"] = dict(A=0.0)
CFG["S6 slot = soup3 instead of S6fall"] = dict(s6_sub=M["soup3"])
CFG["S6 slot = mean(S6fall,s43,s44) probs"] = dict(s6_sub=(S6 + M["s43"] + M["s44"]) / 3.0)
for k, kw in CFG.items():
    if kw is None:
        continue
    P = blend(**kw); pt = mac(P) - bm; d = (boot_auc(P) - bb).mean(1)
    top = np.argsort(-np.abs(pt))[:3]
    res["cfg"][k] = dict(point=float(pt.mean()), sd=float(d.std()), p_gt0=float((d > 0).mean()), per_label=dict(zip(L, np.round(pt, 4).tolist())))
    print("%-34s %+.5f sd %.5f P>0 %.3f | %s" % (k, pt.mean(), d.std(), (d > 0).mean(), ", ".join("%s %+.4f" % (L[j], pt[j]) for j in top if pt[j] != 0)))
# split-half sign stability for the few configs that look best (decided by point estimate: honest about the selection)
best = sorted(res["cfg"].items(), key=lambda t: -t[1]["point"])[:6]
res["split_half"] = {}
for k, _ in best:
    a, h = split_half(CFG[k], {})
    res["split_half"][k] = dict(sign_agree=a, heldout_if_A_pos=h)
    print("split-half %-34s sign agreement %.3f | held-out gain when half A says + %+.5f" % (k, a, h))
json.dump(res, open(OUT, "w"), indent=1)
