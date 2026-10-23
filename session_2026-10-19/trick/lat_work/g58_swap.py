# gold58: (1) is R6 position-based (mirror pass swaps M/L)? (2) which gold58 studies are laterality-suspect, and do any
# member's M/L outputs look swapped there? (3) per-study macro-AUC cost of an M/L swap in S6fall / R6 / both inside the c4
# blend, and the expected hidden-set loss for a given mis-sided fraction. CPU only.
import json
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

D = 'C:/Users/eeshs/Downloads/rsna-knee/'
G = D + 'session_2026-10-18/next/g58'
OUT = D + 'session_2026-10-19/trick/lat_work/'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
C4 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.5, "Synovitis": 0.8, "Baker's": 0.5, "Lateral Meniscus": 0.4, "Fracture": 0.65}
P_ML = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]
g = pd.read_csv(G + "/gold74_eval_set.csv"); g = g[g.source == "original_gold58"].reset_index(drop=True)
ids = g.StudyInstanceUID.astype(str).tolist(); Y = g[L].values.astype(int); n = len(ids)
R6 = pd.read_csv(G + "/R6_gold58_serve.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
R7 = pd.read_csv(G + "/R7OOF.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, ["p|" + l for l in L]].values
R384 = np.load(G + "/gold58_r384_kernelserve_cpu.npy"); R224 = np.load(G + "/gold58_r224_kernelserve.npy")
S6 = pd.read_csv(G + "/n42gpu_gold58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, L].values
r224ids = pd.read_csv(G + "/gold58_r224_ids.csv").StudyInstanceUID.astype(str).tolist()
assert r224ids == ids, 'npy row order differs from gold58 order'
pct = lambda v: rankdata(v) / len(v)

def auc(s, y):
    p = y.sum(); q = len(y) - p
    r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)

def macro(P):
    return np.mean([auc(P[:, j], Y[:, j]) for j in range(12)])

def c4(r6, s6, r384=R384, r224=R224):
    P = np.zeros((n, 12))
    for j, c in enumerate(L):
        rr = .75 * pct(r384[:, j]) + .25 * pct(r224[:, j]); w = C4.get(c, .5)
        b = w * rr + (1 - w) * pct(r6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(s6[:, j])
    return P

res = {}
# ---- (1) R6 position-based? from the worker dump (plain vs raw mirrored pass, fold-mean logits)
dump = json.load(open(G + '/r6g58_box/run_b/worker0/dump.json'))
zp = np.array([np.mean(dump[i]['z_plain'], 0) for i in ids]); zm = np.array([np.mean(dump[i]['z_mirror_raw'], 0) for i in ids])
pos = {}
for a, b in [(2, 3), (4, 5)]:
    same = spearmanr(zm[:, a], zp[:, a])[0]; cross = spearmanr(zm[:, a], zp[:, b])[0]
    same2 = spearmanr(zm[:, b], zp[:, b])[0]; cross2 = spearmanr(zm[:, b], zp[:, a])[0]
    pos[L[a]] = dict(rho_mirror_same=round(same, 3), rho_mirror_cross=round(cross, 3))
    pos[L[b]] = dict(rho_mirror_same=round(same2, 3), rho_mirror_cross=round(cross2, 3))
    # AUC of the raw mirrored pass's column vs gold of the same / the other compartment
    pos[L[a]].update(auc_mirror_raw_vs_own_gold=round(auc(zm[:, a], Y[:, a]), 3), auc_mirror_raw_vs_other_gold=round(auc(zm[:, a], Y[:, b]), 3))
    pos[L[b]].update(auc_mirror_raw_vs_own_gold=round(auc(zm[:, b], Y[:, b]), 3), auc_mirror_raw_vs_other_gold=round(auc(zm[:, b], Y[:, a]), 3))
res['r6_position_based'] = pos
print('R6 mirror pass (raw, unpermuted):', json.dumps(pos, indent=0))

# ---- (2) gold58 laterality status
kc = pd.read_csv(D + 'session_2026-10-15/kcrop/out_kc_side_v1/kc_side_train.csv').set_index('StudyInstanceUID')
rs = pd.read_csv(OUT + 'report_side.csv').set_index('StudyInstanceUID')
gs = pd.DataFrame({'mean_x': kc.loc[ids, 'mean_x'].values, 'tag': kc.loc[ids, 'tag'].values,
                   'rep': rs.loc[ids, 'rep_side'].values, 'lang': rs.loc[ids, 'lang'].values}, index=ids)
gs['x_side'] = np.where(gs.mean_x >= 0, 'L', 'R')
gs['tagged'] = gs.tag.isin(['L', 'R'])
gs['s6_side'] = np.where(gs.tagged, gs.tag, gs.x_side)
gs['side_checked_by_r6'] = (~gs.tagged) | (gs.tagged & (gs.mean_x.abs() >= 20) & (gs.tag != gs.x_side))
print('\ngold58: tagged %d, untagged %d, R6 side check would run on %d (receipt says checked 32, flipped 1)'
      % (gs.tagged.sum(), (~gs.tagged).sum(), gs.side_checked_by_r6.sum()))
print('gold58 |x|<20 untagged:', int(((~gs.tagged) & (gs.mean_x.abs() < 20)).sum()), '; tag != x:', int((gs.tagged & (gs.tag != gs.x_side)).sum()),
      '; report sided:', int(gs.rep.isin(['L', 'R']).sum()), '; report != s6 side:', int((gs.rep.isin(['L', 'R']) & (gs.rep != gs.s6_side)).sum()))

# per-study swap evidence vs gold (M/L pairs), per member: LLR(swapped vs as-is) with rank-probabilities
def llr(M):
    out = np.zeros(n)
    Q = np.column_stack([pct(M[:, j]) for j in range(12)]).clip(.01, .99)
    for a, b in [(2, 3), (4, 5)]:
        for u, v in [(a, b), (b, a)]:
            y = Y[:, u]
            lp = lambda q: y * np.log(q) + (1 - y) * np.log(1 - q)
            out += lp(Q[:, v]) - lp(Q[:, u])
    return out
ev = pd.DataFrame({'R6': llr(R6), 'S6fall': llr(S6), 'r384': llr(R384), 'R7oof': llr(R7)}, index=ids)
gs = gs.join(ev)
pd.set_option('display.width', 250)
sus = gs[(~gs.tagged & (gs.mean_x.abs() < 30)) | (gs.tagged & (gs.tag != gs.x_side)) | (gs.rep.isin(['L', 'R']) & (gs.rep != gs.s6_side))]
print('\nsuspect gold58 studies (+ swap LLR per member; > 0 = swapped fits gold better):')
print(sus[['lang', 'mean_x', 'tag', 'rep', 's6_side', 'R6', 'S6fall', 'r384', 'R7oof']].round(2).to_string())
print('\nstudies where some member has swap LLR > 1:')
print(gs[(gs[['R6', 'S6fall', 'r384', 'R7oof']] > 1).any(axis=1)][['lang', 'mean_x', 'tag', 'rep', 'R6', 'S6fall', 'r384', 'R7oof']].round(2).to_string())
res['gold58_suspect'] = sus[['mean_x', 'tag', 'rep', 's6_side', 'R6', 'S6fall', 'r384', 'R7oof']].round(3).reset_index().to_dict('records')

# ---- (3) cost of a swap, per study, inside c4
base = c4(R6, S6); m0 = macro(base)
print('\nc4 gold58 macro %.4f (R6 %.4f, S6fall %.4f, r384 %.4f)' % (m0, macro(R6), macro(S6), macro(R384)))
costs = {}
for who in ['S6fall', 'R6', 'both', 'R6+S6 (Raptor immune)']:
    d = np.zeros(n)
    for i in range(n):
        r6, s6 = R6.copy(), S6.copy()
        if who in ('S6fall', 'both', 'R6+S6 (Raptor immune)'):
            s6[i] = s6[i, P_ML]
        if who in ('R6', 'both', 'R6+S6 (Raptor immune)'):
            r6[i] = r6[i, P_ML]
        d[i] = macro(c4(r6, s6)) - m0
    costs[who] = dict(mean_per_study=float(d.mean()), median=float(np.median(d)), min=float(d.min()), frac_neg=float((d < -1e-12).mean()))
    print('%-22s per-study swap: mean dMacro %+.5f  median %+.5f  worst %+.5f  frac<0 %.2f' % (who, d.mean(), np.median(d), d.min(), (d < -1e-12).mean()))
# the per-study cost scales with 1/n of the AUC pairs: on a set of N studies one swap costs ~ (58/N) x the gold58 figure.
# expected macro loss on the hidden set with mis-sided fraction f = f x N x (58/N) x mean_per_study = f x 58 x mean_per_study
for f in (0.002, 0.005, 0.01, 0.02, 0.05):
    print('f=%.3f  expected dMacro  S6fall-only %+.5f   R6-only %+.5f   both %+.5f' % (
        f, f * n * costs['S6fall']['mean_per_study'], f * n * costs['R6']['mean_per_study'], f * n * costs['both']['mean_per_study']))
res['swap_cost_gold58'] = costs
# Monte Carlo check of the linear formula at f = 5 % on gold58 (3 random studies swapped in S6fall)
rng = np.random.default_rng(0); mc = []
for _ in range(400):
    k = rng.choice(n, 3, replace=False); s6 = S6.copy(); s6[k] = s6[k][:, P_ML]
    mc.append(macro(c4(R6, s6)) - m0)
print('MC: 3 of 58 S6fall swapped: mean dMacro %+.5f (linear formula %+.5f)' % (np.mean(mc), 3 * costs['S6fall']['mean_per_study']))
res['mc_s6_3of58'] = float(np.mean(mc))
json.dump(res, open(OUT + 'g58_swap.json', 'w'), indent=1, default=str)
