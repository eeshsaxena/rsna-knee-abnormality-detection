# recal_rulers.py (10-20 night, Claude, TRACK 1 recalibration). RESEARCH ONLY, local CPU, no OAI rows (OAI = aggregates already
# on the laptop: val/work/box/perlabel_{PFT,PAREA}.log JSON block). Writes only recal_rulers.log / recal_rulers.json next to itself.
# Offline E-OA predictions for eoa10 / eoa15 / eoa20 / eoa25 / eoa30 / pw20 / pw25 / pw30 on every ruler we have:
#   gold58 c4 (kernel cell-18 rebuild, paired bootstrap), gold74 R7 OOF, gold58 R7 OOF, report OOF (R7 vs report v2 / v4, 3-model mean vs v2),
#   OAI raw and coupling-matched (D5 / s1044, P_FT / P_AREA, regimes raw / R3p-1 / R3p+0 / gap-matched), plus the CLAIMED skeptic models.
import sys, json
import numpy as np
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push')
import census_lib as CL
from census_lib import auc, pct, L

OUTD = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
OA = ['Medial OA', 'Lateral OA', 'PF OA']; JO = [L.index(c) for c in OA]
W_PW = {('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}
SHAPES = [('S10', 'sym', .1), ('S15', 'sym', .15), ('S20', 'sym', .2), ('S25', 'sym', .25), ('S30', 'sym', .3),
          ('PW20', 'pw', .2), ('PW25', 'pw', .25), ('PW30', 'pw', .3)]
log = open(OUTD + 'recal_rulers.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


def eoa(F, mode, a):
    """Kernel cell-18 E-OA on a n x 12 matrix (columns in L order). Returns a copy with the OA trio replaced."""
    G = F.copy(); U = {c: pct(F[:, L.index(c)]) for c in OA}
    for c in OA:
        o = [k for k in OA if k != c]
        if mode == 'sym':
            G[:, L.index(c)] = pct(U[c] - a * (U[o[0]] + U[o[1]]) / 2.0)
        else:
            w = [W_PW[tuple(sorted((c, k), key=OA.index))] for k in o]
            G[:, L.index(c)] = pct(U[c] - a / 2.0 * (w[0] * U[o[0]] + w[1] * U[o[1]]))
    return G


def oa_deltas(F, Y, mode, a):
    """Per-OA-label AUC delta (rows with the label present). E-OA is applied on rows where all 3 OA scores exist."""
    G = eoa(F, mode, a); out = []
    for j in JO:
        ok = ~np.isnan(Y[:, j]); y = Y[ok, j].astype(int)
        out.append(auc(G[ok, j], y) - auc(F[ok, j], y))
    return np.array(out)


D = CL.data()
res = {}
# ---------- 1. gold58 c4 (kernel path rebuild; parity c4 .94113) ----------
C4 = D['C4']; Y58 = D['Y58'].astype(float)
base = np.mean([auc(C4[:, j], D['Y58'][:, j]) for j in range(12)])
P('gold58 c4 parity macro %.5f (target .94113)' % base)
rng = np.random.default_rng(20261020); n58 = len(Y58); B = 2000
BI = [rng.integers(0, n58, n58) for _ in range(B)]
g58 = {}; g58boot = {}
for nm, mode, a in SHAPES:
    G = eoa(C4, mode, a)
    d = np.array([auc(G[:, j], D['Y58'][:, j]) - auc(C4[:, j], D['Y58'][:, j]) for j in JO])
    g58[nm] = dict(per_label=d.tolist(), macro=float(d.sum() / 12))
    bb = []
    for ii in BI:
        s = 0.
        for j in JO:
            y = D['Y58'][ii, j]
            if y.min() == y.max(): s = np.nan; break
            s += auc(G[ii, j], y) - auc(C4[ii, j], y)
        bb.append(s / 12)
    g58boot[nm] = np.array(bb)
    P('gold58 c4 %-5s macro %+.5f  per-label M/L/PF %+.4f %+.4f %+.4f  boot sd %.5f  P>0 %.3f' % (
        nm, d.sum() / 12, *d, np.nanstd(g58boot[nm]), np.nanmean(g58boot[nm] > 0)))
ok = np.all([~np.isnan(g58boot[k]) for k in g58boot], 0)
Mb = np.array([g58boot[k][ok] for k, _, _ in SHAPES])
res['g58'] = g58; res['g58_boot_sd'] = {k: float(np.std(g58boot[k][ok])) for k, _, _ in SHAPES}
res['g58_boot_corr'] = np.corrcoef(Mb).round(3).tolist()
P('gold58 boot corr S10,S20,S30,PW20,PW30:', np.corrcoef(Mb[[0, 2, 4, 5, 7]]).round(2).tolist())

# ---------- 2. R7 OOF on gold74 and gold58 (a single report-trained model, its own OA coupling) ----------
for nm_r, F, Y in (('g74_R7', D['R7_74'], D['Y74']), ('g58_R7', D['R7_58'], Y58)):
    ok = ~np.isnan(F[:, JO]).any(1); F = F[ok]; Y = Y[ok]
    out = {}
    for nm, mode, a in SHAPES:
        d = oa_deltas(F, Y, mode, a); out[nm] = dict(per_label=d.tolist(), macro=float(d.sum() / 12))
    res[nm_r] = out
    P('%s n %d: ' % (nm_r, ok.sum()) + ' | '.join('%s %+.5f' % (k, v['macro']) for k, v in out.items()))

# ---------- 3. report OOF (report-trained models vs report labels: the mechanism says E-OA should LOSE here) ----------
for nm_r, F, Y in (('rep_R7v2', D['F7'], D['Yv2']), ('rep_R7v4', D['F7'], D['Yv4']), ('rep_M3v2', D['M3'], D['Y3v2'])):
    out = {}
    for nm, mode, a in SHAPES:
        d = oa_deltas(F, Y.astype(float), mode, a); out[nm] = dict(per_label=d.tolist(), macro=float(d.sum() / 12))
    res[nm_r] = out
    P('%s n %d: ' % (nm_r, len(F)) + ' | '.join('%s %+.5f' % (k, v['macro']) for k, v in out.items()))

# ---------- 4. OAI aggregates (box JSON already on disk; aggregates only) ----------
VN_SH = ['S10', 'S15', 'S20', 'S25', 'S30', 'PW20', 'PW25', 'PW30']
for prox in ('PFT', 'PAREA'):
    s = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/val/work/box/perlabel_%s.log' % prox, encoding='utf-8').read()
    J = json.loads(s.split('JSON_BEGIN')[1].split('JSON_END')[0].strip()); VN = J['VN']
    for key, vec in J['mean'].items():
        v = dict(zip(VN, vec))
        tot = {sh: sum(v['%s|%s' % (lab, sh)] for lab in 'MLP') for sh in VN_SH}
        res['oai_%s|%s' % (prox, key)] = dict(tot=tot, rho=v['rho'], q=v['q'], phi=v['phi'], gap=v['rho'] - v['q'],
                                             per=dict((sh, [v['%s|%s' % (lab, sh)] for lab in 'MLP']) for sh in VN_SH))
    # gap-matched regime: interpolate the D5 / s1044 curves to the regime whose own gap equals the target gap
    for mdl in ('D5', 's1044'):
        regs = sorted([k for k in J['mean'] if k.startswith(mdl + '|R3p')], key=lambda k: float(k.split('R3p')[1]))
        gaps = np.array([res['oai_%s|%s' % (prox, k)]['gap'] for k in regs]); xs = np.array([float(k.split('R3p')[1]) for k in regs])
        for tg_nm, tg in (('g58gap.102', .102), ('pubgap.128', .128)):
            # gap decreases with the regime index; interpolate on the reversed arrays
            x = float(np.interp(tg, gaps[::-1], xs[::-1]))
            tot = {sh: float(np.interp(x, xs, [res['oai_%s|%s' % (prox, k)]['tot'][sh] for k in regs])) for sh in VN_SH}
            res['oai_%s|%s|%s' % (prox, mdl, tg_nm)] = dict(tot=tot, regime=x)
            P('OAI %s %-5s gap-matched %-10s regime %+.2f: ' % (prox, mdl, tg_nm, x) + ' '.join('%s %+.5f' % (k, tot[k]) for k in ('S10', 'S20', 'S30', 'PW20', 'PW30')))
    for key in ('D5|raw', 'D5|R3p-1.0', 'D5|R3p+0.0', 's1044|raw', 's1044|R3p+0.0'):
        r = res['oai_%s|%s' % (prox, key)]
        P('OAI %s %-14s gap %.3f: ' % (prox, key, r['gap']) + ' '.join('%s %+.5f' % (k, r['tot'][k]) for k in ('S10', 'S20', 'S30', 'PW20', 'PW30')))

json.dump(res, open(OUTD + 'recal_rulers.json', 'w'), indent=0)
P('RECAL_RULERS DONE')
