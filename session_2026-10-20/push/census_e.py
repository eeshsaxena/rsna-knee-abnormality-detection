# census_e.py (10-20 push, Claude). RESEARCH ONLY, local CPU, competition files only. Extra checks for the census shortlist:
# (a) how many of the 66 pair gaps exceed 2 bootstrap sd, vs the ~3.3 expected by chance; (b) max-of-120 noise level for a
# single direction; (c) base dependence (c4 vs eoa20 base, and the A21 in-flight bases sym30 / pw20 / pw30 for OA targets);
# (d) the 16 extra gold74 studies (our reads under the gold convention, NOT expert reads) with R7, the only model served there;
# (e) leave-one-positive-out influence on gold58 (is the effect carried by one or two studies?).
import sys, json, os, tempfile, shutil, hashlib
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
P = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/'
sys.path.insert(0, P)
import census_lib as CL
from cbase import L, AB, pct, R, G
J = {c: j for j, c in enumerate(L)}; OA = ['Medial OA', 'Lateral OA', 'PF OA']
pairs = pd.read_csv(P + 'census_pairs.csv'); dirs = pd.read_csv(P + 'census_dirs.csv')
d = CL.data(); CUR = d['CUR']; C4 = d['C4']; Y58 = d['Y58']; n58 = len(Y58)
auc = CL.auc

# (a) gap z-count
z = pairs.gap58 / pairs.gap58_sd
print('(a) 66 pair gaps (cur rho - gold58 tet): |z| > 2 in %d pairs (chance ~%.1f); z > 2: %s; z < -2: %s' % (
    int((z.abs() > 2).sum()), 66 * .0455, pairs.pair[z > 2].tolist(), pairs.pair[z < -2].tolist()))
print('    mean gap over 66 pairs %+.3f (attenuation predicts < 0 for a calibrated noisy model); outside the OA trio %+.3f' % (
    pairs.gap58.mean(), pairs[~pairs.pair.isin(['MOA-LOA', 'MOA-PFOA', 'LOA-PFOA'])].gap58.mean()))

# (b) max-of-120 noise level
nd = dirs[~dirs.oa_internal]
for a in ('+0.1', '-0.1'):
    se = nd['sd' + a].values; sims = []
    rng = np.random.default_rng(5)
    for _ in range(4000): sims.append((nd['d' + ('%+.2f' % float(a))].mean() + rng.normal(0, se)).max())
    print('(b) a %s: observed max over 120 directions %+.4f; pure-noise max (independent, observed mean + N(0, se)) median %+.4f, 90th pct %+.4f' % (
        a, nd['d%+.2f' % float(a)].max(), np.median(sims), np.quantile(sims, .9)))

# (c) base dependence for OA targets: c4 base and the A21 in-flight variants (exec their cell 18 like the a21 gate)
A21 = R + '/session_2026-10-20/a21/'
def run_nb(path):
    nb = json.load(open(path, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
    fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]; W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
    m = d['members']; ids = d['ids']
    C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
    V = np.zeros((n58, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); rr = .75 * pct(m['R384'][:, j]) + .25 * pct(m['R224'][:, j]); V[:, j] = w * rr + (1 - w) * pct(m['R6'][:, j])
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
    os.makedirs(W + '/s6f/B'); pd.DataFrame(m['S6'], columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
    ns = {}; exec(fn, ns)
    ns['_s6f_blend']({'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}},
                     {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}})
    o = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(float)
    shutil.rmtree(W, ignore_errors=True); return o
BASES = {'c4': C4, 'eoa20': CUR}
for t in ('v224r6c4eoa30j', 'v224r6c4pw20j', 'v224r6c4pw30j'):
    BASES[t.replace('v224r6c4', '').replace('j', '')] = run_nb(A21 + 'kernel_%s/rsna_sub_%s.ipynb' % (t, t))
SHORT = [('Fracture', 'Medial OA', .1), ('Lateral OA', 'Effusion', -.1), ('MCL', 'Medial Meniscus', .1), ('MCL', 'Synovitis', .1),
         ('Lateral Meniscus', 'Medial Meniscus', .1), ('MCL', 'Contusion', .1), ('Medial OA', "Baker's", -.1), ('Lateral OA', 'Medial Meniscus', .1)]
print('\n(c) base dependence: gold58 single-label delta of each shortlisted direction on each base')
for t, o, a in SHORT:
    s = []
    for k, B in BASES.items():
        y = Y58[:, J[t]]; s.append('%s %+.4f' % (k, auc(CL.tr_score(B, J[t], J[o], a), y) - auc(B[:, J[t]], y)))
    print('    %-14s a %+.1f | %s' % (AB[t] + ' <- ' + AB[o], a, ' | '.join(s)))

# (d) the 16 extra gold74 studies (our reads, gold convention), R7 only
g74 = d['Y74']; R7 = d['R7_74']; ext = np.arange(58, 74) if False else None
gl = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); ext = (gl.source != 'original_gold58').values
print('\n(d) extra 16 (manual_verified, our reads): label positives %s' % {AB[c]: int(np.nansum(g74[ext, J[c]])) for c in L})
for t, o, a in SHORT:
    jt, jo = J[t], J[o]
    for nm, rows in (('g58 part', ~ext), ('extra16', ext), ('all74', np.ones(74, bool))):
        ok = rows & ~np.isnan(g74[:, jt]) & ~np.isnan(R7[:, jt]); y = g74[ok, jt].astype(int)
        if y.sum() == 0 or y.sum() == len(y): print('    %-14s %-8s n %d pos %d: undefined' % (AB[t] + ' <- ' + AB[o], nm, ok.sum(), y.sum())); continue
        # transform uses pct over all 74 rows (as a kernel would over the test set)
        sc = CL.tr_score(np.nan_to_num(R7), jt, jo, a)
        print('    %-14s %-8s n %2d pos %2d: R7 delta %+.4f' % (AB[t] + ' <- ' + AB[o], nm, ok.sum(), y.sum(), auc(sc[ok], y) - auc(R7[ok, jt], y)))

# (e) leave-one-positive-out / leave-one-negative-out influence on gold58 (current base)
print('\n(e) influence: gold58 delta with each single study dropped (min / max over the 58 drops) and the share of the delta from the')
print('    top-2 positive studies (pairwise-contribution decomposition)')
for t, o, a in SHORT:
    jt, jo = J[t], J[o]; y = Y58[:, jt]; Dm = CL.K(CL.tr_score(CUR, jt, jo, a)) - CL.K(CUR[:, jt])
    full = CL.wdelta(Dm, y, np.ones((1, n58)))[0]
    W = np.ones((n58, n58)) - np.eye(n58); drops = CL.wdelta(Dm, y, W)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    contrib = Dm[np.ix_(pos, neg)].sum(1) / (len(pos) * len(neg))   # per-positive contribution to the delta
    top2 = np.sort(contrib)[::-1][:2].sum()
    print('    %-14s a %+.1f full %+.4f | drop-one min %+.4f max %+.4f | top-2 positives carry %+.4f (%.0f%%), positives with + / - / 0 contribution %d / %d / %d' % (
        AB[t] + ' <- ' + AB[o], a, full, np.nanmin(drops), np.nanmax(drops), top2, 100 * top2 / full if full else 0,
        (contrib > 1e-12).sum(), (contrib < -1e-12).sum(), (np.abs(contrib) <= 1e-12).sum()))
