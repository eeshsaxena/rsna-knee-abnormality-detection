"""A_blend_model.py (2026-10-08, research_1008 dimension A). MODEL only, laptop-light.

Macro single-d binormal rank-blend model, same as SP/p949/p949_model.py (validated on 11 OOF rank-blend pairs,
error +.0000 to +.0009, always optimistic):  AUC = Phi(d/sqrt2), blend AUC = Phi(w.d / sqrt(2 w'Rw)).
Displayed public reads x are intervals [x-.0005, x+.0005).

Legs: P = public .943 parent stack, F = our 10-member ConvNeXt-T fleet (K9-like, SOLO .944), N = nartaa .949 CoAtNet
(OAI), G = goodpjw2008 2.5D ConvNeXt-T reader (3 folds, .929 alone).

MEASURED anchor reads used for calibration (sources in A_public_assets.md):
  P .943, F .944, P+F (W .50) .946                         (our own submissions)
  N .949                                                      (nartaa 56906959)
  N .60 + P .40 = .949                                        (goodpjw2008 0-949-coatnet-stack-blend notebook table, 10-08)
  G .929; P .70 + G .30 = .944; P .85 + G .15 = .944; P .55 + G .45 = .942   (goodpjw2008 dataset + notebook text)
  N + G per-label routing (mean w_G .145 .. .25) = .950; N .78 + G .22 flat = .950 (kozykappa)   (9 public notebooks)
Unknown: r_FN, r_FG (scanned).
"""
import itertools
import numpy as np
from scipy.stats import norm

def d(a): return np.sqrt(2) * norm.ppf(a)
def blend(s, R, w):
    w = np.asarray(w, float); dv = np.array([d(x) for x in s])
    return norm.cdf((w @ dv) / np.sqrt(2 * (w @ R @ w)))
def b2(a, b, r, wb):
    R = np.array([[1, r], [r, 1]]); return blend((a, b), R, (1 - wb, wb))
def inside(x, disp): return (x >= disp - .0005) & (x < disp + .0005)

RG = np.round(np.arange(.70, .999, .001), 3)
out = []
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

def calib(name, a, da, b, db, reads):
    """reads: list of (w_b, displayed). Returns r values consistent with every read for some a, b in their intervals."""
    ok = set()
    for A in np.linspace(a - .0005, a + .0004999, 5):
        for B in np.linspace(b - .0005, b + .0004999, 5):
            for r in RG:
                if all(inside(b2(A, B, r, wb), disp) for wb, disp in reads):
                    ok.add(r)
    ok = sorted(ok)
    pr(f'{name}: r in [{ok[0]:.3f}, {ok[-1]:.3f}]  (n={len(ok)})' if ok else f'{name}: NO consistent r')
    return ok

rPF = calib('r_PF  (P+F .50 = .946)', .943, 0, .944, 0, [(.5, .946)])
rPN = calib('r_PN  (P .40 + N .60 = .949)', .943, 0, .949, 0, [(.6, .949)])
rPG = calib('r_PG  (P+G at .15/.30 = .944, .45 = .942)', .943, 0, .929, 0, [(.15, .944), (.30, .944), (.45, .942)])
rNG = calib('r_NG  (N+G at .145 / .22 = .950)', .949, 0, .929, 0, [(.145, .950), (.22, .950)])
rNG_loose = calib('r_NG  (only flat .22 = .950)', .949, 0, .929, 0, [(.22, .950)])

mid = lambda v: v[len(v) // 2]
pr(f'\npoints: r_PF {mid(rPF)}, r_PN {mid(rPN)}, r_PG {mid(rPG)}, r_NG {mid(rNG)}')

S = dict(P=.943, F=.944, N=.949, G=.929)
order = 'PFNG'
cands = {
    'N alone': dict(N=1),
    'Apex-like N.80 G.20': dict(N=.8, G=.2),
    'P.40 N.60 (goodpjw .949)': dict(P=.4, N=.6),
    'K9N-a P.10 F.30 N.60': dict(P=.1, F=.3, N=.6),
    'K9N-b P.30 F.30 N.40': dict(P=.3, F=.3, N=.4),
    'F.30 N.70': dict(F=.3, N=.7),
    'X1 P.05 F.20 N.60 G.15': dict(P=.05, F=.2, N=.6, G=.15),
    'X2 F.25 N.60 G.15': dict(F=.25, N=.6, G=.15),
    'X3 F.20 N.65 G.15': dict(F=.2, N=.65, G=.15),
    'X4 P.10 F.20 N.55 G.15': dict(P=.1, F=.2, N=.55, G=.15),
}

def Rmat(rpf, rpn, rpg, rfn, rfg, rng):
    return np.array([[1, rpf, rpn, rpg], [rpf, 1, rfn, rfg], [rpn, rfn, 1, rng], [rpg, rfg, rng, 1]])

def best4(R, step=.05):
    best = (None, -1)
    for wp in np.arange(0, .31, step):
        for wf in np.arange(0, .61, step):
            for wg in np.arange(0, .41, step):
                wn = 1 - wp - wf - wg
                if wn < .3: continue
                b = blend([S[k] for k in order], R, (wp, wf, wn, wg))
                if b > best[1]: best = ((round(wp, 2), round(wf, 2), round(wn, 2), round(wg, 2)), b)
    return best

pr('\nScenario table (r_PF, r_PN, r_PG, r_NG at calibrated points; r_FN, r_FG scanned). AUC is MODEL, optimistic ~+.0005.')
hdr = 'r_FN  r_FG | ' + ' | '.join(k.split(' ')[0] for k in cands) + ' | best4 (P,F,N,G) -> AUC'
pr(hdr)
res = {}
for rfn in (.86, .88, .90, .92, .94):
    for rfg in (.86, .89, .92, .95):
        R = Rmat(mid(rPF), mid(rPN), mid(rPG), rfn, rfg, mid(rNG))
        if np.linalg.eigvalsh(R).min() < -1e-9:
            pr(f'{rfn:.2f}  {rfg:.2f} | not PSD'); continue
        vals = []
        for k, w in cands.items():
            ww = [w.get(x, 0) for x in order]
            vals.append(blend([S[x] for x in order], R, ww)); res.setdefault(k, []).append(vals[-1])
        bw, bb = best4(R)
        res.setdefault('best4', []).append(bb)
        pr(f'{rfn:.2f}  {rfg:.2f} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {bw} -> {bb:.4f}')

pr('\nSummary over PSD scenarios (min / median / max), MODEL:')
for k, v in res.items():
    v = np.array(v); pr(f'  {k:28s} {v.min():.4f} / {np.median(v):.4f} / {v.max():.4f}')

# Robustness: sweep calibrated r over their full intervals for two key comparisons
pr('\nRobustness (all calibrated r over full intervals, r_FN .86-.94, r_FG .86-.95): P(X2 > Apex-like), P(K9N-a > Apex-like)')
cnt = np.zeros(5); tot = 0
ends = lambda v: (v[0], mid(v), v[-1])
for rpf, rpn, rpg, rng in itertools.product(ends(rPF), ends(rPN), ends(rPG), ends(rNG)):
    for rfn in (.86, .90, .94):
        for rfg in (.86, .90, .95):
            R = Rmat(rpf, rpn, rpg, rfn, rfg, rng)
            if np.linalg.eigvalsh(R).min() < -1e-9: continue
            f = lambda w: blend([S[x] for x in order], R, [w.get(x, 0) for x in order])
            apex = f(cands['Apex-like N.80 G.20']); x2 = f(cands['X2 F.25 N.60 G.15'])
            k9a = f(cands['K9N-a P.10 F.30 N.60']); x1 = f(cands['X1 P.05 F.20 N.60 G.15'])
            fn70 = f(cands['F.30 N.70']); x3 = f(cands['X3 F.20 N.65 G.15'])
            cnt += [x2 > apex, k9a > apex, x1 > apex, x3 > k9a, fn70 >= k9a]; tot += 1
cnt = np.append(cnt, 0) if len(cnt) < 5 else cnt
pr(f'  scenarios {tot}: X2>Apex {cnt[0]/tot:.2f}, K9N-a>Apex {cnt[1]/tot:.2f}, X1>Apex {cnt[2]/tot:.2f}, '
   f'X3>K9N-a {cnt[3]/tot:.2f}, F.30N.70>=K9N-a {cnt[4]/tot:.2f}')
open(__file__.replace('.py', '.out'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
