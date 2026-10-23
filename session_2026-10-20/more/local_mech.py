# local_mech.py (10-20 more, Claude): RESEARCH ONLY, local CPU. Lens 1 + 2 on the two local rulers.
#  (A) report-label OOF ruler (mechanism / negative control for E-OA; the only prior support for E1): R7 OOF + s1044 + s1045 OOF,
#      readers v4 / v2 / sol56, c4-shaped 3-member nests and a flat 3-member mean, all 12 labels.
#  (B) gold58 c4 replay with the REAL members (R6 serve, Raptor r384 / r224, S6fall n42gpu): code + consistency check only,
#      NOT a selection ruler. Expected parity: c4 .94102 (float64), E1 kernel-natural .94111 (THINK_1019 sec 8.4).
# Writes only its own .log (stdout redirect). Pre-registration: mechanics.md sec 0.
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import *
OA = ['Medial OA', 'Lateral OA', 'PF OA']; IOA = [L.index(c) for c in OA]
C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
WV = np.array([C4W.get(c, .5) for c in L])
pctc = lambda X: np.column_stack([pct(X[:, j]) for j in range(X.shape[1])])


def eoa(U3, a):
    U = pctc(U3); out = np.empty_like(U)
    for t in range(3):
        o = [k for k in range(3) if k != t]; out[:, t] = pct(U[:, t] - a * (U[:, o[0]] + U[:, o[1]]) / 2.0)
    return out


def nest3(A, B, C, e, place, a):
    # c4 shape with 3 members: V = w pct(B)^e + (1 - w) pct(A)^e  (B in the Raptor slot, A in the R6 slot); final = .8 pct(V)^e + .2 pct(C)^e
    P = {'A': pctc(A), 'B': pctc(B), 'C': pctc(C)}
    if place == 'MEMBER' and a:
        for k in P: P[k][:, IOA] = eoa(P[k][:, IOA], a)
    V = WV * P['B'] ** e + (1 - WV) * P['A'] ** e
    rb = pctc(V)
    if place == 'V224' and a: rb[:, IOA] = eoa(rb[:, IOA], a)
    out = .8 * rb ** e + .2 * P['C'] ** e
    if place == 'FINAL' and a: out[:, IOA] = eoa(out[:, IOA], a)
    return out


def c4real(R384, R224, R6, S6, e, place, a):
    P = {'r384': pctc(R384), 'r224': pctc(R224), 'R6': pctc(R6), 'S6': pctc(S6)}
    if place == 'MEMBER' and a:
        for k in P: P[k][:, IOA] = eoa(P[k][:, IOA], a)
    rr = .75 * P['r384'] ** e + .25 * P['r224'] ** e
    V = WV * rr + (1 - WV) * P['R6'] ** e
    rb = pctc(V)
    if place == 'V224' and a: rb[:, IOA] = eoa(rb[:, IOA], a)
    out = .8 * rb ** e + .2 * P['S6'] ** e
    if place == 'FINAL' and a: out[:, IOA] = eoa(out[:, IOA], a)
    return out


def flat3(Ms, e, place, a):
    P = [pctc(X) for X in Ms]
    if place == 'MEMBER' and a:
        for X in P: X[:, IOA] = eoa(X[:, IOA], a)
    out = np.mean([X ** e for X in P], 0)
    if place == 'FINAL' and a: out[:, IOA] = eoa(out[:, IOA], a)
    return out


def st(x):
    x = np.asarray(x); return '%+.5f (sd %.5f P %.2f)' % (x.mean(), x.std(), (x > 0).mean())


AG = (.1, .2)


def columns(places):
    return [(e, 'NONE', 0.) for e in (1, 2)] + [(e, pl, a) for e in (1, 2) for pl in places for a in AG]


def evaluate(S, lab, boots, tag, places, all_labels=True):
    """S: dict col -> (N, 12) scores; lab (N, 12) with NaN. Per-label AUC per bootstrap, then macro deltas (/12)."""
    cols = list(S); N = lab.shape[0]
    def per_label(ix):
        out = np.full((len(cols), 12), np.nan)
        for j in range(12):
            m = ~np.isnan(lab[ix, j]); y = lab[ix, j][m].astype(int)
            if 5 <= y.sum() <= len(y) - 5:
                for ci, c in enumerate(cols): out[ci, j] = auc(S[c][ix, j][m], y)
        return out
    full = per_label(np.arange(N)); BB = np.array([per_label(ix) for ix in boots])   # (B, ncol, 12)
    ci = {c: i for i, c in enumerate(cols)}
    def mac(c, c0, sel=None):
        sel = range(12) if sel is None else sel
        d = np.nansum(BB[:, ci[c], :][:, list(sel)] - BB[:, ci[c0], :][:, list(sel)], 1) / 12.
        f = np.nansum(full[ci[c], list(sel)] - full[ci[c0], list(sel)]) / 12.
        return f, d
    b1 = (1, 'NONE', 0.); b2 = (2, 'NONE', 0.)
    f, d = mac(b2, b1); print('%s | E1 (pow2) all labels %+.5f boot %s | per-label %s' % (
        tag, f, st(d), np.round(full[ci[b2]] - full[ci[b1]], 4).tolist()))
    f, d = mac(b2, b1, IOA); print('%s | E1 OA trio only %+.5f boot %s' % (tag, f, st(d)))
    for a in AG:
        f1, g1 = mac((1, 'FINAL', a), b1, IOA); f2, g2 = mac((2, 'FINAL', a), b2, IOA); fs, ds = mac((2, 'FINAL', a), (1, 'FINAL', a))
        print('%s | a %.1f | E-OA gain e1 %+.5f %s | e2 %+.5f %s | interaction %+.5f %s | STACK (E1+E-OA)-(E-OA) all labels %+.5f %s' % (
            tag, a, f1, st(g1), f2, st(g2), f2 - f1, st(g2 - g1), fs, st(ds)))
        for pl in places[1:]:
            fp, dp = mac((1, pl, a), (1, 'FINAL', a), IOA)
            print('%s | a %.1f | place %-6s vs FINAL same a %+.5f %s' % (tag, a, pl, fp, st(dp)))
    return full


def oof():
    d7 = r7(); s44, _ = s6seed(1044); s45, _ = s6seed(1045)
    ii = [i for i in s45.index if i in s44.index and i in d7.index]
    A = d7.loc[ii, ['p|' + c for c in L]].values.astype(np.float64); B = s44.loc[ii, L].values.astype(np.float64); C = s45.loc[ii, L].values.astype(np.float64)
    N = len(ii); rng = np.random.default_rng(31); boots = [rng.integers(0, N, N) for _ in range(300)]
    print('=== (A) report OOF ruler | n studies %d | members A R7 OOF, B s1044 OOF, C s1045 OOF ===' % N, flush=True)
    for src in ('v4', 'v2', 'sol56'):
        lab = labels(src).loc[ii, L].values.astype(np.float64)
        for nm, fn, places in (('NEST_B', lambda e, pl, a: nest3(A, B, C, e, pl, a), ['FINAL', 'V224', 'MEMBER']),
                               ('NEST_C', lambda e, pl, a: nest3(A, C, B, e, pl, a), ['FINAL', 'V224', 'MEMBER']),
                               ('FLAT3', lambda e, pl, a: flat3([A, B, C], e, pl, a), ['FINAL', 'MEMBER'])):
            S = {c: fn(*c) for c in columns(places)}
            evaluate(S, lab, boots, 'OOF %-5s %-6s' % (src, nm), places)
            sys.stdout.flush()


def g58():
    g, g58_ = gold(); ids = g58_.StudyInstanceUID.tolist(); Y = g58_[L].values.astype(np.float64); n = len(ids)
    rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)
    R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
    R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy').astype(np.float64); R224 = np.load(G + '/gold58_r224_kernelserve.npy').astype(np.float64)
    places = ['FINAL', 'V224', 'MEMBER']
    S = {c: c4real(R384, R224, R6, S6, *c) for c in columns(places)}
    mac = lambda P: np.mean([auc(P[:, j], Y[:, j].astype(int)) for j in range(12)])
    print('=== (B) gold58 c4 replay, real members (code check only) | n %d ===' % n)
    print('parity: c4 e1 macro %.5f (expect .94102) | E1 e2 macro %.5f (expect .94111)' % (mac(S[(1, 'NONE', 0.)]), mac(S[(2, 'NONE', 0.)])))
    for c in S:
        if c[1] != 'NONE': print('   %s macro %.5f delta vs c4 %+.5f' % (c, mac(S[c]), mac(S[c]) - mac(S[(1, 'NONE', 0.)])))
    rng = np.random.default_rng(20261020); boots = []
    while len(boots) < 2000:
        ix = rng.integers(0, n, n)
        if all(0 < Y[ix, j].sum() < n for j in range(12)): boots.append(ix)
    evaluate(S, Y, boots, 'g58', places)
    # coupling of the members and of c4 on gold58 (Spearman of OA columns), for lens 3 context
    for nm, X in (('R6', R6), ('r384', R384), ('r224', R224), ('S6', S6), ('c4', S[(1, 'NONE', 0.)]), ('c4_E1', S[(2, 'NONE', 0.)])):
        U = pctc(X[:, IOA]); Cm = np.corrcoef(U.T); print('   gold58 OA Spearman %-6s ML %.3f MPF %.3f LPF %.3f mean %.3f' % (nm, Cm[0, 1], Cm[0, 2], Cm[1, 2], (Cm[0, 1] + Cm[0, 2] + Cm[1, 2]) / 3))


if __name__ == '__main__':
    g58()
    oof()
