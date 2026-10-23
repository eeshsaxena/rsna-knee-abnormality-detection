# m5_nested.py (10-19 trick lens, research only, CPU): NESTED validation on gold58 of "pick a radiologist-justified cross-label
# transform on expert labels". Family = pre-registered mechanism pairs (target <- helper, sign), written before running:
#   Synovitis<-Effusion(+)  Effusion<-Synovitis(+)  Effusion<-Baker's(+)  Baker's<-Effusion(+)  Contusion<-ACL(+)  Contusion<-Fracture(+)
#   Fracture<-Contusion(+)  ACL<-Contusion(+)  Medial OA<-Medial Meniscus(+)  Lateral OA<-Lateral Meniscus(+)  Lateral OA<-PF OA(-)
#   (host 733491: lateral patellar facet chondrosis reported as "lateral compartment")  Medial OA<-PF OA(+)  PF OA<-Medial OA(+)
# transform on the FINAL c4 output: s_t' = pct(s_t) + a * sign * pct(s_h), a in {0, .05, .1, .2}.
# Procedures (selection on the inner half ONLY, scored on the outer half, 2-fold CV repeated, stratification-free):
#   P1 per-target free pick of (helper, a) incl. a = 0;   P2 one shared a for ALL family pairs at once;
#   P3 fixed E2 (Synovitis<-Effusion .1, no selection);    P4 fixed all-family at a = .1 (no selection)
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/stack2')
from load import L, auc, pct, gold, G
g, g58 = gold(); ids = g58.StudyInstanceUID.tolist(); Y = g58[L].values.astype(int); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
P = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); b = w * rr + (1 - w) * pct(R6[:, j]); P[:, j] = .8 * pct(b) + .2 * pct(S6[:, j])
FAM = [('Synovitis', 'Effusion', 1), ('Effusion', 'Synovitis', 1), ('Effusion', "Baker's", 1), ("Baker's", 'Effusion', 1),
       ('Contusion', 'ACL', 1), ('Contusion', 'Fracture', 1), ('Fracture', 'Contusion', 1), ('ACL', 'Contusion', 1),
       ('Medial OA', 'Medial Meniscus', 1), ('Lateral OA', 'Lateral Meniscus', 1), ('Lateral OA', 'PF OA', -1),
       ('Medial OA', 'PF OA', 1), ('PF OA', 'Medial OA', 1)]
A = [.05, .1, .2]
ix_ = {c: L.index(c) for c in L}


def t_auc(ix, t, h, sgn, a):
    s = pct(P[ix, ix_[t]]) + (a * sgn * pct(P[ix, ix_[h]]) if a else 0)
    return auc(s, Y[ix, ix_[t]])


def ok(ix): return all(0 < Y[ix, j].sum() < len(ix) for j in range(12))


# full-sample (in-sample) view of the family, for the record
print('in-sample gold58 per-pair delta (target AUC, /12 = macro):')
for t, h, s in FAM:
    full = np.arange(n); b = t_auc(full, t, h, s, 0)
    print('  %-10s <- %-16s %+d  ' % (t, h, s) + '  '.join('a%.2f %+.4f' % (a, t_auc(full, t, h, s, a) - b) for a in A))
rng = np.random.default_rng(20261019); out = {k: [] for k in ['P1', 'P2', 'P3', 'P4']}; picks = {}
targets = sorted(set(t for t, _, _ in FAM))
reps = 0
while reps < 1000:
    perm = rng.permutation(n); tr, te = perm[:29], perm[29:]
    if not (ok(tr) and ok(te)): continue
    reps += 1
    for (inner, outer) in [(tr, te), (te, tr)]:
        # P1: per target, pick the best (helper, a) on inner incl. no-change
        d1 = 0.
        for t in targets:
            cands = [(t_auc(inner, t, h, s, a) - t_auc(inner, t, h, s, 0), h, s, a) for (tt, h, s) in FAM if tt == t for a in A]
            best = max(cands, key=lambda z: z[0])
            if best[0] > 0:
                d1 += t_auc(outer, t, best[1], best[2], best[3]) - t_auc(outer, t, best[1], best[2], 0)
                picks[(t, best[1], best[3])] = picks.get((t, best[1], best[3]), 0) + 1
        out['P1'].append(d1 / 12)
        # P2: one shared a for all family pairs (pairs sharing a target are summed)
        def fam_delta(ix, a):
            dd = 0.
            for t in targets:
                s = pct(P[ix, ix_[t]]) + sum(a * sg * pct(P[ix, ix_[h]]) for (tt, h, sg) in FAM if tt == t)
                dd += auc(s, Y[ix, ix_[t]]) - auc(P[ix, ix_[t]], Y[ix, ix_[t]])
            return dd / 12
        inn = {a: fam_delta(inner, a) for a in A}; a_best = max(inn, key=inn.get)
        out['P2'].append(fam_delta(outer, a_best) if inn[a_best] > 0 else 0.)
        out['P3'].append((t_auc(outer, 'Synovitis', 'Effusion', 1, .1) - t_auc(outer, 'Synovitis', 'Effusion', 1, 0)) / 12)
        out['P4'].append(fam_delta(outer, .1))
for k, v in out.items():
    v = np.array(v); print('%s outer-half macro delta: mean %+.5f  sd %.5f  P>0 %.3f  (n %d)' % (k, v.mean(), v.std(), (v > 0).mean(), len(v)))
print('P1 most frequent inner picks:', sorted(picks.items(), key=lambda z: -z[1])[:8])
