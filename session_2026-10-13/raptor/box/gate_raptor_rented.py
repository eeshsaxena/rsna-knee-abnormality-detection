# gate_raptor_rented.py (box CPU): PREREG_RAPTOR_RENTED gate (sha256 9117ee60..., written 10-13 ~21:00 before any seed read).
# Per seed (fetched + sha-verified by box_fetch.sh into ~/raptor_store/rented/s<seed>/): G0 integrity, G1 SWA gold58 >= .9000.
# Ensemble Rr = per-label rank-mean of the SV predictions of the passing seeds; legs G2-G5 vs N (SV, from serve_check.py on
# the box GPU) and F (K14 R fleet, k14_gate.py helpers). Report-only extras: per-seed SV, seed-to-seed Spearman, [Rr .8 + F_B .2].
# usage: python gate_raptor_rented.py --n_pred N_SV.npy [--seeds "43 44 45 46"] [--out ~/raptor2/GATE_RENTED_RESULT.txt]
import os, sys, json, re, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser('~')
ap = argparse.ArgumentParser(); ap.add_argument('--n_pred', required=True); ap.add_argument('--seeds', default='43 44 45 46')
ap.add_argument('--root', default=f'{H}/raptor_store/rented'); ap.add_argument('--out', default=f'{H}/raptor2/GATE_RENTED_RESULT.txt')
ap.add_argument('--boot', type=int, default=2000)
A = ap.parse_args()
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
L = []
def say(s): print(s, flush=True); L.append(s)
g = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID'); ids = list(g.index); Y = g[T].values.astype(float)
auc = lambda P, ii=slice(None): float(np.mean([roc_auc_score(Y[ii, j], P[ii, j]) for j in range(12)]))
rk = lambda P: np.stack([rankdata(P[:, j]) / len(P) for j in range(12)], 1)
rho = lambda a, b: float(np.mean([spearmanr(a[:, j], b[:, j]).correlation for j in range(12)]))
N = np.load(A.n_pred); assert N.shape == (58, 12) and np.isfinite(N).all()
sys.path.insert(0, f'{H}/k14_test/k14/box'); import k14_gate as kg
M = json.load(open(f'{H}/k14_test/members.json'))['members']
def fleet(sets):
    m = [kg.rd(kg.pred_path(r, f'{H}/k14_test/gold/preds')).loc[ids, T].to_numpy(np.float64) for r in M if r['set'] in sets]
    return np.mean(m, axis=0), len(m)
F, nF = fleet(kg.FLEETS['R']); FB, nFB = fleet(kg.FLEETS['B']); assert nF == 20 and nFB == 20
P = {}
for s in A.seeds.split():
    d = f'{A.root}/s{s}'
    if not os.path.isfile(f'{d}/FETCHED') or 'rc 0' not in open(f'{d}/FETCHED').read():
        say(f'FAIL s{s} G0: not fetched with rc 0 ({open(d + "/FETCHED").read().strip() if os.path.isfile(d + "/FETCHED") else "missing"})'); continue
    tl = open(f'{d}/train.log').read(); sm = json.load(open(f'{d}/summary.json'))
    ep = len(re.findall(r'^=== ep \d+ gold58', tl, re.M)); nonf = sm['nonfinite_loss_steps'] / max(1, sm['steps'])
    sv = glob.glob(f'{d}/sv/*.npy'); S = np.load(sv[0]) if len(sv) == 1 else None
    g0 = ep >= 16 and nonf <= .01 and 'SERVE-LOAD OK' in tl and S is not None and S.shape == (58, 12) and np.isfinite(S).all()
    g1 = sm['swa_gold58'] >= .9000
    say(f'{"PASS" if g0 and g1 else "FAIL"} s{s}: G0 {g0} (epochs {ep}, nonfinite {nonf:.4f}, SV {"ok" if S is not None else "missing"}) | '
        f'G1 SWA gold58 {sm["swa_gold58"]:.4f} (epochs {sm["swa_over"]}) | SV {auc(S) if S is not None else float("nan"):.4f}')
    if g0 and g1: P[s] = S
if len(P) == 0:
    say('RAPTOR_RENTED_GATE FAIL | no seed passed G0 + G1'); open(A.out, 'w').write('\n'.join(L) + '\n'); sys.exit(0)
Rr = np.mean([rk(p) for p in P.values()], axis=0)
mix = (rk(N) + rk(Rr)) / 2
a_R, a_N, a_m = auc(Rr), auc(N), auc(mix); r_RN = rho(Rr, N)
g5S = auc(.8 * rk(mix) + .2 * rk(F)); g5N = auc(.8 * rk(N) + .2 * rk(F))
legs = [('G2 AUC_SV(Rr) >= AUC_SV(N) - .0150', a_R >= a_N - .015, f'Rr {a_R:.4f} ({len(P)} seeds) N {a_N:.4f} d {a_R - a_N:+.4f}'),
        ('G3 rank-mean(N,Rr) >= AUC_SV(N) - .0020', a_m >= a_N - .002, f'{a_m:.4f} d {a_m - a_N:+.4f}'),
        ('G4 mean Spearman(Rr,N) <= .970', r_RN <= .970, f'{r_RN:.4f}'),
        ('G5 [mix .8 + F .2] >= [N .8 + F .2] - .0020', g5S >= g5N - .002, f'{g5S:.4f} vs {g5N:.4f} d {g5S - g5N:+.4f}')]
for name, ok, v in legs:
    say(f'{"PASS" if ok else "FAIL"} {name}: {v}')
rng = np.random.default_rng(13); bs = [rng.integers(0, 58, 58) for _ in range(A.boot)]
def se(x, y):
    d = []
    for b in bs:
        try: d.append(auc(x, b) - auc(y, b))
        except ValueError: pass
    return float(np.std(d)), float(np.mean(np.array(d) > 0))
for nm, (x, y) in {'Rr - N': (Rr, N), 'mix - N': (mix, N), 'G5 Rr-side - N-side': (.8 * rk(mix) + .2 * rk(F), .8 * rk(N) + .2 * rk(F))}.items():
    s_, p_ = se(x, y); say(f'  report: {nm} bootstrap SE {s_:.4f}, P(>0) {p_:.2f}')
ks = list(P)
if len(ks) > 1:
    say('  report: seed-to-seed Spearman ' + ' '.join(f's{a}~s{b} {rho(P[a], P[b]):.3f}' for i, a in enumerate(ks) for b in ks[i + 1:]))
say('  report per-label Rr: ' + ' '.join(f'{t[:6]}:{roc_auc_score(Y[:, j], Rr[:, j]):.3f}' for j, t in enumerate(T)))
say('  report per-label N : ' + ' '.join(f'{t[:6]}:{roc_auc_score(Y[:, j], N[:, j]):.3f}' for j, t in enumerate(T)))
say(f'  report: F (R) alone {auc(F):.4f} | F_B alone {auc(FB):.4f} | OAI-free [Rr .8 + F_B .2] {auc(.8 * rk(Rr) + .2 * rk(FB)):.4f} vs [N .8 + F_B .2] {auc(.8 * rk(N) + .2 * rk(FB)):.4f}')
v = 'PASS' if all(ok for _, ok, _ in legs) else 'FAIL'
say(f'RAPTOR_RENTED_GATE {v} | passing seeds {ks} | prereg sha256 9117ee6071dee8cbe0d00b3e96bdd6b75e17842041c8e2a8abad3b5a990ec937')
open(A.out, 'w').write('\n'.join(L) + '\n')
