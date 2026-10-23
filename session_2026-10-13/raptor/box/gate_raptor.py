# gate_raptor.py (box CPU): the PREREG_RAPTOR_SEED gate (sha256 bc2cef53..., written before any GPU number).
# Inputs: the full run dir (train.log, summary.json, swa.pt) and the SV predictions of S and N on gold58 written by
# serve_check.py (same 58-study order: gold74_eval_set.csv original_gold58 rows in file order). F = K14 "R" fleet
# (C + DO + E + E2, 20 members) gold58 pass-A predictions, equal mean, loaded with k14_gate.py's own helpers.
# Usage: python gate_raptor.py --run ~/raptor_store/run_s42 --s_pred S.npy --n_pred N.npy [--out ~/raptor2/GATE_RESULT.txt]
import os, sys, json, re, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
H = os.path.expanduser('~')
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True); ap.add_argument('--s_pred', required=True); ap.add_argument('--n_pred', required=True)
ap.add_argument('--out', default=f'{H}/raptor2/GATE_RESULT.txt'); ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--planned_epochs', type=int, required=True)
A = ap.parse_args()
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
     "Baker's", 'Contusion', 'Fracture']
L = []
def say(s): print(s, flush=True); L.append(s)
g = pd.read_csv(f'{H}/rsna_train/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
g = g[g.source == 'original_gold58'].set_index('StudyInstanceUID'); ids = list(g.index); Y = g[T].values.astype(float)
S = np.load(A.s_pred); N = np.load(A.n_pred); assert S.shape == N.shape == (58, 12)
# F: K14 R fleet, equal mean of the 20 members' gold58 pass-A predictions (k14_gate.py helpers, read only)
sys.path.insert(0, f'{H}/k14_test/k14/box'); import k14_gate as kg
M = json.load(open(f'{H}/k14_test/members.json'))['members']
Fm = [kg.rd(kg.pred_path(r, f'{H}/k14_test/gold/preds')).loc[ids, T].to_numpy(np.float64) for r in M if r['set'] in kg.FLEETS['R']]
assert len(Fm) == 20, len(Fm); F = np.mean(Fm, axis=0)
auc = lambda P, ii=slice(None): float(np.mean([roc_auc_score(Y[ii, j], P[ii, j]) for j in range(12)]))
rk = lambda P: np.stack([rankdata(P[:, j]) / len(P) for j in range(12)], 1)
mix = (rk(N) + rk(S)) / 2
B_S = .8 * rk(mix) + .2 * rk(F); B_N = .8 * rk(N) + .2 * rk(F)
# G0 / G1 from the run
tl = open(f'{A.run}/train.log').read(); sm = json.load(open(f'{A.run}/summary.json'))
ep_done = len(re.findall(r'^=== ep \d+ gold58', tl, re.M)); nonf = sm['nonfinite_loss_steps'] / max(1, sm['steps'])
g0 = ep_done >= A.planned_epochs and nonf <= .01 and 'SERVE-LOAD OK' in tl and np.isfinite(S).all() and np.isfinite(N).all()
g1v = sm['swa_gold58']
a_S, a_N, a_F = auc(S), auc(N), auc(F)
g3v, g5S, g5N = auc(mix), auc(B_S), auc(B_N)
rho = float(np.mean([spearmanr(S[:, j], N[:, j]).correlation for j in range(12)]))
legs = [('G0 integrity', g0, f'epochs {ep_done}/{A.planned_epochs}, nonfinite {nonf:.4f}'),
        ('G1 SWA gold58 train geometry >= .9050', g1v >= .9050, f'{g1v:.4f}'),
        ('G2 AUC_SV(S) >= AUC_SV(N) - .0150', a_S >= a_N - .0150, f'S {a_S:.4f} N {a_N:.4f} d {a_S - a_N:+.4f}'),
        ('G3 rank-mean(N,S) >= AUC_SV(N) - .0020', g3v >= a_N - .0020, f'{g3v:.4f} d {g3v - a_N:+.4f}'),
        ('G4 mean Spearman(S,N) <= .970', rho <= .970, f'{rho:.4f}'),
        ('G5 [mix .8 + F .2] >= [N .8 + F .2] - .0020', g5S >= g5N - .0020, f'{g5S:.4f} vs {g5N:.4f} d {g5S - g5N:+.4f}')]
for name, ok, v in legs:
    say(f'{"PASS" if ok else "FAIL"} {name}: {v}')
rng = np.random.default_rng(13); bs = [rng.integers(0, 58, 58) for _ in range(A.boot)]
def se(fa, fb):
    d = []
    for b in bs:
        try: d.append(auc(fa, b) - auc(fb, b))
        except ValueError: pass
    return float(np.std(d)), float(np.mean(np.array(d) > 0))
for nm, (x, y) in {'S - N': (S, N), 'mix - N': (mix, N), 'G5 S-side - N-side': (B_S, B_N)}.items():
    s_, p_ = se(x, y); say(f'  report: {nm} bootstrap SE {s_:.4f}, P(>0) {p_:.2f}')
say('  report per-label S: ' + ' '.join(f'{t[:6]}:{roc_auc_score(Y[:, j], S[:, j]):.3f}' for j, t in enumerate(T)))
say('  report per-label N: ' + ' '.join(f'{t[:6]}:{roc_auc_score(Y[:, j], N[:, j]):.3f}' for j, t in enumerate(T)))
say(f'  report: F alone {a_F:.4f} (k14 log .9212) | S train-geometry SWA {g1v:.4f} vs N stored .918952 / CPU re-score .919111')
v = 'PASS' if all(ok for _, ok, _ in legs) else ('PORT-BROKEN' if not (legs[0][1] and legs[1][1]) else 'FAIL')
say(f'RAPTOR_SEED_GATE {v} | prereg sha256 bc2cef530c4ed39f6c778686f9d8f8ce8d23b9b0030937afccfd0c4192aa5c6d')
open(A.out, 'w').write('\n'.join(L) + '\n')
