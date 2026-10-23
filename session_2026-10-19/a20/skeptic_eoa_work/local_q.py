# Skeptic: c4 OA coupling (rho) and cross-predictiveness q on gold58; R7 OOF on train vs report labels. Local CPU only.
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OA = ['Medial OA', 'Lateral OA', 'PF OA']
pct = lambda v: rankdata(v) / len(v)
ns = lambda v: norm.ppf((rankdata(v) - .5) / len(v))
def auc(s, y):
    p = y.sum(); q = len(y) - p; r = rankdata(s); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
def decor(F, a):
    U = np.stack([pct(F[:, k]) for k in range(3)], 1); O = U.copy()
    for k in range(3):
        o = [m for m in range(3) if m != k]; O[:, k] = U[:, k] - a * (U[:, o[0]] + U[:, o[1]]) / 2
    return O
g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str}); g = g[g.source == 'original_gold58'].reset_index(drop=True)
ids = g.StudyInstanceUID.tolist(); n = len(ids)
rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
C4 = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
V = np.zeros((n, 12))
for j, c in enumerate(L):
    w = C4.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
p = R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb'
nb = json.load(open(p, encoding='utf-8')); s = nb['cells'][18]['source']; s = ''.join(s) if isinstance(s, list) else s
fn = s.split('\n\ntry:\n    _S6F_BLEND = ')[0]
W = tempfile.mkdtemp().replace(chr(92), '/'); fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
sha = lambda q: hashlib.sha256(open(q, 'rb').read()).hexdigest()
pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
nsd = {}; exec(fn, nsd); rec = nsd['_s6f_blend'](setup, run)
out = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids]
shutil.rmtree(W, ignore_errors=True)
F = out[OA].values.astype(float); Y = g[OA].values.astype(int)
def stats(F, Y):
    Z = np.stack([ns(F[:, k]) for k in range(3)], 1)
    rho = np.corrcoef(Z.T)[np.triu_indices(3, 1)]
    q = []
    for k in range(3):
        o = [m for m in range(3) if m != k]; ck = np.corrcoef(Z[:, k], Y[:, k])[0, 1]
        q.append(np.mean([np.corrcoef(Z[:, j], Y[:, k])[0, 1] for j in o]) / ck)
    return rho, np.array(q)
def theory(q, r, a):  # ratio of latent corr after/before for out = f_k - a*(f_j+f_l)/2, equicorrelated f
    return (1 - a * q) / np.sqrt(1 - 2 * a * r + a * a * (1 + r) / 2)
def a_be(q, r): return 2 * (r - q) / ((1 + r) / 2 - q * q)
rho, q = stats(F, Y)
base = np.array([auc(F[:, k], Y[:, k]) for k in range(3)])
print('c4 final gold58: rho(ns) ML/MPF/LPF %s mean %.3f | q per label M/L/PF %s | base AUC %s' % (np.round(rho, 3).tolist(), rho.mean(), np.round(q, 3).tolist(), np.round(base, 4).tolist()))
for k in range(3):
    o = [m for m in range(3) if m != k]; rk = np.mean([np.corrcoef(ns(F[:, k]), ns(F[:, j]))[0, 1] for j in o])
    print('  %-10s q %.3f rho_k %.3f a_break-even %.3f' % (OA[k], q[k], rk, a_be(q[k], rk)))
rng = np.random.default_rng(0); QB = []; RB = []; DB = {.1: [], .2: []}
for b in range(2000):
    ii = rng.integers(0, n, n); y = Y[ii]
    if (y.sum(0) < 2).any(): continue
    r_, q_ = stats(F[ii], y); QB.append(q_.mean()); RB.append(r_.mean())
QB = np.array(QB); RB = np.array(RB)
print('bootstrap (gold58): q mean %.3f sd %.3f 90%% [%.3f, %.3f] | rho mean %.3f | P(q < rho) %.3f | P(a_be > .2) %.3f | P(a_be > .1) %.3f' % (
    QB.mean(), QB.std(), *np.quantile(QB, [.05, .95]), RB.mean(), np.mean(QB < RB), np.mean(a_be(QB, RB) > .2), np.mean(a_be(QB, RB) > .1)))
# R7 OOF on train vs report labels (v2 YES/NO only, UNK dropped per label) and on gold58
r7 = pd.read_csv(G + '/R7OOF.csv', dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
F7 = r7[['p|' + c for c in OA]].set_axis(OA, axis=1)
rep = pd.read_csv(R + '/_labels/rsna-knee-llm-labels/report_labels_v2.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
V2 = rep[[c + '__verdict' for c in OA]].set_axis(OA, axis=1)
common = F7.index.intersection(V2.index)
Z7 = np.stack([ns(F7.loc[common, c].values) for c in OA], 1)
print('R7 OOF train n %d rho(ns) ML/MPF/LPF %s' % (len(common), np.round(np.corrcoef(Z7.T)[np.triu_indices(3, 1)], 3).tolist()))
full = V2.loc[common].isin(['YES', 'NO']).all(1).values
Yr = (V2.loc[common][full] == 'YES').values.astype(int); Fr = F7.loc[common][full].values
phr = np.corrcoef(Yr.T)[np.triu_indices(3, 1)]
rr, qr = stats(Fr, Yr)
print('R7 OOF vs report v2 (no UNK in any OA) n %d prev %s label phi %s | rho %s | q %s | theory a_be %s' % (
    len(Yr), np.round(Yr.mean(0), 3).tolist(), np.round(phr, 3).tolist(), np.round(rr, 3).tolist(), np.round(qr, 3).tolist(),
    np.round([a_be(qr[k], rr.mean()) for k in range(3)], 3).tolist()))
bR = np.array([auc(Fr[:, k], Yr[:, k]) for k in range(3)])
for a in (.1, .2):
    O = decor(Fr, a); print('  report-regime E-OA a %.1f OA-mean dAUC %+.5f (theory corr-ratio %s)' % (a, np.mean([auc(O[:, k], Yr[:, k]) for k in range(3)]) - bR.mean(),
        np.round([theory(qr[k], rr.mean(), a) for k in range(3)], 4).tolist()))
F7g = F7.reindex(ids); ok = F7g.notna().all(1).values
r7g, q7g = stats(F7g.values[ok], Y[ok])
print('R7 OOF gold58 n %d rho %s q %s' % (ok.sum(), np.round(r7g, 3).tolist(), np.round(q7g, 3).tolist()))
for a in (.1, .2):
    O = decor(F, a); d = np.mean([auc(O[:, k], Y[:, k]) for k in range(3)]) - base.mean()
    print('c4 gold58 E-OA a %.1f OA-mean dAUC %+.5f macro %+.5f | theory corr ratio per label %s' % (a, d, d / 4, np.round([theory(q[k], rho.mean(), a) for k in range(3)], 4).tolist()))
