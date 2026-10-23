# cbase.py (10-20 push, Claude). RESEARCH ONLY, local CPU. Rebuild the FINAL outputs of the scored c4 (57343435) and a20 eoa20
# notebooks on gold58 by exec'ing their exact cell-18 _s6f_blend code (same method as a21/gate_a21_offline.py; parity: gold58 macro
# c4 .94113, eoa20 .94360). Returns 58 x 12 arrays in label order L plus labels / ids. Writes nothing outside a temp dir.
import json, os, hashlib, tempfile, shutil
import numpy as np, pandas as pd
from scipy.stats import rankdata
R = 'C:/Users/eeshs/Downloads/rsna-knee'
G = R + '/session_2026-10-18/next/g58'
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
     "Contusion", "Fracture"]
AB = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
      "PF OA": "PFOA", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Cont", "Fracture": "Frac"}
pct = lambda v: rankdata(v) / len(v)
NB = {'c4': R + '/session_2026-10-18/a18/kernel_v224r6a25c4j/rsna_sub_v224r6a25c4j.ipynb',
      'eoa20': R + '/session_2026-10-19/a20/kernel_v224r6c4eoa20j/rsna_sub_v224r6c4eoa20j.ipynb'}


def load():
    g = pd.read_csv(G + '/gold74_eval_set.csv', dtype={'StudyInstanceUID': str})
    g58 = g[g.source == 'original_gold58'].reset_index(drop=True)
    ids = g58.StudyInstanceUID.tolist(); n = len(ids)
    rd = lambda f: pd.read_csv(G + '/' + f, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values
    R6 = rd('R6_gold58_serve.csv'); S6 = rd('n42gpu_gold58.csv')
    R384 = np.load(G + '/gold58_r384_kernelserve_cpu.npy'); R224 = np.load(G + '/gold58_r224_kernelserve.npy')
    C4W = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .5, "Synovitis": .8, "Baker's": .5, "Lateral Meniscus": .4, "Fracture": .65}
    V = np.zeros((n, 12))
    for j, c in enumerate(L):
        w = C4W.get(c, .5); rr = .75 * pct(R384[:, j]) + .25 * pct(R224[:, j]); V[:, j] = w * rr + (1 - w) * pct(R6[:, j])
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    out = {}
    for k, p in NB.items():
        nb = json.load(open(p, encoding='utf-8')); s_ = nb['cells'][18]['source']; s_ = ''.join(s_) if isinstance(s_, list) else s_
        fn = s_.split('\n\ntry:\n    _S6F_BLEND = ')[0]
        W = tempfile.mkdtemp().replace(chr(92), '/'); assert fn.count("W = '/kaggle/working'") == 1
        fn = fn.replace("W = '/kaggle/working'", "W = %r" % W)
        pd.DataFrame(V.astype(np.float32), columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/submission_v224flip.csv', index=False)
        os.makedirs(W + '/s6f/B'); pd.DataFrame(S6, columns=L).assign(StudyInstanceUID=ids)[['StudyInstanceUID'] + L].to_csv(W + '/s6f/B/B_probs.csv', index=False)
        setup = {'base_sha256': sha(W + '/submission_v224flip.csv'), 'v224_canary': {'fired': []}}
        run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W + '/s6f/B/B_probs.csv')}}}
        ns = {}; exec(fn, ns); ns['_s6f_blend'](setup, run)
        o = pd.read_csv(W + '/submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[ids, L].values.astype(np.float64)
        out[k] = o; shutil.rmtree(W, ignore_errors=True)
    return dict(g74=g, g58=g58, ids=ids, Y58=g58[L].values.astype(int), Y74=g[L].values.astype(int), ids74=g.StudyInstanceUID.tolist(),
                c4=out['c4'], eoa20=out['eoa20'], members=dict(R6=R6, S6=S6, R384=R384, R224=R224))


if __name__ == '__main__':
    from scipy.stats import rankdata as rk
    def auc(s, y):
        r = rk(s); p = y.sum(); q = len(y) - p; return (r[y == 1].sum() - p * (p + 1) / 2) / (p * q)
    d = load()
    for k in ('c4', 'eoa20'):
        print(k, 'gold58 macro %.5f' % np.mean([auc(d[k][:, j], d['Y58'][:, j]) for j in range(12)]))
