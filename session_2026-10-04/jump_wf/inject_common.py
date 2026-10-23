"""Shared loader + exact re-implementation of the parent's (0.943) blend graph on the gold58 box mimic run.
Formulas transcribed from rsna_sub_k2_fleet.ipynb cells 16 (DINO), 23-24 (A5), 26 (Rad + calibrator), 28 (Raptor,
CoAt family, Raptor x CoAt hybrid, final outer blend). CPU only, read only on the mimic run directory."""
import os, json, base64, zlib
import numpy as np, pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
H = os.path.expanduser('~')
R = H + '/kmimic/runs/gold58_gpu'
W = R + '/kaggle/working'
P = R + '/localLB_parts'
ROOT = R + '/kaggle/input/rsna-knee-abnormality-detection'
FLEET = H + '/k2_test/runs/C_gold58/working/submission_fleetsolo.csv'
TRAIN = H + '/kmimic/kaggle/input/rsna-knee-abnormality-detection/train.csv'

# ---- constants (cell refs)
RAPTOR_ARMS = ['maxspan-v5', 'native384dense-v10', 'maxspan-v5-reverse', 'native384-v8']   # c28 ARMS order
RAPTOR_W = np.array([0.60, 0.10, 0.10, 0.20])                                              # c28 ARMS[*]['w']
COAT_MEMBERS = ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3']              # c28 _members order
PRIVATE_ALPHA = 0.4                                                                       # c28 private_alpha
COATNET_W = {t: 0.60 for t in T}                                                          # c28 _coatnet_weight
COATNET_W.update({'ACL': 0.75, 'Medial Meniscus': 0.80, 'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75})
A5_W = 0.52                                                                               # c23
RAD_ALPHA, RAD_EXCLUDE, RAD_E13_W, RAD_SECOND_ALPHA = 0.55, ("Baker's",), 0.50, 0.20     # c26


def rank_pct(x):            # pandas rank(method='average', pct=True) per column (rsna_rankpct / _k2_rank)
    return pd.DataFrame(np.asarray(x, np.float64)).rank(method='average', pct=True).to_numpy(np.float64)


def rank01(x):              # rsna_rank01: (rank-1)/(n-1)
    x = np.asarray(x, np.float64)
    return (pd.DataFrame(x).rank(method='average').to_numpy() - 1) / (len(x) - 1)


def read_csv(path, ids):
    d = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert list(d.columns) == T, path
    return d.loc[ids, T].to_numpy(np.float64)


def npz_frame(path, ids, key='values'):
    z = np.load(path, allow_pickle=False)
    u = [str(s) for s in z['study_uids']]
    pos = {s: i for i, s in enumerate(u)}
    v = z[key]
    idx = [pos[s] for s in ids]
    return v[..., idx, :] if v.ndim == 3 else v[idx]


def load_all():
    ids = pd.read_csv(ROOT + '/test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    D = {'ids': ids}
    # stored stage outputs (byte copies of _pipeline_stage.csv after each phase, and the Raptor/CoAt files)
    D['st1'] = read_csv(P + '/pipeline_stage_1_dino_stack.csv', ids)
    D['st2'] = read_csv(P + '/pipeline_stage_2_dino_plus_a5.csv', ids)
    D['st3'] = read_csv(P + '/pipeline_stage_3_transformer_side_dino_a5_rad.csv', ids)
    D['st4'] = read_csv(P + '/pipeline_stage_4_final_outer_blend.csv', ids)
    D['parent_sub'] = read_csv(R + '/localLB_parent_preds.csv', ids)
    D['pub_csv'] = read_csv(P + '/raptor_public_rank.csv', ids)
    D['fam_csv'] = read_csv(P + '/coat_family_rank.csv', ids)
    D['side_csv'] = read_csv(P + '/raptor_coat_side_rank.csv', ids)
    # inputs
    man = json.load(open(H + '/kmimic/kaggle/input/rsna-knee-weights/manifest.json'))
    D['dino_ids'] = [m['id'] for m in man['members']]
    D['dino_fold'] = [m['fold'] for m in man['members']]
    D['dino_run'] = [m['source_run'] for m in man['members']]
    D['dino'] = np.stack([npz_frame(W + f'/diagnostics/dino_{m}.npz', ids) for m in D['dino_ids']])   # (20,n,12) f32
    D['a5_rank_mean'] = npz_frame(W + '/diagnostics/a5_rank_mean.npz', ids)
    D['a5_raw'] = npz_frame(W + '/speed_a5_raw.npz', ids, 'raw_probabilities')
    z = np.load(W + '/speed_a5_raw.npz'); pos = {str(s): i for i, s in enumerate(z['study_uids'])}
    D['a5_ok'] = np.array([bool(z['applicable'][pos[s]]) for s in ids])
    for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'):
        D[k] = npz_frame(W + f'/diagnostics/{k}.npz', ids)                                         # (5,n,12) f32
    D['raptor_raw'] = npz_frame(W + '/raptor_raw.npz', ids, 'raw_probabilities')                    # (4,n,12) f32
    cf = {'resgated_top3': W + '/coat_resgated_ep10_top3_predictions.npz',
          'global96_top3': W + '/g96_input/coatnet_global96_top3_predictions.npz',
          'd4_swa3': W + '/d4_input/coatnet_d4_depthzone_swa3_predictions.npz',
          'repairv1_top3': W + '/coatnet_repairv1_top3_predictions.npz'}
    D['coat_prob'] = {}
    for n_, f in cf.items():
        z = np.load(f, allow_pickle=False)
        prob = np.asarray(z['probability_mean'], np.float64) if 'probability_mean' in z.files else \
            np.asarray(z['raw_probabilities'], np.float64).mean(axis=0)
        pos = {str(s): i for i, s in enumerate(z['study_uids'])}
        D['coat_prob'][n_] = prob[[pos[s] for s in ids]]
    D['tser'] = pd.read_csv(ROOT + '/test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    exec(open(os.path.dirname(os.path.abspath(__file__)) + '/inject_radcal_payload.py').read(), globals())
    D['cal'] = json.loads(zlib.decompress(base64.b64decode(_RAD_CAL_PAYLOAD)).decode())   # noqa: F821
    # fleet + labels
    D['fleet'] = read_csv(FLEET, ids)
    tr = pd.read_csv(TRAIN, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    D['Y'] = tr.loc[ids, T].to_numpy(np.float64)
    assert not np.isnan(D['Y']).any()
    return D


# ---------------- graph stages ----------------
def stage1_dino(dino, order=None):           # c16 _combine (all weights 1.0 in manifest) + write_submission rank pct
    # _combine accumulates acc += rank*w member by member in thread-completion order, then acc/tot; float rounding
    # of that running sum decides near-ties, so the exact bytes depend on the (unlogged) completion order.
    acc = np.zeros(dino.shape[1:], np.float64)
    for i in (order if order is not None else range(len(dino))):
        acc += rank_pct(dino[i]) * 1.0
    return rank_pct(acc / float(len(dino)))


def stage2_a5(st1, a5_rank_mean, ok, w=A5_W):     # c23 _a5_blend_available, c24
    base = rank_pct(st1)
    a5 = np.asarray(a5_rank_mean, np.float32).astype(np.float64)      # c23: A5_PREDS = _a5_rank_mean.astype(float32)
    a5r = rank_pct(a5[ok])
    out = base.copy()
    out[ok] = (1 - w) * base[ok] + w * a5r
    return out


def protocol_features(tser, ids):            # c26 _rad_main protocol block
    pr = pd.DataFrame(index=pd.Index(ids, name='StudyInstanceUID'))
    pr['n_series'] = tser.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for pl in ('Sagittal', 'Coronal', 'Axial'):
        part = tser[tser.Anatomical_Plane.astype(str) == pl]
        pr[f'n_{pl[:3]}'] = part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    for flag in ('Fat_Suppression', 'Fluid_Sensitive'):
        marked = tser[pd.to_numeric(tser[flag], errors='coerce').fillna(0) > 0]
        pr[flag[:3]] = marked.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
        for pl in ('Sagittal', 'Coronal', 'Axial'):
            part = marked[marked.Anatomical_Plane.astype(str) == pl]
            pr[f'{flag[:3]}_{pl[:3]}'] = part.groupby('StudyInstanceUID').size().reindex(ids).fillna(0)
    return pr


def stage3_rad(st2, e10, e13, second, cal, protocol):   # c26 _rad_main
    reference_rank = rank_pct(e10.mean(0))               # predict_heads: f32 mean over 5 heads
    e13_rank = rank_pct(e13.mean(0))
    reference_rank = rank_pct((1 - RAD_E13_W) * reference_rank + RAD_E13_W * e13_rank)
    baseline_rank = rank_pct(st2)
    cand = st2.copy()
    for j, t in enumerate(T):
        if t not in RAD_EXCLUDE:
            cand[:, j] = (1 - RAD_ALPHA) * baseline_rank[:, j] + RAD_ALPHA * reference_rank[:, j]
    second_rank = rank_pct(second.mean(0))
    branch = rank_pct((1 - RAD_SECOND_ALPHA) * rank_pct(cand) + RAD_SECOND_ALPHA * second_rank)
    assert list(protocol.columns) == list(cal['protocol_columns'])
    mean = (baseline_rank + reference_rank + second_rank) / 3.
    blocks = [baseline_rank, reference_rank, second_rank, reference_rank - baseline_rank, second_rank - baseline_rank, mean]
    for g in cal['groups']:
        blocks.append(mean[:, [T.index(t) for t in g]].mean(axis=1, keepdims=True))
    blocks.append(protocol.to_numpy(np.float64))
    x = np.concatenate(blocks, axis=1)
    adj = rank_pct(((x - np.asarray(cal['mean'])) / np.asarray(cal['scale'])) @ np.asarray(cal['coef']).T
                   + np.asarray(cal['intercept']))
    vals = branch.copy()
    for j, t in enumerate(T):
        if t in set(cal['gate']):
            vals[:, j] = .6 * vals[:, j] + .4 * adj[:, j]
    return rank_pct(vals)


def raptor_public(arms, w, extra=None, extra_w=0.0):
    """c28 _ke_run_raptor_arms: normalized prob blend of clipped arm probs, then rank01; written as float32.
    extra: optional extra prob arm (the fleet) with raw weight extra_w (normalized together with w)."""
    ws = list(np.asarray(w, np.float64)); mats = [np.clip(a, 0, 1) for a in arms]
    if extra is not None and extra_w > 0:
        ws.append(float(extra_w)); mats.append(np.clip(extra, 0, 1))
    ws = np.asarray(ws, np.float64); ws = ws / ws.sum()
    blend = np.tensordot(ws, np.stack(mats), axes=(0, 0))
    return rank01(blend).astype(np.float32).astype(np.float64)


def coat_family(probs):                      # c28: rank of the members' probability mean
    return rank_pct(sum(probs) / float(len(probs)))


def hybrid(pub, fam, alpha=PRIVATE_ALPHA):   # c28: (1-a)*rank_pct(public Raptor csv) + a*family rank; NOT re-ranked
    return (1.0 - alpha) * rank_pct(pub) + alpha * fam


def final_blend(st3, side, cw=None):         # c28 outer blend: rank_pct((1-w_l) rank_pct(T) + w_l side)
    cw = cw or COATNET_W
    tr = rank_pct(st3); out = np.empty_like(tr)
    for j, t in enumerate(T):
        out[:, j] = (1 - cw[t]) * tr[:, j] + cw[t] * side[:, j]
    return rank_pct(out)


def k2_global(final, F, w):                  # c31 K2: rank((1-W) rank(parent) + W rank(fleet))
    return rank_pct((1 - w) * rank_pct(final) + w * rank_pct(F))


# ---------------- metrics ----------------
def auc_cols(Y, X):
    from sklearn.metrics import roc_auc_score
    return np.array([roc_auc_score(Y[:, j], X[:, j]) for j in range(Y.shape[1])])


def mauc(Y, X):
    return float(auc_cols(Y, X).mean())


def rho_mean(A, B):
    from scipy.stats import spearmanr
    return float(np.nanmean([spearmanr(A[:, j], B[:, j]).correlation for j in range(A.shape[1])]))


class Boot:
    """Paired bootstrap of macro-AUC deltas via multinomial study weights (exact weighted Mann-Whitney)."""
    def __init__(self, Y, B=2000, seed=0):
        self.Y = Y; n = len(Y)
        rng = np.random.default_rng(seed)
        self.C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float64)

    def aucs(self, X):          # (B, 12), NaN where a label lacks a class in the resample
        out = np.full((len(self.C), self.Y.shape[1]), np.nan)
        for j in range(self.Y.shape[1]):
            pos = self.Y[:, j] == 1; neg = ~pos
            s = X[:, j]
            K = (s[pos][:, None] > s[neg][None, :]).astype(float) + 0.5 * (s[pos][:, None] == s[neg][None, :])
            Cp, Cn = self.C[:, pos], self.C[:, neg]
            num = np.einsum('bi,ij,bj->b', Cp, K, Cn)
            den = Cp.sum(1) * Cn.sum(1)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[:, j] = np.where(den > 0, num / den, np.nan)
        return out

    def delta(self, Xnew, Xold):
        a, b = self.aucs(Xnew), self.aucs(Xold)
        d = np.nanmean(a - b, axis=1)
        return float(np.quantile(d, .05)), float(np.quantile(d, .95)), float((d > 0).mean())
