# kc_g1.py -- KC G1 view ablation (BOX_PLAN_CRITIQUE_1015, adopted; PREREG_10-15 Amendment 6 G1).
# Used by (a) the f0 end hook inside fleet_train_kc.py (GPU, on the E* ckpt, fold-0 val ~867 studies, scored vs T_b) and
# (b) kc_g1_ablation.py (box CPU, on the epoch-3 ckpt, ~1.2 h into f0: early warning only, main decides).
# ONE backbone encode per batch, then the attention head is re-run from the cached features for 4 variants:
#   full     present as stored
#   abl_med  present[sag_med, cor_med] = False   (the view softmax masks them to -1e4: equal to removing the views)
#   abl_lat  present[sag_lat, cor_lat] = False
#   swap     features of sag_lat <-> sag_med and cor_lat <-> cor_med exchanged (info only)
# head_from_feats replicates fleet_model.KneeFleetNet.forward after encode (pos=None); a self-check asserts
# max |p_split - p_model| <= 2e-3 on the first batch (fp16 on GPU; ~1e-6 on CPU fp32).
# Pre-registered G1 ablation rule: drop_X(l) = AUC_full(l) - AUC_ablX(l), labels thresholded at .5 (NaN masked);
#   med_margin = mean over {Medial Meniscus, Medial OA} of drop_med - drop_lat
#   lat_margin = mean over {Lateral Meniscus, Lateral OA} of drop_lat - drop_med
#   ablation PASS iff med_margin > 0 AND lat_margin > 0.
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
MED_L = ('Medial Meniscus', 'Medial OA')
LAT_L = ('Lateral Meniscus', 'Lateral OA')
VARIANTS = ('full', 'abl_med', 'abl_lat', 'swap')


def head_from_feats(m, f, present, ctr, B, V, S):
    sp = m.slice_pos[None]
    h = m.proj(f).view(B, V, S, m.dim) + sp + m.view_emb[None, :, None]
    if m.ctr_emb is not None and ctr is not None:
        h = h + m.ctr_emb(ctr)[:, :, None]
    sc = 1.0 / math.sqrt(m.dim)
    a = torch.einsum('td,bvsd->bvts', m.slice_q, m.slice_k(h)) * sc * m.logt_s.exp()
    pooled = torch.einsum('bvts,bvsd->bvtd', a.softmax(-1), h)
    va = torch.einsum('td,bvtd->btv', m.view_q, m.view_k(pooled)) * sc * m.logt_v.exp()
    va = va + m.vbias[None]
    pres = present.clone()
    pres[pres.sum(1) == 0] = True
    va = va.masked_fill(~pres[:, None, :], -1e4).softmax(-1)
    z = torch.einsum('btv,bvtd->btd', va, pooled)
    return (m.drop(z) * m.head_w[None]).sum(-1) + m.head_b


def run_variants(m, loader, dev, amp, view_names, check=True, max_batches=0):
    """-> {variant: probs (N, 12)}; loader yields (x, pr, ct, pos, y) like the trainer's eval DataLoader."""
    vi = {n: i for i, n in enumerate(view_names)}
    med = [vi['sag_med'], vi['cor_med']]; lat = [vi['sag_lat'], vi['cor_lat']]
    perm = list(range(len(view_names)))
    for a, b in (('sag_lat', 'sag_med'), ('cor_lat', 'cor_med')):
        perm[vi[a]], perm[vi[b]] = vi[b], vi[a]
    out = {k: [] for k in VARIANTS}
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        for bi, (x, pr, ct, _pos, _y) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x, pr, ct = x.to(dev), pr.to(dev), ct.to(dev)
            B, V, S = x.shape[:3]
            f = m.encode(x.flatten(0, 2)).float()
            pf = head_from_feats(m, f, pr, ct, B, V, S)
            if check and bi == 0:
                ref = m(x, pr, ct)
                d = (torch.sigmoid(pf.float()) - torch.sigmoid(ref.float())).abs().max().item()
                assert d <= 2e-3, f'kc_g1 split forward drifted from model.forward: max |dp| {d}'
            out['full'].append(torch.sigmoid(pf.float()).cpu().numpy())
            p2 = pr.clone(); p2[:, med] = False
            out['abl_med'].append(torch.sigmoid(head_from_feats(m, f, p2, ct, B, V, S).float()).cpu().numpy())
            p3 = pr.clone(); p3[:, lat] = False
            out['abl_lat'].append(torch.sigmoid(head_from_feats(m, f, p3, ct, B, V, S).float()).cpu().numpy())
            fs = f.view(B, V, S, -1)[:, perm].reshape(B * V * S, -1)
            out['swap'].append(torch.sigmoid(head_from_feats(m, fs, pr[:, perm], ct[:, perm], B, V, S).float()).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _auc(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    yb = y[ok] >= 0.5
    if ok.sum() == 0 or yb.all() or not yb.any():
        return float('nan'), int(yb.sum())
    return float(roc_auc_score(yb, p[ok])), int(yb.sum())


def metrics(P, Y):
    """P: {variant: (N, 12)}, Y: (N, 12) float labels (NaN allowed). -> dict incl. 'ablation_pass'."""
    res = {'n': int(len(Y)), 'labels': {}}
    for l in MED_L + LAT_L:
        j = TARGETS.index(l)
        row = {}
        for k in VARIANTS:
            row[k], npos = _auc(Y[:, j], P[k][:, j])
        row['npos'] = npos
        row['drop_med'] = row['full'] - row['abl_med']
        row['drop_lat'] = row['full'] - row['abl_lat']
        res['labels'][l] = row
    L = res['labels']
    res['med_margin'] = float(np.mean([L[l]['drop_med'] - L[l]['drop_lat'] for l in MED_L]))
    res['lat_margin'] = float(np.mean([L[l]['drop_lat'] - L[l]['drop_med'] for l in LAT_L]))
    res['per_label_ok'] = {l: bool((L[l]['drop_med'] > L[l]['drop_lat']) if l in MED_L else (L[l]['drop_lat'] > L[l]['drop_med']))
                           for l in MED_L + LAT_L}
    res['swap_mean4_full'] = float(np.nanmean([L[l]['full'] for l in MED_L + LAT_L]))
    res['swap_mean4_swapped'] = float(np.nanmean([L[l]['swap'] for l in MED_L + LAT_L]))
    macro = []
    for j in range(len(TARGETS)):
        a, _ = _auc(Y[:, j], P['full'][:, j])
        if np.isfinite(a):
            macro.append(a)
    res['macro_full'] = float(np.mean(macro)) if macro else float('nan')
    res['ablation_pass'] = bool(np.isfinite(res['med_margin']) and np.isfinite(res['lat_margin'])
                                and res['med_margin'] > 0 and res['lat_margin'] > 0)
    return res


def summary_line(res):
    L = res['labels']
    per = ' '.join(f"{l.replace(' ', '')}:dm{L[l]['drop_med']:+.4f}/dl{L[l]['drop_lat']:+.4f}" for l in MED_L + LAT_L)
    return (f"n {res['n']} | med_margin {res['med_margin']:+.4f} lat_margin {res['lat_margin']:+.4f} | {per} | "
            f"swap4 {res['swap_mean4_full']:.4f}->{res['swap_mean4_swapped']:.4f} | macro {res['macro_full']:.4f}")
