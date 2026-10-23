"""Parent shake-up audit, offline part (box, CPU only, read only on the mimic run):
flat outer-map hedge vs the parent's public-tuned per-label map, on the gold58 box mimic run of the parent.

Two brackets for the transformer side T (as in jump_wf/inner_injection.md sec 4):
  in  = the real stage 3 (DINO + A5 + Rad fitted on these 58 studies' gold labels: favours weight on T, i.e. flat)
  oos = pseudo-OOS stage 3 (per study only the DINO fold members / Rad head that held it out: too weak, favours the map)
The Raptor/CoAt side is honestly out of sample on gold58. Fleets at W .5: two proxies of our own fleet (gold58 OOS)."""
import os, sys, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_common import *          # noqa: F401,F403  (copied unchanged from ~/k2_test/analysis_wf)

D = load_all(); Y = D['Y']; ids = D['ids']; n = len(Y)
boot = Boot(Y, B=2000, seed=0)
H_ = os.path.expanduser('~')
FLEETS = {'F1mix50_k2': D['fleet'],
          'K3_f1mix50+ttrv4': read_csv(H_ + '/k3_test/runs/G58_main_2149/working/submission_fleetsolo.csv', ids)}
OUT = {'n': n, 'positives': dict(zip(T, Y.sum(0).astype(int).tolist()))}


def loss(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean(1)


# ---- pseudo-OOS transformer side (identical construction to inject_variants.py)
runs = sorted(set(D['dino_run'])); A = []
for r in runs:
    idx = [i for i, rr in enumerate(D['dino_run']) if rr == r]
    L = np.stack([loss(D['dino'][i]) for i in idx])
    A.append(np.array([D['dino_fold'][idx[k]] for k in np.argmax(L, 0)]))
A = np.stack(A)
dfold = np.array([Counter(A[:, s].tolist()).most_common(1)[0][0] for s in range(n)])
dino_oof = np.zeros((n, 12))
for s in range(n):
    mem = [i for i in range(len(D['dino_ids'])) if D['dino_fold'][i] == dfold[s]]
    dino_oof[s] = D['dino'][mem, s].astype(np.float64).mean(0)
Lrad = sum(np.stack([loss(h) for h in D[k]]) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw'))
rfold = np.argmax(Lrad, 0)
rad_oof = {k: D[k][rfold, np.arange(n)][None].astype(np.float32) for k in ('rad_e10_raw', 'rad_e13_raw', 'rad_legacy_second_raw')}
st2_oof = stage2_a5(rank_pct(dino_oof), D['a5_rank_mean'], D['a5_ok'])
st3_oof = stage3_rad(st2_oof, rad_oof['rad_e10_raw'], rad_oof['rad_e13_raw'], rad_oof['rad_legacy_second_raw'],
                     D['cal'], protocol_features(D['tser'], ids))
TB = {'in': D['st3'], 'oos': st3_oof}
side = D['side_csv']

# ---- sanity: stored stage 3 + stored side with the parent map == stored final (the KF self-check, on 58 studies)
OUT['rebuild_stored_final_max_abs'] = float(np.abs(final_blend(D['st3'], side) - D['st4']).max())
OUT['rebuild_vs_localLB_parent_max_abs'] = float(np.abs(final_blend(D['st3'], side) - D['parent_sub']).max())

# ---- component AUCs per label (where does T carry signal on the mapped labels?)
comp = {'T_in': D['st3'], 'T_oos': st3_oof, 'side': side, 'pub': D['pub_csv'], 'fam': D['fam_csv']}
comp.update({f'fleet_{k}': v for k, v in FLEETS.items()})
OUT['component_auc'] = {k: dict(zip(T, np.round(auc_cols(Y, v), 4).tolist())) | {'macro': round(mauc(Y, v), 4)}
                        for k, v in comp.items()}
OUT['rho_T_side'] = {b: dict(zip(T, [round(float(np.corrcoef(rank_pct(t)[:, j], rank_pct(side)[:, j])[0, 1]), 3)
                                     for j in range(12)])) for b, t in TB.items()}

MAP = dict(COATNET_W)
MAPS = {'map': MAP,
        'flat60': {t: 0.60 for t in T},
        'half': {t: (0.60 + MAP[t]) / 2 for t in T},
        'flat_mean6875': {t: float(np.mean(list(MAP.values()))) for t in T},
        'map_LM_only_flat': dict(MAP, **{'Lateral Meniscus': 0.60})}
CH = [t for t in T if abs(MAP[t] - 0.60) > 1e-12]


def summarize(new, old):
    a, b = auc_cols(Y, new), auc_cols(Y, old)
    lo, hi, p = boot.delta(new, old)
    d = boot.aucs(new) - boot.aucs(old)
    sd = float(np.nanstd(np.nanmean(d, axis=1)))
    return {'d_macro': round(float((a - b).mean()), 5), 'ci90': [round(lo, 5), round(hi, 5)], 'P_gt0': round(p, 3),
            'boot_sd_macro_n58': round(sd, 5), 'scaled_sd_public390': round(sd * (58 / 390) ** .5, 5),
            'd_per_label_changed': {t: round(float(a[T.index(t)] - b[T.index(t)]), 4) for t in CH},
            'mars': round(float(np.abs(new - old).mean()), 5),
            'rho_vs_old_changed': {t: round(float(np.corrcoef(rank_pct(new)[:, T.index(t)],
                                                              rank_pct(old)[:, T.index(t)])[0, 1]), 4) for t in CH}}


res = {}
for b, Tb in TB.items():
    fin = {k: final_blend(Tb, side, m) for k, m in MAPS.items()}
    res[f'{b}/parent'] = {'auc_map': round(mauc(Y, fin['map']), 4),
                          **{k: summarize(fin[k], fin['map']) for k in MAPS if k != 'map'}}
    for fn, F in FLEETS.items():
        bl = {k: k2_global(fin[k], F, 0.5) for k in MAPS}
        res[f'{b}/blendW50_{fn}'] = {'auc_map': round(mauc(Y, bl['map']), 4),
                                     **{k: summarize(bl[k], bl['map']) for k in MAPS if k != 'map'}}
OUT['results'] = res
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flat_audit_out.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != 'component_auc'}, indent=1)[:20000])
print('component AUC macro:', {k: v['macro'] for k, v in OUT['component_auc'].items()})
for k in ('T_in', 'T_oos', 'side'):
    print(k, {t: OUT['component_auc'][k][t] for t in CH})
