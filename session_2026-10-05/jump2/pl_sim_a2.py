# pl_sim_a2.py: fold-0 val 867, students only. Within-report-verdict Spearman (removes the report signal both
# sides share; the residual is image-derived agreement = the part that decides blend lift) vs the Raptor proxy,
# plus student-to-student and ensemble rows. CPU, read-only.
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
H = os.path.expanduser
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
rd = lambda p: pd.read_csv(H(p), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
O = {'F1mix': '~/fleet_runs/f0_s42/oof_f0_s42_raw.csv', 'F1v4': '~/fleet_runs/f0_s42_v4/oof_f0_s42_raw.csv',
     'F1gated': '~/fleet_runs/f0_s42_gated/oof_f0_s42_raw.csv', 'TTRv4': '~/fleet_runs/f0_s1042_ttr/oof_f0_s1042_ema.csv',
     'TTRmix': '~/fleet_runs/f0_s1043_ttr_mix50/oof_f0_s1043_raw.csv'}
O = {k: rd(v)[T] for k, v in O.items()}
ids = sorted(set.intersection(*[set(o.index) for o in O.values()]))
rk = lambda d: d.rank(pct=True)
O['Ev4'] = (rk(O['F1v4'].loc[ids]) + rk(O['TTRv4'].loc[ids])) / 2
O['Emix'] = (rk(O['F1mix'].loc[ids]) + rk(O['TTRmix'].loc[ids])) / 2
O['Eall'] = sum(rk(O[k].loc[ids]) for k in ['F1v4', 'TTRv4', 'F1mix', 'TTRmix']) / 4
rap = rd('~/fleet/teacher_v2_raptorproxy_shard0.csv')[T].loc[ids]
wide = rd('~/rsna_train/teacher_train_probs.csv')[T].loc[ids]
v4 = rd('~/rsna_train/llm_labels_v4_blend.csv')[T].loc[ids]
r2 = rd('~/rsna_train/report_labels_v2.csv')
VER = pd.DataFrame({l: r2[l + '__verdict'] for l in T}).reindex(ids).fillna('UNK')
def ws(A, B):  # within (label, verdict) Spearman, weighted by n
    num = den = 0.0
    for l in T:
        for v in ['YES', 'UNK', 'NO']:
            m = (VER[l] == v).values
            if m.sum() >= 30:
                r = spearmanr(A[l].values[m], B[l].values[m]).correlation
                if np.isfinite(r): num += r * m.sum(); den += m.sum()
    return num / den
def pl(A, B): return float(np.nanmean([spearmanr(A[l], B[l]).correlation for l in T]))
print(f'{"student":8s} rho_rap  ws_rap  ws_wide | ws vs TTRv4  ws vs TTRmix  ws vs F1v4')
for k in ['F1v4', 'TTRv4', 'F1gated', 'F1mix', 'TTRmix', 'Ev4', 'Emix', 'Eall']:
    X = O[k].loc[ids]
    print(f'{k:8s} {pl(X, rap):.3f}   {ws(X, rap):.3f}   {ws(X, wide):.3f}  |  {ws(X, O["TTRv4"].loc[ids]):.3f}       {ws(X, O["TTRmix"].loc[ids]):.3f}        {ws(X, O["F1v4"].loc[ids]):.3f}')
print('ws(rap, wide)', round(ws(rap, wide), 3), '| verdict shares', {v: round(float((VER.values == v).mean()), 3) for v in ['YES', 'UNK', 'NO']})
