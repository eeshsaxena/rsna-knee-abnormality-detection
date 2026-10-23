# select_oai.py (10-20 r5, Claude). Box only; prints AGGREGATES ONLY. Orders the MOAKS-labelled OAI V00 knees that have a
# fleet-store npz so that EVERY PREFIX is a proportional stratified sample (strata = OA P_FT 8-cell x any meniscus tear x
# effusion >= 2): within a stratum knees get random ranks r, key = (r + u_stratum) / n_stratum, sort by key. Proportional (not
# oversampled) so the OA joint distribution, which E-OA depends on, stays the OAI one. Writes ~/r5_oai/ids_oai_all.json.
import os, json
import numpy as np, pandas as pd
H = os.path.expanduser('~')
OA = ['Medial OA', 'Lateral OA', 'PF OA']
COMP = {'Medial OA': ['cmf', 'pmf', 'amt', 'cmt', 'pmt'], 'Lateral OA': ['clf', 'plf', 'alt', 'clt', 'plt'],
        'PF OA': ['mp', 'lp', 'amf', 'alf']}
s = pd.read_csv(H + '/oai_build/oai_label_sources.csv', dtype={'kid': str}).set_index('kid')
have = {f[:-4] for f in os.listdir(H + '/oai_store') if f.endswith('.npz') and '_v' not in f[-8:]}
s = s[s.index.isin(have)]
fin = np.ones(len(s), bool); cell = np.zeros(len(s), int)
for b, (lab, regs) in enumerate(COMP.items()):
    c = s[['m_cm_' + r for r in regs]].astype(float).values
    area = np.floor(c + 1e-6); ft = np.round((c - area) * 10)
    fin &= np.isfinite(c).any(1)
    cell += ((ft >= 1) & np.isfinite(c)).any(1).astype(int) << (2 - b)
men = np.zeros(len(s), int)
for p in ('mm', 'lm'):
    mo = s[[f'm_{p}{x}_morph' for x in ('ah', 'body', 'ph')]].astype(float).values
    men |= (((mo >= 2) & (mo <= 8)).any(1) | (s[f'm_{p}pr_tear'].astype(float).values == 1)).astype(int)
eff = (s['m_eff_wk'].astype(float).values >= 2).astype(int)
s = s[fin]; cell, men, eff = cell[fin], men[fin], eff[fin]
strat = cell * 4 + men * 2 + eff
rng = np.random.default_rng(20261020)
key = np.zeros(len(s))
for g in np.unique(strat):
    ii = np.where(strat == g)[0]
    r = rng.permutation(len(ii))
    key[ii] = (r + rng.random()) / len(ii)
order = np.argsort(key, kind='stable')
ids = s.index[order].tolist()
json.dump(ids, open(H + '/r5_oai/ids_oai_all.json', 'w'))
print('eligible knees (store npz + every OA compartment read):', len(ids), '| strata', len(np.unique(strat)))
for n in (300, 1000, 1500, 2000, len(ids)):
    o = order[:n]
    print(f'prefix {n}: OA P_FT M/L/PF prev {np.round([((cell[o] >> (2 - b)) & 1).mean() for b in range(3)], 3).tolist()} '
          f'| meniscus tear {men[o].mean():.3f} | effusion>=2 {eff[o].mean():.3f} | right-knee frac '
          f'{np.mean([i.endswith("_R") for i in np.array(ids)[:n]]):.3f}')
