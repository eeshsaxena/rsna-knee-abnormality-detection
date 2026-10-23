# census_f.py (10-20 push, Claude): 50/50 mixture of the 'mech' and 'cv' priors (census_d.py) updated by the gold58 observation
# (marginal-likelihood weights); E / P(true > 0) / P(realised private > 0) / P(public read moves) / E[gain of probe + R1], macro units.
# Plus the 2-label combo MCL <- MM + Frac <- MOA (independent labels; effects and read noise add). Local arithmetic only.
import numpy as np, pandas as pd
from scipy.stats import norm
pv = pd.read_csv('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/push/census_posterior.csv')
PRI = {'mech': (.001, .002), 'cv': (-.002, .002)}
rng = np.random.default_rng(7)
def mix(d, se):
    comps = []
    for m0, s0 in PRI.values():
        w = norm.pdf(d, m0, np.sqrt(s0 ** 2 + se ** 2)); k = s0 ** 2 / (s0 ** 2 + se ** 2)
        comps.append((w, m0 + k * (d - m0), np.sqrt(1 / (1 / s0 ** 2 + 1 / se ** 2))))
    W = sum(c[0] for c in comps); return [(c[0] / W, c[1] / 12, c[2] / 12) for c in comps]
def draw(comps, n=400000):
    z = np.concatenate([rng.normal(m, s, int(n * w)) for w, m, s in comps]); return z
rows = []
for r in pv.itertuples():
    z = draw(mix(r.d_g58, r.se)); ep = rng.normal(0, r.n_pub, len(z)); er = rng.normal(0, r.n_priv, len(z))
    rows.append((r.dir, z.mean(), (z > 0).mean(), (z + er > 0).mean(), (z + ep > 0).mean(), np.mean(z * (z + ep > 0)), r.n_pub, r.n_priv))
a = pv.set_index('dir')
z = draw(mix(a.loc['MCL <- MM', 'd_g58'], a.loc['MCL <- MM', 'se'])) + rng.permutation(draw(mix(a.loc['Frac <- MOA', 'd_g58'], a.loc['Frac <- MOA', 'se'])))[:0].sum()
z1 = draw(mix(a.loc['MCL <- MM', 'd_g58'], a.loc['MCL <- MM', 'se'])); z2 = draw(mix(a.loc['Frac <- MOA', 'd_g58'], a.loc['Frac <- MOA', 'se']))
k = min(len(z1), len(z2)); z = rng.permutation(z1)[:k] + rng.permutation(z2)[:k]
npub = np.hypot(a.loc['MCL <- MM', 'n_pub'], a.loc['Frac <- MOA', 'n_pub']); npri = np.hypot(a.loc['MCL <- MM', 'n_priv'], a.loc['Frac <- MOA', 'n_priv'])
ep = rng.normal(0, npub, k); er = rng.normal(0, npri, k)
rows.append(('combo MCL<-MM + Frac<-MOA', z.mean(), (z > 0).mean(), (z + er > 0).mean(), (z + ep > 0).mean(), np.mean(z * (z + ep > 0)), npub, npri))
df = pd.DataFrame(rows, columns=['dir', 'E_macro', 'P_true', 'P_priv', 'P_move', 'probe_gain', 'n_pub', 'n_priv'])
for c in ('E_macro', 'probe_gain', 'n_pub', 'n_priv'): df[c] = df[c].map(lambda x: '%+.1e' % x)
print(df.round(3).to_string(index=False))
