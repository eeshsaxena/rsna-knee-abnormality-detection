import sys, os, importlib.util, numpy as np
def load(p, name):
    sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
H = os.path.expanduser('~')
old = load(f'{H}/kc/kcfix/r3/old/kc_g1.py', 'kc_g1_old'); new = load(f'{H}/kc/code/kc_g1.py', 'kc_g1_new')
T = new.TARGETS; rng = np.random.default_rng(0); N = 867
Y = (rng.random((N, 12)) < .3).astype(float)
def mk(eff):   # eff[label] = (strength full, strength without med views, strength without lat views)
    P = {k: rng.random((N, 12)) * 0.0 for k in new.VARIANTS}
    base = rng.normal(size=(N, 12))
    for k, col in (('full', 0), ('abl_med', 1), ('abl_lat', 2), ('swap', 0)):
        S = np.full(12, 1.0)
        for l, e in eff.items(): S[T.index(l)] = e[col]
        P[k] = 1 / (1 + np.exp(-(base + S * (2 * Y - 1))))
    return P
cases = {
  'canon right, lateral pair flat (noise: lat_margin < 0, med_margin > 0)':
      {'Medial Meniscus': (1.2, .6, 1.2), 'Medial OA': (1.2, .6, 1.2), 'Lateral Meniscus': (1.0, .97, 1.0), 'Lateral OA': (1.0, .98, 1.0)},
  'clear canon FAIL (both margins < 0)':
      {'Medial Meniscus': (1.2, 1.2, .6), 'Medial OA': (1.2, 1.2, .6), 'Lateral Meniscus': (1.2, .6, 1.2), 'Lateral OA': (1.2, .6, 1.2)},
  'clean PASS (both margins > 0)':
      {'Medial Meniscus': (1.2, .6, 1.2), 'Medial OA': (1.2, .6, 1.2), 'Lateral Meniscus': (1.2, 1.2, .6), 'Lateral OA': (1.2, 1.2, .6)},
}
for name, eff in cases.items():
    P = mk(eff); a = old.metrics(P, Y); b = new.metrics(P, Y)
    assert abs(a['med_margin'] - b['med_margin']) < 1e-12 and abs(a['lat_margin'] - b['lat_margin']) < 1e-12
    print(f"{name}: med {a['med_margin']:+.4f} lat {a['lat_margin']:+.4f} | OLD ablation_pass {a['ablation_pass']} | NEW {b['ablation_pass']} | per-pair {b['per_label_ok']}")
P = mk(cases['clean PASS (both margins > 0)']); P['abl_lat'][:] = np.nan
print('non-finite margin: NEW ablation_pass', new.metrics(P, Y)['ablation_pass'])
