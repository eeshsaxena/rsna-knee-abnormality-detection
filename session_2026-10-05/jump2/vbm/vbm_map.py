"""vbm_map.py: check the pre-registered W .50 reading map (blendmath sec 7) and the 'SOLO + blend pins r +-.025' claim."""
import numpy as np
from scipy.stats import norm
def d(a): return np.sqrt(2) * norm.ppf(a)
def blend(P, s, r, W):
    v = (1 - W) ** 2 + W ** 2 + 2 * W * (1 - W) * r
    return norm.cdf(((1 - W) * d(P) + W * d(s)) / np.sqrt(2 * v))
RG = np.linspace(.80, .9999, 4000)
for solo in (.933, .936, .939):
    print(f'fleet solo {solo} (point), parent .943 (interval):')
    for read in (.945, .944, .943, .942, .941):
        ok = np.zeros_like(RG, bool)
        for P in np.linspace(.9425, .9435 - 1e-9, 11):
            for s in np.linspace(solo - .0005, solo + .0005 - 1e-9, 11):
                b = blend(P, s, RG, .5); ok |= (b >= read - .0005) & (b < read + .0005)
        print(f'   W.5 read {read}: r ' + (f'{RG[ok].min():.3f}-{RG[ok].max():.3f}' if ok.any() else 'none'))
# same at W .20: how flat is the read in r?
print('W .20 read vs r at solo .936:', ' '.join(f'r{r:.2f}:{blend(.943, .936, r, .2):.4f}' for r in (.88, .90, .92, .94, .96)))
