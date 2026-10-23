# rank_map.py: MODEL. Public rank on the 10-20 05:53 board (eoa20 = r135, 4 others above it in .959, .960 band 44 teams)
# for eoa20 + delta, using the pooled position posterior x20 (lb_post.py) and an exponential in-band density (k from the
# .959 : .960 counts 65 : 44). Board drift until the read (+12-25 teams/day at .959+) is NOT included.
import numpy as np
d = np.load('lb_post_pooled.npz'); rng = np.random.default_rng(2); B = 40000; i = rng.choice(len(d['x20']), B)
x20 = d['x20'][i]; k = np.maximum(np.log(rng.gamma(66, size=B) / rng.gamma(45, size=B)), 1e-3)
A959 = 65 * k / (1 - np.exp(-k)); A960 = A959 * np.exp(-k)
for dl in (0, .5, 1, 1.5, 2, 2.5, 3, 4, 5):
    x = x20 + dl / 10.0
    n959 = np.minimum(4, A959 * (np.exp(-k * x20) - np.exp(-k * np.minimum(x, 1))) / k)
    n960 = np.where(x > 1, A960 * (1 - np.exp(-k * (x - 1))) / k, 0)
    r = 135 - n959 - n960
    print('eoa20 %+.1fe-4: rank %.0f [%.0f, %.0f] | P(shows .960) %.2f' % (dl, np.median(r), np.quantile(r, .1), np.quantile(r, .9), np.mean(x >= 1)))
