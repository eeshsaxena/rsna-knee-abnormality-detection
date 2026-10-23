# max4b.py: MODEL. "HW at a local optimum" prior: every one-line move loses mu = -|N(m, s)| (gold58 curves: both directions of
# W_RAPTOR and W_S6F lose on gold58). Pick A = argmax public over {HW, wr45, s615, a25}.
import numpy as np
rng = np.random.default_rng(4); M = 1_000_000
sp = np.array([5.1e-4, 4.1e-4, 3.0e-4]); sv = np.array([3.3e-4, 2.7e-4, 2.0e-4])
for m in (1e-4, 3e-4, 6e-4):
    for scale in (1.0, 0.4):
        mu = -np.abs(m + 0.5 * m * rng.standard_normal((M, 3)))
        x = mu + sp * scale * rng.standard_normal((M, 3)); z = mu + sv * scale * rng.standard_normal((M, 3))
        k = np.c_[np.zeros(M), x].argmax(1); gain = np.c_[np.zeros(M), z][np.arange(M), k]
        print(f"loss ~{m:.0e} noise x{scale}: P(A changes) {np.mean(k>0):.2f} | E[priv gain] {gain.mean():+.5f} | P(gain<0) {np.mean(gain<0):.2f}")
