# max4.py: MODEL. A15 makes pick A = API best-id = argmax public over {HW, wr45, s615, a25}. Private gain of that rule vs keeping HW.
import numpy as np
rng = np.random.default_rng(3); M = 1_000_000
sp = np.array([5.1e-4, 4.1e-4, 3.0e-4]); sv = np.array([3.3e-4, 2.7e-4, 2.0e-4])   # wr45, s615, a25 (a25 = GUESS 0.6x wr45)
for name, mean, sd in (("flat N(0,5e-4)", np.zeros(3), np.full(3, 5e-4)),
                       ("gold58-informed", np.array([-5e-4, -1.0e-3, 0.0]), np.array([8e-4, 7e-4, 5e-4]))):
    for scale in (1.0, 0.4):   # 0.4 = prior binormal sim3 noise level instead of the gold58-bootstrap level
        mu = mean + sd * rng.standard_normal((M, 3))
        x = mu + sp * scale * rng.standard_normal((M, 3)); z = mu + sv * scale * rng.standard_normal((M, 3))
        xa = np.c_[np.zeros(M), x]; za = np.c_[np.zeros(M), z]
        k = xa.argmax(1); gain = za[np.arange(M), k]
        print(f"{name:16s} noise x{scale}: P(A changes) {np.mean(k>0):.2f} | P(pick=wr45/s615/a25) {[round(np.mean(k==i),2) for i in (1,2,3)]}"
              f" | E[priv gain] {gain.mean():+.5f} | P(gain<0) {np.mean(gain<0):.2f} | P(gain<-5e-4) {np.mean(gain<-5e-4):.3f} | oracle best E {za.max(1).mean():+.5f}")
