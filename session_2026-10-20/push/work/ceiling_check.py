# ceiling_check.py: MODEL. Does a near-ceiling base AUC (c4 Medial OA ~.97-.99) bend the E-OA curve down faster than the OAI
# models' ~.85-.9 AUCs? Gaussian latent model, equicorrelated trio: corr(Z,Z) r, corr(f,f) rho, corr(f_k,Z_k) c, corr(f_j,Z_k) q c.
# out_k = U_k - a/2 (U_j + U_l), U = Phi(f). Large-n population AUC of out_k vs Y_k = 1[Z_k > tau].
import numpy as np
from scipy.stats import norm, rankdata
rng = np.random.default_rng(0); n = 400000


def curve(c, rho=.816, q=.69, r=.5, prev=.3, grid=(0, .1, .2, .3, .4), pw=None):
    C = np.eye(6)
    for i in range(3):
        for j in range(3):
            if i != j: C[i, j] = r; C[3 + i, 3 + j] = rho
            C[3 + i, j] = C[j, 3 + i] = c if i == j else q * c
    if np.min(np.linalg.eigvalsh(C)) <= 0: return None
    X = rng.standard_normal((n, 6)) @ np.linalg.cholesky(C).T; Z, F = X[:, :3], X[:, 3:]; U = norm.cdf(F)
    Y = (Z[:, 0] > norm.ppf(1 - prev)).astype(int); out = []
    for a in grid:
        s = U[:, 0] - a / 2 * (U[:, 1] + U[:, 2]); R = rankdata(s); p = Y.sum()
        out.append((R[Y == 1].sum() - p * (p + 1) / 2) / (p * (n - p)))
    return np.array(out)


print('rho .816, r .5, prev .3; AUC curve at a = 0/.1/.2/.3/.4 and (A(.3)-A(.2)) / (A(.2)-A(.1))')
for q in (.64, .69, .74):
    for c in (.55, .7, .8, .86):
        A = curve(c, q=q)
        if A is None: print('  q %.2f c %.2f: not PSD' % (q, c)); continue
        d = A - A[0]
        print('  q %.2f c %.2f: base AUC %.4f | d .1/.2/.3/.4 %+.4f %+.4f %+.4f %+.4f | curvature ratio %.2f | best a %.1f' % (
            q, c, A[0], d[1], d[2], d[3], d[4], (d[3] - d[2]) / (d[2] - d[1]) if d[2] != d[1] else np.nan, [0, .1, .2, .3, .4][int(np.argmax(A))]))
