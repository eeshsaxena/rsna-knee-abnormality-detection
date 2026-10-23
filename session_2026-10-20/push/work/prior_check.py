# prior_check.py: MODEL. How well each pre-read expectation of the population E-OA gains explains the public reads.
# Public-sample noise of (S10, S20) at n 390: sd .00025 / .00052, corr .94 (box_abc2_PFT.log, D5 at R3p+0.0 / -0.5; OAI proxy).
import numpy as np
d = np.load('lb_post_pooled.npz'); G = np.stack([d['G10'], d['G20']], 1) * 1e-3
sd = np.array([.00025, .00052]); C = np.array([[sd[0] ** 2, .94 * sd[0] * sd[1]], [.94 * sd[0] * sd[1], sd[1] ** 2]]); Ci = np.linalg.inv(C)
H = {'R5 revision (eoa10 +.0002, eoa20 -.00005)': (.0002, -.00005), 'skeptic ladder MORE_1020 (+.0003, +.0001)': (.0003, .0001),
     'skeptic_eoa (+.0005, +.0007)': (.0005, .0007), 'gold58 c4 x .5 (+.0007, +.0012)': (.0007, .0012), 'ABC posterior mean (+.00064, +.00097)': (.00064, .00097)}
for k, m in H.items():
    r = G - np.array(m); m2 = np.einsum('ij,jk,ik->i', r, Ci, r)
    lik = np.mean(np.exp(-.5 * m2)) / (2 * np.pi * np.sqrt(np.linalg.det(C)))
    # tail: P(noise draw at least as far, Mahalanobis) averaged over LB uncertainty
    from scipy.stats import chi2
    print('%-46s mean Mahalanobis^2 %.1f | P(chi2_2 >= m2) median %.3f | rel. likelihood %.3g' % (k, np.median(m2), np.median(chi2.sf(m2, 2)), lik))
