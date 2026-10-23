"""verify_B rank sensitivity: b_rank.py structure, but the teams ABOVE the band are split into the MEASURED
public groups (10-08 15:10: 71 teams at .951-.953, 82 at >= .954) grown to the deadline (GUESS), and our offset
uses an r_FN prior centred .90 or .915 plus model optimism U(0,.001). GUESS-level, laptop, light."""
import numpy as np
rng = np.random.default_rng(11)
T = 6900; SILVER, BRONZE = int(.05 * T), int(.10 * T)
M = 3000
def band(n):
    k = rng.random(n)
    off = np.where(k < .5, rng.normal(0, .0002, n), np.where(k < .7, rng.normal(-.001, .0003, n), rng.normal(.0005, .0008, n)))
    return off + rng.normal(0, .0004, n)
for mu in (.90, .915):
    for n1, n2, nb in [(71, 82, 900), (150, 110, 900), (250, 130, 900), (400, 150, 900), (250, 130, 600), (250, 130, 1200)]:
        ra, rf = [], []
        for _ in range(M):
            B = band(nb)
            U1 = rng.normal(.002, .0007, n1) + rng.normal(0, .0005, n1)   # N + own blends, public .951-.953 (paired vs band)
            U2 = rng.normal(.006, .002, n2) + rng.normal(0, .0018, n2)    # strong own systems
            Lw = rng.normal(-.004, .0015, 600) + rng.normal(0, .0018, 600)
            others = np.concatenate([B, U1, U2, Lw])
            rFN = np.clip(rng.normal(mu, .025), .84, .97)
            mu_a = .0010 - .031 * (rFN - .90) - rng.uniform(0, .001) + rng.normal(0, .0006)
            a = mu_a + rng.normal(0, .0004)
            f = mu_a + .0002 + .0002 * (rFN > .92) + rng.normal(0, .0004)
            ra.append(1 + int((others > a).sum())); rf.append(1 + int((others > f).sum()))
        ra, rf = np.array(ra), np.array(rf)
        print(f'r_FN~N({mu},.025) n(.951-.953)={n1:3d} n(>=.954)={n2:3d} band={nb:4d}: K9N-a med rank {np.median(ra):5.0f} '
              f'P(silver+) {np.mean(ra <= SILVER):.2f} P(bronze+) {np.mean(ra <= BRONZE):.2f} | FNG med {np.median(rf):5.0f} '
              f'P(silver+) {np.mean(rf <= SILVER):.2f} P(bronze+) {np.mean(rf <= BRONZE):.2f}')
