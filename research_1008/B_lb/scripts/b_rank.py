"""b_rank.py: GUESS-level private-rank model at the deadline (illustrates the 'cliff' against the N band).
All structural inputs are GUESS except where noted; the K9N-a offset curve comes from b_shake.py (MODEL) and the
paired private sds from the calibrated simulation (MODEL). Laptop, light."""
import numpy as np
rng = np.random.default_rng(7)
T = 6900                                  # GUESS: 5,503 teams on 10-08 15:10 UTC (MEASURED) + ~100/day (MEASURED rate) x 14 d
GOLD, SILVER, BRONZE = 10 + int(.002 * T), int(.05 * T), int(.10 * T)
M = 4000
def band(n):                              # GUESS mixture of N-derived blends, offsets vs the N+G cluster's true AUC X
    k = rng.random(n)
    off = np.where(k < .5, rng.normal(0, .0002, n), np.where(k < .7, rng.normal(-.001, .0003, n), rng.normal(.0005, .0008, n)))
    return off + rng.normal(0, .0004, n)  # idiosyncratic private noise vs X (MODEL sd .0003-.0005)
res = {}
for nb in (600, 900, 1200):
    out = {k: [] for k in ('K9N-a', 'FN/FNG', 'K9 clean', 'cluster fork', 'best of K9N-a + FNG', 'best of K9N-a + clean')}
    for _ in range(M):
        B = band(nb)
        U = rng.normal(.004, .002, 120) + rng.normal(0, .0018, 120)        # GUESS own-strong teams above the band
        Lw = rng.normal(-.004, .0015, 600) + rng.normal(0, .0018, 600)      # GUESS near-miss teams below the band
        others = np.concatenate([B, U, Lw])
        rFN = np.clip(rng.normal(.90, .025), .84, .96)                      # GUESS prior on r_FN (one-factor centre ~.89)
        mu_a = .0010 - .031 * (rFN - .90) + rng.normal(0, .0006)            # MODEL offset curve (b_shake) + model error
        a = mu_a + rng.normal(0, .0004)
        f = mu_a + .0002 + .0002 * (rFN > .92) + rng.normal(0, .0004)       # MODEL: FN/FNG ~+.0002, G helps at high r_FN
        c = rng.normal(-.0040, .0008) + rng.normal(0, .0008)                # MODEL clean K9 offset
        cl = rng.normal(0, .0002) + rng.normal(0, .0004)
        rank = lambda x: 1 + int((others > x).sum())
        out['K9N-a'].append(rank(a)); out['FN/FNG'].append(rank(f)); out['K9 clean'].append(rank(c))
        out['cluster fork'].append(rank(cl)); out['best of K9N-a + FNG'].append(min(rank(a), rank(f)))
        out['best of K9N-a + clean'].append(min(rank(a), rank(c)))
    res[nb] = out
print(f'T {T}: gold <= {GOLD}, silver <= {SILVER}, bronze <= {BRONZE}  (Kaggle 1000+ team rules)')
for nb, out in res.items():
    print(f'\nN band size {nb} (GUESS)')
    for k, v in out.items():
        v = np.array(v)
        print(f'  {k:24s} median rank {np.median(v):6.0f}  P(gold) {np.mean(v <= GOLD):.3f}  P(silver+) {np.mean(v <= SILVER):.3f}  '
              f'P(bronze+) {np.mean(v <= BRONZE):.3f}')
