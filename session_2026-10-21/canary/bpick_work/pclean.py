# pclean.py: P(clean hidden run) for an r6pw20 resubmit on 10-22. MODEL on MEASURED counts; weights GUESS.
# MEASURED: S6F-lineage hidden runs before 10-21 all scored at normal displays (.956-.959): 15 in the API list (10-18..10-20) plus
# ~10 earlier from submits.log / timing.md (S6F 57201874 .. a25 57316395) -> ~25 clean of 25. 10-21: 2 clean of 5 (canary .916 x3).
import numpy as np
rng = np.random.default_rng(7); M = 2_000_000
out = open('pclean.log', 'w')
def P(s): print(s); out.write(s + '\n')
p_per = rng.beta(1 + 2, 1 + 3, M)        # 10-21 regime persists: Beta(3, 4) on p_clean
p_base = rng.beta(1 + 25, 1 + 0, M)      # back to the pre-10-21 regime: Beta(26, 1)
for wper in (.3, .5, .6, .7, .9):
    per = rng.random(M) < wper; p = np.where(per, p_per, p_base)
    one = p.mean(); two = (1 - (1 - p) ** 2).mean(); three = (1 - (1 - p) ** 3).mean()
    P('w(persist) %.1f: P(clean) 1 copy %.3f | P(>=1 clean) 2 copies %.3f | 3 copies %.3f' % (wper, one, two, three))
P('regime means: persist %.3f, baseline %.3f' % (p_per.mean(), p_base.mean()))
out.close()
