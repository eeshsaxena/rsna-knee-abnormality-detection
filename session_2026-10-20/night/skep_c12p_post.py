# skep_c12p_post.py (10-20 night, Claude, SKEPTIC C12'): fit the gain-vs-gap law on the pw30-base 4-label box read (skep_box_c12p.log),
# then the corrected posterior for C12' (macro e-4, public units) under two hidden compositions, P(best-id move), private, R1 value.
# MODEL arithmetic on MEASURED inputs (named). Local only.
import re
import numpy as np
from scipy.stats import norm
D = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/'
OUT = open(D + 'skep_c12p_post.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()
rows = []
for ln in open(D + 'skep_box_c12p.log'):
    m = re.search(r'on pw30: a\.05 ([+-][\d.]+) \(sd [\d.]+\) a\.10 ([+-][\d.]+).*pw30 base rho [\d.]+ q [\d.-]+ gap ([+-][\d.]+)', ln)
    if m: rows.append([float(m.group(3)), float(m.group(1)), float(m.group(2))])
R = np.array(rows)
for j, a in ((1, .05), (2, .10)):
    b, c = np.polyfit(R[:, 0], R[:, j], 1); res = R[:, j] - (b * R[:, 0] + c)
    P('box law a %.2f (33 cells, pw30 base): gain_label = %.4f * gap %+.5f | break-even gap %.3f | resid sd %.5f' % (a, b, c, -c / b, res.std()))
b05, c05 = np.polyfit(R[:, 0], R[:, 1], 1)
AMP = .86 / 12 * 1e4  # label -> macro e-4, E-OA gap-matched calibration (CLAIMED recal.md)
SIG = 1.01            # public read sd at a .05 (MEASURED skep_c12p.log D)
SIGP = .59            # private
rng = np.random.default_rng(0); N = 400000
def scen(name, rho_m, rho_s, q_m, q_s, law_s=.0004):
    gap = rng.normal(rho_m, rho_s, N) - rng.normal(q_m, q_s, N)
    mu = (b05 * gap + c05 + rng.normal(0, law_s, N)) * AMP
    return name, mu
SC = [scen('train-like (rho .40+-.07, q .18+-.12)', .40, .07, .18, .12), scen('gold-like (rho .21+-.12, q .15+-.12)', .21, .12, .15, .12),
      scen('recal Route O as stated (rho .206, q .096+-.05)', .206, .02, .096, .05)]
def summ(name, mu, hair=1.0):
    mu = mu * hair; pub = mu + rng.normal(0, SIG, len(mu)); prv = mu + rng.normal(0, SIGP, len(mu)); mv = pub > 0
    P('%-52s haircut %.1f | mu %+.2f [p10 %+.2f, p90 %+.2f] P(mu>0) %.2f | P(move) %.2f | private %+.2f P(priv>0) %.2f | R1 E[priv*move] %+.2f | E[mu|move] %+.2f P(mu<0|move) %.2f' % (
        name, hair, mu.mean(), np.percentile(mu, 10), np.percentile(mu, 90), (mu > 0).mean(), mv.mean(), prv.mean(), (prv > 0).mean(),
        (prv * mv).mean(), mu[mv].mean(), (mu[mv] < 0).mean()))
for nm, mu in SC:
    summ(nm, mu); summ(nm, mu, .7)
for w in (.5, .7):
    k = rng.random(N) < w; mu = np.where(k, SC[0][1], SC[1][1]); summ('mixture P(train-like) %.1f' % w, mu); summ('mixture P(train-like) %.1f' % w, mu, .7)
# ---- skeptic final (added after skep_box_c12p_mem.log): c4's own members R6 / S6fall on OAI give q/rho ~.85-1.2 on the pw30 base,
# so for the train-like case split q into H_noSig (extra RSNA co-firing carries no LOA label signal, trio-style) and H_prop (q scales
# with rho as in the members on OAI). gold-like case: rho .21 (gold58), q .17.
P('---- skeptic final scenarios (macro e-4, public units; law a .05 from the 4-label pw30 box) ----')
S = {'train-like H_noSig (rho .38+-.07, q .18+-.08)': scen('', .38, .07, .18, .08)[1], 'train-like H_prop (rho .38+-.07, q .36+-.08)': scen('', .38, .07, .36, .08)[1],
     'gold-like (rho .21+-.12, q .17+-.08)': scen('', .21, .12, .17, .08)[1]}
for k, v in S.items(): summ(k, v)
u = rng.random(N); mix = np.where(u < .35, S['train-like H_noSig (rho .38+-.07, q .18+-.08)'], np.where(u < .70, S['train-like H_prop (rho .38+-.07, q .36+-.08)'], S['gold-like (rho .21+-.12, q .17+-.08)']))
summ('SKEPTIC MIX (.35 noSig / .35 prop / .30 gold-like)', mix); summ('SKEPTIC MIX (.35 noSig / .35 prop / .30 gold-like)', mix, .7)
