# night_slate_mc.py (10-20 night, Claude, NIGHT synthesis). MODEL only, no data. Writes night_slate_mc.log.
# Value of adding ONE A probe on pw30 to the 10-21 slate {r6pw20 (B, never A), pw25 (A probe)} under R1 (A = API best-id = max public).
# Units: macro e-4 vs pw30, population (= expected private) effect. Inputs:
#   pw25: true ~ N(+0.3, 0.8), paired public read sd 1.4 (CLAIMED implied2 via syn_slate_mc.py / VAL_1020).
#   C12' (LOA <- MM a .05 on pw30): read sd 1.01 (MEASURED skep_c12p.log); scenario mixtures below (CLAIMED skep_c12p_post.log for the
#     three skeptic scenarios, sd from their p10/p90; weights per row).
#   E2s (Syn += .2 Eff): read sd 2.9 (MEASURED syn_more.log); skeptic base / proposer from skp_syn_mc.py.
#   stack C6' + C12': C6' component ~ N(-0.1, 0.8) (skc6 central) added to C12'; read sd 1.4 (CLAIMED recal).
# Read noises of different probes vs pw30 are taken as correlated rho_n (0 or .3: both share pw30's public draw).
import numpy as np
rng = np.random.default_rng(1021); N = 1_000_000
OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/night/night_slate_mc.log', 'w')
def P(s): print(s); OUT.write(s + '\n')
def mix(ws, comps):
    k = rng.choice(len(ws), N, p=np.array(ws) / sum(ws)); m = np.array([c[0] for c in comps])[k]; s = np.array([c[1] for c in comps])[k]
    return rng.normal(m, s)
SK = [(1.54, 1.23), (-0.48, 1.23), (-0.26, 1.65)]          # train-like noSig, train-like prop, gold-like (skep_c12p_post.log)
C12 = {
    'skeptic mix .35/.35/.30, haircut 1.0': lambda: mix([.35, .35, .30], SK),
    'skeptic mix .35/.35/.30, haircut 0.7': lambda: .7 * mix([.35, .35, .30], SK),
    'night adj .45/.30/.25, haircut 0.85': lambda: .85 * mix([.45, .30, .25], SK),
    'pessimistic .20/.45/.35, haircut 0.7': lambda: .7 * mix([.20, .45, .35], SK),
    'OAI-only (members at gold74 coupling) N(-0.9, 0.6)': lambda: rng.normal(-0.9, 0.6, N),
    'RSNA-raked-only (3mean, ratio 1.0) N(+2.1, 0.8)': lambda: rng.normal(2.1, 0.8, N),
}
def slate(t25, p25, tx, px):
    pub = np.stack([np.zeros(N), p25, px], 1); tru = np.stack([np.zeros(N), t25, tx], 1)
    a = pub.argmax(1); return tru[np.arange(N), a], a
for rho_n in (0.0, 0.3):
    P('==== read-noise correlation between probes rho_n %.1f' % rho_n)
    for nm, gen in C12.items():
        t25 = rng.normal(.3, .8, N); e1 = rng.normal(0, 1, N); e2 = rng.normal(0, 1, N)
        p25 = t25 + 1.4 * e1
        tx = gen(); px = tx + 1.01 * (rho_n * e1 + np.sqrt(1 - rho_n ** 2) * e2)
        only25 = np.where(p25 > 0, t25, 0.0)
        aloneX = np.where(px > 0, tx, 0.0)
        both, a = slate(t25, p25, tx, px)
        P('C12\' %-50s mu %+.2f P(mu>0) %.2f | alone: P(move) %.2f R1 %+.2f P(adopt & harmful) %.2f | with pw25: slate %+.2f vs pw25 alone %+.2f '
          '(increment %+.2f), P(C12\' = A) %.2f, P(A harmful) %.2f vs %.2f' % (
              nm, tx.mean(), (tx > 0).mean(), (px > 0).mean(), aloneX.mean(), ((px > 0) & (tx < 0)).mean(), both.mean(), only25.mean(),
              both.mean() - only25.mean(), (a == 2).mean(), (both < 0).mean(), (only25 < 0).mean()))
    # stack and E2s for comparison (skeptic central inputs)
    for nm, gen, sd in (('stack C6\'+C12\' (skeptic mix .35/.35/.30 + C6\' N(-0.1,0.8))', lambda: mix([.35, .35, .30], SK) + rng.normal(-.1, .8, N), 1.4),
                        ('E2s skeptic base (P real .30: N(1.0,.7) else N(-0.8,.8))', lambda: np.where(rng.random(N) < .3, rng.normal(1.0, .7, N), rng.normal(-.8, .8, N)), 2.9),
                        ('E2s proposer (P real .5: N(2.8,1.5) else N(-2.0,1.5))', lambda: np.where(rng.random(N) < .5, rng.normal(2.8, 1.52, N), rng.normal(-2.0, 1.52, N)), 2.9)):
        t25 = rng.normal(.3, .8, N); e1 = rng.normal(0, 1, N); e2 = rng.normal(0, 1, N); p25 = t25 + 1.4 * e1
        tx = gen(); px = tx + sd * (rho_n * e1 + np.sqrt(1 - rho_n ** 2) * e2)
        only25 = np.where(p25 > 0, t25, 0.0); both, a = slate(t25, p25, tx, px)
        P('%-56s mu %+.2f | alone R1 %+.2f | with pw25: increment %+.2f, P(= A) %.2f, P(A harmful) %.2f vs %.2f' % (
            nm, tx.mean(), np.where(px > 0, tx, 0).mean(), both.mean() - only25.mean(), (a == 2).mean(), (both < 0).mean(), (only25 < 0).mean()))
    # C12' and E2s both added (two extra probes) under skeptic inputs
    t25 = rng.normal(.3, .8, N); p25 = t25 + 1.4 * rng.normal(0, 1, N)
    tc = mix([.35, .35, .30], SK); pc = tc + 1.01 * rng.normal(0, 1, N)
    te = np.where(rng.random(N) < .3, rng.normal(1.0, .7, N), rng.normal(-.8, .8, N)); pe = te + 2.9 * rng.normal(0, 1, N)
    pub = np.stack([np.zeros(N), p25, pc, pe], 1); tru = np.stack([np.zeros(N), t25, tc, te], 1); sel = tru[np.arange(N), pub.argmax(1)]
    pub2 = np.stack([np.zeros(N), p25, pc], 1); tru2 = np.stack([np.zeros(N), t25, tc], 1); sel2 = tru2[np.arange(N), pub2.argmax(1)]
    P('4-way {pw30, pw25, C12\', E2s} (skeptic inputs, indep noise): %+.2f vs {pw30, pw25, C12\'} %+.2f vs {pw30, pw25} %+.2f' % (
        sel.mean(), sel2.mean(), np.where(p25 > 0, t25, 0).mean()))
OUT.close()
