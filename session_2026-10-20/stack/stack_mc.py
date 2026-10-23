# stack_mc.py (10-20 ~23:55 IST, Claude, STACK). MODEL only (arithmetic on stated inputs, no data). Expected public / private
# deltas vs pw30 (e-4 macro) of the surviving components and of the proposed stacks, with a SHARED scenario draw (train-like
# hidden mix / extra co-firing carries signal / gold58-like mix) so that lomm05 and MCLmm errors are correlated, as they should be
# (both read the gold74-vs-report coupling through the same raking idea).
# Inputs (sources in inventory.md sec 2):
#   pw25 population mu ~ N(+0.40, 0.80)            (VAL_1020 / implied2 population +0.50, skeptic -0.27..+0.90, stack RAK ~0, OAI pub +1.0)
#   lomm05 by scenario (S1 / S2 / S3): N(+1.10, 1.2) / N(-0.48, 1.2) / N(-0.26, 1.6), haircut .85
#          (S1 = mean of 3mean raked x .73 = 1.54 (NIGHT) and c4-nest raked x .7 = 0.7 (stack_rak2); S2 / S3 as skep_c12p_post)
#   MCLmm by scenario: N(+2.30, 1.0) / N(0.00, 0.8) / N(+1.30, 1.2), haircut .75 (search multiplicity: MM came out of the census)
#          (S1 = stack_rak2 c4-nest raked +3.1..+3.6 x .7; S3 = gold58 +7.6 x .42 shrunk ~60 % toward 0)
#   scenario weights .40 / .35 / .25 (NIGHT used .45 / .30 / .25 for lomm05; S2 is up-weighted here for MCLmm's thinner basis)
#   interaction I(pw25, lomm05) ~ N(+0.16, 0.12)   (OAI +0.25 in 25 / 25 cells, RAK +0.14, REP +0.27, G58 0; x .8)
#   MCLmm is column-disjoint from pw25 / lomm05: I = 0 exactly (macro AUC is a per-column sum).
#   public paired read sd vs pw30: pw25 1.4 (CLAIMED VAL), lomm05 1.01 (CLAIMED skep_c12p), MCLmm 2.5 (MODEL: gold58 sd 7.1 x sqrt(58/390));
#   read errors of different columns independent; private sd = public x sqrt(390 / 910).
#   gap pw30 -> .960 display: lognormal, median 0.75, [p10 0.3, p90 1.8] (CLAIMED implied2).
import numpy as np
rng = np.random.default_rng(1021); N = 400000
sc = rng.choice(3, N, p=[.40, .35, .25])
def scen(means, sds, hc):
    m = np.array(means)[sc]; s = np.array(sds)[sc]; return hc * (m + s * rng.standard_normal(N))
mu = {'pw25': rng.normal(.40, .80, N), 'lomm05': scen([1.10, -.48, -.26], [1.2, 1.2, 1.6], .85), 'MCLmm': scen([2.30, 0., 1.30], [1.0, .8, 1.2], .75)}
Ipl = rng.normal(.16, .12, N)
spub = {'pw25': 1.4, 'lomm05': 1.01, 'MCLmm': 2.5}
epub = {k: rng.normal(0, s, N) for k, s in spub.items()}
k_pr = np.sqrt(390 / 910.)
epri = {k: rng.normal(0, s * k_pr, N) for k, s in spub.items()}
gap = np.exp(rng.normal(np.log(.75), (np.log(1.8) - np.log(.75)) / 1.2816, N))
STACKS = {'pw25 alone': ['pw25'], 'lomm05 alone': ['lomm05'], 'MCLmm alone (S_safe)': ['MCLmm'], 'pw25+MCLmm (S_alt)': ['pw25', 'MCLmm'],
          'lomm05+MCLmm (S_mid)': ['lomm05', 'MCLmm'], 'pw25+lomm05': ['pw25', 'lomm05'], 'pw25+lomm05+MCLmm (S_wide)': ['pw25', 'lomm05', 'MCLmm']}
q = lambda x: '%+.2f [%+.2f, %+.2f]' % (x.mean(), np.percentile(x, 10), np.percentile(x, 90))
print('%-28s | %-24s | %-6s | %-24s | %-8s | %-8s | %s' % ('stack (vs pw30, e-4 macro)', 'population mu [p10,p90]', 'P(mu>0)', 'private [p10,p90]', 'P(pub>0)', 'P(.960)', 'E[priv x 1(pub>0)] (R1 value alone)'))
for nm, parts in STACKS.items():
    m = sum(mu[p] for p in parts) + (Ipl if {'pw25', 'lomm05'} <= set(parts) else 0.)
    pub = m + sum(epub[p] for p in parts); pri = m + sum(epri[p] for p in parts)
    print('%-28s | %-24s | %.2f   | %-24s | %.2f     | %.2f     | %+.2f' % (nm, q(m), (m > 0).mean(), q(pri), (pub > 0).mean(), (pub > gap).mean(), (pri * (pub > 0)).mean()))
print('component P(mu>0): ' + ' | '.join('%s %.2f (mean %+.2f)' % (k, (v > 0).mean(), v.mean()) for k, v in mu.items()))
print('corr(mu lomm05, mu MCLmm) %.2f' % np.corrcoef(mu['lomm05'], mu['MCLmm'])[0, 1])

# ---- slates under R1 (A = max public among pw30 and the new entries). An entry's public delta = sum over its components of
# (mu_c + e_pub_c) (+ I for pw25 + lomm05): the SAME component's read error is shared by every entry that contains it (same column change
# on the same 390 knees). Private of the pick = sum of (mu_c + e_priv_c) (+ I). Values vs keeping pw30 (e-4 macro).
ENT = {'pw25': ['pw25'], 'lomm05': ['lomm05'], 'S_safe=pw30+MCLmm': ['MCLmm'], 'S_alt=pw25+MCLmm': ['pw25', 'MCLmm'],
       'S_mid=pw30+lomm05+MCLmm': ['lomm05', 'MCLmm'], 'S_wide=pw25+lomm05+MCLmm': ['pw25', 'lomm05', 'MCLmm']}
def ent(parts, which):
    e = epub if which == 'pub' else epri
    return sum(mu[p] + e[p] for p in parts) + (Ipl if {'pw25', 'lomm05'} <= set(parts) else 0.)
SLATES = {'NIGHT {pw25, lomm05}': ['pw25', 'lomm05'], '{pw25}': ['pw25'], '{S_safe}': ['S_safe=pw30+MCLmm'], '{S_wide}': ['S_wide=pw25+lomm05+MCLmm'],
          '{S_alt}': ['S_alt=pw25+MCLmm'], '{S_safe, S_wide}': ['S_safe=pw30+MCLmm', 'S_wide=pw25+lomm05+MCLmm'],
          '{S_alt, S_wide}': ['S_alt=pw25+MCLmm', 'S_wide=pw25+lomm05+MCLmm'], '{pw25, S_alt}': ['pw25', 'S_alt=pw25+MCLmm'],
          '{pw25, lomm05, S_wide}': ['pw25', 'lomm05', 'S_wide=pw25+lomm05+MCLmm'],
          '{pw25, S_safe, S_wide}': ['pw25', 'S_safe=pw30+MCLmm', 'S_wide=pw25+lomm05+MCLmm'],
          '{pw25, lomm05, S_safe, S_wide}': ['pw25', 'lomm05', 'S_safe=pw30+MCLmm', 'S_wide=pw25+lomm05+MCLmm']}
print('\nSLATES under R1 (adds to pw30; r6pw20 is B and never A, so it is left out here)')
print('%-34s | %-8s | %-26s | %-10s | %-10s | %s' % ('slate', 'P(move)', 'E private of A vs pw30', 'P(A worse)', 'P(.960)', 'E public gain of best-id'))
for nm, es in SLATES.items():
    PUBm = np.column_stack([np.zeros(N)] + [ent(ENT[e], 'pub') for e in es]); PRIm = np.column_stack([np.zeros(N)] + [ent(ENT[e], 'pri') for e in es])
    a = PUBm.argmax(1); pv = PRIm[np.arange(N), a]; pb = PUBm[np.arange(N), a]
    print('%-34s | %.2f     | %-26s | %.2f       | %.2f       | %+.2f' % (nm, (a > 0).mean(), q(pv), (pv < 0).mean(), (pb > gap).mean(), pb.mean()))
