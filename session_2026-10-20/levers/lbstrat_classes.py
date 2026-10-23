# lbstrat_classes.py (10-20 ~16:00 IST, levers/LB-strategy lens, Claude). Ledger of every CLEAN R1 read since 10-13 by probe class
# (outcomes MEASURED from PREREG_10-13/10-14/10-15 READ lines, submits.log, our_subs2.csv; classification = judgement, stated), the
# class hit rates vs the 50 % null, a sign-only fit of the class mean m (in units of the probe's own public noise sd) at fixed spread
# t, and the R1 value of a NEW probe of each class (E1 is placed in the weak-ruler class: report-OOF + gold58 only). No data access.
import numpy as np
from scipy.stats import norm, binom, beta
LEDGER = {
    # class: [(entry, base, outcome +1 / -1)]
    'P public-proven swap (other teams\' public notebooks / legs)': [
        ('HR7 57137313', 'HV', 1), ('HR7-RR 57140358 (nartaa r224)', 'HR7', 1), ('V224-exact 57171955', 'HFR7-RR', 1),
        ('V224-FLIP 57174021', 'V224-exact', 1), ('S6F 57201874 (nartaa S6fall)', 'FLIP', 1), ('R6 57248084 (R6 bundle)', 'S6F', 1),
        ('HW 57262115 (heliosli public-tuned table)', 'R6', 1)],
    'M own members / fleets / TTA / students': [
        ('HFR7-RR 57146261 (fleet .2)', 'HR7-RR', 1), ('HFR7-RS 57154866 (own Raptor seeds)', 'HFR7-RR', -1), ('RR-F30 57163771', 'HFR7-RR', -1),
        ('RR-F40 57164843', 'HFR7-RR', -1), ('FF20 57199082', 'FLIP', -1), ('FLRM 57200578 (mirror TTA)', 'FLIP', -1),
        ('S6F+S 57222884 (student)', 'S6F', -1), ('COMBO 57250689 (KC1 on R6)', 'R6', -1), ('R7 late 57291336', 'HW', -1)],
    'K knob moves (weights / table / A224 / S6 weight)': [
        ('hw3 57291203 (S6 .3)', 'HW', -1), ('wr45 57310758', 'HW', -1), ('s615 57313454', 'HW', -1), ('a25 57316395 (A224 .25)', 'HW', 1),
        ('c4 57343435 (4-line gold58 stack)', 'a25', 1), ('c4a15 57344133', 'c4', -1), ('fr100 57382011', 'c4', -1), ('pf10 57382016', 'c4', -1),
        ('c2max 57382007', 'c4', -1), ('c2 57382015', 'c4', -1), ('c3max 57382515', 'c4', -1)],
    'E mechanism + independent expert ruler (OAI)': [('eoa10 57413380', 'c4', 1), ('eoa20 57413368', 'c4', 1), ('eoa20 > eoa10 (dose)', 'eoa10', 1)],
}
BY_EVIDENCE = {
    'weak ruler only (gold58 / gold74 / report OOF)': ['c4 (+)', 'fr100 (-)', 'pf10 (-)', 'c2max (-)', 'c2 (-)', 'c3max (-)', 'HFR7-RS (-, gold58 PASS)',
                                                       'COMBO/KC (-, gold74 +.0068)', 'FF20 (-, OOF +.001)', 'S6F+S (-, OOF)'],
    'independent expert ruler + mechanism (OAI)': ['eoa10 (+)', 'eoa20 (+)'],
    'other teams\' public LB': ['HR7', 'HR7-RR', 'V224', 'FLIP', 'S6F', 'R6', 'HW (all +)'],
    'pure exploration of a tuned knob': ['a25 (+)', 'hw3 (-)', 's615 (-)', 'wr45 (-)', 'c4a15 (-)'],
}
out = []
pr = lambda s: (print(s), out.append(s))
pr('UNREAD / excluded (censored by a parallel move or display tie before the frozen-team method): HF20, HF30, HF-R20, HFR7-R20, S 57218523, KC1 57249503,')
pr('S6W 57264835, fr65 57343462, a15 57344106, pf50 57344538 (each < c4; vs a25 unknown).')
for c, rows in LEDGER.items():
    k = sum(1 for r in rows if r[2] > 0); n = len(rows)
    p_null = binom.cdf(k, n, .5) if k <= n / 2 else 1 - binom.cdf(k - 1, n, .5)
    lo, hi = beta.ppf([.1, .9], k + 1, n - k + 1)
    pr('\n%s: %d / %d positive (hit %.2f, Beta(1,1) 80%% CI %.2f-%.2f); one-sided P vs 50%% null %.3f' % (c, k, n, k / n, lo, hi, p_null))
    for r in rows:
        pr('   %s %s vs %s' % ('+' if r[2] > 0 else '-', r[0], r[1]))
pr('\nby evidence source:')
for c, rows in BY_EVIDENCE.items():
    pr('   %-50s %s' % (c, ', '.join(rows)))
# sign-only class mean at fixed spread t: hit = Phi(m / sqrt(1 + t^2)); posterior mean hit from Beta(k+1, n-k+1)
pr('\nsign-only class mean m (sigma units) and R1 value of a NEW probe, E[theta 1(pub > 0)] = sigma [m Phi(a) + t^2/s phi(a)], a = m/s:')
SIG_E1 = 3.7   # E1 public sd e-4 (gold58 bootstrap .00097 scaled to 390, PUSH_1020 sec 2.2, CLAIMED)
for c, (k, n) in {'K knob': (2, 11), 'M own members': (1, 9), 'weak ruler only': (1, 10), 'P public-proven': (7, 7), 'E mechanism+ruler': (3, 3)}.items():
    h = (k + 1) / (n + 2)
    for t in (.5, .8, 1.2):
        s = np.sqrt(1 + t * t); m = norm.ppf(h) * s; a = m / s
        val = m * norm.cdf(a) + t * t / s * norm.pdf(a); wc = norm.pdf(a) / norm.cdf(a) / s
        rho = t / s; p_true_pos_win = None
        pr('   %-18s hit %.2f | t %.1f -> m %+.2f | new-probe R1 value %+.3f sigma (E1-sized sigma 3.7e-4: %+.2fe-4) | public optimism of a win %.2f sigma' % (
            c, h, t, m, val, val * SIG_E1, wc))
open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/levers/lbstrat_classes.log', 'w').write('\n'.join(out) + '\n')
