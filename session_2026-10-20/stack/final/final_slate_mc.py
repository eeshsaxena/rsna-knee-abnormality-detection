# final_slate_mc.py (10-21 ~00:15 IST, Claude, STACK FINAL synthesis). MODEL only: arithmetic on stated inputs, no data read.
# Reuses the world() / slate() model of stack/skeptic/skep_value_mc.py VERBATIM (its source is exec'd up to its first report line,
# with its log handle redirected), and adds the quantities the synthesis needs:
#   (a) whole-slate E private of A, p10 of A's private, P(A < pw30), P(A at dose .3), P(.960 shown) under joint / LB-only / brief gaps;
#   (b) the same under the lomm05 skeptic prior N(0, .9) with I(pw25, lomm05) tied to N(0, .12);
#   (c) increment of S_wide over {pw25, S_pl} per prior.
import numpy as np, os, io
SRC = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
LOGP = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/final/final_slate_mc.log'
src = open(SRC, encoding='utf-8').read()
cut = src.index("P('MODEL skep_value_mc.py")
src = src[:cut].replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
g = {'io': io}; exec(compile(src, SRC, 'exec'), g)
world, slate, N = g['world'], g['slate'], g['N']
L = open(LOGP, 'w')
def P(s=''):
    print(s, flush=True); L.write(s + '\n')
DOSE30 = {'lomm05'}            # entries at A_OA .3 besides pw30 itself (index 0); S_pl* / S_wide / S_alt / pw25 run A_OA .25
SLATES = [['pw25'], ['pw25', 'lomm05'], ['pw25', 'S_pl*'], ['pw25', 'S_pl*', 'S_wide'], ['pw25', 'S_pl*', 'S_alt', 'S_wide'],
          ['pw25', 'lomm05', 'S_alt', 'S_wide'], ['pw25', 'S_wide']]
PRI = ['stack', 'skep2', 'zero', 'neg']
GAPS = (('joint', (.75, 1.8)), ('LB-only', (.5, 1.3)), ('brief', (1.4, 2.5)))
P('MODEL final_slate_mc.py | N %d | e-4 macro vs pw30 57425970 | world()/slate() = stack/skeptic/skep_value_mc.py verbatim | R1: A = max public among {pw30} + probes; r6pw20 = B, never A' % N)
P('S_pl* is the built spl kernel (nb e641e88f); the model treats it as pw25 + lomm05 + I, exactly as skep_value_mc.py.')
for lomm in ('stack', 'skeptic'):
    P('\n== lomm05 prior %s ==' % lomm)
    P('   %-34s | %-7s | %-28s | %-6s | %-8s | %-8s | %-20s' % ('slate (A probes; r6pw20 = B)', 'prior', 'E priv A [p10, p90]', 'P(A<0)', 'P(A@.3)', 'P(A=pw30)', 'P(.960) joint/LB/brief'))
    for pr in PRI:
        Ws = {gn: world(pr, gm, lomm) for gn, gm in GAPS}
        for es in SLATES:
            s = {gn: slate(Ws[gn], es) for gn, _ in GAPS}; sj = s['joint']; a = sj['a']
            at30 = (a == 0) | np.isin(a, [i + 1 for i, e in enumerate(es) if e in DOSE30])
            P('   %-34s | %-7s | %+.2f [%+.2f, %+.2f]     | %.2f   | %.2f     | %.2f      | %.2f / %.2f / %.2f' % (
                '{%s}' % ', '.join(es), pr, sj['priv'].mean(), np.percentile(sj['priv'], 10), np.percentile(sj['priv'], 90), (sj['priv'] < 0).mean(),
                at30.mean(), (a == 0).mean(), s['joint']['p960'], s['LB-only']['p960'], s['brief']['p960']))
    P('   increment of S_wide over {pw25, S_pl*} (joint gap):')
    for pr in ['stack', 'pessimistic', 'skeptic-null', 'recal', 'skep2', 'zero', 'neg']:
        W = world(pr, (.75, 1.8), lomm); b = slate(W, ['pw25', 'S_pl*']); x = slate(W, ['pw25', 'S_pl*', 'S_wide'])
        P('     %-12s %+.2f  (P(.960) %.2f -> %.2f; P(A<0) %.2f -> %.2f)' % (pr, x['priv'].mean() - b['priv'].mean(), b['p960'], x['p960'], (b['priv'] < 0).mean(), (x['priv'] < 0).mean()))
P('FINAL_SLATE_MC DONE')
L.close()
