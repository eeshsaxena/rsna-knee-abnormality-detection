# readtable_mc.py (10-21 ~08:45 IST, Claude, DAY track "read-decision table"). MODEL only. RESEARCH ONLY: local CPU arithmetic,
# no OAI rows, no box, no LB / Kaggle calls, nothing written outside session_2026-10-21/day. Run with `python -B` (no bytecode into
# the read-only val/work or private/dose_work folders).
#
# What it does: re-creates the dose.md / skep_dose.py world (dose_mc.fit() + the same draw construction with the SAME seeds, so the
# pw20 / pw25 / pw30 / R6pw20 numbers reproduce private/dose_work/dose_base.log), adds the A22 stack entries spl / swide / salt from
# the stack skeptic model (stack/skeptic/skep_value_mc.py priors: lomm05 scenario prior, I(pw25, lomm05), MCLmm skep2, read sd
# public / private lomm05 1.00 / .62, MCLmm 2.40 / 1.50, corr(pw25 read, lomm05 read) -.2), and then CONDITIONS on the observable
# A22 read outcomes:
#   W = API best-id after the reads = argmax public over {pw30, pw25, spl, swide, salt}  ('none' = pw30 stays)
#   D = r6pw20 display (957 / 958 / 959) from dose_mc's display model (R6 at u ~ .10 of the .958 band)
# For every (W, D) cell: P(cell); A (R1) = W; A (private-first) = pw20 57425269; B (A22) = r6pw20 if D >= 958 else R6 57248084;
# E private margin of A and of the pair (max), P(silver) on the step curve (10-19 field, picks2 / dose.md) and on the smooth curve
# (shakeup.md), bronze / r150 on the smooth curve; and the conditional value of possible 10-22 entries under R1:
#   built:   lomm05@pw30 (night c12 kernel), smid (pw30 + lomm05 + MCLmm), ssafe (pw30 + MCLmm)
#   unbuilt: pl20 (pw20 + lomm05), wide20 (pw20 + lomm05 + MCLmm)     [would need build + gate + amendment + Eesh GO]
# Units: e-4 private macro margin over the heliosli copy block (picks2 / dose.md ruler). Scenario mix S0 .6 / C+ .1 / C- .1 / SITE .2.
# Args (key=value): any dose_mc arg (prox, psh, ps, ...), plus lomm=stack|skeptic|oai, mcl=skep2|zero|stack, i20=2.0 (I at pw20 as a
# multiple of I(pw25)), rL=.9 (corr of the lomm05 read error between bases), perr=.02 (P(r6pw20 ERROR), GUESS), tag=...
import sys, os, io, json, numpy as np
HERE = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-21/day'
DW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private/dose_work'
SKV = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.py'
SHK = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/private/work/curve_base.json'
own = dict(a.split('=', 1) for a in sys.argv[1:] if a.split('=', 1)[0] in ('lomm', 'mcl', 'i20', 'rL', 'perr', 'tag2'))
sys.argv = ['dose_mc.py'] + [a for a in sys.argv[1:] if a.split('=', 1)[0] not in own]
sys.dont_write_bytecode = True
sys.path.insert(0, DW)
import dose_mc as D
LOMM = own.get('lomm', 'stack'); MCLP = own.get('mcl', 'skep2'); I20 = float(own.get('i20', 2.0)); RL = float(own.get('rL', .9))
PERR = float(own.get('perr', .02)); TAG = D.TAG + ('_' + own['tag2'] if 'tag2' in own else '')
LOG = open(HERE + '/readtable_%s.log' % TAG, 'w')


def P(s=''):
    print(s, flush=True); LOG.write(s + '\n')


# ---------------- 1. dose world: verbatim draw construction of skep_dose.py (same seeds) ----------------
F = D.fit(); rng = np.random.default_rng(D.SEED + 7)
Z = np.load(D.PWK + '/post_base_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); M0 = PV.shape[0]; N = M0 * D.NREP; ib = np.tile(np.arange(M0), D.NREP)
OUR = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW')}
j = rng.choice(len(F['w']), N, p=F['w'] / F['w'].sum())
sh, s, rc4 = F['sh'][j], F['s'][j], F['rc4'][j]; G = {t: F['G'][t][j] for t in D.SH}; P0 = {t: F['P0'][t][j] for t in D.SH}
x20 = F['x20'][j]
SDP = np.sqrt(390 / 910); nc = F['Ls'].shape[0]
Z1 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP; Z2 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP
NP = {t: Z1[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}; NV2 = {t: Z2[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}
del Z1, Z2
kc = rng.normal(D.K_MU, D.K_SD, N) + rng.normal(0, .05, N); epsx = D.EPS * rng.standard_normal(N)
uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)
pubg = kc * P0['PW20'] + epsx + .8 * (G['PW20'] - P0['PW20']) + .6 * F['sdpub']['PW20'] * rng.standard_normal(N)
disp = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958))
dS = rng.normal(0, D.SITE_D, N); rcS = rng.normal(0, D.SITE_RC, N); sS = np.exp(rng.normal(0, D.SITE_S, N))
SCEN = {'S0': dict(d=0.0), 'C+': dict(d=.5), 'C-': dict(d=-.5), 'SITE': dict(d=dS, rc=rcS, sf=sS, nsc=D.SITE_N)}
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}

# ---------------- 2. stack components (skep_value_mc.py priors; separate RNG so section 1 is unchanged) ----------------
src = open(SKV, encoding='utf-8').read(); src = src[:src.index("P('MODEL skep_value_mc.py")]
src = src.replace("OUT = open('C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/stack/skeptic/skep_value_mc.log', 'w')", "OUT = io.StringIO()")
assert 'OUT = io.StringIO()' in src
gv = {'io': io}; exec(compile(src, SKV, 'exec'), gv); gv['N'] = N
rS = np.random.default_rng(2110)
sc = rS.choice(3, N, p=[.40, .35, .25])
if LOMM == 'stack':
    muL = .85 * (np.array([1.10, -.48, -.26])[sc] + np.array([1.2, 1.2, 1.6])[sc] * rS.standard_normal(N)); I = rS.normal(.16, .12, N)
elif LOMM == 'skeptic':
    muL = rS.normal(0., .9, N); I = rS.normal(0., .12, N)
else:                                   # 'oai': lomm05's own skeptic OAI-only mean -0.9 (final_slate_mc3 stress)
    muL = rS.normal(-.9, .9, N); I = rS.normal(0., .12, N)
popg = np.where(sc == 0, False, np.where(sc == 2, True, rS.random(N) < .5))
if MCLP == 'skep2':
    muM = gv['mcl_skep2'](rS, popg)[0]
elif MCLP == 'zero':
    muM = np.zeros(N)
else:
    muM = .75 * (np.array([2.30, 0., 1.30])[sc] + np.array([1.0, .8, 1.2])[sc] * rS.standard_normal(N))
# public read errors (e-4). pw25's own read error vs pw30 is inside dose_mc's G; lomm05's read error is correlated -.2 with it.
e25 = ((G['PW25'] - P0['PW25']) - (G['PW30'] - P0['PW30'])) * 1e4; z25 = (e25 - e25.mean()) / e25.std()
zL = rS.standard_normal(N); eLp = 1.00 * (-.2 * z25 + np.sqrt(1 - .04) * zL)                 # lomm05 read on the pw25 base
eLp_o = 1.00 * (RL * (eLp / 1.00) + np.sqrt(1 - RL ** 2) * rS.standard_normal(N))           # same block on another base (GUESS corr)
eMp = 2.40 * rS.standard_normal(N)                                                          # MCL column is base-independent: shared
eLq = .62 * rS.standard_normal(N); eLq_o = .62 * (RL * eLq / .62 + np.sqrt(1 - RL ** 2) * rS.standard_normal(N))
eMq = 1.50 * rS.standard_normal(N)

# public (e-4, vs c4; pw30 is the current best-id by construction of the fit)
PUB = {'pw30': G['PW30'] * 1e4, 'pw25': G['PW25'] * 1e4}
PUB['spl'] = PUB['pw25'] + muL + I + eLp
PUB['swide'] = PUB['spl'] + muM + eMp
PUB['salt'] = PUB['pw25'] + muM + eMp
PUB['pw20'] = G['PW20'] * 1e4
# 10-22 options (public)
PUB['lomm05'] = PUB['pw30'] + muL + eLp_o
PUB['smid'] = PUB['lomm05'] + muM + eMp
PUB['ssafe'] = PUB['pw30'] + muM + eMp
PUB['pl20'] = PUB['pw20'] + muL + I20 * I + eLp_o
PUB['wide20'] = PUB['pl20'] + muM + eMp
WIN = ['pw30', 'pw25', 'spl', 'swide', 'salt']
W = np.argmax(np.column_stack([PUB[k] for k in WIN]), 1)
pos = lambda x: x20 + (x - G['S20'] * 1e4) * .1                                             # position in the .959 band (1 = .960)
pubW = np.column_stack([PUB[k] for k in WIN])[np.arange(N), W]
show960 = pos(pubW) >= 1.0
OPTS = {'lomm05@pw30 (built)': ['lomm05'], 'smid (built)': ['smid'], 'ssafe (built)': ['ssafe'], 'pl20 (unbuilt)': ['pl20'],
        'wide20 (unbuilt)': ['wide20'], 'pl20 + wide20 (unbuilt)': ['pl20', 'wide20'], 'all 3 built': ['lomm05', 'smid', 'ssafe']}

# ---------------- 3. curves ----------------
CV = np.load(D.P2 + '/curves_base.npz'); g19 = CV['grid'] * 1e4
c19 = lambda k, x: np.interp(x, g19, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
CB = json.load(open(SHK)); ms = np.array(sorted(float(k) for k in CB)); key = {float(k): k for k in CB}


def cshk(name, x):                       # = skep_dose.py cshk
    y = np.array([CB[key[m]][name] for m in ms]); slope = (y[-1] - y[-2]) / (ms[-1] - ms[-2])
    out = np.interp(x, ms, y, left=0.0, right=float(y[-1])); hi = x > ms[-1]; out[hi] = np.minimum(1.0, y[-1] + slope * (x[hi] - ms[-1]))
    out[x < ms[0]] = 0.0
    return out


def met(x):
    return np.column_stack([x, c19('silver', x), c19('r150', x), cshk('P(r<=316)', x), cshk('P(r<=633)', x), cshk('P(r<=150)', x)])


MN = ['E', 'silv_step', 'r150_step', 'silv_smooth', 'bronze_smooth', 'r150_smooth']

# ---------------- 4. private per scenario, conditional accumulation ----------------
ROWS = [(w, d) for w in range(5) for d in (957, 958, 959)]
acc = {}            # key -> mixed metrics array per draw (accumulate weighted per-draw metric, then average within cells)


def add(k, arr, wgt):
    acc[k] = acc.get(k, 0.) + (wgt * arr).astype(np.float32)


for sn, scd in SCEN.items():
    pop = F['popf'](sh + scd['d'], rc4 + scd.get('rc', 0.0), s * scd.get('sf', 1.0)); nsc = scd.get('nsc', 1.0)
    X = {}
    for t, nm in (('PW20', 'pw20'), ('PW25', 'pw25'), ('PW30', 'pw30')):
        X[nm] = (OUR['c4'] + pop[t] + nsc * NP[t]) * 1e4
    X['R6'] = OUR['R6u'] * 1e4
    X['r6pw20'] = (OUR['R6u'] + kc * pop['PW20'] + epsx + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])) * 1e4
    X['spl'] = X['pw25'] + muL + I + nsc * eLq
    X['swide'] = X['spl'] + muM + nsc * eMq
    X['salt'] = X['pw25'] + muM + nsc * eMq
    X['lomm05'] = X['pw30'] + muL + nsc * eLq_o
    X['smid'] = X['lomm05'] + muM + nsc * eMq
    X['ssafe'] = X['pw30'] + muM + nsc * eMq
    X['pl20'] = X['pw20'] + muL + I20 * I + nsc * eLq_o
    X['wide20'] = X['pl20'] + muM + nsc * eMq
    AW = np.column_stack([X[k] for k in WIN])[np.arange(N), W]          # A under R1 (A22)
    Bv = np.where(disp >= 958, X['r6pw20'], X['R6'])                     # B under A22
    BvE = X['R6']                                                        # B if r6pw20 ERRORs
    Bc = X['r6pw20']                                                     # counterfactual: B = r6pw20 whatever it shows
    wg = MIX[sn]
    add('A_R1', met(AW), wg); add('A_PF', met(X['pw20']), wg)
    add('P_R1', met(np.maximum(AW, Bv)), wg); add('P_PF', met(np.maximum(X['pw20'], Bv)), wg)
    add('P_R1_err', met(np.maximum(AW, BvE)), wg); add('P_PF_err', met(np.maximum(X['pw20'], BvE)), wg)
    add('P_R1_Bc', met(np.maximum(AW, Bc)), wg); add('P_PF_Bc', met(np.maximum(X['pw20'], Bc)), wg)
    add('P_pw30fix', met(np.maximum(X['pw30'], Bv)), wg); add('P_pw25fix', met(np.maximum(X['pw25'], Bv)), wg)
    for on, ks in OPTS.items():
        PU2 = np.column_stack([pubW] + [PUB[k] for k in ks]); PR2 = np.column_stack([AW] + [X[k] for k in ks])
        a2 = PU2.argmax(1); A2 = PR2[np.arange(N), a2]
        add('OPT|' + on, met(np.maximum(A2, Bv)), wg); add('OPTA|' + on, A2, wg)
        if sn == 'S0':
            add('OPTsw|' + on, (a2 > 0).astype(float), 1.0)
    if sn == 'S0':
        add('pw20pub_vs_W', PUB['pw20'] - pubW, 1.0)

# ---------------- 5. report ----------------
WN = ['none (pw30 stays)', 'pw25', 'spl', 'swide', 'salt']
P('MODEL readtable_mc.py | tag %s | dose_mc fit prox %s sh %s s %s field %s | ESS %.0f | N %d | lomm05 prior %s, MCLmm prior %s, I20 = %.1f x I, rL %.2f' % (
    TAG, D.PROX, D.PSH, D.PS, D.FS, F['ess'], N, LOMM, MCLP, I20, RL))
P('Scenario mix S0 .6 / C+ .1 / C- .1 / SITE .2 (dose.md). Margin e-4 = private macro over the copy block. B (A22) = r6pw20 if display >= .958 else R6.')
P('Sanity vs dose_base.log / skep_dose_base.log: P(R6pw20 957/958/959) %.3f/%.3f/%.3f (.006/.504/.491); pub pw25 - pw30 %+.2f (+0.11)' % (
    (disp == 957).mean(), (disp == 958).mean(), (disp == 959).mean(), (PUB['pw25'] - PUB['pw30']).mean()))
P('Marginal P(best-id) none/pw25/spl/swide/salt: %s | P(best-id shows .960) %.3f' % (' / '.join('%.3f' % (W == i).mean() for i in range(5)), show960.mean()))

def fmt(v):
    return '%+.2f | %.3f %.3f | %.3f %.3f %.3f' % (v[0], v[1], v[3], v[2], v[5], v[4])


OUTJ = dict(tag=TAG, lomm=LOMM, mcl=MCLP, N=int(N), cells={}, marg={})
def cellrow(mask, label):
    pc = mask.mean()
    r = dict(P=float(pc), P960=float(show960[mask].mean()) if mask.any() else 0.)
    for k in ('A_R1', 'A_PF', 'P_R1', 'P_PF', 'P_R1_err', 'P_PF_err', 'P_R1_Bc', 'P_PF_Bc', 'P_pw30fix', 'P_pw25fix'):
        r[k] = [float(x) for x in acc[k][mask].mean(0)]
    for on in OPTS:
        r['OPT|' + on] = [float(x) for x in acc['OPT|' + on][mask].mean(0)]
        r['OPTA|' + on] = float(acc['OPTA|' + on][mask].mean() - acc['A_R1'][mask, 0].mean())
        r['OPTsw|' + on] = float(acc['OPTsw|' + on][mask].mean())
    r['pw20_minus_W_pub'] = float(acc['pw20pub_vs_W'][mask].mean())
    return r


P('\n== Cells (W = best-id, D = r6pw20 display). Columns: P(cell) | P(A shows .960) || pair R1 {W, B}: E | silver step / smooth | r150 step / smooth | bronze smooth || pair PF {pw20, B}: same || PF - R1 silver step / smooth ==')
for w in range(5):
    for d in (957, 958, 959):
        m = (W == w) & (disp == d)
        r = cellrow(m, None); OUTJ['cells']['%s|%d' % (WIN[w], d)] = r
        a, b = r['P_R1'], r['P_PF']
        P('  %-18s D .%d | P %.3f | .960 %.2f || R1 %s || PF %s || PF-R1 %+.1f / %+.1f pts' % (WN[w], d, r['P'], r['P960'], fmt(a), fmt(b), 100 * (b[1] - a[1]), 100 * (b[3] - a[3])))
P('\n== Marginal by best-id (any display) ==')
for w in range(5):
    m = W == w; r = cellrow(m, None); OUTJ['marg'][WIN[w]] = r
    a, b, aA, bA = r['P_R1'], r['P_PF'], r['A_R1'], r['A_PF']
    P('  %-18s P %.3f | .960 %.2f | E A: R1 %+.2f PF %+.2f (PF-R1 %+.2f) | pair R1 %s || PF %s || PF-R1 %+.1f / %+.1f pts | pub pw20 - W %+.2f' % (
        WN[w], r['P'], r['P960'], aA[0], bA[0], bA[0] - aA[0], fmt(a), fmt(b), 100 * (b[1] - a[1]), 100 * (b[3] - a[3]), r['pw20_minus_W_pub']))
P('\n== Marginal by r6pw20 display ==')
for d in (957, 958, 959):
    m = disp == d; r = cellrow(m, None); OUTJ['marg']['D%d' % d] = r
    a, b, c_, c2 = r['P_R1'], r['P_PF'], r['P_R1_Bc'], r['P_PF_Bc']
    P('  D .%d P %.3f | R1 %s || PF %s || counterfactual B = r6pw20 anyway: R1 %s / PF %s' % (d, r['P'], fmt(a), fmt(b), fmt(c_), fmt(c2)))
m = np.ones(N, bool); r = cellrow(m, None); OUTJ['marg']['all'] = r
P('\n== All draws: R1 %s || PF %s || pw30 fixed %s || pw25 fixed %s' % (fmt(r['P_R1']), fmt(r['P_PF']), fmt(r['P_pw30fix']), fmt(r['P_pw25fix'])))
P('   r6pw20 ERROR (B = R6): R1 %s || PF %s' % (fmt(r['P_R1_err']), fmt(r['P_PF_err'])))

P('\n== 10-22 options under R1 (A = new argmax public; B unchanged). Value = change vs "nothing" in that best-id group: dE pair | d silver step / smooth (pts) | dE of A | P(option becomes A) ==')
for w in list(range(5)) + ['all']:
    m = (W == w) if w != 'all' else np.ones(N, bool)
    base = acc['P_R1'][m].mean(0)
    P('  best-id %s (P %.3f):' % (WN[w] if w != 'all' else 'ALL', m.mean()))
    for on in OPTS:
        v = acc['OPT|' + on][m].mean(0); dA = acc['OPTA|' + on][m].mean() - acc['A_R1'][m, 0].mean(); sw = acc['OPTsw|' + on][m].mean()
        P('     %-26s dE %+.2f | silver %+.1f / %+.1f | dE(A) %+.2f | P(new A) %.2f' % (on, v[0] - base[0], 100 * (v[1] - base[1]), 100 * (v[3] - base[3]), dA, sw))
json.dump(OUTJ, open(HERE + '/readtable_%s.json' % TAG, 'w'), indent=1)
P('READTABLE_MC DONE')
LOG.close()
