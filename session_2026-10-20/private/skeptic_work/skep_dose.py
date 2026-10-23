# skep_dose.py (10-20 ~23:30 IST, Claude, SKEPTIC of private/dose.md). RESEARCH ONLY, local CPU, no OAI rows, no box, nothing submitted.
# 1) Re-run dose_mc.fit() + main()'s draw construction with the SAME seeds, verify the dose_base.log numbers (re-derivation).
# 2) Keep the margin draws and re-score them under (a) the 10-19 field curve dose.md used (near step: silver 0 at +3, .57 at +5, 1 at +8)
#    and (b) the shakeup.md fixed-margin curve (work/curve_base.json: lineage noise + common family shock; silver .41/.63/.80/.92 at
#    +5/+8/+13/+20), plus (c) a probit tau sweep. 3) Decompose pw30 vs pw20 spread (population vs private noise).
import sys, os, json, numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(os.path.dirname(HERE), 'dose_work')
sys.argv = ['dose_mc.py'] + sys.argv[1:]
sys.path.insert(0, DW)
import dose_mc as D
F = D.fit(); rng = np.random.default_rng(D.SEED + 7)
Z = np.load(D.PWK + '/post_base_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); M0 = PV.shape[0]; N = M0 * D.NREP; ib = np.tile(np.arange(M0), D.NREP)
OUR = {k: PV[ib, e[k]] for k in ('c4', 'R6u', 'HW')}
j = rng.choice(len(F['w']), N, p=F['w'] / F['w'].sum())
sh, s, rc4 = F['sh'][j], F['s'][j], F['rc4'][j]; G = {t: F['G'][t][j] for t in D.SH}; P0 = {t: F['P0'][t][j] for t in D.SH}
SDP = np.sqrt(390 / 910); nc = F['Ls'].shape[0]
Z1 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP; Z2 = (rng.standard_normal((N, nc)) @ F['Ls'].T) * SDP
NP = {t: Z1[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}; NV2 = {t: Z2[:, 3 * k_:3 * k_ + 3].sum(1) for k_, t in enumerate(D.SH)}
del Z1, Z2
kc = rng.normal(D.K_MU, D.K_SD, N) + rng.normal(0, .05, N); epsx = D.EPS * rng.standard_normal(N)
uR6 = np.clip(rng.normal(.10, .07, N), .01, .4)
pubg = kc * P0['PW20'] + epsx + .8 * (G['PW20'] - P0['PW20']) + .6 * F['sdpub']['PW20'] * rng.standard_normal(N)
disp = np.where(pubg > (1 - uR6) * 1e-3, 959, np.where(pubg < -uR6 * 1e-3, 957, 958))
r1win = G['PW25'] > G['PW30']
dS = rng.normal(0, D.SITE_D, N); rcS = rng.normal(0, D.SITE_RC, N); sS = np.exp(rng.normal(0, D.SITE_S, N))
# lomm05 (night C12', LOA <- MM a .05 on pw30), separate RNG so the dose_mc draws above are unchanged. NIGHT: E +0.41 [-1.43, +2.20]
# (sd ~1.42) vs pw30, paired public read sd 1.01 (CLAIMED night_slate_mc.log); private paired noise 1.01 x .655. MODEL.
rL = np.random.default_rng(777)
LMU = float(D.args.get('lmu', 0.41)); dL = rL.normal(LMU, 1.42, N) * 1e-4; pubL = G['PW30'] + dL + 1.01e-4 * rL.standard_normal(N); nL = 1.01e-4 * SDP * rL.standard_normal(N)
r1x = np.argmax(np.stack([G['PW30'], G['PW25'], pubL], 1), 1)
print('R1 over {pw30, pw25, lomm05}: P(A = pw30 / pw25 / lomm05) %s | without lomm05 P(pw25) %.3f' % (np.round(np.bincount(r1x, minlength=3) / N, 3), r1win.mean()))
SCEN = {'S0': dict(d=0.0), 'C+': dict(d=.5), 'C-': dict(d=-.5), 'SITE': dict(d=dS, rc=rcS, sf=sS, nsc=D.SITE_N)}
MIX = {'S0': .6, 'C+': .1, 'C-': .1, 'SITE': .2}
print('ESS %.0f N %d | sdpub (e-4) %s' % (F['ess'], N, {k: round(v * 1e4, 2) for k, v in F['sdpub'].items()}))
print('public G vs c4 (e-4) mean: ' + ' '.join('%s %+.2f' % (D.NAME[t], G[t].mean() * 1e4) for t in D.SH))

# curves (x in e-4)
CV = np.load(D.P2 + '/curves_base.npz'); g19 = CV['grid'] * 1e4
def c19(k, x): return np.interp(x, g19, CV[k], left=float(CV[k][0]), right=float(CV[k][-1]))
CB = json.load(open(os.path.join(os.path.dirname(HERE), 'work', 'curve_base.json')))
ms = np.array(sorted(float(k) for k in CB)); key = {float(k): k for k in CB}
def cshk(name, x):
    y = np.array([CB[key[m]][name] for m in ms])
    slope = (y[-1] - y[-2]) / (ms[-1] - ms[-2])
    out = np.interp(x, ms, y, left=0.0, right=float(y[-1]))
    hi = x > ms[-1]; out[hi] = np.minimum(1.0, y[-1] + slope * (x[hi] - ms[-1]))
    out[x < ms[0]] = 0.0
    return out
def cprob(x, mstar, tau):            # probit silver curve with the copy-block cliff at 0
    return np.where(x < 0, 0.0, norm.cdf((x - mstar) / tau))

RES = {}
ENTS = ['c4', 'R6', 'HW', 'eoa10', 'eoa20', 'pw20', 'pw25', 'pw30', 'R6pw20', 'R1', 'lomm05', 'R1L', 'R6pw30']
PAIRS = [('pw20', 'R6pw20'), ('pw30', 'R6pw20'), ('pw25', 'R6pw20'), ('R1', 'R6pw20'), ('eoa20', 'R6pw20'), ('R6', 'pw20'), ('R6', 'pw30'),
         ('pw20', 'pw30'), ('HW', 'pw20'), ('HW', 'pw30'), ('eoa10', 'pw30'), ('pw25', 'pw20'), ('R1L', 'R6pw20'), ('lomm05', 'R6pw20'), ('pw20', 'R6pw30'), ('pw25', 'R6pw30'), ('pw30', 'R6pw30'), ('pw20m0.5', 'R6pw20m0.5'), ('pw20m1', 'R6pw20m1'), ('pw20m1.5', 'R6pw20m1.5')]
TAUS = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]
for sn, sc in SCEN.items():
    pop = F['popf'](sh + sc['d'], rc4 + sc.get('rc', 0.0), s * sc.get('sf', 1.0)); nsc = sc.get('nsc', 1.0)
    X = {'c4': OUR['c4'], 'R6': OUR['R6u'], 'HW': OUR['HW']}
    for t in D.SH: X[D.NAME[t]] = OUR['c4'] + pop[t] + nsc * NP[t]
    r6g = kc * pop['PW20'] + epsx
    raw = OUR['R6u'] + r6g + nsc * (D.RHO * NP['PW20'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW20'])
    X['R6pw20'] = np.where(disp >= 958, raw, X['pw20'])
    X['R6pw30'] = OUR['R6u'] + kc * pop['PW30'] + epsx + nsc * (D.RHO * NP['PW30'] + np.sqrt(1 - D.RHO ** 2) * NV2['PW30'])   # built (27ce4be7), no display guard
    X['R1'] = np.where(r1win, X['pw25'], X['pw30'])
    X['lomm05'] = X['pw30'] + dL + nsc * nL
    X['R1L'] = np.where(r1x == 0, X['pw30'], np.where(r1x == 1, X['pw25'], X['lomm05']))
    X = {k: v * 1e4 for k, v in X.items()}
    for b_ in (0.5, 1.0, 1.5):   # break-even: pw20's true mean b_ e-4 below the model (R6pw20 shares it at k .86). MODEL
        X['pw20m%g' % b_] = X['pw20'] - b_; X['R6pw20m%g' % b_] = X['R6pw20'] - .86 * b_
    R = dict(single={}, pair={}, decomp={})
    def met(x):
        d = dict(E=x.mean(), sd=x.std(), p10=np.percentile(x, 10), Pneg=(x < 0).mean(), Plt5=(x < 5).mean(),
                 s19=c19('silver', x).mean(), r150_19=c19('r150', x).mean(), er19=c19('erank', x).mean(),
                 sS=cshk('P(r<=316)', x).mean(), r150S=cshk('P(r<=150)', x).mean(), r100S=cshk('P(r<=100)', x).mean(), brS=cshk('P(r<=633)', x).mean())
        for tau in TAUS: d['pt%g' % tau] = cprob(x, 6.0, tau).mean()
        return {k: float(v) for k, v in d.items()}
    for k in ENTS: R['single'][k] = met(X[k])
    for a, b in PAIRS: R['pair'][a + '+' + b] = met(np.maximum(X[a], X[b]))
    dp = (pop['PW30'] - pop['PW20']) * 1e4; dn = nsc * (NP['PW30'] - NP['PW20']) * 1e4
    R['decomp'] = dict(sd_pop30=float(pop['PW30'].std() * 1e4), sd_pop20=float(pop['PW20'].std() * 1e4), sd_popdiff=float(dp.std()),
                       sd_noise30=float((nsc * NP['PW30']).std() * 1e4), sd_noise20=float((nsc * NP['PW20']).std() * 1e4), sd_noisediff=float(dn.std()),
                       corr_pop=float(np.corrcoef(pop['PW30'], pop['PW20'])[0, 1]), sd_c4=float(X['c4'].std()),
                       pop30_p10=float(np.percentile(pop['PW30'] * 1e4, 10)), pop20_p10=float(np.percentile(pop['PW20'] * 1e4, 10)),
                       pop30_E=float(pop['PW30'].mean() * 1e4), pop20_E=float(pop['PW20'].mean() * 1e4),
                       corr_s_pop30=float(np.corrcoef(s, pop['PW30'])[0, 1]), corr_sh_popdiff=float(np.corrcoef(sh, dp)[0, 1]))
    RES[sn] = R
    print('== %s' % sn)
    for k in ENTS:
        m = R['single'][k]
        print('  %-7s E %+6.2f sd %5.2f p10 %+6.2f P<0 %.3f P<5 %.3f | 10-19: silver %.3f r150 %.3f E[rank] %5.0f | shakeup: silver %.3f r150 %.3f r100 %.3f bronze %.3f | probit tau 2/6/10 %.3f %.3f %.3f' % (
            k, m['E'], m['sd'], m['p10'], m['Pneg'], m['Plt5'], m['s19'], m['r150_19'], m['er19'], m['sS'], m['r150S'], m['r100S'], m['brS'], m['pt2'], m['pt6'], m['pt10']))
    for p, m in R['pair'].items():
        print('  PAIR %-13s E %+6.2f P<0 %.3f | 10-19: %.3f/%.3f/%4.0f | shakeup: silver %.3f r150 %.3f r100 %.3f bronze %.3f | tau .5/2/4/6/8/10 %s' % (
            p, m['E'], m['Pneg'], m['s19'], m['r150_19'], m['er19'], m['sS'], m['r150S'], m['r100S'], m['brS'], ' '.join('%.3f' % m['pt%g' % t] for t in TAUS)))
    print('  decomp', {k: round(v, 3) for k, v in R['decomp'].items()})
print('== MIX', MIX)
mixed = {}
for grp in ('single', 'pair'):
    for k in RES['S0'][grp]:
        mixed[k] = {q: sum(MIX[sn] * RES[sn][grp][k][q] for sn in MIX) for q in RES['S0'][grp][k]}
for k, m in mixed.items():
    print('  %-14s E %+6.2f P<0 %.3f | 10-19 silver %.3f r150 %.3f E[rank] %5.0f | shakeup silver %.3f r150 %.3f r100 %.3f bronze %.3f | tau .5/2/4/6/8/10 %s' % (
        k, m['E'], m['Pneg'], m['s19'], m['r150_19'], m['er19'], m['sS'], m['r150S'], m['r100S'], m['brS'], ' '.join('%.3f' % m['pt%g' % t] for t in TAUS)))
json.dump(dict(scen=RES, mix=mixed), open(os.path.join(HERE, 'skep_dose_%s.json' % D.TAG), 'w'), indent=1)
