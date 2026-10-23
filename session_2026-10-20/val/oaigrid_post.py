# oaigrid_post.py (10-20 val TRACK A, Claude). Local post-processing of the box AGGREGATES in box_oaigrid_{PFT,PAREA}.log.
# No OAI row / id / per-knee value is involved here: the inputs are means, sds, covariances and likelihood sums over resamples.
# Usage: python oaigrid_post.py > oaigrid_post.log
import json, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A_REQ = [.1, .2, .25, .3, .35, .4, .45, .5, .6]
SHAPE = [.3, .35, .4, .45]


def load(fn):
    t = open(os.path.join(HERE, fn), encoding='utf-8').read().splitlines()
    i = t.index('JSON_BEGIN')
    return json.loads(t[i + 1])


def r3p_shift(rn):
    return float(rn[3:]) if rn.startswith('R3p') else None


def main():
    out = {}
    for P, fn in (('P_FT', 'box_oaigrid_PFT.log'), ('P_AREA', 'box_oaigrid_PAREA.log')):
        E = load(fn); FN = E['FN']; AG = np.array(E['AG']); NF, NA = len(FN), len(AG); R = E['reg']
        AG0 = np.concatenate([[0.], AG])
        ia = lambda a: int(np.argmin(np.abs(AG - a)))
        k = lambda f, a: FN.index(f) * NA + ia(a)
        print('=' * 100); print('PROXY', P, '| n', E['n'], '| regimes', list(R)); print('=' * 100)
        # ---------- 1. regime table + population curves ----------
        print('\n-- 1. population curves (exact weighted, macro e-4 = 3 OA labels / 12) and coupling per regime')
        for rn, r in R.items():
            pop = np.array(r['pop']) * 1e4
            am = [float(AG[np.argmax(pop[f])]) if pop[f].max() > 0 else 0. for f in range(NF)]
            print('%-9s ESS %4.0f phi %.3f [ML %.2f MPF %.2f LPF %.2f] rho %.3f q %.3f gap %+.3f | P(order) %.3f L_joint %.2e | argmax pop %s' % (
                rn, r['ess'], r['phi'], *r['phi_pairs'], r['rho'], r['q'], r['gap'], r['pub']['p_order_all'], r['pub']['L_joint'],
                dict(zip(FN, am))))
            for f in range(NF):
                print('      %-5s %s' % (FN[f], ' '.join('%.2f:%+6.2f' % (a, pop[f, ia(a)]) for a in A_REQ)))
        # ---------- 2. regime posterior on the R3p continuum ----------
        rs = sorted([rn for rn in R if rn.startswith('R3p')], key=r3p_shift); sh = np.array([r3p_shift(x) for x in rs])
        mids = np.concatenate([[sh[0] - (sh[1] - sh[0]) / 2], (sh[1:] + sh[:-1]) / 2, [sh[-1] + (sh[-1] - sh[-2]) / 2]]); wid = np.diff(mids)
        LJ = np.array([R[x]['pub']['L_joint'] for x in rs]); LO = np.array([R[x]['pub']['p_order_all'] for x in rs])
        LM = np.array([R[x]['pub']['L_mag'] for x in rs]); LMO = np.array([R[x]['pub']['L_mag_order'] for x in rs])
        print('\n-- 2. R3p regime likelihoods (public-size n 390 resamples; ABC kernel h 4e-5 on LB posterior (G10, G20); s log-uniform [.5, 2])')
        print('   shift   phi    rho    q      gap   | P(order all) each[S10>0, S20>S10, PW30>S20, PW30>PW20, PW30>S30] | L_mag    L_mag&order  L_joint(+pw30 not .960)')
        for x, s in zip(rs, sh):
            r = R[x]; pb = r['pub']
            print('   %+5.2f  %.3f  %.3f  %.3f  %+.3f | %.3f %s | %.2e  %.2e  %.2e' % (s, r['phi'], r['rho'], r['q'], r['gap'], pb['p_order_all'],
                                                                               np.round(pb['p_order_each'], 3).tolist(), pb['L_mag'], pb['L_mag_order'], pb['L_joint']))
        for x in ('raw', 'R1g74', 'MIX0.25', 'MIX0.50', 'MIX0.75'):
            if x in R:
                pb = R[x]['pub']
                print('   %-7s phi %.3f gap %+.3f | P(order) %.3f %s | L_mag %.2e L_joint %.2e' % (x, R[x]['phi'], R[x]['gap'], pb['p_order_all'],
                                                                                           np.round(pb['p_order_each'], 3).tolist(), pb['L_mag'], pb['L_joint']))
        post = {}
        for pn, prior in (('N01', np.exp(-.5 * sh ** 2) * wid), ('flat', wid.copy())):
            for ln, L in (('joint', LJ), ('order', LO), ('mag&order', LMO)):
                p = prior * L; p = p / p.sum(); post[(pn, ln)] = p
                ms = (p * sh).sum(); sds = np.sqrt((p * (sh - ms) ** 2).sum())
                ph = sum(p[i] * R[x]['phi'] for i, x in enumerate(rs)); gp = sum(p[i] * R[x]['gap'] for i, x in enumerate(rs))
                qq = sum(p[i] * R[x]['q'] for i, x in enumerate(rs)); rh = sum(p[i] * R[x]['rho'] for i, x in enumerate(rs))
                print('   posterior [%s prior, %s lik]: shift %+.2f (sd %.2f) MAP %+.2f | phi %.3f rho %.3f q %.3f gap %+.3f | mass %s' % (
                    pn, ln, ms, sds, sh[np.argmax(p)], ph, rh, qq, gp, ' '.join('%+.2f:%.2f' % (s, v) for s, v in zip(sh, p) if v >= .01)))
        # population ordering feasibility (means, no noise)
        feas = []
        for x, s in zip(rs, sh):
            m = np.array(R[x]['pop'])
            S10, S20, S30 = m[0, ia(.1)], m[0, ia(.2)], m[0, ia(.3)]; PW20, PW30 = m[1, ia(.2)], m[1, ia(.3)]
            ok = (S10 > 0) and (S20 > S10) and (PW30 > S20) and (PW30 > PW20) and (PW30 > S30)
            feas.append((s, ok))
        print('   population-mean ordering holds at shifts:', [s for s, ok in feas if ok])
        # ---------- 3. fitted regime: posterior-mixed curves, optimum a, shape near .3-.45 ----------
        pj = post[('N01', 'joint')]; pjf = post[('flat', 'joint')]
        imap = int(np.argmax(pj)); rmap = rs[imap]
        print('\n-- 3. FITTED REGIME (N01 prior x joint likelihood): MAP %s; posterior-mixed curves' % rmap)
        for pn, pp in (('N01', pj), ('flat', pjf)):
            mix = sum(pp[i] * np.array(R[x]['pop']) for i, x in enumerate(rs)) * 1e4
            smean = sum(pp[i] * float(np.dot(R[x]['pub']['s_post'], np.exp(np.linspace(np.log(.5), np.log(2.), 15)))) for i, x in enumerate(rs))
            print('   [%s] posterior-mixed population curve (OAI units, macro e-4); amplitude s posterior mean %.2f' % (pn, smean))
            for f in range(NF):
                print('      %-5s %s | argmax %.3f' % (FN[f], ' '.join('%.2f:%+6.2f' % (a, mix[f, ia(a)]) for a in A_REQ),
                                                     AG[np.argmax(mix[f])] if mix[f].max() > 0 else 0.))
            # optimum distribution: regime posterior x knee-bootstrap argmax
            for f in range(NF):
                cnt = sum(pp[i] * np.array(R[x]['boot']['knee']['argmax_counts'][f], float) / R[x]['boot']['knee']['B'] for i, x in enumerate(rs))
                cdf = np.cumsum(cnt) / cnt.sum()
                q = [float(AG0[min(len(AG0) - 1, np.searchsorted(cdf, v))]) for v in (.1, .5, .9)]
                print('      optimum a %-5s (regime posterior x knee bootstrap): p10/50/90 %s | P(a* >= .35) %.2f P(a* >= .4) %.2f' % (
                    FN[f], q, cnt[AG0 >= .35 - 1e-9].sum() / cnt.sum(), cnt[AG0 >= .4 - 1e-9].sum() / cnt.sum()))
        # shape near .3-.45 at MAP and at the two neighbours, OAI-size knee bootstrap and public-size sd
        for x in sorted(set([rs[max(0, imap - 1)], rmap, rs[min(len(rs) - 1, imap + 1)]]), key=r3p_shift) + ['R1g74', 'raw']:
            r = R[x]; pop = np.array(r['pop']); C = np.array(r['boot']['knee']['cov']) * 1e-8; CP = np.array(r['pub']['cov']) * 1e-8
            print('   shape %s (ESS %.0f, phi %.3f, gap %+.3f): value e-4 at %s, then paired increments [OAI-size sd | public n390 sd]' % (
                x, r['ess'], r['phi'], r['gap'], SHAPE))
            for f in FN:
                vals = [pop[FN.index(f), ia(a)] * 1e4 for a in SHAPE]
                incs = []
                for a1, a2 in zip(SHAPE[:-1], SHAPE[1:]):
                    i1, i2 = k(f, a1), k(f, a2); d = (pop[FN.index(f), ia(a2)] - pop[FN.index(f), ia(a1)]) * 1e4
                    sd = np.sqrt(C[i1, i1] + C[i2, i2] - 2 * C[i1, i2]) * 1e4; sp = np.sqrt(CP[i1, i1] + CP[i2, i2] - 2 * CP[i1, i2]) * 1e4
                    incs.append('%.2f->%.2f %+.2f [%.2f | %.2f]' % (a1, a2, d, sd, sp))
                ref = k('pw', .3)
                vs = []
                for a in SHAPE:
                    i1 = k(f, a); d = (pop[FN.index(f), ia(a)] - pop[1, ia(.3)]) * 1e4
                    sd = np.sqrt(max(C[i1, i1] + C[ref, ref] - 2 * C[i1, ref], 0)) * 1e4
                    vs.append('%+.2f(%.2f)' % (d, sd))
                print('      %-5s %s | %s | vs pw30 %s' % (f, ' '.join('%+6.2f' % v for v in vals), '; '.join(incs), ' '.join(vs)))
            # per-label shape for sym and pw (label AUC delta, not macro), with paired sd of the step
            LC = np.array(r['boot']['knee']['lab_cov']) * 1e-8; pl = np.array(r['pop_lab'])
            for f in ('sym', 'pw', 'LP2'):
                fi = FN.index(f); s_ = []
                for t, tn in enumerate(['M', 'L', 'PF']):
                    vals = [pl[t, fi, ia(a)] * 1e4 for a in SHAPE]
                    st = []
                    for a1, a2 in zip(SHAPE[:-1], SHAPE[1:]):
                        i1 = t * NF * NA + fi * NA + ia(a1); i2 = t * NF * NA + fi * NA + ia(a2)
                        sd = np.sqrt(max(LC[i1, i1] + LC[i2, i2] - 2 * LC[i1, i2], 0)) * 1e4
                        st.append('%+.1f(%.1f)' % ((pl[t, fi, ia(a2)] - pl[t, fi, ia(a1)]) * 1e4, sd))
                    s_.append('%s %s steps %s' % (tn, '/'.join('%+.1f' % v for v in vals), ' '.join(st)))
                print('      per-label AUC e-4 %-4s | %s' % (f, ' | '.join(s_)))
        # derived: Medial-untouched versions of sym / pw (L + PF label deltas only)
        print('   derived (Medial output untouched, L and PF as in sym / pw), macro e-4, MAP regime %s:' % rmap)
        pl = np.array(R[rmap]['pop_lab'])
        for f in ('sym', 'pw'):
            fi = FN.index(f); v = (pl[1, fi] + pl[2, fi]) / 12 * 1e4
            print('      %s-noM %s | argmax %.3f' % (f, ' '.join('%.2f:%+6.2f' % (a, v[ia(a)]) for a in A_REQ), AG[np.argmax(v)]))
        # ---------- 4. posterior predictive public increments over pw30 ----------
        print('\n-- 4. posterior predictive PUBLIC increment over pw30 (e-4; regime posterior N01 x joint; within-regime ABC on (G10, G20), order, pw30 not .960)')
        for pn, pp in (('N01', pj), ('flat', pjf)):
            print('   [%s prior]' % pn)
            Wr = np.array([pp[i] for i in range(len(rs))])
            for f, a in (('sym', .25), ('sym', .35), ('pw', .2), ('pw', .25), ('pw', .35), ('pw', .4), ('pw', .45), ('pw', .5),
                         ('LP2', .2), ('LP2', .25), ('LP2', .3), ('LP2', .35), ('LP2', .4), ('LPpw', .3), ('LPpw', .4), ('LPpw', .5)):
                fi = FN.index(f); ai = ia(a)
                m = sum(Wr[i] * R[x]['pub']['pp_mean'][fi][ai] for i, x in enumerate(rs))
                sq = sum(Wr[i] * (R[x]['pub']['pp_sd'][fi][ai] ** 2 + R[x]['pub']['pp_mean'][fi][ai] ** 2) for i, x in enumerate(rs))
                pos = sum(Wr[i] * R[x]['pub']['pp_pos'][fi][ai] for i, x in enumerate(rs))
                p9 = sum(Wr[i] * R[x]['pub']['p960'][fi][ai] for i, x in enumerate(rs))
                popinc = sum(Wr[i] * (R[x]['pop'][fi][ai] - R[x]['pop'][1][ia(.3)]) for i, x in enumerate(rs))
                print('      %-4s %.2f: mean %+.2f sd %.2f | P(> pw30) %.2f | P(shows .960) %.2f | population (OAI units, no s) %+.2f' % (
                    f, a, m * 1e4, np.sqrt(max(sq - m * m, 0)) * 1e4, pos, p9, popinc * 1e4))
        out[P] = dict(map=rmap, post_N01=dict(zip(rs, np.round(pj, 4).tolist())), post_flat=dict(zip(rs, np.round(pjf, 4).tolist())))
    json.dump(out, open(os.path.join(HERE, 'oaigrid_post.json'), 'w'), indent=1)


main()
