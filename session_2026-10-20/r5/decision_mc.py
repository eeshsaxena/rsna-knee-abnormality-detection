# decision_mc.py (10-20 r5, decision lens; CPU only, research only): explicit Bayesian read -> final-pick model.
#
# Our side (MODEL, think/picks 10-19): posterior private margins over the heliosli copy block for c4, R6u (57248084), HW, a25, S6F
#   from picks_post.py (post_<OUR>_report.npz, 200k draws; conditioned on every public read up to 10-19).
# E-OA (MODEL): eoaX private = c4 private + dX + private paired noise. Prior on (d10, d20) = PRIOR flag (below).
# Read (MODEL): public paired delta pubX = dX + public paired noise (noise_corr.log, MEASURED sd / corr on report OOF, FPC-corrected).
#   R1 at ~13:00 IST: best-id -> eoa20 iff pub20 > 0 and pub20 > pub10; eoa10 iff pub10 > 0 and pub10 >= pub20; none iff both <= 0.
#   Partial: only one entry scored (other errored); unread: nothing scored (prior).
# Value (MODEL / GUESS field): S_k(x) = P(private rank <= k | max-of-pair margin x) from field_curves.py (curves_<FS>.npz).
# Output: per outcome, every pair from {c4, eoa10, eoa20, R6, a25, HW, S6F}: E[max margin], P(silver), P(bronze+), P(r<=200), P(r<=100).
import sys, json, itertools, numpy as np
R5 = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-20/r5'
PW = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/think/picks_work'
args = dict(a.split('=') for a in sys.argv[1:])
OUR = args.get('our', 'base'); FS = args.get('field', 'base'); PRIOR = args.get('prior', 'more'); NSC = float(args.get('noise', 1.0))
NCORR = float(args.get('ncorr', 0.96)); S6FD = float(args.get('s6fdefl', 1.0)); TAG = args.get('tag', f'{OUR}_{FS}_{PRIOR}_n{NSC}')
REP = int(args.get('rep', 4))                       # E-OA draws per our-side draw (variance reduction for rare outcomes)
rng = np.random.default_rng(int(args.get('seed', 20)))

Z = np.load(f'{PW}/post_{OUR}_report.npz'); ENT = list(Z['ent']); e = {k: i for i, k in enumerate(ENT)}
PV = Z['priv'].astype(float); TV = Z['true'].astype(float)
base = {k: PV[:, e[k]] for k in ('c4', 'R6u', 'HW', 'a25', 'S6F')}
if S6FD != 1.0:   # deflate S6F's idiosyncratic private noise vs R6u (R7 OOF single-fold stand-in overstates it, picks.md sec 4)
    nS = PV[:, e['S6F']] - TV[:, e['S6F']]; nR = PV[:, e['R6u']] - TV[:, e['R6u']]
    base['S6F'] = TV[:, e['S6F']] + nR + S6FD * (nS - nR)
M0 = PV.shape[0]

# ---- E-OA prior on the hidden macro deltas (d10, d20) ----
def prior_draw(n, PRIOR=PRIOR):
    if PRIOR == 'mix':         # Bayesian model average over the 5 priors (weights GUESS); the read re-weights them
        comp = rng.choice(len(MIXN), n, p=MIXW); out = np.zeros((n, 2))
        for i, nm in enumerate(MIXN):
            k = comp == i
            if k.any(): out[k] = prior_draw(int(k.sum()), nm)
        return out, comp
    if PRIOR == 'more':        # MORE_1020 skeptic summary: a.1 +3e-4 (P .75), a.2 +1e-4 (P .55); joint corr .95 (ladder .99, loosened)
        mu = np.array([3e-4, 1e-4]); sd = np.array([4.5e-4, 9e-4]); r = .95
    elif PRIOR == 'orig':      # skeptic_eoa.md 10-19 (pre-ladder, c4 gap map): +5e-4 sd 5e-4 / +7e-4 sd 1.1e-3
        mu = np.array([5e-4, 7e-4]); sd = np.array([5e-4, 1.1e-3]); r = .95
    elif PRIOR == 'null':      # E-OA does nothing on hidden (centred 0, small spread)
        mu = np.array([0., 0.]); sd = np.array([2e-4, 4e-4]); r = .95
    elif PRIOR == 'neg':       # pessimist: hidden coupling at tilt1.5 (ladder tilt1.5 cell means)
        mu = np.array([-1.5e-4, -8.4e-4]); sd = np.array([2.5e-4, 5e-4]); r = .95
    if PRIOR == 'ladder':      # explicit mixture over the MEASURED ladder cells (R1g74 .375, tilt1.0 .375, tilt1.5 .25) + structural spread
        C = json.load(open(R5 + '/ladder_cells.json')); C = [c for c in C if c['cpl'] != 'R0']
        W = np.array([{'R1g74': .375, 'tilt1.0': .375, 'tilt1.5': .25}[c['cpl']] for c in C]); W /= W.sum()
        j = rng.choice(len(C), n, p=W)
        d1 = np.array([c['d1'] for c in C])[j]; d2 = np.array([c['d2'] for c in C])[j]
        s1 = np.array([c['s1'] for c in C])[j]; s2 = np.array([c['s2'] for c in C])[j]
        # c4-vs-proxy transfer: shift by k_a * dgap, dgap ~ N(+.03, .04) (c4 couples more: gold58 gap .10 vs proxies .03-.06), plus
        # 1.5e-4 / 3e-4 structural spread (OAI -> RSNA hidden) shared through z
        dg = rng.normal(.03, .04, n); z = rng.standard_normal(n); u = rng.standard_normal(n)
        return np.stack([d1 + s1 * u + .0039 * dg + 1.5e-4 * z, d2 + s2 * u + .0079 * dg + 3e-4 * z], 1)
    elif PRIOR == 'custom':    # explicit N(m, s) pair from args (prior-width sweep)
        mu = np.array([float(args['m10']), float(args['m20'])]); sd = np.array([float(args['s10']), float(args['s20'])]); r = float(args.get('r', .95))
    if PRIOR not in ('more', 'orig', 'null', 'neg', 'custom'): raise SystemExit('bad prior ' + PRIOR)
    L = np.linalg.cholesky(np.array([[sd[0] ** 2, r * sd[0] * sd[1]], [r * sd[0] * sd[1], sd[1] ** 2]]))
    return mu + rng.standard_normal((n, 2)) @ L.T

MIXN = ['more', 'ladder', 'orig', 'null', 'neg']; MIXW = np.array([.40, .25, .15, .10, .10])

def noise_draw(n, s1, s2, c):
    L = np.linalg.cholesky(np.array([[s1 ** 2, c * s1 * s2], [c * s1 * s2, s2 ** 2]]))
    return rng.standard_normal((n, 2)) @ L.T

N = M0 * REP
ib = np.tile(np.arange(M0), REP)                                          # our-side row for each E-OA draw
if PRIOR == 'mix': D, COMP = prior_draw(N)
else: D = prior_draw(N); COMP = None
NP_ = noise_draw(N, 2.1e-4 * NSC, 4.6e-4 * NSC, NCORR)                   # public 390 (MEASURED .00020 / .00044 corr .96, FPC .95)
NV = noise_draw(N, 1.46e-4 * NSC, 3.15e-4 * NSC, min(.98, NCORR + .02))   # private 910 (MEASURED .00013 / .00028 corr .98, FPC .89)
pub10, pub20 = D[:, 0] + NP_[:, 0], D[:, 1] + NP_[:, 1]
X = {k: v[ib] for k, v in base.items()}
X['eoa10'] = X['c4'] + D[:, 0] + NV[:, 0]; X['eoa20'] = X['c4'] + D[:, 1] + NV[:, 1]
NAME = {'c4': 'c4', 'eoa10': 'eoa10', 'eoa20': 'eoa20', 'R6u': 'R6', 'a25': 'a25', 'HW': 'HW', 'S6F': 'S6F'}
if args.get('extra', '0') == '1':   # HYPOTHETICAL, NOT BUILT: R6 57248084 + E-OA a .1 (R6 gold58 gap .09 vs c4 .10 -> 0.9 x d10 + 1e-4 structural;
    # private E-OA noise corr .6 with eoa10's)
    X['R6e10'] = X['R6u'] + 0.9 * D[:, 0] + 1e-4 * rng.standard_normal(N) + 0.6 * NV[:, 0] + 0.8 * 1.46e-4 * NSC * rng.standard_normal(N)
    NAME['R6e10'] = 'R6e10'
CANDS = list(NAME)

CV = np.load(f'{R5}/curves_{FS}.npz'); grid = CV['grid']
def Sk(k, x): return np.interp(x, grid, CV[k], left=0.0, right=float(CV[k][-1]))

OUT = {
    'O20 eoa20 best-id': (pub20 > 0) & (pub20 > pub10),
    'O10 eoa10 best-id': (pub10 > 0) & (pub10 >= pub20),
    'O0 no move': (pub10 <= 0) & (pub20 <= 0),
    'P10+ only eoa10 scored, moved': pub10 > 0,
    'P10- only eoa10 scored, no move': pub10 <= 0,
    'P20+ only eoa20 scored, moved': pub20 > 0,
    'P20- only eoa20 scored, no move': pub20 <= 0,
    'U unread (prior)': np.ones(N, bool),
}
res = {'tag': TAG, 'args': args, 'N': int(N), 'outcomes': {}}
pairs = list(itertools.combinations(CANDS, 2))
print(f'=== {TAG} | our {OUR} field {FS} prior {PRIOR} noise x{NSC} corr {NCORR} s6fdefl {S6FD} | N {N}')
print('prior: d10 %+.2fe-4 sd %.2f P>0 %.2f | d20 %+.2fe-4 sd %.2f P>0 %.2f | P(d20>d10) %.2f' % (
    D[:, 0].mean() * 1e4, D[:, 0].std() * 1e4, (D[:, 0] > 0).mean(), D[:, 1].mean() * 1e4, D[:, 1].std() * 1e4, (D[:, 1] > 0).mean(), (D[:, 1] > D[:, 0]).mean()))
for oname, m in OUT.items():
    p = m.mean(); w = m
    post = dict(P=float(p), d10=[float(D[w, 0].mean()), float(D[w, 0].std()), float((D[w, 0] > 0).mean())],
                d20=[float(D[w, 1].mean()), float(D[w, 1].std()), float((D[w, 1] > 0).mean())], P_d20_gt_d10=float((D[w, 1] > D[w, 0]).mean()))
    rows = {}
    for a, b in pairs:
        xm = np.maximum(X[a][w], X[b][w])
        rows[f'{NAME[a]}+{NAME[b]}'] = dict(E=float(xm.mean()), silver=float(Sk('silver', xm).mean()), bronze=float(Sk('bronze', xm).mean()),
                                           r200=float(Sk('r200', xm).mean()), r100=float(Sk('r100', xm).mean()), gold=float(Sk('gold', xm).mean()))
    for a in CANDS:
        xm = X[a][w]
        rows[f'{NAME[a]} alone'] = dict(E=float(xm.mean()), silver=float(Sk('silver', xm).mean()), bronze=float(Sk('bronze', xm).mean()),
                                       r200=float(Sk('r200', xm).mean()), r100=float(Sk('r100', xm).mean()), gold=float(Sk('gold', xm).mean()))
    if COMP is not None: post['prior_weights'] = {nm: float((COMP[w] == i).mean()) for i, nm in enumerate(MIXN)}
    post['pairs'] = rows; res['outcomes'][oname] = post
    if COMP is not None: print('   posterior prior weights:', {k: round(v, 3) for k, v in post['prior_weights'].items()})
    print(f'-- {oname}: P {p:.3f} | post d10 {post["d10"][0]*1e4:+.2f}e-4 (P>0 {post["d10"][2]:.2f}) d20 {post["d20"][0]*1e4:+.2f}e-4 '
          f'(P>0 {post["d20"][2]:.2f}) P(d20>d10) {post["P_d20_gt_d10"]:.2f}')
    top = sorted([(k, v) for k, v in rows.items() if '+' in k], key=lambda kv: -kv[1]['silver'])
    show = top[:6] + [kv for kv in top if kv[0] in ('c4+R6', 'eoa10+R6', 'eoa20+R6', 'c4+eoa10', 'c4+eoa20', 'eoa10+eoa20') and kv not in top[:6]]
    for k, v in show:
        print(f'   {k:12s} E {v["E"]*1e4:+.2f}e-4 silver {v["silver"]:.3f} bronze+ {v["bronze"]:.3f} r200 {v["r200"]:.3f} r100 {v["r100"]:.4f}')
json.dump(res, open(f'{R5}/dec_{TAG}.json', 'w'), indent=1)

# ---- magnitude of a move (MODEL): teams passed x ~1.2e-5 (in-band strip, density.md) -> observed public delta with error
# sd sqrt((3e-5)^2 + (0.3 delta)^2) (+-2-3 teams of churn, ~30 % density error). c4 is ~1.1e-4 short of .959 (picks.md / density.md).
if args.get('mag', '0') == '1':
    print('\n== magnitude-conditioned moves (observed public delta bins, x1e-4)')
    BINS = [(0, .5e-4), (.5e-4, 1.1e-4), (1.1e-4, 3e-4), (3e-4, 1)]
    res['mag'] = {}
    for oname, best, other in (('O20 eoa20 best-id', 'eoa20', 'eoa10'), ('O10 eoa10 best-id', 'eoa10', 'eoa20')):
        pb = pub20 if best == 'eoa20' else pub10
        mobs = pb + rng.standard_normal(N) * np.sqrt((3e-5) ** 2 + (0.3 * pb) ** 2)
        for lo, hi in BINS:
            w = OUT[oname] & (mobs > lo) & (mobs <= hi)
            if w.sum() < 2000: continue
            row = {}
            for a, b in ((best, 'R6u'), (other, 'R6u'), ('c4', 'R6u'), (best, 'c4'), (best, 'S6F'), (best, 'HW')):
                xm = np.maximum(X[a][w], X[b][w]); row[f'{NAME[a]}+{NAME[b]}'] = float(Sk('silver', xm).mean())
            key = f'{oname} | obs {lo*1e4:.1f}-{min(hi, 9.9e-4)*1e4:.1f}e-4'
            res['mag'][key] = dict(P=float(w.mean() / OUT[oname].mean()), d10=float(D[w, 0].mean()), d20=float(D[w, 1].mean()),
                                   Pd10=float((D[w, 0] > 0).mean()), Pd20=float((D[w, 1] > 0).mean()), silver=row)
            print(f'  {key}: share {w.mean()/OUT[oname].mean():.2f} | d10 {D[w,0].mean()*1e4:+.2f} (P>0 {(D[w,0]>0).mean():.2f}) '
                  f'd20 {D[w,1].mean()*1e4:+.2f} (P>0 {(D[w,1]>0).mean():.2f}) | silver ' + ' '.join(f'{k} {v:.3f}' for k, v in row.items()))
    json.dump(res, open(f'{R5}/dec_{TAG}.json', 'w'), indent=1)
