#!/usr/bin/env python
# sim_s6f2_runtime.py (10-15, Claude subagent): hidden-run wall + canary probabilities of the two s6f2 kernels. MODEL on MEASURED
# inputs. No em dashes. Laptop CPU, numpy only (~1 min at the default reps).
#   (1) FLRM-AO + S6fall  = V224-FLIP-RM (r224 mirror TTA, +0.06-0.16 h MODEL) + the 3 S6F cells (start guard G, budget 3.3 h,
#       hard deadline NB_T0 + 8.2 h): which G?
#   (2) FLIP-AO + S6fall + S = V224-FLIP + the S fleet stage of kernel_v224flips20 (start guard 7.0 h, wall 8.3 h + 300 s) + the 3
#       S6F cells: which ORDER (S first or S6F first) and which S6F G?
# Model pieces, each copied (not imported; both files run at import) from the reviewed sims:
#   - base V224-FLIP (R7 flip forced on every study) calibrated to its MEASURED 4.57-4.65 h submit-to-score, her R7 deadline 8.3 h and
#     r224 skip at 7.83 h, RM +U(.06, .16) h x s: v224f/sim_ffs_runtime.py section D (r7_flip_ao, run_ao);
#   - S fleet: 1.5 min fixed + 650 studies x 1.62 s x mf x s (MEASURED microbench, 5 x TTR224 <= 325 ms / member / study / T4; the
#     task's 0.31 s / study / GPU / member gives 1.55 s), pre-cells .01 h, HF .005 h, guard 7.0 h, wall 8.3 h + 300 s grace: same file;
#   - S6fall: c ~ U(6.5, 13) s / study / T4 (MODEL range), setup U(.03, .08) h x s, two parts (split_parts order), compact_infer takes no
#     new study after hard - 24 min, the RUN cell kills at the hard deadline, blend .01 h: s6f/sim_s6f_runtime.py (run, s6_part).
# PAIRED design (new here; the source sims draw each config with its own random numbers): every session draws ONE set of
# (s, q, r224, mf, rm, c, setup, per-study arrays) and every config is evaluated on that same set, so config differences are not
# sampling noise (cf. v224f/guard_paired_1015.py). Monotonicity check printed: a higher G can only turn a skip into a run.
#   usage: py -3.12 -I sim_s6f2_runtime.py [--reps 20000] [--seed 0]
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=20000)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()

PRE, FLIP_TOTAL, N_HID = 0.42, 4.62, 1300
R7_DEADLINE, R224_SKIP_AT, LIMIT9 = 8.3, 7.83, 9.0
RM_LO, RM_HI = 0.06, 0.16
FIX_FL, PRECELLS, S_S, WALL, GRACE, HF_H, S_GUARD = 1.5 / 60, 0.01, 1.62, 8.3, 300 / 3600, 0.005, 7.0
C_LO, C_HI, SETUP_LO, SETUP_HI = 6.5, 13.0, 0.03, 0.08
BUDGET, DEADLINE, STOP_MARGIN_H, MIN_RUN_H, CI_MARGIN_H, BLEND_H = 3.3, 8.2, 240 / 3600, 1200 / 3600, (20 * 60 + 4 * 60) / 3600, 0.01
KILL_WAIT_H = 125 / 3600          # RUN cell: p.wait(timeout=120) after the SIGKILL + one 5 s poll


def draw_session(rng, model, n, c_rng):
    if model == 'B':
        s = float(np.exp(rng.normal(0, 0.08)) * (1.3 if rng.random() < 0.05 else 1.0))
    else:                                                          # STRESS (GUESS): sd .12, 9 % x1.3, 3 % x1.5
        s = float(np.exp(rng.normal(0, 0.12)) * rng.choice([1.0, 1.3, 1.5], p=[0.88, 0.09, 0.03]))
    d = dict(s=s, q=rng.uniform(0.0, 0.3), r224=rng.uniform(0.15, 0.30), mf=rng.uniform(1.0, 1.15), rm=rng.uniform(RM_LO, RM_HI),
             c=rng.uniform(*c_rng), setup=rng.uniform(SETUP_LO, SETUP_HI), n=n)
    flip_h = FLIP_TOTAL - d['q'] - PRE - d['r224']
    r7 = []
    for _ in range(2):                                             # r7_flip_ao
        files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n // 2)).astype(int)
        noise = rng.normal(1.0, 0.15, n // 2).clip(0.6, 1.6)
        fh = flip_h * s * n / 1300
        rf = (fh * 3600 - 250) / files.sum()
        r7.append((PRE * s * 3600 + 250 + (rf * files * noise).sum()) / 3600)
    d['r7_raw'] = max(r7)
    parts = []
    for _ in range(2):                                             # s6_part: cumulative per-study ends from the part start
        files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n // 2))
        order = np.argsort(-files, kind='mergesort')
        order = np.concatenate([order[::2], order[1::2]])
        f = files[order]
        noise = rng.normal(1.0, 0.12, n // 2).clip(0.6, 1.6)
        parts.append(np.cumsum(d['c'] * s * f / f.mean() * noise) / 3600)
    d['parts'] = parts
    return d


def base_end(d, rm):
    r7_end = min(d['r7_raw'], R7_DEADLINE + 0.02)
    if r7_end > R224_SKIP_AT:
        return r7_end + 0.01, 'R224_SKIP'                           # her stage 2b skips r224 -> V224 canary Fracture
    b = min(r7_end + d['r224'] * d['s'] + (d['rm'] * d['s'] if rm else 0.0), LIMIT9 - 40 / 60 + 0.01)
    return b, ('R224_TIMEOUT' if b >= LIMIT9 - 40 / 60 else 'ok')


def s_fleet(d, start):
    """S stage of kernel_v224flips20 (SAVE .. HF). Returns (end, outcome)."""
    fs = start + PRECELLS
    if fs > S_GUARD:
        return fs + HF_H, 'SKIPPED'
    fe = fs + FIX_FL + (d['n'] / 2) * S_S * d['mf'] * d['s'] / 3600
    if fe > WALL:
        return WALL + GRACE + HF_H, 'HARDSTOP'
    return fe + HF_H, 'ok'


def s6f(d, start, G):
    """the 3 S6F cells. Returns (end, outcome in S6F / SKIP / NOTIME / CUT / KILLED, coverage)."""
    start = start + 0.002
    if start > G:
        return start + BLEND_H, 'SKIP', 0.0
    hard = min(start + BUDGET, DEADLINE)
    stop_at = hard - STOP_MARGIN_H
    if stop_at - start < MIN_RUN_H:
        return start + BLEND_H, 'NOTIME', 0.0
    ci_deadline = stop_at - 20 / 60
    t0 = start + d['setup'] * d['s']
    done, ends = 0, []
    for cum_rel in d['parts']:
        cum = t0 + cum_rel
        k = int(np.searchsorted(cum, ci_deadline, side='right'))
        k_done = min(len(cum), k + (1 if k < len(cum) else 0))   # the study in flight is finished
        done += k_done
        ends.append(cum[k_done - 1] if k_done else t0)
    end = max(ends)
    if end > hard:
        return hard + KILL_WAIT_H + BLEND_H, 'KILLED', 0.0
    cov = done / d['n']
    return end + BLEND_H, ('S6F' if cov >= 0.98 else 'CUT'), cov


def kernel(d, cfg):
    """one hidden run of a config; returns dict(end, base, s6=outcome, sf=outcome or None, bflag)."""
    b, bflag = base_end(d, cfg['rm'])
    sf = None
    if cfg['order'] == 'S6F':                                      # (1) FLRM / FLIP + S6F
        end, s6, cov = s6f(d, b, cfg['G'])
    elif cfg['order'] == 'S_then_S6F':                             # (2) chosen: flips20 cells, then S6F
        t, sf = s_fleet(d, b)
        end, s6, cov = s6f(d, t, cfg['G'])
    elif cfg['order'] == 'S6F_then_S':                             # (2) alternative: S6F cells, then the S fleet
        t, s6, cov = s6f(d, b, cfg['G'])
        end, sf = s_fleet(d, t)
    elif cfg['order'] == 'S':                                      # flips20 alone (reference)
        end, sf = s_fleet(d, b)
        s6 = None
    else:
        end, s6 = b, None
    return dict(end=end, base=b, s6=s6, sf=sf, bflag=bflag)


CFG1 = {'FLIP + S6F G 5.75 (= the submitted s6f build, reference)': dict(rm=False, order='S6F', G=5.75)}
for G in (5.75, 5.85, 6.0, 6.25, 6.5, 99.0):
    CFG1['FLRM + S6F G %s' % ('%.2f' % G if G < 50 else 'none')] = dict(rm=True, order='S6F', G=G)
CFG2 = {'V224-FLIP alone': dict(rm=False, order='none', G=None), 'FLIP-AO + S (flips20, reference)': dict(rm=False, order='S', G=None)}
for G in (5.75, 6.0, 6.1, 6.25, 6.5, 99.0):
    CFG2['S then S6F, S6F G %s' % ('%.2f' % G if G < 50 else 'none')] = dict(rm=False, order='S_then_S6F', G=G)
for G in (5.75, 6.0):
    CFG2['S6F then S, S6F G %.2f (alternative order)' % G] = dict(rm=False, order='S6F_then_S', G=G)


def summarise(rows, name):
    E = np.array([r['end'] for r in rows])
    s6 = [r['s6'] for r in rows]
    sf = [r['sf'] for r in rows]
    n = len(rows)
    p = lambda xs, v: sum(1 for x in xs if x == v) / n
    out = dict(name=name, median=np.median(E), p90=np.percentile(E, 90), p99=np.percentile(E, 99), max=E.max(),
               gt80=float(np.mean(E > 8.0)), gt83=float(np.mean(E > 8.3)), gt9=float(np.mean(E > 9.0)))
    if any(x is not None for x in s6):
        out.update(s6_used=p(s6, 'S6F'), s6_skip=p(s6, 'SKIP') + p(s6, 'NOTIME'), s6_cut=p(s6, 'CUT') + p(s6, 'KILLED'),
                   s6_killed=p(s6, 'KILLED'))
        out['acl_canary'] = 1.0 - out['s6_used']
    if any(x is not None for x in sf):
        out.update(sf_used=p(sf, 'ok'), sf_skip=p(sf, 'SKIPPED'), sf_hard=p(sf, 'HARDSTOP'))
        out['syn_canary'] = 1.0 - out['sf_used']
    if out.get('s6_used') is not None and out.get('sf_used') is not None:
        out['both'] = sum(1 for a, b in zip(s6, sf) if a == 'S6F' and b == 'ok') / n
    out['frac_canary'] = sum(1 for r in rows if r['bflag'] == 'R224_SKIP') / n
    return out


def fmt(o):
    s = f"  {o['name']}: end median {o['median']:.2f} p90 {o['p90']:.2f} p99 {o['p99']:.2f} max {o['max']:.2f} | P(>8.0) {o['gt80']:.4f} P(>8.3) {o['gt83']:.4f} P(>9) {o['gt9']:.4f}"
    if 's6_used' in o:
        s += f" | S6F used {o['s6_used']:.4f} skip {o['s6_skip']:.4f} cut {o['s6_cut']:.4f} (killed {o['s6_killed']:.4f}) -> P(ACL canary) {o['acl_canary']:.4f}"
    if 'sf_used' in o:
        s += f" | S used {o['sf_used']:.4f} skip {o['sf_skip']:.4f} hard {o['sf_hard']:.4f} -> P(Synovitis canary) {o['syn_canary']:.4f}"
    if 'both' in o:
        s += f" | P(both used) {o['both']:.4f}"
    s += f" | P(Fracture canary, base) {o['frac_canary']:.4f}"
    return s


def block(title, cfgs, model, reps, n=N_HID, c_rng=(C_LO, C_HI), seed_off=0):
    rng = np.random.default_rng(A.seed + seed_off)
    rows = {k: [] for k in cfgs}
    for _ in range(reps):
        d = draw_session(rng, model, n, c_rng)
        for k, cfg in cfgs.items():
            rows[k].append(kernel(d, cfg))
    print(f'\n# {title} (session model {model}, N {n}, c {c_rng}, {reps} paired sessions)')
    res = {}
    for k in cfgs:
        res[k] = summarise(rows[k], k)
        print(fmt(res[k]), flush=True)
    return rows, res


def mono(rows, keys, label):
    """paired check: going up the G list never turns an S6F-used session into a canary."""
    bad = 0
    for a, b in zip(keys, keys[1:]):
        bad += sum(1 for x, y in zip(rows[a], rows[b]) if x['s6'] == 'S6F' and y['s6'] != 'S6F')
    print(f'  paired monotonicity ({label}): sessions used at a lower G but a canary at a higher G: {bad}', flush=True)


print('# sim_s6f2_runtime.py MODEL; kernel h = age from NB_T0; submit-to-score = end + queue q (0-.3 h)')
for model in ('B', 'STRESS'):
    rows, res = block('(1) FLRM-AO + S6fall: S6F start guard G', CFG1, model, A.reps)
    mono(rows, [k for k in CFG1 if k.startswith('FLRM')], 'FLRM')
    rows, res = block('(2) FLIP-AO + S6fall + S: order and S6F start guard G', CFG2, model, A.reps, seed_off=1)
    mono(rows, [k for k in CFG2 if k.startswith('S then S6F')], 'S then S6F')
for n, c_rng in ((1500, (C_LO, C_HI)), (N_HID, (10.0, 13.0)), (N_HID, (13.0, 16.0)), (N_HID, (6.5, 9.0))):
    sub1 = {k: v for k, v in CFG1.items() if k.startswith('FLRM + S6F G 5.75') or k.startswith('FLRM + S6F G 6.00') or k.startswith('FLRM + S6F G 6.25')}
    block('(1) sensitivity', sub1, 'B', A.reps // 4, n=n, c_rng=c_rng, seed_off=2)
    sub2 = {k: v for k, v in CFG2.items() if k.startswith('S then S6F, S6F G 6.00') or k.startswith('S then S6F, S6F G 6.25')
            or k.startswith('S then S6F, S6F G 5.75') or k.startswith('S6F then S, S6F G 5.75')}
    block('(2) sensitivity', sub2, 'B', A.reps // 4, n=n, c_rng=c_rng, seed_off=3)

print('\n# D. bounds (no sampling). S6F: the latest start is G; hard deadline min(start + 3.3, NB_T0 + 8.2 h); compact_infer takes no new')
print('#    study after hard - 24 min; the RUN cell SIGKILLs every S6F process group at hard and waits <= 120 s; blend ~1 min. So S6F')
print('#    cannot push the kernel past ~8.24 h. The S fleet (flips20 cells, unchanged): starts only at age <= 7.0 h, wall 8.3 h + 300 s')
print('#    kill + HF -> <= 8.39 h (inherited from kernel_v224flips20; reached only if the 650-study S stage takes > 1.3 h, i.e. > 4x its')
print('#    MEASURED microbench). In the S-first order that path ends with S6F SKIPPED (age > G): both canaries, kernel <= ~8.40 h < 9 h.')
for G in (5.75, 6.0, 6.25):
    for c in (9.0, 13.0):
        dur = 0.08 + 650 * c / 3600
        print(f'  S6F G {G:.2f}, c {c:4.1f}: S6F wall {dur:.2f} h (s 1.0) / {dur * 1.3:.2f} h (s 1.3); a start AT G ends '
              f'{min(G + dur, DEADLINE - CI_MARGIN_H + c / 3600):.2f} / {min(G + dur * 1.3, DEADLINE - CI_MARGIN_H + 1.3 * c / 3600):.2f} h '
              f'({"used" if G + dur * 1.3 <= DEADLINE - CI_MARGIN_H else "cut (ACL canary) at s 1.3"})')
