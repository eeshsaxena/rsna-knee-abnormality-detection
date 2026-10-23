#!/usr/bin/env python
# sim_ffs_runtime.py (2026-10-15 dir, written 10-14 ~23:55 IST by a Claude subagent; runtime review of V224-FF-S). MODEL.
# Hidden-run wall of V224-FF-S (build_v224fs.py --base ff: flip budget 4.4 h, fleet start guard 5.0 h, fleet wall 8.3 h,
# fleet = K14 R 20 + S 5 TTR224 in ONE stage) versus V224-FF20 (flip budget 4.9 h, guard 5.5 h, 20 members), calibrated
# to MEASURED numbers:
#   - V224-FLIP 57174021 submit-to-COMPLETE 4.60-4.65 h (API submit 08:08:06 UTC, api_chain detection 18:17 IST)
#   - V224-exact 57171955 submit-to-COMPLETE <= 3.28 h (read 15:55 IST); HR7 2.84-3.26 h; HV 3.18-3.35 h (subpoll logs)
#   - fleet 20 (K14 R) hidden: HFR7-RR-F40 4.85-4.92 h vs HR7-RR <= 3.42 h -> fleet adds >= ~1.45 h (wall_model 1.66-1.80 h)
#   - fleet microbench per study per T4, both TTA passes (3 commits, receipts): K14 20 members 8.65-10.18 s; ONE TTR224
#     member 287-325 ms -> S (5 TTR224, same arch/geometry as the DT members) 1.44-1.62 s
#   - driver rule of compact_infer_ta.py (k >= 10: t_k + 1.05 x rate x files left > budget and work > 600 s; k < 10: t_k > budget)
# Unknowns (GUESS): queue q (0-0.3 h), r224 + stage 3 hidden (0.15-0.30 h), session speed spread.
#   usage: python sim_ffs_runtime.py [--reps 400] [--ao-reps 4000] [--only-ao]
# AO 10-15 (PREREG_10-15 Amendment 3 R2 / R3, build_v224fs.py --base flip|flrm): section D (own RNG, so A-C print the same
#   bytes as before) models FLIP-AO + S and FLRM-AO + S: the scored V224-FLIP kernel (R7 flip FORCED on every study, no time-aware
#   driver) + RM (flrm, +0.06-0.16 h MODEL) + SAVE / fleet pre-cells, then a fleet holding ONLY the 5 S members (no K14) behind a
#   start guard G (the fleet starts only if the kernel is <= G h old), wall 8.3 h. Calibrated to the MEASURED V224-FLIP
#   submit-to-score 4.57-4.65 h (= queue + kernel) and the MEASURED S microbench (5 x TTR224 <= 1.62 s / study / T4 = +0.29 h).
#   --only-ao skips A-C.
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reps', type=int, default=400)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ao-reps', type=int, default=4000)
ap.add_argument('--only-ao', action='store_true')
A = ap.parse_args()

PRE = 0.42           # pip wheels + r384 Raptor stage before R7 (MODEL: Danial r384 ~25 min hidden + setup)
FLIP_TOTAL = 4.62    # MEASURED V224-FLIP submit-to-score (midpoint of 4.60-4.65)
N_R7 = 650           # studies per R7 worker (1,300 hidden, 2 T4)
N_FL = 650           # studies per fleet GPU
FIX_FL = 1.5 / 60    # fleet fixed part (MEASURED ~1.04 min on the commit) + S loads / S-MERGE sha (GUESS +0.5 min)
PRECELLS = 0.01      # SAVE, ENV-SNAP, FN-IDS, K12-MERGE, S-MERGE, CLOCK (commit: 8 s)
WALL, GRACE, HARD = 8.3, 300 / 3600, 9.0
K14_S = 9.86         # s / study / slower T4, 20 members (V224-F commit, the slowest of 3 commits' slower-GPU values except ctl 10.18)
S_S = 1.62           # s / study / slower T4, 5 TTR224 members (max measured 324.8 ms x 5)


def r7_worker(rng, n, plain_h, flip_h, pre_h, budget_h, aware=True):
    """One R7 worker under the driver (sim_ta.py worker, unchanged rule). Returns (end_h, flipped share)."""
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    order = np.argsort(-files, kind='mergesort')
    order = np.concatenate([order[::2], order[1::2]])
    f = files[order]
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (flip_h * 3600 - 250) / f.sum()
    rp = (plain_h * 3600 - 250) / f.sum()
    t = pre_h * 3600 + 250
    if aware and t - 250 >= budget_h * 3600:              # budget passed before the worker started: plain only
        return (t + (rp * f * noise).sum()) / 3600, 0.0
    rows = []
    for k in range(n):
        t += rf * f[k] * noise[k]
        rows.append(t)
        if not aware:
            continue
        kk = k + 1
        if kk >= n:
            break
        rem = f[kk:].sum()
        if kk >= 10:
            rate = (rows[-1] - rows[0]) / f[1:kk].sum()
            work = 1.05 * rate * rem
            trig = t + work > budget_h * 3600 and work > 600
        else:
            trig = t > budget_h * 3600
        if trig:
            t += 250 + (rp * f[kk:] * noise[kk:]).sum()
            return t / 3600, kk / n
    return t / 3600, 1.0


def run(rng, s, q, r224, mf, plain_h, cfg, s_s=None, n=None):
    """One hidden run. s = session speed factor (all stages), mf = fleet-only slow factor. Returns a dict."""
    s_s = S_S if s_s is None else s_s
    n = 2 * N_FL if n is None else n
    flip_h = FLIP_TOTAL - q - PRE - r224                    # calibrated so that V224-FLIP (always-on) == 4.62 h at s = 1
    w = [r7_worker(rng, n // 2, plain_h * s * n / 1300, flip_h * s * n / 1300, PRE * s, cfg['budget'], cfg['aware']) for _ in range(2)]
    r7_end = max(e for e, _ in w)
    share = float(np.mean([x for _, x in w]))
    fs = r7_end + r224 * s + PRECELLS
    per = K14_S + (s_s if cfg['S'] else 0.0)
    if fs > cfg['guard']:
        return dict(r7=r7_end, share=share, fs=fs, end=fs + 0.01, fleet='SKIPPED', wall=fs + 0.01)
    fe = fs + (FIX_FL + (n / 2) * per * mf * s / 3600)
    if fe > WALL:
        end = WALL + GRACE + 0.01
        return dict(r7=r7_end, share=share, fs=fs, end=end, fleet='HARDSTOP', wall=end)
    return dict(r7=r7_end, share=share, fs=fs, end=fe + 0.005, fleet='ok', wall=fe + 0.005)


CFGS = {
    'FF-S as built (flip 4.4 / guard 5.0 / 25)': dict(budget=4.4, guard=5.0, S=True, aware=True),
    'FF20 (flip 4.9 / guard 5.5 / 20), 05:36 entry': dict(budget=4.9, guard=5.5, S=False, aware=True),
    'alt FF-S (flip 4.9 / guard 5.5 / 25)': dict(budget=4.9, guard=5.5, S=True, aware=True),
    'alt FF-S (flip 4.65 / guard 5.2 / 25)': dict(budget=4.65, guard=5.2, S=True, aware=True),
    'alt FF-S (flip 4.9 / guard 5.2 / 25)': dict(budget=4.9, guard=5.2, S=True, aware=True),
    'alt FF-S (flip 4.9 / guard 5.1 / 25)': dict(budget=4.9, guard=5.1, S=True, aware=True),
    'FF-S, driver error -> always-on flip (4.4/5.0/25)': dict(budget=99, guard=5.0, S=True, aware=False),
}


# ---- D. AO 10-15: always-on flip base (V224-FLIP / V224-FLRM, both scored or commit-tested kernels) + S-only fleet ----------------
RM_LO, RM_HI = 0.06, 0.16    # MODEL: RM (r224 mirror TTA) adds +3.6-9.6 min hidden (v224rm/FLIPRM_RESULT.md, rm_runtime.py)
HF_H = 0.005                 # HF cell + receipts on ~1,300 rows (commit: < 1 s on 3 rows)
R7_DEADLINE, R224_SKIP_AT, LIMIT9 = 8.3, 7.83, 9.0   # her base guards (cell 6 DEADLINE_HOURS, stage 2b skip / 9 h - 40 min timeout)
AO_CFGS = {
    'V224-FLIP as scored (no fleet)': dict(guard=None, rm=False),
    'V224-FLRM (no fleet)': dict(guard=None, rm=True),
    'FLIP-AO + S, guard 6.0': dict(guard=6.0, rm=False),
    'FLIP-AO + S, guard 6.5': dict(guard=6.5, rm=False),
    'FLIP-AO + S, guard 7.0': dict(guard=7.0, rm=False),
    'FLIP-AO + S, guard 7.5': dict(guard=7.5, rm=False),
    'FLRM-AO + S, guard 6.0': dict(guard=6.0, rm=True),
    'FLRM-AO + S, guard 6.5': dict(guard=6.5, rm=True),
    'FLRM-AO + S, guard 7.0': dict(guard=7.0, rm=True),
}


def r7_flip_ao(rng, n, flip_h, pre_h):
    """One R7 worker with the flip FORCED on every study (V224-FLIP: pinned compact_infer.py --flip-tta, no time-aware driver).
    Same per-study cost model as r7_worker (lognormal file counts, per-study noise, ~250 s compile check), no switch."""
    files = np.maximum(20, rng.lognormal(np.log(220), 0.45, n)).astype(int)
    noise = rng.normal(1.0, 0.15, n).clip(0.6, 1.6)
    rf = (flip_h * 3600 - 250) / files.sum()
    return (pre_h * 3600 + 250 + (rf * files * noise).sum()) / 3600


def run_ao(rng, s, q, r224, mf, cfg, s_s=None, n=None, ftot=None):
    """One hidden run of an AO kernel. Returns base end (V224-FLIP(-RM) part), fleet start, kernel end, fleet outcome."""
    s_s = S_S if s_s is None else s_s
    n = 2 * N_FL if n is None else n
    ftot = FLIP_TOTAL if ftot is None else ftot
    flip_h = ftot - q - PRE - r224                        # always-on flip reproduces the MEASURED V224-FLIP latency at s = 1
    r7_raw = max(r7_flip_ao(rng, n // 2, flip_h * s * n / 1300, PRE * s) for _ in range(2))
    r7_end = min(r7_raw, R7_DEADLINE + 0.02)              # her R7 deadline: unscored studies -> per-study Raptor fallback (not canaried)
    rm = rng.uniform(RM_LO, RM_HI) * s if cfg['rm'] else 0.0
    if r7_end > R224_SKIP_AT:                             # her stage 2b skips r224 -> V224 canary (Fracture 0.5); stage 3 + canary only
        base, bflag = r7_end + 0.01, 'R224_SKIP'
    else:                                                 # her r224 timeout min(left, 2.5 h), left = 9 h - 40 min - age
        base = min(r7_end + r224 * s + rm, LIMIT9 - 40 / 60 + 0.01)
        bflag = 'R224_TIMEOUT' if base >= LIMIT9 - 40 / 60 else ('R7_CUT' if r7_raw > R7_DEADLINE else 'ok')
    if cfg['guard'] is None:
        return dict(base=base, bflag=bflag, fs=None, end=base, fleet='none')
    fs = base + PRECELLS
    if fs > cfg['guard']:                                 # the fleet's own start guard: stage not started -> Synovitis canary
        return dict(base=base, bflag=bflag, fs=fs, end=fs + HF_H, fleet='SKIPPED')
    fe = fs + FIX_FL + (n / 2) * s_s * mf * s / 3600       # S only (5 x TTR224), GPU-bound (decode 4 threads ~0.4 s / study)
    if fe > WALL:
        return dict(base=base, bflag=bflag, fs=fs, end=WALL + GRACE + HF_H, fleet='HARDSTOP')
    return dict(base=base, bflag=bflag, fs=fs, end=fe + HF_H, fleet='ok')


def draw_b(rng):                                          # section B's session distribution (GUESS)
    s = float(np.exp(rng.normal(0, 0.08)) * (1.3 if rng.random() < 0.05 else 1.0))
    return s, rng.uniform(0.0, 0.3), rng.uniform(0.15, 0.30), rng.uniform(1.0, 1.15)


def draw_stress(rng):                                     # heavier tail (GUESS): sd .12, 9 % x1.3, 3 % x1.5 sessions
    s = float(np.exp(rng.normal(0, 0.12)) * rng.choice([1.0, 1.3, 1.5], p=[0.88, 0.09, 0.03]))
    return s, rng.uniform(0.0, 0.3), rng.uniform(0.15, 0.30), rng.uniform(1.0, 1.15)


def ao_mc(rng, cfg, draw, reps, **kw):
    E, FL, FS, SS, BF = [], [], [], [], []
    for _ in range(reps):
        s, q, r224, mf = draw(rng)
        r = run_ao(rng, s, q, r224, mf, cfg, **kw)
        E.append(r['end']); FL.append(r['fleet']); FS.append(r['fs'] if r['fs'] is not None else np.nan); SS.append(r['end'] + q)
        BF.append(r['bflag'])
    return np.array(E), FL, np.array(FS), np.array(SS), BF


def ao_section():
    rng = np.random.default_rng(A.seed + 1015)
    print('\n# D. AO 10-15 (PREREG_10-15 A3 R2 / R3): V224-FLIP or V224-FLRM base (flip FORCED, no driver) + S-only fleet (5 TTR224,')
    print('#    no K14) behind a start guard G, wall 8.3 h. MODEL. Calibration: an always-on flip run at s 1 ends at 4.62 - q h (MEASURED')
    print('#    V224-FLIP 4.57-4.65 h submit-to-score); S = 650 x %.2f s x mf x s (MEASURED microbench, +%.2f h central); RM +%.2f-%.2f h.'
          % (S_S, N_FL * S_S / 3600, RM_LO, RM_HI))
    print('\n## D1 grid (calibration q 0.1, r224+stage3 0.2 h; 40 runs per cell): base end | fleet start | fleet outcome | kernel end')
    for name in ('FLIP-AO + S, guard 6.5', 'FLRM-AO + S, guard 6.5', 'FLIP-AO + S, guard 6.0', 'FLIP-AO + S, guard 7.0'):
        cfg = AO_CFGS[name]
        print(f'  {name}')
        for s in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.5, 1.6):
            for mf in (1.0, 1.15):
                rs = [run_ao(rng, s, 0.1, 0.2, mf, cfg) for _ in range(40)]
                fl = [r['fleet'] for r in rs]
                print(f'    s {s:4.2f} mf {mf:4.2f} | base {np.mean([r["base"] for r in rs]):.2f} | fs {np.mean([r["fs"] for r in rs]):.2f} '
                      f'| fleet ok {fl.count("ok")}/40 skip {fl.count("SKIPPED")} hard {fl.count("HARDSTOP")} '
                      f'| end mean {np.mean([r["end"] for r in rs]):.2f} max {np.max([r["end"] for r in rs]):.2f}')
    for dname, draw in (('B (as section B: s = exp(N(0, .08)) x 1.3 w.p. .05)', draw_b),
                        ('STRESS (s = exp(N(0, .12)) x {1, 1.3, 1.5} w.p. {.88, .09, .03})', draw_stress)):
        print(f'\n## D2 Monte Carlo {dname}, q ~ U(0, .3), r224 ~ U(.15, .30), mf ~ U(1.0, 1.15); {A.ao_reps} sessions per config')
        for name, cfg in AO_CFGS.items():
            E, FL, FS, SS, BF = ao_mc(rng, cfg, draw, A.ao_reps)
            fs_ok = FS[np.isfinite(FS)]
            print(f'  {name}: kernel end median {np.median(E):.2f} p90 {np.percentile(E, 90):.2f} p99 {np.percentile(E, 99):.2f} '
                  f'max {E.max():.2f} | submit-to-score median {np.median(SS):.2f} p90 {np.percentile(SS, 90):.2f} | '
                  + (f'P(fleet skipped) {FL.count("SKIPPED") / len(FL):.4f} P(hard stop) {FL.count("HARDSTOP") / len(FL):.4f} '
                     f'| fleet start p99 {np.percentile(fs_ok, 99):.2f} max {fs_ok.max():.2f} | ' if cfg['guard'] is not None else '')
                  + f'P(end > 8.0 h) {np.mean(E > 8.0):.4f} P(end > 9 h) {np.mean(E > HARD):.4f} | base guards (same in V224-FLIP): '
                  f'R7 cut {BF.count("R7_CUT") / len(BF):.4f} r224 skip / timeout {(BF.count("R224_SKIP") + BF.count("R224_TIMEOUT")) / len(BF):.4f}')
    print('\n## D3 sensitivity (MC as D2-B, %d sessions): S cost / study / T4 (MEASURED <= 1.62; 2.77 = TTR288-like; 4.0 = 2.5x),' % A.ao_reps)
    print('#    hidden study count (1,300 claimed; 1,500 = +15 %), FLIP calibration 4.57 / 4.65 h (the MEASURED window ends)')
    for name in ('FLIP-AO + S, guard 6.5', 'FLRM-AO + S, guard 6.5'):
        cfg = AO_CFGS[name]
        for s_s, n, ftot in ((1.62, 1300, None), (2.77, 1300, None), (4.0, 1300, None), (1.62, 1500, None), (2.77, 1500, None),
                             (1.62, 1300, 4.57), (1.62, 1300, 4.65)):
            E, FL, FS, SS, BF = ao_mc(rng, cfg, draw_b, A.ao_reps, s_s=s_s, n=n, ftot=ftot)
            print(f'  {name} | S {s_s:.2f} s, N {n}, FLIP {ftot or FLIP_TOTAL:.2f} h: kernel end median {np.median(E):.2f} '
                  f'p99 {np.percentile(E, 99):.2f} max {E.max():.2f} | P(skip) {FL.count("SKIPPED") / len(FL):.4f} '
                  f'P(hard) {FL.count("HARDSTOP") / len(FL):.4f} | submit-to-score median {np.median(SS):.2f}')
    print('\n## D4 deterministic (no sampling; R7 per-study noise at its mean): the session speed s* at which the base ends at the guard')
    print('#    (calibration q 0.1 / r224 0.2: base(s) = s x (4.62 - 0.1) [+ RM 0.11 x s]), and the latest clean end = start at G, then the')
    print('#    S fleet at x1.15 x s_f; an S fleet at 4.0 s / study (2.5x) as well')
    for g in (6.0, 6.5, 7.0, 7.5):
        for rmc, lab in ((0.0, 'FLIP'), (0.11, 'FLRM')):
            sstar = (g - PRECELLS) / (FLIP_TOTAL - 0.1 + rmc)
            ends = {sf: g + FIX_FL + N_FL * S_S * 1.15 * sf / 3600 + HF_H for sf in (1.0, 1.3, 1.45, sstar)}
            e4 = g + FIX_FL + N_FL * 4.0 * 1.15 * sstar / 3600 + HF_H
            print(f'  G {g:.1f} {lab}: skip only if s > {sstar:.3f} (P(s > s*) under D2-B ~{np.mean([draw_b(rng)[0] > sstar for _ in range(20000)]):.4f}, '
                  f'STRESS ~{np.mean([draw_stress(rng)[0] > sstar for _ in range(20000)]):.4f}) | start at G, S fleet x1.15 x {{1.0, 1.3, 1.45, s*}}: '
                  f'end {ends[1.0]:.2f} / {ends[1.3]:.2f} / {ends[1.45]:.2f} / {ends[sstar]:.2f} h; S at 4.0 s and s*: {e4:.2f} h '
                  f'-> {"inside 8.0 h" if max(max(ends.values()), e4) <= 8.0 else ("inside the 8.3 h wall" if max(max(ends.values()), e4) <= WALL else "HARD STOP")}')
    print('  hard bound (any G): the fleet wall stops workers at 8.3 h, kills them 300 s later, HF falls back (Synovitis canary):'
          f' kernel <= {WALL + GRACE + HF_H:.2f} h; R7 deadline 8.3 h and her r224 skip at 7.83 h are unchanged (they run before the fleet)')


if A.only_ao:
    ao_section()
    raise SystemExit(0)

rng = np.random.default_rng(A.seed)
print('# A. grid: session speed s x calibration (q queue h, r224 h, R7 plain h); fleet factor mf 1.0 / 1.15. MODEL.')
print('# columns: R7 end | flip share | fleet start | fleet | kernel end (h of kernel age); submit-to-score = end + q')
for q, r224, plain in ((0.10, 0.20, 2.35), (0.30, 0.15, 2.35), (0.0, 0.30, 2.55)):
    print(f'\n## calibration q {q} h, r224+stage3 {r224} h, R7 plain {plain} h (R7 flip {FLIP_TOTAL - q - PRE - r224:.2f} h at s 1)')
    for name, cfg in CFGS.items():
        print(f'  {name}')
        for s in (0.9, 0.95, 1.0, 1.05, 1.15, 1.3, 1.45):
            for mf in (1.0, 1.15):
                rs = [run(rng, s, q, r224, mf, plain, cfg) for _ in range(40)]
                fl = [r['fleet'] for r in rs]
                print(f'    s {s:4.2f} mf {mf:4.2f} | R7 {np.mean([r["r7"] for r in rs]):.2f} | flip {100 * np.mean([r["share"] for r in rs]):3.0f} % '
                      f'| fs {np.mean([r["fs"] for r in rs]):.2f} | fleet ok {fl.count("ok")}/40 skip {fl.count("SKIPPED")} hard {fl.count("HARDSTOP")} '
                      f'| end mean {np.mean([r["end"] for r in rs]):.2f} max {np.max([r["end"] for r in rs]):.2f}')

print('\n# B. Monte Carlo over sessions (GUESS distribution): s = exp(N(0, .08)) x (1.3 with p .05), q ~ U(0, .3), r224 ~ U(.15, .30),')
print('#    R7 plain ~ U(2.25, 2.55) h (V224-exact <= 3.28 h MEASURED), mf ~ U(1.0, 1.15). Kernel end h; P = fraction of runs.')
for name, cfg in CFGS.items():
    E, SH, FL, FS, SS = [], [], [], [], []
    for _ in range(A.reps):
        s = float(np.exp(rng.normal(0, 0.08)) * (1.3 if rng.random() < 0.05 else 1.0))
        q = rng.uniform(0.0, 0.3); r224 = rng.uniform(0.15, 0.30); plain = rng.uniform(2.25, 2.55); mf = rng.uniform(1.0, 1.15)
        r = run(rng, s, q, r224, mf, plain, cfg)
        E.append(r['end']); SH.append(r['share']); FL.append(r['fleet']); FS.append(r['fs']); SS.append(r['end'] + q)
    E = np.array(E); SH = np.array(SH); SS = np.array(SS)
    print(f'  {name}: kernel end median {np.median(E):.2f} p90 {np.percentile(E, 90):.2f} p99 {np.percentile(E, 99):.2f} max {E.max():.2f} '
          f'| submit-to-score median {np.median(SS):.2f} | flip share mean {100 * SH.mean():.0f} %, P(>=90 %) {np.mean(SH >= .9):.2f}, '
          f'P(<=10 %) {np.mean(SH <= .1):.2f} | P(fleet skipped) {FL.count("SKIPPED") / len(FL):.3f} P(hard stop) {FL.count("HARDSTOP") / len(FL):.3f} '
          f'| P(end > 9 h) {np.mean(E > HARD):.3f} | fleet start p99 {np.percentile(FS, 99):.2f}')

print('\n# C. sensitivity of the as-built FF-S (MC as in B): S cost per study per T4 (MEASURED 1.62 s; 2.77 s = if S cost like a')
print('#    TTR288 member; 4.0 s = 2.5x) and hidden study count (1,300 claimed; 1,500 = +15 %).')
cfg = CFGS['FF-S as built (flip 4.4 / guard 5.0 / 25)']
for s_s, n in ((1.62, 1300), (2.77, 1300), (4.0, 1300), (1.62, 1500), (2.77, 1500)):
    E, FL, SH = [], [], []
    for _ in range(A.reps):
        s = float(np.exp(rng.normal(0, 0.08)) * (1.3 if rng.random() < 0.05 else 1.0))
        q = rng.uniform(0.0, 0.3); r224 = rng.uniform(0.15, 0.30); plain = rng.uniform(2.25, 2.55); mf = rng.uniform(1.0, 1.15)
        r = run(rng, s, q, r224, mf, plain, cfg, s_s=s_s, n=n)
        E.append(r['end']); FL.append(r['fleet']); SH.append(r['share'])
    E = np.array(E)
    print(f'  S {s_s:.2f} s/study/T4, N {n}: kernel end median {np.median(E):.2f} p99 {np.percentile(E, 99):.2f} max {E.max():.2f} | '
          f'flip share mean {100 * np.mean(SH):.0f} % | P(fleet skipped) {FL.count("SKIPPED") / len(FL):.3f} P(hard stop) {FL.count("HARDSTOP") / len(FL):.3f}')
# deterministic worst bound (no sampling): flip kept until the switch limit (budget + 600 s of unswitched flip work), r224 at
# x1.45, fleet at x1.45 x 1.15 from that start.
for name, cfg in CFGS.items():
    if not cfg['aware']:
        continue
    r7w = cfg['budget'] + 600 / 3600
    fsw = min(r7w + 0.30 * 1.45 + PRECELLS, cfg['guard'])
    per = K14_S + (S_S if cfg['S'] else 0.0)
    few = fsw + FIX_FL + N_FL * per * 1.15 * 1.45 / 3600
    print(f'  bound {name}: R7 <= {r7w:.2f}, fleet start <= {fsw:.2f}, fleet end at x1.45 x 1.15 {few:.2f} '
          f'-> {"HARD STOP at 8.3 h (+300 s), Synovitis canary, kernel ends ~8.4 h" if few > WALL else "inside the 8.3 h wall"}')

ao_section()
