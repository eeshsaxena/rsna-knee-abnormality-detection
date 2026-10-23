# decide.py (rental_kc, BOX; driver.sh calls it). No em dashes. Prints one JSON line.
#  --smoke: reads the smoke logs pulled after phase 1 (2 trainers per GPU) and phase 2 (1 per GPU) from DIR/p1/smoke and
#           DIR/p2/smoke (log = <out>.log, out from the registry), plus logs/gpu_mon.csv. Per member: iterations per epoch, the
#           last cumulative s/it (includes the loop warmup: conservative), errors / OOM. Per kind (cnxs = ConvNeXt-S AD members,
#           ev2s = KC2 EffNetV2-S fold 0): sit_paired (max over the phase-1 members of that kind), sit_single (phase 2), iters,
#           eval minutes per epoch (MODEL: gold74 x raw+ema for AD; + the fold-0 val set and the G1 hook for KC2) and member
#           hours h1 (single) / h2 (paired). verdict FAIL on a traceback, no progress (< min_it iterations), an OOM on an
#           unpaired GPU, or a missing single-GPU speed for a kind; an OOM only when paired -> pairing disabled.
#  --go:    plan from the measured speeds: members in priority order (--avail), spend so far = spent_before + $/h x elapsed +
#           data, time budget T = min((cap - spent_now) / dph - fetch_h - tail_h, guard_deadline - now - 0.5 h).
#           SEQ plan: one lane per GPU, members assigned greedily (priority order) to the GPU that frees first, admitted only if
#           the plan end x margin <= T. PAIR plan (4 members at once, 2 per GPU): admitted only if max h2 x margin <= T and
#           pairing is not disabled. The plan with more members wins; with equal members PAIR only if its wall < 0.92 x SEQ (SEQ
#           delivers the priority members first). FAIL if no member fits.
#           Output: layout ("0:a,b 1:c,d" = lanes running members in sequence; PAIR "0:a 0:c 1:b 1:d"), members, wall_h,
#           proj_total, deadline_epoch (lane admission line), est_s per kind (h1 for SEQ, h2 for PAIR), guard_tighten.
import argparse, json, os, re, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--go', action='store_true')
ap.add_argument('--reg', default=''); ap.add_argument('--p1', default=''); ap.add_argument('--p2', default='')
ap.add_argument('--dir', default=''); ap.add_argument('--epochs', type=int, default=13); ap.add_argument('--min_it', type=int, default=400)
ap.add_argument('--speeds', default=''); ap.add_argument('--avail', default=''); ap.add_argument('--ng', type=int, default=2)
ap.add_argument('--dph', type=float, default=0); ap.add_argument('--spent_before', type=float, default=0)
ap.add_argument('--cap', type=float, default=7.4); ap.add_argument('--din', type=float, default=0)
ap.add_argument('--launch_epoch', type=int, default=0); ap.add_argument('--guard_deadline', type=int, default=0)
ap.add_argument('--fetch_h', type=float, default=0.3); ap.add_argument('--tail_h', type=float, default=0.3)
ap.add_argument('--margin', type=float, default=1.10); ap.add_argument('--now', type=int, default=0)
A = ap.parse_args()
now = A.now or int(time.time())
kind = lambda k: 'ev2s' if k == 's2011' else 'cnxs'
res = dict(verdict='FAIL', reasons=[])
R = res['reasons']
if A.smoke:
    reg = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(A.reg) if l.strip()}
    per = {}
    for ph, lay in (('p1', A.p1), ('p2', A.p2)):
        items = [it.split(':') for it in lay.split()]
        conc = {}
        for g, k in items:
            conc[g] = conc.get(g, 0) + 1
        for g, k in items:
            p = os.path.join(A.dir, ph, 'smoke', reg[k][1] + '.log')
            t = open(p, errors='replace').read() if os.path.exists(p) else ''
            d = dict(phase=ph, gpu=g, conc=conc[g], key=k, kind=kind(k))
            its = re.findall(r'^ep 0 it (\d+)/(\d+) loss \S+ ([0-9.]+)s/it', t, re.M)
            if its:
                d['it'], d['iters'], d['sit'] = int(its[-1][0]), int(its[-1][1]), float(its[-1][2])
            d['oom'] = bool(re.search(r'out of memory|OutOfMemoryError|CUDNN_STATUS_NOT_ENOUGH_MEMORY|CUBLAS_STATUS_ALLOC_FAILED', t, re.I))
            errs = re.findall(r'^(\w*Error: .*|Traceback .*|!! .*)$', t, re.M)
            d['error'] = errs[-1][:160] if errs else ''
            per[f'{ph}:{k}'] = d
    res['members'] = per
    oom_paired = False
    for name, d in per.items():
        if d['oom']:
            if d['conc'] > 1:
                oom_paired = True
            else:
                R.append(f'{name}: out of memory on an UNPAIRED GPU')
        elif d['error']:
            R.append(f"{name}: {d['error']}")
        elif d.get('it', -1) < A.min_it:
            R.append(f"{name}: no stable progress ({d.get('it', 'no')} iterations)")
    kinds = {}
    for kd in sorted({d['kind'] for d in per.values()}):
        p1 = [d for d in per.values() if d['kind'] == kd and d['phase'] == 'p1' and d['conc'] > 1 and 'sit' in d]
        p2 = [d for d in per.values() if d['kind'] == kd and d['conc'] == 1 and 'sit' in d]
        it_n = max([d['iters'] for d in per.values() if d['kind'] == kd and 'iters' in d] or [0])
        if not p2:
            R.append(f'{kd}: no single-GPU speed measured'); continue
        s1 = max(d['sit'] for d in p2); s2 = max(d['sit'] for d in p1) if p1 else None
        if kd == 'cnxs':   # all-data: gold74 x (raw + ema) per epoch (~0.4 train-iteration per eval study, MODEL) + startup
            ev = lambda s: 2 * 74 * 0.4 * s / 60; hook = 0.0
        else:              # KC2 fold 0: val (~867) + gold74, x (raw + ema), + the G1 hook (4 variants on val) at the end
            ev = lambda s: 2 * (867 + 74) * 0.4 * s / 60; hook = 4 * 867 * 0.4 * s1 / 60
        h = lambda s: (A.epochs * (it_n * s / 60 + ev(s)) + hook + 3.0) / 60
        kinds[kd] = dict(iters=it_n, sit_single=s1, sit_paired=s2, h1=round(h(s1), 3), h2=round(h(s2), 3) if s2 else None,
                         pair_gain=round(2 * h(s1) / h(s2), 3) if s2 else None)
    res['kinds'] = kinds; res['pairing_disabled'] = oom_paired or not all(v.get('h2') for v in kinds.values())
    mem = {}
    gm = os.path.join(A.dir, 'p1', 'logs', 'gpu_mon.csv')
    if os.path.exists(gm):
        for l in open(gm, errors='replace'):
            p = [x.strip() for x in l.split(',')]
            if len(p) >= 5 and p[1].isdigit():
                try:
                    mem[int(p[1])] = max(mem.get(int(p[1]), 0), float(p[3]) / float(p[4]))
                except ValueError:
                    pass
    res['gpu_mem_peak_frac'] = {str(k): round(v, 3) for k, v in sorted(mem.items())}
    if not R:
        res['verdict'] = 'PASS'
elif A.go:
    sp = json.load(open(A.speeds)); K = sp['kinds']
    avail = [k for k in A.avail.split() if kind(k) in K]
    spent_now = A.spent_before + A.dph * (now - A.launch_epoch) / 3600 + A.din
    T = min((A.cap - spent_now) / A.dph - A.fetch_h - A.tail_h, (A.guard_deadline - now) / 3600 - 0.5)
    res.update(spent_now=round(spent_now, 2), time_budget_h=round(T, 2), avail=avail)
    # SEQ: greedy, priority order, to the GPU that frees first
    free = [0.0] * A.ng; lanes = [[] for _ in range(A.ng)]; seq = []
    for k in avail:
        g = min(range(A.ng), key=lambda i: free[i]); end = free[g] + K[kind(k)]['h1']
        if max(max(free), end) * A.margin <= T:
            free[g] = end; lanes[g].append(k); seq.append(k)
    seq_wall = max(free)
    plan = dict(kind='SEQ', members=seq, wall_h=round(seq_wall, 3), layout=' '.join(f'{g}:{",".join(l)}' for g, l in enumerate(lanes) if l),
                est_s={kd: int(3600 * v['h1']) for kd, v in K.items()})
    if len(avail) >= 2 * A.ng and not sp.get('pairing_disabled'):
        pm = avail[:2 * A.ng]; pw = max(K[kind(k)]['h2'] for k in pm)
        if pw * A.margin <= T and (len(pm) > len(seq) or (len(pm) == len(seq) and pw < 0.92 * seq_wall)):   # 10-15 17:30: SEQ unless pairing saves >= 8 % (SEQ delivers the priority members first)
            plan = dict(kind='PAIR', members=pm, wall_h=round(pw, 3), layout=' '.join(f'{i % A.ng}:{k}' for i, k in enumerate(pm)),
                        est_s={kd: int(3600 * v['h2']) for kd, v in K.items() if v.get('h2')})
        res['pair_option'] = dict(members=pm, wall_h=round(pw, 3), fits=bool(pw * A.margin <= T))
    res['seq_option'] = dict(members=seq, wall_h=round(seq_wall, 3))
    res.update(plan)
    if not plan['members']:
        R.append(f'no member fits: time budget {T:.2f} h vs single member {min(v["h1"] for v in K.values()):.2f} h x {A.margin}')
    else:
        w = plan['wall_h']
        res.update(proj_total=round(spent_now + A.dph * (w * A.margin + A.fetch_h + A.tail_h), 2), cap=A.cap,
                   deadline_epoch=int(now + 3600 * T), guard_tighten=int(min(A.guard_deadline, now + 3600 * (w * 1.35 + 1.0))))
        res['verdict'] = 'PASS'
print(json.dumps(res))
