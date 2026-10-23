# decide.py (BOX; driver.sh calls it during / after the 10-min smoke = the first minutes of the real run under watch).
# Reads the trainer logs pulled from the instance (LOGDIR/f<F>_s1047_ttr_r7tb.log) + logs/gpu_mon.csv, prints one JSON line:
#   verdict PASS | FAIL | WAIT, action keep | seq | abort, per-fold speed (s/it, TRAIN-ONLY, eval), fold-time MODEL per
#   concurrency, plan wall, projected TOTAL spend (= spend before this rental + $/h x (elapsed + remaining wall x 1.10 + fetch)
#   + data-in) and the guard-deadline tightening. ABORT if the projection exceeds the cap or the plan cannot end before the
#   guard deadline. 4-GPU layout "0:0 1:1 2:2 3:3 0:4": keep the pair on GPU 0 only if it beats "4 then 1" by >= 8 %
#   (T(2) <= 0.92 x 2 T(1)) and the paired GPU never passed 95 % memory and nothing ran out of memory; else seq = stop f4 and
#   re-run it on the first free GPU ("any:4"). 2-GPU layout "0:0 1:1 0:2 1:3 any:4": keep (OOM -> seq "0:0,2,4 1:1,3").
import argparse, glob, json, os, re, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--logs', required=True); ap.add_argument('--layout', required=True); ap.add_argument('--ng', type=int, required=True)
ap.add_argument('--seed', type=int, default=1047); ap.add_argument('--runtag', default='ttr_r7tb'); ap.add_argument('--epochs', type=int, default=21)
ap.add_argument('--dph', type=float, required=True); ap.add_argument('--spent_before', type=float, required=True)
ap.add_argument('--cap', type=float, default=15.0); ap.add_argument('--din', type=float, default=0.0)
ap.add_argument('--launch_epoch', type=int, required=True); ap.add_argument('--lanes_epoch', type=int, required=True)
ap.add_argument('--guard_deadline', type=int, required=True); ap.add_argument('--now', type=int, default=0)
ap.add_argument('--fetch_h', type=float, default=0.3); ap.add_argument('--final', type=int, default=0)
ap.add_argument('--min_it', type=int, default=200)
A = ap.parse_args()
now = A.now or int(time.time())
lanes = [it.split(':') for it in A.layout.split()]
gpu_of, conc = {}, {}
for g, fs in lanes:
    for f in fs.split(','):
        if g != 'any':
            gpu_of[int(f)] = int(g)
first = {}
for g, fs in lanes:                      # concurrency at the smoke = first fold of every fixed lane on that GPU
    if g != 'any':
        first.setdefault(int(g), []).append(int(fs.split(',')[0]))
for g, fl in first.items():
    for f in fl:
        conc[f] = len(fl)
res = dict(verdict='WAIT', action=None, reasons=[], folds={})
R = res['reasons']
mem = {}
gm = os.path.join(A.logs, 'gpu_mon.csv')
if os.path.exists(gm):
    for l in open(gm, errors='replace'):
        p = [x.strip() for x in l.split(',')]
        if len(p) >= 5 and p[1].isdigit():
            try:
                u, t = float(p[3]), float(p[4]); g = int(p[1]); mem[g] = max(mem.get(g, 0), u / t)
            except ValueError:
                pass
res['gpu_mem_peak_frac'] = {str(k): round(v, 3) for k, v in sorted(mem.items())}
oom_gpus, fatal = set(), []
for f, c in sorted(conc.items()):
    p = os.path.join(A.logs, f'f{f}_s{A.seed}_{A.runtag}.log')
    t = open(p, errors='replace').read() if os.path.exists(p) else ''
    d = dict(conc=c, gpu=gpu_of[f])
    its = re.findall(r'^ep 0 it (\d+)/(\d+) loss \S+ ([0-9.]+)s/it', t, re.M)
    if its:
        d['it'], d['iters'], d['sit'] = int(its[-1][0]), int(its[-1][1]), float(its[-1][2])
    m = re.search(r'^ep 0 TRAIN-ONLY ([0-9.]+) min', t, re.M)
    if m: d['train_min'] = float(m.group(1))
    m = re.search(r'^ep 0 ema .*\| epoch ([0-9.]+) min', t, re.M)
    if m: d['epoch_min'] = float(m.group(1))
    d['val'] = bool(re.search(r'^ep 0 raw .*val AUC', t, re.M))
    oom = bool(re.search(r'out of memory|OutOfMemoryError', t))
    err = bool(re.search(r'^Traceback|^\w*Error:|^!! ', t, re.M))
    d['oom'], d['error'] = oom, err
    if oom:
        oom_gpus.add(gpu_of[f])
    elif err:
        fatal.append(f'f{f}: ' + (re.findall(r'^(\w*Error: .*|!! .*)$', t, re.M) or ['error'])[-1][:160])
    res['folds'][str(f)] = d
eval_min = [v['epoch_min'] - v['train_min'] for v in res['folds'].values() if 'epoch_min' in v and 'train_min' in v]
for f, d in res['folds'].items():          # epoch estimate: measured epoch, else iters x s/it + eval (measured or V100 ratio)
    if 'epoch_min' in d:
        d['ep_min_est'] = d['epoch_min']
    elif 'sit' in d:
        tr = d['iters'] * d['sit'] / 60
        ev = max(eval_min) if eval_min else tr * 1.9 / 8.2          # V100 224: 8.2 min train + 1.9 min eval per epoch (MEASURED)
        d['ep_min_est'] = round(tr + ev, 2)
    if 'ep_min_est' in d:
        d['fold_h_est'] = round((A.epochs * d['ep_min_est'] + 2) / 60, 3)
if fatal:
    res.update(verdict='FAIL', action='abort'); R += fatal
elif any(g in oom_gpus for g in [gpu_of[int(f)] for f, d in res['folds'].items() if d['conc'] == 1]):
    res.update(verdict='FAIL', action='abort'); R.append('out of memory on an UNPAIRED GPU')
else:
    slow = [f for f, d in res['folds'].items() if not d['oom'] and ('sit' not in d or d['it'] < A.min_it)]
    anyval = any(d['val'] for d in res['folds'].values())
    if (slow or not anyval) and not A.final:
        res['verdict'] = 'WAIT'; R.append(f'waiting: folds without a stable s/it {slow}, epoch-0 val line seen {anyval}')
    elif not anyval or [f for f, d in res['folds'].items() if not d['oom'] and 'sit' not in d]:
        res.update(verdict='FAIL', action='abort'); R.append(f'no epoch-0 val line ({anyval}) or folds without progress by the end of the smoke window')
    else:
        T1 = [d['fold_h_est'] for d in res['folds'].values() if d['conc'] == 1 and 'fold_h_est' in d]
        T2 = [d['fold_h_est'] for d in res['folds'].values() if d['conc'] == 2 and 'fold_h_est' in d]
        el = (now - A.lanes_epoch) / 3600
        if A.ng >= 4:
            t1 = max(T1) if T1 else None; t2 = max(T2) if T2 else None
            paired = [g for g, fl in first.items() if len(fl) > 1]
            memok = all(mem.get(g, 0) <= 0.95 for g in paired)
            seq_w = (t1 or t2 / 1.6) - el + (t1 or t2 / 1.6)
            if oom_gpus or not T2:
                res['action'] = 'seq'; R.append('pairing impossible (OOM on the paired GPU)' if oom_gpus else 'no paired fold')
                wall = seq_w
            else:
                keep_w = max(t2, t1 or 0) - el
                res['keep_wall_h'], res['seq_wall_h'] = round(keep_w, 2), round(seq_w, 2)
                if keep_w <= 0.92 * seq_w and memok:
                    res['action'] = 'keep'; wall = keep_w; R.append(f'pair kept: {keep_w:.2f} h vs 4-then-1 {seq_w:.2f} h')
                else:
                    res['action'] = 'seq'; wall = seq_w
                    R.append(f'4-then-1: pair {keep_w:.2f} h vs {seq_w:.2f} h' + ('' if memok else ' (paired GPU memory > 95 %)'))
            res['fold_est_s'] = int(3600 * (t1 or t2))
        elif A.ng >= 2:
            t2 = max(T2 + T1)
            if oom_gpus:
                res['action'] = 'seq'; wall = 3 * (t2 / 1.5) - el; R.append('OOM with 2 per GPU: 3 rounds of 1 per GPU')
            else:   # 10-14 19:55: the extra round only when a fold waits for a GPU (5 folds); 4 folds = one round of 2 per GPU
                nfold = sum(len(fs.split(',')) for g, fs in lanes); extra = nfold > 2 * A.ng
                res['action'] = 'keep'; wall = t2 - el + (t2 * 0.75 if extra else 0.0)
                R.append('2 per GPU, then the 5th on the first free GPU' if extra else f'2 per GPU, one round ({nfold} folds)')
            res['fold_est_s'] = int(3600 * t2 * (0.75 if res['action'] == 'keep' else 1 / 1.5))
        else:
            wall = max(T1 + T2) * 5 - el; res['action'] = 'keep'; res['fold_est_s'] = int(3600 * max(T1 + T2))
        wall = max(wall, 0.05)
        res['wall_rem_h'] = round(wall, 2)
        spent_now = A.spent_before + A.dph * (now - A.launch_epoch) / 3600 + A.din
        proj = spent_now + A.dph * (wall * 1.10 + A.fetch_h)
        end = now + int(3600 * (wall * 1.10 + A.fetch_h))
        res.update(spent_now=round(spent_now, 2), proj_total=round(proj, 2), proj_end=end, cap=A.cap,
                   guard_tighten=int(now + 3600 * (wall * 1.35 + 1.0)))
        if proj > A.cap:
            res.update(verdict='FAIL', action='abort'); R.append(f'projected total ${proj:.2f} > cap ${A.cap:.2f}')
        elif end > A.guard_deadline - 600:
            res.update(verdict='FAIL', action='abort'); R.append('the plan cannot end 10 min before the guard deadline')
        else:
            res['verdict'] = 'PASS'
print(json.dumps(res))
