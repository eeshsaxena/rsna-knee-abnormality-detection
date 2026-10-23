# smoke_verdict.py: reads the smoke logs, writes ~/raptor_store/smoke/verdict.txt with SMOKE_GO / SMOKE_NOGO (PREREG_RAPTOR_SEED
# smoke gate) and the epoch decision for the full run (16 if the projection <= 26 h, else 12; rule fixed by main 10-13).
import re, sys, os, json, statistics
D = os.path.expanduser('~/raptor_store/smoke')
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 6748
out = []


def parse(log):
    t = open(log).read() if os.path.exists(log) else ''
    sps = [float(x) for x in re.findall(r' ([0-9.]+)s/step', t)]
    mem = [float(x) for x in re.findall(r'peak mem ([0-9.]+) GB', t)]
    nonf = [int(x) for x in re.findall(r'nonfinite_loss (\d+)', t)]
    skip = [int(x) for x in re.findall(r'skipped_upd (\d+)', t)]
    ev = re.findall(r'eval (\d+)s\)', t)
    return dict(sps=sps, sps_med=(statistics.median(sps[1:]) if len(sps) > 1 else (sps[0] if sps else None)),
                mem=max(mem) if mem else None, nonfinite=max(nonf) if nonf else None, skipped=max(skip) if skip else None,
                eval_s=int(ev[0]) if ev else None, swa='SERVE-LOAD OK' in t, abort='ABORT' in t, tb='Traceback' in t)


A = parse(f'{D}/A_cl1/train.log'); B = parse(f'{D}/B_cl0/train.log')
sv = open(f'{D}/serve.log').read() if os.path.exists(f'{D}/serve.log') else ''
sv_ok = sv.count('SERVE ') >= 2 and 'Traceback' not in sv
out.append(f'A channels_last=1: {json.dumps(A)}')
out.append(f'B channels_last=0: {json.dumps(B)}')
out.append('serve: ' + ' || '.join(l for l in sv.splitlines() if l.startswith('SERVE')))
best = min([x for x in (A, B) if x['sps_med']], key=lambda x: x['sps_med'], default=None)
cl = 1 if best is A else 0
ok = (A['sps_med'] is not None and best['sps_med'] <= 0.85 and A['mem'] is not None and A['mem'] <= 14.5
      and A['nonfinite'] == 0 and (A['skipped'] or 0) <= 2 and A['swa'] and not A['abort'] and not A['tb'] and sv_ok
      and best['nonfinite'] == 0 and not best['abort'] and not best['tb'] and (best['mem'] or 99) <= 14.5)
ev = A['eval_s'] or 60
proj16 = (best['sps_med'] * ROWS * 16 + 16 * ev) / 3600 if best and best['sps_med'] else None
proj12 = (best['sps_med'] * ROWS * 12 + 12 * ev) / 3600 if best and best['sps_med'] else None
epochs = 16 if proj16 is not None and proj16 <= 26 else 12
out.append(f'best channels_last={cl} s/step {best["sps_med"] if best else None} | projection 16 ep {proj16 and round(proj16, 1)} h, '
           f'12 ep {proj12 and round(proj12, 1)} h -> EPOCHS {epochs} CHANNELS_LAST {cl}')
out.append('SMOKE_GO' if ok else 'SMOKE_NOGO')
open(f'{D}/verdict.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
