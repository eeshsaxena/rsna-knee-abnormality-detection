# offers_rank.py: rank `vastai search offers --raw` JSON for the vastL5 plan (laptop side, read only).
# usage: python offers_rank.py OFFERS.json [--folds 3] [--max_hours 7.0] [--budget 8.10] [--md]
# L5 kit: RTX 3090 speed = 1.57 x V100 MEASURED 10-09 (5.8 h box fold vs 3.7 h on 3090, 16 slots); others MODEL.
# Speeds are MODEL: throughput vs OUR V100-FHHL (measured 7.3 h per TTR320 fold, 21 epochs, 0.27 s/it),
# from public ConvNeXt-T AMP training throughput ratios; "worst" applies x0.8 for CPU/IO contention.
import json, math, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+'); ap.add_argument('--folds', type=int, default=3)
ap.add_argument('--max_hours', type=float, default=8.5)   # L5: one 24-slot fold; must fit the watcher time cap (budget-bound)
ap.add_argument('--budget', type=float, default=8.10)     # spend cap minus the fetch/destroy reserve
ap.add_argument('--setup_h', type=float, default=0.6); ap.add_argument('--fetch_h', type=float, default=0.3)
ap.add_argument('--md', action='store_true'); ap.add_argument('--top', type=int, default=25)
ap.add_argument('--json', action='store_true'); ap.add_argument('--v100_fold_h', type=float, default=5.8); ap.add_argument('--id', type=int, default=0)
A = ap.parse_args()
V100_FOLD_H = A.v100_fold_h
SPEED = {'RTX 4090': 3.0, 'L40S': 2.9, 'RTX 6000Ada': 2.8, 'L40': 2.5, 'A100 PCIE': 3.0, 'A100 SXM4': 3.3,
         'A100X': 3.0, 'RTX 3090': 1.57, 'RTX A6000': 1.75, 'RTX A5000': 1.4}
rows, seen = [], set()
for f in A.files:
    t = open(f, encoding='utf-8', errors='replace').read()
    i = min([k for k in (t.find('['), t.find('{')) if k >= 0], default=-1)
    if i < 0:
        continue
    d = json.loads(t[i:])
    for o in (d if isinstance(d, list) else d.get('offers', [])):
        if o['id'] in seen:
            continue
        seen.add(o['id'])
        sp = SPEED.get(o['gpu_name'])
        if not sp:
            continue
        n = int(o['num_gpus']); rounds = math.ceil(A.folds / n)
        fh, fw = V100_FOLD_H / sp, V100_FOLD_H / (sp * 0.8)
        h_mid = A.setup_h + rounds * fh + A.fetch_h; h_worst = A.setup_h + rounds * fw + A.fetch_h
        dph = float(o['dph_total'])
        xfer = 12 * float(o.get('inet_down_cost') or 0) + 2 * float(o.get('inet_up_cost') or 0)   # GB in / out
        ver = {'verified': 'V', 'unverified': 'U', 'deverified': 'D'}.get(o.get('verification'), '?')
        dc = 'DC' if o.get('hosting_type') == 1 else '--'
        vcpu = o.get('cpu_cores_effective') or 0
        need = (12 if sp >= 2.5 else 8) * min(n, A.folds)          # heavy-aug loader: ~0.46 Xeon-4114 core-s per study
        ok_cpu = vcpu == 0 or vcpu >= need                          # 0 = host did not report
        rows.append(dict(id=o['id'], gpu=o['gpu_name'], n=n, dph=dph, ver=ver, dc=dc, geo=str(o.get('geolocation')),
                         vcpu=vcpu, ram=round((o.get('cpu_ram') or 0) / 1024), down=round(o.get('inet_down') or 0),
                         rel=round(o.get('reliability') or 0, 3), cuda=o.get('cuda_max_good'),
                         disk=round(o.get('disk_space') or 0), rounds=rounds, fold_h=fh, h_mid=h_mid, h_worst=h_worst,
                         c_mid=dph * h_mid + xfer, c_worst=dph * h_worst + xfer, xfer=xfer, ok_cpu=ok_cpu, ram_ok=(o.get('cpu_ram') or 0) >= 48000,
                         fold_s_worst=int(fw * 3600), cuda_ok=float(o.get('cuda_max_good') or 0) >= 12.1,
                         fits=(h_worst <= A.max_hours and dph * h_worst + xfer <= A.budget and ok_cpu)))
# rank: fits first, then verified datacenter, verified, others; then worst-case cost
pref = lambda r: (not r['fits'], {('V', 'DC'): 0, ('V', '--'): 1, ('D', 'DC'): 2}.get((r['ver'], r['dc']), 3), r['c_worst'])
rows.sort(key=pref)
if A.id:
    rows = [r for r in rows if r['id'] == A.id]
if A.json:
    print(json.dumps(rows)); sys.exit(0)
hdr = ['offer', 'gpu', 'n', '$/h', 'ver', 'dc', 'where', 'vCPU', 'RAM GB', 'down Mb/s', 'rel', 'cuda', 'rounds',
       'fold h (MODEL)', 'run h mid/worst', '$ mid/worst', 'fits']
fmt = lambda r: [str(r['id']), r['gpu'], str(r['n']), f"{r['dph']:.3f}", r['ver'], r['dc'], r['geo'][:22],
                 f"{r['vcpu']:.0f}", str(r['ram']), str(r['down']), f"{r['rel']:.3f}", str(r['cuda']), str(r['rounds']),
                 f"{r['fold_h']:.2f}", f"{r['h_mid']:.1f}/{r['h_worst']:.1f}", f"{r['c_mid']:.2f}/{r['c_worst']:.2f}",
                 'yes' if r['fits'] else 'no']
if A.md:
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for r in rows[:A.top]:
        print('| ' + ' | '.join(fmt(r)) + ' |')
else:
    print('  '.join(hdr))
    for r in rows[:A.top]:
        print('  '.join(fmt(r)))
print(f"\n{len(rows)} offers ranked | plan {A.folds} folds | fits = worst-case run <= {A.max_hours} h AND worst-case "
      f"cost <= ${A.budget:.2f} AND vCPU >= 12 per busy 4090/A100/L40-class GPU (8 for 3090/A-series) | ver V=verified D=deverified U=unverified | DC=datacenter")
