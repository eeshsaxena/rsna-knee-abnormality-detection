# offers_rank.py: rank `vastai search offers --raw` JSON for the vastR7T plan (TTR224 frac75 s1044 folds 1-4; laptop side, read only).
# usage: python offers_rank.py OFFERS.json [--folds 3] [--max_hours 7.0] [--budget 8.10] [--md]
# Speeds are MODEL: throughput vs OUR V100-FHHL (MEASURED 3.52 h per TTR224 gc0 fold, 21 epochs, 10.07 min/ep).
# RTX 3090 = 1.5-1.57 x V100 MEASURED 10-09 on the TTR288 recipe; others from public ConvNeXt-T AMP ratios.
# "worst" applies x0.8 for CPU/IO contention. Needs >= 8 vCPU per GPU (Eesh 10-09) and a verified host.
import json, math, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+'); ap.add_argument('--folds', type=int, default=3)
ap.add_argument('--max_hours', type=float, default=8.0)
ap.add_argument('--budget', type=float, default=2.15)     # spend cap minus the fetch/destroy reserve
ap.add_argument('--setup_h', type=float, default=0.45); ap.add_argument('--fetch_h', type=float, default=0.25)
ap.add_argument('--md', action='store_true'); ap.add_argument('--top', type=int, default=25)
ap.add_argument('--json', action='store_true'); ap.add_argument('--v100_fold_h', type=float, default=3.52); ap.add_argument('--id', type=int, default=0)
A = ap.parse_args()
V100_FOLD_H = A.v100_fold_h
# full-epoch speed incl. the ~1.9 min CPU-bound eval (V100 224: 8.2 train + 1.9 eval; 3090 288 MEASURED: eval ~2.1 min)
SPEED = {'RTX 3090': 1.47, 'RTX 3090 Ti': 1.55, 'RTX A5000': 1.2, 'RTX A6000': 1.45, 'A40': 1.35, 'RTX 4090': 2.1,
         'RTX 5090': 2.4, 'L40S': 2.0, 'L40': 1.8, 'RTX 6000Ada': 2.0, 'RTX 5880Ada': 1.9, 'RTX 5000Ada': 1.6,
         'RTX 4500Ada': 1.3, 'RTX 4000Ada': 0.95, 'A100 PCIE': 2.0, 'A100 SXM4': 2.2, 'A100X': 2.0, 'H100 PCIE': 2.4,
         'H100 SXM': 2.6, 'Tesla V100': 1.0, 'Q RTX 8000': 0.85, 'L4': 0.8, 'A10': 0.9}
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
        xfer = 16 * float(o.get('inet_down_cost') or 0) + 2 * float(o.get('inet_up_cost') or 0)   # GB in (store 11.7 + image ~3.5 + pip) / out
        ver = {'verified': 'V', 'unverified': 'U', 'deverified': 'D'}.get(o.get('verification'), '?')
        dc = 'DC' if o.get('hosting_type') == 1 else '--'
        vcpu = o.get('cpu_cores_effective') or 0
        need = 8 * n                                                # Eesh 10-09: >= 8 vCPU per GPU (loader ~5 cores at 3090 speed)
        ok_cpu = vcpu >= need
        rows.append(dict(id=o['id'], gpu=o['gpu_name'], n=n, dph=dph, ver=ver, dc=dc, geo=str(o.get('geolocation')),
                         vcpu=vcpu, ram=round((o.get('cpu_ram') or 0) / 1024), down=round(o.get('inet_down') or 0),
                         rel=round(o.get('reliability') or 0, 3), cuda=o.get('cuda_max_good'),
                         disk=round(o.get('disk_space') or 0), rounds=rounds, fold_h=fh, h_mid=h_mid, h_worst=h_worst,
                         c_mid=dph * h_mid + xfer, c_worst=dph * h_worst + xfer, xfer=xfer, ok_cpu=ok_cpu, ram_ok=(o.get('cpu_ram') or 0) >= 48000,
                         fold_s_worst=int(fw * 3600), cuda_ok=float(o.get('cuda_max_good') or 0) >= 12.1,
                         fits=(h_worst <= A.max_hours and dph * h_worst + xfer <= A.budget and ok_cpu and ver == 'V' and (o.get('gpu_ram') or 0) >= 20000)))
# rank: fits first, then verified datacenter, verified, others; then worst-case cost
pref = lambda r: (not r['fits'], not r['ok_cpu'], {('V', 'DC'): 0, ('V', '--'): 1}.get((r['ver'], r['dc']), 3) if r['c_mid'] <= A.budget else 5, r['c_mid'])
rows.sort(key=pref)
if A.id:
    rows = [r for r in rows if r['id'] == A.id]
if A.json:
    print(json.dumps(rows)); sys.exit(0)
hdr = ['offer', 'gpu', 'n', '$/h', 'ver', 'dc', 'where', 'vCPU', 'RAM GB', 'down Mb/s', 'rel', 'cuda', 'rounds',
       'fold h (MODEL)', 'run h mid/worst', '$ mid/worst', 'data-in $', 'cpu ok', 'fits']
fmt = lambda r: [str(r['id']), r['gpu'], str(r['n']), f"{r['dph']:.3f}", r['ver'], r['dc'], r['geo'][:22],
                 f"{r['vcpu']:.0f}", str(r['ram']), str(r['down']), f"{r['rel']:.3f}", str(r['cuda']), str(r['rounds']),
                 f"{r['fold_h']:.2f}", f"{r['h_mid']:.1f}/{r['h_worst']:.1f}", f"{r['c_mid']:.2f}/{r['c_worst']:.2f}",
                 f"{r['xfer']:.2f}", 'yes' if r['ok_cpu'] else 'NO', 'yes' if r['fits'] else 'no']
if A.md:
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for r in rows[:A.top]:
        print('| ' + ' | '.join(fmt(r)) + ' |')
else:
    print('  '.join(hdr))
    for r in rows[:A.top]:
        print('  '.join(fmt(r)))
print(f"\n{len(rows)} offers ranked | plan {A.folds} folds | fits = worst-case run <= {A.max_hours} h AND worst-case "
      f"cost (incl. data-in) <= ${A.budget:.2f} AND vCPU >= 8 per GPU AND verified AND >= 20 GB | DC=datacenter")
