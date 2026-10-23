# offers_rank.py: rank Vast offers for the RSNA-only Raptor seeds (path b). Reads `vastai search offers --raw` JSON files.
# Hard filters (main 10-13): verified, NOT China-located, reliability >= .98, >= 8 effective vCPU per GPU, inet_down >= 500
# Mbps, gpu_ram >= 24 GB, cuda >= 12.1. MODEL: s/step = V100 measured s/step / speed ratio (GUESS ratios; 3090 1.95x MEASURED
# 10-09 on ConvNeXt), 4,349 studies x 16 epochs per seed, seeds spread over min(n_gpu, n_seeds) parallel lanes.
# usage: python offers_rank.py FILE.json [FILE2.json] --v100_sps 0.65 --seeds 3 --budget 20 --max_h 12 [--json --id OFFER]
import sys, json, math, argparse
ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--v100_sps', type=float, default=0.65)
ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--budget', type=float, default=20.0); ap.add_argument('--max_h', type=float, default=12.0)
ap.add_argument('--json', action='store_true'); ap.add_argument('--id', default=''); ap.add_argument('--top', type=int, default=25)
A = ap.parse_args()
RATIO = {'RTX 4090': 2.4, 'RTX 3090': 1.95, 'A100 SXM4': 2.8, 'A100 PCIE': 2.6, 'A100X': 2.6, 'H100 SXM': 5.0, 'H100 PCIE': 4.0,
         'H100 NVL': 4.2, 'L40S': 2.3, 'L40': 2.0, 'RTX 6000Ada': 2.3, 'RTX A6000': 1.6, 'RTX 5090': 3.0}
STEPS = 4349 * 16; EVAL_H = 16 * 25 / 3600; SETUP_H = 0.75; FETCH_H = 0.35; DATA_GB = 40
rows = []
for f in A.files:
    d = json.load(open(f)); d = d if isinstance(d, list) else d.get('offers', [])
    for o in d:
        g = o.get('gpu_name', ''); n = int(o.get('num_gpus') or 1); geo = str(o.get('geolocation') or '')
        r = RATIO.get(g)
        ok = (r is not None and o.get('verified') is not False and 'CN' not in geo and 'China' not in geo and (o.get('reliability2') or o.get('reliability') or 0) >= .98
              and (o.get('cpu_cores_effective') or 0) / n >= 8 and (o.get('inet_down') or 0) >= 500 and (o.get('gpu_ram') or 0) >= 23000
              and float(o.get('cuda_max_good') or 0) >= 12.1)
        if not ok: continue
        sps = A.v100_sps / r; seed_h = STEPS * sps / 3600 + EVAL_H
        lanes = min(n, A.seeds); rounds = math.ceil(A.seeds / lanes)
        run_h = SETUP_H + rounds * seed_h + FETCH_H
        dph = float(o.get('dph_total') or 0); din = float(o.get('inet_down_cost') or 0) * DATA_GB
        cost = dph * run_h + din
        rows.append(dict(id=o['id'], gpu=g, n=n, dph=round(dph, 3), geo=geo, rel=round(o.get('reliability2') or o.get('reliability') or 0, 3),
                         vcpu=o.get('cpu_cores_effective'), ram=round((o.get('cpu_ram') or 0) / 1024), down=int(o.get('inet_down') or 0),
                         din=round(din, 2), sps=round(sps, 3), seed_h=round(seed_h, 2), rounds=rounds, run_h=round(run_h, 2),
                         cost=round(cost, 2), fits=cost <= A.budget and run_h <= A.max_h, dc=bool(o.get('hosting_type') == 1)))
rows.sort(key=lambda r: (not r['fits'], r['cost']))
if A.id: rows = [r for r in rows if str(r['id']) == str(A.id)]
if A.json: print(json.dumps(rows[:A.top])); sys.exit(0)
print(f'{"id":>10} {"gpu":12} {"n":>2} {"$/h":>6} {"rel":>5} {"vcpu":>5} {"ram":>4} {"down":>5} {"din$":>5} {"s/step":>6} {"seed_h":>6} {"rnd":>3} {"run_h":>5} {"cost$":>6} fits dc geo')
for r in rows[:A.top]:
    print(f'{r["id"]:>10} {r["gpu"]:12} {r["n"]:>2} {r["dph"]:>6} {r["rel"]:>5} {r["vcpu"]:>5} {r["ram"]:>4} {r["down"]:>5} {r["din"]:>5} {r["sps"]:>6} '
          f'{r["seed_h"]:>6} {r["rounds"]:>3} {r["run_h"]:>5} {r["cost"]:>6} {"yes" if r["fits"] else "no ":3} {"DC" if r["dc"] else "  "} {r["geo"]}')
