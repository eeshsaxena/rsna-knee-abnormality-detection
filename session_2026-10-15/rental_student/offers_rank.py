# offers_rank.py (laptop, read only): rank `vastai search offers --raw` JSON for the T_b student (5 TTR224 folds, 21 ep).
# Hard filters (workflow rules 10-14): verified, NOT China / Hong Kong, reliability >= .98, plan GPUs = 4 x RTX 4090 or
# 2 x A100, <= $0.55 per GPU-hour (dph_total incl. the priced storage / n), CUDA >= 12.1, >= 23 GB per GPU, >= 300 Mbps
# down, and >= 8 effective vCPU per CONCURRENT trainer (5 for 4 GPUs: the smoke pairs f4; 4 for 2 GPUs): every trainer runs
# 10 loader workers and the recipe is loader-heavy (10-10 3090 rental, 12.8 vCPU: TRAIN-ONLY 5.7-8.8 min/ep vs V100 8.2,
# GPU util swinging 0-100 %, MEASURED).
# MODEL: V100 3.52 h per fold (MEASURED, 10.07 min/ep). Speed vs V100, conservative because the loader caps it: 4090 1.75,
# A100 1.75 (the 3090 reached 1.44 x MEASURED; 4090 / A100 compute alone would be ~2.1-2.2 x); worst = mid / 1.25. Plan: 4 GPUs -> 2 rounds (4, then 1; the smoke may keep f4 paired = 1 round); 2 GPUs -> 2 per GPU then the
# 5th = T(2) + T(1) with T(2) = 1.6 x T(1) (GUESS pairing gain). Setup 0.45 h (push 11.7 GB at >= 20 MB/s, pip, parity),
# fetch tail 0.3 h. Cost = $/h x run h + data transfer (16 GB in x inet_down_cost + 24 GB out x inet_up_cost: the box pulls
# 5 folds x ~3.9 GB incl. epoch ckpts + logs + a re-pull margin; review 10-14, was data-in only).
# usage: python offers_rank.py FILE.json [FILE2.json] [--cap_left 15] [--max_h 7.0] [--json] [--id OFFER]
import argparse, json, math, sys
ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--cap_left', type=float, default=15.0)
ap.add_argument('--max_h', type=float, default=7.0); ap.add_argument('--per_gpu_max', type=float, default=0.55)
ap.add_argument('--json', action='store_true'); ap.add_argument('--id', default=''); ap.add_argument('--top', type=int, default=25)
ap.add_argument('--v100_fold_h', type=float, default=3.52)
ap.add_argument('--nfolds', type=int, default=5)   # 10-14 21:50: RSTUD_FOLDS count (plan A = 4: f0-f3, f4 on the box)
A = ap.parse_args()
SPEED = {'RTX 4090': 1.75, 'A100 SXM4': 1.75, 'A100 PCIE': 1.75, 'A100X': 1.75}
PLAN = {('RTX 4090', 4), ('A100 SXM4', 2), ('A100 PCIE', 2), ('A100X', 2)}
BADGEO = ('China', ', CN', 'Hong Kong', ', HK')
SETUP_H, FETCH_H, DIN_GB, DOUT_GB = 0.45, 0.3, 16, 24
rows, seen = [], set()
for f in A.files:
    t = open(f, encoding='utf-8', errors='replace').read()
    i = min([k for k in (t.find('['), t.find('{')) if k >= 0], default=-1)
    if i < 0: continue
    d = json.loads(t[i:])
    for o in (d if isinstance(d, list) else d.get('offers', [])):
        if o['id'] in seen: continue
        seen.add(o['id'])
        g = o.get('gpu_name', ''); n = int(o.get('num_gpus') or 0); geo = str(o.get('geolocation') or '')
        dph = float(o.get('dph_total') or 0); rel = float(o.get('reliability2') or o.get('reliability') or 0)
        vcpu = float(o.get('cpu_cores_effective') or 0); why = []
        if (g, n) not in PLAN: why.append('gpu/plan')
        if o.get('verified') is False or o.get('verification') not in (None, 'verified'): why.append('unverified')
        if any(b in geo or geo.endswith(b.strip(', ')) for b in BADGEO) or geo.strip() in ('CN', 'HK'): why.append('CN/HK')
        if rel < 0.98: why.append('rel')
        if n and dph / n > A.per_gpu_max: why.append('$/GPU-h')
        if n and vcpu / (5 if n >= 4 else 4) < 8: why.append('vCPU/trainer')
        if float(o.get('cuda_max_good') or 0) < 12.1: why.append('cuda')
        if float(o.get('gpu_ram') or 0) < 23000: why.append('vram')
        if float(o.get('inet_down') or 0) < 300: why.append('down')
        sp = SPEED.get(g, 1.0); fh = A.v100_fold_h / sp; fw = fh * 1.25
        if n >= 4: run_mid, run_w = SETUP_H + 2 * fh + FETCH_H, SETUP_H + 2 * fw + FETCH_H
        elif A.nfolds <= 2 * n:   # 2 GPUs, <= 2 folds per GPU = ONE round of 2 per GPU: T(2) = 2.25 x the MODEL fold
            # (MEASURED 10-14 19:40, 2 x A100 SXM4, 2 trainers per GPU: 3.9-4.5 h per fold = 1.94-2.24 x 2.01 h)
            run_mid, run_w = SETUP_H + 2.25 * fh + FETCH_H, SETUP_H + 2.25 * fw + FETCH_H
        else: run_mid, run_w = SETUP_H + 2.6 * fh + FETCH_H, SETUP_H + 2.6 * fw + FETCH_H
        din = DIN_GB * float(o.get('inet_down_cost') or 0) + DOUT_GB * float(o.get('inet_up_cost') or 0)   # in + out
        c_mid, c_w = dph * run_mid + din, dph * run_w + din
        fits = not why and c_w <= A.cap_left and run_w <= A.max_h
        rows.append(dict(id=o['id'], gpu=g, n=n, dph=round(dph, 3), per_gpu=round(dph / max(n, 1), 3), geo=geo, rel=round(rel, 3),
                         vcpu=round(vcpu), ram=round((o.get('cpu_ram') or 0) / 1024), down=round(o.get('inet_down') or 0),
                         dc=o.get('hosting_type') == 1, cuda=o.get('cuda_max_good'), fold_h=round(fh, 2), run_mid=round(run_mid, 2),
                         run_w=round(run_w, 2), c_mid=round(c_mid, 2), c_w=round(c_w, 2), din=round(din, 2), fits=fits, why=why,
                         fold_est_s=int(fh * 3600)))
rows.sort(key=lambda r: (not r['fits'], r['c_w'], r['run_w']))
if A.id: rows = [r for r in rows if str(r['id']) == str(A.id)]
if A.json: print(json.dumps(rows[:A.top])); sys.exit(0)
print(f'{"id":>10} {"gpu":10} {"n":>1} {"$/h":>6} {"$/GPU":>6} {"rel":>5} {"vcpu":>4} {"down":>5} {"fold_h":>6} {"run_h mid/worst":>15} {"$ mid/worst":>12} {"din$":>5} fits dc geo / reasons')
for r in rows[:A.top]:
    print(f'{r["id"]:>10} {r["gpu"]:10} {r["n"]:>1} {r["dph"]:>6} {r["per_gpu"]:>6} {r["rel"]:>5} {r["vcpu"]:>4} {r["down"]:>5} {r["fold_h"]:>6} '
          f'{str(r["run_mid"]) + "/" + str(r["run_w"]):>15} {str(r["c_mid"]) + "/" + str(r["c_w"]):>12} {r["din"]:>5} {"yes" if r["fits"] else "no ":3} '
          f'{"DC" if r["dc"] else "  "} {r["geo"]} {",".join(r["why"])}')
print(f'\n{len(rows)} offers | fits = all filters + worst-case cost <= ${A.cap_left:.2f} (cap left) + worst run <= {A.max_h} h')
