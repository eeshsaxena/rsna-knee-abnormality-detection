# offers_rank.py (rental_kc, laptop, read only): rank `vastai search offers --raw` JSON for the KC members. No em dashes.
# Hard filters (main 10-15): verified, NOT Taiwan / China / Hong Kong, reliability >= .98, exactly 2 x A100 (40 or 80 GB),
# dph_total <= --max_dph (1.50), CUDA >= 12.1, >= 300 Mbps down, >= 8 effective vCPU per trainer for 2 trainers (16), host not
# in --bad_hosts. MODEL (refined by the box f0 epoch time once known: --v100_h): one ConvNeXt-S KC256 all-data member =
# 5.1 h on the V100 (critique MODEL 4.9-5.3); A100 single-GPU speedup 2.0 mid / 1.5 worst (SXM4; PCIE x 0.9).
# Plan "2" = 1 member per GPU (the floor the driver can always fall back to); plan "4" = 2 rounds of 2 (or 2 per GPU).
# Run h = setup 0.5 (push ~13.3 GB at >= 20 MB/s, pip, parity) + smoke 0.33 + wait for the G1 gate (--wait_h) + members + fetch
# 0.3 + guard tail 0.25. Cost = $/h x run h + data (13.5 GB in x inet_down_cost + 18 GB out x inet_up_cost).
# fits = filters + plan-2 WORST cost <= --cap_left + plan-2 worst run <= --max_h.
# usage: python offers_rank.py FILE.json [...] [--cap_left 7.25] [--max_h 7] [--wait_h 0.75] [--json] [--id OFFER]
import argparse, json, sys
ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--cap_left', type=float, default=7.25)
ap.add_argument('--max_h', type=float, default=7.0); ap.add_argument('--max_dph', type=float, default=1.50)
ap.add_argument('--json', action='store_true'); ap.add_argument('--id', default=''); ap.add_argument('--top', type=int, default=25)
ap.add_argument('--v100_h', type=float, default=5.1); ap.add_argument('--wait_h', type=float, default=0.75)
ap.add_argument('--bad_hosts', default='513108')
A = ap.parse_args()
SPEED = {'A100 SXM4': 2.0, 'A100 PCIE': 1.8, 'A100X': 1.8}
BADGEO = ('China', 'Hong Kong', 'Taiwan')
BADCC = (', CN', ', HK', ', TW')
SETUP_H, SMOKE_H, FETCH_H, TAIL_H, DIN_GB, DOUT_GB = 0.5, 0.33, 0.3, 0.25, 13.5, 18
bad_hosts = {h.strip() for h in A.bad_hosts.split(',') if h.strip()}
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
        g = o.get('gpu_name', ''); n = int(o.get('num_gpus') or 0); geo = str(o.get('geolocation') or '')
        dph = float(o.get('dph_total') or 0); rel = float(o.get('reliability2') or o.get('reliability') or 0)
        vcpu = float(o.get('cpu_cores_effective') or 0); why = []
        if g not in SPEED or n != 2: why.append('gpu/plan')
        if o.get('verified') is False or o.get('verification') not in (None, 'verified'): why.append('unverified')
        if any(b in geo for b in BADGEO) or any(geo.endswith(c) for c in BADCC) or geo.strip() in ('CN', 'HK', 'TW'): why.append('TW/CN/HK')
        if str(o.get('host_id')) in bad_hosts: why.append('bad host')
        if rel < 0.98: why.append('rel')
        if dph > A.max_dph: why.append('$/h')
        if vcpu < 16: why.append('vCPU')
        if float(o.get('cuda_max_good') or 0) < 12.1: why.append('cuda')
        if float(o.get('gpu_ram') or 0) < 39000: why.append('vram')
        if float(o.get('inet_down') or 0) < 300: why.append('down')
        sp = SPEED.get(g, 1.0); mh = A.v100_h / sp; mw = A.v100_h / (sp * 0.75)
        base = SETUP_H + SMOKE_H + A.wait_h + FETCH_H + TAIL_H
        run_mid, run_w = base + mh, base + mw
        din = DIN_GB * float(o.get('inet_down_cost') or 0) + DOUT_GB * float(o.get('inet_up_cost') or 0)
        c_mid, c_w = dph * run_mid + din, dph * run_w + din
        c4 = dph * (base + 2 * mh) + din
        fits = not why and c_w <= A.cap_left and run_w <= A.max_h
        rows.append(dict(id=o['id'], machine=o.get('machine_id'), host=o.get('host_id'), gpu=g, n=n, dph=round(dph, 3),
                         per_gpu=round(dph / max(n, 1), 3), geo=geo, rel=round(rel, 3), vcpu=round(vcpu), down=round(o.get('inet_down') or 0),
                         dc=o.get('hosting_type') == 1, member_h=round(mh, 2), run_mid=round(run_mid, 2), run_w=round(run_w, 2),
                         c_mid=round(c_mid, 2), c_w=round(c_w, 2), c4_mid=round(c4, 2), din=round(din, 2), fits=fits, why=why,
                         member_est_s=int(mh * 3600)))
rows.sort(key=lambda r: (not r['fits'], r['c_w'], r['run_w']))
if A.id:
    rows = [r for r in rows if str(r['id']) == str(A.id)]
if A.json:
    print(json.dumps(rows[:A.top])); sys.exit(0)
print(f'{"id":>10} {"machine":>7} {"host":>7} {"gpu":10} {"$/h":>6} {"rel":>5} {"vcpu":>4} {"down":>5} {"mem_h":>5} {"run2 mid/worst":>14} {"$2 mid/worst":>12} {"$4 mid":>6} {"data$":>5} fits geo / reasons')
for r in rows[:A.top]:
    print(f'{r["id"]:>10} {str(r["machine"]):>7} {str(r["host"]):>7} {r["gpu"]:10} {r["dph"]:>6} {r["rel"]:>5} {r["vcpu"]:>4} {r["down"]:>5} {r["member_h"]:>5} '
          f'{str(r["run_mid"]) + "/" + str(r["run_w"]):>14} {str(r["c_mid"]) + "/" + str(r["c_w"]):>12} {r["c4_mid"]:>6} {r["din"]:>5} {"yes" if r["fits"] else "no ":3} '
          f'{r["geo"]} {",".join(r["why"])}')
print(f'\n{len(rows)} offers | fits = all filters + plan-2 worst cost <= ${A.cap_left:.2f} (cap left) + worst run <= {A.max_h} h | wait for the gate {A.wait_h} h')
