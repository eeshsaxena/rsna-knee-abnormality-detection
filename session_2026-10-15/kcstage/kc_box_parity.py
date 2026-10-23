#!/usr/bin/env python
# kc_box_parity.py (10-15, Claude subagent, kcstage): A6 G3 "the 3 visible studies' commit dump vs a box CPU run of the same ckpts:
# max |dp| <= 1e-3; the side matches the header rule". Box CPU only (cores 28-39, CUDA hidden), fp32, kc_common / kc_worker frozen
# code (md5 checked), the SAME member files (sha256 == the kernel's KC_MANIFEST.json).
# Input: the GPU commit output dir of an S6F+KC kernel (OUT/kc/dump/<sid>.npz, OUT/kc/kc_probs.csv, OUT/kc/kc_side_test.csv), copied
# to the box (RSNA public test studies only). Per dumped study, independently of the kernel's serve wrapper:
#   side, rule  = kc_common.side_rule(side row mean_x, tag)              == the dump's side / rule == kc_probs kc_side / kc_rule
#   graw        = s<k>_sign * s<k>_mag (sign != 0), geom, flips          == the dump's flips (= kc_build_store's derivation)
#   gok, serve  = kc_common.geom_ok on the vol's present blocks; both if weak or not gok or the side row says both  == dump serve
#   vol sha     = sha256(vol)[:16] == the dump's == kc_probs kc_vol_sha (the served input)
#   views       = kc_common.kc_cut(vol, side | L | R, geom) == the dumped views (byte-equal)
#   probs       = kc_worker.serve_probs(load_members(members, 'cpu'), ...) (fp32 CPU) vs kc_probs.csv (fp16 T4), labels BY NAME
# Writes a JSON the gate (v224s6fkc_check.py --box-parity) reads; prints KC_BOX_PARITY PASS/FAIL.
#   python kc_box_parity.py --out-dir OUT --code ~/kcstage_1015/code --members m1.pt m2.pt --manifest KC_MANIFEST.json --json P.json
import os, sys, json, time, glob, hashlib, argparse
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

PINS_MD5 = {'kc_worker.py': '4af60f6ff7b94b3acab9dceb52162a18', 'kc_common.py': '45255ae0e361ef787b290d7510554a9d',
            'fleet_model.py': '897f83b272fbb144a4d3deab773d9064', 'fleet_model_ttr.py': 'bcdf7d153f339962b97d2736d8baed08'}
SLOTK = {'sagfs': 0, 'sagpd': 1, 'corfs': 2, 'ax': 4}
TOL = 1e-3


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', required=True, help='the fetched kernel output dir (contains kc/dump, kc/kc_probs.csv)')
ap.add_argument('--code', required=True)
ap.add_argument('--members', nargs='+', required=True)
ap.add_argument('--manifest', required=True)
ap.add_argument('--json', required=True)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--per-member', action='store_true', help='also each member alone (info; doubles the CPU time)')
A = ap.parse_args()
for f, h in PINS_MD5.items():
    assert hashlib.md5(open(os.path.join(A.code, f), 'rb').read()).hexdigest() == h, f
sys.path.insert(0, A.code)
import torch
torch.set_num_threads(A.threads)
import kc_common as kc
import kc_worker as kw
man = json.load(open(A.manifest))
want = [m['file'] for m in man['members']]
got = {os.path.basename(p): sha256(p) for p in A.members}
assert [os.path.basename(p) for p in A.members] == want, ('member order / names != manifest', [os.path.basename(p) for p in A.members], want)
assert all(got[m] == man['files'][m] for m in want), ('member sha256 != manifest', got)
t0 = time.time()
ms = kw.load_members(A.members, 'cpu')
t_load = time.time() - t0
probs = pd.read_csv(os.path.join(A.out_dir, 'kc', 'kc_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
dumps = sorted(glob.glob(os.path.join(A.out_dir, 'kc', 'dump', '*.npz')))
L = list(kc.TARGETS)
rows, dp_all = [], []
for p in dumps:
    sid = os.path.basename(p)[:-4]
    t1 = time.time()
    z = np.load(p, allow_pickle=False)
    r = json.loads(str(z['side_row']))
    chk = {}
    side, rule = kc.side_rule(r.get('mean_x') if r.get('mean_x') is not None else float('nan'), r.get('tag') if isinstance(r.get('tag'), str) else '')
    chk['side_rule'] = (side, rule) == (str(z['side']), str(z['rule'])) == (str(r['side']), str(r['rule'])) == \
        (str(probs.at[sid, 'kc_side']), str(probs.at[sid, 'kc_rule']))
    graw = {}
    for b in kc.GEOM_DEFAULT:
        k = SLOTK.get(b)
        sg, mg = (r.get('s%d_sign' % k), r.get('s%d_mag' % k)) if k is not None else (None, None)
        graw[b] = float(sg) * float(mg) if (k is not None and sg is not None and mg is not None and float(sg) != 0) else float('nan')
    geom = {b: (graw[b] if np.isfinite(graw[b]) else kc.GEOM_DEFAULT[b]) for b in kc.GEOM_DEFAULT}
    flips = kc.block_flips(side, geom)
    chk['flips'] = json.loads(str(z['flips'])) == {b: bool(f) for b, f in flips.items()} and all(
        r.get('s%d_flip' % k) is None or bool(int(r['s%d_flip' % k])) == bool(flips[b]) for b, k in SLOTK.items())
    vol = z['vol']
    vsha = hashlib.sha256(vol.tobytes()).hexdigest()[:16]
    chk['vol_sha'] = vsha == str(z['vol_sha']) == str(z['served_vol_sha']) == str(probs.at[sid, 'kc_vol_sha'])
    pres = kc.present_blocks(vol)
    gok = kc.geom_ok(graw, [b for b in ('sagfs', 'sagpd', 'corfs', 'ax') if pres[b]])
    weak = rule != 'mean_x'
    serve = 'both' if (weak or not gok or str(r.get('serve')) == 'both') else 'single'
    chk['serve'] = serve == str(z['serve']) == str(probs.at[sid, 'kc_serve'])
    si = {'serve': serve, 'views': kc.kc_cut(vol, side, geom), 'views_L': kc.kc_cut(vol, 'L', geom), 'views_R': kc.kc_cut(vol, 'R', geom)}
    veq = True
    for nm in ('views', 'views_L', 'views_R'):
        for vn, st in si[nm].items():
            key = '%s__%s' % (nm, vn)
            veq &= (st is None and key not in z.files) or (st is not None and key in z.files and np.array_equal(z[key], st))
    chk['views'] = bool(veq)
    with torch.no_grad():
        pc = kw.serve_probs(ms, si, 'cpu')
        per_member = [kw.serve_probs([m], si, 'cpu').tolist() for m in ms] if A.per_member else None
    pk = probs.loc[sid, L].to_numpy(np.float64)
    pd_ = np.asarray(z['served'], np.float64)
    lab_dump = [str(x) for x in z['labels']]
    chk['dump_labels'] = lab_dump == L and np.array_equal(pd_, pk)
    dp = np.abs(pc.astype(np.float64) - pk)
    dp_all.append(dp)
    rows.append({'sid': sid, 'serve': serve, 'side': side, 'rule': rule, 'checks': chk, 'max_dp': float(dp.max()),
                 'dp_by_label': {l: float(d) for l, d in zip(L, dp)}, 'cpu': [float(x) for x in pc], 'gpu': [float(x) for x in pk],
                 'per_member_cpu': per_member, 'npz_sha256': sha256(p), 'seconds': round(time.time() - t1, 1)})
D = np.stack(dp_all) if dp_all else np.full((1, len(L)), np.inf)
res = {'n': len(rows), 'labels': L, 'max_dp': float(D.max()), 'max_dp_by_label': {l: float(D[:, j].max()) for j, l in enumerate(L)},
       'all_checks': bool(rows) and all(all(r['checks'].values()) for r in rows), 'tol': TOL,
       'PASS': bool(rows) and float(D.max()) <= TOL and all(all(r['checks'].values()) for r in rows),
       'members': {m: got[m] for m in want}, 'manifest_sha256': sha256(A.manifest), 'probs_sha256': sha256(os.path.join(A.out_dir, 'kc', 'kc_probs.csv')),
       'code_md5': PINS_MD5, 'torch': torch.__version__, 'threads': A.threads, 'load_s': round(t_load, 1),
       'device': 'cpu fp32', 'rows': rows}
json.dump(res, open(A.json, 'w'), indent=1)
print('KC_BOX_PARITY %s n %d max |dp| %.2e (label %s) | checks %s | %.0f s' % (
    'PASS' if res['PASS'] else 'FAIL', res['n'], res['max_dp'], max(res['max_dp_by_label'], key=res['max_dp_by_label'].get),
    'all ok' if res['all_checks'] else [r['checks'] for r in rows], time.time() - t0))
sys.exit(0 if res['PASS'] else 1)
