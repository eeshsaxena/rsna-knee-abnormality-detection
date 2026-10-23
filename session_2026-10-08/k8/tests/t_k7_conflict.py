"""t_k7_conflict.py (laptop, CPU; TEST ONLY; 2026-10-06, Claude): the K7-KEY fix for the set B / set C collision
(both fold 0 seed 1044). Run AFTER tests/smoke_k2.py (it reuses its _smoke assets: code64 = k7/code with STORE 64, the
24-study comp_big, the round3 dry family-1 member). Writes only under tests/_k7c.

W  the SHIPPED worker (k7/code/fleet_worker.py == the K5 worker, unchanged) on two tiny REAL TTR members built from the
   same weights: B0 = f0_s1044 TTR ttr_size 48, C0 = f0_s1044 TTR ttr_size 56 (the TTR288 analogue), both A+B:
   W1 pack_fleet_ds.py re-keys C0 to f0_s1044t56 (P-TAG) and the worker serves BOTH: 2 x [KneeFleetNetTTR], every row
      finite, the two rows differ (each member built at its own args ttr_size)
   W2 each member's served rows (P, PA, PB) == the rows of a dataset holding that member ALONE (max|diff| 0.0)
   W3 refusals: pack without the re-key (duplicate member keys); pack of the same file twice even when re-keyed (P-RUN);
      worker on a manifest naming B0 twice; worker on B0 + C0 under its original name (same f0_s1044 key);
      worker on C0 whose args ttr_size was edited to 48 (buffer 56): 'ttr_size buffer != args'
D  build_k2.py preflight (the K7 run key incl. ttr_size) and build_k7.load_members on the REAL checkpoints in k7/_sim/ckpt
   (A f0 s1043, the simulated B f0 s1044, the real TTR288 C f0 s1044):
   D1 B f0 + C f0 re-keyed (PATH=f0_s1044t288) -> 2 rows, staged f0_s1044t288_raw_ttr288-...
   D2 B f0 + C f0 without the re-key -> refused (duplicate member key f0_s1044)
   D3 the same B f0 file twice (second re-keyed) -> refused (one run)
   D4 build_k7.load_members: a members.json listing B f0 twice -> refused; the sim members.json -> C staged as
      f0_s1044t288_..., every other member keeps its file name (byte-identical names to K5/K6)
   D5 old-kit outcome unchanged: the K5 set A alone -> 5 rows, keys f{F}_s1043, no re-key
"""
import glob, json, os, shutil, subprocess, sys
import numpy as np

T = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K7 = os.path.dirname(T); SP = os.path.dirname(K7)
S = f'{T}/_smoke'
O = f'{T}/_k7c'
DRY = f'{SP}/round3/fleet_dry'
PY = sys.executable
ENV = {**os.environ, 'PYTHONWARNINGS': 'ignore', 'PYTHONIOENCODING': 'utf-8', 'CUDA_VISIBLE_DEVICES': '',
       'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
R = []


def ok(name, cond, extra=''):
    R.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"} {name} {extra}', flush=True)


def run(args, env=None):
    return subprocess.run([PY, '-W', 'ignore', *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                          env=env or ENV, timeout=3600)


MAKE = r'''import sys, hashlib, inspect, torch
code, src, dst, ttr, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, code)
import fleet_model as fm, fleet_model_ttr as tt
ck = torch.load(src, map_location='cpu', weights_only=False); a = dict(ck['args'])
kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'], n_contrast=a['n_contrast'],
          contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
m = tt.KneeFleetNetTTR(ttr_size=ttr, **kw)
miss, unexp = m.load_state_dict(ck['model'], strict=False)
assert list(miss) == ['ttr_size_t'] and not unexp
sha = hashlib.sha256((chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) +
                      chr(10).join(inspect.getsource(fm).splitlines())).encode()).hexdigest()[:16]
a.update(seed=seed, model_mod='fleet_model_ttr', ttr_size=ttr, model_sha=sha)
ck2 = dict(ck); ck2['args'] = a; ck2['model'] = m.state_dict(); torch.save(ck2, dst)
print('made', dst.split('/')[-1], 'sha', sha, 'ttr', int(ck2['model']['ttr_size_t']))
'''
EDIT = r'''import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
ck['args'] = dict(ck['args'], ttr_size=int(sys.argv[3])); torch.save(ck, sys.argv[2])
'''


def worker(ds, tag):
    e = dict(ENV, FLEET_ALLOW_CPU='1', CUDA_VISIBLE_DEVICES='-1', FLEET_SHARD='0', FLEET_NSHARD='1', PYTHONPATH=f'{DRY}/shim',
             FLEET_DIR=ds, FLEET_OUT=f'{O}/w_{tag}.npz', RSNA_COMP_ROOT=f'{S}/comp_big', FLEET_MICROBENCH='0')
    r = run([f'{ds}/fleet_worker.py'], env=e)
    open(f'{O}/log_w_{tag}.txt', 'w', encoding='utf-8').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
    return r, (np.load(f'{O}/w_{tag}.npz') if r.returncode == 0 else None)


def pack(out, members, tag=None, tta=None):
    return run([f'{S}/code64/pack_fleet_ds.py', '--out', out, '--store', f'{DRY}/store_64_normal']
               + (['--tag', tag] if tag else []) + (['--tta', tta] if tta else []) + members)


def part_w():
    assert os.path.isfile(f'{S}/code64/fleet_worker.py'), 'run tests/smoke_k2.py first (needs _smoke/code64)'
    import hashlib
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    ok('W0 code64 worker == k7/code worker == k5/code worker (unchanged)',
       h(f'{S}/code64/fleet_worker.py') == h(f'{K7}/code/fleet_worker.py') == h(f'{SP}/k5/code/fleet_worker.py'))
    open(f'{O}/make.py', 'w').write(MAKE); open(f'{O}/edit.py', 'w').write(EDIT)
    f1 = f'{DRY}/fleetds/f0_s42_ema_cnxt64x24_cv5714_best.pt'
    os.makedirs(f'{O}/mem')
    b0 = f'{O}/mem/f0_s1044_raw_ttr48-64x24_cv5714_best.pt'
    c0 = f'{O}/mem/f0_s1044_raw_ttr56-64x24_cv5714_best.pt'
    for p, t in ((b0, 48), (c0, 56)):
        r = run([f'{O}/make.py', f'{S}/code64', f1, p, str(t), '1044']); assert r.returncode == 0, r.stderr[-800:]
        print(r.stdout.strip())
    r = pack(f'{O}/ds_bc', [b0, c0], tag='=,f0_s1044t56', tta='AB,AB')
    names = json.load(open(f'{O}/ds_bc/fleet_manifest.json'))['checkpoints'] if r.returncode == 0 else []
    cn = 'f0_s1044t56_raw_ttr56-64x24_cv5714_best.pt'
    ok('W1a pack re-keys C0 (P-TAG)', r.returncode == 0 and names == [os.path.basename(b0), cn], f'{names} {r.stderr[-300:]}')
    rw, z = worker(f'{O}/ds_bc', 'bc')
    good = rw.returncode == 0 and z is not None
    if good:
        P = z['P']; nm = [str(x) for x in z['names']]
        good = (nm == [os.path.basename(b0), cn] and np.isfinite(P).all() and rw.stdout.count('[KneeFleetNetTTR]') == 2
                and float(np.abs(P[0] - P[1]).max()) > 1e-4)
    ok('W1b worker serves B0 and C0 (same fold+seed, distinct keys): 2 TTR members, all rows finite, rows differ', good,
       '' if rw.returncode == 0 else rw.stderr[-400:])
    rb = pack(f'{O}/ds_b', [b0], tta='AB'); rc = pack(f'{O}/ds_c', [c0], tag='f0_s1044t56', tta='AB')
    _, zb = worker(f'{O}/ds_b', 'b'); _, zc = worker(f'{O}/ds_c', 'c')
    d = []
    for k, zz in ((0, zb), (1, zc)):
        for key in ('P', 'PA', 'PB'):
            d.append(float(np.nanmax(np.abs(z[key][k] - zz[key][0]))))
    ok('W2 each member served exactly as when alone (P, PA, PB)', rb.returncode == rc.returncode == 0 and max(d) == 0.0, f'max|diff| {max(d)}')
    r = pack(f'{O}/ds_x1', [b0, c0], tta='AB,AB')
    ok('W3a pack refuses B0 + C0 without the re-key', r.returncode != 0 and 'duplicate member keys' in r.stderr, r.stderr.strip()[-100:])
    shutil.copy(b0, f'{O}/mem2_b0copy.pt'); os.makedirs(f'{O}/mem2'); shutil.move(f'{O}/mem2_b0copy.pt', f'{O}/mem2/{os.path.basename(b0)}')
    r = pack(f'{O}/ds_x2', [b0, f'{O}/mem2/{os.path.basename(b0)}'], tag='=,f0_s1044x', tta='AB,AB')
    ok('W3b pack refuses the same run twice even when re-keyed (P-RUN)', r.returncode != 0 and 'same run' in r.stderr, r.stderr.strip()[-100:])
    shutil.copytree(f'{O}/ds_b', f'{O}/ds_x3')
    m = json.load(open(f'{O}/ds_x3/fleet_manifest.json')); m['checkpoints'] = m['checkpoints'] * 2
    json.dump(m, open(f'{O}/ds_x3/fleet_manifest.json', 'w'))
    r, _ = worker(f'{O}/ds_x3', 'x3')
    ok('W3c worker refuses a manifest naming B0 twice', r.returncode != 0 and 'duplicate fold/seed members' in r.stderr, r.stderr.strip()[-120:])
    shutil.copytree(f'{O}/ds_bc', f'{O}/ds_x4')
    os.rename(f'{O}/ds_x4/{cn}', f'{O}/ds_x4/{os.path.basename(c0)}')
    m = json.load(open(f'{O}/ds_x4/fleet_manifest.json')); m['checkpoints'] = [os.path.basename(b0), os.path.basename(c0)]
    m['sources'] = {os.path.basename(b0): m['sources'][os.path.basename(b0)], os.path.basename(c0): m['sources'][cn]}
    json.dump(m, open(f'{O}/ds_x4/fleet_manifest.json', 'w'))
    r, _ = worker(f'{O}/ds_x4', 'x4')
    ok('W3d worker refuses B0 + C0 under the same f0_s1044 key (no re-key)', r.returncode != 0 and 'duplicate fold/seed members' in r.stderr, r.stderr.strip()[-120:])
    shutil.copytree(f'{O}/ds_c', f'{O}/ds_x5')
    r0 = run([f'{O}/edit.py', f'{O}/ds_c/{cn}', f'{O}/ds_x5/{cn}', '48'])
    r, _ = worker(f'{O}/ds_x5', 'x5')
    ok('W3e worker refuses C0 whose args ttr_size (48) != its buffer (56)', r0.returncode == 0 and r.returncode != 0
       and 'ttr_size buffer != args' in r.stderr, r.stderr.strip()[-120:])


def part_d():
    ck = f'{K7}/_sim/ckpt'
    a0 = f'{ck}/f0_s1043_raw_ttr224-320x24_cv8835_best.pt'
    b0 = f'{ck}/f0_s1044_raw_ttr224-320x24_cv8835_best.pt'
    (c0,) = glob.glob(f'{ck}/f0_s1044_*_ttr288-*_best.pt')
    store = f'{K7}/_sim/_real/store'
    sys.path.insert(0, K7)
    import build_k2 as bk, build_k7 as b7
    rows = bk.preflight([(b0, None), (c0, 'f0_s1044t288')], store)
    ok('D1 build_k2 preflight B f0 + C f0 (re-keyed) -> 2 rows', [r['staged'] for r in rows] ==
       [os.path.basename(b0), 'f0_s1044t288' + os.path.basename(c0)[len('f0_s1044'):]] and [r['ttr_size'] for r in rows] == [224, 288],
       [r['staged'] for r in rows])
    for name, specs, want in (('D2 B f0 + C f0 without the re-key -> refused', [(b0, None), (c0, None)], 'duplicate member key f0_s1044'),
                              ('D3 the same B f0 file twice (second re-keyed) -> refused', [(b0, None), (b0, 'f0_s1044x')], 'one run')):
        try:
            bk.preflight(specs, store); ok(name, False, 'not refused')
        except AssertionError as e:
            ok(name, want in str(e), str(e)[:120])
    m = json.load(open(f'{K7}/_sim/members.json'))
    mm = dict(m, members=m['members'] + [m['members'][5]])
    json.dump(mm, open(f'{O}/members_dup.json', 'w'))
    try:
        b7.load_members(f'{O}/members_dup.json', ck); ok('D4a load_members refuses B f0 listed twice', False)
    except AssertionError as e:
        ok('D4a load_members refuses B f0 listed twice', 'duplicate' in str(e), str(e)[:100])
    _, mem = b7.load_members(f'{K7}/_sim/members.json', ck)
    st = [(r['set'], r['staged'] == r['file']) for r in mem]
    ok('D4b sim members: only C re-keyed, A/B names unchanged', all(same for s, same in st if s != 'C') and
       [r['staged'] for r in mem if r['set'] == 'C'] == ['f0_s1044t288' + os.path.basename(c0)[len('f0_s1044'):]], st)
    a_all = sorted(glob.glob(f'{ck}/f*_s1043_*_best.pt'))
    rows = bk.preflight([(p, None) for p in a_all], store)
    ok('D5 the K5 set A alone -> 5 rows, keys f{F}_s1043, names unchanged',
       [r['key'] for r in rows] == [f'f{F}_s1043' for F in range(5)] and all(r['staged'] == os.path.basename(r['path']) for r in rows))


if __name__ == '__main__':
    shutil.rmtree(O, ignore_errors=True); os.makedirs(O)
    part_w()
    part_d()
    n = sum(R)
    print(f'TK7CONFLICT {"PASS" if n == len(R) else "FAIL"} {n}/{len(R)}')
    sys.exit(0 if n == len(R) else 1)
