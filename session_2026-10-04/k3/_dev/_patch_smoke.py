import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p = 'tests/smoke_k2.py'
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (p, s.count(old), old[:80])
    s = s.replace(old, new)


rep('''"""K2 laptop CPU smoke (harness only; nothing here ships, nothing is pushed).''',
    '''"""K3 laptop CPU smoke (scratchpad/k3; harness only; nothing here ships, nothing is pushed). = the K2 smoke + K3:
  A+ a tiny REAL TTR member (KneeFleetNetTTR, ttr_size 48 on the 64 px store, seed 1042, model_sha of the shipped
     fleet_model_ttr.py) packed with a dry family-1 member by k3/code/pack_fleet_ds.py (bigjump pack); a stub dataset
     whose stub worker reads the (filtered) manifest; 5 more notebooks (strict, strict + 2 members, TTR solo, stub
     filter, unknown member)
  B+ K3 worker (bigjump) == K2 kit worker on family-1 (P max|diff| 0)
  C+ strict / member-filter / outer-handler scenarios, real workers on the TTR pair
  C2 cross-run: the TTR member's rows in the filtered TTR-solo run == its rows in the unfiltered 2-member run, and the
     family-1 member's rows == the family-1-only run (the filter and the extra family change no prediction)
  D  build_k2.py on the REAL M1 + M2 (k3/ckpt): strict kernel + --kernel_only TTR-solo kernel + refusals
K2 text follows.
K2 laptop CPU smoke (harness only; nothing here ships, nothing is pushed).''')
rep('''STUB = r\'\'\'# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop|blank|views2
import os, sys, time, json
import numpy as np, pandas as pd
mode, shard, nsh = os.environ.get('STUB', 'ok'), int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])''',
    '''STUB = r\'\'\'# STUB fleet worker (K2 smoke only). STUB=ok|fastfail|hang|valve|hardstop|blank|views2|okman
import os, sys, time, json
import numpy as np, pandas as pd
mode, shard, nsh = os.environ.get('STUB', 'ok'), int(os.environ['FLEET_SHARD']), int(os.environ['FLEET_NSHARD'])
MAN = json.load(open(os.environ['FLEET_DIR'] + '/fleet_manifest.json'))['checkpoints']   # okman: names from the manifest''')
rep('''M = 3 if mode == 'valve' else 2''', '''M = 3 if mode == 'valve' else (len(MAN) if mode == 'okman' else 2)''')
rep('''np.savez(tmp, sids=np.array(ids), P=P, NV=NV, names=np.array(['stub_a', 'stub_b', 'stub_c'][:M]),''',
    '''np.savez(tmp, sids=np.array(ids), P=P, NV=NV, names=np.array(MAN if mode == 'okman' else ['stub_a', 'stub_b', 'stub_c'][:M]),''')
# ---- stage(): K3 additions after the K2 notebooks
rep('''    mk.main(['--out', f'{S}/nb/rsna-sub-k2-solo', '--slug', 'rsna-sub-k2-solo', '--main_mode', 'solo'])
    say('A notebooks built: fleet-only (32 cells), combined (35 cells), combined crop-floor-25 (35 cells)')''',
    '''    mk.main(['--out', f'{S}/nb/rsna-sub-k2-solo', '--slug', 'rsna-sub-k2-solo', '--main_mode', 'solo'])
    say('A notebooks built: fleet-only (32 cells), combined (35 cells), combined crop-floor-25 (35 cells)')
    stage_k3(mk)


MAKE_TTR = r\'\'\'# smoke only: a tiny REAL TTR member from the dry family-1 member (same weights, KneeFleetNetTTR at ttr_size 48)
import sys, os, copy, hashlib, inspect, torch
code, src, dst = sys.argv[1:4]
sys.path.insert(0, code)
import fleet_model as fm, fleet_model_ttr as tt
ck = torch.load(src, map_location='cpu', weights_only=False)
a = dict(ck['args'])
kw = dict(backbone=a['backbone'], n_views=a['n_views'], n_slices=a['n_slices'], dim=a['dim'], n_contrast=a['n_contrast'],
          contrast_emb=a['contrast_emb'], pretrained=False, drop_path=0.0, chunk=48)
m = tt.KneeFleetNetTTR(ttr_size=48, **kw)
miss, unexp = m.load_state_dict(ck['model'], strict=False)
assert list(miss) == ['ttr_size_t'] and not unexp, (miss, unexp)
sha = hashlib.sha256((chr(10).join(inspect.getsource(tt).splitlines()) + chr(10) +
                      chr(10).join(inspect.getsource(fm).splitlines())).encode()).hexdigest()[:16]
a.update(seed=1042, model_mod='fleet_model_ttr', ttr_size=48, model_sha=sha)
ck2 = dict(ck); ck2['args'] = a; ck2['model'] = m.state_dict()
torch.save(ck2, dst)
print('TTR', os.path.basename(dst), 'sha', sha, 'ttr_size_t', int(ck2['model']['ttr_size_t']))
\'\'\'


def stage_k3(mk):
    import pandas as pd
    os.makedirs(f'{S}/ttrmem')
    open(f'{S}/make_ttr.py', 'w').write(MAKE_TTR)
    f1 = f'{DRY}/fleetds/f0_s42_ema_cnxt64x24_cv5714_best.pt'
    ttr = f'{S}/ttrmem/f0_s1042_ema_ttr48-64x24_cv5714_best.pt'
    r = run([f'{S}/make_ttr.py', f'{S}/code64', f1, ttr], log=f'{S}/log_make_ttr.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    ok_sha = 'sha f141aeed136dfc01' in r.stdout
    say(f'A+ tiny TTR member: {r.stdout.strip()[-120:]} -> {"PASS" if ok_sha else "FAIL (model_sha != f141aeed136dfc01)"}')
    assert ok_sha
    r = run([f'{S}/code64/pack_fleet_ds.py', '--out', f'{S}/ds/fleet_ttr', '--store', f'{DRY}/store_64_normal', f1, ttr],
            log=f'{S}/log_pack_ttr.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    say('A+ pack (k3 = bigjump pack tool) family-1 + TTR: ' + r.stdout.strip().splitlines()[-1][-150:])
    names = json.load(open(f'{S}/ds/fleet_ttr/fleet_manifest.json'))['checkpoints']
    tm = dict(f1=os.path.basename(f1), ttr=os.path.basename(ttr))
    assert sorted(names) == sorted(tm.values()), names
    json.dump(tm, open(f'{S}/ttr_members.json', 'w'))
    # K2 kit worker twin of fleet_real (family-1 parity K3 worker vs K2 worker)
    shutil.copytree(f'{S}/ds/fleet_real', f'{S}/ds/fleet_k2w')
    shutil.copy2(f'{SP}/k2/code/fleet_worker.py', f'{S}/ds/fleet_k2w/fleet_worker.py')
    # stub dataset whose stub reads the manifest (.pt names, files present so the filtered view links them)
    os.makedirs(f'{S}/ds/fleet_stubman')
    open(f'{S}/ds/fleet_stubman/fleet_worker.py', 'w').write(STUB)
    for n in ('stub_a.pt', 'stub_b.pt'):
        open(f'{S}/ds/fleet_stubman/{n}', 'w').write('stub')
    json.dump({'checkpoints': ['stub_a.pt', 'stub_b.pt']}, open(f'{S}/ds/fleet_stubman/fleet_manifest.json', 'w'))
    W = ['--w_main', '0.25']
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-strict', '--slug', 'rsna-sub-k3-strict', '--main_mode', 'strict'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-strictttr', '--slug', 'rsna-sub-k3-strictttr', '--main_mode', 'strict',
             '--members', f'{tm["f1"]},{tm["ttr"]}'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-ttrsolo', '--slug', 'rsna-sub-k3-ttrsolo', '--main_mode', 'solo',
             '--members', tm['ttr']] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-stubfilter', '--slug', 'rsna-sub-k3-stubfilter', '--main_mode', 'strict',
             '--members', 'stub_b.pt'] + W)
    mk.main(['--out', f'{S}/nb/rsna-sub-k3-unknown', '--slug', 'rsna-sub-k3-unknown', '--main_mode', 'strict',
             '--members', 'nope.pt'] + W)
    # the old notebooks' K2_CFG carries no MEMBERS key (K2 cells byte-identical apart from the shared cell source)
    for nm in ('rsna-sub-k2-fleet', 'rsna-sub-k2-solo'):
        nb = json.load(open(f'{S}/nb/{nm}/{nm.replace("-", "_")}.ipynb', encoding='utf-8'))
        assert 'MEMBERS' not in ''.join(nb['cells'][31]['source']).splitlines()[0], nm
    say('A+ K3 notebooks built: strict W.25, strict + 2-member filter, TTR solo, stub filter, unknown member')''')
# ---- worker parity: + K3 worker vs K2 kit worker
rep('''    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3')):''',
    '''    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3'), ('k2kit', 'fleet_k2w', 'comp_big', '0'),
                              ('ttr', 'fleet_ttr', 'comp_big', '0')):''')
rep('''    say(f'B worker parity K2 vs fleet2 on {len(zk["sids"])} studies: P max|diff| {d} NaN-equal {nan_eq} -> {"PASS" if ok1 else "FAIL"}')''',
    '''    say(f'B worker parity K3 (bigjump) vs fleet2 on {len(zk["sids"])} studies: P max|diff| {d} NaN-equal {nan_eq} -> {"PASS" if ok1 else "FAIL"}')
    zw = outs['k2kit'][0]
    d2 = float(np.nanmax(np.abs(zk['P'] - zw['P'])))
    ok5 = list(zk['names']) == list(zw['names']) and d2 == 0.0 and bool((np.isnan(zk['P']) == np.isnan(zw['P'])).all())
    say(f'B+ worker parity K3 (bigjump) vs the K2 kit worker (family 1, the proven K2 path): P max|diff| {d2} -> {"PASS" if ok5 else "FAIL"}')
    zt = outs['ttr'][0]; tn = [str(x) for x in zt['names']]
    tm = json.load(open(f'{S}/ttr_members.json'))
    i1, it = tn.index(tm['f1']), tn.index(tm['ttr'])
    j1 = [str(x) for x in zk['names']].index(tm['f1'])
    d3 = float(np.nanmax(np.abs(zt['P'][i1] - zk['P'][j1])))
    dt = float(np.nanmax(np.abs(zt['P'][it] - zt['P'][i1])))
    ok6 = (int(zt['n_keep']) == 2 and d3 == 0.0 and np.isfinite(zt['P']).all() and dt > 1e-4
           and 'fleet_model_ttr' not in outs['ttr'][2].split('members:')[0])
    say(f'B+ K3 worker serves family 1 + TTR: keep {int(zt["n_keep"])}/2, family-1 rows == the family-1-only run '
        f'(max|diff| {d3}), TTR differs from its family-1 twin by {dt:.3f} (ttr 48 px resize active) -> {"PASS" if ok6 else "FAIL"}')
    ok1 = ok1 and ok5 and ok6''')
# ---- real build
rep('''def real_build():
    ck = sorted(glob.glob(f'{K2}/_real/ckpt/*.pt'))
    if not ck:
        say('D real-checkpoint build skipped (k2/_real/ckpt empty)'); return True''',
    '''def cross_runs():
    """C2: the member filter and the second family change no prediction (rows compared by member name)."""
    import numpy as np
    tm = json.load(open(f'{S}/ttr_members.json'))

    def rows(sc, name):
        out = {}
        for z in [np.load(f) for f in sorted(glob.glob(f'{S}/run_{sc}/working/_fleet_*.npz'))]:
            nm = [str(x) for x in z['names']]
            for k, sid in enumerate(z['sids']):
                out[str(sid)] = z['P'][nm.index(name), k]
        return out
    try:
        a, b = rows('ttrsolo_ok', tm['ttr']), rows('strict_ttr', tm['ttr'])
        c, d = rows('strict_ttr', tm['f1']), rows('strict_ttr_all', tm['f1'])
    except Exception as e:
        say(f'C2 cross-run check: needs ttrsolo_ok, strict_ttr, strict_ttr_all ({e!r}) -> FAIL'); return False
    d1 = max(float(np.abs(a[s] - b[s]).max()) for s in a)
    d2 = max(float(np.abs(c[s] - d[s]).max()) for s in c)
    ok = sorted(a) == sorted(b) and sorted(c) == sorted(d) and len(a) == 24 and d1 == 0.0 and d2 == 0.0
    say(f'C2 TTR rows filtered (solo) vs unfiltered 2-member run: max|diff| {d1}; family-1 rows filtered pair vs no '
        f'filter: {d2} ({len(a)} studies) -> {"PASS" if ok else "FAIL"}')
    return ok


def real_build():
    ck = sorted(glob.glob(f'{K2}/ckpt/*.pt'))
    m1 = [c for c in ck if os.path.basename(c).startswith('f0_s42_')]
    m2 = [c for c in ck if os.path.basename(c).startswith('f0_s1042_')]
    if len(m1) != 1 or len(m2) != 1:
        say(f'D real build: k3/ckpt must hold exactly one f0_s42_ and one f0_s1042_ file, has {ck} -> FAIL'); return False
    m1, m2 = m1[0], m2[0]
    n1, n2 = os.path.basename(m1), os.path.basename(m2)
    out = f'{S}/real_build'
    r = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', out,
             '--slug', 'rsna-sub-k3-fleet', '--main_mode', 'strict', '--w_main', '0.25', '--members', f'{n1},{n2}',
             '--overwrite'], log=f'{S}/log_real_build.txt')
    ok = r.returncode == 0 and 'upload folder OK' in r.stdout
    r2 = run([f'{K2}/build_k2.py', '--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-ttrsolo', '--main_mode', 'solo',
              '--w_main', '0.25', '--members', n2], log=f'{S}/log_real_build_ko.txt')
    ok = ok and r2.returncode == 0
    summ = json.load(open(f'{out}/BUILD_SUMMARY.json')) if ok else {}
    ks = {k['slug']: k for k in summ.get('kernels', [])}
    ok = ok and set(ks) == {'rsna-sub-k3-fleet', 'rsna-sub-k3-ttrsolo'} \\
        and ks['rsna-sub-k3-fleet']['K2_CFG'].get('MAIN_MODE') == 'strict' \\
        and ks['rsna-sub-k3-fleet']['K2_CFG'].get('MEMBERS') == [n1, n2] \\
        and ks['rsna-sub-k3-ttrsolo']['K2_CFG'].get('MAIN_MODE') == 'solo' \\
        and ks['rsna-sub-k3-ttrsolo']['K2_CFG'].get('MEMBERS') == [n2] \\
        and 'fleet_model_ttr.py' in summ['upload']['files']
    say(f'D build_k2.py on the REAL M1 + M2 ({n1}, {n2}): {"PASS" if ok else "FAIL"} | upload {summ.get("upload", {}).get("mb")} MB '
        f'{len(summ.get("upload", {}).get("files", []))} files | kernels {sorted(ks)}'
        + ('' if ok else f' | {r.stderr[-800:]} {r2.stderr[-800:]}'))
    neg = []
    for nm, args, needle in (
            ('two tags of one run', ['--ckpt', m1, '--ckpt', m1, '--store', f'{K2}/_real/store', '--out', f'{S}/real_build_neg',
                                     '--overwrite'], 'one tag per run'),
            ('an unknown --members name', ['--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out',
                                           f'{S}/real_build_neg', '--overwrite', '--members', 'nope.pt'], 'not in the manifest'),
            ('--kernel_only with an unknown name', ['--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-x',
                                                    '--members', 'f0_s1042_ema_x.pt'], 'not in the manifest'),
            ('--kernel_only on an existing slug', ['--kernel_only', '--out', out, '--slug', 'rsna-sub-k3-ttrsolo',
                                                   '--members', n2], 'already built')):
        rn = run([f'{K2}/build_k2.py'] + args)
        good = rn.returncode != 0 and needle in rn.stderr
        neg.append(good)
        say(f'D build_k2.py refuses {nm}: {"PASS" if good else "FAIL " + rn.stderr[-300:]}')
    return ok and all(neg)


def real_build_k2():
    ck = sorted(glob.glob(f'{K2}/_real/ckpt/*.pt'))
    if not ck:
        say('D real-checkpoint build skipped (k2/_real/ckpt empty)'); return True''')
rep('''ALL = ['fleet_normal', 'fleet_commit3', 'fleet_stub_ok', 'fleet_lowtime', 'fleet_fastfail', 'fleet_valve',
       'fleet_hardstop', 'fleet_hang', 'fleet_stale', 'fleet_writefail', 'fleet_nods', 'fleet_noparent',
       'fleet_badparent', 'fleet_blank', 'fleet_views2', 'fleet_noclock',
       'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf']''',
    '''ALL = ['fleet_normal', 'fleet_commit3', 'fleet_stub_ok', 'fleet_lowtime', 'fleet_fastfail', 'fleet_valve',
       'fleet_hardstop', 'fleet_hang', 'fleet_stale', 'fleet_writefail', 'fleet_nods', 'fleet_noparent',
       'fleet_badparent', 'fleet_blank', 'fleet_views2', 'fleet_noclock',
       'comb_normal', 'comb_stub_ok', 'comb_fleetfail', 'comb_nocrop', 'comb_croplate', 'comb_stale_cf',
       'solo_ok', 'solo_fail',
       'strict_ok', 'strict_fail', 'strict_writefail', 'strict_lowtime', 'strict_nods', 'strict_outer', 'solo_outer',
       'blend_outer', 'strict_ttr', 'strict_ttr_all', 'ttrsolo_ok', 'members_stub_ok', 'members_mismatch',
       'members_unknown']''')
rep('''    say(f'K2 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')
    stage()
    ok_b = worker_parity()
    ok_c = scenarios(names)
    ok_d = real_build()
    say(f'K2 SMOKE {"PASS" if (ok_b and ok_c and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')''',
    '''    say(f'K3 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')
    stage()
    ok_b = worker_parity()
    ok_c = scenarios(names)
    ok_c2 = cross_runs() if {'ttrsolo_ok', 'strict_ttr', 'strict_ttr_all'} <= set(names) else True
    ok_d = real_build() if os.environ.get('K3_SMOKE_SKIP_REAL') != '1' else True
    say(f'K3 SMOKE {"PASS" if (ok_b and ok_c and ok_c2 and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('smoke patched')
