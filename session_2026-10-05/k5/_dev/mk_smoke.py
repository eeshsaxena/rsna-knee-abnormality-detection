# dev helper (not shipped): K5 additions to k5/tests/smoke_k2.py and k5/tests/run_scenario.py (edited in place, once)
import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = {}


def load(p):
    s = open(f'{HERE}/{p}', encoding='utf-8', newline='').read()
    assert '\r\n' not in s and 'K5' not in s, f'{p}: already patched or CRLF'
    return s


def rep(s, old, new, cnt=1):
    assert s.count(old) == cnt, (old[:80], s.count(old))
    return s.replace(old, new)


# ------------------------------------------------------------------ smoke_k2.py
s = load('tests/smoke_k2.py')
s = rep(s, '''"""K3 laptop CPU smoke (scratchpad/k3; harness only; nothing here ships, nothing is pushed). = the K2 smoke + K3:''',
        '''"""K5 laptop CPU smoke (scratchpad/k5) = the K3 smoke below, unchanged, run on the K5 kit (K5 worker, K5 cell, K5
build_k2 / pack), + K5:
  A5 the TTR pair packed with --tta A,AB (family 1 A, TTR AB); a twin with the K3 worker (pre-K5); a twin whose manifest
     asks an unknown tta mode
  B5 K5 worker A-only == the K3 / K4 worker on the TTR pair (P max|diff| 0); AB run: TTR pass A (PA) == the A-only run,
     P == (PA + PB) / 2 exactly, family-1 rows == the A-only run, TTA record (modes, rows, forwards); commit-sized
     microbench with AB: bench passes {f1: 1, ttr: 2}
  C5 scenarios strict_ttr_tta (clean, k2_check --expect_tta), strict_tta_oldworker (manifest asks AB, pre-K5 worker:
     fail loud), strict_tta_bad (unknown mode: fail loud), ttrsolo_tta (solo, TTR AB)
  D  the real build reads M1 + M2 from k3/ckpt (read only; k5/ckpt holds the K5 set) + build_k2.py --tta on them
K3 text follows.
K3 laptop CPU smoke (scratchpad/k3; harness only; nothing here ships, nothing is pushed). = the K2 smoke + K3:''')
s = rep(s, '''    json.dump(tm, open(f'{S}/ttr_members.json', 'w'))
''', '''    json.dump(tm, open(f'{S}/ttr_members.json', 'w'))
    # K5: the same pair with per-member TTA (family 1 A, TTR AB), a pre-K5 worker twin, an unknown-mode twin
    r = run([f'{S}/code64/pack_fleet_ds.py', '--out', f'{S}/ds/fleet_ttr_tta', '--store', f'{DRY}/store_64_normal',
             '--tta', 'A,AB', f1, ttr], log=f'{S}/log_pack_ttr_tta.txt')
    assert r.returncode == 0, r.stderr[-1500:]
    mt = json.load(open(f'{S}/ds/fleet_ttr_tta/fleet_manifest.json'))
    ok_t = {k: v.get('tta') for k, v in mt['sources'].items()} == {tm['f1']: 'A', tm['ttr']: 'AB'}
    say(f'A5 pack --tta A,AB: manifest tta {[(k[:14], v.get("tta")) for k, v in mt["sources"].items()]} -> {"PASS" if ok_t else "FAIL"}')
    assert ok_t
    shutil.copytree(f'{S}/ds/fleet_ttr_tta', f'{S}/ds/fleet_ttr_tta_k3w')
    shutil.copy2(f'{SP}/k3/code/fleet_worker.py', f'{S}/ds/fleet_ttr_tta_k3w/fleet_worker.py')
    shutil.copytree(f'{S}/ds/fleet_ttr_tta', f'{S}/ds/fleet_ttr_badtta')
    mb = json.load(open(f'{S}/ds/fleet_ttr_badtta/fleet_manifest.json'))
    mb['sources'][tm['ttr']]['tta'] = 'ABM'
    json.dump(mb, open(f'{S}/ds/fleet_ttr_badtta/fleet_manifest.json', 'w'), indent=1)
    shutil.copytree(f'{S}/ds/fleet_ttr', f'{S}/ds/fleet_ttr_k3w')          # K3 worker on the A-only TTR pair
    shutil.copy2(f'{SP}/k3/code/fleet_worker.py', f'{S}/ds/fleet_ttr_k3w/fleet_worker.py')
''')
s = rep(s, '''    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3'), ('k2kit', 'fleet_k2w', 'comp_big', '0'),
                              ('ttr', 'fleet_ttr', 'comp_big', '0')):''',
        '''    for tag, ds, comp, mb in (('k2', 'fleet_real', 'comp_big', '0'), ('fleet2', 'fleet_f2orig', 'comp_big', '0'),
                              ('k2_bench', 'fleet_real', 'comp3', '3'), ('k2kit', 'fleet_k2w', 'comp_big', '0'),
                              ('ttr', 'fleet_ttr', 'comp_big', '0'), ('ttr_k3w', 'fleet_ttr_k3w', 'comp_big', '0'),
                              ('ttr_tta', 'fleet_ttr_tta', 'comp_big', '0'), ('ttr_tta_bench', 'fleet_ttr_tta', 'comp3', '3')):''')
s = rep(s, '''    ok4 = worker_timeout(zk)
    return ok1 and ok2 and ok3 and ok4''', '''    ok4 = worker_timeout(zk)
    ok5 = worker_parity_k5(outs, tm)
    return ok1 and ok2 and ok3 and ok4 and ok5


def worker_parity_k5(outs, tm):
    """B5: the K5 worker's A path == the K3/K4 worker; the AB path == pass A + pass B as tta_bench builds them."""
    import numpy as np
    zt, zk3 = outs['ttr'][0], outs['ttr_k3w'][0]
    d0 = float(np.nanmax(np.abs(zt['P'] - zk3['P'])))
    ok0 = (list(zt['names']) == list(zk3['names']) and list(zt['sids']) == list(zk3['sids']) and d0 == 0.0
           and (zt['NV'] == zk3['NV']).all() and bool((np.isnan(zt['P']) == np.isnan(zk3['P'])).all())
           and 'tta' not in zk3.files and json.loads(str(zt['tta']))['modes'] == {tm['f1']: 'A', tm['ttr']: 'A'})
    say(f'B5 K5 worker A-only == K3/K4 worker on family 1 + TTR ({len(zt["sids"])} studies): P max|diff| {d0} -> '
        f'{"PASS" if ok0 else "FAIL"}')
    za = outs['ttr_tta'][0]; nm = [str(x) for x in za['names']]
    i1, it = nm.index(tm['f1']), nm.index(tm['ttr'])
    dA = float(np.nanmax(np.abs(za['PA'][it] - zt['P'][it])))
    d1 = float(np.nanmax(np.abs(za['P'][i1] - zt['P'][i1])))
    ident = bool(np.array_equal(za['P'][it], (za['PA'][it] + za['PB'][it]) / 2))
    dB = float(np.nanmax(np.abs(za['PB'][it] - za['PA'][it])))
    t = json.loads(str(za['tta'])); n = len(za['sids'])
    ok1 = (dA == 0.0 and d1 == 0.0 and ident and dB > 1e-4 and np.isnan(za['PB'][i1]).all()
           and t['modes'] == {tm['f1']: 'A', tm['ttr']: 'AB'} and t['rows'] == n
           and t['fwd'] == {tm['f1']: {'A': n, 'B': 0}, tm['ttr']: {'A': n, 'B': n}} and abs(t['half'] - 0.499) < 1e-12
           and 'TTA modes' in outs['ttr_tta'][2] and 'TTA forwards' in outs['ttr_tta'][2])
    say(f'B5 K5 worker f1 A + TTR AB: TTR pass A == A-only run ({dA}), f1 rows == A-only run ({d1}), P == (PA+PB)/2 '
        f'{ident}, max|PB-PA| {dB:.3f}, record rows {t["rows"]} fwd {t["fwd"]} -> {"PASS" if ok1 else "FAIL"}')
    zb = outs['ttr_tta_bench'][0]; b = json.loads(str(zb['bench'])); tb = json.loads(str(zb['tta']))
    ok2 = (set(b) == {tm['f1'], tm['ttr']} and all(v > 0 for v in b.values())
           and tb['bench_passes'] == {tm['f1']: 1, tm['ttr']: 2} and 'MICROBENCH' in outs['ttr_tta_bench'][2])
    say(f'B5 K5 microbench (3 studies, AB): {b} passes {tb["bench_passes"]} -> {"PASS" if ok2 else "FAIL"}')
    return ok0 and ok1 and ok2''')
s = rep(s, '''def real_build():
    ck = sorted(glob.glob(f'{K2}/ckpt/*.pt'))''', '''def real_build():
    ck = sorted(glob.glob(f'{SP}/k3/ckpt/*.pt'))                         # K5: read only (k5/ckpt holds the K5 set)''')
s = rep(s, '''        say(f'D real build: k3/ckpt must hold exactly one f0_s42_ and one f0_s1042_ file, has {ck} -> FAIL'); return False''',
        '''        say(f'D real build: k3/ckpt must hold exactly one f0_s42_ and one f0_s1042_ file, has {ck} -> FAIL'); return False''')
s = rep(s, '''    neg = []
    # a DIFFERENT run with M1's fold/seed''', '''    # K5: build_k2.py --tta on the real pair (M1 A, M2 AB) writes the manifest tta; a partial map is refused
    out5 = f'{S}/real_build_tta'
    r5 = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', out5,
              '--slug', 'rsna-sub-k5-x', '--main_mode', 'strict', '--w_main', '0.20', '--members', f'{n1},{n2}',
              '--tta', f'{n1}=A,{n2}=AB', '--overwrite'], log=f'{S}/log_real_build_tta.txt')
    ok5 = r5.returncode == 0
    if ok5:
        man5 = json.load(open(f'{out5}/upload/rsna-knee-cnxt-fleet/fleet_manifest.json'))
        ok5 = {k: v.get('tta') for k, v in man5['sources'].items()} == {n1: 'A', n2: 'AB'}
    r6 = run([f'{K2}/build_k2.py', '--ckpt', m1, '--ckpt', m2, '--store', f'{K2}/_real/store', '--out', f'{S}/real_build_neg',
              '--overwrite', '--tta', f'{n2}=AB'])
    ok6 = r6.returncode != 0 and '--tta names' in r6.stderr
    say(f'D5 build_k2.py --tta on the REAL M1 + M2: manifest tta {"OK" if ok5 else "FAIL " + r5.stderr[-300:]}; '
        f'refuses a partial --tta map: {"PASS" if ok6 else "FAIL " + r6.stderr[-300:]}')
    ok = ok and ok5 and ok6
    neg = []
    # a DIFFERENT run with M1's fold/seed''')
s = rep(s, '''       'blend_outer', 'strict_ttr', 'strict_ttr_all', 'ttrsolo_ok', 'members_stub_ok', 'members_mismatch',
       'members_unknown']''', '''       'blend_outer', 'strict_ttr', 'strict_ttr_all', 'ttrsolo_ok', 'members_stub_ok', 'members_mismatch',
       'members_unknown',
       'strict_ttr_tta', 'strict_tta_oldworker', 'strict_tta_bad', 'ttrsolo_tta']                     # K5''')
s = rep(s, '''    say(f'K3 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')''',
        '''    say(f'K5 smoke {time.strftime("%Y-%m-%d %H:%M:%S")} on {PY}')''')
s = rep(s, '''    say(f'K3 SMOKE {"PASS" if (ok_b and ok_c and ok_c2 and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')''',
        '''    say(f'K5 SMOKE {"PASS" if (ok_b and ok_c and ok_c2 and ok_d) else "FAIL"} in {(time.time() - t0) / 60:.1f} min')''')
import ast
ast.parse(s)
S['tests/smoke_k2.py'] = s

# ------------------------------------------------------------------ run_scenario.py
s = load('tests/run_scenario.py')
s = rep(s, '''    'members_unknown':  dict(nb='unknown', fds='stubman', stub='okman', expect_fleet='fail'),''',
        '''    'members_unknown':  dict(nb='unknown', fds='stubman', stub='okman', expect_fleet='fail'),
    # ---- K5: per-member TTA (family 1 A, TTR AB) through the K5 cell gate and k2_check --expect_tta
    'strict_ttr_tta':   dict(nb='strict_ttr', fds='ttr_tta', members='ttr_pair', tta='pair'),
    'strict_tta_oldworker': dict(nb='strict_ttr', fds='ttr_tta_k3w', expect_fleet='fail'),
    'strict_tta_bad':   dict(nb='strict_ttr', fds='ttr_badtta', expect_fleet='fail'),
    'ttrsolo_tta':      dict(nb='ttrsolo', fds='ttr_tta', members='ttr_only', tta='ttr'),''')
s = rep(s, '''           outer=False, members=None)''', '''           outer=False, members=None, tta=None)''')
s = rep(s, '''EXPECT_MEMBERS = {None: None, 'ttr_pair': [TTRM['f1'], TTRM['ttr']], 'ttr_only': [TTRM['ttr']],
                  'stub_b': ['stub_b.pt']}[cfg['members']]''', '''EXPECT_MEMBERS = {None: None, 'ttr_pair': [TTRM['f1'], TTRM['ttr']], 'ttr_only': [TTRM['ttr']],
                  'stub_b': ['stub_b.pt']}[cfg['members']]
EXPECT_TTA = {None: None, 'pair': {TTRM['f1']: 'A', TTRM['ttr']: 'AB'}, 'ttr': {TTRM['ttr']: 'AB'}}[cfg['tta']]   # K5''')
s = rep(s, '''    res, info = k2_check.check(WORK, cfg['expect_fleet'], cfg['expect_crop'], expect_members=EXPECT_MEMBERS,
                               expect_manifest=(f'{S}/ds/fleet_{cfg["fds"]}/fleet_manifest.json'
                                                if cfg['fds'] in ('real', 'ttr') else None))     # K3c''',
        '''    res, info = k2_check.check(WORK, cfg['expect_fleet'], cfg['expect_crop'], expect_members=EXPECT_MEMBERS,
                               expect_manifest=(f'{S}/ds/fleet_{cfg["fds"]}/fleet_manifest.json'
                                                if cfg['fds'] in ('real', 'ttr', 'ttr_tta') else None),   # K3c
                               expect_tta=EXPECT_TTA)                                                     # K5''')
s = rep(s, '''if SC in ('strict_fail', 'members_mismatch'):''', '''if SC in ('strict_ttr_tta', 'ttrsolo_tta'):                               # K5
    t = rec.get('tta') or {}
    extra.append(('K5 receipt: tta record present with the manifest modes and per-member forwards',
                  t.get('modes') == EXPECT_TTA and t.get('rows') == 24 and all(
                      t['fwd'][k] == {'A': 24, 'B': 24 if m == 'AB' else 0} for k, m in EXPECT_TTA.items())))
    rows_ab = {}
    for f in sorted(glob.glob(f'{WORK}/_fleet_*.npz')):
        z = np.load(f); nm = [str(x) for x in z['names']]; j = nm.index(TTRM['ttr'])
        extra.append((f'K5 npz {os.path.basename(f)}: TTR P == (PA + PB) / 2 exactly',
                      bool(np.array_equal(z['P'][j], (z['PA'][j] + z['PB'][j]) / 2))))
if SC == 'strict_tta_oldworker':
    extra.append(('K5 old worker + AB manifest: why names the missing TTA report', 'no worker reports TTA' in why))
if SC == 'strict_tta_bad':
    extra.append(('K5 unknown tta mode: the workers refused it (rc 1)', 'worker rc' in why and '1' in why))
if SC in ('strict_fail', 'members_mismatch'):''')
s = rep(s, '''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown') and not cfg['outer']:''',
        '''if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown') and not cfg['outer']:''')
ast.parse(s)
S['tests/run_scenario.py'] = s
for p, t in S.items():
    open(f'{HERE}/{p}', 'w', encoding='utf-8', newline='').write(t)
print('ok')
