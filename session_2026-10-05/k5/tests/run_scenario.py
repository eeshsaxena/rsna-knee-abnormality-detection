# One K2 smoke scenario per process (harness only; nothing here ships):  python run_scenario.py <scenario>
# Executes the APPENDED cells of a notebook built by make_k2_kernel.py (exact cell text, /kaggle prefixes rewritten
# to a scratch root) in one namespace, as the notebook's last cells would run after the parent. Harness-only
# redirections: parent globals T0 / COMP / ROOT and RSNA_COMP_ROOT are set here (the parent sets them on Kaggle);
# torch.cuda reports 2 devices inside the FLEET cell only (workers forced to CPU with FLEET_ALLOW_CPU=1, laptop GPU
# hidden); cv2 = round3 PIL shim; the crop cell runs on CPU via its own CROP_FLEET_ALLOW_CPU test hook.
import os, sys, io, json, time, shutil, subprocess, glob, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''                  # hide the laptop GPU from this process before torch loads
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
T = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
K2 = os.path.dirname(T)
SP = os.path.dirname(K2)
S = f'{T}/_smoke'
SHIM = f'{SP}/round3/fleet_dry/shim'
sys.path.insert(0, K2); sys.path.insert(0, SHIM)
import numpy as np, pandas as pd
import torch
torch.set_num_threads(2)
import k2_check
TARGETS = k2_check.TARGETS

SC = sys.argv[1]
cfg = dict(nb='fleet', comp='big', fds='real', stub=None, crop=False, age_h=7.0, parent='normal', stale=False,
           write_fail=None, fast_clock=False, noclock=False, expect_fleet='clean', expect_crop='none',
           outer=False, members=None, tta=None)
SCEN = {
    'fleet_normal':     {},
    'fleet_commit3':    dict(comp='3', age_h=0.2),
    'fleet_stub_ok':    dict(fds='stub', stub='ok'),
    'fleet_lowtime':    dict(age_h=8.3, expect_fleet='fail'),
    'fleet_fastfail':   dict(fds='stub', stub='fastfail', expect_fleet='fail'),
    'fleet_valve':      dict(fds='stub', stub='valve', expect_fleet='fail'),
    'fleet_hardstop':   dict(fds='stub', stub='hardstop', expect_fleet='fail'),
    'fleet_hang':       dict(fds='stub', stub='hang', age_h=8.25, fast_clock=True, expect_fleet='fail'),
    'fleet_stale':      dict(fds='stub', stub='fastfail', stale=True, expect_fleet='fail'),
    'fleet_writefail':  dict(fds='stub', stub='ok', write_fail='submission_w40.csv', expect_fleet='fail'),
    'fleet_nods':       dict(fds='none', expect_fleet='fail'),
    'fleet_noparent':   dict(parent='missing', expect_fleet='fail'),
    'fleet_badparent':  dict(parent='badcols', expect_fleet='fail'),
    'fleet_blank':      dict(fds='stub', stub='blank', expect_fleet='fail'),
    'fleet_views2':     dict(fds='stub', stub='views2', expect_fleet='fail'),
    'fleet_noclock':    dict(fds='stub', stub='ok', noclock=True),
    'solo_ok':          dict(nb='solo', fds='stub', stub='ok'),
    'solo_fail':        dict(nb='solo', fds='stub', stub='fastfail', expect_fleet='fail'),
    # ---- K3: strict main mode (K3a) and the member filter (K3b)
    'strict_ok':        dict(nb='strict', fds='stub', stub='ok'),
    'strict_fail':      dict(nb='strict', fds='stub', stub='fastfail', expect_fleet='fail'),
    'strict_writefail': dict(nb='strict', fds='stub', stub='ok', write_fail='submission.csv', expect_fleet='fail'),
    'strict_lowtime':   dict(nb='strict', fds='stub', stub='ok', age_h=8.3, expect_fleet='fail'),
    'strict_nods':      dict(nb='strict', fds='none', expect_fleet='fail'),
    'strict_outer':     dict(nb='strict', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'solo_outer':       dict(nb='solo', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'blend_outer':      dict(nb='fleet', fds='stub', stub='ok', outer=True, expect_fleet='fail'),
    'strict_ttr':       dict(nb='strict_ttr', fds='ttr', members='ttr_pair'),
    'strict_ttr_all':   dict(nb='strict', fds='ttr'),
    'ttrsolo_ok':       dict(nb='ttrsolo', fds='ttr', members='ttr_only'),
    'members_stub_ok':  dict(nb='stubfilter', fds='stubman', stub='okman', members='stub_b'),
    'members_mismatch': dict(nb='stubfilter', fds='stubman', stub='ok', expect_fleet='fail'),
    'members_unknown':  dict(nb='unknown', fds='stubman', stub='okman', expect_fleet='fail'),
    # ---- K5: per-member TTA (family 1 A, TTR AB) through the K5 cell gate and k2_check --expect_tta
    'strict_ttr_tta':   dict(nb='strict_ttr', fds='ttr_tta', members='ttr_pair', tta='pair'),
    'strict_tta_oldworker': dict(nb='strict_ttr', fds='ttr_tta_k3w', expect_fleet='fail'),
    'strict_tta_bad':   dict(nb='strict_ttr', fds='ttr_badtta', expect_fleet='fail'),
    'ttrsolo_tta':      dict(nb='ttrsolo', fds='ttr_tta', members='ttr_only', tta='ttr'),
    # ---- K5b: the kernel's own TTA expectation (K2_CFG TTA) vs the attached manifest
    'strict_tta_expect_ok':   dict(nb='strict_exp', fds='ttr_tta', members='ttr_pair', tta='pair'),
    'strict_tta_expect_k4ds': dict(nb='strict_exp', fds='ttr', expect_fleet='fail'),
    'comb_normal':      dict(nb='combined', crop=True, expect_crop='ok'),
    'comb_stub_ok':     dict(nb='combined', fds='stub', stub='ok', crop=True, expect_crop='ok'),
    'comb_fleetfail':   dict(nb='combined', fds='stub', stub='fastfail', crop=True, expect_fleet='fail', expect_crop='ok'),
    'comb_nocrop':      dict(nb='combined', fds='stub', stub='ok', crop=False, expect_crop='fail'),
    'comb_croplate':    dict(nb='combined_croplate', fds='stub', stub='ok', crop=True, age_h=8.6 - 22 / 60,
                             expect_crop='fail'),
    'comb_stale_cf':    dict(nb='combined', fds='stub', stub='fastfail', crop=False, stale=True,
                             expect_fleet='fail', expect_crop='fail'),
}
cfg.update(SCEN[SC])
R = f'{S}/run_{SC}'
shutil.rmtree(R, ignore_errors=True)
INP, WORK = f'{R}/input', f'{R}/working'
os.makedirs(WORK); os.makedirs(f'{INP}/datasets/eeshsaxena')
COMP = f'{S}/comp_big' if cfg['comp'] == 'big' else f'{S}/comp3'
if cfg['fds'] != 'none':
    shutil.copytree(f'{S}/ds/fleet_{cfg["fds"]}', f'{INP}/datasets/eeshsaxena/rsna-knee-cnxt-fleet')
if cfg['crop']:
    shutil.copytree(f'{S}/ds/crop', f'{INP}/datasets/eeshsaxena/rsna-knee-cropaug-members')
# parent submission: test ids in a DIFFERENT row order (the fleet is reindexed to it)
ids = pd.read_csv(f'{COMP}/test.csv', dtype=str).StudyInstanceUID.tolist()
rng = np.random.default_rng(7)
par = pd.DataFrame(rng.uniform(0, 1, (len(ids), 12)), columns=TARGETS); par.insert(0, 'StudyInstanceUID', ids)
par = par.sample(frac=1, random_state=3).reset_index(drop=True)
par[TARGETS] = par[TARGETS].rank(pct=True)                   # like the parent: per-label rank pct
if cfg['parent'] == 'normal':
    par.to_csv(f'{WORK}/submission.csv', index=False)
elif cfg['parent'] == 'badcols':
    par.rename(columns={'ACL': 'acl'}).to_csv(f'{WORK}/submission.csv', index=False)
par_bytes = open(f'{WORK}/submission.csv', 'rb').read() if os.path.isfile(f'{WORK}/submission.csv') else None
if cfg['stale']:
    for f in ('submission_w20.csv', 'submission_w30.csv', 'submission_w40.csv', 'submission_fleetsolo.csv') + \
            (('submission_cf.csv', 'submission_c15.csv', 'submission_cropsolo.csv') if cfg['nb'] != 'fleet' else ()):
        par.to_csv(f'{WORK}/{f}', index=False)
    json.dump({'clean': True, 'files': {}}, open(f'{WORK}/_k2_stage_fleet.json', 'w'))
# ---- the appended cells, exact text from the built notebook
NB = {'fleet': f'{S}/nb/rsna-sub-k2-fleet/rsna_sub_k2_fleet.ipynb',
      'combined': f'{S}/nb/rsna-sub-k2-fleetcrop/rsna_sub_k2_fleetcrop.ipynb',
      'combined_croplate': f'{S}/nb/rsna-sub-k2-croplate/rsna_sub_k2_croplate.ipynb',
      'solo': f'{S}/nb/rsna-sub-k2-solo/rsna_sub_k2_solo.ipynb',
      'strict': f'{S}/nb/rsna-sub-k3-strict/rsna_sub_k3_strict.ipynb',
      'strict_ttr': f'{S}/nb/rsna-sub-k3-strictttr/rsna_sub_k3_strictttr.ipynb',
      'ttrsolo': f'{S}/nb/rsna-sub-k3-ttrsolo/rsna_sub_k3_ttrsolo.ipynb',
      'stubfilter': f'{S}/nb/rsna-sub-k3-stubfilter/rsna_sub_k3_stubfilter.ipynb',
      'unknown': f'{S}/nb/rsna-sub-k3-unknown/rsna_sub_k3_unknown.ipynb',
      'strict_exp': f'{S}/nb/rsna-sub-k5-strictexp/rsna_sub_k5_strictexp.ipynb'}[cfg['nb']]          # K5b
TTRM = json.load(open(f'{S}/ttr_members.json'))            # written by smoke stage(): the two real-worker member names
EXPECT_MEMBERS = {None: None, 'ttr_pair': [TTRM['f1'], TTRM['ttr']], 'ttr_only': [TTRM['ttr']],
                  'stub_b': ['stub_b.pt']}[cfg['members']]
EXPECT_TTA = {None: None, 'pair': {TTRM['f1']: 'A', TTRM['ttr']: 'AB'}, 'ttr': {TTRM['ttr']: 'AB'}}[cfg['tta']]   # K5
nb = json.load(open(NB, encoding='utf-8'))
cells = [(c['metadata'].get('k2', '?'), ''.join(c['source'])) for c in nb['cells'][31:]]


def redirect(src):
    src = src.replace('/kaggle/input', INP).replace('/kaggle/working', WORK)
    bad = [l for l in src.splitlines() if '/kaggle/' in l and not l.lstrip().startswith('#')]
    assert not bad, bad[:3]
    return src


# ---- harness shims
os.environ['RSNA_COMP_ROOT'] = COMP                          # the parent exports this (repro0946 cell 7)
os.environ['CROP_FLEET_ALLOW_CPU'] = '1'                     # crop cell test hook (CPU members)
for k in ('CROP_FLEET_WORK', 'CROP_FLEET_OVERRIDES', 'CROP_FLEET_INPUT'):
    os.environ.pop(k, None)
_Popen = subprocess.Popen


def _popen(cmd, *a, env=None, **k):
    env = dict(env or os.environ); env['CUDA_VISIBLE_DEVICES'] = '-1'; env['OMP_NUM_THREADS'] = '2'
    env['PYTHONPATH'] = env.get('PYTHONPATH', '') + os.pathsep + SHIM; env['PYTHONWARNINGS'] = 'ignore'
    env['FLEET_ALLOW_CPU'] = '1'
    if cfg['stub']:
        env['STUB'] = cfg['stub']
    return _Popen(cmd, *a, env=env, **k)


subprocess.Popen = _popen
if cfg['write_fail']:
    _rep = os.replace

    def _replace(src, dst):
        if str(dst).replace('\\', '/').endswith('/' + cfg['write_fail']):
            raise OSError(28, 'No space left on device (injected)')
        return _rep(src, dst)
    os.replace = _replace
if cfg['fast_clock']:
    _rt, _t0 = time.time, time.time()
    time.time = lambda: _t0 + (_rt() - _t0) * 200.0
g = {'__name__': '__main__', 'T0': time.time() - cfg['age_h'] * 3600, 'COMP': COMP, 'ROOT': COMP}
if cfg['noclock']:                                           # no parent T0 (Windows has no /proc either): no clock
    del g['T0']
cell_s = {}
_cuda = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache)
OUTER_PAR = None
for name, src in cells:
    if name == 'k2_fleet_stage':
        torch.cuda.is_available = lambda: True; torch.cuda.device_count = lambda: 2; torch.cuda.empty_cache = lambda: None
    t0 = time.perf_counter()
    try:
        if cfg['outer'] and name == 'k2_fleet_stage':
            # K3a outer-handler test (harness only): the cell text minus its last line, then an inner stage that copies
            # the parent, writes a fake fleet main file and raises OUTSIDE the guarded path; then the real _k2_entry()
            body = redirect(src)
            assert body.rstrip().endswith('_k2_receipt = _k2_entry()')
            exec(compile(body.rstrip()[:-len('_k2_receipt = _k2_entry()')], name, 'exec'), g)

            def _boom():
                shutil.copyfile(g['_K2_SUB'], g['_K2_PAR'])
                fake = pd.read_csv(g['_K2_SUB']); fake[TARGETS] = 0.5; fake.to_csv(g['_K2_SUB'], index=False)
                open(f'{WORK}/submission_fleetsolo.csv', 'w').write('x')
                raise RuntimeError('injected outside the guarded path')
            g['_k2_entry_inner'] = _boom
            g['_k2_receipt'] = g['_k2_entry']()
        else:
            exec(compile(redirect(src), name, 'exec'), g)
        escaped = None
    except BaseException as e:                               # must never happen: the cell would fail the notebook
        escaped = repr(e)
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.empty_cache = _cuda
    cell_s[name] = round(time.perf_counter() - t0, 1)
    assert escaped is None, f'exception escaped cell {name}: {escaped}'
if cfg['write_fail']:
    os.replace = _rep
# ---- checks
if cfg['parent'] == 'normal' and cfg['nb'] != 'solo' and not cfg['outer']:
    res, info = k2_check.check(WORK, cfg['expect_fleet'], cfg['expect_crop'], expect_members=EXPECT_MEMBERS,
                               expect_manifest=(f'{S}/ds/fleet_{cfg["fds"]}/fleet_manifest.json'
                                                if cfg['fds'] in ('real', 'ttr', 'ttr_tta') else None),   # K3c
                               expect_tta=EXPECT_TTA)                                                     # K5
else:
    res, info = [], {}
if cfg['nb'] == 'solo':                                     # solo mode: main file = fleet solo, never the parent
    sub, solo = f'{WORK}/submission.csv', f'{WORK}/submission_fleetsolo.csv'
    if cfg['expect_fleet'] == 'clean':
        extra0 = [('solo ok: submission.csv exists and == submission_fleetsolo.csv bytes',
                   os.path.isfile(sub) and os.path.isfile(solo) and open(sub, 'rb').read() == open(solo, 'rb').read()),
                  ('solo ok: submission.csv differs from the parent bytes', os.path.isfile(sub) and open(sub, 'rb').read() != par_bytes)]
    else:
        extra0 = [('solo fail: submission.csv DELETED (submission errors instead of scoring the parent)', not os.path.isfile(sub)),
                  ('solo fail: no fail-loud file left', not [f for f in os.listdir(WORK) if f.startswith('submission_')])]
    res += [(n, bool(c), '') for n, c in extra0]
extra = []
if cfg['parent'] != 'normal':
    extra.append(('bad/missing parent: no fail-loud file exists',
                  not [f for f in os.listdir(WORK) if f.startswith('submission_')]))
if cfg['parent'] == 'missing':
    extra.append(('no parent: nothing written at all', not [f for f in os.listdir(WORK) if f.startswith('submission')]))
    extra.append(('no parent: cell returned a receipt', isinstance(g.get('_k2_receipt'), dict)))
if cfg['parent'] == 'badcols':
    extra.append(('bad parent: submission.csv bytes untouched', open(f'{WORK}/submission.csv', 'rb').read() == par_bytes))
if SC == 'fleet_lowtime':
    extra.append(('lowtime: no worker was launched', not glob.glob(f'{WORK}/_fleet_*.npz')))
    extra.append(('lowtime: why names the wall', 'before the wall' in str(g['_k2_receipt'].get('why'))))
if SC in ('fleet_normal', 'comb_normal'):
    rec = g['_k2_receipt']
    extra.append(('real workers: no microbench on a > 3-study run', all(not b for b in rec.get('bench', []))))
    extra.append(('real workers: decode s/study recorded', (rec.get('decode_s_per_study_per_thread') or 0) > 0))
    extra.append(('real workers: 2 shards, 2 members each, nothing median-filled',
                  rec.get('n_keep') == [2, 2] and rec.get('median_filled') == 0))
if SC == 'fleet_commit3':
    rec = g['_k2_receipt']
    b = rec.get('bench', [])
    extra.append(('commit run: microbench ms/member/study from every worker',
                  len(b) == len(rec.get('gpus', [])) and all(len(x) == 2 and all(v > 0 for v in x.values()) for x in b)))
for nm, wf in (('fleet_fastfail', 'rc'), ('fleet_valve', 'full member count'), ('fleet_hardstop', 'hit the wall'),
               ('fleet_hang', 'after the wall'), ('fleet_nods', 'not attached'), ('fleet_writefail', 'No space left'),
               ('fleet_blank', 'failed studies'), ('fleet_views2', 'views/study')):
    if SC == nm:
        extra.append((f'{nm}: why mentions "{wf}"', wf in str(g['_k2_receipt'].get('why'))))
if SC in ('fleet_hang', 'fleet_fastfail', 'fleet_stale', 'comb_fleetfail'):
    time.sleep(1)
    # every worker the cell started must be dead (the cell kills and reaps them)
    import psutil
    me = psutil.Process()
    kids = [c for c in me.children(recursive=True) if c.is_running() and c.status() != psutil.STATUS_ZOMBIE]
    extra.append(('no worker left running after the cell', not kids))
if SC == 'fleet_noclock':
    rec = g['_k2_receipt']
    extra.append(('noclock: no kernel age, stage given NOCLOCK_MIN (40) minutes',
                  rec.get('kernel_age_h') is None and 39.0 < float(rec.get('left_min') or 0) <= 40.01))
if SC in ('fleet_normal', 'fleet_stub_ok', 'fleet_commit3'):  # commit3: age 0.2 h, the old 150 min cap would show
    rec = g['_k2_receipt']                                   # P6: the wall is the only guard when a clock exists
    extra.append(('wall only: left_min == (8.6 h - kernel age) to 0.1 min',
                  abs(float(rec.get('left_min') or 0) - (8.6 - float(rec.get('kernel_age_h') or 0)) * 60) < 0.1))
if cfg['nb'].startswith('combined') and isinstance(g.get('_k2c_receipt'), dict) and cfg['crop']:
    sb = f'{WORK}/_k2_crop'
    extra.append(('C6: sandbox submission*.csv renamed k1out*.csv',
                  os.path.isdir(sb) and not [f for f in os.listdir(sb) if f.startswith('submission')]
                  and isinstance(g['_k2c_receipt'].get('sandbox_renamed'), list)))
if SC == 'comb_croplate':
    extra.append(('croplate: crop cell refused to start (MIN_START)', 'not started' in str(g['_k2c_receipt'].get('crop_why'))))
if SC == 'comb_nocrop':
    extra.append(('nocrop: crop stage disarmed, crop cell wrote nothing', g.get('_cf_receipt') is None))
if cfg['nb'].startswith('combined'):
    extra.append(('crop cell CF_WORK was the sandbox', str(g.get('CF_WORK', '')).replace('\\', '/').endswith('/_k2_crop')))
    extra.append(('crop env vars cleared after the stage', 'CROP_FLEET_WORK' not in os.environ))
# ---- K3 checks
rec = g.get('_k2_receipt') or {}
why = str(rec.get('why'))
sub, solo = f'{WORK}/submission.csv', f'{WORK}/submission_fleetsolo.csv'
loud_left = [f for f in os.listdir(WORK) if f.startswith('submission_')]
if cfg['nb'] in ('strict', 'strict_ttr', 'stubfilter', 'unknown', 'strict_exp') and not cfg['outer']:   # K5b
    if cfg['expect_fleet'] == 'clean':
        extra.append(('strict ok: submission.csv == submission_w25.csv bytes (P+F at W .25)',
                      os.path.isfile(sub) and open(sub, 'rb').read() == open(f'{WORK}/submission_w25.csv', 'rb').read()))
        extra.append(('strict ok: submission.csv differs from the parent bytes', open(sub, 'rb').read() != par_bytes))
    else:
        extra.append(('strict fail: submission.csv DELETED (never the parent bytes)', not os.path.exists(sub)))
        extra.append(('strict fail: no fail-loud file left', not loud_left))
        extra.append(('strict fail: _k2_parent.csv kept (parent bytes, not submittable)',
                      not os.path.isfile(f'{WORK}/_k2_parent.csv') or open(f'{WORK}/_k2_parent.csv', 'rb').read() == par_bytes))
if SC == 'strict_writefail':
    extra.append(('strict writefail: why names the injected write error', 'No space left' in why))
if SC == 'strict_lowtime':
    extra.append(('strict lowtime: no worker was launched', not glob.glob(f'{WORK}/_fleet_*.npz')))
if SC == 'strict_nods':
    extra.append(('strict nods: why says not attached', 'not attached' in why))
if cfg['outer']:
    left = os.path.isfile(sub) and open(sub, 'rb').read()
    if cfg['nb'] == 'fleet':
        extra.append(('blend outer: submission.csv restored to the parent bytes (K2 behaviour kept)', left == par_bytes))
    else:
        extra.append((f'{cfg["nb"]} outer: submission.csv DELETED (K3a: never the parent, never the fake)', not os.path.exists(sub)))
    extra.append(('outer: fail-loud files removed', not loud_left))
    extra.append(('outer: _k2_entry returned None (exception contained)', g.get('_k2_receipt') is None))
if EXPECT_MEMBERS is not None and cfg['expect_fleet'] == 'clean':
    extra.append((f'members: receipt names == {EXPECT_MEMBERS}, n_keep == {len(EXPECT_MEMBERS)} per worker',
                  sorted(rec.get('names') or []) == sorted(EXPECT_MEMBERS) and rec.get('n_keep') == [len(EXPECT_MEMBERS)] * 2))
    md = rec.get('member_dir')
    extra.append(('members: private FLEET_DIR removed after the stage', bool(md) and not os.path.exists(md)))
    extra.append(('members: nothing extra in the working dir',
                  not [f for f in os.listdir(WORK) if 'k2_members' in f or f.endswith('.pt')]))
if SC == 'strict_ttr_all':
    extra.append(('no filter: both manifest members ran (TTR + family 1)', rec.get('models') == 2 and
                  sorted(rec.get('names') or []) == sorted([TTRM['f1'], TTRM['ttr']]) and rec.get('member_dir') is None))
if SC == 'ttrsolo_ok':
    extra.append(('ttrsolo: submission.csv == submission_fleetsolo.csv bytes (solo mode)',
                  os.path.isfile(sub) and open(sub, 'rb').read() == open(solo, 'rb').read()))
if SC == 'members_mismatch':
    extra.append(('mismatch: why names the member filter', 'member filter selected' in why))
if SC == 'members_unknown':
    extra.append(('unknown: why names the manifest', 'not in the fleet manifest' in why))
    extra.append(('unknown: no worker was launched', not glob.glob(f'{WORK}/_fleet_*.npz')))
    extra.append(('unknown: no private FLEET_DIR created', rec.get('member_dir') is None))
if SC in ('strict_ttr_tta', 'ttrsolo_tta'):                               # K5
    t = rec.get('tta') or {}
    extra.append(('K5 receipt: tta record present with the manifest modes and per-member forwards',
                  t.get('modes') == EXPECT_TTA and t.get('rows') == 24 and all(
                      t['fwd'][k] == {'A': 24, 'B': 24 if m == 'AB' else 0} for k, m in EXPECT_TTA.items())))
    rows_ab = {}
    for f in sorted(glob.glob(f'{WORK}/_fleet_*.npz')):
        z = np.load(f); nm = [str(x) for x in z['names']]; j = nm.index(TTRM['ttr'])
        extra.append((f'K5 npz {os.path.basename(f)}: TTR P == (PA + PB) / 2 exactly',
                      bool(np.array_equal(z['P'][j], (z['PA'][j] + z['PB'][j]) / 2))))
if SC == 'strict_tta_expect_ok':                                         # K5b
    extra.append(('K5b kernel TTA expectation recorded and == the manifest', rec.get('tta_kernel') == EXPECT_TTA))
if SC == 'strict_tta_expect_k4ds':
    extra.append(('K5b manifest without tta under a K5 kernel: why names the kernel expectation',
                  'kernel expects TTA' in why))
if SC == 'strict_tta_oldworker':
    extra.append(('K5 old worker + AB manifest: why names the missing TTA report', 'no worker reports TTA' in why))
if SC == 'strict_tta_bad':
    extra.append(('K5 unknown tta mode: the workers refused it (rc 1)', 'worker rc' in why and '1' in why))
if SC in ('strict_fail', 'members_mismatch'):
    time.sleep(1)
    import psutil
    kids = [c for c in psutil.Process().children(recursive=True) if c.is_running() and c.status() != psutil.STATUS_ZOMBIE]
    extra.append(('no worker left running after the cell', not kids))
res += [(n, bool(c), '') for n, c in extra]
for n, c, d in res:
    print(f'{"PASS" if c else "FAIL"}  {n}  {d}')
out = dict(scenario=SC, passed=sum(c for _, c, _ in res), checks=len(res), ok=all(c for _, c, _ in res), cell_s=cell_s,
           why=str((g.get('_k2_receipt') or {}).get('why'))[:200],
           crop=str((g.get('_k2c_receipt') or {}).get('group_C', ''))[:60] + ' | ' + str((g.get('_k2c_receipt') or {}).get('group_CF', ''))[:60],
           info={k: info.get(k) for k in ('fleet_stage_min', 'worker_loop_min', 'bench', 'decode_s_per_study_per_thread',
                                         'crop_cell_min', 'crop_members')})
print('RESULT', json.dumps(out, default=str))
