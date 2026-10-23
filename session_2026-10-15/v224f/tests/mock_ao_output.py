# mock_ao_output.py (2026-10-15 dir, 10-15, Claude subagent): MOCK commit outputs for the always-on-flip student kernels
# (build_v224fs.py --base flip | flrm --standin -> tests/standin_v224flips20 / tests/standin_v224flrms20), assembled from REAL commits:
#   base part  = the V224-FLIP commit (session_2026-10-14/v224/out_v224flip_v1) or the V224-FLIP-RM commit (v224rm/out_v224flrm_v1):
#                its log up to its last marker, its stage csvs, receipts and worker logs; its submission.csv becomes submission_V224.csv
#                (what SAVE-AO copies);
#   fleet part = the V224-FF20 commit (v224ff/out_v224ff20_v1): its log from V224F_SAVED on and its fleet files, reduced to the 5 DT
#                rows relabelled as the stand-in S (as mock_fs_output.py ffsr does), SAVED knobs / K2 config at the stand-in's AO guard
#                (its BUILD json: 7.0 h / MIN_START_MIN 78 since fix aofix 10-15 F1, was 6.5 h / 108), the s-merge line, manifest /
#                manifest_sha16 / bench under the S names; then the stand-in
#                notebook's own HF-AO cell is EXECUTED on it (its printed lines replace the FF20 HF lines).
# Then: v224aos_check.py on the clean mock (PASS), with --ref-s (PASS), on mutated copies (each must FAIL), cross runs, and the HF-AO
# cell-level tests (clean path == the f / ff HF-S byte for byte; a Fracture canary from RM_GUARD or the V224 CANARY stays blank
# through the S blend; the f / ff HF-S would un-blank RM_GUARD's). Offline, laptop CPU, no GPU, no Kaggle.
#   usage: python tests/mock_ao_output.py flips|flrms      (the cross negatives need both stand-ins and tests/mock_out_v224ffsr20)
import os, sys, json, time, shutil, hashlib, io, contextlib, subprocess, re, tempfile
import numpy as np, pandas as pd

T = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(T); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
MODE = sys.argv[1] if len(sys.argv) > 1 else 'flips'
assert MODE in ('flips', 'flrms'), MODE
BASE = 'flip' if MODE == 'flips' else 'flrm'
ST = 'v224%s20' % MODE
OTHER = 'v224flrms20' if MODE == 'flips' else 'v224flips20'
REF_X = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1')
REF_FLIP = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1')
REF_FLRM = os.path.join(ROOT, 'session_2026-10-15', 'v224rm', 'out_v224flrm_v1')
REF_F = os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'out_v224ff20_v1')
REF_B = REF_FLIP if BASE == 'flip' else REF_FLRM
LAST = 'V224_DONE' if BASE == 'flip' else 'V224RM_DONE'
OUT = os.path.join(T, 'mock_out_' + ST); KD = os.path.join(T, 'standin_' + ST)
FFSR_OUT = os.path.join(T, 'mock_out_v224ffsr20')
BJ = os.path.join(KD, 'BUILD_V224FS.json')
B = json.load(open(BJ, encoding='utf-8'))
NB = json.load(open(os.path.join(KD, 'rsna_sub_%s.ipynb' % ST), encoding='utf-8'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
HF = [src(c) for c in NB['cells'] if src(c).startswith('# [eeshsaxena V224-AO-S HF')]
assert len(HF) == 1; HF = HF[0]
NB_FFSR = json.load(open(os.path.join(T, 'standin_v224ffsr20', 'rsna_sub_v224ffsr20.ipynb'), encoding='utf-8'))
HF_FF = [src(c) for c in NB_FFSR['cells'] if src(c).startswith('# [eeshsaxena V224-F-S HF')][0]       # the f / ff HF-S text
FLEET = [src(c) for c in NB['cells'] if src(c).startswith('K2_CFG = {')][0]
K2 = eval(FLEET.split('\n', 1)[0][len('K2_CFG = '):FLEET.split('\n', 1)[0].index('   # literal')], {})
SC = [src(c) for c in NB['cells'] if src(c).startswith('S_CFG = {')][0]
S_CFG = eval(SC.split('\n', 1)[0][len('S_CFG = '):SC.split('\n', 1)[0].index('   # literal')], {})
GH, GM = B['start_budget_h'], B['min_start_min']                  # the stand-in's AO start guard (fix aofix 10-15: read, not 6.5 / 108)
assert B['base'] == BASE and B['k14'] == [] and K2['MEMBERS'] == B['s'] and K2['MIN_START_MIN'] == GM == round((8.3 - GH) * 60.0, 1)
assert GH in (6.0, 6.5, 7.0), GH
sample = pd.read_csv(os.path.join(ROOT, '_data', 'sample_submission.csv'), dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; test_ids = sample['StudyInstanceUID'].tolist()
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']
sha_ = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str})


def logp(M):
    return os.path.join(M, 'rsna-sub-%s.log' % ST)


# ---- 1. the base part ------------------------------------------------------------------------------------------------------------
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
shutil.copytree(REF_B, OUT, ignore=shutil.ignore_patterns('__pycache__'))
bl = [p for p in os.listdir(OUT) if p.endswith('.log') and 'cufile' not in p]
assert len(bl) == 1
EB = json.load(open(os.path.join(OUT, bl[0]), encoding='utf-8')); os.remove(os.path.join(OUT, bl[0]))
kl = max(i for i, e in enumerate(EB) if LAST in e['data'])
EB = EB[:kl + 1]
os.replace(os.path.join(OUT, 'submission.csv'), os.path.join(OUT, 'submission_V224.csv'))
H_SHA = sha_(os.path.join(OUT, 'submission_V224.csv'))
CANARY = json.load(open(os.path.join(REF_B, 'canary_receipt.json')))      # flip: the CANARY's _canary; flrm: RM_GUARD's _gcr
CAN0 = {k: CANARY[k] for k in ('r224_used', 'R224_OK', 'use_f', 'flip_tta_forced')} | {'fired': [x for x in CANARY['fired'] if x == 'r224->Fracture']}

# ---- 2. the fleet part: files ----------------------------------------------------------------------------------------------------
for f in ('_fn_ids.csv', '_fn_ids.json', '_k2_parent.csv', '_k2_stage_fleet.json', 'submission_w40.csv', 'submission_w50.csv',
          'submission_w60.csv'):
    shutil.copyfile(os.path.join(REF_F, f), os.path.join(OUT, f))
DT = [s.replace('_s1047r7tb_', '_s1043f75oai_') for s in B['s']]
smap = {d: d.replace('_s1043f75oai_', '_s1047r7tb_') for d in DT}
parts = []
for g in (0, 1):
    z = dict(np.load(os.path.join(REF_F, '_fleet_%d.npz' % g), allow_pickle=False)); nm = [str(x) for x in z['names']]
    if g == 0:
        REFN = list(nm)
    for k in ('P', 'PA', 'PB'):
        z[k] = np.stack([z[k][nm.index(d)] for d in DT])
    z['names'] = np.array([smap[d] for d in DT]); z['n_keep'] = np.array(5); parts.append(z)
    np.savez(os.path.join(OUT, '_fleet_%d.npz' % g), **z)
PP = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64); ii = np.concatenate([z['sids'] for z in parts]).astype(str)
solo = pd.DataFrame(PP.mean(0), columns=TG, index=ii).reindex(test_ids); solo.insert(0, 'StudyInstanceUID', test_ids)
solo.to_csv(os.path.join(OUT, 'submission_fleetsolo.csv'), index=False)
fr = json.load(open(os.path.join(REF_F, 'fleet_receipt.json')))
names = [str(x) for x in parts[0]['names']]
assert names == K2['MEMBERS']
K14_LINE = '%d member(s) %r' % (len(REFN), REFN); S_LINE = '5 member(s) %r' % (names,)
assert fr['why'].count(K14_LINE) == 1
SSHA = {n: S_CFG['s_manifest']['sources'][n]['sha256'] for n in B['s']}
fr.update(models=5, names=names, n_keep=[5, 5], members_filter=list(K2['MEMBERS']), tta_expected={m: 'AB' for m in names},
          tta_kernel={m: 'AB' for m in names}, bench=[{smap[d]: b[d] for d in DT} if b else b for b in fr['bench']],
          why=fr['why'].replace(K14_LINE, S_LINE), manifest=list(fr['manifest']) + list(B['s']),
          manifest_sha16=dict(fr['manifest_sha16'], **SSHA))
fr['config'] = dict(fr['config'], MEMBERS=K2['MEMBERS'], TTA=K2['TTA'], MIN_START_MIN=K2['MIN_START_MIN'], WALL_H=K2['WALL_H'])
fr['tta'] = dict(fr['tta'], modes={m: 'AB' for m in names}, fwd={smap[d]: fr['tta']['fwd'][d] for d in DT})
json.dump(fr, open(os.path.join(OUT, 'fleet_receipt.json'), 'w'), indent=1)

# ---- 3. the fleet part: log ------------------------------------------------------------------------------------------------------
EF = json.load(open(os.path.join(REF_F, 'rsna-sub-v224ff20.log'), encoding='utf-8'))
k0 = next(i for i, e in enumerate(EF) if 'V224F_SAVED' in e['data'])
EF = [dict(e) for e in EF[k0:] if not any(m in e['data'] for m in ('HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'))]
m = re.search(r'V224F_SAVED (\{.*\})', EF[0]['data']); sv = json.loads(m.group(1))
sv.update(tag=B['tag'], start_budget_h=B['start_budget_h'], min_start_min=B['min_start_min'], members=5, groups={'K14': 0, 'S': 5},
          s_share=1.0, s_ds=B['s_ds'], ao_base=BASE, sha256=H_SHA)
EF[0]['data'] = EF[0]['data'].replace(m.group(1), json.dumps(sv))
kk = next(i for i, e in enumerate(EF) if '[k12-merge' in e['data'] and 'RESULT OK merged fleet dir' in e['data'])
EF.insert(kk + 1, dict(EF[kk], data='[s-merge 07:55:49] RESULT OK merged fleet dir + S: 30 checkpoints (5 S from /kaggle/input/datasets/'
                                    'eeshsaxena/%s, sha256 verified, 0.4 s)\n' % B['s_ds']))
kc = next(i for i, e in enumerate(EF) if 'RESULT OK fleet stage clean: ' + K14_LINE in e['data'])
EF[kc]['data'] = EF[kc]['data'].replace('RESULT OK fleet stage clean: ' + K14_LINE, 'RESULT OK fleet stage clean: ' + S_LINE)
nw = 0
for e in EF:
    mm = re.match(r'(\[fleet \d\] members: )(.*)$', e['data'], re.S)
    if mm:
        e['data'] = mm.group(1) + ', '.join('%s[KneeFleetNetTTR]' % n for n in names) + '\n'; nw += 1
kr = [i for i, e in enumerate(EF) if e['data'].startswith('[k2-fleet] receipt {')]
assert nw == 4 and len(kr) == 1
EF[kr[0]]['data'] = '[k2-fleet] receipt ' + json.dumps(fr) + '\n'
for e in EF:                                                     # the FF20 fleet part carries no flip-driver line (asserted)
    assert not any(x in e['data'] for x in ('FLIP_TA_', 'compact_infer_ta', 'FLIP_TTA_DISABLED')), e['data'][:120]
E = EB + EF
json.dump(E, open(logp(OUT), 'w', encoding='utf-8'))


# ---- 4. execute the HF-AO cell ---------------------------------------------------------------------------------------------------
def run_hf(M, k2rec, canary, gcr, text=HF, splice=True):
    """Execute an HF cell text in M (as the kernel does after ENV-RESTORE) and splice its printed lines into M's log."""
    H = rd(os.path.join(M, 'submission_V224.csv'))
    svm = json.loads(re.search(r'V224F_SAVED (\{.*\})', next(e['data'] for e in json.load(open(logp(M), encoding='utf-8'))
                                                             if 'V224F_SAVED' in e['data'])).group(1)) if splice else dict(sv)
    g = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': svm,
         '_canary': canary, '_k2_receipt': k2rec}
    if gcr is not None:
        g['_gcr'] = gcr
    for f in ('submission.csv', 'submission_H.csv', 'hf_receipt.json', 'v224f_receipt.json', 'submission_fleet2g.csv'):
        if os.path.isfile(os.path.join(M, f)):
            os.remove(os.path.join(M, f))
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        exec(compile(text.replace('/kaggle/working', M.replace('\\', '/')), 'HF', 'exec'), g)
    for z_ in g.get('_parts') or []:
        z_.close()
    g.clear()
    new = b.getvalue().strip().splitlines()
    if splice:
        EE = json.load(open(logp(M), encoding='utf-8'))
        ix = [i for i, e in enumerate(EE) if any(k in e['data'] for k in ('V224FS_F2G', 'HF_BLEND_OK', 'HF_FALLBACK', 'CANARY_FIRED fleet',
                                                                          'HF_COMPLETE', 'V224F_DONE'))]
        at = ix[0] if ix else len(EE)
        EE = EE[:at] + [dict(EE[-1], data=ln + '\n') for ln in new] + (EE[ix[-1] + 1:] if ix else [])
        json.dump(EE, open(logp(M), 'w', encoding='utf-8'))
    return [ln.split(' ', 1)[0] for ln in new]


GCR0 = CANARY if BASE == 'flrm' else None
printed = run_hf(OUT, dict(fr, clean=True), CAN0, GCR0)
assert printed == ['V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'], printed
print('MOCK OUT', OUT, '| base', os.path.relpath(REF_B, ROOT), '| fleet', os.path.relpath(REF_F, ROOT), '| HF-AO printed:', printed)


def gate(out, build=BJ, extra=()):
    a = [sys.executable, os.path.join(V, 'v224aos_check.py'), out, '--wf', '0.2', '--build', build, '--ref-exact', REF_X,
         '--ref-flip', REF_FLIP] + (['--ref-flrm', REF_FLRM] if json.load(open(build))['base'] == 'flrm' else []) + list(extra)
    r = subprocess.run(a, capture_output=True, text=True, encoding='utf-8')
    lines = r.stdout.strip().splitlines()
    last = lines[-1] if lines else (r.stderr.strip().splitlines() or [''])[-1]
    return r.returncode, last, [ln.strip() for ln in lines if ln.startswith('  FAIL ')], r.stdout


rc, last, fails, full = gate(OUT)
print(full)
res = [rc == 0 and 'V224AOS_CHECK PASS' in last]
print('GATE clean mock:', 'PASS' if res[-1] else 'FAIL', last)
rc, last, fails, full = gate(OUT, extra=['--ref-s', FFSR_OUT])
res.append(rc == 0 and 'V224AOS_CHECK PASS' in last and '--ref-s: S group mean' in full)
print('GATE clean mock + --ref-s mock_out_v224ffsr20 (same 5 S rows):', 'PASS' if res[-1] else 'FAIL', last,
      [ln.strip() for ln in full.splitlines() if '--ref-s' in ln][:1])


def mutate(name, fn, build=BJ, extra=()):
    M = OUT + '_neg'
    if os.path.isdir(M):
        shutil.rmtree(M)
    shutil.copytree(OUT, M); fn(M)
    rc, last, fails, _ = gate(M, build=build, extra=extra)
    ok = rc != 0
    res.append(ok)
    print(f'NEG {name:<58} {"as expected (FAIL)" if ok else "UNEXPECTED PASS"} | {last}')
    for ln in fails:
        print('      ' + ln[:400])
    shutil.rmtree(M)


def jedit(M, fname, **kv):
    p = os.path.join(M, fname); d = json.load(open(p, encoding='utf-8')); d.update(kv); json.dump(d, open(p, 'w'), indent=1)


def log_sub(a, b, every=False):
    def f(M):
        EE = json.load(open(logp(M), encoding='utf-8'))
        hit = [i for i, e in enumerate(EE) if a in e['data']]; assert hit, a
        for i in (hit if every else hit[:1]):
            EE[i]['data'] = EE[i]['data'].replace(a, b)
        json.dump(EE, open(logp(M), 'w', encoding='utf-8'))
    return f


def bump(path, col='ACL', row=1, by=1e-3):
    def f(M):
        d = rd(os.path.join(M, path)); d.loc[row, col] = d.loc[row, col] + by; d.to_csv(os.path.join(M, path), index=False)
    return f


def driver_on(M):                                     # the time-aware driver ran (FF20's FLIP_TTA_FORCED + FLIP_TA_REPORT lines)
    EF20 = json.load(open(os.path.join(REF_F, 'rsna-sub-v224ff20.log'), encoding='utf-8'))
    fl = next(e['data'] for e in EF20 if e['data'].startswith('FLIP_TTA_FORCED'))
    rp = next(e['data'] for e in EF20 if e['data'].startswith('FLIP_TA_REPORT'))
    EE = json.load(open(logp(M), encoding='utf-8'))
    i = next(i for i, e in enumerate(EE) if e['data'].startswith('FLIP_TTA_FORCED')); EE[i]['data'] = fl
    j = next(i for i, e in enumerate(EE) if 'OURS_STAGE_COMPLETE' in e['data'])
    EE.insert(j + 1, dict(EE[j], data=rp)); json.dump(EE, open(logp(M), 'w', encoding='utf-8'))


def flip_off_w1(M):                                   # worker 1 ran WITHOUT --flip-tta (no READY line, no flip_tta block)
    p = os.path.join(M, 'worker1_log.txt')
    t = open(p, encoding='utf-8', errors='replace').read()
    open(p, 'w', encoding='utf-8').write('\n'.join(ln for ln in t.split('\n') if 'FLIP_TTA_READY' not in ln))
    sp = os.path.join(M, 'submission_receipt.json'); d = json.load(open(sp)); d['workers'][1].pop('flip_tta'); json.dump(d, open(sp, 'w'), indent=1)


def npz_s(M):
    for gg in (0, 1):
        z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % gg))); z['P'] = z['P'].copy(); z['P'][-1] *= 0.5
        np.savez(os.path.join(M, '_fleet_%d.npz' % gg), **z)


def k14_present(M):                                   # the fleet ran K14 20 + S 5 (npz, fleet solo, receipt, log consistent)
    zs = []
    for gg in (0, 1):
        r = dict(np.load(os.path.join(REF_F, '_fleet_%d.npz' % gg), allow_pickle=False)); nm_ = [str(x) for x in r['names']]
        for k in ('P', 'PA', 'PB'):
            r[k] = np.concatenate([r[k], np.stack([r[k][nm_.index(d)] for d in DT])])
        r['names'] = np.array(nm_ + [smap[d] for d in DT]); r['n_keep'] = np.array(25); zs.append(r)
        np.savez(os.path.join(M, '_fleet_%d.npz' % gg), **r)
    P2 = np.concatenate([r['P'] for r in zs], axis=1).astype(np.float64); i2 = np.concatenate([r['sids'] for r in zs]).astype(str)
    so = pd.DataFrame(P2.mean(0), columns=TG, index=i2).reindex(test_ids); so.insert(0, 'StudyInstanceUID', test_ids)
    so.to_csv(os.path.join(M, 'submission_fleetsolo.csv'), index=False)
    nn = [str(x) for x in zs[0]['names']]; f0 = json.load(open(os.path.join(REF_F, 'fleet_receipt.json')))
    p = os.path.join(M, 'fleet_receipt.json'); d = json.load(open(p))
    d.update(models=25, names=nn, n_keep=[25, 25], members_filter=nn)
    d['config'] = dict(d['config'], MEMBERS=nn, TTA={x: 'AB' for x in nn})
    d['tta'] = dict(d['tta'], modes={x: 'AB' for x in nn}, fwd=dict(f0['tta']['fwd'], **{smap[x]: f0['tta']['fwd'][x] for x in DT}))
    json.dump(d, open(p, 'w'), indent=1)
    log_sub('RESULT OK fleet stage clean: 5 member(s)', 'RESULT OK fleet stage clean: 25 member(s)')(M)


def not_started(M):                                   # the start guard refused: no npz / solo, HF-AO falls back (Synovitis canary)
    for f in ('_fleet_0.npz', '_fleet_1.npz', 'submission_fleetsolo.csv', 'submission_w40.csv', 'submission_w50.csv', 'submission_w60.csv'):
        if os.path.isfile(os.path.join(M, f)):
            os.remove(os.path.join(M, f))
    why = 'only %.1f min before the 8.3 h wall (< %.1f min): stage not started' % (GM - 8.0, GM)
    jedit(M, 'fleet_receipt.json', clean=False, why=why, kernel_age_h=round(8.3 - (GM - 8.0) / 60.0, 2), left_min=GM - 8.0)
    log_sub('RESULT OK fleet stage clean: 5 member(s)', 'RESULT FLEET STAGE NOT CLEAN (%s) was: 5 member(s)' % why)(M)
    p = run_hf(M, dict(fr, clean=False, why=why), CAN0, GCR0)
    assert 'HF_FALLBACK' in p and (rd(os.path.join(M, 'submission.csv'))['Synovitis'] == 0.5).all(), p


def reblend(wf):
    def f(M):
        Hh = rd(os.path.join(M, 'submission_V224.csv'))
        F_ = rd(os.path.join(M, 'submission_fleet2g.csv')).set_index('StudyInstanceUID').loc[Hh['StudyInstanceUID']]
        o = Hh.copy()
        for c in labels:
            o[c] = ((1.0 - wf) * pd.Series(Hh[c].values.astype(np.float64)).rank(method='average', pct=True).values
                    + wf * pd.Series(F_[c].values.astype(np.float64)).rank(method='average', pct=True).values).astype(np.float32)
        o.to_csv(os.path.join(M, 'submission.csv'), index=False)
        jedit(M, 'hf_receipt.json', submission_sha256=sha_(os.path.join(M, 'submission.csv')))
    return f


def s_sha_wrong(M):                                   # the fleet loaded a different file under one S name (manifest_sha16 != build)
    p = os.path.join(M, 'fleet_receipt.json'); d = json.load(open(p))
    d['manifest_sha16'][B['s'][2]] = '0123456789abcdef'; json.dump(d, open(p, 'w'), indent=1)


def guard_51(M):                                      # built / run with the FF start budget 5.1 h (MIN_START_MIN 192)
    log_sub('"start_budget_h": %s' % json.dumps(GH), '"start_budget_h": 5.1')(M); log_sub('"min_start_min": %s' % json.dumps(GM), '"min_start_min": 192.0')(M)
    p = os.path.join(M, 'fleet_receipt.json'); d = json.load(open(p)); d['config']['MIN_START_MIN'] = 192.0; json.dump(d, open(p, 'w'), indent=1)


def fracture_canary(M):                               # flrm: RM_GUARD fired / flip: the V224 CANARY fired -> Fracture 0.5 in H
    Hs = pd.read_csv(os.path.join(M, 'submission_V224.csv'), dtype=str, keep_default_na=False); Hs['Fracture'] = '0.5'
    Hs.to_csv(os.path.join(M, 'submission_V224.csv'), index=False)
    log_sub('"sha256": "%s"' % H_SHA, '"sha256": "%s"' % sha_(os.path.join(M, 'submission_V224.csv')))(M)
    if BASE == 'flrm':
        gcr = dict(CANARY, fired=['r224 mirror->Fracture'], rm_guard=dict(CANARY['rm_guard'], mirror_ok=False, why='receipt'))
        json.dump(gcr, open(os.path.join(M, 'canary_receipt.json'), 'w'), indent=1)
        log_sub('RM_GUARD_CLEAN', 'RM_GUARD_FIRED r224 used without a valid mirror receipt: Fracture set to 0.5 for all 3 rows')(M)
        p = run_hf(M, dict(fr, clean=True), CAN0, gcr)
    else:
        can = dict(CAN0, fired=['r224->Fracture'], r224_used=False)
        json.dump(can, open(os.path.join(M, 'canary_receipt.json'), 'w'), indent=1)
        log_sub('CANARY_CLEAN r224 used, no column blanked', 'CANARY_FIRED r224 not used (R224_OK=False): Fracture set to 0.5 for all 3 rows')(M)
        p = run_hf(M, dict(fr, clean=True), can, None)
    assert 'HF_BLEND_OK' in p and (rd(os.path.join(M, 'submission.csv'))['Fracture'] == 0.5).all(), p


mutate('time-aware driver ran (FF20 FLIP_TTA_FORCED + FLIP_TA_REPORT)', driver_on)
mutate('flip off on worker 1 (no FLIP_TTA_READY, no flip_tta block)', flip_off_w1)
mutate('R7 = the V224-exact (plain) commit R7 rows', lambda M: shutil.copyfile(os.path.join(REF_X, 'ours_sub.csv'), os.path.join(M, 'ours_sub.csv')))
mutate('submission_V224.csv cell +1e-3', bump('submission_V224.csv'))
mutate('submission.csv cell +1e-3', bump('submission.csv'))
mutate('an S member row x0.5 in the npz', npz_s)
mutate('K14 present: the fleet ran K14 20 + S 5', k14_present)
mutate('canary: fleet not started by the %.1f h guard (HF_FALLBACK)' % GH, not_started)
mutate('wrong weight: w_f .30 (receipt sha ok)', reblend(0.3))
mutate('weight not applied: w_f 0 (receipt sha ok)', reblend(0.0))
mutate('S file version: manifest_sha16 of one S != the build', s_sha_wrong)
mutate('start guard 5.1 h (SAVED + K2 config MIN_START_MIN 192)', guard_51)
mutate(('RM_GUARD fired' if BASE == 'flrm' else 'V224 CANARY fired') + ': Fracture 0.5 kept through the S blend', fracture_canary)
# the kernel dir: one byte changed in a base cell of a copy (build json sha updated, so only the base-cell proof can catch it)
KD2 = tempfile.mkdtemp(prefix='aos_kd_')
shutil.copytree(KD, os.path.join(KD2, 'k'))
nbp = os.path.join(KD2, 'k', 'rsna_sub_%s.ipynb' % ST); nbx = json.load(open(nbp, encoding='utf-8'))
nbx['cells'][5]['source'] = src(nbx['cells'][5]) + ' '
open(nbp, 'w', encoding='utf-8', newline='').write(json.dumps(nbx, ensure_ascii=False, separators=(',', ':')))
bx = json.load(open(os.path.join(KD2, 'k', 'BUILD_V224FS.json'))); bx['notebook_sha256'] = sha_(nbp)
json.dump(bx, open(os.path.join(KD2, 'k', 'BUILD_V224FS.json'), 'w'), indent=1)
mutate('notebook: base cell 5 + 1 byte (build sha updated)', lambda M: None, build=os.path.join(KD2, 'k', 'BUILD_V224FS.json'))
shutil.rmtree(KD2, ignore_errors=True)
mutate('cross: this mock under the %s build json' % OTHER, lambda M: None, build=os.path.join(T, 'standin_' + OTHER, 'BUILD_V224FS.json'))
assert os.path.isdir(FFSR_OUT), 'run tests/mock_fs_output.py ffsr first'


def neg_out(name, out, build=BJ, extra=()):
    rc, last, fails, _ = gate(out, build=build, extra=extra)
    ok = rc != 0
    res.append(ok)
    print(f'NEG {name:<58} {"as expected (FAIL)" if ok else "UNEXPECTED PASS"} | {last}')
    for ln in fails:
        print('      ' + ln[:400])


neg_out('cross: the FFSR mock (time-aware flip, FF20 base) under this build', FFSR_OUT)
SRB = tempfile.mkdtemp(prefix='aos_refs_')
shutil.copytree(FFSR_OUT, os.path.join(SRB, 'r'), ignore=shutil.ignore_patterns('*.csv', '*.log', '*.json'))
npz_s(os.path.join(SRB, 'r'))
neg_out('--ref-s whose S rows differ (an S row x0.5)', OUT, extra=['--ref-s', os.path.join(SRB, 'r')])
shutil.rmtree(SRB, ignore_errors=True)
if BASE == 'flrm':
    a = [sys.executable, os.path.join(V, 'v224aos_check.py'), OUT, '--wf', '0.2', '--build', BJ, '--ref-exact', REF_X, '--ref-flip', REF_FLIP]
    r = subprocess.run(a, capture_output=True, text=True, encoding='utf-8')
    res.append(r.returncode != 0 and 'ref-flrm' in r.stderr)
    print(f'NEG {"flrm build without --ref-flrm (refused)":<58} {"as expected (refused)" if res[-1] else "UNEXPECTED"} | rc {r.returncode}')

# ---- 5. HF-AO cell-level tests (the kernel's own cell, not the gate) ------------------------------------------------------------
cell = []


def cell_case(name, ok, info=''):
    cell.append(bool(ok))
    print(f'CELL {"OK  " if ok else "FAIL"} {name} {info}')


W = tempfile.mkdtemp(prefix='aos_cell_')


def fresh(tag):
    M = os.path.join(W, tag); shutil.copytree(OUT, M); return M


Ma, Mb = fresh('ao_clean'), fresh('ff_clean')
pa = run_hf(Ma, dict(fr, clean=True), CAN0, GCR0, HF, splice=False)
pb = run_hf(Mb, dict(fr, clean=True), CAN0, GCR0, HF_FF, splice=False)
cell_case('clean path: HF-AO submission.csv / submission_fleet2g.csv / submission_H.csv == the f / ff HF-S (FFSR text), byte for byte',
          all(open(os.path.join(Ma, f), 'rb').read() == open(os.path.join(Mb, f), 'rb').read()
              for f in ('submission.csv', 'submission_fleet2g.csv', 'submission_H.csv')) and pa == pb == ['V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'])
ha = json.load(open(os.path.join(Ma, 'hf_receipt.json'))); hb = json.load(open(os.path.join(Mb, 'hf_receipt.json')))
cell_case('clean path: hf_receipt differs only by tag / sha / time + the AO key fired_before_hf []',
          ha.get('fired_before_hf') == [] and 'fired_before_hf' not in hb and
          {k: v for k, v in ha.items() if k not in ('tag', 'fired_before_hf', 'total_seconds', 'submission_sha256')} ==
          {k: v for k, v in hb.items() if k not in ('tag', 'total_seconds', 'submission_sha256')})
for tag, can, gcr in (('rmguard', CAN0, dict(CANARY if BASE == 'flrm' else CAN0, fired=['r224 mirror->Fracture'])),
                      ('canary', dict(CAN0, fired=['r224->Fracture'], r224_used=False), GCR0)):
    Ma, Mb = fresh('ao_' + tag), fresh('ff_' + tag)
    for M_ in (Ma, Mb):
        Hs = pd.read_csv(os.path.join(M_, 'submission_V224.csv'), dtype=str, keep_default_na=False); Hs['Fracture'] = '0.5'
        Hs.to_csv(os.path.join(M_, 'submission_V224.csv'), index=False)
    svx = dict(sv, sha256=sha_(os.path.join(Ma, 'submission_V224.csv')))
    sv_keep, sv = sv, svx
    pa = run_hf(Ma, dict(fr, clean=True), can, gcr, HF, splice=False)
    pb = run_hf(Mb, dict(fr, clean=True), can, gcr, HF_FF, splice=False)
    sv = sv_keep
    fa = rd(os.path.join(Ma, 'submission.csv'))['Fracture']; fb = rd(os.path.join(Mb, 'submission.csv'))['Fracture']
    ra = json.load(open(os.path.join(Ma, 'hf_receipt.json')))
    if tag == 'rmguard':
        cell_case('RM_GUARD fired (_gcr r224 mirror->Fracture, _canary clean): HF-AO keeps Fracture 0.5 after the S blend, blanked '
                  '[Fracture], fired_before_hf lists it', (fa == 0.5).all() and ra.get('blanked') == ['Fracture'] and ra.get('mode') == 'HF'
                  and 'r224 mirror->Fracture' in ra.get('fired_before_hf', []), list(fa))
        cell_case('the same input through the f / ff HF-S: Fracture is UN-BLANKED by the S blend (the gap HF-AO closes)',
                  not (fb == 0.5).all(), list(fb))
    else:
        cell_case('V224 CANARY fired (_canary r224->Fracture): HF-AO and HF-S both keep Fracture 0.5', (fa == 0.5).all() and (fb == 0.5).all())
Mc = fresh('ao_fallback_rm')
Hs = pd.read_csv(os.path.join(Mc, 'submission_V224.csv'), dtype=str, keep_default_na=False); Hs['Fracture'] = '0.5'
Hs.to_csv(os.path.join(Mc, 'submission_V224.csv'), index=False)
sv_keep, sv = sv, dict(sv, sha256=sha_(os.path.join(Mc, 'submission_V224.csv')))
pc = run_hf(Mc, dict(fr, clean=False, why='stage not started'), CAN0, dict(CANARY if BASE == 'flrm' else CAN0, fired=['r224 mirror->Fracture']), HF, splice=False)
sv = sv_keep
sc_ = rd(os.path.join(Mc, 'submission.csv')); rc_ = json.load(open(os.path.join(Mc, 'hf_receipt.json')))
cell_case('fleet not clean + a Fracture canary: HF_FALLBACK, Synovitis AND Fracture 0.5, every other column == H',
          'HF_FALLBACK' in pc and (sc_['Synovitis'] == 0.5).all() and (sc_['Fracture'] == 0.5).all() and rc_.get('blanked') == ['Fracture', 'Synovitis']
          and np.allclose(sc_[[c for c in labels if c not in ('Synovitis', 'Fracture')]].values,
                          rd(os.path.join(Mc, 'submission_V224.csv'))[[c for c in labels if c not in ('Synovitis', 'Fracture')]].values), pc)
shutil.rmtree(W, ignore_errors=True)
print(f'CELL_AO {"PASS" if all(cell) else "FAIL"} {sum(cell)}/{len(cell)}')
print(f'MOCK_AO_GATE {MODE} {"PASS" if all(res) and all(cell) else "FAIL"} {sum(res)}/{len(res)} (2 clean PASS + {len(res) - 2} negatives FAIL '
      f'as expected) + CELL {sum(cell)}/{len(cell)}')
