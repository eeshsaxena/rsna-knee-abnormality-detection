# mock_fs_output.py (2026-10-15 dir, 10-14, Claude subagent): builds a MOCK V224-F-S commit output from the REAL V224-F20 commit
# output (out_v224f20_v1) so that v224fs_check.py can be proven before any S checkpoint exists, then runs the gate on it and on
# mutated copies (negative tests). Offline, laptop, no GPU, no Kaggle.
#   the 25-member worker npz = the commit's 20 K14 rows + 5 stand-in S rows (copies of the 5 DT rows under the S names of the
#   --standin build); fleet solo = the fleet cell's equal-mean formula; the HF-S cell of the stand-in notebook is EXECUTED on it
#   (its printed lines replace the V224-F HF lines of the commit log); V224F_SAVED knobs / fleet receipt / s-merge line as the
#   V224-F-S kernel would write them.
#   usage: python tests/mock_fs_output.py [f|ff|ffsr] (f: V224-F-S from out_v224f20_v1 -> tests/mock_out_v224fs20/, gate v224fs_check;
#          ff: V224-FF-S from v224ff/out_v224ff20_v1 -> tests/mock_out_v224ffs20/, gate v224ffs_check incl. the flip checks;
#          ffsr (FFSR gate test 10-15): V224-FFSR20 = the --replace build (F = S alone, s_share 1.0, MEMBERS = the 5 S only) from
#          v224ff/out_v224ff20_v1 -> tests/mock_out_v224ffsr20/, gate v224ffs_check with tests/standin_v224ffsr20/BUILD_V224FS.json;
#          the 5-member npz = the 5 DT rows of the commit under the S names; its own negative set (the f / ff set is unchanged).
#          The cross negatives use tests/mock_out_v224ffs20/, so run `mock_fs_output.py ff` before `ffsr`.)
import os, sys, json, time, shutil, hashlib, io, contextlib, copy, subprocess, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE); ROOT = os.path.normpath(os.path.join(V, '..', '..'))
BASE = sys.argv[1] if len(sys.argv) > 1 else 'f'
assert BASE in ('f', 'ff', 'ffsr'), BASE
RS = BASE == 'ffsr'                               # REPLACE (FFSR20): the fleet runs the 5 S members only, F = their mean
FT, ST = ('v224f20', 'v224fs20') if BASE == 'f' else ('v224ff20', 'v224ffsr20' if RS else 'v224ffs20')
REF = os.path.join(V, 'out_v224f20_v1') if BASE == 'f' else os.path.join(ROOT, 'session_2026-10-15', 'v224ff', 'out_v224ff20_v1')
OUT = os.path.join(HERE, 'mock_out_' + ST); KD = os.path.join(HERE, 'standin_' + ST)
B = json.load(open(os.path.join(KD, 'BUILD_V224FS.json')))
NB = json.load(open(os.path.join(KD, 'rsna_sub_%s.ipynb' % ST), encoding='utf-8'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
HF = [src(c) for c in NB['cells'] if src(c).startswith('# [eeshsaxena V224-F-S HF')][0]
FLEET = [src(c) for c in NB['cells'] if src(c).startswith('K2_CFG = {')][0]
K2 = eval(FLEET.split('\n', 1)[0][len('K2_CFG = '):FLEET.split('\n', 1)[0].index('   # literal')], {})
NM = len(B['k14']) + len(B['s'])                  # 25 (f, ff); 5 (ffsr)
assert NM == (5 if RS else 25) and (not RS or (B['replace'] is True and B['k14'] == [] and B['s_share'] == 1.0
                                                and K2['MEMBERS'] == B['s'] and B['base'] == 'ff')), (NM, B.get('replace'))
sample = pd.read_csv(os.path.join(ROOT, '_data', 'sample_submission.csv'), dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; test_ids = sample['StudyInstanceUID'].tolist()
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
shutil.copytree(REF, OUT, ignore=shutil.ignore_patterns('__pycache__'))
# ---- npz (25; ffsr: the 5 S only) + fleet solo
if RS:                                            # S = the commit's 5 DT rows under the S names, in the build's MEMBERS order
    DT = [s.replace('_s1047r7tb_', '_s1043f75oai_') for s in B['s']]
else:
    DT = [m for m in B['k14'] if '_s1043f75oai_' in m]
smap = {d: d.replace('_s1043f75oai_', '_s1047r7tb_') for d in DT}
assert sorted(smap.values()) == sorted(B['s'])
parts = []
for g in (0, 1):
    z = dict(np.load(os.path.join(REF, '_fleet_%d.npz' % g), allow_pickle=False)); nm = [str(x) for x in z['names']]
    if g == 0:
        REFN = list(nm)                           # the commit's 20 K14 names (worker order)
    if RS:
        for k in ('P', 'PA', 'PB'):
            z[k] = np.stack([z[k][nm.index(d)] for d in DT])
        z['names'] = np.array([smap[d] for d in DT])
    else:
        z['P'] = np.concatenate([z['P'], np.stack([z['P'][nm.index(d)] for d in DT])]); z['names'] = np.array(nm + [smap[d] for d in DT])
    z['n_keep'] = np.array(NM); parts.append(z); np.savez(os.path.join(OUT, '_fleet_%d.npz' % g), **z)
P = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64); ids = np.concatenate([z['sids'] for z in parts]).astype(str)
solo = pd.DataFrame(P.mean(0), columns=TG, index=ids).reindex(test_ids)
solo.insert(0, 'StudyInstanceUID', test_ids); solo.to_csv(os.path.join(OUT, 'submission_fleetsolo.csv'), index=False)
# ---- fleet receipt
fr = json.load(open(os.path.join(REF, 'fleet_receipt.json')))
names = [str(x) for x in parts[0]['names']]
if RS:
    assert names == K2['MEMBERS']
fr.update(models=NM, names=names, n_keep=[NM, NM])
fr['config'] = dict(fr['config'], MEMBERS=K2['MEMBERS'], TTA=K2['TTA'], MIN_START_MIN=K2['MIN_START_MIN'], WALL_H=K2['WALL_H'])
if RS:
    K14_LINE = '%d member(s) %r' % (len(REFN), REFN); S_LINE = '%d member(s) %r' % (NM, names)
    assert fr['why'].count(K14_LINE) == 1
    fr['tta'] = dict(fr['tta'], modes={m: 'AB' for m in names}, fwd={smap[d]: fr['tta']['fwd'][d] for d in DT})
    fr.update(members_filter=list(K2['MEMBERS']), tta_expected={m: 'AB' for m in names}, tta_kernel={m: 'AB' for m in names},
              bench=[{smap[d]: b[d] for d in DT} if b else b for b in fr['bench']], why=fr['why'].replace(K14_LINE, S_LINE))
else:
    fr['tta'] = dict(fr['tta'], modes={m: 'AB' for m in names}, fwd={m: v for m, v in zip(names, list(fr['tta']['fwd'].values()) +
                                                                                 [fr['tta']['fwd'][d] for d in DT])})
json.dump(fr, open(os.path.join(OUT, 'fleet_receipt.json'), 'w'), indent=1)
# ---- log: SAVED knobs, s-merge line, 25-member (ffsr: 5-member) clean line; then execute HF-S and splice its printed lines
logp = os.path.join(OUT, 'rsna-sub-%s.log' % FT); E = json.load(open(logp, encoding='utf-8'))
os.rename(logp, os.path.join(OUT, 'rsna-sub-%s.log' % ST)); logp = os.path.join(OUT, 'rsna-sub-%s.log' % ST)
if BASE in ('ff', 'ffsr'):                        # the FF-S kernel prints its built flip budget (fix 10-15: 4.9 = FF20's, no rewrite)
    kf = next(i for i, e in enumerate(E) if 'FLIP_TTA_FORCED' in e['data'])
    assert E[kf]['data'].count('"flip_budget_h": 4.9') == 1
    E[kf]['data'] = E[kf]['data'].replace('"flip_budget_h": 4.9', '"flip_budget_h": %r' % B['flip_budget_h'])
ks = next(i for i, e in enumerate(E) if 'V224F_SAVED' in e['data'])
m = re.search(r'V224F_SAVED (\{.*\})', E[ks]['data']); sv = json.loads(m.group(1))
sv.update(tag=B['tag'], start_budget_h=B['start_budget_h'], min_start_min=B['min_start_min'], members=NM,
          groups={'K14': len(B['k14']), 'S': len(B['s'])}, s_share=B['s_share'], s_ds=B['s_ds'])
E[ks]['data'] = E[ks]['data'].replace(m.group(1), json.dumps(sv))
kk = next(i for i, e in enumerate(E) if '[k12-merge' in e['data'] and 'RESULT OK merged fleet dir' in e['data'])
E.insert(kk + 1, dict(E[kk], data='[s-merge 07:36:47] RESULT OK merged fleet dir + S: 30 checkpoints (5 S from /kaggle/input/datasets/'
                                  'eeshsaxena/%s, sha256 verified, 0.4 s)\n' % B['s_ds']))
kc = next(i for i, e in enumerate(E) if 'RESULT OK fleet stage clean: 20 member' in e['data'])
if RS:                                            # REPLACE: the fleet lines name the 5 S members (S-MERGE still links all 30)
    assert E[kc]['data'].count('RESULT OK fleet stage clean: ' + K14_LINE) == 1
    E[kc]['data'] = E[kc]['data'].replace('RESULT OK fleet stage clean: ' + K14_LINE, 'RESULT OK fleet stage clean: ' + S_LINE)
    nw = 0
    for e in E:
        mm = re.match(r'(\[fleet \d\] members: )(.*)$', e['data'], re.S)
        if mm:
            assert mm.group(2).strip() == ', '.join('%s[KneeFleetNetTTR]' % n for n in REFN)
            e['data'] = mm.group(1) + ', '.join('%s[KneeFleetNetTTR]' % n for n in names) + '\n'; nw += 1
    assert nw == 4
    kr = [i for i, e in enumerate(E) if e['data'].startswith('[k2-fleet] receipt {')]
    assert len(kr) == 1
    E[kr[0]]['data'] = '[k2-fleet] receipt ' + json.dumps(fr) + '\n'
else:
    E[kc]['data'] = E[kc]['data'].replace('RESULT OK fleet stage clean: 20 member', 'RESULT OK fleet stage clean: 25 member')
H = pd.read_csv(os.path.join(OUT, 'submission_V224.csv'), dtype={'StudyInstanceUID': str})
g = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': sv,
     '_canary': {'fired': [], 'r224_used': True}, '_k2_receipt': dict(fr, clean=True)}
for f in ('submission.csv', 'submission_H.csv', 'hf_receipt.json', 'v224f_receipt.json'):
    os.remove(os.path.join(OUT, f))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(HF.replace('/kaggle/working', OUT.replace('\\', '/')), 'HF-S', 'exec'), g)
new = buf.getvalue()
hf_i = [i for i, e in enumerate(E) if any(k in e['data'] for k in ('HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'))]
assert hf_i, 'no HF lines in the reference log'
E = E[:hf_i[0]] + [dict(E[hf_i[0]], data=ln + '\n') for ln in new.strip().splitlines()] + E[hf_i[-1] + 1:]
json.dump(E, open(logp, 'w', encoding='utf-8'))
print('MOCK OUT', OUT, '| HF-S printed:', [ln.split(' ', 1)[0] for ln in new.strip().splitlines()])


def gate(out, tag, build=None):
    build = build or os.path.join(KD, 'BUILD_V224FS.json')
    if BASE in ('ff', 'ffsr'):
        r = subprocess.run([sys.executable, os.path.join(V, 'v224ffs_check.py'), out, '--wf', '0.2', '--build', build,
                            '--ref-flip', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224flip_v1'),
                            '--ref-exact', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'), '--ref-fleet', REF],
                           capture_output=True, text=True, encoding='utf-8')
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
        return r.returncode, last, [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith('FAIL')], r.stdout
    r = subprocess.run([sys.executable, os.path.join(V, 'v224fs_check.py'), out, '--wf', '0.2', '--build', build,
                        '--ref-v224', os.path.join(ROOT, 'session_2026-10-14', 'v224', 'out_v224_v1'), '--ref-fleet', REF],
                       capture_output=True, text=True, encoding='utf-8')
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    fails = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith('FAIL')]
    return r.returncode, last, fails, r.stdout


rc, last, fails, full = gate(OUT, 'clean')
print(full)
res = [rc == 0 and 'PASS' in last]
print('GATE clean mock:', 'PASS' if res[-1] else 'FAIL', last)


def mutate(name, fn, allfails=False):
    M = OUT + '_neg'
    if os.path.isdir(M):
        shutil.rmtree(M)
    shutil.copytree(OUT, M); fn(M)
    rc, last, fails, _ = gate(M, name)
    ok = rc != 0
    res.append(ok)
    print(f'NEG {name:<44} {"as expected (FAIL)" if ok else "UNEXPECTED PASS"} | {last} | first FAIL: {fails[:1]}')
    if allfails:                                   # ffsr: every FAIL line (the nested v224fs_check FAIL list sits in the first one)
        for ln in fails:
            print('      ' + ln[:3000])
    shutil.rmtree(M)


def bump(path, col='ACL', row=1, by=1e-3):
    def f(M):
        d = pd.read_csv(os.path.join(M, path), dtype={'StudyInstanceUID': str}); d.loc[row, col] = d.loc[row, col] + by
        d.to_csv(os.path.join(M, path), index=False)
    return f


def npz_s(M):
    for gg in (0, 1):
        z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % gg))); z['P'] = z['P'].copy(); z['P'][-1] *= 0.5
        np.savez(os.path.join(M, '_fleet_%d.npz' % gg), **z)


def log_sub(a, b):
    def f(M):
        p = os.path.join(M, 'rsna-sub-%s.log' % ST); EE = json.load(open(p, encoding='utf-8'))
        hit = [i for i, e in enumerate(EE) if a in e['data']]; assert hit, a
        EE[hit[0]]['data'] = EE[hit[0]]['data'].replace(a, b, 1); json.dump(EE, open(p, 'w', encoding='utf-8'))
    return f


def equal_mean_f2g(M):                                         # the bug FX7 fixes: F = the equal mean of all 25 members
    s_ = pd.read_csv(os.path.join(M, 'submission_fleetsolo.csv'), dtype={'StudyInstanceUID': str})
    s_.to_csv(os.path.join(M, 'submission_fleet2g.csv'), index=False)


if not RS:
    mutate('submission_fleet2g.csv cell +1e-3', bump('submission_fleet2g.csv'))
    mutate('submission.csv cell +1e-3', bump('submission.csv'))
    mutate('fleet2g = the equal 25-member mean (FX7 bug)', equal_mean_f2g)
    mutate('an S member row changed in the npz', npz_s)
    mutate('V224F_SAVED members 20', log_sub('"members": 25', '"members": 20'))
    mutate('no s-merge line', log_sub('[s-merge 07:36:47] RESULT OK', '[x-merge 07:36:47] RESULT OK'))
    mutate('fleet2g file missing', lambda M: os.remove(os.path.join(M, 'submission_fleet2g.csv')))
else:
    # ---- ffsr negatives (FFSR gate test 10-15). Helpers: sha-consistent rewrites, so that only the semantic check can catch them.
    sha_ = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    rd = lambda p: pd.read_csv(p, dtype={'StudyInstanceUID': str})

    def jedit(M, fname, **kv):
        p = os.path.join(M, fname); d = json.load(open(p, encoding='utf-8')); d.update(kv); json.dump(d, open(p, 'w'), indent=1)

    def write_f2g(M, F_):                                      # F_: DataFrame index test_ids, columns labels; + hf_receipt f2g sha
        o = F_.loc[test_ids, labels].reset_index().rename(columns={'index': 'StudyInstanceUID'}); o.columns = cols
        o.to_csv(os.path.join(M, 'submission_fleet2g.csv'), index=False)
        p = os.path.join(M, 'hf_receipt.json'); d = json.load(open(p)); d['f2g'] = dict(d['f2g'], sha256=sha_(os.path.join(M, 'submission_fleet2g.csv')))
        json.dump(d, open(p, 'w'), indent=1)

    def reblend(wf):                                           # submission.csv = (1 - wf) pct(V224) + wf pct(fleet2g) as the HF cell
        def f(M):                                              # computes it (pandas average-tie pct, float32); receipt sha updated
            Hh = rd(os.path.join(M, 'submission_V224.csv'))
            F_ = rd(os.path.join(M, 'submission_fleet2g.csv')).set_index('StudyInstanceUID').loc[Hh['StudyInstanceUID']]
            o = Hh.copy()
            for c in labels:
                o[c] = ((1.0 - wf) * pd.Series(Hh[c].values.astype(np.float64)).rank(method='average', pct=True).values
                        + wf * pd.Series(F_[c].values.astype(np.float64)).rank(method='average', pct=True).values).astype(np.float32)
            o.to_csv(os.path.join(M, 'submission.csv'), index=False)
            jedit(M, 'hf_receipt.json', submission_sha256=sha_(os.path.join(M, 'submission.csv')))
        return f

    def k14_f(M):                                              # the K14 20-member mean = the V224-FF20 commit fleet solo
        return rd(os.path.join(REF, 'submission_fleetsolo.csv')).set_index('StudyInstanceUID').loc[test_ids, labels]

    def f2g_is_k14(M):                                         # F = K14 (wrong group), fleet2g + submission + shas consistent
        write_f2g(M, k14_f(M)); reblend(0.2)(M)

    def f2g_is_ffs_mix(M):                                     # F = .5 K14 + .5 S (the FF-S F) inside the replace kernel
        G = rd(os.path.join(M, 'submission_fleet2g.csv')).set_index('StudyInstanceUID').loc[test_ids, labels]
        write_f2g(M, 0.5 * k14_f(M) + 0.5 * G); reblend(0.2)(M)

    def s_share_half(M):                                       # a .5 split printed + written everywhere (SAVED, HF lines, receipts)
        p = os.path.join(M, 'rsna-sub-%s.log' % ST); EE = json.load(open(p, encoding='utf-8')); n = 0
        for e in EE:
            if '"s_share": 1.0' in e['data']:
                e['data'] = e['data'].replace('"s_share": 1.0', '"s_share": 0.5'); n += 1
        assert n >= 4, n
        json.dump(EE, open(p, 'w', encoding='utf-8'))
        jedit(M, 'hf_receipt.json', s_share=0.5)
        vp = os.path.join(M, 'v224f_receipt.json'); d = json.load(open(vp)); d['s_share'] = 0.5; d['save'] = dict(d['save'], s_share=0.5)
        json.dump(d, open(vp, 'w'), indent=1)

    def k14_present(M):                                        # the fleet ran K14 20 + S 5 (npz, fleet solo, receipt, log consistent)
        zs = []
        for gg in (0, 1):
            r = dict(np.load(os.path.join(REF, '_fleet_%d.npz' % gg), allow_pickle=False)); nm_ = [str(x) for x in r['names']]
            for k in ('P', 'PA', 'PB'):
                r[k] = np.concatenate([r[k], np.stack([r[k][nm_.index(d)] for d in DT])])
            r['names'] = np.array(nm_ + [smap[d] for d in DT]); r['n_keep'] = np.array(25); zs.append(r)
            np.savez(os.path.join(M, '_fleet_%d.npz' % gg), **r)
        PP = np.concatenate([r['P'] for r in zs], axis=1).astype(np.float64); ii = np.concatenate([r['sids'] for r in zs]).astype(str)
        so = pd.DataFrame(PP.mean(0), columns=TG, index=ii).reindex(test_ids); so.insert(0, 'StudyInstanceUID', test_ids)
        so.to_csv(os.path.join(M, 'submission_fleetsolo.csv'), index=False)
        nn = [str(x) for x in zs[0]['names']]; f0 = json.load(open(os.path.join(REF, 'fleet_receipt.json')))
        p = os.path.join(M, 'fleet_receipt.json'); d = json.load(open(p))
        d.update(models=25, names=nn, n_keep=[25, 25], members_filter=nn)
        d['config'] = dict(d['config'], MEMBERS=nn, TTA={x: 'AB' for x in nn})
        d['tta'] = dict(d['tta'], modes={x: 'AB' for x in nn}, fwd=dict(f0['tta']['fwd'], **{smap[x]: f0['tta']['fwd'][x] for x in DT}))
        json.dump(d, open(p, 'w'), indent=1)
        log_sub('RESULT OK fleet stage clean: 5 member(s)', 'RESULT OK fleet stage clean: 25 member(s)')(M)

    def run_hf(M, k2rec):                                      # re-execute the stand-in HF-S cell in M; splice its lines into M's log
        gg = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': sv,
              '_canary': {'fired': [], 'r224_used': True}, '_k2_receipt': k2rec}
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            exec(compile(HF.replace('/kaggle/working', M.replace('\\', '/')), 'HF-S', 'exec'), gg)
        p = os.path.join(M, 'rsna-sub-%s.log' % ST); EE = json.load(open(p, encoding='utf-8'))
        ii = [i for i, e in enumerate(EE) if any(k in e['data'] for k in ('V224FS_F2G', 'HF_BLEND_OK', 'HF_FALLBACK', 'CANARY_FIRED',
                                                                         'HF_COMPLETE', 'V224F_DONE'))]
        EE = EE[:ii[0]] + [dict(EE[ii[0]], data=ln + '\n') for ln in b.getvalue().strip().splitlines()] + EE[ii[-1] + 1:]
        json.dump(EE, open(p, 'w', encoding='utf-8'))
        lines = [ln.split(' ', 1)[0] for ln in b.getvalue().strip().splitlines()]
        assert 'HF_FALLBACK' in lines and 'CANARY_FIRED' in lines, lines
        assert (rd(os.path.join(M, 'submission.csv'))['Synovitis'] == 0.5).all()

    def canary_not_started(M):                                 # fleet time guard refused: no npz / solo / fleet2g / H, HF falls back
        for f in ('_fleet_0.npz', '_fleet_1.npz', 'submission_fleetsolo.csv', 'submission_fleet2g.csv', 'submission_H.csv',
                  'submission_w40.csv', 'submission_w50.csv', 'submission_w60.csv'):
            if os.path.isfile(os.path.join(M, f)):
                os.remove(os.path.join(M, f))
        why = 'only 100.0 min before the 8.3 h wall (< 192.0 min): stage not started'
        jedit(M, 'fleet_receipt.json', clean=False, why=why)
        log_sub('RESULT OK fleet stage clean: 5 member(s)', 'RESULT FLEET STAGE NOT CLEAN (%s) was: 5 member(s)' % why)(M)
        run_hf(M, dict(fr, clean=False, why=why))

    def canary_hf_refused(M):                                  # fleet clean on disk, but the HF cell saw 25 members -> fallback
        run_hf(M, dict(fr, clean=True, models=25, names=REFN + names))

    def neg_cross(name, out, build):                           # a whole mock output under the other build json
        assert os.path.isdir(out), out + ' missing: run `python tests/mock_fs_output.py ff` first'
        rc, last, fails, _ = gate(out, name, build=build)
        ok = rc != 0
        res.append(ok)
        print(f'NEG {name:<44} {"as expected (FAIL)" if ok else "UNEXPECTED PASS"} | {last} | first FAIL: {fails[:1]}')
        for ln in fails:
            print('      ' + ln[:3000])

    mutate('submission_fleet2g.csv cell +1e-3', bump('submission_fleet2g.csv'), True)
    mutate('submission.csv cell +1e-3', bump('submission.csv'), True)
    mutate('an S member row changed in the npz', npz_s, True)
    mutate('fleet2g file missing', lambda M: os.remove(os.path.join(M, 'submission_fleet2g.csv')), True)
    mutate('no s-merge line', log_sub('[s-merge 07:36:47] RESULT OK', '[x-merge 07:36:47] RESULT OK'), True)
    mutate('V224F_SAVED members 25 (K14 + S)', log_sub('"members": 5, "control"', '"members": 25, "control"'), True)
    mutate('V224F_SAVED groups K14 20', log_sub('"groups": {"K14": 0, "S": 5}', '"groups": {"K14": 20, "S": 5}'), True)
    mutate('s_share .5 everywhere (log, hf + v224f receipt)', s_share_half, True)
    mutate('hf_receipt s_share .5 only', lambda M: jedit(M, 'hf_receipt.json', s_share=0.5), True)
    mutate('K14 present: fleet ran K14 20 + S 5', k14_present, True)
    mutate('F = K14 mean (fleet2g, submission, shas ok)', f2g_is_k14, True)
    mutate('F = .5 K14 + .5 S (FF-S F; shas ok)', f2g_is_ffs_mix, True)
    mutate('canary: fleet not started, HF_FALLBACK', canary_not_started, True)
    mutate('canary: HF refused (25 members), HF_FALLBACK', canary_hf_refused, True)
    mutate('weight not applied: w_f 0 (sha ok)', reblend(0.0), True)
    mutate('wrong weight: w_f .30 (sha ok)', reblend(0.3), True)
    neg_cross('cross: FFSR mock under the FF-S build json', OUT, os.path.join(HERE, 'standin_v224ffs20', 'BUILD_V224FS.json'))
    neg_cross('cross: FF-S mock under the FFSR build json', os.path.join(HERE, 'mock_out_v224ffs20'), os.path.join(KD, 'BUILD_V224FS.json'))

    # ---- GAP probe (NOT counted): a wrong-but-self-consistent fleet. Replace mode has no numeric anchor for the fleet (the
    # --ref-fleet check needs K14 rows), so S predictions of the 2 studies of worker 0 swapped (a study-id misalignment), with
    # fleet solo / fleet2g / submission / receipts / log all re-made by the kernel's own HF-S cell, are expected to PASS.
    # FF-S keeps 20 of 25 members anchored to the FF20 commit (1e-3), so the same misalignment there fails on the K14 rows.
    def gap_s_rows_consistent(M):
        zs = []
        for gg in (0, 1):
            z = dict(np.load(os.path.join(M, '_fleet_%d.npz' % gg), allow_pickle=False))
            if gg == 0:
                assert z['P'].shape[1] == 2
                z['P'] = z['P'][:, ::-1, :].copy()
            zs.append(z); np.savez(os.path.join(M, '_fleet_%d.npz' % gg), **z)
        PP = np.concatenate([z['P'] for z in zs], axis=1).astype(np.float64); ii = np.concatenate([z['sids'] for z in zs]).astype(str)
        so = pd.DataFrame(PP.mean(0), columns=TG, index=ii).reindex(test_ids); so.insert(0, 'StudyInstanceUID', test_ids)
        so.to_csv(os.path.join(M, 'submission_fleetsolo.csv'), index=False)
        gg = {'NB_T0': time.time() - 600, 'labels': labels, 'test_ids': test_ids, 'sub': H.copy(), '_K2_TARGETS': TG, '_V224F': sv,
              '_canary': {'fired': [], 'r224_used': True}, '_k2_receipt': dict(fr, clean=True)}
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            exec(compile(HF.replace('/kaggle/working', M.replace('\\', '/')), 'HF-S', 'exec'), gg)
        for z_ in gg.get('_parts') or []:                  # the HF cell keeps its NpzFile handles open (Windows: rmtree fails)
            z_.close()
        gg.clear()
        lines = [ln.split(' ', 1)[0] for ln in b.getvalue().strip().splitlines()]
        assert lines == ['V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'], lines
        p = os.path.join(M, 'rsna-sub-%s.log' % ST); EE = json.load(open(p, encoding='utf-8'))
        ii = [i for i, e in enumerate(EE) if any(k in e['data'] for k in ('V224FS_F2G', 'HF_BLEND_OK', 'HF_COMPLETE', 'V224F_DONE'))]
        EE = EE[:ii[0]] + [dict(EE[ii[0]], data=ln + '\n') for ln in b.getvalue().strip().splitlines()] + EE[ii[-1] + 1:]
        json.dump(EE, open(p, 'w', encoding='utf-8'))
        d = float(np.abs(rd(os.path.join(M, 'submission.csv'))[labels].values - rd(os.path.join(OUT, 'submission.csv'))[labels].values).max())
        print('GAP probe: perturbed-S submission vs the clean mock submission max |diff| %.4f' % d)

    Mg = OUT + '_gap'
    if os.path.isdir(Mg):
        shutil.rmtree(Mg)
    shutil.copytree(OUT, Mg); gap_s_rows_consistent(Mg)
    rc_g, last_g, fails_g, _ = gate(Mg, 'gap')
    print(f'GAP S rows perturbed per study, kernel-consistent (NOT counted): {"PASS = known gap (no numeric anchor in replace mode)" if rc_g == 0 else "FAIL (caught)"} | {last_g}')
    shutil.rmtree(Mg)
print(f'MOCK_FS_GATE {"PASS" if all(res) else "FAIL"} {sum(res)}/{len(res)} (1 clean PASS + {len(res) - 1} negatives FAIL as expected)')
