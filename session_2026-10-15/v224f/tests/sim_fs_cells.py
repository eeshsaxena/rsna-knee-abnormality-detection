# sim_fs_cells.py (2026-10-15 dir, 10-14, Claude subagent): OFFLINE proof of the two NEW V224-F-S cells (S-MERGE, HF-S) of a
# notebook built by build_v224fs.py (default: the --standin build, S = the 5 k14w DT checkpoints relabelled f{F}_s1047r7tb_*).
# No GPU, no Kaggle. Linux (symlinks) - run on the box CPU:  python sim_fs_cells.py DIR   where DIR holds
#   rsna_sub_v224fs20.ipynb, rsna_sub_v224fsr20.ipynb (REPLACE build), sample_submission.csv, v224_submission.csv (the V224-exact
#   commit output), and the V224-F20 commit's fleet outputs _fleet_0.npz, _fleet_1.npz (20 K14 members on the 3 test studies).
# HF-S: the 25-member npz = the commit's 20 real K14 rows + 5 stand-in S rows (copies of the 5 DT rows under the S names), the
#   fleet solo = the fleet cell's own equal-mean formula over the 25 (median fill for failed studies). Expected F = .5 mean(K14)
#   + .5 mean(S), recomputed here independently (own ranker for the blend). Scenarios: clean; a failed study; a permuted
#   _K2_TARGETS order; r224 canary + clean; tampered fleet solo; a missing S member; wrong receipt names; fleet not clean;
#   REPLACE (S only).
# S-MERGE: fake /kaggle/input + /tmp/k12_fleet under a temp dir; the verbatim K12-MERGE body and the S-MERGE body run with
#   test-built configs whose sha256 values match the dummy files. Scenarios: ok (30 checkpoints linked, manifest == merged,
#   every K2_CFG member in it with tta AB); S dataset missing; S file tampered; extra file; K12-MERGE failed; name collision.
import os, sys, json, time, shutil, hashlib, tempfile, io, contextlib, copy
import numpy as np, pandas as pd

D = sys.argv[1]
NB = json.load(open(f'{D}/rsna_sub_v224fs20.ipynb', encoding='utf-8'))
NBR = json.load(open(f'{D}/rsna_sub_v224fsr20.ipynb', encoding='utf-8'))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
C = [src(c) for c in NB['cells']]; CR = [src(c) for c in NBR['cells']]
SAVE, K12, SMERGE, CLOCK, FLEET, HF = C[14], C[17], C[18], C[19], C[20], C[22]
assert SAVE.startswith('# [eeshsaxena V224-F-S SAVE') and K12.startswith('K12M_CFG = {') and SMERGE.startswith('S_CFG = {')
assert CLOCK.startswith('# [eeshsaxena V224-F CLOCK') and FLEET.startswith('K2_CFG = {') and HF.startswith('# [eeshsaxena V224-F-S HF')
lit = lambda line, name: eval(line[len(name + ' = '):line.index('   # literal')], {})
K2 = lit(FLEET.split('\n', 1)[0], 'K2_CFG'); S_CFG = lit(SMERGE.split('\n', 1)[0], 'S_CFG'); K12M = lit(K12.split('\n', 1)[0], 'K12M_CFG')
K2R = lit(CR[20].split('\n', 1)[0], 'K2_CFG')
SN = S_CFG['s_manifest']['checkpoints']; K14 = [m for m in K2['MEMBERS'] if m not in SN]
assert len(K14) == 20 and len(SN) == 5 and K2R['MEMBERS'] == SN
sample = pd.read_csv(f'{D}/sample_submission.csv', dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; test_ids = sample['StudyInstanceUID'].tolist()
TG = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
      'Contusion', 'Fracture']
parts0 = [dict(np.load(f'{D}/_fleet_{g}.npz', allow_pickle=False)) for g in (0, 1)]
names20 = [str(x) for x in parts0[0]['names']]; assert sorted(names20) == sorted(K14)
DT = [m for m in K14 if '_s1043f75oai_' in m]; assert len(DT) == 5
smap = {d: d.replace('_s1043f75oai_', '_s1047r7tb_') for d in DT}; assert sorted(smap.values()) == sorted(SN)
R = []


def pct(v):                                       # independent average-tie percentile rank
    v = np.asarray(v, np.float64); o = np.argsort(v, kind='mergesort'); sv = v[o]; r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return r / len(v)


def make_parts(replace=False, fail_study=None, drop_s=False):
    out = []
    for z in parts0:
        P = z['P'].astype(np.float32); nm = [str(x) for x in z['names']]
        if replace:
            P2 = np.stack([P[nm.index(d)] for d in DT]); nm2 = [smap[d] for d in DT]
        else:
            P2 = np.concatenate([P, np.stack([P[nm.index(d)] for d in DT])]); nm2 = nm + [smap[d] for d in DT]
        if drop_s:
            P2 = P2[:-1]; nm2 = nm2[:-1]
        P2 = P2.copy()
        if fail_study is not None and fail_study in [str(s) for s in z['sids']]:
            P2[3, [str(s) for s in z['sids']].index(fail_study), :] = np.nan
        out.append(dict(z, P=P2, names=np.array(nm2)))
    return out


def eq_solo(parts):                               # the fleet cell's own formula (mean over members, failed -> column median)
    P = np.concatenate([z['P'] for z in parts], axis=1).astype(np.float64); NV = np.concatenate([z['NV'] for z in parts])
    ids = np.concatenate([z['sids'] for z in parts]).astype(str)
    ok = np.isfinite(P).all(axis=(0, 2)) & (NV > 0)
    m = P.mean(0); m[~ok] = np.nan; med = np.nanmedian(m, axis=0); m = np.where(np.isfinite(m), m, med[None])
    return pd.DataFrame(m, columns=TG, index=ids).reindex(test_ids), P, ids, ok


def expect_f(parts, replace=False):
    _, P, ids, ok = eq_solo(parts)
    nm = [str(x) for x in parts[0]['names']]
    fs = P[[nm.index(m) for m in SN]].mean(0)
    f = fs if replace else .5 * P[[nm.index(m) for m in K14]].mean(0) + .5 * fs
    f[~ok] = np.nan; med = np.nanmedian(f, axis=0); f = np.where(np.isfinite(f), f, med[None])
    return pd.DataFrame(f, columns=TG, index=ids).loc[test_ids, labels]


def run_hf(scn, nb_cells=C, replace=False, clean=True, r224=False, fail_study=None, drop_s=False, tamper_solo=False,
           bad_names=False, perm_targets=False, want='HF'):
    W = tempfile.mkdtemp(prefix='v224fs_hf_')
    fix = lambda s: s.replace('/kaggle/working', W)
    sub = pd.read_csv(f'{D}/v224_submission.csv', dtype={'StudyInstanceUID': str}); sub[labels] = sub[labels].astype(np.float32)
    if r224:
        sub['Fracture'] = np.float32(0.5)
    sub.to_csv(f'{W}/submission.csv', index=False); v224_bytes = open(f'{W}/submission.csv', 'rb').read()
    parts = make_parts(replace=replace, fail_study=fail_study, drop_s=drop_s)
    tg = list(TG)
    if perm_targets:                              # the fleet's column order differs from her labels: P columns permuted with it
        perm = list(np.random.default_rng(3).permutation(12)); tg = [TG[i] for i in perm]
        parts = [dict(z, P=z['P'][:, :, perm]) for z in parts]
    for g, z in enumerate(parts):
        np.savez(f'{W}/_fleet_{g}.npz', **z)
    solo, _, _, _ = eq_solo(make_parts(replace=replace, fail_study=fail_study, drop_s=drop_s))
    solo = solo.reset_index().rename(columns={'index': 'StudyInstanceUID'})
    if tamper_solo:
        solo.loc[1, 'ACL'] += 1e-3
    g = {'NB_T0': time.time() - 3.0 * 3600, 'labels': labels, 'test_ids': test_ids, 'sub': sub, '_K2_TARGETS': tg,
         '_canary': {'fired': ['r224->Fracture'] if r224 else [], 'r224_used': not r224}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fix(nb_cells[14]), 'SAVE', 'exec'), g)
        ph = pd.DataFrame(np.full((len(test_ids), 12), 0.5), columns=labels); ph.insert(0, 'StudyInstanceUID', test_ids)
        ph.to_csv(f'{W}/submission.csv', index=False); g['T0'] = time.time()
        exec(compile(fix(nb_cells[19]), 'CLOCK', 'exec'), g)
        names = [str(x) for x in parts[0]['names']]
        M = len(K2R['MEMBERS'] if replace else K2['MEMBERS'])
        if clean:
            solo.to_csv(f'{W}/submission_fleetsolo.csv', index=False)
            g['_k2_receipt'] = {'clean': True, 'models': M, 'why': 'ok', 'n_keep': [M, M],
                                'names': (names[:-5] + ['x%d' % i for i in range(5)]) if bad_names else names}
        else:
            g['_k2_receipt'] = {'clean': False, 'why': "_K2Fail: only 30.0 min before the wall (< 198.0 min): stage not started"}
        exec(compile(fix(nb_cells[22]), 'HF-S', 'exec'), g)
    log = buf.getvalue()
    out = pd.read_csv(f'{W}/submission.csv', dtype={'StudyInstanceUID': str})
    hf = json.load(open(f'{W}/hf_receipt.json')); vr = json.load(open(f'{W}/v224f_receipt.json'))
    H = pd.read_csv(io.BytesIO(v224_bytes), dtype={'StudyInstanceUID': str})
    if want == 'HF':
        F = expect_f(make_parts(replace=replace, fail_study=fail_study), replace=replace)
        exp = np.stack([.8 * pct(H[c]) + .2 * pct(F[c]) for c in labels], 1)
        if r224:
            exp[:, labels.index('Fracture')] = 0.5
        d = float(np.abs(out[labels].values - exp).max())
        F2 = pd.read_csv(f'{W}/submission_fleet2g.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
        d2 = float(np.abs(F2.values - F.values).max())
        eqd = float(np.abs(F.values - solo.set_index('StudyInstanceUID').loc[test_ids, labels].values).max())
        ok = d < 1e-6 and d2 < 1e-9 and hf['mode'] == 'HF' and 'HF_BLEND_OK' in log and 'V224FS_F2G' in log and 'HF_FALLBACK' not in log
        ok = ok and hf['f2g']['eq_mean_vs_fleetsolo_max'] <= 1e-6 and vr['fired'] == (['r224->Fracture'] if r224 else [])
        ok = ok and (eqd > 1e-4 or replace)                 # the two-group F really differs from the equal mean (not replace)
        info = 'final max diff %.2e | fleet2g vs independent %.2e | two-group vs equal mean %.3g' % (d, d2, eqd)
    else:
        Hs = pd.read_csv(io.BytesIO(v224_bytes), dtype=str); Os = pd.read_csv(f'{W}/submission.csv', dtype=str)
        keep = [c for c in cols if c != 'Synovitis']
        ok = (Hs[keep] == Os[keep]).all().all() and (out['Synovitis'] == 0.5).all() and hf['mode'] == 'H_only'
        ok = ok and 'CANARY_FIRED fleet not used' in log and 'fleet->Synovitis' in vr['fired'] and not os.path.exists(f'{W}/submission_fleet2g.csv')
        info = 'why %s' % hf.get('why', '')[:110]
    R.append(ok)
    print(f'HF-S {scn:<34} {"PASS" if ok else "FAIL"} mode={hf["mode"]} {info}', flush=True)
    if not ok:
        print(log[-2500:])
    shutil.rmtree(W, ignore_errors=True)


def run_merge(scn, s_missing=False, s_tamper=False, s_extra=False, k12_fail=False, collide=False, want_ok=True):
    T = tempfile.mkdtemp(prefix='v224fs_sm_'); INP = f'{T}/input'; OUT = f'{T}/k12_fleet'
    fix = lambda s: s.replace('/kaggle/input', INP).replace('/tmp/k12_fleet', OUT)
    sh = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    k12 = copy.deepcopy(K12M); s_cfg = copy.deepcopy(S_CFG)
    b = f'{INP}/{k12["base_ds"]}'; a = f'{INP}/{k12["add_ds"]}'; sd = f'{INP}/{s_cfg["s_ds"]}'
    for d in (b, a) + (() if s_missing else (sd,)):
        os.makedirs(d)
    open(f'{b}/fleet_worker.py', 'w').write('# dummy worker\n'); open(f'{b}/parity_x.npz', 'wb').write(b'p')
    for n in k12['base_manifest']['checkpoints']:
        open(f'{b}/{n}', 'wb').write(('base ' + n).encode())
    json.dump(k12['base_manifest'], open(f'{b}/fleet_manifest.json', 'w'))
    for n in k12['add_manifest']['checkpoints']:
        open(f'{a}/{n}', 'wb').write(('add ' + n).encode()); s = sh(f'{a}/{n}')
        k12['add_manifest']['sources'][n]['sha256_full'] = s; k12['merged_manifest']['sources'][n]['sha256'] = s[:16]
    json.dump(k12['add_manifest'], open(f'{a}/k12w_manifest.json', 'w'))
    if collide:                                   # a file with an S name already in the K12 fleet dir (via the base dataset)
        open(f'{b}/{SN[0]}', 'wb').write(b'collide')
    if not s_missing:
        for n in SN:
            open(f'{sd}/{n}', 'wb').write(('student ' + n).encode()); s = sh(f'{sd}/{n}')
            s_cfg['s_manifest']['sources'][n]['sha256_full'] = s; s_cfg['s_manifest']['sources'][n]['sha256'] = s[:16]
            s_cfg['merged_manifest']['sources'][n]['sha256'] = s[:16]
        json.dump(s_cfg['s_manifest'], open(f'{sd}/{s_cfg["manifest_file"]}', 'w'))
        if s_tamper:
            open(f'{sd}/{SN[2]}', 'ab').write(b'x')
        if s_extra:
            open(f'{sd}/notes.txt', 'w').write('x')
    for n in k12['merged_manifest']['checkpoints']:
        if n in k12['add_manifest']['sources']:
            k12['merged_manifest']['sources'][n]['sha256'] = k12['add_manifest']['sources'][n]['sha256_full'][:16]
    s_cfg['merged_manifest']['sources'].update({n: k12['merged_manifest']['sources'][n] for n in k12['merged_manifest']['checkpoints']})
    if k12_fail:
        open(f'{a}/{k12["add_manifest"]["checkpoints"][0]}', 'ab').write(b'x')
    g = {'K12M_CFG': k12, 'S_CFG': s_cfg}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fix(K12.split('\n', 1)[1]), 'K12-MERGE', 'exec'), g)
        exec(compile(fix(SMERGE.split('\n', 1)[1]), 'S-MERGE', 'exec'), g)
    log = buf.getvalue()
    if want_ok:
        man = json.load(open(f'{OUT}/fleet_manifest.json'))
        links = {f: os.path.realpath(f'{OUT}/{f}') for f in os.listdir(OUT) if f != 'fleet_manifest.json'}
        ok = ('RESULT OK merged fleet dir + S: 30 checkpoints' in log and man == s_cfg['merged_manifest']
              and all(n in links for n in man['checkpoints']) and 'fleet_worker.py' in links
              and all(links[n] == os.path.realpath(f'{sd}/{n}') for n in SN)
              and all(links[n].startswith(os.path.realpath(INP)) for n in man['checkpoints'])
              and all(m in man['checkpoints'] and man['sources'][m]['tta'] == K2['TTA'][m] == 'AB' for m in K2['MEMBERS'])
              and not os.path.exists(OUT + '.k12only') and not os.path.exists(OUT + '.s.tmp')
              and all(os.path.isfile(f'{sd}/{n}') for n in SN))
        info = '%d entries, %d checkpoints in the manifest' % (len(links), len(man['checkpoints']))
    else:
        ok = 'RESULT FAIL' in log.split('[s-merge')[-1] and not os.path.lexists(OUT) and (s_missing or all(os.path.isfile(f'{sd}/{n}') for n in SN))
        info = log.strip().splitlines()[-1][:150]
    R.append(ok)
    print(f'S-MERGE {scn:<31} {"PASS" if ok else "FAIL"} {info}', flush=True)
    if not ok:
        print(log[-2500:])
    shutil.rmtree(T, ignore_errors=True)


print(f'sim_fs_cells: K14 {len(K14)} S {len(SN)} (stand-ins of {DT[0]}...), K2 MIN_START_MIN {K2["MIN_START_MIN"]} WALL_H {K2["WALL_H"]}')
run_hf('clean')
run_hf('a failed study (median fill)', fail_study=test_ids[1])
run_hf('fleet target order permuted', perm_targets=True)
run_hf('r224 canary + clean fleet', r224=True)
run_hf('REPLACE notebook (S only)', nb_cells=CR, replace=True)
run_hf('tampered fleet solo', tamper_solo=True, want='fallback')
run_hf('one S member missing', drop_s=True, want='fallback')
run_hf('receipt names != K14 + S', bad_names=True, want='fallback')
run_hf('fleet not clean (time guard)', clean=False, want='fallback')
run_merge('ok')
run_merge('S dataset missing', s_missing=True, want_ok=False)
run_merge('S file tampered', s_tamper=True, want_ok=False)
run_merge('extra file in S dataset', s_extra=True, want_ok=False)
run_merge('K12-MERGE failed', k12_fail=True, want_ok=False)
run_merge('name collision', collide=True, want_ok=False)
print(f'SIM_FS_CELLS {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}')
