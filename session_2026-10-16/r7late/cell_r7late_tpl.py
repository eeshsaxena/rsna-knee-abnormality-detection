# [eeshsaxena R7 LATE LEG; not in any public notebook] PREREG_10-15 Amendment 14. Runs AFTER the S6F stage, only if time is left:
# heliosli's public R7 ConvNeXt-small bundle (8c7b666e, 4 folds, NO flip TTA) scores the test studies through the SAME worker and
# controller as stage 2 (probe_dual.py + compact_infer.py v4c from SRC), then "ours" = the per-study mean of the R6 (flip) and R7
# probabilities, stage 3 is recomputed with this kernel's own weights (W_RAPTOR / W_RAPTOR_C / A224, the same pct_on rule), and the
# S6F blend is recomputed with W_S6F. The R7 leg is USED only if: V224 canary clean, S6F mode S6F, start age <= G_R7_H, the bundle
# shas match its manifest, the controller ends rc 0, and R7 scored >= 98 % of the test studies. Otherwise the submission written by
# the S6F stage is left exactly as it is (fallback = the base entry, never a canary: this is a final-pick-safe form).
def _r7_late():
    import os, sys, json, time, hashlib, shutil, subprocess, csv
    import numpy as np, pandas as pd
    G_R7_H, DEADLINE_H, KILL_H, COV_MIN, W_S6F = __G_R7_H__, 8.0, 8.25, 0.98, __W_S6F__
    C7_SLUG, B7_ID = 'rsna-knee-compact-r7cnxfsss', '8c7b666e27dc0e8f'
    rec = {'used': False, 'why': None, 'g_r7_h': G_R7_H, 'w_s6f': W_S6F}
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    try:
        age = (time.time() - NB_T0) / 3600.0
        rec['start_age_h'] = round(age, 4)
        if list((globals().get('_canary') or {}).get('fired') or []):
            rec['why'] = 'V224 canary fired'; return rec
        if (globals().get('_S6F_BLEND') or {}).get('mode') != 'S6F':
            rec['why'] = 'S6F not in mode S6F'; return rec
        if age > G_R7_H:
            rec['why'] = 'start age %.2f h > G_R7 %.2f h' % (age, G_R7_H); return rec
        M7 = find_mount(C7_SLUG, 'manifest.json')
        man = json.loads((M7 / 'manifest.json').read_text())
        assert man['bundle_id'] == B7_ID, man['bundle_id']
        B7 = TMP / 'bundle_r7'
        shutil.rmtree(B7, ignore_errors=True); B7.mkdir(parents=True)
        names = list(man['models']) + list(man.get('models2') or []) + [man[k] for k in ('state_spec', 'side_model', 'fluid_stacker', 'state_stacker') if man.get(k)]
        for name in names:
            assert sha(M7 / name) == man['sha256'][name], name
            shutil.copy(M7 / name, B7 / name)
        a = ['--backbone', man['backbone'], '--size', str(man['size']), '--crop', str(man['crop']),
             '--eval-stride', str(man.get('eval_stride', 3)), '--views', man.get('views', 'v1')]
        if man.get('state_spec'):
            a += ['--state-spec', str(B7 / man['state_spec'])]
        if man.get('side_model'):
            a += ['--side-model', str(B7 / man['side_model']), '--side-threshold', str(man.get('side_threshold', 2.0)),
                  '--side-position-mm', str(man.get('side_position_mm', 20.0)), '--side-max-x', str(man.get('side_max_x', 0.0))]
        a += ['--slices', str(man.get('slices', 3)), '--phys-mm', str(man.get('phys_mm', 0.0))]
        assert not man.get('models2') and man['backbone'] != 'ofl_l', 'unexpected R7 manifest shape'
        if man.get('trainer_md5'):
            assert hashlib.md5((SRC / 'compact_train.py').read_bytes()).hexdigest() == man['trainer_md5'], 'trainer md5 differs'
            a += ['--trainer-md5', man['trainer_md5']]
        if man.get('compile'):
            a += ['--compile']
        if man.get('fluid_stacker'):
            a += ['--fluid-stacker', str(B7 / man['fluid_stacker'])]
        if man.get('state_stacker'):
            a += ['--state-stacker', str(B7 / man['state_stacker'])]
        if man.get('views') in ('v3k', 'v4'):
            assert sha(CODE_SRC / 'anatomy.pt') == man['anchor_weights_sha256'], 'anatomy.pt differs'
            a += ['--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
        assert '--flip-tta' not in a
        rec['worker_args'] = a
        pred7 = TMP / 'prediction_r7'
        shutil.rmtree(pred7, ignore_errors=True)
        env7 = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
        t0 = time.time()
        print('R7_LATE_START ' + json.dumps({'age_h': rec['start_age_h']}), flush=True)
        try:
            rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                                 '--code', str(CODE), '--bundle', str(B7), '--arm', man.get('arm', 'E_lrprior'), '--output', str(pred7),
                                 '--amp', 'auto', '--loader-workers', '2', '--deadline', '%.0f' % (NB_T0 + DEADLINE_H * 3600),
                                 '--worker', 'compact_infer.py', '--worker-args', json.dumps(a), '--max-fallback-fraction', '1.0'],
                                env=env7, timeout=max(60, NB_T0 + KILL_H * 3600 - time.time())).returncode
        except subprocess.TimeoutExpired:
            rec['why'] = 'R7 controller killed at the %.2f h wall' % KILL_H; return rec
        rec['rc'], rec['seconds'] = rc, round(time.time() - t0, 1)
        for log in sorted(pred7.glob('worker*/log.txt')):
            shutil.copyfile(log, WORK / ('r7_' + log.parent.name + '_log.txt'))
        if rc != 0:
            rec['why'] = 'R7 controller rc %s' % rc; return rec
        scored7 = set()
        for f in pred7.glob('worker*/predictions.csv'):
            with open(f, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    try:
                        v = [float(r[L]) for L in labels]
                    except (KeyError, ValueError, TypeError):
                        continue
                    if all(0. <= x <= 1. for x in v):
                        scored7.add(r['StudyInstanceUID'])
        cov = len(scored7 & set(test_ids)) / float(len(test_ids))
        rec['coverage'] = round(cov, 4)
        if cov < COV_MIN:
            rec['why'] = 'R7 coverage %.4f < %.2f' % (cov, COV_MIN); return rec
        o7 = pd.read_csv(pred7 / 'submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
        o7.reset_index().to_csv(WORK / 'ours_r7_sub.csv', index=False)
        ok7 = np.array([u in scored7 for u in test_ids])
        A6, A7 = ours.loc[test_ids, labels].to_numpy(np.float64).copy(), o7.to_numpy(np.float64)   # stage-2 R6 (flip) / R7 probs
        m = ok7 & ~our_fail
        A6[m] = 0.5 * (A6[m] + A7[m])                            # both scored -> mean; R7 failed -> R6 alone
        only7 = ok7 & our_fail
        A6[only7] = A7[only7]                                    # R6 failed -> R7 alone
        avg = pd.DataFrame(A6, index=test_ids, columns=labels)
        fail2 = our_fail & ~ok7
        avg.reset_index().to_csv(WORK / 'ours_r67_sub.csv', index=False)
        out = np.zeros((len(test_ids), len(labels)))
        for j, c in enumerate(labels):
            rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(avg[c].values, ~fail2)
            if r224 is not None:
                rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)
            wr = W_RAPTOR_C.get(c, W_RAPTOR)
            b = wr * rr + (1.0 - wr) * ro
            b[rap_fail] = ro[rap_fail]; b[fail2] = rr[fail2]; b[rap_fail & fail2] = 0.5
            out[:, j] = b
        v224 = pd.DataFrame(out.astype(np.float32), columns=labels, index=test_ids)
        v224.reset_index().rename(columns={'index': 'StudyInstanceUID'}).to_csv(WORK / 'submission_v224r67.csv', index=False)
        B = pd.read_csv(WORK / 's6f' / 'B' / 'B_probs.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        prev = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
        shutil.copyfile(WORK / 'submission.csv', WORK / 'submission_pre_r7.csv')
        pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
        fin = prev.set_index('StudyInstanceUID').loc[test_ids].copy()
        for c in labels:
            bs = B.reindex(test_ids)[c].values.astype(np.float64)
            okb = np.isfinite(bs)
            rb = pct(v224[c].values)
            rs = rb.copy()
            if okb.any():
                rs[okb] = pd.Series(bs[okb]).rank(method='average', pct=True).values
            fin[c] = ((1.0 - W_S6F) * rb + W_S6F * rs).astype(np.float32)
        fin = fin.reset_index()[cols]
        assert fin['StudyInstanceUID'].tolist() == test_ids and np.isfinite(fin[labels].values).all()
        assert ((fin[labels].values >= 0) & (fin[labels].values <= 1)).all()
        fin.to_csv(WORK / 'submission.csv', index=False)
        rec.update(used=True, why='used', n=len(test_ids), r6_failed=int(our_fail.sum()), r7_failed=int((~ok7).sum()),
                   submission_sha256=sha(WORK / 'submission.csv'), end_age_h=round((time.time() - NB_T0) / 3600.0, 4))
        return rec
    except Exception as e:
        rec['why'] = 'error %s: %s' % (type(e).__name__, str(e)[:300])
        try:
            if (WORK / 'submission_pre_r7.csv').is_file():
                shutil.copyfile(WORK / 'submission_pre_r7.csv', WORK / 'submission.csv')
                rec['restored_pre_r7'] = True
        except Exception:
            pass
        return rec


try:
    _R7 = _r7_late()
except Exception as _e:
    _R7 = {'used': False, 'why': 'cell error %s' % type(_e).__name__}
try:
    import json as _r7j
    with open('/kaggle/working/r7late_receipt.json', 'w') as _f:
        _r7j.dump(_R7, _f, indent=1, default=str)
    print('R7_LATE_DONE ' + _r7j.dumps({k: _R7.get(k) for k in ('used', 'why', 'start_age_h', 'coverage', 'seconds', 'end_age_h')}), flush=True)
except Exception as _e:
    print('R7_LATE_RECEIPT_ERROR ' + type(_e).__name__, flush=True)
