# Offline sim of the V224-F appended cells SAVE / CLOCK / HF (the verbatim fleet cells are mocked: FN-IDS clobber + a
# _k2_receipt + submission_fleetsolo.csv). Inputs: V224-exact commit output (H) and HFR7-RR commit fleet solo (F), same 3 rows.
import os, sys, json, time, shutil, hashlib, tempfile, io, contextlib
import numpy as np, pandas as pd

ROOT = 'C:/Users/eeshs/Downloads/rsna-knee'
NB = json.load(open(f'{ROOT}/session_2026-10-15/v224f/kernel_v224f20/rsna_sub_v224f20.ipynb', encoding='utf-8'))
cells = [c['source'] for c in NB['cells']]
SAVE, CLOCK, HF = cells[14], cells[18], cells[21]
assert SAVE.startswith('# [eeshsaxena V224-F SAVE') and CLOCK.startswith('# [eeshsaxena V224-F CLOCK') and HF.startswith('# [eeshsaxena V224-F HF')
V224_OUT = f'{ROOT}/session_2026-10-14/v224/out_v224_v1'
F_SRC = f'{ROOT}/session_2026-10-13/hr7rr/out_hfr7rr_v1/submission_fleetsolo.csv'
sample = pd.read_csv(f'{ROOT}/_data/sample_submission.csv', dtype={'StudyInstanceUID': str})
cols = list(sample.columns); labels = cols[1:]; test_ids = sample['StudyInstanceUID'].tolist()


def run(scn, fleet_clean=True, r224_fired=False, tamper_saved=False, bad_fleet_ids=False, members=20):
    W = tempfile.mkdtemp(prefix='v224fsim_')
    fix = lambda s: s.replace('/kaggle/working', W.replace('\\', '/'))
    sub = pd.read_csv(f'{V224_OUT}/submission.csv', dtype={'StudyInstanceUID': str})
    sub[labels] = sub[labels].astype(np.float32)
    if r224_fired:
        sub['Fracture'] = np.float32(0.5)
    sub.to_csv(f'{W}/submission.csv', index=False)
    v224_bytes = open(f'{W}/submission.csv', 'rb').read()
    g = {'NB_T0': time.time() - 3.0 * 3600, 'labels': labels, 'test_ids': test_ids, 'sub': sub, 'n': len(test_ids),
         '_canary': {'fired': ['r224->Fracture'] if r224_fired else [], 'r224_used': not r224_fired}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fix(SAVE), 'SAVE', 'exec'), g)
        # mock FN-IDS: placeholder submission.csv + T0 = now
        ph = pd.DataFrame(np.full((len(test_ids), 12), 0.5), columns=labels); ph.insert(0, 'StudyInstanceUID', test_ids)
        ph.to_csv(f'{W}/submission.csv', index=False)
        g['T0'] = time.time()
        exec(compile(fix(CLOCK), 'CLOCK', 'exec'), g)
        assert abs(g['T0'] - g['NB_T0']) < 1e-6
        if tamper_saved:
            with open(f'{W}/submission_V224.csv', 'a') as fh:
                fh.write('\n')
        if fleet_clean:
            F = pd.read_csv(F_SRC, dtype={'StudyInstanceUID': str})
            if bad_fleet_ids:
                F.loc[0, 'StudyInstanceUID'] = 'x'
            F.to_csv(f'{W}/submission_fleetsolo.csv', index=False)
            g['_k2_receipt'] = {'clean': True, 'models': members, 'why': 'ok', 'n_keep': [members, members]}
        else:
            g['_k2_receipt'] = {'clean': False, 'why': "_K2Fail: only 30.0 min before the wall (< 168.0 min): stage not started"}
        exec(compile(fix(HF), 'HF', 'exec'), g)
    log = buf.getvalue()
    out = pd.read_csv(f'{W}/submission.csv', dtype={'StudyInstanceUID': str})
    hf = json.load(open(f'{W}/hf_receipt.json')); vr = json.load(open(f'{W}/v224f_receipt.json'))
    # independent expectation
    H = pd.read_csv(io.BytesIO(v224_bytes), dtype={'StudyInstanceUID': str})
    pct = lambda v: pd.Series(np.asarray(v, np.float64)).rank(method='average', pct=True).values
    ok_clean = fleet_clean and not bad_fleet_ids and members == 20
    if ok_clean:
        Fd = pd.read_csv(F_SRC, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
        exp = np.stack([(0.8 * pct(H[c]) + 0.2 * pct(Fd[c])) for c in labels], 1)
        if r224_fired:
            exp[:, labels.index('Fracture')] = 0.5
        d = float(np.abs(out[labels].values - exp).max())
        res = d < 1e-6 and hf['mode'] == 'HF' and 'HF_BLEND_OK' in log and 'HF_FALLBACK' not in log
        res = res and os.path.isfile(f'{W}/submission_H.csv') and open(f'{W}/submission_H.csv', 'rb').read() == v224_bytes
        info = 'max diff %.2e' % d
    else:
        # fallback: Synovitis 0.5, (Fracture 0.5 if fired), the other columns string-identical to the V224 output
        Hs = pd.read_csv(io.BytesIO(v224_bytes), dtype=str); Os = pd.read_csv(f'{W}/submission.csv', dtype=str)
        keep = [c for c in cols if c not in ('Synovitis',)]
        res = (Hs[keep] == Os[keep]).all().all() and (out['Synovitis'] == 0.5).all() and hf['mode'] == 'H_only'
        res = res and 'HF_FALLBACK' in log and 'CANARY_FIRED fleet not used' in log and 'fleet->Synovitis' in vr['fired']
        info = 'blanked %s why %s' % (hf.get('blanked'), hf.get('why', '')[:90])
    if tamper_saved:
        res = res and 'V224F_H_FROM_MEMORY' in log and hf['h_source'].startswith('in-memory')
    print(f'SIM {scn:<28} {"PASS" if res else "FAIL"} mode={hf["mode"]} h_source={hf["h_source"][:30]} fired={vr["fired"]} {info}')
    if not res:
        print(log[-3000:])
    shutil.rmtree(W, ignore_errors=True)
    return res


R = [run('clean'), run('fleet not clean (budget)', fleet_clean=False), run('r224 canary + fleet clean', r224_fired=True),
     run('r224 canary + fleet fail', fleet_clean=False, r224_fired=True), run('saved file tampered', tamper_saved=True),
     run('fleet bad ids', bad_fleet_ids=True), run('fleet 19 members', members=19)]
print('SIM', 'PASS' if all(R) else 'FAIL', sum(R), '/', len(R))
