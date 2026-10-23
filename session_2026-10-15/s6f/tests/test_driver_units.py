#!/usr/bin/env python
# test_driver_units.py (10-15, Claude subagent): the S6F driver port on nartaa's REAL commit part outputs. Offline, CPU, -I.
# The built driver (== the bytes the notebook embeds) runs its own main(); only its GPU / dataset steps are replaced (see
# s6f_harness FAKE_PORT). Checks:
#   T1 clean: split_parts == nartaa's part ids; our B_probs.csv is BYTE-identical to nartaa's k1/B/B_probs.csv; coverage 1.0;
#      receipt ok, versions, microbench; part0 argv == nartaa's argv_part0.json except paths and the --deadline value
#   T2 a killed part (no receipt; predictions.csv only): its study scored from the csv (|dp| <= 5e-7), coverage 1.0, no error
#   T3 a timed-out part (deadline): 2 of 3 scored, coverage .667, no retry, receipt ok (the blend decides), no exception
#   T4 a part whose receipt names another compact_infer sha: those studies unscored (collect_part's check kept), coverage 1/3
#   T5 code manifest not the scored run's: driver fails (ok False, exit 3) with the reason; no B_probs.csv
#   usage: py -3.12 -I tests/test_driver_units.py   -> TEST_DRIVER PASS/FAIL n/m
import os, sys, json, time, shutil, hashlib, base64, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s6f_harness as H

R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


TMPB = os.path.join(HERE, '_tmp_driver')
shutil.rmtree(TMPB, ignore_errors=True)
os.makedirs(TMPB)
nb = json.load(open(H.NB_PATH, encoding='utf-8'))
setup = H.src(nb['cells'][15])
b64 = setup.split("DRIVER_B64 = '", 1)[1].split("'", 1)[0]
chk('built driver file == the bytes embedded in notebook cell 15', base64.b64decode(b64) == open(H.DRIVER_BUILT, 'rb').read(),
    hashlib.sha256(base64.b64decode(b64)).hexdigest()[:16])


def run_port(name, mut=None):
    t = os.path.join(TMPB, name)
    os.makedirs(t)
    out, tmp = os.path.join(t, 'out'), os.path.join(t, 'tmp')
    prog = os.path.join(t, 'fake_port.py')
    open(prog, 'w').write(H.FAKE_PORT % dict(driver=H.DRIVER_BUILT, ref=H.REF, data=H.DATA, out=out.replace('\\', '/'),
                                             tmp=tmp.replace('\\', '/'), mut=mut))
    stop = time.time() + 3600
    r = subprocess.run([sys.executable, '-I', prog, '--stop-at', '%.3f' % stop, '--nb-t0', '%.3f' % (time.time() - 4.6 * 3600)],
                       capture_output=True, text=True, timeout=600)
    rec = json.load(open(os.path.join(out, 's6f_driver_receipt.json'))) if os.path.isfile(os.path.join(out, 's6f_driver_receipt.json')) else {}
    return r, rec, out, stop


ref_b = H.ref_bprobs()
ref_raw = open(os.path.join(H.REF, 'k1', 'B', 'B_probs.csv'), 'rb').read()
# ---- T1
r, rec, out, stop = run_port('t1')
print('T1 clean', flush=True)
chk('T1 driver exit 0, receipt ok', r.returncode == 0 and rec.get('ok') is True, (r.returncode, rec.get('error'), r.stderr[-300:]))
mine = open(os.path.join(out, 'B', 'B_probs.csv'), 'rb').read() if os.path.isfile(os.path.join(out, 'B', 'B_probs.csv')) else b''
mine_lf = mine.replace(b'\r\n', b'\n')                 # pandas writes os.linesep: CRLF on this Windows laptop, LF on Kaggle
ml, rl = mine_lf.split(b'\n'), ref_raw.split(b'\n')
zc = [i for i, c in enumerate(rl[0].split(b',')) if c.startswith(b'z|')]
same_z = len(ml) == len(rl) and ml[0] == rl[0] and all([r_.split(b',')[0]] + [r_.split(b',')[i] for i in zc] ==
                                                       [m_.split(b',')[0]] + [m_.split(b',')[i] for i in zc] for m_, r_ in zip(ml[1:], rl[1:]) if r_)
pm = pd.read_csv(os.path.join(out, 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str})
dp = float(np.abs(pm[H.LAB].to_numpy() - ref_b[H.LAB].to_numpy()).max())
nbyte = sum(m_ != r_ for m_, r_ in zip(ml, rl))
chk('T1 B_probs.csv vs nartaa k1/B/B_probs.csv: header, ids, row order and all 12 z| columns byte-identical; P = sigmoid(z) within '
    '1e-15 (numpy exp build: laptop 2.5 vs Kaggle 2.1)', same_z and dp <= 1e-15, 'max |dp| %.1e, lines differing in any byte %d of %d'
    % (dp, nbyte, len(rl) - 1))
B = rec.get('B') or {}
chk('T1 coverage 1.0, 3/3 scored, sources logits x2, no retry', B.get('coverage') == 1.0 and B.get('scored') == 3 and
    B.get('sources') == ['logits', 'logits'] and B.get('retry') is None, {k: B.get(k) for k in ('coverage', 'scored', 'sources')})
chk('T1 probs sha in receipt == file', B.get('probs_sha256') == hashlib.sha256(mine).hexdigest())
mb = B.get('microbench') or {}
chk('T1 microbench fields', all(isinstance(mb.get(k), (int, float)) for k in ('s_per_study_per_gpu_max', 'gpu_s_per_study',
                                                                                 'decode_s_per_study_per_worker', 'windows_per_study')), mb)
chk('T1 receipt: env versions + isolation + B probe + stages', all(k in (rec.get('env') or {}) for k in ('python', 'torch', 'timm', 'pydicom'))
    and (rec.get('env_isolation') or {}).get('PYTHONPATH') == '/tmp/s6f_dec' and (rec.get('B_probe') or {}).get('monai', [None])[0] == '0.6.0'
    and set(rec.get('stages') or {}) == {'env', 'inputs', 'B_probe', 'B'}, list(rec.get('stages') or {}))
mine_argv = json.load(open(os.path.join(out, 'B', 'argv_part0.json')))
ref_argv = json.load(open(os.path.join(H.REF, 'k1', 'B', 'argv_part0.json')))
norm = lambda a: [x.replace('\\', '/').replace(out.replace('\\', '/'), '/kaggle/working/k1')
                  .replace(os.path.join(TMPB, 't1', 'tmp').replace('\\', '/'), '/tmp/k1') for x in a]
ma, ra = norm(mine_argv), norm(ref_argv)
di = [i for i, (x, y) in enumerate(zip(ma, ra)) if x != y]
chk('T1 part0 worker argv == nartaa argv_part0 except the interpreter path and --deadline value', len(ma) == len(ra) and
    all(ma[i - 1] == '--deadline' or i == 0 for i in di), [(ma[i], ra[i]) for i in di])
dl = float(mine_argv[mine_argv.index('--deadline') + 1])
chk('T1 compact_infer --deadline = stop_at - 1200 s', abs(dl - (stop - 1200)) < 1.0, dl - stop)
# ---- T2
r, rec, out, _ = run_port('t2', 'kill_part1')
print('T2 killed part1 (no receipt)', flush=True)
B = rec.get('B') or {}
chk('T2 exit 0, ok, run_error recorded, part1 source predictions_csv', r.returncode == 0 and rec.get('ok') is True and B.get('run_error')
    and B.get('sources') == ['logits', 'predictions_csv'], (r.returncode, B.get('sources'), str(B.get('run_error'))[:80]))
bp = pd.read_csv(os.path.join(out, 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
d = float(np.abs(bp.loc[ref_b['StudyInstanceUID'], H.LAB].to_numpy() - ref_b.set_index('StudyInstanceUID')[H.LAB].to_numpy()).max())
chk('T2 coverage 1.0 and |dp| <= 5e-7 vs nartaa (csv 6-decimal rows)', B.get('coverage') == 1.0 and d <= 5e-7, 'max |dp| %.2e' % d)
# ---- T3
r, rec, out, _ = run_port('t3', 'timed_out_part0')
print('T3 part0 timed out', flush=True)
B = rec.get('B') or {}
bp = pd.read_csv(os.path.join(out, 'B', 'B_probs.csv'), dtype={'StudyInstanceUID': str})
chk('T3 exit 0, ok, coverage 2/3, timed-out part [0], no retry, 1 NaN row', r.returncode == 0 and rec.get('ok') is True and
    abs(B.get('coverage', 0) - 2 / 3) < 1e-9 and B.get('timed_out_parts') == [0] and B.get('retry') is None
    and int(bp[H.LAB].isna().all(axis=1).sum()) == 1, {k: B.get(k) for k in ('coverage', 'timed_out_parts', 'n_unscored')})
# ---- T4
r, rec, out, _ = run_port('t4', 'ci_sha')
print('T4 part0 receipt with another compact_infer sha', flush=True)
B = rec.get('B') or {}
chk('T4 part0 unscored (collect_part refused it), coverage 1/3', rec.get('ok') is True and abs(B.get('coverage', 0) - 1 / 3) < 1e-9
    and str(B.get('sources', [''])[0]).startswith('collect_failed'), (B.get('coverage'), str(B.get('sources'))[:120]))
# ---- T5
r, rec, out, _ = run_port('t5', 'manifest')
print('T5 code manifest mismatch', flush=True)
chk('T5 exit 3, ok False, reason names the manifest, no B_probs.csv', r.returncode == 3 and rec.get('ok') is False and
    'code_manifest.json' in str(rec.get('error')) and not os.path.isfile(os.path.join(out, 'B', 'B_probs.csv')), (r.returncode, rec.get('error')))
chk('driver stdout carries no Traceback / RuntimeError text on any path', not any(w in (r.stdout or '') for w in ('Traceback', 'RuntimeError')))
shutil.rmtree(TMPB, ignore_errors=True)
print('TEST_DRIVER %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
