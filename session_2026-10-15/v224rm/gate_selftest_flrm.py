#!/usr/bin/env python
# gate_selftest_flrm.py (2026-10-14, Claude subagent): positive self-test of v224flrm_check.py BEFORE the real commit output exists.
# Synthesises a V224-FLIP-RM output dir from two real commits on the same 3 studies: the V224-FLIP commit (out_v224flip_v1: log
# up to the r224 stage, flip worker logs + receipts, flip R7 ours_sub, r384 raw / sub) and the V224-F20-RM commit (out_v224rm_v1:
# mirror-TTA r224 raw / plain / mirror / sub + tta receipt + its R224_MIRROR_TTA / R224_STAGE log lines). The notebook's own RM
# cell (11, paths -> temp), stage 3 (14), CANARY (15) and RM_GUARD (16) are then executed there and their stdout appended to the
# log in cell order. Expected: V224FLRM_CHECK PASS. A second synthetic copy with the RM receipt removed after the fact must FAIL.
#   usage: python gate_selftest_flrm.py SCRATCH_DIR
import os, sys, json, shutil, io, contextlib, time, subprocess
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
FLIP_OUT = Path(ROOT) / 'session_2026-10-14' / 'v224' / 'out_v224flip_v1'
RM_OUT = Path(HERE) / 'out_v224rm_v1'
NB = Path(HERE) / 'kernel_v224flrm' / 'rsna_sub_v224flrm.ipynb'
SP = Path(sys.argv[1])
nb = json.load(open(NB, encoding='utf-8'))
S = [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) for c in nb['cells']]


def synth(dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(FLIP_OUT, dst, ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('submission.csv', 'blend_receipt.json', 'canary_receipt.json', 'raptor224_sub.csv'):
        os.remove(dst / f)
    for f in ('raptor224_sub.csv', 'raptor224_raw.csv', 'raptor224_raw_plain.csv', 'raptor224_raw_mirror.csv', 'raptor224_tta_receipt.json'):
        shutil.copy(RM_OUT / f, dst / f)
    data, pred, tmp = dst.parent / (dst.name + '_data'), dst.parent / (dst.name + '_pred'), dst.parent / (dst.name + '_tmp')
    for d in (data, pred / 'worker0', tmp):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copy(Path(ROOT) / '_data' / 'test.csv', data / 'test.csv')
    shutil.copy(Path(ROOT) / '_data' / 'sample_submission.csv', data / 'sample_submission.csv')
    shutil.copy(dst / 'ours_sub.csv', pred / 'worker0' / 'predictions.csv')
    for k, nm in ((3, 'raptor_run.py'), (10, 'raptor224_run.py')):
        open(tmp / nm, 'w', encoding='utf-8', newline='').write(S[k].split('\n', 1)[1])
    g = {'DATA': data, 'WORK': dst, 'pred': pred, 'R224_OK': True, 'RAPTOR_SECONDS': 54.0, 'NB_T0': time.time(), 'json': json,
         'time': time, 'display': lambda *a, **k: None, 'FLIP_TTA_FORCED_ON': True}
    W = str(dst).replace('\\', '/')
    b1, b2 = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(b1):
        exec(compile(S[11].replace("'/tmp/", "'" + str(tmp).replace('\\', '/') + '/'), 'rm', 'exec'), g)
    with contextlib.redirect_stdout(b2):
        exec(compile(S[14], 'stage3', 'exec'), g)
        exec(compile(S[15], 'canary', 'exec'), g)
        exec(compile(S[16].replace("'/kaggle/working'", repr(W)), 'guard', 'exec'), g)
    # log: the FLIP commit log up to (not incl.) its R224_STAGE chunk, then RM cell, the RM commit's r224 lines, then the tail
    fl = json.load(open(FLIP_OUT / 'rsna-sub-v224flip.log', encoding='utf-8'))
    k = next(i for i, e in enumerate(fl) if 'R224_STAGE rc=' in e.get('data', ''))
    rml = '\n'.join(e.get('data', '') for e in json.load(open(RM_OUT / 'rsna-sub-v224rm.log', encoding='utf-8')))
    r224_lines = [ln for ln in rml.splitlines() if 'R224_MIRROR_TTA' in ln or 'R224_STAGE rc=' in ln]
    t = float(fl[k].get('time', 0))
    new = fl[:k] + [{'stream_name': 'stdout', 'time': t, 'data': b1.getvalue()},
                    {'stream_name': 'stdout', 'time': t + 1, 'data': '\n'.join(r224_lines) + '\n'},
                    {'stream_name': 'stdout', 'time': t + 2, 'data': b2.getvalue()}]
    os.remove(dst / 'rsna-sub-v224flip.log')
    json.dump(new, open(dst / 'rsna-sub-v224flrm.log', 'w', encoding='utf-8'))
    return dst


SP.mkdir(parents=True, exist_ok=True)
ok = synth(SP / 'synth_flrm')
r = subprocess.run([sys.executable, os.path.join(HERE, 'v224flrm_check.py'), str(ok)], capture_output=True, text=True, encoding='utf-8')
last = r.stdout.strip().splitlines()[-1]
print('POSITIVE synthetic FLIP-RM:', last, 'rc', r.returncode)
print('\n'.join(ln for ln in r.stdout.splitlines() if ln.startswith('  ')))
bad = synth(SP / 'synth_flrm_bad')
os.remove(bad / 'raptor224_tta_receipt.json')       # the RM receipt vanishes after the run: the gate must notice
r2 = subprocess.run([sys.executable, os.path.join(HERE, 'v224flrm_check.py'), str(bad)], capture_output=True, text=True, encoding='utf-8')
last2 = r2.stdout.strip().splitlines()[-1]
print('NEGATIVE synthetic FLIP-RM without the tta receipt:', last2, 'rc', r2.returncode)
res = r.returncode == 0 and 'PASS' in last and r2.returncode != 0 and 'FAIL' in last2
print('GATE_SELFTEST', 'PASS' if res else 'FAIL')
sys.exit(0 if res else 1)
