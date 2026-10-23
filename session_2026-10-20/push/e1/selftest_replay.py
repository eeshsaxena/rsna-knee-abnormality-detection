# selftest_replay.py (10-20, Claude): self-test of the draft commit gate probe_check_e1eoa.py on REAL member outputs.
# Replays a notebook's own cells 14 / 15 / (16 freeze) / 18 on the member files of a real committed run (visible test, 3 rows)
# and writes a SYNTHETIC out dir (member files, kernel log, DEC / submission receipts copied from the real run; stage 3, final,
# blend / canary / s6f receipts replaced by the replay). T0 first proves the harness: replaying the shipped a20 eoa20 notebook
# on its own commit's members must reproduce that commit's submission_v224flip.csv and submission.csv.
# Usage: python -I selftest_replay.py   (reads only CSV / JSON data from the out dirs; executes only notebook cells we built)
import json, os, sys, time, hashlib, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd
R = 'C:/Users/eeshs/Downloads/rsna-knee'
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
O20 = R + '/session_2026-10-19/a20/out_v224r6c4eoa20j_v1'
NB20 = R + '/session_2026-10-19/a20/kernel_v224r6c4eoa20j/rsna_sub_v224r6c4eoa20j.ipynb'
NBE1 = HERE + '/kernel_v224r6c4eoa20e1j/rsna_sub_v224r6c4eoa20e1j.ipynb'
COPY = ['ours_sub.csv', 'raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv', 'submission_receipt.json', 'dec_receipt.json',
        'dec_envwatch.json', 's6f_setup_receipt.json', 'rsna-sub-v224r6c4eoa20j.log']


def replay(nbp, srcdir):
    nb = json.load(open(nbp, encoding='utf-8')); c14, c15, c18 = (src(nb['cells'][k]) for k in (14, 15, 18))
    ids = pd.read_csv(srcdir + '/submission_v224flip.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    T = tempfile.mkdtemp().replace(chr(92), '/'); D, W, P = Path(T + '/data'), Path(T + '/work'), Path(T + '/pred')
    for d in (D, W, P / 'worker0', W / 's6f' / 'B'): d.mkdir(parents=True)
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(D / 'test.csv', index=False)
    pd.DataFrame(columns=['StudyInstanceUID'] + LAB).to_csv(D / 'sample_submission.csv', index=False)
    for f in ('ours_sub.csv', 'raptor_raw.csv', 'raptor_sub.csv', 'raptor224_sub.csv'): shutil.copyfile(srcdir + '/' + f, W / f)
    shutil.copyfile(srcdir + '/ours_sub.csv', P / 'worker0' / 'predictions.csv')      # the commit's blend receipt: ours_failed 0
    ns = dict(DATA=D, WORK=W, pred=P, R224_OK=True, RAPTOR_SECONDS=0.0, NB_T0=time.time(), time=time, json=json,
              display=lambda *a, **k: None, FLIP_TTA_FORCED_ON=True)
    exec(compile(c14, nbp + '#cell14', 'exec'), ns); exec(compile(c15, nbp + '#cell15', 'exec'), ns)
    shutil.copyfile(W / 'submission.csv', W / 'submission_v224flip.csv')
    setup = {'base_sha256': sha(W / 'submission_v224flip.csv'), 'v224_canary': dict(ns['_canary'])}
    shutil.copyfile(srcdir + '/s6f/B/B_probs.csv', W / 's6f' / 'B' / 'B_probs.csv')
    run = {'status': 'RAN', 'killed': False, 'driver': {'ok': True, 'B': {'probs_sha256': sha(W / 's6f' / 'B' / 'B_probs.csv')}}}
    fn = c18.split('\n\ntry:\n    _S6F_BLEND = ')[0]; assert fn.count("W = '/kaggle/working'") == 1
    ns18 = {}; exec(compile(fn.replace("W = '/kaggle/working'", 'W = %r' % str(W).replace(chr(92), '/')), nbp + '#cell18', 'exec'), ns18)
    rec = ns18['_s6f_blend'](setup, run)
    return T, W, rec, dict(ns['_canary'])


ok = True
def chk(name, cond):
    global ok
    ok = ok and bool(cond); print(('OK   ' if cond else 'FAIL ') + name, flush=True)


# T0: harness vs the real Kaggle commit
T, W, rec, can = replay(NB20, O20)
mx = lambda a, b: float(np.abs(pd.read_csv(a)[LAB].values - pd.read_csv(b)[LAB].values).max())
nl = lambda x: x.replace(b'\r\n', b'\n')                      # Windows pandas writes CRLF, Kaggle (Linux) LF
st3b = nl((W / 'submission.csv').read_bytes()) == open(O20 + '/submission.csv', 'rb').read()
v2b = nl((W / 'submission_v224flip.csv').read_bytes()) == open(O20 + '/submission_v224flip.csv', 'rb').read()
chk('T0 replay of the shipped a20 eoa20 notebook on its own commit members: stage 3 bytes (LF-normalised) == commit %s (maxdiff %.1e), final bytes (LF-normalised) == commit %s (maxdiff %.1e), mode %s' % (
    v2b, mx(W / 'submission_v224flip.csv', O20 + '/submission_v224flip.csv'), st3b, mx(W / 'submission.csv', O20 + '/submission.csv'), rec.get('mode')),
    v2b and st3b and mx(W / 'submission_v224flip.csv', O20 + '/submission_v224flip.csv') < 1e-7 and mx(W / 'submission.csv', O20 + '/submission.csv') < 1e-7 and rec.get('mode') == 'S6F')
shutil.rmtree(T, ignore_errors=True)

# T3 fixture: the E1 candidate replayed on the same members -> SYNTHETIC out dir
DST = HERE + '/selftest/SYNTH_out_v224r6c4eoa20e1j_from_eoa20_v1'
shutil.rmtree(DST, ignore_errors=True); os.makedirs(DST + '/s6f/B')
T, W, rec, can = replay(NBE1, O20)
for f in COPY: shutil.copyfile(O20 + '/' + f, DST + '/' + f)
shutil.copyfile(O20 + '/s6f/B/B_probs.csv', DST + '/s6f/B/B_probs.csv')
for f in ('submission_v224flip.csv', 'submission.csv', 'blend_receipt.json'): shutil.copyfile(W / f, DST + '/' + f)
json.dump(can, open(DST + '/canary_receipt.json', 'w'), indent=1)
s6 = json.load(open(O20 + '/s6f_receipt.json', encoding='utf-8')); s6.update(blend=rec, mode=rec.get('mode'), fired=list(rec.get('fired') or []))
json.dump(s6, open(DST + '/s6f_receipt.json', 'w'), indent=1, default=str)
open(DST + '/SYNTHETIC.txt', 'w').write('SYNTHETIC self-test fixture (10-20, Claude): NOT a Kaggle output. Member files, kernel log and DEC / submission '
                                       'receipts copied from session_2026-10-19/a20/out_v224r6c4eoa20j_v1; stage 3, final and blend / canary / s6f '
                                       'receipts = local replay of kernel_v224r6c4eoa20e1j cells 14 / 15 / 18 (selftest_replay.py).\n')
shutil.rmtree(T, ignore_errors=True)
chk('T3 fixture written: mode %s, e1 %s, a_oa %s, final differs from the eoa20 commit (maxdiff %.3f)' % (
    rec.get('mode'), rec.get('e1'), rec.get('a_oa'), mx(DST + '/submission.csv', O20 + '/submission.csv')),
    rec.get('mode') == 'S6F' and rec.get('e1') == 2.0 and rec.get('a_oa') == 0.2 and mx(DST + '/submission.csv', O20 + '/submission.csv') > 1e-3)
# T4: the same for the a21 pw20 commit (contingency base): harness vs the real commit, then the alt E1-on-pw20 fixture
O21 = R + '/session_2026-10-20/a21/out_v224r6c4pw20j_v1'
NB21 = R + '/session_2026-10-20/a21/kernel_v224r6c4pw20j/rsna_sub_v224r6c4pw20j.ipynb'
NBP = HERE + '/alt/kernel_v224r6c4pw20e1j/rsna_sub_v224r6c4pw20e1j.ipynb'
T, W, rec, can = replay(NB21, O21)
v2b = nl((W / 'submission_v224flip.csv').read_bytes()) == open(O21 + '/submission_v224flip.csv', 'rb').read()
st3b = nl((W / 'submission.csv').read_bytes()) == open(O21 + '/submission.csv', 'rb').read()
chk('T4 replay of the shipped a21 pw20 notebook on its own commit members: stage 3 bytes == commit %s, final bytes == commit %s, mode %s, oa_mode %s' % (
    v2b, st3b, rec.get('mode'), rec.get('oa_mode')), v2b and st3b and rec.get('mode') == 'S6F' and rec.get('oa_mode') == 'pw')
shutil.rmtree(T, ignore_errors=True)
DST2 = HERE + '/selftest/SYNTH_out_v224r6c4pw20e1j_from_pw20_v1'
shutil.rmtree(DST2, ignore_errors=True); os.makedirs(DST2 + '/s6f/B')
T, W, rec, can = replay(NBP, O21)
for f in COPY[:-1] + ['rsna-sub-v224r6c4pw20j.log']: shutil.copyfile(O21 + '/' + f, DST2 + '/' + f)
shutil.copyfile(O21 + '/s6f/B/B_probs.csv', DST2 + '/s6f/B/B_probs.csv')
for f in ('submission_v224flip.csv', 'submission.csv', 'blend_receipt.json'): shutil.copyfile(W / f, DST2 + '/' + f)
json.dump(can, open(DST2 + '/canary_receipt.json', 'w'), indent=1)
s6 = json.load(open(O21 + '/s6f_receipt.json', encoding='utf-8')); s6.update(blend=rec, mode=rec.get('mode'), fired=list(rec.get('fired') or []))
json.dump(s6, open(DST2 + '/s6f_receipt.json', 'w'), indent=1, default=str)
open(DST2 + '/SYNTHETIC.txt', 'w').write('SYNTHETIC self-test fixture (10-20, Claude): NOT a Kaggle output. Members, kernel log and DEC / submission receipts '
                                        'copied from session_2026-10-20/a21/out_v224r6c4pw20j_v1; stage 3, final and blend / canary / s6f receipts = local '
                                        'replay of alt/kernel_v224r6c4pw20e1j cells 14 / 15 / 18 (selftest_replay.py).\n')
shutil.rmtree(T, ignore_errors=True)
chk('T4 fixture written: mode %s, e1 %s, a_oa %s, oa_mode %s' % (rec.get('mode'), rec.get('e1'), rec.get('a_oa'), rec.get('oa_mode')),
    rec.get('mode') == 'S6F' and rec.get('e1') == 2.0 and rec.get('a_oa') == 0.2 and rec.get('oa_mode') == 'pw')
print('SELFTEST_REPLAY', 'PASS' if ok else 'FAIL')
