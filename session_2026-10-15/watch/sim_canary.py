#!/usr/bin/env python
# sim_canary.py (2026-10-14, Claude subagent): offline test of swap_builder.py's GENERIC canary cell. Takes a notebook built
# with --profile generic, extracts ONLY its appended canary cell (our own generated code; nothing of the public notebook is
# executed), points /kaggle/working and the competition mount at temp dirs, and runs 7 cases. Prints SIM_CANARY PASS/FAIL n/m.
#   usage: python sim_canary.py BUILT_NOTEBOOK.ipynb HER_SUBMISSION.csv
# The notebook must have been built with --canary-expr "r224 is None" --canary-expr "not R224_OK" --canary-col Fracture.
import os, sys, io, json, shutil, tempfile, contextlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.normpath(os.path.join(HERE, '..', '..', '_data', 'sample_submission.csv'))
nb_path, her_sub = sys.argv[1], sys.argv[2]
cell = json.load(open(nb_path, encoding='utf-8'))['cells'][-1]
CANARY = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
assert CANARY.startswith('# [eeshsaxena CANARY (generic'), 'last cell is not the generic canary'
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def run(case, ns, sub_df=None, receipt=None, with_sample=True):
    tmp = tempfile.mkdtemp(prefix='simcan_')
    work = os.path.join(tmp, 'working')
    comp = os.path.join(tmp, 'input', 'rsna-knee-abnormality-detection')
    os.makedirs(work)
    os.makedirs(comp)
    if with_sample:
        shutil.copy(SAMPLE, os.path.join(comp, 'sample_submission.csv'))
    sp = os.path.join(work, 'submission.csv')
    if sub_df is None:
        shutil.copy(her_sub, sp)
    else:
        sub_df.to_csv(sp, index=False)
    if receipt is not None:
        json.dump(receipt, open(os.path.join(work, 'blend_receipt.json'), 'w'))
    before = open(sp, 'rb').read()
    code = (CANARY.replace('/kaggle/input/competitions/rsna-knee-abnormality-detection', os.path.join(tmp, 'nope').replace('\\', '/'))
            .replace('/kaggle/input/rsna-knee-abnormality-detection', comp.replace('\\', '/'))
            .replace("'/kaggle/working'", repr(work.replace('\\', '/'))))
    assert '/kaggle/' not in code, 'a /kaggle path was not redirected'
    out, err = io.StringIO(), None
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(code, '<canary:%s>' % case, 'exec'), dict(ns))
        except AssertionError as e:
            err = e
    rc = json.load(open(os.path.join(work, 'canary_receipt.json'))) if os.path.isfile(os.path.join(work, 'canary_receipt.json')) else None
    after = open(sp, 'rb').read()
    shutil.rmtree(tmp, ignore_errors=True)
    return out.getvalue(), err, rc, before, after


her = pd.read_csv(her_sub, dtype={'StudyInstanceUID': str})
labels = [c for c in her.columns if c != 'StudyInstanceUID']
print('SIM_CANARY', nb_path, flush=True)

o, e, rc, b, a = run('A clean', {'r224': object(), 'R224_OK': True})
chk('A clean: CANARY_CLEAN + SWAP_DONE, no error', 'CANARY_CLEAN' in o and 'SWAP_DONE' in o and e is None, o.strip()[-120:])
chk('A clean: submission.csv byte-identical', a == b)
chk('A clean: receipt fired == [], triggers False', rc and rc['fired'] == [] and rc['triggers'] == {'r224 is None': False, 'not R224_OK': False}, rc and rc['triggers'])

o, e, rc, b, a = run('B fired', {'r224': None, 'R224_OK': False})
s = pd.read_csv(io.BytesIO(a), dtype={'StudyInstanceUID': str})
keep = [c for c in labels if c != 'Fracture']
chk('B fired: CANARY_FIRED', 'CANARY_FIRED' in o and e is None, o.strip()[-160:])
chk('B fired: Fracture == 0.5 in every row', bool((s['Fracture'] == 0.5).all()))
chk('B fired: other 11 columns unchanged (values)', float(np.abs(s[keep].values - her[keep].values).max()) == 0.0)
chk('B fired: ids / columns unchanged', s['StudyInstanceUID'].tolist() == her['StudyInstanceUID'].tolist() and list(s.columns) == list(her.columns))
chk('B fired: receipt lists both triggers', rc and sorted(rc['fired']) == ['not R224_OK', 'r224 is None'], rc and rc['fired'])

o, e, rc, b, a = run('C undefined', {})
chk('C undefined variables fire (fail visible)', rc and rc['triggers'] == {'r224 is None': 'undefined', 'not R224_OK': 'undefined'}
    and 'CANARY_FIRED' in o, rc and rc['triggers'])

bad = her.copy()
bad.loc[0, 'StudyInstanceUID'] = '1.2.3'
o, e, rc, b, a = run('D bad ids', {'r224': 1, 'R224_OK': True}, sub_df=bad)
chk('D wrong ids: raises (visible error), no receipt', e is not None and 'ids' in str(e) and rc is None, repr(e)[:100])

nan = her.copy()
nan.loc[1, 'ACL'] = np.nan
o, e, rc, b, a = run('E NaN', {'r224': 1, 'R224_OK': True}, sub_df=nan)
chk('E NaN value: raises', e is not None and 'finite' in str(e), repr(e)[:100])

o, e, rc, b, a = run('F receipt fallbacks', {'r224': 1, 'R224_OK': True}, receipt={'raptor_failed': 2, 'ours_failed': 0, 'mode': 'blend'})
chk('F per-study fallbacks logged, not fired', 'CANARY_RECEIPT_FALLBACKS' in o and 'CANARY_CLEAN' in o and rc
    and rc['checks']['receipt_fallbacks'] == {'blend_receipt.json:raptor_failed': 2} and a == b, rc and rc['checks'])

o, e, rc, b, a = run('G no sample', {'r224': 1, 'R224_OK': True}, with_sample=False)
chk('G no sample: CANARY_NO_SAMPLE logged, still clean', 'CANARY_NO_SAMPLE' in o and 'CANARY_CLEAN' in o and e is None and a == b)

print('SIM_CANARY %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
