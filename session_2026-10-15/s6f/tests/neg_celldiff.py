#!/usr/bin/env python
# neg_celldiff.py (10-15, Claude subagent): the cell-diff proof must FAIL on tampered copies of the built notebook:
#   (a) one character changed in a base cell, (b) the S6F cells moved before the V224 canary (insert, not append),
#   (c) the codec dataset dropped from the metadata, (d) the driver bytes changed (sha no longer the build's), (e) the driver with
#   nartaa's remove_submission() added. And PASS on an untouched copy.  usage: py -3.12 -I tests/neg_celldiff.py
import os, sys, json, shutil, base64, hashlib, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F = os.path.dirname(HERE)
PROOF = os.path.join(S6F, 'celldiff_v224flips6f_proof.py')
KD = os.path.join(S6F, 'kernel_v224flips6f')
NB = os.path.join(KD, 'rsna_sub_v224flips6f.ipynb')
T = os.path.join(HERE, '_tmp_celldiff')
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T)
R = []


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def run(name, fn_nb=None, fn_meta=None, expect_pass=False):
    nb = json.load(open(NB, encoding='utf-8'))
    meta = json.load(open(os.path.join(KD, 'kernel-metadata.json'), encoding='utf-8'))
    if fn_nb:
        fn_nb(nb)
    if fn_meta:
        fn_meta(meta)
    p, m = os.path.join(T, name + '.ipynb'), os.path.join(T, name + '.json')
    open(p, 'wb').write(dumps(nb).encode('utf-8'))
    json.dump(meta, open(m, 'w', encoding='utf-8'), indent=1)
    r = subprocess.run([sys.executable, '-I', PROOF, '--nb', p, '--meta', m], capture_output=True, text=True, timeout=300)
    v = [l for l in r.stdout.splitlines() if l.startswith('S6F_CELLDIFF ') and ('PASS' in l or 'FAIL' in l)][-1:]
    ok = (r.returncode == 0) if expect_pass else (r.returncode != 0)
    R.append(ok)
    fails = [l.strip()[5:70] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    print('  %s %s %s: %s %s' % ('OK  ' if ok else 'FAIL', 'POSITIVE' if expect_pass else 'NEGATIVE', name, v, fails[:3]), flush=True)


def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def tamper_base(nb):
    nb['cells'][13]['source'] = src(nb['cells'][13]).replace('W_RAPTOR, W_OURS = 0.5, 0.5', 'W_RAPTOR, W_OURS = 0.5, 0.50')


def move_before_canary(nb):
    c = nb['cells']
    nb['cells'] = c[:14] + c[15:18] + [c[14]]


def driver_edit(nb, extra):
    s = src(nb['cells'][15])
    b64 = s.split("DRIVER_B64 = '", 1)[1].split("'", 1)[0]
    d = base64.b64decode(b64) + extra
    s = s.replace(b64, base64.b64encode(d).decode('ascii'))
    old = s.split("DRIVER_SHA = '", 1)[1].split("'", 1)[0]
    nb['cells'][15]['source'] = s.replace(old, hashlib.sha256(d).hexdigest())


run('untouched', expect_pass=True)
run('a_base_cell_changed', tamper_base)
run('b_s6f_cells_before_v224_canary', move_before_canary)
run('c_codec_dataset_dropped', fn_meta=lambda m: m['dataset_sources'].remove('glorysing/rsna-knee-offline-codecs-r09'))
run('d_driver_bytes_changed', lambda nb: driver_edit(nb, b'\n# changed\n'))
run('e_driver_with_remove_submission', lambda nb: driver_edit(nb, b'\ndef remove_submission():\n    pass\n'))
shutil.rmtree(T, ignore_errors=True)
print('NEG_CELLDIFF %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
