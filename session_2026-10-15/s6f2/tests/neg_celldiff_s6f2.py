#!/usr/bin/env python
# neg_celldiff_s6f2.py (10-15, Claude subagent): the cell-diff proof must FAIL on mutated copies of each s6f2 kernel (temp files;
# the real kernel dirs are only read). Offline, laptop CPU, -I.
#   per kernel (flrm, flips20): clean copy PASS; then each must FAIL on the intended check:
#   M1 one base cell +1 byte   M2 the S6F cells moved before the base's last cell   M3 codec dataset dropped from the metadata
#   M4 driver bytes changed (b64 of a modified driver, sha constant updated)   M5 BLEND blank rule reverted to the FLIP rule
#   M6 RUN G reverted to the FLIP 5.75   M7 one extra statement in RUN   M8 receipt notebook sha of another build
#   usage: py -3.12 -I tests/neg_celldiff_s6f2.py  -> NEG_CELLDIFF_S6F2 PASS/FAIL n/m
import os, sys, json, copy, base64, hashlib, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
S6F2 = os.path.dirname(HERE)
PROOF = os.path.join(S6F2, 'celldiff_s6f2_proof.py')
TMP = os.path.join(HERE, '_tmp_neg_celldiff')
KS = {'flrm': ('v224flrms6f', 17), 'flips20': ('v224flips6fs20', 24)}
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def run(kind, nb, meta, rec):
    r = subprocess.run([sys.executable, '-I', PROOF, '--kernel', kind, '--nb', nb, '--meta', meta, '--build', rec],
                       capture_output=True, text=True, timeout=300)
    last = ([l for l in r.stdout.splitlines() if l.strip()] or [''])[-1]
    fails = [l.strip()[5:80] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]
    return r.returncode, last, fails


shutil.rmtree(TMP, ignore_errors=True)
for kind, (tag, n) in KS.items():
    NB0 = json.load(open(os.path.join(S6F2, 'kernel_' + tag, 'rsna_sub_%s.ipynb' % tag), encoding='utf-8'))
    META0 = json.load(open(os.path.join(S6F2, 'kernel_' + tag, 'kernel-metadata.json'), encoding='utf-8'))
    REC0 = json.load(open(os.path.join(S6F2, 'build_receipt_%s.json' % tag), encoding='utf-8'))

    def case(name, mut_nb=None, mut_meta=None, mut_rec=None, expect_fail=True, must=()):
        d = os.path.join(TMP, kind, name)
        os.makedirs(d, exist_ok=True)
        nb, meta, rec = copy.deepcopy(NB0), copy.deepcopy(META0), copy.deepcopy(REC0)
        if mut_nb:
            mut_nb(nb)
        if mut_meta:
            mut_meta(meta)
        raw = dumps(nb).encode('utf-8')
        if not mut_rec:
            rec['notebook_sha256'] = hashlib.sha256(raw).hexdigest()     # the receipt follows the mutated notebook unless M8
        else:
            mut_rec(rec)
        p_nb, p_meta, p_rec = os.path.join(d, 'nb.ipynb'), os.path.join(d, 'kernel-metadata.json'), os.path.join(d, 'rec.json')
        open(p_nb, 'wb').write(raw)
        json.dump(meta, open(p_meta, 'w', encoding='utf-8'), indent=1)
        json.dump(rec, open(p_rec, 'w', encoding='utf-8'), indent=1)
        rc, last, fails = run(kind, p_nb, p_meta, p_rec)
        hit = all(any(m in f for f in fails) for m in must)
        ok = (rc != 0 and 'FAIL' in last and hit) if expect_fail else (rc == 0 and 'PASS' in last)
        chk('%s %s %s: %s' % (kind, 'NEGATIVE' if expect_fail else 'POSITIVE', name, last), ok, ' || '.join(fails[:3]))

    def m1(nb):
        c = nb['cells'][5]
        c['source'] = src(c) + ' '

    def m2(nb):
        app = nb['cells'][n:n + 3]
        del nb['cells'][n:n + 3]
        nb['cells'][n - 1:n - 1] = app

    def m4(nb):
        c = nb['cells'][n]
        s = src(c)
        b64 = s.split("DRIVER_B64 = '", 1)[1].split("'", 1)[0]
        old_sha = s.split("DRIVER_SHA = '", 1)[1].split("'", 1)[0]
        drv = base64.b64decode(b64) + b'\n# changed\n'
        s = s.replace(b64, base64.b64encode(drv).decode('ascii')).replace(old_sha, hashlib.sha256(drv).hexdigest())
        c['source'] = s

    def m5(nb):
        c = nb['cells'][n + 2]
        s = src(c)
        i = s.index('    _fired = list(')
        j = s.index('\n', s.index('    blank += [c for c in labels')) + 1
        c['source'] = s[:i] + "    blank = ['Fracture'] if 'r224->Fracture' in ((setup or {}).get('v224_canary') or {}).get('fired', []) else []\n" + s[j:]

    def m6(nb):
        c = nb['cells'][n + 1]
        s = src(c)
        g = '6.0' if kind == 'flrm' else '6.25'
        c['source'] = s.replace('MIN_RUN_S = %s, 3.3, 8.2' % g, 'MIN_RUN_S = 5.75, 3.3, 8.2')

    def m7(nb):
        c = nb['cells'][n + 1]
        c['source'] = src(c) + '\n_S6F_EXTRA = 1\n'

    case('clean_copy', expect_fail=False)
    case('M1_base_cell_plus_1_byte', mut_nb=m1, must=('alignment', 'equal as whole cell dicts', 're-serialises'))
    case('M2_s6f_before_base_end', mut_nb=m2, must=('alignment',))
    case('M3_codec_dataset_dropped', mut_meta=lambda m: m['dataset_sources'].remove('glorysing/rsna-knee-offline-codecs-r09'), must=('kernel-metadata',))
    case('M4_driver_changed', mut_nb=m4, must=('driver embedded',))
    case('M5_flip_blank_rule', mut_nb=m5, must=('undoing exactly',))
    case('M6_G_5.75', mut_nb=m6, must=('undoing exactly',))
    case('M7_extra_statement', mut_nb=m7, must=('undoing exactly',))
    case('M8_receipt_of_another_build', mut_rec=lambda r: r.update(notebook_sha256='0' * 64), must=('build receipt',))
shutil.rmtree(TMP, ignore_errors=True)
print('NEG_CELLDIFF_S6F2 %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
