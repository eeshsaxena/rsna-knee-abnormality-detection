# neg_celldiff.py (10-15, Claude subagent, kcstage): the cell-diff proof must FAIL on broken variants of the built stand-in kernel.
#   py -3.12 -I tests/neg_celldiff.py  -> NEG_CELLDIFF PASS n/m
import os, sys, json, base64, shutil, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
PROOF = os.path.join(KS, 'celldiff_v224s6fkc_proof.py')
KDIR = os.path.join(KS, 'kernel_v224s6fkcstandin')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print('  %s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


def dumps(nb):
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


def run(d):
    r = subprocess.run([sys.executable, '-I', PROOF, d, '--build', os.path.join(KS, 'build_receipt_v224s6fkcstandin.json')],
                       capture_output=True, text=True, timeout=300)
    last = [l for l in r.stdout.splitlines() if l.startswith('KC_CELLDIFF ')][-1:] or ['(none)']
    return r.returncode, last[0], [l.strip()[5:90] for l in r.stdout.splitlines() if l.strip().startswith('FAIL')]


rc, last, _ = run(KDIR)
chk('P0 the built stand-in kernel PASSes', rc == 0 and ' PASS ' in last, last)
tmp = tempfile.mkdtemp(prefix='kcneg_')


def variant(name, fn_nb=None, fn_meta=None):
    d = os.path.join(tmp, name.split()[0])
    shutil.copytree(KDIR, d)
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json'), encoding='utf-8'))
    p = os.path.join(d, meta['code_file'])
    nb = json.loads(open(p, 'rb').read())
    if fn_nb:
        fn_nb(nb)
        open(p, 'wb').write(dumps(nb).encode('utf-8'))
    if fn_meta:
        fn_meta(meta)
        open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8').write(json.dumps(meta, indent=1))
    rc, last, fails = run(d)
    chk('%s -> FAIL' % name, rc != 0 and ' FAIL ' in last, (last, fails[:3]))


def src_set(c, t):
    c['source'] = t


def f1(nb):
    c = nb['cells'][13]; src_set(c, (''.join(c['source']) if isinstance(c['source'], list) else c['source']) + '\n# x')


def f2(nb):
    k = nb['cells'].pop(18); nb['cells'].insert(17, k)


def f3(nb):
    c = nb['cells'][18]; s = c['source']
    i = s.index("DRIVER_B64 = '") + len("DRIVER_B64 = '")
    j = s.index("'", i)
    b = base64.b64decode(s[i:j]) + b'\n# tampered\n'
    src_set(c, s[:i] + base64.b64encode(b).decode() + s[j:])


def f4(meta):
    meta['dataset_sources'] = meta['dataset_sources'][:-1]


def f5(nb):
    c = nb['cells'][19]; src_set(c, c['source'].replace('= 6.9, 1.5, 8.2, 180.0, 1800.0', '= 7.5, 1.5, 8.2, 180.0, 1800.0'))


def f6(nb):
    c = nb['cells'][20]; src_set(c, c['source'] + "\nimport os\nos.environ['PYTHONPATH'] = '/tmp/x'\n")


def f7(nb):
    c = nb['cells'][20]; src_set(c, c['source'].replace('"Baker\'s"', "'ACL'", 1))


def f8(nb):
    nb['cells'] = nb['cells'][:18] + nb['cells'][19:]


variant('N1 a base cell changed', f1)
variant('N2 a KC cell before the S6F BLEND', f2)
variant('N3 embedded driver bytes changed', f3)
variant('N4 KC dataset dropped from the metadata', fn_meta=f4)
variant('N5 RUN cell G_KC 7.5 vs receipt 6.9', f5)
variant('N6 a KC cell sets os.environ PYTHONPATH', f6)
variant("N7 KC canary moved to ACL (= S6F's)", f7)
variant('N8 SETUP cell missing', f8)
shutil.rmtree(tmp, ignore_errors=True)
print('NEG_CELLDIFF %s %d/%d' % ('PASS' if all(R) else 'FAIL', sum(R), len(R)), flush=True)
sys.exit(0 if all(R) else 1)
