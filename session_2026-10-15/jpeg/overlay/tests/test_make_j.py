#!/usr/bin/env python
# test_make_j.py (2026-10-15, Claude subagent): offline tests of make_j.py (laptop CPU, no network).
#  D  determinism: rebuilding the 3 J kernels into temp dirs gives the same notebook / metadata bytes as the built ones;
#  A  the audit passes on the other lineup kernels (dry builds: V224-FLIP-RM, V224-FLIP-RM-AO-S, V224-FF-S, V224-F20);
#  R  refusals (MAKE_J REFUSED, nothing written): pulled dir with id_no, non-empty OUT, wrong base sha, bad slug, slug == base,
#     a J notebook as base, and synthetic bases (one text edit of a real base) where a decoding process would NOT inherit
#     PYTHONPATH: a cell sets / pops os.environ PYTHONPATH, a %%writefile stage script replaces PYTHONPATH for its children, the
#     R7 controller env not built from os.environ, the fleet env drops the inherited PYTHONPATH, R7 probe_dual worker env not from
#     os.environ, a launch with env=<function call>, S6F's PYTHONPATH pop without the anchors that make it safe.
#  J3 (review fix): the 5 forms the pre-fix audit passed (MEASURED: scratchpad audit probe) + 3 spelling variants, each one text
#     edit of a real base: a cell launches [sys.executable, '-I', '-u', script]; a %%writefile script launches with '-E'; a cell
#     slice-assigns sys.path; a %%writefile script reassigns sys.path; the R7 controller env PYTHONPATH = inherited.replace(...);
#     the fleet env PYTHONPATH = fleet_dir + os.pathsep + inherited.replace(...); a combined '-uI' option; os.system('python -E ..').
#  D compares with the CURRENT kernels in r2/ (the review rebuild); --make PATH runs every test against another make_j.py (e.g. a
#  pre-change copy placed so its relative paths resolve) to show which refusals it misses.
import os, sys, json, shutil, hashlib, tempfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OV = os.path.normpath(os.path.join(HERE, '..'))
S15 = os.path.normpath(os.path.join(OV, '..', '..'))
ROOT = os.path.normpath(os.path.join(S15, '..'))
MAKE = os.path.join(OV, 'make_j.py')
KDIR = os.path.join(OV, 'r2')
if '--make' in sys.argv:
    MAKE = os.path.abspath(sys.argv[sys.argv.index('--make') + 1])
BASES = {'v224flipj': os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip'),
         'v224flips6fj': os.path.join(S15, 's6f', 'kernel_v224flips6f'),
         'v224flips20j': os.path.join(S15, 'v224f', 'kernel_v224flips20')}
RES = []


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def mk(args):
    return subprocess.run([sys.executable, MAKE] + args, capture_output=True, text=True)


def rec(name, ok, info=''):
    RES.append(bool(ok))
    print('%s %s %s' % ('OK  ' if ok else 'FAIL', name, info), flush=True)


tmp = tempfile.mkdtemp(prefix='testmakej_')
# ---- D --------------------------------------------------------------------------------------------------------------------------
for tag, base in BASES.items():
    out = os.path.join(tmp, 'D', 'kernel_' + tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = mk([base, out, '--slug', 'eeshsaxena/rsna-sub-' + tag, '--receipt-dir', os.path.dirname(out)])
    built = os.path.join(KDIR, 'kernel_' + tag)
    same = r.returncode == 0 and all(sha(os.path.join(out, f)) == sha(os.path.join(built, f)) for f in os.listdir(built))
    rec('D %s rebuild byte-identical (notebook + metadata)' % tag, same, r.stdout.splitlines()[0][:110] if r.stdout else r.stderr[-300:])
# ---- A --------------------------------------------------------------------------------------------------------------------------
for tag, base in (('v224flrmj', os.path.join(S15, 'v224rm', 'kernel_v224flrm')), ('v224flrms20j', os.path.join(S15, 'v224f', 'kernel_v224flrms20')),
                  ('v224ffs20j', os.path.join(S15, 'v224f', 'kernel_v224ffs20')), ('v224f20jx', os.path.join(S15, 'v224f', 'kernel_v224f20'))):
    r = mk([base, os.path.join(tmp, 'A', tag), '--slug', 'eeshsaxena/rsna-sub-' + tag, '--dry-run'])
    rec('A %s dry build passes the audit' % tag, r.returncode == 0 and 'MAKE_J BUILT' in r.stdout, r.stdout.splitlines()[0][:110] if r.stdout else r.stderr[-200:])


# ---- R --------------------------------------------------------------------------------------------------------------------------
def refuse(name, args, want):
    r = mk(args)
    ok = r.returncode == 1 and 'MAKE_J REFUSED' in r.stdout and want in r.stdout
    out = args[1]
    ok = ok and (not os.path.exists(out) or not os.listdir(out) or name.startswith('non-empty'))
    rec('R %s' % name, ok, (r.stdout.strip() or r.stderr[-300:])[:170])


def synth(base_dir, edits, name):
    """copy a base kernel dir and apply exact text edits [(cell index, old, new)] to its notebook (compact serialisation kept)."""
    d = os.path.join(tmp, 'S', name)
    shutil.copytree(base_dir, d)
    meta = json.load(open(os.path.join(d, 'kernel-metadata.json')))
    p = os.path.join(d, meta['code_file'])
    nb = json.load(open(p, encoding='utf-8'))
    for i, old, new in edits:
        s = nb['cells'][i]['source']
        assert s.count(old) == 1, (name, i, old[:60], s.count(old))
        nb['cells'][i]['source'] = s.replace(old, new)
    open(p, 'wb').write(json.dumps(nb, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    return d


FLIP, S6F, FLIPS = BASES['v224flipj'], BASES['v224flips6fj'], BASES['v224flips20j']
o = lambda n: os.path.join(tmp, 'R', n)
refuse('pulled kernel dir (id_no in metadata)', [os.path.join(S15, 'v224rm', 'pulled_v224flip'), o('r1'), '--slug', 'eeshsaxena/rsna-sub-x1j'], 'id_no')
os.makedirs(o('r2'))
open(os.path.join(o('r2'), 'x.txt'), 'w').write('x')
refuse('non-empty OUT dir', [FLIP, o('r2'), '--slug', 'eeshsaxena/rsna-sub-x2j'], 'not empty')
refuse('wrong --expect-base-sha', [FLIP, o('r3'), '--slug', 'eeshsaxena/rsna-sub-x3j', '--expect-base-sha', '0' * 64], 'expected')
refuse('bad slug', [FLIP, o('r4'), '--slug', 'eeshsaxena/V224FLIPJ'], 'slug')
refuse('slug == base slug', [FLIP, o('r5'), '--slug', 'eeshsaxena/rsna-sub-v224flip'], 'base slug')
refuse('a J notebook as the base', [os.path.join(KDIR, 'kernel_v224flipj'), o('r6'), '--slug', 'eeshsaxena/rsna-sub-v224flipjj'], 'J DEC cell')
refuse('a cell sets os.environ PYTHONPATH', [synth(FLIP, [(13, '# Stage 3:', "import os\nos.environ['PYTHONPATH'] = '/x'\n# Stage 3:")], 's1'),
                                               o('r7'), '--slug', 'eeshsaxena/rsna-sub-x7j'], 'PYTHONPATH')
refuse('a cell pops os.environ PYTHONPATH', [synth(FLIP, [(13, '# Stage 3:', "import os\nos.environ.pop('PYTHONPATH', None)\n# Stage 3:")], 's2'),
                                               o('r8'), '--slug', 'eeshsaxena/rsna-sub-x8j'], 'changes os.environ')
refuse('a %%writefile stage script replaces PYTHONPATH for its children',
       [synth(FLIP, [(3, 'RUN_INFERENCE = True\n', "RUN_INFERENCE = True\nimport os\nos.environ['PYTHONPATH'] = '/opt/x'\n")], 's3'),
        o('r9'), '--slug', 'eeshsaxena/rsna-sub-x9j'], 'replaces PYTHONPATH')
refuse('R7 controller env not built from os.environ',
       [synth(FLIP, [(8, "env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')",
                      "env = {'PYTHONPATH': str(CODE), 'PATH': os.environ['PATH']}")], 's4'), o('r10'), '--slug', 'eeshsaxena/rsna-sub-x10j'],
       'cannot be shown to inherit')
refuse('fleet env drops the inherited PYTHONPATH',
       [synth(FLIPS, [(21, "'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}", "'PYTHONPATH': fleet_dir}")], 's5'),
        o('r11'), '--slug', 'eeshsaxena/rsna-sub-x11j'], 'cannot be shown to inherit')
refuse('R7 probe_dual worker env not from os.environ',
       [synth(FLIP, [(6, 'env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1"',
                      'env = dict(CUDA_VISIBLE_DEVICES=str(gi), OMP_NUM_THREADS="1"')], 's6'), o('r12'), '--slug', 'eeshsaxena/rsna-sub-x12j'],
       'cannot be shown to inherit')
refuse('a launch with env=<function call>',
       [synth(FLIP, [(11, "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600))",
                      "subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=min(_left, 2.5 * 3600), env=_mk_env())")], 's7'),
        o('r13'), '--slug', 'eeshsaxena/rsna-sub-x13j'], 'cannot be shown to inherit')
refuse('S6F PYTHONPATH pop without its safety anchor',
       [synth(S6F, [(15, "stock_pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]",
                     "stock_pp = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != '/x']")], 's8'),
        o('r14'), '--slug', 'eeshsaxena/rsna-sub-x14j'], 'cannot be shown to inherit')
# ---- J3: forms that drop /tmp/_dec which the pre-fix audit passed ------------------------------------------------------------
RAPTOR_RUN = "subprocess.run([sys.executable, '-u', '/tmp/raptor_run.py'], timeout=2 * 3600)"
refuse('J3 form 1: a cell launches python -I (ignores PYTHONPATH)',
       [synth(FLIP, [(4, RAPTOR_RUN, RAPTOR_RUN.replace("'-u'", "'-I', '-u'"))], 'j1'), o('r15'), '--slug', 'eeshsaxena/rsna-sub-x15j'],
       'ignores PYTHONPATH')
refuse('J3 form 2: a %%writefile script launches python -E',
       [synth(FLIP, [(3, 'RUN_INFERENCE = True\n', "RUN_INFERENCE = True\nimport subprocess, sys\nsubprocess.run([sys.executable, '-E', "
                                                    "'/tmp/w.py'])\n")], 'j2'), o('r16'), '--slug', 'eeshsaxena/rsna-sub-x16j'],
       'ignores PYTHONPATH')
refuse('J3 form 3: a cell slice-assigns sys.path',
       [synth(FLIP, [(13, '# Stage 3:', "import sys\nsys.path[:] = [p for p in sys.path if not p.startswith('/tmp')]\n# Stage 3:")], 'j3'),
        o('r17'), '--slug', 'eeshsaxena/rsna-sub-x17j'], 'removes / replaces sys.path')
refuse('J3 form 4: a %%writefile script reassigns sys.path',
       [synth(FLIP, [(3, 'RUN_INFERENCE = True\n', "RUN_INFERENCE = True\nimport sys\nsys.path = [p for p in sys.path if '/tmp/' not in p]\n")],
              'j4'), o('r18'), '--slug', 'eeshsaxena/rsna-sub-x18j'], 'removes / replaces sys.path')
R7PP = "PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', '')"
refuse('J3 form 5: R7 controller env PYTHONPATH = the inherited value with .replace() (dict form)',
       [synth(FLIP, [(8, R7PP, "PYTHONPATH=os.environ.get('PYTHONPATH', '').replace('/tmp/_dec:', '')")], 'j5'), o('r19'), '--slug',
        'eeshsaxena/rsna-sub-x19j'], 'cannot be shown to inherit')
FLPP = "'PYTHONPATH': fleet_dir + os.pathsep + os.environ.get('PYTHONPATH', '')}"
refuse('J3 form 5b: fleet env PYTHONPATH = fleet_dir + os.pathsep + inherited.replace() (** form)',
       [synth(FLIPS, [(21, FLPP, FLPP[:-1] + ".replace('/tmp/_dec:', '')}")], 'j6'), o('r20'), '--slug', 'eeshsaxena/rsna-sub-x20j'],
       'cannot be shown to inherit')
refuse('J3 variant: combined option -uI in a cell launch',
       [synth(FLIP, [(4, RAPTOR_RUN, RAPTOR_RUN.replace("'-u'", "'-uI'"))], 'j7'), o('r21'), '--slug', 'eeshsaxena/rsna-sub-x21j'],
       'ignores PYTHONPATH')
refuse('J3 variant: os.system shell string python -E in a cell',
       [synth(FLIP, [(13, '# Stage 3:', "import os\nos.system('python -E /tmp/w.py')\n# Stage 3:")], 'j8'), o('r22'), '--slug',
        'eeshsaxena/rsna-sub-x22j'], 'ignores PYTHONPATH')
refuse('J3 variant: IPython shell escape !python -I in a cell',
       [synth(FLIP, [(13, '# Stage 3:', "!python -I /tmp/w.py\n# Stage 3:")], 'j9'), o('r23'), '--slug', 'eeshsaxena/rsna-sub-x23j'],
       'ignores PYTHONPATH')
shutil.rmtree(tmp, ignore_errors=True)
ok = all(RES)
print('TEST_MAKE_J %s %d/%d' % ('PASS' if ok else 'FAIL', sum(RES), len(RES)), flush=True)
sys.exit(0 if ok else 1)
