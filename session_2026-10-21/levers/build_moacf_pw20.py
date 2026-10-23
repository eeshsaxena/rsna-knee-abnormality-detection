#!/usr/bin/env python
# build_moacf_pw20.py (10-21 ~12:10 IST, Claude, LEVERS synthesis; same construction as build_moacf_salt.py). LOCAL BUILD ONLY, UNARMED:
# nothing is pushed, committed, registered or submitted. Runs build_moacf.py (regime track, read only) with exactly two asserted text
# changes: BASES = pw20 57425269 only (a21/kernel_v224r6c4pw20j, nb sha 09ba09b9 = the a21 build receipt out_sha256), and its log / index
# file names (build_moacf_pw20.log, build_index_moacf_pw20.json). Same block, same asserts, same offline gate.
# Then rewrites kernels/build_index_moacf_all.json (all 6 MOACF builds, with base / meta / base nb sha from the receipts).
# Role: H2 option only (if H2 sets A = pw20, "A = the better public of pw20 and pw20 + MOACF" needs its own amendment clause).
# Usage: py -3.12 -I build_moacf_pw20.py
import os, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), '/')
SRC = HERE + '/build_moacf.py'
s = open(SRC, encoding='utf-8').read(); s0 = s
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
i0 = s.index('BASES = {'); i1 = s.index('CELL = 18')
old_bases = s[i0:i1]
assert "'swide': (S + '/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j'" in old_bases
rep(old_bases, "BASES = {'pw20': (S + '/a21/kernel_v224r6c4pw20j', 'v224r6c4pw20j', "
               "'09ba09b9529c6962dc17f692a68700eb5ad81e77392360ae18be3a406876abce')}\n")
rep("LOG = open(os.path.join(OUTD, 'build_moacf.log'), 'w', encoding='utf-8')", "LOG = open(os.path.join(OUTD, 'build_moacf_pw20.log'), 'w', encoding='utf-8')")
rep("open(os.path.join(OUTD, 'build_index_moacf.json'), 'w')", "open(os.path.join(OUTD, 'build_index_moacf_pw20.json'), 'w')")
rep("builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())", "builder_sha = %r" % hashlib.sha256(s0.encode('utf-8')).hexdigest())
ns = {'__file__': SRC, '__name__': '__main__'}
exec(compile(s, SRC + ' [pw20 variant via build_moacf_pw20.py]', 'exec'), ns)
K = HERE + '/kernels'
FL = {'pw20': (.2, 0., 0.), 'pw30': (.3, 0., 0.), 'pw25': (.25, 0., 0.), 'spl': (.25, .05, 0.), 'swide': (.25, .05, .1), 'salt': (.25, 0., .1)}
TAG = {'pw20': 'v224r6c4pw20moacf05j', 'pw30': 'v224r6c4pw30moacf05j', 'pw25': 'v224r6c4pw25moacf05j', 'spl': 'v224r6c4pw25lomm05mcl00moacf05j',
       'swide': 'v224r6c4pw25lomm05mcl10moacf05j', 'salt': 'v224r6c4pw25mcl10moacf05j'}
open(K + '/kernel_v224r6c4pw20moacf05j/NOT_ARMED.txt', 'w').write(
    'NOT ARMED. Local build (levers/build_moacf_pw20.py, 10-21). H2 option only. Do not push without Eesh GO + PREREG amendment + registry '
    'merge + moacf_commit_gate.py PASS 29/29 on the fetched commit.\n')
idx = {}
for b, t in TAG.items():
    rc = json.load(open(K + '/build_receipt_%s.json' % t, encoding='utf-8'))
    nbp = K + '/kernel_%s/rsna_sub_%s.ipynb' % (t, t); mp = K + '/kernel_%s/kernel-metadata.json' % t
    nsha = hashlib.sha256(open(nbp, 'rb').read()).hexdigest(); msha = hashlib.sha256(open(mp, 'rb').read()).hexdigest()
    assert nsha == rc['out_nb_sha256'] and msha == rc['meta_sha256'] and rc['offline_gate']['PASS'] and rc['base'] == b, (b, t)
    assert os.path.isfile(K + '/kernel_%s/NOT_ARMED.txt' % t)
    idx['moacf_' + b] = dict(tag=t, base=b, a_moacf=rc['a_moacf'], flags=dict(a_oa=FL[b][0], a_loamm=FL[b][1], a_mclmm=FL[b][2]),
                             nb_sha256=nsha, meta_sha256=msha, base_nb_sha256=rc['base_nb_sha256'], offline_gate=rc['offline_gate']['PASS'],
                             gold58_INFO=rc['gold58_INFO'])
json.dump(idx, open(K + '/build_index_moacf_all.json', 'w', encoding='utf-8'), indent=1)
print('MOACF_INDEX_ALL', {k: v['nb_sha256'][:16] for k, v in idx.items()})
