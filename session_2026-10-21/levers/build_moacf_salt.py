#!/usr/bin/env python
# build_moacf_salt.py (10-21 ~11:30 IST, Claude, LEVERS synthesis). LOCAL BUILD ONLY, UNARMED: nothing is pushed, committed, registered or
# submitted. Runs build_moacf.py (regime track, read only) with exactly two asserted text changes: BASES = the A22 salt entry only
# (val/slate_a22.json: kernel_dir stack/kernels/kernel_v224r6c4pw25mcl10j, nb sha 051232e4), and its log / index file names
# (build_moacf_salt.log, build_index_moacf_salt.json) so the regime track's files are not overwritten. Same block, same asserts, same gate.
# Then writes kernels/build_index_moacf_all.json (all 5 MOACF builds, with base / meta / base nb sha from the receipts).
# Usage: py -3.12 -I build_moacf_salt.py
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
rep(old_bases, "BASES = {'salt': (S + '/stack/kernels/kernel_v224r6c4pw25mcl10j', 'v224r6c4pw25mcl10j', "
               "'051232e424f10840412f9618e56081681e571a6a77815132bac83fa99af2f0d1')}\n")
rep("LOG = open(os.path.join(OUTD, 'build_moacf.log'), 'w', encoding='utf-8')", "LOG = open(os.path.join(OUTD, 'build_moacf_salt.log'), 'w', encoding='utf-8')")
rep("open(os.path.join(OUTD, 'build_index_moacf.json'), 'w')", "open(os.path.join(OUTD, 'build_index_moacf_salt.json'), 'w')")
rep("builder_sha = sha(open(os.path.abspath(__file__), 'rb').read())", "builder_sha = %r" % hashlib.sha256(s0.encode('utf-8')).hexdigest())
ns = {'__file__': SRC, '__name__': '__main__'}
exec(compile(s, SRC + ' [salt variant via build_moacf_salt.py]', 'exec'), ns)
K = HERE + '/kernels'
FL = {'pw30': (.3, 0., 0.), 'pw25': (.25, 0., 0.), 'spl': (.25, .05, 0.), 'swide': (.25, .05, .1), 'salt': (.25, 0., .1)}
TAG = {'pw30': 'v224r6c4pw30moacf05j', 'pw25': 'v224r6c4pw25moacf05j', 'spl': 'v224r6c4pw25lomm05mcl00moacf05j',
       'swide': 'v224r6c4pw25lomm05mcl10moacf05j', 'salt': 'v224r6c4pw25mcl10moacf05j'}
idx = {}
for b, t in TAG.items():
    rc = json.load(open(K + '/build_receipt_%s.json' % t, encoding='utf-8'))
    nbp = K + '/kernel_%s/rsna_sub_%s.ipynb' % (t, t); mp = K + '/kernel_%s/kernel-metadata.json' % t
    nsha = hashlib.sha256(open(nbp, 'rb').read()).hexdigest(); msha = hashlib.sha256(open(mp, 'rb').read()).hexdigest()
    assert nsha == rc['out_nb_sha256'] and msha == rc['meta_sha256'] and rc['offline_gate']['PASS'] and rc['base'] == b, (b, t)
    assert os.path.isfile(K + '/kernel_%s/NOT_ARMED.txt' % t) or b == 'salt'
    idx['moacf_' + b] = dict(tag=t, base=b, a_moacf=rc['a_moacf'], flags=dict(a_oa=FL[b][0], a_loamm=FL[b][1], a_mclmm=FL[b][2]),
                             nb_sha256=nsha, meta_sha256=msha, base_nb_sha256=rc['base_nb_sha256'], offline_gate=rc['offline_gate']['PASS'],
                             gold58_INFO=rc['gold58_INFO'])
open(K + '/kernel_v224r6c4pw25mcl10moacf05j/NOT_ARMED.txt', 'w').write(
    'NOT ARMED. Local build (levers/build_moacf_salt.py, 10-21). Do not push without Eesh GO + PREREG amendment + registry merge + '
    'moacf_commit_gate.py PASS 29/29 on the fetched commit.\n')
json.dump(idx, open(K + '/build_index_moacf_all.json', 'w', encoding='utf-8'), indent=1)
print('MOACF_INDEX_ALL', {k: v['nb_sha256'][:16] for k, v in idx.items()})
