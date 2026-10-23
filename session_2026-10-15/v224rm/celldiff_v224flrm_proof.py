#!/usr/bin/env python
# celldiff_v224flrm_proof.py (2026-10-14, Claude subagent): proof that V224-FLIP-RM (eeshsaxena/rsna-sub-v224flrm) is EXACTLY the
# submitted V224-FLIP notebook (session_2026-10-14/v224/kernel_v224flip, rsna-sub-v224flip v1, ref 57174021) plus ONLY the two
# r224 mirror inserts (RM, RM_GUARD), and that those two inserts are byte-identical to the reviewed + commit-tested v224rm build
# (kernel_v224rm cells 10 and 23). Independent of build_v224rm.py (imports nothing from it). No em dashes.
#   usage: python celldiff_v224flrm_proof.py  -> prints FLRM_CELLDIFF PASS/FAIL n/m; exit 0 only on PASS
import os, sys, json, hashlib, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
FLIP_NB = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'rsna_sub_v224flip.ipynb')
FLIP_META = os.path.join(ROOT, 'session_2026-10-14', 'v224', 'kernel_v224flip', 'kernel-metadata.json')
FLIP_SHA = '73a9d5620d0a637ed171ecec13bcf243bd0af67992b5b4cb8ad7f35bf72b1b1b'
PULLED = os.path.join(HERE, 'pulled_v224flip', 'rsna-sub-v224flip.ipynb')          # kaggle kernels pull of the submitted kernel
RM_NB = os.path.join(HERE, 'kernel_v224rm', 'rsna_sub_v224rm.ipynb')                # reviewed + commit-tested V224-F20-RM v1
NEW_NB = os.path.join(HERE, 'kernel_v224flrm', 'rsna_sub_v224flrm.ipynb')
NEW_META = os.path.join(HERE, 'kernel_v224flrm', 'kernel-metadata.json')
R = []


def chk(name, ok, info=''):
    R.append(bool(ok))
    print(f'  {"OK  " if ok else "FAIL"} {name} {info}', flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s


def dumps(nb):   # varshitha's serialisation (compact, ensure_ascii False), as every V224 kernel file is written
    return json.dumps(nb, ensure_ascii=False, separators=(',', ':'))


flip_raw, new_raw = open(FLIP_NB, 'rb').read(), open(NEW_NB, 'rb').read()
flip, new, rmk = json.loads(flip_raw), json.loads(new_raw), json.load(open(RM_NB, encoding='utf-8'))
print('FLRM_CELLDIFF base', os.path.relpath(FLIP_NB, ROOT), 'new', os.path.relpath(NEW_NB, ROOT), flush=True)
chk('base file == the V224-FLIP kernel that was pushed (sha 73a9d562)', sha(flip_raw) == FLIP_SHA, sha(flip_raw)[:16])
if os.path.isfile(PULLED):
    pl = json.load(open(PULLED, encoding='utf-8'))
    same = [(c['cell_type'], src(c)) for c in pl['cells']] == [(c['cell_type'], src(c)) for c in flip['cells']]
    chk('base cells == kaggle kernels pull of eeshsaxena/rsna-sub-v224flip (the submitted kernel; type + source of all %d cells)'
        % len(flip['cells']), same, '%d vs %d cells' % (len(pl['cells']), len(flip['cells'])))
else:
    chk('pulled copy of the submitted V224-FLIP present', False, PULLED)

FC, NC = flip['cells'], new['cells']
chk('cell counts: base 15, new 17', (len(FC), len(NC)) == (15, 17), (len(FC), len(NC)))
sm = difflib.SequenceMatcher(a=[(c['cell_type'], src(c)) for c in FC], b=[(c['cell_type'], src(c)) for c in NC], autojunk=False)
ops = [op for op in sm.get_opcodes()]
for tag, i1, i2, j1, j2 in ops:
    print(f'    {tag:7s} base[{i1}:{i2}] -> new[{j1}:{j2}]', flush=True)
ins = [j for tag, i1, i2, j1, j2 in ops if tag == 'insert' for j in range(j1, j2)]
other = [op for op in ops if op[0] not in ('equal', 'insert')]
chk('alignment: only EQUAL runs and INSERTs (no replace / delete)', not other, other)
chk('inserted cells are exactly new[11] (after her r224 %%writefile) and new[16] (last)', ins == [11, 16], ins)
chk('new[10] is her r224 %%writefile, new[12] her stage 2b, new[15] the V224 CANARY',
    src(NC[10]).startswith('%%writefile /tmp/raptor224_run.py\n') and src(NC[12]).startswith('# Stage 2b [s5]')
    and src(NC[15]).startswith('# [eeshsaxena CANARY'))
back = json.loads(new_raw)
back['cells'] = [c for k, c in enumerate(NC) if k not in (11, 16)]
chk('new notebook minus cells 11 and 16 re-serialises to the base FILE byte for byte (cell metadata, outputs, notebook metadata)',
    dumps(back).encode('utf-8') == flip_raw, sha(dumps(back).encode('utf-8'))[:16])
chk('notebook-level metadata / nbformat equal', new.get('metadata') == flip.get('metadata') and
    (new.get('nbformat'), new.get('nbformat_minor')) == (flip.get('nbformat'), flip.get('nbformat_minor')))
RC = rmk['cells']
rm_i = [k for k, c in enumerate(RC) if src(c).startswith('# [eeshsaxena V224-RM; the ONE change')]
gd_i = [k for k, c in enumerate(RC) if src(c).startswith('# [eeshsaxena V224-RM GUARD')]
chk('reviewed v224rm kernel has one RM (cell 10) and one RM_GUARD (cell 23)', (rm_i, gd_i) == ([10], [23]), (rm_i, gd_i))
chk('new[11] == reviewed RM cell, byte for byte (whole cell dict)', NC[11] == RC[10], sha(src(NC[11]).encode())[:16])
chk('new[16] == reviewed RM_GUARD cell, byte for byte (whole cell dict)', NC[16] == RC[23], sha(src(NC[16]).encode())[:16])
# the RM cell carries her shas + the patched-script sha, identical to the commit-tested kernel by the equality above; print them
for key in ("_RM_R224_HER_SHA = '", "_RM_R384_HER_SHA = '", "_RM_R224_RM_SHA = '"):
    t = src(NC[11]); i = t.index(key) + len(key)
    print('    info', key.strip(" ='"), t[i:i + 16], flush=True)
# every sha assert / canary of the base is still present, unchanged and in order (implied by the byte proof; listed for review)
needles = ['7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6',
           '8401812c', 'FLIP_TTA_FORCED', 'CANARY_FIRED', 'CANARY_CLEAN', 'V224_DONE']
found = {nd: [k for k, c in enumerate(NC) if nd in src(c)] for nd in needles}
basef = {nd: [k for k, c in enumerate(FC) if nd in src(c)] for nd in needles}
remap = lambda k: k + (k >= 11)          # base index -> new index (one insert before base cell 11)
chk('every base sha / flip / canary needle sits in the same base cells (re-indexed), none only in the inserts',
    all(found[nd][:len(basef[nd])] == [remap(k) for k in basef[nd]] and set(found[nd]) - {remap(k) for k in basef[nd]} <= {11, 16}
        for nd in needles), {nd: found[nd] for nd in needles})
fm, nm = json.load(open(FLIP_META, encoding='utf-8')), json.load(open(NEW_META, encoding='utf-8'))
diffk = sorted(k for k in set(fm) | set(nm) if fm.get(k) != nm.get(k))
chk('kernel-metadata == V224-FLIP except id / title / code_file (same 3 datasets, docker pin, T4 x2, private, no internet)',
    diffk == ['code_file', 'id', 'title'] and nm['id'] == 'eeshsaxena/rsna-sub-v224flrm' and nm['is_private'] is True
    and nm['enable_internet'] is False and nm['enable_gpu'] is True, {k: (fm.get(k), nm.get(k)) for k in diffk})
print('    info new notebook sha256', sha(new_raw), flush=True)
print(f'FLRM_CELLDIFF {"PASS" if all(R) else "FAIL"} {sum(R)}/{len(R)}', flush=True)
sys.exit(0 if all(R) else 1)
