#!/usr/bin/env python
# build_r6eoa.py (2026-10-20, Claude, push lens; LOCAL BUILD ONLY: nothing pushed, committed or submitted).
# Candidate pick B' = R6 57248084 (session_2026-10-15/r6entry/kernel_v224r6flips6fj) + the EXACT A20 E-OA block that is in the scored
# team best eoa20 57413368 (session_2026-10-19/a20/build_a20.py BLOCK, verbatim, same anchor in cell 18, S6F path).
# R6 and c4 cell 18 are identical (MEASURED ex_r6eoa_gate.log), so the patched cell 18 must equal eoa20's patched cell 18 byte for byte.
# Usage: python build_r6eoa.py 0.2   (0 builds the no-op control that must reproduce R6's output byte for byte)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', '..')
SRC = os.path.join(R, 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj')
a = float(sys.argv[1]); tag = 'v224r6flips6feoa%02dj' % round(a * 100); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 3]
BLOCK = [   # verbatim copy of session_2026-10-19/a20/build_a20.py BLOCK (the code inside scored eoa20 57413368 / eoa10 57413380)
    '        A_OA = %r   # [A20 E-OA] OA decorrelation on the final output (OAI expert check G1 SUPPORT, SHIP_A .2); 0 = no-op' % a,
    "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
    '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
    '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}',
    '            for c in _OA:',
    '                _o = [k for k in _OA if k != c]',
    '                out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)',
    "            rec['a_oa'] = A_OA",
    "            print('EOA_APPLIED a_oa %g on %s' % (A_OA, _OA), flush=True)",
]
lines[i + 2:i + 2] = BLOCK
nb['cells'][cell]['source'] = '\n'.join(lines)
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [k for k, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
la, lb = src(A_['cells'][cell]).splitlines(), src(B_['cells'][cell]).splitlines()
assert diff == [cell] and len(A_['cells']) == len(B_['cells']) and lb[:i + 2] == la[:i + 2] and lb[i + 2 + len(BLOCK):] == la[i + 2:]
# cross-check vs the scored A20 kernel at the same a: every cell except 14 identical, and cell 14 differs exactly as R6 vs c4 does
ref = {0.2: 'kernel_v224r6c4eoa20j', 0.1: 'kernel_v224r6c4eoa10j', 0.0: 'kernel_v224r6c4eoa00j'}.get(a)
xref = None
if ref:
    E = json.load(open(os.path.join(R, 'session_2026-10-19', 'a20', ref, 'rsna_sub_%s.ipynb' % ref[7:]), encoding='utf-8'))
    C4 = json.load(open(os.path.join(R, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j', 'rsna_sub_v224r6a25c4j.ipynb'), encoding='utf-8'))
    d_e = [k for k, (x, y) in enumerate(zip(E['cells'], B_['cells'])) if src(x) != src(y)]
    d_c = [k for k, (x, y) in enumerate(zip(C4['cells'], A_['cells'])) if src(x) != src(y)]
    xref = dict(ref=ref, cells_differing_vs_eoa=d_e, cells_differing_c4_vs_R6=d_c, same_pattern=d_e == d_c,
                cell18_equal_to_eoa=src(E['cells'][18]) == src(B_['cells'][18]))
    print('XREF', xref)
json.dump(dict(probe='r6eoa', a_oa=a, tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, inserted_lines=len(BLOCK), xref=xref,
               note='LOCAL BUILD ONLY; push / submit needs Eesh go + PREREG amendment'),
          open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT r6eoa', tag, sha(newb)[:16], '| base R6', sha(raw)[:16], '| +%d lines in cell 18 (S6F path)' % len(BLOCK))
