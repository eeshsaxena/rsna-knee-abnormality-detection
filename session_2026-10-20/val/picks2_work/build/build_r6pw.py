#!/usr/bin/env python
# build_r6pw.py (2026-10-20 val Track D, Claude; LOCAL BUILD ONLY: nothing pushed, committed or submitted).
# Candidate B = R6 57248084 (session_2026-10-15/r6entry/kernel_v224r6flips6fj) + the EXACT A21 pw E-OA block of the scored pw30 57425970 /
# pw20 57425269 (session_2026-10-20/a21/build_a21.py BLOCK for mode 'pw', verbatim, same cell-18 anchor, S6F path).
# R6 and c4 cell 18 are identical (MEASURED ex_r6eoa_gate.log), so the patched cell 18 must equal the scored pw entry's cell 18 byte for byte.
# Usage: python build_r6pw.py 0.3   (0 builds the no-op control that must reproduce R6's output byte for byte)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = 'C:/Users/eeshs/Downloads/rsna-knee'
SRC = os.path.join(R, 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj')
a = float(sys.argv[1]); mode = 'pw'; tag = 'v224r6flips6fpw%02dj' % round(a * 100); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 3]
BLOCK = [   # verbatim copy of session_2026-10-20/a21/build_a21.py BLOCK (mode 'pw'), the code inside scored pw20 57425269 / pw30 57425970
    '        A_OA = %r   # [A21 E-OA %s] OA decorrelation on the final output (A20 eoa20 = team best .959); 0 = no-op' % (a, mode),
    "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
    '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
    '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}',
    '            _W = %s' % ("{('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}" if mode == 'pw' else "None"),
    '            for c in _OA:',
    '                _o = [k for k in _OA if k != c]',
    '                if _W is None:',
    '                    out[c] = pct(_U[c] - A_OA * (_U[_o[0]] + _U[_o[1]]) / 2.0).astype(np.float32)',
    '                else:',
    '                    _w = [_W[tuple(sorted((c, k), key=_OA.index))] for k in _o]',
    '                    out[c] = pct(_U[c] - A_OA / 2.0 * (_w[0] * _U[_o[0]] + _w[1] * _U[_o[1]])).astype(np.float32)',
    "            rec['a_oa'] = A_OA; rec['oa_mode'] = %r" % mode,
    "            print('EOA_APPLIED a_oa %g mode %s on %s' % (A_OA, rec['oa_mode'], _OA), flush=True)",
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
# cross-check vs the scored / built A21 pw kernel at the same a: every cell except 14 identical; cell 14 differs exactly as R6 vs c4 does
ref = {0.3: 'kernel_v224r6c4pw30j', 0.2: 'kernel_v224r6c4pw20j', 0.0: 'kernel_v224r6c4pw00j'}.get(a)
xref = None
if ref:
    E = json.load(open(os.path.join(R, 'session_2026-10-20', 'a21', ref, 'rsna_sub_%s.ipynb' % ref[7:]), encoding='utf-8'))
    C4 = json.load(open(os.path.join(R, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j', 'rsna_sub_v224r6a25c4j.ipynb'), encoding='utf-8'))
    d_e = [k for k, (x, y) in enumerate(zip(E['cells'], B_['cells'])) if src(x) != src(y)]
    d_c = [k for k, (x, y) in enumerate(zip(C4['cells'], A_['cells'])) if src(x) != src(y)]
    xref = dict(ref=ref, cells_differing_vs_pw=d_e, cells_differing_c4_vs_R6=d_c, same_pattern=d_e == d_c,
                cell18_equal_to_pw=src(E['cells'][18]) == src(B_['cells'][18]))
    print('XREF', xref)
json.dump(dict(probe='r6pw', mode=mode, a_oa=a, tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, inserted_lines=len(BLOCK), xref=xref,
               note='LOCAL BUILD ONLY (val Track D); push / submit needs Eesh go + PREREG amendment'),
          open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT r6pw', tag, sha(newb)[:16], '| base R6', sha(raw)[:16], '| +%d lines in cell 18 (S6F path)' % len(BLOCK))
