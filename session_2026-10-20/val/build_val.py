#!/usr/bin/env python
# build_val.py (10-20 ~19:15 IST, Claude, Track C; LOCAL BUILD ONLY: nothing pushed, committed or submitted).
# One builder for both bases, the A21 pair-weighted E-OA block VERBATIM from session_2026-10-20/a21/build_a21.py (same anchor in
# cell 18 _s6f_blend, S6F path, after the blanked-column fill):
#   base c4 : session_2026-10-18/a18/kernel_v224r6a25c4j (57343435, sha b47c29c3)      -> tag v224r6c4pw%02dj   (== a21 / push/shape naming)
#   base r6 : session_2026-10-15/r6entry/kernel_v224r6flips6fj (57248084, sha dfee2ba3) -> tag v224r6flips6fpw%02dj
# For r6 this is push/build_r6eoa.py with the A21 'pw' block in place of the A20 'sym' block (R6 and c4 cell 18 are identical,
# MEASURED push/build_receipt_v224r6flips6feoa20j.json), so R6 + pw<a> cell 18 must equal a21 c4 pw<a> cell 18 byte for byte.
# Usage: python build_val.py {c4|r6} A_OA   (A_OA 0 = no-op control). Writes ONLY under session_2026-10-20/val/kernels.
# Parity (asserted when a reference build exists): c4 notebook + metadata bytes == a21 (pw00 / pw20 / pw30) or push/shape (pw25 / pw35);
# r6: cells differing vs the a21 c4 build at the same a == cells differing c4 vs R6 == [14], and cell 18 equal.
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.join(HERE, 'kernels')
R = os.path.abspath(os.path.join(HERE, '..', '..'))
base = sys.argv[1]; assert base in ('c4', 'r6'); a = float(sys.argv[2]); mode = 'pw'
if base == 'c4':
    SRC = os.path.join(R, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j'); SRC_SHA = 'b47c29c3d72b9fab09393535a369440e5c77d41f2f019d091b65fd7d2306c563'
    tag = 'v224r6c4pw%02dj' % round(a * 100)
else:
    SRC = os.path.join(R, 'session_2026-10-15', 'r6entry', 'kernel_v224r6flips6fj'); SRC_SHA = 'dfee2ba34f4ca107498aa76456a6a9199d27eaea3f08da34918471f1b67d643e'
    tag = 'v224r6flips6fpw%02dj' % round(a * 100)
cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
assert sha(raw) == SRC_SHA, ('base sha changed', sha(raw))
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
# anchor: the blanked-column fill, immediately followed by the validity check
i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 3]
BLOCK = [   # verbatim session_2026-10-20/a21/build_a21.py BLOCK (mode 'pw'), i.e. the code inside scored pw30 57425970 / pw20 57425269
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
out = os.path.join(KD, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [k for k, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
la, lb = src(A_['cells'][cell]).splitlines(), src(B_['cells'][cell]).splitlines()
assert diff == [cell] and len(A_['cells']) == len(B_['cells']) and lb[:i + 2] == la[:i + 2] and lb[i + 2 + len(BLOCK):] == la[i + 2:]
metab = open(os.path.join(out, 'kernel-metadata.json'), 'rb').read()
# parity / cross-reference against the existing builds
REFS = {'v224r6c4pw00j': 'session_2026-10-20/a21', 'v224r6c4pw20j': 'session_2026-10-20/a21', 'v224r6c4pw30j': 'session_2026-10-20/a21',
        'v224r6c4pw25j': 'session_2026-10-20/push/shape', 'v224r6c4pw35j': 'session_2026-10-20/push/shape'}
parity = xref = None
if base == 'c4' and tag in REFS:
    rd_ = os.path.join(R, REFS[tag], 'kernel_' + tag)
    rnb, rmeta = open(os.path.join(rd_, cf), 'rb').read(), open(os.path.join(rd_, 'kernel-metadata.json'), 'rb').read()
    parity = dict(ref=REFS[tag] + '/kernel_' + tag, notebook_identical=rnb == newb, metadata_identical=rmeta == metab, ref_sha256=sha(rnb))
    assert parity['notebook_identical'] and parity['metadata_identical'], parity
    print('PARITY', parity)
if base == 'r6':
    ctag = 'v224r6c4pw%02dj' % round(a * 100); cand = [os.path.join(R, d, 'kernel_' + ctag) for d in ('session_2026-10-20/a21', 'session_2026-10-20/push/shape')]
    cand = [c for c in cand if os.path.isdir(c)]
    if cand:
        E = json.load(open(os.path.join(cand[0], 'rsna_sub_%s.ipynb' % ctag), encoding='utf-8'))
        C4 = json.load(open(os.path.join(R, 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j', 'rsna_sub_v224r6a25c4j.ipynb'), encoding='utf-8'))
        d_e = [k for k, (x, y) in enumerate(zip(E['cells'], B_['cells'])) if src(x) != src(y)]
        d_c = [k for k, (x, y) in enumerate(zip(C4['cells'], A_['cells'])) if src(x) != src(y)]
        xref = dict(ref=os.path.relpath(cand[0], R).replace(chr(92), '/'), cells_differing_vs_c4pw=d_e, cells_differing_c4_vs_R6=d_c,
                    same_pattern=d_e == d_c, cell18_equal_to_c4pw=src(E['cells'][18]) == src(B_['cells'][18]))
        assert xref['same_pattern'] and xref['cell18_equal_to_c4pw'] and d_e == [14], xref
        print('XREF', xref)
json.dump(dict(probe='val_pw', base=base, mode=mode, a_oa=a, tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), meta_sha256=sha(metab), cell=cell,
               inserted_lines=len(BLOCK), parity=parity, xref=xref, note='LOCAL BUILD ONLY; push / submit needs Eesh go + PREREG amendment'),
          open(os.path.join(KD, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', base, mode, tag, sha(newb)[:16], '| base', sha(raw)[:16], '| +%d lines in cell 18 (S6F path)' % len(BLOCK))
