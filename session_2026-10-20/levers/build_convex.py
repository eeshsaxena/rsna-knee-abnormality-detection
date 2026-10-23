#!/usr/bin/env python
# build_convex.py (10-20 levers, Claude): build_a21.py + a CONVEX exponent P_OA on the SUBTRACTED other-compartment pct
# (out_t = pct(U_t - sum_o (A_OA / 2) w(t,o) U_o ** P_OA)); P_OA 1 must reproduce the A21 / A20 kernel OUTPUT byte for byte.
# Evidence: session_2026-10-20/levers/space.md (OAI expert ruler, box_F2 / box_F3). Writes only under session_2026-10-20/levers.
# Usage: python build_convex.py {sym|pw} A_OA P_OA
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j')
mode = sys.argv[1]; assert mode in ('sym', 'pw'); a = float(sys.argv[2]); p = float(sys.argv[3]); assert p >= 1
tag = ('v224r6c4eoa%02dp%gj' if mode == 'sym' else 'v224r6c4pw%02dp%gj') % (round(a * 100), p); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
# anchor: the blanked-column fill, immediately followed by the validity check
i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 3]
BLOCK = [
    '        A_OA = %r   # [LEVERS E-OA %s, convex p] OA decorrelation on the final output; 0 = no-op' % (a, mode),
    '        P_OA = %r   # exponent on the subtracted pct; 1 = the A21 / A20 linear E-OA' % p,
    "        _OA = ['Medial OA', 'Lateral OA', 'PF OA']",
    '        if A_OA and set(_OA) <= set(labels) and not set(_OA) & set(blank):',
    '            _U = {c: pct(out[c].values.astype(np.float64)) for c in _OA}',
    '            _S = _U if P_OA == 1 else {c: _U[c] ** P_OA for c in _OA}',
    '            _W = %s' % ("{('Medial OA', 'Lateral OA'): 0.876, ('Medial OA', 'PF OA'): 0.708, ('Lateral OA', 'PF OA'): 1.416}" if mode == 'pw' else "None"),
    '            for c in _OA:',
    '                _o = [k for k in _OA if k != c]',
    '                if _W is None:',
    '                    out[c] = pct(_U[c] - A_OA * (_S[_o[0]] + _S[_o[1]]) / 2.0).astype(np.float32)',
    '                else:',
    '                    _w = [_W[tuple(sorted((c, k), key=_OA.index))] for k in _o]',
    '                    out[c] = pct(_U[c] - A_OA / 2.0 * (_w[0] * _S[_o[0]] + _w[1] * _S[_o[1]])).astype(np.float32)',
    "            rec['a_oa'] = A_OA; rec['oa_mode'] = %r; rec['p_oa'] = P_OA" % mode,
    "            print('EOA_APPLIED a_oa %g mode %s p_oa %g on %s' % (A_OA, rec['oa_mode'], P_OA, _OA), flush=True)",
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
json.dump(dict(probe='eoa_convex', mode=mode, a_oa=a, p_oa=p, tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, inserted_lines=len(BLOCK)),
          open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', mode, tag, sha(newb)[:16], '| base c4', sha(raw)[:16], '| +%d lines in cell 18 (S6F path)' % len(BLOCK))
