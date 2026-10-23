#!/usr/bin/env python
# build_shape.py (10-20 push, Claude): VERBATIM copy of session_2026-10-20/a21/build_a21.py with only HERE-relative SRC depth changed (push/shape is one level deeper). Writes only under push/shape. Parity: rebuilding pw20 / pw30 / eoa30 must reproduce the a21 receipts' out_sha256.
# build_a21.py (2026-10-20, Claude): PREREG_10-15 Amendment 21 = A20 E-OA builder + mode: 'sym' (A20 formula) or 'pw' (pair weights from the gold74
# excess OA coupling gap, EX_1019 F3: w(LOA,PF) 1.416, w(MOA,LOA) .876, w(MOA,PF) .708; out_t = pct(U_t - sum_o (A_OA / 2) w(t,o) U_o)).
# Origin: build_a20.py (2026-10-19): PREREG_10-15 Amendment 20 candidate = the team best c4 kernel (session_2026-10-18/a18/kernel_v224r6a25c4j,
# 57343435) + E-OA: OA decorrelation on the FINAL output (cell 18 _s6f_blend, S6F path only, after the blanked-column fill):
#   for t in (Medial OA, Lateral OA, PF OA): out_t = pct(pct(f_t) - A_OA * (pct(f_o1) + pct(f_o2)) / 2), f = the final c4 column.
# Evidence: OAI expert MOAKS reads, 2,399 knees, OAI-free report-trained model: OA-mean AUC +.0062 (a .1) / +.0115 (a .2), P>0 1.000,
# 3/3 proxies, 5/5 folds, a -.1 control negative (session_2026-10-19/trick/oai_check/g1_box_run.log, pre-registered rule -> SHIP_A .2);
# gold58 replay +.00142 / +.00239 macro (trick/regime_work/m14_gate.log). Every other column and every other cell is untouched.
# Usage: python build_a21.py {sym|pw} A_OA   (A_OA 0 builds a no-op control that must be byte-identical in output to c4)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', '..', 'session_2026-10-18', 'a18', 'kernel_v224r6a25c4j')
mode = sys.argv[1]; assert mode in ('sym', 'pw'); a = float(sys.argv[2])
tag = ('v224r6c4eoa%02dj' if mode == 'sym' else 'v224r6c4pw%02dj') % round(a * 100); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
# anchor: the blanked-column fill, immediately followed by the validity check
i = [k for k, l in enumerate(lines) if l == '        for c in blank:']; assert len(i) == 1, i; i = i[0]
assert lines[i + 1] == '            out[c] = np.float32(0.5)' and lines[i + 2].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 3]
BLOCK = [
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
json.dump(dict(probe='eoa', mode=mode, a_oa=a, tag=tag, src_sha256=sha(raw), out_sha256=sha(newb), cell=cell, inserted_lines=len(BLOCK)),
          open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('PROBE_BUILT', mode, tag, sha(newb)[:16], '| base c4', sha(raw)[:16], '| +%d lines in cell 18 (S6F path)' % len(BLOCK))
