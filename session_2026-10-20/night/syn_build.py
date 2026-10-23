#!/usr/bin/env python
# syn_build.py (2026-10-20 night, Claude). LOCAL BUILD ONLY, UNARMED: nothing here is pushed, committed or registered.
# Candidate "E2s" = the scored pw30 kernel (session_2026-10-20/a21/kernel_v224r6c4pw30j, 57425970) + ONE change on the FINAL output
# (cell 18 _s6f_blend, S6F path only, right after the A21 E-OA block):
#     out['Synovitis'] = pct(pct(Syn_final) + A_SYN * pct(Eff_final))
# Every other column and every other cell is untouched. Evidence and odds: session_2026-10-20/night/synovitis.md.
# Usage: python syn_build.py A_SYN   (A_SYN 0 builds a no-op control whose output must be byte-identical to pw30's)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'a21', 'kernel_v224r6c4pw30j')
a = float(sys.argv[1]); tag = 'v224r6c4pw30syn%02dj' % round(a * 100); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
# anchor: the last line of the A21 E-OA block, immediately followed by the validity check
i = [k for k, l in enumerate(lines) if l.startswith("            print('EOA_APPLIED a_oa")]; assert len(i) == 1, i; i = i[0]
assert lines[i + 1].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 2]
assert any(l.startswith('        A_OA = 0.3   # [A21 E-OA pw]') for l in lines)
BLOCK = [
    '        A_SYN = %r   # [night E2s] Synovitis borrows Effusion on the final output (synovitis.md); 0 = no-op' % a,
    "        if A_SYN and {'Synovitis', 'Effusion'} <= set(labels) and not {'Synovitis', 'Effusion'} & set(blank):",
    "            out['Synovitis'] = pct(pct(out['Synovitis'].values.astype(np.float64))",
    "                                   + A_SYN * pct(out['Effusion'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_syn'] = A_SYN",
    "            print('SYN_E2_APPLIED a_syn %g' % A_SYN, flush=True)",
]
lines[i + 1:i + 1] = BLOCK
nb['cells'][cell]['source'] = '\n'.join(lines)
out = os.path.join(HERE, 'syn_kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [k for k, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
la, lb = src(A_['cells'][cell]).splitlines(), src(B_['cells'][cell]).splitlines()
assert diff == [cell] and len(A_['cells']) == len(B_['cells']) and lb[:i + 1] == la[:i + 1] and lb[i + 1 + len(BLOCK):] == la[i + 1:]
json.dump(dict(probe='E2s', a_syn=a, tag=tag, base='v224r6c4pw30j (57425970)', src_sha256=sha(raw), out_sha256=sha(newb), cell=cell,
               inserted_lines=len(BLOCK), armed=False, note='local build only; needs Eesh go + PREREG amendment + commit gate before any push'),
          open(os.path.join(HERE, 'syn_build_receipt_%s.json' % tag), 'w'), indent=1)
print('SYN_BUILT', tag, sha(newb)[:16], '| base pw30', sha(raw)[:16], '| +%d lines in cell 18 (S6F path, after E-OA)' % len(BLOCK))
