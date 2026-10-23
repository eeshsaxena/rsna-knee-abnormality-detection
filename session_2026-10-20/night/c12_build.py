#!/usr/bin/env python
# c12_build.py (2026-10-20 night, Claude, NIGHT synthesis). LOCAL BUILD ONLY, UNARMED: nothing here is pushed, committed or registered.
# Candidate C12' = the scored pw30 kernel (session_2026-10-20/a21/kernel_v224r6c4pw30j, 57425970) + ONE change on the FINAL output
# (cell 18 _s6f_blend, S6F path only, right after the A21 E-OA block, same insertion point as night/syn_build.py):
#     out['Lateral OA'] = pct(pct(LOA_final) - A_LOAMM * pct(MM_final))
# Every other column and every other cell is untouched. Evidence and odds: session_2026-10-20/night/NIGHT_1020.md (recal.md C12').
# Usage: python c12_build.py A_LOAMM   (A_LOAMM 0 builds a no-op control whose output must be byte-identical to pw30's)
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'a21', 'kernel_v224r6c4pw30j')
a = float(sys.argv[1]); tag = 'v224r6c4pw30lomm%02dj' % round(a * 100); cell = 18
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
    "        A_LOAMM = %r   # [night C12'] Lateral OA decorrelated from Medial Meniscus on the final output (NIGHT_1020.md); 0 = no-op" % a,
    "        if A_LOAMM and {'Lateral OA', 'Medial Meniscus'} <= set(labels) and not {'Lateral OA', 'Medial Meniscus'} & set(blank):",
    "            out['Lateral OA'] = pct(pct(out['Lateral OA'].values.astype(np.float64))",
    "                                    - A_LOAMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_loamm'] = A_LOAMM",
    "            print('LOAMM_APPLIED a_loamm %g' % A_LOAMM, flush=True)",
]
lines[i + 1:i + 1] = BLOCK
nb['cells'][cell]['source'] = '\n'.join(lines)
out = os.path.join(HERE, 'c12_kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [k for k, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
la, lb = src(A_['cells'][cell]).splitlines(), src(B_['cells'][cell]).splitlines()
assert diff == [cell] and len(A_['cells']) == len(B_['cells']) and lb[:i + 1] == la[:i + 1] and lb[i + 1 + len(BLOCK):] == la[i + 1:]
m0 = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8')); m1 = json.load(open(os.path.join(out, 'kernel-metadata.json'), encoding='utf-8'))
assert {k for k in set(m0) | set(m1) if m0.get(k) != m1.get(k)} <= {'id', 'title', 'code_file'}
json.dump(dict(probe="C12' LOA <- MM", a_loamm=a, tag=tag, base='v224r6c4pw30j (57425970)', src_sha256=sha(raw), out_sha256=sha(newb), cell=cell,
               inserted_lines=len(BLOCK), armed=False, note='local build only; needs Eesh go + PREREG amendment + commit gate before any push'),
          open(os.path.join(HERE, 'c12_build_receipt_%s.json' % tag), 'w'), indent=1)
print('C12_BUILT', tag, sha(newb)[:16], '| base pw30', sha(raw)[:16], '| +%d lines in cell 18 (S6F path, after E-OA) | meta differs only in id/title/code_file' % len(BLOCK))
