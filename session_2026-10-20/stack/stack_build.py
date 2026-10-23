#!/usr/bin/env python
# stack_build.py (2026-10-21 ~00:10 IST, Claude, STACK inventory). LOCAL BUILD ONLY, UNARMED: nothing here is pushed, committed, registered.
# Builds the stack candidates of stack/inventory.md on a SCORED / built E-OA notebook by inserting, in cell 18 (_s6f_blend, S6F path),
# right after the A21 E-OA block (same anchor as night/c12_build.py), in this order:
#   [LOAMM]  out['Lateral OA'] = pct(pct(LOA) - A_LOAMM pct(MM))      (the exact 6-line night/c12_build.py block, A_LOAMM .05)
#   [MCLMM]  out['MCL']        = pct(pct(MCL) - A_MCLMM pct(MM))      (new 6-line block, A_MCLMM .1)
# Every other cell and every other column is untouched. The A_OA dose comes from the base notebook (pw30 = scored 57425970, or the
# built pw25 = val/kernels/kernel_v224r6c4pw25j, sha 64ed7eba). Usage: python stack_build.py {pw30|pw25} A_LOAMM A_MCLMM
import json, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, '..')
BASES = {'pw30': (os.path.join(S, 'a21', 'kernel_v224r6c4pw30j'), '5a903187', 0.3),
         'pw25': (os.path.join(S, 'val', 'kernels', 'kernel_v224r6c4pw25j'), '64ed7eba', 0.25)}
base = sys.argv[1]; al = float(sys.argv[2]); am = float(sys.argv[3])
SRC, SHA8, AOA = BASES[base]
tag = 'v224r6c4%s%s%sj' % (base, ('lomm%02d' % round(al * 100)) if al else '', 'mcl%02d' % round(am * 100)); cell = 18
sha = lambda b: hashlib.sha256(b).hexdigest()
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
raw = open(os.path.join(SRC, meta['code_file']), 'rb').read(); assert sha(raw).startswith(SHA8), (base, sha(raw)[:8]); nb = json.loads(raw)
src = lambda c: ''.join(c['source']) if isinstance(c['source'], list) else c['source']
lines = src(nb['cells'][cell]).split('\n')
i = [k for k, l in enumerate(lines) if l.startswith("            print('EOA_APPLIED a_oa")]; assert len(i) == 1, i; i = i[0]
assert lines[i + 1].startswith('        v = out[labels].values.astype(np.float64)'), lines[i:i + 2]
assert sum(l.startswith('        A_OA = %r   # [A21 E-OA pw]' % AOA) for l in lines) == 1, 'base A_OA line'
LOAMM = [
    "        A_LOAMM = %r   # [night C12'] Lateral OA decorrelated from Medial Meniscus on the final output (NIGHT_1020.md); 0 = no-op" % al,
    "        if A_LOAMM and {'Lateral OA', 'Medial Meniscus'} <= set(labels) and not {'Lateral OA', 'Medial Meniscus'} & set(blank):",
    "            out['Lateral OA'] = pct(pct(out['Lateral OA'].values.astype(np.float64))",
    "                                    - A_LOAMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_loamm'] = A_LOAMM",
    "            print('LOAMM_APPLIED a_loamm %g' % A_LOAMM, flush=True)",
]
MCLMM = [
    "        A_MCLMM = %r   # [stack MCLmm] MCL decorrelated from Medial Meniscus on the final output (stack/inventory.md); 0 = no-op" % am,
    "        if A_MCLMM and {'MCL', 'Medial Meniscus'} <= set(labels) and not {'MCL', 'Medial Meniscus'} & set(blank):",
    "            out['MCL'] = pct(pct(out['MCL'].values.astype(np.float64))",
    "                             - A_MCLMM * pct(out['Medial Meniscus'].values.astype(np.float64))).astype(np.float32)",
    "            rec['a_mclmm'] = A_MCLMM",
    "            print('MCLMM_APPLIED a_mclmm %g' % A_MCLMM, flush=True)",
]
BLOCK = (LOAMM if al else []) + MCLMM
lines[i + 1:i + 1] = BLOCK
nb['cells'][cell]['source'] = '\n'.join(lines)
out = os.path.join(HERE, 'kernel_' + tag); os.makedirs(out, exist_ok=True); cf = 'rsna_sub_%s.ipynb' % tag
json.dump(nb, open(os.path.join(out, cf), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
meta.update(id='eeshsaxena/rsna-sub-' + tag, title='rsna sub ' + tag, code_file=cf)
json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=2)
newb = open(os.path.join(out, cf), 'rb').read(); A_, B_ = json.loads(raw), json.loads(newb)
diff = [k for k, (x, y) in enumerate(zip(A_['cells'], B_['cells'])) if src(x) != src(y)]
la, lb = src(A_['cells'][cell]).splitlines(), src(B_['cells'][cell]).splitlines()
assert diff == [cell] and len(A_['cells']) == len(B_['cells']) and lb[:i + 1] == la[:i + 1] and lb[i + 1 + len(BLOCK):] == la[i + 1:]
m0 = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8')); m1 = json.load(open(os.path.join(out, 'kernel-metadata.json'), encoding='utf-8'))
assert {k for k in set(m0) | set(m1) if m0.get(k) != m1.get(k)} <= {'id', 'title', 'code_file'}
if al:   # the LOAMM lines must be byte-identical to the night lomm05 build's
    nl = src(json.load(open(os.path.join(S, 'night', 'c12_kernel_v224r6c4pw30lomm05j', 'rsna_sub_v224r6c4pw30lomm05j.ipynb'), encoding='utf-8'))['cells'][cell]).splitlines()
    assert all(x in nl for x in LOAMM), 'LOAMM block differs from night/c12 build'
rc = dict(probe='stack', base=base, base_sha256=sha(raw), a_oa=AOA, a_loamm=al, a_mclmm=am, tag=tag, out_sha256=sha(newb),
          meta_sha256=sha(open(os.path.join(out, 'kernel-metadata.json'), 'rb').read()), cell=cell, inserted_lines=len(BLOCK), armed=False,
          note='local build only; needs Eesh go + PREREG amendment + commit gate (stack/stack_probe_check.py) before any push')
json.dump(rc, open(os.path.join(HERE, 'build_receipt_%s.json' % tag), 'w'), indent=1)
print('STACK_BUILT', tag, sha(newb)[:16], '| meta', rc['meta_sha256'][:16], '| base', base, sha(raw)[:16], '| +%d lines in cell 18 after E-OA' % len(BLOCK))
