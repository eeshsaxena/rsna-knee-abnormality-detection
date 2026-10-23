#!/usr/bin/env python
# celldiff.py (2026-10-14, Claude): cell-by-cell diff of a variant kernel vs a reference kernel (notebook + metadata).
#   usage: python celldiff.py REF_DIR VARIANT_DIR
# Prints: cell count / types / notebook-level metadata equality, then for every differing cell the unified diff of its
# source, then the kernel-metadata.json keys that differ. Ends with CELLDIFF <n differing cells> <list of indices>.
import sys, os, json, glob, difflib

ref_dir, var_dir = sys.argv[1], sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
nb = lambda d: json.load(open(glob.glob(os.path.join(d, '*.ipynb'))[0], encoding='utf-8'))
R, V = nb(ref_dir), nb(var_dir)
src = lambda c: ''.join(c['source'])
print('ref', glob.glob(os.path.join(ref_dir, '*.ipynb'))[0])
print('var', glob.glob(os.path.join(var_dir, '*.ipynb'))[0])
print('cells', len(R['cells']), len(V['cells']), '| notebook metadata equal:', R['metadata'] == V['metadata'],
      '| nbformat equal:', (R.get('nbformat'), R.get('nbformat_minor')) == (V.get('nbformat'), V.get('nbformat_minor')))
assert len(R['cells']) == len(V['cells']), 'cell count differs'
diff_idx = []
for i, (a, b) in enumerate(zip(R['cells'], V['cells'])):
    same_nonsrc = {k: v for k, v in a.items() if k != 'source'} == {k: v for k, v in b.items() if k != 'source'}
    if src(a) == src(b) and same_nonsrc:
        continue
    diff_idx.append(i)
    first = src(a).splitlines()[0][:100] if src(a) else ''
    print(f'\n--- cell {i} ({a["cell_type"]}; non-source fields equal: {same_nonsrc}) first line: {first}')
    for l in difflib.unified_diff(src(a).splitlines(), src(b).splitlines(), 'ref', 'var', n=0, lineterm=''):
        print('   ', l[:400])
mr = json.load(open(os.path.join(ref_dir, 'kernel-metadata.json'), encoding='utf-8'))
mv = json.load(open(os.path.join(var_dir, 'kernel-metadata.json'), encoding='utf-8'))
print('\nkernel-metadata.json differing keys:')
for k in sorted(set(mr) | set(mv)):
    if mr.get(k) != mv.get(k):
        print(f'   {k}: {mr.get(k)!r} -> {mv.get(k)!r}')
print(f'CELLDIFF {len(diff_idx)} cells differ: {diff_idx}')
