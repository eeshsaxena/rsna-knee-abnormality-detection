# cell_diff.py (10-21 skeptic, canary track). READ-ONLY: per-cell sha of the A22 kernels + the 10-20 pw20 + R6 (10-15) + the
# guard variant, and a unified diff of cell 18 (spl / salt vs swide). Writes only to stdout. Run: python -I cell_diff.py
import json, hashlib, glob, difflib
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
K = {
 'R6_1015': 'session_2026-10-15/r6entry/kernel_v224r6flips6fj',
 'pw20_1020': 'session_2026-10-20/a21/kernel_v224r6c4pw20j',
 'r6pw20': 'session_2026-10-20/val/kernels/kernel_v224r6flips6fpw20j',
 'pw25': 'session_2026-10-20/val/kernels/kernel_v224r6c4pw25j',
 'spl': 'session_2026-10-20/stack/skeptic/kernels/kernel_v224r6c4pw25lomm05mcl00j',
 'swide': 'session_2026-10-20/stack/kernels/kernel_v224r6c4pw25lomm05mcl10j',
 'salt': 'session_2026-10-20/stack/kernels/kernel_v224r6c4pw25mcl10j',
 'variant': 'session_2026-10-21/canary/kernels/kernel_v224r6flips6fpw20g85j',
}
C = {}
for k, d in K.items():
    f = glob.glob(R + d + '/*.ipynb')
    assert len(f) == 1, (k, f)
    nb = json.load(open(f[0], encoding='utf-8'))
    C[k] = [''.join(c['source']) for c in nb['cells']]
n = max(len(v) for v in C.values())
print('cell | ' + ' | '.join('%-8s' % k[:8] for k in K))
for i in range(n):
    print('%4d | ' % i + ' | '.join('%-8s' % (hashlib.sha256(C[k][i].encode()).hexdigest()[:8] if i < len(C[k]) else '-') for k in K))
for a in ('spl', 'salt', 'pw25', 'r6pw20'):
    print('\n==== cell 18 diff swide -> %s ====' % a)
    for l in difflib.unified_diff(C['swide'][18].splitlines(), C[a][18].splitlines(), lineterm='', n=1):
        print(l)
print('\n==== cell 18 diff pw20_1020 -> r6pw20 ====')
for l in difflib.unified_diff(C['pw20_1020'][18].splitlines(), C['r6pw20'][18].splitlines(), lineterm='', n=1):
    print(l)
print('\n==== cell 14 diff R6_1015 -> r6pw20 ====')
for l in difflib.unified_diff(C['R6_1015'][14].splitlines(), C['r6pw20'][14].splitlines(), lineterm='', n=1):
    print(l)
