"""Independent cell diff of a kernel notebook vs a reference notebook (reviewer tool, read-only).
Usage: python celldiff_rev.py REF.ipynb NEW.ipynb [--full]
Prints: cell alignment (equal / inserted / deleted / replaced), and for replaced cells a unified diff."""
import json, sys, difflib, hashlib

def load(p):
    nb = json.load(open(p, encoding='utf-8'))
    out = []
    for c in nb['cells']:
        s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        out.append((c['cell_type'], s))
    return nb, out

ref_p, new_p = sys.argv[1], sys.argv[2]
full = '--full' in sys.argv
rnb, r = load(ref_p)
nnb, n = load(new_p)
print('REF', ref_p, len(r), 'cells; NEW', new_p, len(n), 'cells')
print('ref metadata == new metadata:', rnb.get('metadata') == nnb.get('metadata'))
sm = difflib.SequenceMatcher(a=[x[1] for x in r], b=[x[1] for x in n], autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        print(f'EQUAL ref[{i1}:{i2}] == new[{j1}:{j2}]')
    elif tag == 'insert':
        for j in range(j1, j2):
            s = n[j][1]
            first = s.strip().splitlines()[0][:110] if s.strip() else '<empty>'
            print(f'INSERT new[{j}] ({n[j][0]}, {len(s)} chars, sha {hashlib.sha256(s.encode()).hexdigest()[:10]}): {first}')
            if full:
                print(s); print('----- end insert -----')
    elif tag == 'delete':
        for i in range(i1, i2):
            print(f'DELETE ref[{i}]: {r[i][1].strip().splitlines()[0][:110]}')
    else:
        print(f'REPLACE ref[{i1}:{i2}] -> new[{j1}:{j2}]')
        a = ''.join(r[i][1] + '\n' for i in range(i1, i2)).splitlines()
        b = ''.join(n[j][1] + '\n' for j in range(j1, j2)).splitlines()
        d = list(difflib.unified_diff(a, b, lineterm='', n=1))
        print('\n'.join(d[:400]))
