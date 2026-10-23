import json, glob, os, sys
for f in glob.glob('*/*.ipynb'):
    nb = json.load(open(f, encoding='utf-8'))
    out = f[:-6] + '.txt'
    with open(out, 'w', encoding='utf-8') as o:
        for i, c in enumerate(nb['cells']):
            src = ''.join(c.get('source', []))
            o.write(f"\n##### CELL {i} [{c['cell_type']}] len={len(src)}\n")
            if len(src) > 30000 and 'base64' in src[:50000]:
                o.write(src[:3000] + '\n... [TRUNCATED long cell] ...\n' + src[-2000:])
            else:
                o.write(src)
            for op in c.get('outputs', []) or []:
                t = op.get('text') or (op.get('data', {}) or {}).get('text/plain')
                if t:
                    t = ''.join(t) if isinstance(t, list) else t
                    o.write('\n----- OUTPUT -----\n' + t[:6000])
    print(os.path.getsize(out), out)
