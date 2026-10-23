import json, sys, re
nb = json.load(open(sys.argv[1], encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        print(f'--- [md {i}] ' + src[:1500])
    else:
        lines = src.splitlines()
        print(f'--- [code {i}] {len(lines)} lines')
        # print lines of interest
        for j, l in enumerate(lines):
            if re.search(r'(weight|W_|w_|blend|rank|LB|0\.9[0-9]|alpha|ROUTE|route|label_w|per.label|mix|FINAL|submission)', l, re.I) and len(l) < 220:
                print(f'   {j:4d}: {l}')
