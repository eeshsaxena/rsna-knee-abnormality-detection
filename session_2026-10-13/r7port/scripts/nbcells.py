import json,sys
nb=json.load(open(sys.argv[1],encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"=== cell {i} {c['cell_type']} len={len(src)}")
    print(src[:600].replace('\n','\n   '))
