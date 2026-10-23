import json,sys,glob,os
for d in sys.argv[1:]:
    f=[x for x in glob.glob(d+'/*.ipynb')][0]
    nb=json.load(open(f,encoding='utf-8'))
    out=[]
    for i,c in enumerate(nb['cells']):
        s=''.join(c['source'])
        out.append(f'# %%%% CELL {i} [{c["cell_type"]}] len={len(s)}\n'+s+'\n')
    open(d+'/code.py','w',encoding='utf-8').write('\n'.join(out))
    print(d,len(nb['cells']))
