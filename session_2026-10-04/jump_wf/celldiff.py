import json,glob,difflib,sys,hashlib
def cells(d):
    f=glob.glob(d+'/*.ipynb')[0]
    nb=json.load(open(f,encoding='utf-8'))
    return [(c['cell_type'],''.join(c['source'])) for c in nb['cells']]
par=cells('../../k2/build/kernel/rsna-sub-k2-fleet')
ph={hashlib.md5(s.encode()).hexdigest():i for i,(t,s) in enumerate(par)}
pcode=[s for t,s in par]
for d in sys.argv[1:]:
    cs=cells(d)
    ident=0; rows=[]
    for i,(t,s) in enumerate(cs):
        h=hashlib.md5(s.encode()).hexdigest()
        if h in ph: ident+=1; continue
        # best match ratio vs parent cells
        best=max(((difflib.SequenceMatcher(None,s[:6000],p[:6000],autojunk=False).quick_ratio(),j) for j,p in enumerate(pcode)),default=(0,-1))
        first=[l for l in s.splitlines() if l.strip()][:1]
        rows.append((i,t,len(s),round(best[0],3),best[1],first[0][:90] if first else ''))
    print(f'== {d} cells {len(cs)} identical-to-parent {ident}')
    for r in rows: print('   ',r)
