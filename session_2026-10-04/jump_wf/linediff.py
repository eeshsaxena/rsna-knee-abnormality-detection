import json,glob,difflib,sys
def lines(d):
    f=glob.glob(d+'/*.ipynb')[0]
    nb=json.load(open(f,encoding='utf-8'))
    L=[]
    for c in nb['cells']:
        if c['cell_type']!='code': continue
        for l in ''.join(c['source']).splitlines():
            l=l.rstrip()
            if l.strip(): L.append(l)
    return L
base=sys.argv[1]; B=lines(base)
for d in sys.argv[2:]:
    A=lines(d)
    sm=difflib.SequenceMatcher(None,B,A,autojunk=False)
    add=[];rem=[]
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op in('replace','insert'): add+=A[j1:j2]
        if op in('replace','delete'): rem+=B[i1:i2]
    print(f'== {d} vs {base}: lines {len(A)} vs {len(B)}; added {len(add)} removed {len(rem)}')
    out=open(d+'/diff_vs_'+base.strip('./').replace('/','_')[-20:]+'.txt','w',encoding='utf-8')
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        out.write(f'@@ {op} base[{i1}:{i2}] new[{j1}:{j2}]\n')
        for l in B[i1:i2]: out.write('- '+l+'\n')
        for l in A[j1:j2]: out.write('+ '+l+'\n')
    out.close()
