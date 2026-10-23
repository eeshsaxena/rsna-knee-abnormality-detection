import ast,sys,hashlib,json,os
src=open(sys.argv[1],encoding='utf-8').read()
out=sys.argv[2]; os.makedirs(out,exist_ok=True)
tree=ast.parse(src)
vals={}
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        n=node.targets[0].id
        if n in ('SOURCES','SOURCE_SHA','CODE_SHA'):
            vals[n]=ast.literal_eval(node.value)
for name,text in vals['SOURCES'].items():
    b=text.encode('utf-8')
    open(os.path.join(out,name),'wb').write(b)
    sha=hashlib.sha256(b).hexdigest(); md5=hashlib.md5(b).hexdigest()
    print(f"{name:28s} {len(b):8d} sha_ok={sha==vals['SOURCE_SHA'][name]} md5={md5}")
json.dump(vals['CODE_SHA'],open(os.path.join(out,'_CODE_SHA.json'),'w'),indent=1)
print('CODE_SHA modules:',list(vals['CODE_SHA']))
