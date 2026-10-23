import json,sys
nb=json.load(open(sys.argv[1],encoding='utf-8'))
i=int(sys.argv[2])
c=nb['cells'][i]
src=''.join(c['source']) if isinstance(c['source'],list) else c['source']
open(sys.argv[3],'w',encoding='utf-8').write(src)
print(len(src))
