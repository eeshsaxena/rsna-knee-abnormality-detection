import os,shutil,sys,glob
dst='out_v224flips6fs20_v1m'
srcs=sorted(glob.glob('out_v224flips6fs20_v1*'),key=os.path.getmtime,reverse=True)
srcs=[s for s in srcs if s!=dst and not s.endswith('bad502')]
def isbad(p):
    try: b=open(p,'rb').read(600)
    except: return True
    return b'Server Error' in b and b'<html' in b.lower()
names=set()
for s in srcs:
    for r,_,fs in os.walk(s):
        for f in fs: names.add(os.path.relpath(os.path.join(r,f),s))
bad=[]
for n in sorted(names):
    for s in srcs:
        p=os.path.join(s,n)
        if os.path.isfile(p) and not isbad(p):
            os.makedirs(os.path.dirname(os.path.join(dst,n)) or dst,exist_ok=True); shutil.copy2(p,os.path.join(dst,n)); break
    else: bad.append(n)
print('MERGE files',len(names),'bad',bad); sys.exit(1 if bad else 0)
