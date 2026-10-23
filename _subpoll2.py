import sys,types,re,unicodedata,time,warnings
warnings.filterwarnings('ignore')
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode()
    return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
refs={int(x) for x in sys.argv[1:]}; comp='rsna-knee-abnormality-detection'; done=set()
for i in range(200):
    try: subs={getattr(s,'ref',None):s for s in api.competition_submissions(comp)}
    except Exception as e: print(f'[{i}] err {str(e)[:60]}',flush=True); time.sleep(30); continue
    for ref in refs:
        if ref in done: continue
        s=subs.get(ref)
        if s is None: continue
        st=str(getattr(s,'status',None)); pub=getattr(s,'public_score',''); err=getattr(s,'error_description','')
        if 'COMPLETE' in st or 'ERROR' in st:
            print(f'[{i}] DONE {ref} {st} public_score={pub!r} err={err!r}',flush=True); done.add(ref)
        else:
            print(f'[{i}] {ref} {st}',flush=True)
    if done>=refs: print('ALL DONE',flush=True); break
    time.sleep(40)
