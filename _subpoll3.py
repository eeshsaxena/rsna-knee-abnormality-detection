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
print(f'watching {refs} (quiet: only prints on completion)',flush=True)
for i in range(150):  # ~150*5min = 12.5h coverage
    try: subs={getattr(s,'ref',None):s for s in api.competition_submissions(comp)}
    except Exception as e: time.sleep(300); continue
    for ref in refs:
        if ref in done: continue
        s=subs.get(ref)
        if s is None: continue
        st=str(getattr(s,'status',None))
        if 'COMPLETE' in st or 'ERROR' in st:
            print(f'DONE {ref} {st.split(".")[-1]} public_score={getattr(s,"public_score","")!r} err={getattr(s,"error_description","")!r}',flush=True); done.add(ref)
    if done>=refs: print('ALL DONE',flush=True); break
    time.sleep(300)
