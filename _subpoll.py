import sys,types,re,unicodedata,time,warnings,os
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS']='ignore'
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode()
    return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
ref=int(sys.argv[1]); comp='rsna-knee-abnormality-detection'
for i in range(150):
    try:
        subs=api.competition_submissions(comp)
        s=next((x for x in subs if getattr(x,'ref',None)==ref), None)
    except Exception as e:
        print(f'[{i}] api err {str(e)[:80]}',flush=True); time.sleep(30); continue
    if s is None: print(f'[{i}] not found',flush=True); time.sleep(30); continue
    st=str(getattr(s,'status',None)); pub=getattr(s,'public_score',''); err=getattr(s,'error_description','')
    print(f'[{i}] {st} public_score={pub!r} err={err!r}',flush=True)
    if 'COMPLETE' in st or 'ERROR' in st:
        print(f'FINAL status={st} public_score={pub!r}',flush=True); break
    time.sleep(30)
