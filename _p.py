import sys,types,re,unicodedata,time,warnings
warnings.filterwarnings('ignore')
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode(); return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
ref=int(sys.argv[1])
for i in range(160):
    try:
        s=next((x for x in api.competition_submissions('rsna-knee-abnormality-detection') if getattr(x,'ref',None)==ref),None)
        st=str(getattr(s,'status',None))
        if 'COMPLETE' in st or 'ERROR' in st:
            print(f'DONE {ref} {st.split(".")[-1]} public_score={getattr(s,"public_score","")!r} err={getattr(s,"error_description","")!r}',flush=True); break
    except Exception: pass
    time.sleep(300)
