import sys,types,re,unicodedata,time
m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode(); return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
m.slugify=slugify; sys.modules['slugify']=m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for i in range(120):
    subs=api.competition_submissions('rsna-knee-abnormality-detection')
    s=[x for x in subs if getattr(x,'ref',None)==55972482]
    if s:
        st=str(getattr(s[0],'status',None)); ps=getattr(s[0],'public_score',None)
        if 'COMPLETE' in st or 'ERROR' in st:
            print(f'DONE status={st} public_score={ps}',flush=True); break
        print(f'poll{i} {st} {ps}',flush=True)
    time.sleep(60)
