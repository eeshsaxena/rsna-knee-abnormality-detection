import sys,types,re,unicodedata,time
m=types.ModuleType('slugify')
def sl(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode(); return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
m.slugify=sl; sys.modules['slugify']=m
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi(); a.authenticate()
for i in range(90):
    for s in a.competition_submissions('rsna-knee-abnormality-detection'):
        if getattr(s,'ref',None)==55979836:
            st=str(getattr(s,'status',None)); ps=getattr(s,'public_score',None)
            if 'COMPLETE' in st or 'ERROR' in st:
                print(f'SCORED status={st} public={ps}',flush=True); sys.exit(0)
    time.sleep(60)
print('timeout',flush=True)
