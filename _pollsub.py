import sys, types, re, unicodedata
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode()
    return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
ref=int(sys.argv[1])
for s in api.competition_submissions("rsna-knee-abnormality-detection")[:15]:
    if getattr(s,'ref',None)==ref:
        st=str(getattr(s,'status',None)); ps=getattr(s,'public_score',None)
        print(f"{ref} {st} public={ps}")
        break
else:
    print(f"{ref} NOTFOUND")
