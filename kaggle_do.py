#!/usr/bin/env python3
"""Kaggle API wrapper that shims the broken slugify stub. Usage:
  python kaggle_do.py push <folder>
  python kaggle_do.py kstatus <owner/kernel>
  python kaggle_do.py koutput <owner/kernel> <dest>
  python kaggle_do.py submit <comp> <kernel> <version> <message>
"""
import sys, types, re, unicodedata
_m=types.ModuleType('slugify')
def slugify(s,*a,**k):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode()
    return re.sub(r'[^a-zA-Z0-9]+','-',s).strip('-').lower()
_m.slugify=slugify; sys.modules['slugify']=_m
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
cmd=sys.argv[1]
if cmd=='push':
    print(api.kernels_push(sys.argv[2]))
elif cmd=='kstatus':
    r=api.kernels_status(sys.argv[2]); print(getattr(r,'status',r))
elif cmd=='koutput':
    api.kernels_output(sys.argv[2], sys.argv[3]); print('downloaded to', sys.argv[3])
elif cmd=='submit':
    comp,kernel,ver,msg=sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
    print(api.competition_submit_code(file_name="submission.csv", message=msg,
                                      competition=comp, kernel=kernel, kernel_version=int(ver)))
elif cmd=='dscreate':
    print(api.dataset_create_new(sys.argv[2], dir_mode='zip', public=False))
elif cmd=='dsversion':
    print(api.dataset_create_version(sys.argv[2], sys.argv[3], dir_mode='zip'))
elif cmd=='dsstatus':
    print(api.dataset_status(sys.argv[2]))
else:
    print('unknown', cmd)
