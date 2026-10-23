# subs_dump.py (10-21 canary track). READ-ONLY: dumps every field of our competition submissions to subs_all.json.
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
out = []
page = None
subs = []
try:
    subs = list(api.competition_submissions('rsna-knee-abnormality-detection', page_size=200))
except TypeError:
    subs = list(api.competition_submissions('rsna-knee-abnormality-detection'))
for s in subs:
    d = {}
    for k in dir(s):
        if k.startswith('_'):
            continue
        try:
            v = getattr(s, k)
        except Exception:
            continue
        if callable(v):
            continue
        d[k] = str(v)
    out.append(d)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('n', len(out))
print(sorted(out[0].keys()) if out else None)
