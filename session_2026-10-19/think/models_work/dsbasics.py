# dsbasics.py: anonymous GetDatasetBasics for a list of dataset refs (read-only). Prints title/subtitle/description head.
import sys, json
sys.path.insert(0, r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/watch')
from kgweb import Web
w = Web()
out = {}
for ref in sys.argv[2:]:
    st, j = w.dataset(ref)
    out[ref] = {'status': st, 'json': j}
    if j is None: print('==', ref, st); continue
    flat = json.dumps(j)
    print('==', ref, st, 'keys:', list(j.keys())[:30])
json.dump(out, open(sys.argv[1], 'w', encoding='utf-8'), indent=1)
