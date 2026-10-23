# kvmtop.py: anonymous GetKernelViewModel for a few refs (read-only); prints best/current version score + run time.
import sys, json
sys.path.insert(0, r'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-15/watch')
from kgweb import Web, summarize_kvm
w = Web(); out = {}
for ref in sys.argv[2:]:
    st, j = w.kvm(ref)
    if j is None: print(ref, st); continue
    s = summarize_kvm(j); out[ref] = s
    print(ref, json.dumps({k: s.get(k) for k in list(s.keys())[:12]}, default=str)[:400])
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
