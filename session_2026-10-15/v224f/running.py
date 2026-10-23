# running.py: count Eesh's kernels that are running / queued now (the 2-concurrent-GPU-sessions rule). Read-only.
#   usage: python running.py [N]   (checks the N most recently run kernels, default 10). Prints RUNNING_COUNT k.
import sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ks = api.kernels_list(mine=True, sort_by='dateRun', page_size=N)
busy = []
for k in ks[:N]:
    try:
        st = str(api.kernels_status(k.ref).status)
    except Exception as e:
        st = 'ERR %s' % type(e).__name__
    print(time.strftime('%H:%M:%S'), k.ref, k.last_run_time, st, flush=True)
    if 'RUNNING' in st.upper() or 'QUEUED' in st.upper():
        busy.append(k.ref)
print('RUNNING_COUNT', len(busy), busy, flush=True)
