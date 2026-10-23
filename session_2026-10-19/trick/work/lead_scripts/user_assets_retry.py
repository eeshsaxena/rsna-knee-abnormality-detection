# Read-only, sequential with backoff: redo users that errored (429) or are missing.
import json, time, sys, os
from kaggle.api.kaggle_api_extended import KaggleApi
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
OUT = W + 'lead_meta/user_assets_top959.json'
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
sc = {r['teamId']: float(r['displayScore']) for r in full['publicLeaderboard']}
rk = {r['teamId']: r['rank'] for r in full['publicLeaderboard']}
users = [(m['userName'], t['teamId'], rk[t['teamId']], t['teamName']) for t in full['teams'] if sc.get(t['teamId'], 0) >= float(sys.argv[1]) for m in (t.get('teamMembers') or [])]
import pandas as pd
cr=set(pd.read_csv(W+'lead_crossers_since1015.csv').teamId)
users.sort(key=lambda x: (0 if x[1] in cr else (1 if x[2]<=70 else 2), x[2]))
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
todo = [u for u in users if u[0] not in out or 'ds_err' in out[u[0]] or ('k_err' in out[u[0]] and not os.environ.get('NOK'))]
print(len(todo), 'todo', flush=True)
api = KaggleApi(); api.authenticate()
def call(f):
    for a in range(6):
        try:
            return f(), None
        except Exception as e:
            err = str(e)[:100]
            time.sleep(60 if "429" in err else 3)
    return None, err
for i, (u, tid, r, tn) in enumerate(todo):
    rec = {'teamId': tid, 'rank': r, 'team': tn, 'datasets': [], 'kernels': []}
    ds, e1 = call(lambda: api.dataset_list(user=u, sort_by='updated') or [])
    if ds is not None:
        rec['datasets'] = [{'ref': d.ref, 'title': d.title, 'updated': str(d.last_updated), 'dl': d.download_count, 'size': d.total_bytes, 'kc': d.kernel_count, 'views': d.view_count, 'lic': d.license_name, 'sub': d.subtitle} for d in ds]
    else: rec['ds_err'] = e1
    time.sleep(1.5)
    ks, e2 = ([], None) if os.environ.get('NOK') else call(lambda: api.kernels_list(user=u, sort_by='dateRun', page_size=40) or [])
    if ks is not None:
        rec['kernels'] = [{'ref': k.ref, 'title': k.title, 'updated': str(k.last_run_time)} for k in ks]
    else: rec['k_err'] = e2
    time.sleep(1.5)
    out[u] = rec
    if i % 10 == 0:
        json.dump(out, open(OUT, 'w')); print(i, u, r, len(rec['datasets']), len(rec['kernels']), flush=True)
json.dump(out, open(OUT, 'w')); print('done')
