# Read-only, parallel: public datasets + kernels for each member of .959+ teams. Resumes from existing JSON.
import json, time, sys, os, threading
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
OUT = W + 'lead_meta/user_assets_top959.json'
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
sc = {r['teamId']: float(r['displayScore']) for r in full['publicLeaderboard']}
rk = {r['teamId']: r['rank'] for r in full['publicLeaderboard']}
users = [(m['userName'], t['teamId'], rk[t['teamId']], t['teamName']) for t in full['teams'] if sc.get(t['teamId'], 0) >= float(sys.argv[1]) for m in (t.get('teamMembers') or [])]
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
todo = [u for u in users if u[0] not in out]
print(len(users), 'users', len(todo), 'todo', flush=True)
loc = threading.local(); lock = threading.Lock()
def get_api():
    if not hasattr(loc, 'api'):
        loc.api = KaggleApi(); loc.api.authenticate()
    return loc.api
def work(x):
    u, tid, r, tn = x; api = get_api()
    rec = {'teamId': tid, 'rank': r, 'team': tn, 'datasets': [], 'kernels': []}
    for a in range(3):
        try:
            ds = api.dataset_list(user=u, sort_by='updated') or []
            rec['datasets'] = [{'ref': d.ref, 'title': d.title, 'updated': str(d.last_updated), 'dl': d.download_count, 'size': d.total_bytes, 'kc': d.kernel_count, 'views': d.view_count, 'lic': d.license_name, 'sub': d.subtitle} for d in ds]
            rec.pop('ds_err', None); break
        except Exception as e:
            rec['ds_err'] = str(e)[:100]; time.sleep(3)
    for a in range(3):
        try:
            ks = api.kernels_list(user=u, sort_by='dateRun', page_size=40) or []
            rec['kernels'] = [{'ref': k.ref, 'title': k.title, 'updated': str(k.last_run_time)} for k in ks]
            rec.pop('k_err', None); break
        except Exception as e:
            rec['k_err'] = str(e)[:100]; time.sleep(3)
    with lock:
        out[u] = rec
        if len(out) % 25 == 0:
            json.dump(out, open(OUT, 'w')); print(len(out), flush=True)
with ThreadPoolExecutor(8) as ex:
    list(ex.map(work, todo))
json.dump(out, open(OUT, 'w')); print('done', len(out))
