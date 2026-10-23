# Read-only: list public datasets + kernels of every member of .959+ teams (and some refs). Writes JSON.
import json, time, sys
from kaggle.api.kaggle_api_extended import KaggleApi
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
api = KaggleApi(); api.authenticate()
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
sc = {r['teamId']: float(r['displayScore']) for r in full['publicLeaderboard']}
rk = {r['teamId']: r['rank'] for r in full['publicLeaderboard']}
users = []
for t in full['teams']:
    if sc.get(t['teamId'], 0) >= float(sys.argv[1]) and rk.get(t['teamId'], 9999) <= int(sys.argv[2]):
        for m in (t.get('teamMembers') or []):
            users.append((m['userName'], t['teamId'], rk[t['teamId']], t['teamName']))
print(len(users), 'users', flush=True)
out = {}
for i, (u, tid, r, tn) in enumerate(users):
    rec = {'teamId': tid, 'rank': r, 'team': tn, 'datasets': [], 'kernels': []}
    for attempt in range(3):
        try:
            ds = api.dataset_list(user=u, sort_by='updated') or []
            rec['datasets'] = [{'ref': d.ref, 'title': d.title, 'updated': str(d.last_updated), 'dl': d.download_count, 'size': d.total_bytes, 'kc': d.kernel_count, 'views': d.view_count, 'lic': d.license_name, 'sub': d.subtitle, 'priv': d.is_private} for d in ds]
            break
        except Exception as e:
            rec['ds_err'] = str(e)[:100]; time.sleep(3)
    for attempt in range(3):
        try:
            ks = api.kernels_list(user=u, sort_by='dateRun', page_size=40) or []
            rec['kernels'] = [{'ref': k.ref, 'title': k.title, 'updated': str(k.last_run_time)} for k in ks]
            break
        except Exception as e:
            rec['k_err'] = str(e)[:100]; time.sleep(3)
    out[u] = rec
    if i % 20 == 0:
        print(i, u, len(rec['datasets']), len(rec['kernels']), flush=True)
        json.dump(out, open(W + 'lead_meta/user_assets_%s.json' % sys.argv[3], 'w'))
json.dump(out, open(W + 'lead_meta/user_assets_%s.json' % sys.argv[3], 'w'))
print('done')
