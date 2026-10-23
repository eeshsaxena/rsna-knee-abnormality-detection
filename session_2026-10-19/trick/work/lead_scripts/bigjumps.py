import json, pandas as pd, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
S = pd.read_parquet(W + 'lead_ts.parquet').sort_values('t')
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
T = pd.DataFrame([{'teamId': t['teamId'], 'teamName': t['teamName'], 'nsub': t['submissionCount'], 'nmem': len(t.get('teamMembers') or []),
                   'members': ','.join(m['userName'] for m in (t.get('teamMembers') or []))} for t in full['teams']]).set_index('teamId')
hist = S.pivot_table(index='teamId', columns='t', values='score', aggfunc='max')
# step jumps between consecutive observed snapshots
ev = []
for tid, h in hist.iterrows():
    h = h.dropna()
    if len(h) < 2: continue
    v = h.values; t = h.index
    for i in range(1, len(v)):
        if v[i] - v[i - 1] >= .003 and v[i] >= .958:
            ev.append((tid, t[i - 1], t[i], v[i - 1], v[i]))
E = pd.DataFrame(ev, columns=['teamId', 't0', 't1', 's0', 's1']).join(T, on='teamId')
E['d'] = E.s1 - E.s0
E = E.sort_values('t1')
pd.set_option('display.width', 250)
print(E[['t0', 't1', 's0', 's1', 'teamName', 'nmem', 'nsub', 'members']].to_string(index=False, max_colwidth=40))
# per window count (normalised per day)
E['win'] = E.t1.dt.strftime('%m-%d')
print(E.groupby('win').size())
E.to_csv(W + 'lead_bigjumps.csv', index=False)
