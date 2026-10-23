import json, glob, pandas as pd, numpy as np
R = 'C:/Users/eeshs/Downloads/rsna-knee/'; W = R + 'session_2026-10-19/trick/work/'
files = glob.glob(R + 'session_2026-10-14/research960/lbprobe/api_lb_*.json') + [R + 'session_2026-10-19/think/work/api_lb_1019T_think.json', R + 'session_2026-10-17/research_1hr/api_lb_1017T_lens.json']
rows = []
for f in files:
    d = json.load(open(f)); t = pd.Timestamp(d['fetched_utc'])
    for x in d['teams']: rows.append((x['teamId'], t, x['submissionCount']))
d = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json')); t = pd.Timestamp(d['_fetched_utc'])
for x in d['teams']: rows.append((x['teamId'], t, x['submissionCount']))
N = pd.DataFrame(rows, columns=['teamId', 't', 'nsub']).pivot_table(index='teamId', columns='t', values='nsub', aggfunc='max')
S = pd.read_parquet(W + 'lead_ts.parquet')
H = S[S.src == 'api'].pivot_table(index='teamId', columns='t', values='score', aggfunc='max')
def nsub_at(tid, t):
    if tid not in N.index: return np.nan
    r = N.loc[tid]; c = [x for x in r.index if x <= t]
    v = r[c].dropna(); return v.iloc[-1] if len(v) else np.nan
cr = pd.read_csv(W + 'lead_crossers_since1015.csv', parse_dates=['first958', 'first959_snap', 'first957'])
cr['subs_958_to_959'] = [nsub_at(r.teamId, r.first959_snap) - nsub_at(r.teamId, r.first958) for r in cr.itertuples()]
print('subs spent between first .958 snapshot and first .959 snapshot, by group')
print(cr.groupby('grp').subs_958_to_959.describe().to_string())
print('group C score now:', cr[cr.grp == 'C_reached958_after_wave'].displayScore.value_counts().sort_index().to_dict())
print('group A score now:', cr[cr.grp == 'A_pre_wave_958'].displayScore.value_counts().sort_index().to_dict())
print('group B score now:', cr[cr.grp == 'B_957_pre_then_958'].displayScore.value_counts().sort_index().to_dict())
# active .958 teams that did NOT cross: teams at .958 now, with >= X subs since 10-16 06:00
tnow = max(N.columns); t0 = pd.Timestamp('2026-10-16 06:22:03.077176')
full = d
L = pd.DataFrame(full['publicLeaderboard']); L['score'] = L.displayScore.astype(float)
L['subs_since_1016'] = [nsub_at(r.teamId, tnow) - nsub_at(r.teamId, t0) for r in L.itertuples()]
at958 = L[L.score == .958]
print('teams at .958 now', len(at958), ' with >=5 subs since 10-16 06:22:', (at958.subs_since_1016 >= 5).sum(), ' >=10:', (at958.subs_since_1016 >= 10).sum(), ' >=15:', (at958.subs_since_1016 >= 15).sum())
c = cr[cr.first959_snap > t0]
c['subs_since_1016'] = [nsub_at(r.teamId, tnow) - nsub_at(r.teamId, t0) for r in c.itertuples()]
print('crossers after 10-16 06:22 (n=%d) subs since then median %.0f' % (len(c), c.subs_since_1016.median()))
for k in [5, 10, 15, 20, 25]:
    a = (at958.subs_since_1016 >= k).sum(); b = (c.subs_since_1016 >= k).sum()
    print(f'  >= {k} subs since 10-16: crossers {b} vs stuck at .958 {a}  -> cross frac {b/(a+b):.3f}')
# our team
me = L[L.teamId == 16799346]; print('our team subs since 10-16', me.subs_since_1016.values, me['rank'].values)
cr.to_csv(W + 'lead_crossers_since1015.csv', index=False)
