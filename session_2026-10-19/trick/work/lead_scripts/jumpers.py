import json, pandas as pd, numpy as np
R = 'C:/Users/eeshs/Downloads/rsna-knee/'; W = R + 'session_2026-10-19/trick/work/'
S = pd.read_parquet(W + 'lead_ts.parquet')
S = S.sort_values('t')
# id->time calibration from our own submissions
o = pd.read_csv(W + 'lead_lb/our_subs2.csv'); o['date'] = pd.to_datetime(o['date'], format='mixed'); o = o.sort_values('ref')
xs = o.ref.values.astype(float); ts = (o.date - pd.Timestamp('2026-10-01')).dt.total_seconds().values / 86400
rate_end = (xs[-1] - xs[-8]) / (ts[-1] - ts[-8])
def id2t(i):
    i = np.asarray(i, float)
    t = np.interp(i, xs, ts)
    t = np.where(i > xs[-1], ts[-1] + (i - xs[-1]) / rate_end, t)
    return pd.Timestamp('2026-10-01') + pd.to_timedelta(t, unit='D')
print('rate_end ids/day', rate_end)
full = json.load(open(W + 'lead_lb/api_lb_full_1019_now.json'))
T = pd.DataFrame([{'teamId': t['teamId'], 'teamName': t['teamName'], 'nsub': t['submissionCount'], 'last': t['lastSubmissionDate'],
                   'nmem': len(t.get('teamMembers') or []), 'members': ','.join(m['userName'] for m in (t.get('teamMembers') or [])),
                   'tiers': ','.join((m.get('tier') or '?')[:2] for m in (t.get('teamMembers') or []))} for t in full['teams']])
L = pd.DataFrame(full['publicLeaderboard']); L['score'] = L.displayScore.astype(float)
L = L.merge(T, on='teamId')
L['best_t'] = id2t(L.submissionId)
top = L[L.score >= .959].copy()
print('now >=.959:', len(top))
snaps = sorted(S.t.unique())
hist = S.pivot_table(index='teamId', columns='t', values='score', aggfunc='max')
def first_cross(tid, thr=.959):
    if tid not in hist.index: return (pd.NaT, np.nan, np.nan)
    h = hist.loc[tid].dropna()
    prev = np.nan; prevt = pd.NaT
    for t, v in h.items():
        if v >= thr: return (t, prev, prevt)
        prev, prevt = v, t
    return (pd.NaT, prev, prevt)
fc = top.teamId.map(lambda x: first_cross(x))
top['first959_snap'] = [a for a, b, c in fc]; top['prev_score'] = [b for a, b, c in fc]; top['prev_t'] = [c for a, b, c in fc]
# score on fixed dates
def score_at(tid, when):
    if tid not in hist.index: return np.nan
    h = hist.loc[tid]; cols = [c for c in h.index if c <= pd.Timestamp(when)]
    if not cols: return np.nan
    v = h[cols].dropna()
    return v.iloc[-1] if len(v) else np.nan
for d in ['2026-10-04 12:00', '2026-10-08 17:00', '2026-10-14 06:00', '2026-10-15 06:00', '2026-10-16 06:30', '2026-10-17 06:00', '2026-10-18 07:00', '2026-10-19 11:00']:
    top['s_' + d[5:10]] = top.teamId.map(lambda x: score_at(x, d))
top.to_csv(W + 'lead_top959.csv', index=False)
# Jump-date histogram by best submission time (UTC day)
top['best_day'] = top.best_t.dt.strftime('%m-%d')
print(top.groupby('best_day').size().to_string())
print('first crossing snapshot by day:')
top['fc_day'] = pd.to_datetime(top.first959_snap).dt.strftime('%m-%d')
print(top.groupby('fc_day', dropna=False).size().to_string())
# for teams crossing after 10-15: prior score distribution
new = top[pd.to_datetime(top.first959_snap) >= pd.Timestamp('2026-10-15')]
print('crossers since 10-15:', len(new)); print(new.prev_score.value_counts(dropna=False).sort_index().to_string())
print('team size of crossers since 10-15:', new.nmem.value_counts().to_dict(), ' all top:', top.nmem.value_counts().to_dict())
print('nsub median crossers', new.nsub.median(), 'top all', top.nsub.median())
