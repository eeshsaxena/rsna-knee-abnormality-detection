import json, pandas as pd, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
S = pd.read_parquet(W + 'lead_ts.parquet').sort_values('t')
S = S[S.src.isin(['api', 'api1008'])].dropna(subset=['submissionId'])
o = pd.read_csv(W + 'lead_lb/our_subs2.csv'); o['date'] = pd.to_datetime(o['date'], format='mixed'); o = o.sort_values('ref')
xs = o.ref.values.astype(float); ts = (o.date - pd.Timestamp('2026-10-01')).dt.total_seconds().values / 86400
rate_end = (xs[-1] - xs[-8]) / (ts[-1] - ts[-8])
def id2t(i):
    i = np.asarray(i, float); t = np.interp(i, xs, ts)
    t = np.where(i > xs[-1], ts[-1] + (i - xs[-1]) / rate_end, t)
    t = np.where(i < xs[0], np.nan, t)
    return pd.Timestamp('2026-10-01') + pd.to_timedelta(t, unit='D')
snaps = sorted(S.t.unique())
prev = None; ev = []
for t in snaps:
    L = S[S.t == t].set_index('teamId')[['submissionId', 'score']]
    if prev is not None:
        j = L.join(prev, rsuffix='_0', how='left')
        ch = j[(j.submissionId != j.submissionId_0)]
        for tid, r in ch.iterrows():
            ev.append((tid, t, r.submissionId, r.score_0, r.score))
    prev = L
E = pd.DataFrame(ev, columns=['teamId', 'snap', 'subId', 's0', 's1'])
E['sub_t'] = id2t(E.subId)
E['d'] = E.s1 - E.s0
E.to_csv(W + 'lead_events.csv', index=False)
print('events', len(E))
E['bin'] = E.sub_t.dt.floor('12h')
g = E.groupby('bin').agg(n=('s1', 'size'), n959=('s1', lambda x: (x >= .959).sum()), n958=('s1', lambda x: (x == .958).sum()),
                         new959=('s1', lambda x: 0))
# new959: events where s0<.959<=s1
E['cross959'] = (E.s1 >= .959) & ((E.s0 < .959) | E.s0.isna())
E['cross958'] = (E.s1 >= .958) & ((E.s0 < .958) | E.s0.isna())
E['big959'] = (E.s1 >= .959) & (E.d >= .003)
g = E.groupby('bin').agg(n=('s1', 'size'), cross958=('cross958', 'sum'), cross959=('cross959', 'sum'), big959=('big959', 'sum'),
                         top_improve=('s1', lambda x: (x >= .960).sum()))
print(g.to_string())
