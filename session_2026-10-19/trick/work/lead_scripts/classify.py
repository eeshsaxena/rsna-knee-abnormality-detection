import pandas as pd, numpy as np
W = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-19/trick/work/'
top = pd.read_csv(W + 'lead_top959.csv', parse_dates=['first959_snap', 'prev_t', 'best_t'])
S = pd.read_parquet(W + 'lead_ts.parquet').sort_values('t')
hist = S.pivot_table(index='teamId', columns='t', values='score', aggfunc='max')
def first_reach(tid, thr):
    if tid not in hist.index: return pd.NaT
    h = hist.loc[tid].dropna(); h = h[h >= thr]
    return h.index[0] if len(h) else pd.NaT
top['first958'] = top.teamId.map(lambda x: first_reach(x, .958))
top['first957'] = top.teamId.map(lambda x: first_reach(x, .957))
new = top[top.first959_snap >= pd.Timestamp('2026-10-15')].copy()
wave = pd.Timestamp('2026-10-15 15:00')
new['grp'] = np.where(new.first958 < wave, 'A_pre_wave_958', np.where(new.first957 < pd.Timestamp('2026-10-15 06:00'), 'B_957_pre_then_958', 'C_reached958_after_wave'))
print(new.grp.value_counts().to_string())
new['gap_h'] = (new.first959_snap - new.first958).dt.total_seconds() / 3600
for g, d in new.groupby('grp'):
    print('\n==', g, len(d), 'nmem', d.nmem.value_counts().to_dict(), 'nsub med', d.nsub.median(), 'tier mix', d.tiers.str.split(',').explode().value_counts().to_dict())
    print(d.sort_values('first959_snap')[['rank', 'displayScore', 'teamName', 'nmem', 'nsub', 'tiers', 'first957', 'first958', 'first959_snap', 'prev_score', 'best_t']].to_string(index=False, max_colwidth=26))
new.to_csv(W + 'lead_crossers_since1015.csv', index=False)
