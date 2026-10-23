# Build team x snapshot score/bestId time series from stored LB snapshots (read-only analysis).
import json, glob, os, pandas as pd, numpy as np
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
W = R + 'session_2026-10-19/trick/work/'
snaps = []
# API snapshots
api_files = glob.glob(R + 'session_2026-10-14/research960/lbprobe/api_lb_*.json') + [
    R + 'session_2026-10-15/silver/pulled2/forum_1015T1030/api_lb_now.json',
    R + 'session_2026-10-17/research_1hr/api_lb_1017T_lens.json',
    R + 'session_2026-10-19/think/work/api_lb_1019T_think.json']
for f in api_files:
    d = json.load(open(f))
    t = pd.Timestamp(d['fetched_utc'])
    L = pd.DataFrame(d['lb'])[['teamId', 'submissionId', 'rank', 'displayScore']]
    L['score'] = L.displayScore.astype(float)
    snaps.append((t, 'api', f, L))
# internal 10-08 (no fetched time: use file mtime)
f = R + 'research_1008/A_lists/lb_1008_internal.json'
d = json.load(open(f))
L = pd.DataFrame(d['publicLeaderboard'])
if 'displayScore' in L:
    L['score'] = L.displayScore.astype(float)
    snaps.append((pd.Timestamp(os.path.getmtime(f), unit='s'), 'api1008', f, L[['teamId', 'submissionId', 'rank', 'displayScore', 'score']]))
# full API now
f = W + 'lead_lb/api_lb_full_1019_now.json'
d = json.load(open(f)); L = pd.DataFrame(d['publicLeaderboard'])
L['score'] = L.displayScore.astype(float)
snaps.append((pd.Timestamp(d['_fetched_utc']), 'api', f, L[['teamId', 'submissionId', 'rank', 'displayScore', 'score']]))
# CSV snapshots
for f in sorted(set(glob.glob(R + 'research_1008/B_lb/hist/*.csv') + glob.glob(R + 'research_1008/A_lists/lb_now/*.csv') + glob.glob(R + 'research_1008/verify_B/lb/*.csv') + glob.glob(W + 'lead_lb/*publicleaderboard*.csv'))):
    ts = os.path.basename(f).split('publicleaderboard-')[1].replace('.csv', '').replace('_', ':')
    t = pd.Timestamp(ts)
    C = pd.read_csv(f, encoding='utf-8-sig')
    L = pd.DataFrame({'teamId': C.TeamId, 'submissionId': np.nan, 'rank': C.Rank, 'displayScore': C.Score.map(lambda x: f'{x:.3f}'), 'score': C.Score})
    snaps.append((t, 'csv', f, L))
snaps.sort(key=lambda x: x[0])
# dedupe near-identical times
rows = []
for t, k, f, L in snaps:
    L = L.copy(); L['t'] = t; L['src'] = k; rows.append(L)
S = pd.concat(rows, ignore_index=True)
S.to_parquet(W + 'lead_ts.parquet')
print(len(snaps), 'snapshots', S.t.min(), S.t.max())
for t, k, f, L in snaps:
    print(t, k, len(L), (L.score >= .959).sum(), (L.score >= .958).sum(), os.path.basename(f))
