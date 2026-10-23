# ruler_counts.py: extra cross-board counts for the pw30 / swide rulers (READ-ONLY on LB snapshots on disk; MEASURED).
import json
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
LP = R + 'session_2026-10-14/research960/lbprobe/'
ME = 16799346
def load(f):
    d = json.load(open(f, encoding='utf-8')); rows = {}
    for r in d['lb']:
        try: s = round(float(r.get('displayScore')), 3)
        except Exception: s = None
        rows[r['teamId']] = dict(rank=r['rank'], score=s, sid=r['submissionId'])
    return rows
A = load(LP + 'api_lb_1020_eoaread2.json'); B = load(LP + 'api_lb_1020_a21read.json'); C = load(R + 'session_2026-10-21/day/lb/api_lb_1021T_day.json')
D = load(LP + 'api_lb_1021_a22read.json')
fro = set(t for t in A if t in C and t != ME and A[t]['sid'] == C[t]['sid'] and A[t]['score'] == .959 and C[t]['score'] == .959)
m = C[ME]['rank']; new = [t for t, r in C.items() if r['score'] == .959 and t != ME and t not in fro]
print('03:11 UTC 10-21: .959 teams not frozen since 05:53 UTC 10-20: %d | above pw30 %d | below %d' % (len(new), sum(C[t]['rank'] < m for t in new), sum(C[t]['rank'] > m for t in new)))
ab = [t for t in fro if A[t]['rank'] < A[ME]['rank']]
print('frozen 05:53 -> 03:11 teams above eoa20 at 05:53: %d | above pw30 at 03:11: %d' % (len(ab), sum(C[t]['rank'] < m for t in ab)))
md = D[ME]['rank']; bel = [t for t, r in D.items() if r['score'] == .960 and r['rank'] > md]
print('12:03 UTC 10-21: teams below swide in .960: %d | frozen .960 since 03:11: %d | were .959 at 03:11: %d' % (len(bel), sum(1 for t in bel if t in C and C[t]['sid'] == D[t]['sid'] and C[t]['score'] == .960), sum(1 for t in bel if t in C and C[t]['score'] == .959)))
