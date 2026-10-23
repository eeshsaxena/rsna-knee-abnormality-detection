# lb_reads.py (10-21 ~19:00 IST, Claude, LASTDAY surface track). READ-ONLY on public LB snapshots already on disk (no API call, no OAI).
# For each E-OA-family read: our row (rank, display, best sid), our display band (size, teams above / below us inside it), and the
# frozen-team cross-board rulers between consecutive best-id reads. Writes reads_lb.json next to this file. MEASURED.
import json, collections, sys
sys.stdout.reconfigure(encoding='utf-8')
R = 'C:/Users/eeshs/Downloads/rsna-knee/'
LP = R + 'session_2026-10-14/research960/lbprobe/'
ME = 16799346
SNAPS = [
    ('c4@1020T0417', R + 'session_2026-10-20/forum/api_lb_1020T0417.json', 'c4', 57343435),
    ('eoa10@1020T0552', LP + 'api_lb_1020_eoaread1.json', 'eoa10', 57413380),
    ('eoa20@1020T0553', LP + 'api_lb_1020_eoaread2.json', 'eoa20', 57413368),
    ('pw30@1020T1320', LP + 'api_lb_1020_a21read.json', 'pw30', 57425970),
    ('pw30@1021T0311', R + 'session_2026-10-21/day/lb/api_lb_1021T_day.json', 'pw30', 57425970),
    ('swide@1021T1203', LP + 'api_lb_1021_a22read.json', 'swide', 57464326),
]


def load(f):
    d = json.load(open(f, encoding='utf-8')); rows = {}
    for r in d['lb']:
        try:
            s = round(float(r.get('displayScore')), 3)
        except Exception:
            s = None
        rows[r['teamId']] = dict(rank=r['rank'], score=s, sid=r['submissionId'])
    return d.get('fetched_utc', '?'), rows


out = {}
L = {}
for tag, f, ent, sid in SNAPS:
    fu, rows = load(f); L[tag] = rows; me = rows[ME]
    band = sorted((r['rank'], t) for t, r in rows.items() if r['score'] == me['score'])
    above = sum(1 for rk, t in band if rk < me['rank']); below = sum(1 for rk, t in band if rk > me['rank'])
    cnt = collections.Counter(r['score'] for r in rows.values() if r['score'] is not None)
    bands = {('%.3f' % s): cnt[s] for s in sorted(cnt, reverse=True) if s >= .957}
    out[tag] = dict(fetched_utc=fu, entry=ent, sid_expected=sid, sid_seen=me['sid'], ok=(me['sid'] == sid), rank=me['rank'],
                    display='%.3f' % me['score'], band_lo=band[0][0], band_hi=band[-1][0], band_n=len(band), above=above, below=below,
                    bands=bands, teams=len(rows))
    print('%-16s %s UTC | sid %s (expected %s) | r%d %s | band r%d-%d n %d | above %d below %d | counts %s' % (
        tag, fu[:16], me['sid'], sid, me['rank'], '%.3f' % me['score'], band[0][0], band[-1][0], len(band), above, below,
        ' '.join('%s:%d' % kv for kv in list(bands.items())[:6])))


def frozen(t0, t1, band_score):
    """teams with the same best sid in both snapshots and display == band_score in both."""
    A, B = L[t0], L[t1]
    return [t for t in A if t in B and t != ME and A[t]['sid'] == B[t]['sid'] and A[t]['score'] == band_score and B[t]['score'] == band_score]


print('\nfrozen cross-board rulers (teams with the same best submission in both snapshots, same display band):')
for t0, t1, sc in (('eoa10@1020T0552', 'eoa20@1020T0553', .959), ('eoa20@1020T0553', 'pw30@1020T1320', .959),
                   ('pw30@1020T1320', 'pw30@1021T0311', .959), ('pw30@1021T0311', 'swide@1021T1203', .960),
                   ('pw30@1020T1320', 'swide@1021T1203', .960)):
    F = frozen(t0, t1, sc); A, B = L[t0], L[t1]; m0, m1 = A[ME]['rank'], B[ME]['rank']
    a0 = [t for t in F if A[t]['rank'] < m0]; b0 = [t for t in F if A[t]['rank'] > m0]
    a1 = [t for t in F if B[t]['rank'] < m1]; b1 = [t for t in F if B[t]['rank'] > m1]
    cross = dict(above_then_below=len(set(a0) & set(b1)), below_then_above=len(set(b0) & set(a1)),
                 above_both=len(set(a0) & set(a1)), below_both=len(set(b0) & set(b1)))
    print('  %s -> %s band %.3f: frozen %d | %s' % (t0, t1, sc, len(F), cross))
    out['frozen %s -> %s' % (t0, t1)] = dict(band=sc, n=len(F), **cross)
json.dump(out, open(R + 'session_2026-10-21/lastday/work/reads_lb.json', 'w'), indent=1)
