# a21_frozen.py (10-20 val Track D; PUBLIC LB snapshots only, read-only). Frozen-team ruler between the eoa20 read (05:53 UTC) and the
# A21 read (13:20 UTC): teams whose best submissionId is unchanged have unchanged unrounded scores, so pw30's position among the frozen
# .959 teams that sat above eoa20 brackets the paired public increment pw30 - eoa20 (MEASURED counts; conversion in abc_a21.py = MODEL).
import json
P = 'C:/Users/eeshs/Downloads/rsna-knee/session_2026-10-14/research960/lbprobe/'
ME = 16799346
def load(fn):
    D = json.load(open(P + fn)); return D['fetched_utc'], sorted(D['lb'], key=lambda r: r['rank'])
t1, L1 = load('api_lb_1020_eoaread2.json'); t2, L2 = load('api_lb_1020_a21read.json')
d1 = {r['teamId']: r for r in L1}; d2 = {r['teamId']: r for r in L2}; m1, m2 = d1[ME], d2[ME]
print('t1 %s ours %s r%d %s | t2 %s ours %s r%d %s' % (t1, m1['submissionId'], m1['rank'], m1['displayScore'], t2, m2['submissionId'], m2['rank'], m2['displayScore']))
for L, t in ((L1, 't1'), (L2, 't2')):
    bands = {}
    for r in L: bands.setdefault(r['displayScore'], []).append(r['rank'])
    print(t, ' '.join('%s n%d r%d-%d' % (s, len(bands[s]), min(bands[s]), max(bands[s])) for s in ('0.961', '0.960', '0.959', '0.958')))
fz = [t for t in d1 if t in d2 and t != ME and d1[t]['submissionId'] == d2[t]['submissionId']]
o1 = sorted(fz, key=lambda t: d1[t]['rank']); o2 = sorted(fz, key=lambda t: d2[t]['rank'])
print('frozen teams %d | relative order of frozen teams identical t1 vs t2: %s' % (len(fz), o1 == o2))
for b in ('0.960', '0.959', '0.958'):
    fb = [t for t in fz if d1[t]['displayScore'] == b]
    print('  frozen %s n %d | above eoa20 at t1 %d | above pw30 at t2 %d' % (b, len(fb), sum(d1[t]['rank'] < m1['rank'] for t in fb), sum(d2[t]['rank'] < m2['rank'] for t in fb)))
ab1 = [r for r in L1 if r['displayScore'] == '0.959' and r['rank'] < m1['rank']]
print('the %d .959 teams above eoa20 at t1 (all teams ruler):' % len(ab1))
for r in ab1:
    q = d2.get(r['teamId']); st = 'frozen' if q and q['submissionId'] == r['submissionId'] else 'moved'
    print('   t1 r%d -> t2 r%d %s %s | %s pw30' % (r['rank'], q['rank'], q['displayScore'], st, 'ABOVE' if q['rank'] < m2['rank'] else 'below'))
ab2 = [r for r in L2 if r['displayScore'] == '0.959' and r['rank'] < m2['rank']]
print('teams above pw30 inside .959 at t2: %d (%s)' % (len(ab2), ', '.join('t2 r%d from t1 r%d %s' % (r['rank'], d1[r['teamId']]['rank'], d1[r['teamId']]['displayScore']) for r in ab2)))
n959 = sum(r['displayScore'] == '0.959' for r in L2)
print('t2 .959 band: %d teams incl. ours; ours has %d above / %d below' % (n959, len(ab2), n959 - 1 - len(ab2)))
print('RESULT (MEASURED): order t1r131 > t1r132 > pw30 > t1r133 > t1r134 > eoa20 (all four frozen); pw30 sits between the 2nd and 3rd of the 4 frozen teams that were above eoa20')
