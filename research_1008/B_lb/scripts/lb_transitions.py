import glob, os, re, csv, collections
D = os.path.join(os.path.dirname(__file__), '..', 'hist')
def load(tag):
    f = [x for x in glob.glob(os.path.join(D, '*.csv')) if tag in x][0]
    out = {}
    with open(f, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            try: s = float(r['Score'])
            except Exception: continue
            out[int(r['TeamId'])] = (int(r['Rank']), s, r['LastSubmissionDate'], int(r['SubmissionCount']), r['TeamName'])
    return out
a = load('10-04T09'); b = load('10-08T03'); c = load('10-08T15')
for (na, A), (nb, B) in [(('10-04', a), ('10-08T03', b)), (('10-08T03', b), ('10-08T15', c))]:
    print(f'\n=== transitions into each score bucket, {na} -> {nb}')
    for sc in [.953, .952, .951, .950, .949, .948]:
        prev = collections.Counter()
        for t, v in B.items():
            if abs(v[1] - sc) < 1e-9:
                p = A.get(t)
                prev['new team' if p is None else f'{p[1]:.3f}'] += 1
        print(f'{sc:.3f}: n={sum(prev.values())}  from: ' + ', '.join(f'{k}:{v}' for k, v in prev.most_common(8)))
# last submission hour histogram for .950 teams in c
print('\n.950 teams (10-08T15) by last-submission date (UTC day-hour)')
h = collections.Counter(v[2][:13] for t, v in c.items() if abs(v[1] - .950) < 1e-9)
for k in sorted(h): print(k, h[k])
print('\n>=.951 teams (10-08T15) last-sub day')
h = collections.Counter(v[2][:10] for t, v in c.items() if v[1] >= .951 - 1e-9)
for k in sorted(h): print(k, h[k])
# submission counts of .950 cluster
sc = [v[3] for t, v in c.items() if abs(v[1] - .950) < 1e-9]
import statistics
print('\n.950 cluster submission count median', statistics.median(sc), 'min', min(sc), 'max', max(sc))
sc = [v[3] for t, v in c.items() if v[1] >= .951]
print('>=.951 submission count median', statistics.median(sc))
# rank range of .950 in c
r = sorted(v[0] for t, v in c.items() if abs(v[1] - .950) < 1e-9); print('.950 rank range', r[0], r[-1])
r = sorted(v[0] for t, v in c.items() if abs(v[1] - .949) < 1e-9); print('.949 rank range', r[0], r[-1])
r = sorted(v[0] for t, v in c.items() if abs(v[1] - .951) < 1e-9); print('.951 rank range', r[0], r[-1])
r = sorted(v[0] for t, v in c.items() if abs(v[1] - .948) < 1e-9); print('.948 rank range', r[0], r[-1])
r = sorted(v[0] for t, v in c.items() if abs(v[1] - .946) < 1e-9); print('.946 rank range', r[0], r[-1])
