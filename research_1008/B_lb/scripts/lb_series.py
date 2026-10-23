"""LB time series: score distribution, mass points, cluster growth, medal cutoffs. Laptop, light."""
import glob, os, re, csv, collections
D = os.path.join(os.path.dirname(__file__), '..', 'hist')
files = sorted(glob.glob(os.path.join(D, '*.csv')))
snaps = []
for f in files:
    ts = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2})', f).group(1)
    rows = []
    with open(f, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            try:
                s = float(r['Score'])
            except Exception:
                continue
            rows.append((int(r['Rank']), int(r['TeamId']), r['TeamName'], r['LastSubmissionDate'], s, int(r['SubmissionCount'])))
    snaps.append((ts, rows))

def cut(rows, rank):
    rs = sorted(rows, key=lambda x: x[0])
    return rs[rank - 1][4] if len(rs) >= rank else None

print('snapshot            teams   top    r10    r20    r50   r100   r200   r272?  10%rank  bronze(10%)  silver(5%)  gold')
for ts, rows in snaps:
    n = len(rows)
    rs = sorted(rows, key=lambda x: x[0])
    # Kaggle medal rules for n >= 1000 teams: gold = top 10 + 0.2%, silver top 5%, bronze top 10%
    g = 10 + int(0.002 * n); sv = int(0.05 * n); b = int(0.10 * n)
    print(f'{ts}  {n:5d}  {rs[0][4]:.3f}  {cut(rows,10):.3f}  {cut(rows,20):.3f}  {cut(rows,50):.3f}  {cut(rows,100):.3f}  {cut(rows,200):.3f}  {cut(rows,272):.3f}   '
          f'gold@{g}={cut(rows,g):.3f} silver@{sv}={cut(rows,sv):.3f} bronze@{b}={cut(rows,b):.3f}')

print('\nmass points (count of teams at each score >= .940) per snapshot')
allsc = [round(x / 1000, 3) for x in range(940, 966)]
hdr = 'score  ' + '  '.join(ts[5:13] for ts, _ in snaps)
print(hdr)
for sc in reversed(allsc):
    line = f'{sc:.3f} '
    for ts, rows in snaps:
        c = sum(1 for r in rows if abs(r[4] - sc) < 1e-9)
        line += f' {c:9d}'
    print(line)
print('\ncount >= score (teams at or above) per snapshot')
for sc in [.964, .960, .959, .958, .955, .952, .951, .950, .949, .948, .947, .946, .945, .944, .943]:
    line = f'>={sc:.3f}'
    for ts, rows in snaps:
        c = sum(1 for r in rows if r[4] >= sc - 1e-9)
        line += f' {c:9d}'
    print(line)

# our team
print('\nour team rows')
for ts, rows in snaps:
    for r in rows:
        if 'eesh' in r[2].lower() or r[1] in ():
            print(ts, r)
