import csv,sys,collections
def load(f):
    rows=[]
    for r in csv.DictReader(open(f,encoding='utf-8-sig')):
        try: s=float(r['Score'])
        except: continue
        rows.append((int(r['Rank']),r['TeamName'],s,r['LastSubmissionDate'],int(r['SubmissionCount']),r['TeamMemberUserNames']))
    return rows
for f in sys.argv[1:]:
    rows=sorted(load(f)); n=len(rows)
    g=10+int(.002*n); sv=int(.05*n); b=int(.1*n)
    c=collections.Counter(round(r[2],3) for r in rows)
    print(f, 'teams',n,'gold',g,rows[g-1][2],'silver',sv,rows[sv-1][2],'bronze',b,rows[b-1][2])
    for s in (.955,.954,.953,.952,.951,.950,.949,.948,.947,.946):
        print(f'  {s:.3f} at {c[s]:4d}  >= {sum(1 for r in rows if r[2]>=s-1e-9):5d}')
    for r in rows:
        if 'eesh' in r[5].lower(): print('  us',r)
    at950=[r[0] for r in rows if abs(r[2]-.95)<1e-9]; print('  .950 ranks',min(at950),max(at950))
