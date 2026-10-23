# Anonymous read-only fetch of competition pages (overview / evaluation / data / rules). No cookies of Eesh.
import sys, json, time, requests
sys.stdout.reconfigure(encoding='utf-8')
BASE='https://www.kaggle.com'
s=requests.Session(); s.get(BASE+'/competitions/rsna-knee-abnormality-detection/overview', timeout=60)
H={'x-xsrf-token': s.cookies.get('XSRF-TOKEN','')}
out=sys.argv[1]
def post(m, body):
    r=s.post(BASE+'/api/i/'+m, json=body, headers=H, timeout=60)
    return r.status_code, (r.json() if r.headers.get('content-type','').startswith('application/json') else r.text[:500])
for m,b in [('competitions.PageService/ListPages',{'competitionId':154281}),
            ('competitions.CompetitionService/GetCompetition',{'competitionName':'rsna-knee-abnormality-detection'})]:
    st,j=post(m,b); print(m, st, (list(j.keys()) if isinstance(j,dict) else j)[:20])
    json.dump(j, open(out+'/'+m.split('/')[-1]+'.json','w',encoding='utf-8'), indent=1, ensure_ascii=False)
    time.sleep(0.5)
